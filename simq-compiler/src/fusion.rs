//! Gate fusion optimization pass
//!
//! This module implements gate fusion, which combines adjacent gates into a
//! single composite gate to reduce the number of full-state passes the
//! simulator has to make. Two algorithms are implemented:
//!
//! - [`find_fusion_chains`] — the original algorithm: fuses adjacent
//!   *single-qubit* gates on the *same wire* into a 2x2 matrix. Always used
//!   when `circuit.num_qubits()` is below `FusionConfig::parallel_threshold_qubits`
//!   (default 18) or `FusionConfig::max_block_width <= 1`.
//! - [`find_fusion_blocks`] — a greedy, single-forward-pass, width-bounded
//!   frontier merge that fuses adjacent gates spanning *multiple* qubits
//!   into one block (up to `FusionConfig::max_block_width` qubits). Only
//!   used above the qubit-count gate above, since multi-qubit fusion only
//!   pays for itself once per-pass memory traffic dominates over per-gate
//!   dispatch overhead (see `simq-sim`'s `parallel_threshold`, default
//!   `1 << 18` amplitudes / 18 qubits — the same crossover this module's
//!   default `parallel_threshold_qubits` reuses).
//!
//! Deliberately **not** implemented: a circuit-wide fusion graph with a
//! per-edge cost model and shortest-path schedule selection, and fusion
//! that crosses a measurement boundary. Both are the specific mechanisms
//! claimed by two granted patents found during this feature's design
//! diligence; this module's greedy width-bounded approach (matching the
//! independently-documented "greedy gate fusion" baseline used elsewhere,
//! e.g. Google's qsim) and its hard stop at any gate with no matrix
//! representation (which includes measurement) are structurally distinct
//! from both.
//!
//! # Theory
//!
//! When multiple gates act on the same qubit(s) sequentially with no
//! intervening operation on those qubits, they can be combined by
//! multiplying their unitary matrices (embedded into a shared basis when
//! they act on different, overlapping qubit subsets):
//!
//! ```text
//! |ψ⟩ → U₃(U₂(U₁|ψ⟩)) = (U₃·U₂·U₁)|ψ⟩
//! ```
//!
//! The fused gate's matrix is the product of the individual (embedded) gate
//! matrices, composed in application order (rightmost/earliest gate is
//! multiplied first).
//!
//! # Example
//!
//! ```ignore
//! use simq_core::Circuit;
//! use simq_gates::standard::{Hadamard, PauliX};
//! use simq_compiler::fusion::fuse_single_qubit_gates;
//!
//! // Create circuit with adjacent single-qubit gates
//! let mut circuit = Circuit::new(2);
//! circuit.add_gate(Hadamard, &[q0]);
//! circuit.add_gate(PauliX, &[q0]);
//! circuit.add_gate(Hadamard, &[q0]);
//!
//! // Apply fusion optimization
//! let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();
//! // Three gates on q0 are fused into one FusedGate
//! assert!(optimized.len() < circuit.len());
//! ```

use crate::matrix_utils::{
    identity_flat, is_identity, is_identity_flat, multiply_2x2, multiply_square, FlatMatrix,
};
use ahash::{AHashMap, AHashSet};
use num_complex::Complex64;
use simq_core::{gate::Gate, Circuit, GateOp, QubitId, Result};
use simq_gates::matrix_ops::embed_gate_matrix_vec;
use smallvec::SmallVec;
use std::fmt;
use std::sync::Arc;

/// Configuration for gate fusion optimization
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Minimum number of gates to fuse (default: 2)
    /// Gates sequences shorter than this won't be fused
    pub min_fusion_size: usize,

    /// Whether to eliminate identity gates (default: true)
    /// If a fused gate results in identity, it will be removed
    pub eliminate_identity: bool,

    /// Epsilon for identity detection (default: 1e-10)
    pub identity_epsilon: f64,

    /// Maximum number of component gates a single fused unit may absorb
    /// (default: None = unlimited). Orthogonal to `max_block_width`: this
    /// limits gate *count* per fused unit, `max_block_width` limits qubit
    /// *span*.
    pub max_fusion_size: Option<usize>,

    /// Maximum number of qubits a single fused block may span (default: 3).
    /// Only takes effect once `circuit.num_qubits() >= parallel_threshold_qubits`
    /// — see `parallel_threshold_qubits` and [`fuse_single_qubit_gates`]'s
    /// dispatch. A value of `1` forces the legacy single-qubit-chain path
    /// unconditionally.
    pub max_block_width: usize,

    /// Circuit qubit count below which fusion always uses the legacy
    /// single-qubit chain path ([`find_fusion_chains`]) regardless of
    /// `max_block_width` (default: 18). Matches `simq-sim`'s
    /// `ExecutionConfig`/`SimulatorConfig` `parallel_threshold` default of
    /// `1 << 18` amplitudes / 18 qubits (see
    /// `simq-sim/src/execution_engine/config.rs` and
    /// `simq-sim/src/config.rs`) — the point where per-pass memory traffic
    /// starts to dominate over per-gate dispatch overhead, which is when
    /// multi-qubit fusion's extra compile-time bookkeeping starts to pay
    /// for itself. `simq-compiler` has no dependency on `simq-sim`, so this
    /// is a separately-defined constant kept in sync by convention, not by
    /// a shared type.
    pub parallel_threshold_qubits: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            min_fusion_size: 2,
            eliminate_identity: true,
            identity_epsilon: 1e-10,
            max_fusion_size: None,
            max_block_width: 3,
            parallel_threshold_qubits: 18,
        }
    }
}

/// A fused quantum gate representing the composition of multiple gates
/// spanning one or more qubits.
///
/// This gate stores the product of multiple (embedded) gate matrices and
/// maintains information about the original gates for debugging and
/// visualization.
#[derive(Clone)]
pub struct FusedGate {
    /// Qubits this block spans, in the fixed canonical order the matrix was
    /// composed against. Whoever executes this gate (the simulator
    /// executor) must pass qubits to the underlying kernel in exactly this
    /// order — see the module-level bit-ordering note on
    /// [`compose_block_matrix`].
    qubits: SmallVec<[QubitId; 4]>,

    /// The composed unitary matrix, flat row-major `2^k x 2^k` where
    /// `k = qubits.len()`. Inline-capacity 64 (an 8x8 / 3-qubit block)
    /// avoids a heap allocation for every block up to the width this crate
    /// currently supports.
    matrix: FlatMatrix,

    /// Names of the original gates in application order
    /// (first gate in vector is applied first)
    component_gates: SmallVec<[String; 4]>,
}

impl FusedGate {
    /// Create a new fused gate from its qubits (canonical order), composed
    /// matrix (flat row-major `2^k x 2^k`), and component gate names.
    pub fn new(
        qubits: SmallVec<[QubitId; 4]>,
        matrix: FlatMatrix,
        component_gates: SmallVec<[String; 4]>,
    ) -> Self {
        Self {
            qubits,
            matrix,
            component_gates,
        }
    }

    /// Get the qubits this block spans, in canonical order.
    pub fn qubits(&self) -> &[QubitId] {
        &self.qubits
    }

    /// Get the composed unitary matrix (flat row-major `2^k x 2^k`)
    pub fn matrix(&self) -> &[Complex64] {
        &self.matrix
    }

    /// Get the names of component gates
    pub fn component_gates(&self) -> &[String] {
        &self.component_gates
    }

    /// Get the number of gates that were fused
    pub fn num_components(&self) -> usize {
        self.component_gates.len()
    }
}

impl Gate for FusedGate {
    fn name(&self) -> &str {
        "FUSED"
    }

    fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    fn is_unitary(&self) -> bool {
        true
    }

    fn description(&self) -> String {
        format!("Fused gate: {}", self.component_gates.join(" → "))
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix.to_vec())
    }
}

impl fmt::Debug for FusedGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FusedGate")
            .field("qubits", &self.qubits)
            .field("components", &self.component_gates)
            .field("num_gates", &self.num_components())
            .finish()
    }
}

// ---------------------------------------------------------------------
// Legacy path: single-qubit chain fusion (unchanged algorithm)
// ---------------------------------------------------------------------

/// A fusion opportunity represents a sequence of gates that can be fused
#[derive(Debug)]
struct FusionChain {
    /// Indices of operations in the circuit
    operation_indices: Vec<usize>,
    /// The qubit this chain operates on
    qubit: QubitId,
}

/// Analyze circuit to find fusion opportunities
///
/// Identifies sequences of adjacent single-qubit gates operating on the same qubit
/// that have no intervening operations on that qubit.
///
/// **This is the original fusion algorithm, kept unmodified.** It is always
/// used when `circuit.num_qubits()` is below `FusionConfig::parallel_threshold_qubits`
/// — see [`fuse_single_qubit_gates`]'s dispatch. This guarantee is
/// structural, not a performance prediction: every currently-published
/// small-circuit benchmark takes this exact code path, unchanged.
///
/// # Arguments
/// * `circuit` - The circuit to analyze
/// * `config` - Fusion configuration
///
/// # Returns
/// Vector of fusion chains, where each chain is a sequence of operation indices
fn find_fusion_chains(circuit: &Circuit, config: &FusionConfig) -> Vec<FusionChain> {
    let num_qubits = circuit.num_qubits();
    let operations: Vec<_> = circuit.operations().collect();

    // Track the last operation index for each qubit
    let mut qubit_last_op: Vec<Option<usize>> = vec![None; num_qubits];

    // Track current fusion chains for each qubit
    let mut current_chains: Vec<Option<Vec<usize>>> = vec![None; num_qubits];

    // Completed fusion chains
    let mut fusion_chains = Vec::new();

    for (op_idx, op) in operations.iter().enumerate() {
        let qubits = op.qubits();

        // Multi-qubit gates break fusion chains on all their qubits
        if qubits.len() > 1 {
            for &qubit in qubits {
                let qubit_idx = qubit.index();
                if let Some(chain) = current_chains[qubit_idx].take() {
                    if chain.len() >= config.min_fusion_size {
                        fusion_chains.push(FusionChain {
                            operation_indices: chain,
                            qubit,
                        });
                    }
                }
                qubit_last_op[qubit_idx] = Some(op_idx);
            }
            continue;
        }

        // Single-qubit gate
        let qubit = qubits[0];
        let qubit_idx = qubit.index();

        // Check if this gate can be fused (must have a matrix representation)
        if op.gate().matrix().is_none() {
            // Non-unitary or unsupported gate, break the chain
            if let Some(chain) = current_chains[qubit_idx].take() {
                if chain.len() >= config.min_fusion_size {
                    fusion_chains.push(FusionChain {
                        operation_indices: chain,
                        qubit,
                    });
                }
            }
            qubit_last_op[qubit_idx] = Some(op_idx);
            continue;
        }

        // Add to or start fusion chain
        match current_chains[qubit_idx].as_mut() {
            Some(chain) => {
                // Check max fusion size
                if let Some(max_size) = config.max_fusion_size {
                    if chain.len() >= max_size {
                        // Finalize current chain and start new one
                        let old_chain = current_chains[qubit_idx].take().unwrap();
                        if old_chain.len() >= config.min_fusion_size {
                            fusion_chains.push(FusionChain {
                                operation_indices: old_chain,
                                qubit,
                            });
                        }
                        current_chains[qubit_idx] = Some(vec![op_idx]);
                    } else {
                        chain.push(op_idx);
                    }
                } else {
                    chain.push(op_idx);
                }
            },
            None => {
                current_chains[qubit_idx] = Some(vec![op_idx]);
            },
        }

        qubit_last_op[qubit_idx] = Some(op_idx);
    }

    // Finalize remaining chains
    for (qubit_idx, chain) in current_chains.into_iter().enumerate() {
        if let Some(chain) = chain {
            if chain.len() >= config.min_fusion_size {
                fusion_chains.push(FusionChain {
                    operation_indices: chain,
                    qubit: QubitId::new(qubit_idx),
                });
            }
        }
    }

    fusion_chains
}

/// Fuse a sequence of single-qubit gates into a single FusedGate
///
/// # Arguments
/// * `gates` - Sequence of gate operations to fuse (in application order)
/// * `config` - Fusion configuration
///
/// # Returns
/// A FusedGate representing the composition, or None if the result is identity
/// and identity elimination is enabled
fn fuse_gates(gates: &[&GateOp], config: &FusionConfig) -> Option<Arc<dyn Gate>> {
    assert!(!gates.is_empty(), "Cannot fuse empty gate sequence");
    assert!(gates.iter().all(|g| g.num_qubits() == 1), "All gates must be single-qubit");

    // Extract matrices from gates
    let matrices: Vec<[[Complex64; 2]; 2]> = gates
        .iter()
        .map(|op| {
            let matrix_vec = op
                .gate()
                .matrix()
                .expect("Gate must have matrix representation");
            // Convert from Vec<Complex64> to [[Complex64; 2]; 2]
            [
                [matrix_vec[0], matrix_vec[1]],
                [matrix_vec[2], matrix_vec[3]],
            ]
        })
        .collect();

    // Compose matrices: rightmost (first applied) gate is multiplied first
    // If gates are [G1, G2, G3], the result is G3 * G2 * G1
    let mut result = matrices[0];
    for matrix in &matrices[1..] {
        result = multiply_2x2(matrix, &result);
    }

    // Check if result is identity
    if config.eliminate_identity && is_identity(&result, config.identity_epsilon) {
        return None;
    }

    // Collect gate names for debugging
    let component_gates: SmallVec<[String; 4]> = gates
        .iter()
        .map(|op| op.gate().name().to_string())
        .collect();

    let qubit = gates[0].qubits()[0];
    let flat: FlatMatrix = FlatMatrix::from_slice(&[
        result[0][0],
        result[0][1],
        result[1][0],
        result[1][1],
    ]);
    let qubits: SmallVec<[QubitId; 4]> = SmallVec::from_slice(&[qubit]);

    Some(Arc::new(FusedGate::new(qubits, flat, component_gates)))
}

/// The legacy single-qubit-chain fusion pass body (verbatim algorithm from
/// before this module supported multi-qubit blocks). Reconstructs the
/// circuit by replacing each fusion chain with one [`FusedGate`].
fn fuse_single_qubit_chains(circuit: &Circuit, config: &FusionConfig) -> Result<Circuit> {
    // Find all fusion opportunities
    let fusion_chains = find_fusion_chains(circuit, config);

    // If no fusion opportunities, return original circuit
    if fusion_chains.is_empty() {
        return Ok(circuit.clone());
    }

    // Build set of operation indices that will be replaced by fused gates
    let fused_indices: AHashSet<usize> = fusion_chains
        .iter()
        .flat_map(|chain| chain.operation_indices.iter().copied())
        .collect();

    // Build the optimized circuit
    let mut optimized = Circuit::with_capacity(circuit.num_qubits(), circuit.len());
    let operations: Vec<_> = circuit.operations().collect();

    // Map from fusion chain start index to fused gate
    let mut fused_gates: AHashMap<usize, (Arc<dyn Gate>, QubitId)> = AHashMap::new();

    for chain in &fusion_chains {
        let chain_ops: Vec<&GateOp> = chain
            .operation_indices
            .iter()
            .map(|&idx| operations[idx])
            .collect();

        if let Some(fused_gate) = fuse_gates(&chain_ops, config) {
            let start_idx = chain.operation_indices[0];
            fused_gates.insert(start_idx, (fused_gate, chain.qubit));
        }
    }

    // Reconstruct circuit
    for (idx, op) in operations.iter().enumerate() {
        if let Some((fused_gate, qubit)) = fused_gates.get(&idx) {
            // Insert fused gate at the position of the first gate in the chain
            optimized.add_gate(Arc::clone(fused_gate), &[*qubit])?;
        } else if !fused_indices.contains(&idx) {
            // Keep original operation if it's not part of any fusion
            optimized.add_gate(Arc::clone(op.gate()), op.qubits())?;
        }
        // Skip operations that were fused (and are not the first in their chain)
    }

    Ok(optimized)
}

// ---------------------------------------------------------------------
// New path: greedy width-bounded multi-qubit block fusion
// ---------------------------------------------------------------------

/// A group of operations fused into a single multi-qubit block.
///
/// Unlike [`FusionChain`] (single qubit, used by [`find_fusion_chains`]), a
/// block can span up to `FusionConfig::max_block_width` qubits and is found
/// by [`find_fusion_blocks`], which is only ever invoked when
/// `circuit.num_qubits() >= FusionConfig::parallel_threshold_qubits` and
/// `FusionConfig::max_block_width > 1` — see [`fuse_single_qubit_gates`]'s
/// dispatch. `pub(crate)` so [`crate::fusion_cache`] can store block
/// *structure* (never matrix values — see that module's docs for why this
/// is exact, not approximate, to cache across circuits with the same shape
/// but different parameters).
#[derive(Debug, Clone)]
pub(crate) struct FusionBlock {
    /// Indices of operations in the circuit, in application order.
    pub(crate) operation_indices: Vec<usize>,
    /// Canonical qubit order for this block (ascending qubit index), fixed
    /// once the block closes. Every component gate's matrix is embedded
    /// into local positions within this order — see
    /// [`compose_block_matrix`] — and the executor must receive qubits in
    /// exactly this order, never re-sorted.
    pub(crate) qubits: SmallVec<[QubitId; 4]>,
}

/// Mutable accumulator for an in-progress (not yet closed) fusion block.
struct FusionBlockBuilder {
    operation_indices: Vec<usize>,
    qubit_set: AHashSet<QubitId>,
}

/// Close block `block_idx` if still open: release its qubit ownership and,
/// if it met `min_size` (component gate count), push it to `finished` with
/// qubits sorted into canonical (ascending index) order. No-op if the block
/// was already closed.
fn close_block(
    blocks: &mut [Option<FusionBlockBuilder>],
    qubit_owner: &mut [Option<usize>],
    finished: &mut Vec<FusionBlock>,
    block_idx: usize,
    min_size: usize,
) {
    let Some(b) = blocks[block_idx].take() else {
        return;
    };
    for &q in &b.qubit_set {
        if qubit_owner[q.index()] == Some(block_idx) {
            qubit_owner[q.index()] = None;
        }
    }
    if b.operation_indices.len() >= min_size {
        let mut qubits: SmallVec<[QubitId; 4]> = b.qubit_set.into_iter().collect();
        qubits.sort_by_key(|q| q.index());
        finished.push(FusionBlock {
            operation_indices: b.operation_indices,
            qubits,
        });
    }
}

/// Analyze a circuit to find multi-qubit fusion blocks using a greedy,
/// single-forward-pass frontier merge — deliberately **not** a circuit-wide
/// fusion graph with a per-edge cost model and shortest-path schedule
/// selection (see the module-level docs: that specific mechanism is what a
/// granted patent found during this feature's diligence claims, and this
/// design avoids it by construction).
///
/// For each operation, in this fixed priority order (no lookahead, no cost
/// comparison):
/// 1. No matrix (e.g. measurement) → close every block it touches. Fusion
///    never crosses this boundary (a separate, on-point patent covers
///    fusing measurement gates specifically; this design never does that).
/// 2. Exactly one open block overlaps its qubits and merging stays within
///    `max_block_width` and `max_fusion_size` → merge into that block.
/// 3. Otherwise, close any block(s) it overlaps (bridging ≥2 blocks closes
///    all of them; exceeding the width/size cap closes the one it would
///    have joined) and open a fresh block for this operation (if its own
///    qubit count fits within `max_block_width`; if not, it is left
///    standalone).
///
/// Only ever called when `config.max_block_width > 1`; the single-qubit
/// path ([`find_fusion_chains`]) is untouched and used otherwise.
pub(crate) fn find_fusion_blocks(circuit: &Circuit, config: &FusionConfig) -> Vec<FusionBlock> {
    let max_width = config.max_block_width.max(1);
    let num_qubits = circuit.num_qubits();
    let operations: Vec<_> = circuit.operations().collect();

    // Which open block (index into `blocks`) currently owns each qubit's wire.
    let mut qubit_owner: Vec<Option<usize>> = vec![None; num_qubits];
    let mut blocks: Vec<Option<FusionBlockBuilder>> = Vec::new();
    let mut finished: Vec<FusionBlock> = Vec::new();

    for (op_idx, op) in operations.iter().enumerate() {
        let qubits = op.qubits();

        if op.gate().matrix().is_none() {
            let touched: AHashSet<usize> =
                qubits.iter().filter_map(|q| qubit_owner[q.index()]).collect();
            for block_idx in touched {
                close_block(&mut blocks, &mut qubit_owner, &mut finished, block_idx, config.min_fusion_size);
            }
            continue;
        }

        let touched: AHashSet<usize> =
            qubits.iter().filter_map(|q| qubit_owner[q.index()]).collect();

        let mut merged = false;
        if touched.len() == 1 {
            let block_idx = *touched.iter().next().unwrap();
            let builder = blocks[block_idx].as_ref().unwrap();

            let at_gate_cap = config
                .max_fusion_size
                .is_some_and(|max| builder.operation_indices.len() >= max);

            let would_grow_to = {
                let mut merged_set: AHashSet<QubitId> = builder.qubit_set.clone();
                merged_set.extend(qubits.iter().copied());
                merged_set.len()
            };

            if !at_gate_cap && would_grow_to <= max_width {
                let b = blocks[block_idx].as_mut().unwrap();
                b.operation_indices.push(op_idx);
                for &q in qubits {
                    b.qubit_set.insert(q);
                    qubit_owner[q.index()] = Some(block_idx);
                }
                merged = true;
            } else {
                close_block(&mut blocks, &mut qubit_owner, &mut finished, block_idx, config.min_fusion_size);
            }
        } else if touched.len() > 1 {
            for block_idx in touched {
                close_block(&mut blocks, &mut qubit_owner, &mut finished, block_idx, config.min_fusion_size);
            }
        }

        if !merged && qubits.len() <= max_width {
            let block_idx = blocks.len();
            let mut qubit_set = AHashSet::new();
            for &q in qubits {
                qubit_set.insert(q);
                qubit_owner[q.index()] = Some(block_idx);
            }
            blocks.push(Some(FusionBlockBuilder {
                operation_indices: vec![op_idx],
                qubit_set,
            }));
        }
        // else (!merged && qubits.len() > max_width): the gate itself is
        // wider than max_block_width — left standalone, never fused. Any
        // block that owned its qubits was already closed above.
    }

    for block_idx in 0..blocks.len() {
        close_block(&mut blocks, &mut qubit_owner, &mut finished, block_idx, config.min_fusion_size);
    }

    finished
}

/// Compose a fusion block's component gate matrices into a single flat
/// `2^k x 2^k` unitary.
///
/// # Bit-ordering convention (the correctness-critical part)
///
/// The dense kernels this crate's fused gates ultimately run through
/// (`simq-sim`'s `two_qubit`/`three_qubit` kernels) use an **argument-order,
/// MSB-first** convention: for `apply_three_qubit_dense(matrix, qubit1,
/// qubit2, qubit3, ...)`, the matrix's local basis index is `m =
/// (bit(qubit1)<<2) | (bit(qubit2)<<1) | bit(qubit3)` — the *first* argument
/// is the *most significant* bit.
///
/// `simq_gates::matrix_ops::embed_gate_matrix_vec`, which this function uses
/// to lift each component gate's small matrix into the block's shared
/// basis, instead uses a **position-index, LSB-first** convention: local
/// position `p` maps to bit `p` (position `0` is the *least significant*
/// bit) — documented and tested in `simq-gates/src/matrix_ops.rs`.
///
/// These are reversed orderings of each other. Since this crate always
/// passes `block.qubits` to the executor/kernel in unchanged canonical
/// (ascending) order — `qubit1 = block.qubits[0]`, ..., `qubitK =
/// block.qubits[K-1]` — reconciling the two conventions requires mapping
/// `block.qubits[i]` to **local position `width - 1 - i`** (not `i`) when
/// building the `qubit_indices` argument to `embed_gate_matrix_vec`. This
/// has been verified analytically (embedding a gate whose own qubits equal
/// `block.qubits` in the same order reduces to the identity embedding,
/// reproducing the gate's original matrix unchanged, as it must) and is
/// additionally covered by fused-vs-sequential-execution correctness tests
/// below that exercise multiple qubit orderings, per this crate's own
/// "don't trust the derivation alone" testing norm.
fn compose_block_matrix(
    block: &FusionBlock,
    operations: &[&GateOp],
    config: &FusionConfig,
) -> Option<Arc<dyn Gate>> {
    let width = block.qubits.len();
    let dim = 1usize << width;

    let mut result: FlatMatrix = identity_flat(dim);
    for &op_idx in &block.operation_indices {
        let op = operations[op_idx];
        let gate_matrix = op.gate().matrix()?;
        let local_qubits: SmallVec<[usize; 4]> = op
            .qubits()
            .iter()
            .map(|q| {
                let i = block
                    .qubits
                    .iter()
                    .position(|bq| bq == q)
                    .expect("component gate's qubit must belong to its block");
                width - 1 - i
            })
            .collect();
        let embedded = embed_gate_matrix_vec(&gate_matrix, width, &local_qubits);
        result = multiply_square(&embedded, &result, dim);
    }

    if config.eliminate_identity && is_identity_flat(&result, dim, config.identity_epsilon) {
        return None;
    }

    let component_gates: SmallVec<[String; 4]> = block
        .operation_indices
        .iter()
        .map(|&idx| operations[idx].gate().name().to_string())
        .collect();

    Some(Arc::new(FusedGate::new(
        block.qubits.clone(),
        result,
        component_gates,
    )))
}

/// Multi-qubit block fusion pass body. Reconstructs the circuit by
/// replacing each fusion block with one [`FusedGate`]. Only called when
/// [`fuse_single_qubit_gates`]'s dispatch selects the multi-qubit path.
///
/// If `cache` is provided, consults it for this circuit's block
/// *structure* (never matrix values) keyed by
/// [`crate::cache::CircuitFingerprint`] before re-running
/// [`find_fusion_blocks`] — see `fusion_cache` module docs for why this is
/// exact, not approximate, across repeated compiles of a same-shaped
/// circuit with different gate parameters (e.g. a VQE/QAOA outer loop).
fn fuse_multi_qubit_blocks(
    circuit: &Circuit,
    config: &FusionConfig,
    cache: Option<&crate::fusion_cache::FusionStructureCache>,
) -> Result<Circuit> {
    let blocks: Arc<Vec<FusionBlock>> = match cache {
        Some(cache) => {
            let fingerprint = crate::cache::CircuitFingerprint::compute(circuit);
            if let Some(cached) = cache.get(fingerprint) {
                cached
            } else {
                let computed = find_fusion_blocks(circuit, config);
                cache.insert(fingerprint, computed.clone());
                Arc::new(computed)
            }
        },
        None => Arc::new(find_fusion_blocks(circuit, config)),
    };

    if blocks.is_empty() {
        return Ok(circuit.clone());
    }

    let operations: Vec<&GateOp> = circuit.operations().collect();

    let fused_indices: AHashSet<usize> = blocks
        .iter()
        .flat_map(|b| b.operation_indices.iter().copied())
        .collect();

    let mut fused_gates: AHashMap<usize, (Arc<dyn Gate>, SmallVec<[QubitId; 4]>)> =
        AHashMap::new();
    for block in blocks.iter() {
        if let Some(fused) = compose_block_matrix(block, &operations, config) {
            let start_idx = block.operation_indices[0];
            fused_gates.insert(start_idx, (fused, block.qubits.clone()));
        }
    }

    let mut optimized = Circuit::with_capacity(circuit.num_qubits(), circuit.len());
    for (idx, op) in operations.iter().enumerate() {
        if let Some((fused_gate, qubits)) = fused_gates.get(&idx) {
            optimized.add_gate(Arc::clone(fused_gate), qubits.as_slice())?;
        } else if !fused_indices.contains(&idx) {
            optimized.add_gate(Arc::clone(op.gate()), op.qubits())?;
        }
    }

    Ok(optimized)
}

// ---------------------------------------------------------------------
// Public entry point: dispatch between the two paths
// ---------------------------------------------------------------------

/// Apply gate fusion optimization to a quantum circuit
///
/// Dispatches between the legacy single-qubit-chain algorithm and the
/// multi-qubit block algorithm based on `circuit.num_qubits()` and
/// `config.max_block_width` — see the module-level docs and
/// [`FusionConfig::parallel_threshold_qubits`]. Below the threshold (or
/// with `max_block_width <= 1`), this is **exactly** the original
/// single-qubit fusion pass, unmodified: the new multi-qubit code is not
/// merely equivalent for small circuits, it is never reached.
///
/// # Arguments
/// * `circuit` - The input circuit to optimize
/// * `config` - Optional fusion configuration (uses default if None)
///
/// # Returns
/// A new optimized circuit with fused gates
///
/// # Errors
/// Returns error if circuit construction fails (should be rare)
///
/// # Example
/// ```ignore
/// use simq_core::Circuit;
/// use simq_compiler::fusion::fuse_single_qubit_gates;
///
/// let circuit = /* ... build circuit ... */;
/// let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();
/// ```
pub fn fuse_single_qubit_gates(circuit: &Circuit, config: Option<FusionConfig>) -> Result<Circuit> {
    fuse_gates_with_cache(circuit, config, None)
}

/// Same as [`fuse_single_qubit_gates`], but consults an optional
/// [`crate::fusion_cache::FusionStructureCache`] for the multi-qubit block
/// path (ignored entirely below `parallel_threshold_qubits`, exactly like
/// the rest of the multi-qubit machinery — see the dispatch below).
///
/// This is the entry point [`crate::passes::GateFusion::with_cache`] wires
/// up to, which in turn is what `simq-sim`'s `Simulator` uses to persist
/// fusion block structure across repeated compiles of a same-shaped
/// circuit (e.g. a VQE/QAOA optimizer's outer loop, which otherwise
/// reconstructs a fresh `Compiler` — and re-derives fusion from scratch —
/// on every single iteration).
pub fn fuse_gates_with_cache(
    circuit: &Circuit,
    config: Option<FusionConfig>,
    cache: Option<&crate::fusion_cache::FusionStructureCache>,
) -> Result<Circuit> {
    let config = config.unwrap_or_default();

    if circuit.num_qubits() < config.parallel_threshold_qubits || config.max_block_width <= 1 {
        return fuse_single_qubit_chains(circuit, &config);
    }

    fuse_multi_qubit_blocks(circuit, &config, cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_gates::standard::{Hadamard, PauliX, PauliY, PauliZ, SGate, TGate};

    fn create_test_circuit_simple() -> Circuit {
        let mut circuit = Circuit::new(2);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);

        // Three gates on q0 that should be fused
        circuit
            .add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0])
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0])
            .unwrap();

        // One gate on q1 (not fused, too small)
        circuit
            .add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[q1])
            .unwrap();

        circuit
    }

    fn create_test_circuit_multiple_qubits() -> Circuit {
        let mut circuit = Circuit::new(3);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        let q2 = QubitId::new(2);

        // Chain on q0
        circuit
            .add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0])
            .unwrap();

        // Chain on q1
        circuit
            .add_gate(Arc::new(PauliY) as Arc<dyn Gate>, &[q1])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[q1])
            .unwrap();

        // Single gate on q2
        circuit
            .add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q2])
            .unwrap();

        // More on q0
        circuit
            .add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[q0])
            .unwrap();

        circuit
    }

    #[test]
    fn test_find_fusion_chains_simple() {
        let circuit = create_test_circuit_simple();
        let config = FusionConfig::default();
        let chains = find_fusion_chains(&circuit, &config);

        // Should find one chain on q0 with 3 gates
        assert_eq!(chains.len(), 1);
        assert_eq!(chains[0].operation_indices.len(), 3);
        assert_eq!(chains[0].qubit, QubitId::new(0));
    }

    #[test]
    fn test_find_fusion_chains_multiple() {
        let circuit = create_test_circuit_multiple_qubits();
        let config = FusionConfig::default();
        let chains = find_fusion_chains(&circuit, &config);

        // Should find chains on q0 and q1 (2 gates each)
        // Single gates on q2 and later q0 are too small to fuse
        assert_eq!(chains.len(), 2);
    }

    #[test]
    fn test_fuse_gates_hadamard_x_hadamard() {
        let q0 = QubitId::new(0);
        let h: Arc<dyn Gate> = Arc::new(Hadamard);
        let x: Arc<dyn Gate> = Arc::new(PauliX);

        let op1 = GateOp::new(Arc::clone(&h), &[q0]).unwrap();
        let op2 = GateOp::new(Arc::clone(&x), &[q0]).unwrap();
        let op3 = GateOp::new(Arc::clone(&h), &[q0]).unwrap();

        let gates = vec![&op1, &op2, &op3];
        let config = FusionConfig::default();

        let fused = fuse_gates(&gates, &config);
        assert!(fused.is_some());

        let fused_gate = fused.unwrap();
        assert_eq!(fused_gate.name(), "FUSED");
        assert_eq!(fused_gate.num_qubits(), 1);
    }

    #[test]
    fn test_fuse_gates_self_inverse() {
        // X * X = I, should be eliminated
        let q0 = QubitId::new(0);
        let x: Arc<dyn Gate> = Arc::new(PauliX);

        let op1 = GateOp::new(Arc::clone(&x), &[q0]).unwrap();
        let op2 = GateOp::new(Arc::clone(&x), &[q0]).unwrap();

        let gates = vec![&op1, &op2];
        let config = FusionConfig::default();

        let fused = fuse_gates(&gates, &config);
        // Should return None because result is identity
        assert!(fused.is_none());
    }

    #[test]
    fn test_fuse_gates_s_s_equals_z() {
        // S * S = Z (not identity)
        let q0 = QubitId::new(0);
        let s: Arc<dyn Gate> = Arc::new(SGate);

        let op1 = GateOp::new(Arc::clone(&s), &[q0]).unwrap();
        let op2 = GateOp::new(Arc::clone(&s), &[q0]).unwrap();

        let gates = vec![&op1, &op2];
        let config = FusionConfig::default();

        let fused = fuse_gates(&gates, &config);
        assert!(fused.is_some());

        // Verify the result is approximately Z
        let fused_gate = fused.unwrap();
        let matrix = fused_gate.matrix().unwrap();

        // Z matrix in flattened form: [1, 0, 0, -1]
        assert!((matrix[0].re - 1.0).abs() < 1e-10);
        assert!(matrix[0].im.abs() < 1e-10);
        assert!(matrix[1].norm() < 1e-10);
        assert!(matrix[2].norm() < 1e-10);
        assert!((matrix[3].re - (-1.0)).abs() < 1e-10);
        assert!(matrix[3].im.abs() < 1e-10);
    }

    #[test]
    fn test_fuse_single_qubit_gates_simple() {
        let circuit = create_test_circuit_simple();
        let original_len = circuit.len();

        let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();

        // Should have fewer operations (3 gates fused into 1, plus 1 unfused)
        assert!(optimized.len() < original_len);
        assert_eq!(optimized.len(), 2); // 1 fused gate on q0, 1 gate on q1
    }

    #[test]
    fn test_fuse_single_qubit_gates_preserves_qubits() {
        let circuit = create_test_circuit_simple();
        let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();

        assert_eq!(optimized.num_qubits(), circuit.num_qubits());
    }

    #[test]
    fn test_fusion_config_min_size() {
        let circuit = create_test_circuit_simple();

        let config = FusionConfig {
            min_fusion_size: 4, // Require at least 4 gates to fuse
            ..Default::default()
        };

        let optimized = fuse_single_qubit_gates(&circuit, Some(config)).unwrap();

        // No fusion should occur (chain has only 3 gates)
        assert_eq!(optimized.len(), circuit.len());
    }

    #[test]
    fn test_fusion_config_no_identity_elimination() {
        // X * X = I
        let mut circuit = Circuit::new(1);
        let q0 = QubitId::new(0);
        let x: Arc<dyn Gate> = Arc::new(PauliX);
        circuit.add_gate(Arc::clone(&x), &[q0]).unwrap();
        circuit.add_gate(Arc::clone(&x), &[q0]).unwrap();

        let config = FusionConfig {
            eliminate_identity: false,
            ..Default::default()
        };

        let optimized = fuse_single_qubit_gates(&circuit, Some(config)).unwrap();

        // Should have 1 fused gate (identity not eliminated)
        assert_eq!(optimized.len(), 1);
    }

    #[test]
    fn test_fusion_config_max_size() {
        let mut circuit = Circuit::new(1);
        let q0 = QubitId::new(0);

        // Add 5 T gates
        let t: Arc<dyn Gate> = Arc::new(TGate);
        for _ in 0..5 {
            circuit.add_gate(Arc::clone(&t), &[q0]).unwrap();
        }

        let config = FusionConfig {
            max_fusion_size: Some(3),
            ..Default::default()
        };

        let optimized = fuse_single_qubit_gates(&circuit, Some(config)).unwrap();

        // Should create 2 fused gates: one with 3 gates, one with 2 gates
        assert_eq!(optimized.len(), 2);
    }

    #[test]
    fn test_fused_gate_description() {
        let q0 = QubitId::new(0);
        let h: Arc<dyn Gate> = Arc::new(Hadamard);
        let x: Arc<dyn Gate> = Arc::new(PauliX);

        let op1 = GateOp::new(Arc::clone(&h), &[q0]).unwrap();
        let op2 = GateOp::new(Arc::clone(&x), &[q0]).unwrap();

        let gates = vec![&op1, &op2];
        let config = FusionConfig::default();

        let fused = fuse_gates(&gates, &config).unwrap();
        let desc = fused.description();

        assert!(desc.contains("H"));
        assert!(desc.contains("X"));
        assert!(desc.contains("→")); // Arrow separator
    }

    #[test]
    fn test_fused_gate_accessors() {
        use num_complex::Complex64;
        use smallvec::smallvec;
        let matrix: FlatMatrix = smallvec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let names: SmallVec<[String; 4]> = smallvec!["H".to_string(), "X".to_string()];
        let qubits: SmallVec<[QubitId; 4]> = smallvec![QubitId::new(0)];
        let gate = FusedGate::new(qubits, matrix, names);

        assert_eq!(gate.component_gates(), &["H", "X"]);
        assert_eq!(gate.num_components(), 2);
        assert!(gate.is_unitary());
        assert_eq!(gate.num_qubits(), 1);
        assert_eq!(gate.name(), "FUSED");
        assert_eq!(gate.qubits(), &[QubitId::new(0)]);

        // Gate trait matrix() method (via trait object to avoid field name collision)
        let gate_dyn: &dyn Gate = &gate;
        let mat_vec = gate_dyn.matrix().unwrap();
        assert_eq!(mat_vec.len(), 4);
    }

    #[test]
    fn test_fuse_single_qubit_gates_no_opportunities() {
        // Circuit with a single gate on one qubit — too short to fuse (min_fusion_size=2)
        let mut circuit = Circuit::new(1);
        circuit
            .add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[QubitId::new(0)])
            .unwrap();

        let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();
        assert_eq!(optimized.len(), circuit.len()); // unchanged
    }

    #[test]
    fn test_fused_gate_debug() {
        use num_complex::Complex64;
        use smallvec::smallvec;
        let matrix: FlatMatrix = smallvec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let names: SmallVec<[String; 4]> = smallvec!["H".to_string()];
        let qubits: SmallVec<[QubitId; 4]> = smallvec![QubitId::new(0)];
        let gate = FusedGate::new(qubits, matrix, names);
        let dbg = format!("{:?}", gate);
        assert!(dbg.contains("FusedGate"));
    }

    // -----------------------------------------------------------------------
    // New tests for previously uncovered lines
    // -----------------------------------------------------------------------

    /// Lines 104-105 (original): FusedGate inherent method `matrix()` (field accessor).
    /// The existing test_fused_gate_accessors called the Gate TRAIT method via
    /// a trait object.  This test calls the inherent struct method directly.
    #[test]
    fn test_fused_gate_matrix_field_accessor() {
        use num_complex::Complex64;
        use smallvec::smallvec;
        let matrix: FlatMatrix = smallvec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ];
        let names: SmallVec<[String; 4]> = smallvec!["Z".to_string()];
        let qubits: SmallVec<[QubitId; 4]> = smallvec![QubitId::new(0)];
        let gate = FusedGate::new(qubits, matrix, names);
        // Call the inherent method (not the Gate trait method)
        let m: &[Complex64] = gate.matrix();
        assert!((m[0].re - 1.0).abs() < 1e-10);
        assert!((m[3].re - (-1.0)).abs() < 1e-10);
    }

    /// A gate with matrix()=None in the middle of a fuseable chain
    /// terminates the chain and pushes it to fusion_chains.
    /// Setup: two H gates (forming a chain of length 2 ≥ min_fusion_size=2) on q0,
    /// followed by a no-matrix gate on q0 → chain is pushed, then fused into one.
    #[test]
    fn test_fuse_single_qubit_gates_no_matrix_gate_breaks_chain() {
        /// A gate that returns no matrix (breaks fusion chains).
        #[derive(Debug)]
        struct LocalNoMatrixGate;

        impl Gate for LocalNoMatrixGate {
            fn name(&self) -> &str {
                "LocalNoMatrix"
            }

            fn num_qubits(&self) -> usize {
                1
            }

            fn matrix(&self) -> Option<Vec<Complex64>> {
                None
            }
        }

        let q0 = QubitId::new(0);
        let mut circuit = Circuit::new(1);
        // Two H gates on q0 → forms a chain of length 2
        circuit
            .add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0])
            .unwrap();
        // No-matrix gate on q0 → breaks the chain; the chain of length 2 is pushed
        circuit
            .add_gate(Arc::new(LocalNoMatrixGate) as Arc<dyn Gate>, &[q0])
            .unwrap();

        // fuse_single_qubit_gates should fuse the first two gates and leave the no-matrix one
        let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();
        // Original: 3 operations.  After fusion: 1 fused gate + 1 no-matrix gate = 2.
        assert_eq!(
            optimized.len(),
            2,
            "Expected 2 operations after fusion: fused(H,X) + LocalNoMatrix"
        );
    }

    // -----------------------------------------------------------------------
    // §6-0 / §8: the <16q structural non-regression guarantee, made checkable
    // -----------------------------------------------------------------------

    /// Below `parallel_threshold_qubits`, `fuse_single_qubit_gates` must
    /// produce byte-identical output to calling `fuse_single_qubit_chains`
    /// (the legacy path) directly — i.e. the new multi-qubit machinery is
    /// provably never reached, not just "equivalent." This is the check
    /// that makes the "<16q is unaffected" claim in the implementation plan
    /// checkable rather than argued.
    #[test]
    fn test_below_threshold_dispatches_to_legacy_path_exactly() {
        // 16 qubits: matches BENCHMARKS.md's largest published row, and is
        // below the default parallel_threshold_qubits (18).
        let mut circuit = Circuit::new(16);
        for q in 0..16 {
            circuit
                .add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[QubitId::new(q)])
                .unwrap();
            circuit
                .add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[QubitId::new(q)])
                .unwrap();
        }
        // A CNOT chain, exactly like the benchmarked VQE/QAOA shapes, to
        // confirm multi-qubit gates don't change the dispatch decision.
        for q in 0..15 {
            circuit
                .add_gate(
                    Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>,
                    &[QubitId::new(q), QubitId::new(q + 1)],
                )
                .unwrap();
        }

        let config = FusionConfig::default(); // max_block_width: 3, threshold: 18
        assert!(circuit.num_qubits() < config.parallel_threshold_qubits);

        let via_dispatch = fuse_single_qubit_gates(&circuit, Some(config.clone())).unwrap();
        let via_legacy_directly = fuse_single_qubit_chains(&circuit, &config).unwrap();

        assert_eq!(via_dispatch.len(), via_legacy_directly.len());
        for (a, b) in via_dispatch.operations().zip(via_legacy_directly.operations()) {
            assert_eq!(a.gate().name(), b.gate().name());
            assert_eq!(a.qubits(), b.qubits());
        }
    }

    /// At/above `parallel_threshold_qubits`, the multi-qubit block path
    /// should actually be reached (a CNOT-adjacent chain should fuse across
    /// the two-qubit boundary, which the legacy path never does).
    #[test]
    fn test_at_threshold_reaches_multi_qubit_block_path() {
        let mut circuit = Circuit::new(18);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit
            .add_gate(
                Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>,
                &[q0, q1],
            )
            .unwrap();
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();

        let config = FusionConfig::default();
        assert!(circuit.num_qubits() >= config.parallel_threshold_qubits);

        let optimized = fuse_single_qubit_gates(&circuit, Some(config)).unwrap();
        // The legacy path could never fuse across the CNOT; the block path
        // should collapse all three gates into one 2-qubit FusedGate.
        assert_eq!(optimized.len(), 1);
        let op = optimized.operations().next().unwrap();
        assert_eq!(op.gate().name(), "FUSED");
        assert_eq!(op.gate().num_qubits(), 2);
    }

    /// Explicit `max_block_width: 1` forces the legacy path even at/above
    /// the qubit threshold (documented override behavior).
    #[test]
    fn test_explicit_width_one_forces_legacy_path_above_threshold() {
        let mut circuit = Circuit::new(18);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit
            .add_gate(
                Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>,
                &[q0, q1],
            )
            .unwrap();
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();

        let config = FusionConfig {
            max_block_width: 1,
            ..Default::default()
        };
        let optimized = fuse_single_qubit_gates(&circuit, Some(config)).unwrap();
        // Legacy behavior: CNOT breaks fusion, H gates on either side are
        // singletons (below min_fusion_size), so nothing fuses at all.
        assert_eq!(optimized.len(), 3);
    }

    // -----------------------------------------------------------------------
    // §5: multi-qubit block fusion correctness
    // -----------------------------------------------------------------------

    fn wide_circuit_config() -> FusionConfig {
        // Force the block path on small test circuits without needing an
        // 18-qubit circuit in every test.
        FusionConfig {
            parallel_threshold_qubits: 0,
            ..Default::default()
        }
    }

    #[test]
    fn test_block_fusion_h_cnot_h_fuses_across_two_qubits() {
        let mut circuit = Circuit::new(2);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit
            .add_gate(
                Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>,
                &[q0, q1],
            )
            .unwrap();
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();

        let optimized = fuse_single_qubit_gates(&circuit, Some(wide_circuit_config())).unwrap();
        assert_eq!(optimized.len(), 1);
        let op = optimized.operations().next().unwrap();
        assert_eq!(op.gate().name(), "FUSED");
        assert_eq!(op.gate().num_qubits(), 2);
        assert_eq!(op.qubits(), &[q0, q1]);
    }

    #[test]
    fn test_block_fusion_respects_max_width() {
        // H(q0)-CNOT(q0,q1)-CNOT(q1,q2): with max_block_width=2, the second
        // CNOT would grow the block to 3 qubits, so it must close the first
        // block and start a new one instead of growing unbounded.
        let mut circuit = Circuit::new(3);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        let q2 = QubitId::new(2);
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit
            .add_gate(Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>, &[q0, q1])
            .unwrap();
        circuit
            .add_gate(Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>, &[q1, q2])
            .unwrap();

        let config = FusionConfig {
            parallel_threshold_qubits: 0,
            max_block_width: 2,
            ..Default::default()
        };
        let blocks = find_fusion_blocks(&circuit, &config);
        for block in &blocks {
            assert!(block.qubits.len() <= 2);
        }
        // Two separate blocks: {q0,q1} (H+CNOT) and the lone CNOT(q1,q2)
        // (dropped for being below min_fusion_size=2, so only 1 finished block).
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].qubits.as_slice(), &[q0, q1]);
    }

    #[test]
    fn test_block_fusion_bridging_gate_closes_prior_blocks() {
        // Two independent single-qubit chains on q0 and q1, then a CNOT
        // bridging them — the bridge must close both prior blocks rather
        // than silently merging or corrupting state.
        let mut circuit = Circuit::new(2);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit.add_gate(Arc::new(PauliY) as Arc<dyn Gate>, &[q1]).unwrap();
        circuit.add_gate(Arc::new(PauliZ) as Arc<dyn Gate>, &[q1]).unwrap();
        circuit
            .add_gate(Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>, &[q0, q1])
            .unwrap();

        let config = wide_circuit_config();
        let blocks = find_fusion_blocks(&circuit, &config);
        // The two 2-gate chains each close as their own block (both meet
        // min_fusion_size=2); the trailing lone CNOT is dropped (size 1).
        assert_eq!(blocks.len(), 2);
        for block in &blocks {
            assert_eq!(block.operation_indices.len(), 2);
        }
    }

    #[test]
    fn test_block_fusion_measurement_still_breaks_chain() {
        /// Regression guard for the deliberate non-goal: fusion must never
        /// extend across a measurement boundary (see module docs / IP
        /// diligence notes).
        #[derive(Debug)]
        struct MockMeasure;
        impl Gate for MockMeasure {
            fn name(&self) -> &str {
                "Measure"
            }
            fn num_qubits(&self) -> usize {
                1
            }
            fn matrix(&self) -> Option<Vec<Complex64>> {
                None
            }
        }

        let mut circuit = Circuit::new(2);
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit.add_gate(Arc::new(MockMeasure) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
        circuit.add_gate(Arc::new(PauliY) as Arc<dyn Gate>, &[q0]).unwrap();
        let _ = q1;

        let optimized = fuse_single_qubit_gates(&circuit, Some(wide_circuit_config())).unwrap();
        // Two 2-gate chains on either side of the measurement, fused
        // separately; the measurement itself passes through unchanged.
        assert_eq!(optimized.len(), 3); // fused(H,X) + Measure + fused(H,Y)
        let names: Vec<&str> = optimized.operations().map(|op| op.gate().name()).collect();
        assert_eq!(names, vec!["FUSED", "Measure", "FUSED"]);
    }

    /// The correctness-critical test: fused vs. sequential (unfused)
    /// execution of a small VQE-shaped circuit must agree numerically, for
    /// multiple qubit orderings, to catch the bit-ordering risk documented
    /// on `compose_block_matrix`.
    #[test]
    fn test_block_fusion_matches_sequential_application_multiple_orderings() {
        // simq-compiler has no dependency on simq-sim (no execution engine
        // here), so correctness is checked via the full circuit unitary
        // (simq_gates::matrix_ops::circuit_matrix, already used by this
        // crate's dependencies) rather than by running a simulator.
        use simq_gates::matrix_ops::circuit_matrix;
        use simq_gates::standard::{RotationY, RotationZ};

        // (q0, q1, q2) permutations to exercise ascending / descending /
        // interleaved orderings against the block's canonical (ascending)
        // qubit order.
        let orderings: [[usize; 3]; 3] = [[0, 1, 2], [2, 1, 0], [1, 0, 2]];

        for perm in orderings {
            let [a, b, c] = perm;
            let (qa, qb, qc) = (QubitId::new(a), QubitId::new(b), QubitId::new(c));

            let build = || -> Circuit {
                let mut circuit = Circuit::new(3);
                circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[qa]).unwrap();
                circuit
                    .add_gate(Arc::new(RotationY::new(0.7)) as Arc<dyn Gate>, &[qa])
                    .unwrap();
                circuit
                    .add_gate(Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>, &[qa, qb])
                    .unwrap();
                circuit
                    .add_gate(Arc::new(RotationZ::new(1.3)) as Arc<dyn Gate>, &[qb])
                    .unwrap();
                circuit
                    .add_gate(Arc::new(simq_gates::standard::CNot) as Arc<dyn Gate>, &[qb, qc])
                    .unwrap();
                circuit
            };

            let unfused = build();
            let fused = fuse_single_qubit_gates(&unfused, Some(wide_circuit_config())).unwrap();

            // Sanity: fusion actually did something (collapsed multiple ops).
            assert!(fused.len() < unfused.len());

            let unfused_matrix = circuit_matrix(&unfused).expect("unfused circuit matrix");
            let fused_matrix = circuit_matrix(&fused).expect("fused circuit matrix");

            assert_eq!(unfused_matrix.len(), fused_matrix.len());
            for (u, f) in unfused_matrix.iter().zip(fused_matrix.iter()) {
                assert!(
                    (u - f).norm() < 1e-10,
                    "fused vs unfused circuit matrix mismatch for ordering {:?}: {} vs {}",
                    perm,
                    u,
                    f
                );
            }
        }
    }
}
