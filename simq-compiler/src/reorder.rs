//! Long-range qubit reordering pre-pass.
//!
//! Computes a qubit relabeling ([`QubitPermutation`]) that clusters
//! long-range-interacting qubits adjacent to each other, and applies it to a
//! [`Circuit`] to produce a relabeled circuit. This is a pure `Circuit ->
//! Circuit'` transformation inserted into the pipeline strictly *before*
//! [`crate::fusion::fuse_gates_with_cache`], which remains completely
//! unmodified.
//!
//! # Patent-diligence boundary
//!
//! This module operates inside the same patent-diligence boundary already
//! established for `fusion.rs` (see that module's docs): the existing
//! greedy, width-bounded fusion design is structurally distinct from two
//! on-point patents claiming a circuit-wide fusion graph with per-edge costs
//! and shortest-path schedule selection, and fusion crossing a measurement
//! boundary. This module must not, even by accident, reconstruct that
//! patented shape.
//!
//! [`compute_reordering`] is a pure function of circuit *topology* only — it
//! reads gate qubit targets and nothing else. It has zero visibility into
//! `fusion.rs`'s internals: it does not import `fusion::FusionBlock`,
//! `FusionConfig`, or any cost/scheduling state from that module, and it
//! runs to completion (see [`compute_reordering`] and [`apply_permutation`])
//! entirely *before* `fuse_gates_with_cache` is ever invoked — never
//! interleaved with it, never informed by its choices. The algorithm below
//! (Cuthill-McKee bandwidth minimization, Cuthill & McKee 1969) is a
//! textbook static graph-labeling technique: no per-edge cost weights, no
//! path/schedule selection, no notion of "schedule" at all — only a fixed
//! topological ordering computed once. This is re-verified against the
//! actual implementation at code-review time, per the companion TDD's
//! explicit blocking requirement; this module's doc comments are a
//! design-time read, not a substitute for that review.
//!
//! # What reordering actually changes (and what it doesn't)
//!
//! [`crate::fusion::find_fusion_blocks`]'s merge/close decisions are driven
//! entirely by qubit *identity* overlap between consecutive operations in
//! the fixed operation sequence (its `qubit_owner` array is an
//! identity-keyed lookup, not a numeric-distance comparison) — so relabeling
//! qubits is an isomorphism that **does not change which operations group
//! into which fusion blocks**. What reordering *does* change:
//!
//! 1. **Execution-time cache locality.** Once a block of qubits is fused,
//!    applying it to the dense/sparse state vector strides over memory at
//!    offsets determined by the block's qubit *indices*. Clustering
//!    interacting qubits to nearby indices reduces that stride, which is
//!    the actual mechanism by which reordering helps (not a change to which
//!    gates fuse).
//! 2. **`fusion.rs`'s block-vs-chain *dispatch* decision**
//!    (`has_long_range_structure`, `fusion.rs:895-908`), which reads raw
//!    qubit-index distance (`|q0 - q1| > 1`). Because reordering's entire
//!    purpose is to bring interacting qubits closer together in index, it
//!    can — for a circuit *below* `FusionConfig::parallel_threshold_qubits`
//!    — flip a circuit that was structurally "long-range" (and thus
//!    correctly dispatched to the multi-qubit block path) into one that no
//!    longer looks long-range post-relabeling, silently redirecting it back
//!    onto the legacy single-qubit-chain path, which never fuses a 2-qubit
//!    gate at all. That would be a strict regression, and worse, a *silent*
//!    one. See [`dispatch_would_survive_reorder`] and
//!    [`compute_reordering`] for how this module avoids it.
//!
//! Reordering is therefore only ever applied when the multi-qubit block
//! path is already guaranteed regardless of any dispatch heuristic —
//! i.e. `circuit.num_qubits() >= PARALLEL_THRESHOLD_QUBITS` — so dispatch
//! stability (see the companion TDD's TC-A8) is structural, not incidental.
//! See [`crate::pipeline`]/callers for where this gating is applied.

use ahash::AHashSet;
use simq_core::{Circuit, QubitId};
use std::collections::{BTreeSet, VecDeque};

/// Mirrors [`crate::fusion::FusionConfig`]'s default
/// `parallel_threshold_qubits` (18) — the qubit count at or above which
/// `fuse_gates_with_cache` always uses the multi-qubit block path
/// regardless of `has_long_range_structure`. This is the *public, documented
/// default* value, not private cost/scheduling state; duplicating it here
/// (rather than depending on `simq-sim` importing `fusion::FusionConfig`
/// directly, which this module intentionally avoids — see module docs) is
/// what lets reordering guarantee it never perturbs fusion's dispatch
/// decision. Kept in sync by convention, exactly as `fusion.rs` itself
/// already does relative to `simq-sim`'s constants.
pub const PARALLEL_THRESHOLD_QUBITS: usize = 18;

/// A bijection between "original" qubit indices (as they appear in a
/// caller-supplied [`Circuit`]) and "relabeled" qubit indices (as they
/// appear in the circuit [`apply_permutation`] produces). Both directions
/// are precomputed so lookups are O(1). No permutation/relabeling utility
/// existed anywhere in the codebase before this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QubitPermutation {
    /// `forward[original_index] = relabeled_index`
    forward: Vec<usize>,
    /// `inverse[relabeled_index] = original_index`
    inverse: Vec<usize>,
}

impl QubitPermutation {
    /// The identity permutation over `num_qubits` wires.
    pub fn identity(num_qubits: usize) -> Self {
        let ids: Vec<usize> = (0..num_qubits).collect();
        Self {
            forward: ids.clone(),
            inverse: ids,
        }
    }

    /// Build a permutation from an explicit forward mapping:
    /// `forward[original_index] = relabeled_index`. Returns `None` if
    /// `forward` is not a bijection on `0..forward.len()`.
    pub fn from_forward(forward: Vec<usize>) -> Option<Self> {
        let n = forward.len();
        let mut inverse = vec![usize::MAX; n];
        for (orig, &new_pos) in forward.iter().enumerate() {
            if new_pos >= n || inverse[new_pos] != usize::MAX {
                return None;
            }
            inverse[new_pos] = orig;
        }
        Some(Self { forward, inverse })
    }

    /// Build a permutation from `order`, where `order[new_position] =
    /// original_qubit_index` — i.e. `order` is the desired new
    /// left-to-right wire ordering, expressed in original qubit indices.
    ///
    /// # Panics
    /// Panics if `order` is not a permutation of `0..order.len()`. This is
    /// an internal invariant of [`compute_reordering`], never a condition
    /// that can arise from caller-supplied data.
    fn from_new_to_original_order(order: Vec<usize>) -> Self {
        let n = order.len();
        let mut forward = vec![usize::MAX; n];
        for (new_pos, &orig) in order.iter().enumerate() {
            assert!(orig < n, "reorder: order entry out of range");
            assert_eq!(forward[orig], usize::MAX, "reorder: order is not a bijection");
            forward[orig] = new_pos;
        }
        debug_assert!(forward.iter().all(|&f| f != usize::MAX));
        Self {
            forward,
            inverse: order,
        }
    }

    /// Number of qubits this permutation is defined over.
    pub fn num_qubits(&self) -> usize {
        self.forward.len()
    }

    /// Whether this permutation is the identity (no relabeling).
    pub fn is_identity(&self) -> bool {
        self.forward.iter().enumerate().all(|(i, &f)| i == f)
    }

    /// Map an original-space qubit to its relabeled position.
    pub fn apply_to(&self, q: QubitId) -> QubitId {
        QubitId::new(self.forward[q.index()])
    }

    /// Map a relabeled-space qubit back to its original position. The
    /// inverse of [`Self::apply_to`].
    pub fn invert_to(&self, q: QubitId) -> QubitId {
        QubitId::new(self.inverse[q.index()])
    }

    /// The forward mapping: `forward()[original_index] = relabeled_index`.
    pub fn forward(&self) -> &[usize] {
        &self.forward
    }

    /// The inverse mapping: `inverse()[relabeled_index] = original_index`.
    pub fn inverse(&self) -> &[usize] {
        &self.inverse
    }
}

/// Whether `circuit`'s current qubit count guarantees
/// `fuse_gates_with_cache` will select the multi-qubit block path
/// regardless of the `has_long_range_structure` heuristic — see module
/// docs. Reordering is only ever applied when this holds, so it can never
/// flip fusion's dispatch decision (the risk this module's docs describe).
pub fn dispatch_would_survive_reorder(circuit: &Circuit) -> bool {
    circuit.num_qubits() >= PARALLEL_THRESHOLD_QUBITS
}

/// Computes a qubit relabeling that clusters long-range-interacting qubits.
/// Pure function of circuit structure only — no fusion cost model, no
/// per-edge weighting, no scheduling decision. See module docs for the
/// patent-diligence rationale this constraint is derived from.
///
/// Returns the identity permutation when [`dispatch_would_survive_reorder`]
/// is false, since below that threshold a nontrivial relabeling risks
/// silently flipping `fusion.rs`'s dispatch decision (see module docs).
pub fn compute_reordering(circuit: &Circuit) -> QubitPermutation {
    let n = circuit.num_qubits();
    if n <= 2 || !dispatch_would_survive_reorder(circuit) {
        return QubitPermutation::identity(n);
    }

    // Build unweighted adjacency: two qubits are adjacent if some operation
    // acts on both. Only reads gate qubit *targets* — never gate.matrix(),
    // gate names, or anything resembling a fusion cost/benefit score.
    let mut adjacency: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for op in circuit.operations() {
        let qubits = op.qubits();
        if qubits.len() < 2 {
            continue;
        }
        for i in 0..qubits.len() {
            for j in (i + 1)..qubits.len() {
                let a = qubits[i].index();
                let b = qubits[j].index();
                adjacency[a].insert(b);
                adjacency[b].insert(a);
            }
        }
    }

    let order = cuthill_mckee_order(&adjacency);
    QubitPermutation::from_new_to_original_order(order)
}

/// Cuthill-McKee bandwidth-reduction ordering over an undirected graph given
/// as an adjacency list. Returns `order` such that `order[new_position] =
/// original_node`. A standard graph-theory technique (Cuthill & McKee,
/// 1969) chosen specifically because it has no per-edge cost weighting and
/// no schedule/path selection — see module docs.
///
/// Deliberately **not** the customary "Reverse" step (RCM): the reversal is
/// only useful for reducing fill-in during sparse matrix factorization, a
/// concern this module doesn't have. Bandwidth (the quantity this module
/// actually cares about) is invariant under reversing an ordering — for any
/// edge `(u, v)` at positions `(p, q)`, `|p - q| == |(n-1-p) - (n-1-q)|` —
/// so dropping the reversal is a no-op for our objective, and its absence
/// is what makes a simple qubit chain (e.g. a GHZ ansatz) reorder to the
/// exact identity permutation instead of a pointless full reversal.
fn cuthill_mckee_order(adjacency: &[BTreeSet<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let component = collect_component(adjacency, start);
        // Root: minimum-degree node in the component (ties broken by
        // smallest index for determinism), a cheap standard stand-in for a
        // true pseudo-peripheral vertex.
        let root = *component
            .iter()
            .min_by_key(|&&node| (adjacency[node].len(), node))
            .expect("component is non-empty");

        let mut queue = VecDeque::new();
        visited[root] = true;
        queue.push_back(root);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            let mut neighbors: Vec<usize> = adjacency[node]
                .iter()
                .copied()
                .filter(|nb| !visited[*nb])
                .collect();
            // BTreeSet iteration is already ascending-index order; break
            // ties in visiting priority by ascending degree (the standard
            // Cuthill-McKee rule) with ascending index as the sub-tie-break.
            neighbors.sort_by_key(|&nb| (adjacency[nb].len(), nb));
            for nb in neighbors {
                if !visited[nb] {
                    visited[nb] = true;
                    queue.push_back(nb);
                }
            }
        }
    }

    debug_assert_eq!(order.len(), n);
    order
}

/// All nodes reachable from `start` via `adjacency`, via a depth-first
/// traversal (order within the returned `Vec` is not meaningful — only
/// membership is used by [`cuthill_mckee_order`]).
fn collect_component(adjacency: &[BTreeSet<usize>], start: usize) -> Vec<usize> {
    let mut seen: AHashSet<usize> = AHashSet::new();
    let mut stack = vec![start];
    seen.insert(start);
    let mut out = vec![start];
    while let Some(node) = stack.pop() {
        for &nb in &adjacency[node] {
            if seen.insert(nb) {
                out.push(nb);
                stack.push(nb);
            }
        }
    }
    out
}

/// Applies `perm` to every gate's qubit operands, producing a relabeled
/// circuit with the same operations (same gates, same order) but each
/// [`QubitId`] mapped through `perm.apply_to`. `perm` and its inverse are
/// cheap, exact, and order-preserving within a wire — this never reorders
/// operations, only renames the wires they act on.
pub fn apply_permutation(circuit: &Circuit, perm: &QubitPermutation) -> Circuit {
    debug_assert_eq!(circuit.num_qubits(), perm.num_qubits());

    if perm.is_identity() {
        return circuit.clone();
    }

    let mut out = Circuit::with_capacity(circuit.num_qubits(), circuit.len());
    for op in circuit.operations() {
        let relabeled: Vec<QubitId> = op.qubits().iter().map(|&q| perm.apply_to(q)).collect();
        out.add_gate(std::sync::Arc::clone(op.gate()), &relabeled)
            .expect("relabeling a valid circuit's operations must stay valid");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::gate::Gate;
    use simq_gates::standard::{CNot, Hadamard};
    use std::sync::Arc;

    fn linear_chain_circuit(n: usize) -> Circuit {
        let mut circuit = Circuit::new(n);
        for q in 0..n {
            circuit
                .add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[QubitId::new(q)])
                .unwrap();
        }
        for q in 0..n - 1 {
            circuit
                .add_gate(
                    Arc::new(CNot) as Arc<dyn Gate>,
                    &[QubitId::new(q), QubitId::new(q + 1)],
                )
                .unwrap();
        }
        circuit
    }

    fn fully_connected_circuit(n: usize) -> Circuit {
        let mut circuit = Circuit::new(n);
        for a in 0..n {
            for b in (a + 1)..n {
                circuit
                    .add_gate(
                        Arc::new(CNot) as Arc<dyn Gate>,
                        &[QubitId::new(a), QubitId::new(b)],
                    )
                    .unwrap();
            }
        }
        circuit
    }

    // TC-A3: apply_permutation composed with the inverse round-trips to
    // identity for arbitrary circuits.
    #[test]
    fn test_permutation_round_trips_to_identity() {
        let circuit = fully_connected_circuit(20);
        let perm = compute_reordering(&circuit);
        let relabeled = apply_permutation(&circuit, &perm);

        // Invert: map every qubit in `relabeled` back through perm.invert_to
        // and confirm we recover the original circuit's structure exactly.
        assert_eq!(relabeled.len(), circuit.len());
        for (orig_op, new_op) in circuit.operations().zip(relabeled.operations()) {
            let recovered: Vec<QubitId> =
                new_op.qubits().iter().map(|&q| perm.invert_to(q)).collect();
            assert_eq!(recovered, orig_op.qubits());
            assert_eq!(orig_op.gate().name(), new_op.gate().name());
        }
    }

    #[test]
    fn test_permutation_forward_inverse_are_mutual_inverses() {
        let circuit = fully_connected_circuit(20);
        let perm = compute_reordering(&circuit);
        for i in 0..perm.num_qubits() {
            let q = QubitId::new(i);
            assert_eq!(perm.invert_to(perm.apply_to(q)), q);
            assert_eq!(perm.apply_to(perm.invert_to(q)), q);
        }
    }

    // TC-A12: a circuit with only local (adjacent) interactions is left at
    // the identity permutation (dropping the customary RCM reversal is what
    // makes this hold exactly, not just "near" — see cuthill_mckee_order's
    // docs).
    #[test]
    fn test_linear_chain_reorders_to_identity() {
        let circuit = linear_chain_circuit(20);
        let perm = compute_reordering(&circuit);
        assert!(perm.is_identity(), "a simple chain should not be perturbed");
    }

    // Below PARALLEL_THRESHOLD_QUBITS, compute_reordering always returns
    // identity — this is what guarantees dispatch stability (TC-A8) rather
    // than it being incidental. See dispatch_would_survive_reorder's docs.
    #[test]
    fn test_below_threshold_is_always_identity() {
        let circuit = fully_connected_circuit(10);
        assert!(circuit.num_qubits() < PARALLEL_THRESHOLD_QUBITS);
        let perm = compute_reordering(&circuit);
        assert!(perm.is_identity());
    }

    #[test]
    fn test_dispatch_would_survive_reorder_matches_threshold() {
        assert!(!dispatch_would_survive_reorder(&Circuit::new(17)));
        assert!(dispatch_would_survive_reorder(&Circuit::new(18)));
    }

    // A circuit whose only long-range interaction is a single edge between
    // far-apart qubits should end up with that edge's endpoints clustered
    // together in the new labeling.
    #[test]
    fn test_far_apart_pair_is_clustered_after_reorder() {
        let mut circuit = Circuit::new(20);
        // A local chain 0..18 plus one long-range edge between qubit 0 and
        // qubit 19, which starts out maximally far apart.
        for q in 0..18 {
            circuit
                .add_gate(
                    Arc::new(CNot) as Arc<dyn Gate>,
                    &[QubitId::new(q), QubitId::new(q + 1)],
                )
                .unwrap();
        }
        circuit
            .add_gate(
                Arc::new(CNot) as Arc<dyn Gate>,
                &[QubitId::new(0), QubitId::new(19)],
            )
            .unwrap();

        let perm = compute_reordering(&circuit);
        let new_0 = perm.apply_to(QubitId::new(0)).index();
        let new_19 = perm.apply_to(QubitId::new(19)).index();
        assert!(
            new_0.abs_diff(new_19) < 19,
            "reordering should reduce the index distance for an interacting pair"
        );
    }

    #[test]
    fn test_apply_permutation_preserves_operation_count_and_order() {
        let circuit = fully_connected_circuit(19);
        let perm = compute_reordering(&circuit);
        let relabeled = apply_permutation(&circuit, &perm);
        assert_eq!(relabeled.len(), circuit.len());
        assert_eq!(relabeled.num_qubits(), circuit.num_qubits());
        for (orig, new) in circuit.operations().zip(relabeled.operations()) {
            assert_eq!(orig.gate().name(), new.gate().name());
            assert_eq!(orig.num_qubits(), new.num_qubits());
        }
    }

    #[test]
    fn test_apply_identity_permutation_is_a_clone() {
        let circuit = linear_chain_circuit(5);
        let perm = QubitPermutation::identity(5);
        let relabeled = apply_permutation(&circuit, &perm);
        for (orig, new) in circuit.operations().zip(relabeled.operations()) {
            assert_eq!(orig.qubits(), new.qubits());
        }
    }

    #[test]
    fn test_qubit_permutation_identity_accessors() {
        let perm = QubitPermutation::identity(4);
        assert!(perm.is_identity());
        assert_eq!(perm.num_qubits(), 4);
        assert_eq!(perm.forward(), &[0, 1, 2, 3]);
        assert_eq!(perm.inverse(), &[0, 1, 2, 3]);
    }
}
