//! Equality-saturation optimization for single-qubit gate chains.
//!
//! [`TemplateSubstitution`](crate::passes::TemplateSubstitution) matches a
//! *fixed* list of gate-name patterns with one greedy, left-to-right,
//! no-backtrack scan: it can rewrite `S,S,S,S` to nothing because that
//! exact 4-gate pattern is in its table, but it cannot discover that
//! `T,T,T,T,T,T,T,T` is also the identity unless someone adds an 8-gate
//! entry for every such compound case by hand — greedy rewriting only ever
//! explores the one rewrite sequence its scan order happens to try first.
//!
//! Equality saturation fixes this by construction: every rewrite rule is
//! applied to *every* equivalent form simultaneously (via an e-graph, which
//! shares equivalent sub-terms instead of picking one), so `T,T -> S` and
//! `S,S,S,S -> I` compose automatically into `T^8 = I` without that fact
//! ever being stated directly. This module runs that search per
//! single-qubit gate chain (the same chain concept
//! [`crate::fusion::find_fusion_chains`] uses) via the `egg` crate, then
//! extracts the lowest-gate-count equivalent sequence found.
//!
//! # Correctness: rewrites are "up to a uniform scalar," and why that's safe
//!
//! Some rules here (the Hadamard-conjugation ones) are only true up to an
//! overall scalar factor on the *whole* replaced sub-block's matrix (e.g.
//! `H,Y,H` implements `-Y`, not `Y`). This is still exactly correct in any
//! circuit, entangled or not: for a unitary `U` and scalar `c`, `U(c|psi>)
//! = c(U|psi>)` for every subsequent gate `U`, so a uniform scalar
//! introduced anywhere in a linear qubit-chain factors out to a pure
//! *global* phase on the final state — never a relative phase between
//! basis states, which is the only kind of phase difference that's
//! actually observable. This is the same assumption
//! `TemplateSubstitution`'s `h-y-h` template already relies on; this module
//! doesn't introduce new risk, it just finds more of the same class of
//! rewrite. Every rule below is checked by `tests::all_rules_preserve_the_circuit_matrix_up_to_a_global_phase`
//! against this crate's own matrix machinery, not just asserted from
//! memory.
//!
//! # Scope
//!
//! Only the 9 fixed (non-parameterized) single-qubit gates
//! `TemplateSubstitution` already targets: H, X, Y, Z, S, S-dagger, T,
//! T-dagger, I. Parameterized gates (RX/RY/RZ/...) are deliberately out of
//! scope: `RY(a),RY(b) = RY(a+b)` is a real identity, but it needs numeric
//! angle composition, not term rewriting, and mixing the two would require
//! a second, differently-verified code path this module doesn't attempt.

use crate::passes::OptimizationPass;
use egg::{
    define_language, rewrite as rw, CostFunction, Extractor, Id, Language, RecExpr, Rewrite, Runner,
};
use simq_core::{gate::Gate, Circuit, QubitId, Result};
use simq_gates::standard::{
    Hadamard, Identity, PauliX, PauliY, PauliZ, SGate, SGateDagger, TGate, TGateDagger,
};
use std::sync::Arc;

define_language! {
    /// A single-qubit gate term: either a named atom (see module docs for
    /// the exact 9-gate scope) or `seq(a, b)` = "apply `a`, then `b`."
    enum GateLang {
        "I" = IdentityAtom,
        "X" = XAtom,
        "Y" = YAtom,
        "Z" = ZAtom,
        "H" = HAtom,
        "S" = SAtom,
        "Sdg" = SdgAtom,
        "T" = TAtom,
        "Tdg" = TdgAtom,
        "seq" = Seq([Id; 2]),
    }
}

/// Gate names this module can represent, in the same order as the
/// [`GateLang`] atoms above. The chain-finder (see [`find_gate_chains`])
/// only chains gates whose name appears here.
const SUPPORTED_GATE_NAMES: [&str; 9] = ["I", "X", "Y", "Z", "H", "S", "S†", "T", "T†"];

fn atom_from_name(name: &str) -> Option<GateLang> {
    match name {
        "I" => Some(GateLang::IdentityAtom),
        "X" => Some(GateLang::XAtom),
        "Y" => Some(GateLang::YAtom),
        "Z" => Some(GateLang::ZAtom),
        "H" => Some(GateLang::HAtom),
        "S" => Some(GateLang::SAtom),
        "S†" => Some(GateLang::SdgAtom),
        "T" => Some(GateLang::TAtom),
        "T†" => Some(GateLang::TdgAtom),
        _ => None,
    }
}

/// Inverse of [`atom_from_name`]; also used to build the replacement
/// [`Gate`] instances (all zero-argument, so no parameters to thread
/// through).
fn gate_for_atom(atom: &GateLang) -> Arc<dyn Gate> {
    match atom {
        GateLang::IdentityAtom => Arc::new(Identity),
        GateLang::XAtom => Arc::new(PauliX),
        GateLang::YAtom => Arc::new(PauliY),
        GateLang::ZAtom => Arc::new(PauliZ),
        GateLang::HAtom => Arc::new(Hadamard),
        GateLang::SAtom => Arc::new(SGate),
        GateLang::SdgAtom => Arc::new(SGateDagger),
        GateLang::TAtom => Arc::new(TGate),
        GateLang::TdgAtom => Arc::new(TGateDagger),
        GateLang::Seq(_) => unreachable!("Seq is not a gate atom"),
    }
}

/// The rewrite rules this pass saturates over. Each is verified against
/// this crate's own gate matrices in `tests` below — see the module docs'
/// "Correctness" section for why the ones with an implicit scalar factor
/// are still exact for circuit simulation purposes.
///
/// # Why there is no associativity rule
///
/// A chain is always built (see [`optimize_chain`]) as a single fixed
/// right-associated tree, `seq(g0, seq(g1, seq(g2, ...)))`, terminated by a
/// trailing `I`. Every rule below matches a window at the *front* of a
/// `?rest` tail rather than a bare adjacent pair (e.g. `(seq X (seq X
/// ?rest))`, not `(seq X X)`) so it fires at any position — egg matches
/// patterns against every e-class, and the `?rest` sub-tree at position `k`
/// in the original chain is exactly the e-class representing the chain's
/// tail starting there, so this covers every window without ever
/// re-parenthesizing anything.
///
/// A bidirectional `(seq (seq ?a ?b) ?c) <=> (seq ?a (seq ?b ?c))`
/// associativity rule was tried first and measured over 600ms per 19-gate
/// chain: general associativity lets an e-graph derive every one of a
/// chain's `Catalan(n-1)` parenthesizations, which is astronomically large
/// even for modest `n` (n=19 => catalan(18) ~ 4.77e8) — a textbook e-graph
/// blowup, not a fundamental requirement of the technique. Fixing the
/// grouping once and writing tail-anchored rules instead needs zero
/// re-association and dropped the same chain to low-microsecond runs (see
/// `tests::optimize_chain_is_fast_on_a_long_chain`).
fn rules() -> Vec<Rewrite<GateLang, ()>> {
    vec![
        // Identity elimination.
        rw!("seq-i-left"; "(seq I ?rest)" => "?rest"),
        rw!("seq-i-right"; "(seq ?a I)" => "?a"),
        // Self-inverse pairs (exact, phase = 1).
        rw!("xx"; "(seq X (seq X ?rest))" => "?rest"),
        rw!("yy"; "(seq Y (seq Y ?rest))" => "?rest"),
        rw!("zz"; "(seq Z (seq Z ?rest))" => "?rest"),
        rw!("hh"; "(seq H (seq H ?rest))" => "?rest"),
        rw!("s-sdg"; "(seq S (seq Sdg ?rest))" => "?rest"),
        rw!("sdg-s"; "(seq Sdg (seq S ?rest))" => "?rest"),
        rw!("t-tdg"; "(seq T (seq Tdg ?rest))" => "?rest"),
        rw!("tdg-t"; "(seq Tdg (seq T ?rest))" => "?rest"),
        // Power identities (exact, phase = 1). `T,T,T,T,T,T,T,T = I` and
        // `S,S,S,S = I` are deliberately NOT listed as their own rules —
        // saturation derives both by re-applying `tt`/`ss` (each collapsing
        // one adjacent pair) followed by `zz`, which is the entire point of
        // this module over `TemplateSubstitution`.
        rw!("ss"; "(seq S (seq S ?rest))" => "(seq Z ?rest)"),
        rw!("tt"; "(seq T (seq T ?rest))" => "(seq S ?rest)"),
        // Hadamard conjugation (H,Y,H is exact only up to a uniform -1
        // scalar — see module docs for why that's still safe here).
        rw!("hxh"; "(seq H (seq X (seq H ?rest)))" => "(seq Z ?rest)"),
        rw!("hzh"; "(seq H (seq Z (seq H ?rest)))" => "(seq X ?rest)"),
        rw!("hyh"; "(seq H (seq Y (seq H ?rest)))" => "(seq Y ?rest)"),
    ]
}

/// Extraction cost: total non-identity atom count. `I` is free (and the
/// `seq-i-*` rules mean a saturated e-graph never needs to keep one
/// wrapped in a `seq` anyway) — this simply biases extraction toward
/// shorter gate sequences, which is what actually reduces simulation cost.
struct GateCost;

impl CostFunction<GateLang> for GateCost {
    type Cost = usize;

    fn cost<C>(&mut self, enode: &GateLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            GateLang::IdentityAtom => 0,
            GateLang::Seq(ids) => ids.iter().map(|&id| costs(id)).sum(),
            _ => 1 + enode.children().iter().map(|&id| costs(id)).sum::<usize>(),
        }
    }
}

/// Flatten an extracted (possibly rebalanced by associativity) `seq` tree
/// back into application order: `seq(a, b)` means "a" happens first.
fn flatten(expr: &RecExpr<GateLang>, id: Id) -> Vec<GateLang> {
    match &expr[id] {
        GateLang::Seq([a, b]) => {
            let mut out = flatten(expr, *a);
            out.extend(flatten(expr, *b));
            out
        },
        GateLang::IdentityAtom => Vec::new(),
        atom => vec![atom.clone()],
    }
}

/// Run equality saturation over one gate-name chain (application order,
/// e.g. `["H", "X", "H"]`) and return the lowest-cost equivalent chain
/// found. Built right-associated with a trailing `I` terminator — see
/// [`rules`]'s docs for why that shape (not general associativity) is what
/// keeps this fast. Bounded iteration/node/time limits (see `Runner`
/// construction) are a defense-in-depth backstop, not load-bearing at this
/// chain length: hitting a limit just stops saturation early; extraction
/// still returns the best expression found so far, never a wrong one.
fn optimize_chain(names: &[&str]) -> Vec<GateLang> {
    if names.is_empty() {
        return Vec::new();
    }

    let mut expr = RecExpr::default();
    let mut tail = expr.add(GateLang::IdentityAtom);
    for &name in names.iter().rev() {
        let atom = atom_from_name(name).expect("caller only passes supported gate names");
        let atom_id = expr.add(atom);
        tail = expr.add(GateLang::Seq([atom_id, tail]));
    }

    let runner = Runner::default()
        .with_iter_limit(30)
        .with_node_limit(50_000)
        .with_time_limit(std::time::Duration::from_millis(500))
        .with_expr(&expr)
        .run(&rules());

    let extractor = Extractor::new(&runner.egraph, GateCost);
    let (_cost, best) = extractor.find_best(runner.roots[0]);
    flatten(&best, Id::from(best.as_ref().len() - 1))
}

/// A chain of same-qubit, chainable-gate operation indices — the same
/// "adjacent on one wire, robust to interleaved other-qubit ops" concept as
/// [`crate::fusion::find_fusion_chains`], scoped to
/// [`SUPPORTED_GATE_NAMES`] instead of "has a matrix at all."
struct GateChain {
    operation_indices: Vec<usize>,
    #[allow(dead_code)] // read by tests; the pass itself only needs the indices
    qubit: QubitId,
}

fn find_gate_chains(circuit: &Circuit) -> Vec<GateChain> {
    let num_qubits = circuit.num_qubits();
    let operations: Vec<_> = circuit.operations().collect();
    let mut current: Vec<Option<Vec<usize>>> = vec![None; num_qubits];
    let mut chains = Vec::new();

    let close =
        |qubit_idx: usize, current: &mut Vec<Option<Vec<usize>>>, chains: &mut Vec<GateChain>| {
            if let Some(chain) = current[qubit_idx].take() {
                if chain.len() >= 2 {
                    chains.push(GateChain {
                        operation_indices: chain,
                        qubit: QubitId::new(qubit_idx),
                    });
                }
            }
        };

    for (op_idx, op) in operations.iter().enumerate() {
        let qubits = op.qubits();
        if qubits.len() == 1 && SUPPORTED_GATE_NAMES.contains(&op.gate().name()) {
            let q = qubits[0].index();
            current[q].get_or_insert_with(Vec::new).push(op_idx);
        } else {
            for &q in qubits {
                close(q.index(), &mut current, &mut chains);
            }
        }
    }
    for q in 0..num_qubits {
        close(q, &mut current, &mut chains);
    }
    chains
}

/// Equality-saturation optimization pass for single-qubit gate chains.
///
/// **Opt-in, not part of the default `O2`/`O3` pipelines** — add via
/// [`crate::pipeline::PipelineBuilder::with_equality_saturation`]. This
/// mirrors this repository's own stated bar for the technique ("keep it
/// opt-in until it matches or beats the current fixed-point pipeline on
/// the full cross-validated suite"): it's a real, tested implementation of
/// e-graph-based circuit optimization, not yet a default because its
/// benefit is concentrated on circuits with redundant fixed-gate chains
/// (e.g. naively-decomposed or template-heavy input) rather than this
/// crate's benchmarked VQE/QAOA/GHZ/QFT workloads, which mostly use
/// parameterized rotation gates outside this pass's scope (see module
/// docs).
#[derive(Debug, Clone, Default)]
pub struct EqualitySaturation;

impl EqualitySaturation {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationPass for EqualitySaturation {
    fn name(&self) -> &str {
        "equality-saturation"
    }

    fn apply(&self, circuit: &mut Circuit) -> Result<bool> {
        let chains = find_gate_chains(circuit);
        if chains.is_empty() {
            return Ok(false);
        }

        let operations: Vec<_> = circuit.operations().collect();
        let mut replacements: Vec<(usize, Vec<Arc<dyn Gate>>)> = Vec::new();
        let mut removed: ahash::AHashSet<usize> = ahash::AHashSet::new();
        let mut modified = false;

        for chain in &chains {
            let names: Vec<&str> = chain
                .operation_indices
                .iter()
                .map(|&idx| operations[idx].gate().name())
                .collect();
            let optimized = optimize_chain(&names);
            if optimized.len() < names.len() {
                modified = true;
                replacements.push((
                    chain.operation_indices[0],
                    optimized.iter().map(gate_for_atom).collect(),
                ));
                removed.extend(chain.operation_indices.iter().skip(1));
                if optimized.is_empty() {
                    removed.insert(chain.operation_indices[0]);
                }
            }
        }

        if !modified {
            return Ok(false);
        }

        let replacements: ahash::AHashMap<usize, Vec<Arc<dyn Gate>>> =
            replacements.into_iter().collect();
        let mut new_circuit = Circuit::with_capacity(circuit.num_qubits(), circuit.len());
        for (idx, op) in operations.iter().enumerate() {
            if let Some(gates) = replacements.get(&idx) {
                for gate in gates {
                    new_circuit.add_gate(Arc::clone(gate), &[op.qubits()[0]])?;
                }
            } else if !removed.contains(&idx) {
                new_circuit.add_gate(Arc::clone(op.gate()), op.qubits())?;
            }
        }
        *circuit = new_circuit;
        Ok(true)
    }

    fn description(&self) -> Option<&str> {
        Some("Equality-saturation optimization of single-qubit gate chains (opt-in; see module docs)")
    }

    fn iterative(&self) -> bool {
        // A shortened chain can't create new cross-chain opportunities
        // (chains never merge across a boundary gate), so one application
        // per compile is enough.
        false
    }

    fn benefit_score(&self) -> f64 {
        0.6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use simq_core::QubitId;
    use simq_gates::matrix_ops::circuit_matrix;
    use std::sync::Arc;

    /// Multiply out a single-qubit gate-name chain's matrix directly via
    /// `simq_gates`, independent of `egraph.rs`'s own atom/cost machinery —
    /// the ground truth every rewrite rule (and the pass's end-to-end
    /// output) is checked against.
    fn chain_matrix(names: &[&str]) -> Vec<num_complex::Complex64> {
        let mut c = Circuit::new(1);
        for &name in names {
            let gate = gate_for_atom(&atom_from_name(name).unwrap());
            c.add_gate(gate, &[QubitId::new(0)]).unwrap();
        }
        circuit_matrix(&c).unwrap()
    }

    /// True if `a` and `b` are equal up to a single global unit-modulus
    /// scalar (the equivalence this module's rules are allowed to use —
    /// see the module docs' correctness section).
    fn equal_up_to_global_phase(
        a: &[num_complex::Complex64],
        b: &[num_complex::Complex64],
    ) -> bool {
        let Some((&a0, &b0)) = a.iter().zip(b.iter()).find(|(x, _)| x.norm() > 1e-9) else {
            return b.iter().all(|v| v.norm() < 1e-9);
        };
        let phase = b0 / a0;
        (phase.norm() - 1.0).abs() < 1e-9
            && a.iter()
                .zip(b)
                .all(|(&x, &y)| (x * phase - y).norm() < 1e-9)
    }

    #[test]
    fn all_rules_preserve_the_circuit_matrix_up_to_a_global_phase() {
        // Each rule's LHS/RHS pattern, spelled out as concrete gate chains
        // (mirroring `rules()` above one-for-one) so a wrong hand-derived
        // identity fails here instead of silently corrupting a circuit.
        let cases: &[(&[&str], &[&str])] = &[
            (&["X", "X"], &["I"]),
            (&["Y", "Y"], &["I"]),
            (&["Z", "Z"], &["I"]),
            (&["H", "H"], &["I"]),
            (&["S", "S†"], &["I"]),
            (&["S†", "S"], &["I"]),
            (&["T", "T†"], &["I"]),
            (&["T†", "T"], &["I"]),
            (&["S", "S"], &["Z"]),
            (&["T", "T"], &["S"]),
            (&["S", "S", "S", "S"], &["I"]),
            (&["H", "X", "H"], &["Z"]),
            (&["H", "Z", "H"], &["X"]),
            (&["H", "Y", "H"], &["Y"]),
        ];
        for (lhs, rhs) in cases {
            let m_lhs = chain_matrix(lhs);
            let m_rhs = chain_matrix(rhs);
            assert!(
                equal_up_to_global_phase(&m_lhs, &m_rhs),
                "{lhs:?} should equal {rhs:?} up to global phase, got {m_lhs:?} vs {m_rhs:?}"
            );
        }
    }

    #[test]
    fn discovers_t_to_the_8_equals_identity_without_a_hardcoded_rule() {
        // No rule anywhere mentions 8 T gates directly -- this must come
        // from composing `tt` (T,T -> S) and `ss4` (S^4 -> I).
        let optimized = optimize_chain(&["T"; 8]);
        assert!(optimized.is_empty(), "expected T^8 to reduce to nothing, got {optimized:?}");
    }

    #[test]
    fn optimize_chain_matches_template_substitution_on_its_own_examples() {
        assert_eq!(optimize_chain(&["S", "S"]), vec![GateLang::ZAtom]);
        assert_eq!(optimize_chain(&["H", "Z", "H"]), vec![GateLang::XAtom]);
        assert!(optimize_chain(&["X", "X"]).is_empty());
    }

    #[test]
    fn finds_a_reduction_a_single_left_to_right_scan_would_miss() {
        // H,X,H,X,H,X,H: no fixed-length template in TemplateSubstitution's
        // table matches this 7-gate run directly, but repeated H-conjugation
        // collapses it. (HXH)X(HXH) = Z X Z = X (Z,X anticommute; Z X Z X = I
        // so Z X Z = X). Equality saturation should find *some* strictly
        // shorter form; the exact minimum isn't asserted; a length-4-or-more
        // reduction is a "greedy would plausibly get stuck" result.
        let optimized = optimize_chain(&["H", "X", "H", "X", "H", "X", "H"]);
        assert!(
            optimized.len() < 7,
            "expected a shorter equivalent for H,X,H,X,H,X,H, got {optimized:?}"
        );
        let names: Vec<&str> = optimized
            .iter()
            .map(|a| match a {
                GateLang::XAtom => "X",
                GateLang::YAtom => "Y",
                GateLang::ZAtom => "Z",
                GateLang::HAtom => "H",
                GateLang::SAtom => "S",
                GateLang::SdgAtom => "S†",
                GateLang::TAtom => "T",
                GateLang::TdgAtom => "T†",
                GateLang::IdentityAtom => "I",
                GateLang::Seq(_) => unreachable!(),
            })
            .collect();
        let reduced_matrix = chain_matrix(&names);
        let original_matrix = chain_matrix(&["H", "X", "H", "X", "H", "X", "H"]);
        assert!(equal_up_to_global_phase(&reduced_matrix, &original_matrix));
    }

    /// Regression guard for the exact blowup `rules()`'s docs describe: a
    /// bidirectional associativity rule once made this take >600ms per
    /// chain (Catalan-many parenthesizations). A generous 50ms budget for
    /// a 19-gate chain leaves ample margin over the low-microsecond times
    /// this module actually runs at while still catching a real regression.
    #[test]
    fn optimize_chain_is_fast_on_a_long_chain() {
        let mut names = vec!["T"; 6];
        names.extend(["H", "X", "H", "X", "H", "X", "H"]);
        names.extend(["S"; 6]);

        let start = std::time::Instant::now();
        let optimized = optimize_chain(&names);
        let elapsed = start.elapsed();

        assert!(
            elapsed < std::time::Duration::from_millis(50),
            "optimize_chain took {elapsed:?} on a 19-gate chain, expected low milliseconds"
        );
        assert!(optimized.len() < names.len());
    }

    #[test]
    fn pass_reduces_gate_count_on_a_redundant_circuit() {
        let mut circuit = Circuit::new(1);
        for _ in 0..8 {
            circuit
                .add_gate(Arc::new(TGate), &[QubitId::new(0)])
                .unwrap();
        }
        let original_len = circuit.len();

        let pass = EqualitySaturation::new();
        let modified = pass.apply(&mut circuit).unwrap();

        assert!(modified);
        assert!(circuit.len() < original_len);
        assert_eq!(circuit.len(), 0, "T^8 should fully cancel to nothing");
    }

    #[test]
    fn pass_leaves_a_two_qubit_gate_untouched_and_does_not_merge_chains_across_it() {
        use simq_gates::standard::CNot;
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliZ), &[QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliZ), &[QubitId::new(1)])
            .unwrap();

        let pass = EqualitySaturation::new();
        let mut optimized = circuit.clone();
        let modified = pass.apply(&mut optimized).unwrap();

        assert!(modified);
        // X,X before the CNOT cancels; the CNOT stays; Z,Z after cancels.
        assert_eq!(optimized.len(), 1);
        assert_eq!(optimized.operations().next().unwrap().gate().name(), "CNOT");
    }

    #[test]
    fn pass_is_a_noop_on_a_circuit_with_no_reducible_chain() {
        use simq_gates::standard::CNot;
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(CNot), &[QubitId::new(0), QubitId::new(1)])
            .unwrap();

        let pass = EqualitySaturation::new();
        let mut optimized = circuit.clone();
        let modified = pass.apply(&mut optimized).unwrap();
        assert!(!modified);
        assert_eq!(optimized.len(), circuit.len());
    }

    #[test]
    fn find_gate_chains_survives_interleaved_other_qubit_ops() {
        // X(q0), H(q1), X(q0): q0's chain must still be found as length 2
        // even though an unrelated q1 op sits between them in the list --
        // the specific weakness TemplateSubstitution's flat scan has that
        // this module's chain-finder avoids (see module docs).
        let mut circuit = Circuit::new(2);
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();
        circuit
            .add_gate(Arc::new(Hadamard), &[QubitId::new(1)])
            .unwrap();
        circuit
            .add_gate(Arc::new(PauliX), &[QubitId::new(0)])
            .unwrap();

        let chains = find_gate_chains(&circuit);
        let q0_chain = chains.iter().find(|c| c.qubit == QubitId::new(0)).unwrap();
        assert_eq!(q0_chain.operation_indices, vec![0, 2]);
    }
}
