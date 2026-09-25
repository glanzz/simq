//! Inverts `simq_compiler::reorder`'s qubit permutation on a simulation
//! result, mapping a state produced by executing a *relabeled* circuit back
//! into the caller's original wire space.
//!
//! This lives in `simq-sim`, not `simq-compiler::reorder` alongside
//! [`compute_reordering`](simq_compiler::compute_reordering)/
//! [`apply_permutation`](simq_compiler::apply_permutation), because it
//! operates on [`AdaptiveState`] (from `simq-state`) — and `simq-compiler`
//! has no dependency on `simq-sim`/`simq-state` (see
//! `simq_compiler::fusion`'s module docs, which establish this as a hard
//! architectural constraint, not an oversight). `simq-sim` already depends
//! on both `simq-compiler` and `simq-state`, so this is the natural home
//! for the half of the round trip that needs both.
//!
//! # Correctness-critical detail
//!
//! [`simq_core`]'s amplitude-index convention (confirmed by
//! `AdaptiveState::get_probability`'s existing tests) is **LSB-first**: bit
//! `k` (weight `2^k`) of a basis-state index holds qubit `k`'s value. A
//! circuit compiled with permutation `perm` (mapping original qubit `q` to
//! relabeled position `perm.apply_to(q)`) produces a final state whose index
//! bit `b` holds the value of *relabeled* qubit `b` — i.e. of *original*
//! qubit `perm.invert_to(b)`. So the original-space index `i_orig`
//! satisfies, for every original qubit `q`:
//!
//! ```text
//! bit_q(i_orig) == bit_{perm.apply_to(q)}(i_relabeled)
//! ```
//!
//! This must run *after* execution completes, never folded into gate
//! application itself — doing it at the wrong pipeline stage produces
//! wrong-but-plausible amplitudes silently, not a panic (this is the
//! highest-risk correctness seam in the reorder feature; see
//! `simq_compiler::reorder`'s module docs).

use simq_compiler::QubitPermutation;
use simq_state::AdaptiveState;

/// Undoes `perm` on a final simulation state, producing an equivalent state
/// indexed in the caller's original wire space. `perm` is the permutation
/// [`simq_compiler::compute_reordering`] produced for the circuit that was
/// executed to obtain `state`. A no-op (cheap clone-free early return) when
/// `perm` is the identity.
pub fn invert_permutation_on_state(
    state: &AdaptiveState,
    perm: &QubitPermutation,
) -> simq_state::error::Result<AdaptiveState> {
    if perm.is_identity() {
        return Ok(clone_state(state));
    }
    debug_assert_eq!(state.num_qubits(), perm.num_qubits());

    let relabeled_amplitudes = state.to_dense_vec();
    let dim = relabeled_amplitudes.len();
    let num_qubits = perm.num_qubits();
    let forward = perm.forward(); // forward[original] = relabeled

    let mut original_amplitudes = vec![num_complex::Complex64::new(0.0, 0.0); dim];
    for (i_relabeled, amp) in relabeled_amplitudes.iter().enumerate() {
        let mut i_orig = 0usize;
        for (q, &new_pos) in forward.iter().enumerate().take(num_qubits) {
            let bit = (i_relabeled >> new_pos) & 1;
            i_orig |= bit << q;
        }
        original_amplitudes[i_orig] = *amp;
    }

    AdaptiveState::from_amplitudes(num_qubits, &original_amplitudes)
}

fn clone_state(state: &AdaptiveState) -> AdaptiveState {
    // AdaptiveState doesn't implement Clone (its variants hold large
    // buffers deliberately not made Clone by accident) — reconstruct
    // explicitly via the dense round trip, matching `to_dense_vec`'s own
    // existing "authoritative amplitude view" contract.
    AdaptiveState::from_amplitudes(state.num_qubits(), &state.to_dense_vec())
        .expect("cloning a valid state must stay valid")
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use simq_compiler::QubitPermutation;

    fn state_from(amps: Vec<Complex64>) -> AdaptiveState {
        let n = amps.len().trailing_zeros() as usize;
        AdaptiveState::from_amplitudes(n, &amps).unwrap()
    }

    #[test]
    fn test_identity_permutation_is_a_no_op() {
        let amps = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];
        let state = state_from(amps.clone());
        let perm = QubitPermutation::identity(2);
        let inverted = invert_permutation_on_state(&state, &perm).unwrap();
        for (a, b) in amps.iter().zip(inverted.to_dense_vec().iter()) {
            assert!((a - b).norm() < 1e-15);
        }
    }

    // TC-A3 (state half): applying a permutation to a circuit and then
    // inverting on the resulting state must round-trip a known basis state
    // to the same basis state in the original labeling.
    #[test]
    fn test_swap_permutation_moves_amplitude_to_expected_original_index() {
        // 3 qubits, perm swaps original qubit 0 <-> relabeled position 2
        // (and vice versa), qubit 1 fixed: forward = [2, 1, 0].
        let perm = QubitPermutation::from_forward(vec![2, 1, 0]).unwrap();

        // In relabeled space, set basis state |q2=1, q1=0, q0=0> = index 4
        // (bit 2 set). Under forward = [2,1,0]: original qubit 0 maps to
        // relabeled bit 2, so this amplitude belongs at original index
        // where bit_0(i_orig) = bit_2(4) = 1, bit_1 = bit_1(4) = 0, bit_2 =
        // bit_0(4) = 0 -> i_orig = 1.
        let mut relabeled_amps = vec![Complex64::new(0.0, 0.0); 8];
        relabeled_amps[4] = Complex64::new(1.0, 0.0);
        let state = state_from(relabeled_amps);

        let inverted = invert_permutation_on_state(&state, &perm).unwrap();
        let dense = inverted.to_dense_vec();
        assert!((dense[1].norm() - 1.0).abs() < 1e-12);
        for (idx, amp) in dense.iter().enumerate() {
            if idx != 1 {
                assert!(amp.norm() < 1e-12);
            }
        }
    }

    #[test]
    fn test_non_identity_round_trips_norm() {
        let val = 1.0 / 2.0_f64.sqrt();
        let amps = vec![
            Complex64::new(val, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(val, 0.0),
        ];
        let state = state_from(amps);
        let perm = QubitPermutation::from_forward(vec![1, 0]).unwrap();
        let inverted = invert_permutation_on_state(&state, &perm).unwrap();
        assert!((inverted.norm() - 1.0).abs() < 1e-10);
    }
}
