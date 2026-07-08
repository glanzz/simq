//! Two-qubit gate application kernels

use super::Matrix4x4;
use crate::execution_engine::error::{ExecutionError, Result};
use num_complex::Complex64;
use rayon::prelude::*;

/// Apply a general two-qubit gate to a dense state vector
///
/// The state uses the little-endian convention (qubit k is bit k of the state
/// index). The 4×4 matrix uses the standard basis ordering |q1 q2⟩, i.e. matrix
/// index `m = (bit(qubit1) << 1) | bit(qubit2)`, matching the sparse kernel and
/// the matrices in `ferriq-gates` (for CNOT, `qubit1` is the control).
///
/// # Arguments
///
/// * `gate` - The 4x4 gate matrix
/// * `qubit1` - Index of the first qubit (control/target1)
/// * `qubit2` - Index of the second qubit (target/target2)
/// * `state` - The state vector to modify
/// * `use_parallel` - Whether to use parallel execution
/// * `parallel_threshold` - Minimum state size for parallel execution
pub fn apply_two_qubit_dense(
    gate: &Matrix4x4,
    qubit1: usize,
    qubit2: usize,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    let num_qubits = (n as f64).log2() as usize;

    if qubit1 >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: ferriq_core::QubitId::new(qubit1),
            max: num_qubits,
        });
    }

    if qubit2 >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: ferriq_core::QubitId::new(qubit2),
            max: num_qubits,
        });
    }

    if qubit1 == qubit2 {
        return Err(ExecutionError::GateApplicationFailed {
            gate: "two-qubit".to_string(),
            qubits: vec![
                ferriq_core::QubitId::new(qubit1),
                ferriq_core::QubitId::new(qubit2),
            ],
            reason: "Qubits must be different".to_string(),
        });
    }

    let (q_min, q_max) = if qubit1 < qubit2 {
        (qubit1, qubit2)
    } else {
        (qubit2, qubit1)
    };

    let num_subspaces = n / 4;

    if use_parallel && n >= parallel_threshold {
        let state_ptr_addr = state.as_mut_ptr() as usize;
        (0..num_subspaces).into_par_iter().for_each(|k| {
            let base = expand_two_qubit_base(k, q_min, q_max);
            let idx = get_two_qubit_indices(base, qubit1, qubit2);

            // Safety: each k maps to a disjoint 4-tuple of in-bounds indices
            unsafe {
                let state_ptr = state_ptr_addr as *mut Complex64;
                let a = [
                    *state_ptr.add(idx[0]),
                    *state_ptr.add(idx[1]),
                    *state_ptr.add(idx[2]),
                    *state_ptr.add(idx[3]),
                ];

                for (out_idx, &out) in idx.iter().enumerate() {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for (in_idx, &amp) in a.iter().enumerate() {
                        sum += gate[out_idx][in_idx] * amp;
                    }
                    *state_ptr.add(out) = sum;
                }
            }
        });
    } else {
        for k in 0..num_subspaces {
            let base = expand_two_qubit_base(k, q_min, q_max);
            let idx = get_two_qubit_indices(base, qubit1, qubit2);

            let a = [state[idx[0]], state[idx[1]], state[idx[2]], state[idx[3]]];

            for (out_idx, &out) in idx.iter().enumerate() {
                let mut sum = Complex64::new(0.0, 0.0);
                for (in_idx, &amp) in a.iter().enumerate() {
                    sum += gate[out_idx][in_idx] * amp;
                }
                state[out] = sum;
            }
        }
    }

    Ok(())
}

/// Expand a compact subspace counter into a base state index that has zero bits
/// at positions `q_min` and `q_max` (with `q_min < q_max`)
#[inline]
fn expand_two_qubit_base(k: usize, q_min: usize, q_max: usize) -> usize {
    let low = (1usize << q_min) - 1;
    let x = (k & low) | ((k & !low) << 1); // insert 0 at bit q_min
    let mid = (1usize << q_max) - 1;
    (x & mid) | ((x & !mid) << 1) // insert 0 at bit q_max
}

/// Get the 4 indices for a 2-qubit gate application
///
/// Index `m` of the returned array corresponds to matrix basis state
/// `m = (bit(qubit1) << 1) | bit(qubit2)` (first qubit is the most
/// significant bit of the matrix index, as in `ferriq-gates` matrices).
#[inline]
fn get_two_qubit_indices(base: usize, qubit1: usize, qubit2: usize) -> [usize; 4] {
    let mask1 = 1 << qubit1;
    let mask2 = 1 << qubit2;

    [
        base,                 // |q1=0, q2=0⟩
        base | mask2,         // |q1=0, q2=1⟩
        base | mask1,         // |q1=1, q2=0⟩
        base | mask1 | mask2, // |q1=1, q2=1⟩
    ]
}

/// Apply CNOT gate (optimized special case)
///
/// CNOT matrix:
/// ```text
/// 1 0 0 0
/// 0 1 0 0
/// 0 0 0 1
/// 0 0 1 0
/// ```
pub fn apply_cnot(
    control: usize,
    target: usize,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    let num_qubits = (n as f64).log2() as usize;

    if control >= num_qubits || target >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: ferriq_core::QubitId::new(control.max(target)),
            max: num_qubits,
        });
    }

    if control == target {
        return Err(ExecutionError::GateApplicationFailed {
            gate: "CNOT".to_string(),
            qubits: vec![
                ferriq_core::QubitId::new(control),
                ferriq_core::QubitId::new(target),
            ],
            reason: "Control and target must be different".to_string(),
        });
    }

    // Little-endian convention: qubit k corresponds to bit k of the state index
    let control_mask = 1 << control;
    let target_mask = 1 << target;

    if use_parallel && n >= parallel_threshold {
        let state_ptr_addr = state.as_mut_ptr() as usize;
        (0..n)
            .into_par_iter()
            .step_by(1)
            .filter(|&i| i & control_mask != 0 && i & target_mask == 0)
            .for_each(|i| unsafe {
                let state_ptr = state_ptr_addr as *mut Complex64;
                let j = i | target_mask;
                let temp = *state_ptr.add(i);
                *state_ptr.add(i) = *state_ptr.add(j);
                *state_ptr.add(j) = temp;
            });
    } else {
        for i in 0..n {
            if i & control_mask != 0 && i & target_mask == 0 {
                let j = i | target_mask;
                state.swap(i, j);
            }
        }
    }

    Ok(())
}

/// Apply CZ (Controlled-Z) gate (optimized special case)
///
/// CZ matrix:
/// ```text
/// 1 0 0  0
/// 0 1 0  0
/// 0 0 1  0
/// 0 0 0 -1
/// ```
pub fn apply_cz(
    control: usize,
    target: usize,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    let num_qubits = (n as f64).log2() as usize;

    if control >= num_qubits || target >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: ferriq_core::QubitId::new(control.max(target)),
            max: num_qubits,
        });
    }

    if control == target {
        return Err(ExecutionError::GateApplicationFailed {
            gate: "CZ".to_string(),
            qubits: vec![
                ferriq_core::QubitId::new(control),
                ferriq_core::QubitId::new(target),
            ],
            reason: "Control and target must be different".to_string(),
        });
    }

    // Little-endian convention: qubit k corresponds to bit k of the state index
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    let both_mask = control_mask | target_mask;

    if use_parallel && n >= parallel_threshold {
        state
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| *i & both_mask == both_mask)
            .for_each(|(_, amp)| {
                *amp = -*amp;
            });
    } else {
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            if i & both_mask == both_mask {
                state[i] = -state[i];
            }
        }
    }

    Ok(())
}

/// Apply SWAP gate (optimized special case)
///
/// SWAP matrix:
/// ```text
/// 1 0 0 0
/// 0 0 1 0
/// 0 1 0 0
/// 0 0 0 1
/// ```
pub fn apply_swap(
    qubit1: usize,
    qubit2: usize,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    let num_qubits = (n as f64).log2() as usize;

    if qubit1 >= num_qubits || qubit2 >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: ferriq_core::QubitId::new(qubit1.max(qubit2)),
            max: num_qubits,
        });
    }

    if qubit1 == qubit2 {
        return Ok(()); // SWAP on same qubit is identity
    }

    // Little-endian convention: qubit k corresponds to bit k of the state index
    let mask1 = 1 << qubit1;
    let mask2 = 1 << qubit2;

    if use_parallel && n >= parallel_threshold {
        let state_ptr_addr = state.as_mut_ptr() as usize;
        (0..n)
            .into_par_iter()
            .filter(|&i| {
                let bit1 = (i & mask1) != 0;
                let bit2 = (i & mask2) != 0;
                bit1 && !bit2
            })
            .for_each(|i| unsafe {
                let j = (i & !mask1) | mask2;
                let state_ptr = state_ptr_addr as *mut Complex64;
                let temp = *state_ptr.add(i);
                *state_ptr.add(i) = *state_ptr.add(j);
                *state_ptr.add(j) = temp;
            });
    } else {
        for i in 0..n {
            let bit1 = (i & mask1) != 0;
            let bit2 = (i & mask2) != 0;

            if bit1 && !bit2 {
                let j = (i & !mask1) | mask2;
                state.swap(i, j);
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_cnot() {
        // |00⟩ → |00⟩
        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        apply_cnot(0, 1, &mut state, false, usize::MAX).unwrap();

        assert_abs_diff_eq!(state[0].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, 0.0, epsilon = 1e-10);

        // Little-endian: index 1 = |q1=0, q0=1⟩. With control=0 set, the target
        // (qubit 1) flips: index 1 → index 3.
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        apply_cnot(0, 1, &mut state, false, usize::MAX).unwrap();

        assert_abs_diff_eq!(state[1].re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[3].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cz() {
        // |11⟩ → -|11⟩
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        apply_cz(0, 1, &mut state, false, usize::MAX).unwrap();

        assert_abs_diff_eq!(state[3].re, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_swap() {
        // |10⟩ → |01⟩
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        apply_swap(0, 1, &mut state, false, usize::MAX).unwrap();

        assert_abs_diff_eq!(state[1].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[2].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qubit_validation() {
        let mut state = vec![Complex64::new(1.0, 0.0); 4];

        let result = apply_cnot(10, 1, &mut state, false, usize::MAX);
        assert!(result.is_err());

        let result = apply_cnot(0, 0, &mut state, false, usize::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn test_cnot_parallel() {
        // control=0 set (index 1) flips target qubit 1: index 1 → index 3
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        apply_cnot(0, 1, &mut state, true, 0).unwrap(); // threshold=0 forces parallel
        assert_abs_diff_eq!(state[3].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cz_parallel() {
        // |11⟩ → -|11⟩ with parallel execution
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        apply_cz(0, 1, &mut state, true, 0).unwrap(); // threshold=0 forces parallel
        assert_abs_diff_eq!(state[3].re, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_swap_parallel() {
        // |10⟩ → |01⟩ with parallel execution
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        apply_swap(0, 1, &mut state, true, 0).unwrap();
        assert_abs_diff_eq!(state[1].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[2].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_swap_same_qubit_is_identity() {
        // SWAP(i, i) is identity
        let mut state = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
        apply_swap(0, 0, &mut state, false, usize::MAX).unwrap();
        assert_abs_diff_eq!(state[1].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_two_qubit_dense_identity() {
        // Identity 4x4 gate should leave state unchanged
        let identity: Matrix4x4 = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];
        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        apply_two_qubit_dense(&identity, 0, 1, &mut state, false, usize::MAX).unwrap();
        assert_abs_diff_eq!(state[0].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_two_qubit_dense_parallel() {
        let identity: Matrix4x4 = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];
        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        apply_two_qubit_dense(&identity, 0, 1, &mut state, true, 0).unwrap(); // threshold=0 forces parallel
        assert_abs_diff_eq!(state[0].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_two_qubit_dense_invalid_qubit() {
        let identity: Matrix4x4 = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];
        let mut state = vec![Complex64::new(1.0, 0.0); 4];
        // qubit1 out of bounds
        assert!(apply_two_qubit_dense(&identity, 5, 1, &mut state, false, usize::MAX).is_err());
        // qubit2 out of bounds
        assert!(apply_two_qubit_dense(&identity, 0, 5, &mut state, false, usize::MAX).is_err());
        // same qubit
        assert!(apply_two_qubit_dense(&identity, 0, 0, &mut state, false, usize::MAX).is_err());
    }

    #[test]
    fn test_cz_invalid_qubits() {
        let mut state = vec![Complex64::new(1.0, 0.0); 4];
        assert!(apply_cz(5, 1, &mut state, false, usize::MAX).is_err());
        assert!(apply_cz(0, 0, &mut state, false, usize::MAX).is_err());
    }

    #[test]
    fn test_swap_invalid_qubit() {
        let mut state = vec![Complex64::new(1.0, 0.0); 4];
        assert!(apply_swap(5, 1, &mut state, false, usize::MAX).is_err());
    }

    fn cnot_matrix() -> Matrix4x4 {
        // CNOT: |00>→|00>, |01>→|01>, |10>→|11>, |11>→|10>
        [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        ]
    }

    #[test]
    fn test_apply_two_qubit_dense_parallel_with_non_trivial_gate() {
        // Use a larger state (8 amplitudes = 3 qubits) so the parallel executor
        // actually processes work across chunks. Apply CNOT-like gate on qubits 0,1.
        // Start with |100> = index 4 (for 3-qubit big-endian: bit2=1, bit1=0, bit0=0)
        let mut state = vec![Complex64::new(0.0, 0.0); 8];
        state[4] = Complex64::new(1.0, 0.0); // |100>

        let cnot = cnot_matrix();
        apply_two_qubit_dense(&cnot, 0, 1, &mut state, true, 0).unwrap();

        // State norm should be preserved
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_general_dense_cnot_matches_specialized() {
        // The general 4x4 path with the CNOT matrix must agree with the
        // specialized apply_cnot kernel, including for non-adjacent qubits.
        let cnot = cnot_matrix();
        for &(control, target, n_qubits) in &[(0, 1, 2), (1, 0, 2), (0, 2, 3), (2, 0, 3), (1, 3, 4)]
        {
            let dim = 1usize << n_qubits;
            // Superposition with distinct amplitudes so any index mix-up shows up
            let mut general: Vec<Complex64> = (0..dim)
                .map(|i| Complex64::new(1.0 + i as f64, 0.5 * i as f64))
                .collect();
            let mut specialized = general.clone();

            apply_two_qubit_dense(&cnot, control, target, &mut general, false, usize::MAX).unwrap();
            apply_cnot(control, target, &mut specialized, false, usize::MAX).unwrap();

            for i in 0..dim {
                assert_abs_diff_eq!(general[i].re, specialized[i].re, epsilon = 1e-10);
                assert_abs_diff_eq!(general[i].im, specialized[i].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_general_dense_non_adjacent_qubits_cover_all_subspaces() {
        // Regression: the old enumeration skipped subspaces for non-adjacent
        // qubits. An X⊗X-style gate on qubits (0, 2) of a 3-qubit state must
        // move every amplitude.
        let o = Complex64::new(0.0, 0.0);
        let i = Complex64::new(1.0, 0.0);
        // X on both qubits: |q1 q2⟩ → |!q1 !q2⟩, i.e. anti-diagonal matrix
        let xx = [[o, o, o, i], [o, o, i, o], [o, i, o, o], [i, o, o, o]];

        let mut state: Vec<Complex64> = (0..8).map(|k| Complex64::new(k as f64, 0.0)).collect();
        apply_two_qubit_dense(&xx, 0, 2, &mut state, false, usize::MAX).unwrap();

        // Flipping bits 0 and 2 maps index k → k ^ 0b101
        for (k, amp) in state.iter().enumerate() {
            assert_abs_diff_eq!(amp.re, (k ^ 0b101) as f64, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_apply_two_qubit_qubit2_less_than_qubit1() {
        // Apply with qubit2 < qubit1 to test (q_min, q_max) order
        let identity: Matrix4x4 = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];
        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        // qubit2=0, qubit1=1 (reversed order)
        apply_two_qubit_dense(&identity, 1, 0, &mut state, false, usize::MAX).unwrap();
        assert_abs_diff_eq!(state[0].re, 1.0, epsilon = 1e-10);
    }
}
