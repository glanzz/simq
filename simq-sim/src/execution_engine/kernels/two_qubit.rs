//! Two-qubit gate application kernels
//!
//! All kernels enumerate the n/4 two-qubit subspaces directly instead of
//! scanning all n indices with a filter, and parallel execution uses
//! coarse-grained rayon tasks (`with_min_len`) so fork/join overhead stays
//! negligible (issue #76).

use super::Matrix4x4;
use crate::execution_engine::error::{ExecutionError, Result};
use num_complex::Complex64;
use rayon::prelude::*;

/// Minimum subspaces per rayon task. Each subspace touches 4 amplitudes, so
/// this keeps tasks at ≥ 128 KiB of state.
const MIN_PAR_SUBSPACES: usize = 1 << 11;

#[inline]
fn check_two_qubits(qubit1: usize, qubit2: usize, n: usize, gate: &str) -> Result<()> {
    let num_qubits = n.trailing_zeros() as usize;

    if qubit1 >= num_qubits || qubit2 >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: simq_core::QubitId::new(qubit1.max(qubit2)),
            max: num_qubits,
        });
    }

    if qubit1 == qubit2 {
        return Err(ExecutionError::GateApplicationFailed {
            gate: gate.to_string(),
            qubits: vec![
                simq_core::QubitId::new(qubit1),
                simq_core::QubitId::new(qubit2),
            ],
            reason: "Qubits must be different".to_string(),
        });
    }

    Ok(())
}

/// Apply a general two-qubit gate to a dense state vector
///
/// The state uses the little-endian convention (qubit k is bit k of the state
/// index). The 4×4 matrix uses the standard basis ordering |q1 q2⟩, i.e. matrix
/// index `m = (bit(qubit1) << 1) | bit(qubit2)`, matching the sparse kernel and
/// the matrices in `simq-gates` (for CNOT, `qubit1` is the control).
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
    check_two_qubits(qubit1, qubit2, n, "two-qubit")?;

    let (q_min, q_max) = if qubit1 < qubit2 {
        (qubit1, qubit2)
    } else {
        (qubit2, qubit1)
    };

    let num_subspaces = n / 4;

    let apply_at = |state_ptr: *mut Complex64, k: usize| {
        let base = expand_two_qubit_base(k, q_min, q_max);
        let idx = get_two_qubit_indices(base, qubit1, qubit2);

        // Safety: each k maps to a disjoint 4-tuple of in-bounds indices
        unsafe {
            let a = [
                *state_ptr.add(idx[0]),
                *state_ptr.add(idx[1]),
                *state_ptr.add(idx[2]),
                *state_ptr.add(idx[3]),
            ];

            for (out_idx, &out) in idx.iter().enumerate() {
                let g = &gate[out_idx];
                *state_ptr.add(out) = g[0] * a[0] + g[1] * a[1] + g[2] * a[2] + g[3] * a[3];
            }
        }
    };

    if use_parallel && n >= parallel_threshold && num_subspaces >= MIN_PAR_SUBSPACES * 2 {
        let state_ptr_addr = state.as_mut_ptr() as usize;
        (0..num_subspaces)
            .into_par_iter()
            .with_min_len(MIN_PAR_SUBSPACES)
            .for_each(|k| apply_at(state_ptr_addr as *mut Complex64, k));
    } else {
        let state_ptr = state.as_mut_ptr();
        for k in 0..num_subspaces {
            apply_at(state_ptr, k);
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
/// significant bit of the matrix index, as in `simq-gates` matrices).
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
    check_two_qubits(control, target, n, "CNOT")?;

    // Little-endian convention: qubit k corresponds to bit k of the state index
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    let (q_min, q_max) = if control < target {
        (control, target)
    } else {
        (target, control)
    };

    let num_subspaces = n / 4;

    if use_parallel && n >= parallel_threshold && num_subspaces >= MIN_PAR_SUBSPACES * 2 {
        let state_ptr_addr = state.as_mut_ptr() as usize;
        (0..num_subspaces)
            .into_par_iter()
            .with_min_len(MIN_PAR_SUBSPACES)
            .for_each(|k| {
                let base = expand_two_qubit_base(k, q_min, q_max);
                let i = base | control_mask;
                let j = i | target_mask;
                // Safety: each k maps to a disjoint in-bounds index pair
                unsafe {
                    let state_ptr = state_ptr_addr as *mut Complex64;
                    std::ptr::swap(state_ptr.add(i), state_ptr.add(j));
                }
            });
    } else {
        for k in 0..num_subspaces {
            let base = expand_two_qubit_base(k, q_min, q_max);
            let i = base | control_mask;
            state.swap(i, i | target_mask);
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
    check_two_qubits(control, target, n, "CZ")?;

    // Little-endian convention: qubit k corresponds to bit k of the state index
    let both_mask = (1usize << control) | (1 << target);
    let (q_min, q_max) = if control < target {
        (control, target)
    } else {
        (target, control)
    };

    let num_subspaces = n / 4;

    if use_parallel && n >= parallel_threshold && num_subspaces >= MIN_PAR_SUBSPACES * 2 {
        let state_ptr_addr = state.as_mut_ptr() as usize;
        (0..num_subspaces)
            .into_par_iter()
            .with_min_len(MIN_PAR_SUBSPACES)
            .for_each(|k| {
                let i = expand_two_qubit_base(k, q_min, q_max) | both_mask;
                // Safety: each k maps to a disjoint in-bounds index
                unsafe {
                    let state_ptr = state_ptr_addr as *mut Complex64;
                    *state_ptr.add(i) = -*state_ptr.add(i);
                }
            });
    } else {
        for k in 0..num_subspaces {
            let i = expand_two_qubit_base(k, q_min, q_max) | both_mask;
            state[i] = -state[i];
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
    let num_qubits = n.trailing_zeros() as usize;

    if qubit1 >= num_qubits || qubit2 >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: simq_core::QubitId::new(qubit1.max(qubit2)),
            max: num_qubits,
        });
    }

    if qubit1 == qubit2 {
        return Ok(()); // SWAP on same qubit is identity
    }

    // Little-endian convention: qubit k corresponds to bit k of the state index
    let mask1 = 1 << qubit1;
    let mask2 = 1 << qubit2;
    let (q_min, q_max) = if qubit1 < qubit2 {
        (qubit1, qubit2)
    } else {
        (qubit2, qubit1)
    };

    let num_subspaces = n / 4;

    if use_parallel && n >= parallel_threshold && num_subspaces >= MIN_PAR_SUBSPACES * 2 {
        let state_ptr_addr = state.as_mut_ptr() as usize;
        (0..num_subspaces)
            .into_par_iter()
            .with_min_len(MIN_PAR_SUBSPACES)
            .for_each(|k| {
                let base = expand_two_qubit_base(k, q_min, q_max);
                // Safety: each k maps to a disjoint in-bounds index pair
                unsafe {
                    let state_ptr = state_ptr_addr as *mut Complex64;
                    std::ptr::swap(state_ptr.add(base | mask1), state_ptr.add(base | mask2));
                }
            });
    } else {
        for k in 0..num_subspaces {
            let base = expand_two_qubit_base(k, q_min, q_max);
            state.swap(base | mask1, base | mask2);
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
        apply_cnot(0, 1, &mut state, true, 0).unwrap();
        assert_abs_diff_eq!(state[1].re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[3].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_general_two_qubit_gate() {
        // CNOT expressed as a general 4x4 matrix (basis |q1 q2⟩)
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        let cnot: Matrix4x4 = [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]];

        // |q1=1, q0=0⟩ = index 2, control = qubit 1 → target qubit 0 flips → index 3
        let mut state = vec![z, z, o, z];
        apply_two_qubit_dense(&cnot, 1, 0, &mut state, false, usize::MAX).unwrap();
        assert_abs_diff_eq!(state[2].re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[3].re, 1.0, epsilon = 1e-10);
    }

    /// Large state: parallel and sequential kernels must agree exactly for
    /// CNOT/CZ/SWAP and the general two-qubit path.
    #[test]
    fn test_parallel_matches_sequential_large_state() {
        let n = 1 << 15;
        let base: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), (i as f64).cos()) / (n as f64).sqrt())
            .collect();

        let sqrt_i = Complex64::new(0.5, 0.5);
        let sqrt_mi = Complex64::new(0.5, -0.5);
        let z = Complex64::new(0.0, 0.0);
        let o = Complex64::new(1.0, 0.0);
        // sqrt(iSWAP)-like unitary to exercise the general kernel
        let gate: Matrix4x4 = [
            [o, z, z, z],
            [z, sqrt_i, sqrt_mi, z],
            [z, sqrt_mi, sqrt_i, z],
            [z, z, z, o],
        ];

        for (q1, q2) in [(0usize, 1usize), (2, 9), (14, 3), (13, 14)] {
            let mut seq = base.clone();
            let mut par = base.clone();
            apply_two_qubit_dense(&gate, q1, q2, &mut seq, false, usize::MAX).unwrap();
            apply_two_qubit_dense(&gate, q1, q2, &mut par, true, 0).unwrap();
            assert_eq!(seq, par, "general ({}, {}) mismatch", q1, q2);

            let mut seq = base.clone();
            let mut par = base.clone();
            apply_cnot(q1, q2, &mut seq, false, usize::MAX).unwrap();
            apply_cnot(q1, q2, &mut par, true, 0).unwrap();
            assert_eq!(seq, par, "cnot ({}, {}) mismatch", q1, q2);

            let mut seq = base.clone();
            let mut par = base.clone();
            apply_cz(q1, q2, &mut seq, false, usize::MAX).unwrap();
            apply_cz(q1, q2, &mut par, true, 0).unwrap();
            assert_eq!(seq, par, "cz ({}, {}) mismatch", q1, q2);

            let mut seq = base.clone();
            let mut par = base.clone();
            apply_swap(q1, q2, &mut seq, false, usize::MAX).unwrap();
            apply_swap(q1, q2, &mut par, true, 0).unwrap();
            assert_eq!(seq, par, "swap ({}, {}) mismatch", q1, q2);
        }
    }
}
