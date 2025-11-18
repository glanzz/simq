//! Two-qubit gate application kernels

use num_complex::Complex64;
use rayon::prelude::*;
use super::Matrix4x4;
use crate::execution_engine::error::{ExecutionError, Result};

/// Apply a general two-qubit gate to a dense state vector
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
            qubit: simq_core::QubitId::new(qubit1),
            max: num_qubits,
        });
    }

    if qubit2 >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: simq_core::QubitId::new(qubit2),
            max: num_qubits,
        });
    }

    if qubit1 == qubit2 {
        return Err(ExecutionError::GateApplicationFailed {
            gate: "two-qubit".to_string(),
            qubits: vec![simq_core::QubitId::new(qubit1), simq_core::QubitId::new(qubit2)],
            reason: "Qubits must be different".to_string(),
        });
    }

    let (q_min, q_max) = if qubit1 < qubit2 {
        (qubit1, qubit2)
    } else {
        (qubit2, qubit1)
    };

    let stride_min = 1 << q_min;
    let stride_max = 1 << q_max;

    if use_parallel && n >= parallel_threshold {
        apply_two_qubit_parallel(gate, stride_min, stride_max, state, qubit1, qubit2);
    } else {
        apply_two_qubit_sequential(gate, stride_min, stride_max, state, qubit1, qubit2);
    }

    Ok(())
}

/// Apply two-qubit gate sequentially
#[inline]
fn apply_two_qubit_sequential(
    gate: &Matrix4x4,
    stride_min: usize,
    stride_max: usize,
    state: &mut [Complex64],
    qubit1: usize,
    qubit2: usize,
) {
    let n = state.len();
    let mut i = 0;

    while i < n {
        for j in 0..stride_min {
            for k in 0..stride_min {
                if (i + j) & stride_min == 0 && (i + j + k * stride_max) & stride_max == 0 {
                    let base = i + j;

                    // Calculate the 4 indices for the 2-qubit subspace
                    let idx = get_two_qubit_indices(base, qubit1, qubit2);

                    // Load amplitudes
                    let a = [
                        state[idx[0]],
                        state[idx[1]],
                        state[idx[2]],
                        state[idx[3]],
                    ];

                    // Apply gate matrix
                    for (out_idx, out) in idx.iter().enumerate() {
                        let mut sum = Complex64::new(0.0, 0.0);
                        for (in_idx, &amp) in a.iter().enumerate() {
                            sum += gate[out_idx][in_idx] * amp;
                        }
                        state[*out] = sum;
                    }
                }
            }
        }
        i += stride_max * 2;
    }
}

/// Apply two-qubit gate in parallel
#[inline]
fn apply_two_qubit_parallel(
    gate: &Matrix4x4,
    stride_min: usize,
    stride_max: usize,
    state: &mut [Complex64],
    qubit1: usize,
    qubit2: usize,
) {
    let chunk_size = stride_max * 2;
    let num_chunks = state.len() / chunk_size;

    (0..num_chunks)
        .into_par_iter()
        .for_each(|chunk_idx| {
            let base_offset = chunk_idx * chunk_size;

            for j in 0..stride_min {
                for k in 0..stride_min {
                    let base = base_offset + j;

                    if (base & stride_min == 0) && ((base + k * stride_max) & stride_max == 0) {
                        let idx = get_two_qubit_indices(base, qubit1, qubit2);

                        // Safety: indices are computed to be within bounds
                        unsafe {
                            let a = [
                                *state.get_unchecked(idx[0]),
                                *state.get_unchecked(idx[1]),
                                *state.get_unchecked(idx[2]),
                                *state.get_unchecked(idx[3]),
                            ];

                            let state_ptr = state.as_mut_ptr();
                            for (out_idx, &out) in idx.iter().enumerate() {
                                let mut sum = Complex64::new(0.0, 0.0);
                                for (in_idx, &amp) in a.iter().enumerate() {
                                    sum += gate[out_idx][in_idx] * amp;
                                }
                                *state_ptr.add(out) = sum;
                            }
                        }
                    }
                }
            }
        });
}

/// Get the 4 indices for a 2-qubit gate application
#[inline]
fn get_two_qubit_indices(base: usize, qubit1: usize, qubit2: usize) -> [usize; 4] {
    let mask1 = 1 << qubit1;
    let mask2 = 1 << qubit2;

    [
        base,                    // |00⟩
        base | mask1,           // |10⟩ or |01⟩ depending on qubit order
        base | mask2,           // |01⟩ or |10⟩ depending on qubit order
        base | mask1 | mask2,   // |11⟩
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
            qubit: simq_core::QubitId::new(control.max(target)),
            max: num_qubits,
        });
    }

    if control == target {
        return Err(ExecutionError::GateApplicationFailed {
            gate: "CNOT".to_string(),
            qubits: vec![simq_core::QubitId::new(control), simq_core::QubitId::new(target)],
            reason: "Control and target must be different".to_string(),
        });
    }

    let control_mask = 1 << control;
    let target_mask = 1 << target;

    if use_parallel && n >= parallel_threshold {
        (0..n)
            .into_par_iter()
            .step_by(1)
            .filter(|&i| i & control_mask != 0 && i & target_mask == 0)
            .for_each(|i| unsafe {
                let state_ptr = state.as_mut_ptr();
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
            qubit: simq_core::QubitId::new(control.max(target)),
            max: num_qubits,
        });
    }

    if control == target {
        return Err(ExecutionError::GateApplicationFailed {
            gate: "CZ".to_string(),
            qubits: vec![simq_core::QubitId::new(control), simq_core::QubitId::new(target)],
            reason: "Control and target must be different".to_string(),
        });
    }

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
            qubit: simq_core::QubitId::new(qubit1.max(qubit2)),
            max: num_qubits,
        });
    }

    if qubit1 == qubit2 {
        return Ok(()); // SWAP on same qubit is identity
    }

    let mask1 = 1 << qubit1;
    let mask2 = 1 << qubit2;

    if use_parallel && n >= parallel_threshold {
        (0..n)
            .into_par_iter()
            .filter(|&i| {
                let bit1 = (i & mask1) != 0;
                let bit2 = (i & mask2) != 0;
                bit1 && !bit2
            })
            .for_each(|i| unsafe {
                let j = (i & !mask1) | mask2;
                let state_ptr = state.as_mut_ptr();
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

        // |10⟩ → |11⟩
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        apply_cnot(0, 1, &mut state, false, usize::MAX).unwrap();

        assert_abs_diff_eq!(state[2].re, 0.0, epsilon = 1e-10);
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
}
