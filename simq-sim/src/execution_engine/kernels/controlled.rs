//! Controlled gate application kernels

use super::Matrix2x2;
use crate::execution_engine::error::{ExecutionError, Result};
use num_complex::Complex64;
use rayon::prelude::*;

/// Apply a controlled single-qubit gate
///
/// Applies a single-qubit gate to the target qubit, conditioned on the control qubit being |1⟩.
pub fn apply_controlled_gate(
    control: usize,
    target: usize,
    gate: &Matrix2x2,
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
            gate: "controlled".to_string(),
            qubits: vec![
                simq_core::QubitId::new(control),
                simq_core::QubitId::new(target),
            ],
            reason: "Control and target must be different".to_string(),
        });
    }

    // Convert qubit indices to bit positions (big-endian: qubit 0 is MSB)
    let control_bit = num_qubits - 1 - control;
    let target_bit = num_qubits - 1 - target;
    let control_mask = 1 << control_bit;
    let target_stride = 1 << target_bit;

    if use_parallel && n >= parallel_threshold {
        apply_controlled_parallel(control_mask, target_stride, gate, state);
    } else {
        apply_controlled_sequential(control_mask, target_stride, gate, state);
    }

    Ok(())
}

#[inline]
fn apply_controlled_sequential(
    control_mask: usize,
    target_stride: usize,
    gate: &Matrix2x2,
    state: &mut [Complex64],
) {
    let n = state.len();
    let mut i = 0;

    while i < n {
        for j in 0..target_stride {
            let idx0 = i + j;
            let idx1 = idx0 + target_stride;

            // Only apply if control qubit is |1⟩
            // Check the control bit on idx0 (same as idx1 since they differ only in target bit)
            if idx0 & control_mask != 0 {
                let a = state[idx0];
                let b = state[idx1];

                state[idx0] = gate[0][0] * a + gate[0][1] * b;
                state[idx1] = gate[1][0] * a + gate[1][1] * b;
            }
        }
        i += target_stride * 2;
    }
}

#[inline]
fn apply_controlled_parallel(
    control_mask: usize,
    target_stride: usize,
    gate: &Matrix2x2,
    state: &mut [Complex64],
) {
    let state_ptr_addr = state.as_ptr() as usize;

    state.par_chunks_mut(target_stride * 2).for_each(|chunk| {
        for j in 0..target_stride.min(chunk.len().saturating_sub(target_stride)) {
            // Reconstruct pointer to avoid borrowing state in closure
            let state_ptr = state_ptr_addr as *const Complex64;
            let base_idx = chunk.as_ptr() as usize - state_ptr as usize;
            let idx0 = base_idx / std::mem::size_of::<Complex64>() + j;

            // Only apply if control qubit is |1⟩
            if idx0 & control_mask != 0 {
                let a = chunk[j];
                let b = chunk[j + target_stride];

                chunk[j] = gate[0][0] * a + gate[0][1] * b;
                chunk[j + target_stride] = gate[1][0] * a + gate[1][1] * b;
            }
        }
    });
}

/// Apply a multi-controlled gate (Toffoli, etc.)
///
/// Applies a single-qubit gate to the target qubit, conditioned on all control qubits being |1⟩.
pub fn apply_multi_controlled(
    controls: &[usize],
    target: usize,
    gate: &Matrix2x2,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    let num_qubits = (n as f64).log2() as usize;

    // Validate qubits
    for &control in controls {
        if control >= num_qubits {
            return Err(ExecutionError::QubitOutOfBounds {
                qubit: simq_core::QubitId::new(control),
                max: num_qubits,
            });
        }
        if control == target {
            return Err(ExecutionError::GateApplicationFailed {
                gate: "multi-controlled".to_string(),
                qubits: vec![
                    simq_core::QubitId::new(control),
                    simq_core::QubitId::new(target),
                ],
                reason: "Control and target must be different".to_string(),
            });
        }
    }

    if target >= num_qubits {
        return Err(ExecutionError::QubitOutOfBounds {
            qubit: simq_core::QubitId::new(target),
            max: num_qubits,
        });
    }

    // Build control mask
    let mut control_mask = 0;
    for &control in controls {
        control_mask |= 1 << control;
    }

    let target_stride = 1 << target;

    if use_parallel && n >= parallel_threshold {
        apply_multi_controlled_parallel(control_mask, controls.len(), target_stride, gate, state);
    } else {
        apply_multi_controlled_sequential(control_mask, controls.len(), target_stride, gate, state);
    }

    Ok(())
}

#[inline]
fn apply_multi_controlled_sequential(
    control_mask: usize,
    num_controls: usize,
    target_stride: usize,
    gate: &Matrix2x2,
    state: &mut [Complex64],
) {
    let n = state.len();
    let mut i = 0;

    while i < n {
        for j in 0..target_stride {
            let idx = i + j;
            // Check if all control qubits are |1⟩
            if (idx & control_mask).count_ones() as usize == num_controls {
                let idx0 = idx;
                let idx1 = idx + target_stride;

                let a = state[idx0];
                let b = state[idx1];

                state[idx0] = gate[0][0] * a + gate[0][1] * b;
                state[idx1] = gate[1][0] * a + gate[1][1] * b;
            }
        }
        i += target_stride * 2;
    }
}

#[inline]
fn apply_multi_controlled_parallel(
    control_mask: usize,
    num_controls: usize,
    target_stride: usize,
    gate: &Matrix2x2,
    state: &mut [Complex64],
) {
    let state_ptr_addr = state.as_mut_ptr() as usize;

    (0..state.len())
        .into_par_iter()
        .step_by(target_stride * 2)
        .for_each(|i| {
            let state_ptr = state_ptr_addr as *mut Complex64;
            for j in 0..target_stride {
                let idx = i + j;
                // Safety: we are checking bounds and using raw pointers to avoid borrow checker issues
                // with disjoint access patterns that the compiler can't see
                unsafe {
                    if idx + target_stride < state.len()
                        && (idx & control_mask).count_ones() as usize == num_controls
                    {
                        let a = *state_ptr.add(idx);
                        let b = *state_ptr.add(idx + target_stride);

                        *state_ptr.add(idx) = gate[0][0] * a + gate[0][1] * b;
                        *state_ptr.add(idx + target_stride) = gate[1][0] * a + gate[1][1] * b;
                    }
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_controlled_gate() {
        // Identity gate for testing
        let gate: Matrix2x2 = [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ]; // X gate

        // |11⟩ state
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        apply_controlled_gate(0, 1, &gate, &mut state, false, usize::MAX).unwrap();

        // Should flip target when control is 1
        assert_abs_diff_eq!(state[2].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[3].re, 0.0, epsilon = 1e-10);
    }
}
