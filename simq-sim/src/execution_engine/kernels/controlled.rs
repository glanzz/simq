//! Controlled gate application kernels

use num_complex::Complex64;
use rayon::prelude::*;
use crate::execution_engine::error::{ExecutionError, Result};
use super::Matrix2x2;

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
            qubits: vec![simq_core::QubitId::new(control), simq_core::QubitId::new(target)],
            reason: "Control and target must be different".to_string(),
        });
    }

    let control_mask = 1 << control;
    let target_stride = 1 << target;

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
            let idx = i + j;
            // Only apply if control qubit is |1⟩
            if idx & control_mask != 0 {
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
fn apply_controlled_parallel(
    control_mask: usize,
    target_stride: usize,
    gate: &Matrix2x2,
    state: &mut [Complex64],
) {
    state
        .par_chunks_mut(target_stride * 2)
        .for_each(|chunk| {
            for j in 0..target_stride.min(chunk.len() - target_stride) {
                let base_idx = chunk.as_ptr() as usize - state.as_ptr() as usize;
                let idx = base_idx / std::mem::size_of::<Complex64>() + j;

                if idx & control_mask != 0 {
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
                qubits: vec![simq_core::QubitId::new(control), simq_core::QubitId::new(target)],
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
    // TODO: Parallel version has borrow checker issues - use sequential for now
    apply_multi_controlled_sequential(control_mask, num_controls, target_stride, gate, state);
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
