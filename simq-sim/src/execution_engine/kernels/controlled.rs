//! Controlled gate application kernels

use num_complex::Complex64;
use rayon::prelude::*;
use crate::execution_engine::error::{ExecutionError, Result};
use super::Matrix2x2;

// Safety wrapper for raw pointers to allow Send+Sync
// This is safe because we partition the work to ensure no data races
#[derive(Copy, Clone)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

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
    let state_len = state.len();
    let state_ptr = state.as_ptr() as usize;
    
    state
        .par_chunks_mut(target_stride * 2)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base_idx = chunk_idx * target_stride * 2;
            
            for j in 0..target_stride.min(chunk.len().saturating_sub(target_stride)) {
                let idx = base_idx + j;

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
    let gate_copy = *gate; // Copy the gate matrix
    
    // Collect indices that need to be modified
    let modifications: Vec<(usize, Complex64, Complex64)> = (0..state.len())
        .into_par_iter()
        .step_by(target_stride * 2)
        .flat_map(|i| {
            let mut mods = Vec::new();
            for j in 0..target_stride {
                let idx = i + j;
                if idx + target_stride < state.len() &&
                   (idx & control_mask).count_ones() as usize == num_controls {
                    let a = state[idx];
                    let b = state[idx + target_stride];
                    let new_a = gate_copy[0][0] * a + gate_copy[0][1] * b;
                    let new_b = gate_copy[1][0] * a + gate_copy[1][1] * b;
                    mods.push((idx, new_a, new_b));
                }
            }
            mods
        })
        .collect();
    
    // Apply modifications sequentially
    for (idx, new_a, new_b) in modifications {
        state[idx] = new_a;
        state[idx + target_stride] = new_b;
    }
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
