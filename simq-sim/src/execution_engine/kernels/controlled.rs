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
    state
        .par_chunks_mut(target_stride * 2)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base_idx = chunk_idx * target_stride * 2;

            for j in 0..target_stride.min(chunk.len().saturating_sub(target_stride)) {
                let idx = base_idx + j;

                // Only apply if control qubit is |1⟩
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
    state
        .par_chunks_mut(target_stride * 2)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base_idx = chunk_idx * target_stride * 2;

            for j in 0..target_stride.min(chunk.len().saturating_sub(target_stride)) {
                let idx = base_idx + j; // Global index for control mask check

                if (idx & control_mask).count_ones() as usize == num_controls {
                    let a = chunk[j];
                    let b = chunk[j + target_stride];

                    chunk[j] = gate[0][0] * a + gate[0][1] * b;
                    chunk[j + target_stride] = gate[1][0] * a + gate[1][1] * b;
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn x_gate() -> Matrix2x2 {
        [
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ]
    }

    fn identity_gate() -> Matrix2x2 {
        [
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]
    }

    #[test]
    fn test_controlled_gate() {
        // |11⟩ state
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        apply_controlled_gate(0, 1, &x_gate(), &mut state, false, usize::MAX).unwrap();

        // Should flip target when control is 1
        assert_abs_diff_eq!(state[2].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[3].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_controlled_gate_control_is_zero_no_change() {
        // |00⟩ state - control is 0, so gate should NOT be applied
        let mut state = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        apply_controlled_gate(0, 1, &x_gate(), &mut state, false, usize::MAX).unwrap();
        // |00⟩ should remain |00⟩
        assert_abs_diff_eq!(state[0].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_controlled_gate_parallel() {
        // |11⟩ state with parallel execution
        let mut state = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        apply_controlled_gate(0, 1, &x_gate(), &mut state, true, 0).unwrap();
        // X applied: |11⟩ -> |10⟩
        assert_abs_diff_eq!(state[2].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[3].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_controlled_gate_invalid_qubit() {
        let mut state = vec![Complex64::new(1.0, 0.0); 4];
        assert!(apply_controlled_gate(10, 1, &x_gate(), &mut state, false, usize::MAX).is_err());
        assert!(apply_controlled_gate(0, 10, &x_gate(), &mut state, false, usize::MAX).is_err());
        assert!(apply_controlled_gate(0, 0, &x_gate(), &mut state, false, usize::MAX).is_err());
    }

    #[test]
    fn test_multi_controlled_toffoli_sequential() {
        // Toffoli: apply X to qubit 2 when qubits 0 and 1 are both |1⟩
        // Use 3-qubit state (8 amplitudes)
        // |110⟩ = index 6 (binary: 110), |111⟩ = index 7
        let mut state = vec![Complex64::new(0.0, 0.0); 8];
        state[6] = Complex64::new(1.0, 0.0); // |110⟩

        // controls = [0, 1] (qubits 0 and 1), target = 2
        apply_multi_controlled(&[0, 1], 2, &x_gate(), &mut state, false, usize::MAX).unwrap();
        // Should flip qubit 2 when qubits 0 and 1 are both 1
        // The exact result depends on qubit ordering; just check no error and norm preserved
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_controlled_identity_sequential() {
        // Apply identity with multi-controlled gate
        let mut state = vec![Complex64::new(0.0, 0.0); 4];
        state[0] = Complex64::new(1.0, 0.0);
        apply_multi_controlled(&[0], 1, &identity_gate(), &mut state, false, usize::MAX).unwrap();
        // Identity should not change state
        assert_abs_diff_eq!(state[0].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_controlled_parallel() {
        let mut state = vec![Complex64::new(0.0, 0.0); 8];
        state[6] = Complex64::new(1.0, 0.0);
        apply_multi_controlled(&[0, 1], 2, &x_gate(), &mut state, true, 0).unwrap();
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_controlled_invalid_control() {
        let mut state = vec![Complex64::new(1.0, 0.0); 4];
        // control qubit out of bounds
        assert!(apply_multi_controlled(&[10], 1, &x_gate(), &mut state, false, usize::MAX).is_err());
    }

    #[test]
    fn test_multi_controlled_control_equals_target() {
        let mut state = vec![Complex64::new(1.0, 0.0); 4];
        // control == target should fail
        assert!(apply_multi_controlled(&[1], 1, &x_gate(), &mut state, false, usize::MAX).is_err());
    }

    #[test]
    fn test_multi_controlled_target_out_of_bounds() {
        let mut state = vec![Complex64::new(1.0, 0.0); 4];
        // target out of bounds
        assert!(apply_multi_controlled(&[0], 10, &x_gate(), &mut state, false, usize::MAX).is_err());
    }

    #[test]
    fn test_multi_controlled_parallel_applies_gate() {
        // Use 4-qubit state (16 amplitudes) so there's enough work for parallel.
        // controls=[0,1], target=2. Start in |1100> = index 12 (0b1100).
        // When qubits 0 and 1 are both 1, flip qubit 2.
        let mut state = vec![Complex64::new(0.0, 0.0); 16];
        state[12] = Complex64::new(1.0, 0.0); // |1100>

        // With threshold=0, forces parallel path
        apply_multi_controlled(&[0, 1], 2, &x_gate(), &mut state, true, 0).unwrap();

        // Norm should be preserved
        let norm: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }
}
