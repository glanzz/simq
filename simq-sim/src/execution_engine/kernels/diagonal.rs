//! Diagonal gate application kernels (phase gates, etc.)

use crate::execution_engine::error::Result;
use num_complex::Complex64;
use rayon::prelude::*;

/// Apply a diagonal gate (phase gate) to specific qubits
///
/// Diagonal gates only modify phase, so they're much faster than general gates.
pub fn apply_diagonal_gate(
    qubits: &[usize],
    phases: &[Complex64],
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();

    if phases.len() != (1 << qubits.len()) {
        return Err(crate::execution_engine::error::ExecutionError::InvalidGateMatrix {
            gate: "diagonal".to_string(),
            reason: format!(
                "Phase array size {} doesn't match 2^{} = {}",
                phases.len(),
                qubits.len(),
                1 << qubits.len()
            ),
        });
    }

    if use_parallel && n >= parallel_threshold {
        state.par_iter_mut().enumerate().for_each(|(i, amp)| {
            let phase_idx = get_phase_index(i, qubits);
            *amp *= phases[phase_idx];
        });
    } else {
        for (i, amp) in state.iter_mut().enumerate() {
            let phase_idx = get_phase_index(i, qubits);
            *amp *= phases[phase_idx];
        }
    }

    Ok(())
}

/// Apply a simple phase gate to a single qubit
pub fn apply_phase_gate(
    qubit: usize,
    phase: Complex64,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    let mask = 1 << qubit;

    if use_parallel && n >= parallel_threshold {
        state
            .par_iter_mut()
            .enumerate()
            .filter(|(i, _)| *i & mask != 0)
            .for_each(|(_, amp)| {
                *amp *= phase;
            });
    } else {
        for i in 0..n {
            if i & mask != 0 {
                state[i] *= phase;
            }
        }
    }

    Ok(())
}

#[inline]
fn get_phase_index(state_index: usize, qubits: &[usize]) -> usize {
    let mut phase_idx = 0;
    for (bit_pos, &qubit) in qubits.iter().enumerate() {
        if state_index & (1 << qubit) != 0 {
            phase_idx |= 1 << bit_pos;
        }
    }
    phase_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_phase_gate() {
        let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];

        let phase = Complex64::new(0.0, 1.0); // i
        apply_phase_gate(0, phase, &mut state, false, usize::MAX).unwrap();

        assert_abs_diff_eq!(state[0].re, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].im, 1.0, epsilon = 1e-10);
    }
}
