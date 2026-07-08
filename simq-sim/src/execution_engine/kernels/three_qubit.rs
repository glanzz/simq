//! Three-qubit gate application kernels
//!
//! Provides a general dense kernel for arbitrary 8×8 (three-qubit) gates such
//! as Toffoli (CCNOT) and Fredkin (CSWAP).

use super::Matrix8x8;
use crate::execution_engine::error::{ExecutionError, Result};
use num_complex::Complex64;
use rayon::prelude::*;

/// Apply a general three-qubit gate to a dense state vector
///
/// The state uses the little-endian convention (qubit k is bit k of the state
/// index). The 8×8 matrix uses the standard basis ordering |q1 q2 q3⟩, i.e.
/// matrix index `m = (bit(qubit1) << 2) | (bit(qubit2) << 1) | bit(qubit3)`,
/// matching the matrices in `simq-gates` (for Toffoli, `qubit1`/`qubit2` are
/// the controls and `qubit3` is the target).
///
/// # Arguments
///
/// * `gate` - The 8x8 gate matrix
/// * `qubit1` - Index of the first qubit
/// * `qubit2` - Index of the second qubit
/// * `qubit3` - Index of the third qubit
/// * `state` - The state vector to modify
/// * `use_parallel` - Whether to use parallel execution
/// * `parallel_threshold` - Minimum state size for parallel execution
pub fn apply_three_qubit_dense(
    gate: &Matrix8x8,
    qubit1: usize,
    qubit2: usize,
    qubit3: usize,
    state: &mut [Complex64],
    use_parallel: bool,
    parallel_threshold: usize,
) -> Result<()> {
    let n = state.len();
    let num_qubits = (n as f64).log2() as usize;

    for &q in &[qubit1, qubit2, qubit3] {
        if q >= num_qubits {
            return Err(ExecutionError::QubitOutOfBounds {
                qubit: simq_core::QubitId::new(q),
                max: num_qubits,
            });
        }
    }

    if qubit1 == qubit2 || qubit1 == qubit3 || qubit2 == qubit3 {
        return Err(ExecutionError::GateApplicationFailed {
            gate: "three-qubit".to_string(),
            qubits: vec![
                simq_core::QubitId::new(qubit1),
                simq_core::QubitId::new(qubit2),
                simq_core::QubitId::new(qubit3),
            ],
            reason: "Qubits must be different".to_string(),
        });
    }

    let mut sorted = [qubit1, qubit2, qubit3];
    sorted.sort_unstable();

    let num_subspaces = n / 8;

    // Each subspace touches 8 amplitudes; keep rayon tasks ≥ 128 KiB of state
    // so fork/join overhead stays negligible (issue #76).
    const MIN_PAR_SUBSPACES: usize = 1 << 10;

    if use_parallel && n >= parallel_threshold && num_subspaces >= MIN_PAR_SUBSPACES * 2 {
        let state_ptr_addr = state.as_mut_ptr() as usize;
        (0..num_subspaces)
            .into_par_iter()
            .with_min_len(MIN_PAR_SUBSPACES)
            .for_each(|k| {
                let base = expand_three_qubit_base(k, &sorted);
                let idx = get_three_qubit_indices(base, qubit1, qubit2, qubit3);

                // Safety: each k maps to a disjoint 8-tuple of in-bounds indices
                unsafe {
                    let state_ptr = state_ptr_addr as *mut Complex64;
                    let mut a = [Complex64::new(0.0, 0.0); 8];
                    for (m, &i) in idx.iter().enumerate() {
                        a[m] = *state_ptr.add(i);
                    }

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
            let base = expand_three_qubit_base(k, &sorted);
            let idx = get_three_qubit_indices(base, qubit1, qubit2, qubit3);

            let mut a = [Complex64::new(0.0, 0.0); 8];
            for (m, &i) in idx.iter().enumerate() {
                a[m] = state[i];
            }

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
/// at the three (ascending) positions in `sorted`
#[inline]
fn expand_three_qubit_base(k: usize, sorted: &[usize; 3]) -> usize {
    let mut x = k;
    for &q in sorted.iter() {
        let mask = (1usize << q) - 1;
        x = (x & mask) | ((x & !mask) << 1); // insert 0 at bit q
    }
    x
}

/// Get the 8 indices for a 3-qubit gate application
///
/// Index `m` of the returned array corresponds to matrix basis state
/// `m = (bit(qubit1) << 2) | (bit(qubit2) << 1) | bit(qubit3)`.
#[inline]
fn get_three_qubit_indices(base: usize, qubit1: usize, qubit2: usize, qubit3: usize) -> [usize; 8] {
    let mask1 = 1 << qubit1;
    let mask2 = 1 << qubit2;
    let mask3 = 1 << qubit3;

    let mut idx = [0usize; 8];
    for (m, slot) in idx.iter_mut().enumerate() {
        let mut i = base;
        if m & 0b100 != 0 {
            i |= mask1;
        }
        if m & 0b010 != 0 {
            i |= mask2;
        }
        if m & 0b001 != 0 {
            i |= mask3;
        }
        *slot = i;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn toffoli_matrix() -> Matrix8x8 {
        let o = Complex64::new(0.0, 0.0);
        let l = Complex64::new(1.0, 0.0);
        let mut m = [[o; 8]; 8];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = l;
        }
        // Swap |110⟩ and |111⟩
        m[6][6] = o;
        m[7][7] = o;
        m[6][7] = l;
        m[7][6] = l;
        m
    }

    #[test]
    fn test_toffoli_flips_target_only_when_both_controls_set() {
        let gate = toffoli_matrix();

        // Controls on qubits 0 and 1, target on qubit 2 (little-endian state).
        // |011⟩ = index 3 (q0=1, q1=1, q2=0) → target flips → index 7.
        let mut state = vec![Complex64::new(0.0, 0.0); 8];
        state[3] = Complex64::new(1.0, 0.0);

        apply_three_qubit_dense(&gate, 0, 1, 2, &mut state, false, usize::MAX).unwrap();

        assert_abs_diff_eq!(state[3].re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[7].re, 1.0, epsilon = 1e-10);

        // Only one control set: index 1 (q0=1) stays put
        let mut state = vec![Complex64::new(0.0, 0.0); 8];
        state[1] = Complex64::new(1.0, 0.0);
        apply_three_qubit_dense(&gate, 0, 1, 2, &mut state, false, usize::MAX).unwrap();
        assert_abs_diff_eq!(state[1].re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_three_qubit_parallel_matches_sequential() {
        let gate = toffoli_matrix();
        // 4-qubit state, gate on non-contiguous qubits (3, 0, 2)
        let mut seq: Vec<Complex64> = (0..16)
            .map(|i| Complex64::new(1.0 + i as f64, 0.25 * i as f64))
            .collect();
        let mut par = seq.clone();

        apply_three_qubit_dense(&gate, 3, 0, 2, &mut seq, false, usize::MAX).unwrap();
        apply_three_qubit_dense(&gate, 3, 0, 2, &mut par, true, 0).unwrap();

        for i in 0..16 {
            assert_abs_diff_eq!(seq[i].re, par[i].re, epsilon = 1e-10);
            assert_abs_diff_eq!(seq[i].im, par[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_three_qubit_validation() {
        let gate = toffoli_matrix();
        let mut state = vec![Complex64::new(1.0, 0.0); 8];
        // out of bounds
        assert!(apply_three_qubit_dense(&gate, 0, 1, 5, &mut state, false, usize::MAX).is_err());
        // duplicate qubits
        assert!(apply_three_qubit_dense(&gate, 0, 1, 1, &mut state, false, usize::MAX).is_err());
    }
}
