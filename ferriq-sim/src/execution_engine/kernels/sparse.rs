//! Sparse state gate application kernels

use super::Matrix2x2;
use crate::execution_engine::error::Result;
use ahash::AHashMap;
use num_complex::Complex64;

/// Apply a single-qubit gate to a sparse state
pub fn apply_single_qubit_sparse(
    gate: &Matrix2x2,
    qubit: usize,
    amplitudes: &mut AHashMap<u64, Complex64>,
    _num_qubits: usize,
) -> Result<()> {
    let mask = 1u64 << qubit;
    let mut new_amplitudes: AHashMap<u64, Complex64> = AHashMap::new();

    // Process all existing amplitudes
    for (&idx, &_amp) in amplitudes.iter() {
        let idx0 = idx & !mask; // Clear qubit bit
        let idx1 = idx | mask; // Set qubit bit

        let amp0 = amplitudes
            .get(&idx0)
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0));
        let amp1 = amplitudes
            .get(&idx1)
            .copied()
            .unwrap_or(Complex64::new(0.0, 0.0));

        let new0 = gate[0][0] * amp0 + gate[0][1] * amp1;
        let new1 = gate[1][0] * amp0 + gate[1][1] * amp1;

        if new0.norm_sqr() > 1e-15 {
            new_amplitudes.insert(idx0, new0);
        }
        if new1.norm_sqr() > 1e-15 {
            new_amplitudes.insert(idx1, new1);
        }
    }

    *amplitudes = new_amplitudes;
    Ok(())
}

/// Apply a three-qubit gate to a sparse state
///
/// The 8×8 matrix uses the standard basis ordering |q1 q2 q3⟩, i.e. matrix
/// index `m = (bit(qubit1) << 2) | (bit(qubit2) << 1) | bit(qubit3)`.
pub fn apply_three_qubit_sparse(
    gate: &[[Complex64; 8]; 8],
    qubit1: usize,
    qubit2: usize,
    qubit3: usize,
    amplitudes: &mut AHashMap<u64, Complex64>,
    _num_qubits: usize,
) -> Result<()> {
    let mask1 = 1u64 << qubit1;
    let mask2 = 1u64 << qubit2;
    let mask3 = 1u64 << qubit3;
    let all_masks = mask1 | mask2 | mask3;
    let mut new_amplitudes: AHashMap<u64, Complex64> = AHashMap::new();

    // Collect all 8-dimensional subspaces containing nonzero amplitudes
    let mut basis_states = std::collections::HashSet::new();
    for &idx in amplitudes.keys() {
        basis_states.insert(idx & !all_masks);
    }

    for &base in &basis_states {
        let mut indices = [0u64; 8];
        for (m, slot) in indices.iter_mut().enumerate() {
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

        let mut amp = [Complex64::new(0.0, 0.0); 8];
        for (m, &i) in indices.iter().enumerate() {
            amp[m] = amplitudes
                .get(&i)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));
        }

        for (out_idx, &idx) in indices.iter().enumerate() {
            let mut sum = Complex64::new(0.0, 0.0);
            for (in_idx, &a) in amp.iter().enumerate() {
                sum += gate[out_idx][in_idx] * a;
            }

            if sum.norm_sqr() > 1e-15 {
                new_amplitudes.insert(idx, sum);
            }
        }
    }

    *amplitudes = new_amplitudes;
    Ok(())
}

/// Apply a two-qubit gate to a sparse state
pub fn apply_two_qubit_sparse(
    gate: &[[Complex64; 4]; 4],
    qubit1: usize,
    qubit2: usize,
    amplitudes: &mut AHashMap<u64, Complex64>,
    _num_qubits: usize,
) -> Result<()> {
    let mask1 = 1u64 << qubit1;
    let mask2 = 1u64 << qubit2;
    let mut new_amplitudes: AHashMap<u64, Complex64> = AHashMap::new();

    // Collect all basis states that need to be updated
    let mut basis_states = std::collections::HashSet::new();
    for &idx in amplitudes.keys() {
        let base = idx & !(mask1 | mask2);
        basis_states.insert(base);
    }

    // Process each 4-dimensional subspace
    for &base in &basis_states {
        let idx00 = base;
        let idx01 = base | mask2;
        let idx10 = base | mask1;
        let idx11 = base | mask1 | mask2;

        let amp = [
            amplitudes
                .get(&idx00)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0)),
            amplitudes
                .get(&idx01)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0)),
            amplitudes
                .get(&idx10)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0)),
            amplitudes
                .get(&idx11)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0)),
        ];

        let indices = [idx00, idx01, idx10, idx11];

        for (out_idx, &idx) in indices.iter().enumerate() {
            let mut sum = Complex64::new(0.0, 0.0);
            for (in_idx, &a) in amp.iter().enumerate() {
                sum += gate[out_idx][in_idx] * a;
            }

            if sum.norm_sqr() > 1e-15 {
                new_amplitudes.insert(idx, sum);
            }
        }
    }

    *amplitudes = new_amplitudes;
    Ok(())
}
