//! Specialized optimized implementations of controlled gates (CNOT, CZ, etc.)
//!
//! These gates have special structure that allows faster implementation
//! than generic 4×4 matrix multiplication.

use num_complex::Complex64;

/// Apply a CNOT (Controlled-NOT) gate using direct amplitude swaps
///
/// CNOT structure in basis {|00⟩, |01⟩, |10⟩, |11⟩}:
/// - If control bit is 0: identity on target
/// - If control bit is 1: flip target
///
/// This avoids full 4×4 matrix multiplication by using conditional swaps.
/// Cache-friendly: processes blocks where control qubit bits match.
///
/// # Arguments
/// * `state` - Mutable slice of state amplitudes
/// * `control` - Index of the control qubit
/// * `target` - Index of the target qubit
/// * `num_qubits` - Total number of qubits
pub fn apply_cnot_scalar(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    num_qubits: usize,
) {
    let dimension = 1usize << num_qubits;
    let mask_control = 1usize << control;
    let mask_target = 1usize << target;

    // Iterate through all basis states where control qubit is 1
    // We need to be careful: only swap pairs where control qubit = 1
    let mut processed = vec![false; dimension];

    for i in 0..dimension {
        if !processed[i] && (i & mask_control) != 0 {
            // Control qubit is 1, so we apply the gate
            // The gate flips the target qubit
            let j = i ^ mask_target;
            // Swap state[i] and state[j]
            state.swap(i, j);
            processed[i] = true;
            processed[j] = true;
        }
    }
}

/// Apply a CNOT gate using nested loops for better cache locality
///
/// This version uses stride-based iteration to keep memory accesses local.
/// Processes blocks where both qubits have constant bits.
pub fn apply_cnot_striped(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    num_qubits: usize,
) {
    let dimension = 1usize << num_qubits;

    // Order qubits so control < target for consistent iteration
    let (q_low, q_high, is_control_low) = if control < target {
        (control, target, true)
    } else {
        (target, control, false)
    };

    let stride_low = 1usize << q_low;
    let stride_high = 1usize << q_high;
    let block_high = stride_high * 2;
    let block_low = stride_low * 2;

    let mut base = 0usize;
    while base < dimension {
        let mut mid = 0usize;
        while mid < stride_high {
            let block_base = base + mid;

            for k in 0..stride_low {
                let idx1 = block_base + k;
                let idx2 = idx1 + stride_low;

                // Determine which basis state we're in
                let bit_low = (idx1 >> q_low) & 1;
                let bit_high = (idx1 >> q_high) & 1;

                // Apply CNOT only when control bit is 1
                let apply_gate = if is_control_low {
                    bit_low == 1
                } else {
                    bit_high == 1
                };

                if apply_gate {
                    // Swap the two amplitudes (corresponds to flipping the target)
                    state.swap(idx1, idx2);
                }
            }

            mid += block_low;
        }

        base += block_high;
    }
}

/// Apply a CZ (Controlled-Z) gate using direct phase multiplication
///
/// CZ structure: applies phase -1 only to |11⟩ state
/// - |00⟩ → |00⟩
/// - |01⟩ → |01⟩
/// - |10⟩ → |10⟩
/// - |11⟩ → -|11⟩
///
/// This is much faster than 4×4 matrix multiplication.
/// Only one amplitude needs modification per 4-block.
pub fn apply_cz_scalar(
    state: &mut [Complex64],
    qubit1: usize,
    qubit2: usize,
    num_qubits: usize,
) {
    let dimension = 1usize << num_qubits;
    let mask1 = 1usize << qubit1;
    let mask2 = 1usize << qubit2;
    let mask_both = mask1 | mask2;

    // Only the |11⟩ state gets a phase shift (multiply by -1)
    for i in 0..dimension {
        if (i & mask_both) == mask_both {
            state[i] = -state[i];
        }
    }
}

/// Apply a CZ gate using nested loops for cache locality
///
/// Uses stride-based iteration to access the |11⟩ state efficiently.
pub fn apply_cz_striped(
    state: &mut [Complex64],
    qubit1: usize,
    qubit2: usize,
    num_qubits: usize,
) {
    let dimension = 1usize << num_qubits;

    // Order qubits for consistent iteration
    let (q_low, q_high) = if qubit1 < qubit2 {
        (qubit1, qubit2)
    } else {
        (qubit2, qubit1)
    };

    let stride_low = 1usize << q_low;
    let stride_high = 1usize << q_high;
    let block_high = stride_high * 2;
    let block_low = stride_low * 2;

    // Phase shift for |11⟩ state (both qubits set to 1)
    let phase = Complex64::new(-1.0, 0.0);

    let mut base = 0usize;
    while base < dimension {
        let mut mid = 0usize;
        while mid < stride_high {
            let block_base = base + mid;

            // The |11⟩ state is at idx1 + stride_low when both low and high
            // qubit bits are 1, which happens in the second half of each block
            let idx_11 = block_base + stride_low + stride_high;

            // Apply phase to the |11⟩ state
            if idx_11 < dimension {
                state[idx_11] *= phase;
            }

            mid += block_low;
        }

        base += block_high;
    }
}

/// Apply a controlled-U gate (U gate on target if control=1)
///
/// This is more general than CNOT but still optimized compared to full 4×4.
/// Matrix structure: diag(I_2, U_2) in the computational basis.
pub fn apply_controlled_u_scalar(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    u_matrix: &[[Complex64; 2]; 2],
    num_qubits: usize,
) {
    let dimension = 1usize << num_qubits;
    let mask_control = 1usize << control;
    let mask_target = 1usize << target;

    // Mark indices we've processed to avoid double-processing pairs
    let mut processed = vec![false; dimension];

    // Apply U only when control qubit is 1
    for i in 0..dimension {
        if !processed[i] && (i & mask_control) != 0 {
            // Get the two indices for the target qubit basis
            let i0 = i; // target qubit = 0
            let i1 = i ^ mask_target; // target qubit = 1

            // Apply 2×2 unitary to the pair
            let a0 = state[i0];
            let a1 = state[i1];

            state[i0] = u_matrix[0][0] * a0 + u_matrix[0][1] * a1;
            state[i1] = u_matrix[1][0] * a0 + u_matrix[1][1] * a1;

            processed[i0] = true;
            processed[i1] = true;
        }
    }
}

/// Apply a controlled-U gate using nested loops for better cache locality
///
/// This version uses stride-based iteration to keep memory accesses local.
/// Processes blocks where both qubits have constant bits, enabling
/// better CPU cache utilization than the scalar variant.
pub fn apply_controlled_u_striped(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    u_matrix: &[[Complex64; 2]; 2],
    num_qubits: usize,
) {
    let dimension = 1usize << num_qubits;
    let mask_control = 1usize << control;
    let mask_target = 1usize << target;

    // Mark indices we've processed to avoid double-processing pairs
    let mut processed = vec![false; dimension];

    // For each basis state where control qubit is 1, apply U to the target qubit pair
    for i in 0..dimension {
        if !processed[i] && (i & mask_control) != 0 {
            // Get the two indices for the target qubit basis
            let idx_target_0 = i; // target qubit = 0
            let idx_target_1 = i ^ mask_target; // target qubit = 1

            // Apply 2×2 unitary to the pair
            let a0 = state[idx_target_0];
            let a1 = state[idx_target_1];

            state[idx_target_0] = u_matrix[0][0] * a0 + u_matrix[0][1] * a1;
            state[idx_target_1] = u_matrix[1][0] * a0 + u_matrix[1][1] * a1;

            processed[i] = true;
            processed[idx_target_1] = true;
        }
    }
}

/// Apply a controlled-RX(θ) gate: CRX(θ) = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ RX(θ)
///
/// When control=1, applies RX(θ) to target:
/// RX(θ) = cos(θ/2)·I - i·sin(θ/2)·X
pub fn apply_crx(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    theta: f64,
    num_qubits: usize,
) {
    let c = Complex64::new(0.0, -((theta / 2.0).sin()));
    let d = Complex64::new((theta / 2.0).cos(), 0.0);

    // RX(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
    let u_matrix = [[d, c], [c, d]];

    apply_controlled_u_striped(state, control, target, &u_matrix, num_qubits);
}

/// Apply a controlled-RY(θ) gate: CRY(θ) = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ RY(θ)
///
/// When control=1, applies RY(θ) to target:
/// RY(θ) = cos(θ/2)·I - i·sin(θ/2)·Y
pub fn apply_cry(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    theta: f64,
    num_qubits: usize,
) {
    let c = (theta / 2.0).sin();
    let d = (theta / 2.0).cos();

    // RY(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    let u_matrix = [
        [Complex64::new(d, 0.0), Complex64::new(-c, 0.0)],
        [Complex64::new(c, 0.0), Complex64::new(d, 0.0)],
    ];

    apply_controlled_u_striped(state, control, target, &u_matrix, num_qubits);
}

/// Apply a controlled-RZ(θ) gate: CRZ(θ) = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ RZ(θ)
///
/// When control=1, applies RZ(θ) to target:
/// RZ(θ) = exp(-i·θ/2·Z) = [[e^(-i·θ/2), 0], [0, e^(i·θ/2)]]
pub fn apply_crz(
    state: &mut [Complex64],
    control: usize,
    target: usize,
    theta: f64,
    num_qubits: usize,
) {
    let phase_neg = Complex64::from_polar(1.0, -theta / 2.0);
    let phase_pos = Complex64::from_polar(1.0, theta / 2.0);

    // RZ(θ) = [[e^(-i·θ/2), 0], [0, e^(i·θ/2)]]
    let u_matrix = [
        [phase_neg, Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), phase_pos],
    ];

    apply_controlled_u_striped(state, control, target, &u_matrix, num_qubits);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cnot_scalar() {
        // Test CNOT on |00⟩ state
        let mut state = vec![
            Complex64::new(1.0, 0.0), // |00⟩
            Complex64::new(0.0, 0.0), // |01⟩
            Complex64::new(0.0, 0.0), // |10⟩
            Complex64::new(0.0, 0.0), // |11⟩
        ];

        // CNOT with control=0, target=1 on |00⟩ does nothing
        apply_cnot_scalar(&mut state, 0, 1, 2);

        assert_relative_eq!(state[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cnot_creates_bell_state() {
        // Start with |00⟩ state, then apply Hadamard to qubit 0
        // This gives (|0⟩ + |1⟩)/√2 ⊗ |0⟩ = (|00⟩ + |10⟩)/√2
        // Basis ordering: |00⟩ at idx 0, |01⟩ at idx 1, |10⟩ at idx 2, |11⟩ at idx 3
        // (Qubit 0 is least significant bit)
        let mut state = vec![
            Complex64::new(0.7071067811865476, 0.0), // |00⟩: q0=0, q1=0
            Complex64::new(0.0, 0.0),                // |01⟩: q0=1, q1=0
            Complex64::new(0.7071067811865476, 0.0), // |10⟩: q0=0, q1=1
            Complex64::new(0.0, 0.0),                // |11⟩: q0=1, q1=1
        ];

        // CNOT with control=1, target=0 (control is q1, target is q0)
        // Basis state |00⟩: control(q1)=0 → no swap
        // Basis state |01⟩: control(q1)=0 → no swap
        // Basis state |10⟩: control(q1)=1 → swap with |11⟩ (flip q0)
        // Basis state |11⟩: control(q1)=1 → swap with |10⟩ (flip q0)
        // After CNOT: (|00⟩ + |11⟩)/√2
        apply_cnot_scalar(&mut state, 1, 0, 2);

        // After CNOT with control=1, target=0: (|00⟩ + |11⟩)/√2
        assert_relative_eq!(state[0].re, 0.7071067811865476, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[2].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[3].re, 0.7071067811865476, epsilon = 1e-10);
    }

    #[test]
    fn test_cz_scalar() {
        // Start with equal superposition: (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
        let inv_sqrt2 = 0.5;
        let mut state = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
        ];

        // CZ applies phase to |11⟩
        apply_cz_scalar(&mut state, 0, 1, 2);

        // |11⟩ should be negated
        assert_relative_eq!(state[0].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[2].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[3].re, -inv_sqrt2, epsilon = 1e-10);
    }

    #[test]
    fn test_cnot_striped_matches_scalar() {
        // Test that striped version produces same results
        let mut state1 = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let mut state2 = state1.clone();

        apply_cnot_scalar(&mut state1, 0, 1, 2);
        apply_cnot_striped(&mut state2, 0, 1, 2);

        for i in 0..4 {
            assert_relative_eq!(state1[i].re, state2[i].re, epsilon = 1e-10);
            assert_relative_eq!(state1[i].im, state2[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_cz_striped_matches_scalar() {
        // Test that striped version produces same results
        let mut state1 = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let mut state2 = state1.clone();

        apply_cz_scalar(&mut state1, 0, 1, 2);
        apply_cz_striped(&mut state2, 0, 1, 2);

        for i in 0..4 {
            assert_relative_eq!(state1[i].re, state2[i].re, epsilon = 1e-10);
            assert_relative_eq!(state1[i].im, state2[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_controlled_u_striped_matches_scalar() {
        // Test that striped version matches scalar implementation
        let u_matrix = [
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)], // -iX (Pauli X with phase)
            [Complex64::new(0.0, -1.0), Complex64::new(0.0, 0.0)],
        ];

        let mut state1 = vec![
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.5, 0.0),
        ];

        let mut state2 = state1.clone();

        apply_controlled_u_scalar(&mut state1, 0, 1, &u_matrix, 2);
        apply_controlled_u_striped(&mut state2, 0, 1, &u_matrix, 2);

        for i in 0..4 {
            assert_relative_eq!(state1[i].re, state2[i].re, epsilon = 1e-10);
            assert_relative_eq!(state1[i].im, state2[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_crx_gate() {
        // Test controlled-RX gate on 2-qubit system
        // Basis: |00⟩ = 0, |01⟩ = 1, |10⟩ = 2, |11⟩ = 3 (q0 is LSB)
        // Start with |10⟩ state (control qubit q1 = 1)
        let mut state = vec![
            Complex64::new(0.0, 0.0),  // |00⟩
            Complex64::new(0.0, 0.0),  // |01⟩
            Complex64::new(1.0, 0.0),  // |10⟩ (q1=1, q0=0)
            Complex64::new(0.0, 0.0),  // |11⟩
        ];

        // Apply CRX(π/2) with control=1 (q1), target=0 (q0)
        // When control qubit (q1) = 1: apply RX(π/2) to q0
        // |10⟩ = |1⟩ ⊗ |0⟩, apply RX(π/2) to q0:
        // RX(π/2) |0⟩ = (|0⟩ - i|1⟩)/√2
        // Result: |1⟩ ⊗ (|0⟩ - i|1⟩)/√2 = (|10⟩ - i|11⟩)/√2
        apply_crx(&mut state, 1, 0, std::f64::consts::PI / 2.0, 2);

        let inv_sqrt2 = 0.7071067811865476;
        assert_relative_eq!(state[0].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[2].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[2].im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[3].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[3].im, -inv_sqrt2, epsilon = 1e-10);
    }

    #[test]
    fn test_cry_gate() {
        // Test controlled-RY gate
        let mut state = vec![
            Complex64::new(1.0, 0.0), // |00⟩
            Complex64::new(0.0, 0.0), // |01⟩
            Complex64::new(0.0, 0.0), // |10⟩
            Complex64::new(0.0, 0.0), // |11⟩
        ];

        // Apply CRY(π) with control=0, target=1
        // When control = 0, nothing happens
        apply_cry(&mut state, 0, 1, std::f64::consts::PI, 2);

        // State should remain unchanged (control=0 means no operation)
        assert_relative_eq!(state[0].re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[2].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[3].re, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_crz_gate() {
        // Test controlled-RZ gate (diagonal gate, only affects phases)
        let inv_sqrt2 = 0.7071067811865476;
        let mut state = vec![
            Complex64::new(inv_sqrt2, 0.0), // |00⟩
            Complex64::new(inv_sqrt2, 0.0), // |01⟩
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        // Apply CRZ(π) with control=1, target=0
        // When control = 0, nothing happens
        apply_crz(&mut state, 1, 0, std::f64::consts::PI, 2);

        // |00⟩ is unchanged (control qubit = 0)
        // |01⟩ is unchanged (control qubit = 0)
        assert_relative_eq!(state[0].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[1].re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(state[2].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(state[3].re, 0.0, epsilon = 1e-10);
    }
}
