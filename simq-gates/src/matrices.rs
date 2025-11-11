//! Pre-computed quantum gate matrices at compile time
//!
//! This module provides constant gate matrices for standard quantum gates,
//! computed at compile time for optimal performance.

use num_complex::Complex64;

// Compile-time constant helpers
const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);
const I: Complex64 = Complex64::new(0.0, 1.0);
const NEG_I: Complex64 = Complex64::new(0.0, -1.0);
const NEG_ONE: Complex64 = Complex64::new(-1.0, 0.0);

// Common mathematical constants
const INV_SQRT2: f64 = 0.7071067811865476; // 1/√2

// Single-qubit gate matrices (2x2)

/// Hadamard gate matrix
/// H = 1/√2 * [[1,  1],
///             [1, -1]]
pub const HADAMARD: [[Complex64; 2]; 2] = [
    [
        Complex64::new(INV_SQRT2, 0.0),
        Complex64::new(INV_SQRT2, 0.0),
    ],
    [
        Complex64::new(INV_SQRT2, 0.0),
        Complex64::new(-INV_SQRT2, 0.0),
    ],
];

/// Pauli-X gate matrix (NOT gate)
/// X = [[0, 1],
///      [1, 0]]
pub const PAULI_X: [[Complex64; 2]; 2] = [
    [ZERO, ONE],
    [ONE, ZERO],
];

/// Pauli-Y gate matrix
/// Y = [[0, -i],
///      [i,  0]]
pub const PAULI_Y: [[Complex64; 2]; 2] = [
    [ZERO, NEG_I],
    [I, ZERO],
];

/// Pauli-Z gate matrix
/// Z = [[1,  0],
///      [0, -1]]
pub const PAULI_Z: [[Complex64; 2]; 2] = [
    [ONE, ZERO],
    [ZERO, NEG_ONE],
];

/// Identity gate matrix
/// I = [[1, 0],
///      [0, 1]]
pub const IDENTITY: [[Complex64; 2]; 2] = [
    [ONE, ZERO],
    [ZERO, ONE],
];

/// S gate matrix (Phase gate, √Z)
/// S = [[1, 0],
///      [0, i]]
pub const S_GATE: [[Complex64; 2]; 2] = [
    [ONE, ZERO],
    [ZERO, I],
];

/// S† gate matrix (adjoint of S gate)
/// S† = [[1,  0],
///       [0, -i]]
pub const S_GATE_DAGGER: [[Complex64; 2]; 2] = [
    [ONE, ZERO],
    [ZERO, NEG_I],
];

/// T gate matrix (π/8 gate, √S)
/// T = [[1, 0],
///      [0, e^(iπ/4)]]
pub const T_GATE: [[Complex64; 2]; 2] = [
    [ONE, ZERO],
    [ZERO, Complex64::new(INV_SQRT2, INV_SQRT2)], // e^(iπ/4) = (1+i)/√2
];

/// T† gate matrix (adjoint of T gate)
/// T† = [[1, 0],
///       [0, e^(-iπ/4)]]
pub const T_GATE_DAGGER: [[Complex64; 2]; 2] = [
    [ONE, ZERO],
    [ZERO, Complex64::new(INV_SQRT2, -INV_SQRT2)], // e^(-iπ/4) = (1-i)/√2
];

/// SX gate matrix (√X gate)
/// SX = 1/2 * [[1+i, 1-i],
///             [1-i, 1+i]]
pub const SX_GATE: [[Complex64; 2]; 2] = [
    [
        Complex64::new(0.5, 0.5),   // (1+i)/2
        Complex64::new(0.5, -0.5),  // (1-i)/2
    ],
    [
        Complex64::new(0.5, -0.5),  // (1-i)/2
        Complex64::new(0.5, 0.5),   // (1+i)/2
    ],
];

/// SX† gate matrix (adjoint of √X gate)
/// SX† = 1/2 * [[1-i, 1+i],
///              [1+i, 1-i]]
pub const SX_GATE_DAGGER: [[Complex64; 2]; 2] = [
    [
        Complex64::new(0.5, -0.5),  // (1-i)/2
        Complex64::new(0.5, 0.5),   // (1+i)/2
    ],
    [
        Complex64::new(0.5, 0.5),   // (1+i)/2
        Complex64::new(0.5, -0.5),  // (1-i)/2
    ],
];

// Two-qubit gate matrices (4x4)

/// CNOT gate matrix (Controlled-NOT)
/// CNOT = [[1, 0, 0, 0],
///         [0, 1, 0, 0],
///         [0, 0, 0, 1],
///         [0, 0, 1, 0]]
pub const CNOT: [[Complex64; 4]; 4] = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ONE],
    [ZERO, ZERO, ONE, ZERO],
];

/// CZ gate matrix (Controlled-Z)
/// CZ = [[1, 0, 0,  0],
///       [0, 1, 0,  0],
///       [0, 0, 1,  0],
///       [0, 0, 0, -1]]
pub const CZ: [[Complex64; 4]; 4] = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, ZERO, ZERO, NEG_ONE],
];

/// SWAP gate matrix
/// SWAP = [[1, 0, 0, 0],
///         [0, 0, 1, 0],
///         [0, 1, 0, 0],
///         [0, 0, 0, 1]]
pub const SWAP: [[Complex64; 4]; 4] = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ONE],
];

/// iSWAP gate matrix
/// iSWAP = [[1, 0, 0, 0],
///          [0, 0, i, 0],
///          [0, i, 0, 0],
///          [0, 0, 0, 1]]
pub const ISWAP: [[Complex64; 4]; 4] = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, I, ZERO],
    [ZERO, I, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ONE],
];

/// CY gate matrix (Controlled-Y)
/// CY = [[1, 0,  0,  0],
///       [0, 1,  0,  0],
///       [0, 0,  0, -i],
///       [0, 0,  i,  0]]
pub const CY: [[Complex64; 4]; 4] = [
    [ONE, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ZERO, NEG_I],
    [ZERO, ZERO, I, ZERO],
];

/// ECR gate matrix (Echoed Cross-Resonance)
/// ECR = 1/√2 * [[0,  0, 1,  i],
///               [0,  0, i,  1],
///               [1, -i, 0,  0],
///               [-i, 1, 0,  0]]
pub const ECR: [[Complex64; 4]; 4] = [
    [
        ZERO,
        ZERO,
        Complex64::new(INV_SQRT2, 0.0),
        Complex64::new(0.0, INV_SQRT2),
    ],
    [
        ZERO,
        ZERO,
        Complex64::new(0.0, INV_SQRT2),
        Complex64::new(INV_SQRT2, 0.0),
    ],
    [
        Complex64::new(INV_SQRT2, 0.0),
        Complex64::new(0.0, -INV_SQRT2),
        ZERO,
        ZERO,
    ],
    [
        Complex64::new(0.0, -INV_SQRT2),
        Complex64::new(INV_SQRT2, 0.0),
        ZERO,
        ZERO,
    ],
];

// Three-qubit gate matrices (8x8)

/// Toffoli gate matrix (CCNOT - double-controlled NOT)
/// Only flips target when both control qubits are |1⟩
pub const TOFFOLI: [[Complex64; 8]; 8] = [
    [ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO],
];

/// Fredkin gate matrix (CSWAP - controlled SWAP)
/// Swaps targets when control qubit is |1⟩
pub const FREDKIN: [[Complex64; 8]; 8] = [
    [ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ONE, ZERO, ZERO],
    [ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ONE],
];

// Parameterized gate matrix generators

/// Generate rotation-X gate matrix for a given angle
/// RX(θ) = [[cos(θ/2),    -i·sin(θ/2)],
///          [-i·sin(θ/2),  cos(θ/2)]]
#[inline]
pub fn rotation_x(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.0;
    let cos_val = half_theta.cos();
    let sin_val = half_theta.sin();

    [
        [
            Complex64::new(cos_val, 0.0),
            Complex64::new(0.0, -sin_val),
        ],
        [
            Complex64::new(0.0, -sin_val),
            Complex64::new(cos_val, 0.0),
        ],
    ]
}

/// Generate rotation-Y gate matrix for a given angle
/// RY(θ) = [[cos(θ/2),  -sin(θ/2)],
///          [sin(θ/2),   cos(θ/2)]]
#[inline]
pub fn rotation_y(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.0;
    let cos_val = half_theta.cos();
    let sin_val = half_theta.sin();

    [
        [
            Complex64::new(cos_val, 0.0),
            Complex64::new(-sin_val, 0.0),
        ],
        [
            Complex64::new(sin_val, 0.0),
            Complex64::new(cos_val, 0.0),
        ],
    ]
}

/// Generate rotation-Z gate matrix for a given angle
/// RZ(θ) = [[e^(-iθ/2),  0       ],
///          [0,          e^(iθ/2)]]
#[inline]
pub fn rotation_z(theta: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.0;

    [
        [
            Complex64::new(half_theta.cos(), -half_theta.sin()),
            ZERO,
        ],
        [
            ZERO,
            Complex64::new(half_theta.cos(), half_theta.sin()),
        ],
    ]
}

/// Generate phase gate matrix for a given angle
/// P(θ) = [[1, 0     ],
///         [0, e^(iθ)]]
#[inline]
pub fn phase(theta: f64) -> [[Complex64; 2]; 2] {
    [
        [ONE, ZERO],
        [ZERO, Complex64::new(theta.cos(), theta.sin())],
    ]
}

/// Generate U1 gate matrix (phase gate, equivalent to P gate with global phase)
/// U1(λ) = [[1, 0      ],
///          [0, e^(iλ)]]
#[inline]
pub fn u1(lambda: f64) -> [[Complex64; 2]; 2] {
    phase(lambda)
}

/// Generate U2 gate matrix (Hadamard-like rotation)
/// U2(φ,λ) = 1/√2 * [[1,        -e^(iλ)    ],
///                   [e^(iφ),    e^(i(φ+λ))]]
#[inline]
pub fn u2(phi: f64, lambda: f64) -> [[Complex64; 2]; 2] {
    let e_phi = Complex64::new(phi.cos(), phi.sin());
    let e_lambda = Complex64::new(lambda.cos(), lambda.sin());
    let e_phi_lambda = Complex64::new((phi + lambda).cos(), (phi + lambda).sin());

    [
        [
            Complex64::new(INV_SQRT2, 0.0),
            -e_lambda * INV_SQRT2,
        ],
        [
            e_phi * INV_SQRT2,
            e_phi_lambda * INV_SQRT2,
        ],
    ]
}

/// Generate U3 gate matrix (universal single-qubit gate)
/// U3(θ,φ,λ) = [[cos(θ/2),              -e^(iλ)·sin(θ/2)    ],
///              [e^(iφ)·sin(θ/2),        e^(i(φ+λ))·cos(θ/2)]]
#[inline]
pub fn u3(theta: f64, phi: f64, lambda: f64) -> [[Complex64; 2]; 2] {
    let half_theta = theta / 2.0;
    let cos_val = half_theta.cos();
    let sin_val = half_theta.sin();

    let e_phi = Complex64::new(phi.cos(), phi.sin());
    let e_lambda = Complex64::new(lambda.cos(), lambda.sin());
    let e_phi_lambda = Complex64::new((phi + lambda).cos(), (phi + lambda).sin());

    [
        [
            Complex64::new(cos_val, 0.0),
            -e_lambda * sin_val,
        ],
        [
            e_phi * sin_val,
            e_phi_lambda * cos_val,
        ],
    ]
}

/// Generate controlled-phase gate matrix
/// CP(θ) = [[1, 0, 0,      0     ],
///          [0, 1, 0,      0     ],
///          [0, 0, 1,      0     ],
///          [0, 0, 0, e^(iθ)]]
#[inline]
pub fn controlled_phase(theta: f64) -> [[Complex64; 4]; 4] {
    [
        [ONE, ZERO, ZERO, ZERO],
        [ZERO, ONE, ZERO, ZERO],
        [ZERO, ZERO, ONE, ZERO],
        [ZERO, ZERO, ZERO, Complex64::new(theta.cos(), theta.sin())],
    ]
}

/// Generate RXX gate matrix (two-qubit XX rotation)
/// RXX(θ) = exp(-i θ/2 X⊗X)
#[inline]
pub fn rxx(theta: f64) -> [[Complex64; 4]; 4] {
    let half_theta = theta / 2.0;
    let cos_val = Complex64::new(half_theta.cos(), 0.0);
    let sin_val = Complex64::new(0.0, -half_theta.sin());

    [
        [cos_val, ZERO, ZERO, sin_val],
        [ZERO, cos_val, sin_val, ZERO],
        [ZERO, sin_val, cos_val, ZERO],
        [sin_val, ZERO, ZERO, cos_val],
    ]
}

/// Generate RYY gate matrix (two-qubit YY rotation)
/// RYY(θ) = exp(-i θ/2 Y⊗Y)
#[inline]
pub fn ryy(theta: f64) -> [[Complex64; 4]; 4] {
    let half_theta = theta / 2.0;
    let cos_val = Complex64::new(half_theta.cos(), 0.0);
    let sin_val = Complex64::new(0.0, half_theta.sin());

    [
        [cos_val, ZERO, ZERO, sin_val],
        [ZERO, cos_val, -sin_val, ZERO],
        [ZERO, -sin_val, cos_val, ZERO],
        [sin_val, ZERO, ZERO, cos_val],
    ]
}

/// Generate RZZ gate matrix (two-qubit ZZ rotation)
/// RZZ(θ) = exp(-i θ/2 Z⊗Z)
#[inline]
pub fn rzz(theta: f64) -> [[Complex64; 4]; 4] {
    let half_theta = theta / 2.0;
    let e_neg = Complex64::new(half_theta.cos(), -half_theta.sin());
    let e_pos = Complex64::new(half_theta.cos(), half_theta.sin());

    [
        [e_neg, ZERO, ZERO, ZERO],
        [ZERO, e_pos, ZERO, ZERO],
        [ZERO, ZERO, e_pos, ZERO],
        [ZERO, ZERO, ZERO, e_neg],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pauli_x_squaring() {
        // X² = I
        let mut result = [[ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    result[i][j] += PAULI_X[i][k] * PAULI_X[k][j];
                }
            }
        }

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result[i][j].re, IDENTITY[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(result[i][j].im, IDENTITY[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_hadamard_self_inverse() {
        // H² = I
        let mut result = [[ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    result[i][j] += HADAMARD[i][k] * HADAMARD[k][j];
                }
            }
        }

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result[i][j].re, IDENTITY[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(result[i][j].im, IDENTITY[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_s_gate_squaring() {
        // S² = Z
        let mut result = [[ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    result[i][j] += S_GATE[i][k] * S_GATE[k][j];
                }
            }
        }

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result[i][j].re, PAULI_Z[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(result[i][j].im, PAULI_Z[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_t_gate_squaring() {
        // T² = S
        let mut result = [[ZERO; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    result[i][j] += T_GATE[i][k] * T_GATE[k][j];
                }
            }
        }

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(result[i][j].re, S_GATE[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(result[i][j].im, S_GATE[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_cnot_identity_on_control_0() {
        // When control qubit is |0⟩, target should not flip
        // |00⟩ -> |00⟩, |01⟩ -> |01⟩
        let state_00 = [ONE, ZERO, ZERO, ZERO];
        let state_01 = [ZERO, ONE, ZERO, ZERO];

        let mut result_00 = [ZERO; 4];
        let mut result_01 = [ZERO; 4];

        for i in 0..4 {
            for j in 0..4 {
                result_00[i] += CNOT[i][j] * state_00[j];
                result_01[i] += CNOT[i][j] * state_01[j];
            }
        }

        assert_eq!(result_00, state_00);
        assert_eq!(result_01, state_01);
    }

    #[test]
    fn test_cnot_flip_on_control_1() {
        // When control qubit is |1⟩, target should flip
        // |10⟩ -> |11⟩, |11⟩ -> |10⟩
        let state_10 = [ZERO, ZERO, ONE, ZERO];
        let state_11 = [ZERO, ZERO, ZERO, ONE];
        let expected_11 = [ZERO, ZERO, ZERO, ONE];
        let expected_10 = [ZERO, ZERO, ONE, ZERO];

        let mut result_10 = [ZERO; 4];
        let mut result_11 = [ZERO; 4];

        for i in 0..4 {
            for j in 0..4 {
                result_10[i] += CNOT[i][j] * state_10[j];
                result_11[i] += CNOT[i][j] * state_11[j];
            }
        }

        assert_eq!(result_10, expected_11);
        assert_eq!(result_11, expected_10);
    }

    #[test]
    fn test_rotation_x_identity() {
        // RX(0) = I
        let rx_0 = rotation_x(0.0);

        for i in 0..2 {
            for j in 0..2 {
                assert_relative_eq!(rx_0[i][j].re, IDENTITY[i][j].re, epsilon = 1e-10);
                assert_relative_eq!(rx_0[i][j].im, IDENTITY[i][j].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_rotation_x_pi() {
        // RX(π) = -iX
        use std::f64::consts::PI;
        let rx_pi = rotation_x(PI);

        for i in 0..2 {
            for j in 0..2 {
                let expected = NEG_I * PAULI_X[i][j];
                assert_relative_eq!(rx_pi[i][j].re, expected.re, epsilon = 1e-10);
                assert_relative_eq!(rx_pi[i][j].im, expected.im, epsilon = 1e-10);
            }
        }
    }
}
