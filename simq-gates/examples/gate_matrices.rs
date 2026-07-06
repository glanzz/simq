//! Example demonstrating compile-time pre-computed gate matrices
//!
//! This example showcases how SimQ pre-computes common quantum gate matrices
//! at compile time for optimal performance.

use simq_core::gate::Gate;
use simq_gates::{matrices, standard::*, CNot, Hadamard, PauliX};
use std::f64::consts::PI;

fn print_matrix_2x2(name: &str, matrix: &[[num_complex::Complex64; 2]; 2]) {
    println!("\n{} gate matrix:", name);
    for row in matrix {
        print!("  [");
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                print!(",  ");
            }
            print!("{:>8.4}{:>+8.4}i", val.re, val.im);
        }
        println!("]");
    }
}

fn print_matrix_4x4(name: &str, matrix: &[[num_complex::Complex64; 4]; 4]) {
    println!("\n{} gate matrix:", name);
    for row in matrix {
        print!("  [");
        for (i, val) in row.iter().enumerate() {
            if i > 0 {
                print!(",  ");
            }
            print!("{:>6.2}{:>+6.2}i", val.re, val.im);
        }
        println!("]");
    }
}

fn main() {
    println!("===========================================");
    println!("SimQ: Compile-time Gate Matrix Computation");
    println!("===========================================");

    // ========================================================================
    // Single-Qubit Gates (Pre-computed at compile time)
    // ========================================================================

    println!("\n\n--- SINGLE-QUBIT GATES (Compile-time Pre-computed) ---");

    // Access matrices directly from the matrices module
    print_matrix_2x2("Hadamard (H)", &matrices::HADAMARD);
    print_matrix_2x2("Pauli-X", &matrices::PAULI_X);
    print_matrix_2x2("Pauli-Y", &matrices::PAULI_Y);
    print_matrix_2x2("Pauli-Z", &matrices::PAULI_Z);
    print_matrix_2x2("S Gate", &matrices::S_GATE);
    print_matrix_2x2("T Gate", &matrices::T_GATE);

    // ========================================================================
    // Two-Qubit Gates (Pre-computed at compile time)
    // ========================================================================

    println!("\n\n--- TWO-QUBIT GATES (Compile-time Pre-computed) ---");

    print_matrix_4x4("CNOT", &matrices::CNOT);
    print_matrix_4x4("CZ", &matrices::CZ);
    print_matrix_4x4("SWAP", &matrices::SWAP);
    print_matrix_4x4("iSWAP", &matrices::ISWAP);

    // ========================================================================
    // Access via Gate Types
    // ========================================================================

    println!("\n\n--- ACCESSING MATRICES VIA GATE TYPES ---");

    // Static method access (returns reference to compile-time constant)
    let h_matrix = Hadamard::matrix();
    println!("\nHadamard gate via Hadamard::matrix():");
    println!("  Matrix dimensions: {}x{}", h_matrix.len(), h_matrix[0].len());
    println!("  H[0][0] = {}", h_matrix[0][0]);

    // Via Gate trait (returns owned vector)
    let x_gate = PauliX;
    if let Some(matrix) = x_gate.matrix() {
        println!("\nPauli-X via Gate trait:");
        println!("  Matrix size: {} elements", matrix.len());
        println!("  X[0] = {}", matrix[0]);
    }

    // ========================================================================
    // Parameterized Gates (Computed on demand)
    // ========================================================================

    println!("\n\n--- PARAMETERIZED GATES (Computed on Demand) ---");

    // RX(π/2) - 90 degree rotation around X-axis
    let rx_gate = RotationX::new(PI / 2.0);
    let rx_matrix = rx_gate.matrix();
    print_matrix_2x2("RX(π/2)", &rx_matrix);

    // RY(π) - 180 degree rotation around Y-axis
    let ry_gate = RotationY::new(PI);
    let ry_matrix = ry_gate.matrix();
    print_matrix_2x2("RY(π)", &ry_matrix);

    // RZ(π/4) - 45 degree rotation around Z-axis
    let rz_gate = RotationZ::new(PI / 4.0);
    let rz_matrix = rz_gate.matrix();
    print_matrix_2x2("RZ(π/4)", &rz_matrix);

    // Phase gate with π/3
    let phase_gate = Phase::new(PI / 3.0);
    let phase_matrix = phase_gate.matrix();
    print_matrix_2x2("P(π/3)", &phase_matrix);

    // ========================================================================
    // Performance Characteristics
    // ========================================================================

    println!("\n\n--- PERFORMANCE CHARACTERISTICS ---");
    println!("\n✓ Standard gates (H, X, Y, Z, CNOT, etc.):");
    println!("  - Matrices pre-computed at compile time");
    println!("  - Zero runtime overhead for matrix access");
    println!("  - Stored as const static data in binary");

    println!("\n✓ Parameterized gates (RX, RY, RZ, Phase):");
    println!("  - Matrices computed on first call");
    println!("  - Inline optimizations for trigonometric functions");
    println!("  - Can be cached by the simulator if needed");

    // ========================================================================
    // Gate Properties
    // ========================================================================

    println!("\n\n--- GATE PROPERTIES ---");

    let h = Hadamard;
    println!("\nHadamard gate:");
    println!("  Name: {}", h.name());
    println!("  Qubits: {}", h.num_qubits());
    println!("  Unitary: {}", h.is_unitary());
    println!("  Hermitian: {}", h.is_hermitian());
    println!("  Description: {}", h.description());

    let cnot = CNot;
    println!("\nCNOT gate:");
    println!("  Name: {}", cnot.name());
    println!("  Qubits: {}", cnot.num_qubits());
    println!("  Unitary: {}", cnot.is_unitary());
    println!("  Hermitian: {}", cnot.is_hermitian());
    println!("  Description: {}", cnot.description());

    let rx = RotationX::new(PI / 2.0);
    println!("\nRX(π/2) gate:");
    println!("  Name: {}", rx.name());
    println!("  Angle: {} rad ({} deg)", rx.angle(), rx.angle().to_degrees());
    println!("  Description: {}", rx.description());

    println!("\n===========================================");
    println!("All matrices are type-safe and verified!");
    println!("===========================================\n");
}
