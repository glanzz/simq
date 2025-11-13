/// Comprehensive example demonstrating custom quantum gate creation and usage
///
/// This example showcases:
/// - Creating static custom gates with validation
/// - Creating parametric custom gates
/// - Using the gate registry
/// - Gate composition and manipulation
/// - Advanced validation features
///
/// Run with: cargo run --example custom_gates_tutorial --release

use simq_gates::custom::{CustomGateBuilder, ParametricCustomGateBuilder};
use simq_gates::gate_registry::GateRegistry;
use simq_core::gate::Gate;
use num_complex::Complex64;
use std::f64::consts::{PI, SQRT_2};

fn main() {
    println!("=== SimQ Custom Quantum Gates Tutorial ===\n");

    example_1_basic_custom_gates();
    example_2_validation_features();
    example_3_parametric_gates();
    example_4_gate_registry();
    example_5_gate_composition();
    example_6_controlled_gates();
}

/// Example 1: Basic custom gate creation
fn example_1_basic_custom_gates() {
    println!("Example 1: Basic Custom Gate Creation\n");
    println!("Creating static custom gates with validation\n");

    // Create a custom Hadamard gate
    let inv_sqrt2 = 1.0 / SQRT_2;
    let hadamard = CustomGateBuilder::new("CustomH")
        .matrix_2x2([
            [Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0)],
            [Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0)],
        ])
        .description("Custom Hadamard gate with normalized entries")
        .build()
        .expect("Failed to create Hadamard gate");

    println!("Created: {}", hadamard.name());
    println!("Description: {}", hadamard.description());
    println!("Qubits: {}", hadamard.num_qubits());
    println!("Is Unitary: {}", hadamard.is_unitary());
    println!("Is Hermitian: {}\n", hadamard.is_hermitian());

    // Create Pauli X gate
    let pauli_x = CustomGateBuilder::new("CustomX")
        .matrix_2x2([
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ])
        .description("Pauli X gate (NOT gate)")
        .build()
        .expect("Failed to create Pauli X gate");

    println!("Created: {}", pauli_x.name());
    println!("Description: {}", pauli_x.description());
    println!("Is Hermitian: {}\n", pauli_x.is_hermitian());

    // Create Pauli Z gate
    let pauli_z = CustomGateBuilder::new("CustomZ")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ])
        .description("Pauli Z gate (Phase flip)")
        .build()
        .expect("Failed to create Pauli Z gate");

    println!("Created: {}", pauli_z.name());
    println!("Description: {}\n", pauli_z.description());

    // Demonstrate error handling
    println!("--- Error Handling ---");
    let invalid_matrix = CustomGateBuilder::new("Invalid")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.5, 0.0)],
        ])
        .build();

    match invalid_matrix {
        Err(e) => println!("Expected error for non-unitary matrix: {}\n", e),
        _ => println!("Unexpected: non-unitary matrix was accepted!\n"),
    }
}

/// Example 2: Validation features
fn example_2_validation_features() {
    println!("Example 2: Gate Validation Features\n");

    // Create a gate with explicit unitarity tolerance
    let hadamard = CustomGateBuilder::new("H")
        .matrix_2x2([
            [Complex64::new(1.0 / SQRT_2, 0.0), Complex64::new(1.0 / SQRT_2, 0.0)],
            [Complex64::new(1.0 / SQRT_2, 0.0), Complex64::new(-1.0 / SQRT_2, 0.0)],
        ])
        .tolerance(1e-12) // Strict tolerance
        .description("Hadamard with strict validation")
        .build()
        .expect("Failed to create Hadamard");

    println!("Created Hadamard with strict tolerance (1e-12)");
    println!("Gate is unitary: {}", hadamard.is_unitary());
    println!("Gate is hermitian: {}", hadamard.is_hermitian());

    // Demonstrate hermitian requirement
    println!("\nTesting hermitian requirement:");

    // Observable (must be hermitian)
    let observable_z = CustomGateBuilder::new("ObservableZ")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ])
        .require_hermitian(true)
        .description("Z observable (hermitian)")
        .build();

    match observable_z {
        Ok(_gate) => println!("✓ Observable Z created successfully (is hermitian)"),
        Err(e) => println!("✗ Error: {}", e),
    }

    // Try to create non-hermitian gate with hermitian requirement
    let non_hermitian = CustomGateBuilder::new("NonHermitian")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ])
        .require_hermitian(true)
        .build();

    match non_hermitian {
        Ok(_) => println!("Unexpected: non-hermitian gate was created"),
        Err(e) => println!("✓ Expected error for non-hermitian matrix: {}\n", e),
    }
}

/// Example 3: Parametric gates
fn example_3_parametric_gates() {
    println!("Example 3: Parametric Custom Gates\n");
    println!("Creating gates whose matrices depend on parameters\n");

    // Create a parametric RX gate
    let mut rx_gate = ParametricCustomGateBuilder::new("ParametricRX", 1)
        .with_parameters(vec!["theta"])
        .with_matrix_fn(|params| {
            let theta = params[0];
            let cos_half = (theta / 2.0).cos();
            let sin_half = (theta / 2.0).sin();
            vec![
                Complex64::new(cos_half, 0.0),
                Complex64::new(0.0, -sin_half),
                Complex64::new(0.0, -sin_half),
                Complex64::new(cos_half, 0.0),
            ]
        })
        .with_initial_params(vec![0.0])
        .build()
        .expect("Failed to create parametric RX gate");

    println!("Created parametric gate: ParametricRX");
    println!("Parameters: {:?}", rx_gate.parameter_names());
    println!("Is Hermitian at θ=0: {}", rx_gate.is_hermitian());

    // Update parameters
    println!("\nUpdating parameters to θ=π/4:");
    rx_gate
        .set_parameters(vec![PI / 4.0])
        .expect("Failed to update parameters");

    println!("Matrix updated successfully");
    println!("Matrix size: {} elements", rx_gate.matrix_vec().len());

    // Create a parametric CPhase gate
    println!("\n--- Parametric CPhase Gate ---");
    let cphase = ParametricCustomGateBuilder::new("ParametricCPhase", 2)
        .with_parameters(vec!["phi"])
        .with_matrix_fn(|params| {
            let phi = params[0];
            let exp_i_phi = Complex64::new(phi.cos(), phi.sin());
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                exp_i_phi,
            ]
        })
        .with_initial_params(vec![PI / 2.0])
        .build()
        .expect("Failed to create parametric CPhase gate");

    println!("Created parametric 2-qubit gate: ParametricCPhase");
    println!("Parameters: {:?}", cphase.parameter_names());
    println!("Initial parameter: φ=π/2\n");
}

/// Example 4: Gate registry
fn example_4_gate_registry() {
    println!("Example 4: Gate Registry\n");
    println!("Managing custom gates with the registry\n");

    let mut registry = GateRegistry::new();

    // Create and register multiple gates
    let pauli_x = CustomGateBuilder::new("X")
        .matrix_2x2([
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ])
        .build()
        .unwrap();

    let pauli_y = CustomGateBuilder::new("Y")
        .matrix_2x2([
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
            [Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
        ])
        .build()
        .unwrap();

    let pauli_z = CustomGateBuilder::new("Z")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
        ])
        .build()
        .unwrap();

    registry.register("my_x", pauli_x);
    registry.register("my_y", pauli_y);
    registry.register("my_z", pauli_z);

    println!("Registered {} gates\n", registry.len());

    // List all gates
    println!("--- Registered Gates ---");
    registry.print_gates();

    // Retrieve specific gates
    println!("\n--- Retrieving Specific Gates ---");
    if let Some(gate) = registry.get("my_x") {
        println!("Found gate: {} ({})", gate.name(), gate.num_qubits());
    }

    // Filter gates by qubit count
    println!("\n--- 1-Qubit Gates ---");
    let one_qubit = registry.gates_for_qubits(1);
    println!("Found {} 1-qubit gates:", one_qubit.len());
    for (name, gate) in one_qubit {
        println!("  - {}: {}", name, gate.name());
    }

    println!();
}

/// Example 5: Gate composition
fn example_5_gate_composition() {
    println!("Example 5: Gate Composition and Manipulation\n");

    // Create basic gates
    let pauli_x = CustomGateBuilder::new("X")
        .matrix_2x2([
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ])
        .build()
        .unwrap();

    // Compose X · X = I
    println!("Composing X · X (should equal identity):");
    let x_squared = pauli_x.compose(&pauli_x).expect("Failed to compose gates");
    println!("Result: {}", x_squared.name());

    // Calculate fidelity with identity
    let identity = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];

    let fidelity = x_squared.fidelity(&identity).expect("Failed to compute fidelity");
    println!("Fidelity with identity: {:.6}", fidelity);

    // Create adjoint
    println!("\nCreating adjoint (hermitian conjugate):");
    let phase_gate = CustomGateBuilder::new("S")
        .matrix_2x2([
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)], // i
        ])
        .build()
        .unwrap();

    let s_dagger = phase_gate.adjoint();
    println!("Original gate: {}", phase_gate.name());
    println!("Adjoint gate: {}", s_dagger.name());

    // Verify S · S† = I
    println!("\nVerifying S · S† = I:");
    let identity_check = phase_gate.compose(&s_dagger).expect("Failed to compose");
    let fidelity = identity_check
        .fidelity(&identity)
        .expect("Failed to compute fidelity");
    println!("Fidelity with identity: {:.6} ✓\n", fidelity);
}

/// Example 6: Controlled gates
fn example_6_controlled_gates() {
    println!("Example 6: Controlled Custom Gates\n");
    println!("Automatically generate controlled variants\n");

    // Create a base gate
    let pauli_x = CustomGateBuilder::new("X")
        .matrix_2x2([
            [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        ])
        .description("Pauli X gate")
        .build()
        .unwrap();

    println!("Base gate: {}", pauli_x.name());
    println!("Qubits: {}", pauli_x.num_qubits());

    // Create controlled version
    let controlled_x = pauli_x.controlled().expect("Failed to create controlled gate");

    println!("\nControlled gate: {}", controlled_x.name());
    println!("Qubits: {}", controlled_x.num_qubits());
    println!("Is Unitary: {}", controlled_x.is_unitary());

    // Create double-controlled version (CX -> CCX)
    let double_controlled = controlled_x
        .controlled()
        .expect("Failed to create double-controlled gate");

    println!("\nDouble-controlled gate: {}", double_controlled.name());
    println!("Qubits: {}", double_controlled.num_qubits());
    println!("Matrix size: {} x {}", 8, 8);
    println!("Is Unitary: {}", double_controlled.is_unitary());

    println!("\n=== Tutorial Complete ===\n");
}
