# Custom Quantum Gates Feature Guide

## Overview

SimQ provides a powerful and flexible system for creating, managing, and manipulating custom quantum gates. This guide covers all aspects of the custom gates feature, from basic creation to advanced usage patterns.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Static Custom Gates](#static-custom-gates)
3. [Parametric Custom Gates](#parametric-custom-gates)
4. [Gate Registry](#gate-registry)
5. [Gate Validation](#gate-validation)
6. [Gate Composition](#gate-composition)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)
9. [Performance Considerations](#performance-considerations)

## Quick Start

### Creating Your First Custom Gate

```rust
use simq_gates::custom::CustomGateBuilder;
use num_complex::Complex64;
use std::f64::consts::SQRT_2;

// Create a Hadamard gate
let inv_sqrt2 = 1.0 / SQRT_2;
let hadamard = CustomGateBuilder::new("MyHadamard")
    .matrix_2x2([
        [Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0)],
        [Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0)],
    ])
    .description("Custom Hadamard gate")
    .build()
    .expect("Failed to create gate");

println!("Created: {}", hadamard.name());
```

### Using a Custom Gate in a Circuit

```rust
use simq_core::circuit_builder::CircuitBuilder;
use std::sync::Arc;

// Create a 3-qubit circuit
let mut builder = CircuitBuilder::<3>::new();
let [q0, q1, q2] = builder.qubits();

// Apply custom gate
let gate = Arc::new(hadamard);
builder.apply_gate(gate, &[q0])?;
```

## Static Custom Gates

Static gates have fixed matrices that don't change at runtime. They're ideal for user-defined operations that remain constant throughout your quantum program.

### Basic Creation

The `CustomGateBuilder` provides a fluent API for creating static gates:

```rust
use simq_gates::custom::CustomGateBuilder;
use num_complex::Complex64;

// Single-qubit gate (2x2 matrix)
let x_gate = CustomGateBuilder::new("X")
    .matrix_2x2([
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ])
    .description("Pauli X gate")
    .build()?;

// Two-qubit gate (4x4 matrix)
let cnot_like = CustomGateBuilder::new("CustomCNOT")
    .matrix_4x4([
        [/* 4x4 matrix entries */]
    ])
    .build()?;
```

### Matrix Input Options

```rust
// Option 1: Flattened vector (row-major order)
let gate1 = CustomGateBuilder::new("G1")
    .num_qubits(1)
    .matrix(vec![
        Complex64::new(a, b), Complex64::new(c, d),
        Complex64::new(e, f), Complex64::new(g, h),
    ])
    .build()?;

// Option 2: 2x2 array (1-qubit)
let gate2 = CustomGateBuilder::new("G2")
    .matrix_2x2([[Complex64::new(a, b), /*...*/]])
    .build()?;

// Option 3: 4x4 array (2-qubit)
let gate3 = CustomGateBuilder::new("G3")
    .matrix_4x4([[Complex64::new(a, b), /*...*/]])
    .build()?;
```

## Parametric Custom Gates

Parametric gates allow you to create gates whose matrices depend on parameters, similar to RX, RY, RZ gates but with arbitrary unitary operations.

### Creating Parametric Gates

```rust
use simq_gates::custom::ParametricCustomGateBuilder;
use std::f64::consts::PI;

let mut rx_gate = ParametricCustomGateBuilder::new("MyRX", 1)
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
    .build()?;

// Update parameters
rx_gate.set_parameters(vec![PI / 4.0])?;
```

### Multi-Parameter Gates

```rust
let mut su3_gate = ParametricCustomGateBuilder::new("SU3", 1)
    .with_parameters(vec!["theta", "phi", "lambda"])
    .with_matrix_fn(|params| {
        let theta = params[0];
        let phi = params[1];
        let lambda = params[2];
        
        // Compute U3 matrix
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let exp_i_phi = Complex64::new(phi.cos(), phi.sin());
        let exp_i_lambda = Complex64::new(lambda.cos(), lambda.sin());
        
        vec![
            Complex64::new(cos_half, 0.0),
            -exp_i_lambda * sin_half,
            exp_i_phi * sin_half,
            exp_i_phi * exp_i_lambda * cos_half,
        ]
    })
    .with_initial_params(vec![0.0, 0.0, 0.0])
    .build()?;
```

## Gate Registry

The `GateRegistry` provides centralized management of custom gates, making it easy to store and retrieve gates throughout your application.

### Basic Usage

```rust
use simq_gates::gate_registry::GateRegistry;

let mut registry = GateRegistry::new();

// Register gates
registry.register("my_x", x_gate);
registry.register("my_h", hadamard);

// Retrieve gates
if let Some(gate) = registry.get("my_x") {
    println!("Found gate: {}", gate.name());
}

// Check existence
if registry.contains("my_h") {
    println!("Hadamard is registered");
}
```

### Advanced Registry Operations

```rust
// List all gate names
let names = registry.gate_names();
println!("Registered gates: {:?}", names);

// Get gates for specific qubit count
let single_qubit_gates = registry.gates_for_qubits(1);
let two_qubit_gates = registry.gates_for_qubits(2);

// Get detailed information
let gate_info = registry.list_gates();
for info in gate_info {
    println!("{}: {} qubits, hermitian: {}", 
             info.name, info.num_qubits, info.is_hermitian);
}

// Print formatted table
registry.print_gates();

// Unregister gates
let old_gate = registry.unregister("my_x");

// Clear all
registry.clear();
```

## Gate Validation

All custom gates are automatically validated to ensure they satisfy quantum mechanical properties.

### Validation Properties

1. **Unitarity**: U†U = I (automatically checked)
2. **Proper Dimensions**: Matrix size must be 2^n × 2^n
3. **No Invalid Values**: No NaN or infinite entries
4. **Determinant Check**: |det(U)| = 1 for unitary matrices

### Validation Options

```rust
// Default validation (tolerance = 1e-10)
let gate = CustomGateBuilder::new("G")
    .matrix_2x2([[/* ... */]])
    .build()?;

// Strict validation
let gate = CustomGateBuilder::new("G")
    .matrix_2x2([[/* ... */]])
    .tolerance(1e-12)
    .build()?;

// Require hermitian property
let observable = CustomGateBuilder::new("Observable")
    .matrix_2x2([[/* ... */]])
    .require_hermitian(true)
    .build()?;
```

### Error Handling

```rust
use simq_gates::custom::CustomGateError;

match CustomGateBuilder::new("Invalid")
    .matrix_2x2([[/* non-unitary matrix */]])
    .build() {
    Ok(gate) => println!("Gate created: {}", gate.name()),
    Err(CustomGateError::NotUnitary { max_deviation, tolerance }) => {
        eprintln!("Matrix not unitary: {} > {}", max_deviation, tolerance);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Gate Composition

Custom gates can be composed, manipulated, and compared.

### Composition

```rust
// Compose two gates: U · V
let composed = gate_u.compose(&gate_v)?;
assert_eq!(composed.num_qubits(), gate_u.num_qubits());
```

### Adjoint (Hermitian Conjugate)

```rust
// Get adjoint gate: U†
let adjoint = gate.adjoint();

// Verify U · U† = I
let identity_check = gate.compose(&adjoint)?;
let fidelity = identity_check.fidelity(&identity_matrix)?;
assert!((fidelity - 1.0).abs() < 1e-10);
```

### Fidelity Computation

```rust
// Compare two gates
let target_matrix = vec![/* target unitary */];
let fidelity = custom_gate.fidelity(&target_matrix)?;

if fidelity > 0.99 {
    println!("Gate has high fidelity with target");
}
```

### Controlled Gates

```rust
// Create controlled version automatically
let cx = x_gate.controlled()?;
assert_eq!(cx.num_qubits(), 2);
assert_eq!(cx.name(), "CX");

// Create multiply-controlled gates
let ccx = cx.controlled()?; // CCX (Toffoli)
assert_eq!(ccx.num_qubits(), 3);
```

## Advanced Features

### Gate Properties

```rust
// Query gate properties
let name = gate.name();
let num_qubits = gate.num_qubits();
let is_unitary = gate.is_unitary();
let is_hermitian = gate.is_hermitian();
let description = gate.description();

// Get matrix representation
if let Some(matrix) = gate.matrix() {
    println!("Matrix has {} elements", matrix.len());
}
```

### Validation Module

```rust
use simq_gates::custom::validation;

// Check trace preservation
let is_tp = validation::is_trace_preserving(&matrix, 1e-10);

// Validate quantum gate properties
validation::validate_quantum_gate(&matrix, num_qubits, 1e-10)?;

// Check completeness relation for Kraus operators
validation::check_completeness_relation(&kraus_operators, 1e-10)?;
```

### Converting Parametric to Static

```rust
// Create static gate from parametric gate with current parameters
let static_gate = parametric_gate.to_static_gate()?;
```

## Best Practices

### 1. Validation Before Use

Always verify your custom gates work correctly:

```rust
// Create gate
let gate = CustomGateBuilder::new("MyGate")
    .matrix_2x2([[/* ... */]])
    .build()?;

// Verify it's unitary and hermitian if needed
assert!(gate.is_unitary());
if required_hermitian {
    assert!(gate.is_hermitian());
}
```

### 2. Use Descriptive Names

```rust
// Good
let toffoli = CustomGateBuilder::new("Toffoli")
    .description("3-qubit controlled-controlled-X gate")
    .matrix_4x4([[/* ... */]])
    .build()?;

// Avoid
let gate = CustomGateBuilder::new("G")
    .matrix_4x4([[/* ... */]])
    .build()?;
```

### 3. Registry for Reusable Gates

```rust
// Create registry once at startup
let mut gate_registry = GateRegistry::new();

// Register all custom gates
for gate in create_custom_gates() {
    registry.register(gate.name(), gate);
}

// Reuse throughout application
let my_gate = gate_registry.get("my_gate")?;
```

### 4. Strict Tolerance for Critical Gates

```rust
// For gates used in error correction or precision measurements
let precise_gate = CustomGateBuilder::new("Precise")
    .matrix_2x2([[/* ... */]])
    .tolerance(1e-14) // Very strict
    .build()?;
```

### 5. Document Gate Purpose

```rust
let gate = CustomGateBuilder::new("ApproximateU3")
    .matrix_2x2([[/* ... */]])
    .description("Approximate U3 gate for hardware XYZ rotation block")
    .build()?;
```

## Performance Considerations

### 1. Gate Validation Overhead

- Single-qubit gates: < 1 microsecond
- Two-qubit gates: ~10 microseconds
- Larger gates: O(2^(3n)) where n is qubit count

For performance-critical code, reuse gates rather than recreating them.

### 2. Matrix Representation

```rust
// Store matrices efficiently
let matrix = gate.matrix_vec(); // Zero-copy reference
let matrix_clone = gate.matrix(); // Creates clone (if needed)
```

### 3. Composition vs. Pre-computed

```rust
// For frequently composed gates, consider pre-computing
let pre_computed = gate1.compose(&gate2)?;

// vs. composing on demand
for _ in 0..1000000 {
    let result = gate1.compose(&gate2)?;
}
```

### 4. Parametric Gate Updates

```rust
// Efficient: reuse gate, update parameters
let mut param_gate = ParametricCustomGateBuilder::new("RX", 1)
    .with_parameters(vec!["theta"])
    .with_matrix_fn(/* ... */)
    .build()?;

for angle in angles {
    param_gate.set_parameters(vec![angle])?;
    // Use gate
}

// Inefficient: create new gate each time
for angle in angles {
    let gate = ParametricCustomGateBuilder::new("RX", 1)
        .with_parameters(vec!["theta"])
        .with_matrix_fn(/* ... */)
        .with_initial_params(vec![angle])
        .build()?;
    // Use gate
}
```

## Examples

### Complete Example: VQE with Custom Gates

```rust
use simq_gates::custom::CustomGateBuilder;
use simq_gates::gate_registry::GateRegistry;
use std::sync::Arc;

// Create custom ansatz gates
let mut registry = GateRegistry::new();

// Register all parameterized layers
for layer in 0..num_layers {
    let ansatz_gate = create_ansatz_gate(layer)?;
    registry.register(format!("ansatz_layer_{}", layer), ansatz_gate);
}

// Use in VQE loop
for iteration in 0..num_iterations {
    // Get gates from registry
    let gate = registry.get(&format!("ansatz_layer_0"))?;
    
    // Apply to circuit
    builder.apply_gate(Arc::new(gate), &qubits)?;
}
```

### Custom Observable

```rust
// Create a custom observable (must be hermitian)
let custom_observable = CustomGateBuilder::new("CustomObservable")
    .matrix_2x2([
        [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)],
    ])
    .require_hermitian(true)
    .description("Custom measurement observable")
    .build()?;

assert!(custom_observable.is_hermitian());
```

## Troubleshooting

### "Matrix is not unitary"

Ensure U†U = I within tolerance:
```rust
// Debug: Check matrix manually
let adjoint = compute_adjoint(my_matrix);
let product = matrix_multiply(&adjoint, my_matrix);
// Should equal identity
```

### "Invalid dimensions"

Matrix size must be 2^n × 2^n for n-qubit gates:
- 1-qubit: 2×2 = 4 elements
- 2-qubit: 4×4 = 16 elements
- 3-qubit: 8×8 = 64 elements

### "Matrix contains NaN or infinite values"

Check for:
- Division by zero
- Invalid trigonometric values
- Overflow in calculations

## See Also

- [Standard Gates](./STANDARD_GATES.md)
- [Matrix Operations](./matrix_ops.rs)
- [Circuit Builder](../simq-core/docs/circuit_builder.md)
