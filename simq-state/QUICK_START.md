# SparseState Quick Start Guide

## Installation

```rust
// Add to Cargo.toml
[dependencies]
simq-state = "0.1"

// In your code
use simq_state::SparseState;
use num_complex::Complex64;
```

## Creating States

```rust
// Initial state |0...0⟩
let state = SparseState::new(10)?;

// Specific basis state |i⟩
let state = SparseState::from_basis_state(10, 42)?;

// From dense vector
let amplitudes = vec![/* 2^n Complex64 */];
let state = SparseState::from_dense_amplitudes(10, &amplitudes)?;
```

## Working with Amplitudes

```rust
// Get amplitude (returns 0 if not stored)
let amp = state.get_amplitude(basis_idx);

// Set amplitude
state.set_amplitude(basis_idx, Complex64::new(0.5, 0.2));

// Normalize to unit norm
state.normalize()?;

// Check norm
let norm = state.norm();
```

## Single-Qubit Gates

```rust
// Hadamard
let h = [0.7071, 0.7071, 0.7071, -0.7071]
    .map(|x| Complex64::new(x, 0.0));
state.apply_single_qubit_gate(&h, qubit)?;

// Pauli X
let x = [
    Complex64::ZERO, Complex64::ONE,
    Complex64::ONE, Complex64::ZERO,
];
state.apply_single_qubit_gate(&x, qubit)?;

// RX(θ)
let rx_theta = [
    Complex64::new(theta.cos()/2.0, 0.0),
    Complex64::new(0.0, -theta.sin()/2.0),
    Complex64::new(0.0, -theta.sin()/2.0),
    Complex64::new(theta.cos()/2.0, 0.0),
];
state.apply_single_qubit_gate(&rx_theta, qubit)?;
```

## Two-Qubit Gates

```rust
// CNOT(control, target)
let cnot = [
    Complex64::ONE,  Complex64::ZERO, Complex64::ZERO, Complex64::ZERO,
    Complex64::ZERO, Complex64::ONE,  Complex64::ZERO, Complex64::ZERO,
    Complex64::ZERO, Complex64::ZERO, Complex64::ZERO, Complex64::ONE,
    Complex64::ZERO, Complex64::ZERO, Complex64::ONE,  Complex64::ZERO,
];
state.apply_two_qubit_gate(&cnot, control, target)?;

// CZ(q0, q1)
let cz = [
    Complex64::ONE,  Complex64::ZERO, Complex64::ZERO, Complex64::ZERO,
    Complex64::ZERO, Complex64::ONE,  Complex64::ZERO, Complex64::ZERO,
    Complex64::ZERO, Complex64::ZERO, Complex64::ONE,  Complex64::ZERO,
    Complex64::ZERO, Complex64::ZERO, Complex64::ZERO, -Complex64::ONE,
];
state.apply_two_qubit_gate(&cz, q0, q1)?;
```

## Measurement

```rust
// Get probabilities (no collapse)
let (p0, p1) = state.measure_probability(qubit)?;

// Measure and collapse
let prob = state.measure_and_collapse(qubit, outcome)?;
// outcome must be 0 or 1

// Expectation value P(|i⟩)
let exp = state.expectation_basis(basis_idx)?;
```

## Monitoring & Analysis

```rust
// Check state properties
println!("Qubits: {}", state.num_qubits());
println!("Dimension: {}", state.dimension());
println!("Non-zero: {}", state.num_amplitudes());
println!("Density: {:.2}%", state.density() * 100.0);
println!("Norm: {:.10}", state.norm());

// Should convert to dense?
if state.should_convert_to_dense() {
    let dense = state.to_dense();
}

// Adjust threshold
state.set_density_threshold(0.20);  // 20%

// Display
println!("{}", state);  // Short form
println!("{:?}", state); // Debug form
```

## Bell State Example

```rust
fn bell_state() -> Result<SparseState> {
    let mut state = SparseState::new(2)?;
    
    // H(0)
    let h = [0.7071, 0.7071, 0.7071, -0.7071]
        .map(|x| Complex64::new(x, 0.0));
    state.apply_single_qubit_gate(&h, 0)?;
    
    // CNOT(0, 1)
    let cnot = [
        Complex64::ONE, Complex64::ZERO, Complex64::ZERO, Complex64::ZERO,
        Complex64::ZERO, Complex64::ONE, Complex64::ZERO, Complex64::ZERO,
        Complex64::ZERO, Complex64::ZERO, Complex64::ZERO, Complex64::ONE,
        Complex64::ZERO, Complex64::ZERO, Complex64::ONE, Complex64::ZERO,
    ];
    state.apply_two_qubit_gate(&cnot, 0, 1)?;
    
    // Result: (|00⟩ + |11⟩)/√2
    Ok(state)
}
```

## Error Handling

```rust
use simq_state::StateError;

match SparseState::new(100) {
    Ok(_) => { /* success */ }
    Err(StateError::InvalidDimension { dimension }) => {
        eprintln!("Too many qubits: {}", dimension);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Performance Tips

1. **Sparse is best for**:
   - Product states (limited entanglement)
   - Shallow circuits (few gates)
   - Initial states

2. **Monitor density**:
   - Call `density()` after gates
   - Convert to dense if > threshold
   - Thresholds are tunable per use case

3. **Batch operations**:
   ```rust
   let mut amps = state.amplitudes_mut();
   // Bulk operations...
   drop(amps);
   state.update_density();
   ```

4. **Reuse states**:
   ```rust
   let mut state = SparseState::new(10)?;
   // Use in loop - avoids reallocation
   ```

## Common Patterns

### Superposition
```rust
let mut state = SparseState::new(n)?;
for i in 0..(1 << n) {
    state.set_amplitude(i as u64, Complex64::new(1.0 / ((1 << n) as f64).sqrt(), 0.0));
}
```

### Measurement Loop
```rust
for qubit in 0..n_qubits {
    let (p0, p1) = state.measure_probability(qubit)?;
    let outcome = if rand::random::<f64>() < p0 { 0 } else { 1 };
    state.measure_and_collapse(qubit, outcome)?;
}
```

### State Reconstruction
```rust
let dense = state.to_dense();
// Verify normalization
let norm: f64 = dense.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
assert!((norm - 1.0).abs() < 1e-10);
```

## Documentation

- **Detailed API**: See `SPARSE_STATE_API_REFERENCE.md`
- **Technical Details**: See `SPARSE_STATE_IMPLEMENTATION.md`
- **Examples**: Run `cargo run --example sparse_state_demo`

## Constants

```rust
const DEFAULT_DENSITY_THRESHOLD: f32 = 0.1;  // 10%
const AMPLITUDE_TOLERANCE: f64 = 1e-14;      // Zero detection
const NORMALIZE_TOLERANCE: f64 = 1e-10;      // Norm check
const MAX_QUBITS: usize = 30;                // Memory limit
```

---

**Ready to use!** Start with `let state = SparseState::new(n)?;` 🚀
