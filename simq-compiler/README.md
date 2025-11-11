# simq-compiler

Circuit optimization and compilation for the SimQ quantum computing SDK.

## Overview

The `simq-compiler` crate provides optimization passes that transform quantum circuits into more efficient representations. The primary focus is on reducing circuit depth and gate count while preserving quantum behavior.

## Features

### Gate Fusion

Gate fusion is an optimization that combines adjacent single-qubit gates operating on the same qubit into a single composite gate. This reduces:

- **Circuit depth**: Fewer sequential operations
- **Simulation time**: Single matrix multiply instead of multiple
- **Circuit complexity**: Easier to reason about and optimize further

#### How It Works

When multiple single-qubit gates act on the same qubit sequentially, they can be combined by multiplying their unitary matrices:

```
|ψ⟩ → U₃(U₂(U₁|ψ⟩)) = (U₃·U₂·U₁)|ψ⟩
```

The fused gate's matrix is the product of the individual gate matrices, applied in reverse order (rightmost gate is applied first).

#### Example

```rust
use simq_compiler::fusion::fuse_single_qubit_gates;
use simq_core::{Circuit, QubitId};
use simq_gates::standard::{Hadamard, PauliX, TGate};
use std::sync::Arc;

// Create a circuit with multiple single-qubit gates
let mut circuit = Circuit::new(1);
let q0 = QubitId::new(0);

circuit.add_gate(Arc::new(Hadamard) as Arc<dyn Gate>, &[q0]).unwrap();
circuit.add_gate(Arc::new(TGate) as Arc<dyn Gate>, &[q0]).unwrap();
circuit.add_gate(Arc::new(PauliX) as Arc<dyn Gate>, &[q0]).unwrap();

// Apply fusion optimization
let optimized = fuse_single_qubit_gates(&circuit, None).unwrap();

// Original: 3 operations
// Optimized: 1 fused operation
assert!(optimized.len() < circuit.len());
```

#### Configuration

The fusion pass can be customized via `FusionConfig`:

```rust
use simq_compiler::fusion::{fuse_single_qubit_gates, FusionConfig};

let config = FusionConfig {
    // Minimum gates required to form a fusion chain
    min_fusion_size: 2,

    // Automatically eliminate identity gates
    eliminate_identity: true,

    // Epsilon for identity detection
    identity_epsilon: 1e-10,

    // Optional limit on fusion chain length
    max_fusion_size: Some(10),
};

let optimized = fuse_single_qubit_gates(&circuit, Some(config)).unwrap();
```

#### Key Features

1. **Automatic Identity Elimination**: Gate sequences that result in identity (e.g., X·X, H·H) are automatically removed.

2. **Parameterized Gate Support**: Rotation gates (RX, RY, RZ) and phase gates are fully supported.

3. **Two-Qubit Gate Aware**: Correctly handles circuits with mixed single and two-qubit gates, breaking fusion chains appropriately.

4. **Configurable Limits**: Control minimum/maximum fusion sizes to balance optimization vs. gate complexity.

5. **Preserves Semantics**: The optimized circuit is mathematically equivalent to the original.

#### Performance

Gate fusion provides significant performance improvements:

- **Circuit Size Reduction**: Typical reduction of 50-80% for circuits with many single-qubit gates
- **Simulation Speedup**: 2-5x faster simulation for fusion-heavy circuits
- **Low Overhead**: Fusion analysis is O(n) in circuit size

Benchmark results (on a typical development machine):

```
gate_fusion/fuseable/10q_20g_total200
                        time:   [4.2 µs 4.3 µs 4.4 µs]

gate_fusion/fuseable/50q_50g_total2500
                        time:   [48 µs 49 µs 50 µs]

matrix_mult_2x2        time:   [8.5 ns 8.6 ns 8.7 ns]
```

#### Limitations

1. **Single-Qubit Only**: Currently only fuses single-qubit gates. Two-qubit gate fusion is planned for future releases.

2. **Sequential Gates**: Only adjacent gates on the same qubit are fused. Gates separated by other operations (even on different qubits) are not fused.

3. **Matrix Representation Required**: Custom gates without matrix representations cannot be fused.

## Architecture

### Matrix Utilities (`matrix_utils.rs`)

Provides efficient 2×2 complex matrix operations:

- `multiply_2x2`: Matrix multiplication for gate composition
- `is_identity`: Identity detection with configurable epsilon
- `matrices_approx_eq`: Approximate equality for floating-point matrices
- `frobenius_norm`: Matrix norm computation

### Fusion Analysis (`fusion.rs`)

Core gate fusion implementation:

- `FusedGate`: Represents a composite gate from multiple single-qubit gates
- `find_fusion_chains`: Analyzes circuits to identify fusion opportunities
- `fuse_gates`: Composes gate matrices into a single fused gate
- `fuse_single_qubit_gates`: Main entry point for the optimization pass

## Testing

The crate includes comprehensive tests covering:

- Basic gate fusion (Clifford gates, rotation gates)
- Identity elimination (self-inverse gates)
- Mixed circuits (single and two-qubit gates)
- Edge cases (empty circuits, single gates)
- Configuration options (min/max sizes, identity elimination)

Run tests:

```bash
cargo test --package simq-compiler
```

## Benchmarking

Performance benchmarks are available to measure fusion overhead and effectiveness:

```bash
cargo bench --package simq-compiler
```

Benchmarks include:

- Various circuit sizes (5-50 qubits)
- Different gate densities (10-50 gates per qubit)
- Rotation-heavy circuits
- Mixed single/two-qubit circuits
- Overhead analysis

## Examples

See [`examples/gate_fusion.rs`](examples/gate_fusion.rs) for comprehensive usage examples:

```bash
cargo run --package simq-compiler --example gate_fusion
```

Examples demonstrate:

1. Simple fusion of Clifford gates
2. Fusion of parameterized rotation gates
3. Mixed circuits with two-qubit gates
4. Identity elimination
5. Custom fusion configurations

## Future Work

Planned enhancements:

1. **Two-Qubit Gate Fusion**: Combine sequences like CX-CX or CZ-CZ
2. **Peephole Optimizations**: Pattern-based gate replacements (e.g., H-X-H → Z)
3. **Commutation-Based Fusion**: Reorder commuting gates to create fusion opportunities
4. **Hardware-Specific Compilation**: Target native gate sets for specific quantum hardware
5. **Circuit Synthesis**: Decompose fused gates into hardware-native operations

## Contributing

Contributions are welcome! Areas for improvement:

- Additional optimization passes
- Performance improvements
- Extended gate fusion heuristics
- Hardware-specific optimizations
- Documentation and examples

## License

MIT OR Apache-2.0 (same as SimQ project)
