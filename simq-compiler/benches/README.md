# Optimization Pass Benchmarks

This directory contains comprehensive benchmarks for the SimQ compiler's optimization passes.

## Running Benchmarks

### Run All Benchmarks
```bash
cargo bench --bench optimization_passes
```

### Run Specific Benchmark Groups

**Individual Pass Benchmarks:**
```bash
# Dead code elimination
cargo bench --bench optimization_passes -- dead_code_elimination

# Template matching
cargo bench --bench optimization_passes -- template_matching

# Gate commutation
cargo bench --bench optimization_passes -- gate_commutation

# Gate fusion pass
cargo bench --bench optimization_passes -- gate_fusion_pass
```

**Full Pipeline Benchmarks:**
```bash
# All optimization levels (O0, O1, O2, O3)
cargo bench --bench optimization_passes -- optimization_levels

# Pass combinations
cargo bench --bench optimization_passes -- pass_combinations

# Scalability tests
cargo bench --bench optimization_passes -- scalability
```

**Legacy Gate Fusion Benchmarks:**
```bash
# Original gate fusion benchmarks
cargo bench --bench optimization_passes -- gate_fusion
cargo bench --bench optimization_passes -- rotation_fusion
```

### Quick Benchmarks
For faster iteration during development, use `--quick` mode:
```bash
cargo bench --bench optimization_passes -- --quick
```

## Benchmark Organization

### Individual Pass Benchmarks

These benchmarks measure the performance of each optimization pass in isolation:

- **`bench_dead_code_elimination`**: Tests removal of self-inverse gate pairs (X-X, H-H, etc.)
  - Circuit sizes: 5, 10, 20, 50 qubits
  - Pair counts: 5, 10, 20 pairs per qubit
  - Total gates: up to 2000 gates

- **`bench_template_matching`**: Tests advanced pattern matching and replacement
  - Patterns: H-Z-H → X, H-X-H → Z, X-Y-X → Y
  - Circuit sizes: 5, 10, 20, 50 qubits
  - Pattern counts: 5, 10, 20 patterns per qubit

- **`bench_gate_commutation`**: Tests gate reordering for optimization
  - Circuit sizes: 5, 10, 20, 50 qubits
  - Interleaved gate patterns

- **`bench_gate_fusion_pass`**: Tests single-qubit gate fusion
  - Circuit sizes: 5, 10, 20, 50 qubits
  - Gates per qubit: 10, 20, 50

### Full Pipeline Benchmarks

These benchmarks measure the performance of complete optimization pipelines:

- **`bench_optimization_levels`**: Compares O0, O1, O2, O3 optimization levels
  - Uses realistic quantum circuits with mixed patterns
  - Circuit sizes: 5, 10, 20 qubits
  - Shows tradeoff between optimization time and quality

- **`bench_pass_combinations`**: Tests synergy between passes
  - Single pass performance vs. combined passes
  - Shows how passes work together in the full pipeline

- **`bench_scalability`**: Tests how compilation time scales with circuit size
  - Circuit sizes: 10, 25, 50, 100 qubits
  - Uses O2 optimization level
  - Smaller sample size for faster results

### Legacy Benchmarks

These maintain compatibility with the original gate fusion implementation:

- **`bench_gate_fusion`**: Original gate fusion benchmarks
- **`bench_rotation_fusion`**: Rotation gate fusion
- **`bench_mixed_circuit`**: Mixed single/two-qubit gates
- **`bench_matrix_multiplication`**: 2x2 matrix operations
- **`bench_fusion_overhead`**: Fusion analysis overhead

## Interpreting Results

### Performance Metrics

Criterion.rs provides detailed statistics:
- **Time**: Mean execution time with confidence intervals
- **Throughput**: Operations per second (for some benchmarks)
- **Change Detection**: Automatically detects performance regressions

### Example Output
```
dead_code_elimination/self_inverse/50q_20pairs_2000gates
                        time:   [114.97 µs 115.28 µs 116.54 µs]
```

This shows:
- Circuit: 50 qubits, 20 pairs per qubit, 2000 total gates
- Mean time: 115.28 µs
- Confidence interval: [114.97 µs, 116.54 µs]

### Optimization Level Performance

Expected performance characteristics:

- **O0**: No optimization (baseline)
  - Fastest compilation time
  - No circuit improvement

- **O1**: Basic optimizations (Dead Code Elimination only)
  - ~10-50x slower than O0
  - Removes redundant gates

- **O2**: Standard optimization (DCE + Template Matching + Fusion)
  - ~100-200x slower than O0
  - Recommended for most use cases
  - Best balance of compilation time and circuit quality

- **O3**: Maximum optimization (O2 + Commutation + extra iterations)
  - ~120-250x slower than O0
  - Most aggressive optimization
  - Use for production deployments

### Scalability

The compiler should exhibit roughly linear scaling with circuit size:
- 10 qubits (~119 gates): ~78 µs
- 20 qubits (~239 gates): ~150 µs
- 50 qubits (~600 gates): ~375 µs (estimated)

## Circuit Generators

The benchmarks use several circuit generators:

- **`create_fuseable_circuit`**: Single-qubit gates that can be fused
- **`create_self_inverse_circuit`**: Self-inverse pairs (X-X, H-H, etc.)
- **`create_template_circuit`**: Template patterns (H-Z-H, H-X-H, etc.)
- **`create_commutation_circuit`**: Interleaved gates on different qubits
- **`create_realistic_circuit`**: Mixed patterns resembling real quantum algorithms
- **`create_rotation_circuit`**: Parameterized rotation gates
- **`create_mixed_circuit`**: Single and two-qubit gates

## Viewing Reports

Criterion generates detailed HTML reports in `target/criterion/`:

```bash
# Open the main report
open target/criterion/report/index.html

# Open a specific benchmark group
open target/criterion/dead_code_elimination/report/index.html
```

## Baseline Comparison

To compare against a baseline:

```bash
# Save current performance as baseline
cargo bench --bench optimization_passes -- --save-baseline main

# Run benchmarks and compare to baseline
cargo bench --bench optimization_passes -- --baseline main
```

## Adding New Benchmarks

To add a new benchmark:

1. Create a circuit generator function
2. Create a benchmark function following the pattern:
```rust
fn bench_my_pass(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_pass");

    for size in [5, 10, 20].iter() {
        let circuit = create_my_circuit(*size);

        group.bench_with_input(
            BenchmarkId::new("description", format!("{}q", size)),
            &circuit,
            |b, circuit| {
                b.iter(|| {
                    let mut c = circuit.clone();
                    // Benchmark code here
                    black_box(c)
                })
            },
        );
    }

    group.finish();
}
```
3. Add to the `criterion_group!` macro

## Performance Goals

Target performance for O2 optimization level:
- 10 qubits: < 100 µs
- 50 qubits: < 500 µs
- 100 qubits: < 2 ms

These targets aim for 8-10x better performance than Qiskit transpilation.

## Troubleshooting

### Long Benchmark Times

If benchmarks take too long:
- Use `--quick` mode for faster iteration
- Reduce sample size with `group.sample_size(10)`
- Run specific benchmark groups only

### Inconsistent Results

If results vary significantly:
- Close other applications
- Ensure the system is not thermally throttling
- Run benchmarks multiple times
- Use `cargo bench -- --noplot` to skip HTML generation

### Memory Issues

For very large circuits:
- Reduce the maximum circuit size
- Reduce the number of iterations
- Run benchmarks sequentially instead of in parallel
