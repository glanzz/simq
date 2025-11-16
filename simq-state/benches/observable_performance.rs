use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_complex::Complex64;
use simq_state::{DenseState, Pauli, PauliObservable, PauliString};

// Create a random state for benchmarking
fn create_random_state(num_qubits: usize, seed: u64) -> DenseState {
    let dimension = 1 << num_qubits;
    let mut state = seed;

    let amplitudes: Vec<Complex64> = (0..dimension)
        .map(|_| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let re = ((state / 65536) % 32768) as f64 / 32768.0 - 0.5;
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let im = ((state / 65536) % 32768) as f64 / 32768.0 - 0.5;
            Complex64::new(re, im)
        })
        .collect();

    let mut dense_state = DenseState::from_amplitudes(num_qubits, &amplitudes).unwrap();
    dense_state.normalize();
    dense_state
}

fn bench_diagonal_pauli_expectation(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagonal_pauli_expectation");

    for &num_qubits in [5, 10, 15, 20].iter() {
        let dimension = 1 << num_qubits;
        group.throughput(Throughput::Elements(dimension as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_qubits", num_qubits)),
            &num_qubits,
            |b, &num_qubits| {
                let state = create_random_state(num_qubits, 42);
                let pauli = PauliString::all_z(num_qubits);

                b.iter(|| {
                    black_box(pauli.expectation_value(&state).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_non_diagonal_pauli_expectation(c: &mut Criterion) {
    let mut group = c.benchmark_group("non_diagonal_pauli_expectation");

    for &num_qubits in [5, 10, 15].iter() {
        let dimension = 1 << num_qubits;
        group.throughput(Throughput::Elements(dimension as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_qubits", num_qubits)),
            &num_qubits,
            |b, &num_qubits| {
                let state = create_random_state(num_qubits, 42);

                // Create XX...X Pauli string
                let paulis = vec![Pauli::X; num_qubits];
                let pauli = PauliString::from_paulis(paulis);

                b.iter(|| {
                    black_box(pauli.expectation_value(&state).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_mixed_pauli_expectation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_pauli_expectation");

    for &num_qubits in [5, 10, 15].iter() {
        let dimension = 1 << num_qubits;
        group.throughput(Throughput::Elements(dimension as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_qubits", num_qubits)),
            &num_qubits,
            |b, &num_qubits| {
                let state = create_random_state(num_qubits, 42);

                // Create alternating X and Z Pauli string
                let paulis: Vec<Pauli> = (0..num_qubits)
                    .map(|i| if i % 2 == 0 { Pauli::X } else { Pauli::Z })
                    .collect();
                let pauli = PauliString::from_paulis(paulis);

                b.iter(|| {
                    black_box(pauli.expectation_value(&state).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_pauli_observable_multi_term(c: &mut Criterion) {
    let mut group = c.benchmark_group("pauli_observable_multi_term");

    for &num_qubits in [5, 10, 15].iter() {
        let dimension = 1 << num_qubits;

        for &num_terms in [1, 5, 10, 20].iter() {
            group.throughput(Throughput::Elements((dimension * num_terms as usize) as u64));

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}q_{}t", num_qubits, num_terms)),
                &(num_qubits, num_terms),
                |b, &(num_qubits, num_terms)| {
                    let state = create_random_state(num_qubits, 42);

                    // Create observable with multiple diagonal terms
                    let mut observable = PauliObservable::new();
                    for i in 0..num_terms {
                        let mut paulis = vec![Pauli::I; num_qubits];
                        if (i as usize) < num_qubits {
                            paulis[i as usize] = Pauli::Z;
                        }
                        observable.add_term(PauliString::from_paulis(paulis), 1.0 / num_terms as f64);
                    }

                    b.iter(|| {
                        black_box(observable.expectation_value(&state).unwrap())
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_hamiltonian_expectation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hamiltonian_expectation");

    for &num_qubits in [5, 10, 15].iter() {
        let dimension = 1 << num_qubits;
        group.throughput(Throughput::Elements(dimension as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_qubits", num_qubits)),
            &num_qubits,
            |b, &num_qubits| {
                let state = create_random_state(num_qubits, 42);

                // Create a typical VQE Hamiltonian: sum of ZZ interactions
                let mut hamiltonian = PauliObservable::new();
                for i in 0..(num_qubits - 1) {
                    let mut paulis = vec![Pauli::I; num_qubits];
                    paulis[i] = Pauli::Z;
                    paulis[i + 1] = Pauli::Z;
                    hamiltonian.add_term(PauliString::from_paulis(paulis), -1.0);
                }

                // Add transverse field
                for i in 0..num_qubits {
                    let mut paulis = vec![Pauli::I; num_qubits];
                    paulis[i] = Pauli::X;
                    hamiltonian.add_term(PauliString::from_paulis(paulis), -0.5);
                }

                b.iter(|| {
                    black_box(hamiltonian.expectation_value(&state).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_single_qubit_observables(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_qubit_observables");

    for &num_qubits in [10, 15, 20].iter() {
        let dimension = 1 << num_qubits;
        group.throughput(Throughput::Elements(dimension as u64));

        // Benchmark Z observable
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_qubits_Z", num_qubits)),
            &num_qubits,
            |b, &num_qubits| {
                let state = create_random_state(num_qubits, 42);
                let observable = PauliObservable::single_z(num_qubits, 0);

                b.iter(|| {
                    black_box(observable.expectation_value(&state).unwrap())
                });
            },
        );

        // Benchmark X observable
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_qubits_X", num_qubits)),
            &num_qubits,
            |b, &num_qubits| {
                let state = create_random_state(num_qubits, 42);
                let mut paulis = vec![Pauli::I; num_qubits];
                paulis[0] = Pauli::X;
                let pauli_string = PauliString::from_paulis(paulis);
                let observable = PauliObservable::from_pauli_string(pauli_string, 1.0);

                b.iter(|| {
                    black_box(observable.expectation_value(&state).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_diagonal_vs_non_diagonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagonal_vs_non_diagonal");

    let num_qubits = 15;
    let state = create_random_state(num_qubits, 42);
    let dimension = 1 << num_qubits;

    group.throughput(Throughput::Elements(dimension as u64));

    // Diagonal (all Z)
    group.bench_function("diagonal_ZZZ", |b| {
        let pauli = PauliString::all_z(num_qubits);
        b.iter(|| {
            black_box(pauli.expectation_value(&state).unwrap())
        });
    });

    // Non-diagonal (all X)
    group.bench_function("non_diagonal_XXX", |b| {
        let paulis = vec![Pauli::X; num_qubits];
        let pauli = PauliString::from_paulis(paulis);
        b.iter(|| {
            black_box(pauli.expectation_value(&state).unwrap())
        });
    });

    // Mixed (alternating)
    group.bench_function("mixed_XZXZ", |b| {
        let paulis: Vec<Pauli> = (0..num_qubits)
            .map(|i| if i % 2 == 0 { Pauli::X } else { Pauli::Z })
            .collect();
        let pauli = PauliString::from_paulis(paulis);
        b.iter(|| {
            black_box(pauli.expectation_value(&state).unwrap())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_diagonal_pauli_expectation,
    bench_non_diagonal_pauli_expectation,
    bench_mixed_pauli_expectation,
    bench_pauli_observable_multi_term,
    bench_hamiltonian_expectation,
    bench_single_qubit_observables,
    bench_diagonal_vs_non_diagonal
);

criterion_main!(benches);
