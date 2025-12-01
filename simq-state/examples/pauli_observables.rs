//! Example demonstrating Pauli observable expectation values
//!
//! This example shows how to compute ⟨ψ|O|ψ⟩ for Pauli observables
//! without state collapse, which is essential for VQE and QAOA algorithms.

use num_complex::Complex64;
use simq_state::{DenseState, Pauli, PauliObservable, PauliString};

fn main() {
    println!("=== Pauli Observable Expectation Values ===\n");

    // Example 1: Single qubit observables
    example_single_qubit();

    // Example 2: Bell state measurements
    example_bell_state();

    // Example 3: VQE Hamiltonian
    example_vqe_hamiltonian();

    // Example 4: QAOA cost function
    example_qaoa_cost();

    // Example 5: Performance demonstration
    example_performance();
}

fn example_single_qubit() {
    println!("Example 1: Single Qubit Observables");
    println!("------------------------------------");

    // |0⟩ state
    let zero_state = DenseState::new(1).unwrap();

    // |+⟩ = (|0⟩ + |1⟩) / √2
    let plus_state = DenseState::from_amplitudes(
        1,
        &[
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ],
    )
    .unwrap();

    // |+i⟩ = (|0⟩ + i|1⟩) / √2
    let plus_i_state = DenseState::from_amplitudes(
        1,
        &[
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(0.0, 1.0 / 2_f64.sqrt()),
        ],
    )
    .unwrap();

    // Pauli observables
    let x_obs = PauliString::from_str("X").unwrap();
    let y_obs = PauliString::from_str("Y").unwrap();
    let z_obs = PauliString::from_str("Z").unwrap();

    println!("State |0⟩:");
    println!("  ⟨X⟩ = {:.4}", x_obs.expectation_value(&zero_state).unwrap());
    println!("  ⟨Y⟩ = {:.4}", y_obs.expectation_value(&zero_state).unwrap());
    println!("  ⟨Z⟩ = {:.4}", z_obs.expectation_value(&zero_state).unwrap());

    println!("\nState |+⟩:");
    println!("  ⟨X⟩ = {:.4}", x_obs.expectation_value(&plus_state).unwrap());
    println!("  ⟨Y⟩ = {:.4}", y_obs.expectation_value(&plus_state).unwrap());
    println!("  ⟨Z⟩ = {:.4}", z_obs.expectation_value(&plus_state).unwrap());

    println!("\nState |+i⟩:");
    println!("  ⟨X⟩ = {:.4}", x_obs.expectation_value(&plus_i_state).unwrap());
    println!("  ⟨Y⟩ = {:.4}", y_obs.expectation_value(&plus_i_state).unwrap());
    println!("  ⟨Z⟩ = {:.4}", z_obs.expectation_value(&plus_i_state).unwrap());

    println!();
}

fn example_bell_state() {
    println!("Example 2: Bell State Observables");
    println!("----------------------------------");

    // Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
    let bell_state = DenseState::from_amplitudes(
        2,
        &[
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ],
    )
    .unwrap();

    println!("Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2");

    // Two-qubit observables
    let zz = PauliString::from_str("ZZ").unwrap();
    let xx = PauliString::from_str("XX").unwrap();
    let yy = PauliString::from_str("YY").unwrap();
    let zi = PauliString::from_str("ZI").unwrap();
    let iz = PauliString::from_str("IZ").unwrap();

    println!("\nCorrelation measurements:");
    println!(
        "  ⟨ZZ⟩ = {:.4} (both qubits same Z eigenvalue)",
        zz.expectation_value(&bell_state).unwrap()
    );
    println!(
        "  ⟨XX⟩ = {:.4} (maximally correlated)",
        xx.expectation_value(&bell_state).unwrap()
    );
    println!("  ⟨YY⟩ = {:.4}", yy.expectation_value(&bell_state).unwrap());

    println!("\nSingle qubit measurements:");
    println!("  ⟨Z⊗I⟩ = {:.4} (first qubit)", zi.expectation_value(&bell_state).unwrap());
    println!("  ⟨I⊗Z⟩ = {:.4} (second qubit)", iz.expectation_value(&bell_state).unwrap());

    println!();
}

fn example_vqe_hamiltonian() {
    println!("Example 3: VQE Hamiltonian (H2 molecule)");
    println!("-----------------------------------------");

    // Simplified H2 Hamiltonian on 2 qubits:
    // H = -1.0*ZZ + 0.5*Z⊗I + 0.5*I⊗Z - 0.5*X⊗X
    let mut hamiltonian = PauliObservable::new();
    hamiltonian.add_term(PauliString::from_str("ZZ").unwrap(), -1.0);
    hamiltonian.add_term(PauliString::from_str("ZI").unwrap(), 0.5);
    hamiltonian.add_term(PauliString::from_str("IZ").unwrap(), 0.5);
    hamiltonian.add_term(PauliString::from_str("XX").unwrap(), -0.5);

    println!("Hamiltonian: H = -1.0·ZZ + 0.5·ZI + 0.5·IZ - 0.5·XX");

    // Test different trial states
    let states = vec![
        ("  |00⟩", DenseState::new(2).unwrap()),
        (
            "  |+⟩|+⟩",
            DenseState::from_amplitudes(
                2,
                &[
                    Complex64::new(0.5, 0.0),
                    Complex64::new(0.5, 0.0),
                    Complex64::new(0.5, 0.0),
                    Complex64::new(0.5, 0.0),
                ],
            )
            .unwrap(),
        ),
        (
            "  |Φ+⟩",
            DenseState::from_amplitudes(
                2,
                &[
                    Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
                ],
            )
            .unwrap(),
        ),
    ];

    println!("\nEnergy expectations for different states:");
    for (name, state) in states {
        let energy = hamiltonian.expectation_value(&state).unwrap();
        println!("{}: E = {:.4}", name, energy);
    }

    println!();
}

fn example_qaoa_cost() {
    println!("Example 4: QAOA Cost Function (MaxCut)");
    println!("---------------------------------------");

    // MaxCut on 4-node graph (linear chain)
    // Cost = -0.5 * (ZZ on edges)
    let mut cost_hamiltonian = PauliObservable::new();
    for i in 0..3 {
        let mut paulis = vec![Pauli::I; 4];
        paulis[i] = Pauli::Z;
        paulis[i + 1] = Pauli::Z;
        cost_hamiltonian.add_term(PauliString::from_paulis(paulis), -0.5);
    }

    println!("Graph: 0---1---2---3 (linear chain)");
    println!("Cost: C = -0.5·(Z₀Z₁ + Z₁Z₂ + Z₂Z₃)");

    // Test QAOA trial state (equal superposition)
    let dimension = 16;
    let amplitude = Complex64::new(1.0 / (dimension as f64).sqrt(), 0.0);
    let equal_superposition = DenseState::from_amplitudes(4, &vec![amplitude; dimension]).unwrap();

    let cost = cost_hamiltonian
        .expectation_value(&equal_superposition)
        .unwrap();
    println!("\nCost for equal superposition: {:.4}", cost);

    // Optimal cut: |0101⟩ or |1010⟩
    let mut optimal_amplitudes = vec![Complex64::new(0.0, 0.0); dimension];
    optimal_amplitudes[0b0101] = Complex64::new(1.0, 0.0);

    let optimal_state = DenseState::from_amplitudes(4, &optimal_amplitudes).unwrap();
    let optimal_cost = cost_hamiltonian.expectation_value(&optimal_state).unwrap();
    println!("Cost for optimal cut |0101⟩: {:.4}", optimal_cost);

    println!();
}

fn example_performance() {
    println!("Example 5: Performance Comparison");
    println!("----------------------------------");

    let num_qubits = 15;
    let dimension = 1 << num_qubits;

    // Create a random state
    let mut seed = 42u64;
    let amplitudes: Vec<Complex64> = (0..dimension)
        .map(|_| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let re = ((seed / 65536) % 32768) as f64 / 32768.0 - 0.5;
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let im = ((seed / 65536) % 32768) as f64 / 32768.0 - 0.5;
            Complex64::new(re, im)
        })
        .collect();

    let mut state = DenseState::from_amplitudes(num_qubits, &amplitudes).unwrap();
    state.normalize();

    println!("State: {} qubits ({} amplitudes)", num_qubits, dimension);

    // Benchmark diagonal observable
    let diagonal_obs = PauliString::all_z(num_qubits);
    let start = std::time::Instant::now();
    let _ = diagonal_obs.expectation_value(&state).unwrap();
    let diagonal_time = start.elapsed();

    println!("\nDiagonal observable (ZZZ...Z):");
    println!("  Time: {:?}", diagonal_time);

    // Benchmark non-diagonal observable
    let non_diagonal_paulis = vec![Pauli::X; num_qubits];
    let non_diagonal_obs = PauliString::from_paulis(non_diagonal_paulis);
    let start = std::time::Instant::now();
    let _ = non_diagonal_obs.expectation_value(&state).unwrap();
    let non_diagonal_time = start.elapsed();

    println!("\nNon-diagonal observable (XXX...X):");
    println!("  Time: {:?}", non_diagonal_time);
    println!(
        "  Speedup (diagonal): {:.2}x",
        non_diagonal_time.as_nanos() as f64 / diagonal_time.as_nanos() as f64
    );

    // Benchmark multi-term Hamiltonian
    let mut hamiltonian = PauliObservable::new();
    for i in 0..(num_qubits - 1) {
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[i] = Pauli::Z;
        paulis[i + 1] = Pauli::Z;
        hamiltonian.add_term(PauliString::from_paulis(paulis), -1.0);
    }

    let start = std::time::Instant::now();
    let energy = hamiltonian.expectation_value(&state).unwrap();
    let hamiltonian_time = start.elapsed();

    println!("\nMulti-term Hamiltonian ({} terms):", num_qubits - 1);
    println!("  Energy: {:.6}", energy);
    println!("  Time: {:?}", hamiltonian_time);
    println!("  Time per term: {:?}", hamiltonian_time / (num_qubits - 1) as u32);
}
