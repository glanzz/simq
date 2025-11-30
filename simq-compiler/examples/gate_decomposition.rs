//! Gate Decomposition Example
//!
//! This example demonstrates the comprehensive gate decomposition capabilities
//! in SimQ, showing how to decompose arbitrary quantum gates into various
//! basis gate sets used by different quantum hardware vendors.
//!
//! Run with: cargo run --package simq-compiler --example gate_decomposition

use simq_compiler::decomposition::{
    clifford_t::{CliffordTDecomposer, GridSynthConfig},
    multi_qubit::MultiQubitDecomposer,
    single_qubit::{EulerAngles, EulerBasis, SingleQubitDecomposer},
    two_qubit::{EntanglementGate, TwoQubitDecomposer},
    BasisGateSet, Decomposer, DecompositionConfig,
};
use std::f64::consts::PI;

fn main() {
    println!("=== SimQ Gate Decomposition Examples ===\n");

    // Example 1: Single-Qubit Decomposition
    single_qubit_example();

    // Example 2: Two-Qubit Decomposition
    two_qubit_example();

    // Example 3: Multi-Qubit Decomposition (Toffoli)
    multi_qubit_example();

    // Example 4: Clifford+T Decomposition
    clifford_t_example();

    // Example 5: Basis Gate Sets
    basis_gate_sets_example();

    // Example 6: Decomposition Configuration
    configuration_example();
}

fn single_qubit_example() {
    println!("1. Single-Qubit Gate Decomposition");
    println!("=====================================\n");

    // ZYZ Euler decomposition
    println!("ZYZ Decomposition (Rz-Ry-Rz):");
    let decomposer = SingleQubitDecomposer::new(EulerBasis::ZYZ);
    println!("  - Basis: {}", decomposer.name());
    println!("  - Universal: Any single-qubit unitary can be written as:");
    println!("    U = e^(iα) Rz(β) Ry(γ) Rz(δ)\n");

    // Example: Hadamard gate angles
    println!("  Example: Hadamard gate");
    let angles = EulerAngles::new(0.0, 0.0, PI / 2.0, PI);
    println!(
        "    α = {:.4}, β = {:.4}, γ = {:.4}, δ = {:.4}",
        angles.alpha, angles.beta, angles.gamma, angles.delta
    );
    println!("    Gate count: {}\n", angles.gate_count());

    // ZXZ decomposition
    println!("ZXZ Decomposition (Rz-Rx-Rz):");
    let _zxz_decomposer = SingleQubitDecomposer::new(EulerBasis::ZXZ);
    println!("  - Useful when X rotations are cheaper than Y rotations");
    println!("  - Common in some superconducting qubit architectures\n");

    // U3 decomposition (IBM)
    println!("U3 Decomposition (IBM Quantum):");
    let _u3_decomposer = SingleQubitDecomposer::new(EulerBasis::U3);
    println!("  - IBM's native single-qubit gate");
    println!("  - U3(θ, φ, λ) = Rz(φ) Ry(θ) Rz(λ)");
    println!("  - Discards global phase\n");

    println!();
}

fn two_qubit_example() {
    println!("2. Two-Qubit Gate Decomposition");
    println!("=================================\n");

    println!("Canonical Decomposition:");
    println!("  - Any 2-qubit unitary requires at most 3 CNOTs");
    println!("  - U = (A₁⊗A₂) CNOT (B₁⊗B₂) CNOT (C₁⊗C₂) CNOT (D₁⊗D₂)\n");

    // SWAP decomposition
    println!("SWAP Gate Decomposition:");
    let decomposer = TwoQubitDecomposer::new(EntanglementGate::CNOT);
    let swap_gates = decomposer.decompose_swap();
    println!("  - SWAP = CNOT₀₁ · CNOT₁₀ · CNOT₀₁");
    println!("  - Requires: 3 CNOTs");
    println!("  - Gate sequence length: {}\n", swap_gates.len());

    // iSWAP decomposition
    println!("iSWAP Gate Decomposition:");
    let iswap_gates = decomposer.decompose_iswap();
    println!("  - iSWAP can be decomposed into 2 CNOTs + local gates");
    println!("  - Total gates: {}", iswap_gates.len());
    println!("  - Native to some superconducting architectures\n");

    // Entangling gate conversions
    println!("Entangling Gate Conversions:");
    println!("  CNOT → CZ:");
    let cnot_to_cz = TwoQubitDecomposer::cnot_to_cz();
    println!("    Requires: {} gates (CNOT = H₁ CZ H₁)", cnot_to_cz.len());

    println!("  CZ → CNOT:");
    let cz_to_cnot = TwoQubitDecomposer::cz_to_cnot();
    println!("    Requires: {} gates\n", cz_to_cnot.len());

    println!();
}

fn multi_qubit_example() {
    println!("3. Multi-Qubit Gate Decomposition");
    println!("===================================\n");

    let decomposer = MultiQubitDecomposer::new();

    // Toffoli (CCNOT)
    println!("Toffoli Gate (CCNOT):");
    let toffoli_gates = decomposer.decompose_toffoli_relative_phase();
    println!("  - 3-qubit controlled-controlled-NOT gate");
    println!("  - Relative-phase decomposition: {} gates", toffoli_gates.len());
    println!("  - Includes: 6 CNOTs, 7 T gates");
    println!("  - Optimal for Clifford+T compilation\n");

    // Toffoli with ancilla
    println!("Toffoli with Ancilla:");
    let toffoli_ancilla = decomposer.decompose_toffoli_with_ancilla();
    println!("  - Uses 1 ancilla qubit");
    println!("  - Gate count: {}", toffoli_ancilla.len());
    println!("  - Trades CNOTs for ancilla qubits\n");

    // Fredkin (CSWAP)
    println!("Fredkin Gate (CSWAP):");
    let fredkin_gates = decomposer.decompose_fredkin();
    println!("  - Controlled-SWAP gate");
    println!("  - Gate count: {}", fredkin_gates.len());
    println!("  - Composed from Toffoli gates\n");

    // Multi-controlled X
    println!("Multi-Controlled X (MCX):");
    for n in 1..=5 {
        let cost = decomposer.estimate_mcx_cost(n);
        println!("  - {}-controlled X: ~{} gates", n, cost);
    }
    println!("  - Linear decomposition: O(n²) gates");
    println!("  - Logarithmic with ancillas: O(n log n) gates\n");

    println!();
}

fn clifford_t_example() {
    println!("4. Clifford+T Decomposition");
    println!("============================\n");

    println!("Fault-Tolerant Quantum Computing:");
    println!("  - Basis: {{H, S, T, CNOT}}");
    println!("  - H: Hadamard (Clifford)");
    println!("  - S: Phase gate √Z (Clifford)");
    println!("  - T: π/8 gate √S (non-Clifford)");
    println!("  - CNOT: Entangling gate (Clifford)\n");

    let config = GridSynthConfig {
        epsilon: 1e-10,
        max_gates: 100,
        optimize_t_count: true,
        optimize_t_depth: false,
    };

    let decomposer = CliffordTDecomposer::with_config(config);

    // Exact rotations
    println!("Exact Rz Rotations:");
    println!("  - Rz(π/4) = T");
    let rz_pi4 = decomposer.decompose_rz(PI / 4.0);
    println!("    Gates: {:?}", rz_pi4);

    println!("  - Rz(π/2) = S");
    let rz_pi2 = decomposer.decompose_rz(PI / 2.0);
    println!("    Gates: {:?}", rz_pi2);

    println!("  - Rz(π) = Z = S²");
    let rz_pi = decomposer.decompose_rz(PI);
    println!("    Gates: {:?}\n", rz_pi);

    // T-count optimization
    println!("T-Count Optimization:");
    println!("  - T gates are expensive in fault-tolerant QC");
    println!("  - Magic state distillation required");
    println!("  - Goal: minimize T-count and T-depth");
    println!("  - Trade-off: T-count vs circuit depth\n");

    // T-count and T-depth
    use simq_compiler::decomposition::clifford_t::CliffordTGate;
    let example_circuit = vec![
        CliffordTGate::H,
        CliffordTGate::T,
        CliffordTGate::T,
        CliffordTGate::S,
        CliffordTGate::T,
        CliffordTGate::H,
    ];

    let t_count = CliffordTDecomposer::count_t_gates(&example_circuit);
    let t_depth = CliffordTDecomposer::count_t_depth(&example_circuit);
    println!("Example Circuit Analysis:");
    println!("  - Total gates: {}", example_circuit.len());
    println!("  - T-count: {}", t_count);
    println!("  - T-depth: {}\n", t_depth);

    println!();
}

fn basis_gate_sets_example() {
    println!("5. Basis Gate Sets");
    println!("===================\n");

    println!("Hardware Vendor Basis Sets:\n");

    // IBM
    println!("IBM Quantum:");
    let ibm = BasisGateSet::IBM;
    let ibm_gate_list = ibm.gates();
    let ibm_gates: Vec<_> = ibm_gate_list.iter().map(|g| g.name()).collect();
    println!("  - Native gates: {:?}", ibm_gates);
    let ibm_entangling = ibm.entangling_gate().map(|g| g.name().to_string());
    println!("  - Entangling gate: {:?}", ibm_entangling);
    println!("  - Description: {}\n", ibm.description());

    // Google
    println!("Google Sycamore:");
    let google = BasisGateSet::Google;
    let google_gate_list = google.gates();
    let google_gates: Vec<_> = google_gate_list.iter().map(|g| g.name()).collect();
    println!("  - Native gates: {:?}", google_gates);
    let google_entangling = google.entangling_gate().map(|g| g.name().to_string());
    println!("  - Entangling gate: {:?}", google_entangling);
    println!("  - Uses √iSWAP as native 2-qubit gate\n");

    // Rigetti
    println!("Rigetti:");
    let rigetti = BasisGateSet::Rigetti;
    let rigetti_gate_list = rigetti.gates();
    let rigetti_gates: Vec<_> = rigetti_gate_list.iter().map(|g| g.name()).collect();
    println!("  - Native gates: {:?}", rigetti_gates);
    let rigetti_entangling = rigetti.entangling_gate().map(|g| g.name().to_string());
    println!("  - Entangling gate: {:?}", rigetti_entangling);
    println!("  - Uses CZ instead of CNOT\n");

    // IonQ
    println!("IonQ (Trapped Ion):");
    let ionq = BasisGateSet::IonQ;
    let ionq_gate_list = ionq.gates();
    let ionq_gates: Vec<_> = ionq_gate_list.iter().map(|g| g.name()).collect();
    println!("  - Native gates: {:?}", ionq_gates);
    let ionq_entangling = ionq.entangling_gate().map(|g| g.name().to_string());
    println!("  - Entangling gate: {:?}", ionq_entangling);
    println!("  - MS: Mølmer-Sørensen gate\n");

    // Clifford+T
    println!("Clifford+T (Fault-Tolerant):");
    let clifford_t = BasisGateSet::CliffordT;
    let clifford_t_gate_list = clifford_t.gates();
    let clifford_t_gates: Vec<_> = clifford_t_gate_list.iter().map(|g| g.name()).collect();
    println!("  - Native gates: {:?}", clifford_t_gates);
    println!("  - Universal for quantum computation");
    println!("  - Discrete gate set: {}\n", clifford_t.is_discrete());

    println!();
}

fn configuration_example() {
    println!("6. Decomposition Configuration");
    println!("===============================\n");

    // Basic configuration
    println!("Basic Configuration:");
    let basic_config = DecompositionConfig {
        basis: BasisGateSet::IBM,
        optimization_level: 1,
        fidelity_threshold: 0.9999,
        ..Default::default()
    };
    println!("  - Basis: IBM Quantum");
    println!("  - Optimization level: {}", basic_config.optimization_level);
    println!("  - Fidelity threshold: {}\n", basic_config.fidelity_threshold);

    // Advanced configuration
    println!("Advanced Configuration:");
    let advanced_config = DecompositionConfig {
        basis: BasisGateSet::CliffordT,
        optimization_level: 3,
        fidelity_threshold: 0.99999,
        max_depth: Some(100),
        max_gates: Some(1000),
        allow_ancillas: true,
        num_ancillas: 5,
        clifford_t_epsilon: 1e-12,
    };
    println!("  - Basis: Clifford+T");
    println!("  - Optimization level: {} (aggressive)", advanced_config.optimization_level);
    println!("  - Max depth: {:?}", advanced_config.max_depth);
    println!("  - Max gates: {:?}", advanced_config.max_gates);
    println!("  - Use ancillas: {}", advanced_config.allow_ancillas);
    println!("  - Num ancillas: {}", advanced_config.num_ancillas);
    println!("  - Clifford+T ε: {}\n", advanced_config.clifford_t_epsilon);

    // Optimization levels
    println!("Optimization Levels:");
    println!("  - Level 0: No optimization (direct decomposition)");
    println!("  - Level 1: Basic (merge adjacent rotations)");
    println!("  - Level 2: Advanced (circuit identities, commutation)");
    println!("  - Level 3: Aggressive (numerical search, Solovay-Kitaev)\n");

    println!();
}
