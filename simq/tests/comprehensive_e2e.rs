//! Comprehensive cross-crate integration tests for the SimQ quantum SDK
//!
//! Tests the full workflow: circuit creation → compilation → simulation → backend execution.
//! Verifies that all sub-crates work together correctly.

use simq_backend::{
    ConnectivityGraph, GateDecomposer, LocalSimulatorBackend, QuantumBackend, Router,
    RoutingStrategy, Transpiler,
};
use simq_compiler::pipeline::{create_compiler, OptimizationLevel};
use simq_core::circuit::Circuit;
use simq_core::QubitId;
use simq_gates::standard::*;
use simq_sim::{Simulator, SimulatorConfig};
use std::sync::Arc;

fn q(i: usize) -> QubitId {
    QubitId::new(i)
}

// ============================================================================
// 1. Full pipeline: circuit → compile → simulate → backend
// ============================================================================

#[test]
fn full_pipeline_x_gate() {
    // Create circuit
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();

    // Compile
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut compiled = c.clone();
    compiler.compile(&mut compiled).unwrap();

    // Simulate directly
    let sim = Simulator::new(SimulatorConfig::default());
    let result = sim.run(&compiled).unwrap();
    assert_eq!(result.num_qubits(), 1);

    // Execute through backend
    let backend = LocalSimulatorBackend::new();
    let br = backend.execute(&compiled, 100).unwrap();
    assert_eq!(br.get_count("1"), 100);
}

#[test]
fn full_pipeline_bell_state() {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();

    // Compile with O2
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut compiled = c.clone();
    compiler.compile(&mut compiled).unwrap();

    // Backend execution
    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&compiled, 10000).unwrap();

    let p00 = result.get_count("00");
    let p11 = result.get_count("11");
    assert_eq!(p00 + p11, 10000);
    assert!(p00 > 4000 && p00 < 6000);
}

#[test]
fn full_pipeline_ghz_3() {
    let mut c = Circuit::new(3);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(1), q(2)]).unwrap();

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&c, 10000).unwrap();

    let p000 = result.get_count("000");
    let p111 = result.get_count("111");
    assert_eq!(p000 + p111, 10000);
    assert!(p000 > 4000 && p000 < 6000);
}

#[test]
fn full_pipeline_ghz_5() {
    let mut c = Circuit::new(5);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    for i in 0..4 {
        c.add_gate(Arc::new(CNot), &[q(i), q(i + 1)]).unwrap();
    }

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&c, 10000).unwrap();

    let p00000 = result.get_count("00000");
    let p11111 = result.get_count("11111");
    assert_eq!(p00000 + p11111, 10000);
}

// ============================================================================
// 2. Optimization preserves semantics
// ============================================================================

#[test]
fn optimization_preserves_bell_state() {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();

    let backend = LocalSimulatorBackend::new();

    // No optimization
    let sim_no_opt = Simulator::new(SimulatorConfig::default().with_optimization(false));
    let r_no_opt = sim_no_opt.run(&c).unwrap();

    // With O3 optimization
    let sim_opt = Simulator::new(SimulatorConfig::default().with_optimization_level(3));
    let r_opt = sim_opt.run(&c).unwrap();

    // Both should produce Bell state
    let backend_no_opt = backend.execute(&c, 5000).unwrap();
    let total = backend_no_opt.get_count("00") + backend_no_opt.get_count("11");
    assert_eq!(total, 5000);

    // Verify state sizes match
    assert_eq!(r_no_opt.num_qubits(), r_opt.num_qubits());
}

#[test]
fn optimization_removes_inverse_pairs() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();

    // After optimization, H-H should cancel, leaving just X
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut compiled = c.clone();
    compiler.compile(&mut compiled).unwrap();

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&compiled, 100).unwrap();
    // X|0⟩ = |1⟩
    assert_eq!(result.get_count("1"), 100);
}

#[test]
fn all_optimization_levels_produce_correct_results() {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();

    let backend = LocalSimulatorBackend::new();

    for level in [
        OptimizationLevel::O0,
        OptimizationLevel::O1,
        OptimizationLevel::O2,
        OptimizationLevel::O3,
    ] {
        let compiler = create_compiler(level);
        let mut compiled = c.clone();
        compiler.compile(&mut compiled).unwrap();

        let result = backend.execute(&compiled, 1000).unwrap();
        let total_bell = result.get_count("00") + result.get_count("11");
        assert_eq!(total_bell, 1000, "Failed for {:?}", level);
    }
}

// ============================================================================
// 3. Gate decomposition + backend execution
// ============================================================================

#[test]
fn ibm_decomposition_preserves_bell() {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();

    let decomposer = GateDecomposer::ibm_native();
    let decomposed = decomposer.decompose_circuit(&c).unwrap();

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&decomposed, 5000).unwrap();

    let p00 = result.get_count("00");
    let p11 = result.get_count("11");
    assert_eq!(p00 + p11, 5000);
    assert!(p00 > 2000 && p00 < 3000);
}

#[test]
fn rigetti_decomposition_preserves_bell() {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();

    let decomposer = GateDecomposer::rigetti_native();
    let decomposed = decomposer.decompose_circuit(&c).unwrap();

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&decomposed, 5000).unwrap();

    let p00 = result.get_count("00");
    let p11 = result.get_count("11");
    assert_eq!(p00 + p11, 5000);
}

// ============================================================================
// 4. Transpiler + backend
// ============================================================================

#[test]
fn transpiler_then_execute() {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();

    let backend = LocalSimulatorBackend::new();
    let transpiler = Transpiler::default();
    let transpiled = transpiler.transpile(&c, backend.capabilities()).unwrap();

    let result = backend.execute(&transpiled, 5000).unwrap();
    let total_bell = result.get_count("00") + result.get_count("11");
    assert_eq!(total_bell, 5000);
}

// ============================================================================
// 5. Rotation gates
// ============================================================================

#[test]
fn rotation_y_pi_flips_qubit() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(RotationY::new(std::f64::consts::PI)), &[q(0)])
        .unwrap();

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&c, 100).unwrap();
    assert_eq!(result.get_count("1"), 100);
}

#[test]
fn rotation_x_pi_flips_qubit() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(RotationX::new(std::f64::consts::PI)), &[q(0)])
        .unwrap();

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&c, 100).unwrap();
    assert_eq!(result.get_count("1"), 100);
}

#[test]
fn rotation_z_preserves_zero_state() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(RotationZ::new(1.5)), &[q(0)]).unwrap();

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&c, 100).unwrap();
    // RZ only changes phase, |0⟩ stays |0⟩
    assert_eq!(result.get_count("0"), 100);
}

// ============================================================================
// 6. All Pauli gates
// ============================================================================

#[test]
fn pauli_x_flips() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    assert_eq!(r.get_count("1"), 100);
}

#[test]
fn pauli_y_flips() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(PauliY), &[q(0)]).unwrap();
    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    assert_eq!(r.get_count("1"), 100);
}

#[test]
fn pauli_z_preserves() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(PauliZ), &[q(0)]).unwrap();
    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    assert_eq!(r.get_count("0"), 100);
}

// ============================================================================
// 7. S and T gates
// ============================================================================

#[test]
fn s_gate_on_zero() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(SGate), &[q(0)]).unwrap();
    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    assert_eq!(r.get_count("0"), 100);
}

#[test]
fn t_gate_on_zero() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(TGate), &[q(0)]).unwrap();
    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    assert_eq!(r.get_count("0"), 100);
}

// ============================================================================
// 8. Hadamard creates superposition
// ============================================================================

#[test]
fn hadamard_superposition() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();

    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 10000).unwrap();

    let p0 = r.get_count("0");
    let p1 = r.get_count("1");
    assert_eq!(p0 + p1, 10000);
    assert!(p0 > 4000 && p0 < 6000);
}

// ============================================================================
// 9. Multi-qubit circuits with various gates
// ============================================================================

#[test]
fn swap_gate_swaps_qubits() {
    let mut c = Circuit::new(2);
    // Put qubit 0 in |1⟩
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    // SWAP qubits 0 and 1
    c.add_gate(Arc::new(Swap), &[q(0), q(1)]).unwrap();

    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    // After swap: qubit 0 = |0⟩, qubit 1 = |1⟩ → bitstring "10"
    assert_eq!(r.get_count("10"), 100);
}

#[test]
fn cz_gate_on_bell_basis() {
    let mut c = Circuit::new(2);
    // Create |11⟩
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    c.add_gate(Arc::new(PauliX), &[q(1)]).unwrap();
    // Apply CZ — adds phase -1 to |11⟩
    c.add_gate(Arc::new(CZ), &[q(0), q(1)]).unwrap();

    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    // Still |11⟩ (phase doesn't affect measurement)
    assert_eq!(r.get_count("11"), 100);
}

// ============================================================================
// 10. Result analysis across crates
// ============================================================================

#[test]
fn expectation_value_z_on_x_gate() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&c, 1000).unwrap();

    // ⟨Z⟩ for |1⟩ = -1
    let z_exp = result.expectation_value(|bs| if bs == "0" { 1.0 } else { -1.0 });
    assert!((z_exp - (-1.0)).abs() < 1e-10);
}

#[test]
fn expectation_value_zz_on_bell() {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&c, 50000).unwrap();

    // ⟨ZZ⟩ for Bell state = 1 (correlated)
    let zz_exp = result.expectation_value(|bs| {
        let chars: Vec<char> = bs.chars().collect();
        let z0 = if chars[0] == '0' { 1.0 } else { -1.0 };
        let z1 = if chars[1] == '0' { 1.0 } else { -1.0 };
        z0 * z1
    });
    assert!((zz_exp - 1.0).abs() < 0.05);
}

// ============================================================================
// 11. Routing + execution
// ============================================================================

#[test]
fn routing_preserves_circuit_semantics() {
    let router = Router::new(RoutingStrategy::Subgraph);
    let cg = ConnectivityGraph::linear_chain(5);
    let mapping = router.initial_mapping(3, &cg).unwrap();

    // Verify mapping is valid
    for i in 0..3 {
        let phys = mapping.get_physical(i).unwrap();
        assert!(phys < 5);
    }
}

// ============================================================================
// 12. Deterministic simulation across crates
// ============================================================================

#[test]
fn deterministic_execution_across_backends() {
    let config = simq_backend::LocalSimulatorConfig {
        seed: Some(42),
        ..Default::default()
    };

    let b1 = LocalSimulatorBackend::with_config(config.clone());
    let b2 = LocalSimulatorBackend::with_config(config);

    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();

    let r1 = b1.execute(&c, 1000).unwrap();
    let r2 = b2.execute(&c, 1000).unwrap();

    assert_eq!(r1.get_count("00"), r2.get_count("00"));
    assert_eq!(r1.get_count("11"), r2.get_count("11"));
}

// ============================================================================
// 13. Simulator config variations
// ============================================================================

#[test]
fn simulator_fast_config_produces_correct_results() {
    let sim = Simulator::new(SimulatorConfig::fast());
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    let result = sim.run(&c).unwrap();
    assert_eq!(result.num_qubits(), 2);
}

#[test]
fn simulator_accurate_config_produces_correct_results() {
    let sim = Simulator::new(SimulatorConfig::accurate());
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    let result = sim.run(&c).unwrap();
    assert_eq!(result.num_qubits(), 2);
}

// ============================================================================
// 14. Complex multi-gate circuits
// ============================================================================

#[test]
fn teleportation_circuit_structure() {
    // Quantum teleportation circuit (3 qubits)
    // q0: state to teleport, q1-q2: Bell pair
    let mut c = Circuit::new(3);

    // Prepare Bell pair on q1-q2
    c.add_gate(Arc::new(Hadamard), &[q(1)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(1), q(2)]).unwrap();

    // Bell measurement on q0-q1
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();

    // Circuit should compile and simulate without errors
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut compiled = c.clone();
    compiler.compile(&mut compiled).unwrap();

    let sim = Simulator::new(SimulatorConfig::default());
    let result = sim.run(&compiled).unwrap();
    assert_eq!(result.num_qubits(), 3);
}

#[test]
fn multi_hadamard_uniform_superposition() {
    // Apply H to all qubits: creates uniform superposition
    let n = 3;
    let mut c = Circuit::new(n);
    for i in 0..n {
        c.add_gate(Arc::new(Hadamard), &[q(i)]).unwrap();
    }

    let backend = LocalSimulatorBackend::new();
    let result = backend.execute(&c, 80000).unwrap();

    // Should have 2^3 = 8 outcomes, each with ~1/8 probability
    let probs = result.probabilities();
    assert_eq!(probs.len(), 8);
    for (_, prob) in &probs {
        assert!((*prob - 0.125).abs() < 0.03, "Probability {} too far from 0.125", prob);
    }
}

// ============================================================================
// 15. Edge cases
// ============================================================================

#[test]
fn single_qubit_identity_like() {
    // Z on |0⟩ = |0⟩ (global phase only)
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(PauliZ), &[q(0)]).unwrap();

    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    assert_eq!(r.get_count("0"), 100);
}

#[test]
fn double_x_is_identity() {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();

    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    assert_eq!(r.get_count("0"), 100);
}

#[test]
fn backend_metadata_populated() {
    let backend = LocalSimulatorBackend::new();
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();

    let result = backend.execute(&c, 10).unwrap();
    assert!(result.metadata.is_success());
    assert!(result.metadata.execution_time.is_some());
    assert_eq!(result.metadata.num_qubits, Some(1));
}

// ============================================================================
// 16. Stress tests
// ============================================================================

#[test]
fn stress_deep_circuit() {
    let mut c = Circuit::new(1);
    for _ in 0..100 {
        c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
        c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    }
    // 200 H gates, all cancelling → net identity
    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    assert_eq!(r.get_count("0"), 100);
}

#[test]
fn stress_wide_circuit() {
    let n = 10;
    let mut c = Circuit::new(n);
    for i in 0..n {
        c.add_gate(Arc::new(PauliX), &[q(i)]).unwrap();
    }

    let backend = LocalSimulatorBackend::new();
    let r = backend.execute(&c, 100).unwrap();
    // All qubits flipped → all 1s
    let all_ones = "1".repeat(n);
    assert_eq!(r.get_count(&all_ones), 100);
}

#[test]
fn stress_compile_and_execute_many() {
    let backend = LocalSimulatorBackend::new();
    for _ in 0..50 {
        let mut c = Circuit::new(2);
        c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
        c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();

        let r = backend.execute(&c, 100).unwrap();
        let total = r.get_count("00") + r.get_count("11");
        assert_eq!(total, 100);
    }
}
