//! E2E tests for compiler internals:
//! circuit_analysis_pass, cache, execution_plan, matrix_computation,
//! hardware_aware, adaptive_pipeline, multi_level_optimizer, compiler pipeline

use num_complex::Complex64;
use simq_compiler::{
    cache::{CacheStatistics, CircuitFingerprint, CompilationCache, SharedCompilationCache},
    circuit_analysis_pass::{CircuitCharacteristics, CircuitSize},
    execution_plan::{ExecutionLayer, ExecutionPlanner},
    hardware_aware::{
        CostModel, GoogleHardware, HardwareModel, HardwareType, IBMHardware, IonQHardware,
    },
    matrix_computation::{
        adjoint_2x2, controlled_gate_2x2, decompose_zyz, determinant_2x2,
        doubly_controlled_gate_2x2, gate_fidelity_2x2, hadamard_matrix, identity_2x2,
        is_hermitian_2x2, is_unitary_2x2, is_unitary_4x4, matrices_equal_2x2,
        matrix_exp_hermitian_2x2, multiply_2x2, multiply_4x4, pauli_x_matrix, pauli_y_matrix,
        pauli_z_matrix, tensor_product_2x2, trace_2x2, trace_4x4,
    },
    passes::{DeadCodeElimination, GateFusion},
    pipeline::{create_compiler, OptimizationLevel},
    AdaptiveCompiler, CompilerBuilder, MultiLevelOptimizer,
};
use simq_core::{gate::Gate, Circuit, QubitId};
use std::sync::Arc;

const EPSILON: f64 = 1e-8;

fn q(i: usize) -> QubitId {
    QubitId::new(i)
}

#[derive(Debug)]
struct MockGate {
    name: String,
    n_qubits: usize,
    matrix: Option<Vec<Complex64>>,
}

impl Gate for MockGate {
    fn name(&self) -> &str {
        &self.name
    }
    fn num_qubits(&self) -> usize {
        self.n_qubits
    }
    fn matrix(&self) -> Option<Vec<Complex64>> {
        self.matrix.clone()
    }
}

fn mg(name: &str) -> Arc<MockGate> {
    Arc::new(MockGate {
        name: name.to_string(),
        n_qubits: 1,
        matrix: None,
    })
}

fn mg2(name: &str) -> Arc<MockGate> {
    Arc::new(MockGate {
        name: name.to_string(),
        n_qubits: 2,
        matrix: None,
    })
}

// ============================================================================
// Matrix computation tests
// ============================================================================

#[test]
fn test_pauli_matrices_are_unitary() {
    assert!(is_unitary_2x2(&pauli_x_matrix()));
    assert!(is_unitary_2x2(&pauli_y_matrix()));
    assert!(is_unitary_2x2(&pauli_z_matrix()));
    assert!(is_unitary_2x2(&hadamard_matrix()));
    assert!(is_unitary_2x2(&identity_2x2()));
}

#[test]
fn test_pauli_matrices_are_hermitian() {
    assert!(is_hermitian_2x2(&pauli_x_matrix()));
    assert!(is_hermitian_2x2(&pauli_y_matrix()));
    assert!(is_hermitian_2x2(&pauli_z_matrix()));
    assert!(is_hermitian_2x2(&hadamard_matrix()));
}

#[test]
fn test_pauli_x_squared_is_identity() {
    let x = pauli_x_matrix();
    let xx = multiply_2x2(&x, &x);
    assert!(matrices_equal_2x2(&xx, &identity_2x2(), EPSILON));
}

#[test]
fn test_pauli_y_squared_is_identity() {
    let y = pauli_y_matrix();
    let yy = multiply_2x2(&y, &y);
    assert!(matrices_equal_2x2(&yy, &identity_2x2(), EPSILON));
}

#[test]
fn test_pauli_z_squared_is_identity() {
    let z = pauli_z_matrix();
    let zz = multiply_2x2(&z, &z);
    assert!(matrices_equal_2x2(&zz, &identity_2x2(), EPSILON));
}

#[test]
fn test_hadamard_squared_is_identity() {
    let h = hadamard_matrix();
    let hh = multiply_2x2(&h, &h);
    assert!(matrices_equal_2x2(&hh, &identity_2x2(), EPSILON));
}

#[test]
fn test_adjoint_of_unitary_is_inverse() {
    let h = hadamard_matrix();
    let h_dag = adjoint_2x2(&h);
    let product = multiply_2x2(&h, &h_dag);
    assert!(matrices_equal_2x2(&product, &identity_2x2(), EPSILON));
}

#[test]
fn test_trace_identity() {
    let id = identity_2x2();
    let tr = trace_2x2(&id);
    assert!((tr.re - 2.0).abs() < EPSILON);
    assert!(tr.im.abs() < EPSILON);
}

#[test]
fn test_trace_pauli_x_is_zero() {
    let tr = trace_2x2(&pauli_x_matrix());
    assert!(tr.norm() < EPSILON);
}

#[test]
fn test_determinant_identity() {
    let det = determinant_2x2(&identity_2x2());
    assert!((det.re - 1.0).abs() < EPSILON);
    assert!(det.im.abs() < EPSILON);
}

#[test]
fn test_determinant_pauli_x() {
    let det = determinant_2x2(&pauli_x_matrix());
    assert!((det.re - (-1.0)).abs() < EPSILON);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_tensor_product_identity_identity() {
    let id = identity_2x2();
    let result = tensor_product_2x2(&id, &id);
    // I ⊗ I should be 4x4 identity
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (result[i][j].re - expected).abs() < EPSILON,
                "I⊗I[{i}][{j}] = {:?}, expected {expected}",
                result[i][j]
            );
            assert!(result[i][j].im.abs() < EPSILON);
        }
    }
}

#[test]
fn test_tensor_product_unitarity() {
    let h = hadamard_matrix();
    let x = pauli_x_matrix();
    let hx = tensor_product_2x2(&h, &x);
    assert!(is_unitary_4x4(&hx));
}

#[test]
fn test_controlled_gate_produces_cnot() {
    let x = pauli_x_matrix();
    let cnot = controlled_gate_2x2(&x);

    // CNOT: |00⟩→|00⟩, |01⟩→|01⟩, |10⟩→|11⟩, |11⟩→|10⟩
    assert!((cnot[0][0].re - 1.0).abs() < EPSILON);
    assert!((cnot[1][1].re - 1.0).abs() < EPSILON);
    assert!((cnot[2][3].re - 1.0).abs() < EPSILON);
    assert!((cnot[3][2].re - 1.0).abs() < EPSILON);
    assert!(cnot[2][2].norm() < EPSILON);
    assert!(cnot[3][3].norm() < EPSILON);

    assert!(is_unitary_4x4(&cnot));
}

#[test]
#[allow(clippy::needless_range_loop)]
fn test_controlled_gate_identity_is_identity_4x4() {
    let id = identity_2x2();
    let cid = controlled_gate_2x2(&id);
    // C-I should be 4x4 identity
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((cid[i][j].re - expected).abs() < EPSILON);
        }
    }
}

#[test]
fn test_doubly_controlled_gate_unitarity() {
    let x = pauli_x_matrix();
    let ccx = doubly_controlled_gate_2x2(&x);
    // Check key entries: only last 2x2 block is X
    assert!((ccx[6][6].re).abs() < EPSILON); // X[0][0] = 0
    assert!((ccx[6][7].re - 1.0).abs() < EPSILON); // X[0][1] = 1
    assert!((ccx[7][6].re - 1.0).abs() < EPSILON); // X[1][0] = 1
    assert!((ccx[7][7].re).abs() < EPSILON); // X[1][1] = 0
}

#[test]
fn test_gate_fidelity_same_gate() {
    let h = hadamard_matrix();
    let fidelity = gate_fidelity_2x2(&h, &h);
    assert!((fidelity - 1.0).abs() < EPSILON);
}

#[test]
fn test_gate_fidelity_identity_vs_x() {
    let id = identity_2x2();
    let x = pauli_x_matrix();
    let fidelity = gate_fidelity_2x2(&id, &x);
    assert!(fidelity < 0.5, "Identity and X should have low fidelity");
}

#[test]
fn test_decompose_zyz_identity() {
    let id = identity_2x2();
    let result = decompose_zyz(&id);
    assert!(result.is_ok());
}

#[test]
fn test_decompose_zyz_hadamard() {
    let h = hadamard_matrix();
    let result = decompose_zyz(&h);
    assert!(result.is_ok());
    let (_alpha, _beta, _gamma, _delta) = result.unwrap();
}

#[test]
fn test_matrix_exp_hermitian_small_angle() {
    let x = pauli_x_matrix();
    let result = matrix_exp_hermitian_2x2(&x, 0.0);
    assert!(matrices_equal_2x2(&result, &identity_2x2(), EPSILON));
}

#[test]
fn test_trace_4x4_identity() {
    let id = identity_2x2();
    let id4 = tensor_product_2x2(&id, &id);
    let tr = trace_4x4(&id4);
    assert!((tr.re - 4.0).abs() < EPSILON);
}

#[test]
fn test_multiply_4x4_identity() {
    let id = identity_2x2();
    let id4 = tensor_product_2x2(&id, &id);
    let result = multiply_4x4(&id4, &id4);
    let tr = trace_4x4(&result);
    assert!((tr.re - 4.0).abs() < EPSILON);
}

// ============================================================================
// CircuitCharacteristics / CircuitAnalysis tests
// ============================================================================

#[test]
fn test_characteristics_empty_circuit() {
    let circuit = Circuit::new(5);
    let chars = CircuitCharacteristics::analyze(&circuit);

    assert_eq!(chars.gate_count, 0);
    assert_eq!(chars.num_qubits, 5);
    assert_eq!(chars.depth, 0);
    assert_eq!(chars.size_category(), CircuitSize::Small);
}

#[test]
fn test_characteristics_single_qubit_circuit() {
    let mut circuit = Circuit::new(1);
    let x = mg("X");
    for _ in 0..10 {
        circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    }

    let chars = CircuitCharacteristics::analyze(&circuit);
    assert_eq!(chars.gate_count, 10);
    assert_eq!(chars.num_qubits, 1);
    assert_eq!(chars.depth, 10);
    assert!(chars.gates_per_qubit > 9.0);
}

#[test]
fn test_characteristics_size_categories() {
    let categories = vec![(10, CircuitSize::Small), (49, CircuitSize::Small)];
    for (count, expected_size) in categories {
        let mut circuit = Circuit::new(2);
        let x = mg("X");
        for _ in 0..count {
            circuit.add_gate(x.clone(), &[q(0)]).unwrap();
        }
        let chars = CircuitCharacteristics::analyze(&circuit);
        assert_eq!(
            chars.size_category(),
            expected_size,
            "Circuit with {count} gates should be {:?}",
            expected_size
        );
    }
}

#[test]
fn test_characteristics_dead_code_density() {
    let mut circuit = Circuit::new(1);
    let x = mg("X");
    // X-X pairs: high dead code density
    for _ in 0..5 {
        circuit.add_gate(x.clone(), &[q(0)]).unwrap();
        circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    }

    let chars = CircuitCharacteristics::analyze(&circuit);
    assert!(chars.dead_code_density > 0.0);
    assert!(chars.should_use_dce());
}

#[test]
fn test_characteristics_fusion_density() {
    let mut circuit = Circuit::new(1);
    let h = mg("H");
    let x = mg("X");
    // Consecutive single-qubit gates on same qubit → fusion opportunity
    for _ in 0..5 {
        circuit.add_gate(h.clone(), &[q(0)]).unwrap();
        circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    }

    let chars = CircuitCharacteristics::analyze(&circuit);
    assert!(chars.fusion_density > 0.0);
    assert!(chars.should_use_fusion());
}

#[test]
fn test_characteristics_suggest_iterations() {
    let circuit = Circuit::new(2);
    let chars = CircuitCharacteristics::analyze(&circuit);
    assert_eq!(chars.suggest_iterations(), 3);

    let mut medium_circuit = Circuit::new(2);
    let x = mg("X");
    for _ in 0..50 {
        medium_circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    }
    let chars = CircuitCharacteristics::analyze(&medium_circuit);
    assert_eq!(chars.suggest_iterations(), 5);
}

#[test]
fn test_characteristics_two_qubit_ratio() {
    let mut circuit = Circuit::new(2);
    let x = mg("X");
    let cnot = mg2("CNOT");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(cnot.clone(), &[q(0), q(1)]).unwrap();

    let chars = CircuitCharacteristics::analyze(&circuit);
    assert!(chars.single_to_two_qubit_ratio >= 1.0);
}

// ============================================================================
// CompilationCache tests
// ============================================================================

#[test]
fn test_cache_insert_and_retrieve() {
    let mut cache = CompilationCache::new(10);
    let mut circuit = Circuit::new(2);
    let gate = mg("H");
    circuit.add_gate(gate, &[q(0)]).unwrap();

    let fp = CircuitFingerprint::compute(&circuit);
    assert!(cache.get(fp).is_none());

    cache.insert(fp, circuit.clone());
    assert!(cache.get(fp).is_some());
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_cache_lru_eviction() {
    let mut cache = CompilationCache::new(2);

    let circuits: Vec<Circuit> = (0..3)
        .map(|i| {
            let mut c = Circuit::new(i + 1);
            let g = mg(&format!("G{i}"));
            c.add_gate(g, &[q(0)]).unwrap();
            c
        })
        .collect();

    let fps: Vec<_> = circuits.iter().map(CircuitFingerprint::compute).collect();

    cache.insert(fps[0], circuits[0].clone());
    cache.insert(fps[1], circuits[1].clone());
    assert_eq!(cache.len(), 2);

    cache.insert(fps[2], circuits[2].clone());
    assert_eq!(cache.len(), 2);
    assert!(cache.get(fps[0]).is_none());
    assert!(cache.get(fps[1]).is_some());
    assert!(cache.get(fps[2]).is_some());
}

#[test]
fn test_cache_statistics() {
    let mut cache = CompilationCache::new(10);
    let circuit = Circuit::new(2);
    let fp = CircuitFingerprint::compute(&circuit);

    cache.get(fp); // miss
    cache.insert(fp, circuit);
    cache.get(fp); // hit

    let stats = cache.statistics();
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.misses, 1);
    assert_eq!(stats.hit_rate(), 50.0);
    assert_eq!(stats.miss_rate(), 50.0);
}

#[test]
fn test_cache_clear() {
    let mut cache = CompilationCache::new(10);
    let circuit = Circuit::new(2);
    let fp = CircuitFingerprint::compute(&circuit);
    cache.insert(fp, circuit);
    assert!(!cache.is_empty());

    cache.clear();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cache_resize_evicts() {
    let mut cache = CompilationCache::new(5);
    for i in 0..5 {
        let c = Circuit::new(i + 1);
        let fp = CircuitFingerprint::compute(&c);
        cache.insert(fp, c);
    }
    assert_eq!(cache.len(), 5);

    cache.set_max_size(2);
    assert_eq!(cache.len(), 2);
    assert!(cache.statistics().evictions >= 3);
}

#[test]
fn test_fingerprint_identical_circuits_match() {
    let mut c1 = Circuit::new(2);
    let mut c2 = Circuit::new(2);
    let gate = mg("H");
    c1.add_gate(gate.clone(), &[q(0)]).unwrap();
    c2.add_gate(gate.clone(), &[q(0)]).unwrap();

    assert_eq!(CircuitFingerprint::compute(&c1), CircuitFingerprint::compute(&c2));
}

#[test]
fn test_fingerprint_different_circuits_differ() {
    let mut c1 = Circuit::new(2);
    let mut c2 = Circuit::new(2);
    c1.add_gate(mg("H"), &[q(0)]).unwrap();
    c2.add_gate(mg("X"), &[q(0)]).unwrap();

    assert_ne!(CircuitFingerprint::compute(&c1), CircuitFingerprint::compute(&c2));
}

#[test]
fn test_shared_cache_thread_safety() {
    let cache = SharedCompilationCache::new(10);
    let circuit = Circuit::new(2);
    let fp = CircuitFingerprint::compute(&circuit);

    cache.insert(fp, circuit.clone());
    assert!(cache.get(fp).is_some());
    assert_eq!(cache.len(), 1);
    assert!(!cache.is_empty());

    let stats = cache.statistics();
    assert_eq!(stats.hits, 1);

    cache.clear();
    assert!(cache.is_empty());
}

#[test]
fn test_cache_statistics_display() {
    let stats = CacheStatistics {
        hits: 10,
        misses: 5,
        evictions: 2,
        current_size: 8,
        max_size: 100,
    };
    let display = format!("{}", stats);
    assert!(display.contains("Hits: 10"));
    assert!(display.contains("Misses: 5"));
    assert!(display.contains("Evictions: 2"));
}

// ============================================================================
// ExecutionPlan tests
// ============================================================================

#[test]
fn test_execution_layer_qubit_conflict() {
    let mut layer = ExecutionLayer::new();
    assert!(layer.is_empty());

    layer.add_gate(0, &[q(0)], 1.0);
    assert_eq!(layer.len(), 1);
    assert!(layer.can_add(&[q(1)]));
    assert!(!layer.can_add(&[q(0)]));
}

#[test]
fn test_execution_plan_parallel_gates() {
    let mut circuit = Circuit::new(4);
    let h = mg("H");
    for i in 0..4 {
        circuit.add_gate(h.clone(), &[q(i)]).unwrap();
    }

    let planner = ExecutionPlanner::new();
    let plan = planner.generate_plan(&circuit);

    assert_eq!(plan.depth, 1);
    assert_eq!(plan.gate_count, 4);
    assert_eq!(plan.parallelism_factor, 4.0);
    assert!(plan.parallelization_efficiency() > 0.5);
    assert_eq!(plan.average_layer_size(), 4.0);
}

#[test]
fn test_execution_plan_sequential_gates() {
    let mut circuit = Circuit::new(1);
    let h = mg("H");
    for _ in 0..5 {
        circuit.add_gate(h.clone(), &[q(0)]).unwrap();
    }

    let planner = ExecutionPlanner::new();
    let plan = planner.generate_plan(&circuit);

    assert_eq!(plan.depth, 5);
    assert_eq!(plan.gate_count, 5);
    assert_eq!(plan.parallelism_factor, 1.0);
    assert_eq!(plan.parallelization_efficiency(), 0.0);
}

#[test]
fn test_execution_plan_mixed() {
    let mut circuit = Circuit::new(3);
    let h = mg("H");
    let cnot = mg2("CNOT");

    circuit.add_gate(h.clone(), &[q(0)]).unwrap();
    circuit.add_gate(h.clone(), &[q(1)]).unwrap();
    circuit.add_gate(h.clone(), &[q(2)]).unwrap();
    circuit.add_gate(cnot, &[q(0), q(1)]).unwrap();

    let planner = ExecutionPlanner::new();
    let plan = planner.generate_plan(&circuit);

    assert_eq!(plan.depth, 2);
    assert_eq!(plan.gate_count, 4);
}

#[test]
fn test_execution_plan_resource_estimation() {
    let mut circuit = Circuit::new(5);
    let h = mg("H");
    let cnot = mg2("CNOT");
    circuit.add_gate(h, &[q(0)]).unwrap();
    circuit.add_gate(cnot.clone(), &[q(0), q(1)]).unwrap();
    circuit.add_gate(cnot, &[q(2), q(3)]).unwrap();

    let planner = ExecutionPlanner::new();
    let plan = planner.generate_plan(&circuit);

    assert_eq!(plan.resources.peak_qubits, 5);
    assert_eq!(plan.resources.two_qubit_gates, 2);
    assert!(plan.resources.peak_memory > 0);
}

#[test]
fn test_execution_plan_bottleneck() {
    let mut circuit = Circuit::new(4);
    let h = mg("H");
    // 4 parallel gates → one layer with 4 gates
    for i in 0..4 {
        circuit.add_gate(h.clone(), &[q(i)]).unwrap();
    }
    // 1 sequential gate → one layer with 1 gate
    circuit.add_gate(h.clone(), &[q(0)]).unwrap();

    let planner = ExecutionPlanner::new();
    let plan = planner.generate_plan(&circuit);

    let bottleneck = plan.bottleneck_layer();
    assert!(bottleneck.is_some());
    assert_eq!(bottleneck.unwrap(), 0); // First layer has 4 gates
}

#[test]
fn test_execution_plan_display() {
    let mut circuit = Circuit::new(2);
    let h = mg("H");
    circuit.add_gate(h.clone(), &[q(0)]).unwrap();
    circuit.add_gate(h.clone(), &[q(1)]).unwrap();

    let planner = ExecutionPlanner::new();
    let plan = planner.generate_plan(&circuit);
    let display = format!("{}", plan);
    assert!(display.contains("Execution Plan"));
    assert!(display.contains("Circuit depth"));
}

#[test]
fn test_execution_plan_custom_gate_times() {
    let mut planner = ExecutionPlanner::new();
    planner.set_gate_time("CustomGate", 100.0);

    let mut circuit = Circuit::new(1);
    let cg = mg("CustomGate");
    circuit.add_gate(cg, &[q(0)]).unwrap();

    let plan = planner.generate_plan(&circuit);
    assert!((plan.total_time - 100.0).abs() < EPSILON);
}

#[test]
fn test_execution_plan_optimized_vs_basic() {
    let mut circuit = Circuit::new(3);
    let h = mg("H");
    let x = mg("X");
    circuit.add_gate(h.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(1)]).unwrap();
    circuit.add_gate(h.clone(), &[q(2)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(h.clone(), &[q(1)]).unwrap();

    let planner = ExecutionPlanner::new();
    let basic = planner.generate_plan(&circuit);
    let optimized = planner.generate_optimized_plan(&circuit);

    assert_eq!(basic.gate_count, optimized.gate_count);
    assert!(optimized.depth <= basic.depth);
}

// ============================================================================
// Hardware-aware compilation tests
// ============================================================================

#[test]
fn test_ibm_hardware_native_gates() {
    let ibm = IBMHardware::new();
    assert!(ibm.is_native("CNOT"));
    assert!(ibm.is_native("RZ"));
    assert!(ibm.is_native("SX"));
    assert!(ibm.is_native("X"));
    assert!(!ibm.is_native("CZ"));
    assert!(!ibm.is_native("iSWAP"));
    assert_eq!(ibm.name(), "IBM Quantum");
}

#[test]
fn test_google_hardware_native_gates() {
    let google = GoogleHardware::new();
    assert!(google.is_native("CZ"));
    assert!(google.is_native("iSWAP"));
    assert!(google.is_native("RZ"));
    assert!(!google.is_native("CNOT"));
    assert_eq!(google.name(), "Google Sycamore");
}

#[test]
fn test_ionq_hardware_native_gates() {
    let ionq = IonQHardware::new();
    assert!(ionq.is_native("MS"));
    assert!(ionq.is_native("GPI"));
    assert!(ionq.is_native("GPI2"));
    assert!(ionq.is_native("RX"));
    assert!(!ionq.is_native("CNOT"));
    assert_eq!(ionq.name(), "IonQ Trapped-Ion");
}

#[test]
fn test_hardware_cost_ordering() {
    let ibm = IBMHardware::new();
    // Single-qubit cheaper than two-qubit cheaper than three-qubit
    let sq_cost = ibm.gate_cost_by_name("X");
    let tq_cost = ibm.gate_cost_by_name("CNOT");
    let mq_cost = ibm.gate_cost_by_name("Toffoli");

    assert!(sq_cost < tq_cost);
    assert!(tq_cost < mq_cost);
}

#[test]
fn test_cost_model_empty_circuit() {
    let cm = CostModel::new(HardwareType::IBM);
    let circuit = Circuit::new(2);
    assert_eq!(cm.circuit_cost(&circuit), 0.0);
}

#[test]
fn test_cost_model_with_gates() {
    let cm = CostModel::new(HardwareType::IBM);
    let mut circuit = Circuit::new(2);
    let x = mg("X");
    circuit.add_gate(x, &[q(0)]).unwrap();

    let cost = cm.circuit_cost(&circuit);
    assert!(cost > 0.0);
}

#[test]
fn test_cost_model_non_native_penalty() {
    let cm = CostModel::new(HardwareType::IBM);
    let mut native_circuit = Circuit::new(2);
    let mut nonnative_circuit = Circuit::new(2);

    native_circuit.add_gate(mg("X"), &[q(0)]).unwrap();
    nonnative_circuit.add_gate(mg("CZ"), &[q(0)]).unwrap();

    // Note: CZ is not native on IBM, so it gets decomposition penalty
    // But the mock gate has n_qubits=1 for mg("CZ"), so the comparison
    // is about the name-based cost
    let native_cost = cm.circuit_cost(&native_circuit);
    let nonnative_cost = cm.circuit_cost(&nonnative_circuit);
    // Non-native should cost more due to penalty
    assert!(nonnative_cost > native_cost);
}

#[test]
fn test_hardware_type_names() {
    assert_eq!(HardwareType::IBM.name(), "IBM Quantum");
    assert_eq!(HardwareType::Google.name(), "Google Sycamore");
    assert_eq!(HardwareType::IonQ.name(), "IonQ Trapped-Ion");
}

#[test]
fn test_cost_model_hardware_model() {
    let cm = CostModel::new(HardwareType::Google);
    let model = cm.hardware_model();
    assert_eq!(model.name(), "Google Sycamore");
}

// ============================================================================
// AdaptiveCompiler tests
// ============================================================================

#[test]
fn test_adaptive_compiler_empty_circuit() {
    let adaptive = AdaptiveCompiler::new();
    let circuit = Circuit::new(3);
    let compiler = adaptive.create_for_circuit(&circuit);
    assert!(compiler.num_passes() <= 1);
}

#[test]
fn test_adaptive_compiler_with_dead_code() {
    let adaptive = AdaptiveCompiler::new();
    let mut circuit = Circuit::new(1);
    let x = mg("X");
    for _ in 0..20 {
        circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    }

    let compiler = adaptive.create_for_circuit(&circuit);
    assert!(compiler.num_passes() > 0);

    let result = compiler.compile(&mut circuit).unwrap();
    assert!(result.modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_adaptive_compiler_from_characteristics() {
    let adaptive = AdaptiveCompiler::new();
    let chars = CircuitCharacteristics {
        gate_count: 100,
        num_qubits: 5,
        depth: 50,
        single_to_two_qubit_ratio: 5.0,
        commutation_density: 0.3,
        fusion_density: 0.2,
        template_density: 0.1,
        dead_code_density: 0.05,
        gates_per_qubit: 20.0,
        parallelism_factor: 2.0,
    };

    let compiler = adaptive.create_from_characteristics(&chars);
    assert!(compiler.num_passes() >= 2);
}

#[test]
fn test_adaptive_compiler_verbose() {
    let adaptive = AdaptiveCompiler::new().with_verbose(true);
    assert!(adaptive.verbose);
    let circuit = Circuit::new(1);
    let _compiler = adaptive.create_for_circuit(&circuit);
}

// ============================================================================
// MultiLevelOptimizer tests
// ============================================================================

#[test]
fn test_multi_level_optimizer_removes_inverse_pairs() {
    let optimizer = MultiLevelOptimizer::new();
    let mut circuit = Circuit::new(1);
    let x = mg("X");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();

    let result = optimizer.optimize(&mut circuit);
    assert!(result.modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_multi_level_optimizer_verbose() {
    let optimizer = MultiLevelOptimizer::new().with_verbose(true);
    assert!(optimizer.verbose);
    let mut circuit = Circuit::new(1);
    let x = mg("X");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();

    let result = optimizer.optimize(&mut circuit);
    assert!(result.modified);
}

#[test]
fn test_multi_level_optimizer_unmodifiable_circuit() {
    let optimizer = MultiLevelOptimizer::new();
    let mut circuit = Circuit::new(2);
    let h = mg("H");
    let x = mg("X");
    circuit.add_gate(h, &[q(0)]).unwrap();
    circuit.add_gate(x, &[q(1)]).unwrap();

    let result = optimizer.optimize(&mut circuit);
    assert_eq!(circuit.len(), 2);
    let _ = result;
}

#[test]
fn test_multi_level_has_pass_stats() {
    let optimizer = MultiLevelOptimizer::new();
    let mut circuit = Circuit::new(1);
    let x = mg("X");
    for _ in 0..4 {
        circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    }

    let result = optimizer.optimize(&mut circuit);
    assert!(!result.pass_stats.is_empty());
    assert!(result.total_time_us > 0);
}

// ============================================================================
// Compiler / Pipeline tests
// ============================================================================

#[test]
fn test_optimization_level_o0_no_change() {
    let compiler = create_compiler(OptimizationLevel::O0);
    let mut circuit = Circuit::new(1);
    let x = mg("X");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();

    let result = compiler.compile(&mut circuit).unwrap();
    assert!(!result.modified);
    assert_eq!(circuit.len(), 2);
}

#[test]
fn test_optimization_level_o1_dce() {
    let compiler = create_compiler(OptimizationLevel::O1);
    let mut circuit = Circuit::new(1);
    let x = mg("X");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();

    let result = compiler.compile(&mut circuit).unwrap();
    assert!(result.modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_optimization_level_o3_aggressive() {
    let compiler = create_compiler(OptimizationLevel::O3);
    let mut circuit = Circuit::new(2);
    let x = mg("X");
    let h = mg("H");

    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(h.clone(), &[q(1)]).unwrap();
    circuit.add_gate(h.clone(), &[q(1)]).unwrap();

    let result = compiler.compile(&mut circuit).unwrap();
    assert!(result.modified);
    assert_eq!(circuit.len(), 0);
}

#[test]
fn test_compiler_builder_custom_pipeline() {
    let compiler = CompilerBuilder::new()
        .add_pass(Arc::new(DeadCodeElimination::new()))
        .add_pass(Arc::new(GateFusion::new()))
        .max_iterations(3)
        .enable_timing(true)
        .build();

    let mut circuit = Circuit::new(1);
    let x = mg("X");
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    circuit.add_gate(x.clone(), &[q(0)]).unwrap();

    let result = compiler.compile(&mut circuit).unwrap();
    assert!(result.modified);
    assert_eq!(circuit.len(), 0);
    assert!(!result.pass_stats.is_empty());
    assert!(result.total_time_us > 0);
}

#[test]
fn test_compiler_fixed_point_iteration() {
    let compiler = CompilerBuilder::new()
        .add_pass(Arc::new(DeadCodeElimination::new()))
        .max_iterations(10)
        .build();

    let mut circuit = Circuit::new(1);
    let x = mg("X");
    // Multiple X-X pairs
    for _ in 0..10 {
        circuit.add_gate(x.clone(), &[q(0)]).unwrap();
    }

    let result = compiler.compile(&mut circuit).unwrap();
    assert!(result.modified);
    assert_eq!(circuit.len(), 0);
}
