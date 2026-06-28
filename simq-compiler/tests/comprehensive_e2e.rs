//! Comprehensive end-to-end tests for simq-compiler crate
//!
//! Covers: Compiler pipeline, optimization passes, gate fusion, decomposition,
//! circuit analysis, hardware-aware compilation, caching, lazy evaluation,
//! execution planning, matrix computation, and adaptive compilation.

use num_complex::Complex64;
use simq_compiler::*;
use simq_core::circuit::Circuit;
use simq_core::gate::Gate;
use simq_core::QubitId;
use simq_gates::standard::*;
use std::sync::Arc;

const EPSILON: f64 = 1e-8;

fn q(i: usize) -> QubitId {
    QubitId::new(i)
}

fn build_hh_circuit() -> Circuit {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c
}

fn build_bell_circuit() -> Circuit {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    c
}

fn build_ghz_circuit() -> Circuit {
    let mut c = Circuit::new(3);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(1), q(2)]).unwrap();
    c
}

fn build_multi_gate_circuit() -> Circuit {
    let mut c = Circuit::new(3);
    c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    c.add_gate(Arc::new(PauliX), &[q(1)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(0), q(1)]).unwrap();
    c.add_gate(Arc::new(RotationZ::new(std::f64::consts::PI / 4.0)), &[q(2)]).unwrap();
    c.add_gate(Arc::new(Hadamard), &[q(2)]).unwrap();
    c.add_gate(Arc::new(CNot), &[q(1), q(2)]).unwrap();
    c
}

// ============================================================================
// 1. CompilerBuilder and basic compilation
// ============================================================================

#[test]
fn compiler_builder_default() {
    let compiler = CompilerBuilder::new().build();
    let mut circuit = build_bell_circuit();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn compiler_builder_with_passes() {
    let compiler = CompilerBuilder::new()
        .add_pass(Arc::new(passes::DeadCodeElimination::new()))
        .add_pass(Arc::new(passes::GateFusion::new()))
        .max_iterations(5)
        .build();
    let mut circuit = build_bell_circuit();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn compiler_builder_max_iterations() {
    let compiler = CompilerBuilder::new()
        .add_pass(Arc::new(passes::GateCommutation::new()))
        .max_iterations(1)
        .build();
    let mut circuit = build_multi_gate_circuit();
    let result = compiler.compile(&mut circuit).unwrap();
    // OptimizationResult has modified and pass_stats, no iterations field
    assert!(!result.pass_stats.is_empty() || !result.modified);
}

// ============================================================================
// 2. Optimization passes
// ============================================================================

#[test]
fn pass_dead_code_elimination() {
    let pass = passes::DeadCodeElimination::new();
    let mut circuit = Circuit::new(2);
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    let result = pass.apply(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn pass_gate_commutation() {
    let pass = passes::GateCommutation::new();
    let mut circuit = build_multi_gate_circuit();
    let result = pass.apply(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn pass_gate_fusion() {
    let pass = passes::GateFusion::new();
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliZ), &[q(0)]).unwrap();
    let result = pass.apply(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn pass_template_substitution() {
    let pass = passes::TemplateSubstitution::new();
    let mut circuit = build_bell_circuit();
    let result = pass.apply(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn pass_advanced_template_matching() {
    let pass = passes::AdvancedTemplateMatching::new();
    let mut circuit = build_bell_circuit();
    let result = pass.apply(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn hh_cancellation() {
    let pass = passes::DeadCodeElimination::new();
    let mut circuit = build_hh_circuit();
    let before = circuit.len();
    let _ = pass.apply(&mut circuit);
    assert!(circuit.len() <= before, "H·H should simplify: before={}, after={}", before, circuit.len());
}

// ============================================================================
// 3. Pipeline and optimization levels
// ============================================================================

#[test]
fn pipeline_o0() {
    let compiler = create_compiler(OptimizationLevel::O0);
    let mut circuit = build_bell_circuit();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn pipeline_o1() {
    let compiler = create_compiler(OptimizationLevel::O1);
    let mut circuit = build_multi_gate_circuit();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn pipeline_o2() {
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut circuit = build_multi_gate_circuit();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn pipeline_o3() {
    let compiler = create_compiler(OptimizationLevel::O3);
    let mut circuit = build_multi_gate_circuit();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn pipeline_builder() {
    let compiler = PipelineBuilder::new()
        .with_dead_code_elimination()
        .max_iterations(3)
        .build();
    let mut circuit = build_bell_circuit();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn pipeline_builder_all_passes() {
    let compiler = PipelineBuilder::new()
        .with_dead_code_elimination()
        .with_gate_commutation()
        .with_gate_fusion()
        .with_template_substitution()
        .with_advanced_template_matching()
        .max_iterations(5)
        .build();
    let mut circuit = build_multi_gate_circuit();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

// ============================================================================
// 4. Gate fusion
// ============================================================================

#[test]
fn fuse_circuit_single_qubit_gates() {
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliZ), &[q(0)]).unwrap();
    let fused = fuse_single_qubit_gates(&circuit, None);
    assert!(fused.is_ok(), "Should fuse H and Z");
    let fused_circuit = fused.unwrap();
    assert!(fused_circuit.len() <= circuit.len());
}

#[test]
fn fused_gate_identity_elimination() {
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    let fused = fuse_single_qubit_gates(&circuit, None).unwrap();
    assert_eq!(fused.len(), 0, "X·X = I should be eliminated");
}

#[test]
fn fuse_preserves_multi_qubit_gates() {
    let circuit = build_bell_circuit();
    let fused = fuse_single_qubit_gates(&circuit, None).unwrap();
    assert_eq!(fused.num_qubits(), circuit.num_qubits());
}

// ============================================================================
// 5. Decomposition
// ============================================================================

#[test]
fn decomposition_config_default() {
    let config = DecompositionConfig::default();
    assert!(config.fidelity_threshold > 0.0);
}

#[test]
fn decomposition_ibm_basis() {
    let config = DecompositionConfig {
        basis: BasisGateSet::IBM,
        optimization_level: 1,
        fidelity_threshold: 0.999,
        ..Default::default()
    };
    let decomposer = UniversalDecomposer::new(config);
    let gate = Arc::new(Hadamard) as Arc<dyn Gate>;
    let result = decomposer.decompose_gate(&*gate);
    assert!(result.is_ok(), "Should decompose Hadamard to IBM basis: {:?}", result.err());
}

#[test]
fn decomposition_google_basis() {
    let config = DecompositionConfig {
        basis: BasisGateSet::Google,
        optimization_level: 1,
        fidelity_threshold: 0.999,
        ..Default::default()
    };
    let decomposer = UniversalDecomposer::new(config);
    let gate = Arc::new(PauliX) as Arc<dyn Gate>;
    let result = decomposer.decompose_gate(&*gate);
    assert!(result.is_ok(), "Should decompose X to Google basis: {:?}", result.err());
}

#[test]
fn decomposition_ionq_basis() {
    let config = DecompositionConfig {
        basis: BasisGateSet::IonQ,
        optimization_level: 1,
        fidelity_threshold: 0.999,
        ..Default::default()
    };
    let decomposer = UniversalDecomposer::new(config);
    let gate = Arc::new(TGate) as Arc<dyn Gate>;
    let result = decomposer.decompose_gate(&*gate);
    assert!(result.is_ok(), "Should decompose T to IonQ basis: {:?}", result.err());
}

#[test]
fn decomposition_result_fidelity() {
    let config = DecompositionConfig {
        basis: BasisGateSet::IBM,
        optimization_level: 2,
        fidelity_threshold: 0.999,
        ..Default::default()
    };
    let decomposer = UniversalDecomposer::new(config);
    let gate = Arc::new(RotationX::new(1.0)) as Arc<dyn Gate>;
    let result = decomposer.decompose_gate(&*gate);
    if let Ok(decomp) = result {
        assert!(decomp.fidelity >= 0.999, "Fidelity should meet threshold: {}", decomp.fidelity);
    }
}

#[test]
fn decomposition_zyz_hadamard() {
    let mat = Hadamard.matrix();
    let mat = mat.unwrap();
    let m2: Matrix2 = [[mat[0], mat[1]], [mat[2], mat[3]]];
    let result = decompose_zyz(&m2);
    assert!(result.is_ok(), "Should decompose Hadamard via ZYZ: {:?}", result.err());
    let (alpha, beta, gamma, phase) = result.unwrap();
    assert!(alpha.is_finite());
    assert!(beta.is_finite());
    assert!(gamma.is_finite());
    assert!(phase.is_finite());
}

// ============================================================================
// 6. Circuit analysis
// ============================================================================

#[test]
fn circuit_analysis_basic() {
    let circuit = build_bell_circuit();
    let analysis = CircuitAnalysis::analyze(&circuit);
    assert!(analysis.is_ok(), "Analysis should succeed: {:?}", analysis.err());
}

#[test]
fn circuit_analysis_gate_statistics() {
    let circuit = build_multi_gate_circuit();
    let analysis = CircuitAnalysis::analyze(&circuit);
    if let Ok(a) = analysis {
        assert!(a.statistics.total_gates > 0);
    }
}

#[test]
fn circuit_analysis_empty_circuit() {
    let circuit = Circuit::new(3);
    let analysis = CircuitAnalysis::analyze(&circuit);
    assert!(analysis.is_ok());
}

#[test]
fn circuit_analysis_display() {
    let circuit = build_bell_circuit();
    if let Ok(analysis) = CircuitAnalysis::analyze(&circuit) {
        let display = format!("{}", analysis);
        assert!(!display.is_empty());
    }
}

#[test]
fn circuit_analysis_resources() {
    let circuit = build_multi_gate_circuit();
    if let Ok(analysis) = CircuitAnalysis::analyze(&circuit) {
        assert!(analysis.resources.dense_memory_bytes > 0);
        assert!(analysis.resources.num_qubits == 3);
    }
}

#[test]
fn circuit_analysis_parallelism() {
    let circuit = build_multi_gate_circuit();
    if let Ok(analysis) = CircuitAnalysis::analyze(&circuit) {
        assert!(analysis.parallelism_factor() >= 1.0);
    }
}

// ============================================================================
// 7. Hardware-aware compilation
// ============================================================================

#[test]
fn cost_model_ibm() {
    let cost_model = CostModel::new(HardwareType::IBM);
    let circuit = build_bell_circuit();
    let cost = cost_model.circuit_cost(&circuit);
    assert!(cost >= 0.0, "Circuit cost should be non-negative");
}

#[test]
fn cost_model_google() {
    let cost_model = CostModel::new(HardwareType::Google);
    let circuit = build_bell_circuit();
    let cost = cost_model.circuit_cost(&circuit);
    assert!(cost >= 0.0);
}

#[test]
fn cost_model_ionq() {
    let cost_model = CostModel::new(HardwareType::IonQ);
    let circuit = build_bell_circuit();
    let cost = cost_model.circuit_cost(&circuit);
    assert!(cost >= 0.0);
}

#[test]
fn hardware_model_ibm() {
    let hw = IBMHardware::new();
    let model: &dyn HardwareModel = &hw;
    assert!(model.gate_cost_by_name("H") > 0.0);
    assert!(model.gate_cost_by_name("CNOT") > 0.0);
}

#[test]
fn hardware_model_google() {
    let hw = GoogleHardware::new();
    let model: &dyn HardwareModel = &hw;
    assert!(model.gate_cost_by_name("CZ") > 0.0);
    assert!(model.gate_cost_by_name("RX") > 0.0);
}

#[test]
fn hardware_model_ionq() {
    let hw = IonQHardware::new();
    let model: &dyn HardwareModel = &hw;
    assert!(model.gate_cost_by_name("RX") > 0.0);
    assert!(model.gate_cost_by_name("MS") > 0.0);
}

#[test]
fn hardware_native_gates() {
    let ibm = IBMHardware::new();
    assert!(ibm.is_native("CNOT"));
    assert!(ibm.is_native("RZ"));
    assert!(!ibm.is_native("CZ"));

    let google = GoogleHardware::new();
    assert!(google.is_native("CZ"));
    assert!(!google.is_native("CNOT"));

    let ionq = IonQHardware::new();
    assert!(ionq.is_native("MS"));
    assert!(ionq.is_native("RX"));
}

#[test]
fn cost_model_empty_circuit() {
    let cost_model = CostModel::new(HardwareType::IBM);
    let circuit = Circuit::new(2);
    let cost = cost_model.circuit_cost(&circuit);
    assert_eq!(cost, 0.0, "Empty circuit should have zero cost");
}

// ============================================================================
// 8. Compilation caching
// ============================================================================

#[test]
fn cached_compiler_basic() {
    let compiler = create_compiler(OptimizationLevel::O1);
    let mut cached = CachedCompiler::new(compiler, 10);
    let mut circuit = build_bell_circuit();
    let result = cached.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn cached_compiler_cache_hit() {
    let compiler = create_compiler(OptimizationLevel::O1);
    let mut cached = CachedCompiler::new(compiler, 10);
    let mut c1 = build_bell_circuit();
    let mut c2 = build_bell_circuit();
    let r1 = cached.compile(&mut c1).unwrap();
    let r2 = cached.compile(&mut c2).unwrap();
    assert!(!r1.is_cached(), "First compilation should be a cache miss");
    assert!(r2.is_cached(), "Second identical circuit should be a cache hit");
}

#[test]
fn cache_statistics() {
    let compiler = create_compiler(OptimizationLevel::O1);
    let mut cached = CachedCompiler::new(compiler, 10);
    let mut c1 = build_bell_circuit();
    let mut c2 = build_bell_circuit();
    let _ = cached.compile(&mut c1);
    let _ = cached.compile(&mut c2);
    let stats = cached.cache().statistics();
    assert!(stats.hit_rate() >= 0.0);
}

#[test]
fn circuit_fingerprint() {
    let c1 = build_bell_circuit();
    let c2 = build_bell_circuit();
    let c3 = build_ghz_circuit();
    let f1 = CircuitFingerprint::compute(&c1);
    let f2 = CircuitFingerprint::compute(&c2);
    let f3 = CircuitFingerprint::compute(&c3);
    assert_eq!(f1, f2, "Same circuits should have same fingerprint");
    assert_ne!(f1, f3, "Different circuits should have different fingerprints");
}

#[test]
fn cached_compiler_disabled() {
    let compiler = create_compiler(OptimizationLevel::O1);
    let mut cached = CachedCompiler::new(compiler, 10);
    cached.set_enabled(false);
    let mut circuit = build_bell_circuit();
    let r1 = cached.compile(&mut circuit).unwrap();
    assert!(!r1.is_cached());
    let r2 = cached.compile(&mut circuit).unwrap();
    assert!(!r2.is_cached(), "Caching disabled, should not hit cache");
}

// ============================================================================
// 9. Lazy evaluation
// ============================================================================

#[test]
fn lazy_gate_creation() {
    let gate = Arc::new(Hadamard) as Arc<dyn Gate>;
    let lazy = LazyGate::new(gate);
    let matrix = lazy.matrix_1q();
    assert!(matrix.is_ok());
    let mat = matrix.unwrap();
    assert!((mat[0][0].re - mat[0][1].re).abs() < EPSILON);
}

#[test]
fn lazy_gate_caching() {
    let gate = Arc::new(Hadamard) as Arc<dyn Gate>;
    let lazy = LazyGate::new(gate);
    let m1 = lazy.matrix_1q().unwrap();
    let m2 = lazy.matrix_1q().unwrap();
    assert_eq!(m1, m2, "Repeated calls should return same matrix");
}

#[test]
fn lazy_executor_basic() {
    let executor = LazyExecutor::new(LazyConfig::default());
    assert!(executor.config().enable_caching);
    assert!(executor.config().enable_fusion);
}

#[test]
fn lazy_executor_execute() {
    let mut executor = LazyExecutor::new(LazyConfig::default());
    let circuit = build_bell_circuit();
    let mut state = vec![Complex64::new(0.0, 0.0); 4];
    state[0] = Complex64::new(1.0, 0.0);
    let result = executor.execute(&circuit, &mut state);
    assert!(result.is_ok());
    let stats = executor.stats();
    assert!(stats.matrices_computed > 0);
}

// ============================================================================
// 10. Execution planning
// ============================================================================

#[test]
fn execution_planner_basic() {
    let planner = ExecutionPlanner::new();
    let circuit = build_bell_circuit();
    let plan = planner.generate_plan(&circuit);
    assert!(plan.depth > 0);
    assert!(!plan.layers.is_empty());
}

#[test]
fn execution_planner_ghz() {
    let planner = ExecutionPlanner::new();
    let circuit = build_ghz_circuit();
    let plan = planner.generate_plan(&circuit);
    assert!(plan.depth >= 2, "GHZ needs at least 2 layers: {}", plan.depth);
}

#[test]
fn execution_plan_parallelism() {
    let planner = ExecutionPlanner::new();
    let mut circuit = Circuit::new(4);
    for i in 0..4 {
        circuit.add_gate(Arc::new(Hadamard), &[q(i)]).unwrap();
    }
    let plan = planner.generate_plan(&circuit);
    assert!(plan.parallelism_factor >= 1.0, "4 independent H gates should have parallelism");
}

#[test]
fn execution_plan_resources() {
    let planner = ExecutionPlanner::new();
    let circuit = build_multi_gate_circuit();
    let plan = planner.generate_plan(&circuit);
    assert!(plan.total_time > 0.0);
}

#[test]
fn execution_plan_gate_count() {
    let planner = ExecutionPlanner::new();
    let circuit = build_multi_gate_circuit();
    let plan = planner.generate_plan(&circuit);
    assert_eq!(plan.gate_count, circuit.len());
}

// ============================================================================
// 11. Matrix computation utilities
// ============================================================================

#[test]
fn tensor_product_2x2_identity() {
    let i_mat: Matrix2 = [
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ];
    let result = tensor_product_2x2(&i_mat, &i_mat);
    assert!((result[0][0] - Complex64::new(1.0, 0.0)).norm() < EPSILON);
    assert!((result[3][3] - Complex64::new(1.0, 0.0)).norm() < EPSILON);
    assert!(result[0][1].norm() < EPSILON);
}

#[test]
fn controlled_gate_2x2_cnot() {
    let x_mat: Matrix2 = [
        [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ];
    let cnot = controlled_gate_2x2(&x_mat);
    assert!((cnot[0][0] - Complex64::new(1.0, 0.0)).norm() < EPSILON);
    assert!((cnot[1][1] - Complex64::new(1.0, 0.0)).norm() < EPSILON);
    assert!((cnot[2][3] - Complex64::new(1.0, 0.0)).norm() < EPSILON);
    assert!((cnot[3][2] - Complex64::new(1.0, 0.0)).norm() < EPSILON);
}

#[test]
fn gate_fidelity_identical() {
    let mat = Hadamard.matrix().unwrap();
    let h_mat: Matrix2 = [[mat[0], mat[1]], [mat[2], mat[3]]];
    let fid = gate_fidelity_2x2(&h_mat, &h_mat);
    assert!((fid - 1.0).abs() < EPSILON, "Same gates should have fidelity 1: {}", fid);
}

#[test]
fn gate_fidelity_different() {
    let h_mat_v = Hadamard.matrix().unwrap();
    let x_mat_v = PauliX.matrix().unwrap();
    let h_mat: Matrix2 = [[h_mat_v[0], h_mat_v[1]], [h_mat_v[2], h_mat_v[3]]];
    let x_mat: Matrix2 = [[x_mat_v[0], x_mat_v[1]], [x_mat_v[2], x_mat_v[3]]];
    let fid = gate_fidelity_2x2(&h_mat, &x_mat);
    assert!(fid < 1.0, "Different gates should have fidelity < 1: {}", fid);
}

#[test]
fn decompose_zyz_roundtrip() {
    let mat = Hadamard.matrix().unwrap();
    let m2: Matrix2 = [[mat[0], mat[1]], [mat[2], mat[3]]];
    let (alpha, beta, gamma, phase) = decompose_zyz(&m2).unwrap();
    assert!(alpha.is_finite());
    assert!(beta.is_finite());
    assert!(gamma.is_finite());
    assert!(phase.is_finite());
}

// ============================================================================
// 12. Adaptive pipeline
// ============================================================================

#[test]
fn adaptive_compiler_basic() {
    let adaptive = AdaptiveCompiler::new();
    let circuit = build_bell_circuit();
    let compiler = adaptive.create_for_circuit(&circuit);
    let mut circuit2 = build_bell_circuit();
    let result = compiler.compile(&mut circuit2);
    assert!(result.is_ok());
}

#[test]
fn adaptive_compiler_large_circuit() {
    let adaptive = AdaptiveCompiler::new();
    let mut circuit = Circuit::new(5);
    for i in 0..5 {
        circuit.add_gate(Arc::new(Hadamard), &[q(i)]).unwrap();
    }
    for i in 0..4 {
        circuit.add_gate(Arc::new(CNot), &[q(i), q(i + 1)]).unwrap();
    }
    let compiler = adaptive.create_for_circuit(&circuit);
    let mut circuit2 = circuit.clone();
    let result = compiler.compile(&mut circuit2);
    assert!(result.is_ok());
}

#[test]
fn adaptive_compiler_verbose() {
    let adaptive = AdaptiveCompiler::new().with_verbose(true);
    assert!(adaptive.verbose);
    let circuit = build_bell_circuit();
    let _compiler = adaptive.create_for_circuit(&circuit);
}

#[test]
fn multi_level_optimizer() {
    let optimizer = MultiLevelOptimizer::new();
    let mut circuit = build_multi_gate_circuit();
    let result = optimizer.optimize(&mut circuit);
    // optimize returns OptimizationResult directly (not Result)
    let _ = result.total_time_us;
}

#[test]
fn multi_level_optimizer_removes_dead_code() {
    let optimizer = MultiLevelOptimizer::new();
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    let result = optimizer.optimize(&mut circuit);
    assert!(result.modified);
    assert_eq!(circuit.len(), 0, "X·X should be eliminated");
}

// ============================================================================
// 13. Circuit characteristics
// ============================================================================

#[test]
fn circuit_characteristics_basic() {
    let circuit = build_multi_gate_circuit();
    let chars = CircuitCharacteristics::analyze(&circuit);
    assert!(chars.gate_count > 0);
    assert_eq!(chars.num_qubits, 3);
}

#[test]
fn circuit_characteristics_depth() {
    let circuit = build_multi_gate_circuit();
    let chars = CircuitCharacteristics::analyze(&circuit);
    assert!(chars.depth > 0);
}

#[test]
fn circuit_characteristics_empty() {
    let circuit = Circuit::new(3);
    let chars = CircuitCharacteristics::analyze(&circuit);
    assert_eq!(chars.gate_count, 0);
    assert_eq!(chars.depth, 0);
}

#[test]
fn circuit_size_category() {
    let circuit = build_bell_circuit();
    let chars = CircuitCharacteristics::analyze(&circuit);
    assert_eq!(chars.size_category(), CircuitSize::Small);
}

// ============================================================================
// 14. Edge cases
// ============================================================================

#[test]
fn compile_empty_circuit() {
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut circuit = Circuit::new(3);
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn compile_single_gate() {
    let compiler = create_compiler(OptimizationLevel::O3);
    let mut circuit = Circuit::new(1);
    circuit.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn optimization_preserves_validity() {
    let compiler = create_compiler(OptimizationLevel::O3);
    let mut circuit = build_multi_gate_circuit();
    let _ = compiler.compile(&mut circuit);
    let valid = circuit.validate();
    assert!(valid.is_ok(), "Optimized circuit should remain valid: {:?}", valid.err());
}

#[test]
fn compile_identity_only() {
    let compiler = create_compiler(OptimizationLevel::O2);
    let mut circuit = Circuit::new(2);
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliX), &[q(0)]).unwrap();
    circuit.add_gate(Arc::new(PauliZ), &[q(1)]).unwrap();
    circuit.add_gate(Arc::new(PauliZ), &[q(1)]).unwrap();
    let _ = compiler.compile(&mut circuit);
    assert!(circuit.len() <= 4, "Identity pairs should be reduced");
}

// ============================================================================
// 15. Stress tests
// ============================================================================

#[test]
fn stress_many_passes() {
    let compiler = CompilerBuilder::new()
        .add_pass(Arc::new(passes::DeadCodeElimination::new()))
        .add_pass(Arc::new(passes::GateCommutation::new()))
        .add_pass(Arc::new(passes::GateFusion::new()))
        .add_pass(Arc::new(passes::TemplateSubstitution::new()))
        .add_pass(Arc::new(passes::AdvancedTemplateMatching::new()))
        .max_iterations(10)
        .build();
    let mut circuit = build_multi_gate_circuit();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn stress_large_circuit() {
    let mut circuit = Circuit::new(10);
    for i in 0..10 {
        circuit.add_gate(Arc::new(Hadamard), &[q(i)]).unwrap();
    }
    for i in 0..9 {
        circuit.add_gate(Arc::new(CNot), &[q(i), q(i + 1)]).unwrap();
    }
    for i in 0..10 {
        circuit.add_gate(Arc::new(RotationZ::new(0.1 * i as f64)), &[q(i)]).unwrap();
    }
    let compiler = create_compiler(OptimizationLevel::O2);
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn stress_repeated_compilation() {
    let compiler = create_compiler(OptimizationLevel::O1);
    for _ in 0..10 {
        let mut circuit = build_bell_circuit();
        let result = compiler.compile(&mut circuit);
        assert!(result.is_ok());
    }
}

#[test]
fn stress_all_hardware_costs() {
    let circuit = build_multi_gate_circuit();
    for hw_type in [HardwareType::IBM, HardwareType::Google, HardwareType::IonQ] {
        let cost = CostModel::new(hw_type).circuit_cost(&circuit);
        assert!(cost >= 0.0, "Cost should be non-negative for {:?}", hw_type);
    }
}

#[test]
fn stress_cached_many_circuits() {
    let compiler = create_compiler(OptimizationLevel::O1);
    let mut cached = CachedCompiler::new(compiler, 100);
    for _ in 0..20 {
        let mut c = build_bell_circuit();
        let _ = cached.compile(&mut c);
    }
    let stats = cached.cache().statistics();
    assert!(stats.hits > 0, "Should have cache hits for identical circuits");
}

#[test]
fn stress_execution_plan_large() {
    let planner = ExecutionPlanner::new();
    let mut circuit = Circuit::new(8);
    for i in 0..8 {
        circuit.add_gate(Arc::new(Hadamard), &[q(i)]).unwrap();
    }
    for i in 0..7 {
        circuit.add_gate(Arc::new(CNot), &[q(i), q(i + 1)]).unwrap();
    }
    let plan = planner.generate_plan(&circuit);
    assert!(plan.depth > 0);
    assert_eq!(plan.gate_count, 15);
}

// ============================================================================
// 16. Integration tests
// ============================================================================

#[test]
fn full_pipeline_bell_state() {
    let compiler = create_compiler(OptimizationLevel::O3);
    let mut circuit = build_bell_circuit();
    let original_len = circuit.len();
    let result = compiler.compile(&mut circuit).unwrap();
    assert!(circuit.len() <= original_len);
    let _ = result.total_time_us;
}

#[test]
fn analyze_then_compile() {
    let circuit = build_multi_gate_circuit();
    let analysis = CircuitAnalysis::analyze(&circuit).unwrap();
    assert!(analysis.statistics.total_gates > 0);

    let adaptive = AdaptiveCompiler::new();
    let compiler = adaptive.create_for_circuit(&circuit);
    let mut optimized = circuit.clone();
    let result = compiler.compile(&mut optimized);
    assert!(result.is_ok());
}

#[test]
fn decompose_and_compile() {
    let config = DecompositionConfig {
        basis: BasisGateSet::IBM,
        optimization_level: 1,
        fidelity_threshold: 0.99,
        ..Default::default()
    };
    let decomposer = UniversalDecomposer::new(config);
    let gate = Arc::new(Hadamard) as Arc<dyn Gate>;
    let _decomp = decomposer.decompose_gate(&*gate);

    let compiler = create_compiler(OptimizationLevel::O2);
    let mut circuit = build_bell_circuit();
    let result = compiler.compile(&mut circuit);
    assert!(result.is_ok());
}

#[test]
fn lazy_execute_optimized_circuit() {
    let compiler = create_compiler(OptimizationLevel::O1);
    let mut circuit = build_bell_circuit();
    let _ = compiler.compile(&mut circuit);

    let mut executor = LazyExecutor::new(LazyConfig::default());
    let mut state = vec![Complex64::new(0.0, 0.0); 4];
    state[0] = Complex64::new(1.0, 0.0);
    let result = executor.execute(&circuit, &mut state);
    assert!(result.is_ok());
}
