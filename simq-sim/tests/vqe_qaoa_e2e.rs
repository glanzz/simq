//! End-to-end tests for VQE/QAOA optimizers

use simq_core::circuit::Circuit;
use simq_core::QubitId;
use simq_gates::standard::*;
use simq_sim::gradient::vqe_qaoa::{
    AdamConfig, AdamOptimizer, ConvergenceStatus, MomentumConfig, MomentumOptimizer,
    OptimizationResult, OptimizationStep, QAOAConfig, QAOAOptimizer, VQEConfig, VQEOptimizer,
    gradient_descent,
};
use simq_sim::{Simulator, SimulatorConfig};
use simq_state::observable::{PauliObservable, PauliString};
use std::sync::Arc;
use std::time::Duration;

fn q(i: usize) -> QubitId {
    QubitId::new(i)
}

fn make_sim() -> Simulator {
    Simulator::new(SimulatorConfig::default().with_optimization(false))
}

fn z_observable() -> PauliObservable {
    PauliObservable::from_pauli_string(PauliString::from_str("Z").unwrap(), 1.0)
}

fn ry_circuit(params: &[f64]) -> Circuit {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(RotationY::new(params[0])), &[q(0)]).unwrap();
    c
}

fn qaoa_circuit(params: &[f64]) -> Circuit {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(RotationY::new(params[0])), &[q(0)]).unwrap();
    c.add_gate(Arc::new(RotationX::new(params[1])), &[q(0)]).unwrap();
    c
}

// ============================================================================
// VQEConfig tests
// ============================================================================

#[test]
fn test_vqe_config_default() {
    let config = VQEConfig::default();
    assert_eq!(config.max_iterations, 1000);
    assert!(config.energy_tolerance > 0.0);
    assert!(config.gradient_tolerance > 0.0);
    assert!(config.learning_rate > 0.0);
    assert!(config.adaptive_learning_rate);
}

#[test]
fn test_vqe_config_custom() {
    let config = VQEConfig {
        max_iterations: 42,
        energy_tolerance: 1e-3,
        gradient_tolerance: 1e-3,
        learning_rate: 0.05,
        adaptive_learning_rate: false,
        ..VQEConfig::default()
    };
    assert_eq!(config.max_iterations, 42);
    assert!(!config.adaptive_learning_rate);
}

// ============================================================================
// VQEOptimizer tests
// ============================================================================

#[test]
fn test_vqe_optimizer_new() {
    let config = VQEConfig {
        max_iterations: 3,
        ..VQEConfig::default()
    };
    let optimizer = VQEOptimizer::new(ry_circuit, config);
    // Just check it constructs without panic
    assert!(optimizer.history().is_empty());
}

#[test]
fn test_vqe_optimizer_runs() {
    let sim = make_sim();
    let obs = z_observable();
    let config = VQEConfig {
        max_iterations: 3,
        ..VQEConfig::default()
    };
    let mut optimizer = VQEOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.num_iterations > 0);
    assert!(result.energy.is_finite());
}

#[test]
fn test_vqe_optimizer_history() {
    let sim = make_sim();
    let obs = z_observable();
    let config = VQEConfig {
        max_iterations: 3,
        ..VQEConfig::default()
    };
    let mut optimizer = VQEOptimizer::new(ry_circuit, config);
    let _ = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(!optimizer.history().is_empty());
    // Each step should have energy and parameter info
    let first = &optimizer.history()[0];
    assert_eq!(first.iteration, 0);
    assert!(!first.parameters.is_empty());
    assert!(first.energy.is_finite());
}

#[test]
fn test_vqe_optimizer_reset() {
    let sim = make_sim();
    let obs = z_observable();
    let config = VQEConfig {
        max_iterations: 3,
        ..VQEConfig::default()
    };
    let mut optimizer = VQEOptimizer::new(ry_circuit, config);
    let _ = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(!optimizer.history().is_empty());
    optimizer.reset();
    assert!(optimizer.history().is_empty());
}

#[test]
fn test_vqe_optimizer_adaptive_lr_disabled() {
    let sim = make_sim();
    let obs = z_observable();
    let config = VQEConfig {
        max_iterations: 3,
        adaptive_learning_rate: false,
        ..VQEConfig::default()
    };
    let mut optimizer = VQEOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(result.energy.is_finite());
}

// ============================================================================
// OptimizationResult tests
// ============================================================================

fn make_step(iteration: usize, energy: f64, params: Vec<f64>) -> OptimizationStep {
    OptimizationStep {
        iteration,
        parameters: params,
        energy,
        gradient: vec![0.0],
        gradient_norm: 0.0,
        energy_change: 0.0,
        step_time: Duration::from_millis(1),
        status: ConvergenceStatus::NotConverged,
    }
}

#[test]
fn test_optimization_result_converged() {
    let result = OptimizationResult {
        parameters: vec![1.0],
        energy: -0.5,
        gradient: vec![0.0],
        status: ConvergenceStatus::EnergyConverged,
        num_iterations: 5,
        total_time: Duration::from_millis(10),
        history: vec![],
    };
    assert!(result.converged());

    let result2 = OptimizationResult {
        status: ConvergenceStatus::GradientConverged,
        ..result.clone()
    };
    assert!(result2.converged());

    let result3 = OptimizationResult {
        status: ConvergenceStatus::FullyConverged,
        ..result.clone()
    };
    assert!(result3.converged());

    let result4 = OptimizationResult {
        status: ConvergenceStatus::NotConverged,
        ..result.clone()
    };
    assert!(!result4.converged());

    let result5 = OptimizationResult {
        status: ConvergenceStatus::MaxIterations,
        ..result.clone()
    };
    assert!(!result5.converged());

    let result6 = OptimizationResult {
        status: ConvergenceStatus::Plateau,
        ..result
    };
    assert!(!result6.converged());
}

#[test]
fn test_optimization_result_best_energy() {
    let step1 = make_step(0, -0.3, vec![0.5]);
    let step2 = make_step(1, -0.8, vec![0.9]);
    let result = OptimizationResult {
        parameters: vec![1.0],
        energy: -0.6,
        gradient: vec![0.0],
        status: ConvergenceStatus::MaxIterations,
        num_iterations: 2,
        total_time: Duration::from_millis(10),
        history: vec![step1, step2],
    };
    assert!((result.best_energy() - (-0.8)).abs() < 1e-12);
}

#[test]
fn test_optimization_result_best_parameters() {
    let step1 = make_step(0, -0.3, vec![0.5]);
    let step2 = make_step(1, -0.8, vec![0.9]);
    let result = OptimizationResult {
        parameters: vec![1.0],
        energy: -0.6,
        gradient: vec![0.0],
        status: ConvergenceStatus::MaxIterations,
        num_iterations: 2,
        total_time: Duration::from_millis(10),
        history: vec![step1, step2],
    };
    let best_params = result.best_parameters();
    assert!((best_params[0] - 0.9).abs() < 1e-12);
}

#[test]
fn test_optimization_result_empty_history() {
    let result = OptimizationResult {
        parameters: vec![1.0],
        energy: -0.5,
        gradient: vec![0.0],
        status: ConvergenceStatus::MaxIterations,
        num_iterations: 0,
        total_time: Duration::from_millis(0),
        history: vec![],
    };
    // With empty history, falls back to final values
    assert!((result.best_energy() - (-0.5)).abs() < 1e-12);
    assert_eq!(result.best_parameters(), &[1.0]);
}

// ============================================================================
// ConvergenceStatus tests
// ============================================================================

#[test]
fn test_convergence_status_variants() {
    let statuses = [
        ConvergenceStatus::NotConverged,
        ConvergenceStatus::EnergyConverged,
        ConvergenceStatus::GradientConverged,
        ConvergenceStatus::FullyConverged,
        ConvergenceStatus::MaxIterations,
        ConvergenceStatus::Plateau,
    ];

    // Test PartialEq
    assert_eq!(ConvergenceStatus::NotConverged, ConvergenceStatus::NotConverged);
    assert_ne!(ConvergenceStatus::NotConverged, ConvergenceStatus::EnergyConverged);

    // Test Clone and Copy
    let s = ConvergenceStatus::EnergyConverged;
    let s2 = s;   // Copy
    let s3 = s.clone(); // Clone
    assert_eq!(s, s2);
    assert_eq!(s, s3);

    // Test Debug
    for status in &statuses {
        let debug_str = format!("{:?}", status);
        assert!(!debug_str.is_empty());
    }
}

// ============================================================================
// QAOAConfig tests
// ============================================================================

#[test]
fn test_qaoa_config_default() {
    let config = QAOAConfig::default();
    assert_eq!(config.num_layers, 1);
    assert_eq!(config.max_iterations, 100);
    assert!(!config.layer_wise);
    assert!(config.energy_tolerance > 0.0);
    assert!(config.gamma_learning_rate > 0.0);
    assert!(config.beta_learning_rate > 0.0);
}

// ============================================================================
// QAOAOptimizer tests
// ============================================================================

#[test]
fn test_qaoa_optimizer_runs() {
    let sim = make_sim();
    let obs = z_observable();
    let config = QAOAConfig {
        num_layers: 1,
        max_iterations: 2,
        ..QAOAConfig::default()
    };
    let mut optimizer = QAOAOptimizer::new(qaoa_circuit, config);
    // 2 params for 1 layer (gamma, beta)
    let result = optimizer.optimize(&sim, &obs, &[0.1, 0.2]);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_qaoa_optimizer_wrong_param_count() {
    let sim = make_sim();
    let obs = z_observable();
    let config = QAOAConfig {
        num_layers: 2,
        max_iterations: 2,
        ..QAOAConfig::default()
    };
    let mut optimizer = QAOAOptimizer::new(qaoa_circuit, config);
    // Wrong: 2 layers need 4 params, but we provide 2
    let result = optimizer.optimize(&sim, &obs, &[0.1, 0.2]);
    assert!(result.is_err());
}

#[test]
fn test_qaoa_optimizer_layer_wise() {
    let sim = make_sim();
    let obs = z_observable();
    let config = QAOAConfig {
        num_layers: 1,
        max_iterations: 2,
        layer_wise: true,
        ..QAOAConfig::default()
    };
    let mut optimizer = QAOAOptimizer::new(qaoa_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.1, 0.2]);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.energy.is_finite());
    // Layer-wise returns FullyConverged
    assert_eq!(result.status, ConvergenceStatus::FullyConverged);
}

#[test]
fn test_qaoa_optimizer_history_and_reset() {
    let sim = make_sim();
    let obs = z_observable();
    let config = QAOAConfig {
        num_layers: 1,
        max_iterations: 2,
        ..QAOAConfig::default()
    };
    let mut optimizer = QAOAOptimizer::new(qaoa_circuit, config);
    let _ = optimizer.optimize(&sim, &obs, &[0.1, 0.2]).unwrap();
    // history may or may not have entries (depends on convergence)
    let _hist_len = optimizer.history().len();
    optimizer.reset();
    assert!(optimizer.history().is_empty());
}

// ============================================================================
// gradient_descent utility function
// ============================================================================

#[test]
fn test_gradient_descent_util() {
    let sim = make_sim();
    let obs = z_observable();
    let result = gradient_descent(&sim, ry_circuit, &obs, &[0.5], 0.01, 3);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.energy.is_finite());
}

// ============================================================================
// AdamOptimizer tests
// ============================================================================

#[test]
fn test_adam_config_default() {
    let config = AdamConfig::default();
    assert_eq!(config.max_iterations, 1000);
    assert!((config.beta1 - 0.9).abs() < 1e-12);
    assert!((config.beta2 - 0.999).abs() < 1e-12);
    assert!(config.epsilon > 0.0);
}

#[test]
fn test_adam_optimizer_new() {
    let config = AdamConfig {
        max_iterations: 3,
        ..AdamConfig::default()
    };
    let optimizer = AdamOptimizer::new(ry_circuit, config);
    assert!(optimizer.history().is_empty());
}

#[test]
fn test_adam_optimizer_runs() {
    let sim = make_sim();
    let obs = z_observable();
    let config = AdamConfig {
        max_iterations: 3,
        ..AdamConfig::default()
    };
    let mut optimizer = AdamOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_adam_optimizer_history() {
    let sim = make_sim();
    let obs = z_observable();
    let config = AdamConfig {
        max_iterations: 3,
        ..AdamConfig::default()
    };
    let mut optimizer = AdamOptimizer::new(ry_circuit, config);
    let _ = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(!optimizer.history().is_empty());
}

#[test]
fn test_adam_optimizer_reset() {
    let sim = make_sim();
    let obs = z_observable();
    let config = AdamConfig {
        max_iterations: 3,
        ..AdamConfig::default()
    };
    let mut optimizer = AdamOptimizer::new(ry_circuit, config);
    let _ = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    optimizer.reset();
    assert!(optimizer.history().is_empty());
}

// ============================================================================
// MomentumOptimizer tests
// ============================================================================

#[test]
fn test_momentum_config_default() {
    let config = MomentumConfig::default();
    assert_eq!(config.max_iterations, 1000);
    assert!((config.momentum - 0.9).abs() < 1e-12);
}

#[test]
fn test_momentum_optimizer_runs() {
    let sim = make_sim();
    let obs = z_observable();
    let config = MomentumConfig {
        max_iterations: 3,
        ..MomentumConfig::default()
    };
    let mut optimizer = MomentumOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_momentum_optimizer_history_and_reset() {
    let sim = make_sim();
    let obs = z_observable();
    let config = MomentumConfig {
        max_iterations: 3,
        ..MomentumConfig::default()
    };
    let mut optimizer = MomentumOptimizer::new(ry_circuit, config);
    let _ = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(!optimizer.history().is_empty());
    optimizer.reset();
    assert!(optimizer.history().is_empty());
}
