//! End-to-end tests for classical optimizers (L-BFGS, Nelder-Mead)

use simq_core::circuit::Circuit;
use simq_core::QubitId;
use simq_gates::standard::*;
use simq_sim::gradient::classical_optimizers::{
    LBFGSConfig, LBFGSOptimizer, NelderMeadConfig, NelderMeadOptimizer,
};
use simq_sim::{Simulator, SimulatorConfig};
use simq_state::observable::{PauliObservable, PauliString};
use std::sync::Arc;

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

// ============================================================================
// L-BFGS tests
// ============================================================================

#[test]
fn test_lbfgs_config_default() {
    let config = LBFGSConfig::default();
    assert_eq!(config.max_iterations, 100);
    assert_eq!(config.memory_size, 10);
    assert!(config.tolerance > 0.0);
    assert!(config.gradient_epsilon > 0.0);
    assert!(config.line_search_tolerance > 0.0);
    assert!(config.max_line_search_iterations > 0);
}

#[test]
fn test_lbfgs_config_custom() {
    let config = LBFGSConfig {
        max_iterations: 50,
        tolerance: 1e-4,
        memory_size: 5,
        ..LBFGSConfig::default()
    };
    assert_eq!(config.max_iterations, 50);
    assert_eq!(config.memory_size, 5);
}

#[test]
fn test_lbfgs_optimizer_new() {
    let config = LBFGSConfig {
        max_iterations: 3,
        ..LBFGSConfig::default()
    };
    let optimizer = LBFGSOptimizer::new(ry_circuit, config);
    assert!(optimizer.history().is_empty());
}

#[test]
fn test_lbfgs_optimizer_runs() {
    let sim = make_sim();
    let obs = z_observable();
    let config = LBFGSConfig {
        max_iterations: 3,
        ..LBFGSConfig::default()
    };
    let mut optimizer = LBFGSOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.energy.is_finite());
    assert!(result.num_iterations > 0);
}

#[test]
fn test_lbfgs_optimizer_history() {
    let sim = make_sim();
    let obs = z_observable();
    let config = LBFGSConfig {
        max_iterations: 3,
        ..LBFGSConfig::default()
    };
    let mut optimizer = LBFGSOptimizer::new(ry_circuit, config);
    let _ = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(!optimizer.history().is_empty());
    let first = &optimizer.history()[0];
    assert_eq!(first.iteration, 0);
    assert!(first.energy.is_finite());
}

#[test]
fn test_lbfgs_optimizer_reset() {
    let sim = make_sim();
    let obs = z_observable();
    let config = LBFGSConfig {
        max_iterations: 3,
        ..LBFGSConfig::default()
    };
    let mut optimizer = LBFGSOptimizer::new(ry_circuit, config);
    let _ = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    optimizer.reset();
    assert!(optimizer.history().is_empty());
}

// ============================================================================
// Nelder-Mead tests
// ============================================================================

#[test]
fn test_nelder_mead_config_default() {
    let config = NelderMeadConfig::default();
    assert_eq!(config.max_iterations, 200);
    assert!(config.tolerance > 0.0);
    assert!((config.alpha - 1.0).abs() < 1e-12);
    assert!((config.gamma - 2.0).abs() < 1e-12);
    assert!((config.rho - 0.5).abs() < 1e-12);
    assert!((config.sigma - 0.5).abs() < 1e-12);
}

#[test]
fn test_nelder_mead_config_custom() {
    let config = NelderMeadConfig {
        max_iterations: 10,
        tolerance: 1e-3,
        ..NelderMeadConfig::default()
    };
    assert_eq!(config.max_iterations, 10);
}

#[test]
fn test_nelder_mead_optimizer_new() {
    let config = NelderMeadConfig {
        max_iterations: 5,
        ..NelderMeadConfig::default()
    };
    let optimizer = NelderMeadOptimizer::new(ry_circuit, config);
    assert!(optimizer.history().is_empty());
}

#[test]
fn test_nelder_mead_optimizer_runs() {
    let sim = make_sim();
    let obs = z_observable();
    let config = NelderMeadConfig {
        max_iterations: 5,
        ..NelderMeadConfig::default()
    };
    let mut optimizer = NelderMeadOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_nelder_mead_history() {
    let sim = make_sim();
    let obs = z_observable();
    let config = NelderMeadConfig {
        max_iterations: 5,
        ..NelderMeadConfig::default()
    };
    let mut optimizer = NelderMeadOptimizer::new(ry_circuit, config);
    let _ = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    // Should have history entries
    assert!(!optimizer.history().is_empty());
    let first = &optimizer.history()[0];
    assert_eq!(first.iteration, 0);
    assert!(first.energy.is_finite());
}

#[test]
fn test_nelder_mead_reset() {
    let sim = make_sim();
    let obs = z_observable();
    let config = NelderMeadConfig {
        max_iterations: 5,
        ..NelderMeadConfig::default()
    };
    let mut optimizer = NelderMeadOptimizer::new(ry_circuit, config);
    let _ = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    optimizer.reset();
    assert!(optimizer.history().is_empty());
}

#[test]
fn test_nelder_mead_two_params() {
    let sim = make_sim();
    let obs = z_observable();

    fn two_param_circuit(params: &[f64]) -> Circuit {
        let mut c = Circuit::new(1);
        c.add_gate(Arc::new(RotationY::new(params[0])), &[QubitId::new(0)])
            .unwrap();
        c.add_gate(Arc::new(RotationX::new(params[1])), &[QubitId::new(0)])
            .unwrap();
        c
    }

    let config = NelderMeadConfig {
        max_iterations: 5,
        ..NelderMeadConfig::default()
    };
    let mut optimizer = NelderMeadOptimizer::new(two_param_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5, 0.3]);
    assert!(result.is_ok());
    let result = result.unwrap();
    assert!(result.energy.is_finite());
    // With 2 params, gradient is length 2
    assert_eq!(result.gradient.len(), 2);
}
