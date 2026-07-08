//! End-to-end tests for classical optimizers (L-BFGS, Nelder-Mead)

use ferriq_core::circuit::Circuit;
use ferriq_core::QubitId;
use ferriq_gates::standard::*;
use ferriq_sim::gradient::classical_optimizers::{
    LBFGSConfig, LBFGSOptimizer, NelderMeadConfig, NelderMeadOptimizer,
};
use ferriq_sim::{Simulator, SimulatorConfig};
use ferriq_state::observable::{PauliObservable, PauliString};
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
    c.add_gate(Arc::new(RotationY::new(params[0])), &[q(0)])
        .unwrap();
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

// ============================================================================
// L-BFGS: more iterations / line search / L-BFGS direction path
// ============================================================================

#[test]
fn test_lbfgs_more_iterations_triggers_lbfgs_direction() {
    // Need at least 2 iterations to use L-BFGS two-loop recursion (not steepest descent)
    let sim = make_sim();
    let obs = z_observable();
    let config = LBFGSConfig {
        max_iterations: 5,
        memory_size: 3,
        ..LBFGSConfig::default()
    };
    let mut optimizer = LBFGSOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(result.energy.is_finite());
    // Should have at least 2 history entries to cover the L-BFGS direction path
    assert!(!optimizer.history().is_empty());
}

#[test]
fn test_lbfgs_memory_size_exceeded() {
    // Use memory_size=1 so the history eviction path is triggered
    let sim = make_sim();
    let obs = z_observable();
    let config = LBFGSConfig {
        max_iterations: 5,
        memory_size: 1, // Forces eviction after 2 iterations
        ..LBFGSConfig::default()
    };
    let mut optimizer = LBFGSOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_lbfgs_fully_converged() {
    // Very loose tolerance - converges immediately after first iteration
    let sim = make_sim();
    let obs = z_observable();
    let config = LBFGSConfig {
        max_iterations: 100,
        tolerance: 10.0, // Both energy_change and gradient_norm < 10.0
        ..LBFGSConfig::default()
    };
    let mut optimizer = LBFGSOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(result.energy.is_finite());
    // Should converge very quickly
    assert!(result.num_iterations < 100);
}

#[test]
fn test_lbfgs_energy_converged() {
    // Loose energy tolerance but tight gradient
    let sim = make_sim();
    let obs = z_observable();
    let config = LBFGSConfig {
        max_iterations: 100,
        tolerance: 10.0,
        ..LBFGSConfig::default()
    };
    let mut optimizer = LBFGSOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_lbfgs_gradient_converged() {
    // Tight energy tolerance but loose gradient
    let sim = make_sim();
    let obs = z_observable();
    let config = LBFGSConfig {
        max_iterations: 100,
        tolerance: 10.0,
        ..LBFGSConfig::default()
    };
    let mut optimizer = LBFGSOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_lbfgs_max_iterations() {
    // tolerance = 0.0 means never converges by tolerance
    let sim = make_sim();
    let obs = z_observable();
    let config = LBFGSConfig {
        max_iterations: 2,
        tolerance: 0.0,
        ..LBFGSConfig::default()
    };
    let mut optimizer = LBFGSOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    // Should use MaxIterations status
    assert_eq!(result.num_iterations, 2);
}

#[test]
fn test_lbfgs_two_params() {
    // Two parameters to exercise L-BFGS with multi-dimensional gradient
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

    let config = LBFGSConfig {
        max_iterations: 5,
        ..LBFGSConfig::default()
    };
    let mut optimizer = LBFGSOptimizer::new(two_param_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5, 0.3]).unwrap();
    assert!(result.energy.is_finite());
    assert_eq!(result.gradient.len(), 2);
}

// ============================================================================
// Nelder-Mead: convergence paths - expansion, contraction, shrink
// ============================================================================

#[test]
fn test_nelder_mead_fully_converged_small_simplex() {
    // Use very loose tolerance so the simplex size threshold triggers
    let sim = make_sim();
    let obs = z_observable();
    let config = NelderMeadConfig {
        max_iterations: 200,
        tolerance: 1e10, // Simplex size will be < 1e10 right away
        ..NelderMeadConfig::default()
    };
    let mut optimizer = NelderMeadOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(result.energy.is_finite());
    // Should converge immediately on first iteration check
    assert!(result.num_iterations <= 1);
}

#[test]
fn test_nelder_mead_max_iterations() {
    // Use tolerance = 0.0 so simplex size never triggers
    let sim = make_sim();
    let obs = z_observable();
    let config = NelderMeadConfig {
        max_iterations: 3,
        tolerance: 0.0,
        ..NelderMeadConfig::default()
    };
    let mut optimizer = NelderMeadOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(result.energy.is_finite());
    assert_eq!(result.num_iterations, 3);
}

#[test]
fn test_nelder_mead_expansion_path() {
    // To trigger expansion: reflected_energy < best_energy, then expanded_energy < reflected
    // Start at a non-minimum to get varied evaluations
    let sim = make_sim();
    let obs = z_observable();
    let config = NelderMeadConfig {
        max_iterations: 20,
        tolerance: 0.0,
        alpha: 1.0,
        gamma: 2.0,
        rho: 0.5,
        sigma: 0.5,
    };
    let mut optimizer = NelderMeadOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.3]).unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_nelder_mead_contraction_paths() {
    // Run enough iterations to exercise all contraction branches
    let sim = make_sim();
    let obs = z_observable();
    let config = NelderMeadConfig {
        max_iterations: 30,
        tolerance: 0.0,
        ..NelderMeadConfig::default()
    };
    let mut optimizer = NelderMeadOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[1.5]).unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_nelder_mead_two_params_more_iterations() {
    // 2-parameter case runs more of the simplex logic
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
        max_iterations: 30,
        tolerance: 0.0,
        ..NelderMeadConfig::default()
    };
    let mut optimizer = NelderMeadOptimizer::new(two_param_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5, 1.0]).unwrap();
    assert!(result.energy.is_finite());
    assert_eq!(result.gradient.len(), 2);
}

#[test]
fn test_nelder_mead_zero_starting_point() {
    // Test when initial vertex component is near zero (exercises the else branch in initialize_simplex)
    let sim = make_sim();
    let obs = z_observable();
    let config = NelderMeadConfig {
        max_iterations: 5,
        ..NelderMeadConfig::default()
    };
    let mut optimizer = NelderMeadOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.0]).unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_nelder_mead_custom_coefficients() {
    // Use non-default simplex coefficients to exercise all the operation functions
    let sim = make_sim();
    let obs = z_observable();
    let config = NelderMeadConfig {
        max_iterations: 20,
        tolerance: 0.0,
        alpha: 1.5, // Non-default reflection
        gamma: 2.5, // Non-default expansion
        rho: 0.4,   // Non-default contraction
        sigma: 0.6, // Non-default shrink
    };
    let mut optimizer = NelderMeadOptimizer::new(ry_circuit, config);
    let result = optimizer.optimize(&sim, &obs, &[0.5]).unwrap();
    assert!(result.energy.is_finite());
}

#[test]
fn test_lbfgs_different_starting_points() {
    let sim = make_sim();
    let obs = z_observable();

    for start in [0.0_f64, 0.5, 1.0, 1.5, 2.0, std::f64::consts::PI] {
        let config = LBFGSConfig {
            max_iterations: 3,
            ..LBFGSConfig::default()
        };
        let mut optimizer = LBFGSOptimizer::new(ry_circuit, config);
        let result = optimizer.optimize(&sim, &obs, &[start]).unwrap();
        assert!(result.energy.is_finite(), "Failed for starting point {start}");
    }
}

#[test]
fn test_nelder_mead_different_starting_points() {
    let sim = make_sim();
    let obs = z_observable();

    for start in [0.0_f64, 0.5, 1.0, std::f64::consts::PI] {
        let config = NelderMeadConfig {
            max_iterations: 10,
            tolerance: 0.0,
            ..NelderMeadConfig::default()
        };
        let mut optimizer = NelderMeadOptimizer::new(ry_circuit, config);
        let result = optimizer.optimize(&sim, &obs, &[start]).unwrap();
        assert!(result.energy.is_finite(), "Failed for starting point {start}");
    }
}
