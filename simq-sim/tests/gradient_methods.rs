use simq_core::circuit::Circuit;
use simq_core::QubitId;
use simq_gates::standard::*;
use simq_sim::gradient::autodiff::{gradient_forward, Dual, HybridAD};
use simq_sim::gradient::batch::{evaluate_batch_expectation, evaluate_multi_observable, grid_search};
use simq_sim::gradient::batch_advanced::{
    latin_hypercube_sampling, line_search, verify_gradients, AdaptiveBatchEvaluator, BatchConfig,
    ImportanceSampler,
};
use simq_sim::gradient::classical_optimizers::{LBFGSConfig, NelderMeadConfig};
use simq_sim::gradient::convergence::{
    energy_logger, target_energy_callback, BestTracker, ConvergenceMonitor, MonitorConfig,
    StepMetrics, StoppingCriterion, TrackedOptimizationResult,
};
use simq_sim::gradient::finite_difference::{
    auto_epsilon, compute_gradient_finite_difference, FiniteDifferenceConfig,
    FiniteDifferenceMethod,
};
use simq_sim::gradient::parameter_shift::{
    batch_parameter_shift, compute_gradient_parameter_shift, ParameterShiftConfig,
};
use simq_sim::gradient::{
    compute_gradient, compute_gradient_auto, GradientConfig, GradientMethod, GradientResult,
};
use simq_sim::{Simulator, SimulatorConfig};
use simq_state::observable::{Pauli, PauliObservable, PauliString};
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

fn zz_observable() -> PauliObservable {
    PauliObservable::from_pauli_string(PauliString::from_str("ZZ").unwrap(), 1.0)
}

fn ry_circuit(params: &[f64]) -> Circuit {
    let mut c = Circuit::new(1);
    c.add_gate(Arc::new(RotationY::new(params[0])), &[q(0)])
        .unwrap();
    c
}

fn ry_two_qubit(params: &[f64]) -> Circuit {
    let mut c = Circuit::new(2);
    c.add_gate(Arc::new(RotationY::new(params[0])), &[q(0)])
        .unwrap();
    c.add_gate(Arc::new(RotationY::new(params[1])), &[q(1)])
        .unwrap();
    c
}

// ============================================================================
// Finite difference gradient
// ============================================================================

#[test]
fn fd_central_single_param() {
    let sim = make_sim();
    let config = FiniteDifferenceConfig {
        method: FiniteDifferenceMethod::Central,
        epsilon: 1e-5,
        parallel: false,
    };
    let result =
        compute_gradient_finite_difference(&sim, ry_circuit, &z_observable(), &[0.5], &config)
            .unwrap();
    assert_eq!(result.gradients.len(), 1);
    assert_eq!(result.num_evaluations, 2);
    assert_eq!(result.method_used, GradientMethod::FiniteDifference);
    // d/dtheta cos(theta) = -sin(theta) at theta=0.5
    assert!((result.gradients[0] - (-0.5_f64.sin())).abs() < 1e-4);
}

#[test]
fn fd_forward_single_param() {
    let sim = make_sim();
    let config = FiniteDifferenceConfig {
        method: FiniteDifferenceMethod::Forward,
        epsilon: 1e-5,
        parallel: false,
    };
    let result =
        compute_gradient_finite_difference(&sim, ry_circuit, &z_observable(), &[0.5], &config)
            .unwrap();
    assert_eq!(result.num_evaluations, 2); // base + 1
    assert!((result.gradients[0] - (-0.5_f64.sin())).abs() < 1e-4);
}

#[test]
fn fd_backward_single_param() {
    let sim = make_sim();
    let config = FiniteDifferenceConfig {
        method: FiniteDifferenceMethod::Backward,
        epsilon: 1e-5,
        parallel: false,
    };
    let result =
        compute_gradient_finite_difference(&sim, ry_circuit, &z_observable(), &[0.5], &config)
            .unwrap();
    assert!((result.gradients[0] - (-0.5_f64.sin())).abs() < 1e-4);
}

#[test]
fn fd_empty_params() {
    let sim = make_sim();
    let config = FiniteDifferenceConfig::default();
    let cb = |_: &[f64]| {
        let mut c = Circuit::new(1);
        c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
        c
    };
    let result =
        compute_gradient_finite_difference(&sim, cb, &z_observable(), &[], &config).unwrap();
    assert!(result.gradients.is_empty());
    assert_eq!(result.num_evaluations, 0);
}

#[test]
fn fd_parallel_matches_sequential() {
    let sim = make_sim();
    let params = vec![0.3, 0.7];
    let seq = compute_gradient_finite_difference(
        &sim,
        ry_two_qubit,
        &zz_observable(),
        &params,
        &FiniteDifferenceConfig {
            method: FiniteDifferenceMethod::Central,
            epsilon: 1e-5,
            parallel: false,
        },
    )
    .unwrap();
    let par = compute_gradient_finite_difference(
        &sim,
        ry_two_qubit,
        &zz_observable(),
        &params,
        &FiniteDifferenceConfig {
            method: FiniteDifferenceMethod::Central,
            epsilon: 1e-5,
            parallel: true,
        },
    )
    .unwrap();
    for (s, p) in seq.gradients.iter().zip(par.gradients.iter()) {
        assert!((s - p).abs() < 1e-8);
    }
}

#[test]
fn auto_epsilon_scaling() {
    let params = vec![0.1, 1.0, 10.0, 100.0];
    let epsilons = auto_epsilon(&params, 1e-7);
    assert_eq!(epsilons[0], 1e-7);
    assert_eq!(epsilons[1], 1e-7);
    assert!((epsilons[2] - 1e-6).abs() < 1e-15);
    assert!((epsilons[3] - 1e-5).abs() < 1e-14);
}

// ============================================================================
// Parameter shift gradient
// ============================================================================

#[test]
fn ps_single_param() {
    let sim = make_sim();
    let config = ParameterShiftConfig::default();
    let result =
        compute_gradient_parameter_shift(&sim, ry_circuit, &z_observable(), &[0.5], &config)
            .unwrap();
    assert_eq!(result.gradients.len(), 1);
    assert_eq!(result.num_evaluations, 2);
    assert_eq!(result.method_used, GradientMethod::ParameterShift);
    assert!((result.gradients[0] - (-0.5_f64.sin())).abs() < 1e-6);
}

#[test]
fn ps_multi_params() {
    let sim = make_sim();
    let config = ParameterShiftConfig::default();
    let result = compute_gradient_parameter_shift(
        &sim,
        ry_two_qubit,
        &zz_observable(),
        &[0.3, 0.7],
        &config,
    )
    .unwrap();
    assert_eq!(result.gradients.len(), 2);
    assert_eq!(result.num_evaluations, 4);
}

#[test]
fn ps_empty_params() {
    let sim = make_sim();
    let config = ParameterShiftConfig::default();
    let cb = |_: &[f64]| {
        let mut c = Circuit::new(1);
        c.add_gate(Arc::new(Hadamard), &[q(0)]).unwrap();
        c
    };
    let result =
        compute_gradient_parameter_shift(&sim, cb, &z_observable(), &[], &config).unwrap();
    assert!(result.is_empty());
}

#[test]
fn ps_agrees_with_fd() {
    let sim = make_sim();
    let params = vec![0.5];
    let ps = compute_gradient_parameter_shift(
        &sim,
        ry_circuit,
        &z_observable(),
        &params,
        &ParameterShiftConfig::default(),
    )
    .unwrap();
    let fd = compute_gradient_finite_difference(
        &sim,
        ry_circuit,
        &z_observable(),
        &params,
        &FiniteDifferenceConfig {
            method: FiniteDifferenceMethod::Central,
            epsilon: 1e-7,
            parallel: false,
        },
    )
    .unwrap();
    assert!((ps.gradients[0] - fd.gradients[0]).abs() < 1e-5);
}

#[test]
fn batch_parameter_shift_multiple_observables() {
    let sim = make_sim();
    let config = ParameterShiftConfig {
        parallel: false,
        ..Default::default()
    };
    let obs1 = z_observable();
    let obs2 = PauliObservable::from_pauli_string(PauliString::from_str("Z").unwrap(), 2.0);
    let observables = vec![obs1, obs2];
    let results =
        batch_parameter_shift(&sim, ry_circuit, &observables, &[0.5], &config).unwrap();
    assert_eq!(results.len(), 2);
    // Second observable has 2x coefficient
    assert!((results[1].gradients[0] - 2.0 * results[0].gradients[0]).abs() < 1e-6);
}

// ============================================================================
// High-level gradient computation
// ============================================================================

#[test]
fn compute_gradient_parameter_shift_method() {
    let sim = make_sim();
    let config = GradientConfig {
        method: GradientMethod::ParameterShift,
        ..Default::default()
    };
    let result =
        compute_gradient(&sim, ry_circuit, &z_observable(), &[0.5], &config).unwrap();
    assert_eq!(result.method_used, GradientMethod::ParameterShift);
}

#[test]
fn compute_gradient_fd_method() {
    let sim = make_sim();
    let config = GradientConfig {
        method: GradientMethod::FiniteDifference,
        ..Default::default()
    };
    let result =
        compute_gradient(&sim, ry_circuit, &z_observable(), &[0.5], &config).unwrap();
    assert_eq!(result.method_used, GradientMethod::FiniteDifference);
}

#[test]
fn compute_gradient_auto_method() {
    let sim = make_sim();
    let result =
        compute_gradient_auto(&sim, ry_circuit, &z_observable(), &[0.5]).unwrap();
    assert!(!result.is_empty());
    assert!(result.norm() > 0.0);
}

#[test]
fn gradient_result_methods() {
    let gr = GradientResult {
        gradients: vec![3.0, 4.0],
        num_evaluations: 4,
        computation_time: Duration::from_millis(10),
        method_used: GradientMethod::ParameterShift,
    };
    assert_eq!(gr.len(), 2);
    assert!(!gr.is_empty());
    assert!((gr.norm() - 5.0).abs() < 1e-10);
    assert_eq!(gr.as_slice(), &[3.0, 4.0]);
}

// ============================================================================
// Batch evaluation
// ============================================================================

#[test]
fn batch_expectation_basic() {
    let sim = make_sim();
    let params = vec![vec![0.0], vec![std::f64::consts::PI]];
    let result =
        evaluate_batch_expectation(&sim, ry_circuit, &z_observable(), &params).unwrap();
    assert_eq!(result.values.len(), 2);
    assert_eq!(result.num_evaluations, 2);
    // RY(0)|0⟩ = |0⟩ → ⟨Z⟩ = 1
    assert!((result.values[0] - 1.0).abs() < 1e-6);
    // RY(π)|0⟩ = |1⟩ → ⟨Z⟩ = -1
    assert!((result.values[1] - (-1.0)).abs() < 1e-6);
}

#[test]
fn multi_observable_evaluation() {
    let sim = make_sim();
    let obs_z = z_observable();
    let mut obs_x = PauliObservable::new();
    obs_x.add_term(
        PauliString::from_paulis(vec![Pauli::X]),
        1.0,
    );
    let observables = vec![obs_z, obs_x];

    let cb = |params: &[f64]| {
        let mut c = Circuit::new(1);
        c.add_gate(Arc::new(RotationY::new(params[0])), &[q(0)])
            .unwrap();
        c
    };

    let values = evaluate_multi_observable(&sim, cb, &observables, &[0.0]).unwrap();
    assert_eq!(values.len(), 2);
    // |0⟩: ⟨Z⟩ = 1, ⟨X⟩ = 0
    assert!((values[0] - 1.0).abs() < 1e-6);
    assert!(values[1].abs() < 1e-6);
}

#[test]
fn grid_search_finds_minimum() {
    let sim = make_sim();
    let result = grid_search(
        &sim,
        ry_circuit,
        &z_observable(),
        &[(0.0, std::f64::consts::TAU)],
        10,
    )
    .unwrap();
    assert_eq!(result.param_grid.len(), 10);
    assert_eq!(result.values.len(), 10);
    // Minimum of cos(theta) is at theta=pi
    assert!((result.optimal_params[0] - std::f64::consts::PI).abs() < 1.0);
    assert!(result.optimal_value < -0.5);
}

// ============================================================================
// Advanced batch evaluation
// ============================================================================

#[test]
fn adaptive_batch_evaluator_basic() {
    let sim = make_sim();
    let config = BatchConfig::default();
    let mut evaluator = AdaptiveBatchEvaluator::new(config);
    let params = vec![vec![0.0], vec![1.0], vec![2.0]];
    let result = evaluator
        .evaluate(&sim, ry_circuit, &z_observable(), &params)
        .unwrap();
    assert_eq!(result.values.len(), 3);
    assert_eq!(result.num_evaluations, 3);
}

#[test]
fn adaptive_batch_empty_input() {
    let sim = make_sim();
    let config = BatchConfig::default();
    let mut evaluator = AdaptiveBatchEvaluator::new(config);
    let params: Vec<Vec<f64>> = vec![];
    let result = evaluator
        .evaluate(&sim, ry_circuit, &z_observable(), &params)
        .unwrap();
    assert!(result.values.is_empty());
    assert_eq!(result.num_evaluations, 0);
}

#[test]
fn latin_hypercube_sampling_correct_dims() {
    let ranges = vec![(0.0, 1.0), (-1.0, 1.0)];
    let samples = latin_hypercube_sampling(&ranges, 20);
    assert_eq!(samples.len(), 20);
    for s in &samples {
        assert_eq!(s.len(), 2);
        assert!(s[0] >= 0.0 && s[0] <= 1.0);
        assert!(s[1] >= -1.0 && s[1] <= 1.0);
    }
}

#[test]
fn latin_hypercube_3d() {
    let ranges = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
    let samples = latin_hypercube_sampling(&ranges, 50);
    assert_eq!(samples.len(), 50);
    assert_eq!(samples[0].len(), 3);
}

#[test]
fn importance_sampler_basic() {
    let centers = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
    let weights = vec![0.5, 0.5];
    let sampler = ImportanceSampler::new(centers, weights, 0.1);
    let samples = sampler.sample(100);
    assert_eq!(samples.len(), 100);
    for s in &samples {
        assert_eq!(s.len(), 2);
    }
}

#[test]
fn line_search_finds_better_point() {
    let sim = make_sim();
    let start = vec![0.0];
    let direction = vec![1.0];
    let step_sizes: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
    let (best_step, best_value) =
        line_search(&sim, ry_circuit, &z_observable(), &start, &direction, &step_sizes).unwrap();
    // cos(theta) minimized at theta=pi, so best_step should be near pi
    assert!(best_step > 2.0);
    assert!(best_value < 0.0);
}

#[test]
fn verify_gradients_agreement() {
    let sim = make_sim();
    let verification = verify_gradients(&sim, ry_circuit, &z_observable(), &[0.5]).unwrap();
    assert!(verification.agrees);
    assert!(verification.max_difference < 1e-5);
    assert_eq!(verification.parameter_shift.len(), 1);
    assert_eq!(verification.finite_difference.len(), 1);
}

// ============================================================================
// Dual number autodiff
// ============================================================================

#[test]
fn dual_constant() {
    let c = Dual::constant(5.0);
    assert_eq!(c.value(), 5.0);
    assert_eq!(c.derivative(), 0.0);
}

#[test]
fn dual_variable() {
    let x = Dual::variable(3.0);
    assert_eq!(x.value(), 3.0);
    assert_eq!(x.derivative(), 1.0);
}

#[test]
fn dual_add() {
    let x = Dual::variable(2.0);
    let y = Dual::constant(3.0);
    let z = x + y;
    assert_eq!(z.value(), 5.0);
    assert_eq!(z.derivative(), 1.0);
}

#[test]
fn dual_add_scalar() {
    let x = Dual::variable(2.0);
    let z = x + 5.0;
    assert_eq!(z.value(), 7.0);
    assert_eq!(z.derivative(), 1.0);
}

#[test]
fn dual_sub() {
    let x = Dual::variable(5.0);
    let y = Dual::constant(3.0);
    let z = x - y;
    assert_eq!(z.value(), 2.0);
    assert_eq!(z.derivative(), 1.0);
}

#[test]
fn dual_mul_product_rule() {
    let x = Dual::variable(3.0);
    let y = x * x;
    assert_eq!(y.value(), 9.0);
    assert_eq!(y.derivative(), 6.0);
}

#[test]
fn dual_mul_scalar() {
    let x = Dual::variable(4.0);
    let z = x * 3.0;
    assert_eq!(z.value(), 12.0);
    assert_eq!(z.derivative(), 3.0);
}

#[test]
fn dual_div() {
    let x = Dual::variable(6.0);
    let y = Dual::constant(3.0);
    let z = x / y;
    assert_eq!(z.value(), 2.0);
    assert!((z.derivative() - 1.0 / 3.0).abs() < 1e-10);
}

#[test]
fn dual_div_scalar() {
    let x = Dual::variable(6.0);
    let z = x / 2.0;
    assert_eq!(z.value(), 3.0);
    assert_eq!(z.derivative(), 0.5);
}

#[test]
fn dual_neg() {
    let x = Dual::variable(3.0);
    let z = -x;
    assert_eq!(z.value(), -3.0);
    assert_eq!(z.derivative(), -1.0);
}

#[test]
fn dual_sin() {
    let x = Dual::variable(0.0);
    let y = x.sin();
    assert!((y.value() - 0.0).abs() < 1e-12);
    assert!((y.derivative() - 1.0).abs() < 1e-12); // cos(0) = 1
}

#[test]
fn dual_cos() {
    let x = Dual::variable(0.0);
    let y = x.cos();
    assert!((y.value() - 1.0).abs() < 1e-12);
    assert!((y.derivative() - 0.0).abs() < 1e-12); // -sin(0) = 0
}

#[test]
fn dual_exp() {
    let x = Dual::variable(1.0);
    let y = x.exp();
    assert!((y.value() - std::f64::consts::E).abs() < 1e-10);
    assert!((y.derivative() - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn dual_ln() {
    let x = Dual::variable(std::f64::consts::E);
    let y = x.ln();
    assert!((y.value() - 1.0).abs() < 1e-10);
    assert!((y.derivative() - 1.0 / std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn dual_powi() {
    let x = Dual::variable(2.0);
    let y = x.powi(3);
    assert_eq!(y.value(), 8.0);
    assert_eq!(y.derivative(), 12.0);
}

#[test]
fn dual_sqrt() {
    let x = Dual::variable(4.0);
    let y = x.sqrt();
    assert_eq!(y.value(), 2.0);
    assert_eq!(y.derivative(), 0.25);
}

#[test]
fn dual_abs_positive() {
    let x = Dual::variable(3.0);
    let y = x.abs();
    assert_eq!(y.value(), 3.0);
    assert_eq!(y.derivative(), 1.0);
}

#[test]
fn dual_abs_negative() {
    let x = Dual::variable(-3.0);
    let y = x.abs();
    assert_eq!(y.value(), 3.0);
    assert_eq!(y.derivative(), -1.0);
}

#[test]
fn dual_display() {
    let x = Dual::new(2.0, 3.0);
    assert_eq!(format!("{}", x), "2 + 3ε");
}

#[test]
fn gradient_forward_linear() {
    // f(x, y) = 3x + 2y
    let f = |vars: &[Dual]| vars[0] * 3.0 + vars[1] * 2.0;
    let grad = gradient_forward(f, &[1.0, 2.0]);
    assert!((grad[0] - 3.0).abs() < 1e-10);
    assert!((grad[1] - 2.0).abs() < 1e-10);
}

#[test]
fn gradient_forward_quadratic() {
    let f = |vars: &[Dual]| vars[0] * vars[0] + vars[0] * vars[1] + vars[1] * vars[1];
    let grad = gradient_forward(f, &[2.0, 3.0]);
    assert!((grad[0] - 7.0).abs() < 1e-10);
    assert!((grad[1] - 8.0).abs() < 1e-10);
}

#[test]
fn gradient_forward_trig() {
    let f = |vars: &[Dual]| vars[0].sin() + vars[1].cos();
    let grad = gradient_forward(f, &[0.0, 0.0]);
    assert!((grad[0] - 1.0).abs() < 1e-10); // cos(0)
    assert!((grad[1] - 0.0).abs() < 1e-10); // -sin(0)
}

// ============================================================================
// HybridAD
// ============================================================================

#[test]
fn hybrid_ad_split_params() {
    let had = HybridAD::new(3, 2);
    let params = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (quantum, classical) = had.split_params(&params);
    assert_eq!(quantum, &[1.0, 2.0, 3.0]);
    assert_eq!(classical, &[4.0, 5.0]);
}

// ============================================================================
// Convergence monitoring
// ============================================================================

#[test]
fn convergence_monitor_tracks_best() {
    let config = MonitorConfig::default()
        .with_energy_tolerance(1e-10)
        .with_gradient_tolerance(1e-10);
    let mut monitor = ConvergenceMonitor::new(config);

    monitor.record(0, 1.0, &[0.5], &[0.1]);
    monitor.record(1, 0.5, &[0.3], &[0.2]);
    monitor.record(2, 0.7, &[0.2], &[0.3]);

    assert!((monitor.best_energy() - 0.5).abs() < 1e-10);
    assert_eq!(monitor.best_iteration(), 1);
    assert_eq!(monitor.best_parameters(), &[0.2]);
    assert_eq!(monitor.num_iterations(), 3);
}

#[test]
fn convergence_monitor_full_convergence() {
    let config = MonitorConfig::default()
        .with_energy_tolerance(1e-3)
        .with_gradient_tolerance(1e-3);
    let mut monitor = ConvergenceMonitor::new(config);

    monitor.record(0, 1.0, &[0.5], &[0.1]);
    monitor.record(1, 1.0005, &[1e-4], &[0.1]);

    assert!(monitor.should_stop());
    assert!(monitor.is_converged());
    assert_eq!(
        monitor.stopping_criterion(),
        StoppingCriterion::FullConvergence
    );
}

#[test]
fn convergence_monitor_energy_tolerance() {
    let config = MonitorConfig::default()
        .with_energy_tolerance(1e-3)
        .with_gradient_tolerance(1e-10);
    let mut monitor = ConvergenceMonitor::new(config);

    monitor.record(0, 1.0, &[0.5], &[0.1]);
    monitor.record(1, 1.0005, &[0.5], &[0.1]);

    assert!(monitor.should_stop());
    assert_eq!(
        monitor.stopping_criterion(),
        StoppingCriterion::EnergyTolerance
    );
}

#[test]
fn convergence_monitor_gradient_tolerance() {
    let config = MonitorConfig::default()
        .with_energy_tolerance(1e-10)
        .with_gradient_tolerance(1e-3);
    let mut monitor = ConvergenceMonitor::new(config);

    monitor.record(0, 1.0, &[0.5], &[0.1]);
    monitor.record(1, 0.5, &[1e-4], &[0.2]);

    assert!(monitor.should_stop());
    assert_eq!(
        monitor.stopping_criterion(),
        StoppingCriterion::GradientTolerance
    );
}

#[test]
fn convergence_monitor_patience() {
    let config = MonitorConfig::default()
        .with_patience(3)
        .with_energy_tolerance(1e-10)
        .with_gradient_tolerance(1e-10);
    let mut monitor = ConvergenceMonitor::new(config);

    monitor.record(0, 1.0, &[0.5], &[0.1]);
    for i in 1..=5 {
        monitor.record(i, 1.1 + i as f64 * 0.01, &[0.5], &[0.1]);
    }

    assert!(monitor.should_stop());
    assert_eq!(monitor.stopping_criterion(), StoppingCriterion::Patience);
}

#[test]
fn convergence_monitor_max_iterations() {
    let config = MonitorConfig::default()
        .with_max_iterations(2)
        .with_energy_tolerance(1e-10)
        .with_gradient_tolerance(1e-10);
    let mut monitor = ConvergenceMonitor::new(config);

    monitor.record(0, 1.0, &[0.5], &[0.1]);
    monitor.record(1, 0.5, &[0.3], &[0.2]);
    monitor.record(2, 0.3, &[0.2], &[0.3]);

    assert!(monitor.should_stop());
    assert_eq!(
        monitor.stopping_criterion(),
        StoppingCriterion::MaxIterations
    );
}

#[test]
fn convergence_monitor_energy_increase_stop() {
    let config = MonitorConfig::default()
        .with_energy_increase_stop(true)
        .with_energy_tolerance(1e-10)
        .with_gradient_tolerance(1e-10);
    let mut monitor = ConvergenceMonitor::new(config);

    monitor.record(0, 1.0, &[0.5], &[0.1]);
    monitor.record(1, 0.5, &[0.3], &[0.2]);
    // 3 consecutive increases
    monitor.record(2, 0.6, &[0.3], &[0.3]);
    monitor.record(3, 0.7, &[0.3], &[0.4]);
    monitor.record(4, 0.8, &[0.3], &[0.5]);

    assert!(monitor.should_stop());
    assert_eq!(
        monitor.stopping_criterion(),
        StoppingCriterion::EnergyIncrease
    );
}

#[test]
fn convergence_monitor_reset() {
    let config = MonitorConfig::default()
        .with_energy_tolerance(1e-3)
        .with_gradient_tolerance(1e-3);
    let mut monitor = ConvergenceMonitor::new(config);

    monitor.record(0, 1.0, &[0.5], &[0.1]);
    monitor.record(1, 1.0005, &[1e-4], &[0.1]);
    assert!(monitor.should_stop());

    monitor.reset();
    assert!(!monitor.should_stop());
    assert_eq!(monitor.num_iterations(), 0);
    assert!(monitor.best_energy().is_infinite());
}

#[test]
fn convergence_monitor_report() {
    let config = MonitorConfig::default()
        .with_energy_tolerance(1e-10)
        .with_gradient_tolerance(1e-10);
    let mut monitor = ConvergenceMonitor::new(config);

    monitor.record(0, 1.0, &[0.5], &[0.1]);
    monitor.record(1, 0.5, &[0.3], &[0.2]);

    let report = monitor.convergence_report();
    assert_eq!(report.num_iterations, 2);
    assert!((report.best_energy - 0.5).abs() < 1e-10);
    assert!((report.initial_energy - 1.0).abs() < 1e-10);
    assert!((report.final_energy - 0.5).abs() < 1e-10);
    let _ = report.summary();
}

#[test]
fn convergence_monitor_callback_user_stop() {
    let config = MonitorConfig::default()
        .with_energy_tolerance(1e-10)
        .with_gradient_tolerance(1e-10)
        .with_callback(|metrics: &StepMetrics| metrics.iteration < 2);

    let mut monitor = ConvergenceMonitor::new(config);
    monitor.record(0, 1.0, &[0.5], &[0.1]);
    monitor.record(1, 0.5, &[0.3], &[0.2]);
    assert!(!monitor.should_stop());

    monitor.record(2, 0.3, &[0.2], &[0.3]);
    assert!(monitor.should_stop());
    assert_eq!(monitor.stopping_criterion(), StoppingCriterion::UserStop);
}

#[test]
fn energy_logger_callback() {
    let (callback, energies) = energy_logger();
    let metrics = StepMetrics {
        iteration: 0,
        energy: 1.5,
        gradient: vec![0.1],
        gradient_norm: 0.1,
        parameters: vec![0.5],
        energy_change: 0.0,
        relative_energy_change: 0.0,
        parameter_change: 0.0,
        step_time: Duration::from_millis(1),
        total_time: Duration::from_millis(1),
    };
    assert!(callback(&metrics));
    let logged = energies.lock().unwrap();
    assert_eq!(logged.len(), 1);
    assert!((logged[0] - 1.5).abs() < 1e-10);
}

#[test]
fn target_energy_callback_stops_below_target() {
    let cb = target_energy_callback(-0.5);
    let metrics_above = StepMetrics {
        iteration: 0,
        energy: 0.5,
        gradient: vec![],
        gradient_norm: 0.0,
        parameters: vec![],
        energy_change: 0.0,
        relative_energy_change: 0.0,
        parameter_change: 0.0,
        step_time: Duration::ZERO,
        total_time: Duration::ZERO,
    };
    assert!(cb(&metrics_above)); // continue: 0.5 > -0.5

    let metrics_below = StepMetrics {
        energy: -1.0,
        ..metrics_above.clone()
    };
    assert!(!cb(&metrics_below)); // stop: -1.0 < -0.5
}

// ============================================================================
// BestTracker
// ============================================================================

#[test]
fn best_tracker_basic() {
    let mut tracker = BestTracker::new();
    assert!(!tracker.has_improved());
    assert!(tracker.best_energy().is_infinite());

    assert!(tracker.update(1.0, &[0.5]));
    assert!(tracker.has_improved());
    assert!((tracker.best_energy() - 1.0).abs() < 1e-10);

    assert!(tracker.update(0.5, &[0.3]));
    assert!((tracker.best_energy() - 0.5).abs() < 1e-10);
    assert_eq!(tracker.best_parameters(), &[0.3]);

    assert!(!tracker.update(0.8, &[0.4]));
    assert!((tracker.best_energy() - 0.5).abs() < 1e-10);
    assert_eq!(tracker.best_iteration(), 1);
    assert_eq!(tracker.num_iterations(), 3);
}

#[test]
fn best_tracker_with_history() {
    let mut tracker = BestTracker::with_history();
    tracker.update(1.0, &[0.5]);
    tracker.update(0.5, &[0.3]);
    tracker.update(0.8, &[0.4]);

    assert_eq!(tracker.history(), &[1.0, 0.5, 0.8]);
    assert!((tracker.total_improvement().unwrap() - 0.5).abs() < 1e-10);
}

#[test]
fn best_tracker_reset() {
    let mut tracker = BestTracker::with_history();
    tracker.update(1.0, &[0.5]);
    tracker.reset();
    assert!(!tracker.has_improved());
    assert!(tracker.history().is_empty());
    assert_eq!(tracker.num_iterations(), 0);
}

// ============================================================================
// TrackedOptimizationResult
// ============================================================================

#[test]
fn tracked_result_improvement() {
    let result = TrackedOptimizationResult {
        final_parameters: vec![0.5],
        final_energy: 0.3,
        best_parameters: vec![0.4],
        best_energy: 0.2,
        best_iteration: 5,
        num_iterations: 10,
        final_gradient: Some(vec![0.01]),
        stopping_criterion: StoppingCriterion::EnergyTolerance,
        total_time: Duration::from_millis(100),
        energy_history: vec![1.0, 0.8, 0.5, 0.3, 0.2, 0.2],
    };
    assert!(result.converged());
    assert!((result.improvement() - 0.8).abs() < 1e-10);
    assert!(result.relative_improvement() > 0.0);
}

#[test]
fn tracked_result_not_converged() {
    let result = TrackedOptimizationResult {
        final_parameters: vec![],
        final_energy: 0.5,
        best_parameters: vec![],
        best_energy: 0.5,
        best_iteration: 0,
        num_iterations: 100,
        final_gradient: None,
        stopping_criterion: StoppingCriterion::MaxIterations,
        total_time: Duration::from_millis(100),
        energy_history: vec![],
    };
    assert!(!result.converged());
    assert!((result.improvement() - 0.0).abs() < 1e-10);
}

// ============================================================================
// StoppingCriterion properties
// ============================================================================

#[test]
fn stopping_criterion_converged_variants() {
    assert!(StoppingCriterion::EnergyTolerance.is_converged());
    assert!(StoppingCriterion::GradientTolerance.is_converged());
    assert!(StoppingCriterion::FullConvergence.is_converged());
    assert!(!StoppingCriterion::MaxIterations.is_converged());
    assert!(!StoppingCriterion::NotConverged.is_converged());
    assert!(!StoppingCriterion::UserStop.is_converged());
}

#[test]
fn stopping_criterion_warning_variants() {
    assert!(StoppingCriterion::Patience.is_warning());
    assert!(StoppingCriterion::Oscillation.is_warning());
    assert!(StoppingCriterion::EnergyIncrease.is_warning());
    assert!(!StoppingCriterion::FullConvergence.is_warning());
    assert!(!StoppingCriterion::MaxIterations.is_warning());
}

#[test]
fn stopping_criterion_descriptions() {
    assert!(!StoppingCriterion::EnergyTolerance.description().is_empty());
    assert!(!StoppingCriterion::GradientTolerance.description().is_empty());
    assert!(!StoppingCriterion::FullConvergence.description().is_empty());
    assert!(!StoppingCriterion::Patience.description().is_empty());
    assert!(!StoppingCriterion::MaxIterations.description().is_empty());
    assert!(!StoppingCriterion::NotConverged.description().is_empty());
}

// ============================================================================
// Classical optimizer configs
// ============================================================================

#[test]
fn lbfgs_config_default() {
    let config = LBFGSConfig::default();
    assert_eq!(config.max_iterations, 100);
    assert_eq!(config.memory_size, 10);
    assert!(config.tolerance > 0.0);
    assert!(config.gradient_epsilon > 0.0);
}

#[test]
fn nelder_mead_config_default() {
    let config = NelderMeadConfig::default();
    assert_eq!(config.max_iterations, 200);
    assert!((config.alpha - 1.0).abs() < 1e-10);
    assert!((config.gamma - 2.0).abs() < 1e-10);
    assert!((config.rho - 0.5).abs() < 1e-10);
    assert!((config.sigma - 0.5).abs() < 1e-10);
}

// ============================================================================
// GradientConfig
// ============================================================================

#[test]
fn gradient_config_default() {
    let config = GradientConfig::default();
    assert_eq!(config.method, GradientMethod::ParameterShift);
    assert!((config.shift - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((config.epsilon - 1e-7).abs() < 1e-15);
    assert!(config.parallel);
    assert!(config.cache_circuits);
}

#[test]
fn batch_config_default() {
    let config = BatchConfig::default();
    assert_eq!(config.max_batch_size, 1000);
    assert!(config.adaptive_sizing);
    assert!(!config.enable_cache);
}
