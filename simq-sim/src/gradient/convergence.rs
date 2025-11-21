//! Convergence Monitoring for Variational Quantum Algorithms
//!
//! This module provides comprehensive tools for monitoring optimization
//! convergence in VQE/QAOA and other variational algorithms.
//!
//! # Features
//!
//! - **Real-time Metrics**: Track energy, gradients, parameters during optimization
//! - **Convergence Detection**: Multiple criteria for determining convergence
//! - **Early Stopping**: Prevent wasteful iterations when progress stalls
//! - **Diagnostics**: Identify optimization issues (barren plateaus, oscillations)
//! - **Callbacks**: User-defined actions triggered during optimization
//!
//! # Example
//!
//! ```ignore
//! use simq_sim::gradient::convergence::{ConvergenceMonitor, MonitorConfig, StoppingCriterion};
//!
//! let config = MonitorConfig::default()
//!     .with_energy_tolerance(1e-6)
//!     .with_gradient_tolerance(1e-6)
//!     .with_patience(10)
//!     .with_callback(|metrics| println!("Energy: {:.6}", metrics.energy));
//!
//! let mut monitor = ConvergenceMonitor::new(config);
//!
//! // During optimization loop:
//! monitor.record(iteration, energy, &gradient, &params);
//! if monitor.should_stop() {
//!     break;
//! }
//!
//! // After optimization:
//! let report = monitor.convergence_report();
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ============================================================================
// Core Types
// ============================================================================

/// Metrics recorded at each optimization step
#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// Iteration number
    pub iteration: usize,
    /// Current energy (cost function value)
    pub energy: f64,
    /// Gradient vector
    pub gradient: Vec<f64>,
    /// Gradient L2 norm
    pub gradient_norm: f64,
    /// Parameter values
    pub parameters: Vec<f64>,
    /// Energy change from previous step
    pub energy_change: f64,
    /// Relative energy change
    pub relative_energy_change: f64,
    /// Parameter change (L2 norm of delta)
    pub parameter_change: f64,
    /// Time for this step
    pub step_time: Duration,
    /// Cumulative time
    pub total_time: Duration,
}

/// Convergence criteria types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StoppingCriterion {
    /// Energy change below threshold
    EnergyTolerance,
    /// Gradient norm below threshold
    GradientTolerance,
    /// Both energy and gradient criteria met
    FullConvergence,
    /// No improvement for `patience` iterations
    Patience,
    /// Energy increased (potential overshoot)
    EnergyIncrease,
    /// Detected oscillation in energy
    Oscillation,
    /// Maximum iterations reached
    MaxIterations,
    /// Maximum time exceeded
    MaxTime,
    /// User-requested stop via callback
    UserStop,
    /// Not yet converged
    NotConverged,
}

impl StoppingCriterion {
    /// Check if this represents successful convergence
    pub fn is_converged(&self) -> bool {
        matches!(
            self,
            StoppingCriterion::EnergyTolerance
                | StoppingCriterion::GradientTolerance
                | StoppingCriterion::FullConvergence
        )
    }

    /// Check if this is a warning condition (not failure, but suboptimal)
    pub fn is_warning(&self) -> bool {
        matches!(
            self,
            StoppingCriterion::Patience
                | StoppingCriterion::Oscillation
                | StoppingCriterion::EnergyIncrease
        )
    }

    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            StoppingCriterion::EnergyTolerance => "Energy change below tolerance",
            StoppingCriterion::GradientTolerance => "Gradient norm below tolerance",
            StoppingCriterion::FullConvergence => "Both energy and gradient converged",
            StoppingCriterion::Patience => "No improvement for patience iterations",
            StoppingCriterion::EnergyIncrease => "Energy increased (potential overshoot)",
            StoppingCriterion::Oscillation => "Detected oscillation in energy",
            StoppingCriterion::MaxIterations => "Maximum iterations reached",
            StoppingCriterion::MaxTime => "Maximum time exceeded",
            StoppingCriterion::UserStop => "User requested stop",
            StoppingCriterion::NotConverged => "Not yet converged",
        }
    }
}

// ============================================================================
// Convergence Monitor Configuration
// ============================================================================

/// Callback function type for monitoring
pub type MonitorCallback = Box<dyn Fn(&StepMetrics) -> bool + Send + Sync>;

/// Configuration for convergence monitoring
pub struct MonitorConfig {
    /// Absolute energy change tolerance
    pub energy_tolerance: f64,
    /// Relative energy change tolerance
    pub relative_energy_tolerance: f64,
    /// Gradient norm tolerance
    pub gradient_tolerance: f64,
    /// Parameter change tolerance
    pub parameter_tolerance: f64,
    /// Number of iterations without improvement before stopping
    pub patience: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Maximum optimization time
    pub max_time: Option<Duration>,
    /// Window size for moving average calculations
    pub window_size: usize,
    /// Threshold for detecting oscillation (std dev / mean)
    pub oscillation_threshold: f64,
    /// Enable early stopping on energy increase
    pub stop_on_energy_increase: bool,
    /// Number of consecutive increases to trigger stop
    pub energy_increase_patience: usize,
    /// Callbacks to execute at each step (return false to stop)
    callbacks: Vec<MonitorCallback>,
    /// Enable detailed logging
    pub verbose: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            energy_tolerance: 1e-6,
            relative_energy_tolerance: 1e-8,
            gradient_tolerance: 1e-6,
            parameter_tolerance: 1e-8,
            patience: 20,
            max_iterations: 1000,
            max_time: None,
            window_size: 10,
            oscillation_threshold: 0.5,
            stop_on_energy_increase: false,
            energy_increase_patience: 3,
            callbacks: Vec::new(),
            verbose: false,
        }
    }
}

impl MonitorConfig {
    /// Create a new configuration with custom energy tolerance
    pub fn with_energy_tolerance(mut self, tol: f64) -> Self {
        self.energy_tolerance = tol;
        self
    }

    /// Set gradient tolerance
    pub fn with_gradient_tolerance(mut self, tol: f64) -> Self {
        self.gradient_tolerance = tol;
        self
    }

    /// Set patience (iterations without improvement)
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// Set maximum time
    pub fn with_max_time(mut self, duration: Duration) -> Self {
        self.max_time = Some(duration);
        self
    }

    /// Add a callback function
    pub fn with_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&StepMetrics) -> bool + Send + Sync + 'static,
    {
        self.callbacks.push(Box::new(callback));
        self
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Enable early stopping on energy increase
    pub fn with_energy_increase_stop(mut self, enabled: bool) -> Self {
        self.stop_on_energy_increase = enabled;
        self
    }
}

// ============================================================================
// Convergence Monitor
// ============================================================================

/// Main convergence monitor for tracking optimization progress
pub struct ConvergenceMonitor {
    config: MonitorConfig,
    /// Full history of metrics
    history: Vec<StepMetrics>,
    /// Recent energies for moving average
    recent_energies: VecDeque<f64>,
    /// Best energy found so far
    best_energy: f64,
    /// Iteration of best energy
    best_iteration: usize,
    /// Parameters at best energy
    best_parameters: Vec<f64>,
    /// Iterations since last improvement
    iterations_without_improvement: usize,
    /// Consecutive energy increases
    consecutive_increases: usize,
    /// Current stopping criterion
    stopping_criterion: StoppingCriterion,
    /// Start time
    start_time: Instant,
    /// Whether monitoring is active
    active: bool,
}

impl ConvergenceMonitor {
    /// Create a new convergence monitor
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            recent_energies: VecDeque::new(),
            best_energy: f64::INFINITY,
            best_iteration: 0,
            best_parameters: Vec::new(),
            iterations_without_improvement: 0,
            consecutive_increases: 0,
            stopping_criterion: StoppingCriterion::NotConverged,
            start_time: Instant::now(),
            active: true,
        }
    }

    /// Create with default configuration
    pub fn default_monitor() -> Self {
        Self::new(MonitorConfig::default())
    }

    /// Record metrics for a step
    pub fn record(
        &mut self,
        iteration: usize,
        energy: f64,
        gradient: &[f64],
        parameters: &[f64],
    ) {
        if !self.active {
            return;
        }

        let gradient_norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();

        // Compute changes from previous step
        let (energy_change, relative_energy_change, parameter_change) = if let Some(prev) =
            self.history.last()
        {
            let e_change = (energy - prev.energy).abs();
            let rel_change = if prev.energy.abs() > 1e-10 {
                e_change / prev.energy.abs()
            } else {
                e_change
            };
            let p_change: f64 = parameters
                .iter()
                .zip(prev.parameters.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            (e_change, rel_change, p_change)
        } else {
            (f64::INFINITY, f64::INFINITY, f64::INFINITY)
        };

        let now = Instant::now();
        let step_time = if let Some(prev) = self.history.last() {
            now.duration_since(self.start_time) - prev.total_time
        } else {
            now.duration_since(self.start_time)
        };

        let metrics = StepMetrics {
            iteration,
            energy,
            gradient: gradient.to_vec(),
            gradient_norm,
            parameters: parameters.to_vec(),
            energy_change,
            relative_energy_change,
            parameter_change,
            step_time,
            total_time: now.duration_since(self.start_time),
        };

        // Update best tracking
        if energy < self.best_energy {
            self.best_energy = energy;
            self.best_iteration = iteration;
            self.best_parameters = parameters.to_vec();
            self.iterations_without_improvement = 0;
            self.consecutive_increases = 0;
        } else {
            self.iterations_without_improvement += 1;
            if self.history.last().map(|m| energy > m.energy).unwrap_or(false) {
                self.consecutive_increases += 1;
            } else {
                self.consecutive_increases = 0;
            }
        }

        // Update recent energies for moving average
        self.recent_energies.push_back(energy);
        if self.recent_energies.len() > self.config.window_size {
            self.recent_energies.pop_front();
        }

        // Execute callbacks
        for callback in &self.config.callbacks {
            if !callback(&metrics) {
                self.stopping_criterion = StoppingCriterion::UserStop;
                self.active = false;
            }
        }

        // Check stopping criteria
        self.check_stopping_criteria(&metrics);

        // Verbose logging
        if self.config.verbose {
            self.log_step(&metrics);
        }

        self.history.push(metrics);
    }

    /// Check all stopping criteria
    fn check_stopping_criteria(&mut self, metrics: &StepMetrics) {
        // Already stopped?
        if self.stopping_criterion != StoppingCriterion::NotConverged {
            return;
        }

        // Max iterations
        if metrics.iteration >= self.config.max_iterations {
            self.stopping_criterion = StoppingCriterion::MaxIterations;
            return;
        }

        // Max time
        if let Some(max_time) = self.config.max_time {
            if metrics.total_time >= max_time {
                self.stopping_criterion = StoppingCriterion::MaxTime;
                return;
            }
        }

        // Full convergence (both criteria)
        let energy_converged = metrics.energy_change < self.config.energy_tolerance;
        let gradient_converged = metrics.gradient_norm < self.config.gradient_tolerance;

        if energy_converged && gradient_converged {
            self.stopping_criterion = StoppingCriterion::FullConvergence;
            return;
        }

        // Individual criteria
        if energy_converged {
            self.stopping_criterion = StoppingCriterion::EnergyTolerance;
            return;
        }

        if gradient_converged {
            self.stopping_criterion = StoppingCriterion::GradientTolerance;
            return;
        }

        // Patience
        if self.iterations_without_improvement >= self.config.patience {
            self.stopping_criterion = StoppingCriterion::Patience;
            return;
        }

        // Energy increase
        if self.config.stop_on_energy_increase
            && self.consecutive_increases >= self.config.energy_increase_patience
        {
            self.stopping_criterion = StoppingCriterion::EnergyIncrease;
            return;
        }

        // Oscillation detection
        if self.detect_oscillation() {
            self.stopping_criterion = StoppingCriterion::Oscillation;
        }
    }

    /// Detect oscillation in energy values
    fn detect_oscillation(&self) -> bool {
        if self.recent_energies.len() < self.config.window_size {
            return false;
        }

        let energies: Vec<f64> = self.recent_energies.iter().copied().collect();
        let mean = energies.iter().sum::<f64>() / energies.len() as f64;
        let variance =
            energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / energies.len() as f64;
        let std_dev = variance.sqrt();

        // Check for alternating increases and decreases
        let mut sign_changes = 0;
        for i in 2..energies.len() {
            let prev_diff = energies[i - 1] - energies[i - 2];
            let curr_diff = energies[i] - energies[i - 1];
            if prev_diff * curr_diff < 0.0 {
                sign_changes += 1;
            }
        }

        // Oscillation if high coefficient of variation and many sign changes
        let cv = if mean.abs() > 1e-10 {
            std_dev / mean.abs()
        } else {
            std_dev
        };

        cv > self.config.oscillation_threshold
            && sign_changes as f64 > 0.5 * energies.len() as f64
    }

    /// Log a step (when verbose)
    fn log_step(&self, metrics: &StepMetrics) {
        println!(
            "[Iter {:4}] Energy: {:12.8} | ΔE: {:10.2e} | |∇|: {:10.2e} | Time: {:?}",
            metrics.iteration,
            metrics.energy,
            metrics.energy_change,
            metrics.gradient_norm,
            metrics.step_time
        );
    }

    /// Check if optimization should stop
    pub fn should_stop(&self) -> bool {
        self.stopping_criterion != StoppingCriterion::NotConverged
    }

    /// Get the stopping criterion
    pub fn stopping_criterion(&self) -> StoppingCriterion {
        self.stopping_criterion
    }

    /// Check if converged successfully
    pub fn is_converged(&self) -> bool {
        self.stopping_criterion.is_converged()
    }

    /// Get best energy found
    pub fn best_energy(&self) -> f64 {
        self.best_energy
    }

    /// Get parameters at best energy
    pub fn best_parameters(&self) -> &[f64] {
        &self.best_parameters
    }

    /// Get iteration of best energy
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }

    /// Get total iterations recorded
    pub fn num_iterations(&self) -> usize {
        self.history.len()
    }

    /// Get total time elapsed
    pub fn total_time(&self) -> Duration {
        self.history
            .last()
            .map(|m| m.total_time)
            .unwrap_or_default()
    }

    /// Get full history
    pub fn history(&self) -> &[StepMetrics] {
        &self.history
    }

    /// Generate a convergence report
    pub fn convergence_report(&self) -> ConvergenceReport {
        let energies: Vec<f64> = self.history.iter().map(|m| m.energy).collect();
        let gradients: Vec<f64> = self.history.iter().map(|m| m.gradient_norm).collect();

        // Compute statistics
        let energy_improvement = if !energies.is_empty() {
            energies[0] - self.best_energy
        } else {
            0.0
        };

        let final_gradient_norm = self
            .history
            .last()
            .map(|m| m.gradient_norm)
            .unwrap_or(f64::NAN);

        let avg_step_time = if !self.history.is_empty() {
            self.total_time().as_secs_f64() / self.history.len() as f64
        } else {
            0.0
        };

        // Detect issues
        let mut warnings = Vec::new();
        let mut diagnostics = Vec::new();

        // Check for barren plateau (very small gradients throughout)
        if gradients.iter().all(|&g| g < 1e-8) {
            warnings.push("Possible barren plateau detected (gradients near zero throughout)");
        }

        // Check for slow convergence
        if self.history.len() > 50 && energy_improvement.abs() < self.config.energy_tolerance * 10.0
        {
            warnings.push("Very slow convergence detected");
        }

        // Check for oscillation history
        if self.stopping_criterion == StoppingCriterion::Oscillation {
            warnings.push("Optimization stopped due to oscillation - consider reducing learning rate");
        }

        // Add diagnostics
        if let Some(last) = self.history.last() {
            diagnostics.push(format!("Final energy: {:.8}", last.energy));
            diagnostics.push(format!("Final gradient norm: {:.2e}", last.gradient_norm));
        }
        diagnostics.push(format!("Best energy at iteration: {}", self.best_iteration));
        diagnostics.push(format!(
            "Iterations without improvement: {}",
            self.iterations_without_improvement
        ));

        ConvergenceReport {
            converged: self.is_converged(),
            stopping_criterion: self.stopping_criterion,
            num_iterations: self.history.len(),
            total_time: self.total_time(),
            initial_energy: energies.first().copied().unwrap_or(f64::NAN),
            final_energy: energies.last().copied().unwrap_or(f64::NAN),
            best_energy: self.best_energy,
            best_iteration: self.best_iteration,
            energy_improvement,
            final_gradient_norm,
            avg_step_time,
            warnings: warnings.iter().map(|s| s.to_string()).collect(),
            diagnostics,
        }
    }

    /// Reset the monitor for a new optimization run
    pub fn reset(&mut self) {
        self.history.clear();
        self.recent_energies.clear();
        self.best_energy = f64::INFINITY;
        self.best_iteration = 0;
        self.best_parameters.clear();
        self.iterations_without_improvement = 0;
        self.consecutive_increases = 0;
        self.stopping_criterion = StoppingCriterion::NotConverged;
        self.start_time = Instant::now();
        self.active = true;
    }
}

// ============================================================================
// Convergence Report
// ============================================================================

/// Detailed report on optimization convergence
#[derive(Debug, Clone)]
pub struct ConvergenceReport {
    /// Whether optimization converged successfully
    pub converged: bool,
    /// Final stopping criterion
    pub stopping_criterion: StoppingCriterion,
    /// Total iterations
    pub num_iterations: usize,
    /// Total time
    pub total_time: Duration,
    /// Initial energy
    pub initial_energy: f64,
    /// Final energy
    pub final_energy: f64,
    /// Best energy found
    pub best_energy: f64,
    /// Iteration of best energy
    pub best_iteration: usize,
    /// Total energy improvement
    pub energy_improvement: f64,
    /// Final gradient norm
    pub final_gradient_norm: f64,
    /// Average time per step
    pub avg_step_time: f64,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Diagnostic information
    pub diagnostics: Vec<String>,
}

impl ConvergenceReport {
    /// Print a formatted report
    pub fn print(&self) {
        println!("\n{:=<60}", "");
        println!("  CONVERGENCE REPORT");
        println!("{:=<60}", "");

        println!("\nStatus: {}", if self.converged { "CONVERGED" } else { "NOT CONVERGED" });
        println!("Stopping criterion: {}", self.stopping_criterion.description());

        println!("\n--- Performance ---");
        println!("Total iterations:    {}", self.num_iterations);
        println!("Total time:          {:?}", self.total_time);
        println!("Avg time per step:   {:.2} ms", self.avg_step_time * 1000.0);

        println!("\n--- Energy ---");
        println!("Initial energy:      {:.8}", self.initial_energy);
        println!("Final energy:        {:.8}", self.final_energy);
        println!("Best energy:         {:.8} (iter {})", self.best_energy, self.best_iteration);
        println!("Improvement:         {:.8}", self.energy_improvement);

        println!("\n--- Gradients ---");
        println!("Final gradient norm: {:.2e}", self.final_gradient_norm);

        if !self.warnings.is_empty() {
            println!("\n--- Warnings ---");
            for warning in &self.warnings {
                println!("⚠️  {}", warning);
            }
        }

        if !self.diagnostics.is_empty() {
            println!("\n--- Diagnostics ---");
            for diag in &self.diagnostics {
                println!("   {}", diag);
            }
        }

        println!("\n{:=<60}\n", "");
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "{} after {} iterations ({:?}): energy {:.8} -> {:.8}",
            if self.converged { "Converged" } else { "Stopped" },
            self.num_iterations,
            self.total_time,
            self.initial_energy,
            self.best_energy
        )
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Simple progress printer callback
pub fn progress_callback(interval: usize) -> MonitorCallback {
    Box::new(move |metrics: &StepMetrics| {
        if metrics.iteration % interval == 0 {
            println!(
                "Iteration {:4}: energy = {:.8}, |grad| = {:.2e}",
                metrics.iteration, metrics.energy, metrics.gradient_norm
            );
        }
        true // Continue optimization
    })
}

/// Energy logging callback
pub fn energy_logger() -> (MonitorCallback, std::sync::Arc<std::sync::Mutex<Vec<f64>>>) {
    let energies = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let energies_clone = energies.clone();

    let callback: MonitorCallback = Box::new(move |metrics: &StepMetrics| {
        if let Ok(mut vec) = energies_clone.lock() {
            vec.push(metrics.energy);
        }
        true
    });

    (callback, energies)
}

/// Create a callback that stops when energy reaches a target
pub fn target_energy_callback(target: f64) -> MonitorCallback {
    Box::new(move |metrics: &StepMetrics| {
        metrics.energy > target // Continue if above target
    })
}

// ============================================================================
// Simple Best Tracker
// ============================================================================

/// Lightweight tracker for best energy and parameters
///
/// Use this when you just need to track the best solution without
/// full convergence monitoring overhead.
///
/// # Example
///
/// ```ignore
/// let mut tracker = BestTracker::new();
///
/// for iteration in 0..100 {
///     let energy = compute_energy(&params);
///     tracker.update(energy, &params);
///     // ... update params ...
/// }
///
/// println!("Best energy: {}", tracker.best_energy());
/// let best_params = tracker.best_parameters();
/// ```
#[derive(Debug, Clone)]
pub struct BestTracker {
    best_energy: f64,
    best_parameters: Vec<f64>,
    best_iteration: usize,
    current_iteration: usize,
    history: Vec<f64>,
    track_history: bool,
}

impl BestTracker {
    /// Create a new best tracker
    pub fn new() -> Self {
        Self {
            best_energy: f64::INFINITY,
            best_parameters: Vec::new(),
            best_iteration: 0,
            current_iteration: 0,
            history: Vec::new(),
            track_history: false,
        }
    }

    /// Create with history tracking enabled
    pub fn with_history() -> Self {
        Self {
            track_history: true,
            ..Self::new()
        }
    }

    /// Update with new energy and parameters
    ///
    /// Returns `true` if this is a new best, `false` otherwise
    pub fn update(&mut self, energy: f64, parameters: &[f64]) -> bool {
        if self.track_history {
            self.history.push(energy);
        }

        let is_new_best = energy < self.best_energy;
        if is_new_best {
            self.best_energy = energy;
            self.best_parameters = parameters.to_vec();
            self.best_iteration = self.current_iteration;
        }

        self.current_iteration += 1;
        is_new_best
    }

    /// Get the best energy found
    pub fn best_energy(&self) -> f64 {
        self.best_energy
    }

    /// Get the parameters at best energy
    pub fn best_parameters(&self) -> &[f64] {
        &self.best_parameters
    }

    /// Get the iteration where best was found
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }

    /// Get total iterations recorded
    pub fn num_iterations(&self) -> usize {
        self.current_iteration
    }

    /// Check if any improvement has been found
    pub fn has_improved(&self) -> bool {
        self.best_energy < f64::INFINITY
    }

    /// Get energy history (if tracking enabled)
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Get improvement from first to best
    pub fn total_improvement(&self) -> Option<f64> {
        self.history.first().map(|first| first - self.best_energy)
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.best_energy = f64::INFINITY;
        self.best_parameters.clear();
        self.best_iteration = 0;
        self.current_iteration = 0;
        self.history.clear();
    }
}

impl Default for BestTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Optimization Result with Best Tracking
// ============================================================================

/// Extended optimization result that guarantees best parameters are returned
#[derive(Debug, Clone)]
pub struct TrackedOptimizationResult {
    /// Final parameters (may not be best)
    pub final_parameters: Vec<f64>,
    /// Final energy
    pub final_energy: f64,
    /// Best parameters found during optimization
    pub best_parameters: Vec<f64>,
    /// Best energy found
    pub best_energy: f64,
    /// Iteration where best was found
    pub best_iteration: usize,
    /// Total iterations
    pub num_iterations: usize,
    /// Final gradient (if available)
    pub final_gradient: Option<Vec<f64>>,
    /// Convergence status
    pub stopping_criterion: StoppingCriterion,
    /// Total time
    pub total_time: Duration,
    /// Energy history
    pub energy_history: Vec<f64>,
}

impl TrackedOptimizationResult {
    /// Check if converged
    pub fn converged(&self) -> bool {
        self.stopping_criterion.is_converged()
    }

    /// Get the improvement from initial to best
    pub fn improvement(&self) -> f64 {
        if let Some(&initial) = self.energy_history.first() {
            initial - self.best_energy
        } else {
            0.0
        }
    }

    /// Get relative improvement
    pub fn relative_improvement(&self) -> f64 {
        if let Some(&initial) = self.energy_history.first() {
            if initial.abs() > 1e-10 {
                (initial - self.best_energy) / initial.abs()
            } else {
                self.improvement()
            }
        } else {
            0.0
        }
    }

    /// Print a summary
    pub fn print_summary(&self) {
        println!("Optimization Result:");
        println!("  Status:          {:?}", self.stopping_criterion);
        println!("  Iterations:      {}", self.num_iterations);
        println!("  Time:            {:?}", self.total_time);
        println!("  Best energy:     {:.8} (iter {})", self.best_energy, self.best_iteration);
        println!("  Final energy:    {:.8}", self.final_energy);
        println!("  Improvement:     {:.8} ({:.2}%)",
            self.improvement(),
            self.relative_improvement() * 100.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_best_tracker() {
        let mut tracker = BestTracker::new();

        tracker.update(1.0, &[0.5]);
        assert!((tracker.best_energy() - 1.0).abs() < 1e-10);

        tracker.update(0.5, &[0.3]);
        assert!((tracker.best_energy() - 0.5).abs() < 1e-10);

        // Worse energy shouldn't update best
        tracker.update(0.8, &[0.4]);
        assert!((tracker.best_energy() - 0.5).abs() < 1e-10);
        assert_eq!(tracker.best_iteration(), 1);
    }

    #[test]
    fn test_best_tracker_with_history() {
        let mut tracker = BestTracker::with_history();

        tracker.update(1.0, &[0.5]);
        tracker.update(0.5, &[0.3]);
        tracker.update(0.8, &[0.4]);

        assert_eq!(tracker.history().len(), 3);
        assert!((tracker.total_improvement().unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_monitor_basic() {
        let config = MonitorConfig::default();
        let mut monitor = ConvergenceMonitor::new(config);

        // Simulate optimization
        let params = vec![0.5, 0.3];
        let gradient = vec![0.1, 0.2];

        monitor.record(0, 1.0, &gradient, &params);
        assert!(!monitor.should_stop());
        assert_eq!(monitor.num_iterations(), 1);
    }

    #[test]
    fn test_convergence_detection() {
        let config = MonitorConfig::default()
            .with_energy_tolerance(1e-4)
            .with_gradient_tolerance(1e-4);

        let mut monitor = ConvergenceMonitor::new(config);

        // First step
        monitor.record(0, 1.0, &[0.1, 0.2], &[0.5, 0.3]);

        // Converged step (small change, small gradient)
        monitor.record(1, 0.9999, &[1e-5, 1e-5], &[0.5, 0.3]);

        assert!(monitor.should_stop());
        assert!(monitor.is_converged());
    }

    #[test]
    fn test_patience() {
        let config = MonitorConfig::default().with_patience(3);

        let mut monitor = ConvergenceMonitor::new(config);

        // Initial step
        monitor.record(0, 1.0, &[0.1], &[0.5]);

        // No improvement for several steps
        for i in 1..=5 {
            monitor.record(i, 1.1, &[0.1], &[0.5]);
        }

        assert!(monitor.should_stop());
        assert_eq!(monitor.stopping_criterion(), StoppingCriterion::Patience);
    }

    #[test]
    fn test_report_generation() {
        let config = MonitorConfig::default();
        let mut monitor = ConvergenceMonitor::new(config);

        monitor.record(0, 1.0, &[0.1], &[0.5]);
        monitor.record(1, 0.5, &[0.05], &[0.4]);

        let report = monitor.convergence_report();
        assert_eq!(report.num_iterations, 2);
        assert!((report.best_energy - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_stopping_criterion_properties() {
        assert!(StoppingCriterion::FullConvergence.is_converged());
        assert!(StoppingCriterion::EnergyTolerance.is_converged());
        assert!(!StoppingCriterion::MaxIterations.is_converged());
        assert!(StoppingCriterion::Patience.is_warning());
    }
}
