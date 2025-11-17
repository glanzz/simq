//! High-performance quantum circuit simulator
//!
//! This crate provides the core simulation engine for SimQ, implementing
//! efficient quantum state evolution with automatic sparse/dense representation
//! switching and parallel execution.
//!
//! # Features
//!
//! - **Hybrid state representation**: Automatically switches between sparse and dense
//! - **Parallel execution**: Multi-threaded gate application for large circuits
//! - **Integrated compilation**: Applies circuit optimization before execution
//! - **Flexible configuration**: Tunable thresholds and execution parameters
//! - **Rich telemetry**: Detailed execution statistics and profiling
//!
//! # Example
//!
//! ```ignore
//! use simq_sim::{Simulator, SimulatorConfig};
//! use simq_core::Circuit;
//!
//! // Create simulator with custom config
//! let config = SimulatorConfig {
//!     sparse_threshold: 0.1,
//!     parallel_threshold: 10,
//!     shots: 1024,
//!     ..Default::default()
//! };
//!
//! let simulator = Simulator::new(config);
//!
//! // Simulate circuit
//! let mut circuit = Circuit::new(3);
//! // ... add gates ...
//!
//! let result = simulator.run(&circuit)?;
//! println!("Final state: {:?}", result.state);
//! ```

pub mod config;
pub mod simulator;
pub mod gpu;
pub mod result;
pub mod error;
pub mod execution_engine;
pub mod statistics;

pub mod vqe_qaoa_helpers;

mod autodiff;

pub use vqe_qaoa_helpers::{
	vqe_gradient_parameter_shift,
	vqe_gradient_finite_difference,
	vqe_batch_expectation,
	qaoa_circuit,
	vqe_hardware_efficient_ansatz,
};

pub use config::SimulatorConfig;
pub use simulator::Simulator;
pub use result::{SimulationResult, MeasurementCounts};
pub use error::{SimulatorError, Result};
pub use statistics::ExecutionStatistics;

pub use autodiff::{
    DifferentiableParameter,
    compute_gradients_ad,
};
