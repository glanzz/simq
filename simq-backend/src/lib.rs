//! Hardware Backend Abstraction for SimQ
//!
//! This crate provides a unified interface for executing quantum circuits on
//! different backends, including:
//! - Local simulators (built-in)
//! - IBM Quantum (via Qiskit Runtime API)
//! - AWS Braket
//! - Azure Quantum
//! - Other cloud providers
//!
//! # Architecture
//!
//! The backend system uses a trait-based abstraction that allows seamless
//! switching between different execution targets while maintaining the same API.

pub mod backend;
pub mod backend_selector;
pub mod capabilities;
pub mod error;
pub mod gate_decomposition;
pub mod result;
pub mod routing;
pub mod transpiler;

#[cfg(feature = "local-simulator")]
pub mod local_simulator;

#[cfg(feature = "ibm-quantum")]
pub mod ibm_quantum;

pub use backend::{BackendType, QuantumBackend};
pub use backend_selector::{BackendFeature, BackendSelector, SelectionCriteria};
pub use capabilities::{BackendCapabilities, ConnectivityGraph, GateSet};
pub use error::{BackendError, Result};
pub use gate_decomposition::{
    analyze_gate_distribution, optimize_inverse_gates, optimize_merge_rotations, GateDecomposer,
};
pub use result::{BackendResult, ExecutionMetadata, JobStatus};
pub use routing::{Router, RoutingStats, RoutingStrategy, SabreRouter, SwapGate};
pub use transpiler::{
    DecompositionRule, DecompositionRules, OptimizationLevel, QubitMapping, SwapStrategy,
    TranspilationCost, Transpiler,
};

#[cfg(feature = "local-simulator")]
pub use local_simulator::{LocalSimulatorBackend, LocalSimulatorConfig};

#[cfg(feature = "ibm-quantum")]
pub use ibm_quantum::{IBMConfig, IBMQuantumBackend};
