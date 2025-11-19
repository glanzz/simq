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
pub mod capabilities;
pub mod result;
pub mod error;
pub mod transpiler;

#[cfg(feature = "local-simulator")]
pub mod local_simulator;

#[cfg(feature = "ibm-quantum")]
pub mod ibm_quantum;

pub use backend::{QuantumBackend, BackendType};
pub use capabilities::{BackendCapabilities, ConnectivityGraph, GateSet};
pub use result::{BackendResult, JobStatus, ExecutionMetadata};
pub use error::{BackendError, Result};
pub use transpiler::Transpiler;

#[cfg(feature = "local-simulator")]
pub use local_simulator::{LocalSimulatorBackend, LocalSimulatorConfig};

#[cfg(feature = "ibm-quantum")]
pub use ibm_quantum::{IBMQuantumBackend, IBMConfig};
