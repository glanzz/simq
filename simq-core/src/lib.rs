//! Core types and traits for SimQ quantum computing SDK
//!
//! This crate provides the foundational types and abstractions for building
//! quantum circuits and programs.
//!
//! # Core Components
//!
//! - **Type System**: Type-safe quantum operations with compile-time validation
//! - **Circuit Building**: Ergonomic APIs for constructing quantum circuits
//! - **Gate Abstraction**: Extensible trait system for quantum gates
//! - **Error Handling**: Comprehensive error types with helpful messages
//!
//! # Quick Start
//!
//! ## Type-Safe Circuit Builder (Compile-time size)
//!
//! ```
//! use simq_core::{CircuitBuilder, Gate};
//! use std::sync::Arc;
//!
//! # #[derive(Debug)]
//! # struct HGate;
//! # impl Gate for HGate {
//! #     fn name(&self) -> &str { "H" }
//! #     fn num_qubits(&self) -> usize { 1 }
//! # }
//! // Create a 3-qubit circuit with compile-time size checking
//! let mut builder = CircuitBuilder::<3>::new();
//! let [q0, q1, q2] = builder.qubits();
//!
//! let h_gate = Arc::new(HGate);
//! builder.apply_gate(h_gate, &[q0]).unwrap();
//!
//! let circuit = builder.build();
//! assert_eq!(circuit.num_qubits(), 3);
//! ```
//!
//! ## Dynamic Circuit Builder (Runtime size)
//!
//! ```
//! use simq_core::{DynamicCircuitBuilder, Gate};
//! use std::sync::Arc;
//!
//! # #[derive(Debug)]
//! # struct HGate;
//! # impl Gate for HGate {
//! #     fn name(&self) -> &str { "H" }
//! #     fn num_qubits(&self) -> usize { 1 }
//! # }
//! // Create a circuit with runtime-determined size
//! let num_qubits = 5; // From config, user input, etc.
//! let mut builder = DynamicCircuitBuilder::new(num_qubits);
//!
//! let h_gate = Arc::new(HGate);
//! builder.apply_gate(h_gate, &[0]).unwrap();
//!
//! let circuit = builder.build();
//! assert_eq!(circuit.num_qubits(), 5);
//! ```

pub mod ascii_renderer;
pub mod circuit;
pub mod circuit_builder;
pub mod dynamic_builder;
pub mod error;
pub mod gate;
pub mod noise;
pub mod parameter;
pub mod parameter_id;
pub mod parameter_registry;
pub mod qubit;
pub mod qubit_ref;
pub mod validation;

#[cfg(feature = "serialization")]
pub mod serialization;

// Re-exports for convenience
pub use ascii_renderer::{render as render_ascii, render_with_config as render_ascii_with_config, AsciiConfig};
pub use circuit::Circuit;
pub use circuit_builder::CircuitBuilder;
pub use dynamic_builder::DynamicCircuitBuilder;
pub use error::QuantumError;
pub use gate::{Gate, GateOp};
pub use noise::{
    AmplitudeDamping, AmplitudeDampingMC, CrosstalkProperties, DepolarizingChannel, DepolarizingMC,
    GateNoise, GateTiming, HardwareNoiseModel, KrausOperator, MonteCarloSampler, NoiseChannel,
    NoiseModel, PauliOperation, PhaseDamping, PhaseDampingMC, QubitProperties, QubitTimeTracker,
    ReadoutError, ReadoutErrorMC, TwoQubitGateProperties,
};
pub use num_complex::Complex64;
pub use parameter::Parameter;
pub use parameter_id::ParameterId;
pub use parameter_registry::ParameterRegistry;
pub use qubit::QubitId;
pub use qubit_ref::Qubit;
pub use validation::{
    DependencyGraph, ParallelismAnalysis, ValidationReport, ValidationResult, ValidationRule,
};

/// Type alias for results in SimQ
pub type Result<T> = std::result::Result<T, QuantumError>;
