//! Quantum noise models and error channels
//!
//! This module provides implementations of common noise channels that model
//! real quantum hardware imperfections:
//!
//! - **Depolarizing noise**: Random Pauli errors
//! - **Amplitude damping**: Energy relaxation (T1 decay)
//! - **Phase damping**: Dephasing errors (T2 decay)
//! - **Readout errors**: Measurement bit-flip errors
//!
//! # Physical Motivation
//!
//! Real quantum computers suffer from various error sources:
//! - Gate imperfections introduce random errors
//! - Qubits lose energy to the environment (T1 relaxation)
//! - Quantum coherence decays over time (T2 dephasing)
//! - Measurement devices have finite fidelity
//!
//! # Usage
//!
//! ```ignore
//! use simq_core::noise::{DepolarizingChannel, AmplitudeDamping};
//!
//! // 1% depolarizing error after each gate
//! let depol = DepolarizingChannel::new(0.01)?;
//!
//! // T1 = 50μs amplitude damping
//! let t1_damping = AmplitudeDamping::new(0.02)?;
//! ```

pub mod channels;
pub mod types;

pub use channels::{
    AmplitudeDamping, DepolarizingChannel, PhaseDamping, ReadoutError,
};
pub use types::{KrausOperator, NoiseChannel, NoiseModel};
