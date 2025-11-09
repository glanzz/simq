//! Circuit serialization for caching and persistence
//!
//! This module provides serialization support for quantum circuits, enabling
//! efficient caching, persistence, and interoperability.

pub mod gate;
pub mod circuit;

#[cfg(feature = "cache")]
pub mod cache;

#[cfg(test)]
mod tests;

pub use gate::{SerializedGate, SerializedGateOp, GateRegistry, StandardGateRegistry};
pub use circuit::{SerializedCircuit, CircuitMetadata};

#[cfg(feature = "cache")]
pub use cache::{CircuitCache, CircuitKey, CacheStats, MemoryCache, FileCache};

/// Serialization format version
pub const CIRCUIT_FORMAT_VERSION: u32 = 1;

