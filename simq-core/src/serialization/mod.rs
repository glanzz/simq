//! Circuit serialization for caching and persistence
//!
//! This module provides serialization support for quantum circuits, enabling
//! efficient caching, persistence, and interoperability.

pub mod circuit;
pub mod gate;

#[cfg(feature = "cache")]
pub mod cache;

#[cfg(test)]
mod tests;

pub use circuit::{CircuitMetadata, SerializedCircuit};
pub use gate::{GateRegistry, SerializedGate, SerializedGateOp, StandardGateRegistry};

#[cfg(feature = "cache")]
pub use cache::{CacheStats, CircuitCache, CircuitKey, FileCache, MemoryCache};

/// Serialization format version
pub const CIRCUIT_FORMAT_VERSION: u32 = 1;
