//! Production-grade execution engine for quantum circuit simulation
//!
//! This module provides a comprehensive execution engine with:
//! - Complete gate application for all gate types
//! - Proper error handling with Result types
//! - Circuit-level parallelization
//! - Adaptive execution strategies
//! - GPU backend support
//! - Checkpointing and resumable execution
//! - Comprehensive validation and verification

pub mod adaptive;
pub mod cache;
pub mod checkpoint;
pub mod config;
pub mod error;
pub mod executor;
pub mod kernels;
pub mod parallel;
pub mod recovery;
pub mod telemetry;
pub mod validation;

pub use checkpoint::{Checkpoint, CheckpointManager};
pub use config::{ExecutionConfig, ExecutionMode, ParallelStrategy};
pub use error::{ExecutionError, Result};
pub use executor::ExecutionEngine;
pub use recovery::RecoveryPolicy;
pub use telemetry::{ExecutionMetrics, ExecutionTelemetry};

/// Re-export legacy types for backward compatibility
pub use executor::ExecutionEngine as LegacyExecutionEngine;
