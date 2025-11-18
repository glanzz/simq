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

pub mod config;
pub mod error;
pub mod executor;
pub mod kernels;
pub mod parallel;
pub mod adaptive;
pub mod checkpoint;
pub mod validation;
pub mod telemetry;
pub mod recovery;
pub mod cache;

pub use config::{ExecutionConfig, ExecutionMode, ParallelStrategy};
pub use error::{ExecutionError, Result};
pub use executor::ExecutionEngine;
pub use telemetry::{ExecutionTelemetry, ExecutionMetrics};
pub use recovery::RecoveryPolicy;
pub use checkpoint::{Checkpoint, CheckpointManager};

/// Re-export legacy types for backward compatibility
pub use executor::ExecutionEngine as LegacyExecutionEngine;
