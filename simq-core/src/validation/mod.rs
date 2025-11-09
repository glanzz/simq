//! Circuit validation and DAG analysis
//!
//! This module provides comprehensive validation for quantum circuits, including
//! DAG consistency checking, cycle detection, and parallelism analysis.

pub mod dag;
pub mod rules;
pub mod report;

pub use dag::{DependencyGraph, DependencyEdge, GateNode, ParallelismAnalysis};
pub use rules::{ValidationRule, ValidationResult, ValidationError, ValidationWarning};
pub use report::ValidationReport;

