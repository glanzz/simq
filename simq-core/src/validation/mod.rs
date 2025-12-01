//! Circuit validation and DAG analysis
//!
//! This module provides comprehensive validation for quantum circuits, including
//! DAG consistency checking, cycle detection, and parallelism analysis.

pub mod dag;
pub mod report;
pub mod rules;

pub use dag::{DependencyEdge, DependencyGraph, GateNode, ParallelismAnalysis};
pub use report::ValidationReport;
pub use rules::{ValidationError, ValidationResult, ValidationRule, ValidationWarning};
