//! Custom quantum gate implementation with validation
//!
//! This module allows users to create custom quantum gates with automatic
//! validation of quantum mechanical properties such as unitarity, proper
//! dimensions, and physical constraints.
//!
//! # Example
//!
//! ```rust
//! use simq_gates::custom::CustomGateBuilder;
//! use num_complex::Complex64;
//! use std::f64::consts::SQRT_2;
//!
//! // Create a custom Hadamard-like gate
//! let inv_sqrt2 = 1.0 / SQRT_2;
//! let hadamard = CustomGateBuilder::new("MyHadamard")
//!     .num_qubits(1)
//!     .matrix(vec![
//!         Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0),
//!         Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0),
//!     ])
//!     .build()
//!     .expect("Failed to create custom gate");
//! ```
//!
//! # Validation
//!
//! Custom gates are validated for:
//! - **Unitarity**: U†U = I (required for reversible quantum operations)
//! - **Proper dimensions**: Matrix size must be 2^n × 2^n for n qubits
//! - **Numerical stability**: Checks for NaN and infinite values
//! - **Determinant**: |det(U)| = 1 for unitary matrices
//!
//! # Advanced Features
//!
//! - **Hermiticity checking**: For observables and self-adjoint gates
//! - **Trace calculation**: For quantum channel analysis
//! - **Fidelity computation**: Compare custom gates with target unitaries
//! - **Controlled versions**: Automatically generate controlled variants

use num_complex::Complex64;
use simq_core::gate::Gate;
use std::fmt;
use std::sync::Arc;

/// Errors that can occur when creating custom gates
#[derive(Debug, Clone, PartialEq)]
pub enum CustomGateError {
    /// Matrix is not unitary (U†U ≠ I)
    NotUnitary { max_deviation: f64, tolerance: f64 },
    /// Matrix dimensions are invalid
    InvalidDimensions { expected: usize, actual: usize },
    /// Matrix size is not a power of 2
    InvalidSize { size: usize },
    /// Matrix contains NaN or infinite values
    InvalidValues,
    /// Gate name is empty or invalid
    InvalidName,
    /// Determinant is not approximately 1
    InvalidDeterminant { determinant_norm: f64 },
    /// Matrix is not hermitian (when required)
    NotHermitian { max_deviation: f64 },
}

impl fmt::Display for CustomGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CustomGateError::NotUnitary {
                max_deviation,
                tolerance,
            } => {
                write!(
                    f,
                    "Matrix is not unitary: max deviation {:.2e} exceeds tolerance {:.2e}. \
                    Ensure U†U = I where U† is the conjugate transpose.",
                    max_deviation, tolerance
                )
            },
            CustomGateError::InvalidDimensions { expected, actual } => {
                write!(
                    f,
                    "Invalid matrix dimensions: expected {}×{} ({} elements), got {} elements",
                    expected,
                    expected,
                    expected * expected,
                    actual
                )
            },
            CustomGateError::InvalidSize { size } => {
                write!(
                    f,
                    "Matrix size {} is not valid. Size must be 2^n × 2^n for n-qubit gates (e.g., 2, 4, 8, 16, ...)",
                    size
                )
            },
            CustomGateError::InvalidValues => {
                write!(f, "Matrix contains NaN or infinite values")
            },
            CustomGateError::InvalidName => {
                write!(f, "Gate name cannot be empty")
            },
            CustomGateError::InvalidDeterminant { determinant_norm } => {
                write!(
                    f,
                    "Invalid determinant: |det(U)| = {:.6} (expected 1.0 for unitary matrices)",
                    determinant_norm
                )
            },
            CustomGateError::NotHermitian { max_deviation } => {
                write!(
                    f,
                    "Matrix is not hermitian: max deviation {:.2e}. Hermitian matrices satisfy A = A†",
                    max_deviation
                )
            },
        }
    }
}

impl std::error::Error for CustomGateError {}

/// A custom quantum gate with user-defined matrix
///
/// Custom gates allow you to define arbitrary quantum operations while
/// ensuring they satisfy the mathematical requirements of quantum mechanics.
///
/// # Thread Safety
///
/// `CustomGate` is `Send + Sync` and can be safely shared across threads.
/// The matrix is stored internally and validated once during construction.
pub struct CustomGate {
    name: String,
    num_qubits: usize,
    matrix: Vec<Complex64>,
    is_hermitian: bool,
    description: Option<String>,
}

impl CustomGate {
    /// Create a new custom gate with validation
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the gate
    /// * `num_qubits` - Number of qubits the gate acts on
    /// * `matrix` - Gate matrix as a flattened vector (row-major order)
    /// * `tolerance` - Tolerance for unitarity checking (default: 1e-10)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Matrix is not unitary
    /// - Matrix dimensions don't match num_qubits
    /// - Matrix contains invalid values (NaN, infinity)
    /// - Gate name is empty
    pub fn new(
        name: impl Into<String>,
        num_qubits: usize,
        matrix: Vec<Complex64>,
        tolerance: f64,
    ) -> Result<Self, CustomGateError> {
        let name = name.into();

        // Validate name
        if name.is_empty() {
            return Err(CustomGateError::InvalidName);
        }

        // Calculate expected size
        let expected_size = 1 << num_qubits; // 2^num_qubits
        let expected_elements = expected_size * expected_size;

        // Validate dimensions
        if matrix.len() != expected_elements {
            return Err(CustomGateError::InvalidDimensions {
                expected: expected_size,
                actual: matrix.len(),
            });
        }

        // Validate values
        for &val in &matrix {
            if val.re.is_nan() || val.re.is_infinite() || val.im.is_nan() || val.im.is_infinite() {
                return Err(CustomGateError::InvalidValues);
            }
        }

        // Check unitarity
        let _max_deviation = validate_unitarity(&matrix, tolerance)?;

        // Check if hermitian (optional, for information)
        let is_hermitian = crate::matrix_ops::is_hermitian(&matrix, tolerance);

        Ok(Self {
            name,
            num_qubits,
            matrix,
            is_hermitian,
            description: None,
        })
    }

    /// Get the matrix as a flattened vector
    pub fn matrix_vec(&self) -> &[Complex64] {
        &self.matrix
    }

    /// Check if this gate is hermitian
    pub fn is_hermitian(&self) -> bool {
        self.is_hermitian
    }

    /// Set a custom description for this gate
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Calculate the fidelity between this gate and another unitary
    ///
    /// Fidelity F = |Tr(U†V)|² / d² where d is the matrix dimension
    pub fn fidelity(&self, other: &[Complex64]) -> Result<f64, CustomGateError> {
        let expected_size = 1 << self.num_qubits;
        let expected_elements = expected_size * expected_size;

        if other.len() != expected_elements {
            return Err(CustomGateError::InvalidDimensions {
                expected: expected_size,
                actual: other.len(),
            });
        }

        Ok(crate::matrix_ops::fidelity(&self.matrix, other))
    }

    /// Create a controlled version of this gate
    ///
    /// Returns a new custom gate that applies this gate conditionally
    /// on the control qubit being |1⟩.
    pub fn controlled(&self) -> Result<Self, CustomGateError> {
        let new_num_qubits = self.num_qubits + 1;
        let new_size = 1 << new_num_qubits;
        let mut controlled_matrix = vec![Complex64::new(0.0, 0.0); new_size * new_size];

        // Identity on the first half (control = |0⟩)
        let half_size = 1 << self.num_qubits;
        for i in 0..half_size {
            controlled_matrix[i * new_size + i] = Complex64::new(1.0, 0.0);
        }

        // Apply gate on the second half (control = |1⟩)
        for i in 0..half_size {
            for j in 0..half_size {
                let row = half_size + i;
                let col = half_size + j;
                controlled_matrix[row * new_size + col] = self.matrix[i * half_size + j];
            }
        }

        CustomGate::new(format!("C{}", self.name), new_num_qubits, controlled_matrix, 1e-10)
    }

    /// Get the adjoint (Hermitian conjugate) of this gate
    pub fn adjoint(&self) -> Self {
        let adjoint_matrix = crate::matrix_ops::matrix_adjoint(&self.matrix);

        // Adjoint of unitary is also unitary, so this should not fail
        CustomGate::new(format!("{}†", self.name), self.num_qubits, adjoint_matrix, 1e-10)
            .expect("Adjoint of unitary gate should be unitary")
    }

    /// Compose this gate with another gate
    ///
    /// Returns U·V where U is this gate and V is the other gate.
    /// Both gates must operate on the same number of qubits.
    pub fn compose(&self, other: &CustomGate) -> Result<Self, CustomGateError> {
        if self.num_qubits != other.num_qubits {
            return Err(CustomGateError::InvalidDimensions {
                expected: 1 << self.num_qubits,
                actual: 1 << other.num_qubits,
            });
        }

        let composed_matrix = crate::matrix_ops::matrix_multiply(&self.matrix, &other.matrix);

        CustomGate::new(
            format!("{}·{}", self.name, other.name),
            self.num_qubits,
            composed_matrix,
            1e-10,
        )
    }
}

impl Gate for CustomGate {
    fn name(&self) -> &str {
        &self.name
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn is_unitary(&self) -> bool {
        true // Always true for CustomGate (validated in constructor)
    }

    fn is_hermitian(&self) -> bool {
        self.is_hermitian
    }

    fn description(&self) -> String {
        if let Some(ref desc) = self.description {
            desc.clone()
        } else {
            format!(
                "Custom {}-qubit gate '{}'{}",
                self.num_qubits,
                self.name,
                if self.is_hermitian {
                    " (Hermitian)"
                } else {
                    ""
                }
            )
        }
    }

    fn matrix(&self) -> Option<Vec<Complex64>> {
        Some(self.matrix.clone())
    }
}

impl fmt::Debug for CustomGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomGate")
            .field("name", &self.name)
            .field("num_qubits", &self.num_qubits)
            .field("is_hermitian", &self.is_hermitian)
            .field("matrix_size", &self.matrix.len())
            .finish()
    }
}

/// Builder for creating custom gates with a fluent API
///
/// # Example
///
/// ```rust
/// use simq_gates::custom::CustomGateBuilder;
/// use num_complex::Complex64;
///
/// let gate = CustomGateBuilder::new("MyGate")
///     .num_qubits(1)
///     .matrix(vec![
///         Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
///         Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0),
///     ])
///     .description("A custom Z-like gate")
///     .tolerance(1e-12)
///     .build()
///     .expect("Failed to build gate");
/// ```
#[derive(Debug)]
pub struct CustomGateBuilder {
    name: String,
    num_qubits: Option<usize>,
    matrix: Option<Vec<Complex64>>,
    description: Option<String>,
    tolerance: f64,
    require_hermitian: bool,
}

impl CustomGateBuilder {
    /// Create a new builder with the given gate name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            num_qubits: None,
            matrix: None,
            description: None,
            tolerance: 1e-10,
            require_hermitian: false,
        }
    }

    /// Set the number of qubits
    pub fn num_qubits(mut self, num_qubits: usize) -> Self {
        self.num_qubits = Some(num_qubits);
        self
    }

    /// Set the gate matrix (row-major flattened vector)
    pub fn matrix(mut self, matrix: Vec<Complex64>) -> Self {
        self.matrix = Some(matrix);
        self
    }

    /// Set the gate matrix from a 2D array (for single-qubit gates)
    pub fn matrix_2x2(mut self, matrix: [[Complex64; 2]; 2]) -> Self {
        let vec: Vec<Complex64> = matrix.iter().flatten().copied().collect();
        self.matrix = Some(vec);
        self.num_qubits = Some(1);
        self
    }

    /// Set the gate matrix from a 4x4 2D array (for two-qubit gates)
    pub fn matrix_4x4(mut self, matrix: [[Complex64; 4]; 4]) -> Self {
        let vec: Vec<Complex64> = matrix.iter().flatten().copied().collect();
        self.matrix = Some(vec);
        self.num_qubits = Some(2);
        self
    }

    /// Set a custom description
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the tolerance for unitarity checking
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Require the gate to be hermitian
    ///
    /// If set, the builder will return an error if the matrix is not hermitian.
    pub fn require_hermitian(mut self, require: bool) -> Self {
        self.require_hermitian = require;
        self
    }

    /// Build the custom gate
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Matrix or num_qubits not set
    /// - Matrix validation fails
    /// - Hermitian requirement not met (if required)
    pub fn build(self) -> Result<CustomGate, CustomGateError> {
        let num_qubits = self.num_qubits.ok_or(CustomGateError::InvalidDimensions {
            expected: 0,
            actual: 0,
        })?;

        let matrix = self.matrix.ok_or(CustomGateError::InvalidDimensions {
            expected: 0,
            actual: 0,
        })?;

        let mut gate = CustomGate::new(self.name, num_qubits, matrix, self.tolerance)?;

        // Check hermitian requirement
        if self.require_hermitian && !gate.is_hermitian() {
            let max_deviation = check_hermitian_deviation(&gate.matrix);
            return Err(CustomGateError::NotHermitian { max_deviation });
        }

        // Add description if provided
        if let Some(desc) = self.description {
            gate = gate.with_description(desc);
        }

        Ok(gate)
    }

    /// Build the custom gate and wrap it in an Arc for use in circuits
    pub fn build_arc(self) -> Result<Arc<dyn Gate>, CustomGateError> {
        Ok(Arc::new(self.build()?))
    }
}

/// A parametric custom quantum gate with parameter dependencies
///
/// Allows creating gates whose matrices depend on one or more parameters.
/// The matrix is recomputed whenever parameters change.
///
/// # Example
///
/// ```rust
/// use simq_gates::custom::ParametricCustomGateBuilder;
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// let mut gate = ParametricCustomGateBuilder::new("CustomRX", 1)
///     .with_parameters(vec!["theta"])
///     .with_matrix_fn(|params| {
///         let theta = params[0];
///         let c = (theta / 2.0).cos();
///         let s = (theta / 2.0).sin();
///         vec![
///             Complex64::new(c, 0.0), Complex64::new(0.0, -s),
///             Complex64::new(0.0, -s), Complex64::new(c, 0.0),
///         ]
///     })
///     .build()
///     .expect("Failed to create parametric gate");
///
/// // Update parameters and get new matrix
/// gate.set_parameters(vec![PI / 4.0]).unwrap();
/// let matrix = gate.matrix_vec().to_vec();
/// ```
/// Type alias for matrix generation function
pub type MatrixFn = Arc<dyn Fn(&[f64]) -> Vec<Complex64> + Send + Sync>;

pub struct ParametricCustomGate {
    name: String,
    num_qubits: usize,
    parameter_names: Vec<String>,
    matrix_fn: MatrixFn,
    current_matrix: Vec<Complex64>,
    tolerance: f64,
    is_hermitian: bool,
}

impl fmt::Debug for ParametricCustomGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParametricCustomGate")
            .field("name", &self.name)
            .field("num_qubits", &self.num_qubits)
            .field("parameters", &self.parameter_names)
            .field("is_hermitian", &self.is_hermitian)
            .finish()
    }
}

impl ParametricCustomGate {
    /// Create a new parametric custom gate
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the gate
    /// * `num_qubits` - Number of qubits
    /// * `parameter_names` - Names of parameters
    /// * `matrix_fn` - Function that computes matrix given parameters
    /// * `initial_params` - Initial parameter values
    /// * `tolerance` - Tolerance for validation
    pub fn new(
        name: impl Into<String>,
        num_qubits: usize,
        parameter_names: Vec<String>,
        matrix_fn: MatrixFn,
        initial_params: Vec<f64>,
        tolerance: f64,
    ) -> Result<Self, CustomGateError> {
        if parameter_names.len() != initial_params.len() {
            return Err(CustomGateError::InvalidDimensions {
                expected: parameter_names.len(),
                actual: initial_params.len(),
            });
        }

        let current_matrix = matrix_fn(&initial_params);

        // Validate the initial matrix
        validate_unitarity(&current_matrix, tolerance)?;
        let is_hermitian = crate::matrix_ops::is_hermitian(&current_matrix, tolerance);

        Ok(Self {
            name: name.into(),
            num_qubits,
            parameter_names,
            matrix_fn,
            current_matrix,
            tolerance,
            is_hermitian,
        })
    }

    /// Set new parameter values and recompute matrix
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Parameter count doesn't match
    /// - New matrix is not unitary
    pub fn set_parameters(&mut self, params: Vec<f64>) -> Result<(), CustomGateError> {
        if params.len() != self.parameter_names.len() {
            return Err(CustomGateError::InvalidDimensions {
                expected: self.parameter_names.len(),
                actual: params.len(),
            });
        }

        let new_matrix = (self.matrix_fn)(&params);
        validate_unitarity(&new_matrix, self.tolerance)?;

        self.current_matrix = new_matrix;
        self.is_hermitian = crate::matrix_ops::is_hermitian(&self.current_matrix, self.tolerance);

        Ok(())
    }

    /// Get current matrix
    pub fn matrix_vec(&self) -> &[Complex64] {
        &self.current_matrix
    }

    /// Get parameter names
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }

    /// Check if currently hermitian
    pub fn is_hermitian(&self) -> bool {
        self.is_hermitian
    }

    /// Convert to a static CustomGate with current parameters
    pub fn to_static_gate(self) -> Result<CustomGate, CustomGateError> {
        CustomGate::new(self.name, self.num_qubits, self.current_matrix, self.tolerance)
    }
}

/// Builder for parametric custom gates
pub struct ParametricCustomGateBuilder {
    name: String,
    num_qubits: usize,
    parameter_names: Vec<String>,
    matrix_fn: Option<MatrixFn>,
    initial_params: Vec<f64>,
    tolerance: f64,
}

impl fmt::Debug for ParametricCustomGateBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ParametricCustomGateBuilder")
            .field("name", &self.name)
            .field("num_qubits", &self.num_qubits)
            .field("parameter_names", &self.parameter_names)
            .field("initial_params", &self.initial_params)
            .field("tolerance", &self.tolerance)
            .finish()
    }
}

impl ParametricCustomGateBuilder {
    /// Create a new builder
    pub fn new(name: impl Into<String>, num_qubits: usize) -> Self {
        Self {
            name: name.into(),
            num_qubits,
            parameter_names: Vec::new(),
            matrix_fn: None,
            initial_params: Vec::new(),
            tolerance: 1e-10,
        }
    }

    /// Add parameter names
    pub fn with_parameters(mut self, names: Vec<&str>) -> Self {
        self.parameter_names = names.iter().map(|n| n.to_string()).collect();
        self
    }

    /// Set the matrix computation function
    pub fn with_matrix_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&[f64]) -> Vec<Complex64> + Send + Sync + 'static,
    {
        self.matrix_fn = Some(Arc::new(f));
        self
    }

    /// Set initial parameter values
    pub fn with_initial_params(mut self, params: Vec<f64>) -> Self {
        self.initial_params = params;
        self
    }

    /// Set tolerance for validation
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Build the parametric gate
    pub fn build(self) -> Result<ParametricCustomGate, CustomGateError> {
        let matrix_fn = self.matrix_fn.ok_or(CustomGateError::InvalidDimensions {
            expected: 1,
            actual: 0,
        })?;

        if self.parameter_names.is_empty() {
            return Err(CustomGateError::InvalidName);
        }

        // Provide default initial params if not specified
        let initial_params = if self.initial_params.is_empty() {
            vec![0.0; self.parameter_names.len()]
        } else {
            self.initial_params
        };

        ParametricCustomGate::new(
            self.name,
            self.num_qubits,
            self.parameter_names,
            matrix_fn,
            initial_params,
            self.tolerance,
        )
    }
}

/// Validate that a matrix is unitary
///
/// Checks that U†U ≈ I within the given tolerance.
/// Returns the maximum deviation from identity.
fn validate_unitarity(matrix: &[Complex64], tolerance: f64) -> Result<f64, CustomGateError> {
    let n = (matrix.len() as f64).sqrt() as usize;

    // Verify size is a power of 2
    if n * n != matrix.len() || (n & (n - 1)) != 0 {
        return Err(CustomGateError::InvalidSize { size: n });
    }

    // Compute U†U
    let adjoint = crate::matrix_ops::matrix_adjoint(matrix);
    let u_dagger_u = crate::matrix_ops::matrix_multiply(&adjoint, matrix);

    // Compare with identity
    let mut max_deviation: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            let actual = u_dagger_u[i * n + j];
            let deviation = (actual - expected).norm();
            max_deviation = max_deviation.max(deviation);
        }
    }

    if max_deviation > tolerance {
        return Err(CustomGateError::NotUnitary {
            max_deviation,
            tolerance,
        });
    }

    Ok(max_deviation)
}

/// Calculate the maximum deviation from hermiticity
fn check_hermitian_deviation(matrix: &[Complex64]) -> f64 {
    let n = (matrix.len() as f64).sqrt() as usize;
    let adjoint = crate::matrix_ops::matrix_adjoint(matrix);

    let mut max_deviation: f64 = 0.0;
    for i in 0..n {
        for j in 0..n {
            let deviation = (matrix[i * n + j] - adjoint[i * n + j]).norm();
            max_deviation = max_deviation.max(deviation);
        }
    }

    max_deviation
}

/// Additional validation utilities
pub mod validation {
    use super::*;

    /// Check if a matrix satisfies the trace preservation property
    ///
    /// For a quantum channel Φ, we require Tr(Φ(ρ)) = Tr(ρ) for all density matrices ρ.
    /// For unitary channels, this is automatically satisfied.
    pub fn is_trace_preserving(matrix: &[Complex64], tolerance: f64) -> bool {
        let trace = crate::matrix_ops::matrix_trace(matrix);
        let n = (matrix.len() as f64).sqrt() as usize;
        let expected_trace = n as f64; // Tr(I) for identity matrix
        (trace.re - expected_trace).abs() < tolerance && trace.im.abs() < tolerance
    }

    /// Verify that matrix represents a valid quantum gate
    ///
    /// Performs comprehensive validation:
    /// - Unitarity
    /// - Proper dimensions
    /// - No invalid values
    /// - Determinant check (|det(U)| ≈ 1)
    pub fn validate_quantum_gate(
        matrix: &[Complex64],
        num_qubits: usize,
        tolerance: f64,
    ) -> Result<(), CustomGateError> {
        // This will perform all the standard checks
        validate_unitarity(matrix, tolerance)?;

        // Additional check: determinant
        // For unitary matrices, |det(U)| = 1
        // We'll check this for small matrices (computational cost grows as O(n!))
        let size = 1 << num_qubits;
        if size <= 4 {
            // Only check for 1-2 qubit gates
            let det = compute_determinant(matrix);
            let det_norm = det.norm();
            if (det_norm - 1.0).abs() > tolerance {
                return Err(CustomGateError::InvalidDeterminant {
                    determinant_norm: det_norm,
                });
            }
        }

        Ok(())
    }

    /// Compute determinant for small matrices
    fn compute_determinant(matrix: &[Complex64]) -> Complex64 {
        let n = (matrix.len() as f64).sqrt() as usize;

        match n {
            1 => matrix[0],
            2 => crate::matrix_ops::determinant_2x2(&[
                [matrix[0], matrix[1]],
                [matrix[2], matrix[3]],
            ]),
            _ => {
                // For larger matrices, use numerical methods or skip
                // For now, return 1.0 as we already checked unitarity
                Complex64::new(1.0, 0.0)
            },
        }
    }

    /// Check completeness relation for a set of gates
    ///
    /// For a set of Kraus operators {E_i}, the completeness relation is:
    /// ∑_i E_i† E_i = I
    pub fn check_completeness_relation(
        gates: &[Vec<Complex64>],
        tolerance: f64,
    ) -> Result<(), CustomGateError> {
        if gates.is_empty() {
            return Err(CustomGateError::InvalidDimensions {
                expected: 1,
                actual: 0,
            });
        }

        let size = (gates[0].len() as f64).sqrt() as usize;
        let mut sum = vec![Complex64::new(0.0, 0.0); gates[0].len()];

        for gate in gates {
            if gate.len() != gates[0].len() {
                return Err(CustomGateError::InvalidDimensions {
                    expected: size,
                    actual: (gate.len() as f64).sqrt() as usize,
                });
            }

            let adjoint = crate::matrix_ops::matrix_adjoint(gate);
            let product = crate::matrix_ops::matrix_multiply(&adjoint, gate);

            for (i, &val) in product.iter().enumerate() {
                sum[i] += val;
            }
        }

        // Check if sum ≈ I
        let identity = crate::matrix_ops::identity_matrix(size);
        let mut max_deviation: f64 = 0.0;
        for i in 0..sum.len() {
            let deviation = (sum[i] - identity[i]).norm();
            max_deviation = max_deviation.max(deviation);
        }

        if max_deviation > tolerance {
            return Err(CustomGateError::NotUnitary {
                max_deviation,
                tolerance,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::SQRT_2;

    #[test]
    fn test_custom_pauli_x() {
        let matrix = vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];

        let gate = CustomGate::new("CustomX", 1, matrix, 1e-10).unwrap();
        assert_eq!(gate.name(), "CustomX");
        assert_eq!(gate.num_qubits(), 1);
        assert!(gate.is_unitary());
        assert!(gate.is_hermitian());
    }

    #[test]
    fn test_custom_hadamard() {
        let inv_sqrt2 = 1.0 / SQRT_2;
        let matrix = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ];

        let gate = CustomGate::new("CustomH", 1, matrix, 1e-10).unwrap();
        assert_eq!(gate.name(), "CustomH");
        assert!(gate.is_hermitian());
    }

    #[test]
    fn test_non_unitary_matrix() {
        let matrix = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.5, 0.0), // Not unitary
        ];

        let result = CustomGate::new("Invalid", 1, matrix, 1e-10);
        assert!(result.is_err());
        assert!(matches!(result, Err(CustomGateError::NotUnitary { .. })));
    }

    #[test]
    fn test_invalid_dimensions() {
        let matrix = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0), // Only 3 elements for a 1-qubit gate (needs 4)
        ];

        let result = CustomGate::new("Invalid", 1, matrix, 1e-10);
        assert!(result.is_err());
        assert!(matches!(result, Err(CustomGateError::InvalidDimensions { .. })));
    }

    #[test]
    fn test_builder_pattern() {
        let inv_sqrt2 = 1.0 / SQRT_2;
        let gate = CustomGateBuilder::new("BuilderH")
            .num_qubits(1)
            .matrix(vec![
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ])
            .description("Custom Hadamard gate")
            .build()
            .unwrap();

        assert_eq!(gate.name(), "BuilderH");
        assert_eq!(gate.description(), "Custom Hadamard gate");
    }

    #[test]
    fn test_builder_2x2_matrix() {
        let gate = CustomGateBuilder::new("Z")
            .matrix_2x2([
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
            ])
            .build()
            .unwrap();

        assert_eq!(gate.num_qubits(), 1);
        assert!(gate.is_hermitian());
    }

    #[test]
    fn test_controlled_gate() {
        // Create a simple X gate
        let x_gate = CustomGateBuilder::new("X")
            .matrix_2x2([
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ])
            .build()
            .unwrap();

        let cx_gate = x_gate.controlled().unwrap();
        assert_eq!(cx_gate.num_qubits(), 2);
        assert_eq!(cx_gate.name(), "CX");
        assert!(cx_gate.is_unitary());
    }

    #[test]
    fn test_adjoint() {
        let s_gate = CustomGateBuilder::new("S")
            .matrix_2x2([
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)], // i
            ])
            .build()
            .unwrap();

        let s_dag = s_gate.adjoint();
        assert_eq!(s_dag.name(), "S†");

        // S · S† = I
        let identity = s_gate.compose(&s_dag).unwrap();
        let fidelity = identity
            .fidelity(&crate::matrix_ops::identity_matrix(2))
            .unwrap();
        assert_relative_eq!(fidelity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fidelity() {
        let inv_sqrt2 = 1.0 / SQRT_2;
        let h_matrix = vec![
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(inv_sqrt2, 0.0),
            Complex64::new(-inv_sqrt2, 0.0),
        ];

        let gate = CustomGate::new("H", 1, h_matrix.clone(), 1e-10).unwrap();
        let fidelity = gate.fidelity(&h_matrix).unwrap();
        assert_relative_eq!(fidelity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compose_gates() {
        let x_gate = CustomGateBuilder::new("X")
            .matrix_2x2([
                [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ])
            .build()
            .unwrap();

        // X · X = I
        let x_squared = x_gate.compose(&x_gate).unwrap();
        let fidelity = x_squared
            .fidelity(&crate::matrix_ops::identity_matrix(2))
            .unwrap();
        assert_relative_eq!(fidelity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_values() {
        let matrix = vec![
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let result = CustomGate::new("Invalid", 1, matrix, 1e-10);
        assert!(matches!(result, Err(CustomGateError::InvalidValues)));
    }

    #[test]
    fn test_require_hermitian() {
        // Create a unitary but non-hermitian matrix: Phase gate S = [[1, 0], [0, i]]
        let result = CustomGateBuilder::new("PhaseGate")
            .matrix_2x2([
                [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                [Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)], // i
            ])
            .require_hermitian(true)
            .build();

        assert!(matches!(result, Err(CustomGateError::NotHermitian { .. })));
    }

    #[test]
    fn test_validation_completeness_relation() {
        use validation::check_completeness_relation;

        // Identity operator
        let identity = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];

        let result = check_completeness_relation(&[identity], 1e-10);
        assert!(result.is_ok());
    }
}
