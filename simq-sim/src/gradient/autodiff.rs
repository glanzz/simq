//! Automatic Differentiation for Quantum Circuits
//!
//! This module provides automatic differentiation (AD) capabilities for gradient
//! computation. While the parameter shift rule is quantum-native and exact, AD
//! is useful for:
//! - Classical post-processing (e.g., optimizers, loss functions)
//! - Hybrid quantum-classical algorithms
//! - Debugging and verification
//!
//! We implement both forward-mode and reverse-mode AD using dual numbers.

use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt;

/// Dual number for forward-mode automatic differentiation
///
/// A dual number has the form: f(x) + f'(x)ε where ε² = 0
/// This allows computing function values and derivatives simultaneously.
///
/// # Example
///
/// ```ignore
/// let x = Dual::variable(2.0);  // x = 2 + 1ε
/// let y = x * x;                 // y = 4 + 4ε
/// assert_eq!(y.value(), 4.0);
/// assert_eq!(y.derivative(), 4.0);  // dy/dx = 2x = 4
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// Function value
    value: f64,
    /// Derivative
    derivative: f64,
}

impl Dual {
    /// Create a new dual number
    pub fn new(value: f64, derivative: f64) -> Self {
        Self { value, derivative }
    }

    /// Create a constant (derivative = 0)
    pub fn constant(value: f64) -> Self {
        Self::new(value, 0.0)
    }

    /// Create a variable (derivative = 1)
    ///
    /// This is the starting point for computing derivatives.
    pub fn variable(value: f64) -> Self {
        Self::new(value, 1.0)
    }

    /// Get the function value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the derivative
    pub fn derivative(&self) -> f64 {
        self.derivative
    }

    /// Compute sin(x)
    pub fn sin(self) -> Self {
        Self::new(
            self.value.sin(),
            self.value.cos() * self.derivative,
        )
    }

    /// Compute cos(x)
    pub fn cos(self) -> Self {
        Self::new(
            self.value.cos(),
            -self.value.sin() * self.derivative,
        )
    }

    /// Compute exp(x)
    pub fn exp(self) -> Self {
        let exp_val = self.value.exp();
        Self::new(exp_val, exp_val * self.derivative)
    }

    /// Compute ln(x)
    pub fn ln(self) -> Self {
        Self::new(
            self.value.ln(),
            self.derivative / self.value,
        )
    }

    /// Compute x^n
    pub fn powi(self, n: i32) -> Self {
        Self::new(
            self.value.powi(n),
            (n as f64) * self.value.powi(n - 1) * self.derivative,
        )
    }

    /// Compute sqrt(x)
    pub fn sqrt(self) -> Self {
        let sqrt_val = self.value.sqrt();
        Self::new(
            sqrt_val,
            self.derivative / (2.0 * sqrt_val),
        )
    }

    /// Compute abs(x)
    pub fn abs(self) -> Self {
        if self.value >= 0.0 {
            self
        } else {
            -self
        }
    }
}

// Arithmetic operations for Dual numbers

impl Add for Dual {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(
            self.value + other.value,
            self.derivative + other.derivative,
        )
    }
}

impl Add<f64> for Dual {
    type Output = Self;

    fn add(self, other: f64) -> Self {
        Self::new(self.value + other, self.derivative)
    }
}

impl Sub for Dual {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(
            self.value - other.value,
            self.derivative - other.derivative,
        )
    }
}

impl Sub<f64> for Dual {
    type Output = Self;

    fn sub(self, other: f64) -> Self {
        Self::new(self.value - other, self.derivative)
    }
}

impl Mul for Dual {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // Product rule: (uv)' = u'v + uv'
        Self::new(
            self.value * other.value,
            self.derivative * other.value + self.value * other.derivative,
        )
    }
}

impl Mul<f64> for Dual {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Self::new(self.value * scalar, self.derivative * scalar)
    }
}

impl Div for Dual {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        // Quotient rule: (u/v)' = (u'v - uv') / v²
        Self::new(
            self.value / other.value,
            (self.derivative * other.value - self.value * other.derivative)
                / (other.value * other.value),
        )
    }
}

impl Div<f64> for Dual {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        Self::new(self.value / scalar, self.derivative / scalar)
    }
}

impl Neg for Dual {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.value, -self.derivative)
    }
}

impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}ε", self.value, self.derivative)
    }
}

/// Compute gradient using forward-mode AD
///
/// This evaluates the function n times (once per parameter) to compute
/// all partial derivatives.
///
/// # Arguments
///
/// * `f` - Function to differentiate
/// * `x` - Point at which to evaluate gradient
///
/// # Returns
///
/// Gradient vector at x
///
/// # Example
///
/// ```ignore
/// // f(x, y) = x² + xy + y²
/// let f = |vars: &[Dual]| vars[0] * vars[0] + vars[0] * vars[1] + vars[1] * vars[1];
///
/// let x = vec![2.0, 3.0];
/// let grad = gradient_forward(f, &x);
///
/// // ∂f/∂x = 2x + y = 7
/// // ∂f/∂y = x + 2y = 8
/// assert_eq!(grad, vec![7.0, 8.0]);
/// ```
pub fn gradient_forward<F>(f: F, x: &[f64]) -> Vec<f64>
where
    F: Fn(&[Dual]) -> Dual,
{
    let n = x.len();
    let mut gradient = Vec::with_capacity(n);

    for i in 0..n {
        // Create dual number vector with one variable
        let dual_x: Vec<Dual> = x
            .iter()
            .enumerate()
            .map(|(j, &val)| {
                if i == j {
                    Dual::variable(val) // This parameter is the variable
                } else {
                    Dual::constant(val) // Others are constants
                }
            })
            .collect();

        // Evaluate function
        let result = f(&dual_x);

        // Extract derivative
        gradient.push(result.derivative());
    }

    gradient
}

/// Tape-based reverse-mode automatic differentiation
///
/// This is more efficient than forward-mode for functions with many inputs
/// and few outputs (like f: R^n -> R).
#[derive(Debug, Clone)]
pub struct ReverseTape {
    operations: Vec<Operation>,
    values: Vec<f64>,
    adjoints: Vec<f64>,
}

#[derive(Debug, Clone)]
enum Operation {
    #[allow(dead_code)]
    Input { index: usize },
    Add { lhs: usize, rhs: usize },
    Mul { lhs: usize, rhs: usize },
    Sin { arg: usize },
    #[allow(dead_code)]
    Cos { arg: usize },
    #[allow(dead_code)]
    Exp { arg: usize },
}

impl ReverseTape {
    /// Create a new tape
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            values: Vec::new(),
            adjoints: Vec::new(),
        }
    }

    /// Add an input variable
    pub fn input(&mut self, value: f64) -> usize {
        let index = self.values.len();
        self.values.push(value);
        self.adjoints.push(0.0);
        self.operations.push(Operation::Input { index });
        index
    }

    /// Add two tape variables
    pub fn add(&mut self, lhs: usize, rhs: usize) -> usize {
        let index = self.values.len();
        let value = self.values[lhs] + self.values[rhs];
        self.values.push(value);
        self.adjoints.push(0.0);
        self.operations.push(Operation::Add { lhs, rhs });
        index
    }

    /// Multiply two tape variables
    pub fn mul(&mut self, lhs: usize, rhs: usize) -> usize {
        let index = self.values.len();
        let value = self.values[lhs] * self.values[rhs];
        self.values.push(value);
        self.adjoints.push(0.0);
        self.operations.push(Operation::Mul { lhs, rhs });
        index
    }

    /// Sine of a tape variable
    pub fn sin(&mut self, arg: usize) -> usize {
        let index = self.values.len();
        let value = self.values[arg].sin();
        self.values.push(value);
        self.adjoints.push(0.0);
        self.operations.push(Operation::Sin { arg });
        index
    }

    /// Compute gradients using reverse-mode AD
    ///
    /// # Arguments
    ///
    /// * `output` - Index of the output variable
    /// * `num_inputs` - Number of input variables
    ///
    /// # Returns
    ///
    /// Gradient with respect to all input variables
    pub fn gradient(&mut self, output: usize, num_inputs: usize) -> Vec<f64> {
        // Initialize all adjoints to 0
        for adj in &mut self.adjoints {
            *adj = 0.0;
        }

        // Seed the output
        self.adjoints[output] = 1.0;

        // Reverse pass
        for op in self.operations.iter().rev() {
            match *op {
                Operation::Input { .. } => {
                    // Nothing to do for inputs
                }
                Operation::Add { lhs, rhs } => {
                    let adj = self.adjoints[self.values.len() - 1];
                    self.adjoints[lhs] += adj;
                    self.adjoints[rhs] += adj;
                }
                Operation::Mul { lhs, rhs } => {
                    let adj = self.adjoints[self.values.len() - 1];
                    self.adjoints[lhs] += adj * self.values[rhs];
                    self.adjoints[rhs] += adj * self.values[lhs];
                }
                Operation::Sin { arg } => {
                    let adj = self.adjoints[self.values.len() - 1];
                    self.adjoints[arg] += adj * self.values[arg].cos();
                }
                Operation::Cos { arg } => {
                    let adj = self.adjoints[self.values.len() - 1];
                    self.adjoints[arg] -= adj * self.values[arg].sin();
                }
                Operation::Exp { arg } => {
                    let adj = self.adjoints[self.values.len() - 1];
                    self.adjoints[arg] += adj * self.values[arg].exp();
                }
            }
        }

        // Extract gradients for inputs
        self.adjoints[..num_inputs].to_vec()
    }
}

impl Default for ReverseTape {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid AD: Combine parameter shift rule with AD for classical post-processing
///
/// This is useful for quantum-classical hybrid algorithms where you have:
/// - Quantum circuit parameters (use parameter shift)
/// - Classical post-processing parameters (use AD)
pub struct HybridAD {
    /// Number of quantum parameters
    pub num_quantum_params: usize,
    /// Number of classical parameters
    pub num_classical_params: usize,
}

impl HybridAD {
    /// Create a new hybrid AD system
    pub fn new(num_quantum_params: usize, num_classical_params: usize) -> Self {
        Self {
            num_quantum_params,
            num_classical_params,
        }
    }

    /// Split parameters into quantum and classical
    pub fn split_params<'a>(&self, params: &'a [f64]) -> (&'a [f64], &'a [f64]) {
        let (quantum, classical) = params.split_at(self.num_quantum_params);
        (quantum, classical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_arithmetic() {
        let x = Dual::variable(3.0);
        let y = Dual::constant(2.0);

        // Test addition
        let z = x + y;
        assert_eq!(z.value(), 5.0);
        assert_eq!(z.derivative(), 1.0);

        // Test multiplication
        let z = x * y;
        assert_eq!(z.value(), 6.0);
        assert_eq!(z.derivative(), 2.0);
    }

    #[test]
    fn test_dual_product_rule() {
        // f(x) = x * x
        let x = Dual::variable(3.0);
        let y = x * x;

        assert_eq!(y.value(), 9.0);
        assert_eq!(y.derivative(), 6.0); // f'(x) = 2x = 6
    }

    #[test]
    fn test_dual_chain_rule() {
        // f(x) = sin(x²)
        let x = Dual::variable(2.0);
        let x_squared = x * x;
        let result = x_squared.sin();

        let expected_value = (4.0_f64).sin();
        let expected_derivative = (4.0_f64).cos() * 4.0; // cos(x²) * 2x

        assert!((result.value() - expected_value).abs() < 1e-10);
        assert!((result.derivative() - expected_derivative).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_forward() {
        // f(x, y) = x² + xy + y²
        let f = |vars: &[Dual]| {
            vars[0] * vars[0] + vars[0] * vars[1] + vars[1] * vars[1]
        };

        let x = vec![2.0, 3.0];
        let grad = gradient_forward(f, &x);

        // ∂f/∂x = 2x + y = 7
        // ∂f/∂y = x + 2y = 8
        assert!((grad[0] - 7.0).abs() < 1e-10);
        assert!((grad[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_forward_complex() {
        // f(x, y, z) = sin(x) * exp(y) + z²
        let f = |vars: &[Dual]| {
            vars[0].sin() * vars[1].exp() + vars[2] * vars[2]
        };

        let x = vec![1.0, 0.0, 2.0];
        let grad = gradient_forward(f, &x);

        // ∂f/∂x = cos(x) * exp(y) = cos(1) ≈ 0.540
        // ∂f/∂y = sin(x) * exp(y) = sin(1) ≈ 0.841
        // ∂f/∂z = 2z = 4
        assert!((grad[0] - 1.0_f64.cos()).abs() < 1e-10);
        assert!((grad[1] - 1.0_f64.sin()).abs() < 1e-10);
        assert!((grad[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_exp() {
        let x = Dual::variable(2.0);
        let y = x.exp();

        assert!((y.value() - 2.0_f64.exp()).abs() < 1e-10);
        assert!((y.derivative() - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn test_dual_ln() {
        let x = Dual::variable(3.0);
        let y = x.ln();

        assert!((y.value() - 3.0_f64.ln()).abs() < 1e-10);
        assert!((y.derivative() - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_powi() {
        let x = Dual::variable(2.0);
        let y = x.powi(3);

        assert_eq!(y.value(), 8.0);
        assert_eq!(y.derivative(), 12.0); // 3 * 2² = 12
    }
}
