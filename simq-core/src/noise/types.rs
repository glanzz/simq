//! Core types and traits for noise modeling

use crate::Result;
use num_complex::Complex64;
use std::fmt;

/// A Kraus operator representing a quantum operation
///
/// Quantum channels are described by a set of Kraus operators {K_i}
/// that satisfy the completeness relation: Σ K_i† K_i = I
///
/// The channel transforms a density matrix ρ as:
/// ρ → Σ_i K_i ρ K_i†
#[derive(Clone, Debug)]
pub struct KrausOperator {
    /// The matrix elements in row-major order
    /// For n-qubit operator, this is a 2^n × 2^n matrix flattened
    pub matrix: Vec<Complex64>,
    /// Dimension of the operator (2^n for n qubits)
    pub dimension: usize,
}

impl KrausOperator {
    /// Create a new Kraus operator from a matrix
    ///
    /// # Arguments
    /// * `matrix` - Flattened matrix in row-major order
    /// * `dimension` - Size of the square matrix (must be power of 2)
    ///
    /// # Errors
    /// Returns error if dimension is not a power of 2 or matrix size doesn't match
    pub fn new(matrix: Vec<Complex64>, dimension: usize) -> Result<Self> {
        // Verify dimension is power of 2
        if !dimension.is_power_of_two() || dimension == 0 {
            return Err(crate::QuantumError::ValidationError(format!(
                "Kraus operator dimension must be power of 2, got {}",
                dimension
            )));
        }

        // Verify matrix size
        if matrix.len() != dimension * dimension {
            return Err(crate::QuantumError::ValidationError(format!(
                "Matrix size {} doesn't match dimension {}×{} = {}",
                matrix.len(),
                dimension,
                dimension,
                dimension * dimension
            )));
        }

        Ok(Self { matrix, dimension })
    }

    /// Get the number of qubits this operator acts on
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.dimension.trailing_zeros() as usize
    }

    /// Get a matrix element at (row, col)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Complex64 {
        self.matrix[row * self.dimension + col]
    }

    /// Compute the adjoint (conjugate transpose) of this operator
    pub fn adjoint(&self) -> Self {
        let mut adj_matrix = vec![Complex64::new(0.0, 0.0); self.matrix.len()];

        for i in 0..self.dimension {
            for j in 0..self.dimension {
                adj_matrix[j * self.dimension + i] = self.matrix[i * self.dimension + j].conj();
            }
        }

        Self {
            matrix: adj_matrix,
            dimension: self.dimension,
        }
    }
}

/// Trait for quantum noise channels
///
/// A noise channel describes how errors affect quantum states.
/// Channels are represented using the Kraus operator formalism.
///
/// # Implementing a Custom Channel
///
/// ```ignore
/// struct CustomNoise { error_rate: f64 }
///
/// impl NoiseChannel for CustomNoise {
///     fn kraus_operators(&self) -> Vec<KrausOperator> {
///         // Return Kraus operators for your channel
///         vec![...]
///     }
///
///     fn num_qubits(&self) -> usize { 1 }
///     fn name(&self) -> &str { "custom" }
/// }
/// ```
pub trait NoiseChannel: Send + Sync + fmt::Debug {
    /// Get the Kraus operators defining this channel
    ///
    /// The operators must satisfy the completeness relation:
    /// Σ_i K_i† K_i = I (within numerical precision)
    fn kraus_operators(&self) -> Vec<KrausOperator>;

    /// Number of qubits this channel acts on
    fn num_qubits(&self) -> usize;

    /// Name of this noise channel (e.g., "depolarizing", "amplitude_damping")
    fn name(&self) -> &str;

    /// Get a description of this channel
    fn description(&self) -> String {
        format!("{}-qubit {} channel", self.num_qubits(), self.name())
    }

    /// Verify the completeness relation Σ K_i† K_i = I
    ///
    /// Returns true if the channel is valid (within tolerance)
    fn verify_completeness(&self, tolerance: f64) -> bool {
        let operators = self.kraus_operators();
        if operators.is_empty() {
            return false;
        }

        let dim = operators[0].dimension;
        let mut sum = vec![Complex64::new(0.0, 0.0); dim * dim];

        // Compute Σ K_i† K_i
        for kraus in &operators {
            let adj = kraus.adjoint();

            // Matrix multiply: adj × kraus
            for i in 0..dim {
                for j in 0..dim {
                    let mut element = Complex64::new(0.0, 0.0);
                    for k in 0..dim {
                        element += adj.get(i, k) * kraus.get(k, j);
                    }
                    sum[i * dim + j] += element;
                }
            }
        }

        // Check if sum ≈ I (identity matrix)
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (sum[i * dim + j] - expected).norm();
                if diff > tolerance {
                    return false;
                }
            }
        }

        true
    }
}

/// A complete noise model that can be applied to a quantum circuit
///
/// Combines multiple noise channels that may be applied at different
/// stages of circuit execution (after gates, during measurement, etc.)
#[derive(Debug)]
pub struct NoiseModel {
    /// Gate noise applied after each gate operation
    pub gate_noise: Option<Box<dyn NoiseChannel>>,

    /// Readout noise applied during measurement
    pub readout_noise: Option<Box<dyn NoiseChannel>>,

    /// Idle noise (T1/T2) applied based on circuit depth/time
    pub idle_noise: Option<Box<dyn NoiseChannel>>,
}

impl NoiseModel {
    /// Create an empty noise model with no errors
    pub fn new() -> Self {
        Self {
            gate_noise: None,
            readout_noise: None,
            idle_noise: None,
        }
    }

    /// Create a noise model with gate errors only
    pub fn with_gate_noise(channel: impl NoiseChannel + 'static) -> Self {
        Self {
            gate_noise: Some(Box::new(channel)),
            readout_noise: None,
            idle_noise: None,
        }
    }

    /// Add gate noise to this model
    pub fn set_gate_noise(&mut self, channel: impl NoiseChannel + 'static) {
        self.gate_noise = Some(Box::new(channel));
    }

    /// Add readout noise to this model
    pub fn set_readout_noise(&mut self, channel: impl NoiseChannel + 'static) {
        self.readout_noise = Some(Box::new(channel));
    }

    /// Add idle/decoherence noise to this model
    pub fn set_idle_noise(&mut self, channel: impl NoiseChannel + 'static) {
        self.idle_noise = Some(Box::new(channel));
    }

    /// Check if this model has any noise
    pub fn has_noise(&self) -> bool {
        self.gate_noise.is_some() || self.readout_noise.is_some() || self.idle_noise.is_some()
    }
}

impl Default for NoiseModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kraus_operator_creation() {
        // Identity operator
        let matrix = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ];
        let op = KrausOperator::new(matrix, 2).unwrap();
        assert_eq!(op.num_qubits(), 1);
        assert_eq!(op.dimension, 2);
    }

    #[test]
    fn test_kraus_operator_invalid_dimension() {
        let matrix = vec![Complex64::new(1.0, 0.0); 9];
        let result = KrausOperator::new(matrix, 3); // 3 is not power of 2
        assert!(result.is_err());
    }

    #[test]
    fn test_kraus_operator_adjoint() {
        // Simple 2x2 matrix
        let matrix = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(4.0, -1.0),
        ];
        let op = KrausOperator::new(matrix, 2).unwrap();
        let adj = op.adjoint();

        // Check conjugate transpose
        assert_eq!(adj.get(0, 0), Complex64::new(1.0, -1.0));
        assert_eq!(adj.get(0, 1), Complex64::new(0.0, -3.0));
        assert_eq!(adj.get(1, 0), Complex64::new(2.0, 0.0));
        assert_eq!(adj.get(1, 1), Complex64::new(4.0, 1.0));
    }

    #[test]
    fn test_noise_model_creation() {
        let model = NoiseModel::new();
        assert!(!model.has_noise());

        let model = NoiseModel::new();
        assert!(!model.has_noise());
    }
}
