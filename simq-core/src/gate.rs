//! Quantum gate definitions and operations

use crate::{QuantumError, QubitId, Result};
use num_complex::Complex64;
use smallvec::SmallVec;
use std::fmt;
use std::sync::Arc;

/// Trait for quantum gate operations
///
/// All quantum gates must implement this trait. Gates are stateless
/// and reusable across multiple circuits.
///
/// # Example
/// ```ignore
/// struct HadamardGate;
///
/// impl Gate for HadamardGate {
///     fn name(&self) -> &str { "H" }
///     fn num_qubits(&self) -> usize { 1 }
///     fn is_unitary(&self) -> bool { true }
/// }
/// ```
pub trait Gate: Send + Sync + fmt::Debug {
    /// The name of the gate (e.g., "H", "CNOT", "RX")
    fn name(&self) -> &str;

    /// Number of qubits this gate acts on
    fn num_qubits(&self) -> usize;

    /// Whether this gate is a unitary operation
    ///
    /// Most gates are unitary. Measurement operations are not.
    fn is_unitary(&self) -> bool {
        true
    }

    /// Whether this gate is hermitian (self-adjoint)
    ///
    /// Hermitian gates are their own inverse.
    fn is_hermitian(&self) -> bool {
        false
    }

    /// Get a description of this gate
    fn description(&self) -> String {
        format!("{}-qubit gate '{}'", self.num_qubits(), self.name())
    }

    /// Get the unitary matrix for this gate as a flattened vector
    ///
    /// The matrix is stored in row-major order. For an n-qubit gate,
    /// the matrix has dimension 2^n × 2^n, and the returned vector
    /// has length (2^n)^2.
    ///
    /// Returns `None` for gates that don't have a simple matrix representation
    /// (e.g., measurement operations, custom oracles).
    ///
    /// # Example
    /// ```ignore
    /// let hadamard = HadamardGate;
    /// if let Some(matrix) = hadamard.matrix() {
    ///     // Matrix is 2x2, so length is 4
    ///     assert_eq!(matrix.len(), 4);
    /// }
    /// ```
    fn matrix(&self) -> Option<Vec<Complex64>> {
        None
    }
}

/// A gate operation applied to specific qubits
///
/// Combines a gate with the qubits it operates on.
///
/// # Example
/// ```
/// # use simq_core::{QubitId, gate::GateOp};
/// # use std::sync::Arc;
/// # #[derive(Debug)]
/// # struct DummyGate;
/// # impl simq_core::gate::Gate for DummyGate {
/// #     fn name(&self) -> &str { "DUMMY" }
/// #     fn num_qubits(&self) -> usize { 1 }
/// # }
/// let gate = Arc::new(DummyGate);
/// let q0 = QubitId::new(0);
/// let op = GateOp::new(gate, &[q0]).unwrap();
/// ```
#[derive(Clone)]
pub struct GateOp {
    gate: Arc<dyn Gate>,
    qubits: SmallVec<[QubitId; 2]>, // Most gates are 1-2 qubits
}

impl GateOp {
    /// Create a new gate operation
    ///
    /// # Errors
    /// Returns error if:
    /// - Qubit count doesn't match gate requirements
    /// - Duplicate qubits specified
    pub fn new(gate: Arc<dyn Gate>, qubits: &[QubitId]) -> Result<Self> {
        // Validate qubit count
        if qubits.len() != gate.num_qubits() {
            return Err(QuantumError::invalid_qubit_count(
                gate.name(),
                gate.num_qubits(),
                qubits.len(),
            ));
        }

        // Check for duplicate qubits
        for i in 0..qubits.len() {
            for j in (i + 1)..qubits.len() {
                if qubits[i] == qubits[j] {
                    return Err(QuantumError::DuplicateQubit(qubits[i]));
                }
            }
        }

        Ok(Self {
            gate,
            qubits: SmallVec::from_slice(qubits),
        })
    }

    /// Get the gate
    #[inline]
    pub fn gate(&self) -> &Arc<dyn Gate> {
        &self.gate
    }

    /// Get the qubits this operation acts on
    #[inline]
    pub fn qubits(&self) -> &[QubitId] {
        &self.qubits
    }

    /// Get the number of qubits
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }
}

impl fmt::Debug for GateOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.gate.name())?;
        for (i, q) in self.qubits.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", q)?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for GateOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock gate for testing
    #[derive(Debug)]
    struct MockGate {
        name: String,
        num_qubits: usize,
    }

    impl Gate for MockGate {
        fn name(&self) -> &str {
            &self.name
        }

        fn num_qubits(&self) -> usize {
            self.num_qubits
        }
    }

    #[test]
    fn test_gate_op_creation() {
        let gate = Arc::new(MockGate {
            name: "H".to_string(),
            num_qubits: 1,
        });
        let q0 = QubitId::new(0);
        let op = GateOp::new(gate, &[q0]).unwrap();

        assert_eq!(op.num_qubits(), 1);
        assert_eq!(op.qubits()[0], q0);
    }

    #[test]
    fn test_gate_op_invalid_qubit_count() {
        let gate = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });
        let q0 = QubitId::new(0);

        let result = GateOp::new(gate, &[q0]);
        assert!(result.is_err());

        if let Err(QuantumError::InvalidQubitCount {
            gate,
            expected,
            actual,
        }) = result
        {
            assert_eq!(gate, "CNOT");
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        } else {
            panic!("Expected InvalidQubitCount error");
        }
    }

    #[test]
    fn test_gate_op_duplicate_qubits() {
        let gate = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });
        let q0 = QubitId::new(0);

        let result = GateOp::new(gate, &[q0, q0]);
        assert!(result.is_err());
        assert!(matches!(result, Err(QuantumError::DuplicateQubit(_))));
    }

    #[test]
    fn test_gate_op_display() {
        let gate = Arc::new(MockGate {
            name: "CNOT".to_string(),
            num_qubits: 2,
        });
        let q0 = QubitId::new(0);
        let q1 = QubitId::new(1);
        let op = GateOp::new(gate, &[q0, q1]).unwrap();

        let display = format!("{}", op);
        assert!(display.contains("CNOT"));
        assert!(display.contains("q0"));
        assert!(display.contains("q1"));
    }
}
