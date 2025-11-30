//! Pauli observable implementation for expectation value computation
//!
//! This module provides efficient computation of expectation values ⟨ψ|O|ψ⟩
//! for Pauli observables without state collapse. This is critical for
//! variational quantum algorithms like VQE and QAOA.
//!
//! # Pauli Operators
//!
//! The four single-qubit Pauli operators:
//! - I: Identity [[1,0],[0,1]]
//! - X: Bit flip [[0,1],[1,0]]
//! - Y: Phase flip [[0,-i],[i,0]]
//! - Z: Phase flip [[1,0],[0,-1]]
//!
//! # Pauli Strings
//!
//! A Pauli string is a tensor product of single-qubit Paulis, e.g., "IXYZ"
//! represents I⊗X⊗Y⊗Z acting on 4 qubits.

use crate::dense_state::DenseState;
use crate::error::{Result, StateError};
use num_complex::Complex64;
use std::fmt;

/// Single-qubit Pauli operator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pauli {
    /// Identity operator
    I,
    /// Pauli X (bit flip)
    X,
    /// Pauli Y (phase flip with i)
    Y,
    /// Pauli Z (phase flip)
    Z,
}

impl Pauli {
    /// Parse a Pauli operator from a character
    pub fn from_char(c: char) -> Result<Self> {
        match c.to_ascii_uppercase() {
            'I' => Ok(Pauli::I),
            'X' => Ok(Pauli::X),
            'Y' => Ok(Pauli::Y),
            'Z' => Ok(Pauli::Z),
            _ => Err(StateError::InvalidDimension { dimension: 0 }),
        }
    }

    /// Convert to character representation
    pub fn to_char(self) -> char {
        match self {
            Pauli::I => 'I',
            Pauli::X => 'X',
            Pauli::Y => 'Y',
            Pauli::Z => 'Z',
        }
    }

    /// Check if this Pauli is diagonal (I or Z)
    pub fn is_diagonal(self) -> bool {
        matches!(self, Pauli::I | Pauli::Z)
    }

    /// Get the eigenvalue for a computational basis state
    /// Returns (+1, -1) for diagonal operators
    pub fn eigenvalue(self, basis_state: bool) -> f64 {
        match self {
            Pauli::I => 1.0,
            Pauli::Z => {
                if basis_state {
                    -1.0
                } else {
                    1.0
                }
            },
            _ => panic!("eigenvalue only valid for diagonal Paulis"),
        }
    }
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// A tensor product of Pauli operators (Pauli string)
///
/// Represents observables like X⊗X⊗I⊗Z (written as "XXIZ")
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliString {
    /// Pauli operators for each qubit
    paulis: Vec<Pauli>,

    /// Overall coefficient
    coeff: i32, // Either +1 or -1 for simplicity

    /// Phase factor as multiple of π/2 (0, 1, 2, 3 for 1, i, -1, -i)
    phase: u8,
}

impl PauliString {
    /// Create a new Pauli string from a string representation
    ///
    /// # Example
    /// ```
    /// use simq_state::PauliString;
    ///
    /// let pauli = PauliString::from_str("XXYZ").unwrap();
    /// assert_eq!(pauli.num_qubits(), 4);
    /// ```
    pub fn from_str(s: &str) -> Result<Self> {
        let paulis: Result<Vec<_>> = s.chars().map(Pauli::from_char).collect();
        Ok(Self {
            paulis: paulis?,
            coeff: 1,
            phase: 0,
        })
    }

    /// Create a Pauli string from a vector of Paulis
    pub fn from_paulis(paulis: Vec<Pauli>) -> Self {
        Self {
            paulis,
            coeff: 1,
            phase: 0,
        }
    }

    /// Create an all-Z Pauli string for a given number of qubits
    pub fn all_z(num_qubits: usize) -> Self {
        Self {
            paulis: vec![Pauli::Z; num_qubits],
            coeff: 1,
            phase: 0,
        }
    }

    /// Create an all-I (identity) Pauli string
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            paulis: vec![Pauli::I; num_qubits],
            coeff: 1,
            phase: 0,
        }
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.paulis.len()
    }

    /// Get the Pauli operator at a specific qubit
    pub fn get(&self, qubit: usize) -> Option<Pauli> {
        self.paulis.get(qubit).copied()
    }

    /// Set coefficient
    pub fn with_coeff(mut self, coeff: i32) -> Self {
        self.coeff = coeff;
        self
    }

    /// Check if this Pauli string is diagonal (all I or Z)
    pub fn is_diagonal(&self) -> bool {
        self.paulis.iter().all(|p| p.is_diagonal())
    }

    /// Compute expectation value ⟨ψ|P|ψ⟩ for this Pauli string
    ///
    /// # Arguments
    /// * `state` - The quantum state
    ///
    /// # Returns
    /// The expectation value (real number)
    pub fn expectation_value(&self, state: &DenseState) -> Result<f64> {
        if self.num_qubits() != state.num_qubits() {
            return Err(StateError::DimensionMismatch {
                expected: self.num_qubits(),
                actual: state.num_qubits(),
            });
        }

        if self.is_diagonal() {
            // Fast path for diagonal operators
            self.expectation_value_diagonal(state)
        } else {
            // General case: apply Pauli string and compute ⟨ψ|P|ψ⟩
            self.expectation_value_general(state)
        }
    }

    /// Compute expectation value for diagonal Pauli string (fast path)
    fn expectation_value_diagonal(&self, state: &DenseState) -> Result<f64> {
        let amplitudes = state.amplitudes();

        let mut expectation = 0.0;

        for (basis_state, amplitude) in amplitudes.iter().enumerate() {
            let probability = amplitude.norm_sqr();

            // Compute eigenvalue for this basis state
            let mut eigenvalue = self.coeff as f64;
            for (qubit, &pauli) in self.paulis.iter().enumerate() {
                if pauli == Pauli::Z {
                    // Check if qubit is |1⟩ in this basis state
                    let bit = (basis_state >> qubit) & 1 == 1;
                    eigenvalue *= pauli.eigenvalue(bit);
                }
                // Pauli::I contributes factor of 1
            }

            expectation += probability * eigenvalue;
        }

        Ok(expectation)
    }

    /// Compute expectation value for general Pauli string
    fn expectation_value_general(&self, state: &DenseState) -> Result<f64> {
        // For non-diagonal operators, we need to apply the Pauli string
        // and compute the inner product ⟨ψ|P|ψ⟩

        let amplitudes = state.amplitudes();
        let num_qubits = state.num_qubits();
        let dimension = 1 << num_qubits;

        // Compute P|ψ⟩
        let mut transformed = vec![Complex64::new(0.0, 0.0); dimension];

        for (basis_state, &amplitude) in amplitudes.iter().enumerate() {
            if amplitude.norm_sqr() < 1e-15 {
                continue; // Skip zero amplitudes
            }

            // Apply Pauli string to this basis state
            let (new_state, phase) = self.apply_to_basis_state(basis_state);
            transformed[new_state] += amplitude * phase;
        }

        // Compute ⟨ψ|P|ψ⟩ = Σ conj(ψ[i]) * transformed[i]
        let expectation: Complex64 = amplitudes
            .iter()
            .zip(transformed.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        // Apply overall coefficient
        let result = expectation.re * self.coeff as f64;

        Ok(result)
    }

    /// Apply Pauli string to a computational basis state
    ///
    /// Returns (new_state, phase_factor)
    fn apply_to_basis_state(&self, basis_state: usize) -> (usize, Complex64) {
        let mut new_state = basis_state;
        let mut phase = Complex64::new(1.0, 0.0);

        for (qubit, &pauli) in self.paulis.iter().enumerate() {
            let bit = (basis_state >> qubit) & 1;

            match pauli {
                Pauli::I => {
                    // Identity: no change
                },
                Pauli::X => {
                    // Flip bit
                    new_state ^= 1 << qubit;
                },
                Pauli::Y => {
                    // Flip bit with phase
                    new_state ^= 1 << qubit;
                    // Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
                    phase *= if bit == 0 {
                        Complex64::new(0.0, 1.0) // i
                    } else {
                        Complex64::new(0.0, -1.0) // -i
                    };
                },
                Pauli::Z => {
                    // Phase flip: |0⟩ → |0⟩, |1⟩ → -|1⟩
                    if bit == 1 {
                        phase *= Complex64::new(-1.0, 0.0);
                    }
                },
            }
        }

        (new_state, phase)
    }
}

impl fmt::Display for PauliString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.coeff == -1 {
            write!(f, "-")?;
        }
        for pauli in &self.paulis {
            write!(f, "{}", pauli)?;
        }
        Ok(())
    }
}

/// A weighted sum of Pauli strings (Pauli observable)
///
/// Represents observables like 0.5*X⊗X + 0.3*Z⊗Z
#[derive(Debug, Clone)]
pub struct PauliObservable {
    /// Terms in the observable (Pauli string, coefficient)
    terms: Vec<(PauliString, f64)>,
}

impl PauliObservable {
    /// Create a new empty observable
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Create an observable from a single Pauli string
    pub fn from_pauli_string(pauli: PauliString, coeff: f64) -> Self {
        Self {
            terms: vec![(pauli, coeff)],
        }
    }

    /// Add a term to the observable
    pub fn add_term(&mut self, pauli: PauliString, coeff: f64) {
        self.terms.push((pauli, coeff));
    }

    /// Create a Z observable for a single qubit
    ///
    /// Measures spin in Z direction for qubit at position `qubit`
    pub fn single_z(num_qubits: usize, qubit: usize) -> Self {
        let mut paulis = vec![Pauli::I; num_qubits];
        paulis[qubit] = Pauli::Z;

        Self::from_pauli_string(PauliString::from_paulis(paulis), 1.0)
    }

    /// Get the number of terms
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Compute expectation value ⟨ψ|O|ψ⟩
    ///
    /// # Arguments
    /// * `state` - The quantum state
    ///
    /// # Returns
    /// The expectation value (real number)
    pub fn expectation_value(&self, state: &DenseState) -> Result<f64> {
        let mut total = 0.0;

        for (pauli, coeff) in &self.terms {
            let term_expectation = pauli.expectation_value(state)?;
            total += coeff * term_expectation;
        }

        Ok(total)
    }

    /// Check if all terms are diagonal
    pub fn is_diagonal(&self) -> bool {
        self.terms.iter().all(|(p, _)| p.is_diagonal())
    }
}

impl Default for PauliObservable {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PauliObservable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (i, (pauli, coeff)) in self.terms.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            write!(f, "{:.4}·{}", coeff, pauli)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pauli_from_char() {
        assert_eq!(Pauli::from_char('I').unwrap(), Pauli::I);
        assert_eq!(Pauli::from_char('X').unwrap(), Pauli::X);
        assert_eq!(Pauli::from_char('Y').unwrap(), Pauli::Y);
        assert_eq!(Pauli::from_char('Z').unwrap(), Pauli::Z);
        assert_eq!(Pauli::from_char('x').unwrap(), Pauli::X); // Case insensitive
    }

    #[test]
    fn test_pauli_is_diagonal() {
        assert!(Pauli::I.is_diagonal());
        assert!(!Pauli::X.is_diagonal());
        assert!(!Pauli::Y.is_diagonal());
        assert!(Pauli::Z.is_diagonal());
    }

    #[test]
    fn test_pauli_string_from_str() {
        let pauli = PauliString::from_str("IXYZ").unwrap();
        assert_eq!(pauli.num_qubits(), 4);
        assert_eq!(pauli.get(0), Some(Pauli::I));
        assert_eq!(pauli.get(1), Some(Pauli::X));
        assert_eq!(pauli.get(2), Some(Pauli::Y));
        assert_eq!(pauli.get(3), Some(Pauli::Z));
    }

    #[test]
    fn test_pauli_string_is_diagonal() {
        assert!(PauliString::from_str("IIZZ").unwrap().is_diagonal());
        assert!(!PauliString::from_str("IXYZ").unwrap().is_diagonal());
        assert!(!PauliString::from_str("XIIZ").unwrap().is_diagonal());
    }

    #[test]
    fn test_expectation_value_z_basis_state() {
        // |0⟩ state
        let state = DenseState::new(1).unwrap();

        // Z observable: expect +1 for |0⟩
        let z_obs = PauliString::from_str("Z").unwrap();
        let expectation = z_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);

        // X observable: expect 0 for |0⟩
        let x_obs = PauliString::from_str("X").unwrap();
        let expectation = x_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expectation_value_x_basis_state() {
        // |+⟩ = (|0⟩ + |1⟩)/√2 state
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ];
        let state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

        // X observable: expect +1 for |+⟩
        let x_obs = PauliString::from_str("X").unwrap();
        let expectation = x_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);

        // Z observable: expect 0 for |+⟩
        let z_obs = PauliString::from_str("Z").unwrap();
        let expectation = z_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_expectation_value_bell_state() {
        // Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ];
        let state = DenseState::from_amplitudes(2, &amplitudes).unwrap();

        // ZZ observable: expect +1 (both qubits have same Z eigenvalue)
        let zz_obs = PauliString::from_str("ZZ").unwrap();
        let expectation = zz_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);

        // XX observable: expect +1 for Bell state
        let xx_obs = PauliString::from_str("XX").unwrap();
        let expectation = xx_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);

        // ZI observable: expect 0 (equal superposition of Z eigenvalues)
        let zi_obs = PauliString::from_str("ZI").unwrap();
        let expectation = zi_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pauli_observable_single_term() {
        let state = DenseState::new(1).unwrap();

        let mut obs = PauliObservable::new();
        obs.add_term(PauliString::from_str("Z").unwrap(), 1.0);

        let expectation = obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pauli_observable_multiple_terms() {
        // |+⟩ state
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
        ];
        let state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

        // Observable: 0.5*X + 0.3*Z
        let mut obs = PauliObservable::new();
        obs.add_term(PauliString::from_str("X").unwrap(), 0.5);
        obs.add_term(PauliString::from_str("Z").unwrap(), 0.3);

        // Expect: 0.5*1.0 + 0.3*0.0 = 0.5
        let expectation = obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_pauli_y_operator() {
        // |0⟩ state
        let state = DenseState::new(1).unwrap();

        // Y|0⟩ = i|1⟩, so ⟨0|Y|0⟩ = 0
        let y_obs = PauliString::from_str("Y").unwrap();
        let expectation = y_obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 0.0, epsilon = 1e-10);

        // |+i⟩ = (|0⟩ + i|1⟩)/√2 state (eigenstate of Y with +1)
        let amplitudes = vec![
            Complex64::new(1.0 / 2_f64.sqrt(), 0.0),
            Complex64::new(0.0, 1.0 / 2_f64.sqrt()),
        ];
        let plus_i_state = DenseState::from_amplitudes(1, &amplitudes).unwrap();

        let expectation = y_obs.expectation_value(&plus_i_state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_to_basis_state() {
        let pauli_x = PauliString::from_str("X").unwrap();

        // X|0⟩ = |1⟩
        let (new_state, phase) = pauli_x.apply_to_basis_state(0);
        assert_eq!(new_state, 1);
        assert_relative_eq!(phase.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(phase.im, 0.0, epsilon = 1e-10);

        // X|1⟩ = |0⟩
        let (new_state, phase) = pauli_x.apply_to_basis_state(1);
        assert_eq!(new_state, 0);
        assert_relative_eq!(phase.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_single_z_observable() {
        let state = DenseState::new(3).unwrap();

        // Measure Z on qubit 1 (should be +1 for |000⟩)
        let obs = PauliObservable::single_z(3, 1);
        let expectation = obs.expectation_value(&state).unwrap();
        assert_relative_eq!(expectation, 1.0, epsilon = 1e-10);
    }
}
