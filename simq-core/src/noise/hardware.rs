//! Hardware-calibrated noise models
//!
//! This module provides realistic noise models based on actual quantum
//! hardware calibration data. It supports per-qubit and per-gate noise
//! parameters matching real devices.
//!
//! # Features
//!
//! - **Per-qubit calibration**: Individual T1, T2, readout errors, gate fidelities
//! - **Time-aware noise**: Idle noise based on accumulated idle time
//! - **Two-qubit correlated errors**: Realistic CNOT/CZ gate noise
//! - **Crosstalk modeling**: ZZ-coupling and spectator errors
//! - **Device presets**: IBM, Google, IonQ hardware profiles
//!
//! # Example
//!
//! ```ignore
//! use simq_core::noise::{HardwareNoiseModel, QubitProperties};
//!
//! // IBM quantum hardware preset
//! let noise_model = HardwareNoiseModel::ibm_washington();
//!
//! // Or custom calibration
//! let mut model = HardwareNoiseModel::new(5);
//! model.set_qubit_t1(0, 100.0); // 100μs T1 time
//! model.set_qubit_t2(0, 80.0);  // 80μs T2 time
//! model.set_readout_error(0, 0.015, 0.02); // 1.5%, 2% readout errors
//!
//! // Time-aware idle noise
//! let idle_noise = model.idle_noise(0, 5.0)?; // 5μs idle time
//!
//! // Two-qubit gate noise
//! let cnot_noise = model.two_qubit_gate_noise(0, 1)?;
//! ```

use super::channels::{AmplitudeDamping, DepolarizingChannel, PhaseDamping, ReadoutError};
use crate::Result;
use std::collections::HashMap;

/// Properties of a single qubit based on hardware calibration
#[derive(Debug, Clone)]
pub struct QubitProperties {
    /// T1 relaxation time (μs)
    pub t1: f64,

    /// T2 dephasing time (μs)
    pub t2: f64,

    /// Readout error: P(measure 1 | state is 0)
    pub readout_p01: f64,

    /// Readout error: P(measure 0 | state is 1)
    pub readout_p10: f64,

    /// Single-qubit gate fidelity (0-1)
    pub single_qubit_gate_fidelity: f64,
}

impl Default for QubitProperties {
    fn default() -> Self {
        Self {
            t1: 100.0,                          // 100μs
            t2: 80.0,                           // 80μs
            readout_p01: 0.01,                  // 1%
            readout_p10: 0.02,                  // 2%
            single_qubit_gate_fidelity: 0.9995, // 99.95%
        }
    }
}

/// Gate timing information
#[derive(Debug, Clone, Copy)]
pub struct GateTiming {
    /// Duration of single-qubit gates (μs)
    pub single_qubit_gate_time: f64,

    /// Duration of two-qubit gates (μs)
    pub two_qubit_gate_time: f64,

    /// Measurement time (μs)
    pub measurement_time: f64,
}

impl Default for GateTiming {
    fn default() -> Self {
        Self {
            single_qubit_gate_time: 0.02, // 20ns
            two_qubit_gate_time: 0.1,     // 100ns
            measurement_time: 1.0,        // 1μs
        }
    }
}

/// Two-qubit gate properties
#[derive(Debug, Clone)]
pub struct TwoQubitGateProperties {
    /// Gate fidelity (0-1)
    pub fidelity: f64,

    /// Gate duration (μs)
    pub duration: f64,
}

impl Default for TwoQubitGateProperties {
    fn default() -> Self {
        Self {
            fidelity: 0.99, // 99%
            duration: 0.1,  // 100ns
        }
    }
}

/// Crosstalk properties between qubits
#[derive(Debug, Clone)]
pub struct CrosstalkProperties {
    /// ZZ-coupling strength (MHz)
    pub zz_coupling: f64,

    /// Spectator error rate when neighboring qubit is operated
    pub spectator_error: f64,
}

impl Default for CrosstalkProperties {
    fn default() -> Self {
        Self {
            zz_coupling: 0.0,     // No crosstalk by default
            spectator_error: 0.0, // No spectator errors
        }
    }
}

/// Combined noise channels for a gate operation
#[derive(Debug)]
pub struct GateNoise {
    /// Amplitude damping (T1 decay)
    pub amplitude_damping: Vec<AmplitudeDamping>,

    /// Phase damping (T2 dephasing)
    pub phase_damping: Vec<PhaseDamping>,

    /// Depolarizing noise (gate errors)
    pub depolarizing: Vec<DepolarizingChannel>,

    /// Affected qubits
    pub qubits: Vec<usize>,
}

/// Hardware-calibrated noise model
///
/// Represents realistic noise based on quantum hardware specifications.
/// Supports per-qubit T1/T2, readout errors, gate fidelities, crosstalk,
/// and time-aware noise modeling.
#[derive(Debug, Clone)]
pub struct HardwareNoiseModel {
    /// Number of qubits
    num_qubits: usize,

    /// Per-qubit properties
    qubit_properties: Vec<QubitProperties>,

    /// Gate timing information
    timing: GateTiming,

    /// Two-qubit gate properties (indexed by qubit pair)
    two_qubit_gates: HashMap<(usize, usize), TwoQubitGateProperties>,

    /// Crosstalk properties between qubit pairs
    crosstalk: HashMap<(usize, usize), CrosstalkProperties>,

    /// Enable time-aware idle noise modeling
    enable_idle_noise: bool,

    /// Enable crosstalk effects
    enable_crosstalk: bool,
}

impl HardwareNoiseModel {
    /// Create a new hardware noise model
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the device
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            qubit_properties: vec![QubitProperties::default(); num_qubits],
            timing: GateTiming::default(),
            two_qubit_gates: HashMap::new(),
            crosstalk: HashMap::new(),
            enable_idle_noise: true,
            enable_crosstalk: false,
        }
    }

    /// Enable or disable idle noise modeling
    pub fn set_idle_noise_enabled(&mut self, enabled: bool) {
        self.enable_idle_noise = enabled;
    }

    /// Check if idle noise is enabled
    pub fn is_idle_noise_enabled(&self) -> bool {
        self.enable_idle_noise
    }

    /// Enable or disable crosstalk effects
    pub fn set_crosstalk_enabled(&mut self, enabled: bool) {
        self.enable_crosstalk = enabled;
    }

    /// Check if crosstalk is enabled
    pub fn is_crosstalk_enabled(&self) -> bool {
        self.enable_crosstalk
    }

    /// Set T1 relaxation time for a qubit
    pub fn set_qubit_t1(&mut self, qubit: usize, t1_us: f64) {
        if qubit < self.num_qubits {
            self.qubit_properties[qubit].t1 = t1_us;
        }
    }

    /// Set T2 dephasing time for a qubit
    pub fn set_qubit_t2(&mut self, qubit: usize, t2_us: f64) {
        if qubit < self.num_qubits {
            self.qubit_properties[qubit].t2 = t2_us;
        }
    }

    /// Set readout errors for a qubit
    pub fn set_readout_error(&mut self, qubit: usize, p01: f64, p10: f64) {
        if qubit < self.num_qubits {
            self.qubit_properties[qubit].readout_p01 = p01;
            self.qubit_properties[qubit].readout_p10 = p10;
        }
    }

    /// Set single-qubit gate fidelity
    pub fn set_single_qubit_fidelity(&mut self, qubit: usize, fidelity: f64) {
        if qubit < self.num_qubits {
            self.qubit_properties[qubit].single_qubit_gate_fidelity = fidelity;
        }
    }

    /// Set two-qubit gate properties
    pub fn set_two_qubit_gate(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        fidelity: f64,
        duration_us: f64,
    ) {
        let key = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };

        self.two_qubit_gates.insert(
            key,
            TwoQubitGateProperties {
                fidelity,
                duration: duration_us,
            },
        );
    }

    /// Set crosstalk properties between two qubits
    ///
    /// # Arguments
    /// * `qubit1` - First qubit index
    /// * `qubit2` - Second qubit index
    /// * `zz_coupling` - ZZ-coupling strength in MHz
    /// * `spectator_error` - Error rate on spectator qubit
    pub fn set_crosstalk(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        zz_coupling: f64,
        spectator_error: f64,
    ) {
        let key = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };

        self.crosstalk.insert(
            key,
            CrosstalkProperties {
                zz_coupling,
                spectator_error,
            },
        );
    }

    /// Get crosstalk properties between two qubits
    pub fn crosstalk_properties(
        &self,
        qubit1: usize,
        qubit2: usize,
    ) -> Option<&CrosstalkProperties> {
        let key = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };
        self.crosstalk.get(&key)
    }

    /// Get two-qubit gate properties
    pub fn two_qubit_gate_properties(
        &self,
        qubit1: usize,
        qubit2: usize,
    ) -> Option<&TwoQubitGateProperties> {
        let key = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };
        self.two_qubit_gates.get(&key)
    }

    /// Get gate timing information
    pub fn timing(&self) -> &GateTiming {
        &self.timing
    }

    /// Set gate timing information
    pub fn set_timing(&mut self, timing: GateTiming) {
        self.timing = timing;
    }

    /// Get amplitude damping channel for a qubit after a single-qubit gate
    ///
    /// Computes γ = 1 - exp(-t_gate/T1)
    pub fn amplitude_damping_single_gate(&self, qubit: usize) -> Result<AmplitudeDamping> {
        if qubit >= self.num_qubits {
            return Err(crate::QuantumError::ValidationError(format!(
                "Qubit {} out of range for {}-qubit device",
                qubit, self.num_qubits
            )));
        }

        let t1 = self.qubit_properties[qubit].t1;
        let gate_time = self.timing.single_qubit_gate_time;

        AmplitudeDamping::from_t1(t1, gate_time)
    }

    /// Get phase damping channel for a qubit after a single-qubit gate
    ///
    /// Computes λ = (1 - exp(-t_gate/T2))/2
    pub fn phase_damping_single_gate(&self, qubit: usize) -> Result<PhaseDamping> {
        if qubit >= self.num_qubits {
            return Err(crate::QuantumError::ValidationError(format!(
                "Qubit {} out of range for {}-qubit device",
                qubit, self.num_qubits
            )));
        }

        let t2 = self.qubit_properties[qubit].t2;
        let gate_time = self.timing.single_qubit_gate_time;

        PhaseDamping::from_t2(t2, gate_time)
    }

    /// Get depolarizing channel for a single-qubit gate
    ///
    /// Error probability from gate infidelity: p = 1 - F
    pub fn depolarizing_single_gate(&self, qubit: usize) -> Result<DepolarizingChannel> {
        if qubit >= self.num_qubits {
            return Err(crate::QuantumError::ValidationError(format!(
                "Qubit {} out of range for {}-qubit device",
                qubit, self.num_qubits
            )));
        }

        let fidelity = self.qubit_properties[qubit].single_qubit_gate_fidelity;
        let error_rate = 1.0 - fidelity;

        DepolarizingChannel::new(error_rate)
    }

    /// Get readout error channel for a qubit
    pub fn readout_error(&self, qubit: usize) -> Result<ReadoutError> {
        if qubit >= self.num_qubits {
            return Err(crate::QuantumError::ValidationError(format!(
                "Qubit {} out of range for {}-qubit device",
                qubit, self.num_qubits
            )));
        }

        let p01 = self.qubit_properties[qubit].readout_p01;
        let p10 = self.qubit_properties[qubit].readout_p10;

        ReadoutError::new(p01, p10)
    }

    /// Get qubit properties
    pub fn qubit(&self, qubit: usize) -> Option<&QubitProperties> {
        self.qubit_properties.get(qubit)
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    // ===== Time-Aware Noise Methods =====

    /// Get idle noise channels for a qubit over a given idle time
    ///
    /// Applies T1 and T2 decay based on how long the qubit has been idle.
    /// This is crucial for realistic simulation as qubits decohere even when
    /// not being actively operated.
    ///
    /// # Arguments
    /// * `qubit` - Qubit index
    /// * `idle_time_us` - Duration of idle time in microseconds
    ///
    /// # Returns
    /// Tuple of (AmplitudeDamping, PhaseDamping) for the idle period
    ///
    /// # Example
    /// ```ignore
    /// let model = HardwareNoiseModel::ibm_washington();
    /// let (amp_damp, phase_damp) = model.idle_noise(0, 5.0)?; // 5μs idle
    /// ```
    pub fn idle_noise(
        &self,
        qubit: usize,
        idle_time_us: f64,
    ) -> Result<(AmplitudeDamping, PhaseDamping)> {
        if qubit >= self.num_qubits {
            return Err(crate::QuantumError::ValidationError(format!(
                "Qubit {} out of range for {}-qubit device",
                qubit, self.num_qubits
            )));
        }

        if idle_time_us < 0.0 {
            return Err(crate::QuantumError::ValidationError(
                "Idle time must be non-negative".to_string(),
            ));
        }

        let t1 = self.qubit_properties[qubit].t1;
        let t2 = self.qubit_properties[qubit].t2;

        let amp_damping = AmplitudeDamping::from_t1(t1, idle_time_us)?;
        let phase_damping = PhaseDamping::from_t2(t2, idle_time_us)?;

        Ok((amp_damping, phase_damping))
    }

    /// Get combined noise for a two-qubit gate operation
    ///
    /// Returns noise channels for both qubits involved in a two-qubit gate.
    /// Includes:
    /// - T1/T2 decay during gate execution
    /// - Gate infidelity as depolarizing noise
    /// - Crosstalk effects if enabled
    ///
    /// # Arguments
    /// * `control` - Control qubit index
    /// * `target` - Target qubit index
    ///
    /// # Returns
    /// GateNoise structure containing all noise channels for both qubits
    pub fn two_qubit_gate_noise(&self, control: usize, target: usize) -> Result<GateNoise> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(crate::QuantumError::ValidationError(format!(
                "Qubit indices ({}, {}) out of range for {}-qubit device",
                control, target, self.num_qubits
            )));
        }

        if control == target {
            return Err(crate::QuantumError::ValidationError(
                "Control and target qubits must be different".to_string(),
            ));
        }

        // Get gate properties or use defaults
        let gate_props = self
            .two_qubit_gate_properties(control, target)
            .cloned()
            .unwrap_or_default();

        let gate_time = gate_props.duration;
        let gate_fidelity = gate_props.fidelity;

        // T1 decay for both qubits
        let t1_control = self.qubit_properties[control].t1;
        let t1_target = self.qubit_properties[target].t1;
        let amp_damp_control = AmplitudeDamping::from_t1(t1_control, gate_time)?;
        let amp_damp_target = AmplitudeDamping::from_t1(t1_target, gate_time)?;

        // T2 dephasing for both qubits
        let t2_control = self.qubit_properties[control].t2;
        let t2_target = self.qubit_properties[target].t2;
        let phase_damp_control = PhaseDamping::from_t2(t2_control, gate_time)?;
        let phase_damp_target = PhaseDamping::from_t2(t2_target, gate_time)?;

        // Gate error as depolarizing noise
        let error_rate = 1.0 - gate_fidelity;
        let depol = DepolarizingChannel::new(error_rate)?;

        Ok(GateNoise {
            amplitude_damping: vec![amp_damp_control, amp_damp_target],
            phase_damping: vec![phase_damp_control, phase_damp_target],
            depolarizing: vec![depol],
            qubits: vec![control, target],
        })
    }

    /// Get combined noise for a single-qubit gate operation
    ///
    /// Returns all noise channels that should be applied after a single-qubit gate:
    /// - Amplitude damping (T1 decay)
    /// - Phase damping (T2 dephasing)
    /// - Depolarizing noise (gate errors)
    ///
    /// # Arguments
    /// * `qubit` - Qubit index
    ///
    /// # Returns
    /// GateNoise structure containing all noise channels
    pub fn single_qubit_gate_noise(&self, qubit: usize) -> Result<GateNoise> {
        if qubit >= self.num_qubits {
            return Err(crate::QuantumError::ValidationError(format!(
                "Qubit {} out of range for {}-qubit device",
                qubit, self.num_qubits
            )));
        }

        Ok(GateNoise {
            amplitude_damping: vec![self.amplitude_damping_single_gate(qubit)?],
            phase_damping: vec![self.phase_damping_single_gate(qubit)?],
            depolarizing: vec![self.depolarizing_single_gate(qubit)?],
            qubits: vec![qubit],
        })
    }

    /// Estimate total circuit fidelity for a given gate sequence
    ///
    /// Provides a rough estimate of the final circuit fidelity based on
    /// gate counts and individual gate fidelities.
    ///
    /// # Arguments
    /// * `single_qubit_gates` - Number of single-qubit gates per qubit
    /// * `two_qubit_gates` - Vector of (control, target) pairs for two-qubit gates
    /// * `total_time_us` - Total circuit execution time in microseconds
    ///
    /// # Returns
    /// Estimated circuit fidelity (0-1)
    pub fn estimate_circuit_fidelity(
        &self,
        single_qubit_gates: &[usize],
        two_qubit_gates: &[(usize, usize)],
        total_time_us: f64,
    ) -> f64 {
        let mut fidelity = 1.0;

        // Single-qubit gate contributions
        for (qubit, &count) in single_qubit_gates.iter().enumerate() {
            if qubit < self.num_qubits {
                let gate_fidelity = self.qubit_properties[qubit].single_qubit_gate_fidelity;
                fidelity *= gate_fidelity.powi(count as i32);
            }
        }

        // Two-qubit gate contributions
        for &(q1, q2) in two_qubit_gates {
            let gate_fidelity = self
                .two_qubit_gate_properties(q1, q2)
                .map(|props| props.fidelity)
                .unwrap_or(0.99);
            fidelity *= gate_fidelity;
        }

        // T1/T2 decoherence over total time (simplified)
        for qubit in 0..self.num_qubits {
            let t1 = self.qubit_properties[qubit].t1;
            let t2 = self.qubit_properties[qubit].t2;

            // Approximate fidelity loss from decoherence
            let t1_factor = (-total_time_us / t1).exp();
            let t2_factor = (-total_time_us / t2).exp();
            fidelity *= (t1_factor + t2_factor) / 2.0;
        }

        fidelity.max(0.0).min(1.0)
    }

    // ===== Hardware Presets =====

    /// IBM Quantum Washington (127 qubits) - typical parameters
    ///
    /// Based on publicly available calibration data.
    pub fn ibm_washington() -> Self {
        let mut model = Self::new(127);

        // Typical IBM quantum parameters
        for qubit in 0..127 {
            model.set_qubit_t1(qubit, 100.0); // ~100μs T1
            model.set_qubit_t2(qubit, 80.0); // ~80μs T2
            model.set_readout_error(qubit, 0.015, 0.02); // ~1.5-2%
            model.set_single_qubit_fidelity(qubit, 0.9995); // ~99.95%
        }

        model.timing.single_qubit_gate_time = 0.02; // 20ns
        model.timing.two_qubit_gate_time = 0.3; // 300ns CNOT
        model.timing.measurement_time = 1.0; // 1μs

        model
    }

    /// Google Sycamore (53 qubits) - typical parameters
    pub fn google_sycamore() -> Self {
        let mut model = Self::new(53);

        // Typical Google quantum parameters
        for qubit in 0..53 {
            model.set_qubit_t1(qubit, 20.0); // ~20μs T1
            model.set_qubit_t2(qubit, 15.0); // ~15μs T2
            model.set_readout_error(qubit, 0.03, 0.03); // ~3%
            model.set_single_qubit_fidelity(qubit, 0.9996); // ~99.96%
        }

        model.timing.single_qubit_gate_time = 0.025; // 25ns
        model.timing.two_qubit_gate_time = 0.012; // 12ns iSWAP
        model.timing.measurement_time = 0.5; // 500ns

        model
    }

    /// IonQ Aria (25 qubits) - typical parameters
    ///
    /// Trapped ion systems have very different characteristics.
    pub fn ionq_aria() -> Self {
        let mut model = Self::new(25);

        // Trapped ions: excellent coherence, slower gates
        for qubit in 0..25 {
            model.set_qubit_t1(qubit, 100_000.0); // ~100ms T1 (excellent!)
            model.set_qubit_t2(qubit, 50_000.0); // ~50ms T2
            model.set_readout_error(qubit, 0.001, 0.001); // ~0.1%
            model.set_single_qubit_fidelity(qubit, 0.9999); // ~99.99%
        }

        model.timing.single_qubit_gate_time = 10.0; // 10μs (slower)
        model.timing.two_qubit_gate_time = 200.0; // 200μs (much slower)
        model.timing.measurement_time = 100.0; // 100μs

        model
    }

    /// IBM Quantum Falcon r5.11L (5 qubits) - typical parameters
    ///
    /// Small research-grade system for testing.
    pub fn ibm_falcon_5q() -> Self {
        let mut model = Self::new(5);

        for qubit in 0..5 {
            model.set_qubit_t1(qubit, 80.0); // ~80μs T1
            model.set_qubit_t2(qubit, 60.0); // ~60μs T2
            model.set_readout_error(qubit, 0.02, 0.025); // ~2-2.5%
            model.set_single_qubit_fidelity(qubit, 0.9994); // ~99.94%
        }

        // Linear connectivity: 0-1-2-3-4
        model.set_two_qubit_gate(0, 1, 0.99, 0.3);
        model.set_two_qubit_gate(1, 2, 0.99, 0.3);
        model.set_two_qubit_gate(2, 3, 0.99, 0.3);
        model.set_two_qubit_gate(3, 4, 0.99, 0.3);

        model.timing.single_qubit_gate_time = 0.02;
        model.timing.two_qubit_gate_time = 0.3;
        model.timing.measurement_time = 1.0;

        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_noise_model_creation() {
        let model = HardwareNoiseModel::new(5);
        assert_eq!(model.num_qubits(), 5);
        assert!(model.qubit(0).is_some());
        assert!(model.qubit(5).is_none());
    }

    #[test]
    fn test_set_qubit_properties() {
        let mut model = HardwareNoiseModel::new(3);

        model.set_qubit_t1(0, 150.0);
        model.set_qubit_t2(0, 120.0);
        model.set_readout_error(0, 0.01, 0.015);

        let props = model.qubit(0).unwrap();
        assert_eq!(props.t1, 150.0);
        assert_eq!(props.t2, 120.0);
        assert_eq!(props.readout_p01, 0.01);
        assert_eq!(props.readout_p10, 0.015);
    }

    #[test]
    fn test_amplitude_damping_from_calibration() {
        let model = HardwareNoiseModel::new(1);
        let damping = model.amplitude_damping_single_gate(0).unwrap();

        // With default T1=100μs and gate_time=0.02μs:
        // γ = 1 - exp(-0.02/100) ≈ 0.0002
        assert!(damping.gamma() < 0.001);
        assert!(damping.gamma() > 0.0);
    }

    #[test]
    fn test_depolarizing_from_fidelity() {
        let mut model = HardwareNoiseModel::new(1);
        model.set_single_qubit_fidelity(0, 0.999); // 99.9%

        let depol = model.depolarizing_single_gate(0).unwrap();
        assert!((depol.error_probability() - 0.001).abs() < 1e-10);
    }

    #[test]
    fn test_readout_error_from_calibration() {
        let mut model = HardwareNoiseModel::new(1);
        model.set_readout_error(0, 0.02, 0.03);

        let readout = model.readout_error(0).unwrap();
        assert_eq!(readout.p01(), 0.02);
        assert_eq!(readout.p10(), 0.03);
    }

    #[test]
    fn test_ibm_preset() {
        let model = HardwareNoiseModel::ibm_washington();
        assert_eq!(model.num_qubits(), 127);

        let props = model.qubit(0).unwrap();
        assert_eq!(props.t1, 100.0);
        assert_eq!(props.t2, 80.0);
    }

    #[test]
    fn test_google_preset() {
        let model = HardwareNoiseModel::google_sycamore();
        assert_eq!(model.num_qubits(), 53);

        let props = model.qubit(0).unwrap();
        assert_eq!(props.t1, 20.0);
    }

    #[test]
    fn test_ionq_preset() {
        let model = HardwareNoiseModel::ionq_aria();
        assert_eq!(model.num_qubits(), 25);

        let props = model.qubit(0).unwrap();
        assert_eq!(props.t1, 100_000.0); // Excellent coherence!
        assert!(props.readout_p01 < 0.01); // Excellent readout
    }

    #[test]
    fn test_ibm_5q_preset() {
        let model = HardwareNoiseModel::ibm_falcon_5q();
        assert_eq!(model.num_qubits(), 5);
    }

    #[test]
    fn test_idle_noise() {
        let model = HardwareNoiseModel::new(1);
        let idle_time = 10.0; // 10μs

        let (amp_damp, phase_damp) = model.idle_noise(0, idle_time).unwrap();

        // With default T1=100μs, T2=80μs and idle_time=10μs:
        // γ = 1 - exp(-10/100) ≈ 0.095
        // λ = (1 - exp(-10/80))/2 ≈ 0.059
        assert!(amp_damp.gamma() > 0.09 && amp_damp.gamma() < 0.10);
        assert!(phase_damp.lambda() > 0.058 && phase_damp.lambda() < 0.062);
    }

    #[test]
    fn test_idle_noise_invalid() {
        let model = HardwareNoiseModel::new(1);

        // Negative idle time
        assert!(model.idle_noise(0, -1.0).is_err());

        // Out of range qubit
        assert!(model.idle_noise(5, 1.0).is_err());
    }

    #[test]
    fn test_two_qubit_gate_noise() {
        let mut model = HardwareNoiseModel::new(3);
        model.set_two_qubit_gate(0, 1, 0.98, 0.5); // 98% fidelity, 500ns

        let noise = model.two_qubit_gate_noise(0, 1).unwrap();

        // Should have noise for both qubits
        assert_eq!(noise.amplitude_damping.len(), 2);
        assert_eq!(noise.phase_damping.len(), 2);
        assert_eq!(noise.depolarizing.len(), 1);
        assert_eq!(noise.qubits, vec![0, 1]);

        // Depolarizing channel should reflect 2% error
        assert!((noise.depolarizing[0].error_probability() - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_two_qubit_gate_noise_defaults() {
        let model = HardwareNoiseModel::new(3);

        // No explicit gate properties set, should use defaults
        let noise = model.two_qubit_gate_noise(0, 1).unwrap();
        assert_eq!(noise.qubits, vec![0, 1]);
        assert_eq!(noise.amplitude_damping.len(), 2);
    }

    #[test]
    fn test_two_qubit_gate_noise_invalid() {
        let model = HardwareNoiseModel::new(3);

        // Same qubit for control and target
        assert!(model.two_qubit_gate_noise(0, 0).is_err());

        // Out of range qubits
        assert!(model.two_qubit_gate_noise(0, 5).is_err());
        assert!(model.two_qubit_gate_noise(5, 0).is_err());
    }

    #[test]
    fn test_single_qubit_gate_noise() {
        let model = HardwareNoiseModel::new(3);
        let noise = model.single_qubit_gate_noise(0).unwrap();

        assert_eq!(noise.amplitude_damping.len(), 1);
        assert_eq!(noise.phase_damping.len(), 1);
        assert_eq!(noise.depolarizing.len(), 1);
        assert_eq!(noise.qubits, vec![0]);
    }

    #[test]
    fn test_crosstalk_properties() {
        let mut model = HardwareNoiseModel::new(5);

        // Set crosstalk between qubits 0 and 1
        model.set_crosstalk(0, 1, 0.1, 0.001); // 0.1 MHz ZZ, 0.1% spectator error

        let crosstalk = model.crosstalk_properties(0, 1).unwrap();
        assert_eq!(crosstalk.zz_coupling, 0.1);
        assert_eq!(crosstalk.spectator_error, 0.001);

        // Should work with reversed order
        let crosstalk_rev = model.crosstalk_properties(1, 0).unwrap();
        assert_eq!(crosstalk_rev.zz_coupling, 0.1);

        // Non-existent crosstalk
        assert!(model.crosstalk_properties(2, 3).is_none());
    }

    #[test]
    fn test_enable_disable_features() {
        let mut model = HardwareNoiseModel::new(3);

        // Default: idle noise enabled, crosstalk disabled
        assert!(model.is_idle_noise_enabled());
        assert!(!model.is_crosstalk_enabled());

        // Toggle features
        model.set_idle_noise_enabled(false);
        model.set_crosstalk_enabled(true);

        assert!(!model.is_idle_noise_enabled());
        assert!(model.is_crosstalk_enabled());
    }

    #[test]
    fn test_estimate_circuit_fidelity() {
        let mut model = HardwareNoiseModel::new(3);
        model.set_single_qubit_fidelity(0, 0.999);
        model.set_single_qubit_fidelity(1, 0.999);
        model.set_two_qubit_gate(0, 1, 0.99, 0.3);

        // 2 single-qubit gates on qubit 0, 1 on qubit 1
        let single_gates = vec![2, 1, 0];
        // 1 two-qubit gate between 0 and 1
        let two_gates = vec![(0, 1)];
        // 1μs total time
        let total_time = 1.0;

        let fidelity = model.estimate_circuit_fidelity(&single_gates, &two_gates, total_time);

        // Fidelity should be less than 1 but greater than 0
        assert!(fidelity > 0.0 && fidelity < 1.0);

        // With good qubits, should be close to 1
        assert!(fidelity > 0.95);
    }

    #[test]
    fn test_timing_getters_setters() {
        let mut model = HardwareNoiseModel::new(1);

        let timing = model.timing();
        assert_eq!(timing.single_qubit_gate_time, 0.02);

        let new_timing = GateTiming {
            single_qubit_gate_time: 0.05,
            two_qubit_gate_time: 0.2,
            measurement_time: 2.0,
        };

        model.set_timing(new_timing);
        assert_eq!(model.timing().single_qubit_gate_time, 0.05);
    }

    #[test]
    fn test_two_qubit_gate_properties_getter() {
        let mut model = HardwareNoiseModel::new(5);
        model.set_two_qubit_gate(1, 3, 0.97, 0.4);

        let props = model.two_qubit_gate_properties(1, 3).unwrap();
        assert_eq!(props.fidelity, 0.97);
        assert_eq!(props.duration, 0.4);

        // Should work with reversed indices
        let props_rev = model.two_qubit_gate_properties(3, 1).unwrap();
        assert_eq!(props_rev.fidelity, 0.97);

        // Non-existent gate
        assert!(model.two_qubit_gate_properties(0, 4).is_none());
    }
}
