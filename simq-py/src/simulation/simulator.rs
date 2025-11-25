use pyo3::prelude::*;
use simq_core::Circuit;
use simq_state::AdaptiveState;
use crate::core::circuit::PyCircuit;
use super::result::PySimulationResult;
use crate::noise::hardware::PyHardwareNoiseModel;
use std::collections::HashMap;
use num_complex::Complex64;
use rand::Rng;
use simq_core::noise::types::NoiseChannel;

#[pyclass(name = "SimulatorConfig")]
#[derive(Clone)]
pub struct PySimulatorConfig {
    pub shots: usize,
    pub seed: Option<u64>,
    pub noise_model: Option<PyHardwareNoiseModel>,
}

#[pymethods]
impl PySimulatorConfig {
    #[new]
    #[pyo3(signature = (shots=1024, seed=None, noise_model=None))]
    fn new(shots: usize, seed: Option<u64>, noise_model: Option<PyHardwareNoiseModel>) -> Self {
        Self { shots, seed, noise_model }
    }
    
    #[getter]
    fn shots(&self) -> usize {
        self.shots
    }
    
    #[setter]
    fn set_shots(&mut self, shots: usize) {
        self.shots = shots;
    }
}

#[pyclass(name = "Simulator")]
pub struct PySimulator {
    config: PySimulatorConfig,
}

#[pymethods]
impl PySimulator {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PySimulatorConfig>) -> Self {
        let config = config.unwrap_or_else(|| PySimulatorConfig::new(1024, None, None));
        Self { config }
    }

    fn run(&self, circuit: &PyCircuit) -> PyResult<PySimulationResult> {
        self.run_single_shot(circuit)
    }

    fn run_with_shots(&self, circuit: &PyCircuit, shots: usize) -> PyResult<HashMap<String, usize>> {
        let mut counts = HashMap::new();
        
        for _ in 0..shots {
            let mut result = self.run_single_shot(circuit)?;
            let outcome = self.measure_all_qubits(&mut result.state)?;
            *counts.entry(outcome).or_insert(0) += 1;
        }
        
        Ok(counts)
    }
}

impl PySimulator {
    fn run_single_shot(&self, circuit: &PyCircuit) -> PyResult<PySimulationResult> {
        let num_qubits = circuit.inner.num_qubits();
        let mut state = AdaptiveState::new(num_qubits)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        for gate_op in circuit.inner.operations() {
            let gate = gate_op.gate();
            let qubits = gate_op.qubits();
            
            // Apply gate
            if let Some(matrix_flat) = gate.matrix() {
                let n = qubits.len();
                
                if n == 1 {
                    let matrix = [
                        [matrix_flat[0], matrix_flat[1]],
                        [matrix_flat[2], matrix_flat[3]],
                    ];
                    state.apply_single_qubit_gate(&matrix, qubits[0].into())
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                } else if n == 2 {
                    let matrix = [
                        [matrix_flat[0], matrix_flat[1], matrix_flat[2], matrix_flat[3]],
                        [matrix_flat[4], matrix_flat[5], matrix_flat[6], matrix_flat[7]],
                        [matrix_flat[8], matrix_flat[9], matrix_flat[10], matrix_flat[11]],
                        [matrix_flat[12], matrix_flat[13], matrix_flat[14], matrix_flat[15]],
                    ];
                    state.apply_two_qubit_gate(&matrix, qubits[0].into(), qubits[1].into())
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                } else {
                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                        format!("{}-qubit gates not yet supported", n)
                    ));
                }
            }
            
            // Apply noise if enabled
            if let Some(noise_model) = &self.config.noise_model {
                self.apply_noise(&mut state, noise_model, qubits)?;
            }
        }
        
        Ok(PySimulationResult { state })
    }

    fn apply_noise(&self, state: &mut AdaptiveState, noise_model: &PyHardwareNoiseModel, qubits: &[simq_core::QubitId]) -> PyResult<()> {
        let n = qubits.len();
        let gate_noise = if n == 1 {
            noise_model.inner.single_qubit_gate_noise(qubits[0].into())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        } else if n == 2 {
            noise_model.inner.two_qubit_gate_noise(qubits[0].into(), qubits[1].into())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        } else {
            return Ok(());
        };

        let mut rng = rand::thread_rng();

        // Apply amplitude damping
        for (i, channel) in gate_noise.amplitude_damping.iter().enumerate() {
            let qubit: usize = gate_noise.qubits[i];
            let gamma = channel.gamma();
            
            let (_p0, p1) = state.measure_probability(qubit)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
            let p_jump = gamma * p1;
            let kraus_ops = channel.kraus_operators();
            let op_idx = if rng.gen::<f64>() < p_jump { 1 } else { 0 };
            let op = &kraus_ops[op_idx];
            
            self.apply_kraus_op(state, qubit, op)?;
        }
        
        // Apply phase damping
        for (i, channel) in gate_noise.phase_damping.iter().enumerate() {
            let qubit: usize = gate_noise.qubits[i];
            let lambda = channel.lambda();
            let kraus_ops = channel.kraus_operators();
            let op_idx = if rng.gen::<f64>() < lambda { 1 } else { 0 };
            let op = &kraus_ops[op_idx];
            self.apply_kraus_op(state, qubit, op)?;
        }
        
        // Apply depolarizing
        for (i, channel) in gate_noise.depolarizing.iter().enumerate() {
            let qubit_idx = if gate_noise.depolarizing.len() == gate_noise.qubits.len() {
                i
            } else {
                gate_noise.qubits.len() - 1 
            };
            
            let qubit = gate_noise.qubits[qubit_idx];
            let p = channel.error_probability();
            let r = rng.gen::<f64>();
            let op_idx = if r < 1.0 - p {
                0 
            } else if r < 1.0 - p + p/3.0 {
                1 
            } else if r < 1.0 - p + 2.0*p/3.0 {
                2 
            } else {
                3 
            };
            
            let kraus_ops = channel.kraus_operators();
            let op = &kraus_ops[op_idx];
            self.apply_kraus_op(state, qubit, op)?;
        }
        
        Ok(())
    }

    fn apply_kraus_op(&self, state: &mut AdaptiveState, qubit: usize, op: &simq_core::noise::types::KrausOperator) -> PyResult<()> {
        let data = &op.matrix;
        let matrix = [
            [data[0], data[1]],
            [data[2], data[3]],
        ];
        
        state.apply_single_qubit_gate(&matrix, qubit)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
        state.normalize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
        Ok(())
    }

    fn measure_all_qubits(&self, state: &mut AdaptiveState) -> PyResult<String> {
        let num_qubits = state.num_qubits();
        let mut outcome = String::with_capacity(num_qubits);
        let mut rng = rand::thread_rng();
        
        for i in 0..num_qubits {
            let mut bit = state.measure_qubit(i, rng.gen())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
            // Apply readout error if noise model is present
            if let Some(noise_model) = &self.config.noise_model {
                if let Ok(readout) = noise_model.inner.readout_error(i) {
                     let p01 = readout.p01();
                     let p10 = readout.p10();
                     
                     let r: f64 = rng.gen();
                     if bit == 0 {
                         if r < p01 { bit = 1; }
                     } else {
                         if r < p10 { bit = 0; }
                     }
                }
            }

            outcome.push(if bit == 1 { '1' } else { '0' });
        }
        
        // Reverse to match standard notation (q_n ... q_0)
        Ok(outcome.chars().rev().collect())
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulatorConfig>()?;
    m.add_class::<PySimulator>()?;
    Ok(())
}
