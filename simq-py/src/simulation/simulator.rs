use pyo3::prelude::*;
use simq_core::Circuit;
use simq_state::AdaptiveState;
use crate::core::circuit::PyCircuit;
use super::result::PySimulationResult;

#[pyclass(name = "SimulatorConfig")]
#[derive(Clone)]
pub struct PySimulatorConfig {
    pub shots: usize,
    pub seed: Option<u64>,
}

#[pymethods]
impl PySimulatorConfig {
    #[new]
    #[pyo3(signature = (shots=1024, seed=None))]
    fn new(shots: usize, seed: Option<u64>) -> Self {
        Self { shots, seed }
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
        let config = config.unwrap_or_else(|| PySimulatorConfig::new(1024, None));
        Self { config }
    }

    fn run(&self, circuit: &PyCircuit) -> PyResult<PySimulationResult> {
        // Create initial state
        let num_qubits = circuit.inner.num_qubits();
        let mut state = AdaptiveState::new(num_qubits)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        // Apply all gates in the circuit
        for gate_op in circuit.inner.operations() {
            let gate = gate_op.gate();
            let qubits = gate_op.qubits();
            
            // Get the gate matrix
            if let Some(matrix_flat) = gate.matrix() {
                let n = qubits.len();
                
                if n == 1 {
                    // Single-qubit gate
                    let matrix = [
                        [matrix_flat[0], matrix_flat[1]],
                        [matrix_flat[2], matrix_flat[3]],
                    ];
                    state.apply_single_qubit_gate(&matrix, qubits[0].into())
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                } else if n == 2 {
                    // Two-qubit gate
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
        }
        
        Ok(PySimulationResult { state })
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulatorConfig>()?;
    m.add_class::<PySimulator>()?;
    Ok(())
}
