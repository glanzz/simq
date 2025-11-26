//! Python bindings for circuit compilation and optimization

use pyo3::prelude::*;
use simq_compiler::{create_compiler, OptimizationLevel, Compiler, CircuitAnalysis};
use std::sync::Arc;
use crate::core::circuit::PyCircuit;

#[pyclass(name = "OptimizationLevel")]
#[derive(Clone, Copy)]
pub enum PyOptimizationLevel {
    O0 = 0,
    O1 = 1,
    O2 = 2,
    O3 = 3,
}

impl From<PyOptimizationLevel> for OptimizationLevel {
    fn from(level: PyOptimizationLevel) -> Self {
        match level {
            PyOptimizationLevel::O0 => OptimizationLevel::O0,
            PyOptimizationLevel::O1 => OptimizationLevel::O1,
            PyOptimizationLevel::O2 => OptimizationLevel::O2,
            PyOptimizationLevel::O3 => OptimizationLevel::O3,
        }
    }
}

#[pyclass(name = "Compiler")]
pub struct PyCompiler {
    inner: Compiler,
}

#[pymethods]
impl PyCompiler {
    #[new]
    #[pyo3(signature = (level=PyOptimizationLevel::O2))]
    fn new(level: PyOptimizationLevel) -> Self {
        let compiler = create_compiler(level.into());
        Self {
            inner: compiler,
        }
    }

    fn compile(&self, py: Python, circuit: &PyCircuit) -> PyResult<PyCircuit> {
        // Clone the circuit to avoid modifying the original
        let mut cloned_circuit = circuit.inner().clone();
        
        // Run compilation
        // We release GIL because compilation might be expensive
        py.allow_threads(|| {
            self.inner.compile(&mut cloned_circuit)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        // Return new PyCircuit
        Ok(PyCircuit::from(cloned_circuit))
    }
}

#[pyclass(name = "CircuitAnalysis")]
pub struct PyCircuitAnalysis {
    inner: CircuitAnalysis,
}

#[pymethods]
impl PyCircuitAnalysis {
    #[staticmethod]
    fn analyze(circuit: &PyCircuit) -> PyResult<Self> {
        let analysis = CircuitAnalysis::analyze(circuit.inner())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner: analysis })
    }
    
    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }
    
    #[getter]
    fn gate_count(&self) -> usize {
        self.inner.statistics.total_gates
    }
    
    #[getter]
    fn depth(&self) -> usize {
        self.inner.statistics.depth
    }
    
    #[getter]
    fn multi_qubit_gate_count(&self) -> usize {
        self.inner.statistics.multi_qubit_gates
    }
    
    #[getter]
    fn estimated_time_us(&self) -> f64 {
        self.inner.resources.estimated_time_us
    }
    
    #[getter]
    fn dense_memory_bytes(&self) -> usize {
        self.inner.resources.dense_memory_bytes
    }
    
    #[getter]
    fn parallelism_factor(&self) -> f64 {
        self.inner.parallelism_factor()
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOptimizationLevel>()?;
    m.add_class::<PyCompiler>()?;
    m.add_class::<PyCircuitAnalysis>()?;
    Ok(())
}
