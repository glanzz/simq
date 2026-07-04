use pyo3::prelude::*;
use simq_gates::standard as gates;
use std::sync::Arc;
use simq_core::Gate;
use num_complex::Complex64;

/// Helper macro to define a Python class for a parameterized gate
macro_rules! define_param_gate {
    ($py_name:ident, $rust_name:ident, $doc:expr, $num_params:expr) => {
        #[pyclass(name = $doc)]
        #[derive(Clone)]
        pub struct $py_name {
            pub inner: Arc<dyn Gate>,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new(theta: f64) -> Self {
                Self {
                    inner: Arc::new(gates::$rust_name::new(theta)),
                }
            }

            fn name(&self) -> String {
                self.inner.name().to_string()
            }

            fn num_qubits(&self) -> usize {
                self.inner.num_qubits()
            }
            
            fn matrix(&self) -> PyResult<Vec<Vec<Complex64>>> {
                let flat_matrix = self.inner.matrix().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Gate does not have a matrix representation")
                })?;
                
                let size = (flat_matrix.len() as f64).sqrt() as usize;
                let mut result = Vec::with_capacity(size);
                for i in 0..size {
                    let mut row = Vec::with_capacity(size);
                    for j in 0..size {
                        row.push(flat_matrix[i * size + j]);
                    }
                    result.push(row);
                }
                Ok(result)
            }
        }
    };
}

// Single-qubit parameterized gates
define_param_gate!(PyRX, RotationX, "RXGate", 1);
define_param_gate!(PyRY, RotationY, "RYGate", 1);
define_param_gate!(PyRZ, RotationZ, "RZGate", 1);
define_param_gate!(PyPhase, Phase, "PhaseGate", 1);

// U3 Gate (3 parameters)
#[pyclass(name = "U3Gate")]
#[derive(Clone)]
pub struct PyU3 {
    pub inner: Arc<dyn Gate>,
}

#[pymethods]
impl PyU3 {
    #[new]
    fn new(theta: f64, phi: f64, lambda: f64) -> Self {
        Self {
            inner: Arc::new(gates::U3::new(theta, phi, lambda)),
        }
    }

    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }
    
    fn matrix(&self) -> PyResult<Vec<Vec<Complex64>>> {
        let flat_matrix = self.inner.matrix().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Gate does not have a matrix representation")
        })?;
        
        let size = (flat_matrix.len() as f64).sqrt() as usize;
        let mut result = Vec::with_capacity(size);
        for i in 0..size {
            let mut row = Vec::with_capacity(size);
            for j in 0..size {
                row.push(flat_matrix[i * size + j]);
            }
            result.push(row);
        }
        Ok(result)
    }
}

// Two-qubit parameterized gates
define_param_gate!(PyRXX, RXX, "RXXGate", 1);
define_param_gate!(PyRYY, RYY, "RYYGate", 1);
define_param_gate!(PyRZZ, RZZ, "RZZGate", 1);
define_param_gate!(PyCPhase, CPhase, "CPhaseGate", 1);

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRX>()?;
    m.add_class::<PyRY>()?;
    m.add_class::<PyRZ>()?;
    m.add_class::<PyPhase>()?;
    m.add_class::<PyU3>()?;
    
    m.add_class::<PyRXX>()?;
    m.add_class::<PyRYY>()?;
    m.add_class::<PyRZZ>()?;
    m.add_class::<PyCPhase>()?;
    
    Ok(())
}
