use pyo3::prelude::*;
use simq_gates::custom::CustomGate;
use simq_core::Gate;
use std::sync::Arc;
use num_complex::Complex64;

#[pyclass(name = "CustomGate")]
#[derive(Clone)]
pub struct PyCustomGate {
    pub inner: Arc<CustomGate>,
}

#[pymethods]
impl PyCustomGate {
    #[new]
    fn new(name: String, matrix: Vec<Vec<Complex64>>) -> PyResult<Self> {
        let rows = matrix.len();
        if rows == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("Matrix cannot be empty"));
        }
        let cols = matrix[0].len();
        if rows != cols {
            return Err(pyo3::exceptions::PyValueError::new_err("Matrix must be square"));
        }
        
        // Check if dimension is a power of 2
        if !rows.is_power_of_two() {
            return Err(pyo3::exceptions::PyValueError::new_err("Matrix dimension must be a power of 2"));
        }
        
        let num_qubits = rows.trailing_zeros() as usize;

        let mut data = Vec::with_capacity(rows * cols);
        for row in matrix {
            if row.len() != cols {
                return Err(pyo3::exceptions::PyValueError::new_err("All rows must have the same length"));
            }
            data.extend(row);
        }

        let gate = CustomGate::new(name, num_qubits, data, 1e-10)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(gate),
        })
    }

    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }
    
    fn matrix(&self) -> PyResult<Vec<Vec<Complex64>>> {
        let data = self.inner.matrix_vec();
        let size = (data.len() as f64).sqrt() as usize;
        
        let mut result = Vec::with_capacity(size);
        for i in 0..size {
            let mut row = Vec::with_capacity(size);
            for j in 0..size {
                row.push(data[i * size + j]);
            }
            result.push(row);
        }
        Ok(result)
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCustomGate>()?;
    Ok(())
}
