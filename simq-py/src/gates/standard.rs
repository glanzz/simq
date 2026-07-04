use pyo3::prelude::*;
use simq_gates::standard as gates;
use std::sync::Arc;
use simq_core::Gate;
use num_complex::Complex64;

/// Helper macro to define a Python class for a standard gate
macro_rules! define_gate {
    ($py_name:ident, $rust_name:ident, $doc:expr) => {
        #[pyclass(name = $doc)]
        #[derive(Clone)]
        pub struct $py_name {
            pub inner: Arc<dyn Gate>,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            fn new() -> Self {
                Self {
                    inner: Arc::new(gates::$rust_name),
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

// Single-qubit gates
define_gate!(PyHadamard, Hadamard, "HGate");
define_gate!(PyPauliX, PauliX, "XGate");
define_gate!(PyPauliY, PauliY, "YGate");
define_gate!(PyPauliZ, PauliZ, "ZGate");
define_gate!(PyS, SGate, "SGate");
define_gate!(PySdg, SGateDagger, "SdgGate");
define_gate!(PyT, TGate, "TGate");
define_gate!(PyTdg, TGateDagger, "TdgGate");
define_gate!(PySX, SXGate, "SXGate");
define_gate!(PySXdg, SXGateDagger, "SXdgGate");

// Two-qubit gates
define_gate!(PyCNot, CNot, "CXGate");
define_gate!(PyCZ, CZ, "CZGate");
// define_gate!(PyCY, CY, "CYGate");
// define_gate!(PyCH, CH, "CHGate");
define_gate!(PySwap, Swap, "SwapGate");
define_gate!(PyISwap, ISwap, "iSwapGate");
define_gate!(PyECR, ECR, "ECRGate");

// Three-qubit gates
define_gate!(PyToffoli, Toffoli, "ToffoliGate");
define_gate!(PyFredkin, Fredkin, "FredkinGate");

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHadamard>()?;
    m.add_class::<PyPauliX>()?;
    m.add_class::<PyPauliY>()?;
    m.add_class::<PyPauliZ>()?;
    m.add_class::<PyS>()?;
    m.add_class::<PySdg>()?;
    m.add_class::<PyT>()?;
    m.add_class::<PyTdg>()?;
    m.add_class::<PySX>()?;
    m.add_class::<PySXdg>()?;
    
    m.add_class::<PyCNot>()?;
    m.add_class::<PyCZ>()?;
    // m.add_class::<PyCY>()?;
    // m.add_class::<PyCH>()?;
    m.add_class::<PySwap>()?;
    m.add_class::<PyISwap>()?;
    m.add_class::<PyECR>()?;
    
    m.add_class::<PyToffoli>()?;
    m.add_class::<PyFredkin>()?;
    
    Ok(())
}
