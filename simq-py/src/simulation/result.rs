use pyo3::prelude::*;
use simq_state::AdaptiveState;
use numpy::PyArray1;
use num_complex::Complex64;

#[pyclass(name = "SimulationResult")]
pub struct PySimulationResult {
    pub state: AdaptiveState,
}

#[pymethods]
impl PySimulationResult {
    #[getter]
    fn state_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        let dense_vec = self.state.to_dense_vec();
        PyArray1::from_vec_bound(py, dense_vec)
    }

    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let num_qubits = self.state.num_qubits();
        let dim = 1 << num_qubits;
        let mut probs = Vec::with_capacity(dim);
        
        for i in 0..dim {
            probs.push(self.state.get_probability(i).unwrap_or(0.0));
        }
        
        Ok(PyArray1::from_vec_bound(py, probs))
    }
}

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySimulationResult>()?;
    Ok(())
}
