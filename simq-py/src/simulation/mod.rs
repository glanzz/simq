use pyo3::prelude::*;

pub mod simulator;
pub mod result;

pub fn register(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let simulation_module = PyModule::new_bound(py, "simulation")?;
    
    simulator::register(py, &simulation_module)?;
    result::register(py, &simulation_module)?;
    
    m.add_submodule(&simulation_module)?;
    Ok(())
}
