use pyo3::prelude::*;

pub mod channels;
pub mod hardware;

pub fn register(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let noise_module = PyModule::new_bound(_py, "noise")?;
    
    channels::register(_py, &noise_module)?;
    hardware::register(_py, &noise_module)?;
    
    m.add_submodule(&noise_module)?;
    Ok(())
}
