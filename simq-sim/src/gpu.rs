//! GPU backend for SimQ using wgpu

#[cfg(feature = "gpu")]

#[cfg(feature = "gpu")]
use wgpu;
use num_complex::Complex;

#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    // NOTE: This should be async in real use, but for now, provide a sync stub for integration
    pub fn new() -> Result<Self, String> {
        // TODO: Implement actual wgpu initialization (async)
        Err("GPU backend not yet implemented".to_string())
    }

    pub fn apply_single_qubit_dense_gpu(
        &self,
        _gate: [[Complex<f64>; 2]; 2],
        _qubit: usize,
        _state: &mut [Complex<f64>],
    ) -> Result<(), String> {
        // TODO: Implement GPU kernel using wgpu compute shader
        Err("GPU kernel not yet implemented".to_string())
    }
}

#[cfg(not(feature = "gpu"))]
#[derive(Clone)]
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    pub fn new() -> Result<Self, String> {
        Err("GPU backend not enabled".to_string())
    }
}
