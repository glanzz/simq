//! GPU backend for Ferriq using wgpu
//!
//! **Status: not implemented.** No wgpu device initialization, kernel
//! dispatch, or read-back has been written yet. Every entry point returns an
//! error so callers cannot mistake CPU execution (or, worse, *no* execution)
//! for GPU acceleration. An earlier version of this module dispatched a
//! compute pass and then returned `Ok` while leaving the state untouched —
//! that silent no-op is exactly what this module now refuses to do.
//!
//! The WGSL kernels under `shaders/` are kept as a starting point for a real
//! implementation.

#[cfg(feature = "gpu")]
use num_complex::Complex;

/// Error message shared by every unimplemented GPU entry point
#[cfg(feature = "gpu")]
const GPU_UNIMPLEMENTED: &str = "GPU backend is not implemented: gate kernels have no device \
     initialization or result read-back; run with the CPU execution engine instead";

#[cfg(feature = "gpu")]
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// GPU contexts cannot be constructed yet.
    pub fn new() -> Result<Self, String> {
        Err(GPU_UNIMPLEMENTED.to_string())
    }

    /// Unimplemented: fails loudly instead of returning `Ok` with an
    /// unmodified state.
    pub fn apply_single_qubit_dense_gpu(
        &self,
        _gate: [[Complex<f64>; 2]; 2],
        _qubit: usize,
        _state: &mut [Complex<f64>],
    ) -> Result<(), String> {
        Err(GPU_UNIMPLEMENTED.to_string())
    }
}

#[cfg(not(feature = "gpu"))]
#[derive(Clone)]
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    pub fn new() -> Result<Self, String> {
        Err(
            "GPU backend not enabled (build with the `gpu` feature; note the backend itself is \
             also not implemented yet)"
                .to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_context_construction_fails_loudly() {
        let err = GpuContext::new()
            .err()
            .expect("GPU context must not construct");
        assert!(err.contains("not"), "error should explain unavailability: {}", err);
    }
}
