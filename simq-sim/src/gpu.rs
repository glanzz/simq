//! GPU backend for SimQ using wgpu.
//!
//! Real `wgpu` device acquisition, compute-pipeline construction, and
//! gate dispatch/read-back for single- and two-qubit dense gates. Device,
//! queue, and pipeline objects are constructed once per [`GpuContext`] and
//! reused across every gate dispatch — never rebuilt per gate, since
//! `wgpu` pipeline/device construction costs orders of magnitude more than
//! a single gate application (see the companion TDD's §4).
//!
//! # No silent fallback
//!
//! Every failure mode here — no compatible adapter, an adapter lacking
//! double-precision shader support, device request failure, buffer
//! allocation failure — is a hard, descriptive [`Result::Err`]. This module
//! never dispatches a compute pass and returns `Ok` while leaving the state
//! untouched (the specific bug an earlier version of this module had, and
//! the reason every entry point used to unconditionally error instead).
//! [`ExecutionMode::Gpu`](crate::execution_engine::ExecutionMode) has no
//! automatic CPU fallback for a construction or dispatch failure, by
//! design — see that type's docs.
//!
//! # Precision
//!
//! These kernels operate in double precision (`f64`) to match the CPU
//! `DenseState` reference's own `Complex64` arithmetic at the project's
//! 1e-12 cross-validation tolerance, requiring `wgpu::Features::SHADER_F64`
//! (mapped to Vulkan's `shaderFloat64`; unsupported on Metal and most DX12
//! drivers as of `naga` 0.19 — see [`GpuContext::new`]'s docs). An adapter
//! that can't provide it is rejected outright at construction, never
//! silently downgraded to `f32` — the project already has a documented,
//! honest negative result on `f32` precision tradeoffs elsewhere (see
//! `simq-state`'s `single_precision_state` module) and this backend does
//! not repeat that tradeoff invisibly.
//!
//! # What is not yet implemented
//!
//! Only single- and two-qubit dense gates are supported. A `FusedGate`
//! block spanning 3+ qubits (which `simq-compiler`'s multi-qubit fusion can
//! produce at `optimization_level >= 2`) is explicitly out of scope for
//! this pass — see the companion TDD §3.2's "stretch, not required" note —
//! and [`ExecutionEngine::execute_gpu`](crate::execution_engine::ExecutionEngine)
//! returns a descriptive error for it rather than guessing.
//!
//! # Verification status
//!
//! The device-acquisition path (through [`GpuContext::new`] failing when no
//! adapter is available) is exercised by this module's own tests in every
//! environment, including this one. The WGSL kernels themselves
//! (`shaders/single_qubit_gate.wgsl`, `shaders/two_qubit_gate.wgsl`) have
//! **not** been validated against real GPU hardware or driver — no adapter
//! is available in the environment this was written in (`wgpu`'s
//! `request_adapter` genuinely returns `None` here, which is what
//! [`test_gpu_context_construction_fails_loudly_without_hardware`]
//! exercises). Per the companion TDD §8, kernel correctness
//! (TC-B3/TC-B4) and the BENCHMARKS.md entry are explicitly gated on real
//! GPU-equipped CI existing, and must be verified there before being
//! trusted.

#[cfg(feature = "gpu")]
mod real {
    use bytemuck::{Pod, Zeroable};
    use num_complex::Complex64;
    use std::borrow::Cow;

    const SINGLE_QUBIT_SHADER: &str = include_str!("shaders/single_qubit_gate.wgsl");
    const TWO_QUBIT_SHADER: &str = include_str!("shaders/two_qubit_gate.wgsl");
    const WORKGROUP_SIZE: u32 = 64;

    /// Error message prefix shared by every GPU failure mode, so callers
    /// (and tests) can recognize "this is the GPU backend talking" without
    /// string-matching a specific failure reason.
    const GPU_ERROR_PREFIX: &str = "GPU backend";

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct GpuComplex {
        re: f64,
        im: f64,
    }

    /// A live `wgpu` device, queue, and the compute pipelines for this
    /// crate's dense gate kernels. Expensive to construct (adapter/device
    /// negotiation, shader compilation); construct once and reuse across
    /// every gate dispatch in a run — see module docs.
    pub struct GpuContext {
        pub device: wgpu::Device,
        pub queue: wgpu::Queue,
        single_qubit_pipeline: wgpu::ComputePipeline,
        single_qubit_layout: wgpu::BindGroupLayout,
        two_qubit_pipeline: wgpu::ComputePipeline,
        two_qubit_layout: wgpu::BindGroupLayout,
    }

    impl GpuContext {
        /// Acquires a `wgpu` adapter/device and builds this backend's
        /// compute pipelines. Fails descriptively (never panics, never
        /// returns `Ok` with a half-initialized context) when:
        /// - no `wgpu`-compatible adapter exists on this machine (no GPU,
        ///   or no Vulkan/Metal/DX12 driver installed);
        /// - the adapter lacks `wgpu::Features::SHADER_F64` (see module
        ///   docs on why f64 is required, not optional, here);
        /// - device request itself fails (e.g. resource limits).
        pub fn new() -> Result<Self, String> {
            pollster::block_on(Self::new_async())
        }

        async fn new_async() -> Result<Self, String> {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .ok_or_else(|| {
                    format!(
                        "{GPU_ERROR_PREFIX}: no wgpu-compatible adapter found on this machine \
                         (no GPU, or no Vulkan/Metal/DX12 driver installed); run with the CPU \
                         execution engine instead"
                    )
                })?;

            if !adapter.features().contains(wgpu::Features::SHADER_F64) {
                let info = adapter.get_info();
                return Err(format!(
                    "{GPU_ERROR_PREFIX}: adapter '{}' ({:?}) does not support double-precision \
                     compute shaders (wgpu::Features::SHADER_F64). SimQ's GPU kernels require \
                     f64 to match the CPU DenseState reference at this project's 1e-12 \
                     cross-validation tolerance; there is no f32 fallback. Run with the CPU \
                     execution engine instead.",
                    info.name, info.backend
                ));
            }

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("simq-gpu-device"),
                        required_features: wgpu::Features::SHADER_F64,
                        required_limits: wgpu::Limits::default(),
                    },
                    None,
                )
                .await
                .map_err(|e| format!("{GPU_ERROR_PREFIX}: device request failed: {e}"))?;

            let (single_qubit_pipeline, single_qubit_layout) =
                build_pipeline(&device, SINGLE_QUBIT_SHADER, "simq_single_qubit_gate", 3);
            let (two_qubit_pipeline, two_qubit_layout) =
                build_pipeline(&device, TWO_QUBIT_SHADER, "simq_two_qubit_gate", 3);

            Ok(Self {
                device,
                queue,
                single_qubit_pipeline,
                single_qubit_layout,
                two_qubit_pipeline,
                two_qubit_layout,
            })
        }

        /// Applies a single-qubit dense gate to `state` on the GPU,
        /// blocking until the result is read back into `state`. `state.len()`
        /// must be a power of two and `qubit` must be a valid bit position
        /// for it.
        pub fn apply_single_qubit_dense_gpu(
            &self,
            gate: [[Complex64; 2]; 2],
            qubit: usize,
            state: &mut [Complex64],
        ) -> Result<(), String> {
            let dim = state.len();
            validate_dense_state(dim, GPU_ERROR_PREFIX)?;
            let num_qubits = dim.trailing_zeros() as usize;
            if qubit >= num_qubits {
                return Err(format!(
                    "{GPU_ERROR_PREFIX}: qubit index {qubit} out of range for a {num_qubits}-qubit state"
                ));
            }

            let gate_flat: [f64; 8] = [
                gate[0][0].re, gate[0][0].im, gate[0][1].re, gate[0][1].im,
                gate[1][0].re, gate[1][0].im, gate[1][1].re, gate[1][1].im,
            ];
            let params: [u32; 1] = [qubit as u32];
            let workgroups = ceil_div((dim / 2) as u32, WORKGROUP_SIZE);

            self.dispatch(
                &self.single_qubit_pipeline,
                &self.single_qubit_layout,
                &gate_flat,
                &params,
                workgroups,
                state,
            )
        }

        /// Applies a two-qubit dense gate to `state` on the GPU, blocking
        /// until the result is read back into `state`. The 4x4 `gate`
        /// matrix uses the same argument-order convention as
        /// `simq_state::DenseState::apply_two_qubit_gate`: local basis
        /// index `2*bit(qubit1) + bit(qubit2)`.
        pub fn apply_two_qubit_dense_gpu(
            &self,
            gate: [[Complex64; 4]; 4],
            qubit1: usize,
            qubit2: usize,
            state: &mut [Complex64],
        ) -> Result<(), String> {
            let dim = state.len();
            validate_dense_state(dim, GPU_ERROR_PREFIX)?;
            let num_qubits = dim.trailing_zeros() as usize;
            if qubit1 >= num_qubits || qubit2 >= num_qubits {
                return Err(format!(
                    "{GPU_ERROR_PREFIX}: qubit indices ({qubit1}, {qubit2}) out of range for a \
                     {num_qubits}-qubit state"
                ));
            }
            if qubit1 == qubit2 {
                return Err(format!(
                    "{GPU_ERROR_PREFIX}: qubit1 and qubit2 must differ, both were {qubit1}"
                ));
            }
            if dim < 4 {
                return Err(format!(
                    "{GPU_ERROR_PREFIX}: a two-qubit gate needs at least 2 qubits of state"
                ));
            }

            let mut gate_flat = [0.0f64; 32];
            for (r, row) in gate.iter().enumerate() {
                for (c, entry) in row.iter().enumerate() {
                    let base = (r * 4 + c) * 2;
                    gate_flat[base] = entry.re;
                    gate_flat[base + 1] = entry.im;
                }
            }
            let params: [u32; 2] = [qubit1 as u32, qubit2 as u32];
            let workgroups = ceil_div((dim / 4) as u32, WORKGROUP_SIZE);

            self.dispatch(
                &self.two_qubit_pipeline,
                &self.two_qubit_layout,
                &gate_flat,
                &params,
                workgroups,
                state,
            )
        }

        /// Shared upload -> dispatch -> read-back path for both kernels.
        /// `gate_bytes`/`param_bytes` are uploaded as `storage, read`
        /// buffers at bindings 1/2 respectively (see the WGSL sources);
        /// `state` is uploaded/downloaded at binding 0.
        fn dispatch<G: Pod, P: Pod>(
            &self,
            pipeline: &wgpu::ComputePipeline,
            layout: &wgpu::BindGroupLayout,
            gate_data: &[G],
            param_data: &[P],
            workgroups: u32,
            state: &mut [Complex64],
        ) -> Result<(), String> {
            if workgroups == 0 {
                // A 1-qubit state has no partner amplitude for a
                // single-qubit gate's stride-based indexing; nothing to
                // dispatch, and the caller's dim>=2 check already rejects
                // this — reachable only for a defensively-sized 0 case.
                return Ok(());
            }

            let device = &self.device;
            let queue = &self.queue;

            let gpu_state: Vec<GpuComplex> =
                state.iter().map(|c| GpuComplex { re: c.re, im: c.im }).collect();
            let state_bytes = bytemuck::cast_slice(&gpu_state);
            let state_size = state_bytes.len() as wgpu::BufferAddress;

            let state_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("simq-gpu-state"),
                size: state_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&state_buf, 0, state_bytes);

            let gate_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("simq-gpu-gate"),
                size: std::mem::size_of_val(gate_data) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&gate_buf, 0, bytemuck::cast_slice(gate_data));

            let param_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("simq-gpu-params"),
                size: std::mem::size_of_val(param_data) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&param_buf, 0, bytemuck::cast_slice(param_data));

            let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("simq-gpu-readback"),
                size: state_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("simq-gpu-bind-group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: state_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: gate_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: param_buf.as_entire_binding() },
                ],
            });

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("simq-gpu-encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("simq-gpu-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&state_buf, 0, &staging_buf, 0, state_size);
            queue.submit(Some(encoder.finish()));

            let (tx, rx) = std::sync::mpsc::channel();
            staging_buf.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .map_err(|e| format!("{GPU_ERROR_PREFIX}: read-back channel closed: {e}"))?
                .map_err(|e| format!("{GPU_ERROR_PREFIX}: read-back mapping failed: {e}"))?;

            {
                let view = staging_buf.slice(..).get_mapped_range();
                let read_back: &[GpuComplex] = bytemuck::cast_slice(&view);
                if read_back.len() != state.len() {
                    return Err(format!(
                        "{GPU_ERROR_PREFIX}: read-back length mismatch: expected {}, got {}",
                        state.len(),
                        read_back.len()
                    ));
                }
                for (dst, src) in state.iter_mut().zip(read_back.iter()) {
                    *dst = Complex64::new(src.re, src.im);
                }
            }
            staging_buf.unmap();

            Ok(())
        }
    }

    fn validate_dense_state(dim: usize, prefix: &str) -> Result<(), String> {
        if dim == 0 || (dim & (dim - 1)) != 0 {
            return Err(format!(
                "{prefix}: state length {dim} is not a power of two (dense statevectors must be)"
            ));
        }
        Ok(())
    }

    fn ceil_div(a: u32, b: u32) -> u32 {
        a.div_ceil(b)
    }

    /// Builds a compute pipeline (and its bind group layout: 3 storage
    /// buffer bindings — read_write state, read-only gate, read-only
    /// params) from WGSL source. `label` is used for both the shader
    /// module and the pipeline, for debugger/error-message readability.
    fn build_pipeline(
        device: &wgpu::Device,
        source: &str,
        label: &str,
        num_bindings: u32,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        });

        let mut entries = Vec::with_capacity(num_bindings as usize);
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        for binding in 1..num_bindings {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });

        (pipeline, bind_group_layout)
    }
}

#[cfg(feature = "gpu")]
pub use real::GpuContext;

#[cfg(not(feature = "gpu"))]
#[derive(Clone)]
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    pub fn new() -> Result<Self, String> {
        Err(
            "GPU backend not enabled (build with the `gpu` feature)".to_string(),
        )
    }

    /// Unreachable without the `gpu` feature: [`Self::new`] above always
    /// errors first. Exists only so callers (`ExecutionEngine::execute_gpu`)
    /// compile identically whether or not the feature is enabled.
    pub fn apply_single_qubit_dense_gpu(
        &self,
        _gate: [[num_complex::Complex<f64>; 2]; 2],
        _qubit: usize,
        _state: &mut [num_complex::Complex<f64>],
    ) -> Result<(), String> {
        Err("GPU backend not enabled (build with the `gpu` feature)".to_string())
    }

    /// See [`Self::apply_single_qubit_dense_gpu`].
    pub fn apply_two_qubit_dense_gpu(
        &self,
        _gate: [[num_complex::Complex<f64>; 4]; 4],
        _qubit1: usize,
        _qubit2: usize,
        _state: &mut [num_complex::Complex<f64>],
    ) -> Result<(), String> {
        Err("GPU backend not enabled (build with the `gpu` feature)".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // This environment has no wgpu-compatible adapter (confirmed: no
    // /dev/dri, no Vulkan ICD registered), so with the `gpu` feature
    // enabled this genuinely exercises GpuContext::new()'s real
    // request_adapter() call failing — not merely a compile check. Without
    // the `gpu` feature, it exercises the feature-gate stub instead. Either
    // way, construction must fail descriptively, never silently succeed
    // with a half-initialized context.
    #[test]
    fn test_gpu_context_construction_fails_loudly_without_hardware() {
        let err = GpuContext::new()
            .err()
            .expect("GPU context must not construct without a real adapter");
        assert!(
            err.to_lowercase().contains("gpu") || err.to_lowercase().contains("adapter"),
            "error should explain unavailability: {err}"
        );
    }
}
