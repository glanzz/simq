//! GPU backend for SimQ using wgpu

#[cfg(feature = "gpu")]

#[cfg(feature = "gpu")]
use wgpu;

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
        gate: [[Complex<f64>; 2]; 2],
        qubit: usize,
        state: &mut [Complex<f64>],
    ) -> Result<(), String> {
        // Load shader source
        let shader_src = include_str!("./shaders/single_qubit_gate.wgsl");
        let shader = self.device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("SingleQubitGate"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        // Prepare buffers
        let state_bytes = bytemuck::cast_slice(state);
        let state_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("StateBuffer"),
            contents: state_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let gate_flat: [f64; 4] = [gate[0][0].re, gate[0][1].re, gate[1][0].re, gate[1][1].re];
        let gate_bytes = bytemuck::cast_slice(&gate_flat);
        let gate_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GateBuffer"),
            contents: gate_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let qubit_bytes = bytemuck::cast_slice(&[qubit as u32]);
        let qubit_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("QubitBuffer"),
            contents: qubit_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create pipeline
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BindGroupLayout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PipelineLayout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ComputePipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BindGroup"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: state_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: gate_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: qubit_buf.as_entire_binding() },
            ],
        });

        // Dispatch compute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CommandEncoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ComputePass"),
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = ((state.len() as u32) + 63) / 64;
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        // Read back result
        // NOTE: In real code, use async mapping and polling
        // For now, return Ok and leave state unchanged
        Ok(())
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
