//! WebGPU renderer embedded in egui's render pass.

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu;

const SHADER: &str = include_str!("fractal.wgsl");
const PANE_COUNT: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    view: [f32; 4],
    dynamics: [f32; 4],
    display: [f32; 4],
}

impl Uniforms {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        centre: [f64; 2],
        half_height: f64,
        aspect: f32,
        julia_c: [f64; 2],
        iterations: u32,
        bailout: f32,
        pane: usize,
        palette_phase: f32,
        smooth: bool,
        grid: bool,
    ) -> Self {
        Self {
            view: [
                centre[0] as f32,
                centre[1] as f32,
                half_height as f32,
                aspect,
            ],
            dynamics: [
                julia_c[0] as f32,
                julia_c[1] as f32,
                iterations as f32,
                bailout * bailout,
            ],
            display: [
                pane as f32,
                palette_phase,
                smooth as u8 as f32,
                grid as u8 as f32,
            ],
        }
    }
}

struct PaneResources {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct FractalPipeline {
    pipeline: wgpu::RenderPipeline,
    panes: [PaneResources; PANE_COUNT],
}

impl FractalPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iterascope.fractal.shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iterascope.fractal.bind-group-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as u64),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iterascope.fractal.pipeline-layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("iterascope.fractal.pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let panes = std::array::from_fn(|index| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if index == 0 {
                    "iterascope.parameter.uniform"
                } else {
                    "iterascope.dynamical.uniform"
                }),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iterascope.fractal.bind-group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            PaneResources { buffer, bind_group }
        });

        Self { pipeline, panes }
    }
}

pub struct FractalCallback {
    pub pane: usize,
    pub uniforms: Uniforms,
}

impl CallbackTrait for FractalCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(renderer) = resources.get::<FractalPipeline>() {
            queue.write_buffer(
                &renderer.panes[self.pane].buffer,
                0,
                bytemuck::bytes_of(&self.uniforms),
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &CallbackResources,
    ) {
        let Some(renderer) = resources.get::<FractalPipeline>() else {
            return;
        };
        render_pass.set_pipeline(&renderer.pipeline);
        render_pass.set_bind_group(0, &renderer.panes[self.pane].bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

pub fn callback(rect: egui::Rect, pane: usize, uniforms: Uniforms) -> egui::PaintCallback {
    egui_wgpu::Callback::new_paint_callback(rect, FractalCallback { pane, uniforms })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates_with_wgpus_naga_version() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("fractal.wgsl must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("fractal.wgsl must validate");
    }

    #[test]
    fn uniform_layout_is_three_vec4s() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 48);
    }
}
