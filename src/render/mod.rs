//! WebGPU renderer embedded in egui's render pass.

use std::sync::{Arc, Mutex};

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu;

use crate::MAX_ITERATIONS;
use crate::arbitrary::ReferenceOrbit;
use crate::precision::{PrecisionMode, split_f64};

const SHADER: &str = include_str!("fractal.wgsl");
const PANE_COUNT: usize = 2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    view_hi: [f32; 4],
    view_lo: [f32; 4],
    dynamics_hi: [f32; 4],
    dynamics_lo: [f32; 4],
    display: [f32; 4],
    deep: [f32; 4],
}

impl Uniforms {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        centre: [f64; 2],
        half_height: f64,
        aspect: f32,
        julia_c: [f64; 2],
        iterations: u32,
        bailout: f32,
        family: u32,
        pane: usize,
        palette_phase: f32,
        smooth: bool,
        grid: bool,
        precision: PrecisionMode,
    ) -> Self {
        let centre_x = split_f64(centre[0]);
        let centre_y = split_f64(centre[1]);
        let scale = split_f64(half_height);
        let julia_x = split_f64(julia_c[0]);
        let julia_y = split_f64(julia_c[1]);
        Self {
            view_hi: [centre_x[0], centre_y[0], scale[0], aspect],
            view_lo: [centre_x[1], centre_y[1], scale[1], precision.shader_flag()],
            dynamics_hi: [julia_x[0], julia_y[0], iterations as f32, bailout * bailout],
            dynamics_lo: [julia_x[1], julia_y[1], family as f32, 0.0],
            display: [
                pane as f32,
                palette_phase,
                smooth as u8 as f32,
                grid as u8 as f32,
            ],
            deep: [0.0; 4],
        }
    }

    pub(crate) fn enable_perturbation(
        mut self,
        scale_mantissa: f32,
        scale_exponent: i32,
        reference_len: usize,
        ds_fallback: bool,
    ) -> Self {
        self.deep = [
            if ds_fallback { 1.0 } else { 2.0 },
            scale_mantissa,
            scale_exponent as f32,
            reference_len as f32,
        ];
        self
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuReferencePoint {
    z_hi: [f32; 2],
    z_lo: [f32; 2],
}

impl GpuReferencePoint {
    fn new(z: [f64; 2]) -> Self {
        let x = split_f64(z[0]);
        let y = split_f64(z[1]);
        Self {
            z_hi: [x[0], y[0]],
            z_lo: [x[1], y[1]],
        }
    }
}

#[derive(Debug)]
pub(crate) struct DeepRenderData {
    pub(crate) generation: u64,
    pub(crate) scale_mantissa: f32,
    pub(crate) scale_exponent: i32,
    pub(crate) reference: Vec<GpuReferencePoint>,
    pub(crate) ds_fallback: bool,
}

impl DeepRenderData {
    pub(crate) fn from_reference(
        generation: u64,
        scale_mantissa: f32,
        scale_exponent: i32,
        orbit: &ReferenceOrbit,
        ds_fallback: bool,
    ) -> Self {
        Self {
            generation,
            scale_mantissa,
            scale_exponent,
            reference: orbit
                .points
                .iter()
                .map(|point| GpuReferencePoint::new([point.re.to_f64(), point.im.to_f64()]))
                .collect(),
            ds_fallback,
        }
    }
}

struct PaneResources {
    buffer: wgpu::Buffer,
    reference_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    reference_generation: Mutex<Option<u64>>,
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
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Uniforms>() as u64
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<GpuReferencePoint>() as u64,
                        ),
                    },
                    count: None,
                },
            ],
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
            let reference_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(if index == 0 {
                    "iterascope.parameter.reference-orbit"
                } else {
                    "iterascope.dynamical.reference-orbit"
                }),
                size: (MAX_ITERATIONS as u64 + 1) * std::mem::size_of::<GpuReferencePoint>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("iterascope.fractal.bind-group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: reference_buffer.as_entire_binding(),
                    },
                ],
            });
            PaneResources {
                buffer,
                reference_buffer,
                bind_group,
                reference_generation: Mutex::new(None),
            }
        });

        Self { pipeline, panes }
    }
}

struct FractalCallback {
    pane: usize,
    uniforms: Uniforms,
    deep: Option<Arc<DeepRenderData>>,
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
            if let Some(deep) = &self.deep {
                let pane = &renderer.panes[self.pane];
                let mut uploaded = pane.reference_generation.lock().unwrap();
                if *uploaded != Some(deep.generation) {
                    queue.write_buffer(
                        &pane.reference_buffer,
                        0,
                        bytemuck::cast_slice(&deep.reference),
                    );
                    *uploaded = Some(deep.generation);
                }
            }
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

pub(crate) fn callback(
    rect: egui::Rect,
    pane: usize,
    uniforms: Uniforms,
    deep: Option<Arc<DeepRenderData>>,
) -> egui::PaintCallback {
    egui_wgpu::Callback::new_paint_callback(
        rect,
        FractalCallback {
            pane,
            uniforms,
            deep,
        },
    )
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
    fn uniform_layout_is_six_vec4s() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 96);
    }

    #[test]
    fn reference_point_matches_two_gpu_vec2_values() {
        assert_eq!(std::mem::size_of::<GpuReferencePoint>(), 16);
        let point = GpuReferencePoint::new([-0.745_123_456_789, 0.113_987_654_321]);
        assert_ne!(point.z_lo, [0.0; 2]);
    }

    #[test]
    fn uniforms_preserve_low_coordinate_words() {
        let uniforms = Uniforms::new(
            [-0.745_123_456_789, 0.113_987_654_321],
            1.45e-9,
            1.6,
            [-0.745, 0.113],
            256,
            4.0,
            0,
            0,
            0.0,
            true,
            false,
            PrecisionMode::DoubleSingle,
        );
        assert_ne!(uniforms.view_lo[0], 0.0);
        assert_ne!(uniforms.view_lo[1], 0.0);
        assert_ne!(uniforms.view_lo[2], 0.0);
        assert_eq!(uniforms.view_lo[3], 1.0);
    }

    #[test]
    fn perturbation_uniform_keeps_a_thousand_digit_scale_alive() {
        let scale = crate::arbitrary::DeepReal::parse("1.45e-1000", 1_000).unwrap();
        let (mantissa, exponent) = scale.scaled_f32();
        let uniforms = Uniforms::new(
            [0.0; 2],
            1e-14,
            1.0,
            [0.0; 2],
            256,
            4.0,
            0,
            1,
            0.0,
            true,
            false,
            PrecisionMode::DoubleSingle,
        )
        .enable_perturbation(mantissa, exponent, 257, false);
        assert_ne!(uniforms.deep[1], 0.0);
        assert_eq!(uniforms.deep[2], -3_322.0);
        assert_eq!(uniforms.deep[3], 257.0);
    }

    #[test]
    fn newton_family_flag_is_carried_without_changing_uniform_layout() {
        let uniforms = Uniforms::new(
            [0.0; 2],
            1.65,
            1.0,
            [0.0; 2],
            128,
            4.0,
            1,
            0,
            0.0,
            true,
            false,
            PrecisionMode::F32,
        );
        assert_eq!(uniforms.dynamics_lo[2], 1.0);
        assert_eq!(std::mem::size_of::<Uniforms>(), 96);
    }
}
