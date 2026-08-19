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
    family_a: [f32; 4],
    family_b: [f32; 4],
    numerics: [f32; 4],
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
        interior_shading: bool,
        precision: PrecisionMode,
        family_words: [f32; 8],
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
            dynamics_lo: [
                julia_x[1],
                julia_y[1],
                family as f32,
                interior_shading as u8 as f32,
            ],
            display: [
                pane as f32,
                palette_phase,
                smooth as u8 as f32,
                grid as u8 as f32,
            ],
            deep: [0.0; 4],
            family_a: [
                family_words[0],
                family_words[1],
                family_words[2],
                family_words[3],
            ],
            family_b: [
                family_words[4],
                family_words[5],
                family_words[6],
                family_words[7],
            ],
            // Four opaque copies of 1.0 used by the shader to protect
            // compensated arithmetic from fast-math reassociation.
            numerics: [1.0; 4],
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
    /// Reference data from an `f64` orbit for views below the
    /// arbitrary-precision handoff.
    pub(crate) fn from_f64_orbit(
        generation: u64,
        half_height: f64,
        points: &[[f64; 2]],
        ds_fallback: bool,
    ) -> Self {
        let exponent = half_height.log2().floor() as i32;
        let mantissa = (half_height / 2f64.powi(exponent)) as f32;
        Self {
            generation,
            scale_mantissa: mantissa,
            scale_exponent: exponent,
            reference: points
                .iter()
                .map(|point| GpuReferencePoint::new(*point))
                .collect(),
            ds_fallback,
        }
    }

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
    fn uniform_layout_is_nine_vec4s() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 144);
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
            true,
            PrecisionMode::DoubleSingle,
            [0.0; 8],
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
            true,
            PrecisionMode::DoubleSingle,
            [0.0; 8],
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
            true,
            PrecisionMode::F32,
            [0.0; 8],
        );
        assert_eq!(uniforms.dynamics_lo[2], 1.0);
        assert_eq!(std::mem::size_of::<Uniforms>(), 144);
        assert_eq!(uniforms.numerics[0], 1.0);
    }

    #[test]
    fn shader_family_codes_match_the_catalogue() {
        use crate::family::FractalFamily;
        for family in FractalFamily::ALL {
            let needle = format!(": u32 = {}u;", family.shader_flag());
            let line = SHADER
                .lines()
                .find(|line| line.starts_with("const FAMILY_") && line.ends_with(&needle))
                .unwrap_or_else(|| panic!("no shader constant for {family:?}"));
            let constant = line
                .trim_start_matches("const FAMILY_")
                .split(':')
                .next()
                .unwrap()
                .replace('_', "")
                .to_lowercase();
            let expected = family.document_id().replace('-', "");
            let expected = match expected.as_str() {
                "newtoncubic" => "newton".to_owned(),
                "magnet1" => "magnetone".to_owned(),
                "magnet2" => "magnettwo".to_owned(),
                "barnsley1" => "barnsleyone".to_owned(),
                "barnsley2" => "barnsleytwo".to_owned(),
                other => other.to_owned(),
            };
            assert_eq!(
                constant,
                expected,
                "family code {} is mislabeled",
                family.shader_flag()
            );
        }
    }

    #[test]
    fn family_words_fill_the_trailing_uniform_vectors() {
        let words = crate::family::FamilyParameters::default().uniform_words(true);
        let uniforms = Uniforms::new(
            [0.0; 2],
            1.0,
            1.0,
            [0.0; 2],
            64,
            4.0,
            2,
            1,
            0.0,
            true,
            false,
            true,
            PrecisionMode::F32,
            words,
        );
        assert_eq!(uniforms.family_a[0], 3.0);
        assert_eq!(uniforms.family_b[3], 1.0);
        assert_eq!(uniforms.family_b[1].to_bits(), 0b10);
        assert_eq!(uniforms.family_b[2], 2.0);
    }

    /// Headless GPU harness shared by the ignored gallery tests.
    struct GpuHarness {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: FractalPipeline,
        texture: wgpu::Texture,
        readback: wgpu::Buffer,
        width: u32,
        height: u32,
    }

    impl GpuHarness {
        fn new(width: u32, height: u32) -> Self {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                }))
                .expect("a GPU adapter");
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                    .expect("a GPU device");
            let format = wgpu::TextureFormat::Rgba8Unorm;
            let pipeline = FractalPipeline::new(&device, format);
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("gallery"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gallery.readback"),
                size: (width * 4 * height) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            Self {
                device,
                queue,
                pipeline,
                texture,
                readback,
                width,
                height,
            }
        }

        fn aspect(&self) -> f32 {
            self.width as f32 / self.height as f32
        }

        /// Renders one pane and returns RGB rows, top first.
        fn render(
            &self,
            pane: usize,
            uniforms: Uniforms,
            reference: Option<&[GpuReferencePoint]>,
        ) -> Vec<u8> {
            let bytes_per_row = self.width * 4;
            self.queue.write_buffer(
                &self.pipeline.panes[pane].buffer,
                0,
                bytemuck::bytes_of(&uniforms),
            );
            if let Some(reference) = reference {
                self.queue.write_buffer(
                    &self.pipeline.panes[pane].reference_buffer,
                    0,
                    bytemuck::cast_slice(reference),
                );
            }
            let view = self.texture.create_view(&Default::default());
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("gallery.pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.pipeline.pipeline);
                pass.set_bind_group(0, &self.pipeline.panes[pane].bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(self.height),
                    },
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit([encoder.finish()]);
            let (sender, receiver) = std::sync::mpsc::channel();
            self.readback
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    sender.send(result).unwrap()
                });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            receiver.recv().unwrap().unwrap();
            let pixels = self.readback.slice(..).get_mapped_range().to_vec();
            self.readback.unmap();
            // The quad maps uv.y = 0 to the bottom of the viewport; return
            // rows top-first so images are upright.
            let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);
            for row in pixels.chunks(bytes_per_row as usize).rev() {
                for pixel in row.chunks(4) {
                    rgb.extend_from_slice(&pixel[..3]);
                }
            }
            rgb
        }

        fn write_ppm(&self, path: &str, rgb: &[u8]) {
            let mut ppm = format!("P6\n{} {}\n255\n", self.width, self.height).into_bytes();
            ppm.extend_from_slice(rgb);
            std::fs::write(path, ppm).unwrap();
            eprintln!("wrote {path}");
        }
    }

    /// Renders every family's default views through the real GPU pipeline
    /// and writes them as PPM images to `$ITERASCOPE_RENDER_DIR`. Ignored by
    /// default because it needs a GPU; run with
    /// `ITERASCOPE_RENDER_DIR=out cargo test --release gpu_family_gallery -- --ignored`.
    #[test]
    #[ignore]
    fn gpu_family_gallery() {
        use crate::family::{FamilyParameters, FractalFamily, Linkage};

        let Ok(directory) = std::env::var("ITERASCOPE_RENDER_DIR") else {
            eprintln!("set ITERASCOPE_RENDER_DIR to write the gallery");
            return;
        };
        std::fs::create_dir_all(&directory).unwrap();
        let gpu = GpuHarness::new(512, 352);

        let parameters = FamilyParameters::default();
        let precisions = [PrecisionMode::F32, PrecisionMode::DoubleSingle];
        // Optional extra dynamical-plane renders: "family:re:im,family:re:im".
        let extras: Vec<(FractalFamily, [f64; 2])> = std::env::var("ITERASCOPE_RENDER_EXTRA")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|item| {
                        let mut parts = item.split(':');
                        let family = FractalFamily::from_document_id(parts.next()?)?;
                        let re = parts.next()?.parse().ok()?;
                        let im = parts.next()?.parse().ok()?;
                        Some((family, [re, im]))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut jobs: Vec<(FractalFamily, usize, PrecisionMode, [f64; 2], String)> = Vec::new();
        for family in FractalFamily::ALL {
            for pane in 0..2 {
                for precision in precisions {
                    if precision == PrecisionMode::DoubleSingle && !family.supports_double_single()
                    {
                        continue;
                    }
                    let name = format!(
                        "{:02}-{}-{}-{}",
                        family.shader_flag(),
                        family.document_id(),
                        if pane == 0 { "left" } else { "right" },
                        precision.label().to_lowercase().replace(' ', "-"),
                    );
                    jobs.push((family, pane, precision, family.default_parameter(), name));
                }
            }
        }
        for (index, (family, c)) in extras.iter().enumerate() {
            let name = format!(
                "extra-{index:02}-{}-{:+.3}{:+.3}i",
                family.document_id(),
                c[0],
                c[1]
            );
            jobs.push((*family, 1, PrecisionMode::F32, *c, name));
        }
        for (family, pane, precision, parameter, name) in jobs {
            let plane = if pane == 0 {
                family.default_parameter_view()
            } else {
                family.default_dynamical_view()
            };
            let dynamical = pane == 1 || family.linkage() == Linkage::OverviewDetail;
            let iterations = if family.is_newton() { 128 } else { 256 };
            let uniforms = Uniforms::new(
                plane.centre,
                plane.half_height,
                gpu.aspect(),
                parameter,
                iterations,
                4.0,
                family.shader_flag(),
                pane,
                0.0,
                true,
                false,
                true,
                precision,
                parameters.uniform_words(dynamical),
            );
            let rgb = gpu.render(pane, uniforms, None);
            gpu.write_ppm(&format!("{directory}/{name}.ppm"), &rgb);
        }
    }

    /// Finds a point on the boundary between bounded and finished orbits by
    /// scanning a horizontal line through `centre` and bisecting in f64.
    fn boundary_point(
        family: crate::family::FractalFamily,
        parameters: &crate::family::FamilyParameters,
        centre: [f64; 2],
        half_width: f64,
        dynamical: bool,
        parameter: [f64; 2],
        iterations: u32,
    ) -> Option<[f64; 2]> {
        use crate::family::{OrbitFate, diagnose, initial_state_with};
        let bounded = |point: [f64; 2]| {
            diagnose(
                family,
                parameters,
                initial_state_with(family, point, dynamical, parameter),
                iterations,
                4.0,
            )
            .fate
                == OrbitFate::Bounded
        };
        // Scan slightly off the view's centre line so symmetry axes (which
        // are often degenerate, e.g. the real axis of the Magnet maps) are
        // avoided.
        let y = centre[1] + 0.37 * half_width;
        let samples = 512;
        let mut previous = None;
        for index in 0..=samples {
            let x = centre[0] + (index as f64 / samples as f64 * 2.0 - 1.0) * half_width;
            let state = bounded([x, y]);
            if let Some((last_x, last_state)) = previous
                && last_state != state
            {
                let (mut a, mut b) = (last_x, x);
                for _ in 0..70 {
                    let middle = 0.5 * (a + b);
                    if bounded([middle, y]) == last_state {
                        a = middle;
                    } else {
                        b = middle;
                    }
                }
                return Some([0.5 * (a + b), y]);
            }
            previous = Some((x, state));
        }
        None
    }

    /// Renders a view around a boundary point of every perturbation-capable
    /// family twice — with the double-single path and with the perturbation
    /// path driven by an arbitrary-precision reference orbit — at a
    /// magnification where both are valid, and reports how many pixels
    /// disagree. Run with
    /// `ITERASCOPE_RENDER_DIR=out cargo test --release gpu_perturbation_matches_double_single -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_perturbation_matches_double_single() {
        use crate::arbitrary::{DeepComplex, DeepReal, DeepState, DeepView, ReferenceOrbit};
        use crate::family::{FamilyParameters, FractalFamily, Linkage};

        let directory = std::env::var("ITERASCOPE_RENDER_DIR").ok();
        if let Some(directory) = &directory {
            std::fs::create_dir_all(directory).unwrap();
        }
        let gpu = GpuHarness::new(384, 256);
        let parameters = FamilyParameters::default();
        let zoom_exponent: u32 = std::env::var("ITERASCOPE_DEEP_ZOOM_EXPONENT")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6);
        let family_filter = std::env::var("ITERASCOPE_FAMILY").ok();
        let mut report = Vec::new();
        let mut collapsed = Vec::new();
        for family in FractalFamily::ALL {
            if !family.supports_deep_zoom() {
                continue;
            }
            if let Some(filter) = &family_filter
                && !filter.split(',').any(|id| id == family.document_id())
            {
                continue;
            }
            for pane in 0..2 {
                let dynamical = pane == 1 || family.linkage() == Linkage::OverviewDetail;
                let plane = if pane == 0 {
                    family.default_parameter_view()
                } else {
                    family.default_dynamical_view()
                };
                let iterations = if family.is_newton() { 128 } else { 256 };
                let Some(point) = boundary_point(
                    family,
                    &parameters,
                    plane.centre,
                    plane.half_height,
                    dynamical,
                    family.default_parameter(),
                    iterations,
                ) else {
                    report.push(format!("{family:?} pane {pane}: no boundary point found"));
                    continue;
                };
                let half_height = 1.45 / 10f64.powi(zoom_exponent as i32);
                let words = parameters.uniform_words(dynamical);
                let ds_uniforms = Uniforms::new(
                    point,
                    half_height,
                    gpu.aspect(),
                    family.default_parameter(),
                    iterations,
                    4.0,
                    family.shader_flag(),
                    pane,
                    0.0,
                    true,
                    false,
                    true,
                    PrecisionMode::DoubleSingle,
                    words,
                );
                let ds = gpu.render(pane, ds_uniforms, None);
                let mut f32_uniforms = ds_uniforms;
                f32_uniforms.view_lo[3] = PrecisionMode::F32.shader_flag();
                let plain = gpu.render(pane, f32_uniforms, None);

                // Below the handoff the navigation layer never builds deep
                // views, so assemble one directly at 48-digit precision.
                let view = DeepView {
                    centre: DeepComplex::from_f64(point, 40).unwrap(),
                    half_height: DeepReal::parse(&format!("{half_height:.17e}"), 40).unwrap(),
                    zoom_exponent: 40,
                    magnification_log10: zoom_exponent as f64,
                };
                let julia = DeepComplex::from_f64(family.default_parameter(), 40).unwrap();
                let initial = DeepState::initial(family, &view.centre, dynamical, &julia).unwrap();
                let orbit =
                    ReferenceOrbit::family(family, &parameters, initial, iterations, 4.0).unwrap();
                let (mantissa, exponent) = view.half_height.scaled_f32();
                let data = DeepRenderData::from_reference(1, mantissa, exponent, &orbit, true);
                let perturbed_uniforms = Uniforms::new(
                    point,
                    half_height,
                    gpu.aspect(),
                    family.default_parameter(),
                    iterations,
                    4.0,
                    family.shader_flag(),
                    pane,
                    0.0,
                    true,
                    false,
                    true,
                    PrecisionMode::DoubleSingle,
                    words,
                )
                .enable_perturbation(
                    mantissa,
                    exponent,
                    data.reference.len(),
                    true,
                );
                let perturbed = gpu.render(pane, perturbed_uniforms, Some(&data.reference));
                // The same view through the f64 reference used below the handoff.
                let f64_orbit = crate::family::reference_orbit_f64(
                    family,
                    &parameters,
                    crate::family::initial_state_with(
                        family,
                        point,
                        dynamical,
                        family.default_parameter(),
                    ),
                    iterations,
                    4.0,
                );
                let f64_data =
                    DeepRenderData::from_f64_orbit(2, half_height, &f64_orbit.points, true);
                let f64_uniforms = Uniforms::new(
                    point,
                    half_height,
                    gpu.aspect(),
                    family.default_parameter(),
                    iterations,
                    4.0,
                    family.shader_flag(),
                    pane,
                    0.0,
                    true,
                    false,
                    true,
                    PrecisionMode::DoubleSingle,
                    words,
                )
                .enable_perturbation(
                    f64_data.scale_mantissa,
                    f64_data.scale_exponent,
                    f64_data.reference.len(),
                    true,
                );
                let f64_perturbed = gpu.render(pane, f64_uniforms, Some(&f64_data.reference));

                let pixels = (gpu.width * gpu.height) as usize;
                let mut differing = 0usize;
                for index in 0..pixels {
                    let a = &ds[index * 3..index * 3 + 3];
                    let b = &perturbed[index * 3..index * 3 + 3];
                    let delta: i32 = (0..3).map(|k| (a[k] as i32 - b[k] as i32).abs()).sum();
                    if delta > 60 {
                        differing += 1;
                    }
                }
                let fraction = differing as f64 / pixels as f64;
                let mut f64_differing = 0usize;
                for index in 0..pixels {
                    let a = &f64_perturbed[index * 3..index * 3 + 3];
                    let b = &perturbed[index * 3..index * 3 + 3];
                    let delta: i32 = (0..3).map(|k| (a[k] as i32 - b[k] as i32).abs()).sum();
                    if delta > 60 {
                        f64_differing += 1;
                    }
                }
                let f64_fraction = f64_differing as f64 / pixels as f64;
                let distinct_ds: std::collections::HashSet<&[u8]> = ds.chunks(3).collect();
                let distinct_perturbed: std::collections::HashSet<&[u8]> =
                    perturbed.chunks(3).collect();
                if distinct_ds.len() > 8 && distinct_perturbed.len() <= 1 {
                    collapsed.push(format!("{family:?} pane {pane}"));
                }
                report.push(format!(
                    "{:26} pane {pane}: reference {} points (ends {:?}), DS vs AP {:.2}%, f64 vs AP {:.2}% pixels differ",
                    format!("{family:?}"),
                    orbit.points.len(),
                    orbit.escape_iteration,
                    100.0 * fraction,
                    100.0 * f64_fraction
                ));
                if let Some(directory) = &directory {
                    let stem = format!(
                        "{directory}/deep-{:02}-{}-{pane}",
                        family.shader_flag(),
                        family.document_id()
                    );
                    gpu.write_ppm(&format!("{stem}-ds.ppm"), &ds);
                    gpu.write_ppm(&format!("{stem}-f32.ppm"), &plain);
                    gpu.write_ppm(&format!("{stem}-pert.ppm"), &perturbed);
                    gpu.write_ppm(&format!("{stem}-f64pert.ppm"), &f64_perturbed);
                    // CPU f64 reference classification of the same view:
                    // bounded = dark, escaped = warm ramp, converged = cool ramp.
                    use crate::family::{OrbitFate, diagnose, initial_state_with};
                    let mut cpu = Vec::with_capacity((gpu.width * gpu.height * 3) as usize);
                    for row in 0..gpu.height {
                        for column in 0..gpu.width {
                            let local = [
                                ((column as f64 + 0.5) / gpu.width as f64 * 2.0 - 1.0)
                                    * gpu.aspect() as f64,
                                1.0 - (row as f64 + 0.5) / gpu.height as f64 * 2.0,
                            ];
                            let world = [
                                point[0] + local[0] * half_height,
                                point[1] + local[1] * half_height,
                            ];
                            let result = diagnose(
                                family,
                                &parameters,
                                initial_state_with(
                                    family,
                                    world,
                                    dynamical,
                                    family.default_parameter(),
                                ),
                                iterations,
                                4.0,
                            );
                            let ramp = ((result.iterations % 32) * 8) as u8;
                            match result.fate {
                                OrbitFate::Bounded => cpu.extend_from_slice(&[10, 12, 20]),
                                OrbitFate::Escaped => cpu.extend_from_slice(&[200, ramp, 40]),
                                OrbitFate::Converged => cpu.extend_from_slice(&[40, ramp, 220]),
                                OrbitFate::NonFinite => cpu.extend_from_slice(&[255, 255, 255]),
                            }
                        }
                    }
                    gpu.write_ppm(&format!("{stem}-cpu.ppm"), &cpu);
                }
            }
        }
        for line in &report {
            eprintln!("{line}");
        }
        assert!(
            collapsed.is_empty(),
            "perturbation collapsed to a uniform image for {collapsed:?}"
        );
    }

    /// Checks on the real GPU that compensated double-single arithmetic is
    /// not optimized back to f32 by the shader compiler (Metal compiles with
    /// fast math). Run with `cargo test --release gpu_double_single_self_test -- --ignored`.
    #[test]
    #[ignore]
    fn gpu_double_single_self_test() {
        let gpu = GpuHarness::new(64, 64);
        let uniforms = Uniforms::new(
            [0.3, 0.2],
            1e-9,
            1.0,
            [0.0; 2],
            16,
            4.0,
            99,
            0,
            0.0,
            true,
            false,
            true,
            PrecisionMode::DoubleSingle,
            [0.0; 8],
        );
        let rgb = gpu.render(0, uniforms, None);
        let distinct: std::collections::HashSet<&[u8]> = rgb.chunks(3).collect();
        assert_eq!(
            distinct.into_iter().collect::<Vec<_>>(),
            vec![&[255u8, 255, 255][..]],
            "double-single addition, multiplication and view transform must all survive"
        );
    }

    /// Locates a repelling fixed point of the family's dynamical-plane map by
    /// arbitrary-precision Newton iteration with a finite-difference
    /// derivative. Julia sets are self-similar around repelling fixed points,
    /// so a view centred there shows structure at every magnification.
    fn repelling_fixed_point(
        family: crate::family::FractalFamily,
        parameters: &crate::family::FamilyParameters,
        c: &crate::arbitrary::DeepComplex,
        start: [f64; 2],
        precision_exponent: u32,
    ) -> Option<(crate::arbitrary::DeepComplex, f64)> {
        use crate::arbitrary::{DeepComplex, DeepState, deep_step};
        let mut z = DeepComplex::from_f64(start, precision_exponent).ok()?;
        let step = |z: &DeepComplex| {
            deep_step(
                family,
                parameters,
                &DeepState {
                    z: z.clone(),
                    z_prev: z.clone(),
                    c: c.clone(),
                },
            )
            .z
        };
        let mut multiplier = 0.0;
        for _ in 0..80 {
            let g = step(&z).sub(&z);
            let h = z.real_like(1e-25);
            let shifted = z.add(&h);
            let g_shifted = step(&shifted).sub(&shifted);
            let derivative = g_shifted.sub(&g).div(&h);
            let d = derivative.to_f64_pair();
            if !d[0].is_finite() || !d[1].is_finite() || d[0].hypot(d[1]) < 1e-12 {
                return None;
            }
            // f'(z) = g'(z) + 1
            multiplier = (d[0] + 1.0).hypot(d[1]);
            let delta = g.div(&derivative);
            z = z.sub(&delta);
            let size = delta.to_f64_pair();
            if size[0].hypot(size[1]) < 1e-60 {
                break;
            }
        }
        let residual = step(&z).sub(&z).to_f64_pair();
        if residual[0].hypot(residual[1]) > 1e-30 || multiplier <= 1.05 {
            return None;
        }
        Some((z, multiplier))
    }

    /// Renders several families at 10^30 magnification, far beyond the
    /// double-single handoff, centred on a repelling fixed point of the
    /// dynamical plane (or, for the Mandelbox, a boundary point of the
    /// parameter plane bisected in arbitrary precision), and checks that the
    /// perturbation path still resolves structure. Run with
    /// `ITERASCOPE_RENDER_DIR=out cargo test --release gpu_deep_zoom_resolves_structure -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_deep_zoom_resolves_structure() {
        use crate::arbitrary::{DeepComplex, DeepReal, DeepState, DeepView, ReferenceOrbit};
        use crate::family::{FamilyParameters, FractalFamily};

        let directory = std::env::var("ITERASCOPE_RENDER_DIR").ok();
        let gpu = GpuHarness::new(384, 256);
        let parameters = FamilyParameters::default();
        let zoom_exponent = 30u32;
        let precision_exponent = zoom_exponent + 40;
        let iterations = 512u32;
        let mut failures = Vec::new();
        let mut report = Vec::new();
        // Families whose dynamical planes have repelling fixed points found
        // by holomorphic Newton iteration, plus the Mandelbox via bisection.
        // Anti-holomorphic (Tricorn), convergence-coloured (Nova, Newton) and
        // piecewise-linear (Barnsley) families are validated against the
        // double-single path at 10^6 by the comparison test instead.
        for family in [
            FractalFamily::Quadratic,
            FractalFamily::Multibrot,
            FractalFamily::BurningShip,
            FractalFamily::Celtic,
            FractalFamily::Lambda,
            FractalFamily::MagnetOne,
            FractalFamily::Mandelbox,
        ] {
            let plane = family.default_parameter_view();
            let c_f64 = family.default_parameter();
            let c = DeepComplex::from_f64(c_f64, precision_exponent).unwrap();
            let mut centre = None;
            let mut dynamical = true;
            if family == FractalFamily::Mandelbox {
                // Piecewise maps have fold lines everywhere; bisect a boundary
                // point of the parameter plane instead of seeking a fixed point.
                dynamical = false;
                let y = DeepReal::from_f64(
                    plane.centre[1] + 0.37 * plane.half_height,
                    precision_exponent,
                )
                .unwrap();
                let bounded = |x: &DeepReal| {
                    let c = DeepComplex {
                        re: x.clone(),
                        im: y.clone(),
                    };
                    let initial = DeepState::initial(family, &c, false, &c).unwrap();
                    ReferenceOrbit::family(family, &parameters, initial, iterations, 4.0)
                        .unwrap()
                        .escape_iteration
                        .is_none()
                };
                let samples = 256;
                let mut previous: Option<(f64, bool)> = None;
                for index in 0..=samples {
                    let x = plane.centre[0]
                        + (index as f64 / samples as f64 * 2.0 - 1.0) * plane.half_height;
                    let state = bounded(&DeepReal::from_f64(x, precision_exponent).unwrap());
                    if let Some((last_x, last_state)) = previous
                        && last_state != state
                    {
                        let mut a = DeepReal::from_f64(last_x, precision_exponent).unwrap();
                        let mut b = DeepReal::from_f64(x, precision_exponent).unwrap();
                        let half = a.constant_like(0.5);
                        for _ in 0..110 {
                            let middle = a.add(&b).mul(&half);
                            if bounded(&middle) == last_state {
                                a = middle;
                            } else {
                                b = middle;
                            }
                        }
                        centre = Some(DeepComplex {
                            re: a.add(&b).mul(&half),
                            im: y.clone(),
                        });
                        break;
                    }
                    previous = Some((x, state));
                }
            } else {
                // Prefer the most strongly repelling fixed point: structure
                // appears after log(1e30)/log(multiplier) iterations.
                let mut best: Option<(DeepComplex, f64)> = None;
                for start in [
                    [0.6, 0.4],
                    [-0.5, 0.7],
                    [1.1, -0.3],
                    [-1.2, 0.1],
                    [0.2, -0.9],
                    [0.05, 0.05],
                    [-0.9, -0.8],
                    [1.6, 0.9],
                ] {
                    if let Some((point, multiplier)) =
                        repelling_fixed_point(family, &parameters, &c, start, precision_exponent)
                        && best
                            .as_ref()
                            .is_none_or(|(_, best_multiplier)| multiplier > *best_multiplier)
                    {
                        best = Some((point, multiplier));
                    }
                }
                if let Some((point, multiplier)) = best {
                    report.push(format!(
                        "{family:?}: fixed point multiplier {multiplier:.3}"
                    ));
                    centre = Some(point);
                }
            }
            let Some(centre) = centre else {
                report.push(format!("{family:?}: no deep centre found"));
                continue;
            };
            let view = DeepView {
                centre: centre.clone(),
                half_height: DeepReal::parse(&format!("1.45e-{zoom_exponent}"), precision_exponent)
                    .unwrap(),
                zoom_exponent: precision_exponent,
                magnification_log10: zoom_exponent as f64,
            };
            let initial = DeepState::initial(family, &view.centre, dynamical, &c).unwrap();
            let orbit =
                ReferenceOrbit::family(family, &parameters, initial, iterations, 4.0).unwrap();
            let (mantissa, exponent) = view.half_height.scaled_f32();
            let data = DeepRenderData::from_reference(1, mantissa, exponent, &orbit, false);
            let pane = if dynamical { 1 } else { 0 };
            let uniforms = Uniforms::new(
                view.centre_preview(),
                view.half_height_preview(),
                gpu.aspect(),
                c_f64,
                iterations,
                4.0,
                family.shader_flag(),
                pane,
                0.0,
                true,
                false,
                true,
                PrecisionMode::DoubleSingle,
                parameters.uniform_words(dynamical),
            )
            .enable_perturbation(mantissa, exponent, data.reference.len(), false);
            let rgb = gpu.render(pane, uniforms, Some(&data.reference));
            let distinct: std::collections::HashSet<&[u8]> = rgb.chunks(3).collect();
            report.push(format!(
                "{:26} at 1e{zoom_exponent}: centre {:?}, reference {} points (ends {:?}), {} distinct colours",
                format!("{family:?}"),
                centre.to_f64_pair(),
                orbit.points.len(),
                orbit.escape_iteration,
                distinct.len()
            ));
            if let Some(directory) = &directory {
                std::fs::create_dir_all(directory).unwrap();
                gpu.write_ppm(
                    &format!("{directory}/deep30-{}.ppm", family.document_id()),
                    &rgb,
                );
            }
            if distinct.len() <= 8 {
                failures.push(format!("{family:?}"));
            }
        }
        for line in &report {
            eprintln!("{line}");
        }
        assert!(
            failures.is_empty(),
            "perturbation at 1e{zoom_exponent} rendered a uniform image for {failures:?}"
        );
    }
}
