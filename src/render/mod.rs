//! WebGPU renderer embedded in egui's render pass.

use std::sync::{Arc, Mutex};

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use eframe::wgpu;

use crate::MAX_ITERATIONS;
use crate::arbitrary::DeepComplex;
#[cfg(test)]
use crate::arbitrary::ReferenceOrbit;
use crate::colouring::{self, LayerStack};
use crate::precision::{PrecisionMode, split_f64};

/// Raw WGSL with the delta-algebra template; see [`shader_source`].
const SHADER: &str = include_str!("fractal.wgsl");
/// Two interactive panes plus one set of resources reserved for the
/// image-sequence exporter, which renders between UI frames on the same
/// queue and must not disturb the panes' uploaded state.
const PANE_COUNT: usize = 3;
pub(crate) const EXPORT_PANE: usize = 2;

const TEMPLATE_BEGIN: &str = "// BEGIN DELTA TEMPLATE\n";
const TEMPLATE_END: &str = "// END DELTA TEMPLATE\n";

/// The shader with orbit statistics enabled (see [`shader_source_variant`]).
#[cfg(test)]
pub(crate) fn shader_source() -> String {
    shader_source_variant(true)
}

/// Expands the delta-algebra template in `fractal.wgsl` into its scaled
/// (arbitrary-precision depth) and plain-f32 (f64-reference range)
/// instantiations and returns the compilable shader source. With
/// `orbit_stats` false the per-iteration colouring accumulators are compiled
/// out, which keeps the default configuration as fast as it was before the
/// colour stage existed.
pub(crate) fn shader_source_variant(orbit_stats: bool) -> String {
    const MARKER: &str = "const ORBIT_STATS: bool = true; // ORBIT_STATS_MARKER";
    assert!(
        SHADER.contains(MARKER),
        "fractal.wgsl has the ORBIT_STATS marker"
    );
    let shader = SHADER.replace(
        MARKER,
        if orbit_stats {
            "const ORBIT_STATS: bool = true;"
        } else {
            "const ORBIT_STATS: bool = false;"
        },
    );
    let begin = shader
        .find(TEMPLATE_BEGIN)
        .expect("fractal.wgsl has a delta template begin marker");
    let end = shader
        .find(TEMPLATE_END)
        .expect("fractal.wgsl has a delta template end marker");
    let template = &shader[begin + TEMPLATE_BEGIN.len()..end];
    let scaled = instantiate_template(
        template,
        "_scaled",
        "ScaledComplex",
        "ScaledReal",
        &[
            ("dc_mul_plain", "scaled_mul_plain"),
            ("dc_mul_real", "sc_mul_real"),
            ("dc_from_reals", "sc_from_reals"),
            ("dc_from_f32", "sc_from_f32"),
            ("dc_to_f32", "scaled_to_f32"),
            ("dc_normalize", "scaled_normalize"),
            ("dc_pixel_delta", "sc_pixel_delta"),
            ("dc_zero", "sc_zero"),
            ("dc_neg", "sc_neg"),
            ("dc_sub", "sc_sub"),
            ("dc_add", "scaled_add"),
            ("dc_mul", "scaled_complex_mul"),
            ("dc_scale", "sc_scale"),
            ("dc_x", "sc_x"),
            ("dc_y", "sc_y"),
            ("dr_times_complex", "sr_times_complex"),
            ("dr_from_f32", "sr_from_f32"),
            ("dr_to_f32", "sr_to_f32"),
            ("dr_zero", "sr_zero"),
            ("dr_add", "sr_add"),
            ("dr_neg", "sr_neg"),
            ("dr_mul", "sr_mul"),
            ("dr_scale", "sr_scale"),
        ],
    );
    let plain = instantiate_template(
        template,
        "_f32",
        "vec2<f32>",
        "f32",
        &[
            ("dc_mul_plain", "fc_mul_plain"),
            ("dc_mul_real", "fc_mul_real"),
            ("dc_from_reals", "fc_from_reals"),
            ("dc_from_f32", "fc_from_f32"),
            ("dc_to_f32", "fc_to_f32"),
            ("dc_normalize", "fc_normalize"),
            ("dc_pixel_delta", "fc_pixel_delta"),
            ("dc_zero", "fc_zero"),
            ("dc_neg", "fc_neg"),
            ("dc_sub", "fc_sub"),
            ("dc_add", "fc_add"),
            ("dc_mul", "fc_mul"),
            ("dc_scale", "fc_scale"),
            ("dc_x", "fc_x"),
            ("dc_y", "fc_y"),
            ("dr_times_complex", "fr_times_complex"),
            ("dr_from_f32", "fr_from_f32"),
            ("dr_to_f32", "fr_to_f32"),
            ("dr_zero", "fr_zero"),
            ("dr_add", "fr_add"),
            ("dr_neg", "fr_neg"),
            ("dr_mul", "fr_mul"),
            ("dr_scale", "fr_scale"),
        ],
    );
    let mut source = String::with_capacity(shader.len() + template.len());
    source.push_str(&shader[..begin]);
    source.push_str("// --- scaled instantiation ---\n");
    source.push_str(&scaled);
    source.push_str("// --- f32 instantiation ---\n");
    source.push_str(&plain);
    source.push_str(&shader[end + TEMPLATE_END.len()..]);
    source
}

/// Replaces the template's placeholder identifiers. Replacement is done on
/// whole identifiers only, longest names first, so `dc_mul_plain` is not
/// clobbered by `dc_mul`.
fn instantiate_template(
    template: &str,
    suffix: &str,
    complex_type: &str,
    real_type: &str,
    operations: &[(&str, &str)],
) -> String {
    let mut rules: Vec<(&str, &str)> = operations.to_vec();
    rules.push(("DC", complex_type));
    rules.push(("DR", real_type));
    rules.sort_by_key(|(from, _)| std::cmp::Reverse(from.len()));
    let mut output = String::with_capacity(template.len() + 256);
    let bytes = template.as_bytes();
    let is_identifier = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let mut index = 0;
    while index < template.len() {
        if !is_identifier(bytes[index]) {
            output.push(bytes[index] as char);
            index += 1;
            continue;
        }
        let start = index;
        while index < template.len() && is_identifier(bytes[index]) {
            index += 1;
        }
        let word = &template[start..index];
        if let Some(stripped) = word.strip_suffix("__T") {
            output.push_str(stripped);
            output.push_str(suffix);
        } else if let Some((_, to)) = rules.iter().find(|(from, _)| *from == word) {
            output.push_str(to);
        } else {
            output.push_str(word);
        }
    }
    output
}

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
    reference: [f32; 4],
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
        smooth: bool,
        grid: bool,
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
            dynamics_lo: [julia_x[1], julia_y[1], family as f32, 0.0],
            display: [pane as f32, 0.0, smooth as u8 as f32, grid as u8 as f32],
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
            reference: [0.0; 4],
        }
    }

    /// Enables perturbation rendering around the uploaded reference orbit.
    /// `reference_offset` is the reference point's position relative to the
    /// view centre in the shader's local units (x in units of the half-height
    /// times the aspect ratio, y in half-heights); zero for a centred
    /// reference.
    pub(crate) fn enable_perturbation(
        mut self,
        scale_mantissa: f32,
        scale_exponent: i32,
        reference_len: usize,
        ds_fallback: bool,
        reference_offset: [f32; 2],
    ) -> Self {
        self.deep = [
            if ds_fallback { 1.0 } else { 2.0 },
            scale_mantissa,
            scale_exponent as f32,
            reference_len as f32,
        ];
        self.reference = [reference_offset[0], reference_offset[1], 0.0, 0.0];
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

/// The colour stage's uniform block; layout shared with `ColouringUniforms`
/// in `fractal.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ColouringUniforms {
    words: [[f32; 4]; colouring::GPU_WORDS],
}

impl ColouringUniforms {
    /// `pixel_log` is the natural logarithm of one pixel's height in world
    /// units; distance estimates are expressed in pixels.
    pub(crate) fn new(layers: &LayerStack, pixel_log: f32) -> Self {
        Self {
            words: layers.gpu_words(pixel_log),
        }
    }

    /// Whether a selected algorithm needs the per-iteration accumulators,
    /// i.e. the orbit-statistics shader variant.
    pub(crate) fn needs_orbit_stats(&self) -> bool {
        self.words[colouring::NEEDS_WORD]
            .iter()
            .any(|flag| *flag > 0.5)
    }
}

impl Default for ColouringUniforms {
    fn default() -> Self {
        Self::new(&LayerStack::default(), 0.0)
    }
}

/// The rasterised gradients of a layer stack's visible layers, concatenated
/// bottom first, ready for upload. The generation lets the renderer skip
/// re-uploading a table it already holds.
#[derive(Debug)]
pub(crate) struct GradientTable {
    pub(crate) generation: u64,
    pub(crate) entries: Vec<[f32; 4]>,
}

impl GradientTable {
    pub(crate) fn new(generation: u64, layers: &LayerStack) -> Self {
        Self {
            generation,
            entries: layers.lookup_tables(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct DeepRenderData {
    pub(crate) generation: u64,
    pub(crate) scale_mantissa: f32,
    pub(crate) scale_exponent: i32,
    /// Shared so a reference can be re-described (new scale, new offset)
    /// without copying or re-uploading its points.
    pub(crate) reference: Arc<Vec<GpuReferencePoint>>,
    pub(crate) ds_fallback: bool,
    /// Reference point relative to the view centre in shader local units.
    pub(crate) reference_offset: [f32; 2],
}

impl DeepRenderData {
    /// Reference data from an `f64` orbit for views below the
    /// arbitrary-precision handoff.
    pub(crate) fn from_f64_orbit(
        generation: u64,
        half_height: f64,
        points: &[[f64; 2]],
        ds_fallback: bool,
        reference_offset: [f32; 2],
    ) -> Self {
        let exponent = half_height.log2().floor() as i32;
        let mantissa = (half_height / 2f64.powi(exponent)) as f32;
        Self {
            generation,
            scale_mantissa: mantissa,
            scale_exponent: exponent,
            reference: Arc::new(
                points
                    .iter()
                    .map(|point| GpuReferencePoint::new(*point))
                    .collect(),
            ),
            ds_fallback,
            reference_offset,
        }
    }

    /// The same reference orbit described for another view: new pixel scale
    /// and the reference point's offset from the new centre. Keeps the
    /// generation so the GPU does not re-upload the points.
    pub(crate) fn redescribed(
        &self,
        scale_mantissa: f32,
        scale_exponent: i32,
        reference_offset: [f32; 2],
        ds_fallback: bool,
    ) -> Self {
        Self {
            generation: self.generation,
            scale_mantissa,
            scale_exponent,
            reference: Arc::clone(&self.reference),
            ds_fallback,
            reference_offset,
        }
    }

    pub(crate) fn from_points(
        generation: u64,
        scale_mantissa: f32,
        scale_exponent: i32,
        points: &[DeepComplex],
        ds_fallback: bool,
    ) -> Self {
        Self {
            generation,
            scale_mantissa,
            scale_exponent,
            reference: Arc::new(
                points
                    .iter()
                    .map(|point| GpuReferencePoint::new([point.re.to_f64(), point.im.to_f64()]))
                    .collect(),
            ),
            ds_fallback,
            reference_offset: [0.0; 2],
        }
    }

    #[cfg(test)]
    pub(crate) fn from_reference(
        generation: u64,
        scale_mantissa: f32,
        scale_exponent: i32,
        orbit: &ReferenceOrbit,
        ds_fallback: bool,
    ) -> Self {
        Self::from_points(
            generation,
            scale_mantissa,
            scale_exponent,
            &orbit.points,
            ds_fallback,
        )
    }
}

/// Reduced-resolution target used while input is active.
struct PreviewTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    size: (u32, u32),
}

struct PaneResources {
    buffer: wgpu::Buffer,
    reference_buffer: wgpu::Buffer,
    colouring_buffer: wgpu::Buffer,
    gradient_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    reference_generation: Mutex<Option<u64>>,
    gradient_generation: Mutex<Option<u64>>,
    /// Whether the uploaded colouring needs the orbit-statistics pipeline.
    orbit_stats: Mutex<bool>,
    preview: Mutex<Option<PreviewTarget>>,
}

pub struct FractalPipeline {
    /// Fractal pipeline with the colouring accumulators compiled out.
    pipeline: wgpu::RenderPipeline,
    /// Fractal pipeline gathering orbit statistics for the trap, average
    /// and distance-estimate colourings.
    stats_pipeline: wgpu::RenderPipeline,
    /// The same two variants targeting `Rgba8Unorm` for the exporter, whose
    /// readback format must not depend on the window surface.
    export_pipeline: wgpu::RenderPipeline,
    export_stats_pipeline: wgpu::RenderPipeline,
    /// Draws a preview texture onto the pane.
    blit_pipeline: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    target_format: wgpu::TextureFormat,
    panes: [PaneResources; PANE_COUNT],
    /// Offscreen target reused across exported frames of one size.
    export_target: Mutex<Option<ExportTarget>>,
}

/// Texture and readback buffer for one export frame size.
struct ExportTarget {
    texture: wgpu::Texture,
    readback: wgpu::Buffer,
    size: (u32, u32),
    padded_bytes_per_row: u32,
}

/// Fullscreen-triangle blit of a sampled texture; used to present preview
/// renders made at reduced resolution while input is active.
const BLIT_SHADER: &str = r#"
struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var preview_texture: texture_2d<f32>;
@group(0) @binding(1) var preview_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let corner = corners[vertex_index];
    var out: VertexOut;
    out.position = vec4<f32>(corner, 0.0, 1.0);
    // Texture rows run top-down while clip space runs bottom-up.
    out.uv = vec2<f32>(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(preview_texture, preview_sampler, in.uv);
}
"#;

impl FractalPipeline {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iterascope.fractal.shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source_variant(false).into()),
        });
        let stats_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iterascope.fractal.shader.orbit-stats"),
            source: wgpu::ShaderSource::Wgsl(shader_source_variant(true).into()),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<ColouringUniforms>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
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
        let fractal_pipeline =
            |label: &str, module: &wgpu::ShaderModule, format: wgpu::TextureFormat| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    fragment: Some(wgpu::FragmentState {
                        module,
                        entry_point: Some("fs_main"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                })
            };
        let pipeline = fractal_pipeline("iterascope.fractal.pipeline", &shader, target_format);
        let stats_pipeline = fractal_pipeline(
            "iterascope.fractal.pipeline.orbit-stats",
            &stats_shader,
            target_format,
        );
        let export_pipeline = fractal_pipeline(
            "iterascope.fractal.pipeline.export",
            &shader,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let export_stats_pipeline = fractal_pipeline(
            "iterascope.fractal.pipeline.export.orbit-stats",
            &stats_shader,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("iterascope.blit.shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("iterascope.blit.bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("iterascope.blit.pipeline-layout"),
            bind_group_layouts: &[Some(&blit_layout)],
            immediate_size: 0,
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("iterascope.blit.pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
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
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("iterascope.blit.sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
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
            // The colour stage starts with the default colouring and
            // gradient so a pane renders before the application uploads
            // its own.
            let colouring_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iterascope.colouring"),
                size: std::mem::size_of::<ColouringUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            colouring_buffer
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::bytes_of(&ColouringUniforms::default()));
            colouring_buffer.unmap();
            let gradient_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iterascope.gradient"),
                size: (colouring::MAX_LAYERS
                    * colouring::LOOKUP_TABLE_LEN
                    * std::mem::size_of::<[f32; 4]>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            {
                let default_table = LayerStack::default().lookup_tables();
                let bytes: &[u8] = bytemuck::cast_slice(&default_table);
                let mut mapped = gradient_buffer.slice(..).get_mapped_range_mut();
                mapped.slice(..bytes.len()).copy_from_slice(bytes);
            }
            gradient_buffer.unmap();
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
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: colouring_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: gradient_buffer.as_entire_binding(),
                    },
                ],
            });
            PaneResources {
                buffer,
                reference_buffer,
                colouring_buffer,
                gradient_buffer,
                bind_group,
                reference_generation: Mutex::new(None),
                gradient_generation: Mutex::new(None),
                orbit_stats: Mutex::new(false),
                preview: Mutex::new(None),
            }
        });

        Self {
            pipeline,
            stats_pipeline,
            export_pipeline,
            export_stats_pipeline,
            blit_pipeline,
            blit_layout,
            sampler,
            target_format,
            panes,
            export_target: Mutex::new(None),
        }
    }

    /// Renders one export frame at `size` and returns tightly packed RGBA
    /// rows, top first. Blocks until the GPU finishes; the exporter calls
    /// this between UI frames, one frame per update. `reference` carries a
    /// generation so an orbit shared by many frames uploads once.
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_export(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uniforms: &Uniforms,
        colouring: &ColouringUniforms,
        gradient: &GradientTable,
        reference: Option<(u64, &[GpuReferencePoint])>,
        size: (u32, u32),
    ) -> Vec<u8> {
        let resources = &self.panes[EXPORT_PANE];
        queue.write_buffer(&resources.buffer, 0, bytemuck::bytes_of(uniforms));
        self.upload_colouring(queue, EXPORT_PANE, colouring, Some(gradient));
        if let Some((generation, points)) = reference {
            let mut uploaded = resources.reference_generation.lock().unwrap();
            if *uploaded != Some(generation) {
                queue.write_buffer(&resources.reference_buffer, 0, bytemuck::cast_slice(points));
                *uploaded = Some(generation);
            }
        }

        let mut slot = self.export_target.lock().unwrap();
        if slot.as_ref().is_none_or(|target| target.size != size) {
            let padded_bytes_per_row = (size.0 * 4).div_ceil(256) * 256;
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("iterascope.export"),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("iterascope.export.readback"),
                size: (padded_bytes_per_row * size.1) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            *slot = Some(ExportTarget {
                texture,
                readback,
                size,
                padded_bytes_per_row,
            });
        }
        let target = slot.as_ref().unwrap();

        let view = target.texture.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("iterascope.export.pass"),
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
            if colouring.needs_orbit_stats() {
                pass.set_pipeline(&self.export_stats_pipeline);
            } else {
                pass.set_pipeline(&self.export_pipeline);
            }
            pass.set_bind_group(0, &resources.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &target.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.padded_bytes_per_row),
                    rows_per_image: Some(size.1),
                },
            },
            wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::channel();
        target
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let padded = target.readback.slice(..).get_mapped_range().to_vec();
        target.readback.unmap();

        // Texture row 0 is the top of the viewport (clip y = +1), which the
        // fullscreen triangle maps to uv.y = 1 and the fragment shader to
        // positive imaginary values — the buffer is already top-first, so
        // only the copy padding needs stripping.
        let tight_row = (size.0 * 4) as usize;
        let mut rgba = Vec::with_capacity(tight_row * size.1 as usize);
        for row in padded
            .chunks(target.padded_bytes_per_row as usize)
            .take(size.1 as usize)
        {
            rgba.extend_from_slice(&row[..tight_row]);
        }
        rgba
    }

    /// Uploads the colour stage for a pane: the uniform block every time,
    /// the gradient table only when its generation changed.
    fn upload_colouring(
        &self,
        queue: &wgpu::Queue,
        pane: usize,
        colouring: &ColouringUniforms,
        gradient: Option<&GradientTable>,
    ) {
        let resources = &self.panes[pane];
        queue.write_buffer(
            &resources.colouring_buffer,
            0,
            bytemuck::bytes_of(colouring),
        );
        *resources.orbit_stats.lock().unwrap() = colouring.needs_orbit_stats();
        if let Some(gradient) = gradient {
            let mut uploaded = resources.gradient_generation.lock().unwrap();
            if *uploaded != Some(gradient.generation) {
                let limit = colouring::MAX_LAYERS * colouring::LOOKUP_TABLE_LEN;
                let entries = &gradient.entries[..gradient.entries.len().min(limit)];
                queue.write_buffer(&resources.gradient_buffer, 0, bytemuck::cast_slice(entries));
                *uploaded = Some(gradient.generation);
            }
        }
    }

    /// The fractal pipeline matching the pane's uploaded colouring.
    fn fractal_pipeline(&self, pane: usize) -> &wgpu::RenderPipeline {
        if *self.panes[pane].orbit_stats.lock().unwrap() {
            &self.stats_pipeline
        } else {
            &self.pipeline
        }
    }

    /// Returns the pane's preview target, recreating it when the requested
    /// size changes.
    fn preview_target(&self, device: &wgpu::Device, pane: usize, size: (u32, u32)) {
        let mut slot = self.panes[pane].preview.lock().unwrap();
        if slot.as_ref().is_some_and(|target| target.size == size) {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iterascope.preview"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("iterascope.preview.bind-group"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        *slot = Some(PreviewTarget {
            _texture: texture,
            view,
            bind_group,
            size,
        });
    }
}

struct FractalCallback {
    pane: usize,
    uniforms: Uniforms,
    colouring: ColouringUniforms,
    gradient: Arc<GradientTable>,
    deep: Option<Arc<DeepRenderData>>,
    /// Preview reduction factor: 1 renders directly at full resolution;
    /// larger values render into a texture of 1/factor the size and blit it.
    preview_scale: u32,
    /// Pane size in physical pixels.
    pixel_size: (u32, u32),
}

impl FractalCallback {
    fn preview_size(&self) -> (u32, u32) {
        (
            self.pixel_size.0.div_ceil(self.preview_scale).max(1),
            self.pixel_size.1.div_ceil(self.preview_scale).max(1),
        )
    }
}

impl CallbackTrait for FractalCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(renderer) = resources.get::<FractalPipeline>() {
            queue.write_buffer(
                &renderer.panes[self.pane].buffer,
                0,
                bytemuck::bytes_of(&self.uniforms),
            );
            renderer.upload_colouring(queue, self.pane, &self.colouring, Some(&self.gradient));
            if let Some(deep) = &self.deep {
                let pane = &renderer.panes[self.pane];
                let mut uploaded = pane.reference_generation.lock().unwrap();
                if *uploaded != Some(deep.generation) {
                    queue.write_buffer(
                        &pane.reference_buffer,
                        0,
                        bytemuck::cast_slice(deep.reference.as_slice()),
                    );
                    *uploaded = Some(deep.generation);
                }
            }
            if self.preview_scale > 1 {
                // Render the fractal at reduced resolution now; `paint` only
                // has to blit the result into the pane.
                renderer.preview_target(device, self.pane, self.preview_size());
                let slot = renderer.panes[self.pane].preview.lock().unwrap();
                if let Some(target) = slot.as_ref() {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("iterascope.preview.pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &target.view,
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
                    pass.set_pipeline(renderer.fractal_pipeline(self.pane));
                    pass.set_bind_group(0, &renderer.panes[self.pane].bind_group, &[]);
                    pass.draw(0..3, 0..1);
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
        if self.preview_scale > 1 {
            let slot = renderer.panes[self.pane].preview.lock().unwrap();
            if let Some(target) = slot.as_ref() {
                render_pass.set_pipeline(&renderer.blit_pipeline);
                render_pass.set_bind_group(0, &target.bind_group, &[]);
                render_pass.draw(0..3, 0..1);
                return;
            }
        }
        render_pass.set_pipeline(renderer.fractal_pipeline(self.pane));
        render_pass.set_bind_group(0, &renderer.panes[self.pane].bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn callback(
    rect: egui::Rect,
    pane: usize,
    uniforms: Uniforms,
    colouring: ColouringUniforms,
    gradient: Arc<GradientTable>,
    deep: Option<Arc<DeepRenderData>>,
    preview_scale: u32,
    pixels_per_point: f32,
) -> egui::PaintCallback {
    let pixel_size = (
        (rect.width() * pixels_per_point).round().max(1.0) as u32,
        (rect.height() * pixels_per_point).round().max(1.0) as u32,
    );
    egui_wgpu::Callback::new_paint_callback(
        rect,
        FractalCallback {
            pane,
            uniforms,
            colouring,
            gradient,
            deep,
            preview_scale: preview_scale.max(1),
            pixel_size,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_shader_variant_compiles_the_accumulators_out() {
        let plain = shader_source_variant(false);
        let stats = shader_source_variant(true);
        assert!(plain.contains("const ORBIT_STATS: bool = false;"));
        assert!(stats.contains("const ORBIT_STATS: bool = true;"));
        assert!(!plain.contains("ORBIT_STATS_MARKER"));
        naga::front::wgsl::parse_str(&plain).expect("plain variant parses");
    }

    #[test]
    fn shader_validates_with_wgpus_naga_version() {
        let source = shader_source();
        assert!(source.contains("fn perturb_step_scaled("));
        assert!(source.contains("fn perturb_step_f32("));
        assert!(!source.contains("__T"), "template suffix left unexpanded");
        assert!(!source.contains(" DC("), "template type left unexpanded");
        let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|error| {
            panic!(
                "expanded fractal.wgsl must parse: {}",
                error.emit_to_string(&source)
            )
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("fractal.wgsl must validate");
    }

    #[test]
    fn uniform_layout_is_ten_vec4s() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 160);
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
            true,
            false,
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
            true,
            false,
            PrecisionMode::DoubleSingle,
            [0.0; 8],
        )
        .enable_perturbation(mantissa, exponent, 257, false, [0.0; 2]);
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
            true,
            false,
            PrecisionMode::F32,
            [0.0; 8],
        );
        assert_eq!(uniforms.dynamics_lo[2], 1.0);
        assert_eq!(std::mem::size_of::<Uniforms>(), 160);
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
            true,
            false,
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
                pass.set_pipeline(self.pipeline.fractal_pipeline(pane));
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
            // Texture row 0 is the top of the viewport (+Im); rows are
            // already top-first.
            let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);
            for row in pixels.chunks(bytes_per_row as usize) {
                for pixel in row.chunks(4) {
                    rgb.extend_from_slice(&pixel[..3]);
                }
            }
            rgb
        }

        /// Renders through the preview path: fractal into a reduced target,
        /// then blit into the main texture. Returns RGB rows, top first.
        fn render_preview(&self, pane: usize, uniforms: Uniforms, scale: u32) -> Vec<u8> {
            let bytes_per_row = self.width * 4;
            self.queue.write_buffer(
                &self.pipeline.panes[pane].buffer,
                0,
                bytemuck::bytes_of(&uniforms),
            );
            let size = (
                self.width.div_ceil(scale).max(1),
                self.height.div_ceil(scale).max(1),
            );
            self.pipeline.preview_target(&self.device, pane, size);
            let view = self.texture.create_view(&Default::default());
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let slot = self.pipeline.panes[pane].preview.lock().unwrap();
                let target = slot.as_ref().unwrap();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("preview.pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target.view,
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
                pass.set_pipeline(self.pipeline.fractal_pipeline(pane));
                pass.set_bind_group(0, &self.pipeline.panes[pane].bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            {
                let slot = self.pipeline.panes[pane].preview.lock().unwrap();
                let target = slot.as_ref().unwrap();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("blit.pass"),
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
                pass.set_pipeline(&self.pipeline.blit_pipeline);
                pass.set_bind_group(0, &target.bind_group, &[]);
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
            let mut rgb = Vec::with_capacity((self.width * self.height * 3) as usize);
            for row in pixels.chunks(bytes_per_row as usize) {
                for pixel in row.chunks(4) {
                    rgb.extend_from_slice(&pixel[..3]);
                }
            }
            rgb
        }

        /// Uploads a colour stage for a pane; later renders of that pane use
        /// it until replaced.
        fn set_colouring(
            &self,
            pane: usize,
            colouring: &crate::colouring::Colouring,
            pixel_log: f32,
        ) {
            self.set_layers(pane, &LayerStack::single(colouring.clone()), pixel_log);
        }

        fn set_layers(&self, pane: usize, layers: &LayerStack, pixel_log: f32) {
            let table = GradientTable::new(
                self.pipeline.panes[pane]
                    .gradient_generation
                    .lock()
                    .unwrap()
                    .map_or(1, |generation| generation + 1),
                layers,
            );
            self.pipeline.upload_colouring(
                &self.queue,
                pane,
                &ColouringUniforms::new(layers, pixel_log),
                Some(&table),
            );
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
                true,
                false,
                precision,
                parameters.uniform_words(dynamical),
            );
            let rgb = gpu.render(pane, uniforms, None);
            gpu.write_ppm(&format!("{directory}/{name}.ppm"), &rgb);
        }
    }

    /// Renders the quadratic family through every colouring algorithm, at the
    /// default view and around an f64 reference at 1e12, and writes the
    /// images to `$ITERASCOPE_RENDER_DIR` when set. Every algorithm must
    /// produce a varied image (no collapsed or NaN-black output), and the
    /// distance estimate must stay varied at depth, where it is evaluated in
    /// logarithms. Run with
    /// `ITERASCOPE_RENDER_DIR=out cargo test --release gpu_colouring_gallery -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_colouring_gallery() {
        use crate::colouring::{ColouringAlgorithm, Gradient, TrapShape};
        use crate::family::{
            FamilyParameters, FractalFamily, initial_state_with, reference_orbit_f64,
        };
        let directory = std::env::var("ITERASCOPE_RENDER_DIR").ok();
        if let Some(directory) = &directory {
            std::fs::create_dir_all(directory).unwrap();
        }
        let gpu = GpuHarness::new(512, 352);
        let parameters = FamilyParameters::default();
        let family = FractalFamily::Quadratic;
        let iterations = 512;
        let bailout = 1e6_f32;
        let plane = family.default_parameter_view();
        let deep_point = boundary_point(
            family,
            &parameters,
            plane.centre,
            plane.half_height,
            false,
            family.default_parameter(),
            iterations,
        )
        .unwrap();
        let deep_half_height = 1.45 / 1e12;
        let orbit = reference_orbit_f64(
            family,
            &parameters,
            initial_state_with(family, deep_point, false, family.default_parameter()),
            iterations,
            bailout as f64,
        );
        let deep =
            DeepRenderData::from_f64_orbit(1, deep_half_height, &orbit.points, true, [0.0; 2]);

        let render = |colouring: &crate::colouring::Colouring, deep_zoom: bool| -> Vec<u8> {
            let (centre, half_height) = if deep_zoom {
                (deep_point, deep_half_height)
            } else {
                (plane.centre, plane.half_height)
            };
            let pixel_log = (2.0 * half_height / gpu.height as f64).ln() as f32;
            gpu.set_colouring(0, colouring, pixel_log);
            let mut uniforms = Uniforms::new(
                centre,
                half_height,
                gpu.aspect(),
                family.default_parameter(),
                iterations,
                bailout,
                family.shader_flag(),
                0,
                true,
                false,
                if deep_zoom {
                    PrecisionMode::DoubleSingle
                } else {
                    PrecisionMode::F32
                },
                parameters.uniform_words(false),
            );
            if deep_zoom {
                uniforms = uniforms.enable_perturbation(
                    deep.scale_mantissa,
                    deep.scale_exponent,
                    deep.reference.len(),
                    true,
                    [0.0; 2],
                );
                gpu.render(0, uniforms, Some(deep.reference.as_slice()))
            } else {
                gpu.render(0, uniforms, None)
            }
        };
        let distinct = |rgb: &[u8]| -> usize {
            let set: std::collections::HashSet<&[u8]> = rgb.chunks(3).collect();
            set.len()
        };

        for algorithm in ColouringAlgorithm::ALL {
            for deep_zoom in [false, true] {
                let mut colouring = crate::colouring::Colouring {
                    gradient: Gradient::random(5),
                    ..crate::colouring::Colouring::default()
                };
                colouring.outside.set_algorithm(algorithm);
                colouring
                    .inside
                    .set_algorithm(ColouringAlgorithm::OrbitTrap);
                colouring.inside.trap_shape = TrapShape::Cross;
                if algorithm == ColouringAlgorithm::Decomposition {
                    colouring.outside.sectors = 2;
                }
                if deep_zoom
                    && matches!(
                        algorithm,
                        ColouringAlgorithm::TriangleInequality | ColouringAlgorithm::Stripes
                    )
                {
                    // Deep orbits share hundreds of identical leading terms,
                    // so the averages vary little; artists raise the density.
                    colouring.outside.density = 60.0;
                }
                let rgb = render(&colouring, deep_zoom);
                let colours = distinct(&rgb);
                let black = rgb
                    .chunks(3)
                    .filter(|p| p[0] < 4 && p[1] < 4 && p[2] < 4)
                    .count();
                eprintln!(
                    "{:?} {}: {colours} distinct colours, {:.1}% near-black",
                    algorithm,
                    if deep_zoom { "at 1e12" } else { "default view" },
                    100.0 * black as f64 / (rgb.len() / 3) as f64,
                );
                if let Some(directory) = &directory {
                    let name = format!(
                        "colouring-{:?}-{}",
                        algorithm,
                        if deep_zoom { "deep" } else { "default" }
                    )
                    .to_lowercase();
                    gpu.write_ppm(&format!("{directory}/{name}.ppm"), &rgb);
                }
                let minimum = match algorithm {
                    // Two sectors through a random gradient: few outside
                    // colours, plus the inside trap shading.
                    ColouringAlgorithm::Decomposition => 8,
                    ColouringAlgorithm::Solid => 8,
                    // The closest approach to the trap happens in the part
                    // of the orbit every deep pixel shares, so the outside
                    // trap is flat at depth; only the inside trap varies.
                    // Skipping the shared prefix restores it — see below.
                    ColouringAlgorithm::OrbitTrap if deep_zoom => 8,
                    _ => 200,
                };
                assert!(
                    colours >= minimum,
                    "{algorithm:?} (deep {deep_zoom}) collapsed to {colours} colours"
                );
                assert!(
                    black < rgb.len() / 3 / 2,
                    "{algorithm:?} (deep {deep_zoom}) is mostly black"
                );
            }
        }

        // Diagnostics: the deep reference's length and the escape band of
        // sample pixels across the deep view.
        use crate::family::{OrbitFate, diagnose};
        eprintln!(
            "deep reference: {} points, escape {:?}",
            deep.reference.len(),
            orbit.escape_iteration
        );
        for offset in [-0.9f64, -0.45, 0.0, 0.45, 0.9] {
            let world = [
                deep_point[0] + offset * deep_half_height * gpu.aspect() as f64,
                deep_point[1] + offset * deep_half_height,
            ];
            let result = diagnose(
                family,
                &parameters,
                initial_state_with(family, world, false, family.default_parameter()),
                iterations,
                bailout as f64,
            );
            eprintln!(
                "  sample {offset:+.2}: fate {:?} at iteration {}",
                result.fate, result.iterations
            );
            let _ = OrbitFate::Bounded;
        }

        // Skipping the shared leading iterations restores the orbit trap's
        // variety at depth. Deep pixels differ only in their last few dozen
        // iterations (the escape band above sits at ~497-512), and a
        // chaotic orbit keeps revisiting the trap, so the shared prefix
        // dominates the minimum until the skip reaches almost the escape
        // time — exactly how the control is used: raise it until structure
        // appears.
        let mut best = 0usize;
        for skip in [0u32, 448, 480, 496, 504] {
            let mut colouring = crate::colouring::Colouring {
                gradient: Gradient::random(5),
                ..crate::colouring::Colouring::default()
            };
            colouring
                .outside
                .set_algorithm(ColouringAlgorithm::OrbitTrap);
            let mut stack = LayerStack::single(colouring);
            stack.layers[0].skip_iterations = skip;
            let pixel_log = (2.0 * deep_half_height / gpu.height as f64).ln() as f32;
            gpu.set_layers(0, &stack, pixel_log);
            let mut uniforms = Uniforms::new(
                deep_point,
                deep_half_height,
                gpu.aspect(),
                family.default_parameter(),
                iterations,
                bailout,
                family.shader_flag(),
                0,
                true,
                false,
                PrecisionMode::DoubleSingle,
                parameters.uniform_words(false),
            );
            uniforms = uniforms.enable_perturbation(
                deep.scale_mantissa,
                deep.scale_exponent,
                deep.reference.len(),
                true,
                [0.0; 2],
            );
            let rgb = gpu.render(0, uniforms, Some(deep.reference.as_slice()));
            let colours = distinct(&rgb);
            eprintln!("deep outside trap with skip {skip}: {colours} distinct colours");
            if let Some(directory) = &directory {
                gpu.write_ppm(&format!("{directory}/trap-skip-{skip:03}.ppm"), &rgb);
            }
            if skip > 0 {
                best = best.max(colours);
            }
        }
        assert!(
            best > 300,
            "no skip revived the deep trap (best {best} colours)"
        );
    }

    /// Renders the quadratic Julia plane at 1e4000× (Ultra Fractal 5's
    /// limit) around an arbitrary-precision reference at a repelling fixed
    /// point, through the iteration, distance-estimate, stripe and
    /// triangle-inequality colourings, and checks each stays varied and free
    /// of NaN (black) output. Ignored: needs a GPU and a long AP orbit. Run
    /// with `ITERASCOPE_RENDER_DIR=out cargo test --release gpu_colouring_at_uf_limit -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_colouring_at_uf_limit() {
        use crate::arbitrary::{DeepComplex, DeepReal, DeepState, DeepView, ReferenceOrbit};
        use crate::colouring::ColouringAlgorithm;
        use crate::family::{FamilyParameters, FractalFamily};

        let directory = std::env::var("ITERASCOPE_RENDER_DIR").ok();
        let gpu = GpuHarness::new(384, 256);
        let parameters = FamilyParameters::default();
        let family = FractalFamily::Quadratic;
        let zoom_exponent = 4000u32;
        let precision_exponent = zoom_exponent + 40;
        // Structure around a repelling fixed point appears after
        // log(1e4000)/log(multiplier) iterations.
        let iterations = 24_000u32;
        let c_f64 = family.default_parameter();
        let c = DeepComplex::from_f64(c_f64, precision_exponent).unwrap();
        let mut best: Option<(DeepComplex, f64)> = None;
        for start in [
            [0.6, 0.4],
            [-0.5, 0.7],
            [1.1, -0.3],
            [-1.2, 0.1],
            [0.2, -0.9],
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
        let (mut centre, multiplier) = best.expect("a repelling fixed point");
        // The generic helper converges to roughly 1e-85; refine with the
        // quadratic's exact Newton step (z² + c − z) / (2z − 1), which doubles
        // the correct digits each time.
        let one = centre.real_like(1.0);
        let two = centre.real_like(2.0);
        // A fixed count: an f64 projection of the step cannot tell 1e-400
        // from zero, so there is nothing cheap to test for convergence.
        for _ in 0..20 {
            let g = centre.mul(&centre).add(&c).sub(&centre);
            let derivative = two.mul(&centre).sub(&one);
            centre = centre.sub(&g.div(&derivative));
        }
        let residual = centre.mul(&centre).add(&c).sub(&centre).to_f64_pair();
        eprintln!("fixed point residual (f64 projection) {residual:?}");
        let expected_structure_at =
            (zoom_exponent as f64 * std::f64::consts::LN_10) / multiplier.ln();
        eprintln!(
            "fixed point multiplier {multiplier:.3}; structure expected after ~{expected_structure_at:.0} iterations"
        );
        let view = DeepView {
            centre: centre.clone(),
            half_height: DeepReal::parse(&format!("1.45e-{zoom_exponent}"), precision_exponent)
                .unwrap(),
            zoom_exponent: precision_exponent,
            magnification_log10: zoom_exponent as f64,
        };
        let start = std::time::Instant::now();
        let initial = DeepState::initial(family, &view.centre, true, &c).unwrap();
        let orbit = ReferenceOrbit::family(family, &parameters, initial, iterations, 4.0).unwrap();
        eprintln!(
            "AP reference: {} points in {:.1} s",
            orbit.points.len(),
            start.elapsed().as_secs_f64()
        );
        let (mantissa, exponent) = view.half_height.scaled_f32();
        let data = DeepRenderData::from_reference(1, mantissa, exponent, &orbit, false);
        let uniforms = Uniforms::new(
            view.centre_preview(),
            view.half_height_preview(),
            gpu.aspect(),
            c_f64,
            iterations,
            4.0,
            family.shader_flag(),
            1,
            true,
            false,
            PrecisionMode::DoubleSingle,
            parameters.uniform_words(true),
        )
        .enable_perturbation(mantissa, exponent, data.reference.len(), false, [0.0; 2]);
        // ln of one pixel's height at 1e4000: far outside f64 range as a
        // value, finite as a logarithm.
        let pixel_log = (std::f64::consts::LN_10 * (1.45_f64.log10() - zoom_exponent as f64)
            + (2.0 / gpu.height as f64).ln()) as f32;
        assert!(pixel_log.is_finite());

        for (label, algorithm, density) in [
            ("iteration", ColouringAlgorithm::Iteration, 0.035),
            ("distance", ColouringAlgorithm::DistanceEstimate, 0.25),
            ("stripes", ColouringAlgorithm::Stripes, 60.0),
            ("triangle", ColouringAlgorithm::TriangleInequality, 60.0),
        ] {
            let mut colouring = crate::colouring::Colouring::default();
            colouring.outside.set_algorithm(algorithm);
            colouring.outside.density = density;
            gpu.set_colouring(1, &colouring, pixel_log);
            let start = std::time::Instant::now();
            let rgb = gpu.render(1, uniforms, Some(data.reference.as_slice()));
            let elapsed = start.elapsed();
            let distinct: std::collections::HashSet<&[u8]> = rgb.chunks(3).collect();
            let black = rgb
                .chunks(3)
                .filter(|p| p[0] < 4 && p[1] < 4 && p[2] < 4)
                .count();
            eprintln!(
                "{label} at 1e{zoom_exponent}: {} distinct colours, {:.1}% near-black, {:.1} ms",
                distinct.len(),
                100.0 * black as f64 / (rgb.len() / 3) as f64,
                elapsed.as_secs_f64() * 1e3
            );
            if let Some(directory) = &directory {
                std::fs::create_dir_all(directory).unwrap();
                gpu.write_ppm(&format!("{directory}/uf-limit-{label}.ppm"), &rgb);
            }
            assert!(
                distinct.len() >= 200,
                "{label} collapsed at 1e{zoom_exponent}"
            );
            assert!(
                black < rgb.len() / 3 / 2,
                "{label} is mostly black at 1e{zoom_exponent}"
            );
        }
    }

    /// Exports a five-frame zoom path through `render_export` — plain f32,
    /// f64-reference perturbation and arbitrary-precision perturbation, at a
    /// width whose rows need copy padding — and checks that every frame is
    /// varied, that consecutive frames differ (the zoom actually moves), and
    /// that the padded readback carries no artefacts. Run with
    /// `ITERASCOPE_RENDER_DIR=out cargo test --release gpu_zoom_export_frames -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_zoom_export_frames() {
        use crate::animation::{self, ZoomAnimation};
        use crate::arbitrary::{DeepComplex, DeepReal, DeepState, DeepView, ReferenceOrbit};
        use crate::family::{FamilyParameters, FractalFamily};

        let directory = std::env::var("ITERASCOPE_RENDER_DIR").ok();
        if let Some(directory) = &directory {
            std::fs::create_dir_all(directory).unwrap();
        }
        let gpu = GpuHarness::new(64, 64); // harness unused; device via its fields
        let parameters = FamilyParameters::default();
        let family = FractalFamily::Quadratic;
        let iterations = 2_048u32;
        // 322 × 4 = 1288 bytes per row: not a multiple of 256, so the copy
        // path must pad and the readback must strip the padding.
        let size = (322u32, 240u32);
        let zoom_exponent = 30u32;
        let precision_exponent = zoom_exponent + 40;
        let c_f64 = family.default_parameter();
        let c = DeepComplex::from_f64(c_f64, precision_exponent).unwrap();
        let (centre, _) = [[0.6, 0.4], [-0.5, 0.7], [1.1, -0.3]]
            .iter()
            .filter_map(|start| {
                repelling_fixed_point(family, &parameters, &c, *start, precision_exponent)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("a repelling fixed point");
        let view = DeepView {
            centre: centre.clone(),
            half_height: DeepReal::parse(&format!("1.45e-{zoom_exponent}"), precision_exponent)
                .unwrap(),
            zoom_exponent: precision_exponent,
            magnification_log10: zoom_exponent as f64,
        };
        let initial = DeepState::initial(family, &view.centre, true, &c).unwrap();
        let orbit = ReferenceOrbit::family(family, &parameters, initial, iterations, 4.0).unwrap();
        let centre_f64 = view.centre_preview();
        let f64_points: Vec<[f64; 2]> = orbit
            .points
            .iter()
            .map(|point| [point.re.to_f64(), point.im.to_f64()])
            .collect();
        let ap = DeepRenderData::from_points(11, 1.0, 0, &orbit.points, false).reference;
        let f64_reference =
            DeepRenderData::from_f64_orbit(12, 1.0, &f64_points, true, [0.0; 2]).reference;

        let animation = ZoomAnimation {
            duration_seconds: 0.2,
            fps: 25,
            width: size.0,
            height: size.1,
            start_magnification_log10: 0.0,
            end_magnification_log10: zoom_exponent as f64,
            ease: false,
            gradient_sweep_turns: 0.25,
            encode_video: false,
        };
        assert_eq!(animation.frame_count(), 5);
        let handoff_log = crate::arbitrary::ARBITRARY_HANDOFF_ZOOM.log10();
        let layers_base = LayerStack::default();
        let gradient = GradientTable::new(13, &layers_base);

        let mut previous: Option<Vec<u8>> = None;
        for frame in 0..animation.frame_count() {
            let magnification = animation.magnification_log10_at(frame);
            let (mantissa, exponent) = animation::frame_scale(magnification);
            let reference = if magnification > handoff_log {
                Some((11u64, ap.as_slice(), false))
            } else {
                Some((12u64, f64_reference.as_slice(), true))
            };
            let mut uniforms = Uniforms::new(
                centre_f64,
                animation::frame_half_height_f64(magnification),
                size.0 as f32 / size.1 as f32,
                c_f64,
                iterations,
                4.0,
                family.shader_flag(),
                1,
                true,
                false,
                PrecisionMode::DoubleSingle,
                parameters.uniform_words(true),
            );
            if let Some((_, points, ds_fallback)) = reference {
                uniforms = uniforms.enable_perturbation(
                    mantissa,
                    exponent,
                    points.len(),
                    ds_fallback,
                    [0.0; 2],
                );
            }
            let mut layers = layers_base.clone();
            layers.layers[0].colouring.outside.offset += animation.gradient_offset_at(frame);
            let rgba = gpu.pipeline.render_export(
                &gpu.device,
                &gpu.queue,
                &uniforms,
                &ColouringUniforms::new(&layers, animation::frame_pixel_log(magnification, size.1)),
                &gradient,
                reference.map(|(generation, points, _)| (generation, points)),
                size,
            );
            assert_eq!(rgba.len(), (size.0 * size.1 * 4) as usize);
            let distinct: std::collections::HashSet<&[u8]> = rgba.chunks(4).collect();
            eprintln!(
                "frame {frame} at 1e{magnification:.1}: {} distinct colours",
                distinct.len()
            );
            assert!(
                distinct.len() > 50,
                "frame {frame} at 1e{magnification:.1} collapsed"
            );
            // Alpha is opaque everywhere (no padding bleed into the rows).
            assert!(rgba.chunks(4).all(|pixel| pixel[3] == 255));
            if let Some(previous) = &previous {
                assert_ne!(
                    previous, &rgba,
                    "frame {frame} identical to its predecessor"
                );
            }
            if let Some(directory) = &directory {
                let rgb: Vec<u8> = rgba
                    .chunks(4)
                    .flat_map(|pixel| pixel[..3].to_vec())
                    .collect();
                let mut ppm = format!("P6\n{} {}\n255\n", size.0, size.1).into_bytes();
                ppm.extend_from_slice(&rgb);
                std::fs::write(format!("{directory}/export-frame-{frame}.ppm"), ppm).unwrap();
            }
            previous = Some(rgba);
        }
    }

    /// Layer compositing: a single-layer stack must reproduce the
    /// pre-layer renderer byte for byte; adding layers must change the
    /// image; a fully transparent top layer must not; merge modes must
    /// differ from one another. Run with
    /// `ITERASCOPE_RENDER_DIR=out cargo test --release gpu_layer_composite -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_layer_composite() {
        use crate::colouring::{ColouringAlgorithm, Gradient, Layer, MergeMode, TrapShape};
        use crate::family::{FamilyParameters, FractalFamily};

        let directory = std::env::var("ITERASCOPE_RENDER_DIR").ok();
        let gpu = GpuHarness::new(512, 352);
        let parameters = FamilyParameters::default();
        let family = FractalFamily::Quadratic;
        let plane = family.default_parameter_view();
        let uniforms = Uniforms::new(
            plane.centre,
            plane.half_height,
            gpu.aspect(),
            family.default_parameter(),
            512,
            4.0,
            family.shader_flag(),
            0,
            true,
            false,
            PrecisionMode::F32,
            parameters.uniform_words(false),
        );
        let pixel_log = (2.0 * plane.half_height / gpu.height as f64).ln() as f32;
        let render = |layers: &LayerStack| -> Vec<u8> {
            gpu.set_layers(0, layers, pixel_log);
            gpu.render(0, uniforms, None)
        };

        // 1. One layer == the plain colouring path.
        let base = crate::colouring::Colouring::default();
        let single = render(&LayerStack::single(base.clone()));
        gpu.set_colouring(0, &base, pixel_log);
        let plain = gpu.render(0, uniforms, None);
        assert_eq!(single, plain, "a single-layer stack changed the image");

        // 2. A stripes layer on top changes the image; at opacity 0 it
        // does not; hidden it does not.
        let mut stripes = Layer {
            merge_mode: MergeMode::Multiply,
            ..Layer::default()
        };
        stripes.colouring.gradient = Gradient::random(9);
        stripes
            .colouring
            .outside
            .set_algorithm(ColouringAlgorithm::Stripes);
        stripes
            .colouring
            .inside
            .set_algorithm(ColouringAlgorithm::OrbitTrap);
        stripes.colouring.inside.trap_shape = TrapShape::Cross;
        let mut stack = LayerStack::single(base.clone());
        stack.layers.push(stripes);
        let composed = render(&stack);
        assert_ne!(composed, single, "the second layer had no effect");
        let distinct: std::collections::HashSet<&[u8]> = composed.chunks(3).collect();
        eprintln!("two-layer composite: {} distinct colours", distinct.len());
        assert!(distinct.len() > 200);
        if let Some(directory) = &directory {
            std::fs::create_dir_all(directory).unwrap();
            gpu.write_ppm(&format!("{directory}/layers-single.ppm"), &single);
            gpu.write_ppm(&format!("{directory}/layers-composite.ppm"), &composed);
        }
        stack.layers[1].opacity = 0.0;
        assert_eq!(render(&stack), single, "opacity 0 still changed the image");
        stack.layers[1].opacity = 1.0;
        stack.layers[1].visible = false;
        assert_eq!(
            render(&stack),
            single,
            "a hidden layer still changed the image"
        );
        stack.layers[1].visible = true;

        // 3. Merge modes are distinct.
        let mut by_mode: Vec<Vec<u8>> = Vec::new();
        for mode in MergeMode::ALL {
            stack.layers[1].merge_mode = mode;
            let image = render(&stack);
            for (previous_mode, previous) in MergeMode::ALL.iter().zip(&by_mode) {
                assert_ne!(
                    &image, previous,
                    "{mode:?} renders identically to {previous_mode:?}"
                );
            }
            by_mode.push(image);
        }

        // 4. Masks: a solid-white mask between base and top changes
        // nothing; a solid-black mask hides the top layer entirely; a
        // half-strength mask sits in between.
        let two_layer = render(&stack);
        let mut masked = stack.clone();
        let mut mask = Layer {
            mask: true,
            ..Layer::default()
        };
        mask.colouring
            .outside
            .set_algorithm(ColouringAlgorithm::Solid);
        mask.colouring
            .inside
            .set_algorithm(ColouringAlgorithm::Solid);
        mask.colouring.outside.solid = [1.0, 1.0, 1.0];
        mask.colouring.inside.solid = [1.0, 1.0, 1.0];
        masked.layers.insert(1, mask);
        assert_eq!(
            render(&masked),
            two_layer,
            "a solid-white mask changed the image"
        );
        masked.layers[1].colouring.outside.solid = [0.0; 3];
        masked.layers[1].colouring.inside.solid = [0.0; 3];
        let black_masked = render(&masked);
        let base_only = render(&LayerStack::single(base.clone()));
        // The masked stack still carries the top layer's accumulators and so
        // renders through the orbit-statistics pipeline variant, whose
        // instruction scheduling wiggles the last bit; compare with a small
        // tolerance rather than byte-for-byte.
        let differing = black_masked
            .chunks(3)
            .zip(base_only.chunks(3))
            .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 2))
            .count();
        assert!(
            differing < black_masked.len() / 3 / 200,
            "a solid-black mask did not hide the layer above ({differing} pixels differ)"
        );
        masked.layers[1].opacity = 0.5;
        let half_masked = render(&masked);
        assert_ne!(half_masked, two_layer);
        assert_ne!(half_masked, base_only);
        // A gradient-driven mask produces a spatially varying blend.
        masked.layers[1].opacity = 1.0;
        masked.layers[1]
            .colouring
            .outside
            .set_algorithm(ColouringAlgorithm::Iteration);
        let varying = render(&masked);
        assert_ne!(varying, two_layer);
        assert_ne!(varying, base_only);
        let distinct: std::collections::HashSet<&[u8]> = varying.chunks(3).collect();
        eprintln!(
            "gradient-masked composite: {} distinct colours",
            distinct.len()
        );
        assert!(distinct.len() > 200);
        if let Some(directory) = &directory {
            gpu.write_ppm(&format!("{directory}/layers-masked.ppm"), &varying);
        }

        // 5. The stack survives the maximum depth: all eight layers.
        while stack.layers.len() < crate::colouring::MAX_LAYERS {
            let mut layer = stack.layers[1].clone();
            layer.opacity = 0.35;
            stack.layers.push(layer);
        }
        let full = render(&stack);
        let distinct: std::collections::HashSet<&[u8]> = full.chunks(3).collect();
        eprintln!("eight-layer composite: {} distinct colours", distinct.len());
        assert!(distinct.len() > 200);
        if let Some(directory) = &directory {
            gpu.write_ppm(&format!("{directory}/layers-eight.ppm"), &full);
        }
    }

    /// Exported buffers are top-first with positive imaginary values up, and
    /// a `region_view` of a quadrant reproduces the crop of the whole frame.
    /// This pins the orientation convention: the readback must not flip rows
    /// (texture row 0 is already the top of the viewport). Run with
    /// `cargo test --release gpu_export_orientation_and_regions -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_export_orientation_and_regions() {
        use crate::animation::region_view;
        use crate::family::{FamilyParameters, FractalFamily};
        let gpu = GpuHarness::new(64, 64);
        let parameters = FamilyParameters::default();
        let family = FractalFamily::Quadratic;
        let full = (512u32, 352u32);
        let layers = LayerStack::default();
        let gradient = GradientTable::new(41, &layers);
        let centre = [-0.6f64, 0.0];
        let half = 1.45f64;
        let render = |centre: [f64; 2], half: f64, aspect: f32, size: (u32, u32)| -> Vec<u8> {
            let uniforms = Uniforms::new(
                centre,
                half,
                aspect,
                [0.0; 2],
                512,
                4.0,
                family.shader_flag(),
                0,
                true,
                false,
                PrecisionMode::F32,
                parameters.uniform_words(false),
            );
            gpu.pipeline.render_export(
                &gpu.device,
                &gpu.queue,
                &uniforms,
                &ColouringUniforms::new(&layers, -5.0),
                &gradient,
                None,
                size,
            )
        };
        let whole = render(centre, half, full.0 as f32 / full.1 as f32, full);
        // Crop the top-right quadrant of the whole.
        let (tw, th) = (256usize, 176usize);
        let mut crop = Vec::new();
        for y in 0..th {
            let base = (y * full.0 as usize + 256) * 4;
            crop.extend_from_slice(&whole[base..base + tw * 4]);
        }
        // Manual quadrant uniforms.
        let manual = render(
            [centre[0] + 1.0545234, centre[1] + 0.725],
            0.725,
            256.0 / 176.0,
            (256, 176),
        );
        // region_view quadrant.
        let region = region_view(0.0, full, (256, 0), (256, 176));
        eprintln!("region = {region:?}");
        let via_region = render(
            [
                centre[0] + region.centre_shift[0],
                centre[1] + region.centre_shift[1],
            ],
            region.half_height_f64,
            region.aspect,
            (256, 176),
        );
        let diff = |a: &[u8], b: &[u8], label: &str| {
            let differing = a
                .chunks(4)
                .zip(b.chunks(4))
                .filter(|(x, y)| x.iter().zip(y.iter()).any(|(p, q)| p.abs_diff(*q) > 2))
                .count();
            eprintln!(
                "{label}: {:.2}% differ",
                100.0 * differing as f64 / (a.len() / 4) as f64
            );
        };
        diff(&crop, &manual, "crop vs manual");
        diff(&crop, &via_region, "crop vs region_view");
        diff(&manual, &via_region, "manual vs region_view");

        // Empirical convention check: which crop does which shift match?
        let crop_at = |x0: usize, y0: usize| -> Vec<u8> {
            let mut out = Vec::new();
            for y in 0..th {
                let base = ((y0 + y) * full.0 as usize + x0) * 4;
                out.extend_from_slice(&whole[base..base + tw * 4]);
            }
            out
        };
        let count_close = |a: &[u8], b: &[u8]| -> f64 {
            let same = a
                .chunks(4)
                .zip(b.chunks(4))
                .filter(|(x, y)| x.iter().zip(y.iter()).all(|(p, q)| p.abs_diff(*q) <= 2))
                .count();
            100.0 * same as f64 / (a.len() / 4) as f64
        };
        let mut matched = Vec::new();
        for (sx, sy) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
            let tile = render(
                [centre[0] + sx * 1.0545454, centre[1] + sy * 0.725],
                0.725,
                256.0 / 176.0,
                (256, 176),
            );
            for (x0, y0, name) in [
                (0usize, 0usize, "TL"),
                (256, 0, "TR"),
                (0, 176, "BL"),
                (256, 176, "BR"),
            ] {
                let close = count_close(&crop_at(x0, y0), &tile);
                if close > 50.0 {
                    eprintln!("shift ({sx:+},{sy:+}) matches crop {name}: {close:.1}% close");
                    matched.push(((sx as i8, sy as i8), name));
                }
            }
        }
        // +x,+y must be the top-right crop: rows top-first, +Im up.
        assert!(matched.contains(&((1, 1), "TR")), "{matched:?}");
        assert!(matched.contains(&((-1, -1), "BL")), "{matched:?}");
    }

    /// Tiled rendering: a frame assembled from 2×2 regions rendered around
    /// the same reference orbit must match the frame rendered whole — at a
    /// shallow f32 view and at 1e30 through the arbitrary-precision
    /// perturbation path — up to the last-bit rounding of per-tile local
    /// coordinates. Run with
    /// `ITERASCOPE_RENDER_DIR=out cargo test --release gpu_tiled_render_matches_whole -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_tiled_render_matches_whole() {
        use crate::animation::region_view;
        use crate::arbitrary::{DeepComplex, DeepReal, DeepState, DeepView, ReferenceOrbit};
        use crate::family::{FamilyParameters, FractalFamily};

        let directory = std::env::var("ITERASCOPE_RENDER_DIR").ok();
        let gpu = GpuHarness::new(64, 64);
        let parameters = FamilyParameters::default();
        let family = FractalFamily::Quadratic;
        let iterations = 2_048u32;
        let full = (512u32, 352u32);
        let layers = LayerStack::default();
        let gradient = GradientTable::new(31, &layers);
        let zoom_exponent = 30u32;
        let precision_exponent = zoom_exponent + 40;
        let c_f64 = family.default_parameter();
        let c = DeepComplex::from_f64(c_f64, precision_exponent).unwrap();
        let (fixed_point, _) = [[0.6, 0.4], [-0.5, 0.7], [1.1, -0.3]]
            .iter()
            .filter_map(|start| {
                repelling_fixed_point(family, &parameters, &c, *start, precision_exponent)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("a repelling fixed point");
        let view = DeepView {
            centre: fixed_point.clone(),
            half_height: DeepReal::parse(&format!("1.45e-{zoom_exponent}"), precision_exponent)
                .unwrap(),
            zoom_exponent: precision_exponent,
            magnification_log10: zoom_exponent as f64,
        };
        let initial = DeepState::initial(family, &view.centre, true, &c).unwrap();
        let orbit = ReferenceOrbit::family(family, &parameters, initial, iterations, 4.0).unwrap();
        let ap = DeepRenderData::from_points(32, 1.0, 0, &orbit.points, false).reference;
        let deep_centre = view.centre_preview();

        // (magnification, julia?, label): a shallow parameter-plane view on
        // the plain f32 path and the deep Julia view on the AP path.
        for (magnification, julia, label) in [(0.0f64, false, "shallow"), (30.0, true, "deep")] {
            let render_region = |origin: (u32, u32), size: (u32, u32)| -> Vec<u8> {
                let region = region_view(magnification, full, origin, size);
                let centre = if julia { deep_centre } else { [-0.6, 0.0] };
                let mut uniforms = Uniforms::new(
                    [
                        centre[0] + region.centre_shift[0],
                        centre[1] + region.centre_shift[1],
                    ],
                    region.half_height_f64,
                    region.aspect,
                    c_f64,
                    iterations,
                    4.0,
                    family.shader_flag(),
                    usize::from(julia),
                    true,
                    false,
                    if julia {
                        PrecisionMode::DoubleSingle
                    } else {
                        PrecisionMode::F32
                    },
                    parameters.uniform_words(julia),
                );
                let mut reference = None;
                if julia {
                    uniforms = uniforms.enable_perturbation(
                        region.scale_mantissa,
                        region.scale_exponent,
                        ap.len(),
                        false,
                        region.reference_offset,
                    );
                    reference = Some((32u64, ap.as_slice()));
                }
                gpu.pipeline.render_export(
                    &gpu.device,
                    &gpu.queue,
                    &uniforms,
                    &ColouringUniforms::new(&layers, -5.0),
                    &gradient,
                    reference,
                    size,
                )
            };

            let whole = render_region((0, 0), full);
            let mut assembled = vec![0u8; whole.len()];
            for (row_offset, tile_height) in crate::animation::tile_spans(full.1, 200) {
                for (column_offset, tile_width) in crate::animation::tile_spans(full.0, 300) {
                    let tile =
                        render_region((column_offset, row_offset), (tile_width, tile_height));
                    for y in 0..tile_height as usize {
                        let source = y * tile_width as usize * 4..(y + 1) * tile_width as usize * 4;
                        let target = ((row_offset as usize + y) * full.0 as usize
                            + column_offset as usize)
                            * 4;
                        assembled[target..target + tile_width as usize * 4]
                            .copy_from_slice(&tile[source]);
                    }
                }
            }
            let differing = whole
                .chunks(4)
                .zip(assembled.chunks(4))
                .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 2))
                .count();
            let fraction = differing as f64 / (whole.len() / 4) as f64;
            eprintln!(
                "{label}: {:.3}% of pixels differ between whole and tiled",
                100.0 * fraction
            );
            if let Some(directory) = &directory {
                std::fs::create_dir_all(directory).unwrap();
                for (name, rgba) in [("whole", &whole), ("tiled", &assembled)] {
                    let rgb: Vec<u8> = rgba.chunks(4).flat_map(|p| p[..3].to_vec()).collect();
                    let mut ppm = format!("P6\n{} {}\n255\n", full.0, full.1).into_bytes();
                    ppm.extend_from_slice(&rgb);
                    std::fs::write(format!("{directory}/tiling-{label}-{name}.ppm"), ppm).unwrap();
                }
            }
            assert!(
                fraction < 0.01,
                "{label}: tiled render differs from whole on {:.2}% of pixels",
                100.0 * fraction
            );
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
                    true,
                    false,
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
                    true,
                    false,
                    PrecisionMode::DoubleSingle,
                    words,
                )
                .enable_perturbation(
                    mantissa,
                    exponent,
                    data.reference.len(),
                    true,
                    [0.0; 2],
                );
                let perturbed =
                    gpu.render(pane, perturbed_uniforms, Some(data.reference.as_slice()));
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
                let f64_data = DeepRenderData::from_f64_orbit(
                    2,
                    half_height,
                    &f64_orbit.points,
                    true,
                    [0.0; 2],
                );
                let f64_uniforms = Uniforms::new(
                    point,
                    half_height,
                    gpu.aspect(),
                    family.default_parameter(),
                    iterations,
                    4.0,
                    family.shader_flag(),
                    pane,
                    true,
                    false,
                    PrecisionMode::DoubleSingle,
                    words,
                )
                .enable_perturbation(
                    f64_data.scale_mantissa,
                    f64_data.scale_exponent,
                    f64_data.reference.len(),
                    true,
                    [0.0; 2],
                );
                let f64_perturbed =
                    gpu.render(pane, f64_uniforms, Some(f64_data.reference.as_slice()));
                // And once more around an off-centre reference point; the
                // image must not depend on where the reference sits.
                let offset = [0.31f32, -0.22f32];
                let offset_point = [
                    point[0] + offset[0] as f64 * half_height,
                    point[1] + offset[1] as f64 * half_height,
                ];
                let offset_orbit = crate::family::reference_orbit_f64(
                    family,
                    &parameters,
                    crate::family::initial_state_with(
                        family,
                        offset_point,
                        dynamical,
                        family.default_parameter(),
                    ),
                    iterations,
                    4.0,
                );
                let offset_data = DeepRenderData::from_f64_orbit(
                    3,
                    half_height,
                    &offset_orbit.points,
                    true,
                    offset,
                );
                let offset_uniforms = Uniforms::new(
                    point,
                    half_height,
                    gpu.aspect(),
                    family.default_parameter(),
                    iterations,
                    4.0,
                    family.shader_flag(),
                    pane,
                    true,
                    false,
                    PrecisionMode::DoubleSingle,
                    words,
                )
                .enable_perturbation(
                    offset_data.scale_mantissa,
                    offset_data.scale_exponent,
                    offset_data.reference.len(),
                    true,
                    offset,
                );
                let offset_perturbed = gpu.render(
                    pane,
                    offset_uniforms,
                    Some(offset_data.reference.as_slice()),
                );
                // Same off-centre reference through the scaled instantiation.
                let mut scaled_offset_uniforms = offset_uniforms;
                scaled_offset_uniforms.deep[0] = 2.0;
                let scaled_offset_perturbed = gpu.render(
                    pane,
                    scaled_offset_uniforms,
                    Some(offset_data.reference.as_slice()),
                );

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
                let mut offset_differing = 0usize;
                for index in 0..pixels {
                    let a = &offset_perturbed[index * 3..index * 3 + 3];
                    let b = &f64_perturbed[index * 3..index * 3 + 3];
                    let delta: i32 = (0..3).map(|k| (a[k] as i32 - b[k] as i32).abs()).sum();
                    if delta > 60 {
                        offset_differing += 1;
                    }
                }
                let offset_fraction = offset_differing as f64 / pixels as f64;
                let scaled_offset_differing = scaled_offset_perturbed
                    .chunks(3)
                    .zip(f64_perturbed.chunks(3))
                    .filter(|(a, b)| {
                        (0..3)
                            .map(|k| (a[k] as i32 - b[k] as i32).abs())
                            .sum::<i32>()
                            > 60
                    })
                    .count();
                let scaled_offset_fraction = scaled_offset_differing as f64 / pixels as f64;
                let distinct_ds: std::collections::HashSet<&[u8]> = ds.chunks(3).collect();
                let distinct_perturbed: std::collections::HashSet<&[u8]> =
                    perturbed.chunks(3).collect();
                if distinct_ds.len() > 8 && distinct_perturbed.len() <= 1 {
                    collapsed.push(format!("{family:?} pane {pane}"));
                }
                report.push(format!(
                    "{:26} pane {pane}: reference {} points (ends {:?}), DS vs AP {:.2}%, f64 vs AP {:.2}%, off-centre vs centred {:.2}% (scaled {:.2}%) pixels differ",
                    format!("{family:?}"),
                    orbit.points.len(),
                    orbit.escape_iteration,
                    100.0 * fraction,
                    100.0 * f64_fraction,
                    100.0 * offset_fraction,
                    100.0 * scaled_offset_fraction
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
            true,
            false,
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
                true,
                false,
                PrecisionMode::DoubleSingle,
                parameters.uniform_words(dynamical),
            )
            .enable_perturbation(
                mantissa,
                exponent,
                data.reference.len(),
                false,
                [0.0; 2],
            );
            let rgb = gpu.render(pane, uniforms, Some(data.reference.as_slice()));
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

    /// Timing probe: wall time of one 1024x1024 render per path.
    #[test]
    #[ignore]
    fn gpu_timing_probe() {
        use crate::arbitrary::{DeepComplex, DeepReal, DeepState, DeepView, ReferenceOrbit};
        use crate::family::{FamilyParameters, FractalFamily};
        let gpu = GpuHarness::new(1024, 1024);
        let parameters = FamilyParameters::default();
        let iterations = 512u32;
        for family in [
            FractalFamily::Quadratic,
            FractalFamily::Multibrot,
            FractalFamily::BurningShip,
            FractalFamily::MagnetOne,
            FractalFamily::Mandelbox,
        ] {
            let plane = family.default_parameter_view();
            let centre = plane.centre;
            for (label, precision, perturb) in [
                ("f32", PrecisionMode::F32, false),
                ("ds", PrecisionMode::DoubleSingle, false),
                ("pert-f64", PrecisionMode::DoubleSingle, true),
            ] {
                let mut uniforms = Uniforms::new(
                    centre,
                    plane.half_height * 1e-6,
                    gpu.aspect(),
                    family.default_parameter(),
                    iterations,
                    4.0,
                    family.shader_flag(),
                    0,
                    true,
                    false,
                    precision,
                    parameters.uniform_words(false),
                );
                let mut reference = None;
                if perturb {
                    let orbit = crate::family::reference_orbit_f64(
                        family,
                        &parameters,
                        crate::family::initial_state_with(
                            family,
                            centre,
                            false,
                            family.default_parameter(),
                        ),
                        iterations,
                        4.0,
                    );
                    let data = DeepRenderData::from_f64_orbit(
                        1,
                        plane.half_height * 1e-6,
                        &orbit.points,
                        true,
                        [0.0; 2],
                    );
                    uniforms = uniforms.enable_perturbation(
                        data.scale_mantissa,
                        data.scale_exponent,
                        data.reference.len(),
                        true,
                        [0.0; 2],
                    );
                    reference = Some(data);
                }
                // Warm up, then time.
                let _ = gpu.render(
                    0,
                    uniforms,
                    reference.as_ref().map(|d| d.reference.as_slice()),
                );
                let start = std::time::Instant::now();
                let _ = gpu.render(
                    0,
                    uniforms,
                    reference.as_ref().map(|d| d.reference.as_slice()),
                );
                let elapsed = start.elapsed();
                eprintln!("{family:?} {label}: {:.1} ms", elapsed.as_secs_f64() * 1e3);

                // The same render through the orbit-statistics variant with
                // every accumulator live, to record what the trap, average
                // and distance-estimate colourings cost on top.
                let mut heavy = crate::colouring::Colouring::default();
                heavy
                    .outside
                    .set_algorithm(crate::colouring::ColouringAlgorithm::TriangleInequality);
                heavy
                    .inside
                    .set_algorithm(crate::colouring::ColouringAlgorithm::OrbitTrap);
                let mut extra = crate::colouring::Colouring::default();
                extra
                    .outside
                    .set_algorithm(crate::colouring::ColouringAlgorithm::Stripes);
                extra
                    .inside
                    .set_algorithm(crate::colouring::ColouringAlgorithm::DistanceEstimate);
                // Needs flags are the union of both sides; merge by taking
                // the outside of `heavy` and inside of `extra` plus traps.
                let mut all = heavy.clone();
                all.inside = extra.inside;
                gpu.set_colouring(0, &heavy, -10.0);
                let _ = gpu.render(
                    0,
                    uniforms,
                    reference.as_ref().map(|d| d.reference.as_slice()),
                );
                let start = std::time::Instant::now();
                let _ = gpu.render(
                    0,
                    uniforms,
                    reference.as_ref().map(|d| d.reference.as_slice()),
                );
                let with_stats = start.elapsed();
                eprintln!(
                    "{family:?} {label} + orbit stats (TIA/trap): {:.1} ms",
                    with_stats.as_secs_f64() * 1e3
                );
                gpu.set_colouring(0, &all, -10.0);
                let _ = gpu.render(
                    0,
                    uniforms,
                    reference.as_ref().map(|d| d.reference.as_slice()),
                );
                let start = std::time::Instant::now();
                let _ = gpu.render(
                    0,
                    uniforms,
                    reference.as_ref().map(|d| d.reference.as_slice()),
                );
                eprintln!(
                    "{family:?} {label} + orbit stats (TIA/DE): {:.1} ms",
                    start.elapsed().as_secs_f64() * 1e3
                );
                gpu.set_colouring(0, &crate::colouring::Colouring::default(), -10.0);
            }
        }
        // CPU reference orbit costs.
        for exponent in [20u32, 100, 300, 1000] {
            let c = DeepComplex::from_f64([-0.745, 0.113], exponent).unwrap();
            for family in [
                FractalFamily::Quadratic,
                FractalFamily::BurningShip,
                FractalFamily::MagnetOne,
            ] {
                let start = std::time::Instant::now();
                let initial = DeepState::initial(family, &c, false, &c).unwrap();
                let orbit =
                    ReferenceOrbit::family(family, &parameters, initial, 2000, 4.0).unwrap();
                eprintln!(
                    "AP reference {family:?} 1e{exponent}, {} points: {:.1} ms",
                    orbit.points.len(),
                    start.elapsed().as_secs_f64() * 1e3
                );
            }
            let view = DeepView {
                centre: c.clone(),
                half_height: DeepReal::parse("1e-20", exponent).unwrap(),
                zoom_exponent: exponent,
                magnification_log10: 20.0,
            };
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let _ = (
                    view.centre.re.exact_decimal(),
                    view.centre.im.exact_decimal(),
                    view.half_height.exact_decimal(),
                );
            }
            eprintln!(
                "exact_decimal x3 at 1e{exponent}: {:.3} ms each",
                start.elapsed().as_secs_f64() * 1e3 / 10.0
            );
        }
    }

    /// The preview path (reduced render + blit) must reproduce the direct
    /// render exactly at scale 1 (orientation, format round trip) and
    /// closely at scale 2.
    #[test]
    #[ignore]
    fn gpu_preview_blit_matches_direct_render() {
        use crate::family::{FamilyParameters, FractalFamily};
        let gpu = GpuHarness::new(384, 256);
        let parameters = FamilyParameters::default();
        let family = FractalFamily::BurningShip;
        let plane = family.default_parameter_view();
        let uniforms = Uniforms::new(
            plane.centre,
            plane.half_height,
            gpu.aspect(),
            family.default_parameter(),
            256,
            4.0,
            family.shader_flag(),
            0,
            true,
            false,
            PrecisionMode::F32,
            parameters.uniform_words(false),
        );
        let direct = gpu.render(0, uniforms, None);
        let preview_full = gpu.render_preview(0, uniforms, 1);
        assert_eq!(
            direct, preview_full,
            "scale-1 preview must be pixel identical"
        );
        let preview_half = gpu.render_preview(0, uniforms, 2);
        // Compare downsampled: average 4x4 blocks of both images.
        let (w, h) = (gpu.width as usize, gpu.height as usize);
        let block = 4;
        let mut total = 0.0;
        let mut count = 0;
        for by in 0..h / block {
            for bx in 0..w / block {
                let mut a = [0.0f64; 3];
                let mut b = [0.0f64; 3];
                for y in 0..block {
                    for x in 0..block {
                        let i = ((by * block + y) * w + bx * block + x) * 3;
                        for k in 0..3 {
                            a[k] += direct[i + k] as f64;
                            b[k] += preview_half[i + k] as f64;
                        }
                    }
                }
                total += (0..3).map(|k| (a[k] - b[k]).abs()).sum::<f64>() / (block * block) as f64;
                count += 1;
            }
        }
        let mean_difference = total / count as f64;
        eprintln!("mean block difference scale 2: {mean_difference:.2}");
        assert!(
            mean_difference < 40.0,
            "half-resolution preview is misaligned: {mean_difference}"
        );
    }

    /// A pan must move the image by exactly the panned amount, whatever
    /// reference point the frame happens to use. Renders a view, pans by a
    /// whole number of pixels, renders again with a differently placed
    /// reference (as the app's candidate search may do), and checks that the
    /// second image is the first one shifted.
    #[test]
    #[ignore]
    fn gpu_pan_moves_image_by_exactly_the_pan() {
        use crate::family::{
            FamilyParameters, FractalFamily, initial_state_with, reference_orbit_f64,
        };
        let gpu = GpuHarness::new(384, 256);
        let parameters = FamilyParameters::default();
        let family = FractalFamily::BurningShip;
        let plane = family.default_parameter_view();
        let half_height = plane.half_height * 1e-9;
        let aspect = gpu.aspect();
        let iterations = 256;
        let render = |centre: [f64; 2], offset: [f32; 2], ds_fallback: bool| -> Vec<u8> {
            let reference_point = [
                centre[0] + offset[0] as f64 * half_height,
                centre[1] + offset[1] as f64 * half_height,
            ];
            let orbit = reference_orbit_f64(
                family,
                &parameters,
                initial_state_with(family, reference_point, false, family.default_parameter()),
                iterations,
                4.0,
            );
            let data =
                DeepRenderData::from_f64_orbit(1, half_height, &orbit.points, ds_fallback, offset);
            let uniforms = Uniforms::new(
                centre,
                half_height,
                aspect,
                family.default_parameter(),
                iterations,
                4.0,
                family.shader_flag(),
                0,
                true,
                false,
                PrecisionMode::DoubleSingle,
                parameters.uniform_words(false),
            )
            .enable_perturbation(
                data.scale_mantissa,
                data.scale_exponent,
                data.reference.len(),
                ds_fallback,
                offset,
            );
            gpu.render(0, uniforms, Some(data.reference.as_slice()))
        };
        // Pixel size in world units (height maps to 2 * half_height).
        let world_per_pixel = 2.0 * half_height / gpu.height as f64;
        let shift_pixels = 7i64;
        let centre_a = plane.centre;
        let centre_b = [
            centre_a[0] + shift_pixels as f64 * world_per_pixel,
            centre_a[1],
        ];
        // The re-described case keeps the very same reference point for both
        // views: the offset changes by exactly the pan (in local units).
        let pan_local = shift_pixels as f32 * 2.0 / gpu.height as f32;
        for (label, offset_a, offset_b, ds_fallback) in [
            ("centred f32", [0.0f32; 2], [0.0f32; 2], true),
            (
                "same reference f32",
                [0.3, 0.2],
                [0.3 - pan_local, 0.2],
                true,
            ),
            (
                "same reference scaled",
                [0.3, 0.2],
                [0.3 - pan_local, 0.2],
                false,
            ),
            (
                "off-centre f32",
                [0.8 * aspect, -0.4],
                [-0.4 * aspect, 0.8],
                true,
            ),
            (
                "off-centre scaled",
                [0.8 * aspect, -0.4],
                [-0.4 * aspect, 0.8],
                false,
            ),
        ] {
            let a = render(centre_a, offset_a, ds_fallback);
            let b = render(centre_b, offset_b, ds_fallback);
            // b is the view moved right by `shift_pixels`: b[x] == a[x + shift].
            let (w, h) = (gpu.width as i64, gpu.height as i64);
            let mut differing = 0usize;
            let mut compared = 0usize;
            for y in 0..h {
                for x in 0..(w - shift_pixels) {
                    let ia = ((y * w + x + shift_pixels) * 3) as usize;
                    let ib = ((y * w + x) * 3) as usize;
                    let delta: i32 = (0..3)
                        .map(|k| (a[ia + k] as i32 - b[ib + k] as i32).abs())
                        .sum();
                    if delta > 60 {
                        differing += 1;
                    }
                    compared += 1;
                }
            }
            let fraction = differing as f64 / compared as f64;
            eprintln!("{label}: {:.2}% of shifted pixels differ", 100.0 * fraction);
            assert!(
                fraction < 0.05,
                "{label}: pan did not translate the image ({:.2}% differ)",
                100.0 * fraction
            );
        }
    }

    /// Pans the quadratic instrument by a whole number of pixels at 1e11 and
    /// compares the shifted images for its double-single path and for an
    /// f64-reference perturbation render of the same views.
    #[test]
    #[ignore]
    fn gpu_quadratic_pan_consistency_at_deep_ds_zoom() {
        use crate::family::{
            FamilyParameters, FractalFamily, initial_state_with, reference_orbit_f64,
        };
        let gpu = GpuHarness::new(384, 256);
        let parameters = FamilyParameters::default();
        let family = FractalFamily::Quadratic;
        let iterations = 512;
        // A boundary point of the Mandelbrot set found by bisection, so the
        // view has structure.
        let plane = family.default_parameter_view();
        let point = boundary_point(
            family,
            &parameters,
            plane.centre,
            plane.half_height,
            false,
            family.default_parameter(),
            iterations,
        )
        .unwrap();
        for zoom in [1e9f64, 1e11] {
            let half_height = 1.45 / zoom;
            let world_per_pixel = 2.0 * half_height / gpu.height as f64;
            let shift_pixels = 5i64;
            let centre_a = point;
            let centre_b = [point[0] + shift_pixels as f64 * world_per_pixel, point[1]];
            let render_ds = |centre: [f64; 2]| {
                let uniforms = Uniforms::new(
                    centre,
                    half_height,
                    gpu.aspect(),
                    family.default_parameter(),
                    iterations,
                    4.0,
                    family.shader_flag(),
                    0,
                    true,
                    false,
                    PrecisionMode::DoubleSingle,
                    parameters.uniform_words(false),
                );
                gpu.render(0, uniforms, None)
            };
            let render_f64 = |centre: [f64; 2]| {
                let orbit = reference_orbit_f64(
                    family,
                    &parameters,
                    initial_state_with(family, centre, false, family.default_parameter()),
                    iterations,
                    4.0,
                );
                let data =
                    DeepRenderData::from_f64_orbit(1, half_height, &orbit.points, true, [0.0; 2]);
                let uniforms = Uniforms::new(
                    centre,
                    half_height,
                    gpu.aspect(),
                    family.default_parameter(),
                    iterations,
                    4.0,
                    family.shader_flag(),
                    0,
                    true,
                    false,
                    PrecisionMode::DoubleSingle,
                    parameters.uniform_words(false),
                )
                .enable_perturbation(
                    data.scale_mantissa,
                    data.scale_exponent,
                    data.reference.len(),
                    true,
                    [0.0; 2],
                );
                gpu.render(0, uniforms, Some(data.reference.as_slice()))
            };
            for (label, a, b) in [
                ("DS", render_ds(centre_a), render_ds(centre_b)),
                ("f64 reference", render_f64(centre_a), render_f64(centre_b)),
            ] {
                let (w, h) = (gpu.width as i64, gpu.height as i64);
                let mut differing = 0usize;
                let mut compared = 0usize;
                for y in 0..h {
                    for x in 0..(w - shift_pixels) {
                        let ia = ((y * w + x + shift_pixels) * 3) as usize;
                        let ib = ((y * w + x) * 3) as usize;
                        let delta: i32 = (0..3)
                            .map(|k| (a[ia + k] as i32 - b[ib + k] as i32).abs())
                            .sum();
                        if delta > 60 {
                            differing += 1;
                        }
                        compared += 1;
                    }
                }
                let distinct: std::collections::HashSet<&[u8]> = a.chunks(3).collect();
                eprintln!(
                    "quadratic {label} at {zoom:e}: {:.2}% of shifted pixels differ ({} distinct colours)",
                    100.0 * differing as f64 / compared as f64,
                    distinct.len()
                );
            }
        }
    }
}
