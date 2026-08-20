use std::sync::Arc;
use std::time::Duration;

use web_time::Instant;

use crate::animation::{self, ZoomAnimation};
#[cfg(not(target_arch = "wasm32"))]
use crate::arbitrary::ReferenceOrbit;
use crate::arbitrary::{
    ARBITRARY_HANDOFF_ZOOM, DeepComplex, DeepReal, DeepState, DeepView, MAX_DECIMAL_ZOOM_EXPONENT,
    ReferenceOrbitBuilder,
};
use crate::colouring::{
    Colouring, ColouringAlgorithm, ColouringSide, Gradient, Interpolation, Layer, LayerStack,
    MAX_LAYERS, MergeMode, Transfer, TrapShape, presets,
};
use crate::experiment::{
    ComplexDocument, ComputationDocument, DeepComplexDocument, DeepPlaneDocument, DisplayDocument,
    ExperimentDocument, FORMAT_ID, FORMAT_VERSION, FamilyParametersDocument, MAX_BAILOUT,
    PlaneDocument,
};
use crate::family::{
    FamilyParameters, FractalFamily, Linkage, MAX_DEGREE, MIN_DEGREE, OrbitFate, PlaneDefault,
    diagnose, initial_state_with, lyapunov_exponent, reference_orbit_f64,
    validate_lyapunov_sequence,
};
use crate::newton::{NewtonResult, ROOTS};
use crate::orbit::{CriticalOrbit, CriticalOrbitCache, OrbitInput};
use crate::precision::{
    DoubleSingle, DsValidity, PathProbeResult, PrecisionMode, ProbeCache, ProbeInput, ProbeResult,
    ValidityLevel,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::render::GpuReferencePoint;
use crate::render::{
    self, ColouringUniforms, DeepRenderData, FractalPipeline, GradientTable, Uniforms,
};

const PANEL_WIDTH: f32 = 286.0;
const HEADER_HEIGHT: f32 = 33.0;
const PANE_GAP: f32 = 6.0;
/// Trackpad pinch is reported every frame as a multiplicative scale. Applying
/// it one-for-one feels abrupt on a high-resolution trackpad, so temper it
/// without introducing lag or accumulating hidden state.
const PINCH_SENSITIVITY: f64 = 0.65;
/// Shift multiplies logarithmic zoom velocity. This preserves direction and
/// reversibility while crossing hundreds of decimal orders in a short gesture.
const ACCELERATED_ZOOM_POWER: f64 = 64.0;
/// Limit one accelerated input sample to twenty decimal orders so a noisy
/// trackpad event cannot jump uncontrollably across the entire deep range.
const MAX_ACCELERATED_DECADES_PER_EVENT: f64 = 20.0;
/// Automatic progressive navigation advances only after the current stage has
/// been submitted for painting. Ten decades per stage reaches extreme targets
/// quickly while preserving a visible sequence of intermediate renders.
const PROGRESSIVE_ZOOM_DECADES_PER_STAGE: f64 = 10.0;
/// Leave the GPU idle between progressive stages instead of continuously
/// queuing full-screen deep renders faster than WebGPU can present them.
const PROGRESSIVE_ZOOM_STAGE_INTERVAL: Duration = Duration::from_millis(750);
/// Orbit disagreement alone enables the more expensive DS path only once a
/// view is meaningfully zoomed. Classification disagreement and coordinate
/// collapse always override this floor.
const PROBE_DS_MIN_ZOOM: f64 = 256.0;
/// Arbitrary-precision reference orbits are extended across frames so a
/// long, high-precision orbit never freezes the interface.
const DEEP_REFERENCE_FRAME_BUDGET: Duration = Duration::from_millis(6);
/// Linear reduction of the render resolution while input is active (a
/// factor of 3 means one ninth of the fragments) and while a reference orbit
/// is still being extended.
const PREVIEW_SCALE_INTERACTING: u32 = 3;
const PREVIEW_SCALE_BUILDING: u32 = 2;

const BG: egui::Color32 = egui::Color32::from_rgb(10, 13, 18);
const PANEL: egui::Color32 = egui::Color32::from_rgb(22, 24, 29);
const PANEL_DEEP: egui::Color32 = egui::Color32::from_rgb(15, 17, 21);
const BORDER: egui::Color32 = egui::Color32::from_rgb(53, 57, 64);
const TEXT: egui::Color32 = egui::Color32::from_rgb(202, 206, 214);
const MUTED: egui::Color32 = egui::Color32::from_rgb(130, 137, 150);
const CREAM: egui::Color32 = egui::Color32::from_rgb(231, 220, 188);
const BLUE: egui::Color32 = egui::Color32::from_rgb(101, 157, 195);
const CORAL: egui::Color32 = egui::Color32::from_rgb(163, 78, 65);

#[derive(Clone, Copy)]
struct PlaneView {
    centre: [f64; 2],
    half_height: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct DeepReferenceKey {
    family: FractalFamily,
    parameters: FamilyParameters,
    centre_re: String,
    centre_im: String,
    half_height: String,
    julia_re: String,
    julia_im: String,
    iterations: u32,
    bailout: u32,
    pane: usize,
}

impl DeepReferenceKey {
    #[allow(clippy::too_many_arguments)]
    fn new(
        family: FractalFamily,
        parameters: &FamilyParameters,
        pane: usize,
        view: &DeepView,
        julia_c: &DeepComplex,
        iterations: u32,
        bailout: f32,
    ) -> Self {
        Self {
            family,
            parameters: parameters.clone(),
            centre_re: view.centre.re.exact_decimal(),
            centre_im: view.centre.im.exact_decimal(),
            half_height: view.half_height.exact_decimal(),
            julia_re: julia_c.re.exact_decimal(),
            julia_im: julia_c.im.exact_decimal(),
            iterations,
            bailout: bailout.to_bits(),
            pane,
        }
    }
}

struct DeepReferenceCache {
    key: DeepReferenceKey,
    /// The view the reference was built for; during navigation the same
    /// orbit is re-described relative to the current view.
    reference_centre: DeepComplex,
    reference_half_height: DeepReal,
    data: Arc<DeepRenderData>,
    /// Still-extending builder while the orbit is incomplete.
    builder: Option<ReferenceOrbitBuilder>,
    ds_fallback: bool,
}

impl DeepReferenceKey {
    /// Same family, parameters, dynamics and pane — only the view differs.
    fn same_dynamics(&self, other: &Self) -> bool {
        self.family == other.family
            && self.parameters == other.parameters
            && self.julia_re == other.julia_re
            && self.julia_im == other.julia_im
            && self.iterations == other.iterations
            && self.bailout == other.bailout
            && self.pane == other.pane
    }
}

/// Cache key for the `f64` reference orbit used below the handoff.
#[derive(Clone, Debug, PartialEq)]
struct F64ReferenceKey {
    family: FractalFamily,
    parameters: FamilyParameters,
    centre: [u64; 2],
    half_height: u64,
    julia: [u64; 2],
    iterations: u32,
    bailout: u32,
    pane: usize,
}

struct F64ReferenceCache {
    key: F64ReferenceKey,
    data: Arc<DeepRenderData>,
}

/// A centred arbitrary-precision reference being built for a view that is
/// currently served by a re-described older reference.
struct PendingDeepReference {
    key: DeepReferenceKey,
    centre: DeepComplex,
    half_height: DeepReal,
    ds_fallback: bool,
    scale_mantissa: f32,
    scale_exponent: i32,
    builder: ReferenceOrbitBuilder,
}

impl PlaneView {
    const fn new(centre: [f64; 2], half_height: f64) -> Self {
        Self {
            centre,
            half_height,
        }
    }

    const fn from_default(default: PlaneDefault) -> Self {
        Self::new(default.centre, default.half_height)
    }

    fn point_at(&self, rect: egui::Rect, position: egui::Pos2) -> [f64; 2] {
        let nx = ((position.x - rect.center().x) / (rect.height() * 0.5)) as f64;
        let ny = ((rect.center().y - position.y) / (rect.height() * 0.5)) as f64;
        [
            self.centre[0] + nx * self.half_height,
            self.centre[1] + ny * self.half_height,
        ]
    }

    fn pan(&mut self, rect: egui::Rect, delta: egui::Vec2) {
        let units_per_point = 2.0 * self.half_height / rect.height().max(1.0) as f64;
        self.centre[0] -= delta.x as f64 * units_per_point;
        self.centre[1] += delta.y as f64 * units_per_point;
    }

    fn pan_tenth(&mut self, rect: egui::Rect, delta: [f64; 2]) {
        let aspect = rect.width() as f64 / rect.height().max(1.0) as f64;
        self.centre[0] += delta[0] * 0.2 * self.half_height * aspect;
        self.centre[1] += delta[1] * 0.2 * self.half_height;
    }

    fn zoom(&mut self, factor: f64) {
        let handoff_half_height = 1.45 / ARBITRARY_HANDOFF_ZOOM;
        self.half_height = (self.half_height * factor).clamp(handoff_half_height, 1e6);
    }

    fn zoom_from(&mut self, focus: Option<[f64; 2]>, factor: f64) {
        if let Some(focus) = focus {
            self.centre = focus;
        }
        self.zoom(factor);
    }

    fn magnification(&self) -> f64 {
        1.45 / self.half_height
    }
}

pub struct App {
    family: FractalFamily,
    parameter: PlaneView,
    dynamical: PlaneView,
    julia_c: [f64; 2],
    progressive_julia_zoom_target_exponent: u32,
    progressive_julia_zoom_active: bool,
    progressive_julia_next_stage_at: Option<Instant>,
    /// Selected starting value of overview/detail instruments.
    selected_z: [f64; 2],
    family_parameters: FamilyParameters,
    lyapunov_sequence_draft: String,
    iterations: u32,
    bailout: f32,
    grid: bool,
    layers: LayerStack,
    /// Single composited image instead of the two linked panes.
    single_image: bool,
    /// The Ultra Fractal-style switch picker: a parameter-plane window with
    /// a live Julia preview for choosing `c` while composing a single image.
    switch_picker_open: bool,
    /// The left (instrument) and right (studio) control panes; each
    /// collapses to a slim strip independently.
    left_panel_open: bool,
    right_panel_open: bool,
    /// Rasterised gradients shared with the render callbacks; rebuilt when
    /// the visible layers' gradients differ from `gradient_table_source`.
    gradient_table: Arc<GradientTable>,
    gradient_table_source: Vec<Gradient>,
    gradient_editor_open: bool,
    gradient_selected_stop: usize,
    gradient_import_text: String,
    gradient_message: Option<(String, bool)>,
    gradient_random_seed: u64,
    render_state: eframe::egui_wgpu::RenderState,
    animation: ZoomAnimation,
    still_width: u32,
    still_height: u32,
    still_supersample: u32,
    export_directory: String,
    #[cfg(not(target_arch = "wasm32"))]
    export: Option<ExportJob>,
    #[cfg(not(target_arch = "wasm32"))]
    still: Option<StillJob>,
    #[cfg(not(target_arch = "wasm32"))]
    export_generation: u64,
    export_message: Option<(String, bool)>,
    zoom_focus: [Option<[f64; 2]>; 2],
    active_pane: usize,
    pending_pan_steps: [f64; 2],
    experiment_editor_open: bool,
    experiment_json: String,
    experiment_message: Option<(String, bool)>,
    orbit_inspector_open: bool,
    show_orbit_overlay: bool,
    orbit_step: usize,
    orbit_cache: CriticalOrbitCache,
    precision_modes: [PrecisionMode; 2],
    deep_active: [bool; 2],
    ds_validity: [DsValidity; 2],
    probes: [ProbeCache; 2],
    deep_views: [Option<DeepView>; 2],
    deep_julia_c: Option<DeepComplex>,
    deep_references: [Option<DeepReferenceCache>; 2],
    deep_pending: [Option<PendingDeepReference>; 2],
    f64_references: [Option<F64ReferenceCache>; 2],
    /// Set during a frame in which a pane's arbitrary-precision reference is
    /// still being extended; the pane then requests another repaint.
    deep_reference_building: [bool; 2],
    /// Whether the pane is currently rendered by perturbation around an
    /// `f64` reference (below the handoff) rather than arbitrary precision.
    f64_reference_active: [bool; 2],
    deep_generation: u64,
    ui_update_ms: f32,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Option<Self> {
        let render_state = cc.wgpu_render_state.as_ref()?;
        render_state
            .renderer
            .write()
            .callback_resources
            .insert(FractalPipeline::new(
                &render_state.device,
                render_state.target_format,
            ));

        configure_style(&cc.egui_ctx);

        Some(Self {
            family: FractalFamily::Quadratic,
            parameter: PlaneView::from_default(FractalFamily::Quadratic.default_parameter_view()),
            dynamical: PlaneView::from_default(FractalFamily::Quadratic.default_dynamical_view()),
            julia_c: FractalFamily::Quadratic.default_parameter(),
            progressive_julia_zoom_target_exponent: 0,
            progressive_julia_zoom_active: false,
            progressive_julia_next_stage_at: None,
            selected_z: FractalFamily::NewtonCubic.default_parameter(),
            family_parameters: FamilyParameters::default(),
            lyapunov_sequence_draft: FamilyParameters::default().lyapunov_sequence,
            iterations: 256,
            bailout: 4.0,
            grid: false,
            layers: LayerStack::default(),
            single_image: true,
            switch_picker_open: false,
            left_panel_open: true,
            right_panel_open: true,
            gradient_table: Arc::new(GradientTable::new(0, &LayerStack::default())),
            gradient_table_source: vec![Gradient::default()],
            gradient_editor_open: false,
            gradient_selected_stop: 0,
            gradient_import_text: String::new(),
            gradient_message: None,
            gradient_random_seed: 1,
            render_state: render_state.clone(),
            animation: ZoomAnimation::default(),
            still_width: 3840,
            still_height: 2160,
            still_supersample: 2,
            export_directory: default_export_directory(),
            #[cfg(not(target_arch = "wasm32"))]
            export: None,
            #[cfg(not(target_arch = "wasm32"))]
            still: None,
            #[cfg(not(target_arch = "wasm32"))]
            export_generation: 0,
            export_message: None,
            zoom_focus: [None, None],
            active_pane: 0,
            pending_pan_steps: [0.0; 2],
            experiment_editor_open: false,
            experiment_json: String::new(),
            experiment_message: None,
            orbit_inspector_open: false,
            show_orbit_overlay: true,
            orbit_step: 0,
            orbit_cache: CriticalOrbitCache::default(),
            precision_modes: [PrecisionMode::F32; 2],
            deep_active: [false; 2],
            ds_validity: [DsValidity::default(); 2],
            probes: [ProbeCache::default(); 2],
            deep_views: [None, None],
            deep_julia_c: None,
            deep_references: [None, None],
            deep_pending: [None, None],
            f64_references: [None, None],
            deep_reference_building: [false; 2],
            f64_reference_active: [false; 2],
            deep_generation: 0,
            ui_update_ms: 0.0,
        })
    }

    /// The left pane: the scientific instrument — family, document,
    /// parameters, computation, navigation and diagnostics.
    fn instrument_controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .id_salt("iterascope.scroll.instrument")
            .show(ui, |ui| {
            section(ui, "Experiment", |ui| {
                let previous_family = self.family;
                egui::ComboBox::from_id_salt("iterascope.family")
                    .selected_text(self.family.name())
                    .width(240.0)
                    .show_ui(ui, |ui| {
                        let mut groups: Vec<&'static str> = Vec::new();
                        for family in FractalFamily::ALL {
                            if !groups.contains(&family.group()) {
                                groups.push(family.group());
                            }
                        }
                        for (index, group) in groups.into_iter().enumerate() {
                            if index > 0 {
                                ui.add_space(2.0);
                            }
                            ui.label(egui::RichText::new(group).small().color(MUTED));
                            for family in FractalFamily::ALL {
                                if family.group() == group {
                                    ui.selectable_value(&mut self.family, family, family.name());
                                }
                            }
                        }
                    });
                if self.family != previous_family {
                    self.reset_for_family();
                    if self.family.converges() != previous_family.converges() {
                        for layer in &mut self.layers.layers {
                            layer.colouring.outside = if self.family.converges() {
                                ColouringSide::default_basins()
                            } else {
                                ColouringSide::default()
                            };
                        }
                    }
                }
                ui.label(
                    egui::RichText::new(self.family.name())
                        .color(CREAM)
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(self.family.formula())
                        .monospace()
                        .color(TEXT),
                );
                ui.add_space(4.0);
                ui.label(egui::RichText::new(self.family.description()).color(MUTED));
            });

            section(ui, "Document", |ui| {
                if ui.button("Export / Import JSON").clicked() {
                    self.refresh_experiment_json();
                    self.experiment_editor_open = true;
                }
                ui.label(
                    egui::RichText::new(
                        "Versioned experiment documents reproduce both views and their scientific settings.",
                    )
                    .small()
                    .color(MUTED),
                );
            });

            if self.family.linkage() == Linkage::ParameterDynamical {
                let symbol = self.family.parameter_symbol();
                section(ui, "Selected parameter", |ui| {
                    let mut parameter_changed = false;
                    parameter_changed |= coordinate_row(ui, &format!("Re({symbol})"), &mut self.julia_c[0]);
                    parameter_changed |= coordinate_row(ui, &format!("Im({symbol})"), &mut self.julia_c[1]);
                    if self.family.supports_deep_zoom() {
                        ui.horizontal(|ui| {
                            ui.label("Progressive Julia target 10^");
                            ui.add(
                                egui::DragValue::new(
                                    &mut self.progressive_julia_zoom_target_exponent,
                                )
                                .range(0..=MAX_DECIMAL_ZOOM_EXPONENT)
                                .speed(1),
                            )
                            .on_hover_text(
                                "Select and centre a feature, then traverse rendered stages toward this exponent",
                            );
                        });
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(!self.progressive_julia_zoom_active, egui::Button::new("Start"))
                                .clicked()
                            {
                                self.progressive_julia_zoom_active = true;
                                self.progressive_julia_next_stage_at = None;
                            }
                            if ui
                                .add_enabled(self.progressive_julia_zoom_active, egui::Button::new("Stop"))
                                .clicked()
                            {
                                self.progressive_julia_zoom_active = false;
                                self.progressive_julia_next_stage_at = None;
                            }
                            let current = self.magnification_log10(1);
                            ui.label(
                                egui::RichText::new(format!("current 10^{current:.2}"))
                                    .small()
                                    .color(if self.progressive_julia_zoom_active {
                                        BLUE
                                    } else {
                                        MUTED
                                    }),
                            );
                        });
                    }
                    let presets = self.family.presets();
                    for row in presets.chunks(2) {
                        ui.horizontal(|ui| {
                            for preset in row {
                                parameter_changed |=
                                    preset_button(ui, preset.label, preset.c, &mut self.julia_c);
                            }
                        });
                    }
                    if parameter_changed {
                        self.deep_julia_c = None;
                        self.reframe_dynamical_plane();
                    }
                });
            } else {
                let symbol = self.family.parameter_symbol();
                section(ui, "Selected initial value", |ui| {
                    coordinate_row(ui, &format!("Re({symbol})"), &mut self.selected_z[0]);
                    coordinate_row(ui, &format!("Im({symbol})"), &mut self.selected_z[1]);
                    if ui.button(format!("Center detail on {symbol}")).clicked() {
                        self.dynamical.centre = self.selected_z;
                        self.zoom_focus[1] = None;
                    }
                    ui.label(
                        egui::RichText::new(format!(
                            "Click the overview to select {symbol} and open a linked detail region."
                        ))
                        .small()
                        .color(MUTED),
                    );
                });
            }

            if self.family.has_family_parameters() {
                section(ui, "Family parameters", |ui| {
                    if self.family.uses_degree() {
                        ui.add(
                            egui::Slider::new(
                                &mut self.family_parameters.degree,
                                MIN_DEGREE..=MAX_DEGREE,
                            )
                            .text(if self.family == FractalFamily::Nova {
                                "degree p"
                            } else {
                                "degree d"
                            }),
                        );
                    }
                    if self.family.uses_relaxation() {
                        ui.add(
                            egui::Slider::new(&mut self.family_parameters.nova_relaxation, 0.1..=4.0)
                                .text("relaxation R"),
                        );
                    }
                    if self.family.uses_lyapunov_sequence() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("sequence").color(MUTED));
                            let response = ui.add(
                                egui::TextEdit::singleline(&mut self.lyapunov_sequence_draft)
                                    .desired_width(150.0)
                                    .font(egui::TextStyle::Monospace),
                            );
                            if response.changed() {
                                let draft = self.lyapunov_sequence_draft.to_ascii_uppercase();
                                if validate_lyapunov_sequence(&draft).is_ok() {
                                    self.family_parameters.lyapunov_sequence = draft;
                                }
                            }
                        });
                        let valid = validate_lyapunov_sequence(&self.lyapunov_sequence_draft);
                        match valid {
                            Ok(()) => ui.label(
                                egui::RichText::new(format!(
                                    "r alternates a/b following {}; first quarter of iterations discarded.",
                                    self.family_parameters.lyapunov_sequence
                                ))
                                .small()
                                .color(MUTED),
                            ),
                            Err(error) => ui.label(egui::RichText::new(error).small().color(CORAL)),
                        };
                        ui.horizontal(|ui| {
                            if ui.small_button("AB").clicked() {
                                self.set_lyapunov_sequence("AB");
                            }
                            if ui.small_button("Zircon Zity").clicked() {
                                self.set_lyapunov_sequence("BBBBBBAAAAAA");
                                self.parameter = PlaneView::new([3.7, 2.95], 0.45);
                                self.dynamical = PlaneView::new([3.7, 2.95], 0.45);
                                self.selected_z = [3.7, 2.95];
                                self.zoom_focus = [None, None];
                            }
                            if ui.small_button("Jellyfish").clicked() {
                                self.set_lyapunov_sequence("AABAB");
                            }
                        });
                    }
                    if self.family.uses_mandelbox() {
                        ui.add(
                            egui::Slider::new(&mut self.family_parameters.mandelbox_scale, -4.0..=4.0)
                                .text("scale s"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.family_parameters.mandelbox_min_radius,
                                0.01..=2.0,
                            )
                            .text("min radius"),
                        );
                        ui.add(
                            egui::Slider::new(
                                &mut self.family_parameters.mandelbox_fixed_radius,
                                0.1..=4.0,
                            )
                            .text("fixed radius"),
                        );
                    }
                });
            }

            section(ui, "Computation", |ui| {
                let iteration_range = self.family.min_iterations()..=self.family.max_iterations();
                ui.add(
                    egui::Slider::new(&mut self.iterations, iteration_range)
                        .logarithmic(true)
                        .text("iterations"),
                );
                if self.family.uses_bailout() {
                    ui.add(
                        egui::Slider::new(&mut self.bailout, 2.0..=MAX_BAILOUT)
                            .logarithmic(true)
                            .text("bailout"),
                    );
                }
                if self.family.is_escape_time() {
                    let (left, right) = self.pane_labels();
                    precision_row(
                        ui,
                        left,
                        self.precision_modes[0],
                        self.ds_validity[0],
                        self.deep_active[0],
                        self.f64_reference_active[0],
                    );
                    precision_row(
                        ui,
                        right,
                        self.precision_modes[1],
                        self.ds_validity[1],
                        self.deep_active[1],
                        self.f64_reference_active[1],
                    );
                    let note = if self.family.is_quadratic() {
                        "Precision switches automatically after instability."
                    } else if self.family.supports_deep_zoom() {
                        "Past the f32 pixel grid the GPU perturbs an f64 reference orbit of the view centre; beyond 1.14e14× the reference is arbitrary-precision. No orbit probe exists for this family."
                    } else {
                        "This family renders in f32 only; structure below the f32 pixel grid is not resolved."
                    };
                    ui.label(egui::RichText::new(note).small().color(MUTED));
                } else {
                    ui.label(
                        egui::RichText::new(
                            "Newton rendering is capped at 2,048 iterations per pixel.",
                        )
                        .small()
                        .color(MUTED),
                    );
                }
            });

            if self.family.is_quadratic() {
                let orbit_input = OrbitInput {
                c: self.julia_c,
                iterations: self.iterations,
                bailout: self.bailout as f64,
            };
            let orbit = self.orbit_cache.update(orbit_input);
            let orbit_status = match orbit.escape_iteration {
                Some(iteration) => format!("Critical orbit escapes at n = {iteration}"),
                None => format!("No escape through n = {}", orbit.last_iteration()),
            };
            let orbit_last = orbit.last_iteration();
            let orbit_colour = if orbit.escape_iteration.is_some() {
                CREAM
            } else {
                BLUE
            };
            let mut orbit_step = self.orbit_step.min(orbit_last);
            let mut show_orbit_overlay = self.show_orbit_overlay;
            let mut open_orbit_inspector = false;
            let mut centre_on_orbit_step = false;
            section(ui, "Critical orbit", |ui| {
                ui.label(egui::RichText::new(orbit_status).color(orbit_colour));
                ui.checkbox(&mut show_orbit_overlay, "Show in Julia plane");
                ui.horizontal(|ui| {
                    if ui.button("<").on_hover_text("Previous orbit point").clicked() {
                        orbit_step = orbit_step.saturating_sub(1);
                    }
                    if ui.button(">").on_hover_text("Next orbit point").clicked() {
                        orbit_step = (orbit_step + 1).min(orbit_last);
                    }
                    ui.monospace(format!("z_{orbit_step}"));
                });
                ui.add(egui::Slider::new(&mut orbit_step, 0..=orbit_last).text("iteration n"));
                if ui.button("Center Julia on selected z_n").clicked() {
                    centre_on_orbit_step = true;
                }
                if ui.button("Inspect orbit").clicked() {
                    open_orbit_inspector = true;
                }
                ui.label(
                    egui::RichText::new("f64 diagnostic for z_0 = 0 and z_(n+1) = z_n^2 + c.")
                        .small()
                        .color(MUTED),
                );
            });
            self.orbit_step = orbit_step;
            self.show_orbit_overlay = show_orbit_overlay;
            self.orbit_inspector_open |= open_orbit_inspector;
            if centre_on_orbit_step {
                let orbit = self.orbit_cache.update(orbit_input);
                self.dynamical.centre = orbit.points[orbit_step].z;
                self.deep_views[1] = None;
                self.zoom_focus[1] = None;
            }
            } else if self.family.is_newton() {
                let result = NewtonResult::calculate(self.selected_z, self.iterations);
                section(ui, "Newton diagnostic", |ui| {
                    let status = match result.root {
                        Some(root) => format!("Converges to root {}", root + 1),
                        None if result.singular => "Derivative singularity".to_owned(),
                        None => format!("No convergence through n = {}", result.iterations),
                    };
                    ui.label(
                        egui::RichText::new(status).color(if result.root.is_some() {
                            CREAM
                        } else {
                            CORAL
                        }),
                    );
                    diagnostic_row(ui, "iterations", &result.iterations.to_string());
                    if let Some(root) = result.root {
                        diagnostic_row(
                            ui,
                            "root",
                            &format!("{:+.9e} {:+.9e}i", ROOTS[root][0], ROOTS[root][1]),
                        );
                    }
                    diagnostic_row(ui, "residual |p(z)|", &format!("{:.9e}", result.residual));
                    diagnostic_row(ui, "last step |Δz|", &format!("{:.9e}", result.last_step));
                    diagnostic_row(ui, "Re(final z)", &format!("{:+.12e}", result.value[0]));
                    diagnostic_row(ui, "Im(final z)", &format!("{:+.12e}", result.value[1]));
                    ui.label(
                        egui::RichText::new(
                            "CPU f64 diagnostic for the selected starting value z₀.",
                        )
                        .small()
                        .color(MUTED),
                    );
                });
            } else if self.family == FractalFamily::Lyapunov {
                let exponent = lyapunov_exponent(
                    &self.family_parameters,
                    self.selected_z[0],
                    self.selected_z[1],
                    self.iterations,
                );
                section(ui, "Lyapunov diagnostic", |ui| {
                    let (status, colour) = if exponent.is_infinite() {
                        ("Orbit leaves [0, 1]: parameters outside the logistic range".to_owned(), CORAL)
                    } else if exponent < 0.0 {
                        (format!("Stable: λ = {exponent:+.6}"), CREAM)
                    } else {
                        (format!("Chaotic: λ = {exponent:+.6}"), BLUE)
                    };
                    ui.label(egui::RichText::new(status).color(colour));
                    diagnostic_row(ui, "a", &format!("{:+.9}", self.selected_z[0]));
                    diagnostic_row(ui, "b", &format!("{:+.9}", self.selected_z[1]));
                    diagnostic_row(ui, "sequence", &self.family_parameters.lyapunov_sequence);
                    diagnostic_row(ui, "iterations", &self.iterations.to_string());
                    ui.label(
                        egui::RichText::new(
                            "CPU f64 exponent of x -> r x (1 - x) from x₀ = ½, transient discarded.",
                        )
                        .small()
                        .color(MUTED),
                    );
                });
            } else {
                let dynamical = self.family.linkage() == Linkage::OverviewDetail;
                let world = if dynamical { self.selected_z } else { self.julia_c };
                let state = initial_state_with(self.family, world, dynamical, self.julia_c);
                let result = diagnose(
                    self.family,
                    &self.family_parameters,
                    state,
                    self.iterations,
                    self.bailout as f64,
                );
                let title = if dynamical {
                    "Selected orbit"
                } else {
                    "Critical orbit"
                };
                section(ui, title, |ui| {
                    let (status, colour) = match result.fate {
                        OrbitFate::Escaped => {
                            (format!("Escapes at n = {}", result.iterations), CREAM)
                        }
                        OrbitFate::Converged => {
                            (format!("Converges at n = {}", result.iterations), CREAM)
                        }
                        OrbitFate::NonFinite => {
                            (format!("Non-finite at n = {}", result.iterations), CORAL)
                        }
                        OrbitFate::Bounded => {
                            (format!("Bounded through n = {}", result.iterations), BLUE)
                        }
                    };
                    ui.label(egui::RichText::new(status).color(colour));
                    diagnostic_row(ui, "Re(z₀)", &format!("{:+.9e}", state.z[0]));
                    diagnostic_row(ui, "Im(z₀)", &format!("{:+.9e}", state.z[1]));
                    diagnostic_row(ui, "|final z|", &format!("{:.9e}", result.magnitude));
                    diagnostic_row(ui, "Re(final z)", &format!("{:+.12e}", result.z[0]));
                    diagnostic_row(ui, "Im(final z)", &format!("{:+.12e}", result.z[1]));
                    ui.label(
                        egui::RichText::new(if dynamical {
                            "CPU f64 orbit of the selected starting value under the displayed map."
                        } else {
                            "CPU f64 critical orbit for the selected parameter; the same recurrence the shader iterates."
                        })
                        .small()
                        .color(MUTED),
                    );
                });
            }

            section(ui, "View status", |ui| {
                let (left, right) = self.pane_labels();
                zoom_row(
                    ui,
                    left,
                    self.deep_views[0].as_ref().map_or_else(
                        || format!("{:.6e}", self.parameter.magnification()),
                        DeepView::magnification_label,
                    ),
                );
                zoom_row(
                    ui,
                    right,
                    self.deep_views[1].as_ref().map_or_else(
                        || format!("{:.6e}", self.dynamical.magnification()),
                        DeepView::magnification_label,
                    ),
                );
                if self.family.is_quadratic() {
                    probe_rows(
                        ui,
                        "P",
                        self.probes[0].last_result(),
                        self.ds_validity[0],
                    );
                    probe_rows(
                        ui,
                        "J",
                        self.probes[1].last_result(),
                        self.ds_validity[1],
                    );
                    ui.label(
                        egui::RichText::new(
                            "Nine CPU samples compare both GPU arithmetic paths with f64 after a settled view change.",
                        )
                        .small()
                        .color(MUTED),
                    );
                } else if self.family.is_newton() {
                    ui.label(
                        egui::RichText::new(
                            "Overview classifies roots; detail emphasizes convergence time.",
                        )
                        .small()
                        .color(MUTED),
                    );
                } else {
                    ui.label(
                        egui::RichText::new(
                            "No orbit probe exists for this family; precision labels reflect coordinate resolution only.",
                        )
                        .small()
                        .color(MUTED),
                    );
                }
            });

            section(ui, "Navigation", |ui| {
                let parameter_dynamical = self.family.linkage() == Linkage::ParameterDynamical;
                let (left, right) = self.pane_labels();
                ui.label(
                    egui::RichText::new(if parameter_dynamical {
                        "Click: centre + ×2 zoom · right-click: centre only"
                    } else {
                        "Overview click: select the point + open detail · right-click: centre only"
                    })
                    .color(TEXT),
                );
                ui.label(
                    egui::RichText::new(if parameter_dynamical {
                        "Wheel or pinch to zoom. Drag to pan. Left clicks also choose the parameter."
                    } else {
                        "Wheel or pinch to zoom. Drag to pan. Click detail to centre + ×2."
                    })
                    .small()
                    .color(MUTED),
                );
                if self.family.supports_deep_zoom() {
                    ui.label(
                        egui::RichText::new(
                            "Hold Shift while zooming for accelerated deep navigation.",
                        )
                        .small()
                        .color(BLUE),
                    );
                }
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Fine pan target").color(TEXT));
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.active_pane, 0, left);
                    ui.selectable_value(&mut self.active_pane, 1, right);
                });
                ui.horizontal(|ui| {
                    if ui.button("<").on_hover_text("Pan view left by 1/10").clicked() {
                        self.pending_pan_steps[0] -= 1.0;
                    }
                    if ui.button("^").on_hover_text("Pan view up by 1/10").clicked() {
                        self.pending_pan_steps[1] += 1.0;
                    }
                    if ui.button("v").on_hover_text("Pan view down by 1/10").clicked() {
                        self.pending_pan_steps[1] -= 1.0;
                    }
                    if ui.button(">").on_hover_text("Pan view right by 1/10").clicked() {
                        self.pending_pan_steps[0] += 1.0;
                    }
                });
                ui.label(
                    egui::RichText::new(
                        "Buttons and arrow keys pan by 1/10 of the displayed range (Shift + arrows: 1/100). Hover a pane or select its target above.",
                    )
                    .small()
                    .color(MUTED),
                );
                ui.horizontal(|ui| {
                    if ui.button(format!("Reset {}", left.to_lowercase())).clicked() {
                        self.parameter = PlaneView::from_default(self.family.default_parameter_view());
                        self.deep_views[0] = None;
                        self.zoom_focus[0] = None;
                    }
                    if ui.button(format!("Reset {}", right.to_lowercase())).clicked() {
                        if parameter_dynamical {
                            self.reframe_dynamical_plane();
                        } else {
                            self.dynamical = PlaneView::from_default(self.family.default_dynamical_view());
                            self.zoom_focus[1] = None;
                        }
                    }
                });
            });
        });
    }

    /// Short names of the two panes for the current instrument.
    /// The right pane: the studio — layers, colouring and export.
    fn studio_controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical()
            .id_salt("iterascope.scroll.studio")
            .show(ui, |ui| {
                section(ui, "Layers", |ui| self.layer_list(ui));

                section(ui, "Colouring", |ui| self.colouring_controls(ui));

                #[cfg(not(target_arch = "wasm32"))]
                section(ui, "Still image", |ui| self.still_controls(ui));

                section(ui, "Animation", |ui| self.animation_controls(ui));
            });
    }

    fn pane_labels(&self) -> (&'static str, &'static str) {
        match (self.family.linkage(), self.family.is_quadratic()) {
            (Linkage::ParameterDynamical, true) => ("Parameter", "Julia"),
            (Linkage::ParameterDynamical, false) => ("Parameter", "Dynamical"),
            (Linkage::OverviewDetail, _) => ("Overview", "Detail"),
        }
    }

    /// Whether a pane iterates from z₀ = pixel (dynamical or detail plane)
    /// rather than treating the pixel as the parameter.
    fn pane_is_dynamical(&self, pane: usize) -> bool {
        pane == 1 || self.family.linkage() == Linkage::OverviewDetail
    }

    fn set_lyapunov_sequence(&mut self, sequence: &str) {
        self.family_parameters.lyapunov_sequence = sequence.to_owned();
        self.lyapunov_sequence_draft = sequence.to_owned();
    }

    fn experiment_document(&self) -> ExperimentDocument {
        let overview_detail = self.family.linkage() == Linkage::OverviewDetail;
        ExperimentDocument {
            format: FORMAT_ID.to_owned(),
            version: FORMAT_VERSION,
            family: self.family.document_id().to_owned(),
            parameter_plane: PlaneDocument {
                centre: ComplexDocument {
                    re: self.parameter.centre[0],
                    im: self.parameter.centre[1],
                },
                half_height: self.parameter.half_height,
            },
            dynamical_plane: PlaneDocument {
                centre: ComplexDocument {
                    re: self.dynamical.centre[0],
                    im: self.dynamical.centre[1],
                },
                half_height: self.dynamical.half_height,
            },
            parameter_c: ComplexDocument {
                re: self.julia_c[0],
                im: self.julia_c[1],
            },
            newton_initial_z: self.family.is_newton().then_some(ComplexDocument {
                re: self.selected_z[0],
                im: self.selected_z[1],
            }),
            initial_z: (overview_detail && !self.family.is_newton()).then_some(ComplexDocument {
                re: self.selected_z[0],
                im: self.selected_z[1],
            }),
            family_parameters: FamilyParametersDocument::for_family(
                self.family,
                &self.family_parameters,
            ),
            computation: ComputationDocument {
                iterations: self.iterations,
                bailout: self.bailout,
            },
            display: DisplayDocument {
                smooth_escape_time: self.layers.layers[0].colouring.outside.smooth,
                coordinate_grid: self.grid,
                palette_phase: 0.0,
                critical_orbit_overlay: self.show_orbit_overlay,
                interior_shading: true,
            },
            colouring: None,
            layers: Some(self.layers.layers.clone()),
            progressive_julia_zoom_target_exponent: if self.family.supports_deep_zoom() {
                self.progressive_julia_zoom_target_exponent
            } else {
                0
            },
            deep_parameter_plane: if self.family.supports_deep_zoom() {
                self.deep_views[0].as_ref().map(deep_plane_document)
            } else {
                None
            },
            deep_dynamical_plane: if self.family.supports_deep_zoom() {
                self.deep_views[1].as_ref().map(deep_plane_document)
            } else {
                None
            },
            deep_parameter_c: if self.family.supports_deep_zoom() {
                self.deep_julia_c.as_ref().map(|value| DeepComplexDocument {
                    re: value.re.exact_decimal(),
                    im: value.im.exact_decimal(),
                })
            } else {
                None
            },
        }
    }

    fn refresh_experiment_json(&mut self) {
        match self.experiment_document().to_pretty_json() {
            Ok(json) => {
                self.experiment_json = json;
                self.experiment_message = None;
            }
            Err(error) => self.experiment_message = Some((error, true)),
        }
    }

    fn apply_experiment(&mut self, document: ExperimentDocument) {
        self.family = FractalFamily::from_document_id(&document.family).unwrap_or_default();
        self.parameter = PlaneView::new(
            [
                document.parameter_plane.centre.re,
                document.parameter_plane.centre.im,
            ],
            document.parameter_plane.half_height,
        );
        self.dynamical = PlaneView::new(
            [
                document.dynamical_plane.centre.re,
                document.dynamical_plane.centre.im,
            ],
            document.dynamical_plane.half_height,
        );
        self.julia_c = [document.parameter_c.re, document.parameter_c.im];
        if let Some(value) = document.newton_initial_z.or(document.initial_z) {
            self.selected_z = [value.re, value.im];
        } else if self.family.linkage() == Linkage::OverviewDetail {
            self.selected_z = self.family.default_parameter();
        }
        self.family_parameters = FamilyParameters::default();
        if let Some(parameters) = &document.family_parameters {
            parameters.apply_to(&mut self.family_parameters);
        }
        self.lyapunov_sequence_draft = self.family_parameters.lyapunov_sequence.clone();
        self.progressive_julia_zoom_target_exponent =
            document.progressive_julia_zoom_target_exponent;
        self.progressive_julia_zoom_active = false;
        self.progressive_julia_next_stage_at = None;
        self.iterations = document.computation.iterations;
        self.bailout = document.computation.bailout;
        self.grid = document.display.coordinate_grid;
        self.show_orbit_overlay = document.display.critical_orbit_overlay;
        self.layers = if let Some(layers) = document.layers {
            LayerStack::from_layers(layers)
        } else {
            LayerStack::single(document.colouring.unwrap_or_else(|| {
                // Documents from before format version 5: the palette phase
                // was the gradient offset and the only colouring choice was
                // the smoothing of the outside iteration count.
                let mut colouring = Colouring::default();
                if self.family.converges() {
                    colouring.outside = ColouringSide::default_basins();
                }
                colouring.outside.smooth = document.display.smooth_escape_time;
                colouring.outside.offset = document.display.palette_phase;
                colouring
            }))
        };
        self.gradient_selected_stop = 0;

        self.zoom_focus = [None, None];
        self.pending_pan_steps = [0.0; 2];
        self.orbit_step = 0;
        self.precision_modes = [PrecisionMode::F32; 2];
        self.ds_validity = [DsValidity::default(); 2];
        self.probes = [ProbeCache::default(); 2];
        self.deep_views = [None, None];
        self.deep_julia_c = None;
        self.deep_active = [false; 2];
        if let Some(plane) = document.deep_parameter_plane {
            self.deep_views[0] = DeepView::parse(
                &plane.centre.re,
                &plane.centre.im,
                &plane.half_height,
                plane.magnification_log10,
            )
            .ok();
        }
        if let Some(plane) = document.deep_dynamical_plane {
            self.deep_views[1] = DeepView::parse(
                &plane.centre.re,
                &plane.centre.im,
                &plane.half_height,
                plane.magnification_log10,
            )
            .ok();
        }
        if let Some(value) = document.deep_parameter_c {
            let exponent = self.deep_views[0]
                .as_ref()
                .map_or(15, |view| view.zoom_exponent);
            self.deep_julia_c = DeepComplex::parse(&value.re, &value.im, exponent).ok();
        }
    }

    fn experiment_editor(&mut self, ctx: &egui::Context) {
        if !self.experiment_editor_open {
            return;
        }

        let mut open = self.experiment_editor_open;
        let mut close_requested = false;
        egui::Window::new("Experiment document")
            .open(&mut open)
            .default_width(680.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(format!(
                        "IteraScope JSON · format version {FORMAT_VERSION}"
                    ))
                        .monospace()
                        .color(CREAM),
                );
                ui.label(
                    egui::RichText::new(
                        "Copy this document to export it. To import, replace the text with another IteraScope document and choose Load JSON.",
                    )
                    .small()
                    .color(MUTED),
                );
                ui.add_space(6.0);
                ui.add(
                    egui::TextEdit::multiline(&mut self.experiment_json)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(22),
                );
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button("Refresh current").clicked() {
                        self.refresh_experiment_json();
                    }
                    if ui.button("Copy JSON").clicked() {
                        ui.ctx().copy_text(self.experiment_json.clone());
                        self.experiment_message = Some(("JSON copied to clipboard".to_owned(), false));
                    }
                    if ui.button("Load JSON").clicked() {
                        match ExperimentDocument::from_json(&self.experiment_json) {
                            Ok(document) => {
                                self.apply_experiment(document);
                                self.experiment_message =
                                    Some(("Experiment loaded".to_owned(), false));
                            }
                            Err(error) => {
                                self.experiment_message = Some((format!("Import failed: {error}"), true));
                            }
                        }
                    }
                    if ui.button("Close").clicked() {
                        close_requested = true;
                    }
                });
                if let Some((message, error)) = &self.experiment_message {
                    let colour = if *error { CORAL } else { BLUE };
                    ui.label(egui::RichText::new(message).color(colour));
                }
            });
        self.experiment_editor_open = open && !close_requested;
    }

    fn orbit_inspector(&mut self, ctx: &egui::Context) {
        if !self.orbit_inspector_open {
            return;
        }

        let input = OrbitInput {
            c: self.julia_c,
            iterations: self.iterations,
            bailout: self.bailout as f64,
        };
        let orbit = self.orbit_cache.update(input);
        let centre_request = show_orbit_inspector(
            ctx,
            orbit,
            input.c,
            &mut self.orbit_step,
            &mut self.orbit_inspector_open,
        );
        if let Some(centre) = centre_request {
            self.dynamical.centre = centre;
            self.zoom_focus[1] = None;
        }
    }

    fn workspace(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap().shrink(6.0);
        if self.single_image {
            let pane = self.active_pane.min(1);
            self.pane(ui, rect, pane);
            return;
        }
        if rect.width() > 780.0 {
            let width = (rect.width() - PANE_GAP) * 0.5;
            let left = egui::Rect::from_min_size(rect.min, egui::vec2(width, rect.height()));
            let right = egui::Rect::from_min_size(
                egui::pos2(left.max.x + PANE_GAP, rect.min.y),
                egui::vec2(width, rect.height()),
            );
            self.pane(ui, left, 0);
            self.pane(ui, right, 1);
        } else {
            let height = (rect.height() - PANE_GAP) * 0.5;
            let top = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), height));
            let bottom = egui::Rect::from_min_size(
                egui::pos2(rect.min.x, top.max.y + PANE_GAP),
                egui::vec2(rect.width(), height),
            );
            self.pane(ui, top, 0);
            self.pane(ui, bottom, 1);
        }
    }

    fn pane(&mut self, ui: &mut egui::Ui, outer: egui::Rect, pane: usize) {
        ui.painter().rect_filled(outer, 3.0, PANEL_DEEP);
        ui.painter().rect_stroke(
            outer,
            3.0,
            egui::Stroke::new(1.0, BORDER),
            egui::StrokeKind::Inside,
        );

        let header = egui::Rect::from_min_max(
            outer.min,
            egui::pos2(outer.max.x, outer.min.y + HEADER_HEIGHT),
        );
        let viewport = egui::Rect::from_min_max(
            egui::pos2(outer.min.x + 1.0, header.max.y),
            egui::pos2(outer.max.x - 1.0, outer.max.y - 1.0),
        );

        let (title, subtitle) = self.family.pane_titles(pane);
        ui.painter().text(
            egui::pos2(header.min.x + 11.0, header.center().y),
            egui::Align2::LEFT_CENTER,
            title,
            egui::FontId::new(11.0, egui::FontFamily::Monospace),
            TEXT,
        );
        if outer.width() > 520.0 {
            ui.painter().text(
                egui::pos2(header.center().x, header.center().y),
                egui::Align2::CENTER_CENTER,
                subtitle,
                egui::FontId::new(14.0, egui::FontFamily::Proportional),
                egui::Color32::WHITE,
            );
        }
        let response = ui.allocate_rect(viewport, egui::Sense::click_and_drag());
        if response.hovered() {
            self.active_pane = pane;
        }
        let pinch_delta = ui.input(|input| input.zoom_delta());
        let pinching = (pinch_delta - 1.0).abs() > 0.001;
        let mut interacting = pinching || response.dragged();

        let mut fine_pan = [0.0; 2];
        if pane == self.active_pane {
            fine_pan = self.pending_pan_steps;
            self.pending_pan_steps = [0.0; 2];
            if !ui.ctx().egui_wants_keyboard_input() {
                // Arrow keys pan by 1/10 of the view; with Shift held by 1/100.
                let (keyboard_pan, fine) = ui.input(|input| {
                    (
                        [
                            (input.key_pressed(egui::Key::ArrowRight) as i8
                                - input.key_pressed(egui::Key::ArrowLeft) as i8)
                                as f64,
                            (input.key_pressed(egui::Key::ArrowUp) as i8
                                - input.key_pressed(egui::Key::ArrowDown) as i8)
                                as f64,
                        ],
                        input.modifiers.shift,
                    )
                });
                let step = if fine { 0.1 } else { 1.0 };
                fine_pan[0] += keyboard_pan[0] * step;
                fine_pan[1] += keyboard_pan[1] * step;
            }
        }
        if fine_pan != [0.0; 2] {
            interacting = true;
            if pane == 1 {
                self.progressive_julia_zoom_active = false;
                self.progressive_julia_next_stage_at = None;
            }
            if let Some(deep) = &mut self.deep_views[pane] {
                let aspect = viewport.width() as f64 / viewport.height().max(1.0) as f64;
                let _ = deep.pan_local([fine_pan[0] * 0.2 * aspect, fine_pan[1] * 0.2]);
            } else if pane == 0 {
                self.parameter.pan_tenth(viewport, fine_pan);
            } else {
                self.dynamical.pan_tenth(viewport, fine_pan);
            }
            self.zoom_focus[pane] = None;
        }

        if response.dragged() && !pinching {
            if pane == 1 {
                self.progressive_julia_zoom_active = false;
                self.progressive_julia_next_stage_at = None;
            }
            let delta = ui.input(|input| input.pointer.delta());
            if let Some(deep) = &mut self.deep_views[pane] {
                let height = viewport.height().max(1.0) as f64;
                let _ = deep.pan_local([
                    -2.0 * delta.x as f64 / height,
                    2.0 * delta.y as f64 / height,
                ]);
            } else if pane == 0 {
                self.parameter.pan(viewport, delta);
            } else {
                self.dynamical.pan(viewport, delta);
            }
            self.zoom_focus[pane] = None;
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            let factor = if pinching {
                Some(pinch_zoom_factor(pinch_delta))
            } else if scroll.abs() > 0.0 {
                Some((-scroll as f64 * 0.0025).exp())
            } else {
                None
            };
            if let Some(factor) = factor {
                interacting = true;
                if pane == 1 {
                    self.progressive_julia_zoom_active = false;
                    self.progressive_julia_next_stage_at = None;
                }
                let accelerated = ui.input(|input| input.modifiers.shift);
                self.zoom_pane(pane, accelerate_zoom_factor(factor, accelerated));
            }
        }
        let primary_click = response.clicked() && !response.dragged();
        let secondary_click = response.secondary_clicked() && !response.dragged();
        if primary_click || secondary_click {
            if let Some(position) = response.interact_pointer_pos() {
                interacting = true;
                if pane == 1 {
                    self.progressive_julia_zoom_active = false;
                    self.progressive_julia_next_stage_at = None;
                }
                let accelerated = ui.input(|input| input.modifiers.shift);
                // A secondary (right) click recentres without zooming, so a
                // region can be framed precisely before magnifying it.
                let click_zoom = if secondary_click {
                    1.0
                } else {
                    accelerate_zoom_factor(0.5, accelerated)
                };
                let overview_detail = self.family.linkage() == Linkage::OverviewDetail;
                if let Some(deep) = &mut self.deep_views[pane] {
                    // Deep navigation: recentre and zoom in arbitrary precision.
                    let local = local_coordinate(viewport, position);
                    let _ = deep.recenter_local(local);
                    if overview_detail {
                        if pane == 0 && secondary_click {
                            // Recentre the overview only.
                        } else if pane == 0 {
                            self.selected_z = deep.centre_preview();
                            // Open a linked detail region around the selected
                            // point at the same depth plus one more decade.
                            let mut detail = deep.clone();
                            let _ = detail.zoom(0.22);
                            self.deep_views[1] = Some(detail);
                            self.zoom_focus[1] = None;
                        } else {
                            self.selected_z = deep.centre_preview();
                            let _ = deep.zoom(click_zoom);
                        }
                    } else {
                        let _ = deep.zoom(click_zoom);
                        if pane == 0 {
                            self.deep_julia_c = Some(deep.centre.clone());
                            self.julia_c = deep.centre_preview();
                            self.reframe_dynamical_plane();
                        }
                    }
                    self.zoom_focus[pane] = None;
                } else if overview_detail {
                    let point = if pane == 0 {
                        self.parameter.point_at(viewport, position)
                    } else {
                        self.dynamical.point_at(viewport, position)
                    };
                    if pane == 0 && secondary_click {
                        // Recentre the overview only.
                        self.parameter.centre = point;
                        self.zoom_focus[0] = None;
                    } else if pane == 0 {
                        self.selected_z = point;
                        self.dynamical.centre = point;
                        self.dynamical.half_height = (self.parameter.half_height * 0.22)
                            .clamp(1.45 / ARBITRARY_HANDOFF_ZOOM, 1e6);
                        self.deep_views[1] = None;
                        self.zoom_focus[0] = Some(point);
                        self.zoom_focus[1] = None;
                    } else {
                        self.selected_z = point;
                        self.dynamical.centre = point;
                        self.zoom_focus[1] = Some(point);
                        self.zoom_pane(1, click_zoom);
                    }
                } else {
                    let point = if pane == 0 {
                        self.parameter.point_at(viewport, position)
                    } else {
                        self.dynamical.point_at(viewport, position)
                    };
                    self.zoom_focus[pane] = Some(point);
                    if pane == 0 {
                        self.julia_c = point;
                        self.deep_julia_c = None;
                        self.parameter.centre = point;
                        self.zoom_pane(0, click_zoom);
                        if let Some(deep) = &self.deep_views[0] {
                            self.deep_julia_c = Some(deep.centre.clone());
                        }
                        self.reframe_dynamical_plane();
                    } else {
                        self.dynamical.centre = point;
                        self.zoom_pane(1, click_zoom);
                    }
                }
            }
        }
        if interacting {
            // While input is active a deep view is rendered around its
            // existing reference orbit, re-described for the moved view.
            // Schedule exactly one settled frame so the centred replacement
            // reference can start building after input stops.
            ui.ctx().request_repaint();
        }

        let view = if pane == 0 {
            self.parameter
        } else {
            self.dynamical
        };
        let aspect = viewport.width() / viewport.height().max(1.0);
        let deep_view = self.deep_views[pane].clone();
        let magnification = deep_view.as_ref().map_or_else(
            || view.magnification(),
            |deep| 10.0_f64.powf(deep.magnification_log10),
        );
        let probe_input = ProbeInput {
            centre: view.centre,
            half_height: view.half_height,
            aspect: aspect as f64,
            julia_c: self.julia_c,
            iterations: self.iterations,
            bailout: self.bailout as f64,
            pane,
        };
        let probe = if !self.family.is_quadratic() || deep_view.is_some() {
            ProbeResult::default()
        } else if interacting {
            self.probes[pane].current(probe_input).unwrap_or_default()
        } else {
            self.probes[pane].update(probe_input)
        };
        let dynamics_parameter = (self.family.linkage() == Linkage::ParameterDynamical
            && pane == 1)
            .then_some(self.julia_c);
        let coordinate_limited = view_is_f32_limited(
            &view,
            dynamics_parameter,
            viewport,
            ui.ctx().pixels_per_point(),
        ) || (self.family.is_quadratic()
            && pane == 1
            && julia_critical_roundoff_risk(
                &view,
                self.julia_c,
                viewport,
                ui.ctx().pixels_per_point(),
            ));
        let precision = if self.family.supports_double_single() {
            choose_precision(magnification, coordinate_limited, probe)
        } else {
            PrecisionMode::F32
        };
        self.precision_modes[pane] = precision;
        let ds_validity = DsValidity::from_probe(
            probe.ds,
            ds_coordinate_ratio(
                &view,
                dynamics_parameter,
                viewport,
                ui.ctx().pixels_per_point(),
            ),
        );
        self.ds_validity[pane] = ds_validity;
        let f32_only_limited = !self.family.supports_double_single() && coordinate_limited;

        let deep = if !self.family.supports_deep_zoom() {
            None
        } else if deep_view.is_some() {
            self.deep_reference(pane, deep_view.as_ref(), !interacting)
        } else if precision == PrecisionMode::DoubleSingle {
            // Below the handoff every family renders by perturbation around
            // an f64 reference orbit: exact reference fates, no GPU
            // compensated arithmetic on the critical path.
            self.f64_reference(pane, &view, aspect)
        } else {
            None
        };
        self.f64_reference_active[pane] = deep.is_some() && deep_view.is_none();
        self.deep_active[pane] = deep.is_some();
        let building = self.deep_reference_building[pane];
        if building {
            self.deep_reference_building[pane] = false;
            ui.ctx().request_repaint();
        }
        // Render at reduced resolution while input is active (and, more
        // mildly, while a reference orbit is still being extended) so each
        // frame stays cheap and the view follows the input instead of
        // lurching after long frames. The settled frame renders in full.
        let preview_scale = if interacting {
            PREVIEW_SCALE_INTERACTING
        } else if building {
            PREVIEW_SCALE_BUILDING
        } else {
            1
        };

        let (precision_text, zoom_colour) = if deep.is_some() && deep_view.is_some() {
            ("AP PERT", BLUE)
        } else if deep.is_some() {
            ("F64 PERT", BLUE)
        } else {
            match precision {
                PrecisionMode::DoubleSingle => (
                    ds_validity.label(),
                    match ds_validity.level {
                        ValidityLevel::Stable => BLUE,
                        ValidityLevel::Risk => CREAM,
                        ValidityLevel::Limit => CORAL,
                    },
                ),
                PrecisionMode::F32 if probe.f32.unstable() => ("F32 UNSTABLE", CORAL),
                PrecisionMode::F32 if f32_only_limited => ("F32 LIMIT", CORAL),
                PrecisionMode::F32 => ("F32", CREAM),
            }
        };
        ui.painter().text(
            egui::pos2(header.max.x - 10.0, header.center().y),
            egui::Align2::RIGHT_CENTER,
            format!(
                "{precision_text}  ·  ZOOM ×{}",
                deep_view.as_ref().map_or_else(
                    || format!("{magnification:.4e}"),
                    |deep| deep.magnification_label()
                )
            ),
            egui::FontId::new(11.0, egui::FontFamily::Monospace),
            zoom_colour,
        );

        // While a deep view is active the shader's DS centre must track the
        // arbitrary-precision centre (the reference parameter of the
        // perturbation path), not the frozen handoff view.
        let shader_view = deep_view.as_ref().map_or(view, |deep| {
            PlaneView::new(deep.centre_preview(), deep.half_height_preview())
        });
        let mut uniforms = Uniforms::new(
            shader_view.centre,
            shader_view.half_height,
            aspect,
            self.julia_c,
            self.iterations,
            self.bailout,
            self.family.shader_flag(),
            pane,
            self.layers.layers[0].colouring.outside.smooth,
            self.grid,
            precision,
            self.family_parameters
                .uniform_words(self.pane_is_dynamical(pane)),
        );
        if let Some(data) = &deep {
            uniforms = uniforms.enable_perturbation(
                data.scale_mantissa,
                data.scale_exponent,
                data.reference.len(),
                data.ds_fallback,
                data.reference_offset,
            );
        }
        // Natural log of one pixel's height in world units, from the
        // magnification so it stays finite far below f64 range.
        let pixel_height_points = viewport.height().max(1.0) as f64;
        let pixel_log = std::f64::consts::LN_10
            * (1.45_f64.log10() - self.magnification_log10(pane))
            + (2.0 / (pixel_height_points * ui.ctx().pixels_per_point() as f64)).ln();
        let colouring_uniforms = ColouringUniforms::new(&self.layers, pixel_log as f32);
        let gradient = self.gradient_table();
        ui.painter().add(render::callback(
            viewport,
            pane,
            uniforms,
            colouring_uniforms,
            gradient,
            deep,
            preview_scale,
            ui.ctx().pixels_per_point(),
        ));

        if self.family.is_quadratic() && pane == 1 && self.show_orbit_overlay && deep_view.is_none()
        {
            let orbit = self.orbit_cache.update(OrbitInput {
                c: self.julia_c,
                iterations: self.iterations,
                bailout: self.bailout as f64,
            });
            let selected = self.orbit_step.min(orbit.last_iteration());
            self.orbit_step = selected;
            draw_orbit_overlay(ui, viewport, &view, orbit, selected);
        }

        if let Some(focus) = self.zoom_focus[pane]
            && deep_view.is_none()
        {
            self.draw_focus_marker(ui, viewport, &view, focus);
        }
        if deep_view.is_none() {
            self.draw_readout(ui, viewport, response.hover_pos(), &view, precision);
        } else if let Some(deep_view) = &deep_view {
            self.draw_deep_readout(ui, viewport, response.hover_pos(), deep_view);
        }
    }

    /// The rasterised gradients for the render callbacks, rebuilt only when
    /// a visible layer's gradient (or the visible set) changed.
    fn gradient_table(&mut self) -> Arc<GradientTable> {
        let sources: Vec<Gradient> = self
            .layers
            .visible()
            .map(|layer| layer.colouring.gradient.clone())
            .collect();
        if sources != self.gradient_table_source {
            let generation = self.gradient_table.generation + 1;
            self.gradient_table = Arc::new(GradientTable::new(generation, &self.layers));
            self.gradient_table_source = sources;
        }
        Arc::clone(&self.gradient_table)
    }

    /// The layer list: visibility, blend settings and stacking order, shown
    /// top layer first as in Ultra Fractal.
    fn layer_list(&mut self, ui: &mut egui::Ui) {
        let count = self.layers.layers.len();
        let mut remove: Option<usize> = None;
        let mut swap: Option<(usize, usize)> = None;
        for index in (0..count).rev() {
            let active = index == self.layers.active;
            let bottom = index == 0;
            ui.horizontal(|ui| {
                {
                    let layer = &mut self.layers.layers[index];
                    let eye = if layer.visible { "👁" } else { "—" };
                    if ui
                        .add(egui::Button::new(eye).min_size(egui::vec2(24.0, 0.0)))
                        .on_hover_text("Show or hide this layer")
                        .clicked()
                    {
                        layer.visible = !layer.visible;
                    }
                }
                // Reorder and delete sit right-aligned on the name row so
                // the blend row below keeps its full width for the slider.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(count > 1, egui::Button::new("✕").small())
                        .on_hover_text("Delete this layer")
                        .clicked()
                    {
                        remove = Some(index);
                    }
                    if ui
                        .add_enabled(index > 0, egui::Button::new("▼").small())
                        .on_hover_text("Move down")
                        .clicked()
                    {
                        swap = Some((index, index - 1));
                    }
                    if ui
                        .add_enabled(index + 1 < count, egui::Button::new("▲").small())
                        .on_hover_text("Move up")
                        .clicked()
                    {
                        swap = Some((index, index + 1));
                    }
                    let layer = &self.layers.layers[index];
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        if ui
                            .selectable_label(active, &layer.name)
                            .on_hover_text("Select this layer for editing")
                            .clicked()
                        {
                            self.layers.active = index;
                            self.gradient_selected_stop = 0;
                        }
                    });
                });
            });
            ui.horizontal(|ui| {
                ui.add_space(26.0);
                let layer = &mut self.layers.layers[index];
                if ui
                    .selectable_label(layer.mask, "M")
                    .on_hover_text(
                        "Mask: this layer paints nothing; its luminance (times its opacity) multiplies the opacity of the layer above it",
                    )
                    .clicked()
                {
                    layer.mask = !layer.mask;
                }
                if layer.mask {
                    ui.label(egui::RichText::new("mask ↑").small().color(CREAM));
                } else if bottom {
                    ui.label(egui::RichText::new("base").small().color(MUTED))
                        .on_hover_text(
                            "The bottom layer composites over black; its merge mode is ignored",
                        );
                } else {
                    egui::ComboBox::from_id_salt(("iterascope.layer.mode", index))
                        .selected_text(layer.merge_mode.name())
                        .width(88.0)
                        .show_ui(ui, |ui| {
                            for mode in MergeMode::ALL {
                                ui.selectable_value(&mut layer.merge_mode, mode, mode.name());
                            }
                        });
                }
                // The opacity slider takes whatever width remains.
                ui.spacing_mut().slider_width = (ui.available_width() - 16.0).max(40.0);
                ui.add(
                    egui::Slider::new(&mut layer.opacity, 0.0..=1.0)
                        .show_value(false)
                        .text(""),
                )
                .on_hover_text(format!("Opacity {:.0}%", layer.opacity * 100.0));
            });
        }
        if let Some((a, b)) = swap {
            self.layers.layers.swap(a, b);
            if self.layers.active == a {
                self.layers.active = b;
            } else if self.layers.active == b {
                self.layers.active = a;
            }
        }
        if let Some(index) = remove
            && self.layers.layers.len() > 1
        {
            self.layers.layers.remove(index);
            if self.layers.active >= self.layers.layers.len() {
                self.layers.active = self.layers.layers.len() - 1;
            }
        }
        ui.horizontal(|ui| {
            let full = self.layers.layers.len() >= MAX_LAYERS;
            if ui
                .add_enabled(!full, egui::Button::new("Duplicate"))
                .on_hover_text("Copy the active layer above itself")
                .clicked()
            {
                let mut copy = self.layers.active_layer().clone();
                copy.name = format!("{} copy", copy.name);
                let at = self.layers.active + 1;
                self.layers.layers.insert(at, copy);
                self.layers.active = at;
            }
            if ui
                .add_enabled(!full, egui::Button::new("Add"))
                .on_hover_text("Add a default layer on top")
                .clicked()
            {
                let mut layer = Layer {
                    name: format!("Layer {}", self.layers.layers.len() + 1),
                    ..Layer::default()
                };
                if self.family.converges() {
                    layer.colouring.outside = ColouringSide::default_basins();
                }
                self.layers.layers.push(layer);
                self.layers.active = self.layers.layers.len() - 1;
            }
            let name = &mut self.layers.active_layer_mut().name;
            ui.add(egui::TextEdit::singleline(name).desired_width(110.0));
        });
    }

    fn colouring_controls(&mut self, ui: &mut egui::Ui) {
        let family = self.family;
        if self.layers.layers.len() > 1 {
            ui.label(
                egui::RichText::new(format!(
                    "Editing layer: {}",
                    self.layers.active_layer().name
                ))
                .small()
                .color(CREAM),
            );
        }
        // Gradient preview; click to open the editor.
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 20.0), egui::Sense::click());
        paint_gradient(
            ui.painter(),
            rect,
            &self.layers.active_colouring().gradient,
            96,
        );
        ui.painter().rect_stroke(
            rect,
            2.0,
            egui::Stroke::new(1.0, if response.hovered() { CREAM } else { BORDER }),
            egui::StrokeKind::Inside,
        );
        if response.clicked() {
            self.gradient_editor_open = true;
        }
        response.on_hover_text("Open the gradient editor");
        ui.horizontal(|ui| {
            if ui.button("Edit gradient…").clicked() {
                self.gradient_editor_open = true;
            }
            egui::ComboBox::from_id_salt("iterascope.gradient.preset")
                .selected_text("Presets")
                .width(110.0)
                .show_ui(ui, |ui| {
                    for name in presets::NAMES {
                        if ui.selectable_label(false, name).clicked()
                            && let Some(gradient) = presets::by_name(name)
                        {
                            self.layers.active_colouring_mut().gradient = gradient;
                            self.gradient_selected_stop = 0;
                        }
                    }
                });
        });
        ui.add_space(4.0);

        {
            let layer = self.layers.active_layer_mut();
            let uses_accumulators = [&layer.colouring.outside, &layer.colouring.inside]
                .iter()
                .any(|side| {
                    matches!(
                        side.algorithm,
                        ColouringAlgorithm::OrbitTrap
                            | ColouringAlgorithm::Stripes
                            | ColouringAlgorithm::TriangleInequality
                    )
                });
            if uses_accumulators {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("skip first").color(MUTED));
                    ui.add(
                        egui::DragValue::new(&mut layer.skip_iterations)
                            .range(0..=49_999)
                            .speed(5),
                    )
                    .on_hover_text(
                        "Iterations the trap, stripe and triangle accumulators ignore. Deep-zoom pixels share their leading iterations, which flattens these colourings; skip the shared prefix to restore their variety.",
                    );
                    ui.label(egui::RichText::new("iterations").color(MUTED));
                });
            }
        }

        let outside_label = if family.converges() {
            "Outside · converged"
        } else if family == FractalFamily::Lyapunov {
            "Stable regions"
        } else {
            "Outside · escaped"
        };
        egui::CollapsingHeader::new(outside_label)
            .id_salt("iterascope.colouring.outside")
            .default_open(true)
            .show(ui, |ui| {
                colouring_side_controls(
                    ui,
                    &mut self.layers.active_colouring_mut().outside,
                    family,
                    true,
                );
            });
        if family != FractalFamily::Lyapunov {
            egui::CollapsingHeader::new("Inside · bounded")
                .id_salt("iterascope.colouring.inside")
                .default_open(false)
                .show(ui, |ui| {
                    colouring_side_controls(
                        ui,
                        &mut self.layers.active_colouring_mut().inside,
                        family,
                        false,
                    );
                });
        }
        ui.add_space(2.0);
        ui.checkbox(&mut self.grid, "Coordinate grid");
    }

    fn gradient_editor(&mut self, ctx: &egui::Context) {
        if !self.gradient_editor_open {
            return;
        }
        let mut open = self.gradient_editor_open;
        egui::Window::new("Gradient")
            .open(&mut open)
            .default_width(560.0)
            .resizable(true)
            .show(ctx, |ui| {
                self.gradient_editor_contents(ui);
            });
        self.gradient_editor_open = open;
    }

    fn gradient_editor_contents(&mut self, ui: &mut egui::Ui) {
        let gradient = &mut self.layers.active_colouring_mut().gradient;
        gradient.normalise();
        if self.gradient_selected_stop >= gradient.stops.len() {
            self.gradient_selected_stop = 0;
        }

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Name").color(MUTED));
            ui.add(egui::TextEdit::singleline(&mut gradient.name).desired_width(180.0));
            ui.add_space(8.0);
            egui::ComboBox::from_id_salt("iterascope.gradient.editor.preset")
                .selected_text("Presets")
                .width(110.0)
                .show_ui(ui, |ui| {
                    for name in presets::NAMES {
                        if ui.selectable_label(false, name).clicked()
                            && let Some(preset) = presets::by_name(name)
                        {
                            *gradient = preset;
                            self.gradient_selected_stop = 0;
                        }
                    }
                });
            if ui.button("Random").clicked() {
                self.gradient_random_seed = self
                    .gradient_random_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *gradient = Gradient::random(self.gradient_random_seed >> 20);
                self.gradient_selected_stop = 0;
            }
        });
        ui.add_space(6.0);

        // The gradient bar with draggable stop markers beneath it.
        let bar_height = 44.0;
        let marker_height = 14.0;
        let (rect, response) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), bar_height + marker_height + 2.0),
            egui::Sense::click(),
        );
        let bar = egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), bar_height));
        paint_gradient(ui.painter(), bar, gradient, 256);
        ui.painter().rect_stroke(
            bar,
            2.0,
            egui::Stroke::new(1.0, BORDER),
            egui::StrokeKind::Inside,
        );
        let rotation = gradient.rotation;
        let to_x = |position: f32| bar.min.x + wrap_turns(position + rotation) * bar.width();
        if response.double_clicked()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let position = ((pointer.x - bar.min.x) / bar.width()).clamp(0.0, 0.9999);
            self.gradient_selected_stop = gradient.insert_stop(position);
        }
        response.on_hover_text("Double-click to add a stop; drag the markers to move stops");
        let mut drag: Option<(usize, f32)> = None;
        for (index, stop) in gradient.stops.iter().enumerate() {
            let x = to_x(stop.position);
            let marker = egui::Rect::from_center_size(
                egui::pos2(x, bar.max.y + 1.0 + marker_height * 0.5),
                egui::vec2(10.0, marker_height),
            );
            let id = ui.id().with(("gradient-stop", index));
            let marker_response = ui.interact(marker, id, egui::Sense::click_and_drag());
            if marker_response.clicked() || marker_response.drag_started() {
                self.gradient_selected_stop = index;
            }
            if marker_response.dragged() {
                drag = Some((index, marker_response.drag_delta().x / bar.width()));
            }
            let selected = index == self.gradient_selected_stop;
            let colour = egui::Color32::from_rgb(
                (stop.colour[0] * 255.0).round() as u8,
                (stop.colour[1] * 255.0).round() as u8,
                (stop.colour[2] * 255.0).round() as u8,
            );
            // Triangle pointing at the bar, filled with the stop colour.
            let points = vec![
                egui::pos2(x, bar.max.y + 1.0),
                egui::pos2(x - 5.0, bar.max.y + 1.0 + marker_height),
                egui::pos2(x + 5.0, bar.max.y + 1.0 + marker_height),
            ];
            ui.painter().add(egui::Shape::convex_polygon(
                points,
                colour,
                egui::Stroke::new(
                    if selected { 2.0 } else { 1.0 },
                    if selected { egui::Color32::WHITE } else { TEXT },
                ),
            ));
            if selected {
                ui.painter().vline(
                    x,
                    bar.y_range(),
                    egui::Stroke::new(1.0, egui::Color32::from_white_alpha(160)),
                );
            }
        }
        if let Some((index, delta)) = drag {
            let stop = &mut gradient.stops[index];
            stop.position = wrap_turns(stop.position + delta);
            // Keep the dragged stop selected through the re-sort.
            let position = stop.position;
            let colour = stop.colour;
            gradient.normalise();
            if let Some(found) = gradient
                .stops
                .iter()
                .position(|s| s.position == position && s.colour == colour)
            {
                self.gradient_selected_stop = found;
            }
        }
        ui.add_space(4.0);

        // Selected stop.
        let count = gradient.stops.len();
        let index = self.gradient_selected_stop.min(count - 1);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(format!("Stop {} of {count}", index + 1)).color(CREAM));
            let stop = &mut gradient.stops[index];
            let mut rgb = [
                (stop.colour[0] * 255.0).round() as u8,
                (stop.colour[1] * 255.0).round() as u8,
                (stop.colour[2] * 255.0).round() as u8,
            ];
            if egui::color_picker::color_edit_button_srgb(ui, &mut rgb).changed() {
                stop.colour = [
                    rgb[0] as f32 / 255.0,
                    rgb[1] as f32 / 255.0,
                    rgb[2] as f32 / 255.0,
                ];
            }
            let mut position = stop.position;
            if ui
                .add(
                    egui::DragValue::new(&mut position)
                        .speed(0.002)
                        .range(0.0..=0.9999)
                        .fixed_decimals(4)
                        .prefix("position "),
                )
                .changed()
            {
                stop.position = position;
                let colour = stop.colour;
                gradient.normalise();
                if let Some(found) = gradient
                    .stops
                    .iter()
                    .position(|s| s.position == wrap_turns(position) && s.colour == colour)
                {
                    self.gradient_selected_stop = found;
                }
            }
            if ui
                .button("Add")
                .on_hover_text("Insert a stop halfway to the next one")
                .clicked()
            {
                let this = gradient.stops[index].position;
                let next = gradient.stops[(index + 1) % count].position;
                let mut mid = if count == 1 {
                    this + 0.5
                } else if next > this {
                    0.5 * (this + next)
                } else {
                    0.5 * (this + next + 1.0)
                };
                mid = wrap_turns(mid + gradient.rotation);
                self.gradient_selected_stop = gradient.insert_stop(mid);
            }
            if ui
                .add_enabled(count > 1, egui::Button::new("Remove"))
                .clicked()
            {
                gradient.remove_stop(index);
                self.gradient_selected_stop = index.saturating_sub(1);
            }
        });
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("iterascope.gradient.interpolation")
                .selected_text(gradient.interpolation.name())
                .width(110.0)
                .show_ui(ui, |ui| {
                    for mode in Interpolation::ALL {
                        ui.selectable_value(&mut gradient.interpolation, mode, mode.name());
                    }
                });
            ui.checkbox(&mut gradient.smooth, "Smooth")
                .on_hover_text("Cubic blending between stops instead of linear");
            if ui.button("Reverse").clicked() {
                gradient.reverse();
            }
            if ui
                .button("Distribute")
                .on_hover_text("Space the stops evenly")
                .clicked()
            {
                gradient.distribute_evenly();
            }
        });
        ui.add(egui::Slider::new(&mut gradient.rotation, 0.0..=1.0).text("rotation"));
        ui.add_space(6.0);

        egui::CollapsingHeader::new("Import / export")
            .id_salt("iterascope.gradient.import")
            .default_open(false)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Paste an Ultra Fractal .ugr gradient or a Fractint .map palette and choose Import; or drop the file onto the window. Export copies the gradient as .ugr text.",
                    )
                    .small()
                    .color(MUTED),
                );
                ui.add(
                    egui::TextEdit::multiline(&mut self.gradient_import_text)
                        .code_editor()
                        .desired_width(f32::INFINITY)
                        .desired_rows(6),
                );
                ui.horizontal(|ui| {
                    if ui.button("Import").clicked() {
                        match Gradient::parse(&self.gradient_import_text) {
                            Ok(mut gradients) => {
                                let found = gradients.len();
                                *gradient = gradients.swap_remove(0);
                                self.gradient_selected_stop = 0;
                                self.gradient_message = Some((
                                    if found > 1 {
                                        format!("Imported the first of {found} gradients")
                                    } else {
                                        format!("Imported {}", gradient.name)
                                    },
                                    false,
                                ));
                            }
                            Err(error) => {
                                self.gradient_message = Some((format!("Import failed: {error}"), true));
                            }
                        }
                    }
                    if ui.button("Export (.ugr to clipboard)").clicked() {
                        let text = gradient.to_ugr();
                        self.gradient_import_text = text.clone();
                        ui.ctx().copy_text(text);
                        self.gradient_message = Some(("Gradient copied as .ugr".to_owned(), false));
                    }
                });
                if let Some((message, error)) = &self.gradient_message {
                    ui.label(egui::RichText::new(message).color(if *error { CORAL } else { BLUE }));
                }
            });

        // Dropped files anywhere in the window import as gradients.
        let dropped: Vec<String> = ui.ctx().input(|input| {
            input
                .raw
                .dropped_files
                .iter()
                .filter_map(|file| {
                    file.bytes
                        .as_ref()
                        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                        .or_else(|| {
                            file.path
                                .as_ref()
                                .and_then(|path| std::fs::read_to_string(path).ok())
                        })
                })
                .collect()
        });
        if let Some(text) = dropped.into_iter().next() {
            match Gradient::parse(&text) {
                Ok(mut gradients) => {
                    *gradient = gradients.swap_remove(0);
                    self.gradient_selected_stop = 0;
                    self.gradient_message = Some((format!("Imported {}", gradient.name), false));
                }
                Err(error) => {
                    self.gradient_message = Some((format!("Import failed: {error}"), true));
                }
            }
        }
    }

    fn animation_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new(
                "A zoom dive to the current view's centre: the magnification moves between the start and end exponents at constant (optionally eased) logarithmic speed. One reference orbit serves every frame.",
            )
            .small()
            .color(MUTED),
        );
        ui.add(
            egui::Slider::new(&mut self.animation.duration_seconds, 1.0..=180.0)
                .logarithmic(true)
                .text("duration s"),
        );
        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("iterascope.animation.fps")
                .selected_text(format!("{} fps", self.animation.fps))
                .width(76.0)
                .show_ui(ui, |ui| {
                    for fps in [24u32, 25, 30, 50, 60] {
                        ui.selectable_value(&mut self.animation.fps, fps, format!("{fps} fps"));
                    }
                });
            ui.add(
                egui::DragValue::new(&mut self.animation.width)
                    .range(16..=animation::MAX_DIMENSION)
                    .speed(8),
            );
            ui.label("×");
            ui.add(
                egui::DragValue::new(&mut self.animation.height)
                    .range(16..=animation::MAX_DIMENSION)
                    .speed(8),
            );
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("zoom 10^").color(MUTED));
            ui.add(
                egui::DragValue::new(&mut self.animation.start_magnification_log10)
                    .range(-3.0..=MAX_DECIMAL_ZOOM_EXPONENT as f64)
                    .speed(0.25)
                    .fixed_decimals(2),
            );
            ui.label("→ 10^");
            ui.add(
                egui::DragValue::new(&mut self.animation.end_magnification_log10)
                    .range(-3.0..=MAX_DECIMAL_ZOOM_EXPONENT as f64)
                    .speed(0.25)
                    .fixed_decimals(2),
            );
            if ui
                .button("= view")
                .on_hover_text("Set the end magnification to the active pane's current zoom")
                .clicked()
            {
                self.animation.end_magnification_log10 = self.magnification_log10(self.active_pane);
            }
        });
        ui.checkbox(&mut self.animation.ease, "Ease in and out");
        ui.add(
            egui::Slider::new(&mut self.animation.gradient_sweep_turns, -4.0..=4.0)
                .text("gradient sweep (turns)"),
        );
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.animation_export_controls(ui);
        }
        #[cfg(target_arch = "wasm32")]
        ui.label(
            egui::RichText::new("Image-sequence export runs in the native application.")
                .small()
                .color(CREAM),
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn still_controls(&mut self, ui: &mut egui::Ui) {
        ui.label(
            egui::RichText::new(
                "Render the active pane's current view as a PNG with supersampled anti-aliasing; sizes past 8192 render as tiles around one reference orbit.",
            )
            .small()
            .color(MUTED),
        );
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.still_width)
                    .range(16..=animation::MAX_STILL_DIMENSION)
                    .speed(16),
            );
            ui.label("×");
            ui.add(
                egui::DragValue::new(&mut self.still_height)
                    .range(16..=animation::MAX_STILL_DIMENSION)
                    .speed(16),
            );
            egui::ComboBox::from_id_salt("iterascope.still.supersample")
                .selected_text(format!("{0}×{0} AA", self.still_supersample.clamp(1, 3)))
                .width(84.0)
                .show_ui(ui, |ui| {
                    for factor in [1u32, 2, 3] {
                        ui.selectable_value(
                            &mut self.still_supersample,
                            factor,
                            format!("{factor}×{factor} AA"),
                        );
                    }
                });
        });
        if let Some(job) = &self.still {
            let total = job.tile_count();
            ui.add(
                egui::ProgressBar::new(job.next_tile as f32 / total as f32)
                    .text(format!("tile {}/{total}", job.next_tile)),
            );
            if ui.button("Cancel").clicked() {
                self.still = None;
                self.export_message = Some(("Still render cancelled".to_owned(), true));
            }
        } else if ui
            .add_enabled(self.export.is_none(), egui::Button::new("Render still"))
            .clicked()
        {
            self.start_still();
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn animation_export_controls(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(
            &mut self.animation.encode_video,
            "Encode MP4 with ffmpeg (if installed)",
        );
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Into").color(MUTED));
            ui.add(
                egui::TextEdit::singleline(&mut self.export_directory).desired_width(f32::INFINITY),
            );
        });
        if let Some(job) = &self.export {
            let total = job.animation.frame_count();
            let done = job.next_frame;
            let elapsed = job.started.elapsed().as_secs_f64();
            let remaining = if done > 0 {
                format!(
                    " · ~{:.0} s left",
                    elapsed / done as f64 * (total - done) as f64
                )
            } else {
                String::new()
            };
            ui.add(
                egui::ProgressBar::new(done as f32 / total as f32)
                    .text(format!("frame {done}/{total}{remaining}")),
            );
            if ui.button("Cancel").clicked() {
                self.export = None;
                self.export_message = Some(("Export cancelled".to_owned(), true));
            }
        } else {
            let frames = self.animation.frame_count();
            ui.label(
                egui::RichText::new(format!(
                    "{frames} frames at {}×{}",
                    self.animation.width, self.animation.height
                ))
                .small()
                .color(MUTED),
            );
            if ui.button("Render image sequence").clicked() {
                self.start_export();
            }
        }
        if let Some((message, error)) = &self.export_message {
            let colour = if *error { CORAL } else { BLUE };
            ui.label(egui::RichText::new(message).color(colour));
        }
    }

    /// Freezes the current view, dynamics and colouring into an export job
    /// and builds its reference orbit. The centre never moves during the
    /// animation, so the orbit of the (deep) centre serves every frame:
    /// re-described in scale for arbitrary-precision frames, projected to
    /// `f64` for frames below the handoff.
    #[cfg(not(target_arch = "wasm32"))]
    fn start_export(&mut self) {
        if self.export.is_some() {
            return;
        }
        self.export_message = None;
        let mut animation = self.animation.clone();
        // Video encoders want even dimensions.
        animation.width &= !1;
        animation.height &= !1;
        if let Err(error) = animation.validate() {
            self.export_message = Some((error, true));
            return;
        }
        let pane = self.active_pane;
        let current = self.magnification_log10(pane);
        let mut clamped = false;
        for target in [
            &mut animation.start_magnification_log10,
            &mut animation.end_magnification_log10,
        ] {
            if *target > current + 1e-9 {
                *target = current;
                clamped = true;
            }
        }

        let scene = match self.freeze_scene(pane) {
            Ok(scene) => scene,
            Err(error) => {
                self.export_message = Some((error, true));
                return;
            }
        };
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0);
        let directory = std::path::PathBuf::from(self.export_directory.trim())
            .join(format!("zoom-{}-{stamp}", self.family.document_id()));
        if let Err(error) = std::fs::create_dir_all(&directory) {
            self.export_message = Some((format!("cannot create {directory:?}: {error}"), true));
            return;
        }

        self.export = Some(ExportJob {
            animation,
            scene,
            directory,
            next_frame: 0,
            started: Instant::now(),
        });
        if clamped {
            self.export_message = Some((
                "Magnification clamped to the current view (zoom further first to go deeper)"
                    .to_owned(),
                false,
            ));
        }
    }

    /// Freezes the pane's dynamics, layer stack and reference orbit for
    /// offline rendering. The centre never moves, so one orbit — the
    /// arbitrary-precision orbit of a deep centre, or an `f64` orbit
    /// otherwise — serves any magnification up to the current view's.
    #[cfg(not(target_arch = "wasm32"))]
    fn freeze_scene(&mut self, pane: usize) -> Result<FrozenScene, String> {
        let dynamical = self.pane_is_dynamical(pane);
        self.export_generation += 1;
        let base = self.export_generation * 4;
        let mut ap_reference = None;
        let f64_points;
        let centre;
        if let Some(view) = &self.deep_views[pane] {
            let julia_c = self.deep_julia_c.clone().unwrap_or_else(|| {
                DeepComplex::from_f64(self.julia_c, view.zoom_exponent)
                    .expect("finite Julia parameter")
            });
            let initial = DeepState::initial(self.family, &view.centre, dynamical, &julia_c)?;
            let orbit = ReferenceOrbit::family(
                self.family,
                &self.family_parameters,
                initial,
                self.iterations,
                self.bailout as f64,
            )?;
            centre = view.centre_preview();
            f64_points = orbit
                .points
                .iter()
                .map(|point| [point.re.to_f64(), point.im.to_f64()])
                .collect::<Vec<_>>();
            ap_reference = Some((
                base + 1,
                Arc::clone(
                    &DeepRenderData::from_points(base + 1, 1.0, 0, &orbit.points, false).reference,
                ),
            ));
        } else {
            centre = if pane == 0 {
                self.parameter.centre
            } else {
                self.dynamical.centre
            };
            if self.family.supports_deep_zoom() {
                f64_points = reference_orbit_f64(
                    self.family,
                    &self.family_parameters,
                    initial_state_with(self.family, centre, dynamical, self.julia_c),
                    self.iterations,
                    self.bailout as f64,
                )
                .points;
            } else {
                f64_points = Vec::new();
            }
        }
        let f64_reference = (!f64_points.is_empty()).then(|| {
            (
                base + 2,
                Arc::clone(
                    &DeepRenderData::from_f64_orbit(base + 2, 1.0, &f64_points, true, [0.0; 2])
                        .reference,
                ),
            )
        });
        Ok(FrozenScene {
            family: self.family,
            dynamical,
            family_words: self.family_parameters.uniform_words(dynamical),
            iterations: self.iterations,
            bailout: self.bailout,
            julia_c: self.julia_c,
            centre,
            layers: self.layers.clone(),
            gradient: Arc::new(GradientTable::new(base + 3, &self.layers)),
            ap_reference,
            f64_reference,
        })
    }

    /// Starts a still render of the current view: the frame renders at
    /// `supersample`× the requested size — in tiles when the supersampled
    /// frame exceeds the 8192-pixel texture limit — and is box-filtered
    /// down in linear light.
    #[cfg(not(target_arch = "wasm32"))]
    fn start_still(&mut self) {
        if self.still.is_some() {
            return;
        }
        self.export_message = None;
        let width = (self.still_width & !1).clamp(16, animation::MAX_STILL_DIMENSION);
        let height = (self.still_height & !1).clamp(16, animation::MAX_STILL_DIMENSION);
        let supersample = self.still_supersample.clamp(1, 3);
        let pane = self.active_pane;
        let magnification = self.magnification_log10(pane);
        let scene = match self.freeze_scene(pane) {
            Ok(scene) => scene,
            Err(error) => {
                self.export_message = Some((error, true));
                return;
            }
        };
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0);
        let directory = std::path::PathBuf::from(self.export_directory.trim());
        if let Err(error) = std::fs::create_dir_all(&directory) {
            self.export_message = Some((format!("cannot create {directory:?}: {error}"), true));
            return;
        }
        let path = directory.join(format!("still-{}-{stamp}.png", self.family.document_id()));
        // Tiles are laid out in final pixels so downsampling blocks never
        // straddle a tile edge; each supersampled tile fits the texture.
        let max_tile = animation::MAX_DIMENSION / supersample;
        self.still = Some(StillJob {
            scene,
            magnification,
            width,
            height,
            supersample,
            columns: animation::tile_spans(width, max_tile),
            rows: animation::tile_spans(height, max_tile),
            next_tile: 0,
            image: vec![0; width as usize * height as usize * 4],
            path,
            started: Instant::now(),
        });
    }

    /// Renders one still tile per UI update.
    #[cfg(not(target_arch = "wasm32"))]
    fn advance_still(&mut self, ctx: &egui::Context) {
        let render_state = self.render_state.clone();
        let Some(job) = &mut self.still else {
            return;
        };
        let tile = job.next_tile;
        let column = tile % job.columns.len();
        let row = tile / job.columns.len();
        let (column_offset, tile_width) = job.columns[column];
        let (row_offset, tile_height) = job.rows[row];
        let supersample = job.supersample;
        let rendered = match job.scene.render_region(
            &render_state,
            job.magnification,
            (job.width * supersample, job.height * supersample),
            (column_offset * supersample, row_offset * supersample),
            (tile_width * supersample, tile_height * supersample),
            0.0,
        ) {
            Ok(rendered) => rendered,
            Err(error) => {
                self.still = None;
                self.export_message = Some((error, true));
                return;
            }
        };
        let tile_rgba = downsample_srgb(&rendered, tile_width * supersample, supersample);
        for y in 0..tile_height as usize {
            let source = y * tile_width as usize * 4..(y + 1) * tile_width as usize * 4;
            let target =
                ((row_offset as usize + y) * job.width as usize + column_offset as usize) * 4;
            job.image[target..target + tile_width as usize * 4].copy_from_slice(&tile_rgba[source]);
        }

        job.next_tile += 1;
        if job.next_tile >= job.tile_count() {
            let job = self.still.take().unwrap();
            if let Err(error) = write_png(&job.path, job.width, job.height, &job.image) {
                self.export_message = Some((format!("cannot write {:?}: {error}", job.path), true));
                return;
            }
            self.export_message = Some((
                format!(
                    "Wrote {path} ({width}×{height}, {ss}×{ss} anti-aliasing, {tiles} tiles) in {seconds:.1} s",
                    path = job.path.display(),
                    width = job.width,
                    height = job.height,
                    ss = job.supersample,
                    tiles = job.tile_count(),
                    seconds = job.started.elapsed().as_secs_f64(),
                ),
                false,
            ));
        }
        ctx.request_repaint();
    }

    /// Renders one export frame per UI update so the interface stays
    /// responsive; the export pane's GPU resources are separate from the
    /// interactive panes'.
    #[cfg(not(target_arch = "wasm32"))]
    fn advance_export(&mut self, ctx: &egui::Context) {
        let render_state = self.render_state.clone();
        let Some(job) = &mut self.export else {
            return;
        };
        let frame = job.next_frame;
        let total = job.animation.frame_count();
        let magnification = job.animation.magnification_log10_at(frame);
        let rgba = match job.scene.render_frame(
            &render_state,
            magnification,
            (job.animation.width, job.animation.height),
            job.animation.gradient_offset_at(frame),
        ) {
            Ok(rgba) => rgba,
            Err(error) => {
                self.export = None;
                self.export_message = Some((error, true));
                return;
            }
        };

        let path = job.directory.join(format!("frame-{frame:05}.png"));
        if let Err(error) = write_png(&path, job.animation.width, job.animation.height, &rgba) {
            self.export = None;
            self.export_message = Some((format!("cannot write {path:?}: {error}"), true));
            return;
        }

        job.next_frame += 1;
        if job.next_frame >= total {
            let directory = job.directory.clone();
            let fps = job.animation.fps;
            let encode = job.animation.encode_video;
            let elapsed = job.started.elapsed().as_secs_f64();
            self.export = None;
            let mut message = format!(
                "Wrote {total} frames to {} in {elapsed:.0} s",
                directory.display()
            );
            if encode {
                match encode_video(&directory, fps) {
                    Ok(output) => message = format!("{message}; encoded {output}"),
                    Err(error) => message = format!("{message}; video not encoded: {error}"),
                }
            }
            self.export_message = Some((message, false));
        }
        ctx.request_repaint();
    }

    /// The Ultra Fractal-style switch picker: a parameter-plane window with
    /// crosshair, scroll zoom and a live Julia thumbnail. Clicking chooses
    /// `c`; the composited image behind the window follows immediately.
    /// Only offered while the single image shows the dynamical plane — the
    /// picker renders through the parameter pane's GPU resources, which are
    /// otherwise idle in that layout.
    fn switch_picker(&mut self, ctx: &egui::Context) {
        if !self.single_image
            || self.active_pane != 1
            || self.family.linkage() != Linkage::ParameterDynamical
        {
            self.switch_picker_open = false;
            return;
        }
        if !self.switch_picker_open {
            return;
        }
        let mut open = self.switch_picker_open;
        egui::Window::new("Choose c")
            .open(&mut open)
            .default_width(460.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Click to set the Julia parameter; scroll to zoom the plane.",
                    )
                    .small()
                    .color(MUTED),
                );
                let width = 440.0f32;
                let height = width / 1.45;
                let (rect, response) = ui
                    .allocate_exact_size(egui::vec2(width, height), egui::Sense::click_and_drag());
                let aspect = (rect.width() / rect.height().max(1.0)).max(0.1);
                let view = self.parameter;
                let pixel_log = (2.0 * view.half_height
                    / (rect.height().max(1.0) as f64 * ctx.pixels_per_point() as f64))
                    .ln() as f32;
                let uniforms = Uniforms::new(
                    view.centre,
                    view.half_height,
                    aspect,
                    self.julia_c,
                    self.iterations.min(2_048),
                    self.bailout,
                    self.family.shader_flag(),
                    0,
                    self.layers.layers[0].colouring.outside.smooth,
                    false,
                    PrecisionMode::F32,
                    self.family_parameters.uniform_words(false),
                );
                let colouring_uniforms = ColouringUniforms::new(&self.layers, pixel_log);
                let gradient = self.gradient_table();
                ui.painter().add(render::callback(
                    rect,
                    0,
                    uniforms,
                    colouring_uniforms,
                    Arc::clone(&gradient),
                    None,
                    1,
                    ctx.pixels_per_point(),
                ));
                ui.painter().rect_stroke(
                    rect,
                    2.0,
                    egui::Stroke::new(1.0, BORDER),
                    egui::StrokeKind::Inside,
                );

                let hover = response
                    .hover_pos()
                    .map(|position| (position, self.parameter.point_at(rect, position)));
                if let Some((position, point)) = hover {
                    // Crosshair at the cursor.
                    ui.painter().vline(
                        position.x,
                        rect.y_range(),
                        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(60)),
                    );
                    ui.painter().hline(
                        rect.x_range(),
                        position.y,
                        egui::Stroke::new(1.0, egui::Color32::from_white_alpha(60)),
                    );
                    // Scroll zooms about the cursor.
                    let scroll = ui.input(|input| input.smooth_scroll_delta.y) as f64;
                    if scroll != 0.0 {
                        let factor = 0.997f64.powf(scroll).clamp(0.2, 5.0);
                        let view = &mut self.parameter;
                        view.centre[0] = point[0] + (view.centre[0] - point[0]) * factor;
                        view.centre[1] = point[1] + (view.centre[1] - point[1]) * factor;
                        view.zoom(factor);
                    }
                    if response.clicked() {
                        self.julia_c = point;
                        self.deep_julia_c = None;
                        self.reframe_dynamical_plane();
                    }
                }
                // Marker at the currently selected c.
                if let Some(position) = complex_to_screen(&self.parameter, rect, self.julia_c)
                    && rect.contains(position)
                {
                    ui.painter()
                        .circle_stroke(position, 5.0, egui::Stroke::new(1.5, CREAM));
                }

                ui.horizontal(|ui| {
                    let c = hover.map_or(self.julia_c, |(_, point)| point);
                    ui.label(
                        egui::RichText::new(format!("c = {:+.6} {:+.6}i", c[0], c[1]))
                            .monospace()
                            .color(TEXT),
                    );
                    if ui.button("Reset view").clicked() {
                        self.parameter =
                            PlaneView::from_default(self.family.default_parameter_view());
                    }
                });

                // Live Julia preview of the hovered (else selected) c.
                let preview_c = hover.map_or(self.julia_c, |(_, point)| point);
                let preview_height = 120.0f32;
                let preview_width = preview_height * 1.45;
                let (preview_rect, _) = ui.allocate_exact_size(
                    egui::vec2(preview_width, preview_height),
                    egui::Sense::hover(),
                );
                let preview_view = self.family.default_dynamical_view();
                let preview_uniforms = Uniforms::new(
                    preview_view.centre,
                    preview_view.half_height,
                    preview_width / preview_height,
                    preview_c,
                    self.iterations.min(512),
                    self.bailout,
                    self.family.shader_flag(),
                    1,
                    self.layers.layers[0].colouring.outside.smooth,
                    false,
                    PrecisionMode::F32,
                    self.family_parameters.uniform_words(true),
                );
                let preview_pixel_log = (2.0 * preview_view.half_height
                    / (preview_rect.height().max(1.0) as f64 * ctx.pixels_per_point() as f64))
                    .ln() as f32;
                ui.painter().add(render::callback(
                    preview_rect,
                    render::EXPORT_PANE,
                    preview_uniforms,
                    ColouringUniforms::new(&self.layers, preview_pixel_log),
                    gradient,
                    None,
                    1,
                    ctx.pixels_per_point(),
                ));
                ui.painter().rect_stroke(
                    preview_rect,
                    2.0,
                    egui::Stroke::new(1.0, BORDER),
                    egui::StrokeKind::Inside,
                );
                if hover.is_some() {
                    ctx.request_repaint();
                }
            });
        self.switch_picker_open = open;
    }

    fn reframe_dynamical_plane(&mut self) {
        self.dynamical = PlaneView::from_default(self.family.default_dynamical_view());
        self.deep_views[1] = None;
        self.progressive_julia_zoom_active = false;
        self.progressive_julia_next_stage_at = None;
        self.zoom_focus[1] = None;
        self.orbit_step = 0;
    }

    fn reset_for_family(&mut self) {
        self.progressive_julia_zoom_active = false;
        self.progressive_julia_next_stage_at = None;
        self.deep_views = [None, None];
        self.deep_julia_c = None;
        self.deep_references = [None, None];
        self.deep_pending = [None, None];
        self.f64_references = [None, None];
        self.f64_reference_active = [false; 2];
        self.deep_active = [false; 2];
        self.zoom_focus = [None, None];
        self.pending_pan_steps = [0.0; 2];
        self.probes = [ProbeCache::default(); 2];
        self.ds_validity = [DsValidity::default(); 2];
        self.precision_modes = [PrecisionMode::F32; 2];
        self.active_pane = 0;
        self.orbit_inspector_open = false;
        self.parameter = PlaneView::from_default(self.family.default_parameter_view());
        self.dynamical = PlaneView::from_default(self.family.default_dynamical_view());
        match self.family.linkage() {
            Linkage::ParameterDynamical => {
                self.julia_c = self.family.default_parameter();
            }
            Linkage::OverviewDetail => {
                self.selected_z = self.family.default_parameter();
            }
        }
        self.iterations = self
            .iterations
            .clamp(self.family.min_iterations(), self.family.max_iterations());
        self.lyapunov_sequence_draft = self.family_parameters.lyapunov_sequence.clone();
    }

    fn magnification_log10(&self, pane: usize) -> f64 {
        self.deep_views[pane].as_ref().map_or_else(
            || {
                let view = if pane == 0 {
                    self.parameter
                } else {
                    self.dynamical
                };
                view.magnification().log10()
            },
            |view| view.magnification_log10,
        )
    }

    fn advance_progressive_julia_zoom(&mut self) -> Option<Duration> {
        if !self.progressive_julia_zoom_active {
            return None;
        }
        let now = Instant::now();
        if let Some(next_stage) = self.progressive_julia_next_stage_at
            && next_stage > now
        {
            return Some(next_stage - now);
        }
        let current = self.magnification_log10(1);
        let target = self.progressive_julia_zoom_target_exponent as f64;
        let Some(factor) = progressive_zoom_factor(current, target) else {
            self.progressive_julia_zoom_active = false;
            self.progressive_julia_next_stage_at = None;
            return None;
        };
        self.zoom_pane(1, factor);
        self.progressive_julia_next_stage_at = Some(now + PROGRESSIVE_ZOOM_STAGE_INTERVAL);
        Some(PROGRESSIVE_ZOOM_STAGE_INTERVAL)
    }

    fn zoom_pane(&mut self, pane: usize, factor: f64) {
        if !self.family.supports_deep_zoom() {
            let view = if pane == 0 {
                &mut self.parameter
            } else {
                &mut self.dynamical
            };
            view.zoom_from(self.zoom_focus[pane], factor);
            self.zoom_focus[pane] = None;
            return;
        }
        let handoff_log = ARBITRARY_HANDOFF_ZOOM.log10();
        if let Some(deep) = &mut self.deep_views[pane] {
            let next_log = deep.magnification_log10 - factor.log10();
            if next_log < handoff_log {
                let centre = deep.centre_preview();
                let half_height = 1.45 / 10.0_f64.powf(next_log);
                self.deep_views[pane] = None;
                let view = if pane == 0 {
                    &mut self.parameter
                } else {
                    &mut self.dynamical
                };
                view.centre = centre;
                view.half_height = half_height;
                self.zoom_focus[pane] = None;
            } else {
                let _ = deep.zoom(factor);
            }
            return;
        }

        let view = if pane == 0 {
            &mut self.parameter
        } else {
            &mut self.dynamical
        };
        let desired_half_height = view.half_height * factor;
        let handoff_half_height = 1.45 / ARBITRARY_HANDOFF_ZOOM;
        if factor < 1.0 && desired_half_height <= handoff_half_height {
            if let Some(focus) = self.zoom_focus[pane] {
                view.centre = focus;
            }
            view.half_height = handoff_half_height;
            if let Ok(mut deep) = DeepView::at_handoff(view.centre) {
                let remaining_factor = desired_half_height / handoff_half_height;
                if remaining_factor < 1.0 {
                    let _ = deep.zoom(remaining_factor);
                }
                self.deep_views[pane] = Some(deep);
                self.zoom_focus[pane] = None;
            }
        } else {
            view.zoom_from(self.zoom_focus[pane], factor);
        }
    }

    /// Builds (or reuses) the `f64` reference orbit of the pane's view centre
    /// for perturbation rendering below the arbitrary-precision handoff.
    /// Builds (or reuses) the `f64` reference orbit used for perturbation
    /// rendering below the arbitrary-precision handoff. If the view centre's
    /// orbit ends early, a coarse grid of candidates across the view is tried
    /// and the longest-lived one becomes the reference, so as few pixels as
    /// possible outlive it.
    fn f64_reference(
        &mut self,
        pane: usize,
        view: &PlaneView,
        aspect: f32,
    ) -> Option<Arc<DeepRenderData>> {
        if !view.half_height.is_finite() || view.half_height <= 0.0 {
            return None;
        }
        let key = F64ReferenceKey {
            family: self.family,
            parameters: self.family_parameters.clone(),
            centre: [view.centre[0].to_bits(), view.centre[1].to_bits()],
            half_height: view.half_height.to_bits(),
            julia: [self.julia_c[0].to_bits(), self.julia_c[1].to_bits()],
            iterations: self.iterations,
            bailout: self.bailout.to_bits(),
            pane,
        };
        if let Some(cached) = &self.f64_references[pane]
            && cached.key == key
        {
            return Some(Arc::clone(&cached.data));
        }
        let dynamical = self.pane_is_dynamical(pane);
        let orbit_at = |offset: [f32; 2]| {
            let world = [
                view.centre[0] + offset[0] as f64 * view.half_height,
                view.centre[1] + offset[1] as f64 * view.half_height,
            ];
            reference_orbit_f64(
                self.family,
                &self.family_parameters,
                initial_state_with(self.family, world, dynamical, self.julia_c),
                self.iterations,
                self.bailout as f64,
            )
        };
        let mut offset = [0.0f32; 2];
        let mut orbit = orbit_at(offset);
        // Candidate grid size: the search costs up to grid² orbits, so it is
        // reduced at high iteration counts and skipped at extreme ones
        // (pixels outliving the reference still continue correctly in f32).
        let grid: i32 = if self.iterations <= 2_048 {
            5
        } else if self.iterations <= 8_192 {
            3
        } else {
            1
        };
        if orbit.escape_iteration.is_some() && grid > 1 {
            // Try a grid across the view (local units: x spans ±aspect).
            let mut best_length = orbit.points.len();
            let half = grid / 2;
            let spacing = 0.8 / half as f32;
            for row in 0..grid {
                for column in 0..grid {
                    if row == half && column == half {
                        continue;
                    }
                    let candidate = [
                        (column - half) as f32 * spacing * aspect,
                        (row - half) as f32 * spacing,
                    ];
                    let candidate_orbit = orbit_at(candidate);
                    if candidate_orbit.points.len() > best_length {
                        best_length = candidate_orbit.points.len();
                        offset = candidate;
                        orbit = candidate_orbit;
                        if orbit.escape_iteration.is_none() {
                            break;
                        }
                    }
                }
                if orbit.escape_iteration.is_none() {
                    break;
                }
            }
        }
        self.deep_generation = self.deep_generation.wrapping_add(1).max(1);
        let data = Arc::new(DeepRenderData::from_f64_orbit(
            self.deep_generation,
            view.half_height,
            &orbit.points,
            true,
            offset,
        ));
        self.f64_references[pane] = Some(F64ReferenceCache {
            key,
            data: Arc::clone(&data),
        });
        Some(data)
    }

    /// Arbitrary-precision reference orbit for a deep view.
    ///
    /// Orbits are built incrementally under a per-frame time budget; while
    /// incomplete the GPU renders with the points available so far (pixels
    /// beyond the available length continue in f32) and the pane requests
    /// another repaint. While the view moves, the existing orbit is simply
    /// re-described relative to the new view (perturbation does not need a
    /// centred reference), so navigation is immediate; a fresh centred
    /// reference is rebuilt once input settles and swapped in when complete.
    fn deep_reference(
        &mut self,
        pane: usize,
        view: Option<&DeepView>,
        may_build: bool,
    ) -> Option<Arc<DeepRenderData>> {
        let view = view?;
        let julia_c = self.deep_julia_c.clone().unwrap_or_else(|| {
            DeepComplex::from_f64(self.julia_c, view.zoom_exponent).expect("finite Julia parameter")
        });
        let key = DeepReferenceKey::new(
            self.family,
            &self.family_parameters,
            pane,
            view,
            &julia_c,
            self.iterations,
            self.bailout,
        );
        let (scale_mantissa, scale_exponent) = view.half_height.scaled_f32();
        let ds_fallback =
            view.magnification_log10 <= ARBITRARY_HANDOFF_ZOOM.log10() + f64::EPSILON * 8.0;

        let same_view = self.deep_references[pane]
            .as_ref()
            .is_some_and(|cached| cached.key == key);
        let reusable = !same_view
            && self.deep_references[pane]
                .as_ref()
                .is_some_and(|cached| cached.key.same_dynamics(&key));
        if reusable {
            // Re-describe the existing orbit for the new view: offset of the
            // reference point from the new centre, in local units.
            let cached = self.deep_references[pane].as_ref().unwrap();
            let scale = view.half_height.with_zoom_exponent(view.zoom_exponent);
            let dx = cached
                .reference_centre
                .re
                .with_zoom_exponent(view.zoom_exponent)
                .sub(&view.centre.re)
                .div(&scale)
                .to_f64();
            let dy = cached
                .reference_centre
                .im
                .with_zoom_exponent(view.zoom_exponent)
                .sub(&view.centre.im)
                .div(&scale)
                .to_f64();
            let ratio = cached.reference_half_height.to_f64() / view.half_height.to_f64();
            let offset_ok = dx.is_finite()
                && dy.is_finite()
                && dx.abs() < 64.0
                && dy.abs() < 64.0
                && ratio.is_finite()
                && (1e-6..=1e6).contains(&ratio);
            if offset_ok {
                let redescribed = Arc::new(cached.data.redescribed(
                    scale_mantissa,
                    scale_exponent,
                    [dx as f32, dy as f32],
                    ds_fallback,
                ));
                if !may_build {
                    return Some(redescribed);
                }
                // Settled on a new view: keep serving the re-described orbit
                // while a fresh centred reference is built for this view.
                let pending_matches = self.deep_pending[pane]
                    .as_ref()
                    .is_some_and(|pending| pending.key == key);
                if !pending_matches {
                    let dynamical = self.pane_is_dynamical(pane);
                    let initial =
                        DeepState::initial(self.family, &view.centre, dynamical, &julia_c).ok()?;
                    let builder = ReferenceOrbitBuilder::new(
                        self.family,
                        &self.family_parameters,
                        initial,
                        self.iterations,
                        self.bailout as f64,
                    )
                    .ok()?;
                    self.deep_pending[pane] = Some(PendingDeepReference {
                        key,
                        centre: view.centre.clone(),
                        half_height: view.half_height.clone(),
                        ds_fallback,
                        scale_mantissa,
                        scale_exponent,
                        builder,
                    });
                }
                self.advance_pending_reference(pane);
                if self.deep_pending[pane].is_none() {
                    // The centred reference completed within budget.
                    let cached = self.deep_references[pane].as_ref()?;
                    return Some(Arc::clone(&cached.data));
                }
                return Some(redescribed);
            }
        }

        if !same_view {
            if !may_build {
                // Keep the last completed deep frame on screen while navigation
                // changes and a replacement reference orbit is not ready yet.
                if let Some(cached) = &self.deep_references[pane] {
                    return Some(Arc::clone(&cached.data));
                }
            }
            let dynamical = self.pane_is_dynamical(pane);
            let initial =
                DeepState::initial(self.family, &view.centre, dynamical, &julia_c).ok()?;
            let builder = ReferenceOrbitBuilder::new(
                self.family,
                &self.family_parameters,
                initial,
                self.iterations,
                self.bailout as f64,
            )
            .ok()?;
            self.deep_generation = self.deep_generation.wrapping_add(1).max(1);
            let data = Arc::new(DeepRenderData::from_points(
                self.deep_generation,
                scale_mantissa,
                scale_exponent,
                &builder.points,
                ds_fallback,
            ));
            self.deep_pending[pane] = None;
            self.deep_references[pane] = Some(DeepReferenceCache {
                key,
                reference_centre: view.centre.clone(),
                reference_half_height: view.half_height.clone(),
                data,
                builder: Some(builder),
                ds_fallback,
            });
        }
        // Extend the current orbit (and any pending replacement) under budget.
        self.advance_pending_reference(pane);
        let cached = self.deep_references[pane].as_mut()?;
        if let Some(builder) = cached.builder.as_mut() {
            let start = Instant::now();
            let mut complete = builder.is_complete();
            while !complete && start.elapsed() < DEEP_REFERENCE_FRAME_BUDGET {
                complete = builder.advance(64);
            }
            self.deep_generation = self.deep_generation.wrapping_add(1).max(1);
            cached.data = Arc::new(DeepRenderData::from_points(
                self.deep_generation,
                cached.data.scale_mantissa,
                cached.data.scale_exponent,
                &builder.points,
                cached.ds_fallback,
            ));
            if complete {
                cached.builder = None;
            } else {
                self.deep_reference_building[pane] = true;
            }
        }
        Some(Arc::clone(&cached.data))
    }

    /// Advances the pending centred reference for a pane under the frame
    /// budget and swaps it into the cache once complete.
    fn advance_pending_reference(&mut self, pane: usize) {
        let Some(pending) = self.deep_pending[pane].as_mut() else {
            return;
        };
        let start = Instant::now();
        let mut complete = pending.builder.is_complete();
        while !complete && start.elapsed() < DEEP_REFERENCE_FRAME_BUDGET {
            complete = pending.builder.advance(64);
        }
        if complete {
            let pending = self.deep_pending[pane].take().unwrap();
            self.deep_generation = self.deep_generation.wrapping_add(1).max(1);
            let data = Arc::new(DeepRenderData::from_points(
                self.deep_generation,
                pending.scale_mantissa,
                pending.scale_exponent,
                &pending.builder.points,
                pending.ds_fallback,
            ));
            self.deep_references[pane] = Some(DeepReferenceCache {
                key: pending.key,
                reference_centre: pending.centre,
                reference_half_height: pending.half_height,
                data,
                builder: None,
                ds_fallback: pending.ds_fallback,
            });
        } else {
            self.deep_reference_building[pane] = true;
        }
    }

    fn draw_focus_marker(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        view: &PlaneView,
        focus: [f64; 2],
    ) {
        let aspect = rect.width() as f64 / rect.height().max(1.0) as f64;
        let nx = (focus[0] - view.centre[0]) / (view.half_height * aspect);
        let ny = (focus[1] - view.centre[1]) / view.half_height;
        let position = egui::pos2(
            rect.center().x + nx as f32 * rect.width() * 0.5,
            rect.center().y - ny as f32 * rect.height() * 0.5,
        );
        if rect.contains(position) {
            ui.painter()
                .circle_stroke(position, 6.0, egui::Stroke::new(1.5, CREAM));
            ui.painter().circle_filled(position, 1.5, CREAM);
        }
    }

    fn draw_readout(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        pointer: Option<egui::Pos2>,
        view: &PlaneView,
        precision: PrecisionMode,
    ) {
        let Some(pointer) = pointer.filter(|position| rect.contains(*position)) else {
            return;
        };
        let sample = coordinate_sample(view, rect, pointer, ui.ctx().pixels_per_point(), precision);
        let text = format!(
            "NAV f64    {:+.15e}  {:+.15e}i\n\
             {:<10} {:+.15e}  {:+.15e}i\n\
             Δ GPU-NAV  {:+.3e}  {:+.3e}i\n\
             PIXEL Δz   {:.3e}",
            sample.navigation[0],
            sample.navigation[1],
            precision.label(),
            sample.rendered[0],
            sample.rendered[1],
            sample.delta[0],
            sample.delta[1],
            sample.world_per_pixel,
        );
        let galley = ui.painter().layout_no_wrap(
            text,
            egui::FontId::new(12.0, egui::FontFamily::Monospace),
            TEXT,
        );
        let box_rect = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + 8.0, rect.max.y - galley.size().y - 13.0),
            galley.size() + egui::vec2(12.0, 7.0),
        );
        ui.painter().rect_filled(
            box_rect,
            2.0,
            egui::Color32::from_rgba_unmultiplied(10, 13, 18, 220),
        );
        ui.painter()
            .galley(box_rect.min + egui::vec2(6.0, 3.0), galley, TEXT);
    }

    fn draw_deep_readout(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        pointer: Option<egui::Pos2>,
        view: &DeepView,
    ) {
        let mut sampled = view.clone();
        if let Some(pointer) = pointer.filter(|pointer| rect.contains(*pointer)) {
            let _ = sampled.recenter_local(local_coordinate(rect, pointer));
        }
        let text = format!(
            "NAV AP   {}  {}i\nSCALE    {}\nPRECISION {} decimal digits",
            sampled.centre.re.scientific(40),
            sampled.centre.im.scientific(40),
            view.half_height.scientific(8),
            view.half_height.precision(),
        );
        let font = egui::FontId::new(12.0, egui::FontFamily::Monospace);
        let galley =
            ui.painter()
                .layout_no_wrap(text, font, egui::Color32::from_rgb(218, 222, 230));
        let background = egui::Rect::from_min_size(
            egui::pos2(rect.min.x + 8.0, rect.max.y - galley.size().y - 12.0),
            galley.size() + egui::vec2(12.0, 8.0),
        );
        ui.painter().rect_filled(
            background,
            3.0,
            egui::Color32::from_rgba_unmultiplied(8, 11, 17, 232),
        );
        ui.painter().galley(
            background.min + egui::vec2(6.0, 4.0),
            galley,
            egui::Color32::WHITE,
        );
    }
}

fn complex_to_screen(view: &PlaneView, rect: egui::Rect, z: [f64; 2]) -> Option<egui::Pos2> {
    if !z[0].is_finite() || !z[1].is_finite() || view.half_height <= 0.0 {
        return None;
    }
    let aspect = rect.width() as f64 / rect.height().max(1.0) as f64;
    let nx = (z[0] - view.centre[0]) / (view.half_height * aspect);
    let ny = (z[1] - view.centre[1]) / view.half_height;
    if !nx.is_finite() || !ny.is_finite() || nx.abs() > 8.0 || ny.abs() > 8.0 {
        return None;
    }
    Some(egui::pos2(
        rect.center().x + nx as f32 * rect.width() * 0.5,
        rect.center().y - ny as f32 * rect.height() * 0.5,
    ))
}

fn draw_orbit_overlay(
    ui: &egui::Ui,
    rect: egui::Rect,
    view: &PlaneView,
    orbit: &CriticalOrbit,
    selected: usize,
) {
    if orbit.points.is_empty() {
        return;
    }
    let painter = ui.painter_at(rect);
    const ORBIT_TAIL_POINTS: usize = 9;
    let trail_rgb = [255, 74, 190];
    let trail_colour = egui::Color32::from_rgb(trail_rgb[0], trail_rgb[1], trail_rgb[2]);
    let trail_shadow = egui::Color32::from_rgba_unmultiplied(5, 8, 12, 210);
    let visible_range = orbit_tail_range(orbit.points.len(), selected, ORBIT_TAIL_POINTS);
    let visible_points = &orbit.points[visible_range];
    let fade_denominator = visible_points.len().saturating_sub(1).max(1) as f32;

    for (index, pair) in visible_points.windows(2).enumerate() {
        if let (Some(from), Some(to)) = (
            complex_to_screen(view, rect, pair[0].z),
            complex_to_screen(view, rect, pair[1].z),
        ) {
            let age = (index + 1) as f32 / fade_denominator;
            let alpha = (55.0 + 200.0 * age).round() as u8;
            let colour = egui::Color32::from_rgba_unmultiplied(
                trail_rgb[0],
                trail_rgb[1],
                trail_rgb[2],
                alpha,
            );
            painter.line_segment([from, to], egui::Stroke::new(4.0, trail_shadow));
            painter.line_segment([from, to], egui::Stroke::new(2.25, colour));
        }
    }

    for (index, point) in visible_points.iter().enumerate() {
        if let Some(position) = complex_to_screen(view, rect, point.z) {
            if rect.contains(position) {
                let age = index as f32 / fade_denominator;
                let alpha = (80.0 + 175.0 * age).round() as u8;
                let point_colour = egui::Color32::from_rgba_unmultiplied(
                    trail_rgb[0],
                    trail_rgb[1],
                    trail_rgb[2],
                    alpha,
                );
                let radius = if point.iteration as usize == selected {
                    3.5
                } else {
                    2.5
                };
                painter.circle_filled(position, radius + 1.5, trail_shadow);
                painter.circle_filled(position, radius, point_colour);
            }
        }
    }

    if let Some(escape) = orbit.escape_iteration {
        if escape as usize <= selected {
            if let Some(position) = complex_to_screen(view, rect, orbit.points[escape as usize].z) {
                if rect.contains(position) {
                    let offset = egui::vec2(5.0, 5.0);
                    painter.line_segment(
                        [position - offset, position + offset],
                        egui::Stroke::new(2.0, CORAL),
                    );
                    painter.line_segment(
                        [
                            position + egui::vec2(-offset.x, offset.y),
                            position + egui::vec2(offset.x, -offset.y),
                        ],
                        egui::Stroke::new(2.0, CORAL),
                    );
                }
            }
        }
    }

    if let Some(point) = orbit.points.get(selected) {
        if let Some(position) = complex_to_screen(view, rect, point.z) {
            if rect.contains(position) {
                painter.circle_filled(position, 4.0, CORAL);
                painter.circle_stroke(position, 9.0, egui::Stroke::new(2.5, CREAM));
                painter.text(
                    position + egui::vec2(12.0, -11.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("z_{}", point.iteration),
                    egui::FontId::new(11.0, egui::FontFamily::Monospace),
                    egui::Color32::WHITE,
                );
            }
        }
    }

    let badge = egui::Rect::from_min_size(rect.min + egui::vec2(8.0, 8.0), egui::vec2(112.0, 24.0));
    painter.rect_filled(
        badge,
        3.0,
        egui::Color32::from_rgba_unmultiplied(10, 13, 18, 220),
    );
    painter.rect_stroke(
        badge,
        3.0,
        egui::Stroke::new(1.0, trail_colour),
        egui::StrokeKind::Inside,
    );
    painter.text(
        badge.center(),
        egui::Align2::CENTER_CENTER,
        format!("ORBIT  z_{selected}"),
        egui::FontId::new(11.0, egui::FontFamily::Monospace),
        egui::Color32::WHITE,
    );
}

fn orbit_tail_range(
    point_count: usize,
    selected: usize,
    tail_points: usize,
) -> std::ops::Range<usize> {
    let end = selected.saturating_add(1).min(point_count);
    let start = end.saturating_sub(tail_points.max(1));
    start..end
}

fn show_orbit_inspector(
    ctx: &egui::Context,
    orbit: &CriticalOrbit,
    c: [f64; 2],
    selected_step: &mut usize,
    open: &mut bool,
) -> Option<[f64; 2]> {
    let last = orbit.last_iteration();
    *selected_step = (*selected_step).min(last);
    let mut centre_request = None;

    egui::Window::new("Critical orbit inspector")
        .open(open)
        .default_pos(egui::pos2(PANEL_WIDTH + 24.0, 52.0))
        .default_size(egui::vec2(680.0, 700.0))
        .max_width(760.0)
        .max_height(760.0)
        .resizable(true)
        .show(ctx, |ui| {
            ui.monospace(format!("c = {:+.17e} {:+.17e}i", c[0], c[1]));
            match orbit.escape_iteration {
                Some(iteration) => {
                    ui.label(
                        egui::RichText::new(format!("Escaped at iteration {iteration}"))
                            .color(CREAM)
                            .strong(),
                    );
                    if let Some(smooth) = orbit.smooth_escape_iteration {
                        ui.label(
                            egui::RichText::new(format!("Smooth escape iteration: {smooth:.12}"))
                                .monospace()
                                .color(TEXT),
                        );
                    }
                }
                None => {
                    ui.label(
                        egui::RichText::new(format!("No escape through iteration {last}"))
                            .color(BLUE)
                            .strong(),
                    );
                    ui.label(
                        egui::RichText::new(
                            "Finite iteration does not prove that the orbit is bounded.",
                        )
                        .small()
                        .color(MUTED),
                    );
                }
            }

            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("|<").on_hover_text("First point").clicked() {
                    *selected_step = 0;
                }
                if ui.button("<").on_hover_text("Previous point").clicked() {
                    *selected_step = selected_step.saturating_sub(1);
                }
                if ui.button(">").on_hover_text("Next point").clicked() {
                    *selected_step = (*selected_step + 1).min(last);
                }
                if ui
                    .button(">|")
                    .on_hover_text("Last computed point")
                    .clicked()
                {
                    *selected_step = last;
                }
                if let Some(escape) = orbit.escape_iteration {
                    if ui.button("Escape").clicked() {
                        *selected_step = escape as usize;
                    }
                }
            });
            ui.add(egui::Slider::new(selected_step, 0..=last).text("iteration n"));

            let point = orbit.points[*selected_step];
            if ui.button("Center Julia view on z_n").clicked() {
                centre_request = Some(point.z);
            }
            let derivative_magnitude =
                point.parameter_derivative[0].hypot(point.parameter_derivative[1]);
            egui::Grid::new("iterascope.orbit.point")
                .num_columns(2)
                .spacing([14.0, 4.0])
                .show(ui, |ui| {
                    ui.label("n");
                    ui.monospace(point.iteration.to_string());
                    ui.end_row();
                    ui.label("Re(z_n)");
                    ui.monospace(format!("{:+.17e}", point.z[0]));
                    ui.end_row();
                    ui.label("Im(z_n)");
                    ui.monospace(format!("{:+.17e}", point.z[1]));
                    ui.end_row();
                    ui.label("|z_n|");
                    ui.monospace(format!("{:.17e}", point.magnitude));
                    ui.end_row();
                    ui.label("arg(z_n)");
                    ui.monospace(format!("{:+.17e} rad", point.z[1].atan2(point.z[0])));
                    ui.end_row();
                    ui.label("Re(dz_n/dc)");
                    ui.monospace(format!("{:+.17e}", point.parameter_derivative[0]));
                    ui.end_row();
                    ui.label("Im(dz_n/dc)");
                    ui.monospace(format!("{:+.17e}", point.parameter_derivative[1]));
                    ui.end_row();
                    ui.label("|dz_n/dc|");
                    ui.monospace(format!("{:.17e}", derivative_magnitude));
                    ui.end_row();
                });

            ui.separator();
            ui.label(egui::RichText::new("Nearby orbit points").color(TEXT));
            let start = selected_step.saturating_sub(3);
            let end = (*selected_step + 4).min(orbit.points.len());
            egui::Grid::new("iterascope.orbit.nearby")
                .striped(true)
                .spacing([12.0, 3.0])
                .show(ui, |ui| {
                    ui.strong("n");
                    ui.strong("Re(z_n)");
                    ui.strong("Im(z_n)");
                    ui.strong("|z_n|");
                    ui.end_row();
                    for point in &orbit.points[start..end] {
                        ui.monospace(point.iteration.to_string());
                        ui.monospace(format!("{:+.9e}", point.z[0]));
                        ui.monospace(format!("{:+.9e}", point.z[1]));
                        ui.monospace(format!("{:.9e}", point.magnitude));
                        ui.end_row();
                    }
                });
        });
    centre_request
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let update_start = Instant::now();

        egui::Panel::top("iterascope.topbar")
            .exact_size(34.0)
            .frame(egui::Frame::new().fill(egui::Color32::from_rgb(12, 14, 18)))
            .show(ui, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("ITERASCOPE").strong().color(TEXT));
                    ui.label(egui::RichText::new("Complex dynamics laboratory").color(MUTED));
                    ui.add_space(10.0);
                    // View selector: either plane alone, full window, or
                    // both linked panes side by side.
                    let (left, right) = self.pane_labels();
                    if ui
                        .selectable_label(self.single_image && self.active_pane == 0, left)
                        .on_hover_text("This plane alone, full window")
                        .clicked()
                    {
                        self.single_image = true;
                        self.active_pane = 0;
                    }
                    if ui
                        .selectable_label(self.single_image && self.active_pane == 1, right)
                        .on_hover_text("This plane alone, full window")
                        .clicked()
                    {
                        self.single_image = true;
                        self.active_pane = 1;
                    }
                    if ui
                        .selectable_label(!self.single_image, "Both")
                        .on_hover_text("The linked panes side by side")
                        .clicked()
                    {
                        self.single_image = false;
                    }
                    if self.family.linkage() == Linkage::ParameterDynamical
                        && ui
                            .selectable_label(self.switch_picker_open, "Pick c…")
                            .on_hover_text(
                                "Choose the Julia parameter from the parameter plane, with a live preview",
                            )
                            .clicked()
                    {
                        // The picker draws through the parameter pane's GPU
                        // resources, so it lives in the single Julia view.
                        if self.switch_picker_open {
                            self.switch_picker_open = false;
                        } else {
                            self.single_image = true;
                            self.active_pane = 1;
                            self.switch_picker_open = true;
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new(format!("UI {:.1} ms", self.ui_update_ms))
                                .monospace()
                                .color(MUTED),
                        );
                        badge(ui, "ON DEMAND", BLUE);
                    });
                });
            });

        // Both control panes share the same default width and can be
        // resized by dragging their inner edge; collapsed they become an
        // exact slim strip.
        let left_open = self.left_panel_open;
        let mut left = egui::Panel::left("iterascope.controls");
        left = if left_open {
            left.resizable(true)
                .default_size(PANEL_WIDTH)
                .size_range(PANEL_WIDTH..=520.0)
        } else {
            left.exact_size(COLLAPSED_PANEL_WIDTH)
        };
        left.frame(egui::Frame::new().fill(PANEL).inner_margin(if left_open {
            egui::Margin::symmetric(12, 10)
        } else {
            egui::Margin::symmetric(2, 10)
        }))
        .show(ui, |ui| {
            if panel_header(ui, "INSTRUMENT", true, &mut self.left_panel_open) {
                self.instrument_controls(ui);
            }
        });

        let right_open = self.right_panel_open;
        let mut right = egui::Panel::right("iterascope.studio");
        right = if right_open {
            right
                .resizable(true)
                .default_size(PANEL_WIDTH)
                .size_range(PANEL_WIDTH..=520.0)
        } else {
            right.exact_size(COLLAPSED_PANEL_WIDTH)
        };
        right
            .frame(egui::Frame::new().fill(PANEL).inner_margin(if right_open {
                egui::Margin::symmetric(12, 10)
            } else {
                egui::Margin::symmetric(2, 10)
            }))
            .show(ui, |ui| {
                if panel_header(ui, "STUDIO", false, &mut self.right_panel_open) {
                    self.studio_controls(ui);
                }
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BG))
            .show(ui, |ui| self.workspace(ui));

        // The current stage has now produced its paint callback. Advance the
        // authoritative view afterwards so every stage is presented for at
        // least one frame before its successor is prepared.
        if let Some(delay) = self.advance_progressive_julia_zoom() {
            ui.ctx().request_repaint_after(delay);
        }

        self.experiment_editor(ui.ctx());
        self.orbit_inspector(ui.ctx());
        self.gradient_editor(ui.ctx());
        self.switch_picker(ui.ctx());
        #[cfg(not(target_arch = "wasm32"))]
        self.advance_export(ui.ctx());
        #[cfg(not(target_arch = "wasm32"))]
        self.advance_still(ui.ctx());

        let elapsed = (Instant::now() - update_start).as_secs_f32() * 1000.0;
        self.ui_update_ms = if self.ui_update_ms == 0.0 {
            elapsed
        } else {
            self.ui_update_ms + (elapsed - self.ui_update_ms) * 0.2
        };
    }

    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct ExportJob {
    animation: ZoomAnimation,
    scene: FrozenScene,
    directory: std::path::PathBuf,
    next_frame: usize,
    started: Instant,
}

/// Everything an offline render needs, frozen at the moment the user asked
/// for it: the dynamics, the layer stack, and the reference orbit of the
/// fixed centre — re-described in scale for any magnification a frame asks
/// for.
#[cfg(not(target_arch = "wasm32"))]
struct FrozenScene {
    family: FractalFamily,
    dynamical: bool,
    family_words: [f32; 8],
    iterations: u32,
    bailout: f32,
    julia_c: [f64; 2],
    /// `f64` projection of the (possibly arbitrary-precision) centre; the
    /// perturbation scale carries the depth.
    centre: [f64; 2],
    layers: LayerStack,
    gradient: Arc<GradientTable>,
    /// Arbitrary-precision reference for frames beyond the handoff.
    ap_reference: Option<(u64, Arc<Vec<GpuReferencePoint>>)>,
    /// The same orbit projected to `f64` for frames below the handoff.
    f64_reference: Option<(u64, Arc<Vec<GpuReferencePoint>>)>,
}

#[cfg(not(target_arch = "wasm32"))]
impl FrozenScene {
    /// Renders one frame of this scene at `magnification` through the
    /// export pipeline. Reference selection mirrors the interactive
    /// renderer: plain f32 for families without perturbation, the f64
    /// reference below the handoff, the arbitrary-precision reference
    /// beyond it. `sweep` is added to every layer's gradient offsets.
    fn render_frame(
        &self,
        render_state: &eframe::egui_wgpu::RenderState,
        magnification: f64,
        size: (u32, u32),
        sweep: f32,
    ) -> Result<Vec<u8>, String> {
        self.render_region(render_state, magnification, size, (0, 0), size, sweep)
    }

    /// Renders one rectangular region of a frame — the whole frame is the
    /// trivial region. Tiles render around the same reference orbit as the
    /// frame (the frame centre moves to the region's `reference_offset`),
    /// so tiling is exact at any magnification.
    #[allow(clippy::too_many_arguments)]
    fn render_region(
        &self,
        render_state: &eframe::egui_wgpu::RenderState,
        magnification: f64,
        full_size: (u32, u32),
        origin: (u32, u32),
        region_size: (u32, u32),
        sweep: f32,
    ) -> Result<Vec<u8>, String> {
        let handoff_log = ARBITRARY_HANDOFF_ZOOM.log10();
        let region = animation::region_view(magnification, full_size, origin, region_size);
        let reference = if magnification > handoff_log {
            self.ap_reference
                .as_ref()
                .map(|r| (r.0, r.1.as_slice(), false))
        } else {
            self.f64_reference
                .as_ref()
                .map(|r| (r.0, r.1.as_slice(), true))
                .or_else(|| {
                    self.ap_reference
                        .as_ref()
                        .map(|r| (r.0, r.1.as_slice(), false))
                })
        };
        let mut uniforms = Uniforms::new(
            [
                self.centre[0] + region.centre_shift[0],
                self.centre[1] + region.centre_shift[1],
            ],
            region.half_height_f64,
            region.aspect,
            self.julia_c,
            self.iterations,
            self.bailout,
            self.family.shader_flag(),
            usize::from(self.dynamical),
            self.layers.layers[0].colouring.outside.smooth,
            false,
            if reference.is_some() {
                PrecisionMode::DoubleSingle
            } else {
                PrecisionMode::F32
            },
            self.family_words,
        );
        if let Some((_, points, ds_fallback)) = reference {
            uniforms = uniforms.enable_perturbation(
                region.scale_mantissa,
                region.scale_exponent,
                points.len(),
                ds_fallback,
                region.reference_offset,
            );
        }
        let mut layers = self.layers.clone();
        if sweep != 0.0 {
            for layer in &mut layers.layers {
                layer.colouring.outside.offset += sweep;
                layer.colouring.inside.offset += sweep;
            }
        }
        let colouring_uniforms = ColouringUniforms::new(
            &layers,
            animation::frame_pixel_log(magnification, full_size.1),
        );
        let renderer = render_state.renderer.read();
        let pipeline = renderer
            .callback_resources
            .get::<FractalPipeline>()
            .ok_or_else(|| "renderer not initialised".to_owned())?;
        Ok(pipeline.render_export(
            &render_state.device,
            &render_state.queue,
            &uniforms,
            &colouring_uniforms,
            &self.gradient,
            reference.map(|(generation, points, _)| (generation, points)),
            region_size,
        ))
    }
}

/// A still render in progress: tiles of the supersampled frame render one
/// per UI update, are box-filtered down in linear light, and land in the
/// final image buffer.
#[cfg(not(target_arch = "wasm32"))]
struct StillJob {
    scene: FrozenScene,
    magnification: f64,
    width: u32,
    height: u32,
    supersample: u32,
    /// (offset, size) spans of the final image, per axis.
    columns: Vec<(u32, u32)>,
    rows: Vec<(u32, u32)>,
    next_tile: usize,
    /// Final-resolution RGBA, filled tile by tile.
    image: Vec<u8>,
    path: std::path::PathBuf,
    started: Instant,
}

#[cfg(not(target_arch = "wasm32"))]
impl StillJob {
    fn tile_count(&self) -> usize {
        self.columns.len() * self.rows.len()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn default_export_directory() -> String {
    std::env::var("HOME")
        .map(|home| format!("{home}/iterascope-exports"))
        .unwrap_or_else(|_| "iterascope-exports".to_owned())
}

#[cfg(target_arch = "wasm32")]
fn default_export_directory() -> String {
    String::new()
}

/// Box-filters `factor`×`factor` blocks of RGBA sRGB pixels down to one, in
/// linear light so anti-aliased edges keep their brightness. `width` is the
/// supersampled row width; the output is `width/factor` wide.
#[cfg(not(target_arch = "wasm32"))]
fn downsample_srgb(rgba: &[u8], width: u32, factor: u32) -> Vec<u8> {
    if factor <= 1 {
        return rgba.to_vec();
    }
    // sRGB → linear lookup for every byte value.
    let to_linear: Vec<f32> = (0..256)
        .map(|value| {
            let v = value as f32 / 255.0;
            if v <= 0.04045 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    let from_linear = |linear: f32| -> u8 {
        let v = if linear <= 0.003_130_8 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    let width = width as usize;
    let factor = factor as usize;
    let out_width = width / factor;
    let height = rgba.len() / (width * 4);
    let out_height = height / factor;
    let samples = (factor * factor) as f32;
    let mut out = Vec::with_capacity(out_width * out_height * 4);
    for row in 0..out_height {
        for column in 0..out_width {
            let mut sum = [0.0f32; 3];
            for dy in 0..factor {
                let base = ((row * factor + dy) * width + column * factor) * 4;
                for dx in 0..factor {
                    let pixel = &rgba[base + dx * 4..base + dx * 4 + 3];
                    sum[0] += to_linear[pixel[0] as usize];
                    sum[1] += to_linear[pixel[1] as usize];
                    sum[2] += to_linear[pixel[2] as usize];
                }
            }
            out.push(from_linear(sum[0] / samples));
            out.push(from_linear(sum[1] / samples));
            out.push(from_linear(sum[2] / samples));
            out.push(255);
        }
    }
    out
}

#[cfg(not(target_arch = "wasm32"))]
fn write_png(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|error| error.to_string())?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .and_then(|mut writer| writer.write_image_data(rgba))
        .map_err(|error| error.to_string())
}

/// Encodes `frame-%05d.png` in `directory` into `zoom.mp4` with ffmpeg.
#[cfg(not(target_arch = "wasm32"))]
fn encode_video(directory: &std::path::Path, fps: u32) -> Result<String, String> {
    // A GUI launch may not inherit the shell's PATH; try the common
    // package-manager locations after the bare name.
    let ffmpeg = [
        "ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    .into_iter()
    .find(|candidate| {
        std::process::Command::new(candidate)
            .arg("-version")
            .output()
            .is_ok_and(|probe| probe.status.success())
    })
    .ok_or_else(|| "ffmpeg not found".to_owned())?;
    let output = std::process::Command::new(ffmpeg)
        .args([
            "-y",
            "-framerate",
            &fps.to_string(),
            "-i",
            "frame-%05d.png",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "zoom.mp4",
        ])
        .current_dir(directory)
        .output()
        .map_err(|error| format!("ffmpeg failed to start ({error})"))?;
    if output.status.success() {
        Ok("zoom.mp4".to_owned())
    } else {
        Err(format!(
            "ffmpeg failed: {}",
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .last()
                .unwrap_or("unknown error")
        ))
    }
}

fn wrap_turns(value: f32) -> f32 {
    let wrapped = value - value.floor();
    if wrapped >= 1.0 || !wrapped.is_finite() {
        0.0
    } else {
        wrapped
    }
}

/// Paints a gradient into `rect` as `columns` vertical strips.
fn paint_gradient(painter: &egui::Painter, rect: egui::Rect, gradient: &Gradient, columns: usize) {
    let mut mesh = egui::Mesh::default();
    let width = rect.width() / columns as f32;
    for column in 0..columns {
        let t = (column as f32 + 0.5) / columns as f32;
        let c = gradient.colour_at(t);
        let colour = egui::Color32::from_rgb(
            (c[0] * 255.0).round() as u8,
            (c[1] * 255.0).round() as u8,
            (c[2] * 255.0).round() as u8,
        );
        let strip = egui::Rect::from_min_max(
            egui::pos2(rect.min.x + column as f32 * width, rect.min.y),
            egui::pos2(rect.min.x + (column as f32 + 1.0) * width + 0.5, rect.max.y),
        );
        mesh.add_colored_rect(strip, colour);
    }
    painter.add(egui::Shape::mesh(mesh));
}

fn colouring_side_controls(
    ui: &mut egui::Ui,
    side: &mut ColouringSide,
    family: FractalFamily,
    outside: bool,
) {
    let salt = if outside { "outside" } else { "inside" };
    let mut algorithm = side.algorithm;
    egui::ComboBox::from_id_salt(("iterascope.colouring.algorithm", salt))
        .selected_text(algorithm.name())
        .width(230.0)
        .show_ui(ui, |ui| {
            for candidate in ColouringAlgorithm::ALL {
                ui.selectable_value(&mut algorithm, candidate, candidate.name())
                    .on_hover_text(candidate.description());
            }
        });
    side.set_algorithm(algorithm);
    ui.label(
        egui::RichText::new(algorithm.description())
            .small()
            .color(MUTED),
    );

    match algorithm {
        ColouringAlgorithm::Solid => {
            let mut rgb = [
                (side.solid[0] * 255.0).round() as u8,
                (side.solid[1] * 255.0).round() as u8,
                (side.solid[2] * 255.0).round() as u8,
            ];
            ui.horizontal(|ui| {
                ui.label("Colour");
                if egui::color_picker::color_edit_button_srgb(ui, &mut rgb).changed() {
                    side.solid = [
                        rgb[0] as f32 / 255.0,
                        rgb[1] as f32 / 255.0,
                        rgb[2] as f32 / 255.0,
                    ];
                }
            });
            return;
        }
        ColouringAlgorithm::Iteration => {
            if outside {
                ui.checkbox(
                    &mut side.smooth,
                    if family.converges() {
                        "Smooth convergence time"
                    } else {
                        "Smooth escape time"
                    },
                );
            }
        }
        ColouringAlgorithm::Decomposition => {
            ui.add(
                egui::Slider::new(&mut side.sectors, 0..=64)
                    .text("sectors")
                    .clamping(egui::SliderClamping::Never),
            );
        }
        ColouringAlgorithm::Stripes => {
            ui.add(egui::Slider::new(&mut side.stripe_frequency, 0.5..=32.0).text("frequency"));
        }
        ColouringAlgorithm::TriangleInequality => {
            if outside && family.uses_bailout() {
                ui.label(
                    egui::RichText::new("Raise the bailout (Computation) for smoother results.")
                        .small()
                        .color(MUTED),
                );
            }
        }
        ColouringAlgorithm::DistanceEstimate => {
            if !family.has_distance_estimate() {
                ui.label(
                    egui::RichText::new(format!(
                        "{} has no derivative in the shader; falling back to the iteration count.",
                        family.name()
                    ))
                    .small()
                    .color(CORAL),
                );
            }
        }
        ColouringAlgorithm::OrbitTrap => {
            egui::ComboBox::from_id_salt(("iterascope.colouring.trap", salt))
                .selected_text(side.trap_shape.name())
                .width(150.0)
                .show_ui(ui, |ui| {
                    for shape in TrapShape::ALL {
                        ui.selectable_value(&mut side.trap_shape, shape, shape.name());
                    }
                });
            ui.horizontal(|ui| {
                ui.label("centre");
                ui.add(
                    egui::DragValue::new(&mut side.trap_centre[0])
                        .speed(0.01)
                        .fixed_decimals(3),
                );
                ui.add(
                    egui::DragValue::new(&mut side.trap_centre[1])
                        .speed(0.01)
                        .fixed_decimals(3),
                );
            });
            if side.trap_shape.uses_size() {
                ui.add(egui::Slider::new(&mut side.trap_size, 0.0..=4.0).text("size"));
            }
        }
    }

    ui.add(
        egui::Slider::new(&mut side.density, 0.001..=100.0)
            .logarithmic(true)
            .text("density"),
    );
    ui.add(egui::Slider::new(&mut side.offset, 0.0..=1.0).text("offset"));
    egui::ComboBox::from_id_salt(("iterascope.colouring.transfer", salt))
        .selected_text(side.transfer.name())
        .width(150.0)
        .show_ui(ui, |ui| {
            for transfer in Transfer::ALL {
                ui.selectable_value(&mut side.transfer, transfer, transfer.name());
            }
        });
    if outside && algorithm != ColouringAlgorithm::Iteration {
        ui.add(egui::Slider::new(&mut side.shading, 0.0..=1.0).text("iteration shading"))
            .on_hover_text("Darken slowly escaping or converging pixels");
    }
}

fn configure_style(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL;
    visuals.window_fill = PANEL;
    visuals.extreme_bg_color = PANEL_DEEP;
    visuals.faint_bg_color = egui::Color32::from_rgb(28, 31, 36);
    visuals.widgets.noninteractive.fg_stroke.color = TEXT;
    visuals.widgets.inactive.fg_stroke.color = TEXT;
    visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(48, 51, 57);
    visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(61, 65, 72);
    visuals.widgets.active.bg_fill = egui::Color32::from_rgb(72, 77, 84);
    visuals.selection.bg_fill = BLUE.gamma_multiply(0.55);
    visuals.hyperlink_color = BLUE;
    ctx.set_visuals(visuals);

    ctx.all_styles_mut(|style| {
        style.spacing.slider_width = 126.0;
        style.spacing.item_spacing = egui::vec2(7.0, 6.0);
        for font in style.text_styles.values_mut() {
            font.size = (font.size * 1.12).round();
        }
    });
}

const COLLAPSED_PANEL_WIDTH: f32 = 22.0;

/// Header row of a side pane: the title and a chevron that collapses the
/// pane to a slim strip. Returns whether the pane's contents should render.
fn panel_header(ui: &mut egui::Ui, title: &str, left: bool, open: &mut bool) -> bool {
    if *open {
        ui.horizontal(|ui| {
            let chevron = if left { "◀" } else { "▶" };
            let collapse = |ui: &mut egui::Ui, open: &mut bool| {
                if ui
                    .add(egui::Button::new(chevron).small().frame(false))
                    .on_hover_text(format!("Collapse the {} pane", title.to_lowercase()))
                    .clicked()
                {
                    *open = false;
                }
            };
            if !left {
                collapse(ui, open);
            }
            ui.label(egui::RichText::new(title).small().strong().color(MUTED));
            if left {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    collapse(ui, open);
                });
            }
        });
        ui.add_space(2.0);
        *open
    } else {
        let chevron = if left { "▶" } else { "◀" };
        ui.vertical_centered(|ui| {
            if ui
                .add(egui::Button::new(chevron).small().frame(false))
                .on_hover_text(format!("Expand the {} pane", title.to_lowercase()))
                .clicked()
            {
                *open = true;
            }
        });
        false
    }
}

fn section(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(egui::RichText::new(title).color(TEXT).strong())
        .default_open(true)
        .show(ui, |ui| {
            ui.add_space(2.0);
            body(ui);
            ui.add_space(4.0);
        });
    ui.separator();
}

fn coordinate_row(ui: &mut egui::Ui, label: &str, value: &mut f64) -> bool {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).monospace().color(MUTED));
        ui.add(
            egui::DragValue::new(value)
                .speed(0.0001)
                .range(-8.0..=8.0)
                .max_decimals(12),
        )
        .changed()
    })
    .inner
}

fn zoom_row(ui: &mut egui::Ui, label: &str, magnification: String) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(format!("×{magnification}"))
                    .monospace()
                    .color(CREAM),
            );
        });
    });
}

fn diagnostic_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.monospace(value);
        });
    });
}

fn precision_row(
    ui: &mut egui::Ui,
    label: &str,
    precision: PrecisionMode,
    validity: DsValidity,
    deep_active: bool,
    f64_reference: bool,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (text, colour) = if deep_active && f64_reference {
                ("F64 PERT", BLUE)
            } else if deep_active {
                ("AP PERT", BLUE)
            } else {
                match precision {
                    PrecisionMode::F32 => (precision.label(), CREAM),
                    PrecisionMode::DoubleSingle => (
                        validity.label(),
                        match validity.level {
                            ValidityLevel::Stable => BLUE,
                            ValidityLevel::Risk => CREAM,
                            ValidityLevel::Limit => CORAL,
                        },
                    ),
                }
            };
            badge(ui, text, colour);
        });
    });
}

fn probe_rows(ui: &mut egui::Ui, label: &str, result: Option<ProbeResult>, validity: DsValidity) {
    path_probe_row(ui, &format!("{label} f32"), result.map(|value| value.f32));
    path_probe_row(ui, &format!("{label} DS"), result.map(|value| value.ds));
    ui.label(
        egui::RichText::new(validity.summary())
            .small()
            .color(match validity.level {
                ValidityLevel::Stable => MUTED,
                ValidityLevel::Risk => CREAM,
                ValidityLevel::Limit => CORAL,
            }),
    );
}

fn path_probe_row(ui: &mut egui::Ui, label: &str, result: Option<PathProbeResult>) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (text, colour) = match result {
                Some(result) if result.unstable() => (result.summary(), CORAL),
                Some(result) => (result.summary(), MUTED),
                None => ("waiting for settled view".to_owned(), MUTED),
            };
            ui.label(egui::RichText::new(text).small().color(colour));
        });
    });
}

fn f32_spacing(value: f64) -> f64 {
    let value = (value as f32).abs();
    if value == 0.0 {
        return f32::from_bits(1) as f64;
    }
    (f32::from_bits(value.to_bits() + 1) - value) as f64
}

fn pinch_zoom_factor(zoom_delta: f32) -> f64 {
    // Spreading fingers reports >1 and should reduce half-height (zoom in).
    // Clamp a single frame so an OS gesture spike cannot discard the view.
    (zoom_delta as f64).clamp(0.5, 2.0).powf(-PINCH_SENSITIVITY)
}

fn accelerate_zoom_factor(factor: f64, accelerated: bool) -> f64 {
    if !accelerated {
        return factor;
    }
    let maximum_log = MAX_ACCELERATED_DECADES_PER_EVENT * std::f64::consts::LN_10;
    (factor.ln() * ACCELERATED_ZOOM_POWER)
        .clamp(-maximum_log, maximum_log)
        .exp()
}

fn progressive_zoom_factor(current_log10: f64, target_log10: f64) -> Option<f64> {
    let remaining = target_log10 - current_log10;
    if remaining.abs() < 1e-9 {
        return None;
    }
    let stage = remaining.clamp(
        -PROGRESSIVE_ZOOM_DECADES_PER_STAGE,
        PROGRESSIVE_ZOOM_DECADES_PER_STAGE,
    );
    Some(10.0_f64.powf(-stage))
}

fn local_coordinate(rect: egui::Rect, position: egui::Pos2) -> [f64; 2] {
    let half_height = (rect.height() * 0.5).max(0.5);
    [
        ((position.x - rect.center().x) / half_height) as f64,
        ((rect.center().y - position.y) / half_height) as f64,
    ]
}

fn deep_plane_document(view: &DeepView) -> DeepPlaneDocument {
    DeepPlaneDocument {
        centre: DeepComplexDocument {
            re: view.centre.re.exact_decimal(),
            im: view.centre.im.exact_decimal(),
        },
        half_height: view.half_height.exact_decimal(),
        magnification_log10: view.magnification_log10,
    }
}

#[derive(Clone, Copy, Debug)]
struct CoordinateSample {
    navigation: [f64; 2],
    rendered: [f64; 2],
    delta: [f64; 2],
    world_per_pixel: f64,
}

fn coordinate_sample(
    view: &PlaneView,
    rect: egui::Rect,
    pointer: egui::Pos2,
    pixels_per_point: f32,
    precision: PrecisionMode,
) -> CoordinateSample {
    let ppp = pixels_per_point.max(1e-6) as f64;
    let width = (rect.width() as f64 * ppp).round().max(1.0);
    let height = (rect.height() as f64 * ppp).round().max(1.0);
    // Report the fragment that is actually shaded, not an infinitely precise
    // point between pixels. Pixel centres are half-integer physical positions.
    let pixel_x = (((pointer.x - rect.min.x) as f64 * ppp).floor() + 0.5).clamp(0.5, width - 0.5);
    let pixel_y = (((pointer.y - rect.min.y) as f64 * ppp).floor() + 0.5).clamp(0.5, height - 0.5);
    let local = [
        (2.0 * pixel_x - width) / height,
        1.0 - 2.0 * pixel_y / height,
    ];

    let navigation = [
        view.centre[0] + local[0] * view.half_height,
        view.centre[1] + local[1] * view.half_height,
    ];
    // Mirror the selected WGSL expression after the interpolant has been
    // rounded to f32. Back-convert only for display and delta calculation.
    let rendered = match precision {
        PrecisionMode::F32 => {
            let scale = view.half_height as f32;
            [
                (view.centre[0] as f32 + local[0] as f32 * scale) as f64,
                (view.centre[1] as f32 + local[1] as f32 * scale) as f64,
            ]
        }
        PrecisionMode::DoubleSingle => {
            let scale = DoubleSingle::from_f64(view.half_height);
            [
                DoubleSingle::from_f64(view.centre[0])
                    .add(DoubleSingle::from_f32(local[0] as f32).mul(scale))
                    .as_f64(),
                DoubleSingle::from_f64(view.centre[1])
                    .add(DoubleSingle::from_f32(local[1] as f32).mul(scale))
                    .as_f64(),
            ]
        }
    };
    let delta = [rendered[0] - navigation[0], rendered[1] - navigation[1]];

    CoordinateSample {
        navigation,
        rendered,
        delta,
        world_per_pixel: 2.0 * view.half_height / height,
    }
}

fn view_is_f32_limited(
    view: &PlaneView,
    dynamics_parameter: Option<[f64; 2]>,
    rect: egui::Rect,
    pixels_per_point: f32,
) -> bool {
    let physical_height = (rect.height() * pixels_per_point).max(1.0) as f64;
    let world_per_pixel = 2.0 * view.half_height / physical_height;
    let mut arithmetic_spacing = f32_spacing(view.centre[0]).max(f32_spacing(view.centre[1]));
    if let Some(c) = dynamics_parameter {
        // In a Julia iteration, small spatial differences eventually meet the
        // much larger fixed c in z²+c. Include c's spacing so a view centred
        // near zero does not incorrectly claim unlimited useful precision.
        arithmetic_spacing = arithmetic_spacing
            .max(f32_spacing(c[0]))
            .max(f32_spacing(c[1]));
    }
    // Warn before adjacent pixels fully collapse onto the same f32 value.
    arithmetic_spacing >= world_per_pixel * 0.5
}

fn julia_critical_roundoff_risk(
    view: &PlaneView,
    c: [f64; 2],
    rect: egui::Rect,
    pixels_per_point: f32,
) -> bool {
    let aspect = rect.width() as f64 / rect.height().max(1.0) as f64;
    let critical_point_visible = view.centre[0].abs() <= view.half_height * aspect
        && view.centre[1].abs() <= view.half_height;
    if !critical_point_visible {
        return false;
    }

    // Near z = 0, f32 can round z² out of z²+c and collapse a neighborhood
    // onto the exactly bounded critical orbit. Promote before that false basin
    // reaches the size of a displayed pixel.
    let c_spacing = f32_spacing(c[0]).max(f32_spacing(c[1]));
    let collapse_radius = c_spacing.sqrt();
    let physical_height = (rect.height() * pixels_per_point).max(1.0) as f64;
    let world_per_pixel = 2.0 * view.half_height / physical_height;
    collapse_radius >= world_per_pixel * 0.5
}

fn ds_coordinate_ratio(
    view: &PlaneView,
    dynamics_parameter: Option<[f64; 2]>,
    rect: egui::Rect,
    pixels_per_point: f32,
) -> f64 {
    let physical_height = (rect.height() * pixels_per_point).max(1.0) as f64;
    let world_per_pixel = 2.0 * view.half_height / physical_height;
    let mut spacing = ds_spacing(view.centre[0]).max(ds_spacing(view.centre[1]));
    if let Some(c) = dynamics_parameter {
        spacing = spacing.max(ds_spacing(c[0])).max(ds_spacing(c[1]));
    }
    spacing / world_per_pixel.max(f64::MIN_POSITIVE)
}

fn ds_spacing(value: f64) -> f64 {
    // A normalized double-single carries about one additional f32 mantissa
    // beneath the high word. This conservative estimate intentionally warns
    // before the theoretical best case.
    f32_spacing(value) * 2.0_f64.powi(-24)
}

fn choose_precision(
    magnification: f64,
    coordinate_limited: bool,
    probe: ProbeResult,
) -> PrecisionMode {
    let classification_failed = probe.f32.classification_mismatches > 0;
    let orbit_failed = magnification >= PROBE_DS_MIN_ZOOM && probe.f32.unstable();
    if coordinate_limited || classification_failed || orbit_failed {
        PrecisionMode::DoubleSingle
    } else {
        PrecisionMode::F32
    }
}

fn preset_button(ui: &mut egui::Ui, label: &str, c: [f64; 2], selected: &mut [f64; 2]) -> bool {
    if ui.small_button(label).clicked() {
        *selected = c;
        true
    } else {
        false
    }
}

fn badge(ui: &mut egui::Ui, text: &str, colour: egui::Color32) {
    egui::Frame::new()
        .fill(colour.gamma_multiply(0.23))
        .stroke(egui::Stroke::new(1.0, colour.gamma_multiply(0.7)))
        .corner_radius(3.0)
        .inner_margin(egui::Margin::symmetric(6, 2))
        .show(ui, |ui| {
            ui.label(egui::RichText::new(text).monospace().small().color(colour));
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn downsampling_averages_in_linear_light_and_keeps_flat_colours() {
        // A flat colour must survive exactly.
        let flat: Vec<u8> = std::iter::repeat_n([10u8, 128, 240, 255], 16)
            .flatten()
            .collect();
        let out = downsample_srgb(&flat, 4, 2);
        assert_eq!(out.len(), 2 * 2 * 4);
        assert!(out.chunks(4).all(|pixel| pixel == [10, 128, 240, 255]));
        // A black/white checkerboard averages to sRGB mid-grey (~188), not
        // the naive byte average 128 — the linear-light hallmark.
        let mut board = Vec::new();
        for row in 0..2 {
            for column in 0..2 {
                let value = if (row + column) % 2 == 0 { 255u8 } else { 0 };
                board.extend_from_slice(&[value, value, value, 255]);
            }
        }
        let out = downsample_srgb(&board, 2, 2);
        assert_eq!(out.len(), 4);
        assert!(
            (out[0] as i32 - 188).abs() <= 1,
            "linear-light average was {}",
            out[0]
        );
        // Factor 1 is the identity.
        assert_eq!(downsample_srgb(&flat, 4, 1), flat);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn png_writer_produces_a_decodable_file() {
        let directory = std::env::temp_dir().join("iterascope-png-test");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("frame-00000.png");
        let rgba: Vec<u8> = (0..4 * 4 * 4).map(|i| (i * 7 % 256) as u8).collect();
        write_png(&path, 4, 4, &rgba).unwrap();
        let decoder =
            png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let mut reader = decoder.read_info().unwrap();
        let mut buffer = vec![0; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buffer).unwrap();
        assert_eq!((info.width, info.height), (4, 4));
        assert_eq!(&buffer[..rgba.len()], &rgba[..]);
        std::fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn click_target_can_recentre_and_zoom_two_times() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let pointer = egui::pos2(625.0, 172.0);
        let mut view = PlaneView::new([-0.5, 0.0], 1.45);
        let selected = view.point_at(rect, pointer);
        view.zoom_from(Some(selected), 0.5);
        assert_eq!(view.centre, selected);
        assert!((view.half_height - 0.725).abs() < 1e-12);
    }

    #[test]
    fn zoom_preserves_the_chosen_centre() {
        let mut view = PlaneView::new([-0.745, 0.113], 1.45);
        let centre = view.centre;
        view.zoom_from(None, 0.4);
        assert_eq!(view.centre, centre);
        assert!((view.half_height - 0.58).abs() < 1e-12);
    }

    #[test]
    fn fine_pan_uses_tenths_of_the_displayed_range() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let mut view = PlaneView::new([-0.228_155_5, 1.115_143], 1.45e-8);
        let original = view.centre;
        let tenth_width = 2.0 * view.half_height * (800.0 / 600.0) * 0.1;
        let tenth_height = 2.0 * view.half_height * 0.1;

        view.pan_tenth(rect, [1.0, -1.0]);

        assert!((view.centre[0] - (original[0] + tenth_width)).abs() < 1e-18);
        assert!((view.centre[1] - (original[1] - tenth_height)).abs() < 1e-18);
    }

    #[test]
    fn orbit_projection_matches_the_dynamical_view() {
        let rect = egui::Rect::from_min_size(egui::pos2(10.0, 20.0), egui::vec2(800.0, 600.0));
        let view = PlaneView::new([0.25, -0.5], 1.5);
        let aspect = 800.0 / 600.0;

        assert_eq!(
            complex_to_screen(&view, rect, view.centre),
            Some(rect.center())
        );
        assert_eq!(
            complex_to_screen(
                &view,
                rect,
                [view.centre[0] + view.half_height * aspect, view.centre[1]],
            ),
            Some(egui::pos2(rect.right(), rect.center().y)),
        );
        assert_eq!(
            complex_to_screen(
                &view,
                rect,
                [view.centre[0], view.centre[1] + view.half_height],
            ),
            Some(egui::pos2(rect.center().x, rect.top())),
        );
    }

    #[test]
    fn orbit_overlay_keeps_only_the_recent_tail() {
        assert_eq!(orbit_tail_range(129, 0, 9), 0..1);
        assert_eq!(orbit_tail_range(129, 7, 9), 0..8);
        assert_eq!(orbit_tail_range(129, 27, 9), 19..28);
        assert_eq!(orbit_tail_range(5, usize::MAX, 9), 0..5);
    }

    #[test]
    fn first_zoom_uses_the_selected_focus() {
        let mut view = PlaneView::new([-0.5, 0.0], 1.45);
        let focus = [-0.745, 0.113];
        view.zoom_from(Some(focus), 0.5);
        assert_eq!(view.centre, focus);
        assert!((view.half_height - 0.725).abs() < 1e-12);
    }

    #[test]
    fn f32_limit_warning_tracks_world_units_per_pixel() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let ordinary = PlaneView::new([-0.745, 0.113], 1.45);
        let deep = PlaneView::new([-0.745, 0.113], 1.45e-8);
        assert!(!view_is_f32_limited(&ordinary, None, rect, 2.0));
        assert!(view_is_f32_limited(&deep, None, rect, 2.0));
    }

    #[test]
    fn julia_warning_accounts_for_the_fixed_parameter() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let near_zero = PlaneView::new([0.0, 0.0], 1.45e-8);
        assert!(!view_is_f32_limited(&near_zero, None, rect, 2.0));
        assert!(view_is_f32_limited(
            &near_zero,
            Some([-0.745, 0.113]),
            rect,
            2.0,
        ));
    }

    #[test]
    fn julia_cancellation_risk_prevents_the_false_basin_around_zero() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 1000.0));
        let close_view = PlaneView::new([0.0, 0.0], 1.45 / 766.0);
        assert!(julia_critical_roundoff_risk(
            &close_view,
            [0.0, 1.0],
            rect,
            1.0,
        ));

        let initial_view = PlaneView::new([0.0, 0.0], 1.45);
        assert!(!julia_critical_roundoff_risk(
            &initial_view,
            [0.0, 1.0],
            rect,
            1.0,
        ));
    }

    #[test]
    fn coordinate_sample_reports_f32_rounding_and_pixel_scale() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let view = PlaneView::new([-0.745123456789, 0.113987654321], 1.45e-5);
        let sample = coordinate_sample(&view, rect, rect.center(), 2.0, PrecisionMode::F32);
        assert_eq!(sample.rendered[0], sample.navigation[0] as f32 as f64);
        assert_eq!(sample.rendered[1], sample.navigation[1] as f32 as f64);
        assert_eq!(sample.delta[0], sample.rendered[0] - sample.navigation[0]);
        assert_eq!(sample.delta[1], sample.rendered[1] - sample.navigation[1]);
        assert!((sample.world_per_pixel - 2.0 * 1.45e-5 / 1200.0).abs() < 1e-18);
    }

    #[test]
    fn pinch_direction_and_sensitivity_are_predictable() {
        assert!(pinch_zoom_factor(1.2) < 1.0);
        assert!(pinch_zoom_factor(0.8) > 1.0);
        assert_eq!(pinch_zoom_factor(1.0), 1.0);
        assert!(pinch_zoom_factor(2.0) > 0.5);
    }

    #[test]
    fn shift_acceleration_is_fast_symmetric_and_bounded() {
        let zoom_in = accelerate_zoom_factor(0.5, true);
        let zoom_out = accelerate_zoom_factor(2.0, true);
        assert!(zoom_in < 1e-19);
        assert!(zoom_out > 1e19);
        assert!((zoom_in * zoom_out - 1.0).abs() < 1e-12);
        assert_eq!(accelerate_zoom_factor(0.5, false), 0.5);
        assert!(accelerate_zoom_factor(1e-100, true) >= 1e-20 * (1.0 - 1e-12));
        assert!(accelerate_zoom_factor(1e100, true) <= 1e20 * (1.0 + 1e-12));
    }

    #[test]
    fn progressive_zoom_uses_staged_symmetric_decades_and_lands_exactly() {
        assert_eq!(progressive_zoom_factor(0.0, 5_000.0), Some(1e-10));
        assert_eq!(progressive_zoom_factor(5_000.0, 0.0), Some(1e10));
        assert_eq!(progressive_zoom_factor(995.0, 1_000.0), Some(1e-5));
        assert_eq!(progressive_zoom_factor(1_000.0, 1_000.0), None);
    }

    #[test]
    fn automatic_precision_responds_to_probe_and_coordinate_limits() {
        assert_eq!(
            choose_precision(1.0, false, ProbeResult::default()),
            PrecisionMode::F32
        );
        assert_eq!(
            choose_precision(1.0, true, ProbeResult::default()),
            PrecisionMode::DoubleSingle
        );
        assert_eq!(
            choose_precision(
                512.0,
                false,
                ProbeResult {
                    f32: PathProbeResult {
                        unstable_samples: 2,
                        ..Default::default()
                    },
                    ..Default::default()
                }
            ),
            PrecisionMode::DoubleSingle
        );
    }

    #[test]
    fn double_single_readout_is_closer_to_navigation_coordinate() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let view = PlaneView::new([-0.745_123_456_789, 0.113_987_654_321], 1.45e-9);
        let f32_sample = coordinate_sample(&view, rect, rect.center(), 2.0, PrecisionMode::F32);
        let ds_sample =
            coordinate_sample(&view, rect, rect.center(), 2.0, PrecisionMode::DoubleSingle);
        assert!(ds_sample.delta[0].abs() < f32_sample.delta[0].abs());
        assert!(ds_sample.delta[1].abs() < f32_sample.delta[1].abs());
    }
}
