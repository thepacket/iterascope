use std::sync::Arc;

use web_time::Instant;

use crate::MAX_ITERATIONS;
use crate::arbitrary::{
    ARBITRARY_HANDOFF_ZOOM, DeepComplex, DeepView, MAX_DECIMAL_ZOOM_EXPONENT, ReferenceOrbit,
};
use crate::experiment::{
    ComplexDocument, ComputationDocument, DeepComplexDocument, DeepPlaneDocument, DisplayDocument,
    ExperimentDocument, FAMILY_ID, FORMAT_ID, FORMAT_VERSION, PlaneDocument,
};
use crate::orbit::{CriticalOrbit, CriticalOrbitCache, OrbitInput};
use crate::precision::{
    DoubleSingle, DsValidity, PathProbeResult, PrecisionMode, ProbeCache, ProbeInput, ProbeResult,
    ValidityLevel,
};
use crate::render::{self, DeepRenderData, FractalPipeline, Uniforms};

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
/// Orbit disagreement alone enables the more expensive DS path only once a
/// view is meaningfully zoomed. Classification disagreement and coordinate
/// collapse always override this floor.
const PROBE_DS_MIN_ZOOM: f64 = 256.0;

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

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeepReferenceKey {
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
    fn new(
        pane: usize,
        view: &DeepView,
        julia_c: &DeepComplex,
        iterations: u32,
        bailout: f32,
    ) -> Self {
        Self {
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
    data: Arc<DeepRenderData>,
}

impl PlaneView {
    const fn new(centre: [f64; 2], half_height: f64) -> Self {
        Self {
            centre,
            half_height,
        }
    }

    fn reset_parameter(&mut self) {
        *self = Self::new([-0.5, 0.0], 1.45);
    }

    fn reset_dynamical(&mut self) {
        *self = Self::new([0.0, 0.0], 1.45);
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
    parameter: PlaneView,
    dynamical: PlaneView,
    julia_c: [f64; 2],
    progressive_julia_zoom_target_exponent: u32,
    progressive_julia_zoom_active: bool,
    iterations: u32,
    bailout: f32,
    palette_phase: f32,
    smooth: bool,
    grid: bool,
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
    deep_generation: u64,
    frame_ms: f32,
    frame_start: Instant,
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
            parameter: PlaneView::new([-0.5, 0.0], 1.45),
            dynamical: PlaneView::new([0.0, 0.0], 1.45),
            julia_c: [-0.745, 0.113],
            progressive_julia_zoom_target_exponent: 0,
            progressive_julia_zoom_active: false,
            iterations: 256,
            bailout: 4.0,
            palette_phase: 0.0,
            smooth: true,
            grid: false,
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
            deep_generation: 0,
            frame_ms: 0.0,
            frame_start: Instant::now(),
        })
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            section(ui, "Experiment", |ui| {
                ui.label(
                    egui::RichText::new("Quadratic family")
                        .color(CREAM)
                        .strong(),
                );
                ui.label(
                    egui::RichText::new("f₍c₎(z) = z² + c")
                        .monospace()
                        .color(TEXT),
                );
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(
                        "Select c in the parameter plane to inspect its dynamical plane.",
                    )
                    .color(MUTED),
                );
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

            section(ui, "Selected parameter", |ui| {
                let mut parameter_changed = false;
                parameter_changed |= coordinate_row(ui, "Re(c)", &mut self.julia_c[0]);
                parameter_changed |= coordinate_row(ui, "Im(c)", &mut self.julia_c[1]);
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
                    }
                    if ui
                        .add_enabled(self.progressive_julia_zoom_active, egui::Button::new("Stop"))
                        .clicked()
                    {
                        self.progressive_julia_zoom_active = false;
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
                ui.horizontal(|ui| {
                    parameter_changed |=
                        preset_button(ui, "Seahorse", [-0.745, 0.113], &mut self.julia_c);
                    parameter_changed |=
                        preset_button(ui, "Dendrite", [0.0, 1.0], &mut self.julia_c);
                });
                ui.horizontal(|ui| {
                    parameter_changed |=
                        preset_button(ui, "Rabbit", [-0.123, 0.745], &mut self.julia_c);
                    parameter_changed |=
                        preset_button(ui, "Basilica", [-1.0, 0.0], &mut self.julia_c);
                });
                if parameter_changed {
                    self.deep_julia_c = None;
                }
                if parameter_changed {
                    self.reframe_dynamical_plane();
                }
            });

            section(ui, "Computation", |ui| {
                ui.add(
                    egui::Slider::new(&mut self.iterations, 32..=MAX_ITERATIONS)
                        .logarithmic(true)
                        .text("iterations"),
                );
                ui.add(
                    egui::Slider::new(&mut self.bailout, 2.0..=32.0)
                        .logarithmic(true)
                        .text("bailout"),
                );
                precision_row(
                    ui,
                    "Parameter",
                    self.precision_modes[0],
                    self.ds_validity[0],
                    self.deep_active[0],
                );
                precision_row(
                    ui,
                    "Julia",
                    self.precision_modes[1],
                    self.ds_validity[1],
                    self.deep_active[1],
                );
                ui.label(
                    egui::RichText::new("Precision switches automatically after instability.")
                        .small()
                        .color(MUTED),
                );
            });

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

            section(ui, "View status", |ui| {
                zoom_row(
                    ui,
                    "Parameter",
                    self.deep_views[0].as_ref().map_or_else(
                        || format!("{:.6e}", self.parameter.magnification()),
                        DeepView::magnification_label,
                    ),
                );
                zoom_row(
                    ui,
                    "Julia",
                    self.deep_views[1].as_ref().map_or_else(
                        || format!("{:.6e}", self.dynamical.magnification()),
                        DeepView::magnification_label,
                    ),
                );
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
            });

            section(ui, "Display", |ui| {
                ui.checkbox(&mut self.smooth, "Smooth escape-time colouring");
                ui.checkbox(&mut self.grid, "Coordinate grid");
                ui.add(
                    egui::Slider::new(&mut self.palette_phase, -1.0..=1.0).text("palette phase"),
                );
                if ui.button("Reset palette").clicked() {
                    self.palette_phase = 0.0;
                }
            });

            section(ui, "Navigation", |ui| {
                ui.label(egui::RichText::new("Click: centre + ×2 zoom").color(TEXT));
                ui.label(
                    egui::RichText::new(
                        "Wheel or pinch to zoom. Drag to pan. Left clicks also choose c.",
                    )
                    .small()
                    .color(MUTED),
                );
                ui.label(
                    egui::RichText::new(
                        "Hold Shift while zooming for accelerated deep navigation.",
                    )
                    .small()
                    .color(BLUE),
                );
                ui.add_space(4.0);
                ui.label(egui::RichText::new("Fine pan target").color(TEXT));
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.active_pane, 0, "Parameter");
                    ui.selectable_value(&mut self.active_pane, 1, "Julia");
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
                        "Buttons and arrow keys pan by 1/10 of the displayed range. Hover a pane or select its target above.",
                    )
                    .small()
                    .color(MUTED),
                );
                ui.horizontal(|ui| {
                    if ui.button("Reset parameter").clicked() {
                        self.parameter.reset_parameter();
                        self.deep_views[0] = None;
                        self.zoom_focus[0] = None;
                    }
                    if ui.button("Reset Julia").clicked() {
                        self.reframe_dynamical_plane();
                    }
                });
            });
        });
    }

    fn experiment_document(&self) -> ExperimentDocument {
        ExperimentDocument {
            format: FORMAT_ID.to_owned(),
            version: FORMAT_VERSION,
            family: FAMILY_ID.to_owned(),
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
            computation: ComputationDocument {
                iterations: self.iterations,
                bailout: self.bailout,
            },
            display: DisplayDocument {
                smooth_escape_time: self.smooth,
                coordinate_grid: self.grid,
                palette_phase: self.palette_phase,
                critical_orbit_overlay: self.show_orbit_overlay,
            },
            progressive_julia_zoom_target_exponent: self.progressive_julia_zoom_target_exponent,
            deep_parameter_plane: self.deep_views[0].as_ref().map(deep_plane_document),
            deep_dynamical_plane: self.deep_views[1].as_ref().map(deep_plane_document),
            deep_parameter_c: self.deep_julia_c.as_ref().map(|value| DeepComplexDocument {
                re: value.re.exact_decimal(),
                im: value.im.exact_decimal(),
            }),
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
        self.progressive_julia_zoom_target_exponent =
            document.progressive_julia_zoom_target_exponent;
        self.progressive_julia_zoom_active = false;
        self.iterations = document.computation.iterations;
        self.bailout = document.computation.bailout;
        self.smooth = document.display.smooth_escape_time;
        self.grid = document.display.coordinate_grid;
        self.palette_phase = document.display.palette_phase;
        self.show_orbit_overlay = document.display.critical_orbit_overlay;

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

        let (title, subtitle) = if pane == 0 {
            ("PARAMETER PLANE", "c -> bounded critical orbit")
        } else {
            ("DYNAMICAL PLANE", "z -> z^2 + c")
        };
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
                let keyboard_pan = ui.input(|input| {
                    [
                        (input.key_pressed(egui::Key::ArrowRight) as i8
                            - input.key_pressed(egui::Key::ArrowLeft) as i8)
                            as f64,
                        (input.key_pressed(egui::Key::ArrowUp) as i8
                            - input.key_pressed(egui::Key::ArrowDown) as i8)
                            as f64,
                    ]
                });
                fine_pan[0] += keyboard_pan[0];
                fine_pan[1] += keyboard_pan[1];
            }
        }
        if fine_pan != [0.0; 2] {
            interacting = true;
            if pane == 1 {
                self.progressive_julia_zoom_active = false;
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
                }
                let accelerated = ui.input(|input| input.modifiers.shift);
                self.zoom_pane(pane, accelerate_zoom_factor(factor, accelerated));
            }
        }
        if response.clicked() && !response.dragged() {
            if let Some(position) = response.interact_pointer_pos() {
                interacting = true;
                if pane == 1 {
                    self.progressive_julia_zoom_active = false;
                }
                if let Some(deep) = &mut self.deep_views[pane] {
                    let local = local_coordinate(viewport, position);
                    let _ = deep.recenter_local(local);
                    let accelerated = ui.input(|input| input.modifiers.shift);
                    let _ = deep.zoom(accelerate_zoom_factor(0.5, accelerated));
                    self.zoom_focus[pane] = None;
                    if pane == 0 {
                        self.deep_julia_c = Some(deep.centre.clone());
                        self.julia_c = deep.centre_preview();
                        self.reframe_dynamical_plane();
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
                        let accelerated = ui.input(|input| input.modifiers.shift);
                        self.zoom_pane(0, accelerate_zoom_factor(0.5, accelerated));
                        if let Some(deep) = &self.deep_views[0] {
                            self.deep_julia_c = Some(deep.centre.clone());
                        }
                        self.reframe_dynamical_plane();
                    } else {
                        self.dynamical.centre = point;
                        let accelerated = ui.input(|input| input.modifiers.shift);
                        self.zoom_pane(1, accelerate_zoom_factor(0.5, accelerated));
                    }
                }
            }
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
        let probe = if deep_view.is_some() {
            ProbeResult::default()
        } else if interacting {
            self.probes[pane].current(probe_input).unwrap_or_default()
        } else {
            self.probes[pane].update(probe_input)
        };
        let coordinate_limited = view_is_f32_limited(
            &view,
            (pane == 1).then_some(self.julia_c),
            viewport,
            ui.ctx().pixels_per_point(),
        ) || (pane == 1
            && julia_critical_roundoff_risk(
                &view,
                self.julia_c,
                viewport,
                ui.ctx().pixels_per_point(),
            ));
        let precision = choose_precision(magnification, coordinate_limited, probe);
        self.precision_modes[pane] = precision;
        let ds_validity = DsValidity::from_probe(
            probe.ds,
            ds_coordinate_ratio(
                &view,
                (pane == 1).then_some(self.julia_c),
                viewport,
                ui.ctx().pixels_per_point(),
            ),
        );
        self.ds_validity[pane] = ds_validity;

        let deep = self.deep_reference(pane, deep_view.as_ref(), !interacting);
        self.deep_active[pane] = deep.is_some();

        let (precision_text, zoom_colour) = if deep.is_some() {
            ("AP PERT", BLUE)
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

        let mut uniforms = Uniforms::new(
            view.centre,
            view.half_height,
            aspect,
            self.julia_c,
            self.iterations,
            self.bailout,
            pane,
            self.palette_phase,
            self.smooth,
            self.grid,
            precision,
        );
        if let Some(data) = &deep {
            uniforms = uniforms.enable_perturbation(
                data.scale_mantissa,
                data.scale_exponent,
                data.reference.len(),
                data.ds_fallback,
            );
        }
        ui.painter()
            .add(render::callback(viewport, pane, uniforms, deep));

        if pane == 1 && self.show_orbit_overlay && deep_view.is_none() {
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

    fn reframe_dynamical_plane(&mut self) {
        self.dynamical.reset_dynamical();
        self.deep_views[1] = None;
        self.progressive_julia_zoom_active = false;
        self.zoom_focus[1] = None;
        self.orbit_step = 0;
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

    fn advance_progressive_julia_zoom(&mut self) {
        if !self.progressive_julia_zoom_active {
            return;
        }
        let current = self.magnification_log10(1);
        let target = self.progressive_julia_zoom_target_exponent as f64;
        let Some(factor) = progressive_zoom_factor(current, target) else {
            self.progressive_julia_zoom_active = false;
            return;
        };
        self.zoom_pane(1, factor);
    }

    fn zoom_pane(&mut self, pane: usize, factor: f64) {
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
        let key = DeepReferenceKey::new(pane, view, &julia_c, self.iterations, self.bailout);
        if let Some(cached) = &self.deep_references[pane]
            && cached.key == key
        {
            return Some(Arc::clone(&cached.data));
        }
        if !may_build {
            // Keep the last completed deep frame on screen while navigation
            // changes and a replacement reference orbit is not ready yet.
            // Returning `None` here exposes the frozen DS handoff snapshot,
            // which appears as a distracting 1.14e14× flash.
            if let Some(cached) = &self.deep_references[pane] {
                return Some(Arc::clone(&cached.data));
            }
            // On the first transition there is no deep frame to retain. Build
            // synchronously so the previously presented shallow frame stays
            // visible until the first AP frame can replace it atomically.
        }

        let orbit = if pane == 0 {
            ReferenceOrbit::quadratic_parameter(&view.centre, self.iterations, self.bailout as f64)
        } else {
            ReferenceOrbit::quadratic_julia(
                view.centre.clone(),
                &julia_c,
                self.iterations,
                self.bailout as f64,
            )
        }
        .ok()?;
        let (scale_mantissa, scale_exponent) = view.half_height.scaled_f32();
        self.deep_generation = self.deep_generation.wrapping_add(1).max(1);
        let data = Arc::new(DeepRenderData::from_reference(
            self.deep_generation,
            scale_mantissa,
            scale_exponent,
            &orbit,
            view.magnification_log10 <= ARBITRARY_HANDOFF_ZOOM.log10() + f64::EPSILON * 8.0,
        ));
        self.deep_references[pane] = Some(DeepReferenceCache {
            key,
            data: Arc::clone(&data),
        });
        Some(data)
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
        let now = Instant::now();
        let elapsed = (now - self.frame_start).as_secs_f32() * 1000.0;
        self.frame_start = now;
        self.frame_ms += (elapsed - self.frame_ms) * 0.08;

        egui::Panel::top("iterascope.topbar")
            .exact_size(34.0)
            .frame(egui::Frame::new().fill(egui::Color32::from_rgb(12, 14, 18)))
            .show(ui, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("ITERASCOPE").strong().color(TEXT));
                    ui.label(egui::RichText::new("Complex dynamics laboratory").color(MUTED));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_space(8.0);
                        ui.label(
                            egui::RichText::new(format!("{:.1} ms", self.frame_ms))
                                .monospace()
                                .color(MUTED),
                        );
                        badge(ui, "LIVE", CORAL);
                    });
                });
            });

        egui::Panel::left("iterascope.controls")
            .exact_size(PANEL_WIDTH)
            .frame(
                egui::Frame::new()
                    .fill(PANEL)
                    .inner_margin(egui::Margin::symmetric(12, 10)),
            )
            .show(ui, |ui| self.controls(ui));

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BG))
            .show(ui, |ui| self.workspace(ui));

        // The current stage has now produced its paint callback. Advance the
        // authoritative view afterwards so every stage is presented for at
        // least one frame before its successor is prepared.
        self.advance_progressive_julia_zoom();

        self.experiment_editor(ui.ctx());
        self.orbit_inspector(ui.ctx());

        ui.ctx().request_repaint();
    }

    #[cfg(target_arch = "wasm32")]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
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
                .range(-2.0..=2.0)
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

fn precision_row(
    ui: &mut egui::Ui,
    label: &str,
    precision: PrecisionMode,
    validity: DsValidity,
    deep_active: bool,
) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let (text, colour) = if deep_active {
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
