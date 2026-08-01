use web_time::Instant;

use crate::precision::{DoubleSingle, PrecisionMode, ProbeCache, ProbeInput, ProbeResult};
use crate::render::{self, FractalPipeline, Uniforms};

const PANEL_WIDTH: f32 = 286.0;
const HEADER_HEIGHT: f32 = 33.0;
const PANE_GAP: f32 = 6.0;
/// Trackpad pinch is reported every frame as a multiplicative scale. Applying
/// it one-for-one feels abrupt on a high-resolution trackpad, so temper it
/// without introducing lag or accumulating hidden state.
const PINCH_SENSITIVITY: f64 = 0.65;
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

    fn zoom(&mut self, factor: f64) {
        self.half_height = (self.half_height * factor).clamp(1e-14, 1e6);
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
    iterations: u32,
    bailout: f32,
    palette_phase: f32,
    smooth: bool,
    grid: bool,
    zoom_focus: [Option<[f64; 2]>; 2],
    precision_modes: [PrecisionMode; 2],
    probes: [ProbeCache; 2],
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
            iterations: 256,
            bailout: 4.0,
            palette_phase: 0.0,
            smooth: true,
            grid: false,
            zoom_focus: [None, None],
            precision_modes: [PrecisionMode::F32; 2],
            probes: [ProbeCache::default(); 2],
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

            section(ui, "Selected parameter", |ui| {
                let mut parameter_changed = false;
                parameter_changed |= coordinate_row(ui, "Re(c)", &mut self.julia_c[0]);
                parameter_changed |= coordinate_row(ui, "Im(c)", &mut self.julia_c[1]);
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
                    self.reframe_dynamical_plane();
                }
            });

            section(ui, "Computation", |ui| {
                ui.add(
                    egui::Slider::new(&mut self.iterations, 32..=2048)
                        .logarithmic(true)
                        .text("iterations"),
                );
                ui.add(
                    egui::Slider::new(&mut self.bailout, 2.0..=32.0)
                        .logarithmic(true)
                        .text("bailout"),
                );
                precision_row(ui, "Parameter", self.precision_modes[0]);
                precision_row(ui, "Julia", self.precision_modes[1]);
                ui.label(
                    egui::RichText::new("Precision switches automatically after instability.")
                        .small()
                        .color(MUTED),
                );
            });

            section(ui, "View status", |ui| {
                zoom_row(ui, "Parameter", self.parameter.magnification());
                zoom_row(ui, "Julia", self.dynamical.magnification());
                probe_row(ui, "P probe", self.probes[0].last_result());
                probe_row(ui, "J probe", self.probes[1].last_result());
                ui.label(
                    egui::RichText::new(
                        "Nine CPU orbit samples validate f32 only when a settled view changes.",
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
                ui.horizontal(|ui| {
                    if ui.button("Reset parameter").clicked() {
                        self.parameter.reset_parameter();
                        self.zoom_focus[0] = None;
                    }
                    if ui.button("Reset Julia").clicked() {
                        self.dynamical.reset_dynamical();
                        self.zoom_focus[1] = None;
                    }
                });
            });
        });
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
        let pinch_delta = ui.input(|input| input.zoom_delta());
        let pinching = (pinch_delta - 1.0).abs() > 0.001;
        let mut interacting = pinching || response.dragged();
        if response.dragged() && !pinching {
            let delta = ui.input(|input| input.pointer.delta());
            if pane == 0 {
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
                if pane == 0 {
                    self.parameter.zoom_from(self.zoom_focus[0], factor);
                } else {
                    self.dynamical.zoom_from(self.zoom_focus[1], factor);
                }
            }
        }
        if response.clicked() && !response.dragged() {
            if let Some(position) = response.interact_pointer_pos() {
                interacting = true;
                let point = if pane == 0 {
                    self.parameter.point_at(viewport, position)
                } else {
                    self.dynamical.point_at(viewport, position)
                };
                self.zoom_focus[pane] = Some(point);
                if pane == 0 {
                    self.julia_c = point;
                    self.parameter.zoom_from(Some(point), 0.5);
                    self.reframe_dynamical_plane();
                } else {
                    self.dynamical.zoom_from(Some(point), 0.5);
                }
            }
        }

        let view = if pane == 0 {
            self.parameter
        } else {
            self.dynamical
        };
        let aspect = viewport.width() / viewport.height().max(1.0);
        let magnification = view.magnification();
        let probe_input = ProbeInput {
            centre: view.centre,
            half_height: view.half_height,
            aspect: aspect as f64,
            julia_c: self.julia_c,
            iterations: self.iterations,
            bailout: self.bailout as f64,
            pane,
        };
        let probe = if interacting {
            self.probes[pane].current(probe_input).unwrap_or_default()
        } else {
            self.probes[pane].update(probe_input)
        };
        let coordinate_limited = view_is_f32_limited(
            &view,
            (pane == 1).then_some(self.julia_c),
            viewport,
            ui.ctx().pixels_per_point(),
        );
        let precision = choose_precision(magnification, coordinate_limited, probe);
        self.precision_modes[pane] = precision;

        let (precision_text, zoom_colour) = match precision {
            PrecisionMode::DoubleSingle => ("DS ~48-BIT", BLUE),
            PrecisionMode::F32 if probe.unstable() => ("F32 UNSTABLE", CORAL),
            PrecisionMode::F32 => ("F32", CREAM),
        };
        ui.painter().text(
            egui::pos2(header.max.x - 10.0, header.center().y),
            egui::Align2::RIGHT_CENTER,
            format!("{precision_text}  ·  ZOOM ×{magnification:.4e}"),
            egui::FontId::new(11.0, egui::FontFamily::Monospace),
            zoom_colour,
        );

        let uniforms = Uniforms::new(
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
        ui.painter().add(render::callback(viewport, pane, uniforms));

        if let Some(focus) = self.zoom_focus[pane] {
            self.draw_focus_marker(ui, viewport, &view, focus);
        }
        self.draw_readout(ui, viewport, response.hover_pos(), &view, precision);
    }

    fn reframe_dynamical_plane(&mut self) {
        self.dynamical.reset_dynamical();
        self.zoom_focus[1] = None;
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

fn zoom_row(ui: &mut egui::Ui, label: &str, magnification: f64) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(format!("×{magnification:.6e}"))
                    .monospace()
                    .color(CREAM),
            );
        });
    });
}

fn precision_row(ui: &mut egui::Ui, label: &str, precision: PrecisionMode) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            badge(
                ui,
                precision.label(),
                if precision == PrecisionMode::DoubleSingle {
                    BLUE
                } else {
                    CREAM
                },
            );
        });
    });
}

fn probe_row(ui: &mut egui::Ui, label: &str, result: Option<ProbeResult>) {
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

fn choose_precision(
    magnification: f64,
    coordinate_limited: bool,
    probe: ProbeResult,
) -> PrecisionMode {
    let classification_failed = probe.classification_mismatches > 0;
    let orbit_failed = magnification >= PROBE_DS_MIN_ZOOM && probe.unstable();
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
                    unstable_samples: 2,
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
