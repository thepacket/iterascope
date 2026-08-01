use web_time::Instant;

use crate::render::{self, FractalPipeline, Uniforms};

const PANEL_WIDTH: f32 = 286.0;
const HEADER_HEIGHT: f32 = 33.0;
const PANE_GAP: f32 = 6.0;

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

    fn zoom_at(&mut self, rect: egui::Rect, position: egui::Pos2, factor: f64) {
        let before = self.point_at(rect, position);
        self.half_height = (self.half_height * factor).clamp(1e-14, 1e6);
        let after = self.point_at(rect, position);
        self.centre[0] += before[0] - after[0];
        self.centre[1] += before[1] - after[1];
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
                coordinate_row(ui, "Re(c)", &mut self.julia_c[0]);
                coordinate_row(ui, "Im(c)", &mut self.julia_c[1]);
                ui.horizontal(|ui| {
                    preset_button(ui, "Seahorse", [-0.745, 0.113], &mut self.julia_c);
                    preset_button(ui, "Dendrite", [0.0, 1.0], &mut self.julia_c);
                });
                ui.horizontal(|ui| {
                    preset_button(ui, "Rabbit", [-0.123, 0.745], &mut self.julia_c);
                    preset_button(ui, "Basilica", [-1.0, 0.0], &mut self.julia_c);
                });
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
                ui.horizontal(|ui| {
                    ui.label("precision");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        badge(ui, "GPU f32", BLUE);
                    });
                });
                ui.label(
                    egui::RichText::new("Deep precision arrives in the next rendering stage.")
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
                ui.label(egui::RichText::new("Drag to pan · wheel to zoom").color(TEXT));
                ui.label(
                    egui::RichText::new("Click the parameter plane to choose c.")
                        .small()
                        .color(MUTED),
                );
                ui.horizontal(|ui| {
                    if ui.button("Reset parameter").clicked() {
                        self.parameter.reset_parameter();
                    }
                    if ui.button("Reset Julia").clicked() {
                        self.dynamical.reset_dynamical();
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
            ("PARAMETER PLANE", "c ↦ bounded critical orbit")
        } else {
            ("DYNAMICAL PLANE", "z ↦ z² + c")
        };
        let magnification = if pane == 0 {
            self.parameter.magnification()
        } else {
            self.dynamical.magnification()
        };

        ui.painter().text(
            egui::pos2(header.min.x + 11.0, header.center().y),
            egui::Align2::LEFT_CENTER,
            title,
            egui::FontId::new(11.0, egui::FontFamily::Monospace),
            TEXT,
        );
        ui.painter().text(
            egui::pos2(header.center().x, header.center().y),
            egui::Align2::CENTER_CENTER,
            subtitle,
            egui::FontId::new(11.0, egui::FontFamily::Proportional),
            MUTED,
        );
        ui.painter().text(
            egui::pos2(header.max.x - 10.0, header.center().y),
            egui::Align2::RIGHT_CENTER,
            format!("×{magnification:.3e}"),
            egui::FontId::new(10.5, egui::FontFamily::Monospace),
            MUTED,
        );

        let response = ui.allocate_rect(viewport, egui::Sense::click_and_drag());
        if response.dragged() {
            let delta = ui.input(|input| input.pointer.delta());
            if pane == 0 {
                self.parameter.pan(viewport, delta);
            } else {
                self.dynamical.pan(viewport, delta);
            }
        }
        if response.hovered() {
            let scroll = ui.input(|input| input.smooth_scroll_delta.y);
            if scroll.abs() > 0.0 {
                if let Some(position) = response.hover_pos() {
                    let factor = (-scroll as f64 * 0.0025).exp();
                    if pane == 0 {
                        self.parameter.zoom_at(viewport, position, factor);
                    } else {
                        self.dynamical.zoom_at(viewport, position, factor);
                    }
                }
            }
        }
        if pane == 0 && response.clicked() && !response.dragged() {
            if let Some(position) = response.interact_pointer_pos() {
                self.julia_c = self.parameter.point_at(viewport, position);
            }
        }

        let view = if pane == 0 {
            self.parameter
        } else {
            self.dynamical
        };
        let aspect = viewport.width() / viewport.height().max(1.0);
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
        );
        ui.painter().add(render::callback(viewport, pane, uniforms));

        if pane == 0 {
            self.draw_parameter_marker(ui, viewport);
        }
        self.draw_readout(ui, viewport, response.hover_pos(), &view);
    }

    fn draw_parameter_marker(&self, ui: &egui::Ui, rect: egui::Rect) {
        let aspect = rect.width() as f64 / rect.height().max(1.0) as f64;
        let nx =
            (self.julia_c[0] - self.parameter.centre[0]) / (self.parameter.half_height * aspect);
        let ny = (self.julia_c[1] - self.parameter.centre[1]) / self.parameter.half_height;
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
    ) {
        let Some(pointer) = pointer.filter(|position| rect.contains(*position)) else {
            return;
        };
        let world = view.point_at(rect, pointer);
        let text = format!("{:+.9}  {:+.9}i", world[0], world[1]);
        let galley = ui.painter().layout_no_wrap(
            text,
            egui::FontId::new(10.5, egui::FontFamily::Monospace),
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

fn coordinate_row(ui: &mut egui::Ui, label: &str, value: &mut f64) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(label).monospace().color(MUTED));
        ui.add(
            egui::DragValue::new(value)
                .speed(0.0001)
                .range(-2.0..=2.0)
                .max_decimals(12),
        );
    });
}

fn preset_button(ui: &mut egui::Ui, label: &str, c: [f64; 2], selected: &mut [f64; 2]) {
    if ui.small_button(label).clicked() {
        *selected = c;
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
    fn zoom_keeps_the_point_under_the_cursor_fixed() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
        let pointer = egui::pos2(625.0, 172.0);
        let mut view = PlaneView::new([-0.5, 0.0], 1.45);
        let before = view.point_at(rect, pointer);
        view.zoom_at(rect, pointer, 0.4);
        let after = view.point_at(rect, pointer);
        assert!((before[0] - after[0]).abs() < 1e-12);
        assert!((before[1] - after[1]).abs() < 1e-12);
    }
}
