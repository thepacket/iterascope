//! Gradients and colouring algorithms — the colour stage that turns the
//! statistics of an orbit into a pixel.
//!
//! Iteration produces a [`GenericResult`](crate::render) per pixel: whether
//! the orbit escaped, converged or stayed bounded, the (smoothed) iteration
//! count, the final `z`, and a few accumulators gathered along the orbit
//! (orbit-trap distance, triangle-inequality and stripe averages, the
//! derivative for distance estimation). The colour stage maps that to a
//! gradient position through an independent *outside* (escaped or converged)
//! and *inside* (bounded) algorithm, then looks the position up in a single
//! cyclic [`Gradient`]. This mirrors the Ultra Fractal layer model: one
//! gradient per layer, two colouring algorithms.
//!
//! The CPU owns the editable model ([`Gradient`], [`Colouring`]); the GPU
//! receives a rasterised lookup table ([`Gradient::lookup_table`]) and a
//! small uniform block ([`Colouring::gpu_words`]).

use serde::{Deserialize, Serialize};

use crate::family::FractalFamily;

/// Entries in the rasterised gradient uploaded to the GPU.
pub(crate) const LOOKUP_TABLE_LEN: usize = 1024;

/// Ultra Fractal gradients are defined on 400 positions.
const UGR_POSITIONS: f32 = 400.0;

// ---------------------------------------------------------------------------
// Gradient
// ---------------------------------------------------------------------------

/// How colours between two control points are blended.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Interpolation {
    /// Straight lines in RGB.
    #[default]
    Rgb,
    /// Hue, saturation and lightness, taking the shorter way round the hue
    /// circle.
    HslShort,
    /// Hue, saturation and lightness, taking the longer way round.
    HslLong,
}

impl Interpolation {
    pub(crate) const ALL: [Self; 3] = [Self::Rgb, Self::HslShort, Self::HslLong];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Rgb => "RGB",
            Self::HslShort => "HSL (short)",
            Self::HslLong => "HSL (long)",
        }
    }
}

/// One control point: a position on the cyclic gradient and an sRGB colour.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct GradientStop {
    /// Position in `[0, 1)`.
    pub(crate) position: f32,
    /// sRGB components in `[0, 1]`.
    pub(crate) colour: [f32; 3],
}

impl GradientStop {
    pub(crate) const fn new(position: f32, colour: [f32; 3]) -> Self {
        Self { position, colour }
    }

    pub(crate) fn from_u8(position: f32, rgb: [u8; 3]) -> Self {
        Self::new(
            position,
            [
                rgb[0] as f32 / 255.0,
                rgb[1] as f32 / 255.0,
                rgb[2] as f32 / 255.0,
            ],
        )
    }
}

/// A cyclic colour gradient: the colour at position 1 wraps to position 0,
/// so a colouring algorithm can sweep through it any number of times.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Gradient {
    pub(crate) name: String,
    /// Control points, kept sorted by position.
    pub(crate) stops: Vec<GradientStop>,
    #[serde(default)]
    pub(crate) interpolation: Interpolation,
    /// Cubic (Catmull-Rom) rather than linear blending between stops.
    #[serde(default)]
    pub(crate) smooth: bool,
    /// Rotation of the whole gradient, in turns. Positive values move the
    /// colours towards higher positions.
    #[serde(default)]
    pub(crate) rotation: f32,
}

impl Default for Gradient {
    fn default() -> Self {
        presets::default_gradient()
    }
}

impl Gradient {
    pub(crate) fn new(name: &str, stops: Vec<GradientStop>) -> Self {
        let mut gradient = Self {
            name: name.to_owned(),
            stops,
            interpolation: Interpolation::Rgb,
            smooth: false,
            rotation: 0.0,
        };
        gradient.normalise();
        gradient
    }

    /// Sorts the stops and wraps their positions into `[0, 1)`. A gradient
    /// always keeps at least one stop.
    pub(crate) fn normalise(&mut self) {
        if self.stops.is_empty() {
            self.stops.push(GradientStop::new(0.0, [0.5; 3]));
        }
        for stop in &mut self.stops {
            stop.position = wrap_unit(stop.position);
            for channel in &mut stop.colour {
                *channel = channel.clamp(0.0, 1.0);
            }
        }
        self.stops
            .sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
        self.rotation = wrap_unit(self.rotation);
    }

    /// Colour at `position` (any real number; the gradient is cyclic),
    /// before the gradient rotation is applied.
    pub(crate) fn colour_at_unrotated(&self, position: f32) -> [f32; 3] {
        let n = self.stops.len();
        if n == 1 {
            return self.stops[0].colour;
        }
        let t = wrap_unit(position);
        // Index of the last stop at or before t (cyclically).
        let mut index = n - 1;
        for (i, stop) in self.stops.iter().enumerate() {
            if stop.position <= t {
                index = i;
            } else {
                break;
            }
        }
        let a = self.stops[index];
        let b = self.stops[(index + 1) % n];
        let start = a.position;
        let mut end = b.position;
        if index == n - 1 {
            end += 1.0;
        }
        let mut local = t;
        if local < start {
            local += 1.0;
        }
        let span = end - start;
        let fraction = if span <= 1e-6 {
            0.0
        } else {
            ((local - start) / span).clamp(0.0, 1.0)
        };
        if self.smooth {
            let before = self.stops[(index + n - 1) % n].colour;
            let after = self.stops[(index + 2) % n].colour;
            let mut out = [0.0; 3];
            for channel in 0..3 {
                out[channel] = catmull_rom(
                    before[channel],
                    a.colour[channel],
                    b.colour[channel],
                    after[channel],
                    fraction,
                )
                .clamp(0.0, 1.0);
            }
            if self.interpolation == Interpolation::Rgb {
                return out;
            }
            // Smooth HSL: blend the cubic RGB result with the HSL path so hue
            // sweeps stay saturated while the curve stays continuous.
            let hsl = blend_hsl(a.colour, b.colour, fraction, self.interpolation);
            return [
                0.5 * (out[0] + hsl[0]),
                0.5 * (out[1] + hsl[1]),
                0.5 * (out[2] + hsl[2]),
            ];
        }
        match self.interpolation {
            Interpolation::Rgb => [
                a.colour[0] + (b.colour[0] - a.colour[0]) * fraction,
                a.colour[1] + (b.colour[1] - a.colour[1]) * fraction,
                a.colour[2] + (b.colour[2] - a.colour[2]) * fraction,
            ],
            mode => blend_hsl(a.colour, b.colour, fraction, mode),
        }
    }

    /// Colour at `position` including the rotation.
    pub(crate) fn colour_at(&self, position: f32) -> [f32; 3] {
        self.colour_at_unrotated(position - self.rotation)
    }

    /// Rasterises the gradient into `LOOKUP_TABLE_LEN` RGBA entries for the
    /// GPU; the shader interpolates linearly between entries and wraps.
    pub(crate) fn lookup_table(&self) -> Vec<[f32; 4]> {
        (0..LOOKUP_TABLE_LEN)
            .map(|i| {
                let c = self.colour_at(i as f32 / LOOKUP_TABLE_LEN as f32);
                [c[0], c[1], c[2], 1.0]
            })
            .collect()
    }

    /// Inserts a stop at `position` with the gradient's current colour there
    /// and returns its index.
    pub(crate) fn insert_stop(&mut self, position: f32) -> usize {
        let colour = self.colour_at(position);
        self.stops.push(GradientStop::new(
            wrap_unit(position - self.rotation),
            colour,
        ));
        self.normalise();
        self.stops
            .iter()
            .position(|stop| {
                (stop.position - wrap_unit(position - self.rotation)).abs() < 1e-6
                    && stop.colour == colour
            })
            .unwrap_or(0)
    }

    /// Removes a stop unless it is the last one.
    pub(crate) fn remove_stop(&mut self, index: usize) {
        if self.stops.len() > 1 && index < self.stops.len() {
            self.stops.remove(index);
        }
    }

    /// Mirrors the gradient: position p becomes 1 − p.
    pub(crate) fn reverse(&mut self) {
        for stop in &mut self.stops {
            stop.position = wrap_unit(1.0 - stop.position);
        }
        self.rotation = wrap_unit(-self.rotation);
        self.normalise();
    }

    /// Spreads the stops evenly over the gradient, keeping their order.
    pub(crate) fn distribute_evenly(&mut self) {
        let n = self.stops.len();
        for (i, stop) in self.stops.iter_mut().enumerate() {
            stop.position = i as f32 / n as f32;
        }
    }

    /// A random gradient from a small seed; the same seed always produces
    /// the same gradient so a document can record it by value.
    pub(crate) fn random(seed: u64) -> Self {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            // SplitMix64.
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f32 / (1u64 << 53) as f32
        };
        let count = 3 + (next() * 4.0) as usize;
        let base_hue = next();
        let scheme = next();
        let mut stops = Vec::with_capacity(count);
        for i in 0..count {
            // Analogous hues with occasional complements, alternating light
            // and dark stops so the gradient has contrast.
            let hue = if scheme < 0.35 {
                base_hue + next() * 0.2
            } else if scheme < 0.7 {
                base_hue + (i as f32 / count as f32) * 0.5
            } else {
                base_hue + (i % 2) as f32 * 0.5 + next() * 0.1
            };
            let saturation = 0.55 + 0.45 * next();
            let lightness = if i % 2 == 0 {
                0.08 + 0.25 * next()
            } else {
                0.55 + 0.4 * next()
            };
            let colour = hsl_to_rgb([wrap_unit(hue), saturation, lightness]);
            stops.push(GradientStop::new(
                i as f32 / count as f32 + 0.04 * next(),
                colour,
            ));
        }
        let mut gradient = Self::new(&format!("Random {seed}"), stops);
        gradient.smooth = next() < 0.5;
        gradient
    }

    /// Serialises the gradient as an Ultra Fractal `.ugr` entry.
    pub(crate) fn to_ugr(&self) -> String {
        let mut out = String::new();
        let name = if self.name.trim().is_empty() {
            "IteraScope"
        } else {
            self.name.trim()
        };
        out.push_str(&format!("{name} {{\ngradient:\n"));
        out.push_str(&format!(
            "  title=\"{name}\" smooth={} rotation={}\n",
            if self.smooth { "yes" } else { "no" },
            (self.rotation * UGR_POSITIONS).round() as i32,
        ));
        for stop in &self.stops {
            let index = (stop.position * UGR_POSITIONS).round() as i32;
            let r = (stop.colour[0] * 255.0).round() as u32;
            let g = (stop.colour[1] * 255.0).round() as u32;
            let b = (stop.colour[2] * 255.0).round() as u32;
            out.push_str(&format!(
                "  index={index} color={}\n",
                r + (g << 8) + (b << 16)
            ));
        }
        out.push_str("opacity:\n  smooth=no\n  index=0 opacity=255\n}\n");
        out
    }

    /// Parses every gradient in an Ultra Fractal `.ugr` file, or a Fractint
    /// `.map` palette (256 lines of `r g b`). Returns an error when nothing
    /// usable was found.
    pub(crate) fn parse(text: &str) -> Result<Vec<Gradient>, String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err("the text is empty".to_owned());
        }
        if trimmed.contains("gradient:") {
            return parse_ugr(trimmed);
        }
        parse_map(trimmed)
    }
}

fn parse_ugr(text: &str) -> Result<Vec<Gradient>, String> {
    let mut gradients = Vec::new();
    let mut name = String::new();
    let mut in_gradient = false;
    let mut in_opacity = false;
    let mut stops: Vec<GradientStop> = Vec::new();
    let mut smooth = false;
    let mut rotation = 0.0;
    let mut title: Option<String> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.ends_with('{') && !in_gradient {
            name = line.trim_end_matches('{').trim().to_owned();
            continue;
        }
        if line == "gradient:" {
            in_gradient = true;
            in_opacity = false;
            stops.clear();
            smooth = false;
            rotation = 0.0;
            title = None;
            continue;
        }
        if line == "opacity:" {
            in_opacity = true;
            continue;
        }
        if line.starts_with('}') {
            if in_gradient && !stops.is_empty() {
                let mut gradient = Gradient::new(title.as_deref().unwrap_or(&name), stops.clone());
                gradient.smooth = smooth;
                gradient.rotation = wrap_unit(rotation / UGR_POSITIONS);
                gradients.push(gradient);
            }
            in_gradient = false;
            in_opacity = false;
            continue;
        }
        if !in_gradient || in_opacity {
            continue;
        }
        let mut index: Option<f32> = None;
        let mut colour: Option<u32> = None;
        for token in split_ugr_tokens(line) {
            let Some((key, value)) = token.split_once('=') else {
                continue;
            };
            match key {
                "title" => title = Some(value.trim_matches('"').to_owned()),
                "smooth" => smooth = value == "yes",
                "rotation" => rotation = value.parse().unwrap_or(0.0),
                "index" => index = value.parse().ok(),
                "color" => colour = value.parse().ok(),
                _ => {}
            }
        }
        if let (Some(index), Some(colour)) = (index, colour) {
            // Ultra Fractal stores COLORREF-style integers: red in the low
            // byte, blue in the high byte.
            let rgb = [
                (colour & 0xFF) as u8,
                ((colour >> 8) & 0xFF) as u8,
                ((colour >> 16) & 0xFF) as u8,
            ];
            stops.push(GradientStop::from_u8(index / UGR_POSITIONS, rgb));
        }
    }
    if gradients.is_empty() {
        return Err("no gradient control points found in the .ugr text".to_owned());
    }
    Ok(gradients)
}

/// Splits a `.ugr` line into `key=value` tokens, keeping quoted titles
/// (which may contain spaces) intact.
fn split_ugr_tokens(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for ch in line.chars() {
        match ch {
            '"' => {
                quoted = !quoted;
                current.push(ch);
            }
            ' ' | '\t' if !quoted => {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn parse_map(text: &str) -> Result<Vec<Gradient>, String> {
    let mut colours = Vec::new();
    for line in text.lines() {
        let numbers: Vec<u8> = line
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|s| !s.is_empty())
            .take(3)
            .filter_map(|s| s.parse::<u32>().ok())
            .map(|v| v.min(255) as u8)
            .collect();
        if numbers.len() == 3 {
            colours.push([numbers[0], numbers[1], numbers[2]]);
        }
    }
    if colours.len() < 2 {
        return Err(
            "expected a .ugr gradient or a .map palette with at least two `r g b` lines".to_owned(),
        );
    }
    let count = colours.len();
    let stops = colours
        .into_iter()
        .enumerate()
        .map(|(i, rgb)| GradientStop::from_u8(i as f32 / count as f32, rgb))
        .collect();
    Ok(vec![Gradient::new("Imported map", stops)])
}

fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

fn blend_hsl(a: [f32; 3], b: [f32; 3], fraction: f32, mode: Interpolation) -> [f32; 3] {
    let ha = rgb_to_hsl(a);
    let hb = rgb_to_hsl(b);
    // Grey endpoints have no hue; borrow the other side's so the sweep does
    // not spin through red.
    let hue_a = if ha[1] < 1e-4 { hb[0] } else { ha[0] };
    let hue_b = if hb[1] < 1e-4 { hue_a } else { hb[0] };
    let mut delta = hue_b - hue_a;
    delta -= delta.round();
    if mode == Interpolation::HslLong && delta.abs() > 1e-4 {
        delta -= delta.signum();
    }
    let hue = wrap_unit(hue_a + delta * fraction);
    hsl_to_rgb([
        hue,
        ha[1] + (hb[1] - ha[1]) * fraction,
        ha[2] + (hb[2] - ha[2]) * fraction,
    ])
}

pub(crate) fn rgb_to_hsl(rgb: [f32; 3]) -> [f32; 3] {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let lightness = 0.5 * (max + min);
    let chroma = max - min;
    if chroma < 1e-6 {
        return [0.0, 0.0, lightness];
    }
    let saturation = chroma / (1.0 - (2.0 * lightness - 1.0).abs()).max(1e-6);
    let hue = if max == rgb[0] {
        ((rgb[1] - rgb[2]) / chroma).rem_euclid(6.0)
    } else if max == rgb[1] {
        (rgb[2] - rgb[0]) / chroma + 2.0
    } else {
        (rgb[0] - rgb[1]) / chroma + 4.0
    } / 6.0;
    [hue, saturation.min(1.0), lightness]
}

pub(crate) fn hsl_to_rgb(hsl: [f32; 3]) -> [f32; 3] {
    let [h, s, l] = hsl;
    let chroma = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = wrap_unit(h) * 6.0;
    let x = chroma * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let m = l - 0.5 * chroma;
    [
        (r + m).clamp(0.0, 1.0),
        (g + m).clamp(0.0, 1.0),
        (b + m).clamp(0.0, 1.0),
    ]
}

fn wrap_unit(value: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let wrapped = value - value.floor();
    if wrapped >= 1.0 { 0.0 } else { wrapped }
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

pub(crate) mod presets {
    use super::{Gradient, GradientStop, Interpolation};

    pub(crate) const NAMES: [&str; 10] = [
        "Ultramarine",
        "Classic",
        "Ember",
        "Ocean",
        "Grayscale",
        "Spectrum",
        "Forest",
        "Neon",
        "Sepia",
        "Ice",
    ];

    pub(crate) fn default_gradient() -> Gradient {
        by_name("Ultramarine").unwrap()
    }

    pub(crate) fn by_name(name: &str) -> Option<Gradient> {
        let stops = |list: &[(f32, [u8; 3])]| -> Vec<GradientStop> {
            list.iter()
                .map(|(p, rgb)| GradientStop::from_u8(*p, *rgb))
                .collect()
        };
        let mut gradient = match name {
            // Deep blue through white to orange — the familiar look of
            // smooth escape-time renders.
            "Ultramarine" => Gradient::new(
                name,
                stops(&[
                    (0.0, [0, 7, 100]),
                    (0.16, [32, 107, 203]),
                    (0.42, [237, 255, 255]),
                    (0.6425, [255, 170, 0]),
                    (0.8575, [0, 2, 0]),
                ]),
            ),
            // The cosine palette IteraScope used before gradients existed.
            "Classic" => {
                let list: Vec<GradientStop> = (0..16)
                    .map(|i| {
                        let t = i as f32 / 16.0;
                        let a = [0.47, 0.49, 0.52];
                        let b = [0.42, 0.39, 0.36];
                        let c = [1.00, 0.82, 0.68];
                        let d = [0.06, 0.18, 0.36];
                        let mut colour = [0.0; 3];
                        for k in 0..3 {
                            colour[k] = (a[k]
                                + b[k] * (std::f32::consts::TAU * (c[k] * t + d[k])).cos())
                            .clamp(0.0, 1.0);
                        }
                        GradientStop::new(t, colour)
                    })
                    .collect();
                let mut g = Gradient::new(name, list);
                g.smooth = true;
                g
            }
            "Ember" => Gradient::new(
                name,
                stops(&[
                    (0.0, [8, 4, 10]),
                    (0.2, [120, 18, 20]),
                    (0.45, [240, 110, 20]),
                    (0.65, [255, 230, 120]),
                    (0.8, [255, 255, 255]),
                    (0.92, [120, 60, 30]),
                ]),
            ),
            "Ocean" => Gradient::new(
                name,
                stops(&[
                    (0.0, [2, 8, 30]),
                    (0.25, [10, 60, 120]),
                    (0.5, [40, 170, 200]),
                    (0.7, [220, 250, 240]),
                    (0.85, [20, 110, 140]),
                ]),
            ),
            "Grayscale" => Gradient::new(name, stops(&[(0.0, [0, 0, 0]), (0.5, [255, 255, 255])])),
            "Spectrum" => {
                let mut g = Gradient::new(
                    name,
                    stops(&[
                        (0.0, [255, 0, 0]),
                        (1.0 / 3.0, [0, 255, 0]),
                        (2.0 / 3.0, [0, 0, 255]),
                    ]),
                );
                g.interpolation = Interpolation::HslShort;
                g
            }
            "Forest" => Gradient::new(
                name,
                stops(&[
                    (0.0, [6, 14, 8]),
                    (0.3, [30, 90, 40]),
                    (0.55, [150, 200, 90]),
                    (0.7, [250, 245, 200]),
                    (0.85, [90, 60, 30]),
                ]),
            ),
            "Neon" => Gradient::new(
                name,
                stops(&[
                    (0.0, [10, 0, 20]),
                    (0.25, [255, 0, 140]),
                    (0.5, [0, 230, 255]),
                    (0.75, [255, 240, 0]),
                ]),
            ),
            "Sepia" => Gradient::new(
                name,
                stops(&[
                    (0.0, [30, 20, 10]),
                    (0.4, [140, 100, 60]),
                    (0.7, [240, 225, 190]),
                    (0.9, [90, 60, 35]),
                ]),
            ),
            "Ice" => Gradient::new(
                name,
                stops(&[
                    (0.0, [10, 20, 40]),
                    (0.3, [70, 120, 190]),
                    (0.55, [200, 235, 255]),
                    (0.75, [255, 255, 255]),
                    (0.9, [120, 170, 220]),
                ]),
            ),
            _ => return None,
        };
        if matches!(name, "Ember" | "Ocean" | "Forest" | "Ice" | "Sepia") {
            gradient.smooth = true;
        }
        Some(gradient)
    }
}

// ---------------------------------------------------------------------------
// Colouring algorithms
// ---------------------------------------------------------------------------

/// What a pixel's orbit statistics are reduced to before the gradient lookup.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ColouringAlgorithm {
    /// Escape or convergence time, fractional when smoothing is on.
    #[default]
    Iteration,
    /// Argument of the final `z`: continuous, or quantised into sectors
    /// (two sectors is classic binary decomposition).
    Decomposition,
    /// Triangle-inequality average of the orbit.
    TriangleInequality,
    /// Average of a sine of the orbit's arguments (stripe average colouring).
    Stripes,
    /// Exterior distance estimate in pixels (polynomial families).
    DistanceEstimate,
    /// Smallest distance from the orbit to a trap shape.
    OrbitTrap,
    /// A single colour.
    Solid,
}

impl ColouringAlgorithm {
    pub(crate) const ALL: [Self; 7] = [
        Self::Iteration,
        Self::Decomposition,
        Self::TriangleInequality,
        Self::Stripes,
        Self::DistanceEstimate,
        Self::OrbitTrap,
        Self::Solid,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Iteration => "Iteration count",
            Self::Decomposition => "Decomposition (angle)",
            Self::TriangleInequality => "Triangle inequality average",
            Self::Stripes => "Stripe average",
            Self::DistanceEstimate => "Distance estimate",
            Self::OrbitTrap => "Orbit trap",
            Self::Solid => "Solid colour",
        }
    }

    pub(crate) const fn description(self) -> &'static str {
        match self {
            Self::Iteration => "Gradient position advances with the (smoothed) iteration count.",
            Self::Decomposition => {
                "Colour by the argument of the final z; sectors = 0 is continuous, 2 is binary decomposition. Separates Newton and Nova basins."
            }
            Self::TriangleInequality => {
                "Average over the orbit of where |z| falls between the triangle-inequality bounds; a large bailout gives the smoothest result."
            }
            Self::Stripes => {
                "Average of a sine of the orbit's argument; the frequency sets the number of stripes per turn."
            }
            Self::DistanceEstimate => {
                "Exterior distance to the set in pixels, from the orbit's derivative. Exact for the quadratic, Multibrot and lambda families; others fall back to the iteration count."
            }
            Self::OrbitTrap => {
                "Closest approach of the orbit to a trap shape in the dynamical plane."
            }
            Self::Solid => "A single colour, independent of the orbit.",
        }
    }

    const fn code(self) -> u32 {
        match self {
            Self::Iteration => 0,
            Self::Decomposition => 1,
            Self::TriangleInequality => 2,
            Self::Stripes => 3,
            Self::DistanceEstimate => 4,
            Self::OrbitTrap => 5,
            Self::Solid => 6,
        }
    }

    /// A density that puts the algorithm's typical value range on a useful
    /// scale when it is first selected.
    pub(crate) const fn default_density(self) -> f32 {
        match self {
            Self::Iteration => 0.035,
            Self::Decomposition => 1.0,
            Self::TriangleInequality => 1.0,
            Self::Stripes => 1.0,
            Self::DistanceEstimate => 0.25,
            Self::OrbitTrap => 1.0,
            Self::Solid => 1.0,
        }
    }

    pub(crate) const fn default_transfer(self) -> Transfer {
        match self {
            Self::DistanceEstimate => Transfer::Log,
            Self::OrbitTrap => Transfer::Sqrt,
            _ => Transfer::Linear,
        }
    }
}

/// Curve applied to the algorithm's value before density and offset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Transfer {
    #[default]
    Linear,
    Sqrt,
    CubeRoot,
    Log,
}

impl Transfer {
    pub(crate) const ALL: [Self; 4] = [Self::Linear, Self::Sqrt, Self::CubeRoot, Self::Log];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Sqrt => "Square root",
            Self::CubeRoot => "Cube root",
            Self::Log => "Logarithm",
        }
    }

    const fn code(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::Sqrt => 1,
            Self::CubeRoot => 2,
            Self::Log => 3,
        }
    }
}

/// Shape of the orbit trap, centred at [`ColouringSide::trap_centre`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TrapShape {
    /// Distance to the centre point.
    #[default]
    Point,
    /// Distance to the nearer of the two axes through the centre.
    Cross,
    /// Distance to a circle of the given radius.
    Circle,
    /// Distance to the boundary of an axis-aligned square.
    Square,
    /// Distance to the horizontal line through the centre.
    Horizontal,
    /// Distance to the vertical line through the centre.
    Vertical,
}

impl TrapShape {
    pub(crate) const ALL: [Self; 6] = [
        Self::Point,
        Self::Cross,
        Self::Circle,
        Self::Square,
        Self::Horizontal,
        Self::Vertical,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Point => "Point",
            Self::Cross => "Cross",
            Self::Circle => "Circle",
            Self::Square => "Square",
            Self::Horizontal => "Horizontal line",
            Self::Vertical => "Vertical line",
        }
    }

    const fn code(self) -> u32 {
        match self {
            Self::Point => 0,
            Self::Cross => 1,
            Self::Circle => 2,
            Self::Square => 3,
            Self::Horizontal => 4,
            Self::Vertical => 5,
        }
    }

    pub(crate) const fn uses_size(self) -> bool {
        matches!(self, Self::Circle | Self::Square)
    }
}

/// Settings of one colouring algorithm — used once for the outside (escaped
/// or converged orbits) and once for the inside (bounded orbits).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ColouringSide {
    pub(crate) algorithm: ColouringAlgorithm,
    pub(crate) transfer: Transfer,
    /// Gradient cycles per unit of transferred value.
    pub(crate) density: f32,
    /// Gradient position added to every pixel, in turns.
    pub(crate) offset: f32,
    /// Fractional iteration counts (escape- or convergence-time smoothing).
    pub(crate) smooth: bool,
    /// Decomposition sectors; 0 keeps the angle continuous.
    pub(crate) sectors: u32,
    /// Stripes per turn of the argument.
    pub(crate) stripe_frequency: f32,
    pub(crate) trap_shape: TrapShape,
    pub(crate) trap_centre: [f32; 2],
    pub(crate) trap_size: f32,
    /// 0 = flat; 1 = fully darken slowly escaping or converging pixels
    /// (the classic root-basin look when combined with decomposition).
    pub(crate) shading: f32,
    /// Colour for [`ColouringAlgorithm::Solid`], sRGB.
    pub(crate) solid: [f32; 3],
}

impl Default for ColouringSide {
    fn default() -> Self {
        Self {
            algorithm: ColouringAlgorithm::Iteration,
            transfer: Transfer::Linear,
            density: ColouringAlgorithm::Iteration.default_density(),
            offset: 0.0,
            smooth: true,
            sectors: 0,
            stripe_frequency: 5.0,
            trap_shape: TrapShape::Point,
            trap_centre: [0.0, 0.0],
            trap_size: 0.5,
            shading: 0.0,
            solid: [0.025, 0.040, 0.058],
        }
    }
}

impl ColouringSide {
    /// The default inside colouring: a dark solid, as in Ultra Fractal.
    pub(crate) fn default_inside() -> Self {
        Self {
            algorithm: ColouringAlgorithm::Solid,
            ..Self::default()
        }
    }

    /// The default for convergence families (Newton, Nova, Magnet): basins
    /// by the argument of the limit, darkened by convergence time.
    pub(crate) fn default_basins() -> Self {
        Self {
            algorithm: ColouringAlgorithm::Decomposition,
            density: 1.0,
            sectors: 0,
            shading: 0.75,
            ..Self::default()
        }
    }

    /// Switches algorithm and adopts its natural density and transfer.
    pub(crate) fn set_algorithm(&mut self, algorithm: ColouringAlgorithm) {
        if self.algorithm == algorithm {
            return;
        }
        self.algorithm = algorithm;
        self.density = algorithm.default_density();
        self.transfer = algorithm.default_transfer();
    }

    pub(crate) fn validate(&self, name: &str) -> Result<(), String> {
        if !self.density.is_finite() || !(1e-6..=1e6).contains(&self.density) {
            return Err(format!(
                "{name}.density must be finite and between 1e-6 and 1e6"
            ));
        }
        if !self.offset.is_finite() {
            return Err(format!("{name}.offset must be finite"));
        }
        if self.sectors > 4096 {
            return Err(format!("{name}.sectors must be at most 4096"));
        }
        if !self.stripe_frequency.is_finite() || !(0.0..=1000.0).contains(&self.stripe_frequency) {
            return Err(format!(
                "{name}.stripe_frequency must be between 0 and 1000"
            ));
        }
        if !self.trap_centre[0].is_finite()
            || !self.trap_centre[1].is_finite()
            || !self.trap_size.is_finite()
            || self.trap_size < 0.0
        {
            return Err(format!(
                "{name}.trap_centre and trap_size must be finite and non-negative"
            ));
        }
        if !self.shading.is_finite() || !(0.0..=1.0).contains(&self.shading) {
            return Err(format!("{name}.shading must be between 0 and 1"));
        }
        if self
            .solid
            .iter()
            .any(|c| !c.is_finite() || !(0.0..=1.0).contains(c))
        {
            return Err(format!("{name}.solid components must be between 0 and 1"));
        }
        Ok(())
    }

    fn gpu_words(&self) -> [[f32; 4]; 4] {
        [
            [
                self.algorithm.code() as f32,
                self.transfer.code() as f32,
                self.density,
                self.offset,
            ],
            [
                self.smooth as u8 as f32,
                self.sectors as f32,
                self.stripe_frequency,
                self.trap_shape.code() as f32,
            ],
            [
                self.trap_centre[0],
                self.trap_centre[1],
                self.trap_size,
                self.shading,
            ],
            [self.solid[0], self.solid[1], self.solid[2], 0.0],
        ]
    }
}

/// The whole colour stage of one image (later: one layer).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Colouring {
    pub(crate) gradient: Gradient,
    pub(crate) outside: ColouringSide,
    pub(crate) inside: ColouringSide,
}

impl Default for Colouring {
    fn default() -> Self {
        Self {
            gradient: Gradient::default(),
            outside: ColouringSide::default(),
            inside: ColouringSide::default_inside(),
        }
    }
}

/// Most layers one image may composite; must match `MAX_LAYERS` in
/// `fractal.wgsl`.
pub(crate) const MAX_LAYERS: usize = 8;
/// `vec4<f32>` words per layer in the uniform block: outside a–d,
/// inside a–d, blend.
const LAYER_WORDS: usize = 9;
/// Register-resident statistics accumulator slots in the shader; the first
/// `STATS_SLOTS` stats-bearing layers (in stack order) get one each, and
/// any further stats-bearing layer colours as if its statistics were
/// unavailable. Must match `MAX_STATS_SLOTS` in `fractal.wgsl`.
pub(crate) const STATS_SLOTS: usize = 4;
/// `vec4<f32>` words per stats slot: (layer, skip, flags, stripe
/// frequency), the outside trap (shape, centre, size), the inside trap.
const SLOT_WORDS: usize = 3;
/// Number of `vec4<f32>` words in the colouring uniform block; must match
/// `ColouringUniforms` in `fractal.wgsl`: header, needs union, the layer
/// array, then the stats-slot plans.
pub(crate) const GPU_WORDS: usize = 2 + MAX_LAYERS * LAYER_WORDS + STATS_SLOTS * SLOT_WORDS;
/// Index of the word holding the union of accumulator flags (trap, triangle
/// inequality, stripes, derivative).
pub(crate) const NEEDS_WORD: usize = 1;
/// First word of the stats-slot plans.
const SLOTS_BASE: usize = 2 + MAX_LAYERS * LAYER_WORDS;

impl Colouring {
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.gradient.stops.is_empty() {
            return Err("gradient needs at least one stop".to_owned());
        }
        if self.gradient.stops.len() > 4096 {
            return Err("gradient has too many stops (maximum 4096)".to_owned());
        }
        for stop in &self.gradient.stops {
            if !stop.position.is_finite()
                || stop
                    .colour
                    .iter()
                    .any(|c| !c.is_finite() || !(0.0..=1.0).contains(c))
            {
                return Err(
                    "gradient stops need finite positions and colours between 0 and 1".to_owned(),
                );
            }
        }
        if !self.gradient.rotation.is_finite() {
            return Err("gradient rotation must be finite".to_owned());
        }
        self.outside.validate("outside")?;
        self.inside.validate("inside")
    }

    /// Whether any selected algorithm needs the per-iteration accumulator.
    fn needs(&self, algorithm: ColouringAlgorithm) -> bool {
        self.outside.algorithm == algorithm || self.inside.algorithm == algorithm
    }

    /// Whether any selected algorithm feeds the per-iteration statistics
    /// accumulators — and therefore occupies one of the shader's
    /// [`STATS_SLOTS`] register slots.
    pub(crate) fn uses_statistics(&self) -> bool {
        self.needs(ColouringAlgorithm::OrbitTrap)
            || self.needs(ColouringAlgorithm::TriangleInequality)
            || self.needs(ColouringAlgorithm::Stripes)
    }
}

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

/// How a layer combines with the layers beneath it. The bottom layer's mode
/// is ignored (it composites over black by its opacity alone).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MergeMode {
    #[default]
    Normal,
    Add,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    Difference,
}

impl MergeMode {
    pub(crate) const ALL: [Self; 8] = [
        Self::Normal,
        Self::Add,
        Self::Multiply,
        Self::Screen,
        Self::Overlay,
        Self::Darken,
        Self::Lighten,
        Self::Difference,
    ];

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Add => "Add",
            Self::Multiply => "Multiply",
            Self::Screen => "Screen",
            Self::Overlay => "Overlay",
            Self::Darken => "Darken",
            Self::Lighten => "Lighten",
            Self::Difference => "Difference",
        }
    }

    pub(crate) const fn code(self) -> u32 {
        match self {
            Self::Normal => 0,
            Self::Add => 1,
            Self::Multiply => 2,
            Self::Screen => 3,
            Self::Overlay => 4,
            Self::Darken => 5,
            Self::Lighten => 6,
            Self::Difference => 7,
        }
    }
}

/// An arbitrary-precision location for a detached scene beyond the f64
/// handoff: exact decimal strings, captured from a deep view. The scene's
/// f64 fields then hold the projection for display.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct LayerSceneDeep {
    pub(crate) centre_re: String,
    pub(crate) centre_im: String,
    pub(crate) half_height: String,
    pub(crate) magnification_log10: f64,
    /// The Julia parameter at full precision; absent, the scene's f64
    /// parameter is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) julia: Option<(String, String)>,
}

impl LayerSceneDeep {
    /// The working precision for parsing this location.
    pub(crate) fn zoom_exponent(&self) -> u32 {
        (self.magnification_log10.ceil().max(0.0) as u32).saturating_add(40)
    }

    pub(crate) fn parse_view(&self) -> Result<crate::arbitrary::DeepView, String> {
        crate::arbitrary::DeepView::parse(
            &self.centre_re,
            &self.centre_im,
            &self.half_height,
            self.magnification_log10,
        )
    }
}

/// A layer's own scene: its formula, plane, parameter and location,
/// independent of the image the stack belongs to. Detached scenes render as
/// their own pass; locations are exact through the `f64`-reference range and,
/// with a captured [`LayerSceneDeep`], through arbitrary precision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct LayerScene {
    pub(crate) family: FractalFamily,
    /// Iterate from z₀ = pixel (dynamical plane) rather than treating the
    /// pixel as the parameter.
    pub(crate) dynamical: bool,
    /// The Julia parameter (dynamical plane) in the complex plane.
    pub(crate) julia_c: [f64; 2],
    pub(crate) centre: [f64; 2],
    pub(crate) half_height: f64,
    pub(crate) iterations: u32,
    pub(crate) bailout: f32,
    /// Family-specific settings, in the experiment document's shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) family_parameters: Option<crate::experiment::FamilyParametersDocument>,
    /// The location at arbitrary precision, when captured beyond the f64
    /// handoff; the f64 fields then hold its projection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) deep: Option<LayerSceneDeep>,
}

impl Default for LayerScene {
    fn default() -> Self {
        let family = FractalFamily::Quadratic;
        let view = family.default_dynamical_view();
        Self {
            family,
            dynamical: true,
            julia_c: family.default_parameter(),
            centre: view.centre,
            half_height: view.half_height,
            iterations: 256,
            bailout: 4.0,
            family_parameters: None,
            deep: None,
        }
    }
}

impl LayerScene {
    /// The smallest half-height a detached scene may use: the `f64`
    /// reference-orbit path is exact to the arbitrary-precision handoff.
    pub(crate) const MIN_HALF_HEIGHT: f64 = 1.45 / crate::arbitrary::ARBITRARY_HANDOFF_ZOOM;

    pub(crate) fn validate(&self, index: usize) -> Result<(), String> {
        if !self.centre[0].is_finite()
            || !self.centre[1].is_finite()
            || !self.julia_c[0].is_finite()
            || !self.julia_c[1].is_finite()
        {
            return Err(format!("layer {index}: scene coordinates must be finite"));
        }
        if !self.half_height.is_finite()
            || !(Self::MIN_HALF_HEIGHT..=1e6).contains(&self.half_height)
        {
            return Err(format!(
                "layer {index}: scene half_height must be between {:.3e} and 1e6 (detached layers stop at the arbitrary-precision handoff)",
                Self::MIN_HALF_HEIGHT
            ));
        }
        if !(1..=50_000).contains(&self.iterations) {
            return Err(format!(
                "layer {index}: scene iterations must be between 1 and 50000"
            ));
        }
        if !self.bailout.is_finite() || !(2.0..=1e10).contains(&self.bailout) {
            return Err(format!(
                "layer {index}: scene bailout must be between 2 and 1e10"
            ));
        }
        if let Some(parameters) = &self.family_parameters {
            parameters
                .validate()
                .map_err(|error| format!("layer {index}: {error}"))?;
        }
        if let Some(deep) = &self.deep {
            if !self.family.supports_deep_zoom() {
                return Err(format!(
                    "layer {index}: {} scenes cannot go beyond f32/f64 range",
                    self.family.name()
                ));
            }
            deep.parse_view()
                .map_err(|error| format!("layer {index}: deep location: {error}"))?;
        }
        Ok(())
    }

    /// The runtime family parameters this scene describes.
    pub(crate) fn parameters(&self) -> crate::family::FamilyParameters {
        let mut parameters = crate::family::FamilyParameters::default();
        if let Some(document) = &self.family_parameters {
            document.apply_to(&mut parameters);
        }
        parameters
    }
}

/// One layer of the composited image: a complete colour stage plus how it
/// merges with the layers beneath. By default layers share the image's
/// location, family and iteration — the orbit is computed once per pixel
/// and coloured once per layer. A layer with its own [`LayerScene`] instead
/// renders as a separate pass with its own formula and location.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Layer {
    pub(crate) name: String,
    pub(crate) visible: bool,
    pub(crate) opacity: f32,
    pub(crate) merge_mode: MergeMode,
    /// A mask paints nothing: the luminance of its colour, scaled by its
    /// opacity, multiplies the opacity of the next non-mask layer above it.
    pub(crate) mask: bool,
    /// Iterations the layer's orbit-trap, stripe and triangle-inequality
    /// accumulators ignore at the start of every orbit. Deep-zoom pixels
    /// share hundreds of identical leading iterations, which flattens those
    /// colourings; skipping the shared prefix restores their variety.
    pub(crate) skip_iterations: u32,
    /// This layer's own formula and location; `None` shares the image's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) scene: Option<LayerScene>,
    pub(crate) colouring: Colouring,
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            name: "Layer".to_owned(),
            visible: true,
            opacity: 1.0,
            merge_mode: MergeMode::Normal,
            mask: false,
            skip_iterations: 0,
            scene: None,
            colouring: Colouring::default(),
        }
    }
}

impl Layer {
    pub(crate) fn validate(&self, index: usize) -> Result<(), String> {
        if !self.opacity.is_finite() || !(0.0..=1.0).contains(&self.opacity) {
            return Err(format!("layer {index}: opacity must be between 0 and 1"));
        }
        if self.skip_iterations >= 50_000 {
            return Err(format!(
                "layer {index}: skip_iterations must be below 50000"
            ));
        }
        if let Some(scene) = &self.scene {
            scene.validate(index)?;
        }
        self.colouring
            .validate()
            .map_err(|error| format!("layer {index}: {error}"))
    }
}

/// The image's layer stack, bottom first. Always holds at least one layer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerStack {
    pub(crate) layers: Vec<Layer>,
    pub(crate) active: usize,
}

impl Default for LayerStack {
    fn default() -> Self {
        Self::single(Colouring::default())
    }
}

impl LayerStack {
    pub(crate) fn single(colouring: Colouring) -> Self {
        Self {
            layers: vec![Layer {
                colouring,
                ..Layer::default()
            }],
            active: 0,
        }
    }

    pub(crate) fn from_layers(layers: Vec<Layer>) -> Self {
        let mut stack = Self { layers, active: 0 };
        if stack.layers.is_empty() {
            stack.layers.push(Layer::default());
        }
        stack.layers.truncate(MAX_LAYERS);
        stack
    }

    pub(crate) fn active_layer(&self) -> &Layer {
        &self.layers[self.active.min(self.layers.len() - 1)]
    }

    pub(crate) fn active_layer_mut(&mut self) -> &mut Layer {
        let index = self.active.min(self.layers.len() - 1);
        self.active = index;
        &mut self.layers[index]
    }

    pub(crate) fn active_colouring(&self) -> &Colouring {
        &self.active_layer().colouring
    }

    pub(crate) fn active_colouring_mut(&mut self) -> &mut Colouring {
        &mut self.active_layer_mut().colouring
    }

    /// The layers that contribute to the image (visible and not fully
    /// transparent), bottom first — the order the shader composites and the
    /// order the gradient tables are packed. Skipping transparent layers
    /// also keeps their accumulators out of the needs union, so an opacity-0
    /// layer costs nothing.
    pub(crate) fn visible(&self) -> impl Iterator<Item = &Layer> {
        self.layers
            .iter()
            .filter(|layer| layer.visible && layer.opacity > 0.0)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn validate(&self) -> Result<(), String> {
        if self.layers.is_empty() {
            return Err("the layer stack needs at least one layer".to_owned());
        }
        if self.layers.len() > MAX_LAYERS {
            return Err(format!("at most {MAX_LAYERS} layers are supported"));
        }
        for (index, layer) in self.layers.iter().enumerate() {
            layer.validate(index)?;
        }
        Ok(())
    }

    /// The gradient tables of the visible layers, concatenated bottom first
    /// for the shader's single lookup buffer.
    pub(crate) fn lookup_tables(&self) -> Vec<[f32; 4]> {
        let mut entries = Vec::with_capacity(LOOKUP_TABLE_LEN);
        for layer in self.visible() {
            entries.extend(layer.colouring.gradient.lookup_table());
        }
        if entries.is_empty() {
            entries.extend(Gradient::default().lookup_table());
        }
        entries
    }

    /// The uniform block of one layer rendered alone (a detached-scene
    /// pass): full opacity, Normal mode, no mask — merging happens in the
    /// compositing pass afterwards. `gradient_base` is the layer's table
    /// offset (in entries) inside the shared gradient buffer.
    pub(crate) fn single_pass_words(
        layer: &Layer,
        pixel_log: f32,
        gradient_base: usize,
    ) -> [[f32; 4]; GPU_WORDS] {
        let mut alone = LayerStack::single(layer.colouring.clone());
        alone.layers[0].skip_iterations = layer.skip_iterations;
        let mut words = alone.gpu_words(pixel_log);
        words[0][3] = gradient_base as f32;
        words
    }

    /// The uniform block for the shader. `pixel_log` is the natural log of
    /// one pixel's height in world units (distance estimates are expressed
    /// in pixels). Only visible layers are uploaded; the bottom layer's
    /// merge mode is forced in the shader, not here.
    pub(crate) fn gpu_words(&self, pixel_log: f32) -> [[f32; 4]; GPU_WORDS] {
        let mut words = [[0.0f32; 4]; GPU_WORDS];
        // Unused stats slots carry the no-layer sentinel.
        for slot in 0..STATS_SLOTS {
            words[SLOTS_BASE + slot * SLOT_WORDS][0] = 255.0;
        }
        let mut needs = [false; 4];
        let mut count = 0usize;
        let mut slot = 0usize;
        for layer in self.visible().take(MAX_LAYERS) {
            let colouring = &layer.colouring;
            let outside = colouring.outside.gpu_words();
            let inside = colouring.inside.gpu_words();
            let base = 2 + count * LAYER_WORDS;
            words[base..base + 4].copy_from_slice(&outside);
            words[base + 4..base + 8].copy_from_slice(&inside);
            // The two spare `d.w` slots carry the layer's mask flag and its
            // accumulator skip; see `outside_d`/`inside_d` in fractal.wgsl.
            words[base + 3][3] = layer.mask as u8 as f32;
            words[base + 7][3] = layer.skip_iterations.min(49_999) as f32;
            let needs_trap = colouring.needs(ColouringAlgorithm::OrbitTrap);
            let needs_stripes = colouring.needs(ColouringAlgorithm::Stripes);
            let needs_tia = colouring.needs(ColouringAlgorithm::TriangleInequality);
            words[base + 8] = [
                layer.merge_mode.code() as f32,
                layer.opacity,
                needs_trap as u8 as f32,
                needs_stripes as u8 as f32,
            ];
            needs[0] |= needs_trap;
            needs[1] |= needs_tia;
            needs[2] |= needs_stripes;
            needs[3] |= colouring.needs(ColouringAlgorithm::DistanceEstimate);
            // Assign the layer a stats slot: (layer, skip, flag bits, stripe
            // frequency), then the outside and inside trap descriptions —
            // everything the per-iteration accumulator update needs, so the
            // shader never touches the per-layer words in its hot loop.
            if slot < STATS_SLOTS && (needs_trap || needs_stripes || needs_tia) {
                let slot_base = SLOTS_BASE + slot * SLOT_WORDS;
                let frequency = if colouring.outside.algorithm == ColouringAlgorithm::Stripes {
                    outside[1][2]
                } else {
                    inside[1][2]
                };
                let flags =
                    u8::from(needs_trap) | (u8::from(needs_tia) << 1) | (u8::from(needs_stripes) << 2);
                words[slot_base] = [
                    count as f32,
                    layer.skip_iterations.min(49_999) as f32,
                    f32::from(flags),
                    frequency,
                ];
                words[slot_base + 1] = [outside[1][3], outside[2][0], outside[2][1], outside[2][2]];
                words[slot_base + 2] = [inside[1][3], inside[2][0], inside[2][1], inside[2][2]];
                slot += 1;
            }
            count += 1;
        }
        if count == 0 {
            // Nothing visible: upload one fully transparent layer so the
            // image renders black rather than stale.
            words[2 + 8] = [0.0, 0.0, 0.0, 0.0];
            count = 1;
        }
        words[0] = [count as f32, pixel_log, LOOKUP_TABLE_LEN as f32, 0.0];
        // words[0][3] stays 0: the shared-scene pass reads the gradient
        // table from its start.
        words[NEEDS_WORD] = [
            needs[0] as u8 as f32,
            needs[1] as u8 as f32,
            needs[2] as u8 as f32,
            needs[3] as u8 as f32,
        ];
        words
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-3)
    }

    #[test]
    fn gradient_interpolates_and_wraps() {
        let gradient = Gradient::new(
            "two",
            vec![
                GradientStop::new(0.0, [0.0, 0.0, 0.0]),
                GradientStop::new(0.5, [1.0, 1.0, 1.0]),
            ],
        );
        assert!(close(gradient.colour_at(0.25), [0.5; 3]));
        assert!(close(gradient.colour_at(0.75), [0.5; 3]));
        assert!(close(gradient.colour_at(1.25), [0.5; 3]));
        assert!(close(gradient.colour_at(-0.5), [1.0; 3]));
        assert!(close(gradient.colour_at(0.0), [0.0; 3]));
    }

    #[test]
    fn rotation_shifts_colours_towards_higher_positions() {
        let mut gradient = Gradient::new(
            "two",
            vec![
                GradientStop::new(0.0, [0.0, 0.0, 0.0]),
                GradientStop::new(0.5, [1.0, 1.0, 1.0]),
            ],
        );
        gradient.rotation = 0.25;
        assert!(close(gradient.colour_at(0.25), [0.0; 3]));
        assert!(close(gradient.colour_at(0.75), [1.0; 3]));
    }

    #[test]
    fn lookup_table_matches_direct_evaluation() {
        let gradient = Gradient::default();
        let table = gradient.lookup_table();
        assert_eq!(table.len(), LOOKUP_TABLE_LEN);
        for (i, entry) in table.iter().enumerate().step_by(97) {
            let direct = gradient.colour_at(i as f32 / LOOKUP_TABLE_LEN as f32);
            assert!(close([entry[0], entry[1], entry[2]], direct));
            assert_eq!(entry[3], 1.0);
        }
    }

    #[test]
    fn hsl_round_trips() {
        for rgb in [
            [0.2, 0.6, 0.9],
            [0.9, 0.1, 0.1],
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ] {
            assert!(close(hsl_to_rgb(rgb_to_hsl(rgb)), rgb), "{rgb:?}");
        }
    }

    #[test]
    fn hsl_short_path_does_not_pass_through_the_complement() {
        let gradient = {
            let mut g = Gradient::new(
                "hue",
                vec![
                    GradientStop::new(0.0, [1.0, 0.0, 0.0]),
                    GradientStop::new(0.5, [1.0, 0.0, 1.0]),
                ],
            );
            g.interpolation = Interpolation::HslShort;
            g
        };
        // Halfway from red to magenta the short path is around hue 330°
        // (pinkish red), never green.
        let mid = gradient.colour_at(0.25);
        assert!(mid[0] > 0.9 && mid[1] < 0.1, "{mid:?}");
    }

    #[test]
    fn ugr_round_trip_preserves_stops_and_colours() {
        let mut original = presets::by_name("Ember").unwrap();
        original.rotation = 0.1;
        let text = original.to_ugr();
        let parsed = Gradient::parse(&text).unwrap();
        assert_eq!(parsed.len(), 1);
        let gradient = &parsed[0];
        assert_eq!(gradient.name, "Ember");
        assert!(gradient.smooth);
        assert_eq!(gradient.stops.len(), original.stops.len());
        assert!((gradient.rotation - 0.1).abs() < 2.0 / 400.0);
        for (a, b) in gradient.stops.iter().zip(&original.stops) {
            assert!((a.position - b.position).abs() < 1.0 / 400.0);
            assert!(close(a.colour, b.colour));
        }
    }

    #[test]
    fn ugr_parser_reads_ultra_fractal_layout() {
        let text = r#"Blues {
gradient:
  title="Blues" smooth=yes rotation=20
  index=0 color=8388608
  index=200 color=16777215
  index=399 color=255
opacity:
  smooth=no
  index=0 opacity=255
}
Reds {
gradient:
  title="Reds and more" smooth=no
  index=0 color=255
  index=100 color=65280
}
"#;
        let gradients = Gradient::parse(text).unwrap();
        assert_eq!(gradients.len(), 2);
        assert_eq!(gradients[0].name, "Blues");
        assert_eq!(gradients[0].stops.len(), 3);
        // 8388608 = 0x800000: blue 128 in COLORREF byte order.
        assert!(close(
            gradients[0].stops[0].colour,
            [0.0, 0.0, 128.0 / 255.0]
        ));
        assert!(close(gradients[0].stops[2].colour, [1.0, 0.0, 0.0]));
        assert!((gradients[0].rotation - 0.05).abs() < 1e-6);
        assert_eq!(gradients[1].name, "Reds and more");
        assert!(close(gradients[1].stops[1].colour, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn map_parser_reads_fractint_palettes() {
        let text = (0..256)
            .map(|i| format!("{i} {} {}", 255 - i, i / 2))
            .collect::<Vec<_>>()
            .join("\n");
        let gradients = Gradient::parse(&text).unwrap();
        assert_eq!(gradients[0].stops.len(), 256);
        assert!(close(
            gradients[0].stops[255].colour,
            [1.0, 0.0, 127.0 / 255.0]
        ));
        assert!(Gradient::parse("nothing here").is_err());
    }

    #[test]
    fn insert_and_remove_keep_the_gradient_sorted_and_non_empty() {
        let mut gradient = Gradient::new(
            "two",
            vec![
                GradientStop::new(0.0, [0.0, 0.0, 0.0]),
                GradientStop::new(0.5, [1.0, 1.0, 1.0]),
            ],
        );
        let index = gradient.insert_stop(0.25);
        assert_eq!(index, 1);
        assert!(close(gradient.stops[1].colour, [0.5; 3]));
        gradient.remove_stop(0);
        gradient.remove_stop(0);
        gradient.remove_stop(0);
        assert_eq!(gradient.stops.len(), 1);
    }

    #[test]
    fn every_preset_parses_and_random_is_deterministic() {
        for name in presets::NAMES {
            let gradient = presets::by_name(name).unwrap();
            assert!(gradient.stops.len() >= 2, "{name}");
            assert_eq!(gradient.name, name);
        }
        assert_eq!(Gradient::random(7), Gradient::random(7));
        assert_ne!(Gradient::random(7), Gradient::random(8));
    }

    #[test]
    fn gpu_words_encode_the_visible_layers_and_the_needs_union() {
        let mut bottom = Colouring::default();
        bottom.inside.set_algorithm(ColouringAlgorithm::OrbitTrap);
        assert_eq!(bottom.inside.transfer, Transfer::Sqrt);
        let mut top = Layer {
            opacity: 0.5,
            merge_mode: MergeMode::Screen,
            ..Layer::default()
        };
        top.colouring
            .outside
            .set_algorithm(ColouringAlgorithm::Stripes);
        let mut hidden = Layer {
            visible: false,
            ..Layer::default()
        };
        hidden
            .colouring
            .outside
            .set_algorithm(ColouringAlgorithm::DistanceEstimate);
        let mut stack = LayerStack::single(bottom);
        stack.layers.push(hidden);
        stack.layers.push(top);
        let words = stack.gpu_words(-3.0);
        // Header: two visible layers, pixel log, entries per table.
        assert_eq!(words[0][0], 2.0);
        assert_eq!(words[0][1], -3.0);
        assert_eq!(words[0][2], LOOKUP_TABLE_LEN as f32);
        // Needs union: traps (bottom) and stripes (top); no distance,
        // because that layer is hidden.
        assert_eq!(words[NEEDS_WORD], [1.0, 0.0, 1.0, 0.0]);
        // Layer 0: outside iteration, inside trap; per-layer trap flag set.
        assert_eq!(words[2][0], 0.0);
        assert_eq!(words[2 + 4][0], 5.0);
        assert_eq!(words[2 + 8][2], 1.0);
        // Layer 1 (the visible top layer, packed second): stripes outside,
        // Screen at half opacity, stripes flag set, trap flag clear.
        assert_eq!(words[2 + 9][0], 3.0);
        assert_eq!(words[2 + 9 + 8], [3.0, 0.5, 0.0, 1.0]);
        stack.validate().unwrap();
        // The combined gradient table packs one table per visible layer.
        assert_eq!(stack.lookup_tables().len(), 2 * LOOKUP_TABLE_LEN);
    }

    #[test]
    fn mask_flag_and_skip_land_in_the_spare_d_words() {
        let mut mask = Layer {
            mask: true,
            opacity: 0.7,
            ..Layer::default()
        };
        mask.colouring
            .outside
            .set_algorithm(ColouringAlgorithm::OrbitTrap);
        mask.skip_iterations = 250;
        let mut stack = LayerStack::default();
        stack.layers.push(mask);
        let words = stack.gpu_words(0.0);
        // Layer 0: no mask, no skip.
        assert_eq!(words[2 + 3][3], 0.0);
        assert_eq!(words[2 + 7][3], 0.0);
        // Layer 1: mask flag in outside_d.w, skip in inside_d.w; the solid
        // colours in d.rgb are untouched.
        assert_eq!(words[2 + 9 + 3][3], 1.0);
        assert_eq!(words[2 + 9 + 7][3], 250.0);
        stack.validate().unwrap();
        stack.layers[1].skip_iterations = 60_000;
        assert!(stack.validate().is_err());
        stack.layers[1].skip_iterations = 0;

        // Round trip through JSON keeps the new fields; old documents
        // without them default to no mask, no skip.
        let json = serde_json::to_string(&stack.layers).unwrap();
        let back: Vec<Layer> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, stack.layers);
        let old: Layer = serde_json::from_str("{\"name\":\"x\"}").unwrap();
        assert!(!old.mask);
        assert_eq!(old.skip_iterations, 0);
    }

    #[test]
    fn deep_scene_locations_round_trip_and_validate() {
        let mut scene = LayerScene {
            family: FractalFamily::Quadratic,
            ..LayerScene::default()
        };
        scene.deep = Some(LayerSceneDeep {
            centre_re: "-7.45e-1".to_owned(),
            centre_im: "1.13e-1".to_owned(),
            half_height: "1.45e-200".to_owned(),
            magnification_log10: 200.0,
            julia: Some(("-7.45e-1".to_owned(), "1.13e-1".to_owned())),
        });
        scene.validate(0).unwrap();
        let json = serde_json::to_string(&scene).unwrap();
        let back: LayerScene = serde_json::from_str(&json).unwrap();
        assert_eq!(back, scene);
        assert_eq!(back.deep.as_ref().unwrap().zoom_exponent(), 240);
        back.deep.as_ref().unwrap().parse_view().unwrap();

        // A corrupt decimal fails validation with the layer index.
        scene.deep.as_mut().unwrap().centre_re = "not a number".to_owned();
        let error = scene.validate(3).unwrap_err();
        assert!(error.contains("layer 3"), "{error}");

        // Families without deep-zoom support cannot carry deep locations.
        let mut shallow = LayerScene {
            family: FractalFamily::Exponential,
            deep: Some(LayerSceneDeep {
                centre_re: "0".to_owned(),
                centre_im: "0".to_owned(),
                half_height: "1.45e-20".to_owned(),
                magnification_log10: 20.0,
                julia: None,
            }),
            ..LayerScene::default()
        };
        assert!(shallow.validate(0).is_err());
        shallow.deep = None;
        shallow.validate(0).unwrap();
    }

    #[test]
    fn layer_stack_guards_its_shape() {
        let mut stack = LayerStack::from_layers(Vec::new());
        assert_eq!(stack.layers.len(), 1);
        stack.active = 7;
        assert_eq!(stack.active_layer_mut().name, "Layer");
        assert_eq!(stack.active, 0);
        stack.layers = vec![Layer::default(); MAX_LAYERS + 2];
        assert!(stack.validate().is_err());
        let stack = LayerStack::from_layers(vec![Layer::default(); MAX_LAYERS + 2]);
        assert_eq!(stack.layers.len(), MAX_LAYERS);
        // All layers hidden: the uniforms still describe one (transparent)
        // layer so the shader has defined input.
        let mut hidden = LayerStack::default();
        hidden.layers[0].visible = false;
        let words = hidden.gpu_words(0.0);
        assert_eq!(words[0][0], 1.0);
        assert_eq!(words[2 + 8][1], 0.0);
        assert_eq!(hidden.lookup_tables().len(), LOOKUP_TABLE_LEN);
    }

    #[test]
    fn colouring_round_trips_through_json() {
        let mut colouring = Colouring::default();
        colouring.outside.set_algorithm(ColouringAlgorithm::Stripes);
        colouring.gradient = Gradient::random(3);
        let json = serde_json::to_string(&colouring).unwrap();
        let back: Colouring = serde_json::from_str(&json).unwrap();
        assert_eq!(back, colouring);
        // Missing fields take their defaults.
        let partial: Colouring =
            serde_json::from_str(r#"{"outside":{"algorithm":"stripes"}}"#).unwrap();
        assert_eq!(partial.outside.algorithm, ColouringAlgorithm::Stripes);
        assert_eq!(partial.inside, ColouringSide::default_inside());
    }
}
