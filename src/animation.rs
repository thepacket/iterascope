//! Zoom-path animation: a camera path through log-magnification, sampled
//! into frames for the image-sequence exporter.
//!
//! The base model is the one deep-zoom videos actually use: a fixed centre
//! (the deep target the user navigated to) and a magnification that moves
//! through decades of zoom at constant — optionally eased — logarithmic
//! speed. Because the centre never moves, one reference orbit serves every
//! frame: the arbitrary-precision orbit of the centre is re-described (new
//! scale mantissa and exponent) per frame instead of being rebuilt, so
//! exporting a 10^1000× dive costs one orbit, not one per frame.
//!
//! With two or more captured [`ZoomWaypoint`]s the camera instead flies a
//! [`CameraPath`]: the centre drifts between waypoints while the
//! magnification follows them. Each path segment anchors on its deeper
//! endpoint's reference orbit, and the drifting centre is expressed as a
//! small screen-space offset from that anchor — the same off-centre-reference
//! mechanism the tiled still exporter uses — so drift stays exact at any
//! depth, one reference orbit per segment.
//!
//! An optional gradient sweep advances both colouring offsets across the
//! animation for the classic slowly-turning-palette look.
//!
//! Parameter curves animate the dynamics themselves: the Julia parameter
//! morphs along a keyed path through the parameter plane, and continuous
//! family settings (Nova relaxation, the Mandelbox radii and scale) follow
//! keyed scalar curves. Animated dynamics invalidate the reference orbit,
//! so the exporter rebuilds it per frame from the frozen scene's recipe —
//! f64 below the arbitrary-precision handoff, full precision beyond it.

use crate::arbitrary::DeepReal;
use crate::family::{FamilyParameters, FractalFamily};
use serde::{Deserialize, Serialize};

pub(crate) const MAX_DIMENSION: u32 = 8_192;
/// Largest still-image dimension; images beyond `MAX_DIMENSION` render as
/// tiles around the same reference orbit.
pub(crate) const MAX_STILL_DIMENSION: u32 = 16_384;
pub(crate) const MAX_FRAMES: usize = 100_000;

/// The user-editable animation settings; the exporter freezes a copy when
/// rendering starts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ZoomAnimation {
    pub(crate) duration_seconds: f32,
    pub(crate) fps: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// log10 of the magnification of the first frame.
    pub(crate) start_magnification_log10: f64,
    /// log10 of the magnification of the last frame.
    pub(crate) end_magnification_log10: f64,
    /// Smoothstep ease-in/out of the logarithmic zoom speed.
    pub(crate) ease: bool,
    /// Gradient offset added over the whole animation, in turns.
    pub(crate) gradient_sweep_turns: f32,
    /// Invoke ffmpeg on the finished sequence when it is installed.
    pub(crate) encode_video: bool,
    /// Scale each frame's iteration budget with its magnification: the
    /// budget the user set applies to the deepest frame, and shallower
    /// frames — whose structure resolves in far fewer iterations — use
    /// proportionally fewer, which speeds long dives up considerably.
    pub(crate) scale_iterations: bool,
    /// Captured camera waypoints. With fewer than two the animation is the
    /// classic fixed-centre dive between the start and end exponents; with
    /// two or more, the camera flies the waypoint path and the exponents
    /// above are ignored.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) waypoints: Vec<ZoomWaypoint>,
    /// Morphs the Julia parameter along a keyed path (Julia-plane exports).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) julia_curve: Option<JuliaCurve>,
    /// Keyed curves over continuous family settings.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) family_curves: Vec<FamilyCurve>,
}

/// One waypoint of a keyframed camera path: a location and magnification
/// captured from a view. `centre` is the f64 projection used for display and
/// the shallow render paths; `exact` carries the full decimal centre when
/// the waypoint was captured beyond f64 resolution.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct ZoomWaypoint {
    pub(crate) magnification_log10: f64,
    pub(crate) centre: [f64; 2],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) exact: Option<(String, String)>,
}

pub(crate) const MAX_WAYPOINTS: usize = 64;
/// The farthest a segment's shallow endpoint may sit from its deep anchor,
/// in units of the shallow half-height: the offset reaches the shader as an
/// f32, whose resolution at this magnitude is still ~1/100 pixel.
const MAX_PAN_HALF_HEIGHTS: f64 = 20_000.0;
/// Perceptual exchange rate between panning and zooming, decades per
/// half-height: zooming at r decades/s moves a mid-frame pixel about as fast
/// as panning at r·ln 10 half-heights/s, so a pan of x half-heights reads
/// like x/ln 10 decades when apportioning time along the path.
const PAN_DECADES_PER_HALF_HEIGHT: f64 = std::f64::consts::LOG10_E;

impl Default for ZoomAnimation {
    fn default() -> Self {
        Self {
            duration_seconds: 12.0,
            fps: 30,
            width: 1920,
            height: 1080,
            start_magnification_log10: 0.0,
            end_magnification_log10: 3.0,
            ease: true,
            gradient_sweep_turns: 0.0,
            encode_video: true,
            scale_iterations: true,
            waypoints: Vec::new(),
            julia_curve: None,
            family_curves: Vec::new(),
        }
    }
}

pub(crate) const MAX_CURVE_KEYS: usize = 64;

/// One key of a scalar parameter curve: a value pinned at a fraction of the
/// animation's duration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct CurveKey {
    pub(crate) time: f32,
    pub(crate) value: f64,
}

/// One key of the Julia-parameter curve: the complex parameter at a time
/// fraction. The morph path is a (optionally eased) polyline through the
/// parameter plane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct JuliaKey {
    pub(crate) time: f32,
    pub(crate) c: [f64; 2],
}

/// A keyed morph of the Julia parameter. Between keys the parameter
/// interpolates linearly (smoothstep-eased per key pair when `smooth`);
/// before the first and after the last key it holds. Keys may be stored in
/// any order — evaluation sorts by time.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct JuliaCurve {
    pub(crate) keys: Vec<JuliaKey>,
    pub(crate) smooth: bool,
}

impl JuliaCurve {
    /// The Julia parameter at linear progress `u ∈ [0, 1]`.
    pub(crate) fn at(&self, u: f64) -> [f64; 2] {
        let (before, after, t) = keyed_span(self.keys.iter().map(|key| key.time), u, self.smooth);
        let (a, b) = (self.keys[before].c, self.keys[after].c);
        [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        validate_key_times("julia curve", self.keys.iter().map(|key| key.time))?;
        for (index, key) in self.keys.iter().enumerate() {
            if !key.c[0].is_finite()
                || !key.c[1].is_finite()
                || key.c[0].abs() > 8.0
                || key.c[1].abs() > 8.0
            {
                return Err(format!(
                    "julia curve key {index}: c must be finite with |re|, |im| at most 8"
                ));
            }
        }
        Ok(())
    }
}

/// The continuous family settings a curve may drive. The Multibrot/Nova
/// degree stays fixed: it is integer-valued in the iteration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum FamilyCurveTarget {
    #[default]
    NovaRelaxation,
    MandelboxScale,
    MandelboxMinRadius,
    MandelboxFixedRadius,
}

impl FamilyCurveTarget {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::NovaRelaxation => "Nova relaxation",
            Self::MandelboxScale => "Mandelbox scale",
            Self::MandelboxMinRadius => "Mandelbox min radius",
            Self::MandelboxFixedRadius => "Mandelbox fixed radius",
        }
    }

    /// The valid value range, mirroring `FamilyParameters::validate`.
    pub(crate) fn range(self) -> (f64, f64) {
        match self {
            Self::NovaRelaxation => (0.1, 4.0),
            Self::MandelboxScale => (-4.0, 4.0),
            Self::MandelboxMinRadius => (0.01, 2.0),
            Self::MandelboxFixedRadius => (0.1, 4.0),
        }
    }

    pub(crate) fn applies_to(self, family: FractalFamily) -> bool {
        match self {
            Self::NovaRelaxation => family.uses_relaxation(),
            Self::MandelboxScale | Self::MandelboxMinRadius | Self::MandelboxFixedRadius => {
                family.uses_mandelbox()
            }
        }
    }

    pub(crate) fn apply(self, parameters: &mut FamilyParameters, value: f64) {
        match self {
            Self::NovaRelaxation => parameters.nova_relaxation = value,
            Self::MandelboxScale => parameters.mandelbox_scale = value,
            Self::MandelboxMinRadius => parameters.mandelbox_min_radius = value,
            Self::MandelboxFixedRadius => parameters.mandelbox_fixed_radius = value,
        }
    }

    pub(crate) fn current(self, parameters: &FamilyParameters) -> f64 {
        match self {
            Self::NovaRelaxation => parameters.nova_relaxation,
            Self::MandelboxScale => parameters.mandelbox_scale,
            Self::MandelboxMinRadius => parameters.mandelbox_min_radius,
            Self::MandelboxFixedRadius => parameters.mandelbox_fixed_radius,
        }
    }
}

/// A keyed curve over one continuous family setting; the same hold/lerp
/// semantics as [`JuliaCurve`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct FamilyCurve {
    pub(crate) target: FamilyCurveTarget,
    pub(crate) keys: Vec<CurveKey>,
    pub(crate) smooth: bool,
}

impl FamilyCurve {
    /// The setting's value at linear progress `u ∈ [0, 1]`.
    pub(crate) fn at(&self, u: f64) -> f64 {
        let (before, after, t) = keyed_span(self.keys.iter().map(|key| key.time), u, self.smooth);
        let (a, b) = (self.keys[before].value, self.keys[after].value);
        a + (b - a) * t
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let label = self.target.label();
        validate_key_times(label, self.keys.iter().map(|key| key.time))?;
        let (low, high) = self.target.range();
        for (index, key) in self.keys.iter().enumerate() {
            if !key.value.is_finite() || !(low..=high).contains(&key.value) {
                return Err(format!(
                    "{label} curve key {index}: value must be between {low} and {high}"
                ));
            }
        }
        Ok(())
    }
}

/// The key pair bracketing progress `u` and the interpolation fraction
/// between them, over keys sorted by time (holds at both ends). Indices
/// refer to the unsorted key list, so callers index their own storage.
fn keyed_span(times: impl Iterator<Item = f32>, u: f64, smooth: bool) -> (usize, usize, f64) {
    let mut order: Vec<(usize, f32)> = times.enumerate().collect();
    debug_assert!(!order.is_empty(), "curves hold at least one key");
    order.sort_by(|a, b| a.1.total_cmp(&b.1));
    let first = order[0];
    if u <= f64::from(first.1) {
        return (first.0, first.0, 0.0);
    }
    for pair in order.windows(2) {
        let (before, after) = (pair[0], pair[1]);
        if u <= f64::from(after.1) {
            let span = f64::from(after.1) - f64::from(before.1);
            let mut t = if span > 0.0 {
                ((u - f64::from(before.1)) / span).clamp(0.0, 1.0)
            } else {
                1.0
            };
            if smooth {
                t = t * t * (3.0 - 2.0 * t);
            }
            // Land exactly on the key: `a + (b − a)·1` misses `b` by an ulp.
            if t >= 1.0 {
                return (after.0, after.0, 0.0);
            }
            return (before.0, after.0, t);
        }
    }
    let last = order[order.len() - 1];
    (last.0, last.0, 0.0)
}

fn validate_key_times(label: &str, times: impl Iterator<Item = f32>) -> Result<(), String> {
    let times: Vec<f32> = times.collect();
    if times.is_empty() || times.len() > MAX_CURVE_KEYS {
        return Err(format!(
            "{label}: a curve holds between 1 and {MAX_CURVE_KEYS} keys"
        ));
    }
    for (index, time) in times.iter().enumerate() {
        if !time.is_finite() || !(0.0..=1.0).contains(time) {
            return Err(format!("{label} key {index}: time must be between 0 and 1"));
        }
    }
    Ok(())
}

impl ZoomAnimation {
    /// Number of frames in the sequence. The last frame lands exactly on the
    /// end magnification, so `duration × fps` playback ends on the target.
    pub(crate) fn frame_count(&self) -> usize {
        ((self.duration_seconds.max(0.1) * self.fps.max(1) as f32).round() as usize)
            .clamp(2, MAX_FRAMES)
    }

    /// Linear progress of `frame` through the sequence, in `[0, 1]`.
    fn progress(&self, frame: usize) -> f64 {
        let last = (self.frame_count() - 1) as f64;
        (frame as f64 / last).clamp(0.0, 1.0)
    }

    /// Eased progress: smoothstep when easing is on.
    fn eased(&self, u: f64) -> f64 {
        if self.ease {
            u * u * (3.0 - 2.0 * u)
        } else {
            u
        }
    }

    /// Eased progress of `frame`, the sampling parameter for [`CameraPath`].
    pub(crate) fn eased_progress(&self, frame: usize) -> f64 {
        self.eased(self.progress(frame))
    }

    /// Linear (un-eased) progress of `frame`, the time base of the gradient
    /// sweep and the parameter curves.
    pub(crate) fn linear_progress(&self, frame: usize) -> f64 {
        self.progress(frame)
    }

    /// Whether any curve animates the dynamics — which makes the exporter
    /// rebuild the reference orbit per frame.
    pub(crate) fn has_dynamics_curves(&self) -> bool {
        self.julia_curve.is_some() || !self.family_curves.is_empty()
    }

    /// Whether the animation flies a waypoint path rather than the
    /// fixed-centre dive.
    pub(crate) fn path_active(&self) -> bool {
        self.waypoints.len() >= 2
    }

    /// log10 of the magnification of `frame`.
    pub(crate) fn magnification_log10_at(&self, frame: usize) -> f64 {
        let u = self.eased(self.progress(frame));
        self.start_magnification_log10
            + (self.end_magnification_log10 - self.start_magnification_log10) * u
    }

    /// Gradient offset added at `frame`, in turns. Linear (not eased): a
    /// turning palette should turn at constant speed.
    pub(crate) fn gradient_offset_at(&self, frame: usize) -> f32 {
        self.gradient_sweep_turns * self.progress(frame) as f32
    }

    /// The deepest magnification any frame reaches.
    pub(crate) fn max_magnification_log10(&self) -> f64 {
        if self.path_active() {
            self.waypoints
                .iter()
                .map(|waypoint| waypoint.magnification_log10)
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            self.start_magnification_log10
                .max(self.end_magnification_log10)
        }
    }

    /// The iteration budget of a frame rendered at `magnification_log10`
    /// (the exporters sample the magnification first — off the fixed dive
    /// or off the waypoint path — then budget from it). The budget the user
    /// set applies to the deepest frame of the animation; with iteration
    /// scaling on, shallower frames interpolate linearly in
    /// log-magnification down to a floor — escape times near a boundary
    /// grow roughly linearly with the zoom exponent, so early frames of a
    /// deep dive need only a fraction of the final budget.
    pub(crate) fn frame_iterations_at(&self, user_iterations: u32, magnification_log10: f64) -> u32 {
        if !self.scale_iterations {
            return user_iterations;
        }
        let floor = user_iterations.min(256).max(32);
        let deepest = self.max_magnification_log10();
        if deepest <= 1e-9 {
            return user_iterations;
        }
        let share = (magnification_log10 / deepest).clamp(0.0, 1.0);
        let budget = floor as f64 + (user_iterations as f64 - floor as f64) * share;
        (budget.round() as u32).clamp(floor.min(user_iterations), user_iterations)
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if !self.duration_seconds.is_finite() || !(0.1..=3600.0).contains(&self.duration_seconds) {
            return Err("duration must be between 0.1 and 3600 seconds".to_owned());
        }
        if !(1..=120).contains(&self.fps) {
            return Err("fps must be between 1 and 120".to_owned());
        }
        if self.frame_count() > MAX_FRAMES {
            return Err(format!("the sequence exceeds {MAX_FRAMES} frames"));
        }
        if !(16..=MAX_DIMENSION).contains(&self.width)
            || !(16..=MAX_DIMENSION).contains(&self.height)
        {
            return Err(format!(
                "frame size must be between 16 and {MAX_DIMENSION} pixels"
            ));
        }
        if !self.start_magnification_log10.is_finite()
            || !self.end_magnification_log10.is_finite()
            || !(-3.0..=5_000.0).contains(&self.start_magnification_log10)
            || !(-3.0..=5_000.0).contains(&self.end_magnification_log10)
        {
            return Err("magnification exponents must be between -3 and 5000".to_owned());
        }
        if !self.gradient_sweep_turns.is_finite() || self.gradient_sweep_turns.abs() > 100.0 {
            return Err("gradient sweep must be between -100 and 100 turns".to_owned());
        }
        if self.waypoints.len() > MAX_WAYPOINTS {
            return Err(format!("a camera path may hold at most {MAX_WAYPOINTS} waypoints"));
        }
        for (index, waypoint) in self.waypoints.iter().enumerate() {
            if !waypoint.magnification_log10.is_finite()
                || !(-3.0..=5_000.0).contains(&waypoint.magnification_log10)
            {
                return Err(format!(
                    "waypoint {index}: magnification exponent must be between -3 and 5000"
                ));
            }
            if !waypoint.centre[0].is_finite() || !waypoint.centre[1].is_finite() {
                return Err(format!("waypoint {index}: centre must be finite"));
            }
            if let Some((re, im)) = &waypoint.exact
                && (re.trim().is_empty() || im.trim().is_empty())
            {
                return Err(format!("waypoint {index}: exact centre must not be empty"));
            }
        }
        if let Some(curve) = &self.julia_curve {
            curve.validate()?;
        }
        for curve in &self.family_curves {
            curve.validate()?;
        }
        Ok(())
    }
}

/// One frame of a [`CameraPath`], everything the exporter needs beyond the
/// per-waypoint reference orbits it holds itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PathFrame {
    pub(crate) magnification_log10: f64,
    /// Index of the waypoint whose reference orbit anchors this frame.
    pub(crate) anchor: usize,
    /// Frame centre relative to the anchor, in units of the frame
    /// half-height (screen space): world offset = `screen_offset × h`.
    pub(crate) screen_offset: [f64; 2],
}

struct CameraSegment {
    m_from: f64,
    m_to: f64,
    /// Offset of the shallow endpoint's centre from the deep endpoint's, in
    /// units of the shallow half-height.
    pan: [f64; 2],
    /// Whether the segment's deeper endpoint — its reference anchor — is
    /// the destination waypoint.
    anchor_is_to: bool,
    length: f64,
}

/// A precomputed flight path through the animation's waypoints.
///
/// Each segment anchors on its deeper endpoint: that waypoint's reference
/// orbit serves every frame of the segment, and the frame centre approaches
/// it along `centre(v) = anchor + pan·v·h(v)`, where `v` is the progress
/// from the anchor and `h(v)` the frame half-height (log-interpolated). The
/// centre's screen-space offset from the anchor is therefore `pan·v` —
/// linear, bounded by the pan distance and independent of depth — which is
/// exactly the off-centre-reference form the tiled still exporter already
/// renders, so drift is as exact as tiling at any magnification. For a pure
/// zoom (`pan = 0`) it degenerates to the fixed-centre dive; for a pure pan
/// (equal magnifications) to a linear glide.
///
/// Frame time is apportioned across segments by a perceptual arc length
/// mixing decades of zoom with screen-space pan, so the visual speed stays
/// steady through waypoints.
pub(crate) struct CameraPath {
    segments: Vec<CameraSegment>,
    /// Normalised cumulative-length boundaries, `segments.len() + 1` values
    /// from 0 to 1.
    cumulative: Vec<f64>,
}

impl CameraPath {
    pub(crate) fn new(waypoints: &[ZoomWaypoint]) -> Result<Self, String> {
        if waypoints.len() < 2 {
            return Err("a camera path needs at least two waypoints".to_owned());
        }
        let mut segments = Vec::with_capacity(waypoints.len() - 1);
        for (index, pair) in waypoints.windows(2).enumerate() {
            let (from, to) = (&pair[0], &pair[1]);
            let anchor_is_to = to.magnification_log10 >= from.magnification_log10;
            let (deep, shallow) = if anchor_is_to { (to, from) } else { (from, to) };
            let pan = waypoint_pan(deep, shallow)
                .map_err(|error| format!("waypoints {index}–{}: {error}", index + 1))?;
            let pan_norm = pan[0].hypot(pan[1]);
            if pan_norm > MAX_PAN_HALF_HEIGHTS {
                return Err(format!(
                    "waypoints {index}–{}: the pan between them spans {pan_norm:.0} half-heights \
                     of the shallower view (at most {MAX_PAN_HALF_HEIGHTS:.0}); add a waypoint at \
                     a lower magnification between them",
                    index + 1
                ));
            }
            let dm = to.magnification_log10 - from.magnification_log10;
            let length = dm
                .hypot(pan_norm * PAN_DECADES_PER_HALF_HEIGHT)
                .max(1e-9);
            segments.push(CameraSegment {
                m_from: from.magnification_log10,
                m_to: to.magnification_log10,
                pan,
                anchor_is_to,
                length,
            });
        }
        let total: f64 = segments.iter().map(|segment| segment.length).sum();
        let mut cumulative = Vec::with_capacity(segments.len() + 1);
        cumulative.push(0.0);
        let mut sum = 0.0;
        for segment in &segments {
            sum += segment.length;
            cumulative.push(sum / total);
        }
        cumulative[segments.len()] = 1.0;
        Ok(Self {
            segments,
            cumulative,
        })
    }

    /// The camera at eased progress `s ∈ [0, 1]` along the path.
    pub(crate) fn at(&self, s: f64) -> PathFrame {
        let s = s.clamp(0.0, 1.0);
        let index = self
            .cumulative
            .windows(2)
            .position(|bounds| s <= bounds[1])
            .unwrap_or(self.segments.len() - 1);
        let span = self.cumulative[index + 1] - self.cumulative[index];
        let t = if span > 0.0 {
            ((s - self.cumulative[index]) / span).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let segment = &self.segments[index];
        let v = if segment.anchor_is_to { 1.0 - t } else { t };
        PathFrame {
            magnification_log10: segment.m_from + (segment.m_to - segment.m_from) * t,
            anchor: index + usize::from(segment.anchor_is_to),
            screen_offset: [segment.pan[0] * v, segment.pan[1] * v],
        }
    }

    /// The sorted waypoint indices that anchor at least one segment — the
    /// waypoints whose reference orbits the exporter must compute.
    pub(crate) fn anchor_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = self
            .segments
            .iter()
            .enumerate()
            .map(|(index, segment)| index + usize::from(segment.anchor_is_to))
            .collect();
        indices.sort_unstable();
        indices.dedup();
        indices
    }
}

/// The offset of `shallow`'s centre from `deep`'s, in units of the shallow
/// half-height. Computed in decimal arbitrary precision — the difference of
/// two deep centres cancels catastrophically in f64 — then lifted into f64
/// range by the exact power of ten before the final small division.
fn waypoint_pan(deep: &ZoomWaypoint, shallow: &ZoomWaypoint) -> Result<[f64; 2], String> {
    let ms = shallow.magnification_log10;
    let zoom_exponent = ms.ceil().max(0.0) as u32 + 40;
    let parse = |waypoint: &ZoomWaypoint| -> Result<(DeepReal, DeepReal), String> {
        match &waypoint.exact {
            Some((re, im)) => Ok((
                DeepReal::parse(re, zoom_exponent)?,
                DeepReal::parse(im, zoom_exponent)?,
            )),
            None => Ok((
                DeepReal::from_f64(waypoint.centre[0], zoom_exponent)?,
                DeepReal::from_f64(waypoint.centre[1], zoom_exponent)?,
            )),
        }
    };
    let (deep_re, deep_im) = parse(deep)?;
    let (shallow_re, shallow_im) = parse(shallow)?;
    // h_shallow = 1.45 × 10^−ms. Divide in two steps so every intermediate
    // stays representable: lift the exact decimal difference by 10^⌈ms⌉
    // (an exact power of ten), then divide by the remaining f64 factor
    // 1.45 × 10^{⌈ms⌉−ms} ∈ [1.45, 14.5).
    let scale = DeepReal::parse(&format!("1e{}", ms.ceil() as i64), zoom_exponent)?;
    let residual = 1.45 * 10f64.powf(ms.ceil() - ms);
    Ok([
        shallow_re.sub(&deep_re).mul(&scale).to_f64() / residual,
        shallow_im.sub(&deep_im).mul(&scale).to_f64() / residual,
    ])
}

/// Scale of one frame in the shader's mantissa × 2^exponent form, computed
/// from logarithms so it survives magnifications far beyond f64 range.
pub(crate) fn frame_scale(magnification_log10: f64) -> (f32, i32) {
    // half_height = 1.45 × 10^-m.
    let log2_half = 1.45_f64.log2() - magnification_log10 * std::f64::consts::LOG2_10;
    let exponent = log2_half.floor();
    let mantissa = 2f64.powf(log2_half - exponent) as f32;
    (mantissa, exponent as i32)
}

/// The frame's half-height as an f64 for the shader's view uniforms; zero
/// beyond f64 range, matching `DeepReal::to_f64` for deep interactive views.
pub(crate) fn frame_half_height_f64(magnification_log10: f64) -> f64 {
    if magnification_log10 > 300.0 {
        return 0.0;
    }
    1.45 * 10f64.powf(-magnification_log10)
}

/// Natural log of one pixel's height in world units at this magnification,
/// for the distance-estimate colouring.
pub(crate) fn frame_pixel_log(magnification_log10: f64, frame_height: u32) -> f32 {
    (std::f64::consts::LN_10 * (1.45_f64.log10() - magnification_log10)
        + (2.0 / frame_height.max(1) as f64).ln()) as f32
}

/// The view of one rectangular region of a frame, expressed so the region
/// renders around the *same* reference orbit as the whole frame: the scale
/// shrinks by the region's share of the frame height, and the reference —
/// the frame centre — moves to `reference_offset` in the region's local
/// units. This keeps tiling exact at any magnification: the perturbation
/// deltas are algebraically identical to the whole frame's, while the `f64`
/// centre shift only serves the shallow non-perturbation paths (it
/// underflows harmlessly at depth, where those paths are unused).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RegionView {
    /// Region centre offset from the frame centre, world units (`f64` range
    /// permitting; zero beyond it).
    pub(crate) centre_shift: [f64; 2],
    pub(crate) half_height_f64: f64,
    pub(crate) scale_mantissa: f32,
    pub(crate) scale_exponent: i32,
    pub(crate) aspect: f32,
    /// Frame centre relative to the region centre, in region-local units
    /// (x and y in units of the region half-height).
    pub(crate) reference_offset: [f32; 2],
    /// Region half-height as a share of the frame half-height.
    pub(crate) share: f32,
    /// Region centre in frame-local units (x spans ±frame aspect, y ±1) —
    /// with `share`, the affine map the transformations stage uses to
    /// evaluate warps in frame coordinates.
    pub(crate) frame_centre_local: [f32; 2],
}

/// `frame_reference_offset` is the reference point's position relative to
/// the frame centre, in units of the frame half-height — zero when the
/// reference orbit sits at the frame centre, `−screen_offset` when a
/// [`CameraPath`] frame drifts away from its anchor.
pub(crate) fn region_view(
    magnification_log10: f64,
    full_size: (u32, u32),
    origin: (u32, u32),
    region_size: (u32, u32),
    frame_reference_offset: [f64; 2],
) -> RegionView {
    let (full_width, full_height) = (full_size.0.max(1) as f64, full_size.1.max(1) as f64);
    let (region_width, region_height) = (region_size.0.max(1) as f64, region_size.1.max(1) as f64);
    let frame_aspect = full_width / full_height;
    // Region centre in frame-local units (x spans ±aspect, y up, +1 at the
    // top row of pixels).
    let centre_x = ((origin.0 as f64 + region_width * 0.5) / full_width * 2.0 - 1.0) * frame_aspect;
    let centre_y = 1.0 - (origin.1 as f64 + region_height * 0.5) / full_height * 2.0;
    // Region half-height as a share of the frame half-height.
    let share = region_height / full_height;

    let half_full = frame_half_height_f64(magnification_log10);
    let (full_mantissa, full_exponent) = frame_scale(magnification_log10);
    // scale_region = scale_full × share, renormalised into [1, 2).
    let scaled = full_mantissa as f64 * share;
    let shift = scaled.log2().floor() as i32;
    RegionView {
        centre_shift: [centre_x * half_full, centre_y * half_full],
        half_height_f64: half_full * share,
        scale_mantissa: (scaled / 2f64.powi(shift)) as f32,
        scale_exponent: full_exponent + shift,
        aspect: (region_width / region_height) as f32,
        reference_offset: [
            ((frame_reference_offset[0] - centre_x) / share) as f32,
            ((frame_reference_offset[1] - centre_y) / share) as f32,
        ],
        share: share as f32,
        frame_centre_local: [centre_x as f32, centre_y as f32],
    }
}

/// Splits `extent` pixels into the fewest tiles of at most `max_tile`,
/// as evenly as possible: (offset, size) pairs covering the extent exactly.
pub(crate) fn tile_spans(extent: u32, max_tile: u32) -> Vec<(u32, u32)> {
    let max_tile = max_tile.max(1);
    let count = extent.div_ceil(max_tile).max(1);
    let base = extent / count;
    let remainder = extent % count;
    let mut spans = Vec::with_capacity(count as usize);
    let mut offset = 0;
    for index in 0..count {
        let size = base + u32::from(index < remainder);
        spans.push((offset, size));
        offset += size;
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(start: f64, end: f64, ease: bool) -> ZoomAnimation {
        ZoomAnimation {
            duration_seconds: 10.0,
            fps: 30,
            start_magnification_log10: start,
            end_magnification_log10: end,
            ease,
            ..ZoomAnimation::default()
        }
    }

    #[test]
    fn endpoints_are_exact_and_progress_is_monotonic() {
        for ease in [false, true] {
            let animation = path(0.0, 120.0, ease);
            let frames = animation.frame_count();
            assert_eq!(frames, 300);
            assert_eq!(animation.magnification_log10_at(0), 0.0);
            assert_eq!(animation.magnification_log10_at(frames - 1), 120.0);
            let mut previous = f64::NEG_INFINITY;
            for frame in 0..frames {
                let m = animation.magnification_log10_at(frame);
                assert!(m >= previous, "ease={ease} frame {frame}");
                previous = m;
            }
        }
    }

    #[test]
    fn easing_slows_the_ends_and_zoom_out_paths_descend() {
        let eased = path(0.0, 100.0, true);
        let linear = path(0.0, 100.0, false);
        // The first eased step is smaller than the first linear step.
        assert!(eased.magnification_log10_at(1) < linear.magnification_log10_at(1));
        // Midpoint agrees (smoothstep is symmetric).
        let mid = eased.frame_count() / 2;
        assert!((eased.magnification_log10_at(mid) - 50.0).abs() < 0.5);
        let out = path(80.0, 10.0, true);
        assert!(out.magnification_log10_at(0) > out.magnification_log10_at(out.frame_count() - 1));
        assert_eq!(out.max_magnification_log10(), 80.0);
    }

    #[test]
    fn gradient_sweep_is_linear_and_ends_at_the_requested_turns() {
        let mut animation = path(0.0, 10.0, true);
        animation.gradient_sweep_turns = 0.5;
        let frames = animation.frame_count();
        assert_eq!(animation.gradient_offset_at(0), 0.0);
        assert!((animation.gradient_offset_at(frames - 1) - 0.5).abs() < 1e-6);
        let quarter = animation.gradient_offset_at(frames / 4);
        assert!((quarter - 0.125).abs() < 0.01);
    }

    #[test]
    fn iteration_budget_hits_the_user_value_at_depth_and_the_floor_at_the_top() {
        let animation = path(0.0, 300.0, false);
        let frames = animation.frame_count();
        let budget_at =
            |frame: usize| animation.frame_iterations_at(2_400, animation.magnification_log10_at(frame));
        assert_eq!(budget_at(frames - 1), 2_400);
        assert_eq!(budget_at(0), 256);
        let mid = budget_at(frames / 2);
        assert!((1_200..=1_450).contains(&mid), "{mid}");
        // Monotonic along a zoom-in.
        let mut previous = 0;
        for frame in 0..frames {
            let budget = budget_at(frame);
            assert!(budget >= previous);
            previous = budget;
        }
        // Disabled scaling and degenerate paths keep the full budget.
        let mut fixed = path(0.0, 300.0, false);
        fixed.scale_iterations = false;
        assert_eq!(fixed.frame_iterations_at(2_400, 0.0), 2_400);
        let flat = path(0.0, 0.0, false);
        assert_eq!(flat.frame_iterations_at(2_400, 0.0), 2_400);
        // Small budgets never scale below themselves.
        assert_eq!(path(0.0, 100.0, false).frame_iterations_at(64, 0.0), 64);
    }

    #[test]
    fn frame_scale_matches_f64_in_range_and_survives_beyond_it() {
        for m in [0.0, 3.0, 13.9, 250.0] {
            let (mantissa, exponent) = frame_scale(m);
            let value = mantissa as f64 * 2f64.powi(exponent);
            let expected = 1.45 * 10f64.powf(-m);
            assert!(
                (value - expected).abs() / expected < 1e-6,
                "m={m}: {value} vs {expected}"
            );
        }
        let (mantissa, exponent) = frame_scale(4000.0);
        assert!((1.0..2.0).contains(&mantissa));
        // log2(1.45e-4000) ≈ -13287.
        assert!((-13290..=-13280).contains(&exponent));
        assert_eq!(frame_half_height_f64(4000.0), 0.0);
        assert!(frame_half_height_f64(10.0) > 0.0);
        assert!(frame_pixel_log(4000.0, 1080).is_finite());
    }

    #[test]
    fn region_view_of_the_whole_frame_is_the_frame() {
        let region = region_view(12.0, (1920, 1080), (0, 0), (1920, 1080), [0.0; 2]);
        let (mantissa, exponent) = frame_scale(12.0);
        assert_eq!(region.scale_mantissa, mantissa);
        assert_eq!(region.scale_exponent, exponent);
        assert_eq!(region.reference_offset, [0.0, 0.0]);
        assert_eq!(region.centre_shift, [0.0, 0.0]);
        assert!((region.aspect - 1920.0 / 1080.0).abs() < 1e-6);
        assert_eq!(region.half_height_f64, frame_half_height_f64(12.0));
    }

    #[test]
    fn region_views_reconstruct_the_frame_pixel_grid() {
        // A pixel's world position computed through any region containing it
        // must match the position computed through the whole frame.
        let m = 6.0;
        let full = (640u32, 480u32);
        let half = frame_half_height_f64(m);
        let aspect = full.0 as f64 / full.1 as f64;
        let world_of = |px: f64, py: f64| -> [f64; 2] {
            let lx = (px / full.0 as f64 * 2.0 - 1.0) * aspect;
            let ly = 1.0 - py / full.1 as f64 * 2.0;
            [lx * half, ly * half]
        };
        for (origin, size) in [
            ((0u32, 0u32), (320u32, 240u32)),
            ((320, 0), (320, 240)),
            ((0, 240), (320, 240)),
            ((320, 240), (320, 240)),
            ((160, 120), (321, 199)),
        ] {
            let region = region_view(m, full, origin, size, [0.0; 2]);
            let scale = region.scale_mantissa as f64 * 2f64.powi(region.scale_exponent);
            for (px, py) in [(0.3, 0.7), (0.9, 0.1)] {
                // The sample point in region-local units.
                let sample_x = origin.0 as f64 + px * size.0 as f64;
                let sample_y = origin.1 as f64 + py * size.1 as f64;
                let lx = (px * 2.0 - 1.0) * region.aspect as f64;
                let ly = 1.0 - py * 2.0;
                // World = frame centre + (local − reference_offset) × scale,
                // exactly the shader's perturbation delta.
                let world = [
                    (lx - region.reference_offset[0] as f64) * scale,
                    (ly - region.reference_offset[1] as f64) * scale,
                ];
                let expected = world_of(sample_x, sample_y);
                for axis in 0..2 {
                    assert!(
                        (world[axis] - expected[axis]).abs() < half * 1e-6,
                        "{origin:?} {size:?} axis {axis}: {} vs {}",
                        world[axis],
                        expected[axis]
                    );
                }
            }
        }
    }

    #[test]
    fn tile_spans_cover_exactly_with_the_fewest_even_tiles() {
        assert_eq!(tile_spans(8192, 8192), vec![(0, 8192)]);
        assert_eq!(tile_spans(10000, 8192), vec![(0, 5000), (5000, 5000)]);
        let spans = tile_spans(16384, 2730);
        assert_eq!(spans.len(), 7);
        assert_eq!(spans.iter().map(|(_, size)| size).sum::<u32>(), 16384);
        assert_eq!(
            spans.last().map(|(offset, size)| offset + size),
            Some(16384)
        );
        assert!(spans.iter().all(|(_, size)| *size <= 2730));
        assert_eq!(tile_spans(5, 2), vec![(0, 2), (2, 2), (4, 1)]);
    }

    fn waypoint(m: f64, centre: [f64; 2]) -> ZoomWaypoint {
        ZoomWaypoint {
            magnification_log10: m,
            centre,
            exact: None,
        }
    }

    /// The world-space camera centre a [`PathFrame`] describes, given the
    /// anchor waypoints' f64 centres — the shallow-range reconstruction the
    /// exporter performs.
    fn frame_centre(path_frame: &PathFrame, waypoints: &[ZoomWaypoint]) -> [f64; 2] {
        let h = frame_half_height_f64(path_frame.magnification_log10);
        let anchor = waypoints[path_frame.anchor].centre;
        [
            anchor[0] + path_frame.screen_offset[0] * h,
            anchor[1] + path_frame.screen_offset[1] * h,
        ]
    }

    #[test]
    fn path_segments_anchor_deep_and_hit_both_endpoints() {
        let waypoints = [waypoint(1.0, [-0.5, 0.1]), waypoint(3.0, [-0.52, 0.13])];
        let path = CameraPath::new(&waypoints).unwrap();
        let start = path.at(0.0);
        assert_eq!(start.magnification_log10, 1.0);
        assert_eq!(start.anchor, 1);
        let reconstructed = frame_centre(&start, &waypoints);
        for axis in 0..2 {
            assert!(
                (reconstructed[axis] - waypoints[0].centre[axis]).abs() < 1e-12,
                "axis {axis}: {} vs {}",
                reconstructed[axis],
                waypoints[0].centre[axis]
            );
        }
        // The start sits (0.02, −0.03)/0.145 half-heights from the anchor.
        assert!((start.screen_offset[0] - 0.02 / 0.145).abs() < 1e-9);
        assert!((start.screen_offset[1] - -0.03 / 0.145).abs() < 1e-9);
        let end = path.at(1.0);
        assert_eq!(end.magnification_log10, 3.0);
        assert_eq!(end.anchor, 1);
        assert_eq!(end.screen_offset, [0.0, 0.0]);
        // A zoom-out segment anchors on its start.
        let out = CameraPath::new(&[waypoint(3.0, [-0.52, 0.13]), waypoint(1.0, [-0.5, 0.1])])
            .unwrap();
        assert_eq!(out.at(0.0).anchor, 0);
        assert_eq!(out.at(0.0).screen_offset, [0.0, 0.0]);
        assert_eq!(out.at(1.0).magnification_log10, 1.0);
    }

    #[test]
    fn pure_pan_glides_linearly_at_constant_magnification() {
        let waypoints = [waypoint(2.0, [-0.5, 0.0]), waypoint(2.0, [-0.47, 0.02])];
        let path = CameraPath::new(&waypoints).unwrap();
        for (s, share) in [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (1.0, 1.0)] {
            let frame = path.at(s);
            assert_eq!(frame.magnification_log10, 2.0);
            let centre = frame_centre(&frame, &waypoints);
            for axis in 0..2 {
                let expected = waypoints[0].centre[axis]
                    + (waypoints[1].centre[axis] - waypoints[0].centre[axis]) * share;
                assert!(
                    (centre[axis] - expected).abs() < 1e-12,
                    "s={s} axis {axis}: {} vs {expected}",
                    centre[axis]
                );
            }
        }
    }

    #[test]
    fn multi_segment_paths_are_continuous_at_waypoints() {
        let waypoints = [
            waypoint(0.5, [-0.6, 0.0]),
            waypoint(2.5, [-0.58, 0.015]),
            waypoint(2.5, [-0.55, 0.02]),
            waypoint(1.0, [-0.53, 0.01]),
        ];
        let path = CameraPath::new(&waypoints).unwrap();
        for s in [0.001, 0.25, 0.4999, 0.5001, 0.75, 0.999] {
            let before = path.at(s - 1e-6);
            let after = path.at(s + 1e-6);
            assert!(
                (before.magnification_log10 - after.magnification_log10).abs() < 1e-3,
                "magnification jumps at s={s}"
            );
            let a = frame_centre(&before, &waypoints);
            let b = frame_centre(&after, &waypoints);
            let h = frame_half_height_f64(before.magnification_log10);
            for axis in 0..2 {
                assert!(
                    (a[axis] - b[axis]).abs() < h * 1e-2,
                    "centre jumps at s={s} axis {axis}: {} vs {}",
                    a[axis],
                    b[axis]
                );
            }
        }
        // Every waypoint is visited exactly, in order.
        for (index, s) in [(0usize, 0.0f64), (3, 1.0)] {
            let frame = path.at(s);
            let centre = frame_centre(&frame, &waypoints);
            for axis in 0..2 {
                assert!((centre[axis] - waypoints[index].centre[axis]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn deep_waypoint_pans_use_the_exact_decimals() {
        // Two centres differing at the 51st decimal digit — far beyond f64
        // resolution, so their f64 projections coincide.
        let deep = ZoomWaypoint {
            magnification_log10: 52.0,
            centre: [0.1, 0.2],
            exact: Some(("0.1".to_owned(), "0.2".to_owned())),
        };
        let mut shallow_re = "0.1".to_owned();
        shallow_re.push_str(&"0".repeat(49));
        shallow_re.push('1');
        let shallow = ZoomWaypoint {
            magnification_log10: 50.0,
            centre: [0.1, 0.2],
            exact: Some((shallow_re, "0.2".to_owned())),
        };
        let path = CameraPath::new(&[shallow.clone(), deep.clone()]).unwrap();
        let start = path.at(0.0);
        assert_eq!(start.anchor, 1);
        // pan = 1e-51 / (1.45e-50) ≈ 0.0689655.
        assert!(
            (start.screen_offset[0] - 1e-51 / 1.45e-50).abs() < 1e-9,
            "{}",
            start.screen_offset[0]
        );
        assert_eq!(start.screen_offset[1], 0.0);
        // The same pair through f64 projections would cancel to zero.
        let blunt = CameraPath::new(&[
            waypoint(50.0, [0.1, 0.2]),
            waypoint(52.0, [0.1, 0.2]),
        ])
        .unwrap();
        assert_eq!(blunt.at(0.0).screen_offset, [0.0, 0.0]);
    }

    #[test]
    fn path_time_is_apportioned_by_perceptual_length() {
        // One decade then nine: the first waypoint boundary sits near s=0.1.
        let waypoints = [
            waypoint(0.0, [-0.5, 0.0]),
            waypoint(1.0, [-0.5, 0.0]),
            waypoint(10.0, [-0.5, 0.0]),
        ];
        let path = CameraPath::new(&waypoints).unwrap();
        let boundary = path.at(0.1);
        assert!((boundary.magnification_log10 - 1.0).abs() < 0.05);
        // A long pan takes time even without any zoom.
        let pan_heavy = CameraPath::new(&[
            waypoint(1.0, [-0.5, 0.0]),
            waypoint(1.0, [-0.5 + 0.29 * 40.0, 0.0]), // 80 half-heights at h=0.145
            waypoint(4.0, [-0.5 + 0.29 * 40.0, 0.0]),
        ])
        .unwrap();
        // Pan length ≈ 80·log10(e) ≈ 35 decades-equivalent vs 3 decades of
        // zoom, so the halfway point is still inside the pan segment.
        let mid = pan_heavy.at(0.5);
        assert_eq!(mid.magnification_log10, 1.0, "still panning at s=0.5");
    }

    #[test]
    fn paths_reject_oversized_pans_and_undersized_waypoint_lists() {
        assert!(CameraPath::new(&[waypoint(1.0, [0.0, 0.0])]).is_err());
        let too_far = CameraPath::new(&[
            waypoint(6.0, [0.0, 0.0]),
            waypoint(6.0, [1.0, 0.0]), // 1/1.45e-6 ≈ 690k half-heights
        ]);
        assert!(too_far.is_err());
    }

    #[test]
    fn path_activation_and_budgets_follow_the_waypoints() {
        let mut animation = path(0.0, 3.0, false);
        assert!(!animation.path_active());
        animation.waypoints = vec![waypoint(0.0, [-0.5, 0.0]), waypoint(120.0, [-0.5, 0.0])];
        assert!(animation.path_active());
        assert_eq!(animation.max_magnification_log10(), 120.0);
        assert_eq!(animation.frame_iterations_at(2_400, 120.0), 2_400);
        assert_eq!(animation.frame_iterations_at(2_400, 0.0), 256);
        animation.validate().unwrap();
        animation.waypoints[0].magnification_log10 = f64::NAN;
        assert!(animation.validate().is_err());
    }

    #[test]
    fn curves_hold_lerp_and_smooth_between_keys() {
        let curve = FamilyCurve {
            target: FamilyCurveTarget::NovaRelaxation,
            keys: vec![
                CurveKey {
                    time: 0.25,
                    value: 1.0,
                },
                CurveKey {
                    time: 0.75,
                    value: 3.0,
                },
            ],
            smooth: false,
        };
        // Holds before the first and after the last key.
        assert_eq!(curve.at(0.0), 1.0);
        assert_eq!(curve.at(1.0), 3.0);
        // Linear between keys.
        assert!((curve.at(0.5) - 2.0).abs() < 1e-12);
        assert!((curve.at(0.375) - 1.5).abs() < 1e-12);
        // Smoothstep keeps the midpoint but eases the ends.
        let eased = FamilyCurve {
            smooth: true,
            ..curve.clone()
        };
        assert!((eased.at(0.5) - 2.0).abs() < 1e-12);
        assert!(eased.at(0.3) < curve.at(0.3));
        assert!(eased.at(0.7) > curve.at(0.7));
        // A single key is a constant override.
        let constant = FamilyCurve {
            target: FamilyCurveTarget::MandelboxScale,
            keys: vec![CurveKey {
                time: 0.4,
                value: -2.0,
            }],
            smooth: false,
        };
        for u in [0.0, 0.4, 1.0] {
            assert_eq!(constant.at(u), -2.0);
        }
        // Unsorted key storage evaluates in time order.
        let unsorted = FamilyCurve {
            target: FamilyCurveTarget::NovaRelaxation,
            keys: vec![
                CurveKey {
                    time: 0.75,
                    value: 3.0,
                },
                CurveKey {
                    time: 0.25,
                    value: 1.0,
                },
            ],
            smooth: false,
        };
        assert!((unsorted.at(0.5) - 2.0).abs() < 1e-12);
        assert_eq!(unsorted.at(0.0), 1.0);
    }

    #[test]
    fn julia_curves_morph_through_the_parameter_plane() {
        let curve = JuliaCurve {
            keys: vec![
                JuliaKey {
                    time: 0.0,
                    c: [-0.8, 0.156],
                },
                JuliaKey {
                    time: 0.5,
                    c: [-0.4, 0.6],
                },
                JuliaKey {
                    time: 1.0,
                    c: [0.285, 0.01],
                },
            ],
            smooth: false,
        };
        assert_eq!(curve.at(0.0), [-0.8, 0.156]);
        assert_eq!(curve.at(0.5), [-0.4, 0.6]);
        assert_eq!(curve.at(1.0), [0.285, 0.01]);
        let quarter = curve.at(0.25);
        assert!((quarter[0] - -0.6).abs() < 1e-12);
        assert!((quarter[1] - 0.378).abs() < 1e-12);
        curve.validate().unwrap();
    }

    #[test]
    fn curve_validation_rejects_bad_keys_and_animation_carries_them() {
        assert!(
            JuliaCurve {
                keys: Vec::new(),
                smooth: false
            }
            .validate()
            .is_err()
        );
        assert!(
            JuliaCurve {
                keys: vec![JuliaKey {
                    time: 1.5,
                    c: [0.0, 0.0]
                }],
                smooth: false
            }
            .validate()
            .is_err()
        );
        assert!(
            JuliaCurve {
                keys: vec![JuliaKey {
                    time: 0.5,
                    c: [9.0, 0.0]
                }],
                smooth: false
            }
            .validate()
            .is_err()
        );
        // Family targets enforce their own value ranges.
        assert!(
            FamilyCurve {
                target: FamilyCurveTarget::NovaRelaxation,
                keys: vec![CurveKey {
                    time: 0.0,
                    value: 5.0
                }],
                smooth: false,
            }
            .validate()
            .is_err()
        );
        let mut animation = path(0.0, 3.0, false);
        assert!(!animation.has_dynamics_curves());
        animation.family_curves.push(FamilyCurve {
            target: FamilyCurveTarget::MandelboxMinRadius,
            keys: vec![CurveKey {
                time: 0.0,
                value: 0.5,
            }],
            smooth: false,
        });
        assert!(animation.has_dynamics_curves());
        animation.validate().unwrap();
        animation.family_curves[0].keys[0].value = 3.0;
        assert!(animation.validate().is_err());
    }

    #[test]
    fn region_reference_offset_composes_the_path_drift() {
        let m = 6.0;
        let full = (640u32, 480u32);
        let drift = [1.75f64, -0.6];
        let plain = region_view(m, full, (160, 120), (320, 240), [0.0; 2]);
        let drifted = region_view(m, full, (160, 120), (320, 240), drift);
        // Only the reference offset moves — by drift/share — and the pixel
        // grid mapping (centre, scale) is untouched.
        assert_eq!(plain.centre_shift, drifted.centre_shift);
        assert_eq!(plain.scale_mantissa, drifted.scale_mantissa);
        assert_eq!(plain.scale_exponent, drifted.scale_exponent);
        let share = 240.0 / 480.0;
        for axis in 0..2 {
            let expected = plain.reference_offset[axis] as f64 + drift[axis] / share;
            assert!(
                (drifted.reference_offset[axis] as f64 - expected).abs() < 1e-5,
                "axis {axis}: {} vs {expected}",
                drifted.reference_offset[axis]
            );
        }
    }

    #[test]
    fn validation_rejects_out_of_range_settings() {
        let good = ZoomAnimation::default();
        good.validate().unwrap();
        for bad in [
            ZoomAnimation {
                fps: 0,
                ..good.clone()
            },
            ZoomAnimation {
                width: 8,
                ..good.clone()
            },
            ZoomAnimation {
                duration_seconds: 4000.0,
                ..good.clone()
            },
            ZoomAnimation {
                end_magnification_log10: 6000.0,
                ..good.clone()
            },
            ZoomAnimation {
                gradient_sweep_turns: f32::NAN,
                ..good.clone()
            },
        ] {
            assert!(bad.validate().is_err());
        }
        // 3600 s at 120 fps stays within the frame budget only by clamping.
        let long = ZoomAnimation {
            duration_seconds: 3600.0,
            fps: 120,
            ..good
        };
        assert!(long.frame_count() <= MAX_FRAMES);
    }
}
