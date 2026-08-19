//! Zoom-path animation: a camera path through log-magnification, sampled
//! into frames for the image-sequence exporter.
//!
//! The path model is deliberately the one deep-zoom videos actually use: a
//! fixed centre (the deep target the user navigated to) and a magnification
//! that moves through decades of zoom at constant — optionally eased —
//! logarithmic speed. Because the centre never moves, one reference orbit
//! serves every frame: the arbitrary-precision orbit of the centre is
//! re-described (new scale mantissa and exponent) per frame instead of being
//! rebuilt, so exporting a 10^1000× dive costs one orbit, not one per frame.
//! An optional gradient sweep advances both colouring offsets across the
//! animation for the classic slowly-turning-palette look.

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
}

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
        }
    }
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
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn max_magnification_log10(&self) -> f64 {
        self.start_magnification_log10
            .max(self.end_magnification_log10)
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
        Ok(())
    }
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
}

pub(crate) fn region_view(
    magnification_log10: f64,
    full_size: (u32, u32),
    origin: (u32, u32),
    region_size: (u32, u32),
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
        reference_offset: [(-centre_x / share) as f32, (-centre_y / share) as f32],
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
        let region = region_view(12.0, (1920, 1080), (0, 0), (1920, 1080));
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
            let region = region_view(m, full, origin, size);
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
