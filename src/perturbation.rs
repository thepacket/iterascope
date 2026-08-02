//! Cached reference orbits for GPU perturbation rendering.
//!
//! One `f64` orbit is computed when the parameter view changes. Its high and low
//! `f32` words are uploaded to WebGPU; every fragment then evolves only its
//! small displacement from this shared orbit.

use std::sync::Arc;

use crate::precision::split_f64;

pub(crate) const MAX_REFERENCE_POINTS: usize = 4097;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ReferencePoint {
    /// Reference z as `[re_hi, im_hi, re_lo, im_lo]`.
    pub(crate) value: [f32; 4],
}

impl ReferencePoint {
    fn from_f64(z: [f64; 2]) -> Self {
        let re = split_f64(z[0]);
        let im = split_f64(z[1]);
        Self {
            value: [re[0], im[0], re[1], im[1]],
        }
    }

    #[cfg(test)]
    fn as_f64(self) -> [f64; 2] {
        [
            self.value[0] as f64 + self.value[2] as f64,
            self.value[1] as f64 + self.value[3] as f64,
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ReferenceInput {
    pub(crate) centre: [f64; 2],
    pub(crate) julia_c: [f64; 2],
    pub(crate) iterations: u32,
    pub(crate) pane: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReferenceKey {
    centre: [u64; 2],
    julia_c: [u64; 2],
    iterations: u32,
    pane: usize,
}

impl From<ReferenceInput> for ReferenceKey {
    fn from(input: ReferenceInput) -> Self {
        Self {
            centre: [input.centre[0].to_bits(), input.centre[1].to_bits()],
            julia_c: [input.julia_c[0].to_bits(), input.julia_c[1].to_bits()],
            iterations: input.iterations,
            pane: input.pane,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ReferenceOrbit {
    pub(crate) points: Arc<[ReferencePoint]>,
    pub(crate) revision: u64,
    pub(crate) complete: bool,
}

impl ReferenceOrbit {
    pub(crate) fn point_count(&self) -> u32 {
        self.points.len() as u32
    }
}

#[derive(Debug, Default)]
pub(crate) struct ReferenceOrbitCache {
    key: Option<ReferenceKey>,
    orbit: ReferenceOrbit,
    next_revision: u64,
}

impl ReferenceOrbitCache {
    pub(crate) fn update(&mut self, input: ReferenceInput) -> &ReferenceOrbit {
        let key = input.into();
        if self.key != Some(key) {
            self.next_revision = self.next_revision.wrapping_add(1).max(1);
            self.orbit = build_reference_orbit(input, self.next_revision);
            self.key = Some(key);
        }
        &self.orbit
    }

    pub(crate) fn invalidate(&mut self) {
        self.key = None;
    }
}

fn build_reference_orbit(input: ReferenceInput, revision: u64) -> ReferenceOrbit {
    let requested_points = (input.iterations as usize + 1).min(MAX_REFERENCE_POINTS);
    let (mut z, c) = if input.pane == 0 {
        ([0.0, 0.0], input.centre)
    } else {
        (input.centre, input.julia_c)
    };
    let mut points = Vec::with_capacity(requested_points);
    points.push(ReferencePoint::from_f64(z));

    while points.len() < requested_points {
        z = [z[0] * z[0] - z[1] * z[1] + c[0], 2.0 * z[0] * z[1] + c[1]];
        if !z[0].is_finite() || !z[1].is_finite() {
            break;
        }
        points.push(ReferencePoint::from_f64(z));
    }

    ReferenceOrbit {
        complete: points.len() == requested_points,
        points: points.into(),
        revision,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(z: [f64; 2]) -> [f64; 2] {
        [z[0] * z[0] - z[1] * z[1], 2.0 * z[0] * z[1]]
    }

    fn multiply(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
        [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
    }

    fn perturb(reference: [f64; 2], delta: [f64; 2], delta_c: [f64; 2]) -> [f64; 2] {
        let coupling = multiply(reference, delta);
        let delta_square = square(delta);
        [
            2.0 * coupling[0] + delta_square[0] + delta_c[0],
            2.0 * coupling[1] + delta_square[1] + delta_c[1],
        ]
    }

    #[test]
    fn parameter_reference_starts_at_the_critical_point() {
        let orbit = build_reference_orbit(
            ReferenceInput {
                centre: [1.0, 0.0],
                julia_c: [0.0, 0.0],
                iterations: 3,
                pane: 0,
            },
            1,
        );
        let values: Vec<_> = orbit.points.iter().map(|point| point.as_f64()).collect();
        assert_eq!(values, [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [5.0, 0.0]]);
        assert!(orbit.complete);
    }

    #[test]
    fn julia_reference_starts_at_the_view_centre() {
        let orbit = build_reference_orbit(
            ReferenceInput {
                centre: [0.5, 0.0],
                julia_c: [0.0, 0.0],
                iterations: 2,
                pane: 1,
            },
            1,
        );
        let values: Vec<_> = orbit.points.iter().map(|point| point.as_f64()).collect();
        assert_eq!(values, [[0.5, 0.0], [0.25, 0.0], [0.0625, 0.0]]);
    }

    #[test]
    fn parameter_perturbation_matches_direct_iteration() {
        let reference_c = [-0.745, 0.113];
        let delta_c = [2.0e-12, -3.0e-12];
        let pixel_c = [reference_c[0] + delta_c[0], reference_c[1] + delta_c[1]];
        let mut reference = [0.0, 0.0];
        let mut delta = [0.0, 0.0];
        let mut direct = [0.0, 0.0];

        for _ in 0..20 {
            delta = perturb(reference, delta, delta_c);
            let reference_square = square(reference);
            reference = [
                reference_square[0] + reference_c[0],
                reference_square[1] + reference_c[1],
            ];
            let direct_square = square(direct);
            direct = [direct_square[0] + pixel_c[0], direct_square[1] + pixel_c[1]];
            assert!((reference[0] + delta[0] - direct[0]).abs() < 2e-14);
            assert!((reference[1] + delta[1] - direct[1]).abs() < 2e-14);
        }
    }

    #[test]
    fn julia_perturbation_matches_direct_iteration() {
        let c = [-0.123, 0.745];
        let mut reference = [0.25, -0.1];
        let mut delta = [4.0e-12, 7.0e-12];
        let mut direct = [reference[0] + delta[0], reference[1] + delta[1]];

        for _ in 0..20 {
            delta = perturb(reference, delta, [0.0, 0.0]);
            let reference_square = square(reference);
            reference = [reference_square[0] + c[0], reference_square[1] + c[1]];
            let direct_square = square(direct);
            direct = [direct_square[0] + c[0], direct_square[1] + c[1]];
            assert!((reference[0] + delta[0] - direct[0]).abs() < 2e-13);
            assert!((reference[1] + delta[1] - direct[1]).abs() < 2e-13);
        }
    }

    #[test]
    fn cache_reuses_an_unchanged_reference() {
        let input = ReferenceInput {
            centre: [-0.745, 0.113],
            julia_c: [-0.745, 0.113],
            iterations: 256,
            pane: 0,
        };
        let mut cache = ReferenceOrbitCache::default();
        let first = cache.update(input).revision;
        let second = cache.update(input).revision;
        assert_eq!(first, second);

        let changed = cache
            .update(ReferenceInput {
                centre: [-0.745, 0.114],
                ..input
            })
            .revision;
        assert_ne!(first, changed);

        cache.invalidate();
        let invalidated = cache.update(input).revision;
        assert_ne!(changed, invalidated);
    }
}
