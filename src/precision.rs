//! Numerical precision policy and lightweight orbit diagnostics.
//!
//! Rendering remains GPU-first. The CPU probe evaluates only nine orbits when
//! a settled view changes; it never renders pixels or builds reference images.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PrecisionMode {
    #[default]
    F32,
    DoubleSingle,
}

impl PrecisionMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::F32 => "GPU f32",
            Self::DoubleSingle => "GPU DS ~48-bit",
        }
    }

    pub(crate) fn shader_flag(self) -> f32 {
        match self {
            Self::F32 => 0.0,
            Self::DoubleSingle => 1.0,
        }
    }
}

/// Two `f32` values representing one number as their unevaluated sum.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct DoubleSingle {
    pub(crate) hi: f32,
    pub(crate) lo: f32,
}

impl DoubleSingle {
    pub(crate) fn from_f64(value: f64) -> Self {
        let hi = value as f32;
        let lo = (value - hi as f64) as f32;
        Self { hi, lo }
    }

    pub(crate) fn from_f32(value: f32) -> Self {
        Self { hi: value, lo: 0.0 }
    }

    pub(crate) fn as_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }

    fn normalize(hi: f32, lo: f32) -> Self {
        let sum = hi + lo;
        // Keep the compensation behind an explicit fused operation. GPU
        // shader compilers may otherwise reassociate the subtraction pattern
        // used by quick-two-sum and silently reduce DS arithmetic to f32.
        let error = (-1.0_f32).mul_add(sum - hi, lo);
        Self { hi: sum, lo: error }
    }

    pub(crate) fn add(self, other: Self) -> Self {
        let sum = self.hi + other.hi;
        let virtual_b = sum - self.hi;
        let a_error = (-1.0_f32).mul_add(sum - virtual_b, self.hi);
        let b_error = (-1.0_f32).mul_add(virtual_b, other.hi);
        let error = a_error + b_error + self.lo + other.lo;
        Self::normalize(sum, error)
    }

    pub(crate) fn mul(self, other: Self) -> Self {
        let product = self.hi * other.hi;
        let error = self.hi.mul_add(other.hi, -product)
            + self.hi * other.lo
            + self.lo * other.hi
            + self.lo * other.lo;
        Self::normalize(product, error)
    }
}

pub(crate) fn split_f64(value: f64) -> [f32; 2] {
    let value = DoubleSingle::from_f64(value);
    [value.hi, value.lo]
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ProbeInput {
    pub(crate) centre: [f64; 2],
    pub(crate) half_height: f64,
    pub(crate) aspect: f64,
    pub(crate) julia_c: [f64; 2],
    pub(crate) iterations: u32,
    pub(crate) bailout: f64,
    pub(crate) pane: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProbeResult {
    pub(crate) unstable_samples: u8,
    pub(crate) classification_mismatches: u8,
    pub(crate) max_escape_delta: f64,
    pub(crate) max_orbit_delta: f64,
}

impl ProbeResult {
    pub(crate) fn unstable(self) -> bool {
        self.classification_mismatches > 0 || self.unstable_samples >= 2
    }

    pub(crate) fn summary(self) -> String {
        if self.classification_mismatches > 0 {
            format!(
                "{} / 9 classifications disagree",
                self.classification_mismatches
            )
        } else if self.unstable_samples > 0 {
            format!("{} / 9 sampled orbits unstable", self.unstable_samples)
        } else {
            "9 / 9 sampled orbits agree".to_owned()
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ProbeKey {
    centre: [u64; 2],
    half_height: u64,
    aspect: u64,
    julia_c: [u64; 2],
    iterations: u32,
    bailout: u64,
    pane: usize,
}

impl From<ProbeInput> for ProbeKey {
    fn from(input: ProbeInput) -> Self {
        Self {
            centre: [input.centre[0].to_bits(), input.centre[1].to_bits()],
            half_height: input.half_height.to_bits(),
            aspect: input.aspect.to_bits(),
            julia_c: [input.julia_c[0].to_bits(), input.julia_c[1].to_bits()],
            iterations: input.iterations,
            bailout: input.bailout.to_bits(),
            pane: input.pane,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProbeCache {
    key: Option<ProbeKey>,
    result: ProbeResult,
}

impl ProbeCache {
    pub(crate) fn update(&mut self, input: ProbeInput) -> ProbeResult {
        let key = input.into();
        if self.key != Some(key) {
            self.result = run_probe(input);
            self.key = Some(key);
        }
        self.result
    }

    pub(crate) fn current(&self, input: ProbeInput) -> Option<ProbeResult> {
        (self.key == Some(input.into())).then_some(self.result)
    }

    pub(crate) fn last_result(&self) -> Option<ProbeResult> {
        self.key.map(|_| self.result)
    }
}

#[derive(Clone, Copy)]
struct OrbitOutcome {
    escaped: bool,
    smooth_iteration: f64,
    z: [f64; 2],
}

fn run_probe(input: ProbeInput) -> ProbeResult {
    let mut result = ProbeResult::default();
    // Interior points plus near-corner points cover the visible plane without
    // making the probe proportional to its pixel dimensions.
    for y in [-0.72, 0.0, 0.72] {
        for x in [-0.72, 0.0, 0.72] {
            let world = [
                input.centre[0] + x * input.aspect * input.half_height,
                input.centre[1] + y * input.half_height,
            ];
            let (z0, c) = if input.pane == 0 {
                ([0.0, 0.0], world)
            } else {
                (world, input.julia_c)
            };
            let precise = orbit_f64(z0, c, input.iterations, input.bailout);
            let gpu = orbit_f32(z0, c, input.iterations, input.bailout);

            if precise.escaped != gpu.escaped {
                result.classification_mismatches += 1;
                result.unstable_samples += 1;
                continue;
            }

            if precise.escaped {
                let delta = (precise.smooth_iteration - gpu.smooth_iteration).abs();
                result.max_escape_delta = result.max_escape_delta.max(delta);
                if delta > 0.2 {
                    result.unstable_samples += 1;
                }
            } else {
                let delta =
                    ((precise.z[0] - gpu.z[0]).powi(2) + (precise.z[1] - gpu.z[1]).powi(2)).sqrt();
                result.max_orbit_delta = result.max_orbit_delta.max(delta);
                if delta > 1e-3 * (1.0 + precise.z[0].hypot(precise.z[1])) {
                    result.unstable_samples += 1;
                }
            }
        }
    }
    result
}

fn orbit_f64(z0: [f64; 2], c: [f64; 2], iterations: u32, bailout: f64) -> OrbitOutcome {
    let mut z = z0;
    let bailout_squared = bailout * bailout;
    for iteration in 1..=iterations {
        z = [z[0] * z[0] - z[1] * z[1] + c[0], 2.0 * z[0] * z[1] + c[1]];
        let magnitude_squared = z[0] * z[0] + z[1] * z[1];
        if magnitude_squared > bailout_squared {
            let log_zn = 0.5 * magnitude_squared.max(1.000_001).ln();
            return OrbitOutcome {
                escaped: true,
                smooth_iteration: iteration as f64 + 1.0 - log_zn.max(1e-12).log2(),
                z,
            };
        }
    }
    OrbitOutcome {
        escaped: false,
        smooth_iteration: iterations as f64,
        z,
    }
}

fn orbit_f32(z0: [f64; 2], c: [f64; 2], iterations: u32, bailout: f64) -> OrbitOutcome {
    let mut z = [z0[0] as f32, z0[1] as f32];
    let c = [c[0] as f32, c[1] as f32];
    let bailout_squared = (bailout as f32) * (bailout as f32);
    for iteration in 1..=iterations {
        z = [z[0] * z[0] - z[1] * z[1] + c[0], 2.0 * z[0] * z[1] + c[1]];
        let magnitude_squared = z[0] * z[0] + z[1] * z[1];
        if magnitude_squared > bailout_squared {
            let log_zn = 0.5 * magnitude_squared.max(1.000_001).ln();
            return OrbitOutcome {
                escaped: true,
                smooth_iteration: (iteration as f32 + 1.0 - log_zn.max(1e-6).log2()) as f64,
                z: [z[0] as f64, z[1] as f64],
            };
        }
    }
    OrbitOutcome {
        escaped: false,
        smooth_iteration: iterations as f64,
        z: [z[0] as f64, z[1] as f64],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_reconstructs_about_forty_eight_bits() {
        let value = -0.745_123_456_789_012_3;
        let split = DoubleSingle::from_f64(value);
        assert!((split.as_f64() - value).abs() < 2e-15);
        assert_ne!(split.lo, 0.0);
    }

    #[test]
    fn double_single_operations_improve_on_plain_f32() {
        let a = DoubleSingle::from_f64(1.000_000_119_209_289_6);
        let b = DoubleSingle::from_f64(-1.0);
        let sum = a.add(b).as_f64();
        assert!((sum - 1.192_092_896e-7).abs() < 1e-14);

        let product = DoubleSingle::from_f64(0.745_123_456_789)
            .mul(DoubleSingle::from_f64(0.113_987_654_321))
            .as_f64();
        let expected = 0.745_123_456_789 * 0.113_987_654_321;
        assert!((product - expected).abs() < 2e-14);
    }

    #[test]
    fn probe_cache_reuses_an_exact_view() {
        let input = ProbeInput {
            centre: [-0.5, 0.0],
            half_height: 1.45,
            aspect: 1.6,
            julia_c: [-0.745, 0.113],
            iterations: 64,
            bailout: 4.0,
            pane: 0,
        };
        let mut cache = ProbeCache::default();
        let first = cache.update(input);
        assert_eq!(
            cache.current(input).unwrap().unstable_samples,
            first.unstable_samples
        );

        let changed = ProbeInput {
            half_height: 0.725,
            ..input
        };
        assert!(cache.current(changed).is_none());
    }

    #[test]
    fn initial_parameter_view_does_not_trigger_the_probe() {
        let result = run_probe(ProbeInput {
            centre: [-0.5, 0.0],
            half_height: 1.45,
            aspect: 1.6,
            julia_c: [-0.745, 0.113],
            iterations: 256,
            bailout: 4.0,
            pane: 0,
        });
        assert_eq!(result.classification_mismatches, 0);
        assert!(!result.unstable());
    }
}
