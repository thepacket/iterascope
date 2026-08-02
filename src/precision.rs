//! Numerical precision policy and lightweight orbit diagnostics.
//!
//! Rendering remains GPU-first. The CPU probe evaluates only nine orbits when
//! a settled view changes; it never renders pixels or builds reference images.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PrecisionMode {
    #[default]
    F32,
    DoubleSingle,
    Perturbation,
}

impl PrecisionMode {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::F32 => "GPU f32",
            Self::DoubleSingle => "GPU DS ~48-bit",
            Self::Perturbation => "GPU PERT f64-ref",
        }
    }

    pub(crate) fn shader_flag(self) -> f32 {
        match self {
            Self::F32 => 0.0,
            Self::DoubleSingle => 1.0,
            Self::Perturbation => 2.0,
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

    pub(crate) fn sub(self, other: Self) -> Self {
        self.add(Self {
            hi: -other.hi,
            lo: -other.lo,
        })
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct PathProbeResult {
    pub(crate) unstable_samples: u8,
    pub(crate) classification_mismatches: u8,
    pub(crate) non_finite_samples: u8,
    pub(crate) max_escape_delta: f64,
    pub(crate) max_orbit_delta: f64,
}

impl PathProbeResult {
    pub(crate) fn unstable(self) -> bool {
        self.non_finite_samples > 0
            || self.classification_mismatches > 0
            || self.unstable_samples >= 2
    }

    pub(crate) fn summary(self) -> String {
        if self.non_finite_samples > 0 {
            format!("{} / 9 non-finite orbits", self.non_finite_samples)
        } else if self.classification_mismatches > 0 {
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ProbeResult {
    pub(crate) f32: PathProbeResult,
    pub(crate) ds: PathProbeResult,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ValidityLevel {
    #[default]
    Stable,
    Risk,
    Limit,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ValidityReason {
    #[default]
    SamplesAgree,
    CoordinateRisk,
    CoordinateCollapse,
    OrbitDivergence(u8),
    ClassificationMismatch(u8),
    NonFinite(u8),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DsValidity {
    pub(crate) level: ValidityLevel,
    pub(crate) reason: ValidityReason,
}

impl DsValidity {
    pub(crate) fn from_probe(probe: PathProbeResult, coordinate_ratio: f64) -> Self {
        if coordinate_ratio >= 0.5 {
            return Self {
                level: ValidityLevel::Limit,
                reason: ValidityReason::CoordinateCollapse,
            };
        }
        if probe.non_finite_samples > 0 {
            return Self {
                level: ValidityLevel::Limit,
                reason: ValidityReason::NonFinite(probe.non_finite_samples),
            };
        }
        if probe.classification_mismatches >= 3 {
            return Self {
                level: ValidityLevel::Limit,
                reason: ValidityReason::ClassificationMismatch(probe.classification_mismatches),
            };
        }
        if probe.classification_mismatches > 0 {
            return Self {
                level: ValidityLevel::Risk,
                reason: ValidityReason::ClassificationMismatch(probe.classification_mismatches),
            };
        }
        if probe.unstable_samples >= 5 {
            return Self {
                level: ValidityLevel::Limit,
                reason: ValidityReason::OrbitDivergence(probe.unstable_samples),
            };
        }
        if probe.unstable_samples >= 2 {
            return Self {
                level: ValidityLevel::Risk,
                reason: ValidityReason::OrbitDivergence(probe.unstable_samples),
            };
        }
        if coordinate_ratio >= 0.0625 {
            return Self {
                level: ValidityLevel::Risk,
                reason: ValidityReason::CoordinateRisk,
            };
        }
        Self::default()
    }

    pub(crate) fn label(self) -> &'static str {
        match (self.level, self.reason) {
            (ValidityLevel::Stable, _) => "DS STABLE",
            (ValidityLevel::Risk, _) => "DS RISK",
            (ValidityLevel::Limit, ValidityReason::CoordinateCollapse) => "DS COORD LIMIT",
            (ValidityLevel::Limit, _) => "RENDER LIMIT",
        }
    }

    pub(crate) fn render_limited(self) -> bool {
        self.level == ValidityLevel::Limit
            && !matches!(self.reason, ValidityReason::CoordinateCollapse)
    }

    pub(crate) fn summary(self) -> String {
        match self.reason {
            ValidityReason::SamplesAgree => "DS samples agree with f64".to_owned(),
            ValidityReason::CoordinateRisk => "DS pixel spacing approaching limit".to_owned(),
            ValidityReason::CoordinateCollapse => "DS adjacent coordinates may collapse".to_owned(),
            ValidityReason::OrbitDivergence(count) => {
                format!("{count} / 9 DS orbits diverge from f64")
            }
            ValidityReason::ClassificationMismatch(count) => {
                format!("{count} / 9 DS classifications disagree")
            }
            ValidityReason::NonFinite(count) => format!("{count} / 9 DS orbits non-finite"),
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
            let f32_outcome = orbit_f32(z0, c, input.iterations, input.bailout);
            let local_offset = [
                (x as f32 * input.aspect as f32) * input.half_height as f32,
                y as f32 * input.half_height as f32,
            ];
            let ds_outcome = orbit_adaptive_ds(input, local_offset);

            compare_outcome(&mut result.f32, precise, f32_outcome);
            compare_outcome(&mut result.ds, precise, ds_outcome);
        }
    }
    result
}

fn compare_outcome(result: &mut PathProbeResult, precise: OrbitOutcome, candidate: OrbitOutcome) {
    if !candidate.smooth_iteration.is_finite()
        || !candidate.z[0].is_finite()
        || !candidate.z[1].is_finite()
    {
        result.non_finite_samples += 1;
        result.unstable_samples += 1;
        return;
    }
    if precise.escaped != candidate.escaped {
        result.classification_mismatches += 1;
        result.unstable_samples += 1;
        return;
    }
    if precise.escaped {
        let delta = (precise.smooth_iteration - candidate.smooth_iteration).abs();
        result.max_escape_delta = result.max_escape_delta.max(delta);
        if delta > 0.2 {
            result.unstable_samples += 1;
        }
    } else {
        let delta = ((precise.z[0] - candidate.z[0]).powi(2)
            + (precise.z[1] - candidate.z[1]).powi(2))
        .sqrt();
        result.max_orbit_delta = result.max_orbit_delta.max(delta);
        if delta > 1e-3 * (1.0 + precise.z[0].hypot(precise.z[1])) {
            result.unstable_samples += 1;
        }
    }
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

fn orbit_adaptive_ds(input: ProbeInput, local_offset: [f32; 2]) -> OrbitOutcome {
    let zero = [DoubleSingle::default(); 2];
    let centre = [
        DoubleSingle::from_f64(input.centre[0]),
        DoubleSingle::from_f64(input.centre[1]),
    ];
    let julia_c = [
        DoubleSingle::from_f64(input.julia_c[0]),
        DoubleSingle::from_f64(input.julia_c[1]),
    ];
    let (mut reference_z, mut reference_c, mut delta_z, mut delta_c) = if input.pane == 0 {
        (zero, centre, [0.0; 2], local_offset)
    } else {
        (centre, julia_c, local_offset, [0.0; 2])
    };
    let bailout_squared = (input.bailout as f32) * (input.bailout as f32);
    let mut approximate = ds2_approx(reference_z);

    for iteration in 1..=input.iterations.min(4096) {
        let reference_before = ds2_approx(reference_z);
        let coupling = complex_mul_f32(reference_before, delta_z);
        let delta_square = complex_square_f32(delta_z);
        delta_z = [
            2.0 * coupling[0] + delta_square[0] + delta_c[0],
            2.0 * coupling[1] + delta_square[1] + delta_c[1],
        ];
        let square = ds_complex_square(reference_z);
        reference_z = [square[0].add(reference_c[0]), square[1].add(reference_c[1])];
        let reference_approximate = ds2_approx(reference_z);
        approximate = [
            reference_approximate[0] + delta_z[0],
            reference_approximate[1] + delta_z[1],
        ];

        let reference_size = reference_approximate[0].hypot(reference_approximate[1]);
        let delta_size = delta_z[0].hypot(delta_z[1]);
        let cancellation_scale = reference_size.max(delta_size);
        let delta_is_large = delta_size > 0.03125 * (1.0 + reference_size);
        let cancellation = cancellation_scale > 1e-4
            && approximate[0].hypot(approximate[1]) < 0.01 * cancellation_scale;
        if delta_is_large || cancellation {
            reference_z = ds2_add_f32(reference_z, delta_z);
            reference_c = ds2_add_f32(reference_c, delta_c);
            delta_z = [0.0; 2];
            delta_c = [0.0; 2];
            approximate = ds2_approx(reference_z);
        }

        let magnitude_squared = approximate[0] * approximate[0] + approximate[1] * approximate[1];
        if magnitude_squared > bailout_squared {
            let log_zn = 0.5 * magnitude_squared.max(1.000_001).ln();
            return OrbitOutcome {
                escaped: true,
                smooth_iteration: (iteration as f32 + 1.0 - log_zn.max(1e-6).log2()) as f64,
                z: [approximate[0] as f64, approximate[1] as f64],
            };
        }
    }

    OrbitOutcome {
        escaped: false,
        smooth_iteration: input.iterations as f64,
        z: [approximate[0] as f64, approximate[1] as f64],
    }
}

fn ds2_approx(value: [DoubleSingle; 2]) -> [f32; 2] {
    [value[0].hi + value[0].lo, value[1].hi + value[1].lo]
}

fn ds2_add_f32(value: [DoubleSingle; 2], offset: [f32; 2]) -> [DoubleSingle; 2] {
    [
        value[0].add(DoubleSingle::from_f32(offset[0])),
        value[1].add(DoubleSingle::from_f32(offset[1])),
    ]
}

fn ds_complex_square(value: [DoubleSingle; 2]) -> [DoubleSingle; 2] {
    let xx = value[0].mul(value[0]);
    let yy = value[1].mul(value[1]);
    let xy = value[0].mul(value[1]);
    [xx.sub(yy), DoubleSingle::from_f32(2.0).mul(xy)]
}

fn complex_square_f32(value: [f32; 2]) -> [f32; 2] {
    [
        value[0] * value[0] - value[1] * value[1],
        2.0 * value[0] * value[1],
    ]
}

fn complex_mul_f32(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
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
            cache.current(input).unwrap().f32.unstable_samples,
            first.f32.unstable_samples
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
        assert_eq!(result.f32.classification_mismatches, 0);
        assert!(!result.f32.unstable());
        assert_eq!(result.ds.classification_mismatches, 0);
        assert!(!result.ds.unstable());
    }

    #[test]
    fn adaptive_ds_probe_improves_the_reported_deep_view() {
        let result = run_probe(ProbeInput {
            centre: [-0.015_945_7, 1.013_91],
            half_height: 1.45 / 119_577.0,
            aspect: 0.71,
            julia_c: [-0.015_945_7, 1.013_91],
            iterations: 2048,
            bailout: 6.1,
            pane: 0,
        });
        assert_eq!(result.ds.non_finite_samples, 0);
        assert!(result.ds.classification_mismatches <= result.f32.classification_mismatches);
    }

    #[test]
    fn ds_validity_distinguishes_risk_and_limit() {
        assert_eq!(
            DsValidity::from_probe(PathProbeResult::default(), 0.0).level,
            ValidityLevel::Stable
        );
        assert_eq!(
            DsValidity::from_probe(
                PathProbeResult {
                    classification_mismatches: 1,
                    ..Default::default()
                },
                0.0,
            )
            .level,
            ValidityLevel::Risk
        );
        let coordinate_limit = DsValidity::from_probe(PathProbeResult::default(), 0.5);
        assert_eq!(coordinate_limit.level, ValidityLevel::Limit);
        assert_eq!(coordinate_limit.label(), "DS COORD LIMIT");
        assert!(!coordinate_limit.render_limited());

        let arithmetic_limit = DsValidity::from_probe(
            PathProbeResult {
                non_finite_samples: 1,
                ..Default::default()
            },
            0.0,
        );
        assert_eq!(arithmetic_limit.label(), "RENDER LIMIT");
        assert!(arithmetic_limit.render_limited());
    }
}
