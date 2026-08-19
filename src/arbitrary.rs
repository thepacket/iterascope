//! Arbitrary-precision coordinates for deep navigation and reference orbits.
//!
//! The raster renderer still uses the stable GPU paths. This module is the
//! authoritative numerical foundation for the scaled perturbation renderer:
//! coordinates never pass through `f64` after the deep-zoom handoff.

use std::str::FromStr;

use dashu_float::DBig;

use crate::family::{
    FamilyParameters, FractalFamily, MAGNET_CONVERGENCE, MANDELBOX_BAILOUT_FACTOR,
    NOVA_CONVERGENCE, NOVA_ESCAPE,
};

/// The experimentally confirmed end of the current stable renderer.
pub const ARBITRARY_HANDOFF_ZOOM: f64 = 1.14e14;
/// Concrete acceptance target for the deep renderer.
pub const TARGET_DECIMAL_ZOOM_EXPONENT: u32 = 1_000;
/// Current user-selectable arbitrary-precision navigation ceiling.
pub const MAX_DECIMAL_ZOOM_EXPONENT: u32 = 5_000;

const GUARD_DECIMAL_DIGITS: u32 = 32;
const MIN_DECIMAL_DIGITS: u32 = 48;

/// Decimal precision used at a given magnification.
///
/// One digit is required per order of magnification. Guard digits cover
/// coordinate transforms and reference-orbit error growth. The policy is
/// intentionally decimal because experiment documents use exact decimal
/// strings and the product requirement is stated as `10^1000`.
pub fn decimal_digits_for_zoom_exponent(exponent: u32) -> usize {
    exponent
        .saturating_add(GUARD_DECIMAL_DIGITS)
        .max(MIN_DECIMAL_DIGITS) as usize
}

pub fn binary_bits_for_zoom_exponent(exponent: u32) -> usize {
    // ceil(decimal_digits * log2(10)); the slightly high rational bound keeps
    // this deterministic and avoids an f64 decision in the precision policy.
    let digits = decimal_digits_for_zoom_exponent(exponent) as u64;
    ((digits * 3_322 + 999) / 1_000) as usize
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeepReal {
    value: DBig,
    decimal_digits: usize,
}

impl DeepReal {
    pub fn parse(value: &str, zoom_exponent: u32) -> Result<Self, String> {
        let decimal_digits = decimal_digits_for_zoom_exponent(zoom_exponent);
        let value = DBig::from_str(value)
            .map_err(|error| error.to_string())?
            .with_precision(decimal_digits)
            .value();
        Ok(Self {
            value,
            decimal_digits,
        })
    }

    pub fn from_f64(value: f64, zoom_exponent: u32) -> Result<Self, String> {
        if !value.is_finite() {
            return Err("deep coordinate must be finite".to_owned());
        }
        // Seventeen significant digits round-trip every finite f64. Parsing
        // that decimal avoids making binary64 part of subsequent operations.
        Self::parse(&format!("{value:.17e}"), zoom_exponent)
    }

    pub fn precision(&self) -> usize {
        self.decimal_digits
    }

    pub fn exact_decimal(&self) -> String {
        format!("{:e}", self.value)
    }

    pub fn scientific(&self, fractional_digits: usize) -> String {
        format!("{:.*e}", fractional_digits, self.value)
    }

    /// A lossy projection for diagnostics and GPU reference-orbit upload.
    /// It must never be used to update an authoritative deep coordinate.
    pub fn to_f64(&self) -> f64 {
        self.value.to_f64().value()
    }

    /// Return `mantissa × 2^exponent` without first squeezing the value into
    /// an IEEE float. The mantissa is zero or has magnitude in `[1, 2)`.
    pub fn scaled_f32(&self) -> (f32, i32) {
        if self.value == DBig::ZERO {
            return (0.0, 0);
        }
        let binary = self.value.to_binary().value();
        let exponent = binary.repr().exponent() + binary.repr().digits() as isize - 1;
        let normalized = binary >> exponent;
        let exponent = i32::try_from(exponent).expect("deep exponent exceeds GPU i32 range");
        (normalized.to_f32().value(), exponent)
    }

    pub fn add(&self, other: &Self) -> Self {
        self.rounded(&self.value + &other.value)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.rounded(&self.value - &other.value)
    }

    pub fn mul(&self, other: &Self) -> Self {
        self.rounded(&self.value * &other.value)
    }

    pub fn mul_i32(&self, factor: i32) -> Self {
        self.rounded(&self.value * factor)
    }

    pub fn div(&self, other: &Self) -> Self {
        self.rounded(&self.value / &other.value)
    }

    pub fn neg(&self) -> Self {
        self.rounded(&self.value * -1)
    }

    pub fn abs(&self) -> Self {
        if self.value < DBig::ZERO {
            self.neg()
        } else {
            self.clone()
        }
    }

    pub fn is_zero(&self) -> bool {
        self.value == DBig::ZERO
    }

    pub fn is_negative(&self) -> bool {
        self.value < DBig::ZERO
    }

    /// A constant carried at the same precision as `self`. Only used for
    /// small exactly-representable parameters such as family settings.
    pub fn constant_like(&self, value: f64) -> Self {
        let parsed = DBig::from_str(&format!("{value:.17e}"))
            .expect("finite f64 formats as a decimal")
            .with_precision(self.decimal_digits)
            .value();
        Self {
            value: parsed,
            decimal_digits: self.decimal_digits,
        }
    }

    pub fn with_zoom_exponent(&self, zoom_exponent: u32) -> Self {
        let decimal_digits = decimal_digits_for_zoom_exponent(zoom_exponent);
        let value = self.value.clone().with_precision(decimal_digits).value();
        Self {
            value,
            decimal_digits,
        }
    }

    fn rounded(&self, value: DBig) -> Self {
        let value = value.with_precision(self.decimal_digits).value();
        Self {
            value,
            decimal_digits: self.decimal_digits,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeepComplex {
    pub re: DeepReal,
    pub im: DeepReal,
}

impl DeepComplex {
    pub fn parse(re: &str, im: &str, zoom_exponent: u32) -> Result<Self, String> {
        Ok(Self {
            re: DeepReal::parse(re, zoom_exponent)?,
            im: DeepReal::parse(im, zoom_exponent)?,
        })
    }

    pub fn from_f64(value: [f64; 2], zoom_exponent: u32) -> Result<Self, String> {
        Ok(Self {
            re: DeepReal::from_f64(value[0], zoom_exponent)?,
            im: DeepReal::from_f64(value[1], zoom_exponent)?,
        })
    }

    pub fn square_add(&self, c: &Self) -> Self {
        let re = self.re.mul(&self.re).sub(&self.im.mul(&self.im)).add(&c.re);
        let im = self.re.mul(&self.im).mul_i32(2).add(&c.im);
        Self { re, im }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re.add(&other.re),
            im: self.im.add(&other.im),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self {
            re: self.re.sub(&other.re),
            im: self.im.sub(&other.im),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self {
            re: self.re.mul(&other.re).sub(&self.im.mul(&other.im)),
            im: self.re.mul(&other.im).add(&self.im.mul(&other.re)),
        }
    }

    pub fn div(&self, other: &Self) -> Self {
        let denominator = other.re.mul(&other.re).add(&other.im.mul(&other.im));
        if denominator.is_zero() {
            // A pole: the f64 reference produces a non-finite value here and
            // the orbit is classified as escaped. Mirror that with a value far
            // beyond every escape radius.
            return self.real_like(1e300);
        }
        Self {
            re: self
                .re
                .mul(&other.re)
                .add(&self.im.mul(&other.im))
                .div(&denominator),
            im: self
                .im
                .mul(&other.re)
                .sub(&self.re.mul(&other.im))
                .div(&denominator),
        }
    }

    pub fn scale_i32(&self, factor: i32) -> Self {
        Self {
            re: self.re.mul_i32(factor),
            im: self.im.mul_i32(factor),
        }
    }

    pub fn scale_real(&self, factor: &DeepReal) -> Self {
        Self {
            re: self.re.mul(factor),
            im: self.im.mul(factor),
        }
    }

    pub fn square(&self) -> Self {
        Self {
            re: self.re.mul(&self.re).sub(&self.im.mul(&self.im)),
            im: self.re.mul(&self.im).mul_i32(2),
        }
    }

    pub fn pow(&self, degree: u32) -> Self {
        let mut result = self.clone();
        for _ in 1..degree {
            result = result.mul(self);
        }
        result
    }

    pub fn real_like(&self, value: f64) -> Self {
        Self {
            re: self.re.constant_like(value),
            im: self.re.constant_like(0.0),
        }
    }

    pub fn to_f64_pair(&self) -> [f64; 2] {
        [self.re.to_f64(), self.im.to_f64()]
    }

    fn with_zoom_exponent(&self, zoom_exponent: u32) -> Self {
        Self {
            re: self.re.with_zoom_exponent(zoom_exponent),
            im: self.im.with_zoom_exponent(zoom_exponent),
        }
    }

    fn zero_like(value: &Self) -> Result<Self, String> {
        let exponent = value
            .re
            .precision()
            .saturating_sub(GUARD_DECIMAL_DIGITS as usize) as u32;
        Ok(Self {
            re: DeepReal::parse("0", exponent)?,
            im: DeepReal::parse("0", exponent)?,
        })
    }

    fn magnitude_squared_f64(&self) -> f64 {
        self.re.mul(&self.re).add(&self.im.mul(&self.im)).to_f64()
    }
}

/// Arbitrary-precision orbit state: the same triple the shader and the CPU
/// `f64` reference track for every family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeepState {
    pub z: DeepComplex,
    pub z_prev: DeepComplex,
    pub c: DeepComplex,
}

impl DeepState {
    /// Arbitrary-precision counterpart of `family::initial_state_with`.
    pub(crate) fn initial(
        family: FractalFamily,
        world: &DeepComplex,
        dynamical: bool,
        parameter: &DeepComplex,
    ) -> Result<Self, String> {
        let zero = DeepComplex::zero_like(world)?;
        if dynamical {
            let z_prev = if family == FractalFamily::Manowar {
                world.clone()
            } else {
                zero
            };
            return Ok(Self {
                z: world.clone(),
                z_prev,
                c: parameter.clone(),
            });
        }
        let c = world.clone();
        let (z, z_prev) = match family {
            FractalFamily::Lambda => (world.real_like(0.5), zero),
            FractalFamily::Manowar => (c.clone(), c.clone()),
            FractalFamily::Nova => (world.real_like(1.0), zero),
            FractalFamily::BarnsleyOne | FractalFamily::BarnsleyTwo => (c.clone(), zero),
            _ => (zero.clone(), zero),
        };
        Ok(Self { z, z_prev, c })
    }
}

/// One arbitrary-precision reference orbit shared by every GPU pixel.
///
/// Its CPU cost is O(iterations), never O(pixels × iterations).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceOrbit {
    pub points: Vec<DeepComplex>,
    /// Iteration at which the reference escaped or converged, if it did.
    pub escape_iteration: Option<u32>,
}

/// Arbitrary-precision transcription of `family::step` for the families that
/// support perturbation rendering. The `f64` implementation remains the
/// definition of record; `deep_step_matches_f64_reference` checks agreement.
pub(crate) fn deep_step(
    family: FractalFamily,
    parameters: &FamilyParameters,
    state: &DeepState,
) -> DeepState {
    let DeepState { z, z_prev, c } = state;
    let one = z.real_like(1.0);
    let two = z.real_like(2.0);
    let mut next_prev = z_prev.clone();
    let mut next_c = c.clone();
    let (x, y) = (&z.re, &z.im);
    let xx = x.mul(x);
    let yy = y.mul(y);
    let xy = x.mul(y);
    let real_square = xx.sub(&yy);
    let next = match family {
        FractalFamily::Quadratic => z.square_add(c),
        FractalFamily::Multibrot => z.pow(parameters.degree).add(c),
        FractalFamily::Tricorn => DeepComplex {
            re: real_square.add(&c.re),
            im: xy.mul_i32(-2).add(&c.im),
        },
        FractalFamily::PerpendicularMandelbrot => DeepComplex {
            re: real_square.add(&c.re),
            im: x.abs().mul(y).mul_i32(-2).add(&c.im),
        },
        FractalFamily::BurningShip => DeepComplex {
            re: real_square.add(&c.re),
            im: xy.abs().mul_i32(2).add(&c.im),
        },
        FractalFamily::PerpendicularBurningShip => DeepComplex {
            re: real_square.add(&c.re),
            im: x.mul(&y.abs()).mul_i32(-2).add(&c.im),
        },
        FractalFamily::Celtic => DeepComplex {
            re: real_square.abs().add(&c.re),
            im: xy.mul_i32(2).add(&c.im),
        },
        FractalFamily::PerpendicularCeltic => DeepComplex {
            re: real_square.abs().add(&c.re),
            im: x.abs().mul(y).mul_i32(-2).add(&c.im),
        },
        FractalFamily::Buffalo => DeepComplex {
            re: real_square.abs().add(&c.re),
            im: xy.abs().mul_i32(2).add(&c.im),
        },
        FractalFamily::PerpendicularBuffalo => DeepComplex {
            re: real_square.abs().add(&c.re),
            im: x.mul(&y.abs()).mul_i32(-2).add(&c.im),
        },
        FractalFamily::Lambda => c.mul(&z.mul(&one.sub(z))),
        FractalFamily::Phoenix => {
            next_prev = z.clone();
            let square = z.square();
            DeepComplex {
                re: square.re.add(&c.re).add(&c.im.mul(&z_prev.re)),
                im: square.im.add(&c.im.mul(&z_prev.im)),
            }
        }
        FractalFamily::Manowar => {
            next_prev = z.clone();
            z.square().add(z_prev).add(c)
        }
        FractalFamily::Spider => {
            let next = z.square_add(c);
            next_c = c.scale_real(&z.re.constant_like(0.5)).add(&next);
            next
        }
        FractalFamily::MagnetOne => {
            let numerator = z.square().add(c).sub(&one);
            let denominator = z.scale_i32(2).add(c).sub(&two);
            numerator.div(&denominator).square()
        }
        FractalFamily::MagnetTwo => {
            let c1 = c.sub(&one);
            let c2 = c.sub(&two);
            let c12 = c1.mul(&c2);
            let z2 = z.square();
            let z3 = z2.mul(z);
            let numerator = z3.add(&c1.mul(z).scale_i32(3)).add(&c12);
            let denominator = z2
                .scale_i32(3)
                .add(&c2.mul(z).scale_i32(3))
                .add(&c12)
                .add(&one);
            numerator.div(&denominator).square()
        }
        FractalFamily::Nova | FractalFamily::NewtonCubic => {
            let (p, relaxation) = if family == FractalFamily::NewtonCubic {
                (3, 1.0)
            } else {
                (parameters.degree, parameters.nova_relaxation)
            };
            let inverse = one.div(&z.pow(p - 1));
            let newton_step = z
                .sub(&inverse)
                .scale_real(&z.re.constant_like(1.0 / p as f64));
            let shifted = z.sub(&newton_step.scale_real(&z.re.constant_like(relaxation)));
            if family == FractalFamily::NewtonCubic {
                shifted
            } else {
                shifted.add(c)
            }
        }
        FractalFamily::BarnsleyOne => {
            if x.is_negative() {
                z.add(&one).mul(c)
            } else {
                z.sub(&one).mul(c)
            }
        }
        FractalFamily::BarnsleyTwo => {
            let test = x.mul(&c.im).add(&c.re.mul(y));
            if test.is_negative() {
                z.add(&one).mul(c)
            } else {
                z.sub(&one).mul(c)
            }
        }
        FractalFamily::Mandelbox => {
            let fold = |value: &DeepReal| -> DeepReal {
                let approx = value.to_f64();
                if approx > 1.0 {
                    value.constant_like(2.0).sub(value)
                } else if approx < -1.0 {
                    value.constant_like(-2.0).sub(value)
                } else {
                    value.clone()
                }
            };
            let folded = DeepComplex {
                re: fold(x),
                im: fold(y),
            };
            let radius_squared = folded.re.mul(&folded.re).add(&folded.im.mul(&folded.im));
            let min_squared = parameters.mandelbox_min_radius * parameters.mandelbox_min_radius;
            let fixed_squared =
                parameters.mandelbox_fixed_radius * parameters.mandelbox_fixed_radius;
            let radius_approx = radius_squared.to_f64();
            let ball = if radius_approx < min_squared {
                folded.scale_real(&x.constant_like(fixed_squared / min_squared))
            } else if radius_approx < fixed_squared {
                folded.scale_real(&x.constant_like(fixed_squared).div(&radius_squared))
            } else {
                folded
            };
            ball.scale_real(&x.constant_like(parameters.mandelbox_scale))
                .add(c)
        }
        // Families without a perturbation path are never iterated here.
        FractalFamily::Exponential
        | FractalFamily::Sine
        | FractalFamily::Cosine
        | FractalFamily::Collatz
        | FractalFamily::Lyapunov => z.clone(),
    };
    DeepState {
        z: next,
        z_prev: next_prev,
        c: next_c,
    }
}

impl ReferenceOrbit {
    pub fn quadratic_parameter(
        c: &DeepComplex,
        iterations: u32,
        bailout: f64,
    ) -> Result<Self, String> {
        Self::quadratic(DeepComplex::zero_like(c)?, c, iterations, bailout)
    }

    pub fn quadratic_julia(
        initial: DeepComplex,
        c: &DeepComplex,
        iterations: u32,
        bailout: f64,
    ) -> Result<Self, String> {
        Self::quadratic(initial, c, iterations, bailout)
    }

    /// Reference orbit of any perturbation-capable family, terminating on the
    /// same escape or convergence tests as the shader and `family::diagnose`.
    pub(crate) fn family(
        family: FractalFamily,
        parameters: &FamilyParameters,
        initial: DeepState,
        iterations: u32,
        bailout: f64,
    ) -> Result<Self, String> {
        if !bailout.is_finite() || bailout <= 0.0 {
            return Err("reference-orbit bailout must be positive and finite".to_owned());
        }
        if !family.supports_deep_zoom() {
            return Err(format!(
                "{} has no arbitrary-precision reference orbit",
                family.name()
            ));
        }
        if family == FractalFamily::Quadratic {
            return Self::quadratic(initial.z, &initial.c, iterations, bailout);
        }
        let escape_squared = match family {
            FractalFamily::Mandelbox => {
                let radius = bailout * MANDELBOX_BAILOUT_FACTOR;
                radius * radius
            }
            FractalFamily::Nova => NOVA_ESCAPE * NOVA_ESCAPE,
            FractalFamily::NewtonCubic => 1e36,
            _ => bailout * bailout,
        };
        let mut state = initial;
        let mut points = Vec::with_capacity(iterations as usize + 1);
        points.push(state.z.clone());
        for iteration in 1..=iterations {
            let previous = state.z.to_f64_pair();
            state = deep_step(family, parameters, &state);
            points.push(state.z.clone());
            let z = state.z.to_f64_pair();
            let magnitude_squared = z[0] * z[0] + z[1] * z[1];
            let converged = match family {
                FractalFamily::MagnetOne | FractalFamily::MagnetTwo => {
                    (z[0] - 1.0).hypot(z[1]) < MAGNET_CONVERGENCE
                }
                FractalFamily::Nova => {
                    (z[0] - previous[0]).hypot(z[1] - previous[1]) < NOVA_CONVERGENCE
                }
                FractalFamily::NewtonCubic => {
                    let z2 = [z[0] * z[0] - z[1] * z[1], 2.0 * z[0] * z[1]];
                    let z3 = [z2[0] * z[0] - z2[1] * z[1], z2[0] * z[1] + z2[1] * z[0]];
                    (z3[0] - 1.0).hypot(z3[1]) <= 1e-6
                }
                _ => false,
            };
            if !magnitude_squared.is_finite() || magnitude_squared > escape_squared || converged {
                return Ok(Self {
                    points,
                    escape_iteration: Some(iteration),
                });
            }
        }
        Ok(Self {
            points,
            escape_iteration: None,
        })
    }

    fn quadratic(
        mut z: DeepComplex,
        c: &DeepComplex,
        iterations: u32,
        bailout: f64,
    ) -> Result<Self, String> {
        if !bailout.is_finite() || bailout <= 0.0 {
            return Err("reference-orbit bailout must be positive and finite".to_owned());
        }

        let mut points = Vec::with_capacity(iterations as usize + 1);
        points.push(z.clone());
        let bailout_squared = bailout * bailout;
        for iteration in 1..=iterations {
            z = z.square_add(c);
            points.push(z.clone());
            if z.magnitude_squared_f64() > bailout_squared {
                return Ok(Self {
                    points,
                    escape_iteration: Some(iteration),
                });
            }
        }
        Ok(Self {
            points,
            escape_iteration: None,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepView {
    pub centre: DeepComplex,
    pub half_height: DeepReal,
    pub zoom_exponent: u32,
    pub magnification_log10: f64,
}

impl DeepView {
    pub fn parse(
        centre_re: &str,
        centre_im: &str,
        half_height: &str,
        magnification_log10: f64,
    ) -> Result<Self, String> {
        let handoff_log = ARBITRARY_HANDOFF_ZOOM.log10();
        if !magnification_log10.is_finite()
            || !(handoff_log..=MAX_DECIMAL_ZOOM_EXPONENT as f64).contains(&magnification_log10)
        {
            return Err(format!(
                "deep magnification log10 must be between {handoff_log} and {}",
                MAX_DECIMAL_ZOOM_EXPONENT
            ));
        }
        let zoom_exponent = magnification_log10.ceil() as u32;
        let half_height = DeepReal::parse(half_height, zoom_exponent)?;
        if half_height.value <= DBig::ZERO {
            return Err("deep half-height must be positive".to_owned());
        }
        Ok(Self {
            centre: DeepComplex::parse(centre_re, centre_im, zoom_exponent)?,
            half_height,
            zoom_exponent,
            magnification_log10,
        })
    }

    pub fn at_handoff(centre: [f64; 2]) -> Result<Self, String> {
        let zoom_exponent = ARBITRARY_HANDOFF_ZOOM.log10().ceil() as u32;
        Ok(Self {
            centre: DeepComplex::from_f64(centre, zoom_exponent)?,
            half_height: DeepReal::parse(
                &format!("{:.17e}", 1.45 / ARBITRARY_HANDOFF_ZOOM),
                zoom_exponent,
            )?,
            zoom_exponent,
            magnification_log10: ARBITRARY_HANDOFF_ZOOM.log10(),
        })
    }

    pub fn target_scale(centre: [f64; 2]) -> Result<Self, String> {
        let zoom_exponent = TARGET_DECIMAL_ZOOM_EXPONENT;
        Ok(Self {
            centre: DeepComplex::from_f64(centre, zoom_exponent)?,
            half_height: DeepReal::parse("1.45e-1000", zoom_exponent)?,
            zoom_exponent,
            magnification_log10: TARGET_DECIMAL_ZOOM_EXPONENT as f64,
        })
    }

    pub fn at_zoom_exponent(centre: [f64; 2], zoom_exponent: u32) -> Result<Self, String> {
        let minimum = ARBITRARY_HANDOFF_ZOOM.log10().ceil() as u32;
        if !(minimum..=MAX_DECIMAL_ZOOM_EXPONENT).contains(&zoom_exponent) {
            return Err(format!(
                "deep zoom exponent must be between {minimum} and {MAX_DECIMAL_ZOOM_EXPONENT}"
            ));
        }
        Ok(Self {
            centre: DeepComplex::from_f64(centre, zoom_exponent)?,
            half_height: DeepReal::parse(&format!("1.45e-{zoom_exponent}"), zoom_exponent)?,
            zoom_exponent,
            magnification_log10: zoom_exponent as f64,
        })
    }

    pub fn zoom(&mut self, factor: f64) -> Result<(), String> {
        if !factor.is_finite() || factor <= 0.0 {
            return Err("deep zoom factor must be positive and finite".to_owned());
        }
        let next_log =
            (self.magnification_log10 - factor.log10()).min(MAX_DECIMAL_ZOOM_EXPONENT as f64);
        let next_exponent = next_log.ceil() as u32;
        self.centre = self.centre.with_zoom_exponent(next_exponent);
        self.half_height = self.half_height.with_zoom_exponent(next_exponent);
        if next_log >= MAX_DECIMAL_ZOOM_EXPONENT as f64 {
            self.half_height = DeepReal::parse("1.45e-5000", next_exponent)?;
        } else {
            let factor = DeepReal::from_f64(factor, next_exponent)?;
            self.half_height = self.half_height.mul(&factor);
        }
        self.zoom_exponent = next_exponent;
        self.magnification_log10 = next_log;
        Ok(())
    }

    pub fn recenter_local(&mut self, local: [f64; 2]) -> Result<(), String> {
        let x = DeepReal::from_f64(local[0], self.zoom_exponent)?;
        let y = DeepReal::from_f64(local[1], self.zoom_exponent)?;
        self.centre.re = self.centre.re.add(&self.half_height.mul(&x));
        self.centre.im = self.centre.im.add(&self.half_height.mul(&y));
        Ok(())
    }

    pub fn pan_local(&mut self, local: [f64; 2]) -> Result<(), String> {
        self.recenter_local(local)
    }

    pub fn centre_preview(&self) -> [f64; 2] {
        [self.centre.re.to_f64(), self.centre.im.to_f64()]
    }

    pub fn half_height_preview(&self) -> f64 {
        self.half_height.to_f64()
    }

    pub fn magnification_label(&self) -> String {
        let exponent = self.magnification_log10.floor();
        let mantissa = 10.0_f64.powf(self.magnification_log10 - exponent);
        format!("{mantissa:.6}e{exponent:.0}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_precision_includes_one_thousand_digits_and_guards() {
        assert_eq!(decimal_digits_for_zoom_exponent(1_000), 1_032);
        assert!(binary_bits_for_zoom_exponent(1_000) >= 3_428);
    }

    #[test]
    fn target_scale_is_representable() {
        let view = DeepView::target_scale([-0.745, 0.113]).unwrap();
        assert_eq!(view.half_height.exact_decimal(), "1.45e-1000");
        assert_eq!(view.half_height.precision(), 1_032);
    }

    #[test]
    fn target_scale_converts_to_a_nonzero_scaled_gpu_number() {
        let scale = DeepReal::parse("1.45e-1000", 1_000).unwrap();
        let (mantissa, exponent) = scale.scaled_f32();
        assert!((1.0..2.0).contains(&mantissa));
        assert_eq!(exponent, -3_322);
    }

    #[test]
    fn handoff_matches_the_confirmed_renderer_boundary() {
        let view = DeepView::at_handoff([-0.745, 0.113]).unwrap();
        assert_eq!(view.zoom_exponent, 15);
        assert_eq!(view.half_height.precision(), 48);
        assert!((view.magnification_log10 - ARBITRARY_HANDOFF_ZOOM.log10()).abs() < 1e-12);
    }

    #[test]
    fn deep_zoom_and_recentre_never_round_trip_through_f64() {
        let mut view = DeepView::target_scale([-0.745, 0.113]).unwrap();
        let before = view.centre.clone();
        view.recenter_local([0.25, -0.5]).unwrap();
        assert_eq!(
            view.centre.re.sub(&before.re).exact_decimal(),
            "3.625e-1001"
        );
        assert_eq!(
            view.centre.im.sub(&before.im).exact_decimal(),
            "-7.25e-1001"
        );
    }

    #[test]
    fn zoom_clamps_at_the_configured_maximum() {
        let mut view = DeepView::at_handoff([-0.745, 0.113]).unwrap();
        for _ in 0..20 {
            view.zoom(1e-300).unwrap();
        }
        assert_eq!(view.magnification_log10, 5_000.0);
        assert_eq!(view.half_height.exact_decimal(), "1.45e-5000");
    }

    #[test]
    fn direct_five_thousand_exponent_view_is_representable() {
        let view = DeepView::at_zoom_exponent([0.0, 0.0], 5_000).unwrap();
        assert_eq!(view.half_height.exact_decimal(), "1.45e-5000");
        assert_eq!(view.half_height.precision(), 5_032);
        let (mantissa, exponent) = view.half_height.scaled_f32();
        assert!((1.0..2.0).contains(&mantissa));
        assert!(exponent < -16_000);
    }

    #[test]
    fn unit_scale_coordinate_retains_a_ten_to_minus_one_thousand_offset() {
        let centre = DeepReal::parse("-0.745", 1_000).unwrap();
        let offset = DeepReal::parse("1e-1000", 1_000).unwrap();
        let moved = centre.add(&offset);
        assert_eq!(moved.sub(&centre), offset);
    }

    #[test]
    fn quadratic_reference_step_uses_arbitrary_precision() {
        let z = DeepComplex {
            re: DeepReal::parse("1e-500", 1_000).unwrap(),
            im: DeepReal::parse("-2e-500", 1_000).unwrap(),
        };
        let c = DeepComplex {
            re: DeepReal::parse("-0.745", 1_000).unwrap(),
            im: DeepReal::parse("0.113", 1_000).unwrap(),
        };
        let next = z.square_add(&c);
        assert_eq!(next.re.sub(&c.re).exact_decimal(), "-3e-1000");
        assert_eq!(next.im.sub(&c.im).exact_decimal(), "-4e-1000");
    }

    #[test]
    fn reference_orbit_is_linear_in_iterations_not_pixels() {
        let c = DeepComplex::from_f64([0.0, 0.0], 1_000).unwrap();
        let orbit = ReferenceOrbit::quadratic_parameter(&c, 2_048, 4.0).unwrap();
        assert_eq!(orbit.points.len(), 2_049);
        assert_eq!(orbit.escape_iteration, None);
    }

    #[test]
    fn deep_step_matches_f64_reference_for_every_perturbation_family() {
        use crate::family::{OrbitState, initial_state_with, step};
        let parameters = FamilyParameters::default();
        for family in FractalFamily::ALL {
            if !family.supports_deep_zoom() {
                continue;
            }
            let world = [0.31, -0.47];
            let parameter = family.default_parameter();
            for dynamical in [false, true] {
                if !dynamical && family.linkage() == crate::family::Linkage::OverviewDetail {
                    continue;
                }
                let deep_world = DeepComplex::from_f64(world, 40).unwrap();
                let deep_parameter = DeepComplex::from_f64(parameter, 40).unwrap();
                let mut deep =
                    DeepState::initial(family, &deep_world, dynamical, &deep_parameter).unwrap();
                let mut exact: OrbitState = initial_state_with(family, world, dynamical, parameter);
                for iteration in 0..12 {
                    let a = deep.z.to_f64_pair();
                    let b = exact.z;
                    let scale = 1.0 + b[0].abs().max(b[1].abs());
                    assert!(
                        (a[0] - b[0]).abs() / scale < 1e-9 && (a[1] - b[1]).abs() / scale < 1e-9,
                        "{family:?} dynamical={dynamical} iteration {iteration}: deep {a:?} vs f64 {b:?}"
                    );
                    if !b[0].is_finite() || scale > 1e6 {
                        break;
                    }
                    deep = deep_step(family, &parameters, &deep);
                    exact = step(family, &parameters, exact);
                }
            }
        }
    }

    #[test]
    fn family_reference_orbit_stops_on_convergence() {
        let parameters = FamilyParameters::default();
        let c = DeepComplex::from_f64([1.5, 0.0], 40).unwrap();
        let initial = DeepState::initial(FractalFamily::MagnetOne, &c, false, &c).unwrap();
        let orbit =
            ReferenceOrbit::family(FractalFamily::MagnetOne, &parameters, initial, 256, 4.0)
                .unwrap();
        assert_eq!(orbit.escape_iteration, Some(1));
        assert_eq!(orbit.points.len(), 2);
    }

    #[test]
    fn escaping_reference_orbit_stops_early() {
        let c = DeepComplex::from_f64([1.0, 0.0], 1_000).unwrap();
        let orbit = ReferenceOrbit::quadratic_parameter(&c, 2_048, 4.0).unwrap();
        assert_eq!(orbit.escape_iteration, Some(3));
        assert_eq!(orbit.points.len(), 4);
    }
}
