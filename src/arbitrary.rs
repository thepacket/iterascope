//! Arbitrary-precision coordinates for deep navigation and reference orbits.
//!
//! The raster renderer still uses the stable GPU paths. This module is the
//! authoritative numerical foundation for the scaled perturbation renderer:
//! coordinates never pass through `f64` after the deep-zoom handoff.

use std::str::FromStr;

use dashu_float::DBig;

/// The experimentally confirmed end of the current stable renderer.
pub const ARBITRARY_HANDOFF_ZOOM: f64 = 1.14e14;
/// Concrete acceptance target for the deep renderer.
pub const TARGET_DECIMAL_ZOOM_EXPONENT: u32 = 1_000;

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

    /// A lossy projection for diagnostics and GPU reference-orbit upload.
    /// It must never be used to update an authoritative deep coordinate.
    pub fn to_f64(&self) -> f64 {
        self.value.to_f64().value()
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

/// One arbitrary-precision reference orbit shared by every GPU pixel.
///
/// Its CPU cost is O(iterations), never O(pixels × iterations).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceOrbit {
    pub points: Vec<DeepComplex>,
    pub escape_iteration: Option<u32>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeepView {
    pub centre: DeepComplex,
    pub half_height: DeepReal,
    pub zoom_exponent: u32,
}

impl DeepView {
    pub fn at_handoff(centre: [f64; 2]) -> Result<Self, String> {
        let zoom_exponent = ARBITRARY_HANDOFF_ZOOM.log10().ceil() as u32;
        Ok(Self {
            centre: DeepComplex::from_f64(centre, zoom_exponent)?,
            half_height: DeepReal::parse(
                &format!("{:.17e}", 1.45 / ARBITRARY_HANDOFF_ZOOM),
                zoom_exponent,
            )?,
            zoom_exponent,
        })
    }

    pub fn target_scale(centre: [f64; 2]) -> Result<Self, String> {
        let zoom_exponent = TARGET_DECIMAL_ZOOM_EXPONENT;
        Ok(Self {
            centre: DeepComplex::from_f64(centre, zoom_exponent)?,
            half_height: DeepReal::parse("1.45e-1000", zoom_exponent)?,
            zoom_exponent,
        })
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
    fn handoff_matches_the_confirmed_renderer_boundary() {
        let view = DeepView::at_handoff([-0.745, 0.113]).unwrap();
        assert_eq!(view.zoom_exponent, 15);
        assert_eq!(view.half_height.precision(), 48);
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
    fn escaping_reference_orbit_stops_early() {
        let c = DeepComplex::from_f64([1.0, 0.0], 1_000).unwrap();
        let orbit = ReferenceOrbit::quadratic_parameter(&c, 2_048, 4.0).unwrap();
        assert_eq!(orbit.escape_iteration, Some(3));
        assert_eq!(orbit.points.len(), 4);
    }
}
