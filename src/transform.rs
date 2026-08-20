//! The transformations stage: screen-space warps of the pixel's view-local
//! coordinate, applied before the dynamics see it.
//!
//! Each transformation maps the local coordinate (x spans ±aspect, y spans
//! ±1, origin at the view centre) onto another local coordinate; the warped
//! image is the fractal seen through that lens. Because the warp happens
//! before any precision path derives a world position or perturbation
//! delta, it only chooses *where near the reference* each pixel samples —
//! so every transformation, affine or not, is exact at any magnification.
//! The shader applies the same chain (`apply_transformations` in
//! fractal.wgsl); the `apply` functions here are the CPU mirror used for
//! pointer interactions, and both must stay in step.

use serde::{Deserialize, Serialize};

pub(crate) const MAX_TRANSFORMATIONS: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum TransformationKind {
    /// Rotate the view by `a` radians.
    #[default]
    Rotation,
    /// Fold across the axis through the centre at angle `a`: the half-plane
    /// below the axis shows the mirror image of the half above.
    Mirror,
    /// `a`-fold mirrored wedges around the centre, rotated by `b`.
    Kaleidoscope,
    /// Rotate each point by `a` radians per unit of its radius.
    Twist,
    /// Invert through the circle of radius `a` around the centre.
    Inversion,
}

impl TransformationKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Rotation => "Rotation",
            Self::Mirror => "Mirror",
            Self::Kaleidoscope => "Kaleidoscope",
            Self::Twist => "Twist",
            Self::Inversion => "Inversion",
        }
    }

    /// The kind code the shader switches on; 0 is reserved for "none".
    fn shader_code(self) -> f32 {
        match self {
            Self::Rotation => 1.0,
            Self::Mirror => 2.0,
            Self::Kaleidoscope => 3.0,
            Self::Twist => 4.0,
            Self::Inversion => 5.0,
        }
    }

    /// A freshly added transformation of this kind, with tasteful defaults.
    pub(crate) fn default_transformation(self) -> Transformation {
        let (a, b) = match self {
            Self::Rotation => (0.0, 0.0),
            Self::Mirror => (0.0, 0.0),
            Self::Kaleidoscope => (6.0, 0.0),
            Self::Twist => (1.0, 0.0),
            Self::Inversion => (0.8, 0.0),
        };
        Transformation { kind: self, a, b }
    }
}

/// One transformation of the chain; `a` and `b` are the kind's parameters
/// (see [`TransformationKind`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Transformation {
    pub(crate) kind: TransformationKind,
    pub(crate) a: f64,
    pub(crate) b: f64,
}

fn rotate(p: [f64; 2], angle: f64) -> [f64; 2] {
    let (sin, cos) = angle.sin_cos();
    [cos * p[0] - sin * p[1], sin * p[0] + cos * p[1]]
}

impl Transformation {
    /// The CPU mirror of the shader's warp, in f64 for pointer picking.
    pub(crate) fn apply(&self, p: [f64; 2]) -> [f64; 2] {
        match self.kind {
            TransformationKind::Rotation => rotate(p, self.a),
            TransformationKind::Mirror => {
                let mut q = rotate(p, -self.a);
                q[1] = q[1].abs();
                rotate(q, self.a)
            }
            TransformationKind::Kaleidoscope => {
                let radius = p[0].hypot(p[1]);
                let sector = std::f64::consts::PI / self.a.max(1.0);
                let span = 2.0 * sector;
                let mut theta = p[1].atan2(p[0]) - self.b;
                theta -= (theta / span).floor() * span;
                if theta > sector {
                    theta = span - theta;
                }
                theta += self.b;
                [radius * theta.cos(), radius * theta.sin()]
            }
            TransformationKind::Twist => rotate(p, self.a * p[0].hypot(p[1])),
            TransformationKind::Inversion => {
                let scale = (self.a * self.a) / (p[0] * p[0] + p[1] * p[1]).max(1e-12);
                [p[0] * scale, p[1] * scale]
            }
        }
    }

    /// The `(kind, a, b, 0)` word the shader consumes.
    pub(crate) fn shader_word(&self) -> [f32; 4] {
        [self.kind.shader_code(), self.a as f32, self.b as f32, 0.0]
    }

    pub(crate) fn validate(&self, index: usize) -> Result<(), String> {
        if !self.a.is_finite() || !self.b.is_finite() {
            return Err(format!("transformation {index}: parameters must be finite"));
        }
        let full_turns = 4.0 * std::f64::consts::TAU;
        match self.kind {
            TransformationKind::Rotation | TransformationKind::Mirror => {
                if self.a.abs() > full_turns {
                    return Err(format!(
                        "transformation {index}: angle must be within ±4 turns"
                    ));
                }
            }
            TransformationKind::Kaleidoscope => {
                if !(2.0..=64.0).contains(&self.a) {
                    return Err(format!(
                        "transformation {index}: kaleidoscope needs between 2 and 64 sectors"
                    ));
                }
                if self.b.abs() > full_turns {
                    return Err(format!(
                        "transformation {index}: rotation must be within ±4 turns"
                    ));
                }
            }
            TransformationKind::Twist => {
                if self.a.abs() > 64.0 {
                    return Err(format!(
                        "transformation {index}: twist must be within ±64 radians per unit"
                    ));
                }
            }
            TransformationKind::Inversion => {
                if !(0.01..=16.0).contains(&self.a) {
                    return Err(format!(
                        "transformation {index}: inversion radius must be between 0.01 and 16"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Applies the whole chain, first transformation first — matching the
/// shader's loop order.
pub(crate) fn apply_chain(transformations: &[Transformation], p: [f64; 2]) -> [f64; 2] {
    transformations
        .iter()
        .fold(p, |point, transformation| transformation.apply(point))
}

/// The chain as shader words plus its length.
pub(crate) fn shader_words(
    transformations: &[Transformation],
) -> ([[f32; 4]; MAX_TRANSFORMATIONS], u32) {
    let mut words = [[0.0f32; 4]; MAX_TRANSFORMATIONS];
    let count = transformations.len().min(MAX_TRANSFORMATIONS);
    for (word, transformation) in words.iter_mut().zip(transformations.iter()) {
        *word = transformation.shader_word();
    }
    (words, count as u32)
}

pub(crate) fn validate_chain(transformations: &[Transformation]) -> Result<(), String> {
    if transformations.len() > MAX_TRANSFORMATIONS {
        return Err(format!(
            "at most {MAX_TRANSFORMATIONS} transformations may be chained"
        ));
    }
    for (index, transformation) in transformations.iter().enumerate() {
        transformation.validate(index)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f64; 2], b: [f64; 2]) -> bool {
        (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12
    }

    #[test]
    fn rotation_and_mirror_behave_geometrically() {
        let quarter = Transformation {
            kind: TransformationKind::Rotation,
            a: std::f64::consts::FRAC_PI_2,
            b: 0.0,
        };
        assert!(close(quarter.apply([1.0, 0.0]), [0.0, 1.0]));
        // Mirror across the x-axis folds the lower half-plane up and leaves
        // the upper half alone.
        let mirror = Transformation {
            kind: TransformationKind::Mirror,
            a: 0.0,
            b: 0.0,
        };
        assert!(close(mirror.apply([0.3, -0.4]), [0.3, 0.4]));
        assert!(close(mirror.apply([0.3, 0.4]), [0.3, 0.4]));
        // A tilted mirror axis maps its own line to itself.
        let tilted = Transformation {
            kind: TransformationKind::Mirror,
            a: std::f64::consts::FRAC_PI_4,
            b: 0.0,
        };
        let on_axis = [0.5, 0.5];
        assert!(close(tilted.apply(on_axis), on_axis));
        // Reflection across the 45° line swaps the coordinates.
        assert!(close(tilted.apply([0.5, -0.5]), [-0.5, 0.5]));
    }

    #[test]
    fn kaleidoscope_folds_preserve_radius_and_sector_symmetry() {
        let kaleidoscope = Transformation {
            kind: TransformationKind::Kaleidoscope,
            a: 6.0,
            b: 0.0,
        };
        for angle_degrees in 0..360 {
            let theta = f64::from(angle_degrees).to_radians();
            let p = [0.7 * theta.cos(), 0.7 * theta.sin()];
            let q = kaleidoscope.apply(p);
            // Radius is preserved and the folded angle lands in the first
            // half-sector [0, π/6].
            assert!((q[0].hypot(q[1]) - 0.7).abs() < 1e-12);
            let folded = q[1].atan2(q[0]);
            assert!(
                (-1e-12..=std::f64::consts::PI / 6.0 + 1e-12).contains(&folded),
                "angle {angle_degrees}: folded to {folded}"
            );
        }
        // Points a full sector apart fold onto the same point.
        let sector = std::f64::consts::PI / 3.0;
        let a = kaleidoscope.apply([0.4 * 0.2f64.cos(), 0.4 * 0.2f64.sin()]);
        let b = kaleidoscope.apply([
            0.4 * (0.2 + sector).cos(),
            0.4 * (0.2 + sector).sin(),
        ]);
        assert!(close(a, b));
    }

    #[test]
    fn twist_and_inversion_have_their_defining_invariants() {
        let twist = Transformation {
            kind: TransformationKind::Twist,
            a: 1.5,
            b: 0.0,
        };
        // The centre is fixed and radius is preserved.
        assert!(close(twist.apply([0.0, 0.0]), [0.0, 0.0]));
        let q = twist.apply([0.6, 0.0]);
        assert!((q[0].hypot(q[1]) - 0.6).abs() < 1e-12);
        assert!((q[1].atan2(q[0]) - 1.5 * 0.6).abs() < 1e-12);
        // Inversion through radius r maps the circle r to itself and is an
        // involution.
        let inversion = Transformation {
            kind: TransformationKind::Inversion,
            a: 0.8,
            b: 0.0,
        };
        let on_circle = [0.8 * 0.3f64.cos(), 0.8 * 0.3f64.sin()];
        assert!(close(inversion.apply(on_circle), on_circle));
        let p = [0.2, 0.5];
        assert!(close(inversion.apply(inversion.apply(p)), p));
    }

    #[test]
    fn chains_apply_in_order_and_validate() {
        let chain = [
            TransformationKind::Rotation.default_transformation(),
            TransformationKind::Kaleidoscope.default_transformation(),
        ];
        validate_chain(&chain).unwrap();
        // Rotation by 0 is the identity, so the chain equals the
        // kaleidoscope alone.
        let p = [0.31, -0.44];
        assert!(close(apply_chain(&chain, p), chain[1].apply(p)));
        let (words, count) = shader_words(&chain);
        assert_eq!(count, 2);
        assert_eq!(words[0][0], 1.0);
        assert_eq!(words[1][0], 3.0);
        assert_eq!(words[1][1], 6.0);
        assert_eq!(words[2], [0.0; 4]);
        // Validation rejects out-of-range parameters and oversized chains.
        let bad = Transformation {
            kind: TransformationKind::Kaleidoscope,
            a: 1.0,
            b: 0.0,
        };
        assert!(bad.validate(0).is_err());
        let long = [TransformationKind::Rotation.default_transformation(); 5];
        assert!(validate_chain(&long).is_err());
    }
}
