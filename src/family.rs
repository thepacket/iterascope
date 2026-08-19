//! Fractal family catalogue.
//!
//! Every instrument IteraScope can display is described here: its document
//! identity, the shader code that selects its WGSL branch, its default views
//! and parameters, and a CPU `f64` reference implementation of the same
//! dynamics. The CPU implementation is the definition of record: the shader
//! branches are written to agree with it, and the pointwise diagnostics in the
//! control panel are computed from it.

use crate::MAX_ITERATIONS;

/// Escape radius used for the exponential family: `Re(z) > EXP_ESCAPE`.
pub(crate) const EXP_ESCAPE: f64 = 50.0;
/// Escape test for the trigonometric families: `|Im(z)| > TRIG_ESCAPE`.
pub(crate) const TRIG_ESCAPE: f64 = 50.0;
/// Collatz escape tests: `|Im(z)| > COLLATZ_IMAG_ESCAPE` or `|z| > COLLATZ_RADIUS_ESCAPE`.
pub(crate) const COLLATZ_IMAG_ESCAPE: f64 = 20.0;
pub(crate) const COLLATZ_RADIUS_ESCAPE: f64 = 1e6;
/// Magnet maps converge to the superattracting fixed point `z = 1`.
pub(crate) const MAGNET_CONVERGENCE: f64 = 1e-4;
/// Nova iteration stops once a Newton step is smaller than this.
pub(crate) const NOVA_CONVERGENCE: f64 = 1e-5;
/// Nova orbits pass through large values legitimately; only a very large
/// modulus counts as escape.
pub(crate) const NOVA_ESCAPE: f64 = 1e6;
/// Mandelbox orbits escape beyond `MANDELBOX_BAILOUT_FACTOR × bailout`.
pub(crate) const MANDELBOX_BAILOUT_FACTOR: f64 = 4.0;
/// Maximum length of a Lyapunov forcing sequence (it is packed into a `u32`).
pub(crate) const MAX_LYAPUNOV_SEQUENCE: usize = 32;
pub(crate) const MIN_DEGREE: u32 = 2;
pub(crate) const MAX_DEGREE: u32 = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum FractalFamily {
    #[default]
    Quadratic,
    NewtonCubic,
    Multibrot,
    Tricorn,
    PerpendicularMandelbrot,
    BurningShip,
    PerpendicularBurningShip,
    Celtic,
    PerpendicularCeltic,
    Buffalo,
    PerpendicularBuffalo,
    Lambda,
    Phoenix,
    Manowar,
    Spider,
    MagnetOne,
    MagnetTwo,
    Exponential,
    Sine,
    Cosine,
    Collatz,
    Lyapunov,
    Nova,
    BarnsleyOne,
    BarnsleyTwo,
    Mandelbox,
}

/// How the two panes of an instrument relate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Linkage {
    /// Left pane is a parameter plane; clicking selects the parameter shown
    /// in the right-hand dynamical plane.
    ParameterDynamical,
    /// Both panes show the same plane; the left pane is an overview and the
    /// right pane a linked detail region around the selected point.
    OverviewDetail,
}

/// Family-specific numerical settings. All fields are always present so the
/// experiment document and the uniform buffer stay fixed-shape; a family
/// simply ignores the ones it does not use.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FamilyParameters {
    /// Multibrot exponent `d` and Nova polynomial degree `p`.
    pub(crate) degree: u32,
    /// Nova relaxation `R`.
    pub(crate) nova_relaxation: f64,
    /// Lyapunov forcing sequence over the alphabet `{A, B}`.
    pub(crate) lyapunov_sequence: String,
    pub(crate) mandelbox_scale: f64,
    pub(crate) mandelbox_min_radius: f64,
    pub(crate) mandelbox_fixed_radius: f64,
}

impl Default for FamilyParameters {
    fn default() -> Self {
        Self {
            degree: 3,
            nova_relaxation: 1.0,
            lyapunov_sequence: "AB".to_owned(),
            mandelbox_scale: -1.5,
            mandelbox_min_radius: 0.5,
            mandelbox_fixed_radius: 1.0,
        }
    }
}

impl FamilyParameters {
    /// Packs the Lyapunov sequence into a bit mask (bit `n` set means `B` at
    /// position `n`) and its length. Invalid characters are treated as `A`.
    pub(crate) fn lyapunov_bits(&self) -> (u32, u32) {
        let mut bits = 0u32;
        let mut length = 0u32;
        for (index, character) in self
            .lyapunov_sequence
            .chars()
            .take(MAX_LYAPUNOV_SEQUENCE)
            .enumerate()
        {
            if character.eq_ignore_ascii_case(&'b') {
                bits |= 1 << index;
            }
            length = index as u32 + 1;
        }
        (bits, length.max(1))
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        if !(MIN_DEGREE..=MAX_DEGREE).contains(&self.degree) {
            return Err(format!(
                "degree must be between {MIN_DEGREE} and {MAX_DEGREE}"
            ));
        }
        if !self.nova_relaxation.is_finite() || !(0.1..=4.0).contains(&self.nova_relaxation) {
            return Err("nova_relaxation must be finite and between 0.1 and 4".to_owned());
        }
        validate_lyapunov_sequence(&self.lyapunov_sequence)?;
        if !self.mandelbox_scale.is_finite() || !(-4.0..=4.0).contains(&self.mandelbox_scale) {
            return Err("mandelbox_scale must be finite and between -4 and 4".to_owned());
        }
        if !self.mandelbox_min_radius.is_finite()
            || !(0.01..=2.0).contains(&self.mandelbox_min_radius)
        {
            return Err("mandelbox_min_radius must be finite and between 0.01 and 2".to_owned());
        }
        if !self.mandelbox_fixed_radius.is_finite()
            || !(0.1..=4.0).contains(&self.mandelbox_fixed_radius)
        {
            return Err("mandelbox_fixed_radius must be finite and between 0.1 and 4".to_owned());
        }
        Ok(())
    }

    /// Uniform words consumed by the shader (two `vec4<f32>`).
    pub(crate) fn uniform_words(&self, dynamical: bool) -> [f32; 8] {
        let (bits, length) = self.lyapunov_bits();
        [
            self.degree as f32,
            self.nova_relaxation as f32,
            self.mandelbox_scale as f32,
            self.mandelbox_min_radius as f32,
            self.mandelbox_fixed_radius as f32,
            f32::from_bits(bits),
            length as f32,
            dynamical as u8 as f32,
        ]
    }
}

pub(crate) fn validate_lyapunov_sequence(sequence: &str) -> Result<(), String> {
    if sequence.is_empty() || sequence.chars().count() > MAX_LYAPUNOV_SEQUENCE {
        return Err(format!(
            "lyapunov_sequence must contain between 1 and {MAX_LYAPUNOV_SEQUENCE} symbols"
        ));
    }
    if !sequence
        .chars()
        .all(|character| matches!(character, 'A' | 'B' | 'a' | 'b'))
    {
        return Err("lyapunov_sequence may only contain the symbols A and B".to_owned());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PlaneDefault {
    pub(crate) centre: [f64; 2],
    pub(crate) half_height: f64,
}

const fn plane(centre: [f64; 2], half_height: f64) -> PlaneDefault {
    PlaneDefault {
        centre,
        half_height,
    }
}

pub(crate) struct Preset {
    pub(crate) label: &'static str,
    pub(crate) c: [f64; 2],
}

const fn preset(label: &'static str, c: [f64; 2]) -> Preset {
    Preset { label, c }
}

const QUADRATIC_PRESETS: &[Preset] = &[
    preset("Seahorse", [-0.745, 0.113]),
    preset("Dendrite", [0.0, 1.0]),
    preset("Rabbit", [-0.123, 0.745]),
    preset("Basilica", [-1.0, 0.0]),
];
const MULTIBROT_PRESETS: &[Preset] = &[
    preset("Spirals", [0.377, 0.676]),
    preset("Dendrite", [0.24, 1.077]),
];
const PHOENIX_PRESETS: &[Preset] = &[
    preset("Fan", [-0.26, 0.977]),
    preset("Ushiki", [0.566_67, -0.5]),
];
const LAMBDA_PRESETS: &[Preset] = &[
    preset("Spiral", [2.872, 0.622]),
    preset("Seahorse", [2.995, -0.113]),
    preset("Rabbit", [2.553, -0.959]),
    preset("Period 3", [3.83, 0.0]),
];
const EXPONENTIAL_PRESETS: &[Preset] =
    &[preset("Bouquet", [0.3, 0.0]), preset("Omega", [-1.0, 0.0])];
const NOVA_PRESETS: &[Preset] = &[preset("Shift", [0.2, 0.35]), preset("Newton", [0.0, 0.0])];
const BARNSLEY_PRESETS: &[Preset] = &[
    preset("Fractint", [0.6, 1.1]),
    preset("Fronds", [0.923, 0.797]),
];

impl FractalFamily {
    pub(crate) const ALL: [Self; 26] = [
        Self::Quadratic,
        Self::NewtonCubic,
        Self::Multibrot,
        Self::Tricorn,
        Self::PerpendicularMandelbrot,
        Self::BurningShip,
        Self::PerpendicularBurningShip,
        Self::Celtic,
        Self::PerpendicularCeltic,
        Self::Buffalo,
        Self::PerpendicularBuffalo,
        Self::Lambda,
        Self::Phoenix,
        Self::Manowar,
        Self::Spider,
        Self::MagnetOne,
        Self::MagnetTwo,
        Self::Exponential,
        Self::Sine,
        Self::Cosine,
        Self::Collatz,
        Self::Lyapunov,
        Self::Nova,
        Self::BarnsleyOne,
        Self::BarnsleyTwo,
        Self::Mandelbox,
    ];

    /// Stable identifier written to experiment documents.
    pub(crate) const fn document_id(self) -> &'static str {
        match self {
            Self::Quadratic => "quadratic",
            Self::NewtonCubic => "newton-cubic",
            Self::Multibrot => "multibrot",
            Self::Tricorn => "tricorn",
            Self::PerpendicularMandelbrot => "perpendicular-mandelbrot",
            Self::BurningShip => "burning-ship",
            Self::PerpendicularBurningShip => "perpendicular-burning-ship",
            Self::Celtic => "celtic",
            Self::PerpendicularCeltic => "perpendicular-celtic",
            Self::Buffalo => "buffalo",
            Self::PerpendicularBuffalo => "perpendicular-buffalo",
            Self::Lambda => "lambda",
            Self::Phoenix => "phoenix",
            Self::Manowar => "manowar",
            Self::Spider => "spider",
            Self::MagnetOne => "magnet-1",
            Self::MagnetTwo => "magnet-2",
            Self::Exponential => "exponential",
            Self::Sine => "sine",
            Self::Cosine => "cosine",
            Self::Collatz => "collatz",
            Self::Lyapunov => "lyapunov",
            Self::Nova => "nova",
            Self::BarnsleyOne => "barnsley-1",
            Self::BarnsleyTwo => "barnsley-2",
            Self::Mandelbox => "mandelbox",
        }
    }

    pub(crate) fn from_document_id(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|family| family.document_id() == value)
    }

    /// Value of `dynamics_lo.z` selecting the shader branch.
    pub(crate) fn shader_flag(self) -> u32 {
        Self::ALL
            .iter()
            .position(|family| *family == self)
            .expect("every family is listed in ALL") as u32
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Quadratic => "Quadratic",
            Self::NewtonCubic => "Newton cubic",
            Self::Multibrot => "Multibrot",
            Self::Tricorn => "Tricorn",
            Self::PerpendicularMandelbrot => "Perpendicular Mandelbrot",
            Self::BurningShip => "Burning Ship",
            Self::PerpendicularBurningShip => "Perpendicular Burning Ship",
            Self::Celtic => "Celtic",
            Self::PerpendicularCeltic => "Perpendicular Celtic",
            Self::Buffalo => "Buffalo",
            Self::PerpendicularBuffalo => "Perpendicular Buffalo",
            Self::Lambda => "Lambda (logistic)",
            Self::Phoenix => "Phoenix",
            Self::Manowar => "Manowar",
            Self::Spider => "Spider",
            Self::MagnetOne => "Magnet I",
            Self::MagnetTwo => "Magnet II",
            Self::Exponential => "Exponential",
            Self::Sine => "Sine",
            Self::Cosine => "Cosine",
            Self::Collatz => "Collatz",
            Self::Lyapunov => "Lyapunov (Markus)",
            Self::Nova => "Nova",
            Self::BarnsleyOne => "Barnsley 1",
            Self::BarnsleyTwo => "Barnsley 2",
            Self::Mandelbox => "Mandelbox (2D)",
        }
    }

    pub(crate) const fn group(self) -> &'static str {
        match self {
            Self::Quadratic | Self::Multibrot | Self::Lambda => "Polynomial",
            Self::NewtonCubic | Self::Nova => "Root finding",
            Self::Tricorn
            | Self::PerpendicularMandelbrot
            | Self::BurningShip
            | Self::PerpendicularBurningShip
            | Self::Celtic
            | Self::PerpendicularCeltic
            | Self::Buffalo
            | Self::PerpendicularBuffalo => "Absolute-value variants",
            Self::Phoenix | Self::Manowar | Self::Spider => "Memory and drifting maps",
            Self::MagnetOne | Self::MagnetTwo => "Rational (magnet)",
            Self::Exponential | Self::Sine | Self::Cosine | Self::Collatz => "Transcendental",
            Self::Lyapunov => "Real forced maps",
            Self::BarnsleyOne | Self::BarnsleyTwo => "Piecewise",
            Self::Mandelbox => "Folding",
        }
    }

    pub(crate) const fn formula(self) -> &'static str {
        match self {
            Self::Quadratic => "f₍c₎(z) = z² + c",
            Self::NewtonCubic => "N(z) = z - (z³ - 1) / (3z²)",
            Self::Multibrot => "z ← zᵈ + c",
            Self::Tricorn => "z ← z̄² + c",
            Self::PerpendicularMandelbrot => "z ← (|x| - iy)² + c",
            Self::BurningShip => "z ← (|x| + i|y|)² + c",
            Self::PerpendicularBurningShip => "z ← (x - i|y|)² + c",
            Self::Celtic => "z ← |Re z²| + i Im z² + c",
            Self::PerpendicularCeltic => "z ← |x² - y²| - 2i|x|y + c",
            Self::Buffalo => "z ← |Re z²| + i|Im z²| + c",
            Self::PerpendicularBuffalo => "z ← |x² - y²| - 2ix|y| + c",
            Self::Lambda => "z ← λ z (1 - z)",
            Self::Phoenix => "zₙ₊₁ = zₙ² + Re c + (Im c) zₙ₋₁",
            Self::Manowar => "zₙ₊₁ = zₙ² + zₙ₋₁ + c",
            Self::Spider => "z ← z² + c,  c ← c/2 + z",
            Self::MagnetOne => "z ← ((z² + c - 1) / (2z + c - 2))²",
            Self::MagnetTwo => {
                "z ← ((z³ + 3(c-1)z + (c-1)(c-2)) / (3z² + 3(c-2)z + (c-1)(c-2) + 1))²"
            }
            Self::Exponential => "z ← c eᶻ",
            Self::Sine => "z ← c sin z",
            Self::Cosine => "z ← c cos z",
            Self::Collatz => "z ← ¼(2 + 7z - (2 + 5z) cos πz)",
            Self::Lyapunov => "xₙ₊₁ = rₙ xₙ (1 - xₙ),  rₙ ∈ {a, b}",
            Self::Nova => "z ← z - R (zᵖ - 1) / (p zᵖ⁻¹) + c",
            Self::BarnsleyOne => "z ← (z ∓ 1) c  by sign of Re z",
            Self::BarnsleyTwo => "z ← (z ∓ 1) c  by sign of Im(zc)",
            Self::Mandelbox => "v ← s · ballfold(boxfold(v)) + c",
        }
    }

    pub(crate) const fn description(self) -> &'static str {
        match self {
            Self::Quadratic => "Select c in the parameter plane to inspect its dynamical plane.",
            Self::NewtonCubic => "Compare root basins with a linked convergence-detail view.",
            Self::Multibrot => {
                "Degree-d generalisation of the Mandelbrot set; critical orbit starts at 0."
            }
            Self::Tricorn => {
                "Anti-holomorphic conjugate square; the Mandelbar has three-fold symmetry."
            }
            Self::PerpendicularMandelbrot => {
                "Absolute value on Re z only; mirror conventions follow Kalles Fraktaler."
            }
            Self::BurningShip => "Componentwise absolute value before squaring.",
            Self::PerpendicularBurningShip => "Absolute value on Im z only.",
            Self::Celtic => "Absolute value applied to the real part of z².",
            Self::PerpendicularCeltic => "Celtic real part with the perpendicular imaginary part.",
            Self::Buffalo => "Componentwise absolute value applied to z².",
            Self::PerpendicularBuffalo => {
                "Celtic real part with the perpendicular burning-ship imaginary part."
            }
            Self::Lambda => "Logistic map in the complex λ plane; critical orbit starts at ½.",
            Self::Phoenix => "Two-step memory map; the parameter plane is (p, q) = (Re c, Im c).",
            Self::Manowar => "Two-step memory map started from z₀ = z₋₁ = c.",
            Self::Spider => "The parameter drifts with the orbit.",
            Self::MagnetOne => "Orbits escape or converge to the superattracting point z = 1.",
            Self::MagnetTwo => "Higher-order magnet renormalisation map.",
            Self::Exponential => "Orbit of the singular value 0; escape when Re z > 50. f32 only.",
            Self::Sine => "Orbit of the critical point π/2; escape when |Im z| > 50. f32 only.",
            Self::Cosine => "Orbit of the critical point 0; escape when |Im z| > 50. f32 only.",
            Self::Collatz => {
                "Complex interpolation of the Collatz map on the same plane in both panes. f32 only."
            }
            Self::Lyapunov => "Lyapunov exponent of the forced logistic map over (a, b). f32 only.",
            Self::Nova => {
                "Relaxed Newton iteration with an additive parameter; orbits converge or escape."
            }
            Self::BarnsleyOne => "Piecewise-linear map switching on the sign of Re z.",
            Self::BarnsleyTwo => "Piecewise-linear map switching on the sign of Im(z·c).",
            Self::Mandelbox => {
                "Box fold, ball fold and scale in the plane; escape radius is 4 × bailout."
            }
        }
    }

    pub(crate) const fn linkage(self) -> Linkage {
        match self {
            Self::NewtonCubic | Self::Collatz | Self::Lyapunov => Linkage::OverviewDetail,
            _ => Linkage::ParameterDynamical,
        }
    }

    pub(crate) const fn is_quadratic(self) -> bool {
        matches!(self, Self::Quadratic)
    }

    pub(crate) const fn is_newton(self) -> bool {
        matches!(self, Self::NewtonCubic)
    }

    /// Everything except the Newton instrument is coloured by escape or
    /// convergence time (the Lyapunov plane by its exponent).
    pub(crate) const fn is_escape_time(self) -> bool {
        !matches!(self, Self::NewtonCubic)
    }

    /// Families with an exact perturbation recurrence and an
    /// arbitrary-precision reference orbit, so they can continue past the
    /// double-single handoff into deep zoom.
    pub(crate) const fn supports_deep_zoom(self) -> bool {
        !matches!(
            self,
            Self::Exponential | Self::Sine | Self::Cosine | Self::Collatz | Self::Lyapunov
        )
    }

    /// Families whose orbits are iterated in compensated double-single
    /// arithmetic when the f32 coordinate grid becomes too coarse.
    pub(crate) const fn supports_double_single(self) -> bool {
        !matches!(
            self,
            Self::Exponential | Self::Sine | Self::Cosine | Self::Collatz | Self::Lyapunov
        )
    }

    /// Families whose orbits end by converging to a root or a fixed point,
    /// so the argument of the limit separates their basins.
    pub(crate) const fn converges(self) -> bool {
        matches!(
            self,
            Self::NewtonCubic | Self::Nova | Self::MagnetOne | Self::MagnetTwo
        )
    }

    /// Families whose derivative the shader tracks, so the distance-estimate
    /// colouring is exact. Must agree with `family_has_derivative` in
    /// fractal.wgsl.
    pub(crate) const fn has_distance_estimate(self) -> bool {
        matches!(self, Self::Quadratic | Self::Multibrot | Self::Lambda)
    }

    pub(crate) const fn uses_bailout(self) -> bool {
        !matches!(
            self,
            Self::NewtonCubic
                | Self::Exponential
                | Self::Sine
                | Self::Cosine
                | Self::Collatz
                | Self::Lyapunov
        )
    }

    pub(crate) const fn uses_degree(self) -> bool {
        matches!(self, Self::Multibrot | Self::Nova)
    }

    pub(crate) const fn uses_relaxation(self) -> bool {
        matches!(self, Self::Nova)
    }

    pub(crate) const fn uses_lyapunov_sequence(self) -> bool {
        matches!(self, Self::Lyapunov)
    }

    pub(crate) const fn uses_mandelbox(self) -> bool {
        matches!(self, Self::Mandelbox)
    }

    pub(crate) const fn has_family_parameters(self) -> bool {
        self.uses_degree() || self.uses_lyapunov_sequence() || self.uses_mandelbox()
    }

    pub(crate) const fn min_iterations(self) -> u32 {
        if self.is_newton() { 8 } else { 32 }
    }

    pub(crate) const fn max_iterations(self) -> u32 {
        if self.is_newton() {
            2_048
        } else {
            MAX_ITERATIONS
        }
    }

    pub(crate) const fn default_parameter_view(self) -> PlaneDefault {
        match self {
            Self::Quadratic | Self::Manowar | Self::Spider | Self::Phoenix => {
                plane([-0.5, 0.0], 1.45)
            }
            Self::NewtonCubic => plane([0.0, 0.0], 1.65),
            Self::Multibrot => plane([0.0, 0.0], 1.45),
            Self::Tricorn | Self::PerpendicularMandelbrot => plane([-0.3, 0.0], 1.6),
            Self::BurningShip | Self::PerpendicularBurningShip => plane([-0.4, -0.4], 1.6),
            Self::Celtic
            | Self::PerpendicularCeltic
            | Self::Buffalo
            | Self::PerpendicularBuffalo => plane([-0.4, 0.0], 1.6),
            Self::Lambda => plane([1.0, 0.0], 2.4),
            Self::MagnetOne => plane([1.5, 0.0], 2.2),
            Self::MagnetTwo => plane([1.5, 0.0], 2.8),
            Self::Exponential => plane([0.5, 0.0], 3.0),
            Self::Sine => plane([1.0, 0.0], 2.2),
            Self::Cosine => plane([0.0, 0.0], 2.4),
            Self::Collatz => plane([0.0, 0.0], 2.5),
            Self::Lyapunov => plane([2.0, 2.0], 2.0),
            Self::Nova => plane([0.0, 0.0], 1.8),
            Self::BarnsleyOne | Self::BarnsleyTwo => plane([0.0, 0.0], 2.0),
            Self::Mandelbox => plane([0.0, 0.0], 3.0),
        }
    }

    pub(crate) const fn default_dynamical_view(self) -> PlaneDefault {
        match self {
            Self::NewtonCubic => plane([0.0, 0.0], 1.65),
            Self::Lambda => plane([0.5, 0.0], 0.8),
            Self::MagnetOne | Self::MagnetTwo => plane([1.0, 0.0], 3.0),
            Self::Exponential => plane([2.5, 0.0], 3.2),
            Self::Sine | Self::Cosine => plane([0.0, 0.0], 3.2),
            Self::Collatz => plane([0.0, 0.0], 2.5),
            Self::Lyapunov => plane([2.0, 2.0], 2.0),
            Self::Nova => plane([0.0, 0.0], 1.8),
            Self::BarnsleyOne | Self::BarnsleyTwo => plane([0.0, 0.0], 2.0),
            Self::Mandelbox => plane([0.0, 0.0], 3.0),
            _ => plane([0.0, 0.0], 1.45),
        }
    }

    /// Default dynamical-plane parameter, or the selected point for
    /// overview/detail instruments.
    pub(crate) const fn default_parameter(self) -> [f64; 2] {
        match self {
            Self::Quadratic => [-0.745, 0.113],
            Self::NewtonCubic => [0.5, 0.5],
            Self::Multibrot => [0.377, 0.676],
            Self::Tricorn => [0.343, 0.801],
            Self::PerpendicularMandelbrot => [-0.792, 0.138],
            Self::BurningShip => [0.394, -0.594],
            Self::PerpendicularBurningShip => [0.243, -0.538],
            Self::Celtic => [-1.194, 0.249],
            Self::PerpendicularCeltic => [-0.665, 1.022],
            Self::Buffalo => [-0.892, 0.028],
            Self::PerpendicularBuffalo => [-1.119, 0.083],
            Self::Lambda => [2.872, 0.622],
            Self::Phoenix => [-0.26, 0.977],
            Self::Manowar => [-0.1, 0.75],
            Self::Spider => [-0.534, 1.027],
            Self::MagnetOne => [0.237, 1.119],
            Self::MagnetTwo => [-0.253, 1.084],
            Self::Exponential => [0.3, 0.0],
            Self::Sine => [-1.86, 0.722],
            Self::Cosine => [3.007, 0.373],
            Self::Collatz => [1.0, 0.2],
            Self::Lyapunov => [3.2, 3.6],
            Self::Nova => [0.2, 0.35],
            Self::BarnsleyOne => [0.6, 1.1],
            Self::BarnsleyTwo => [0.923, 0.797],
            Self::Mandelbox => [2.167, 2.217],
        }
    }

    pub(crate) const fn presets(self) -> &'static [Preset] {
        match self {
            Self::Quadratic => QUADRATIC_PRESETS,
            Self::Multibrot => MULTIBROT_PRESETS,
            Self::Phoenix => PHOENIX_PRESETS,
            Self::Lambda => LAMBDA_PRESETS,
            Self::Exponential => EXPONENTIAL_PRESETS,
            Self::Nova => NOVA_PRESETS,
            Self::BarnsleyOne | Self::BarnsleyTwo => BARNSLEY_PRESETS,
            _ => &[],
        }
    }

    /// Label for the selected value in the control panel.
    pub(crate) const fn parameter_symbol(self) -> &'static str {
        match self {
            Self::Lambda => "λ",
            Self::Phoenix => "c = p + iq",
            Self::Lyapunov => "(a, b)",
            Self::NewtonCubic | Self::Collatz => "z₀",
            _ => "c",
        }
    }

    pub(crate) const fn pane_titles(self, pane: usize) -> (&'static str, &'static str) {
        match (self, pane) {
            (Self::Quadratic, 0) => ("PARAMETER PLANE", "c -> bounded critical orbit"),
            (Self::Quadratic, _) => ("DYNAMICAL PLANE", "z -> z^2 + c"),
            (Self::NewtonCubic, 0) => ("ROOT BASINS", "z^3 - 1: attracting root"),
            (Self::NewtonCubic, _) => ("CONVERGENCE DETAIL", "iterations and boundary sensitivity"),
            (Self::Collatz, 0) => ("COLLATZ PLANE", "escape time of the interpolated map"),
            (Self::Collatz, _) => ("COLLATZ DETAIL", "linked region around z0"),
            (Self::Lyapunov, 0) => ("LYAPUNOV PLANE", "(a, b): exponent of the forced map"),
            (Self::Lyapunov, _) => ("LYAPUNOV DETAIL", "linked region around (a, b)"),
            (Self::Lambda, 0) => ("PARAMETER PLANE", "lambda -> bounded critical orbit"),
            (Self::Lambda, _) => ("DYNAMICAL PLANE", "z -> lambda z (1 - z)"),
            (Self::MagnetOne | Self::MagnetTwo, 0) => {
                ("PARAMETER PLANE", "c -> escape or convergence to 1")
            }
            (Self::MagnetOne | Self::MagnetTwo, _) => {
                ("DYNAMICAL PLANE", "escape / convergence time")
            }
            (Self::Nova, 0) => ("PARAMETER PLANE", "c -> convergence of z0 = 1"),
            (Self::Nova, _) => ("DYNAMICAL PLANE", "convergence time and attracting root"),
            (_, 0) => ("PARAMETER PLANE", "c -> bounded critical orbit"),
            (_, _) => ("DYNAMICAL PLANE", "escape time of z0"),
        }
    }
}

/// Outcome of a finite CPU orbit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OrbitFate {
    Bounded,
    Escaped,
    Converged,
    NonFinite,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct EscapeDiagnostic {
    pub(crate) fate: OrbitFate,
    pub(crate) iterations: u32,
    pub(crate) z: [f64; 2],
    pub(crate) magnitude: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OrbitState {
    pub(crate) z: [f64; 2],
    pub(crate) z_prev: [f64; 2],
    pub(crate) c: [f64; 2],
}

/// Starting state for a pixel. `dynamical` selects the dynamical/detail
/// plane (z₀ = world, c = parameter) rather than the parameter plane
/// (c = world).
pub(crate) fn initial_state_with(
    family: FractalFamily,
    world: [f64; 2],
    dynamical: bool,
    parameter: [f64; 2],
) -> OrbitState {
    if dynamical {
        let z_prev = if family == FractalFamily::Manowar {
            world
        } else {
            [0.0, 0.0]
        };
        return OrbitState {
            z: world,
            z_prev,
            c: parameter,
        };
    }
    let c = world;
    let (z, z_prev) = match family {
        FractalFamily::Lambda => ([0.5, 0.0], [0.0, 0.0]),
        FractalFamily::Manowar => (c, c),
        FractalFamily::Sine => ([std::f64::consts::FRAC_PI_2, 0.0], [0.0, 0.0]),
        FractalFamily::Nova => ([1.0, 0.0], [0.0, 0.0]),
        FractalFamily::BarnsleyOne | FractalFamily::BarnsleyTwo => (c, [0.0, 0.0]),
        _ => ([0.0, 0.0], [0.0, 0.0]),
    };
    OrbitState { z, z_prev, c }
}

fn mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn div(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let denominator = b[0] * b[0] + b[1] * b[1];
    [
        (a[0] * b[0] + a[1] * b[1]) / denominator,
        (a[1] * b[0] - a[0] * b[1]) / denominator,
    ]
}

fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn scale(a: [f64; 2], s: f64) -> [f64; 2] {
    [a[0] * s, a[1] * s]
}

fn square(z: [f64; 2]) -> [f64; 2] {
    [z[0] * z[0] - z[1] * z[1], 2.0 * z[0] * z[1]]
}

fn pow(z: [f64; 2], degree: u32) -> [f64; 2] {
    let mut result = z;
    for _ in 1..degree {
        result = mul(result, z);
    }
    result
}

fn exp(z: [f64; 2]) -> [f64; 2] {
    let magnitude = z[0].exp();
    [magnitude * z[1].cos(), magnitude * z[1].sin()]
}

fn sin(z: [f64; 2]) -> [f64; 2] {
    [z[0].sin() * z[1].cosh(), z[0].cos() * z[1].sinh()]
}

fn cos(z: [f64; 2]) -> [f64; 2] {
    [z[0].cos() * z[1].cosh(), -z[0].sin() * z[1].sinh()]
}

fn box_fold(value: f64) -> f64 {
    if value > 1.0 {
        2.0 - value
    } else if value < -1.0 {
        -2.0 - value
    } else {
        value
    }
}

/// One step of the family's map. Mirrors `family_step_f32` in the shader.
pub(crate) fn step(
    family: FractalFamily,
    parameters: &FamilyParameters,
    state: OrbitState,
) -> OrbitState {
    let OrbitState { z, z_prev, c } = state;
    let (x, y) = (z[0], z[1]);
    let one = [1.0, 0.0];
    let mut next_prev = z_prev;
    let mut next_c = c;
    let next = match family {
        FractalFamily::Quadratic => add(square(z), c),
        FractalFamily::NewtonCubic => {
            let z2 = square(z);
            let polynomial = sub(mul(z2, z), one);
            sub(z, div(polynomial, scale(z2, 3.0)))
        }
        FractalFamily::Multibrot => add(pow(z, parameters.degree), c),
        FractalFamily::Tricorn => add([x * x - y * y, -2.0 * x * y], c),
        FractalFamily::PerpendicularMandelbrot => add([x * x - y * y, -2.0 * x.abs() * y], c),
        FractalFamily::BurningShip => add([x * x - y * y, 2.0 * (x * y).abs()], c),
        FractalFamily::PerpendicularBurningShip => add([x * x - y * y, -2.0 * x * y.abs()], c),
        FractalFamily::Celtic => add([(x * x - y * y).abs(), 2.0 * x * y], c),
        FractalFamily::PerpendicularCeltic => add([(x * x - y * y).abs(), -2.0 * x.abs() * y], c),
        FractalFamily::Buffalo => add([(x * x - y * y).abs(), 2.0 * (x * y).abs()], c),
        FractalFamily::PerpendicularBuffalo => add([(x * x - y * y).abs(), -2.0 * x * y.abs()], c),
        FractalFamily::Lambda => mul(c, mul(z, sub(one, z))),
        FractalFamily::Phoenix => {
            next_prev = z;
            add(add(square(z), [c[0], 0.0]), scale(z_prev, c[1]))
        }
        FractalFamily::Manowar => {
            next_prev = z;
            add(add(square(z), z_prev), c)
        }
        FractalFamily::Spider => {
            let next = add(square(z), c);
            next_c = add(scale(c, 0.5), next);
            next
        }
        FractalFamily::MagnetOne => {
            let numerator = sub(add(square(z), c), one);
            let denominator = sub(add(scale(z, 2.0), c), [2.0, 0.0]);
            square(div(numerator, denominator))
        }
        FractalFamily::MagnetTwo => {
            let c1 = sub(c, one);
            let c2 = sub(c, [2.0, 0.0]);
            let c12 = mul(c1, c2);
            let z2 = square(z);
            let z3 = mul(z2, z);
            let numerator = add(add(z3, scale(mul(c1, z), 3.0)), c12);
            let denominator = add(add(add(scale(z2, 3.0), scale(mul(c2, z), 3.0)), c12), one);
            square(div(numerator, denominator))
        }
        FractalFamily::Exponential => mul(c, exp(z)),
        FractalFamily::Sine => mul(c, sin(z)),
        FractalFamily::Cosine => mul(c, cos(z)),
        FractalFamily::Collatz => {
            let cosine = cos(scale(z, std::f64::consts::PI));
            let term = mul(add([2.0, 0.0], scale(z, 5.0)), cosine);
            scale(sub(add([2.0, 0.0], scale(z, 7.0)), term), 0.25)
        }
        FractalFamily::Lyapunov => z,
        FractalFamily::Nova => {
            // (z^p - 1) / (p z^(p-1)) = z / p - z^(1-p) / p, written with an
            // inverse power so large |z| cannot overflow the numerator.
            let p = parameters.degree;
            let inverse = div(one, pow(z, p - 1));
            let newton_step = scale(sub(z, inverse), 1.0 / p as f64);
            add(sub(z, scale(newton_step, parameters.nova_relaxation)), c)
        }
        FractalFamily::BarnsleyOne => {
            if x >= 0.0 {
                mul(sub(z, one), c)
            } else {
                mul(add(z, one), c)
            }
        }
        FractalFamily::BarnsleyTwo => {
            if x * c[1] + c[0] * y >= 0.0 {
                mul(sub(z, one), c)
            } else {
                mul(add(z, one), c)
            }
        }
        FractalFamily::Mandelbox => {
            let folded = [box_fold(x), box_fold(y)];
            let radius_squared = folded[0] * folded[0] + folded[1] * folded[1];
            let min_squared = parameters.mandelbox_min_radius * parameters.mandelbox_min_radius;
            let fixed_squared =
                parameters.mandelbox_fixed_radius * parameters.mandelbox_fixed_radius;
            let ball = if radius_squared < min_squared {
                scale(folded, fixed_squared / min_squared)
            } else if radius_squared < fixed_squared {
                scale(folded, fixed_squared / radius_squared)
            } else {
                folded
            };
            add(scale(ball, parameters.mandelbox_scale), c)
        }
    };
    OrbitState {
        z: next,
        z_prev: next_prev,
        c: next_c,
    }
}

/// Iterates a pixel's orbit in `f64` and reports how it ends. Mirrors the
/// generic shader loop, including its escape and convergence tests.
pub(crate) fn diagnose(
    family: FractalFamily,
    parameters: &FamilyParameters,
    state: OrbitState,
    iterations: u32,
    bailout: f64,
) -> EscapeDiagnostic {
    let mut state = state;
    let bailout_squared = match family {
        FractalFamily::Mandelbox => {
            let radius = bailout * MANDELBOX_BAILOUT_FACTOR;
            radius * radius
        }
        FractalFamily::Nova => NOVA_ESCAPE * NOVA_ESCAPE,
        _ => bailout * bailout,
    };
    for iteration in 1..=iterations {
        let previous = state.z;
        state = step(family, parameters, state);
        let z = state.z;
        let magnitude_squared = z[0] * z[0] + z[1] * z[1];
        if !magnitude_squared.is_finite() {
            return EscapeDiagnostic {
                fate: OrbitFate::NonFinite,
                iterations: iteration,
                z,
                magnitude: f64::INFINITY,
            };
        }
        let magnitude = magnitude_squared.sqrt();
        let escaped = match family {
            FractalFamily::Exponential => z[0] > EXP_ESCAPE,
            FractalFamily::Sine | FractalFamily::Cosine => z[1].abs() > TRIG_ESCAPE,
            FractalFamily::Collatz => {
                z[1].abs() > COLLATZ_IMAG_ESCAPE || magnitude > COLLATZ_RADIUS_ESCAPE
            }
            _ => magnitude_squared > bailout_squared,
        };
        if escaped {
            return EscapeDiagnostic {
                fate: OrbitFate::Escaped,
                iterations: iteration,
                z,
                magnitude,
            };
        }
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
        if converged {
            return EscapeDiagnostic {
                fate: OrbitFate::Converged,
                iterations: iteration,
                z,
                magnitude,
            };
        }
    }
    let z = state.z;
    EscapeDiagnostic {
        fate: OrbitFate::Bounded,
        iterations,
        z,
        magnitude: z[0].hypot(z[1]),
    }
}

/// Outcome of an `f64` reference orbit: the visited points and, if the orbit
/// escaped or converged, the iteration at which it did.
pub(crate) struct F64ReferenceOrbit {
    pub(crate) points: Vec<[f64; 2]>,
    /// Iteration at which the reference escaped or converged; the GPU learns
    /// this from the point count, so it is only inspected by tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) escape_iteration: Option<u32>,
}

/// Iterates a reference orbit in `f64`, applying the same termination rules
/// as `diagnose`. Used to drive GPU perturbation below the arbitrary-precision
/// handoff, where `f64` resolves every representable view.
pub(crate) fn reference_orbit_f64(
    family: FractalFamily,
    parameters: &FamilyParameters,
    state: OrbitState,
    iterations: u32,
    bailout: f64,
) -> F64ReferenceOrbit {
    let mut state = state;
    let mut points = Vec::with_capacity(iterations as usize + 1);
    points.push(state.z);
    let bailout_squared = match family {
        FractalFamily::Mandelbox => {
            let radius = bailout * MANDELBOX_BAILOUT_FACTOR;
            radius * radius
        }
        FractalFamily::Nova => NOVA_ESCAPE * NOVA_ESCAPE,
        FractalFamily::NewtonCubic => 1e36,
        _ => bailout * bailout,
    };
    for iteration in 1..=iterations {
        let previous = state.z;
        state = step(family, parameters, state);
        let z = state.z;
        points.push(z);
        let magnitude_squared = z[0] * z[0] + z[1] * z[1];
        let escaped = match family {
            FractalFamily::Exponential => z[0] > EXP_ESCAPE,
            FractalFamily::Sine | FractalFamily::Cosine => z[1].abs() > TRIG_ESCAPE,
            FractalFamily::Collatz => {
                z[1].abs() > COLLATZ_IMAG_ESCAPE || magnitude_squared.sqrt() > COLLATZ_RADIUS_ESCAPE
            }
            _ => magnitude_squared > bailout_squared,
        };
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
        if !magnitude_squared.is_finite() || escaped || converged {
            return F64ReferenceOrbit {
                points,
                escape_iteration: Some(iteration),
            };
        }
    }
    F64ReferenceOrbit {
        points,
        escape_iteration: None,
    }
}

/// Lyapunov exponent of the forced logistic map at `(a, b)`, discarding the
/// first quarter of the iterations as a transient. Mirrors the shader.
pub(crate) fn lyapunov_exponent(
    parameters: &FamilyParameters,
    a: f64,
    b: f64,
    iterations: u32,
) -> f64 {
    let (bits, length) = parameters.lyapunov_bits();
    let warmup = iterations / 4;
    let mut x = 0.5_f64;
    let mut sum = 0.0;
    let mut count = 0u32;
    for iteration in 0..iterations {
        let use_b = (bits >> (iteration % length)) & 1 == 1;
        let r = if use_b { b } else { a };
        x = r * x * (1.0 - x);
        if !x.is_finite() || !(0.0..=1.0).contains(&x) {
            return f64::INFINITY;
        }
        if iteration >= warmup {
            sum += (r * (1.0 - 2.0 * x)).abs().max(1e-300).ln();
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    sum / count as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orbit(family: FractalFamily, world: [f64; 2], dynamical: bool) -> EscapeDiagnostic {
        let parameters = FamilyParameters::default();
        diagnose(
            family,
            &parameters,
            initial_state_with(family, world, dynamical, family.default_parameter()),
            512,
            4.0,
        )
    }

    #[test]
    fn document_ids_are_unique_and_round_trip() {
        for (index, family) in FractalFamily::ALL.into_iter().enumerate() {
            assert_eq!(family.shader_flag(), index as u32);
            assert_eq!(
                FractalFamily::from_document_id(family.document_id()),
                Some(family)
            );
            for other in FractalFamily::ALL {
                if other != family {
                    assert_ne!(other.document_id(), family.document_id());
                }
            }
        }
        assert_eq!(FractalFamily::Quadratic.shader_flag(), 0);
        assert_eq!(FractalFamily::NewtonCubic.shader_flag(), 1);
    }

    #[test]
    fn the_quadratic_step_is_the_mandelbrot_recurrence() {
        let parameters = FamilyParameters::default();
        let state = OrbitState {
            z: [0.5, -0.25],
            z_prev: [0.0; 2],
            c: [-0.1, 0.65],
        };
        let next = step(FractalFamily::Quadratic, &parameters, state);
        assert!((next.z[0] - (0.25 - 0.0625 - 0.1)).abs() < 1e-15);
        assert!((next.z[1] - (2.0 * 0.5 * -0.25 + 0.65)).abs() < 1e-15);
    }

    #[test]
    fn multibrot_degree_two_matches_the_quadratic_family() {
        let parameters = FamilyParameters {
            degree: 2,
            ..FamilyParameters::default()
        };
        let state = OrbitState {
            z: [0.3, 0.7],
            z_prev: [0.0; 2],
            c: [-0.4, 0.2],
        };
        let quadratic = step(FractalFamily::Quadratic, &parameters, state);
        let multibrot = step(FractalFamily::Multibrot, &parameters, state);
        assert_eq!(quadratic.z, multibrot.z);
    }

    #[test]
    fn burning_ship_and_tricorn_agree_with_mandelbrot_in_the_first_quadrant() {
        let parameters = FamilyParameters::default();
        let state = OrbitState {
            z: [0.4, 0.3],
            z_prev: [0.0; 2],
            c: [0.1, 0.1],
        };
        let mandelbrot = step(FractalFamily::Quadratic, &parameters, state);
        let ship = step(FractalFamily::BurningShip, &parameters, state);
        assert_eq!(mandelbrot.z, ship.z);
        let tricorn = step(FractalFamily::Tricorn, &parameters, state);
        assert_eq!(tricorn.z[0], mandelbrot.z[0]);
        assert!((tricorn.z[1] - (-2.0 * 0.4 * 0.3 + 0.1)).abs() < 1e-15);
    }

    #[test]
    fn collatz_map_reproduces_integer_collatz_steps() {
        let parameters = FamilyParameters::default();
        for (input, expected) in [(6.0, 3.0), (7.0, 22.0), (1.0, 4.0), (16.0, 8.0)] {
            let next = step(
                FractalFamily::Collatz,
                &parameters,
                OrbitState {
                    z: [input, 0.0],
                    z_prev: [0.0; 2],
                    c: [0.0; 2],
                },
            );
            assert!(
                (next.z[0] - expected).abs() < 1e-9,
                "{input} -> {:?}",
                next.z
            );
            assert!(next.z[1].abs() < 1e-9);
        }
    }

    #[test]
    fn magnet_one_converges_to_one_inside_its_basin() {
        // c = 3/2: g(0) = (c - 1) / (c - 2) = -1, so z₁ = 1 exactly.
        let result = orbit(FractalFamily::MagnetOne, [1.5, 0.0], false);
        assert_eq!(result.fate, OrbitFate::Converged);
        assert_eq!(result.iterations, 1);
        assert!((result.z[0] - 1.0).abs() < MAGNET_CONVERGENCE);
        // c = 2 puts the pole of the map at z₀ = 0; the orbit leaves through
        // infinity and is reported as non-finite rather than bounded.
        let pole = orbit(FractalFamily::MagnetOne, [2.0, 0.0], false);
        assert!(matches!(
            pole.fate,
            OrbitFate::NonFinite | OrbitFate::Escaped
        ));
    }

    #[test]
    fn nova_converges_to_a_cube_root_of_unity_without_shift() {
        let parameters = FamilyParameters::default();
        let result = diagnose(
            FractalFamily::Nova,
            &parameters,
            OrbitState {
                z: [0.4, 0.6],
                z_prev: [0.0; 2],
                c: [0.0, 0.0],
            },
            256,
            4.0,
        );
        assert_eq!(result.fate, OrbitFate::Converged);
        assert!((result.magnitude - 1.0).abs() < 1e-3);
    }

    #[test]
    fn lyapunov_exponent_is_negative_for_the_period_one_logistic_map() {
        let parameters = FamilyParameters::default();
        // a = b = 2.5: fixed point 0.6 with multiplier |r(1 - 2x)| = 0.5.
        let exponent = lyapunov_exponent(&parameters, 2.5, 2.5, 2_000);
        assert!((exponent - 0.5_f64.ln()).abs() < 1e-6, "{exponent}");
        // Just below r = 4 the map is chaotic with a positive exponent. (At
        // exactly r = 4 the critical point x₀ = ½ lands on the repelling
        // fixed point 0, which is why the seed is not a measurement of ln 2.)
        let chaotic = lyapunov_exponent(&parameters, 3.99, 3.99, 20_000);
        assert!(chaotic > 0.4 && chaotic < 0.75, "{chaotic}");
        // Out of range parameters diverge.
        assert!(lyapunov_exponent(&parameters, 4.5, 4.5, 100).is_infinite());
    }

    #[test]
    fn lyapunov_sequence_packs_into_bits() {
        let parameters = FamilyParameters {
            lyapunov_sequence: "BBAAB".to_owned(),
            ..FamilyParameters::default()
        };
        assert_eq!(parameters.lyapunov_bits(), (0b10011, 5));
        assert!(validate_lyapunov_sequence("").is_err());
        assert!(validate_lyapunov_sequence("ABC").is_err());
        assert!(validate_lyapunov_sequence(&"A".repeat(33)).is_err());
        assert!(validate_lyapunov_sequence("BBBBBBAAAAAA").is_ok());
    }

    #[test]
    fn default_views_contain_both_bounded_and_escaping_points() {
        // A coarse CPU raster of every escape-time family's default views
        // guards against degenerate defaults (an all-black or all-escaped
        // instrument) and exercises each step function.
        let parameters = FamilyParameters::default();
        for family in FractalFamily::ALL {
            if !family.is_escape_time() || family == FractalFamily::Lyapunov {
                continue;
            }
            for (dynamical, view) in [
                (false, family.default_parameter_view()),
                (true, family.default_dynamical_view()),
            ] {
                if family.linkage() == Linkage::OverviewDetail && !dynamical {
                    continue;
                }
                let mut bounded = 0;
                let mut finished = 0;
                let samples = 48;
                for j in 0..samples {
                    for i in 0..samples {
                        let world = [
                            view.centre[0]
                                + (i as f64 / (samples - 1) as f64 * 2.0 - 1.0)
                                    * view.half_height
                                    * 1.4,
                            view.centre[1]
                                + (j as f64 / (samples - 1) as f64 * 2.0 - 1.0) * view.half_height,
                        ];
                        let dynamical_plane =
                            dynamical || family.linkage() == Linkage::OverviewDetail;
                        let result = diagnose(
                            family,
                            &parameters,
                            initial_state_with(
                                family,
                                world,
                                dynamical_plane,
                                family.default_parameter(),
                            ),
                            256,
                            4.0,
                        );
                        match result.fate {
                            OrbitFate::Bounded => bounded += 1,
                            OrbitFate::Escaped | OrbitFate::Converged => finished += 1,
                            OrbitFate::NonFinite => {}
                        }
                    }
                }
                let total = samples * samples;
                let bounded_fraction = bounded as f64 / total as f64;
                let finished_fraction = finished as f64 / total as f64;
                assert!(
                    finished_fraction > 0.02,
                    "{family:?} dynamical={dynamical}: nothing escapes or converges"
                );
                // Convergence-coloured families and the area-preserving
                // Manowar map legitimately have almost no bounded pixels.
                let may_be_unbounded = matches!(
                    family,
                    FractalFamily::Nova
                        | FractalFamily::MagnetOne
                        | FractalFamily::MagnetTwo
                        | FractalFamily::Manowar
                );
                assert!(
                    bounded_fraction > 0.01 || may_be_unbounded,
                    "{family:?} dynamical={dynamical}: nothing is bounded"
                );
            }
        }
    }

    #[test]
    fn lyapunov_default_view_has_stable_and_chaotic_regions() {
        let parameters = FamilyParameters::default();
        let view = FractalFamily::Lyapunov.default_parameter_view();
        let mut stable = 0;
        let mut chaotic = 0;
        for j in 0..32 {
            for i in 0..32 {
                let a = view.centre[0] + (i as f64 / 31.0 * 2.0 - 1.0) * view.half_height * 0.98;
                let b = view.centre[1] + (j as f64 / 31.0 * 2.0 - 1.0) * view.half_height * 0.98;
                let exponent = lyapunov_exponent(&parameters, a, b, 400);
                if exponent < 0.0 {
                    stable += 1;
                } else if exponent.is_finite() {
                    chaotic += 1;
                }
            }
        }
        assert!(stable > 50, "{stable}");
        assert!(chaotic > 20, "{chaotic}");
    }

    #[test]
    fn f64_reference_orbit_agrees_with_diagnose() {
        let parameters = FamilyParameters::default();
        for family in FractalFamily::ALL {
            if !family.supports_deep_zoom() {
                continue;
            }
            for (world, dynamical) in [([0.31, -0.47], true), ([-0.2, 0.55], false)] {
                if !dynamical && family.linkage() == Linkage::OverviewDetail {
                    continue;
                }
                let state =
                    initial_state_with(family, world, dynamical, family.default_parameter());
                let orbit = reference_orbit_f64(family, &parameters, state, 300, 4.0);
                let result = diagnose(family, &parameters, state, 300, 4.0);
                let expected = match result.fate {
                    OrbitFate::Bounded => None,
                    _ => Some(result.iterations),
                };
                assert_eq!(
                    orbit.escape_iteration, expected,
                    "{family:?} dynamical={dynamical}"
                );
                assert_eq!(
                    orbit.points.len() as u32,
                    expected.unwrap_or(300) + 1,
                    "{family:?} point count"
                );
                assert_eq!(*orbit.points.last().unwrap(), result.z);
            }
        }
    }

    #[test]
    fn parameter_validation_rejects_out_of_range_values() {
        let mut parameters = FamilyParameters::default();
        assert!(parameters.validate().is_ok());
        parameters.degree = 9;
        assert!(parameters.validate().is_err());
        parameters.degree = 3;
        parameters.mandelbox_scale = 9.0;
        assert!(parameters.validate().is_err());
    }
}
