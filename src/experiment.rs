//! Versioned, deterministic experiment documents.

use serde::{Deserialize, Serialize};

use crate::arbitrary::MAX_DECIMAL_ZOOM_EXPONENT;
use crate::colouring::{Colouring, Layer, MAX_LAYERS};
use crate::family::{FamilyParameters, FractalFamily, Linkage};

pub(crate) const FORMAT_ID: &str = "iterascope-experiment";
pub(crate) const FORMAT_VERSION: u32 = 7;
/// Largest escape radius a document may ask for. Large radii give the
/// triangle-inequality and stripe colourings their smoothest results.
pub(crate) const MAX_BAILOUT: f32 = 1e10;
#[cfg(test)]
const QUADRATIC_FAMILY_ID: &str = FractalFamily::Quadratic.document_id();
#[cfg(test)]
const NEWTON_FAMILY_ID: &str = FractalFamily::NewtonCubic.document_id();

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExperimentDocument {
    pub(crate) format: String,
    pub(crate) version: u32,
    pub(crate) family: String,
    pub(crate) parameter_plane: PlaneDocument,
    pub(crate) dynamical_plane: PlaneDocument,
    pub(crate) parameter_c: ComplexDocument,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) newton_initial_z: Option<ComplexDocument>,
    /// Selected point of non-Newton overview/detail instruments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) initial_z: Option<ComplexDocument>,
    /// Family-specific numerical settings; absent for families without any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) family_parameters: Option<FamilyParametersDocument>,
    pub(crate) computation: ComputationDocument,
    pub(crate) display: DisplayDocument,
    /// Gradient and colouring algorithms of a single-layer image (format
    /// version 5); superseded by `layers`. Absent in older documents, which
    /// are coloured with the defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) colouring: Option<Colouring>,
    /// The layer stack, bottom first (format version 6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) layers: Option<Vec<Layer>>,
    #[serde(
        default,
        alias = "initial_julia_zoom_exponent",
        skip_serializing_if = "is_zero"
    )]
    pub(crate) progressive_julia_zoom_target_exponent: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) deep_parameter_plane: Option<DeepPlaneDocument>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) deep_dynamical_plane: Option<DeepPlaneDocument>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) deep_parameter_c: Option<DeepComplexDocument>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FamilyParametersDocument {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) degree: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) nova_relaxation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) lyapunov_sequence: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) mandelbox_scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) mandelbox_min_radius: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) mandelbox_fixed_radius: Option<f64>,
}

impl FamilyParametersDocument {
    /// Records only the settings the family actually uses, so documents stay
    /// minimal and reproducible.
    pub(crate) fn for_family(family: FractalFamily, parameters: &FamilyParameters) -> Option<Self> {
        if !family.has_family_parameters() {
            return None;
        }
        Some(Self {
            degree: family.uses_degree().then_some(parameters.degree),
            nova_relaxation: family
                .uses_relaxation()
                .then_some(parameters.nova_relaxation),
            lyapunov_sequence: family
                .uses_lyapunov_sequence()
                .then(|| parameters.lyapunov_sequence.clone()),
            mandelbox_scale: family
                .uses_mandelbox()
                .then_some(parameters.mandelbox_scale),
            mandelbox_min_radius: family
                .uses_mandelbox()
                .then_some(parameters.mandelbox_min_radius),
            mandelbox_fixed_radius: family
                .uses_mandelbox()
                .then_some(parameters.mandelbox_fixed_radius),
        })
    }

    /// Applies the recorded settings on top of the current parameters.
    pub(crate) fn apply_to(&self, parameters: &mut FamilyParameters) {
        if let Some(value) = self.degree {
            parameters.degree = value;
        }
        if let Some(value) = self.nova_relaxation {
            parameters.nova_relaxation = value;
        }
        if let Some(value) = &self.lyapunov_sequence {
            parameters.lyapunov_sequence = value.clone();
        }
        if let Some(value) = self.mandelbox_scale {
            parameters.mandelbox_scale = value;
        }
        if let Some(value) = self.mandelbox_min_radius {
            parameters.mandelbox_min_radius = value;
        }
        if let Some(value) = self.mandelbox_fixed_radius {
            parameters.mandelbox_fixed_radius = value;
        }
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        let mut parameters = FamilyParameters::default();
        self.apply_to(&mut parameters);
        parameters
            .validate()
            .map_err(|error| format!("family_parameters: {error}"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeepPlaneDocument {
    pub(crate) centre: DeepComplexDocument,
    pub(crate) half_height: String,
    pub(crate) magnification_log10: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DeepComplexDocument {
    pub(crate) re: String,
    pub(crate) im: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlaneDocument {
    pub(crate) centre: ComplexDocument,
    pub(crate) half_height: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ComplexDocument {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ComputationDocument {
    pub(crate) iterations: u32,
    pub(crate) bailout: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DisplayDocument {
    pub(crate) smooth_escape_time: bool,
    pub(crate) coordinate_grid: bool,
    /// Gradient offset of documents before format version 5; newer documents
    /// record the offset inside `colouring`.
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub(crate) palette_phase: f32,
    #[serde(default = "default_true")]
    pub(crate) critical_orbit_overlay: bool,
    /// Shade bounded orbits by their minimum modulus (format version 4).
    #[serde(default = "default_true")]
    pub(crate) interior_shading: bool,
}

const fn default_true() -> bool {
    true
}

const fn is_zero(value: &u32) -> bool {
    *value == 0
}

fn is_zero_f32(value: &f32) -> bool {
    *value == 0.0
}

impl ExperimentDocument {
    pub(crate) fn to_pretty_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|error| error.to_string())
    }

    pub(crate) fn from_json(json: &str) -> Result<Self, String> {
        let document: Self = serde_json::from_str(json).map_err(|error| error.to_string())?;
        document.validate()?;
        Ok(document)
    }

    fn validate(&self) -> Result<(), String> {
        if self.format != FORMAT_ID {
            return Err(format!("unsupported document format {:?}", self.format));
        }
        if !(1..=FORMAT_VERSION).contains(&self.version) {
            return Err(format!(
                "unsupported document version {}; supported versions are 1 through {}",
                self.version, FORMAT_VERSION,
            ));
        }
        let Some(family) = FractalFamily::from_document_id(&self.family) else {
            return Err(format!("unsupported fractal family {:?}", self.family));
        };
        validate_plane("parameter_plane", self.parameter_plane)?;
        validate_plane("dynamical_plane", self.dynamical_plane)?;
        validate_complex("parameter_c", self.parameter_c)?;
        if let Some(value) = self.newton_initial_z {
            validate_complex("newton_initial_z", value)?;
        }
        if let Some(value) = self.initial_z {
            validate_complex("initial_z", value)?;
        }
        let minimum_iterations = family.min_iterations();
        let maximum_iterations = family.max_iterations();
        if !(minimum_iterations..=maximum_iterations).contains(&self.computation.iterations) {
            return Err(format!(
                "iterations must be between {minimum_iterations} and {maximum_iterations}"
            ));
        }
        if !self.computation.bailout.is_finite()
            || !(2.0..=MAX_BAILOUT).contains(&self.computation.bailout)
        {
            return Err("bailout must be finite and between 2 and 1e10".to_owned());
        }
        if let Some(colouring) = &self.colouring {
            colouring
                .validate()
                .map_err(|error| format!("colouring: {error}"))?;
            if self.version < 5 {
                return Err("colouring requires document version 5".to_owned());
            }
        }
        if let Some(layers) = &self.layers {
            if layers.is_empty() || layers.len() > MAX_LAYERS {
                return Err(format!(
                    "layers must hold between 1 and {MAX_LAYERS} layers"
                ));
            }
            for (index, layer) in layers.iter().enumerate() {
                layer.validate(index)?;
            }
            if self.version < 6 {
                return Err("layers require document version 6".to_owned());
            }
            if self.version < 7 && layers.iter().any(|layer| layer.scene.is_some()) {
                return Err("per-layer scenes require document version 7".to_owned());
            }
            if self.colouring.is_some() {
                return Err("a document records either layers or colouring, not both".to_owned());
            }
        }
        if !self.display.palette_phase.is_finite()
            || !(-1.0..=1.0).contains(&self.display.palette_phase)
        {
            return Err("palette_phase must be finite and between -1 and 1".to_owned());
        }
        if self.progressive_julia_zoom_target_exponent > MAX_DECIMAL_ZOOM_EXPONENT {
            return Err(format!(
                "progressive_julia_zoom_target_exponent must be between 0 and {MAX_DECIMAL_ZOOM_EXPONENT}"
            ));
        }
        if !family.supports_deep_zoom()
            && (self.progressive_julia_zoom_target_exponent != 0
                || self.deep_parameter_plane.is_some()
                || self.deep_dynamical_plane.is_some()
                || self.deep_parameter_c.is_some())
        {
            return Err(format!(
                "deep arbitrary-precision state is not valid for {} experiments",
                family.name()
            ));
        }
        if family.is_newton() && self.newton_initial_z.is_none() {
            return Err("Newton experiments require newton_initial_z".to_owned());
        }
        if !family.is_newton() && self.newton_initial_z.is_some() {
            return Err("newton_initial_z is only valid for Newton experiments".to_owned());
        }
        let uses_initial_z = !family.is_newton() && family.linkage() == Linkage::OverviewDetail;
        if uses_initial_z && self.initial_z.is_none() {
            return Err(format!("{} experiments require initial_z", family.name()));
        }
        if !uses_initial_z && self.initial_z.is_some() {
            return Err("initial_z is only valid for overview/detail experiments".to_owned());
        }
        if let Some(parameters) = &self.family_parameters {
            if !family.has_family_parameters() {
                return Err(format!(
                    "family_parameters are not valid for {} experiments",
                    family.name()
                ));
            }
            parameters.validate()?;
        }
        if let Some(plane) = &self.deep_parameter_plane {
            validate_deep_plane("deep_parameter_plane", plane)?;
        }
        if let Some(plane) = &self.deep_dynamical_plane {
            validate_deep_plane("deep_dynamical_plane", plane)?;
        }
        if let Some(value) = &self.deep_parameter_c {
            let exponent = self
                .deep_parameter_plane
                .as_ref()
                .map_or(15, |plane| plane.magnification_log10.ceil() as u32);
            crate::arbitrary::DeepComplex::parse(&value.re, &value.im, exponent)
                .map_err(|error| format!("deep_parameter_c: {error}"))?;
        }
        if self.version == 1
            && (self.deep_parameter_plane.is_some()
                || self.deep_dynamical_plane.is_some()
                || self.deep_parameter_c.is_some()
                || self.progressive_julia_zoom_target_exponent != 0)
        {
            return Err("deep arbitrary-precision state requires document version 2".to_owned());
        }
        if self.version < 3 && self.newton_initial_z.is_some() {
            return Err("Newton state requires document version 3".to_owned());
        }
        if self.version < 4
            && (!(family.is_quadratic() || family.is_newton())
                || self.initial_z.is_some()
                || self.family_parameters.is_some())
        {
            return Err(format!(
                "{} experiments require document version 4",
                family.name()
            ));
        }
        Ok(())
    }
}

fn validate_deep_plane(name: &str, plane: &DeepPlaneDocument) -> Result<(), String> {
    crate::arbitrary::DeepView::parse(
        &plane.centre.re,
        &plane.centre.im,
        &plane.half_height,
        plane.magnification_log10,
    )
    .map(|_| ())
    .map_err(|error| format!("{name}: {error}"))
}

fn validate_plane(name: &str, plane: PlaneDocument) -> Result<(), String> {
    validate_complex(&format!("{name}.centre"), plane.centre)?;
    if !plane.half_height.is_finite() || !(1e-14..=1e6).contains(&plane.half_height) {
        return Err(format!(
            "{name}.half_height must be finite and between 1e-14 and 1e6"
        ));
    }
    Ok(())
}

fn validate_complex(name: &str, value: ComplexDocument) -> Result<(), String> {
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(format!("{name} coordinates must be finite"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MAX_ITERATIONS;

    fn example() -> ExperimentDocument {
        ExperimentDocument {
            format: FORMAT_ID.to_owned(),
            version: FORMAT_VERSION,
            family: QUADRATIC_FAMILY_ID.to_owned(),
            parameter_plane: PlaneDocument {
                centre: ComplexDocument {
                    re: -0.743_643_887_037_151,
                    im: 0.131_825_904_205_33,
                },
                half_height: 1.45e-9,
            },
            dynamical_plane: PlaneDocument {
                centre: ComplexDocument { re: 0.0, im: 0.0 },
                half_height: 0.725,
            },
            parameter_c: ComplexDocument {
                re: -0.743_643_887_037_151,
                im: 0.131_825_904_205_33,
            },
            newton_initial_z: None,
            initial_z: None,
            family_parameters: None,
            computation: ComputationDocument {
                iterations: 2048,
                bailout: 4.0,
            },
            display: DisplayDocument {
                smooth_escape_time: true,
                coordinate_grid: false,
                palette_phase: 0.25,
                critical_orbit_overlay: true,
                interior_shading: true,
            },
            colouring: None,
            layers: None,
            progressive_julia_zoom_target_exponent: 0,
            deep_parameter_plane: None,
            deep_dynamical_plane: None,
            deep_parameter_c: None,
        }
    }

    #[test]
    fn pretty_json_round_trips_exactly() {
        let document = example();
        let json = document.to_pretty_json().unwrap();
        assert_eq!(ExperimentDocument::from_json(&json).unwrap(), document);
    }

    #[test]
    fn thousand_digit_deep_state_round_trips_as_decimal_strings() {
        let mut document = example();
        document.deep_dynamical_plane = Some(DeepPlaneDocument {
            centre: DeepComplexDocument {
                re: "-7.45e-1".to_owned(),
                im: "1.13e-1".to_owned(),
            },
            half_height: "1.45e-1000".to_owned(),
            magnification_log10: 1_000.0,
        });
        let json = document.to_pretty_json().unwrap();
        let restored = ExperimentDocument::from_json(&json).unwrap();
        assert_eq!(restored, document);
        assert!(json.contains("1.45e-1000"));
    }

    #[test]
    fn colouring_round_trips_and_requires_version_five() {
        use crate::colouring::{ColouringAlgorithm, Gradient};
        let mut document = example();
        let mut colouring = Colouring {
            gradient: Gradient::random(11),
            ..Colouring::default()
        };
        colouring
            .outside
            .set_algorithm(ColouringAlgorithm::TriangleInequality);
        colouring
            .inside
            .set_algorithm(ColouringAlgorithm::OrbitTrap);
        document.colouring = Some(colouring);
        document.computation.bailout = 1e8;
        document.version = 5;
        let json = document.to_pretty_json().unwrap();
        assert_eq!(ExperimentDocument::from_json(&json).unwrap(), document);

        document.version = 4;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("version 5"), "{error}");

        document.version = 5;
        document.colouring.as_mut().unwrap().outside.density = -1.0;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("colouring: outside.density"), "{error}");

        document.computation.bailout = 1e11;
        assert!(ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).is_err());
    }

    #[test]
    fn layer_stacks_round_trip_and_require_version_six() {
        use crate::colouring::{ColouringAlgorithm, Gradient, MergeMode};
        let mut document = example();
        let mut top = Layer {
            name: "Stripes".to_owned(),
            opacity: 0.6,
            merge_mode: MergeMode::Multiply,
            ..Layer::default()
        };
        top.colouring.gradient = Gradient::random(21);
        top.colouring
            .outside
            .set_algorithm(ColouringAlgorithm::Stripes);
        document.layers = Some(vec![Layer::default(), top]);
        let json = document.to_pretty_json().unwrap();
        assert_eq!(ExperimentDocument::from_json(&json).unwrap(), document);

        document.version = 5;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("version 6"), "{error}");

        document.version = 6;
        document.layers = Some(vec![Layer::default(); MAX_LAYERS + 1]);
        assert!(ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).is_err());
        document.layers = Some(vec![Layer {
            opacity: 2.0,
            ..Layer::default()
        }]);
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("opacity"), "{error}");
    }

    #[test]
    fn detached_layer_scenes_round_trip_and_require_version_seven() {
        use crate::colouring::LayerScene;
        let mut document = example();
        let mut layer = Layer {
            name: "Backdrop".to_owned(),
            ..Layer::default()
        };
        layer.scene = Some(LayerScene {
            family: crate::family::FractalFamily::BurningShip,
            dynamical: false,
            centre: [-0.5, -0.5],
            half_height: 1.0,
            iterations: 400,
            ..LayerScene::default()
        });
        document.layers = Some(vec![Layer::default(), layer]);
        let json = document.to_pretty_json().unwrap();
        assert_eq!(ExperimentDocument::from_json(&json).unwrap(), document);

        document.version = 6;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("version 7"), "{error}");

        document.version = 7;
        document.layers.as_mut().unwrap()[1]
            .scene
            .as_mut()
            .unwrap()
            .half_height = 1e-20;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("half_height"), "{error}");
    }

    #[test]
    fn older_documents_without_colouring_still_load() {
        let mut document = example();
        document.version = 4;
        let json = document.to_pretty_json().unwrap();
        assert!(!json.contains("colouring"));
        assert_eq!(ExperimentDocument::from_json(&json).unwrap(), document);
    }

    #[test]
    fn unknown_versions_are_rejected() {
        let mut document = example();
        document.version += 1;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("unsupported document version"));
    }

    #[test]
    fn older_version_one_documents_enable_the_orbit_overlay() {
        let mut document = example();
        document.version = 1;
        let mut value = serde_json::to_value(document).unwrap();
        value["display"]
            .as_object_mut()
            .unwrap()
            .remove("critical_orbit_overlay");
        let loaded =
            ExperimentDocument::from_json(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(loaded.display.critical_orbit_overlay);
    }

    #[test]
    fn iteration_limit_accepts_fifty_thousand_and_rejects_more() {
        let mut document = example();
        document.computation.iterations = MAX_ITERATIONS;
        assert!(ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).is_ok());

        document.computation.iterations = MAX_ITERATIONS + 1;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("50000"));
    }

    #[test]
    fn progressive_julia_target_accepts_five_thousand_and_rejects_more() {
        let mut document = example();
        document.progressive_julia_zoom_target_exponent = MAX_DECIMAL_ZOOM_EXPONENT;
        assert!(ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).is_ok());

        document.progressive_julia_zoom_target_exponent = MAX_DECIMAL_ZOOM_EXPONENT + 1;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("5000"));
    }

    #[test]
    fn newton_experiment_round_trips_and_rejects_quadratic_deep_state() {
        let mut document = example();
        document.family = NEWTON_FAMILY_ID.to_owned();
        document.computation.iterations = 128;
        document.newton_initial_z = Some(ComplexDocument { re: 0.5, im: 0.5 });
        let json = document.to_pretty_json().unwrap();
        assert_eq!(ExperimentDocument::from_json(&json).unwrap(), document);

        // Newton now supports deep zoom through the Nova perturbation path, so
        // its documents may carry deep state; a family without a reference
        // orbit may not.
        document.progressive_julia_zoom_target_exponent = 100;
        assert!(ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).is_ok());
        document.family = FractalFamily::Collatz.document_id().to_owned();
        document.newton_initial_z = None;
        document.initial_z = Some(ComplexDocument { re: 0.5, im: 0.5 });
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("not valid for Collatz"));
    }

    #[test]
    fn escape_time_families_round_trip_with_their_parameters() {
        let mut document = example();
        document.family = FractalFamily::Nova.document_id().to_owned();
        document.family_parameters = Some(FamilyParametersDocument {
            degree: Some(4),
            nova_relaxation: Some(1.5),
            lyapunov_sequence: None,
            mandelbox_scale: None,
            mandelbox_min_radius: None,
            mandelbox_fixed_radius: None,
        });
        let json = document.to_pretty_json().unwrap();
        assert!(json.contains("\"nova\""));
        assert!(!json.contains("mandelbox_scale"));
        assert_eq!(ExperimentDocument::from_json(&json).unwrap(), document);

        // Out-of-range parameters are rejected before any state changes.
        document.family_parameters.as_mut().unwrap().degree = Some(12);
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("degree"));

        // Families without parameters reject stray parameter blocks.
        document.family = FractalFamily::BurningShip.document_id().to_owned();
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("family_parameters"));
        document.family_parameters = None;
        assert!(ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).is_ok());

        // Deep state is accepted by perturbation-capable families only.
        document.progressive_julia_zoom_target_exponent = 50;
        assert!(ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).is_ok());
        document.family = FractalFamily::Sine.document_id().to_owned();
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("deep arbitrary-precision state"));
    }

    #[test]
    fn overview_detail_families_record_the_selected_point() {
        let mut document = example();
        document.family = FractalFamily::Lyapunov.document_id().to_owned();
        document.family_parameters = Some(FamilyParametersDocument {
            degree: None,
            nova_relaxation: None,
            lyapunov_sequence: Some("BBBBBBAAAAAA".to_owned()),
            mandelbox_scale: None,
            mandelbox_min_radius: None,
            mandelbox_fixed_radius: None,
        });
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("initial_z"));
        document.initial_z = Some(ComplexDocument { re: 3.2, im: 3.6 });
        let json = document.to_pretty_json().unwrap();
        assert_eq!(ExperimentDocument::from_json(&json).unwrap(), document);

        document
            .family_parameters
            .as_mut()
            .unwrap()
            .lyapunov_sequence = Some("ABC".to_owned());
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("lyapunov_sequence"));
    }

    #[test]
    fn new_families_require_document_version_four() {
        let mut document = example();
        document.family = FractalFamily::Tricorn.document_id().to_owned();
        document.version = 3;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("version 4"));
        document.version = 4;
        assert!(ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).is_ok());
    }

    #[test]
    fn newton_iteration_limit_matches_the_shader_limit() {
        let mut document = example();
        document.family = NEWTON_FAMILY_ID.to_owned();
        document.newton_initial_z = Some(ComplexDocument { re: 0.5, im: 0.5 });
        document.computation.iterations = 2_048;
        assert!(ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).is_ok());

        document.computation.iterations = 2_049;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("2048"));
    }
}
