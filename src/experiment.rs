//! Versioned, deterministic experiment documents.

use serde::{Deserialize, Serialize};

use crate::MAX_ITERATIONS;
use crate::arbitrary::MAX_DECIMAL_ZOOM_EXPONENT;

pub(crate) const FORMAT_ID: &str = "iterascope-experiment";
pub(crate) const FORMAT_VERSION: u32 = 3;
pub(crate) const QUADRATIC_FAMILY_ID: &str = "quadratic";
pub(crate) const NEWTON_FAMILY_ID: &str = "newton-cubic";

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
    pub(crate) computation: ComputationDocument,
    pub(crate) display: DisplayDocument,
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
    pub(crate) palette_phase: f32,
    #[serde(default = "default_true")]
    pub(crate) critical_orbit_overlay: bool,
}

const fn default_true() -> bool {
    true
}

const fn is_zero(value: &u32) -> bool {
    *value == 0
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
        if self.family != QUADRATIC_FAMILY_ID && self.family != NEWTON_FAMILY_ID {
            return Err(format!("unsupported fractal family {:?}", self.family));
        }
        validate_plane("parameter_plane", self.parameter_plane)?;
        validate_plane("dynamical_plane", self.dynamical_plane)?;
        validate_complex("parameter_c", self.parameter_c)?;
        if let Some(value) = self.newton_initial_z {
            validate_complex("newton_initial_z", value)?;
        }
        let minimum_iterations = if self.family == NEWTON_FAMILY_ID {
            8
        } else {
            32
        };
        let maximum_iterations = if self.family == NEWTON_FAMILY_ID {
            2_048
        } else {
            MAX_ITERATIONS
        };
        if !(minimum_iterations..=maximum_iterations).contains(&self.computation.iterations) {
            return Err(format!(
                "iterations must be between {minimum_iterations} and {maximum_iterations}"
            ));
        }
        if !self.computation.bailout.is_finite()
            || !(2.0..=32.0).contains(&self.computation.bailout)
        {
            return Err("bailout must be finite and between 2 and 32".to_owned());
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
        if self.family == NEWTON_FAMILY_ID
            && (self.progressive_julia_zoom_target_exponent != 0
                || self.deep_parameter_plane.is_some()
                || self.deep_dynamical_plane.is_some()
                || self.deep_parameter_c.is_some())
        {
            return Err("deep quadratic state is not valid for Newton experiments".to_owned());
        }
        if self.family == NEWTON_FAMILY_ID && self.newton_initial_z.is_none() {
            return Err("Newton experiments require newton_initial_z".to_owned());
        }
        if self.family == QUADRATIC_FAMILY_ID && self.newton_initial_z.is_some() {
            return Err("newton_initial_z is only valid for Newton experiments".to_owned());
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
            computation: ComputationDocument {
                iterations: 2048,
                bailout: 4.0,
            },
            display: DisplayDocument {
                smooth_escape_time: true,
                coordinate_grid: false,
                palette_phase: 0.25,
                critical_orbit_overlay: true,
            },
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

        document.progressive_julia_zoom_target_exponent = 100;
        let error = ExperimentDocument::from_json(&document.to_pretty_json().unwrap()).unwrap_err();
        assert!(error.contains("not valid for Newton"));
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
