//! Versioned, deterministic experiment documents.

use serde::{Deserialize, Serialize};

pub(crate) const FORMAT_ID: &str = "iterascope-experiment";
pub(crate) const FORMAT_VERSION: u32 = 1;
pub(crate) const FAMILY_ID: &str = "quadratic";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExperimentDocument {
    pub(crate) format: String,
    pub(crate) version: u32,
    pub(crate) family: String,
    pub(crate) parameter_plane: PlaneDocument,
    pub(crate) dynamical_plane: PlaneDocument,
    pub(crate) parameter_c: ComplexDocument,
    pub(crate) computation: ComputationDocument,
    pub(crate) display: DisplayDocument,
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
        if self.version != FORMAT_VERSION {
            return Err(format!(
                "unsupported document version {}; expected {}",
                self.version, FORMAT_VERSION
            ));
        }
        if self.family != FAMILY_ID {
            return Err(format!("unsupported fractal family {:?}", self.family));
        }
        validate_plane("parameter_plane", self.parameter_plane)?;
        validate_plane("dynamical_plane", self.dynamical_plane)?;
        validate_complex("parameter_c", self.parameter_c)?;
        if !(32..=2048).contains(&self.computation.iterations) {
            return Err("iterations must be between 32 and 2048".to_owned());
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
        Ok(())
    }
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
            family: FAMILY_ID.to_owned(),
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
        }
    }

    #[test]
    fn pretty_json_round_trips_exactly() {
        let document = example();
        let json = document.to_pretty_json().unwrap();
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
        let document = example();
        let mut value = serde_json::to_value(document).unwrap();
        value["display"]
            .as_object_mut()
            .unwrap()
            .remove("critical_orbit_overlay");
        let loaded =
            ExperimentDocument::from_json(&serde_json::to_string(&value).unwrap()).unwrap();
        assert!(loaded.display.critical_orbit_overlay);
    }
}
