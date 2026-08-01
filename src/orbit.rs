//! Critical-orbit calculation and cached scientific diagnostics.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OrbitInput {
    pub(crate) c: [f64; 2],
    pub(crate) iterations: u32,
    pub(crate) bailout: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OrbitPoint {
    pub(crate) iteration: u32,
    pub(crate) z: [f64; 2],
    pub(crate) magnitude: f64,
    /// Parameter sensitivity dz_n/dc for z_0 = 0 and z_(n+1) = z_n² + c.
    pub(crate) parameter_derivative: [f64; 2],
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CriticalOrbit {
    pub(crate) points: Vec<OrbitPoint>,
    pub(crate) escape_iteration: Option<u32>,
    pub(crate) smooth_escape_iteration: Option<f64>,
}

impl CriticalOrbit {
    pub(crate) fn calculate(input: OrbitInput) -> Self {
        let mut points = Vec::with_capacity(input.iterations as usize + 1);
        let mut z = [0.0_f64; 2];
        let mut derivative = [0.0_f64; 2];
        points.push(point(0, z, derivative));

        let bailout_squared = input.bailout * input.bailout;
        for iteration in 1..=input.iterations {
            derivative = [
                2.0 * (z[0] * derivative[0] - z[1] * derivative[1]) + 1.0,
                2.0 * (z[0] * derivative[1] + z[1] * derivative[0]),
            ];
            z = [
                z[0] * z[0] - z[1] * z[1] + input.c[0],
                2.0 * z[0] * z[1] + input.c[1],
            ];
            let orbit_point = point(iteration, z, derivative);
            points.push(orbit_point);

            let magnitude_squared = z[0] * z[0] + z[1] * z[1];
            if !magnitude_squared.is_finite() || magnitude_squared > bailout_squared {
                let smooth = if magnitude_squared.is_finite() {
                    let log_zn = 0.5 * magnitude_squared.max(1.000_001).ln();
                    Some(iteration as f64 + 1.0 - log_zn.max(1e-12).log2())
                } else {
                    None
                };
                return Self {
                    points,
                    escape_iteration: Some(iteration),
                    smooth_escape_iteration: smooth,
                };
            }
        }

        Self {
            points,
            escape_iteration: None,
            smooth_escape_iteration: None,
        }
    }

    pub(crate) fn last_iteration(&self) -> usize {
        self.points.len().saturating_sub(1)
    }
}

fn point(iteration: u32, z: [f64; 2], parameter_derivative: [f64; 2]) -> OrbitPoint {
    OrbitPoint {
        iteration,
        z,
        magnitude: z[0].hypot(z[1]),
        parameter_derivative,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OrbitKey {
    c: [u64; 2],
    iterations: u32,
    bailout: u64,
}

impl From<OrbitInput> for OrbitKey {
    fn from(input: OrbitInput) -> Self {
        Self {
            c: [input.c[0].to_bits(), input.c[1].to_bits()],
            iterations: input.iterations,
            bailout: input.bailout.to_bits(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CriticalOrbitCache {
    key: Option<OrbitKey>,
    orbit: CriticalOrbit,
}

impl CriticalOrbitCache {
    pub(crate) fn update(&mut self, input: OrbitInput) -> &CriticalOrbit {
        let key = input.into();
        if self.key != Some(key) {
            self.orbit = CriticalOrbit::calculate(input);
            self.key = Some(key);
        }
        &self.orbit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_parameter_has_a_stationary_critical_orbit() {
        let orbit = CriticalOrbit::calculate(OrbitInput {
            c: [0.0, 0.0],
            iterations: 32,
            bailout: 4.0,
        });
        assert_eq!(orbit.escape_iteration, None);
        assert_eq!(orbit.points.len(), 33);
        assert!(orbit.points.iter().all(|point| point.z == [0.0, 0.0]));
    }

    #[test]
    fn c_one_escapes_at_the_expected_iteration() {
        let orbit = CriticalOrbit::calculate(OrbitInput {
            c: [1.0, 0.0],
            iterations: 32,
            bailout: 4.0,
        });
        assert_eq!(orbit.escape_iteration, Some(3));
        assert_eq!(orbit.points[3].z, [5.0, 0.0]);
        assert_eq!(orbit.points[3].parameter_derivative, [13.0, 0.0]);
        assert!(orbit.smooth_escape_iteration.is_some());
    }

    #[test]
    fn cache_reuses_an_identical_input() {
        let input = OrbitInput {
            c: [-1.0, 0.0],
            iterations: 64,
            bailout: 4.0,
        };
        let mut cache = CriticalOrbitCache::default();
        let first_length = cache.update(input).points.len();
        assert_eq!(cache.update(input).points.len(), first_length);
        assert_eq!(cache.update(input).escape_iteration, None);
    }
}
