//! Newton iteration diagnostics for the polynomial p(z) = z^3 - 1.

pub(crate) const ROOTS: [[f64; 2]; 3] = [
    [1.0, 0.0],
    [-0.5, 0.866_025_403_784_438_6],
    [-0.5, -0.866_025_403_784_438_6],
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct NewtonResult {
    pub(crate) initial: [f64; 2],
    pub(crate) value: [f64; 2],
    pub(crate) root: Option<usize>,
    pub(crate) iterations: u32,
    pub(crate) residual: f64,
    pub(crate) last_step: f64,
    pub(crate) singular: bool,
}

impl NewtonResult {
    pub(crate) fn calculate(initial: [f64; 2], max_iterations: u32) -> Self {
        let mut z = initial;
        let mut last_step = 0.0;
        for iteration in 0..max_iterations {
            let z2 = complex_mul(z, z);
            let z3 = complex_mul(z2, z);
            let polynomial = [z3[0] - 1.0, z3[1]];
            let residual = polynomial[0].hypot(polynomial[1]);
            if residual <= 1e-12 {
                return Self::converged(initial, z, iteration, residual, last_step);
            }

            let derivative = [3.0 * z2[0], 3.0 * z2[1]];
            let denominator = derivative[0] * derivative[0] + derivative[1] * derivative[1];
            if !denominator.is_finite() || denominator <= 1e-28 {
                return Self {
                    initial,
                    value: z,
                    root: None,
                    iterations: iteration,
                    residual,
                    last_step,
                    singular: true,
                };
            }
            let correction = complex_div(polynomial, derivative, denominator);
            last_step = correction[0].hypot(correction[1]);
            z = [z[0] - correction[0], z[1] - correction[1]];
            if !z[0].is_finite() || !z[1].is_finite() {
                return Self {
                    initial,
                    value: z,
                    root: None,
                    iterations: iteration + 1,
                    residual: f64::INFINITY,
                    last_step,
                    singular: true,
                };
            }
        }

        let z2 = complex_mul(z, z);
        let z3 = complex_mul(z2, z);
        let residual = (z3[0] - 1.0).hypot(z3[1]);
        Self {
            initial,
            value: z,
            root: (residual <= 1e-12).then(|| nearest_root(z)),
            iterations: max_iterations,
            residual,
            last_step,
            singular: false,
        }
    }

    fn converged(
        initial: [f64; 2],
        value: [f64; 2],
        iterations: u32,
        residual: f64,
        last_step: f64,
    ) -> Self {
        Self {
            initial,
            value,
            root: Some(nearest_root(value)),
            iterations,
            residual,
            last_step,
            singular: false,
        }
    }
}

fn complex_mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn complex_div(a: [f64; 2], b: [f64; 2], denominator: f64) -> [f64; 2] {
    [
        (a[0] * b[0] + a[1] * b[1]) / denominator,
        (a[1] * b[0] - a[0] * b[1]) / denominator,
    ]
}

fn nearest_root(z: [f64; 2]) -> usize {
    ROOTS
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| squared_distance(z, **a).total_cmp(&squared_distance(z, **b)))
        .map_or(0, |(index, _)| index)
}

fn squared_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_seed_regions_converge_to_three_distinct_roots() {
        let seeds = [[1.2, 0.1], [-0.7, 0.8], [-0.7, -0.8]];
        for (expected, seed) in seeds.into_iter().enumerate() {
            let result = NewtonResult::calculate(seed, 64);
            assert_eq!(result.root, Some(expected));
            assert!(result.residual <= 1e-12);
        }
    }

    #[test]
    fn origin_is_reported_as_a_derivative_singularity() {
        let result = NewtonResult::calculate([0.0, 0.0], 64);
        assert!(result.singular);
        assert_eq!(result.root, None);
    }

    #[test]
    fn exact_root_requires_no_newton_step() {
        let result = NewtonResult::calculate([1.0, 0.0], 64);
        assert_eq!(result.root, Some(0));
        assert_eq!(result.iterations, 0);
        assert_eq!(result.residual, 0.0);
    }
}
