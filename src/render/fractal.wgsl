struct Uniforms {
    // Centre xy and half-height z, high words. w = aspect ratio.
    view_hi: vec4<f32>,
    // Matching low words. w = 1 for double-single rendering.
    view_lo: vec4<f32>,
    // Julia c high words, maximum iterations, bailout squared.
    dynamics_hi: vec4<f32>,
    // Julia c low words; z = 0 quadratic / 1 Newton cubic; w reserved.
    dynamics_lo: vec4<f32>,
    // x = 0 parameter plane / 1 dynamical plane, y = palette phase,
    // z = smooth colouring, w = show grid.
    display: vec4<f32>,
    // x = perturbation enabled, y = scale mantissa, z = base-2 scale
    // exponent, w = number of uploaded reference points.
    deep: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct ReferencePoint {
    z_hi: vec2<f32>,
    z_lo: vec2<f32>,
};

@group(0) @binding(1) var<storage, read> reference_orbit: array<ReferencePoint>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct Ds {
    hi: f32,
    lo: f32,
};

struct Ds2 {
    x: Ds,
    y: Ds,
};

struct EscapeResult {
    escaped: bool,
    iteration: u32,
    z: vec2<f32>,
};

struct ScaledComplex {
    mantissa: vec2<f32>,
    exponent: i32,
};

struct NewtonResult {
    converged: bool,
    root: u32,
    iteration: u32,
    continuous_iteration: f32,
    residual: f32,
    z: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let corner = corners[vertex_index];
    var out: VertexOut;
    out.position = vec4<f32>(corner, 0.0, 1.0);
    out.uv = corner * 0.5 + vec2<f32>(0.5);
    return out;
}

fn ds_normalize(hi: f32, lo: f32) -> Ds {
    let sum = hi + lo;
    // The fma calls are deliberate optimization barriers. Reassociation of
    // the compensated subtraction identities reduces DS arithmetic to f32 on
    // otherwise valid WebGPU backends.
    let error = fma(-1.0, sum - hi, lo);
    return Ds(sum, error);
}

fn ds_add(a: Ds, b: Ds) -> Ds {
    let sum = a.hi + b.hi;
    let virtual_b = sum - a.hi;
    let a_error = fma(-1.0, sum - virtual_b, a.hi);
    let b_error = fma(-1.0, virtual_b, b.hi);
    let error = a_error + b_error + a.lo + b.lo;
    return ds_normalize(sum, error);
}

fn ds_sub(a: Ds, b: Ds) -> Ds {
    return ds_add(a, Ds(-b.hi, -b.lo));
}

fn ds_mul(a: Ds, b: Ds) -> Ds {
    let product = a.hi * b.hi;
    // fma recovers the product's rounding residue on WebGPU hardware. Cross
    // terms then retain the low words carried from the CPU or prior operation.
    let error = fma(a.hi, b.hi, -product)
        + a.hi * b.lo
        + a.lo * b.hi
        + a.lo * b.lo;
    return ds_normalize(product, error);
}

fn ds_div(a: Ds, b: Ds) -> Ds {
    let estimate = a.hi / b.hi;
    let remainder = ds_sub(a, ds_mul(b, Ds(estimate, 0.0)));
    let correction = (remainder.hi + remainder.lo) / b.hi;
    return ds_add(Ds(estimate, 0.0), Ds(correction, 0.0));
}

fn ds_approx(a: Ds) -> f32 {
    return a.hi + a.lo;
}

fn ds2_approx(a: Ds2) -> vec2<f32> {
    return vec2<f32>(ds_approx(a.x), ds_approx(a.y));
}

fn ds2_add_f32(a: Ds2, b: vec2<f32>) -> Ds2 {
    return Ds2(
        ds_add(a.x, Ds(b.x, 0.0)),
        ds_add(a.y, Ds(b.y, 0.0)),
    );
}

fn ds2_sub(a: Ds2, b: Ds2) -> Ds2 {
    return Ds2(ds_sub(a.x, b.x), ds_sub(a.y, b.y));
}

fn ds_complex_mul(a: Ds2, b: Ds2) -> Ds2 {
    return Ds2(
        ds_sub(ds_mul(a.x, b.x), ds_mul(a.y, b.y)),
        ds_add(ds_mul(a.x, b.y), ds_mul(a.y, b.x)),
    );
}

fn ds_complex_div(a: Ds2, b: Ds2) -> Ds2 {
    let denominator = ds_add(ds_mul(b.x, b.x), ds_mul(b.y, b.y));
    return Ds2(
        ds_div(ds_add(ds_mul(a.x, b.x), ds_mul(a.y, b.y)), denominator),
        ds_div(ds_sub(ds_mul(a.y, b.x), ds_mul(a.x, b.y)), denominator),
    );
}

fn ds_complex_square(z: Ds2) -> Ds2 {
    let xx = ds_mul(z.x, z.x);
    let yy = ds_mul(z.y, z.y);
    let xy = ds_mul(z.x, z.y);
    return Ds2(
        ds_sub(xx, yy),
        ds_mul(Ds(2.0, 0.0), xy),
    );
}

fn complex_square(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
}

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y,
                     a.x * b.y + a.y * b.x);
}

fn complex_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let denominator = dot(b, b);
    return vec2<f32>(
        (a.x * b.x + a.y * b.y) / denominator,
        (a.y * b.x - a.x * b.y) / denominator,
    );
}

fn nearest_newton_root(z: vec2<f32>) -> u32 {
    let distance_a = distance(z, vec2<f32>(1.0, 0.0));
    let distance_b = distance(z, vec2<f32>(-0.5, 0.8660254));
    let distance_c = distance(z, vec2<f32>(-0.5, -0.8660254));
    if (distance_b < distance_a && distance_b <= distance_c) {
        return 1u;
    }
    if (distance_c < distance_a && distance_c < distance_b) {
        return 2u;
    }
    return 0u;
}

fn continuous_newton_iteration(
    iteration: u32,
    previous_residual: f32,
    residual: f32,
) -> f32 {
    if (iteration == 0u) {
        return 0.0;
    }
    let threshold_log = log(1e-6);
    let previous_log = log(max(previous_residual, 1e-30));
    let current_log = log(max(residual, 1e-30));
    let denominator = max(previous_log - current_log, 1e-20);
    let fraction = clamp(
        (previous_log - threshold_log) / denominator,
        0.0,
        1.0,
    );
    return f32(iteration - 1u) + fraction;
}

fn iterate_newton(initial: vec2<f32>) -> NewtonResult {
    var z = initial;
    var previous_residual = 1e30;
    let requested = u32(clamp(u.dynamics_hi.z, 1.0, 50000.0));
    let max_iterations = min(requested, 2048u);
    for (var i = 0u; i < 2048u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        let z2 = complex_mul(z, z);
        let z3 = complex_mul(z2, z);
        let polynomial = z3 - vec2<f32>(1.0, 0.0);
        let residual = length(polynomial);
        if (residual <= 1e-6) {
            return NewtonResult(
                true,
                nearest_newton_root(z),
                i,
                continuous_newton_iteration(i, previous_residual, residual),
                residual,
                z,
            );
        }
        let derivative = 3.0 * z2;
        if (dot(derivative, derivative) <= 1e-18) {
            return NewtonResult(false, 0u, i, f32(i), residual, z);
        }
        previous_residual = residual;
        z = z - complex_div(polynomial, derivative);
        if (any(abs(z) > vec2<f32>(1e18))) {
            return NewtonResult(false, 0u, i + 1u, f32(i + 1u), 1e18, z);
        }
    }
    let z2 = complex_mul(z, z);
    let polynomial = complex_mul(z2, z) - vec2<f32>(1.0, 0.0);
    let residual = length(polynomial);
    if (residual <= 1e-6) {
        return NewtonResult(
            true,
            nearest_newton_root(z),
            max_iterations,
            continuous_newton_iteration(max_iterations, previous_residual, residual),
            residual,
            z,
        );
    }
    return NewtonResult(false, 0u, max_iterations, f32(max_iterations), residual, z);
}

fn iterate_newton_ds(initial: Ds2) -> NewtonResult {
    var z = initial;
    var previous_residual = 1e30;
    let requested = u32(clamp(u.dynamics_hi.z, 1.0, 50000.0));
    let max_iterations = min(requested, 2048u);
    for (var i = 0u; i < 2048u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        let z2 = ds_complex_mul(z, z);
        let z3 = ds_complex_mul(z2, z);
        let polynomial = Ds2(ds_sub(z3.x, Ds(1.0, 0.0)), z3.y);
        let polynomial_approx = ds2_approx(polynomial);
        let residual = length(polynomial_approx);
        if (residual <= 1e-6) {
            let approximate = ds2_approx(z);
            return NewtonResult(
                true,
                nearest_newton_root(approximate),
                i,
                continuous_newton_iteration(i, previous_residual, residual),
                residual,
                approximate,
            );
        }
        let derivative = Ds2(
            ds_mul(Ds(3.0, 0.0), z2.x),
            ds_mul(Ds(3.0, 0.0), z2.y),
        );
        let derivative_approx = ds2_approx(derivative);
        if (dot(derivative_approx, derivative_approx) <= 1e-18) {
            return NewtonResult(false, 0u, i, f32(i), residual, ds2_approx(z));
        }
        previous_residual = residual;
        z = ds2_sub(z, ds_complex_div(polynomial, derivative));
        let approximate = ds2_approx(z);
        if (any(abs(approximate) > vec2<f32>(1e18))) {
            return NewtonResult(false, 0u, i + 1u, f32(i + 1u), 1e18, approximate);
        }
    }
    let z2 = ds_complex_mul(z, z);
    let z3 = ds_complex_mul(z2, z);
    let polynomial = Ds2(ds_sub(z3.x, Ds(1.0, 0.0)), z3.y);
    let residual = length(ds2_approx(polynomial));
    let approximate = ds2_approx(z);
    if (residual <= 1e-6) {
        return NewtonResult(
            true,
            nearest_newton_root(approximate),
            max_iterations,
            continuous_newton_iteration(max_iterations, previous_residual, residual),
            residual,
            approximate,
        );
    }
    return NewtonResult(
        false,
        0u,
        max_iterations,
        f32(max_iterations),
        residual,
        approximate,
    );
}

fn scaled_normalize(value: ScaledComplex) -> ScaledComplex {
    let size = max(abs(value.mantissa.x), abs(value.mantissa.y));
    if (size == 0.0) {
        return ScaledComplex(vec2<f32>(0.0), 0);
    }
    let shift = i32(floor(log2(size)));
    return ScaledComplex(
        value.mantissa * exp2(-f32(shift)),
        value.exponent + shift,
    );
}

fn scaled_add(a: ScaledComplex, b: ScaledComplex) -> ScaledComplex {
    if (all(a.mantissa == vec2<f32>(0.0))) { return b; }
    if (all(b.mantissa == vec2<f32>(0.0))) { return a; }
    if (a.exponent >= b.exponent) {
        let difference = min(a.exponent - b.exponent, 150);
        return scaled_normalize(ScaledComplex(
            a.mantissa + b.mantissa * exp2(-f32(difference)),
            a.exponent,
        ));
    }
    let difference = min(b.exponent - a.exponent, 150);
    return scaled_normalize(ScaledComplex(
        a.mantissa * exp2(-f32(difference)) + b.mantissa,
        b.exponent,
    ));
}

fn scaled_complex_mul(a: ScaledComplex, b: ScaledComplex) -> ScaledComplex {
    return scaled_normalize(ScaledComplex(
        complex_mul(a.mantissa, b.mantissa),
        a.exponent + b.exponent,
    ));
}

fn scaled_mul_plain(a: ScaledComplex, b: vec2<f32>) -> ScaledComplex {
    return scaled_normalize(ScaledComplex(complex_mul(a.mantissa, b), a.exponent));
}

fn scaled_to_f32(a: ScaledComplex) -> vec2<f32> {
    if (a.exponent < -126) { return vec2<f32>(0.0); }
    if (a.exponent > 120) {
        return a.mantissa * 1e20;
    }
    return a.mantissa * exp2(f32(a.exponent));
}

fn reference_value(index: u32) -> vec2<f32> {
    let point = reference_orbit[index];
    return point.z_hi + point.z_lo;
}

fn iterate_f32(world: vec2<f32>, julia: bool) -> EscapeResult {
    var z = select(vec2<f32>(0.0), world, julia);
    let c = select(world, u.dynamics_hi.xy, julia);
    let max_iterations = u32(clamp(u.dynamics_hi.z, 1.0, 50000.0));
    var escaped = false;
    var iteration = 0u;
    for (var i = 0u; i < 50000u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        z = complex_square(z) + c;
        iteration = i + 1u;
        if (dot(z, z) > u.dynamics_hi.w) {
            escaped = true;
            break;
        }
    }
    return EscapeResult(escaped, iteration, z);
}

fn continue_f32(
    initial_z: vec2<f32>,
    c: vec2<f32>,
    first_iteration: u32,
    max_iterations: u32,
) -> EscapeResult {
    var z = initial_z;
    for (var i = first_iteration; i < 50000u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        z = complex_square(z) + c;
        if (dot(z, z) > u.dynamics_hi.w) {
            return EscapeResult(true, i + 1u, z);
        }
    }
    return EscapeResult(false, max_iterations, z);
}

fn iterate_ds(centre: Ds2, local_offset: vec2<f32>, julia: bool) -> EscapeResult {
    let zero = Ds2(Ds(0.0, 0.0), Ds(0.0, 0.0));
    let julia_c = Ds2(
        Ds(u.dynamics_hi.x, u.dynamics_lo.x),
        Ds(u.dynamics_hi.y, u.dynamics_lo.y),
    );
    // Keep the small, pixel-varying part separate from the large reference
    // coordinates. This centred recurrence prevents a WebGPU backend from
    // rounding adjacent pixels onto the reference centre even if compensated
    // additions are optimized aggressively.
    var reference_z = zero;
    var reference_c = centre;
    var delta_z = vec2<f32>(0.0);
    var delta_c = local_offset;
    if (julia) {
        reference_z = centre;
        reference_c = julia_c;
        delta_z = local_offset;
        delta_c = vec2<f32>(0.0);
    }

    let max_iterations = u32(clamp(u.dynamics_hi.z, 1.0, 50000.0));
    var escaped = false;
    var iteration = 0u;
    var approximate = ds2_approx(reference_z) + delta_z;
    for (var i = 0u; i < 50000u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        let reference_before = ds2_approx(reference_z);
        delta_z = 2.0 * complex_mul(reference_before, delta_z)
            + complex_square(delta_z)
            + delta_c;
        let square = ds_complex_square(reference_z);
        reference_z = Ds2(
            ds_add(square.x, reference_c.x),
            ds_add(square.y, reference_c.y),
        );
        approximate = ds2_approx(reference_z) + delta_z;

        // A centred delta is accurate while it stays small. Promote this
        // fragment to a full per-pixel DS orbit before a large delta or
        // cancellation can erase structure. By this point neighbouring
        // orbits have separated enough that the promotion cannot collapse
        // them back to the original f32 coordinate grid.
        let reference_size = length(ds2_approx(reference_z));
        let delta_size = length(delta_z);
        let cancellation_scale = max(reference_size, delta_size);
        let delta_is_large = delta_size > 0.03125 * (1.0 + reference_size);
        let cancellation = cancellation_scale > 1e-4
            && length(approximate) < 0.01 * cancellation_scale;
        if (delta_is_large || cancellation) {
            reference_z = ds2_add_f32(reference_z, delta_z);
            reference_c = ds2_add_f32(reference_c, delta_c);
            delta_z = vec2<f32>(0.0);
            delta_c = vec2<f32>(0.0);
            approximate = ds2_approx(reference_z);
        }
        iteration = i + 1u;
        if (dot(approximate, approximate) > u.dynamics_hi.w) {
            escaped = true;
            break;
        }
    }
    return EscapeResult(escaped, iteration, approximate);
}

fn iterate_perturbation(
    centre: Ds2,
    local: vec2<f32>,
    local_offset: vec2<f32>,
    julia: bool,
) -> EscapeResult {
    let max_iterations = u32(clamp(u.dynamics_hi.z, 1.0, 50000.0));
    let reference_len = u32(max(u.deep.w, 1.0));
    let available_iterations = min(max_iterations, reference_len - 1u);
    let pixel_delta = scaled_normalize(ScaledComplex(
        local * u.deep.y,
        i32(u.deep.z),
    ));
    var delta_z = ScaledComplex(vec2<f32>(0.0), 0);
    var delta_c = pixel_delta;
    if (julia) {
        delta_z = pixel_delta;
        delta_c = ScaledComplex(vec2<f32>(0.0), 0);
    }

    var approximate = reference_value(0u);
    for (var i = 0u; i < 50000u; i = i + 1u) {
        if (i >= available_iterations) { break; }
        let reference_before = reference_value(i);
        let linear = scaled_mul_plain(delta_z, 2.0 * reference_before);
        let quadratic = scaled_complex_mul(delta_z, delta_z);
        delta_z = scaled_add(scaled_add(linear, quadratic), delta_c);
        approximate = reference_value(i + 1u) + scaled_to_f32(delta_z);
        if (dot(approximate, approximate) > u.dynamics_hi.w) {
            return EscapeResult(true, i + 1u, approximate);
        }
    }

    if (available_iterations < max_iterations && u.deep.x < 1.5) {
        // An escaping reference can end before a nearby fragment. At the
        // initial handoff the proven DS renderer is still a safe per-fragment
        // fallback; deeper operation will replace this with orbit rebasing.
        return iterate_ds(centre, local_offset, julia);
    }
    if (available_iterations < max_iterations) {
        // Beyond the handoff the absolute pixel coordinate no longer fits in
        // DS. Continue from the already-separated perturbed state instead of
        // restarting from a collapsed coordinate. Automatic reference
        // rebasing will ultimately replace this conservative visual tail.
        let c = select(u.view_hi.xy, u.dynamics_hi.xy, julia);
        return continue_f32(approximate, c, available_iterations, max_iterations);
    }
    return EscapeResult(false, max_iterations, approximate);
}

fn palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.47, 0.49, 0.52);
    let b = vec3<f32>(0.42, 0.39, 0.36);
    let c = vec3<f32>(1.00, 0.82, 0.68);
    let d = vec3<f32>(0.06, 0.18, 0.36);
    return a + b * cos(6.2831853 * (c * t + d));
}

fn newton_colour(result: NewtonResult, detail: bool) -> vec3<f32> {
    if (!result.converged) {
        let warning = clamp(log2(1.0 + result.residual) * 0.025, 0.0, 0.18);
        return vec3<f32>(0.025 + warning, 0.031, 0.043);
    }
    var root_colour = vec3<f32>(0.20, 0.86, 0.92);
    if (result.root == 1u) {
        root_colour = vec3<f32>(0.98, 0.43, 0.30);
    } else if (result.root == 2u) {
        root_colour = vec3<f32>(0.96, 0.79, 0.30);
    }
    if (u.display.z < 0.5) {
        return root_colour;
    }
    let speed = exp(-0.055 * result.continuous_iteration);
    if (!detail) {
        return root_colour * (0.24 + 0.76 * speed);
    }
    let convergence = palette(0.043 * result.continuous_iteration + u.display.y);
    return mix(convergence, root_colour, 0.28) * (0.38 + 0.62 * speed);
}

fn grid(world: vec2<f32>, scale: f32) -> f32 {
    let decade = pow(10.0, floor(log2(max(scale, 1e-18)) / log2(10.0)));
    let cell = max(decade, 1e-18);
    let q = abs(fract(world / cell + 0.5) - 0.5) / fwidth(world / cell);
    return 1.0 - min(min(q.x, q.y), 1.0);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Top of the viewport maps to positive imaginary values.
    let local = vec2<f32>((in.uv.x * 2.0 - 1.0) * u.view_hi.w,
                         (in.uv.y * 2.0 - 1.0));
    let centre_ds = Ds2(
        Ds(u.view_hi.x, u.view_lo.x),
        Ds(u.view_hi.y, u.view_lo.y),
    );
    let scale_ds = Ds(u.view_hi.z, u.view_lo.z);
    let local_offset = local * ds_approx(scale_ds);
    let world_ds = Ds2(
        ds_add(centre_ds.x, ds_mul(Ds(local.x, 0.0), scale_ds)),
        ds_add(centre_ds.y, ds_mul(Ds(local.y, 0.0), scale_ds)),
    );
    let world_ds_approx = ds2_approx(world_ds);
    let world_f32 = u.view_hi.xy + local * u.view_hi.z;

    let julia = u.display.x > 0.5;
    let double_single = u.view_lo.w > 0.5;
    let perturbation = u.deep.x > 0.5;
    let newton = u.dynamics_lo.z > 0.5;
    if (newton) {
        var result: NewtonResult;
        if (double_single) {
            result = iterate_newton_ds(world_ds);
        } else {
            result = iterate_newton(world_f32);
        }
        var colour = newton_colour(result, u.display.x > 0.5);
        if (u.display.w > 0.5) {
            colour = mix(
                colour,
                vec3<f32>(0.78, 0.81, 0.87),
                grid(world_ds_approx, ds_approx(scale_ds)) * 0.12,
            );
        }
        return vec4<f32>(colour, 1.0);
    }
    var result: EscapeResult;
    var world = world_f32;
    if (perturbation) {
        result = iterate_perturbation(centre_ds, local, local_offset, julia);
        world = world_ds_approx;
    } else if (double_single) {
        result = iterate_ds(centre_ds, local_offset, julia);
        world = world_ds_approx;
    } else {
        result = iterate_f32(world_f32, julia);
    }

    var colour = vec3<f32>(0.025, 0.031, 0.043);
    if (result.escaped) {
        var value = f32(result.iteration);
        if (u.display.z > 0.5) {
            let magnitude_squared = max(dot(result.z, result.z), 1.000001);
            let log_zn = 0.5 * log(magnitude_squared);
            value = value + 1.0 - log2(max(log_zn, 1e-6));
        }
        let t = 0.035 * value + u.display.y;
        colour = palette(t);
        colour *= 0.82 + 0.18 * cos(6.2831853 * fract(value * 0.05));
    } else {
        colour = vec3<f32>(0.025, 0.040, 0.058);
    }

    if (u.display.w > 0.5) {
        colour = mix(
            colour,
            vec3<f32>(0.72, 0.76, 0.82),
            grid(world, ds_approx(scale_ds)) * 0.12,
        );
    }
    return vec4<f32>(colour, 1.0);
}
