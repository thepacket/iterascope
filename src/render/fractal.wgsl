struct Uniforms {
    // Centre xy and half-height z, high words. w = aspect ratio.
    view_hi: vec4<f32>,
    // Matching low words. w = 1 for double-single rendering.
    view_lo: vec4<f32>,
    // Julia c high words, maximum iterations, bailout squared.
    dynamics_hi: vec4<f32>,
    // Julia c low words; zw reserved.
    dynamics_lo: vec4<f32>,
    // x = 0 parameter plane / 1 dynamical plane, y = palette phase,
    // z = smooth colouring, w = show grid.
    display: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

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

fn palette(t: f32) -> vec3<f32> {
    let a = vec3<f32>(0.47, 0.49, 0.52);
    let b = vec3<f32>(0.42, 0.39, 0.36);
    let c = vec3<f32>(1.00, 0.82, 0.68);
    let d = vec3<f32>(0.06, 0.18, 0.36);
    return a + b * cos(6.2831853 * (c * t + d));
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
    var result: EscapeResult;
    var world = world_f32;
    if (double_single) {
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
