struct Uniforms {
    // Centre xy and half-height z, high words. w = aspect ratio.
    view_hi: vec4<f32>,
    // Matching low words. w = 1 for double-single rendering.
    view_lo: vec4<f32>,
    // Julia c high words, maximum iterations, bailout squared.
    dynamics_hi: vec4<f32>,
    // Julia c low words; z = family code (see FAMILY_* below);
    // w = 1 to shade bounded orbits by their minimum modulus.
    dynamics_lo: vec4<f32>,
    // x = 0 parameter plane / 1 dynamical plane, y = palette phase,
    // z = smooth colouring, w = show grid.
    display: vec4<f32>,
    // x = perturbation enabled, y = scale mantissa, z = base-2 scale
    // exponent, w = number of uploaded reference points.
    deep: vec4<f32>,
    // Family parameters: x = degree d / p, y = Nova relaxation R,
    // z = Mandelbox scale, w = Mandelbox minimum radius.
    family_a: vec4<f32>,
    // x = Mandelbox fixed radius, y = Lyapunov sequence bits (bitcast u32),
    // z = Lyapunov sequence length, w = 1 when this pane is a dynamical or
    // detail plane (z0 = pixel) rather than a parameter plane (c = pixel).
    family_b: vec4<f32>,
    // Four copies of 1.0 uploaded at runtime so the shader compiler cannot
    // prove their value. Multiplying consecutive rounded intermediates by
    // *different* components blocks the fast-math reassociation and common-
    // factor extraction that otherwise collapse double-single arithmetic back
    // to f32 (Metal compiles with fast math enabled).
    numerics: vec4<f32>,
    // xy = reference point relative to the view centre, in the same local
    // units as the fragment coordinate (x spans ±aspect, y spans ±1) — the
    // reference orbit may be chosen away from the centre so it lives longer.
    reference: vec4<f32>,
};

// Family codes. They must match `FractalFamily::shader_flag` in family.rs.
const FAMILY_QUADRATIC: u32 = 0u;
const FAMILY_NEWTON: u32 = 1u;
const FAMILY_MULTIBROT: u32 = 2u;
const FAMILY_TRICORN: u32 = 3u;
const FAMILY_PERPENDICULAR_MANDELBROT: u32 = 4u;
const FAMILY_BURNING_SHIP: u32 = 5u;
const FAMILY_PERPENDICULAR_BURNING_SHIP: u32 = 6u;
const FAMILY_CELTIC: u32 = 7u;
const FAMILY_PERPENDICULAR_CELTIC: u32 = 8u;
const FAMILY_BUFFALO: u32 = 9u;
const FAMILY_PERPENDICULAR_BUFFALO: u32 = 10u;
const FAMILY_LAMBDA: u32 = 11u;
const FAMILY_PHOENIX: u32 = 12u;
const FAMILY_MANOWAR: u32 = 13u;
const FAMILY_SPIDER: u32 = 14u;
const FAMILY_MAGNET_ONE: u32 = 15u;
const FAMILY_MAGNET_TWO: u32 = 16u;
const FAMILY_EXPONENTIAL: u32 = 17u;
const FAMILY_SINE: u32 = 18u;
const FAMILY_COSINE: u32 = 19u;
const FAMILY_COLLATZ: u32 = 20u;
const FAMILY_LYAPUNOV: u32 = 21u;
const FAMILY_NOVA: u32 = 22u;
const FAMILY_BARNSLEY_ONE: u32 = 23u;
const FAMILY_BARNSLEY_TWO: u32 = 24u;
const FAMILY_MANDELBOX: u32 = 25u;
// Not a family: selects the double-single self-test fragment.
const FAMILY_SELF_TEST: u32 = 99u;

// Escape and convergence thresholds shared with family.rs.
const EXP_ESCAPE: f32 = 50.0;
const TRIG_ESCAPE: f32 = 50.0;
const COLLATZ_IMAG_ESCAPE: f32 = 20.0;
const COLLATZ_RADIUS_ESCAPE_SQUARED: f32 = 1e12;
const MAGNET_CONVERGENCE: f32 = 1e-4;
const NOVA_CONVERGENCE: f32 = 1e-5;
const NOVA_ESCAPE_SQUARED: f32 = 1e12;
const MANDELBOX_BAILOUT_FACTOR: f32 = 4.0;
const PI: f32 = 3.14159265358979;

const RESULT_BOUNDED: u32 = 0u;
const RESULT_ESCAPED: u32 = 1u;
const RESULT_CONVERGED: u32 = 2u;

@group(0) @binding(0) var<uniform> u: Uniforms;

struct ReferencePoint {
    z_hi: vec2<f32>,
    z_lo: vec2<f32>,
};

@group(0) @binding(1) var<storage, read> reference_orbit: array<ReferencePoint>;

// Colour stage settings; see `Colouring::gpu_words` in colouring.rs.
// Most layers one image may composite; must match `colouring::MAX_LAYERS`.
const MAX_LAYERS: u32 = 8u;

// One layer's colour stage. See `Layer` / `ColouringSide::gpu_words` in
// colouring.rs.
struct LayerColouring {
    // Outside (escaped or converged) algorithm:
    // a = [algorithm, transfer, density, offset],
    // b = [smooth, decomposition sectors, stripe frequency, trap shape],
    // c = [trap centre x, trap centre y, trap size, shading],
    // d = [solid colour rgb, unused].
    outside_a: vec4<f32>,
    outside_b: vec4<f32>,
    outside_c: vec4<f32>,
    outside_d: vec4<f32>,
    // Inside (bounded) algorithm, same layout.
    inside_a: vec4<f32>,
    inside_b: vec4<f32>,
    inside_c: vec4<f32>,
    inside_d: vec4<f32>,
    // x = merge mode, y = opacity, z = layer needs orbit traps,
    // w = layer needs the stripe accumulator.
    blend: vec4<f32>,
};

struct ColouringUniforms {
    // x = visible layer count, y = ln(pixel height in world units),
    // z = gradient table entries per layer.
    header: vec4<f32>,
    // Union over layers of the accumulators needed: x = orbit trap,
    // y = triangle inequality, z = stripes, w = derivative.
    needs: vec4<f32>,
    // Visible layers, bottom first.
    layers: array<LayerColouring, 8>,
};
@group(0) @binding(2) var<uniform> colouring: ColouringUniforms;
// Rasterised cyclic gradients, RGBA; one table of header.z entries per
// visible layer, bottom first.
@group(0) @binding(3) var<storage, read> gradient_table: array<vec4<f32>>;

fn layer_count() -> u32 {
    return clamp(u32(colouring.header.x + 0.5), 1u, MAX_LAYERS);
}

// Whether the iteration loops gather orbit statistics (orbit traps,
// triangle-inequality and stripe averages, the derivative). The renderer
// compiles the shader twice — see `render::shader_source_variant` — and uses
// the plain variant whenever no selected algorithm needs them, so the
// accumulators cost nothing in the default configuration.
const ORBIT_STATS: bool = true; // ORBIT_STATS_MARKER

const ALGORITHM_ITERATION: u32 = 0u;
const ALGORITHM_DECOMPOSITION: u32 = 1u;
const ALGORITHM_TRIANGLE: u32 = 2u;
const ALGORITHM_STRIPES: u32 = 3u;
const ALGORITHM_DISTANCE: u32 = 4u;
const ALGORITHM_TRAP: u32 = 5u;
const ALGORITHM_SOLID: u32 = 6u;

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

struct ScaledComplex {
    mantissa: vec2<f32>,
    exponent: i32,
};

// Statistics gathered along an orbit for the colour stage. Every iteration
// loop feeds its f32 orbit values through `observe`; which accumulators are
// live is decided per frame by `colouring.needs`.
struct OrbitStats {
    // Best (smallest) trap metric per layer, for the layer's outside and
    // inside trap shapes.
    trap_outside: array<f32, 8>,
    trap_inside: array<f32, 8>,
    // Triangle-inequality terms: running sum, the most recent term and the
    // number of terms, so the last term can be blended by the fractional
    // iteration count. Shared across layers (depends only on the family).
    tia_sum: f32,
    tia_last: f32,
    tia_count: f32,
    // Stripe-average terms per layer (the frequency is a layer setting).
    stripe_sum: array<f32, 8>,
    stripe_last: array<f32, 8>,
    stripe_count: f32,
    // Derivative of z_n with respect to the pixel coordinate, for distance
    // estimation; scaled so it survives deep zooms. Shared across layers.
    derivative: ScaledComplex,
};

struct EscapeResult {
    escaped: bool,
    iteration: u32,
    z: vec2<f32>,
    stats: OrbitStats,
};

struct GenericState {
    z: vec2<f32>,
    z_prev: vec2<f32>,
    c: vec2<f32>,
};

struct GenericStateDs {
    z: Ds2,
    z_prev: Ds2,
    c: Ds2,
};

struct GenericResult {
    kind: u32,
    iteration: u32,
    z: vec2<f32>,
    // Continuous iteration count (escape- or convergence-time smoothed when
    // smooth colouring is enabled).
    value: f32,
    stats: OrbitStats,
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

// Values the compiler must treat as unknown; every component is 1.0. The
// compensated identities below multiply each rounded intermediate by one of
// them, rotating through the four so that no two adjacent terms share a
// factor. A single factor is not enough: with fast math the optimizer pulls
// the common factor out, reassociates the remaining additions and evaluates
// `a - (s - v) + (b - v)` as `(a + b)(1 - k)`, which is exactly zero at run
// time. The GPU self-test in render/mod.rs checks this protection.
fn k1(value: f32) -> f32 { return value * u.numerics.x; }
fn k2(value: f32) -> f32 { return value * u.numerics.y; }
fn k3(value: f32) -> f32 { return value * u.numerics.z; }
fn k4(value: f32) -> f32 { return value * u.numerics.w; }

fn ds_normalize(hi: f32, lo: f32) -> Ds {
    let sum = k1(hi + lo);
    let recovered = k2(sum - hi);
    let error = k3(lo - recovered);
    return Ds(sum, error);
}

fn ds_add(a: Ds, b: Ds) -> Ds {
    let sum = k1(a.hi + b.hi);
    let virtual_b = k2(sum - a.hi);
    let virtual_a = k3(sum - virtual_b);
    let a_error = k4(a.hi - virtual_a);
    let b_error = k1(b.hi - virtual_b);
    let high_error = k2(a_error + b_error);
    let low_error = k3(a.lo + b.lo);
    let error = k4(high_error + low_error);
    return ds_normalize(sum, error);
}

fn ds_sub(a: Ds, b: Ds) -> Ds {
    return ds_add(a, Ds(-b.hi, -b.lo));
}

fn ds_mul(a: Ds, b: Ds) -> Ds {
    let product = k1(a.hi * b.hi);
    // fma recovers the product's rounding residue; its first factor is
    // wrapped so the optimizer cannot match it against `product`.
    let residue = k2(fma(k3(a.hi), b.hi, -product));
    let cross = k4(k1(a.hi * b.lo) + k2(a.lo * b.hi));
    let low = k3(a.lo * b.lo);
    let error = k4(k1(residue + cross) + low);
    return ds_normalize(product, error);
}

fn ds_div(a: Ds, b: Ds) -> Ds {
    let estimate = k1(a.hi / b.hi);
    let remainder = ds_sub(a, ds_mul(b, Ds(estimate, 0.0)));
    let correction = k2(k3(remainder.hi + remainder.lo) / b.hi);
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

// Power-of-two scaling through the float exponent field; no transcendental
// functions. `exp2i` is exact for -126 <= n <= 127 and returns 0 below.
fn exp2i(n: i32) -> f32 {
    if (n < -126) { return 0.0; }
    return bitcast<f32>(u32(min(n, 127) + 127) << 23u);
}

// Exponent of the largest power of two not exceeding `size` (size > 0).
fn floor_log2(size: f32) -> i32 {
    return i32((bitcast<u32>(size) >> 23u) & 255u) - 127;
}

fn scaled_normalize(value: ScaledComplex) -> ScaledComplex {
    let size = max(abs(value.mantissa.x), abs(value.mantissa.y));
    if (size == 0.0) {
        return ScaledComplex(vec2<f32>(0.0), 0);
    }
    let shift = floor_log2(size);
    return ScaledComplex(
        value.mantissa * exp2i(-shift),
        value.exponent + shift,
    );
}

fn scaled_add(raw_a: ScaledComplex, raw_b: ScaledComplex) -> ScaledComplex {
    let a = scaled_normalize(raw_a);
    let b = scaled_normalize(raw_b);
    if (all(a.mantissa == vec2<f32>(0.0))) { return b; }
    if (all(b.mantissa == vec2<f32>(0.0))) { return a; }
    if (a.exponent >= b.exponent) {
        return scaled_normalize(ScaledComplex(
            a.mantissa + b.mantissa * exp2i(b.exponent - a.exponent),
            a.exponent,
        ));
    }
    return scaled_normalize(ScaledComplex(
        a.mantissa * exp2i(a.exponent - b.exponent) + b.mantissa,
        b.exponent,
    ));
}

// Products and scalings leave their result unnormalized: mantissas of
// normalized operands stay within a few binades, f32 has range to spare, and
// every consumer that needs a canonical form (additions, conversions and the
// end of each perturbation step) normalizes itself.
fn scaled_complex_mul(a: ScaledComplex, b: ScaledComplex) -> ScaledComplex {
    return ScaledComplex(complex_mul(a.mantissa, b.mantissa), a.exponent + b.exponent);
}

fn scaled_mul_plain(a: ScaledComplex, b: vec2<f32>) -> ScaledComplex {
    return ScaledComplex(complex_mul(a.mantissa, b), a.exponent);
}

fn scaled_to_f32(value: ScaledComplex) -> vec2<f32> {
    let a = scaled_normalize(value);
    if (a.exponent > 120) {
        return a.mantissa * 1e20;
    }
    return a.mantissa * exp2i(a.exponent);
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
    var stats = stats_new(julia);
    for (var i = 0u; i < 50000u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        let z_prev = z;
        z = complex_square(z) + c;
        iteration = i + 1u;
        observe(&stats, FAMILY_QUADRATIC, z, z_prev, c, iteration, julia);
        let magnitude_squared = dot(z, z);
        if (magnitude_squared > u.dynamics_hi.w) {
            escaped = true;
            break;
        }
    }
    return EscapeResult(escaped, iteration, z, stats);
}

fn continue_f32(
    initial_z: vec2<f32>,
    c: vec2<f32>,
    first_iteration: u32,
    max_iterations: u32,
    initial_stats: OrbitStats,
    julia: bool,
) -> EscapeResult {
    var z = initial_z;
    var stats = initial_stats;
    for (var i = first_iteration; i < 50000u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        let z_prev = z;
        z = complex_square(z) + c;
        observe(&stats, FAMILY_QUADRATIC, z, z_prev, c, i + 1u, julia);
        let magnitude_squared = dot(z, z);
        if (magnitude_squared > u.dynamics_hi.w) {
            return EscapeResult(true, i + 1u, z, stats);
        }
    }
    return EscapeResult(false, max_iterations, z, stats);
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
    var stats = stats_new(julia);
    var approximate = ds2_approx(reference_z) + delta_z;
    for (var i = 0u; i < 50000u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        let z_prev = approximate;
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
        observe(&stats, FAMILY_QUADRATIC, approximate, z_prev,
            ds2_approx(reference_c) + delta_c, iteration, julia);
        let magnitude_squared = dot(approximate, approximate);
        if (magnitude_squared > u.dynamics_hi.w) {
            escaped = true;
            break;
        }
    }
    return EscapeResult(escaped, iteration, approximate, stats);
}

fn iterate_perturbation(
    centre: Ds2,
    local: vec2<f32>,
    local_offset: vec2<f32>,
    reference_c: vec2<f32>,
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

    let c = select(reference_c + scaled_to_f32(delta_c), u.dynamics_hi.xy, julia);
    var approximate = reference_value(0u) + scaled_to_f32(delta_z);
    var stats = stats_new(julia);
    for (var i = 0u; i < 50000u; i = i + 1u) {
        if (i >= available_iterations) { break; }
        let reference_before = reference_value(i);
        let z_prev = approximate;
        let linear = scaled_mul_plain(delta_z, 2.0 * reference_before);
        let quadratic = scaled_complex_mul(delta_z, delta_z);
        delta_z = scaled_add(scaled_add(linear, quadratic), delta_c);
        approximate = reference_value(i + 1u) + scaled_to_f32(delta_z);
        observe(&stats, FAMILY_QUADRATIC, approximate, z_prev, c, i + 1u, julia);
        let magnitude_squared = dot(approximate, approximate);
        if (magnitude_squared > u.dynamics_hi.w) {
            return EscapeResult(true, i + 1u, approximate, stats);
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
        return continue_f32(approximate, c, available_iterations, max_iterations, stats, julia);
    }
    return EscapeResult(false, max_iterations, approximate, stats);
}

// ---------------------------------------------------------------------------
// Generic escape-time families.
//
// `family_step_f32` and `family_step_ds` are transcriptions of
// `family::step` in family.rs; that CPU f64 implementation is the definition
// of record for every map below.
// ---------------------------------------------------------------------------

fn complex_exp(z: vec2<f32>) -> vec2<f32> {
    let magnitude = exp(z.x);
    return vec2<f32>(magnitude * cos(z.y), magnitude * sin(z.y));
}

fn complex_sin(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

fn complex_cos(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}

fn complex_pow(z: vec2<f32>, degree: u32) -> vec2<f32> {
    var result = z;
    for (var i = 1u; i < 8u; i = i + 1u) {
        if (i >= degree) { break; }
        result = complex_mul(result, z);
    }
    return result;
}

fn box_fold(value: f32) -> f32 {
    if (value > 1.0) { return 2.0 - value; }
    if (value < -1.0) { return -2.0 - value; }
    return value;
}

fn family_degree() -> u32 {
    return u32(clamp(u.family_a.x, 2.0, 8.0));
}

fn family_initial_f32(family: u32, world: vec2<f32>, dynamical: bool) -> GenericState {
    if (dynamical) {
        var z_prev = vec2<f32>(0.0);
        if (family == FAMILY_MANOWAR) { z_prev = world; }
        return GenericState(world, z_prev, u.dynamics_hi.xy);
    }
    let c = world;
    var z = vec2<f32>(0.0);
    var z_prev = vec2<f32>(0.0);
    switch family {
        case FAMILY_LAMBDA: { z = vec2<f32>(0.5, 0.0); }
        case FAMILY_MANOWAR: { z = c; z_prev = c; }
        case FAMILY_SINE: { z = vec2<f32>(0.5 * PI, 0.0); }
        case FAMILY_NOVA: { z = vec2<f32>(1.0, 0.0); }
        case FAMILY_BARNSLEY_ONE, FAMILY_BARNSLEY_TWO: { z = c; }
        default: {}
    }
    return GenericState(z, z_prev, c);
}

fn family_step_f32(family: u32, state: GenericState) -> GenericState {
    let z = state.z;
    let c = state.c;
    let x = z.x;
    let y = z.y;
    let one = vec2<f32>(1.0, 0.0);
    var z_prev = state.z_prev;
    var next_c = c;
    var next = z;
    switch family {
        case FAMILY_MULTIBROT: { next = complex_pow(z, family_degree()) + c; }
        case FAMILY_TRICORN: { next = vec2<f32>(x * x - y * y, -2.0 * x * y) + c; }
        case FAMILY_PERPENDICULAR_MANDELBROT: {
            next = vec2<f32>(x * x - y * y, -2.0 * abs(x) * y) + c;
        }
        case FAMILY_BURNING_SHIP: { next = vec2<f32>(x * x - y * y, 2.0 * abs(x * y)) + c; }
        case FAMILY_PERPENDICULAR_BURNING_SHIP: {
            next = vec2<f32>(x * x - y * y, -2.0 * x * abs(y)) + c;
        }
        case FAMILY_CELTIC: { next = vec2<f32>(abs(x * x - y * y), 2.0 * x * y) + c; }
        case FAMILY_PERPENDICULAR_CELTIC: {
            next = vec2<f32>(abs(x * x - y * y), -2.0 * abs(x) * y) + c;
        }
        case FAMILY_BUFFALO: { next = vec2<f32>(abs(x * x - y * y), 2.0 * abs(x * y)) + c; }
        case FAMILY_PERPENDICULAR_BUFFALO: {
            next = vec2<f32>(abs(x * x - y * y), -2.0 * x * abs(y)) + c;
        }
        case FAMILY_LAMBDA: { next = complex_mul(c, complex_mul(z, one - z)); }
        case FAMILY_PHOENIX: {
            next = complex_square(z) + vec2<f32>(c.x, 0.0) + c.y * z_prev;
            z_prev = z;
        }
        case FAMILY_MANOWAR: {
            next = complex_square(z) + z_prev + c;
            z_prev = z;
        }
        case FAMILY_SPIDER: {
            next = complex_square(z) + c;
            next_c = 0.5 * c + next;
        }
        case FAMILY_MAGNET_ONE: {
            let numerator = complex_square(z) + c - one;
            let denominator = 2.0 * z + c - vec2<f32>(2.0, 0.0);
            next = complex_square(complex_div(numerator, denominator));
        }
        case FAMILY_MAGNET_TWO: {
            let c1 = c - one;
            let c2 = c - vec2<f32>(2.0, 0.0);
            let c12 = complex_mul(c1, c2);
            let z2 = complex_square(z);
            let z3 = complex_mul(z2, z);
            let numerator = z3 + 3.0 * complex_mul(c1, z) + c12;
            let denominator = 3.0 * z2 + 3.0 * complex_mul(c2, z) + c12 + one;
            next = complex_square(complex_div(numerator, denominator));
        }
        case FAMILY_EXPONENTIAL: { next = complex_mul(c, complex_exp(z)); }
        case FAMILY_SINE: { next = complex_mul(c, complex_sin(z)); }
        case FAMILY_COSINE: { next = complex_mul(c, complex_cos(z)); }
        case FAMILY_COLLATZ: {
            let cosine = complex_cos(PI * z);
            let term = complex_mul(vec2<f32>(2.0, 0.0) + 5.0 * z, cosine);
            next = 0.25 * (vec2<f32>(2.0, 0.0) + 7.0 * z - term);
        }
        case FAMILY_NOVA, FAMILY_NEWTON: {
            var p = family_degree();
            var relaxation = u.family_a.y;
            if (family == FAMILY_NEWTON) {
                p = 3u;
                relaxation = 1.0;
            }
            let inverse = complex_div(one, complex_pow(z, p - 1u));
            let newton_step = (z - inverse) / f32(p);
            next = z - relaxation * newton_step;
            if (family == FAMILY_NOVA) { next = next + c; }
        }
        case FAMILY_BARNSLEY_ONE: {
            if (x >= 0.0) {
                next = complex_mul(z - one, c);
            } else {
                next = complex_mul(z + one, c);
            }
        }
        case FAMILY_BARNSLEY_TWO: {
            if (x * c.y + c.x * y >= 0.0) {
                next = complex_mul(z - one, c);
            } else {
                next = complex_mul(z + one, c);
            }
        }
        case FAMILY_MANDELBOX: {
            let folded = vec2<f32>(box_fold(x), box_fold(y));
            let radius_squared = dot(folded, folded);
            let min_squared = u.family_a.w * u.family_a.w;
            let fixed_squared = u.family_b.x * u.family_b.x;
            var ball = folded;
            if (radius_squared < min_squared) {
                ball = folded * (fixed_squared / min_squared);
            } else if (radius_squared < fixed_squared) {
                ball = folded * (fixed_squared / radius_squared);
            }
            next = u.family_a.z * ball + c;
        }
        default: { next = complex_square(z) + c; }
    }
    return GenericState(next, z_prev, next_c);
}

// --- double-single helpers for the generic families -----------------------

fn ds_from_f32(value: f32) -> Ds {
    return Ds(value, 0.0);
}

fn ds2_from_f32(value: vec2<f32>) -> Ds2 {
    return Ds2(Ds(value.x, 0.0), Ds(value.y, 0.0));
}

fn ds2_add(a: Ds2, b: Ds2) -> Ds2 {
    return Ds2(ds_add(a.x, b.x), ds_add(a.y, b.y));
}

fn ds_neg(a: Ds) -> Ds {
    return Ds(-a.hi, -a.lo);
}

fn ds_abs(a: Ds) -> Ds {
    if (a.hi < 0.0 || (a.hi == 0.0 && a.lo < 0.0)) {
        return ds_neg(a);
    }
    return a;
}

fn ds_scale(a: Ds, s: f32) -> Ds {
    return ds_mul(a, Ds(s, 0.0));
}

fn ds2_scale(a: Ds2, s: f32) -> Ds2 {
    return Ds2(ds_scale(a.x, s), ds_scale(a.y, s));
}

fn ds2_scale_ds(a: Ds2, s: Ds) -> Ds2 {
    return Ds2(ds_mul(a.x, s), ds_mul(a.y, s));
}

fn ds_complex_pow(z: Ds2, degree: u32) -> Ds2 {
    var result = z;
    for (var i = 1u; i < 8u; i = i + 1u) {
        if (i >= degree) { break; }
        result = ds_complex_mul(result, z);
    }
    return result;
}

fn ds_box_fold(value: Ds) -> Ds {
    if (value.hi > 1.0) { return ds_sub(Ds(2.0, 0.0), value); }
    if (value.hi < -1.0) { return ds_sub(Ds(-2.0, 0.0), value); }
    return value;
}

fn family_initial_ds(family: u32, world: Ds2, dynamical: bool) -> GenericStateDs {
    let zero = Ds2(Ds(0.0, 0.0), Ds(0.0, 0.0));
    if (dynamical) {
        var z_prev = zero;
        if (family == FAMILY_MANOWAR) { z_prev = world; }
        let c = Ds2(
            Ds(u.dynamics_hi.x, u.dynamics_lo.x),
            Ds(u.dynamics_hi.y, u.dynamics_lo.y),
        );
        return GenericStateDs(world, z_prev, c);
    }
    let c = world;
    var z = zero;
    var z_prev = zero;
    switch family {
        case FAMILY_LAMBDA: { z = ds2_from_f32(vec2<f32>(0.5, 0.0)); }
        case FAMILY_MANOWAR: { z = c; z_prev = c; }
        case FAMILY_NOVA: { z = ds2_from_f32(vec2<f32>(1.0, 0.0)); }
        case FAMILY_BARNSLEY_ONE, FAMILY_BARNSLEY_TWO: { z = c; }
        default: {}
    }
    return GenericStateDs(z, z_prev, c);
}

fn family_step_ds(family: u32, state: GenericStateDs) -> GenericStateDs {
    let z = state.z;
    let c = state.c;
    let one = ds2_from_f32(vec2<f32>(1.0, 0.0));
    let two = ds2_from_f32(vec2<f32>(2.0, 0.0));
    var z_prev = state.z_prev;
    var next_c = c;
    var next = z;
    // Shared real products for the absolute-value variants.
    let xx = ds_mul(z.x, z.x);
    let yy = ds_mul(z.y, z.y);
    let xy = ds_mul(z.x, z.y);
    let real_square = ds_sub(xx, yy);
    switch family {
        case FAMILY_MULTIBROT: { next = ds2_add(ds_complex_pow(z, family_degree()), c); }
        case FAMILY_TRICORN: {
            next = ds2_add(Ds2(real_square, ds_scale(xy, -2.0)), c);
        }
        case FAMILY_PERPENDICULAR_MANDELBROT: {
            next = ds2_add(Ds2(real_square, ds_scale(ds_mul(ds_abs(z.x), z.y), -2.0)), c);
        }
        case FAMILY_BURNING_SHIP: {
            next = ds2_add(Ds2(real_square, ds_scale(ds_abs(xy), 2.0)), c);
        }
        case FAMILY_PERPENDICULAR_BURNING_SHIP: {
            next = ds2_add(Ds2(real_square, ds_scale(ds_mul(z.x, ds_abs(z.y)), -2.0)), c);
        }
        case FAMILY_CELTIC: {
            next = ds2_add(Ds2(ds_abs(real_square), ds_scale(xy, 2.0)), c);
        }
        case FAMILY_PERPENDICULAR_CELTIC: {
            next = ds2_add(
                Ds2(ds_abs(real_square), ds_scale(ds_mul(ds_abs(z.x), z.y), -2.0)),
                c,
            );
        }
        case FAMILY_BUFFALO: {
            next = ds2_add(Ds2(ds_abs(real_square), ds_scale(ds_abs(xy), 2.0)), c);
        }
        case FAMILY_PERPENDICULAR_BUFFALO: {
            next = ds2_add(
                Ds2(ds_abs(real_square), ds_scale(ds_mul(z.x, ds_abs(z.y)), -2.0)),
                c,
            );
        }
        case FAMILY_LAMBDA: {
            next = ds_complex_mul(c, ds_complex_mul(z, ds2_sub(one, z)));
        }
        case FAMILY_PHOENIX: {
            let square = ds_complex_square(z);
            next = Ds2(
                ds_add(ds_add(square.x, c.x), ds_mul(c.y, z_prev.x)),
                ds_add(square.y, ds_mul(c.y, z_prev.y)),
            );
            z_prev = z;
        }
        case FAMILY_MANOWAR: {
            next = ds2_add(ds2_add(ds_complex_square(z), z_prev), c);
            z_prev = z;
        }
        case FAMILY_SPIDER: {
            next = ds2_add(ds_complex_square(z), c);
            next_c = ds2_add(ds2_scale(c, 0.5), next);
        }
        case FAMILY_MAGNET_ONE: {
            let numerator = ds2_sub(ds2_add(ds_complex_square(z), c), one);
            let denominator = ds2_sub(ds2_add(ds2_scale(z, 2.0), c), two);
            next = ds_complex_square(ds_complex_div(numerator, denominator));
        }
        case FAMILY_MAGNET_TWO: {
            let c1 = ds2_sub(c, one);
            let c2 = ds2_sub(c, two);
            let c12 = ds_complex_mul(c1, c2);
            let z2 = ds_complex_square(z);
            let z3 = ds_complex_mul(z2, z);
            let numerator = ds2_add(ds2_add(z3, ds2_scale(ds_complex_mul(c1, z), 3.0)), c12);
            let denominator = ds2_add(
                ds2_add(ds2_add(ds2_scale(z2, 3.0), ds2_scale(ds_complex_mul(c2, z), 3.0)), c12),
                one,
            );
            next = ds_complex_square(ds_complex_div(numerator, denominator));
        }
        case FAMILY_NOVA, FAMILY_NEWTON: {
            var p = family_degree();
            var relaxation = u.family_a.y;
            if (family == FAMILY_NEWTON) {
                p = 3u;
                relaxation = 1.0;
            }
            let inverse = ds_complex_div(one, ds_complex_pow(z, p - 1u));
            let newton_step = ds2_scale(ds2_sub(z, inverse), 1.0 / f32(p));
            next = ds2_sub(z, ds2_scale(newton_step, relaxation));
            if (family == FAMILY_NOVA) { next = ds2_add(next, c); }
        }
        case FAMILY_BARNSLEY_ONE: {
            if (ds_approx(z.x) >= 0.0) {
                next = ds_complex_mul(ds2_sub(z, one), c);
            } else {
                next = ds_complex_mul(ds2_add(z, one), c);
            }
        }
        case FAMILY_BARNSLEY_TWO: {
            let test = ds_add(ds_mul(z.x, c.y), ds_mul(c.x, z.y));
            if (ds_approx(test) >= 0.0) {
                next = ds_complex_mul(ds2_sub(z, one), c);
            } else {
                next = ds_complex_mul(ds2_add(z, one), c);
            }
        }
        case FAMILY_MANDELBOX: {
            let folded = Ds2(ds_box_fold(z.x), ds_box_fold(z.y));
            let radius_squared = ds_add(ds_mul(folded.x, folded.x), ds_mul(folded.y, folded.y));
            let min_squared = u.family_a.w * u.family_a.w;
            let fixed_squared = u.family_b.x * u.family_b.x;
            var ball = folded;
            let radius_approx = ds_approx(radius_squared);
            if (radius_approx < min_squared) {
                ball = ds2_scale(folded, fixed_squared / min_squared);
            } else if (radius_approx < fixed_squared) {
                ball = ds2_scale_ds(folded, ds_div(Ds(fixed_squared, 0.0), radius_squared));
            }
            next = ds2_add(ds2_scale(ball, u.family_a.z), c);
        }
        default: { next = ds2_add(ds_complex_square(z), c); }
    }
    return GenericStateDs(next, z_prev, next_c);
}

// --- shared termination and smoothing --------------------------------------

fn family_escape_radius_squared(family: u32) -> f32 {
    if (family == FAMILY_MANDELBOX) {
        return u.dynamics_hi.w * MANDELBOX_BAILOUT_FACTOR * MANDELBOX_BAILOUT_FACTOR;
    }
    if (family == FAMILY_NOVA) {
        return NOVA_ESCAPE_SQUARED;
    }
    if (family == FAMILY_NEWTON) {
        return 1e36;
    }
    return u.dynamics_hi.w;
}

fn family_escaped(family: u32, z: vec2<f32>, radius_squared: f32) -> bool {
    let magnitude_squared = dot(z, z);
    if (magnitude_squared != magnitude_squared) { return true; }
    switch family {
        case FAMILY_EXPONENTIAL: { return z.x > EXP_ESCAPE; }
        case FAMILY_SINE, FAMILY_COSINE: { return abs(z.y) > TRIG_ESCAPE; }
        case FAMILY_COLLATZ: {
            return abs(z.y) > COLLATZ_IMAG_ESCAPE
                || magnitude_squared > COLLATZ_RADIUS_ESCAPE_SQUARED;
        }
        default: { return magnitude_squared > radius_squared; }
    }
}

// Residual whose decay below a threshold marks convergence, or -1 when the
// family has no convergence criterion.
fn family_residual(family: u32, z: vec2<f32>, previous: vec2<f32>) -> f32 {
    switch family {
        case FAMILY_MAGNET_ONE, FAMILY_MAGNET_TWO: { return length(z - vec2<f32>(1.0, 0.0)); }
        case FAMILY_NOVA: { return length(z - previous); }
        case FAMILY_NEWTON: {
            let z2 = complex_mul(z, z);
            return length(complex_mul(z2, z) - vec2<f32>(1.0, 0.0));
        }
        default: { return -1.0; }
    }
}

fn family_convergence_threshold(family: u32) -> f32 {
    if (family == FAMILY_NOVA) { return NOVA_CONVERGENCE; }
    if (family == FAMILY_NEWTON) { return 1e-6; }
    return MAGNET_CONVERGENCE;
}

// Smoothed escape iteration. Polynomial-like maps use the degree-d
// normalisation; linear maps (Barnsley, Mandelbox) use their contraction
// ratio; transcendental maps are left unsmoothed.
fn smooth_escape_value(family: u32, iteration: u32, z: vec2<f32>) -> f32 {
    let value = f32(iteration);
    if (u.display.z < 0.5) { return value; }
    let magnitude_squared = max(dot(z, z), 1.000001);
    switch family {
        // Transcendental maps have no polynomial normalisation and the
        // piecewise-linear Barnsley maps make log-ratio smoothing ill
        // conditioned near |c| = 1, so these keep integer escape times.
        case FAMILY_EXPONENTIAL, FAMILY_SINE, FAMILY_COSINE, FAMILY_COLLATZ,
            FAMILY_BARNSLEY_ONE, FAMILY_BARNSLEY_TWO: { return value; }
        case FAMILY_MANDELBOX: {
            let ratio = max(abs(u.family_a.z), 1.0001);
            let radius = sqrt(family_escape_radius_squared(family));
            return value - log(sqrt(magnitude_squared) / radius) / log(ratio);
        }
        case FAMILY_MULTIBROT: {
            let log_zn = 0.5 * log(magnitude_squared);
            return value + 1.0 - log(max(log_zn, 1e-6)) / log(f32(family_degree()));
        }
        default: {
            let log_zn = 0.5 * log(magnitude_squared);
            return value + 1.0 - log2(max(log_zn, 1e-6));
        }
    }
}

fn smooth_convergence_value(iteration: u32, previous_residual: f32, residual: f32, threshold: f32) -> f32 {
    if (u.display.z < 0.5 || iteration == 0u) { return f32(iteration); }
    let threshold_log = log(threshold);
    let previous_log = log(max(previous_residual, 1e-30));
    let current_log = log(max(residual, 1e-30));
    let denominator = max(previous_log - current_log, 1e-20);
    let fraction = clamp((previous_log - threshold_log) / denominator, 0.0, 1.0);
    return f32(iteration - 1u) + fraction;
}

fn iterate_family_f32(family: u32, world: vec2<f32>, dynamical: bool) -> GenericResult {
    var state = family_initial_f32(family, world, dynamical);
    let max_iterations = u32(clamp(u.dynamics_hi.z, 1.0, 50000.0));
    let radius_squared = family_escape_radius_squared(family);
    let threshold = family_convergence_threshold(family);
    var previous_residual = 1e30;
    var stats = stats_new(dynamical);
    for (var i = 0u; i < 50000u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        let previous = state.z;
        state = family_step_f32(family, state);
        let iteration = i + 1u;
        observe(&stats, family, state.z, previous, state.c, iteration, dynamical);
        if (family_escaped(family, state.z, radius_squared)) {
            return GenericResult(
                RESULT_ESCAPED,
                iteration,
                state.z,
                smooth_escape_value(family, iteration, state.z),
                stats,
            );
        }
        let residual = family_residual(family, state.z, previous);
        if (residual >= 0.0 && residual < threshold) {
            return GenericResult(
                RESULT_CONVERGED,
                iteration,
                state.z,
                smooth_convergence_value(iteration, previous_residual, residual, threshold),
                stats,
            );
        }
        previous_residual = residual;
    }
    return GenericResult(RESULT_BOUNDED, max_iterations, state.z, f32(max_iterations), stats);
}

// Double-single rendering of the generic families.
//
// Like the quadratic `iterate_ds`, this is a centred recurrence: every pixel
// iterates the *view centre's* orbit in double-single (identical for all
// pixels) and carries its own offset as an exact perturbation delta in scaled
// arithmetic, rebasing the delta into the reference once it grows. Adjacent
// pixels therefore never depend on compensated arithmetic resolving their
// tiny coordinate difference — a plain per-pixel DS orbit silently collapses
// to the f32 grid whenever the shader compiler's fast-math reassociation
// defeats the compensated identities, which it does on Metal.
fn iterate_family_ds(
    family: u32,
    centre: Ds2,
    local_offset: vec2<f32>,
    dynamical: bool,
) -> GenericResult {
    var reference = family_initial_ds(family, centre, dynamical);
    let max_iterations = u32(clamp(u.dynamics_hi.z, 1.0, 50000.0));
    let radius_squared = family_escape_radius_squared(family);
    let threshold = family_convergence_threshold(family);
    let pixel_delta = local_offset;

    var deltas: PerturbState_f32;
    if (dynamical) {
        var dz_prev = vec2<f32>(0.0);
        if (family == FAMILY_MANOWAR) { dz_prev = pixel_delta; }
        deltas = PerturbState_f32(pixel_delta, dz_prev, vec2<f32>(0.0), ds2_approx(reference.c));
    } else {
        var dz = vec2<f32>(0.0);
        var dz_prev = vec2<f32>(0.0);
        if (family == FAMILY_MANOWAR || family == FAMILY_BARNSLEY_ONE || family == FAMILY_BARNSLEY_TWO) {
            dz = pixel_delta;
        }
        if (family == FAMILY_MANOWAR) { dz_prev = pixel_delta; }
        deltas = PerturbState_f32(dz, dz_prev, pixel_delta, ds2_approx(reference.c));
    }

    var previous_residual = 1e30;
    var stats = stats_new(dynamical);
    var approximate = ds2_approx(reference.z) + deltas.dz;
    for (var i = 0u; i < 50000u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        let previous = approximate;
        let z = ds2_approx(reference.z);
        let z_prev = ds2_approx(reference.z_prev);
        reference = family_step_ds(family, reference);
        let z_next = ds2_approx(reference.z);
        deltas = perturb_step_f32(family, deltas, z, z_prev, z_next);
        // The reference parameter follows the double-single reference (only
        // the Spider map moves it).
        deltas.c_ref = ds2_approx(reference.c);
        var dz_f32 = deltas.dz;
        approximate = z_next + dz_f32;

        // Promote the pixel to its own reference before the delta grows large
        // or cancellation erases structure (same rule as the quadratic path).
        let reference_size = length(z_next);
        let delta_size = length(dz_f32);
        let cancellation_scale = max(reference_size, delta_size);
        let delta_is_large = delta_size > 0.03125 * (1.0 + reference_size);
        let cancellation = cancellation_scale > 1e-4
            && length(approximate) < 0.01 * cancellation_scale;
        if (delta_is_large || cancellation) {
            reference.z = ds2_add_f32(reference.z, dz_f32);
            reference.z_prev = ds2_add_f32(reference.z_prev, deltas.dz_prev);
            reference.c = ds2_add_f32(reference.c, deltas.dc);
            deltas = PerturbState_f32(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0), ds2_approx(reference.c));
            approximate = ds2_approx(reference.z);
        }

        let iteration = i + 1u;
        observe(&stats, family, approximate, previous, deltas.c_ref + deltas.dc, iteration, dynamical);
        if (family_escaped(family, approximate, radius_squared)) {
            return GenericResult(
                RESULT_ESCAPED,
                iteration,
                approximate,
                smooth_escape_value(family, iteration, approximate),
                stats,
            );
        }
        let residual = family_residual(family, approximate, previous);
        if (residual >= 0.0 && residual < threshold) {
            return GenericResult(
                RESULT_CONVERGED,
                iteration,
                approximate,
                smooth_convergence_value(iteration, previous_residual, residual, threshold),
                stats,
            );
        }
        previous_residual = residual;
    }
    return GenericResult(RESULT_BOUNDED, max_iterations, approximate, f32(max_iterations), stats);
}

// --- Lyapunov plane --------------------------------------------------------

// Lyapunov exponent of x -> r x (1 - x) with r forced by the A/B sequence,
// discarding the first quarter of the iterations. Returns a large positive
// value when the orbit leaves [0, 1]. Mirrors family::lyapunov_exponent.
fn lyapunov_exponent(a: f32, b: f32) -> f32 {
    let iterations = u32(clamp(u.dynamics_hi.z, 1.0, 50000.0));
    let warmup = iterations / 4u;
    let bits = bitcast<u32>(u.family_b.y);
    let length = max(u32(u.family_b.z), 1u);
    var x = 0.5;
    var sum = 0.0;
    var count = 0u;
    for (var i = 0u; i < 50000u; i = i + 1u) {
        if (i >= iterations) { break; }
        let use_b = ((bits >> (i % length)) & 1u) == 1u;
        let r = select(a, b, use_b);
        x = r * x * (1.0 - x);
        if (!(x >= 0.0 && x <= 1.0)) {
            return 1e3;
        }
        if (i >= warmup) {
            sum = sum + log(max(abs(r * (1.0 - 2.0 * x)), 1e-30));
            count = count + 1u;
        }
    }
    if (count == 0u) { return 0.0; }
    return sum / f32(count);
}

// The Lyapunov plane has no orbit result: stable regions (negative
// exponent) sweep through the gradient by the outside density and offset,
// chaotic regions darken with the exponent.
fn lyapunov_colour(exponent: f32) -> vec3<f32> {
    if (exponent >= 1e3) {
        return vec3<f32>(0.02, 0.024, 0.03);
    }
    if (exponent < 0.0) {
        let strength = clamp(-exponent / 1.5, 0.0, 1.0);
        let outside_a = colouring.layers[0].outside_a;
        let position = transfer(u32(outside_a.y + 0.5), -exponent)
            * outside_a.z / 0.035 * 0.18 + outside_a.w;
        let base = vec3<f32>(0.06, 0.05, 0.04);
        return mix(base, gradient_colour(position, 0u), pow(strength, 0.65));
    }
    let strength = clamp(exponent / 1.2, 0.0, 1.0);
    let near_zero = vec3<f32>(0.30, 0.36, 0.52);
    let deep_chaos = vec3<f32>(0.03, 0.05, 0.11);
    return mix(near_zero, deep_chaos, pow(strength, 0.5));
}

// ---------------------------------------------------------------------------
// Perturbation rendering for the generic families.
//
// Every family below has an exact recurrence for the difference
// δ_{n+1} = f(Z_n + δ_n) − f(Z_n) between a pixel orbit and the
// arbitrary-precision reference orbit Z. The deltas live in scaled
// (mantissa, base-2 exponent) arithmetic so they survive far below f32 range;
// reference values are f32 projections of the uploaded orbit.
// ---------------------------------------------------------------------------

struct ScaledReal {
    mantissa: f32,
    exponent: i32,
};


fn sr_normalize(value: ScaledReal) -> ScaledReal {
    let size = abs(value.mantissa);
    if (size == 0.0) { return ScaledReal(0.0, 0); }
    let shift = floor_log2(size);
    return ScaledReal(value.mantissa * exp2i(-shift), value.exponent + shift);
}

fn sr_from_f32(value: f32) -> ScaledReal {
    return sr_normalize(ScaledReal(value, 0));
}

fn sr_to_f32(raw: ScaledReal) -> f32 {
    let value = sr_normalize(raw);
    if (value.exponent > 120) { return value.mantissa * 1e20; }
    return value.mantissa * exp2i(value.exponent);
}

fn sr_add(raw_a: ScaledReal, raw_b: ScaledReal) -> ScaledReal {
    let a = sr_normalize(raw_a);
    let b = sr_normalize(raw_b);
    if (a.mantissa == 0.0) { return b; }
    if (b.mantissa == 0.0) { return a; }
    if (a.exponent >= b.exponent) {
        return sr_normalize(ScaledReal(a.mantissa + b.mantissa * exp2i(b.exponent - a.exponent), a.exponent));
    }
    return sr_normalize(ScaledReal(a.mantissa * exp2i(a.exponent - b.exponent) + b.mantissa, b.exponent));
}

fn sr_neg(a: ScaledReal) -> ScaledReal {
    return ScaledReal(-a.mantissa, a.exponent);
}

fn sr_mul(a: ScaledReal, b: ScaledReal) -> ScaledReal {
    return ScaledReal(a.mantissa * b.mantissa, a.exponent + b.exponent);
}

fn sr_scale(a: ScaledReal, s: f32) -> ScaledReal {
    return ScaledReal(a.mantissa * s, a.exponent);
}

fn sc_x(a: ScaledComplex) -> ScaledReal {
    return ScaledReal(a.mantissa.x, a.exponent);
}

fn sc_y(a: ScaledComplex) -> ScaledReal {
    return ScaledReal(a.mantissa.y, a.exponent);
}

fn sc_from_reals(raw_x: ScaledReal, raw_y: ScaledReal) -> ScaledComplex {
    let x = sr_normalize(raw_x);
    let y = sr_normalize(raw_y);
    if (x.mantissa == 0.0) { return scaled_normalize(ScaledComplex(vec2<f32>(0.0, y.mantissa), y.exponent)); }
    if (y.mantissa == 0.0) { return scaled_normalize(ScaledComplex(vec2<f32>(x.mantissa, 0.0), x.exponent)); }
    if (x.exponent >= y.exponent) {
        return scaled_normalize(ScaledComplex(
            vec2<f32>(x.mantissa, y.mantissa * exp2i(y.exponent - x.exponent)),
            x.exponent,
        ));
    }
    return scaled_normalize(ScaledComplex(
        vec2<f32>(x.mantissa * exp2i(x.exponent - y.exponent), y.mantissa),
        y.exponent,
    ));
}

fn sc_from_f32(value: vec2<f32>) -> ScaledComplex {
    return scaled_normalize(ScaledComplex(value, 0));
}

fn sc_zero() -> ScaledComplex {
    return ScaledComplex(vec2<f32>(0.0), 0);
}

fn sc_neg(a: ScaledComplex) -> ScaledComplex {
    return ScaledComplex(-a.mantissa, a.exponent);
}

fn sc_scale(a: ScaledComplex, s: f32) -> ScaledComplex {
    return ScaledComplex(a.mantissa * s, a.exponent);
}

// Scaled real times an f32 complex value.
fn sr_times_complex(r: ScaledReal, z: vec2<f32>) -> ScaledComplex {
    return ScaledComplex(z * r.mantissa, r.exponent);
}

// Scaled real times a scaled complex value.
fn sc_mul_real(a: ScaledComplex, r: ScaledReal) -> ScaledComplex {
    return ScaledComplex(a.mantissa * r.mantissa, a.exponent + r.exponent);
}

fn sc_sub(a: ScaledComplex, b: ScaledComplex) -> ScaledComplex {
    return scaled_add(a, sc_neg(b));
}

fn complex_inverse(z: vec2<f32>) -> vec2<f32> {
    let denominator = max(dot(z, z), 1e-38);
    return vec2<f32>(z.x, -z.y) / denominator;
}

// |a + d| − |a| evaluated without cancellation. When |d| is far below |a|
// the result is exactly ±d and stays scaled; only when the pixel crosses the
// fold does the O(1) difference appear.



fn box_branch(value: f32) -> i32 {
    if (value > 1.0) { return 1; }
    if (value < -1.0) { return -1; }
    return 0;
}



// Continues a generic orbit in plain f32 from a known state; used after the
// reference orbit ends beyond the double-single handoff.
fn continue_family_f32(
    family: u32,
    initial: GenericState,
    first_iteration: u32,
    max_iterations: u32,
    initial_stats: OrbitStats,
    initial_previous_residual: f32,
    dynamical: bool,
) -> GenericResult {
    var state = initial;
    let radius_squared = family_escape_radius_squared(family);
    let threshold = family_convergence_threshold(family);
    var previous_residual = initial_previous_residual;
    var stats = initial_stats;
    for (var i = first_iteration; i < 50000u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        let previous = state.z;
        state = family_step_f32(family, state);
        let iteration = i + 1u;
        observe(&stats, family, state.z, previous, state.c, iteration, dynamical);
        if (family_escaped(family, state.z, radius_squared)) {
            return GenericResult(RESULT_ESCAPED, iteration, state.z,
                smooth_escape_value(family, iteration, state.z), stats);
        }
        let residual = family_residual(family, state.z, previous);
        if (residual >= 0.0 && residual < threshold) {
            return GenericResult(RESULT_CONVERGED, iteration, state.z,
                smooth_convergence_value(iteration, previous_residual, residual, threshold), stats);
        }
        previous_residual = residual;
    }
    return GenericResult(RESULT_BOUNDED, max_iterations, state.z, f32(max_iterations), stats);
}

// Plain-f32 delta helpers for the `_f32` instantiation.
fn fc_zero() -> vec2<f32> { return vec2<f32>(0.0); }
fn fc_from_f32(value: vec2<f32>) -> vec2<f32> { return value; }
fn fc_neg(a: vec2<f32>) -> vec2<f32> { return -a; }
fn fc_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return a - b; }
fn fc_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return a + b; }
fn fc_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return complex_mul(a, b); }
fn fc_mul_plain(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return complex_mul(a, b); }
fn fc_scale(a: vec2<f32>, s: f32) -> vec2<f32> { return a * s; }
fn fc_to_f32(a: vec2<f32>) -> vec2<f32> { return a; }
fn fc_x(a: vec2<f32>) -> f32 { return a.x; }
fn fc_y(a: vec2<f32>) -> f32 { return a.y; }
fn fc_from_reals(x: f32, y: f32) -> vec2<f32> { return vec2<f32>(x, y); }
fn fc_mul_real(a: vec2<f32>, r: f32) -> vec2<f32> { return a * r; }
fn fr_times_complex(r: f32, z: vec2<f32>) -> vec2<f32> { return z * r; }
fn fc_normalize(a: vec2<f32>) -> vec2<f32> { return a; }
fn fc_pixel_delta(local: vec2<f32>) -> vec2<f32> { return local * u.deep.y * exp2i(i32(u.deep.z)); }
fn fr_zero() -> f32 { return 0.0; }
fn fr_from_f32(value: f32) -> f32 { return value; }
fn fr_to_f32(value: f32) -> f32 { return value; }
fn fr_add(a: f32, b: f32) -> f32 { return a + b; }
fn fr_neg(a: f32) -> f32 { return -a; }
fn fr_mul(a: f32, b: f32) -> f32 { return a * b; }
fn fr_scale(a: f32, s: f32) -> f32 { return a * s; }

fn sr_zero() -> ScaledReal {
    return ScaledReal(0.0, 0);
}

fn sc_pixel_delta(local: vec2<f32>) -> ScaledComplex {
    return scaled_normalize(ScaledComplex(local * u.deep.y, i32(u.deep.z)));
}

// ---------------------------------------------------------------------------
// Delta algebra template.
//
// Everything between the BEGIN/END markers is instantiated twice by
// `render::shader_source()` at shader-load time: once with scaled
// (mantissa, exponent) deltas for arbitrary-precision depth (suffix
// `_scaled`, types ScaledComplex/ScaledReal, `sc_*`/`sr_*` helpers) and once
// with plain f32 deltas for the f64-reference range below the handoff
// (suffix `_f32`, types vec2<f32>/f32, `fc_*`/`fr_*` helpers), where deltas
// never leave f32 range and the plain version runs several times faster.
// `DC`/`DR` are the complex/real delta types, `dc_*`/`dr_*` the operations
// and the double-underscore-T suffix marks names to be instantiated. The raw file does not compile as-is.
// ---------------------------------------------------------------------------
// BEGIN DELTA TEMPLATE
struct PerturbState__T {
    dz: DC,
    dz_prev: DC,
    dc: DC,
    // Reference parameter (drifts for the Spider map).
    c_ref: vec2<f32>,
};

fn diffabs__T(reference: f32, delta: DR) -> DR {
    let delta_f32 = dr_to_f32(delta);
    if (reference >= 0.0) {
        if (reference + delta_f32 >= 0.0) { return delta; }
        return dr_add(dr_from_f32(-2.0 * reference), dr_neg(delta));
    }
    if (reference + delta_f32 < 0.0) { return dr_neg(delta); }
    return dr_add(dr_from_f32(2.0 * reference), delta);
}

// 2 Z δ + δ².
fn quadratic_delta__T(z: vec2<f32>, delta: DC) -> DC {
    return dc_add(dc_mul_plain(delta, 2.0 * z), dc_mul(delta, delta));
}

// Σ_{k=1}^{d} C(d, k) Z^{d−k} δ^k  =  (Z + δ)^d − Z^d.
fn binomial_delta__T(z: vec2<f32>, delta: DC, degree: u32) -> DC {
    // Horner form in δ: Σ C(d,k) Z^(d−k) δ^k = δ (c₁Z^(d−1) + δ (c₂Z^(d−2) + ... )).
    // Evaluated from the highest power down so only one running power of Z
    // and one multiplication by δ per term are needed.
    var accumulator = dc_zero();
    var coefficient = 1.0;
    // C(d, d) = 1 for the δ^d term; build coefficients downward.
    var z_power = vec2<f32>(1.0, 0.0);
    for (var k = 0u; k < 8u; k = k + 1u) {
        if (k >= degree) { break; }
        // term index t = degree - k (from δ^d down to δ^1), coefficient C(d, t)
        let t = degree - k;
        accumulator = dc_add(
            dc_mul(accumulator, delta),
            dc_mul_plain(dc_from_f32(z_power), vec2<f32>(coefficient, 0.0)),
        );
        // Next (lower) term: multiply Z power, update C(d, t-1) = C(d, t) * t / (d - t + 1)
        z_power = complex_mul(z_power, z);
        coefficient = coefficient * f32(t) / f32(degree - t + 1u);
    }
    return dc_mul(accumulator, delta);
}

// Component-wise box-fold difference box(X + dx) − box(X).
fn box_fold_delta__T(reference: f32, delta: DR) -> DR {
    let pixel = reference + dr_to_f32(delta);
    let reference_branch = box_branch(reference);
    if (box_branch(pixel) == reference_branch) {
        if (reference_branch == 0) { return delta; }
        return dr_neg(delta);
    }
    return dr_from_f32(box_fold(pixel) - box_fold(reference));
}

fn perturb_step__T(
    family: u32,
    state: PerturbState__T,
    z: vec2<f32>,
    z_prev: vec2<f32>,
    z_next: vec2<f32>,
) -> PerturbState__T {
    let delta = state.dz;
    let dc = state.dc;
    let c = state.c_ref;
    let x = z.x;
    let y = z.y;
    let dx = dc_x(delta);
    let dy = dc_y(delta);
    var next = dc_zero();
    var next_prev = state.dz_prev;
    var next_dc = dc;
    var next_c = c;
    switch family {
        case FAMILY_MULTIBROT: {
            next = dc_add(binomial_delta__T(z, delta, family_degree()), dc);
        }
        case FAMILY_TRICORN, FAMILY_PERPENDICULAR_MANDELBROT, FAMILY_BURNING_SHIP,
            FAMILY_PERPENDICULAR_BURNING_SHIP, FAMILY_CELTIC, FAMILY_PERPENDICULAR_CELTIC,
            FAMILY_BUFFALO, FAMILY_PERPENDICULAR_BUFFALO: {
            // Δ(x² − y²) and Δ(xy) without cancellation.
            let real_delta = dr_add(
                dr_add(dr_scale(dx, 2.0 * x), dr_mul(dx, dx)),
                dr_neg(dr_add(dr_scale(dy, 2.0 * y), dr_mul(dy, dy))),
            );
            let product_delta = dr_add(dr_add(dr_scale(dy, x), dr_scale(dx, y)), dr_mul(dx, dy));
            var real_part = real_delta;
            if (family == FAMILY_CELTIC || family == FAMILY_PERPENDICULAR_CELTIC
                || family == FAMILY_BUFFALO || family == FAMILY_PERPENDICULAR_BUFFALO) {
                real_part = diffabs__T(x * x - y * y, real_delta);
            }
            var imaginary_part = dr_zero();
            if (family == FAMILY_TRICORN) {
                imaginary_part = dr_scale(product_delta, -2.0);
            } else if (family == FAMILY_CELTIC) {
                imaginary_part = dr_scale(product_delta, 2.0);
            } else if (family == FAMILY_BURNING_SHIP || family == FAMILY_BUFFALO) {
                imaginary_part = dr_scale(diffabs__T(x * y, product_delta), 2.0);
            } else if (family == FAMILY_PERPENDICULAR_MANDELBROT || family == FAMILY_PERPENDICULAR_CELTIC) {
                // Δ(|x| y) = diffabs__T(X, dx) (Y + dy) + |X| dy
                let abs_x_delta = diffabs__T(x, dx);
                imaginary_part = dr_scale(
                    dr_add(dr_add(dr_scale(abs_x_delta, y), dr_mul(abs_x_delta, dy)), dr_scale(dy, abs(x))),
                    -2.0,
                );
            } else {
                // Perpendicular Burning Ship / Buffalo: Δ(x |y|) = X dY + dx |Y| + dx dY
                let abs_y_delta = diffabs__T(y, dy);
                imaginary_part = dr_scale(
                    dr_add(dr_add(dr_scale(abs_y_delta, x), dr_scale(dx, abs(y))), dr_mul(dx, abs_y_delta)),
                    -2.0,
                );
            }
            next = dc_add(dc_from_reals(real_part, imaginary_part), dc);
        }
        case FAMILY_LAMBDA: {
            // f = λ (z − z²):  Δ = Λ w + δλ ((Z − Z²) + w),  w = δ(1 − 2Z) − δ².
            let w = dc_sub(dc_mul_plain(delta, vec2<f32>(1.0, 0.0) - 2.0 * z), dc_mul(delta, delta));
            let base = z - complex_square(z);
            next = dc_add(
                dc_add(dc_mul_plain(w, c), dc_mul_plain(dc, base)),
                dc_mul(dc, w),
            );
        }
        case FAMILY_PHOENIX: {
            let dp = dc_from_reals(dc_x(dc), dr_zero());
            let dq = dc_y(dc);
            next = dc_add(quadratic_delta__T(z, delta), dp);
            next = dc_add(next, dc_scale(state.dz_prev, c.y));
            next = dc_add(next, dr_times_complex(dq, z_prev));
            next = dc_add(next, dc_mul_real(state.dz_prev, dq));
            next_prev = delta;
        }
        case FAMILY_MANOWAR: {
            next = dc_add(dc_add(quadratic_delta__T(z, delta), state.dz_prev), dc);
            next_prev = delta;
        }
        case FAMILY_SPIDER: {
            next = dc_add(quadratic_delta__T(z, delta), dc);
            next_dc = dc_add(dc_scale(dc, 0.5), next);
            next_c = 0.5 * c + z_next;
        }
        case FAMILY_MAGNET_ONE: {
            let one = vec2<f32>(1.0, 0.0);
            let numerator = complex_square(z) + c - one;
            let denominator = 2.0 * z + c - vec2<f32>(2.0, 0.0);
            let numerator_delta = dc_add(quadratic_delta__T(z, delta), dc);
            let denominator_delta = dc_add(dc_scale(delta, 2.0), dc);
            let pixel_denominator = denominator + dc_to_f32(denominator_delta);
            let g = complex_div(numerator, denominator);
            let quotient_delta = dc_sub(
                dc_mul_plain(numerator_delta, denominator),
                dc_mul_plain(denominator_delta, numerator),
            );
            let dg = dc_mul_plain(
                quotient_delta,
                complex_inverse(complex_mul(denominator, pixel_denominator)),
            );
            next = quadratic_delta__T(g, dg);
        }
        case FAMILY_MAGNET_TWO: {
            let one = vec2<f32>(1.0, 0.0);
            let c1 = c - one;
            let c2 = c - vec2<f32>(2.0, 0.0);
            let c12 = complex_mul(c1, c2);
            let z2 = complex_square(z);
            let numerator = complex_mul(z2, z) + 3.0 * complex_mul(c1, z) + c12;
            let denominator = 3.0 * z2 + 3.0 * complex_mul(c2, z) + c12 + one;
            let delta2 = dc_mul(delta, delta);
            let delta3 = dc_mul(delta2, delta);
            // Terms shared by numerator and denominator: δc (3Z + 2C − 3) + 3 δc δ + δc².
            let shared_terms = dc_add(
                dc_add(
                    dc_mul_plain(dc, 3.0 * z + 2.0 * c - vec2<f32>(3.0, 0.0)),
                    dc_scale(dc_mul(dc, delta), 3.0),
                ),
                dc_mul(dc, dc),
            );
            let numerator_delta = dc_add(
                dc_add(
                    dc_add(dc_mul_plain(delta, 3.0 * z2 + 3.0 * c1), dc_scale(dc_mul_plain(delta2, z), 3.0)),
                    delta3,
                ),
                shared_terms,
            );
            let denominator_delta = dc_add(
                dc_add(dc_mul_plain(delta, 6.0 * z + 3.0 * c2), dc_scale(delta2, 3.0)),
                shared_terms,
            );
            let pixel_denominator = denominator + dc_to_f32(denominator_delta);
            let g = complex_div(numerator, denominator);
            let quotient_delta = dc_sub(
                dc_mul_plain(numerator_delta, denominator),
                dc_mul_plain(denominator_delta, numerator),
            );
            let dg = dc_mul_plain(
                quotient_delta,
                complex_inverse(complex_mul(denominator, pixel_denominator)),
            );
            next = quadratic_delta__T(g, dg);
        }
        case FAMILY_NOVA, FAMILY_NEWTON: {
            var p = family_degree();
            var relaxation = u.family_a.y;
            if (family == FAMILY_NEWTON) {
                p = 3u;
                relaxation = 1.0;
            }
            // (Z+δ)^(1−p) − Z^(1−p) = −B / (Z^(p−1) (Z+δ)^(p−1)).
            let b = binomial_delta__T(z, delta, p - 1u);
            let z_power = complex_pow(z, p - 1u);
            let pixel_power = complex_pow(z + dc_to_f32(delta), p - 1u);
            let inverse_delta = dc_neg(dc_mul_plain(
                b,
                complex_inverse(complex_mul(z_power, pixel_power)),
            ));
            next = dc_add(
                dc_scale(delta, 1.0 - relaxation / f32(p)),
                dc_scale(inverse_delta, relaxation / f32(p)),
            );
            if (family == FAMILY_NOVA) {
                next = dc_add(next, dc);
            }
        }
        case FAMILY_BARNSLEY_ONE, FAMILY_BARNSLEY_TWO: {
            let one = vec2<f32>(1.0, 0.0);
            let pixel_z = z + dc_to_f32(delta);
            let pixel_c = c + dc_to_f32(dc);
            var reference_branch = x >= 0.0;
            var pixel_branch = pixel_z.x >= 0.0;
            if (family == FAMILY_BARNSLEY_TWO) {
                reference_branch = x * c.y + c.x * y >= 0.0;
                pixel_branch = pixel_z.x * pixel_c.y + pixel_c.x * pixel_z.y >= 0.0;
            }
            let reference_base = select(z + one, z - one, reference_branch);
            let pixel_base = select(z + one, z - one, pixel_branch);
            // (base + δ)(C + δc) − reference_base C
            next = dc_add(
                dc_add(
                    dc_from_f32(complex_mul(pixel_base - reference_base, c)),
                    dc_mul_plain(delta, c),
                ),
                dc_add(dc_mul_plain(dc, pixel_base), dc_mul(dc, delta)),
            );
        }
        case FAMILY_MANDELBOX: {
            let folded = vec2<f32>(box_fold(x), box_fold(y));
            let fold_delta = dc_from_reals(box_fold_delta__T(x, dx), box_fold_delta__T(y, dy));
            let radius_squared = dot(folded, folded);
            // Δ(r²) = 2 F·ΔF + |ΔF|²
            let radius_delta = dr_add(
                dr_add(dr_scale(dc_x(fold_delta), 2.0 * folded.x), dr_scale(dc_y(fold_delta), 2.0 * folded.y)),
                dr_add(dr_mul(dc_x(fold_delta), dc_x(fold_delta)), dr_mul(dc_y(fold_delta), dc_y(fold_delta))),
            );
            let pixel_radius_squared = radius_squared + dr_to_f32(radius_delta);
            let min_squared = u.family_a.w * u.family_a.w;
            let fixed_squared = u.family_b.x * u.family_b.x;
            var reference_branch = 2;
            if (radius_squared < min_squared) { reference_branch = 0; }
            else if (radius_squared < fixed_squared) { reference_branch = 1; }
            var pixel_branch = 2;
            if (pixel_radius_squared < min_squared) { pixel_branch = 0; }
            else if (pixel_radius_squared < fixed_squared) { pixel_branch = 1; }
            var ball_delta = fold_delta;
            if (reference_branch != pixel_branch) {
                let pixel_folded = folded + dc_to_f32(fold_delta);
                var pixel_ball = pixel_folded;
                if (pixel_branch == 0) { pixel_ball = pixel_folded * (fixed_squared / min_squared); }
                else if (pixel_branch == 1) { pixel_ball = pixel_folded * (fixed_squared / pixel_radius_squared); }
                var reference_ball = folded;
                if (reference_branch == 0) { reference_ball = folded * (fixed_squared / min_squared); }
                else if (reference_branch == 1) { reference_ball = folded * (fixed_squared / radius_squared); }
                ball_delta = dc_from_f32(pixel_ball - reference_ball);
            } else if (reference_branch == 0) {
                ball_delta = dc_scale(fold_delta, fixed_squared / min_squared);
            } else if (reference_branch == 1) {
                // fixed² [ΔF r² − F Δ(r²)] / (r² r²_pixel)
                let numerator = dc_sub(
                    dc_scale(fold_delta, radius_squared),
                    dr_times_complex(radius_delta, folded),
                );
                ball_delta = dc_scale(numerator, fixed_squared / max(radius_squared * pixel_radius_squared, 1e-38));
            }
            next = dc_add(dc_scale(ball_delta, u.family_a.z), dc);
        }
        default: {
            next = dc_add(quadratic_delta__T(z, delta), dc);
        }
    }
    return PerturbState__T(
        dc_normalize(next),
        dc_normalize(next_prev),
        dc_normalize(next_dc),
        next_c,
    );
}

fn iterate_family_perturbation__T(
    family: u32,
    reference_c: vec2<f32>,
    local: vec2<f32>,
    dynamical: bool,
) -> GenericResult {
    let max_iterations = u32(clamp(u.dynamics_hi.z, 1.0, 50000.0));
    let reference_len = u32(max(u.deep.w, 1.0));
    let available_iterations = min(max_iterations, reference_len - 1u);
    let pixel_delta = dc_pixel_delta(local);
    let radius_squared = family_escape_radius_squared(family);
    let threshold = family_convergence_threshold(family);

    var state: PerturbState__T;
    let z0 = reference_value(0u);
    if (dynamical) {
        var dz_prev = dc_zero();
        if (family == FAMILY_MANOWAR) { dz_prev = pixel_delta; }
        state = PerturbState__T(pixel_delta, dz_prev, dc_zero(), u.dynamics_hi.xy);
    } else {
        var dz = dc_zero();
        var dz_prev = dc_zero();
        if (family == FAMILY_MANOWAR || family == FAMILY_BARNSLEY_ONE || family == FAMILY_BARNSLEY_TWO) {
            dz = pixel_delta;
        }
        if (family == FAMILY_MANOWAR) { dz_prev = pixel_delta; }
        state = PerturbState__T(dz, dz_prev, pixel_delta, reference_c);
    }
    var initial_prev = vec2<f32>(0.0);
    if (family == FAMILY_MANOWAR) { initial_prev = z0; }

    var approximate = z0 + dc_to_f32(state.dz);
    var previous_residual = 1e30;
    var stats = stats_new(dynamical);
    var z_prev = initial_prev;
    var z = z0;
    for (var i = 0u; i < 50000u; i = i + 1u) {
        if (i >= available_iterations) { break; }
        let z_next = reference_value(i + 1u);
        let previous = approximate;
        state = perturb_step__T(family, state, z, z_prev, z_next);
        z_prev = z;
        z = z_next;
        approximate = z_next + dc_to_f32(state.dz);
        let iteration = i + 1u;
        observe(&stats, family, approximate, previous, state.c_ref + dc_to_f32(state.dc), iteration, dynamical);
        if (family_escaped(family, approximate, radius_squared)) {
            return GenericResult(RESULT_ESCAPED, iteration, approximate,
                smooth_escape_value(family, iteration, approximate), stats);
        }
        let residual = family_residual(family, approximate, previous);
        if (residual >= 0.0 && residual < threshold) {
            return GenericResult(RESULT_CONVERGED, iteration, approximate,
                smooth_convergence_value(iteration, previous_residual, residual, threshold), stats);
        }
        previous_residual = residual;
    }

    if (available_iterations < max_iterations) {
        // The reference escaped or converged before this pixel did. Continue
        // from the separated perturbed state in plain f32: by now the pixel's
        // orbit differs from the reference by far more than f32 resolution.
        // Automatic reference rebasing will ultimately replace this tail.
        let tail = GenericState(
            approximate,
            z_prev + dc_to_f32(state.dz_prev),
            state.c_ref + dc_to_f32(state.dc),
        );
        return continue_family_f32(family, tail, available_iterations, max_iterations, stats, previous_residual, dynamical);
    }
    return GenericResult(RESULT_BOUNDED, max_iterations, approximate, f32(max_iterations), stats);
}
// END DELTA TEMPLATE

// ---------------------------------------------------------------------------
// Colour stage.
//
// `observe` runs once per iteration in every loop above and keeps the
// accumulators the selected algorithms need; `colour_stage` reduces a result
// to a gradient position through the outside or inside algorithm and looks
// it up in the rasterised gradient.
// ---------------------------------------------------------------------------

fn stats_new(dynamical: bool) -> OrbitStats {
    // In the dynamical plane the pixel is z0, so dz0/dz0 = 1; in the
    // parameter plane the families with derivative support start from a
    // constant, so dz0/dc = 0 and the first step contributes ∂f/∂c.
    var stats: OrbitStats;
    for (var layer = 0u; layer < MAX_LAYERS; layer = layer + 1u) {
        stats.trap_outside[layer] = 1e30;
        stats.trap_inside[layer] = 1e30;
        stats.stripe_sum[layer] = 0.0;
        stats.stripe_last[layer] = 0.0;
    }
    stats.tia_sum = 0.0;
    stats.tia_last = 0.0;
    stats.tia_count = 0.0;
    stats.stripe_count = 0.0;
    stats.derivative = ScaledComplex(vec2<f32>(select(0.0, 1.0, dynamical), 0.0), 0);
    return stats;
}

// Metric of z against a trap shape. Point traps accumulate the squared
// distance (no square root per iteration); `trap_distance` undoes that.
fn trap_metric(shape: u32, centre: vec2<f32>, size: f32, z: vec2<f32>) -> f32 {
    let d = z - centre;
    switch shape {
        case 1u: { return min(abs(d.x), abs(d.y)); }
        case 2u: { return abs(length(d) - size); }
        case 3u: { return abs(max(abs(d.x), abs(d.y)) - size); }
        case 4u: { return abs(d.y); }
        case 5u: { return abs(d.x); }
        default: { return dot(d, d); }
    }
}

fn trap_distance(shape: u32, metric: f32) -> f32 {
    if (shape == 0u) { return sqrt(max(metric, 0.0)); }
    return metric;
}

// Whether `derivative_step` knows the family's derivative; other families
// fall back to the iteration count when distance estimation is selected.
fn family_has_derivative(family: u32) -> bool {
    return family == FAMILY_QUADRATIC || family == FAMILY_MULTIBROT || family == FAMILY_LAMBDA;
}

// One step of dz_{n+1} = f'(z_n) dz_n (+ ∂f/∂c in the parameter plane).
fn derivative_step(
    family: u32,
    derivative: ScaledComplex,
    z_prev: vec2<f32>,
    c: vec2<f32>,
    dynamical: bool,
) -> ScaledComplex {
    var multiplier = 2.0 * z_prev;
    var offset = vec2<f32>(1.0, 0.0);
    switch family {
        case FAMILY_MULTIBROT: {
            let degree = family_degree();
            multiplier = f32(degree) * complex_pow(z_prev, degree - 1u);
        }
        case FAMILY_LAMBDA: {
            let one = vec2<f32>(1.0, 0.0);
            multiplier = complex_mul(c, one - 2.0 * z_prev);
            offset = complex_mul(z_prev, one - z_prev);
        }
        default: {}
    }
    // Renormalise every step: the product alone would overflow the f32
    // mantissa within a hundred iterations of a repelling orbit.
    var next = scaled_normalize(scaled_mul_plain(derivative, multiplier));
    if (!dynamical) {
        next = scaled_add(next, ScaledComplex(offset, 0));
    }
    return next;
}

fn observe(
    stats: ptr<function, OrbitStats>,
    family: u32,
    z: vec2<f32>,
    z_prev: vec2<f32>,
    c: vec2<f32>,
    iteration: u32,
    dynamical: bool,
) {
    if (!ORBIT_STATS) { return; }
    let count = layer_count();
    if (colouring.needs.x > 0.5) {
        for (var layer = 0u; layer < MAX_LAYERS; layer = layer + 1u) {
            if (layer >= count) { break; }
            let l = colouring.layers[layer];
            if (l.blend.z < 0.5) { continue; }
            (*stats).trap_outside[layer] = min((*stats).trap_outside[layer], trap_metric(
                u32(l.outside_b.w + 0.5), l.outside_c.xy, l.outside_c.z, z));
            (*stats).trap_inside[layer] = min((*stats).trap_inside[layer], trap_metric(
                u32(l.inside_b.w + 0.5), l.inside_c.xy, l.inside_c.z, z));
        }
    }
    if (colouring.needs.y > 0.5 && iteration > 1u) {
        // |z_{n+1}| lies between ||z_n|^d − |c|| and |z_n|^d + |c| for
        // z_{n+1} = z_n^d + c; record where in that range it fell.
        let degree = family_degree();
        var power = dot(z_prev, z_prev);
        if (degree != 2u) {
            power = pow(max(length(z_prev), 1e-20), f32(degree));
        }
        let c_size = length(c);
        let low = abs(power - c_size);
        let high = power + c_size;
        let span = high - low;
        var term = 0.0;
        if (span > 1e-12) {
            term = clamp((length(z) - low) / span, 0.0, 1.0);
        }
        (*stats).tia_sum = (*stats).tia_sum + term;
        (*stats).tia_last = term;
        (*stats).tia_count = (*stats).tia_count + 1.0;
    }
    if (colouring.needs.z > 0.5) {
        let angle = atan2(z.y, z.x);
        for (var layer = 0u; layer < MAX_LAYERS; layer = layer + 1u) {
            if (layer >= count) { break; }
            let l = colouring.layers[layer];
            if (l.blend.w < 0.5) { continue; }
            // The layer's outside frequency when its outside uses stripes,
            // otherwise its inside frequency.
            let frequency = select(l.inside_b.z, l.outside_b.z,
                u32(l.outside_a.x + 0.5) == ALGORITHM_STRIPES);
            let term = 0.5 + 0.5 * sin(frequency * angle);
            (*stats).stripe_sum[layer] = (*stats).stripe_sum[layer] + term;
            (*stats).stripe_last[layer] = term;
        }
        (*stats).stripe_count = (*stats).stripe_count + 1.0;
    }
    if (colouring.needs.w > 0.5 && family_has_derivative(family)) {
        (*stats).derivative = derivative_step(family, (*stats).derivative, z_prev, c, dynamical);
    }
}

fn gradient_colour(position: f32, layer: u32) -> vec3<f32> {
    let n = max(u32(colouring.header.z + 0.5), 1u);
    let offset = layer * n;
    let scaled = fract(position) * f32(n);
    let base = floor(scaled);
    let i0 = min(u32(base), n - 1u);
    let i1 = (i0 + 1u) % n;
    return mix(
        gradient_table[offset + i0].rgb,
        gradient_table[offset + i1].rgb,
        scaled - base,
    );
}

fn transfer(code: u32, value: f32) -> f32 {
    let v = max(value, 0.0);
    switch code {
        case 1u: { return sqrt(v); }
        case 2u: { return pow(v, 1.0 / 3.0); }
        case 3u: { return log(1.0 + v); }
        default: { return v; }
    }
}

// Weight of an escaped orbit's final term in an average: 1 when the orbit
// only just crossed the escape radius (its neighbour in the next band has
// the same term in full), 0 when it nearly escaped one iteration earlier.
// Derived from the family's own smoothing term, so it matches whatever
// radius and degree the smoothing uses; integer colouring gives 1.
fn escape_band_weight(family: u32, z: vec2<f32>) -> f32 {
    let radius = sqrt(family_escape_radius_squared(family));
    let at_radius = smooth_escape_value(family, 0u, vec2<f32>(radius, 0.0));
    let at_z = smooth_escape_value(family, 0u, z);
    return clamp(1.0 + at_z - at_radius, 0.0, 1.0);
}

// Weight of the final accumulator term for any finished orbit. Convergence
// smoothing already yields a fraction that is 0 when the previous iterate
// had all but converged.
fn final_term_weight(family: u32, result: GenericResult) -> f32 {
    if (result.kind == RESULT_ESCAPED) {
        return escape_band_weight(family, result.z);
    }
    if (result.kind == RESULT_CONVERGED && u.display.z > 0.5) {
        return fract(result.value);
    }
    return 1.0;
}

// Blends the running average of an orbit accumulator with the average that
// excludes the final term, so the colouring stays continuous across
// iteration bands.
fn blended_average(sum: f32, last: f32, count: f32, weight: f32) -> f32 {
    if (count < 1.0) { return 0.0; }
    let all = sum / count;
    if (count < 2.0) { return all; }
    let previous = (sum - last) / (count - 1.0);
    return mix(previous, all, weight);
}

fn colour_side(
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    d: vec4<f32>,
    family: u32,
    result: GenericResult,
    inside: bool,
    layer: u32,
) -> vec3<f32> {
    let algorithm = u32(a.x + 0.5);
    if (algorithm == ALGORITHM_SOLID) {
        return d.rgb;
    }
    var value = result.value;
    switch algorithm {
        case ALGORITHM_DECOMPOSITION: {
            var angle = atan2(result.z.y, result.z.x) / 6.2831853 + 0.5;
            let sectors = b.y;
            if (sectors >= 1.0) {
                angle = floor(angle * sectors) / sectors;
            }
            value = angle;
        }
        case ALGORITHM_TRIANGLE: {
            if (ORBIT_STATS) {
                value = blended_average(result.stats.tia_sum, result.stats.tia_last,
                    result.stats.tia_count, final_term_weight(family, result));
            }
        }
        case ALGORITHM_STRIPES: {
            if (ORBIT_STATS) {
                value = blended_average(result.stats.stripe_sum[layer],
                    result.stats.stripe_last[layer],
                    result.stats.stripe_count, final_term_weight(family, result));
            }
        }
        case ALGORITHM_DISTANCE: {
            if (ORBIT_STATS && family_has_derivative(family) && result.kind == RESULT_ESCAPED) {
                // Exterior distance |z| ln|z| / |dz| expressed in pixels,
                // evaluated in logarithms so deep zooms stay in range.
                let magnitude = max(length(result.z), 1.000001);
                let mantissa = max(length(result.stats.derivative.mantissa), 1e-30);
                let log_derivative = log(mantissa) + f32(result.stats.derivative.exponent) * 0.69314718;
                let log_distance = log(magnitude) + log(max(log(magnitude), 1e-6)) - log_derivative;
                value = exp(clamp(log_distance - colouring.header.y, -40.0, 40.0));
            } else if (result.kind != RESULT_ESCAPED) {
                value = 0.0;
            }
        }
        case ALGORITHM_TRAP: {
            if (ORBIT_STATS) {
                let shape = u32(b.w + 0.5);
                let metric = select(result.stats.trap_outside[layer],
                    result.stats.trap_inside[layer], inside);
                value = select(trap_distance(shape, metric), 0.0, metric >= 1e29);
            }
        }
        default: {}
    }
    let position = transfer(u32(a.y + 0.5), value) * a.z + a.w;
    var colour = gradient_colour(position, layer);
    if (c.w > 0.0) {
        // Darken slowly escaping or converging pixels.
        let speed = exp(-0.05 * result.value);
        colour = colour * mix(1.0, 0.3 + 0.7 * speed, c.w);
    }
    return colour;
}

// One layer's colour for this orbit result.
fn layer_colour(layer: u32, family: u32, result: GenericResult) -> vec3<f32> {
    let l = colouring.layers[layer];
    if (result.kind == RESULT_BOUNDED) {
        return colour_side(l.inside_a, l.inside_b, l.inside_c, l.inside_d,
            family, result, true, layer);
    }
    return colour_side(l.outside_a, l.outside_b, l.outside_c, l.outside_d,
        family, result, false, layer);
}

// Photoshop-style merge of one layer over the accumulated colour.
fn blend_layer(dst: vec3<f32>, src: vec3<f32>, mode: u32, opacity: f32) -> vec3<f32> {
    var blended: vec3<f32>;
    switch mode {
        case 1u: { blended = min(dst + src, vec3<f32>(1.0)); }
        case 2u: { blended = dst * src; }
        case 3u: { blended = vec3<f32>(1.0) - (vec3<f32>(1.0) - dst) * (vec3<f32>(1.0) - src); }
        case 4u: {
            blended = select(
                vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - dst) * (vec3<f32>(1.0) - src),
                2.0 * dst * src,
                dst < vec3<f32>(0.5),
            );
        }
        case 5u: { blended = min(dst, src); }
        case 6u: { blended = max(dst, src); }
        case 7u: { blended = abs(dst - src); }
        default: { blended = src; }
    }
    return mix(dst, blended, clamp(opacity, 0.0, 1.0));
}

// The composited colour of the visible layer stack, bottom first. The
// bottom layer's merge mode is ignored: it composites over black by its
// opacity alone, so a single-layer stack reproduces the pre-layer renderer
// exactly at opacity 1.
fn colour_stage(family: u32, result: GenericResult) -> vec3<f32> {
    let count = layer_count();
    var colour = layer_colour(0u, family, result) * clamp(colouring.layers[0].blend.y, 0.0, 1.0);
    for (var layer = 1u; layer < MAX_LAYERS; layer = layer + 1u) {
        if (layer >= count) { break; }
        let l = colouring.layers[layer];
        colour = blend_layer(
            colour,
            layer_colour(layer, family, result),
            u32(l.blend.x + 0.5),
            l.blend.y,
        );
    }
    return colour;
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
    // Pixel position relative to the (possibly off-centre) reference point,
    // and the reference's own parameter for parameter-plane families.
    let reference_local = local - u.reference.xy;
    let reference_c = ds2_approx(centre_ds) + u.reference.xy * ds_approx(scale_ds);
    let family = u32(u.dynamics_lo.z + 0.5);
    let newton = family == FAMILY_NEWTON;
    let dynamical = u.family_b.w > 0.5;
    if (family == FAMILY_SELF_TEST) {
        // Double-single self-test used by the ignored GPU test in
        // render/mod.rs: red = compensated addition keeps a 1e-9 offset below
        // 1.0, green = compensated multiplication keeps the cross term,
        // blue = the pixel coordinate survives the DS view transform. Any
        // channel at zero means the shader compiler's fast-math optimizations
        // have broken the compensated arithmetic.
        let one = Ds(1.0, 0.0);
        let tiny = Ds(u.view_hi.z, 0.0);
        let sum = ds_add(one, tiny);
        let back = ds_sub(sum, one);
        let add_ok = abs(ds_approx(back) - u.view_hi.z) < 1e-11;
        let square = ds_mul(sum, sum);
        let back2 = ds_sub(square, one);
        let mul_ok = abs(ds_approx(back2) - 2.0 * u.view_hi.z) < 2e-11;
        let world_ok = abs(ds_approx(ds_sub(world_ds.x, centre_ds.x)) - local.x * u.view_hi.z) < 1e-11;
        // Division: (1 + t) / (1 - t) - 1 ≈ 2t.
        let quotient = ds_div(sum, ds_sub(one, tiny));
        let back3 = ds_sub(quotient, one);
        let div_ok = abs(ds_approx(back3) - 2.0 * u.view_hi.z) < 4e-11;
        // Complex division keeps the offset too: (1 + t, t) / (1, 0).
        let complex_quotient = ds_complex_div(Ds2(sum, tiny), Ds2(one, Ds(0.0, 0.0)));
        let complex_ok = abs(ds_approx(ds_sub(complex_quotient.x, one)) - u.view_hi.z) < 1e-11
            && abs(ds_approx(complex_quotient.y) - u.view_hi.z) < 1e-11;
        return vec4<f32>(
            select(0.0, 1.0, add_ok && div_ok),
            select(0.0, 1.0, mul_ok && complex_ok),
            select(0.0, 1.0, world_ok),
            1.0,
        );
    }
    if (family == FAMILY_LYAPUNOV) {
        var colour = lyapunov_colour(lyapunov_exponent(world_f32.x, world_f32.y));
        if (u.display.w > 0.5) {
            colour = mix(
                colour,
                vec3<f32>(0.78, 0.81, 0.87),
                grid(world_ds_approx, ds_approx(scale_ds)) * 0.12,
            );
        }
        return vec4<f32>(colour, 1.0);
    }
    // Every family except Newton takes the generic path; the quadratic keeps
    // its own double-single and arbitrary-precision recurrences but joins
    // the generic f32-delta perturbation below the handoff.
    let quadratic_f64_reference = family == FAMILY_QUADRATIC && perturbation && u.deep.x < 1.5;
    if (!newton && (family != FAMILY_QUADRATIC || quadratic_f64_reference)) {
        var result: GenericResult;
        var world = world_f32;
        if (perturbation && u.deep.x < 1.5) {
            result = iterate_family_perturbation_f32(family, reference_c, reference_local, dynamical);
            world = world_ds_approx;
        } else if (perturbation) {
            result = iterate_family_perturbation_scaled(family, reference_c, reference_local, dynamical);
            world = world_ds_approx;
        } else if (double_single) {
            result = iterate_family_ds(family, centre_ds, local_offset, dynamical);
            world = world_ds_approx;
        } else {
            result = iterate_family_f32(family, world_f32, dynamical);
        }
        var colour = colour_stage(family, result);
        if (u.display.w > 0.5) {
            colour = mix(
                colour,
                vec3<f32>(0.72, 0.76, 0.82),
                grid(world, ds_approx(scale_ds)) * 0.12,
            );
        }
        return vec4<f32>(colour, 1.0);
    }
    if (newton) {
        var result: GenericResult;
        if (perturbation && u.deep.x < 1.5) {
            result = iterate_family_perturbation_f32(FAMILY_NEWTON, reference_c, reference_local, true);
        } else if (perturbation) {
            result = iterate_family_perturbation_scaled(FAMILY_NEWTON, reference_c, reference_local, true);
        } else if (double_single) {
            result = iterate_family_ds(FAMILY_NEWTON, centre_ds, local_offset, true);
        } else {
            result = iterate_family_f32(FAMILY_NEWTON, world_f32, true);
        }
        var colour = colour_stage(FAMILY_NEWTON, result);
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
        result = iterate_perturbation(centre_ds, reference_local, local_offset, reference_c, julia);
        world = world_ds_approx;
    } else if (double_single) {
        result = iterate_ds(centre_ds, local_offset, julia);
        world = world_ds_approx;
    } else {
        result = iterate_f32(world_f32, julia);
    }

    var generic = GenericResult(RESULT_BOUNDED, result.iteration, result.z,
        f32(result.iteration), result.stats);
    if (result.escaped) {
        generic.kind = RESULT_ESCAPED;
        generic.value = smooth_escape_value(FAMILY_QUADRATIC, result.iteration, result.z);
    }
    var colour = colour_stage(FAMILY_QUADRATIC, generic);

    if (u.display.w > 0.5) {
        colour = mix(
            colour,
            vec3<f32>(0.72, 0.76, 0.82),
            grid(world, ds_approx(scale_ds)) * 0.12,
        );
    }
    return vec4<f32>(colour, 1.0);
}
