struct Uniforms {
    // xy = centre, z = half-height, w = aspect ratio
    view: vec4<f32>,
    // xy = Julia parameter, z = maximum iterations, w = bailout squared
    dynamics: vec4<f32>,
    // x = 0 parameter plane / 1 dynamical plane, y = palette phase,
    // z = smooth colouring, w = show grid
    display: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
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

fn complex_square(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
}

fn palette(t: f32) -> vec3<f32> {
    // The cream / slate-blue / muted-coral family shared with 3DM, expressed
    // as a continuous scientific palette so iteration bands remain legible.
    let a = vec3<f32>(0.47, 0.49, 0.52);
    let b = vec3<f32>(0.42, 0.39, 0.36);
    let c = vec3<f32>(1.00, 0.82, 0.68);
    let d = vec3<f32>(0.06, 0.18, 0.36);
    return a + b * cos(6.2831853 * (c * t + d));
}

fn grid(world: vec2<f32>) -> f32 {
    // A scale-aware decimal grid. Derivatives keep the line near one pixel.
    let decade = pow(10.0, floor(log2(max(u.view.z, 1e-18)) / log2(10.0)));
    let cell = max(decade, 1e-18);
    let q = abs(fract(world / cell + 0.5) - 0.5) / fwidth(world / cell);
    return 1.0 - min(min(q.x, q.y), 1.0);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let local = vec2<f32>((in.uv.x * 2.0 - 1.0) * u.view.w,
                         (1.0 - in.uv.y * 2.0));
    let world = u.view.xy + local * u.view.z;

    let julia = u.display.x > 0.5;
    var z = select(vec2<f32>(0.0), world, julia);
    let c = select(world, u.dynamics.xy, julia);
    let max_iterations = u32(clamp(u.dynamics.z, 1.0, 4096.0));
    var escaped = false;
    var iteration = 0u;

    for (var i = 0u; i < 4096u; i = i + 1u) {
        if (i >= max_iterations) { break; }
        z = complex_square(z) + c;
        iteration = i + 1u;
        if (dot(z, z) > u.dynamics.w) {
            escaped = true;
            break;
        }
    }

    var colour = vec3<f32>(0.025, 0.031, 0.043);
    if (escaped) {
        var value = f32(iteration);
        if (u.display.z > 0.5) {
            let log_zn = 0.5 * log(max(dot(z, z), 1.000001));
            value = value + 1.0 - log2(max(log_zn, 1e-6));
        }
        let t = 0.035 * value + u.display.y;
        colour = palette(t);
        // A little luminance structure preserves fine escape-time detail.
        colour *= 0.82 + 0.18 * cos(6.2831853 * fract(value * 0.05));
    } else {
        // Interior is not absolute black: the dark blue makes the set boundary
        // readable against the application's charcoal canvas.
        colour = vec3<f32>(0.025, 0.040, 0.058);
    }

    if (u.display.w > 0.5) {
        colour = mix(colour, vec3<f32>(0.72, 0.76, 0.82), grid(world) * 0.12);
    }

    return vec4<f32>(colour, 1.0);
}
