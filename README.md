# IteraScope

**An open laboratory for complex dynamics.**

IteraScope is a GPU-first scientific viewer for exploring iterated maps. The
first instrument links the parameter and dynamical planes of the quadratic
family

\[
f_c(z) = z^2 + c.
\]

The same Rust application runs natively and in a WebGPU-capable browser. egui
and the fractal renderer share one `wgpu` device; the escape-time calculation
runs entirely in WGSL.

## Current prototype

- Linked Mandelbrot parameter plane and Julia dynamical plane
- Click-to-select the Julia parameter
- Click-to-centre 2× zoom and drag navigation in either plane
- Adjustable iteration limit, bailout, palette phase and smooth colouring
- Optional scale-aware coordinate grid
- Live coordinate and magnification readouts
- Automatic GPU precision switching between fast `f32` and a centred
  double-single recurrence with adaptive per-pixel rebasing (approximately
  48-bit reference coordinates)
- A cached 3×3 CPU instability probe comparing `f32` and `f64` orbit behavior
- Responsive side-by-side or stacked layout
- Native and WASM entry points from the same codebase
- Offline WGSL parsing and validation using the exact Naga version used by wgpu

The interface reports the active arithmetic for each pane. Navigation remains
in CPU `f64`; rendering starts with fast GPU `f32` and switches automatically
to GPU double-single when coordinate resolution is at risk or the lightweight
orbit probe detects divergent escape behavior. Perturbation and
reference-orbit infrastructure remain subsequent rendering stages.

## Run natively

```sh
cargo run
```

## Run in a browser

Install the WASM target and [Trunk](https://trunkrs.dev/), then:

```sh
rustup target add wasm32-unknown-unknown
cargo install trunk
trunk serve
```

Open `http://127.0.0.1:8080` in a browser with WebGPU enabled.

For a production bundle, with Binaryen's `wasm-opt` installed when available:

```sh
./build-release.sh
```

## Deploy on Fly.io

IteraScope deploys as a static bundle served by Caddy. The multi-stage
[`Dockerfile`](Dockerfile) compiles the WASM application, optimizes it with
Binaryen, and precompresses the browser assets. [`fly.toml`](fly.toml) runs the
small serving container in Toronto (`yyz`), enforces HTTPS for WebGPU, performs
HTTP health checks, and stops the Machine when it is idle.

For the first deployment:

```sh
fly auth login
fly apps create iterascope
fly deploy
```

Subsequent deployments only require:

```sh
fly deploy
```

The app will be available at `https://iterascope.fly.dev`. If the global Fly
app name is no longer available, choose another name with `fly apps create` and
change the `app` value in `fly.toml` to match.

## Validate

```sh
cargo check
cargo test
cargo check --target wasm32-unknown-unknown
```

## Near-term roadmap

1. Deterministic experiment documents and view import/export
2. Orbit inspector and critical-orbit diagnostics
3. Perturbation with a small arbitrary-precision CPU reference orbit
4. Period detection, multipliers and Newton refinement

## Project status

IteraScope is at the first executable-prototype stage. Its numerical results
are exploratory, not yet certified. Precision mode, thresholds and algorithmic
assumptions will remain visible as the scientific toolset grows.

## License

IteraScope is open-source software released under the [MIT License](LICENSE).
