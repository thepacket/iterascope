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

## What works today

- Linked Mandelbrot parameter plane and Julia dynamical plane
- Click-to-centre 2× zoom; parameter-plane clicks also select the Julia parameter
- Wheel, trackpad pinch and drag navigation in either plane
- Shift-modified click, wheel or pinch for accelerated logarithmic zoom in
  either direction
- Automatic progressive Julia navigation to a selected `10^n` target, up to
  `10^5000×`, with visible intermediate renders and Start/Stop controls
- Keyboard arrows and four on-screen buttons for deterministic fine panning by
  one tenth of the displayed range
- Adjustable iteration limit up to 50,000, bailout, palette phase and smooth colouring
- Optional scale-aware coordinate grid
- Live magnification plus navigation, rendered-coordinate, rounding-delta and
  pixel-scale readouts
- Versioned JSON experiment documents with cross-platform copy/paste import and export
- Cached `f64` critical-orbit inspector with escape, smooth-escape and
  parameter-sensitivity diagnostics
- Synchronized selected-step overlay in the Julia plane, using a short fading
  critical-orbit tail rather than an unreadable full trajectory
- Automatic GPU precision switching between fast `f32` and a centred
  double-single recurrence with adaptive per-pixel rebasing (approximately
  48-bit reference coordinates)
- A cached 3×3 CPU instability probe comparing both GPU arithmetic paths with
  `f64`, including adaptive rebasing and non-finite detection
- Explicit `DS STABLE`, `DS RISK`, and `DS LIMIT` states with diagnostic reasons
- Responsive side-by-side or stacked layout
- Native and WASM entry points from the same codebase
- Offline WGSL parsing and validation using the exact Naga version used by wgpu

## Using the laboratory

Click a point in either plane to make it the centre and immediately zoom by
2×. Clicking in the parameter plane additionally sets `c`, so the Julia plane
on the right updates to the corresponding dynamical system. Drag to pan, use
the wheel or trackpad to zoom, or select a **Fine pan target** and use the arrow
keys or `< ^ v >` controls. Each fine-pan step moves exactly one tenth of the
currently displayed horizontal or vertical range.

The **Critical orbit** section computes

\[
z_0 = 0, \qquad z_{n+1} = z_n^2 + c.
\]

Select an iteration to inspect its complex coordinate, magnitude, argument and
derivative with respect to `c`. The Julia overlay connects only the selected
point and its eight immediate predecessors. These straight segments indicate
iteration order; they are not continuous curves along the Julia set. The
selected orbit point can also become the centre of the Julia view.

### On-demand rendering and timing

IteraScope renders on demand. An unchanged view does not continuously consume
CPU and GPU resources. A repaint is requested for user input, the settled
replacement following deep navigation, and each active progressive-zoom
stage. The previous completed deep image remains visible while its replacement
reference is prepared. Progressive stages are spaced by 750 ms so WebGPU is
not continuously fed full-screen deep renders faster than they can be
presented.

The top-right **ON DEMAND** indicator reports the smoothed CPU time spent in
the most recent UI update. It is not a refresh interval, frame rate, or direct
measurement of shader execution time. WebGPU command completion is
asynchronous, so separate reference-orbit timings, GPU timestamps where
supported, and presented-frame timings are still planned for detailed
performance diagnostics.

## Numerical model and limits

The interface reports the active arithmetic for each pane. Navigation remains
in CPU `f64`; rendering starts with fast GPU `f32` and switches automatically
to GPU double-single when coordinate resolution is at risk or the lightweight
orbit probe detects divergent escape behaviour. The `DS STABLE`, `DS RISK` and
`DS LIMIT` labels describe agreement with sampled `f64` orbits and coordinate
resolution—they are more meaningful than visual smoothness alone.

The current stable raster path hands off at the experimentally confirmed
`1.14e14×` boundary. IteraScope now has pure-Rust arbitrary-precision decimal
coordinates, a zoom-dependent precision policy and arbitrary-precision
quadratic reference orbits sized for the `1e1000×` acceptance target. At the
handoff, those cached reference orbits now drive an exponent-scaled GPU
perturbation path; an early-ending reference falls back per fragment to the
stable DS renderer at the handoff. Beyond it, the pane centre, scale, click
coordinates and tenth-range pans remain in arbitrary precision through the
`1e1000×` acceptance target, with navigation currently available through
`1e5000×`; experiment JSON records exact decimal values and the configured
progressive Julia target. CPU
work scales with reference precision and iteration count rather than pixel
count. Automatic reference rebasing and formal glitch validation remain in
development.

At extreme depth, a uniformly black view is not yet proof that the sampled
region is mathematically interior. Progressive zoom retains a finite selected
centre, which can eventually fall entirely to one side of a boundary, and the
current perturbation path does not yet implement automatic rebasing or glitch
detection. Increasing the iteration limit also rebuilds the arbitrary-
precision reference orbit synchronously and can ask every displayed pixel to
execute up to 50,000 shader iterations. High iteration counts can therefore
temporarily make the application unresponsive, especially for non-escaping
views.

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

## Experiment documents

Open **Document → Export / Import JSON** to capture the complete reproducible
experiment state. The versioned document records the quadratic family, both
plane centres and scales, the selected parameter, computation limits, and
display settings, including critical-orbit overlay visibility. Runtime
diagnostics, selected inspector step and frame timing are deliberately not
stored. Copy the JSON to export it; paste another IteraScope document into the
editor and choose **Load JSON** to import it. Imports are validated before any
live state is changed.

## Near-term roadmap

1. Automatic reference rebasing and perturbation glitch detection
2. Deterministic validation against direct arbitrary-precision sample orbits
3. Progressive/background reference-orbit generation at very high iteration
   counts

## Project status

IteraScope is an executable scientific prototype. Its numerical results are
exploratory, not certified. Precision modes, thresholds and algorithmic
assumptions remain visible so a plausible-looking image is not silently
presented as a trustworthy computation.

## License

IteraScope is open-source software released under the [MIT License](LICENSE).
