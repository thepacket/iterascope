# IteraScope

![IteraScope: the quadratic instrument, with the Mandelbrot parameter plane on
the left and the linked Julia dynamical plane for the selected c on the
right](docs/hero.png)

**An open laboratory for complex dynamics.**

Note: I am considering moving this new born project toward something like Ultra Fractal,
something artistic, not "scientific".

IteraScope is a GPU-first scientific laboratory for exploring iterated maps.
Its reference instrument links the parameter and dynamical planes of the
quadratic family

\[
f_c(z) = z^2 + c,
\]

with an arbitrary-precision deep-zoom path. Around it sits a catalogue of
twenty-five further escape-time, convergence-time and root-finding
instruments, from the Burning Ship and Magnet maps to the Lyapunov plane of
the forced logistic map, all rendered by the same WGSL shader and all backed by
a CPU `f64` reference implementation of the same recurrence.

The same Rust application runs natively and in a WebGPU-capable browser. egui
and the fractal renderer share one `wgpu` device; every per-pixel iteration
runs entirely in WGSL. CPU calculations are reserved for pointwise
diagnostics, precision validation and arbitrary-precision reference orbits
rather than full-frame pixel rendering.

## Scientific instruments

Two instrument layouts exist. **Parameter/dynamical** instruments colour the
left pane by the fate of a critical (or otherwise distinguished) orbit for
each parameter `c`, and the right pane by the fate of every starting value
`z₀` under the selected parameter; clicking the left pane selects `c`.
**Overview/detail** instruments show the same plane in both panes; clicking
the overview selects a point and opens a linked, magnified detail region.

| Family | Recurrence | Layout | Start of the left-pane orbit | Terminates on | Deep zoom |
| --- | --- | --- | --- | --- | --- |
| Quadratic | `z ← z² + c` | parameter/dynamical | `z₀ = 0` | `\|z\| > bailout` | f64 + AP perturbation |
| Newton cubic | `z ← z − (z³−1)/(3z²)` | overview/detail | pixel | residual `< 10⁻⁶` | f64 + AP perturbation |
| Multibrot | `z ← zᵈ + c`, `2 ≤ d ≤ 8` | parameter/dynamical | `z₀ = 0` | bailout | f64 + AP perturbation |
| Tricorn | `z ← z̄² + c` | parameter/dynamical | `z₀ = 0` | bailout | f64 + AP perturbation |
| Perpendicular Mandelbrot | `(x² − y², −2\|x\|y) + c` | parameter/dynamical | `z₀ = 0` | bailout | f64 + AP perturbation |
| Burning Ship | `(x² − y², 2\|xy\|) + c` | parameter/dynamical | `z₀ = 0` | bailout | f64 + AP perturbation |
| Perpendicular Burning Ship | `(x² − y², −2x\|y\|) + c` | parameter/dynamical | `z₀ = 0` | bailout | f64 + AP perturbation |
| Celtic | `(\|x² − y²\|, 2xy) + c` | parameter/dynamical | `z₀ = 0` | bailout | f64 + AP perturbation |
| Perpendicular Celtic | `(\|x² − y²\|, −2\|x\|y) + c` | parameter/dynamical | `z₀ = 0` | bailout | f64 + AP perturbation |
| Buffalo | `(\|x² − y²\|, 2\|xy\|) + c` | parameter/dynamical | `z₀ = 0` | bailout | f64 + AP perturbation |
| Perpendicular Buffalo | `(\|x² − y²\|, −2x\|y\|) + c` | parameter/dynamical | `z₀ = 0` | bailout | f64 + AP perturbation |
| Lambda (logistic) | `z ← λ z (1 − z)` | parameter/dynamical | `z₀ = ½` | bailout | f64 + AP perturbation |
| Phoenix | `zₙ₊₁ = zₙ² + Re c + (Im c) zₙ₋₁` | parameter/dynamical | `z₀ = z₋₁ = 0` | bailout | f64 + AP perturbation |
| Manowar | `zₙ₊₁ = zₙ² + zₙ₋₁ + c` | parameter/dynamical | `z₀ = z₋₁ = c` | bailout | f64 + AP perturbation |
| Spider | `z ← z² + c`, `c ← c/2 + z` | parameter/dynamical | `z₀ = 0` | bailout | f64 + AP perturbation |
| Magnet I | `z ← ((z² + c − 1)/(2z + c − 2))²` | parameter/dynamical | `z₀ = 0` | bailout or `\|z − 1\| < 10⁻⁴` | f64 + AP perturbation |
| Magnet II | `z ← ((z³ + 3(c−1)z + (c−1)(c−2)) / (3z² + 3(c−2)z + (c−1)(c−2) + 1))²` | parameter/dynamical | `z₀ = 0` | bailout or `\|z − 1\| < 10⁻⁴` | f64 + AP perturbation |
| Exponential | `z ← c eᶻ` | parameter/dynamical | `z₀ = 0` | `Re z > 50` | f32 only |
| Sine | `z ← c sin z` | parameter/dynamical | `z₀ = π/2` | `\|Im z\| > 50` | f32 only |
| Cosine | `z ← c cos z` | parameter/dynamical | `z₀ = 0` | `\|Im z\| > 50` | f32 only |
| Collatz | `z ← ¼(2 + 7z − (2 + 5z) cos πz)` | overview/detail | pixel | `\|Im z\| > 20` or `\|z\| > 10⁶` | f32 only |
| Lyapunov (Markus) | `xₙ₊₁ = rₙ xₙ(1 − xₙ)`, `rₙ ∈ {a, b}` by sequence | overview/detail | `x₀ = ½` | Lyapunov exponent after `n` iterations | f32 only |
| Nova | `z ← z − R (zᵖ − 1)/(p zᵖ⁻¹) + c` | parameter/dynamical | `z₀ = 1` | `\|Δz\| < 10⁻⁵` or `\|z\| > 10⁶` | f64 + AP perturbation |
| Barnsley 1 | `z ← (z ∓ 1) c` by sign of `Re z` | parameter/dynamical | `z₀ = c` | bailout | f64 + AP perturbation |
| Barnsley 2 | `z ← (z ∓ 1) c` by sign of `Im(zc)` | parameter/dynamical | `z₀ = c` | bailout | f64 + AP perturbation |
| Mandelbox (2D) | `v ← s · ballfold(boxfold(v)) + c` | parameter/dynamical | `v₀ = 0` | `\|v\| > 4 × bailout` | f64 + AP perturbation |

`x` and `y` denote `Re z` and `Im z`. The sign conventions of the
absolute-value variants follow the Kalles Fraktaler family; the opposite sign
of an imaginary part produces the mirror image in the conjugate parameter, not
a different set. Buffalo uses the componentwise absolute value of `z²`. The
Manowar map has complex Jacobian determinant `−1`, so its bounded set has no
attracting interior; it is rendered for its escape-time level sets. Magnet,
Nova and Newton orbits finish by converging, so their default colouring is the
argument of the limit (the basin) darkened by convergence time; the Lyapunov
plane is coloured by the sign and size of the exponent, the stable regions
through the gradient and the chaotic ones in darkening blues.

The **Deep zoom** column lists the precision paths available past plain
`f32`: every listed family switches to GPU perturbation around a CPU
reference orbit — `f64` below the `1.14e14×` handoff, arbitrary precision
beyond it — up to the `10^5000×` navigation ceiling. The quadratic
instrument additionally keeps its compensated double-single recurrence as a
fallback and as the subject of its orbit probes. The
transcendental families and the Lyapunov plane stop at `f32`: their reference
orbits would need arbitrary-precision exponentials and trigonometry, which the
decimal arithmetic layer does not yet provide.

Every instrument links a global classification view to a local diagnostic
view and exports the selected value, viewports, family parameters, numerical
settings and display state together as a versioned experiment document. The
control panel always shows a CPU `f64` diagnostic for the selected point
computed from the reference implementation in `src/family.rs`, which is the
definition of record for each recurrence; the shader branches are transcribed
from it.

## What works today

- Linked Mandelbrot parameter plane and Julia dynamical plane
- Newton basin instrument for `z³ - 1`, with three-root classification and a
  linked convergence-detail pane
- Twenty-four further escape-time and convergence-time families (Multibrot,
  eight absolute-value variants, Lambda, Phoenix, Manowar, Spider, Magnet I/II,
  exponential, sine, cosine, Collatz, Lyapunov, Nova, Barnsley 1/2 and the 2D
  Mandelbox) with per-family parameters, presets and a CPU `f64` orbit
  diagnostic for the selected point
- Deep zoom (arbitrary-precision navigation plus GPU perturbation) for every
  family except the transcendental ones and the Lyapunov plane
- Headless GPU tests that render every family through the real pipeline
  (`gpu_family_gallery`), compare the perturbation path against the
  double-single path and a CPU `f64` raster at `10⁶×`
  (`gpu_perturbation_matches_double_single`), render seven families at `10³⁰×`
  around arbitrary-precision repelling fixed points
  (`gpu_deep_zoom_resolves_structure`), and check that compensated
  arithmetic survives the shader compiler (`gpu_double_single_self_test`);
  all are `#[ignore]`d and run with
  `ITERASCOPE_RENDER_DIR=out cargo test --release <name> -- --ignored`
- Pointwise Newton diagnostics reporting the attracting root, iteration count,
  polynomial residual, final value, last step and derivative singularities
- Click-to-centre 2× zoom; parameter-plane clicks also select the Julia
  parameter; right-click recentres without zooming
- Wheel, trackpad pinch and drag navigation in either plane
- Shift-modified click, wheel or pinch for accelerated logarithmic zoom in
  either direction
- Automatic progressive Julia navigation to a selected `10^n` target, up to
  `10^5000×`, with visible intermediate renders and Start/Stop controls
- Keyboard arrows and four on-screen buttons for deterministic fine panning by
  one tenth of the displayed range (Shift + arrows: one hundredth)
- Adjustable iteration limit up to 50,000 and escape radius up to `1e10`
- A colour stage in the Ultra Fractal mould: cyclic gradients
  (control points, RGB or HSL blending, cubic smoothing, rotation, presets,
  random generation, `.ugr` and Fractint `.map` import, `.ugr` export) and
  independent outside/inside colouring algorithms — (smooth) iteration count,
  decomposition by argument, triangle-inequality average, stripe average,
  exterior distance estimate and orbit traps (point, cross, circle, square,
  lines) — each with density, offset, transfer curve and iteration shading.
  Every algorithm works through the perturbation paths at any depth; the
  distance estimate is evaluated in logarithms and verified at `1e4000×`
  (Ultra Fractal 5's limit; IteraScope navigates to `1e5000×`)
- Layers: up to eight complete colour stages composited over one iteration
  pass — each layer has its own gradient, algorithms, opacity and merge mode
  (Normal, Add, Multiply, Screen, Overlay, Darken, Lighten, Difference) —
  with a single-image workspace (the default layout) that gives the
  composited image the full window. Layers share the image's location and
  family, so one reference orbit serves the whole stack and the deep-zoom
  engine is untouched; a single-layer stack renders byte-identically to, and
  as fast as, the pre-layer renderer
- An Ultra Fractal-style switch picker: while the single image shows the
  dynamical plane, **Pick c…** opens a parameter-plane window with
  crosshair, scroll zoom and a live Julia thumbnail of the hovered
  parameter; clicking sets `c` and the composited image follows immediately
- Still-image export (native app): the current view rendered to PNG at up
  to 16384×16384 with 2×2 or 3×3 supersampled anti-aliasing box-filtered in
  linear light; sizes whose supersampled frame exceeds the 8192-pixel
  texture limit render as tiles around the same reference orbit, seamless
  at any magnification
- Two independently collapsible, drag-resizable control panes: the left
  **Instrument** pane (family, document, parameters, computation,
  navigation, diagnostics) and the right **Studio** pane (layers, colouring,
  still and animation export)
- Optional scale-aware coordinate grid
- Zoom-path animation: a dive to the current centre between two magnification
  exponents at constant or eased logarithmic speed, with an optional gradient
  sweep, exported (native app) as a PNG image sequence at up to 8192×8192 and
  optionally encoded to MP4 with ffmpeg. One reference orbit serves every
  frame — the arbitrary-precision orbit of the centre is re-described per
  frame, so a `10^1000×` dive costs one orbit, not one per frame
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
2×; right-click to make it the centre without zooming, which is the way to
frame a region precisely before magnifying it. Clicking in the parameter
plane additionally sets `c`, so the Julia plane on the right updates to the
corresponding dynamical system. Drag to pan, use the wheel or trackpad to
zoom, or select a **Fine pan target** and use the arrow keys or `< ^ v >`
controls. Each fine-pan step moves exactly one tenth of the currently
displayed horizontal or vertical range; holding Shift with the arrow keys
moves one hundredth.

### Animation

The **Still image** section renders the active pane's current view to a PNG
at a chosen resolution (up to 16384×16384) with supersampled anti-aliasing:
the frame is rendered at two or three times the requested size and
box-filtered down in linear light, so gradient edges keep their brightness.
When the supersampled frame exceeds the GPU's 8192-pixel texture limit it is
rendered as tiles, one per interface update with a progress bar. Tiles render
around the same reference orbit as the whole frame — the frame centre simply
becomes an off-centre reference — so the perturbation deltas are
algebraically identical to the whole frame's and tiling is exact at any
magnification, with no seams. It uses the same frozen-scene machinery as the
animation exporter: one reference orbit at the current centre, valid at the
full depth of the view.

The controls are split across two independently collapsible panes: the left
**Instrument** pane holds the scientific state (family, document, parameters,
computation, navigation and diagnostics), the right **Studio** pane the
artistic state (layers, colouring, still and animation export). The chevron
in each pane's header collapses it to a slim strip, and each pane can be
widened by dragging its inner edge.

The **Animation** section renders the classic deep-zoom video: a dive to the
active pane's centre, from a start to an end magnification exponent (defaults:
`10^0` to the current view via **End = view**) at constant logarithmic speed,
optionally eased at both ends, with an optional constant-speed gradient sweep.
The exporter writes `frame-00000.png …` at the chosen resolution and frame
rate into a new directory under the configured folder, one frame per UI
update so the interface stays live, and — when ffmpeg is installed — encodes
`zoom.mp4` when the sequence completes. Because the centre is fixed, the
arbitrary-precision reference orbit is computed once and re-described in
scale for every frame; frames below the `1.14e14×` handoff use the same orbit
projected to `f64`, and the switch between the paths is seamless. Export runs
in the native application; the browser build shows the settings but cannot
write files.

### Layers

The **Layers** section holds the image's layer stack, shown top first. The
orbit is iterated once per pixel; every visible layer then colours that same
result with its own gradient and algorithms and is merged over the layers
beneath it by its opacity and merge mode (the bottom layer composites over
black, its mode ignored). Duplicate the active layer, restyle it — a stripe
average multiplied over an iteration-count base, an orbit trap screened on
top — and reorder or hide layers freely; the **Colouring** section always
edits the active layer. Layers currently share the image's location, family
and iteration settings (per-layer formulas and locations are future work),
which is what keeps the whole stack exact at any magnification: one
reference orbit drives every layer. The **Image** toggle in the top bar
swaps the two linked panes for the composited image alone, full-window — the
default layout; **Panes** brings the linked scientific layout back. While the
image shows the dynamical plane, **Pick c…** in the top bar opens the switch
picker: the parameter plane with a crosshair, scroll zoom, a marker on the
current `c` and a live Julia thumbnail of the hovered parameter — click to
choose and the composited image follows immediately. From a parameter-plane
image, **Open Julia** jumps to the dynamical plane of the selected `c`.

### Colouring

The **Colouring** section holds one gradient and two colouring algorithms, as
in Ultra Fractal: **Outside** colours orbits that escaped or converged,
**Inside** colours orbits that reached the iteration limit. Each algorithm
reduces the orbit to a value — the smoothed iteration count, the argument of
the final `z` (continuous or in sectors; two sectors is binary decomposition),
the triangle-inequality or stripe average over the orbit, the exterior
distance estimate in pixels, or the closest approach to an orbit trap — and
maps it to a gradient position through a transfer curve, a density and an
offset. The averages interpolate their final term by the family's own
smoothing so they stay continuous across iteration bands; the distance
estimate follows the orbit's derivative in scaled arithmetic and is evaluated
in logarithms, so it is exact at any depth for the quadratic, Multibrot and
lambda families. Large escape radii (up to `1e10`, under **Computation**) give
the averages their smoothest results.

Click the gradient bar to open the editor: drag the markers to move control
points, double-click the bar to add one, pick the colour and position of the
selected point, choose RGB or HSL blending, cubic smoothing, rotation, and
reverse or redistribute the points. The editor offers presets, a random
generator, Ultra Fractal `.ugr` and Fractint `.map` import (paste the text or
drop the file) and `.ugr` export.

The accumulators behind orbit traps, the averages and the distance estimate
live inside every iteration loop, including the perturbation paths. They are
compiled into a second shader variant that is only used while one of those
algorithms is selected, so the default configuration renders exactly as fast
as before the colour stage existed — the deep-zoom engine is not the price of
the artistic features.

The **Critical orbit** section computes

\[
z_0 = 0, \qquad z_{n+1} = z_n^2 + c.
\]

Select an iteration to inspect its complex coordinate, magnitude, argument and
derivative with respect to `c`. The Julia overlay connects only the selected
point and its eight immediate predecessors. These straight segments indicate
iteration order; they are not continuous curves along the Julia set. The
selected orbit point can also become the centre of the Julia view.

### Newton basins

Choose **Newton** in the Experiment section to study Newton's method applied
to

\[
p(z) = z^3 - 1, \qquad
N(z) = z - \frac{z^3 - 1}{3z^2}.
\]

By default the left pane colours each starting value by the argument of the
root to which it converges (the **Decomposition** colouring), with brightness
encoding convergence speed; switch the outside colouring to **Iteration
count** to emphasize convergence time and expose sensitive basin boundaries.
Clicking the overview selects `z₀` and opens a linked region in the right
pane. Convergence time is a continuous estimate of where the residual crossed
the convergence threshold, avoiding false bands from whole iteration counts. The CPU `f64` diagnostic independently reports
the exact integer iteration count alongside the selected orbit's root,
residual, last Newton step and final complex value; `z₀ = 0` is explicitly
identified as a derivative singularity.

Newton mode is currently limited to 2,048 iterations and the stable
`f32`/double-single viewport range. Its double-single path keeps the starting
coordinate, polynomial, derivative, complex division and Newton update in
compensated arithmetic so magnified basin boundaries do not collapse onto an
`f32` coordinate grid. The arbitrary-precision perturbation path remains
specific to the quadratic family.

### Other escape-time families

Choose any other family from the Experiment drop-down. Parameter/dynamical
instruments behave like the quadratic one: click the left pane to choose the
parameter, adjust it numerically or through presets, and read the **Critical
orbit** diagnostic, which iterates the same recurrence on the CPU in `f64`
and reports whether the orbit escapes, converges, becomes non-finite or stays
bounded through the iteration limit. Overview/detail instruments (Newton,
Collatz, Lyapunov) select a starting point instead. Families with parameters
expose them in **Family parameters**: the Multibrot and Nova degree, the Nova
relaxation `R`, the Lyapunov forcing sequence over `{A, B}` (up to 32
symbols, with the first quarter of the iterations discarded as a transient)
and the Mandelbox scale, minimum radius and fixed radius.

Default parameters for the dynamical planes were chosen just inside the
boundary of each family's connectedness locus, so the default Julia sets are
thin and filamentary rather than filled discs; the presets offer a few
alternatives, and clicking anywhere in the parameter plane selects another.
Bounded orbits take the **Inside** colouring, a dark solid by default; an
inside **Orbit trap** (point at the origin) reveals the basins of attracting
cycles.

Precision handling is deliberately explicit. Once the `f32` coordinate grid
becomes coarser than a pixel, every family renders by GPU perturbation
around a reference orbit: below the `1.14e14×` handoff the reference is
computed on the CPU in `f64` (`F64 PERT`), beyond it in arbitrary precision
(`AP PERT`). (The GPU double-single recurrence that previously carried the
quadratic instrument through that range was found, on Metal, to lose
structure well before the handoff — a flat image at `10^11×` where the
`f64` reference resolves over a thousand distinct escape times — so it now
serves only as a fallback; the CPU probes still characterise it.) In both cases
the navigation layer keeps exact coordinates, the CPU builds the reference
with `family::reference_orbit_f64` or `arbitrary::deep_step` (exact
transcriptions of the `f64` definition, checked by test), and the shader
iterates an exact delta recurrence for that family in scaled
mantissa/exponent arithmetic —
binomial expansions for `zᵈ`, the `diffabs` identity for every absolute-value
variant, exact rational differences for the Magnet and Nova maps, branch-aware
differences for the Barnsley and Mandelbox folds. Newton's basins reuse the
Nova recurrence. The transcendental, Collatz and Lyapunov families render in
`f32` only and show `F32 LIMIT` once the pixel grid collapses.

A note on compensated (double-single) arithmetic on the GPU: Metal compiles
WGSL with fast math, and its reassociation silently reduced every compensated
sum back to `f32` (the GPU self-test in `render/mod.rs` reproduces this). The
DS primitives now route each rounded intermediate through one of four opaque
`1.0` uniforms so no two adjacent terms share a factor the optimizer can pull
out, and the self-test verifies addition, multiplication, division and the
view transform on the real device. Because that protection is only as good
as the compiler's behaviour at each inlined call site, the generic families
no longer depend on it for image coherence: their remaining DS path (used
only when a reference orbit ends early at the handoff) is a centred
recurrence like the quadratic one — a shared centre orbit plus exact
per-pixel deltas with rebasing — rather than a plain per-pixel DS orbit.

### On-demand rendering and timing

IteraScope renders on demand. An unchanged view does not continuously consume
CPU and GPU resources. A repaint is requested for user input, the settled
replacement following deep navigation, each active progressive-zoom stage,
and each slice of an arbitrary-precision reference orbit that is still being
extended. The previous completed deep image remains visible while its
replacement reference is prepared. Progressive stages are spaced by 750 ms so
WebGPU is not continuously fed full-screen deep renders faster than they can
be presented.

While input is active (dragging, zooming, stepping, clicking) each pane is
rendered at one third of its resolution into a texture and scaled up, so a
frame stays cheap at any depth and the view follows the pointer instead of
lurching after long frames; the settled frame renders at full resolution
(a milder half-resolution preview is used while a reference orbit is still
being extended).

Reference orbits are built so that navigation stays responsive. Below the
handoff the `f64` reference is chosen as the longest-lived orbit among the
view centre and a coarse grid of candidates, so few pixels outlive it (those
that do continue in plain `f32` from their already-separated state).
Arbitrary-precision references are extended across frames under a 6 ms
budget: the GPU renders with the points available so far and refines as the
orbit grows, instead of freezing the interface for a long high-precision
orbit. While a deep view is being dragged, zoomed or stepped, its existing
reference orbit is kept and merely re-described relative to the moved view
(perturbation does not require a centred reference), so the image follows
the input immediately; a fresh centred reference is built once input
settles and swapped in when complete. The delta recurrence itself runs in plain `f32` below the handoff and
in scaled mantissa/exponent arithmetic only at arbitrary-precision depth;
the two are instantiated from one template at shader-load time.

The top-right **ON DEMAND** indicator reports the smoothed CPU time spent in
the most recent UI update. It is not a refresh interval, frame rate, or direct
measurement of shader execution time. WebGPU command completion is
asynchronous, so separate reference-orbit timings, GPU timestamps where
supported, and presented-frame timings are still planned for detailed
performance diagnostics.

## Numerical model and limits

The interface reports the active arithmetic for each pane. Viewport navigation
starts in CPU `f64` and moves to arbitrary-precision decimal coordinates for
quadratic deep zoom. Rendering starts with fast GPU `f32` and switches
automatically to perturbation around a CPU `f64` reference orbit when
coordinate resolution is at risk or the lightweight orbit probe detects
divergent escape behaviour. The
`DS STABLE`, `DS RISK` and `DS LIMIT` labels describe agreement with sampled
`f64` orbits and coordinate resolution—they are more meaningful than visual
smoothness alone.

The current stable raster path hands off at the experimentally confirmed
`1.14e14×` boundary. IteraScope now has pure-Rust arbitrary-precision decimal
coordinates, a zoom-dependent precision policy and arbitrary-precision
reference orbits for every perturbation-capable family, sized for the
`1e1000×` acceptance target. At the
handoff, those cached reference orbits now drive an exponent-scaled GPU
perturbation path; an early-ending reference falls back per fragment to the
stable DS renderer at the handoff. Beyond it, the pane centre, scale, click
coordinates and tenth-range pans remain in arbitrary precision through the
`1e1000×` acceptance target, with navigation currently available through
`1e5000×`; experiment JSON records exact decimal values and the configured
progressive Julia target. CPU work scales with reference precision and
iteration count rather than pixel count. Automatic reference rebasing and
formal glitch validation remain in development.

At extreme depth, a uniformly dark view (or, with an inside orbit trap, a
smoothly shaded one) is not yet proof that the sampled region is
mathematically interior. Progressive zoom retains a finite selected
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

The ignored `gpu_*` tests in `src/render/mod.rs` run on the real GPU and
write PPM renders to `$ITERASCOPE_RENDER_DIR`: a gallery of every family, the
perturbation-versus-double-single-versus-CPU comparison, deep-zoom structure
at `1e30×`, preview-blit and pan consistency, a gallery of every colouring
algorithm at the default view and around an `f64` reference at `1e12×`, the
iteration, distance, stripe and triangle colourings around an
arbitrary-precision reference at `1e4000×`, a five-frame zoom-path export
crossing every precision path at a row-padded width, a layer-compositing test
(a single-layer stack must reproduce the pre-layer renderer byte for byte,
every merge mode must be distinct, and an eight-layer stack must survive),
and a timing probe that also records the cost of the orbit-statistics
variant:

```sh
ITERASCOPE_RENDER_DIR=out cargo test --release gpu_ -- --ignored --nocapture
```

## Experiment documents

Open **Document → Export / Import JSON** to capture the complete reproducible
experiment state. The versioned document (format version 6) records the
active family, both plane centres and scales, the selected parameter or
starting value, the family parameters the active family uses, computation
limits, the layer stack with each layer's gradient and colouring algorithms,
and display settings, including critical-orbit overlay visibility. Version-5
documents load their single colouring as a one-layer stack; documents from
before version 5 load with the default colouring (their palette phase becomes
the outside offset). Runtime
diagnostics, selected inspector step and frame timing are deliberately not
stored. Copy the JSON to export it; paste another IteraScope document into the
editor and choose **Load JSON** to import it. Imports are validated before any
live state is changed.

## Near-term roadmap

1. Automatic reference rebasing and perturbation glitch detection
2. Deterministic validation against direct arbitrary-precision sample orbits
3. Progressive/background reference-orbit generation at very high iteration
   counts
4. Parameterized Newton polynomials and rational maps with critical-point
   analysis
5. Orbit probes for the non-quadratic escape-time families, and
   arbitrary-precision exponential/trigonometric reference orbits so the
   transcendental families can join the deep-zoom path
6. Towards generative fractal art: per-layer formulas, locations and masks
   (layers, the switch picker, anti-aliased tiled stills and the single-image
   default landed); keyframed centre drift and per-parameter animation
   curves; orbit-trap options that skip the shared leading iterations of
   deep orbits; derivatives for the remaining families' distance estimates

## Project status

IteraScope is an executable scientific prototype. Its numerical results are
exploratory, not certified. Precision modes, thresholds and algorithmic
assumptions remain visible so a plausible-looking image is not silently
presented as a trustworthy computation.

## License

IteraScope is open-source software released under the [MIT License](LICENSE).
