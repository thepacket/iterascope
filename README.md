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
| Quadratic | `z ← z² + c` | parameter/dynamical | `z₀ = 0` | `\|z\| > bailout` | DS, AP perturbation |
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
attracting interior; it is rendered for its escape-time level sets. Magnet and
Nova orbits are coloured by convergence time (Nova additionally by the
argument of the attracting point); the Lyapunov plane is coloured by the sign
and size of the exponent, warm for stable and cool for chaotic forcing.

The **Deep zoom** column lists the precision paths available past plain
`f32`: the quadratic instrument uses compensated double-single (`DS`, about
48 bits, validated by its orbit probes) up to the `1.14e14×` handoff; every
other listed family switches directly to GPU perturbation around a CPU
reference orbit — `f64` below the handoff, arbitrary precision beyond it —
which carries it to the same `10^5000×` navigation ceiling. The
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
- Adjustable iteration limit up to 50,000, bailout, palette phase and smooth colouring
- Optional interior shading of bounded orbits by the minimum modulus they
  reach (an orbit trap at the origin), exposing basin structure without
  claiming that a dark pixel is proven interior
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
2×; right-click to make it the centre without zooming, which is the way to
frame a region precisely before magnifying it. Clicking in the parameter
plane additionally sets `c`, so the Julia plane on the right updates to the
corresponding dynamical system. Drag to pan, use the wheel or trackpad to
zoom, or select a **Fine pan target** and use the arrow keys or `< ^ v >`
controls. Each fine-pan step moves exactly one tenth of the currently
displayed horizontal or vertical range; holding Shift with the arrow keys
moves one hundredth.

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

The left pane colours each starting value by the root to which it converges.
Brightness encodes convergence speed. Clicking the overview selects `z₀` and
opens a linked region in the right pane, where convergence time is emphasized
to expose sensitive basin boundaries. Image colours use a continuous estimate
of where the residual crossed the convergence threshold, avoiding false bands
from whole iteration counts. The CPU `f64` diagnostic independently reports
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
Bounded orbits are shaded by the smallest modulus they reach (**Display →
Interior shading**), which reveals the basins of attracting cycles; switch it
off to recover the uniform dark interior.

Precision handling is deliberately explicit. Once the `f32` coordinate grid
becomes coarser than a pixel, the non-quadratic families render by GPU
perturbation around a reference orbit of the view centre: below the
`1.14e14×` handoff the reference is computed on the CPU in `f64`
(`F64 PERT`), beyond it in arbitrary precision (`AP PERT`). In both cases
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
automatically to GPU double-single when coordinate resolution is at risk or
the lightweight orbit probe detects divergent escape behaviour. The
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

At extreme depth, a uniformly dark view (or, with interior shading enabled, a
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

## Experiment documents

Open **Document → Export / Import JSON** to capture the complete reproducible
experiment state. The versioned document (format version 4) records the
active family, both plane centres and scales, the selected parameter or
starting value, the family parameters the active family uses, computation
limits, and display settings, including critical-orbit overlay visibility. Runtime
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

## Project status

IteraScope is an executable scientific prototype. Its numerical results are
exploratory, not certified. Precision modes, thresholds and algorithmic
assumptions remain visible so a plausible-looking image is not silently
presented as a trustworthy computation.

## License

IteraScope is open-source software released under the [MIT License](LICENSE).
