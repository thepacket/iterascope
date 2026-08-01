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
- Cursor-centred zoom and drag navigation in either plane
- Adjustable iteration limit, bailout, palette phase and smooth colouring
- Optional scale-aware coordinate grid
- Live coordinate and magnification readouts
- Responsive side-by-side or stacked layout
- Native and WASM entry points from the same codebase
- Offline WGSL parsing and validation using the exact Naga version used by wgpu

This first rendering path is intentionally `f32`. The interface reports that
fact instead of implying precision it does not yet provide. Double-single
arithmetic, perturbation and reference-orbit infrastructure are subsequent
rendering stages.

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

## Validate

```sh
cargo check
cargo test
cargo check --target wasm32-unknown-unknown
```

## Near-term roadmap

1. Deterministic experiment documents and view import/export
2. Orbit inspector and critical-orbit diagnostics
3. Double-single GPU arithmetic for the intermediate zoom range
4. Perturbation with a small arbitrary-precision CPU reference orbit
5. Period detection, multipliers and Newton refinement

## Project status

IteraScope is at the first executable-prototype stage. Its numerical results
are exploratory, not yet certified. Precision mode, thresholds and algorithmic
assumptions will remain visible as the scientific toolset grows.

## License

IteraScope is open-source software released under the [MIT License](LICENSE).
