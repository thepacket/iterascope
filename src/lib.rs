//! IteraScope — an open laboratory for complex dynamics.
//!
//! The reference instrument links the parameter and dynamical planes of the
//! quadratic family; `family.rs` catalogues the further escape-time,
//! convergence-time and root-finding instruments. The fractals run in WGSL on
//! the same WebGPU device egui uses, while the CPU maintains the view and
//! experiment parameters and provides `f64` reference diagnostics.
//! `colouring.rs` holds the colour stage — one cyclic gradient and
//! independent outside/inside colouring algorithms — whose settings and
//! rasterised gradient the shader reads alongside the view uniforms.

mod app;
pub mod arbitrary;
mod colouring;
mod experiment;
mod family;
mod newton;
mod orbit;
mod precision;
pub mod render;

pub(crate) const MAX_ITERATIONS: u32 = 50_000;

pub use app::App;
