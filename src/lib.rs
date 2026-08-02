//! IteraScope — an open laboratory for complex dynamics.
//!
//! The initial instrument links the parameter and dynamical planes of the
//! quadratic family. The fractals run in WGSL on the same WebGPU device egui
//! uses, while the CPU only maintains the view and experiment parameters.

mod app;
pub mod arbitrary;
mod experiment;
mod orbit;
mod precision;
pub mod render;

pub(crate) const MAX_ITERATIONS: u32 = 50_000;

pub use app::App;
