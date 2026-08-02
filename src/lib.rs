//! IteraScope — an open laboratory for complex dynamics.
//!
//! The initial instrument links the parameter and dynamical planes of the
//! quadratic family. The fractals run in WGSL on the same WebGPU device egui
//! uses, while the CPU only maintains the view and experiment parameters.

mod app;
mod experiment;
mod orbit;
mod perturbation;
mod precision;
pub mod render;

pub use app::App;
