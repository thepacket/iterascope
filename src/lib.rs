//! IteraScope — an open laboratory for complex dynamics.
//!
//! The initial instrument links the parameter and dynamical planes of the
//! quadratic family. The fractals run in WGSL on the same WebGPU device egui
//! uses, while the CPU only maintains the view and experiment parameters.

mod app;
pub mod render;

pub use app::App;
