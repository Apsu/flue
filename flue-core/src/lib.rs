pub mod device_map;
pub mod loader;
mod util;

mod clip;
mod flux;
mod t5;

pub use device_map::*;
pub use flux::FluxLoader;
use image::DynamicImage;
pub use loader::*;
use serde::{Deserialize, Serialize};
pub(crate) use util::*;

// Define the request/response types.
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq, PartialOrd)]
pub struct GenerationRequest {
    pub prompt: String,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub steps: Option<usize>,
    pub guidance: Option<f64>,
    pub seed: Option<u64>,
    // Other parameters (quantized, model variant, cpu) can be added as needed.
}

pub trait ModelLike: Send + Sync {
    fn run(&mut self, request: GenerationRequest) -> anyhow::Result<DynamicImage>;
}
