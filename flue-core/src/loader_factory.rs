use anyhow::{anyhow, Result};
use hf_hub::api::tokio::Api;

use crate::{DeviceMap, FluxLoader, Loader, ModelLike};
use std::sync::Arc;

use crate::flux;

/// Enum of supported model types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Flux,
    T5,
    Clip,
    // Add more model types as they become available
}

impl ModelType {
    /// Detect model type from model name
    pub fn from_name(model_name: &str) -> Option<Self> {
        let name_upper = model_name.to_uppercase();

        if name_upper.contains("FLUX") {
            Some(ModelType::Flux)
        } else if name_upper.contains("T5") {
            Some(ModelType::T5)
        } else if name_upper.contains("CLIP") {
            Some(ModelType::Clip)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub enum ModelVariant {
    Flux(flux::FluxVariant),
    // T5(t5::T5Variant),
    // Clip(clip::ClipVariant),
}

impl ModelVariant {
    /// Detect model variant from model name
    pub fn from_name(model_name: &str) -> Option<Self> {
        let name_upper = model_name.to_uppercase();

        if name_upper.contains("FLUX") {
            Some(ModelVariant::Flux(if name_upper.contains("SCHNELL") {
                flux::FluxVariant::Schnell
            } else if name_upper.contains("DEV") {
                flux::FluxVariant::Dev
            } else {
                flux::FluxVariant::Schnell // Default to Schnell if no specific variant is found
            }))
        } else {
            None
        }
    }
}

/// Load a model based on its name, automatically detecting the appropriate loader
pub async fn load_model(
    model_name: &str,
    api: Api,
    device_map: DeviceMap,
) -> Result<Arc<dyn ModelLike>> {
    // Get model type and variant or return error if unsupported
    let model_type = ModelType::from_name(model_name)
        .ok_or_else(|| anyhow!("Unsupported model type: {}", model_name))?;
    let model_variant = ModelVariant::from_name(model_name)
        .ok_or_else(|| anyhow!("Unsupported model variant: {}", model_name))?;

    println!(
        "Loading model: {} (detected type: {:?}/variant: {:?})",
        model_name, model_type, model_variant
    );

    match model_type {
        ModelType::Flux => {
            let model = FluxLoader::load(model_variant, api, device_map).await?;
            Ok(Arc::new(model))
        }
        // Uncomment as other loaders are implemented
        // ModelType::T5 => {
        //     let model = T5Loader::load(api, device_map).await?;
        //     Ok(Arc::new(model))
        // },
        // ModelType::Clip => {
        //     let model = ClipLoader::load(api, device_map).await?;
        //     Ok(Arc::new(model))
        // },
        _ => Err(anyhow!(
            "Model type {:?}/variant {:?} is recognized but not yet implemented",
            model_type,
            model_variant
        )),
    }
}
