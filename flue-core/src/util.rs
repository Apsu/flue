use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Tensor};
use image::DynamicImage;

use crate::DeviceMap;

pub fn select_best_device(device_map: DeviceMap) -> Result<Device> {
    match device_map {
        DeviceMap::ForceCpu => Ok(Device::Cpu),
        DeviceMap::Ordinal(ordinal) if cuda_is_available() => Ok(Device::new_cuda(ordinal)?),
        DeviceMap::Ordinal(ordinal) if metal_is_available() => Ok(Device::new_metal(ordinal)?),
        DeviceMap::Ordinal(_) => {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
            }
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                println!(
                    "Running on CPU, to run on GPU, build this example with `--features cuda`"
                );
            }
            Ok(Device::Cpu)
        }
    }
}

/// Converts a tensor with shape (3, height, width) into a base64-encoded PNG.
pub fn tensor_to_image(img: &Tensor) -> Result<DynamicImage> {
    let (channels, height, width) = img.dims3()?;
    if channels != 3 {
        anyhow::bail!("tensor_to_base64_png expects an image with 3 channels");
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let buffer = image::ImageBuffer::from_raw(width as u32, height as u32, pixels)
        .ok_or_else(|| candle_core::Error::msg("error converting tensor to image buffer"))?;
    Ok(DynamicImage::ImageRgb8(buffer))
}
