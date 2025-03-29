use base64::prelude::*;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Result, Tensor};
use std::io::Cursor;

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!("Running on CPU, to run on GPU(metal), build this example with `--features metal`");
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

/// Converts a tensor with shape (3, height, width) into a base64-encoded PNG.
pub fn tensor_to_base64_png(img: &Tensor) -> Result<String> {
    let (channels, height, width) = img.dims3()?;
    if channels != 3 {
        candle_core::bail!("tensor_to_base64_png expects an image with 3 channels");
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let buffer = image::ImageBuffer::from_raw(width as u32, height as u32, pixels)
        .ok_or_else(|| candle_core::Error::msg("error converting tensor to image buffer"))?;
    let dyn_img = image::DynamicImage::ImageRgb8(buffer);
    let mut bytes = Vec::new();
    dyn_img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(candle_core::Error::wrap)?;
    Ok(BASE64_STANDARD.encode(&bytes))
}
