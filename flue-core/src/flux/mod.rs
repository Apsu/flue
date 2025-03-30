use anyhow::{Context, Error, Result};
use autoencoder::AutoEncoder;
use candle_core::{DType, Device, IndexOp};
use candle_nn::Module;
use hf_hub::api::tokio::Api;
use image::DynamicImage;
use model::Flux;
use tokenizers::Tokenizer;

mod autoencoder;
mod model;
mod sampling;

use crate::{
    clip::{ClipTextConfig, ClipTextTransformer},
    select_best_device,
    t5::{self, T5EncoderModel},
    tensor_to_image, DeviceMap, GenerationRequest, Loader, ModelLike,
};

pub struct FluxModel {
    device: Device,
    dtype: DType,
    t5_model: T5EncoderModel,
    t5_tokenizer: Tokenizer,
    clip_model: ClipTextTransformer,
    clip_tokenizer: Tokenizer,
    autoencoder: AutoEncoder,
    flux_model: Flux,
}

impl ModelLike for FluxModel {
    fn run(&mut self, request: GenerationRequest) -> anyhow::Result<DynamicImage> {
        // Set defaults.
        let width = request.width.unwrap_or(1360);
        let height = request.height.unwrap_or(768);
        let steps = request.steps.unwrap_or(4);
        let guidance = request.guidance.unwrap_or(0.0);

        // Optionally set seed for reproducibility.
        if let Some(seed) = request.seed {
            self.device.set_seed(seed)?;
        }

        // --- Generate noise image ---
        let noise_img =
            sampling::get_noise(1, height, width, &self.device)?.to_dtype(self.dtype)?;

        // --- Compute T5 embedding using the preloaded T5 model and tokenizer ---
        let mut t5_tokens = self
            .t5_tokenizer
            .encode(request.prompt.as_str(), true)
            .map_err(Error::msg)? // Encode the prompt
            .get_ids()
            .to_vec();
        t5_tokens.resize(256, 0);
        let input_token_ids = candle_core::Tensor::new(&*t5_tokens, &self.device)?.unsqueeze(0)?;
        let t5_emb = self.t5_model.forward(&input_token_ids)?;

        // --- Compute CLIP embedding using the preloaded CLIP model and tokenizer ---
        let clip_tokens = self
            .clip_tokenizer
            .encode(request.prompt.as_str(), true)
            .map_err(Error::msg)? // Encode the prompt
            .get_ids()
            .to_vec();
        let input_token_ids_clip =
            candle_core::Tensor::new(&*clip_tokens, &self.device)?.unsqueeze(0)?;
        let clip_emb = self.clip_model.forward(&input_token_ids_clip)?;

        // --- Create sampling self and schedule ---
        let sampling_self = sampling::State::new(&t5_emb, &clip_emb, &noise_img)?;
        let timesteps = sampling::get_schedule(steps, None);

        // --- Run denoising via the preloaded Flux model ---
        let latent_img = sampling::denoise(
            &self.flux_model,
            &sampling_self.img,
            &sampling_self.img_ids,
            &sampling_self.txt,
            &sampling_self.txt_ids,
            &sampling_self.vec,
            &timesteps,
            guidance,
        )?;

        let unpacked = sampling::unpack(&latent_img, height, width)?;
        println!("Generated latent image");

        // --- Decode the latent image using the preloaded autoencoder ---
        let decoded = self.autoencoder.decode(&unpacked)?;
        println!("Decoded image");

        // --- Postprocessing: clamp, scale, convert type, and convert to base64 PNG ---
        let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
        let img_tensor = img.i(0)?;

        tensor_to_image(&img_tensor)
    }
}

pub struct FluxLoader;

impl Loader for FluxLoader {
    type Model = FluxModel;

    async fn load(api: Api, device_map: DeviceMap) -> Result<Self::Model> {
        // Configure device.
        let device = select_best_device(device_map).context("failed to set up device")?;
        let dtype = device.bf16_default_to_f32();

        // --- Load T5 Model and Tokenizer ---
        let t5_repo = api.repo(hf_hub::Repo::with_revision(
            "google/t5-v1_1-xxl".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/2".to_string(),
        ));
        let t5_model_file = t5_repo
            .get("model.safetensors")
            .await
            .context("failed to load T5 model file")?;
        let t5_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[t5_model_file], dtype, &device)
                .context("failed to build T5 var builder")?
        };
        let config_filename = t5_repo
            .get("config.json")
            .await
            .context("failed to get T5 config")?;
        let config_str =
            std::fs::read_to_string(&config_filename).context("failed to read T5 config")?;
        let t5_config: t5::Config =
            serde_json::from_str(&config_str).context("failed to parse T5 config")?;
        let t5_model =
            T5EncoderModel::load(t5_vb, &t5_config).context("failed to load T5 model")?;
        let t5_tokenizer_filename = api
            .model("lmz/mt5-tokenizers".to_string())
            .get("t5-v1_1-xxl.tokenizer.json")
            .await
            .context("failed to get T5 tokenizer")?;
        let t5_tokenizer = tokenizers::Tokenizer::from_file(t5_tokenizer_filename)
            .map_err(anyhow::Error::msg)
            .context("failed to load T5 tokenizer")?;

        // --- Load CLIP Model and Tokenizer ---
        let clip_repo = api.repo(hf_hub::Repo::model(
            "openai/clip-vit-large-patch14".to_string(),
        ));
        let clip_model_file = clip_repo
            .get("model.safetensors")
            .await
            .context("failed to get CLIP model file")?;
        let clip_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[clip_model_file], dtype, &device)
                .context("failed to build CLIP var builder")?
        };
        let clip_config = ClipTextConfig {
            vocab_size: 49408,
            projection_dim: 768,
            intermediate_size: 3072,
            embed_dim: 768,
            max_position_embeddings: 77,
            num_hidden_layers: 12,
            num_attention_heads: 12,
        };
        let clip_model = ClipTextTransformer::new(clip_vb.pp("text_model"), &clip_config)
            .context("failed to load CLIP model")?;
        let clip_tokenizer_filename = clip_repo
            .get("tokenizer.json")
            .await
            .context("failed to get CLIP tokenizer")?;
        let clip_tokenizer = tokenizers::Tokenizer::from_file(clip_tokenizer_filename)
            .map_err(anyhow::Error::msg)
            .context("failed to load CLIP tokenizer")?;

        // --- Load Autoencoder ---
        let bf_repo = {
            let name = "black-forest-labs/FLUX.1-schnell";
            api.repo(hf_hub::Repo::model(name.to_string()))
        };
        let autoencoder_model_file = bf_repo
            .get("ae.safetensors")
            .await
            .context("failed to get autoencoder model file")?;
        let autoencoder_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[autoencoder_model_file],
                dtype,
                &device,
            )
            .context("failed to build autoencoder var builder")?
        };
        let autoencoder_config = autoencoder::Config::schnell();
        let autoencoder = AutoEncoder::new(&autoencoder_config, autoencoder_vb)
            .context("failed to load autoencoder")?;

        // --- Load Flux Model (non-quantized) ---
        let flux_model_file = bf_repo
            .get("flux1-schnell.safetensors")
            .await
            .context("failed to get flux model file")?;
        let flux_vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[flux_model_file], dtype, &device)
                .context("failed to build flux var builder")?
        };
        let flux_config = model::Config::schnell();
        let flux_model = Flux::new(&flux_config, flux_vb).context("failed to load flux model")?;

        Ok(FluxModel {
            device,
            dtype,
            t5_model,
            t5_tokenizer,
            clip_model,
            clip_tokenizer,
            autoencoder,
            flux_model,
        })
    }
}
