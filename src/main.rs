use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Router
};
use candle_core::{DType, Device, Error, IndexOp};
use candle_nn::Module;
use candle_transformers::models::{
    clip::text_model::{self, ClipTextTransformer},
    flux::{self, autoencoder::AutoEncoder, model::{self, Flux}},
    t5::{self, T5EncoderModel},
    };
use tokenizers::Tokenizer;
use tokio::{
    net::TcpListener,
    self,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

// Define the request/response types.
#[derive(Deserialize)]
struct GenerationRequest {
    prompt: String,
    width: Option<usize>,
    height: Option<usize>,
    steps: Option<usize>,
    guidance: Option<f64>,
    seed: Option<u64>,
    // Other parameters (quantized, model variant, cpu) can be added as needed.
}

#[derive(Serialize)]
struct GenerationResponse {
    image: String,
}

// Application state containing the preloaded models and device settings.
#[derive(Clone)]
struct AppState {
    device: Device,
    dtype: DType,
    t5_model: Arc<Mutex<T5EncoderModel>>,
    t5_tokenizer: Tokenizer,
    clip_model: Arc<Mutex<ClipTextTransformer>>,
    clip_tokenizer: Tokenizer,
    autoencoder: Arc<Mutex<AutoEncoder>>,
    flux_model: Arc<Mutex<Flux>>,
}

async fn generate_image_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerationRequest>,
) -> impl IntoResponse {
    match generate_image(req, &state) {
        Ok(img_base64) => Json(GenerationResponse { image: img_base64 }).into_response(),
        Err(e) => {
            eprintln!("Error generating image: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {:?}", e)).into_response()
        }
    }
}

/// This function uses the preloaded models from `state` to generate an image.
fn generate_image(
    params: GenerationRequest,
    state: &AppState,
) -> Result<String, Box<dyn std::error::Error>> {
    // Set defaults.
    let width = params.width.unwrap_or(1360);
    let height = params.height.unwrap_or(768);
    let steps = params.steps.unwrap_or(4);
    let guidance = params.guidance.unwrap_or(0.0);

    // Optionally set seed for reproducibility.
    if let Some(seed) = params.seed {
        state.device.set_seed(seed)?;
    }

    // --- Generate noise image ---
    let noise_img = flux::sampling::get_noise(1, height, width, &state.device)?
        .to_dtype(state.dtype)?;

    // --- Compute T5 embedding using the preloaded T5 model and tokenizer ---
    let mut t5_tokens = state
        .t5_tokenizer
        .encode(params.prompt.as_str(), true)
        .map_err(Error::msg)? // Encode the prompt
        .get_ids()
        .to_vec();
    t5_tokens.resize(256, 0);
    let input_token_ids = candle_core::Tensor::new(&*t5_tokens, &state.device)?.unsqueeze(0)?;
    let t5_emb = state.t5_model.lock().unwrap().forward(&input_token_ids)?;

    // --- Compute CLIP embedding using the preloaded CLIP model and tokenizer ---
    let clip_tokens = state
        .clip_tokenizer
        .encode(params.prompt.as_str(), true)
        .map_err(Error::msg)? // Encode the prompt
        .get_ids()
        .to_vec();
    let input_token_ids_clip =
        candle_core::Tensor::new(&*clip_tokens, &state.device)?.unsqueeze(0)?;
    let clip_emb = state.clip_model.lock().unwrap().forward(&input_token_ids_clip)?;

    // --- Create sampling state and schedule ---
    let sampling_state = flux::sampling::State::new(&t5_emb, &clip_emb, &noise_img)?;
    let timesteps = flux::sampling::get_schedule(steps, None);

    // --- Run denoising via the preloaded Flux model ---
    let latent_img = flux::sampling::denoise(
        &*state.flux_model.lock().unwrap(),
        &sampling_state.img,
        &sampling_state.img_ids,
        &sampling_state.txt,
        &sampling_state.txt_ids,
        &sampling_state.vec,
        &timesteps,
        guidance,
    )?;

    let unpacked = flux::sampling::unpack(&latent_img, height, width)?;
    println!("Generated latent image");

    // --- Decode the latent image using the preloaded autoencoder ---
    let decoded = state.autoencoder.lock().unwrap().decode(&unpacked)?;
    println!("Decoded image");

    // --- Postprocessing: clamp, scale, convert type, and convert to base64 PNG ---
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?
        .to_dtype(DType::U8)?;
    let img_tensor = img.i(0)?;
    let img_base64 = flue::tensor_to_base64_png(&img_tensor)?;
    Ok(img_base64)
}

#[tokio::main]
async fn main() {
    #[cfg(feature = "cuda")]
    {
        candle_core::quantized::cuda::set_force_dmmv(false);
        candle_core::quantized::cuda::set_memory_pool(true, 20 << 30); // Enable CUDA memory pool if using CUDA
    }
    // --- Load models once at startup ---

    // Create the HF hub API instance.
    let api = hf_hub::api::sync::Api::new().expect("failed to create hf hub API");

    // Configure device.
    let cpu = false; // change if needed
    let device = flue::device(cpu).expect("failed to set up device");
    let dtype = device.bf16_default_to_f32();

    // --- Load T5 Model and Tokenizer ---
    let t5_repo = api.repo(hf_hub::Repo::with_revision(
        "google/t5-v1_1-xxl".to_string(),
        hf_hub::RepoType::Model,
        "refs/pr/2".to_string(),
    ));
    let t5_model_file = t5_repo.get("model.safetensors").expect("failed to load T5 model file");
    let t5_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[t5_model_file], dtype, &device)
            .expect("failed to build T5 var builder")
    };
    let config_filename = t5_repo.get("config.json").expect("failed to get T5 config");
    let config_str = std::fs::read_to_string(&config_filename).expect("failed to read T5 config");
    let t5_config: t5::Config =
        serde_json::from_str(&config_str).expect("failed to parse T5 config");
    let t5_model =
        T5EncoderModel::load(t5_vb, &t5_config)
            .expect("failed to load T5 model");
    let t5_tokenizer_filename = api
        .model("lmz/mt5-tokenizers".to_string())
        .get("t5-v1_1-xxl.tokenizer.json")
        .expect("failed to get T5 tokenizer");
    let t5_tokenizer = tokenizers::Tokenizer::from_file(t5_tokenizer_filename)
        .expect("failed to load T5 tokenizer");

    // --- Load CLIP Model and Tokenizer ---
    let clip_repo = api
        .repo(hf_hub::Repo::model("openai/clip-vit-large-patch14".to_string()));
    let clip_model_file = clip_repo
        .get("model.safetensors")
        .expect("failed to get CLIP model file");
    let clip_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[clip_model_file], dtype, &device)
            .expect("failed to build CLIP var builder")
    };
    let clip_config = text_model::ClipTextConfig {
        vocab_size: 49408,
        projection_dim: 768,
        activation: text_model::Activation::QuickGelu,
        intermediate_size: 3072,
        embed_dim: 768,
        max_position_embeddings: 77,
        pad_with: None,
        num_hidden_layers: 12,
        num_attention_heads: 12,
    };
    let clip_model = ClipTextTransformer::new(
        clip_vb.pp("text_model"),
        &clip_config,
    )
    .expect("failed to load CLIP model");
    let clip_tokenizer_filename = clip_repo
        .get("tokenizer.json")
        .expect("failed to get CLIP tokenizer");
    let clip_tokenizer = tokenizers::Tokenizer::from_file(clip_tokenizer_filename)
        .expect("failed to load CLIP tokenizer");

    // --- Load Autoencoder ---
    let bf_repo = {
        let name = "black-forest-labs/FLUX.1-schnell";
        api.repo(hf_hub::Repo::model(name.to_string()))
    };
    let autoencoder_model_file = bf_repo
        .get("ae.safetensors")
        .expect("failed to get autoencoder model file");
    let autoencoder_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[autoencoder_model_file], dtype, &device)
            .expect("failed to build autoencoder var builder")
    };
    let autoencoder_config = flux::autoencoder::Config::schnell();
    let autoencoder = AutoEncoder::new(&autoencoder_config, autoencoder_vb)
        .expect("failed to load autoencoder");

    // --- Load Flux Model (non-quantized) ---
    let flux_model_file = bf_repo
        .get("flux1-schnell.safetensors")
        .expect("failed to get flux model file");
    let flux_vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[flux_model_file], dtype, &device)
            .expect("failed to build flux var builder")
    };
    let flux_config = model::Config::schnell();
    let flux_model =
        Flux::new(&flux_config, flux_vb).expect("failed to load flux model");

    // Build application state and wrap in Arc.
    let app_state = AppState {
        device,
        dtype,
        t5_model: Arc::new(Mutex::new(t5_model)),
        t5_tokenizer,
        clip_model: Arc::new(Mutex::new(clip_model)),
        clip_tokenizer,
        autoencoder: Arc::new(Mutex::new(autoencoder)),
        flux_model: Arc::new(Mutex::new(flux_model)),
    };
    let shared_state = Arc::new(app_state);

    // --- Build axum router with shared state ---
    let app = Router::new()
        .route("/v1/images/generations", post(generate_image_handler))
        .with_state(shared_state);

    let listener = TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("Starting server on {}", listener.local_addr().unwrap());
    axum::serve(listener, app.into_make_service())
        .await
        .unwrap();
}
