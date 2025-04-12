use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Router,
};
use base64::{prelude::BASE64_STANDARD, Engine};
use clap::Parser;
use flue_core::{load_model, DeviceMap, GenerationRequest, ModelLike};
use hf_hub::api::tokio::Api;
use image::DynamicImage;
use serde::Serialize;
use std::{io::Cursor, sync::Arc};
use tokio::{self, net::TcpListener};

// Define command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about = "Flue image generation server")]
struct Args {
    /// Use CPU instead of GPU
    #[arg(long)]
    cpu: bool,

    /// Model to use
    #[arg(long, default_value = "black-forest-labs/FLUX.1-schnell")]
    model: String,

    /// Host address to bind the server to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to bind the server to
    #[arg(long, default_value_t = 8000)]
    port: u16,
}

/// Converts a tensor with shape (3, height, width) into a base64-encoded PNG.
fn image_to_base64_png(img: &DynamicImage) -> Result<String> {
    let mut bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)?;
    Ok(BASE64_STANDARD.encode(&bytes))
}

#[derive(Serialize)]
struct GenerationResponse {
    image: String,
    gen_time: f64, // Time in seconds
}

// Application state containing the preloaded models and device settings.
#[derive(Clone)]
struct AppState(Arc<dyn ModelLike>);

async fn generate_image_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerationRequest>,
) -> impl IntoResponse {
    match generate_image(req, &state).await {
        Ok((img_base64, gen_time)) => Json(GenerationResponse {
            image: img_base64,
            gen_time,
        })
        .into_response(),
        Err(e) => {
            eprintln!("Error generating image: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {:?}", e)).into_response()
        }
    }
}

/// This function uses the preloaded models from `state` to generate an image (base64).
/// Returns both the base64 image and the generation time in seconds.
async fn generate_image(params: GenerationRequest, state: &AppState) -> Result<(String, f64)> {
    let start_time = std::time::Instant::now();
    let image = state.0.run(params)?;
    let gen_time = start_time.elapsed().as_secs_f64();
    let base64_image = image_to_base64_png(&image)?;
    println!("Generation time: {:.2} seconds", gen_time);
    Ok((base64_image, gen_time))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Use device map based on CPU flag
    let device_map = if args.cpu {
        println!("Using CPU for inference");
        DeviceMap::ForceCpu
    } else {
        DeviceMap::default()
    };

    // Use the factory to load the appropriate model
    let model = load_model(&args.model, Api::new()?, device_map).await?;

    // Build application state and wrap in Arc.
    let app_state = AppState(model);
    let shared_state = Arc::new(app_state);

    // --- Build axum router with shared state ---
    let app = Router::new()
        .route("/v1/images/generations", post(generate_image_handler))
        .with_state(shared_state);

    // --- Start the server ---
    let bind_address = format!("{}:{}", args.host, args.port);
    let listener = TcpListener::bind(&bind_address).await.unwrap();
    println!("Started server on {}", listener.local_addr().unwrap());
    axum::serve(listener, app.into_make_service()).await?;

    Ok(())
}
