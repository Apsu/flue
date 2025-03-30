use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::post,
    Router,
};
use base64::{prelude::BASE64_STANDARD, Engine};
use flue_core::{DeviceMap, FluxLoader, GenerationRequest, Loader, ModelLike};
use hf_hub::api::tokio::Api;
use image::DynamicImage;
use serde::Serialize;
use std::{
    io::Cursor,
    sync::{Arc, Mutex},
};
use tokio::{self, net::TcpListener};

/// Converts a tensor with shape (3, height, width) into a base64-encoded PNG.
fn image_to_base64_png(img: &DynamicImage) -> Result<String> {
    let mut bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(candle_core::Error::wrap)?;
    Ok(BASE64_STANDARD.encode(&bytes))
}

#[derive(Serialize)]
struct GenerationResponse {
    image: String,
}

// Application state containing the preloaded models and device settings.
#[derive(Clone)]
struct AppState(Arc<Mutex<dyn ModelLike>>);

async fn generate_image_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerationRequest>,
) -> impl IntoResponse {
    match generate_image(req, &state).await {
        Ok(img_base64) => Json(GenerationResponse { image: img_base64 }).into_response(),
        Err(e) => {
            eprintln!("Error generating image: {:?}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {:?}", e)).into_response()
        }
    }
}

/// This function uses the preloaded models from `state` to generate an image (base64).
async fn generate_image(params: GenerationRequest, state: &AppState) -> Result<String> {
    let image = state.0.lock().unwrap().run(params)?;
    image_to_base64_png(&image)
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = FluxLoader::load(Api::new()?, DeviceMap::default()).await?;

    // Build application state and wrap in Arc.
    let app_state = AppState(Arc::new(Mutex::new(model)));
    let shared_state = Arc::new(app_state);

    // --- Build axum router with shared state ---
    let app = Router::new()
        .route("/v1/images/generations", post(generate_image_handler))
        .with_state(shared_state);

    let listener = TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("Starting server on {}", listener.local_addr().unwrap());
    axum::serve(listener, app.into_make_service()).await?;

    Ok(())
}
