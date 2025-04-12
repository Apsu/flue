# Flue: Fast Lightweight Unified Engine for Image Diffusion
Flue is a high-performance API server designed for efficiently loading and running Text-to-Image diffusion models. Built in Rust, Flue leverages the HuggingFace Candle, Transformers, and Hub libraries, delivering fast, scalable, and optimized model inference.

## Features
- Unified API: Easily load and serve multiple diffusion models through a consistent interface.
- GPU Acceleration: Supports single and multi-GPU deployments across one or multiple nodes, leveraging CUDA and Metal.
- Optimized Pipelines:
    - FlashAttention
	- Quantization
	- Compiled computation graphs
	- Advanced parallelization techniques for maximum throughput and efficiency.
- Model Support: Currently supports Flux model family, with planned expansions to additional Candle-supported diffusion models.

## Installation

Prerequisites
- Rust
- CUDA toolkit (if using NVIDIA GPUs)
- Metal (if deploying on Apple hardware)

### Quickstart

Clone and build Flue:
```sh
git clone https://github.com/Apsu/flue.git
cd flue
cargo build --release --features cuda
```

> Note: `cudnn` feature can be added if you have it installed for potential improved performance

### Run the API server:

```sh
./target/release/flue-server
```

## Configuration

Flue accepts command-line arguments to configure the engine.

```sh
--model "path or HF repo"   # Default: black-forest-labs/FLUX.1-schnell
--host 0.0.0.0              # Default 127.0.0.1
--port 1234                 # Default: 8080
--cpu                       # Force CPU offloading
```

> Note: CPU offloading is currently all-or-nothing, and takes a lot of RAM and CPU for even a small request!

## Usage

Send an HTTP POST request to generate images:

```sh
curl -X POST http://localhost:8080/v1/images/generations \
-H "Content-Type: application/json" \
-d '{
  "prompt": "A futuristic city skyline at sunset",
  "width": 512,
  "height": 512,
  "steps": 4,
  "guidance_scale": 7.5
}'
```

> Note: FLUX.1-schnell ignores guidance_scale and requires less steps than FLUX.1-dev

## Roadmap

- ✅ Flux models support
- ✅ FlashAttention integration
- ⬜ Broader model compatibility (Stable Diffusion, Hunyuan, etc.)
- ⬜ Distributed inference across multiple GPUs/nodes
- ⬜ Parallel optimization methods like PipeFusion, Ulysses, Ring, and DistVAE
- ⬜ Graph compilation and save/load
- ⬜ Advanced caching and resource management

## Contributing

Contributions are welcome! Please open issues and pull requests directly on this repository.

## License

Flue is licensed under [MIT](LICENSE).
