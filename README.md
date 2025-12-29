# Delayed Streams Modeling: Kyutai STT & TTS

This repo contains instructions and examples of how to run
[Kyutai Speech-To-Text](#kyutai-speech-to-text)
and [Kyutai Text-To-Speech](#kyutai-text-to-speech) models.
See also [Unmute](https://github.com/kyutai-labs/unmute), a voice AI system built using Kyutai STT and Kyutai TTS.

But wait, what is "Delayed Streams Modeling"? It is a technique for solving many streaming X-to-Y tasks (with X, Y in `{speech, text}`)
that formalize the approach we had with Moshi and Hibiki. See our [pre-print about DSM](https://arxiv.org/abs/2509.08753).

## Directory Structure

This repository is organized by component type and programming language:

```
delayed-streams-modeling/
├── server/              # Server/backend components
│   ├── rust/            # Rust server code
│   │   └── moshi/       # Moshi server workspace
│   ├── typescript/      # TypeScript server code
│   │   └── auth-server/ # Authentication server
├── client/              # Client/frontend components
│   └── rust/            # Rust client applications
│       ├── kyutai-client-core/ # Shared auth/WebSocket helpers
│       ├── kyutai-stt-client/  # STT client library
│       ├── kyutai-stt-cli/     # STT CLI
│       └── tts-rs/             # TTS standalone client
├── tools/               # Development tools
│   ├── bf16-to-fp16/    # Checkpoint conversion helper
│   ├── gpu-check/       # GPU capability inspector
│   ├── log-formatter/   # Log cleanup and normalization
│   ├── quant-bench/     # Quantization benchmarking (Rust)
│   ├── s3-upload/       # Log upload helper
│   ├── sm75-prep/       # Pre-Ampere checkpoint prep
│   ├── smoke-test/      # Smoke testing utilities
│   └── token-gen/       # JWT token generator
├── configs/             # Configuration files
│   ├── stt/             # STT server configs
│   ├── tts/             # TTS server configs
│   └── models/          # Model JSON presets
├── ops/                 # Operational and deployment scripts
├── docs/                # Documentation
└── audio/               # Sample audio files
```

## CUDA 13.1 build shim

If you are using CUDA 13.1, `cudarc` only recognizes toolkits up to 12.9. Use the shim below to spoof the version and set CUDA paths for local builds:

```bash
source ops/setup_env.sh
cargo check -p moshi-server --features cuda
```

For convenience:

```bash
ops/check_cuda.sh
```

`ops/run-moshi-server.sh` will source the shim automatically when it detects CUDA 13.1.

## Kyutai Speech-To-Text

<a href="https://huggingface.co/collections/kyutai/speech-to-text-685403682cf8a23ab9466886" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KyutaiSTT-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

**More details can be found on the [project page](https://kyutai.org/next/stt).**

Kyutai STT models are optimized for real-time usage, can be batched for efficiency, and return word level timestamps.
We provide two models:
- `kyutai/stt-1b-en_fr`, an English and French model with ~1B parameters, a 0.5 second delay, and a [semantic VAD](https://kyutai.org/next/stt#semantic-vad).
- `kyutai/stt-2.6b-en`, an English-only model with ~2.6B parameters and a 2.5 second delay.

These speech-to-text models have several advantages:
- Streaming inference: the models can process audio in chunks, which allows
  for real-time transcription, and is great for interactive applications.
- Easy batching for maximum efficiency: a H100 can process 400 streams in
  real-time.
- They return word-level timestamps.
- The 1B model has a semantic Voice Activity Detection (VAD) component that
  can be used to detect when the user is speaking. This is especially useful
  for building voice agents.

### Implementations overview

The Rust implementation is intended for production deployment. It provides
streaming access over websockets and is the same server used to run
[Unmute](https://unmute.sh/); on a L40S GPU, we can serve 64 simultaneous
connections at a real-time factor of 3x.

<details>
<summary>Rust server</summary>

<a href="https://huggingface.co/kyutai/stt-2.6b-en-candle" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

The Rust implementation provides a server that can process multiple streaming
queries in parallel. Depending on the amount of memory on your GPU, you may
have to adjust the batch size from the config file. For a L40S GPU, a batch size
of 64 works well and requests can be processed at 3x real-time speed.

In order to run the server, install the [moshi-server
crate](https://crates.io/crates/moshi-server) via the following command. The
server code can be found in the
[kyutai-labs/moshi](https://github.com/kyutai-labs/moshi/tree/main/rust/moshi-server)
repository.
```bash
cargo install --features cuda moshi-server
```

For detailed compilation instructions and troubleshooting for 8GB VRAM cards (e.g. RTX 2070), see [docs/MOSHI_SERVER_SETUP.md](docs/MOSHI_SERVER_SETUP.md).

Then the server can be started via the following command using the config file
from this repository. Configs live under `configs/stt/` (see `configs/README.md`).
For `kyutai/stt-1b-en_fr`, use `configs/stt/config-stt-en_fr-hf.toml`,
and for `kyutai/stt-2.6b-en`, use `configs/stt/config-stt-en-hf.toml`,

```bash
moshi-server worker --config configs/stt/config-stt-en_fr-hf.toml
```

Once the server has started, use the Rust clients in `client/rust` to connect.
</details>

<details>
<summary>Rust standalone</summary>
<a href="https://huggingface.co/kyutai/stt-2.6b-en-candle" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

A standalone Rust example is provided in `client/rust/kyutai-stt-cli`.
This can be used as follows:
```bash
cargo run -p kyutai-stt-cli -r -- --auth-token <JWT> file ../../../audio/bria.mp3
```
</details>


## Kyutai Text-to-Speech

<a href="https://huggingface.co/collections/kyutai/text-to-speech-6866192e7e004ed04fd39e29" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KyutaiTTS-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

**More details can be found on the [project page](https://kyutai.org/next/tts).**

We provide a Rust implementation for production. The Rust server provides
streaming access to the model over websockets and is the same server used to
run Unmute.

<details>
<summary>Rust server</summary>


The Rust implementation provides a server that can process multiple streaming
queries in parallel.

In order to run the server, install the [moshi-server
crate](https://crates.io/crates/moshi-server) via the following command:
```bash
cargo install --features cuda moshi-server
```

For detailed compilation instructions, see [docs/MOSHI_SERVER_SETUP.md](docs/MOSHI_SERVER_SETUP.md).

Once installed, the server can be started via the following command using the config file
from this repository.

```bash
moshi-server worker --config configs/tts/config-tts.toml
```

Once the server has started, use the Rust clients in `client/rust` to connect.

You can configure the server by modifying `configs/tts/config-tts.toml`. See comments in that file to see what options are available.
TTS configs live under `configs/tts/` (see `configs/README.md` for layout).
</details>

## FAQ

Checkout the [Frequently Asked Questions](FAQ.md) section before opening an issue.

## License

The present code is provided under the Apache license for the Rust backend.
The web client code is provided under the MIT license.
Note that parts of this code is based on [AudioCraft](https://github.com/facebookresearch/audiocraft), released under
the MIT license.

The weights for the speech-to-text models are released under the CC-BY 4.0 license.

## Citation

Please cite the following paper.
```
@techreport{kyutai2025streaming,
      title={Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling}, 
      author={Neil Zeghidour and Eugene Kharitonov and Manu Orsini and Václav Volhejn and Gabriel de Marmiesse and Edouard Grave and Patrick Pérez and Laurent Mazaré and Alexandre Défossez},
      year={2025},
      eprint={2509.08753},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.08753}, 
}
```
