<a href="https://huggingface.co/collections/kyutai/speech-to-text-685403682cf8a23ab9466886" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KyutaiSTT-blue" style="display: inline-block; vertical-align: middle;"/>
</a>
<a target="_blank" href="https://colab.research.google.com/drive/1mc0Q-FoHxU2pEvId8rTdS4q1r1zorJhS?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


This repo contains instructions and examples of how to run Kyutai Speech-To-Text models.
These models are powered by delayed streams modeling (DSM),
a flexible formulation for streaming, multimodal sequence-to-sequence learning.

Text-to-speech models based on DSM coming soon!
[Sign up here](https://docs.google.com/forms/d/15sB4zyfuwyXTii4OM74hFGkk4DlDNynJ9xywnaEzE4I/edit)
to be notified when we open-source text-to-speech and [Unmute](https://unmute.sh).

## Kyutai Speech-To-Text

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

You can retrieve the sample files used in the following snippets via:
```bash
wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
wget https://github.com/kyutai-labs/moshi/raw/refs/heads/main/data/sample_fr_hibiki_crepes.mp3
```

### PyTorch implementation
<a href="https://huggingface.co/kyutai/stt-2.6b-en" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>
<a target="_blank" href="https://colab.research.google.com/drive/1mc0Q-FoHxU2pEvId8rTdS4q1r1zorJhS?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This requires the [moshi package](https://pypi.org/project/moshi/)
with version 0.2.6 or later, which can be installed via pip.

```bash
python -m moshi.run_inference --hf-repo kyutai/stt-2.6b-en bria.mp3
```

If you have [uv](https://docs.astral.sh/uv/) installed, you can skip the installation step and run directly:
```bash
uvx --with moshi python -m moshi.run_inference --hf-repo kyutai/stt-2.6b-en bria.mp3
```
It will install the moshi package in a temporary environment and run the speech-to-text.

Additionally, we provide two scripts that highlight different usage scenarios. The first script illustrates how to extract word-level timestamps from the model's outputs:

```bash
uv run \
  scripts/streaming_stt_timestamps.py \
  --hf-repo kyutai/stt-2.6b-en \
  --file bria.mp3
```

The second script can be used to run a model on an existing Hugging Face dataset and calculate its performance metrics: 
```bash
uv run scripts/streaming_stt.py  \
  --dataset meanwhile  \
  --hf-repo kyutai/stt-2.6b-en
```

### Rust server
<a href="https://huggingface.co/kyutai/stt-2.6b-en-candle" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

The Rust implementation provides a server that can process multiple streaming
queries in parallel. Dependening on the amount of memory on your GPU, you may
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

Then the server can be started via the following command using the config file
from this repository.
For `kyutai/stt-1b-en_fr`, use `configs/config-stt-en_fr.hf.toml`,
and for `kyutai/stt-2.6b-en`, use `configs/config-stt-en-hf.toml`,

```bash
moshi-server worker --config configs/config-stt-en_fr-hf.toml
```

Once the server has started you can run a streaming inference with the following
script.
```bash
uv run scripts/asr-streaming-query.py bria.mp3
```

The script limits the decoding speed to simulates real-time processing of the audio. 
Faster processing can be triggered by setting 
the real-time factor, e.g. `--rtf 500` will process
the data as fast as possible.

### Rust standalone
<a href="https://huggingface.co/kyutai/stt-2.6b-en-candle" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

A standalone Rust example script is provided in the `stt-rs` directory in this repo.
This can be used as follows:
```bash
cd stt-rs
cargo run --features cuda -r -- bria.mp3
```
You can get the timestamps by adding the `--timestamps` flag, and see the output
of the semantic VAD by adding the `--vad` flag.

### MLX implementation
<a href="https://huggingface.co/kyutai/stt-2.6b-en-mlx" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is Apple's ML framework that allows you to use
hardware acceleration on Apple silicon.

This requires the [moshi-mlx package](https://pypi.org/project/moshi-mlx/)
with version 0.2.6 or later, which can be installed via pip.

```bash
python -m moshi_mlx.run_inference --hf-repo kyutai/stt-2.6b-en-mlx bria.mp3 --temp 0
```

If you have [uv](https://docs.astral.sh/uv/) installed, you can skip the installation step and run directly:
```bash
uvx --with moshi-mlx python -m moshi_mlx.run_inference --hf-repo kyutai/stt-2.6b-en-mlx bria.mp3 --temp 0
```
It will install the moshi package in a temporary environment and run the speech-to-text.

The MLX models can also be used in swift using the [moshi-swift
codebase](https://github.com/kyutai-labs/moshi-swift), the 1b model has been
tested to work fine on an iPhone 16 Pro.

## Text-to-Speech

We're in the process of open-sourcing our TTS models. Check back for updates!

## License

The present code is provided under the MIT license for the Python parts, and Apache license for the Rust backend.
The web client code is provided under the MIT license.
Note that parts of this code is based on [AudioCraft](https://github.com/facebookresearch/audiocraft), released under
the MIT license.

The weights for the speech-to-text models are released under the CC-BY 4.0 license.
