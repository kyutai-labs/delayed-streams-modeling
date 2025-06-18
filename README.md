# delayed-streams-modeling
Delayed Streams Modeling (DSM) is a flexible formulation for streaming, multimodal sequence-to-sequence learning.

## Speech-to-text

DSM can be used to build streaming speech-to-text models. These models can be
batched for efficiency, return word level timestamps,  and are great for
interactive applications. We provide two such models, these models are
characterized by their size as well as the delay it takes for audio to be
transcribed into text. We provide two such models:
- An English and French model with ~1b parameters using a 0.5 second delay,
  `kyutai/stt-1b-en_fr`.
- An English only model with ~2.6b parameters using a 2.5 second delay,
  `kyutai/stt-2.6b-en`.

More details can be found on the [project page](https://kyutai.org/next/stt).

### PyTorch implementation
<a href="https://huggingface.co/kyutai/stt-2.6b-en" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>
<a target="_blank" href="https://colab.research.google.com/drive/1mc0Q-FoHxU2pEvId8rTdS4q1r1zorJhS?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This requires the [moshi package](https://pypi.org/project/moshi/)
with version 0.2.5 or later, which can be installed via pip.

```bash
# wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
python -m moshi.run_inference --hf-repo kyutai/stt-2.6b-en bria.mp3
```

If you have `uv` installed, you can skip the installation step and run directly:
```bash
uvx --with moshi python -m moshi.run_inference --hf-repo kyutai/stt-2.6b-en bria.mp3
```
It will install the moshi package in a temporary environment and run the speech-to-text.

### MLX implementation
<a href="https://huggingface.co/kyutai/stt-2.6b-en-mlx" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

This requires the [moshi-mlx package](https://pypi.org/project/moshi-mlx/)
with version 0.2.5 or later, which can be installed via pip.

```bash
# wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
python -m moshi_mlx.run_inference --hf-repo kyutai/stt-2.6b-en-mlx bria.mp3 --temp 0
```

If you have `uv` installed, you can skip the installation step and run directly:
```bash
uvx --with moshi-mlx python -m moshi_mlx.run_inference --hf-repo kyutai/stt-2.6b-en-mlx bria.mp3 --temp 0

```
It will install the moshi package in a temporary environment and run the speech-to-text.

### Rust implementation
<a href="https://huggingface.co/kyutai/stt-2.6b-en-candle" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

A standalone Rust example is provided in the `stt-rs` directory in this repo.
This can be used as follows:
```bash
cd stt-rs
cargo run --features cuda -r -- bria.mp3
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
```bash
moshi-server worker --config configs/config-stt-hf.toml
```

Once the server has started you can run a streaming inference with the following
script.
```bash
# wget https://github.com/metavoiceio/metavoice-src/raw/main/assets/bria.mp3
uv run scripts/asr-streaming-query.py bria.mp3
```

The script simulates some real-time processing of the audio. Faster processing
can be triggered by setting the real-time factor, e.g. `--rtf 500` will process
the data as fast as possible.

## Text-to-Speech

We're in the process of open-sourcing our TTS models. Check back for updates!

## License

The present code is provided under the MIT license for the Python parts, and Apache license for the Rust backend.
The web client code is provided under the MIT license.
Note that parts of this code is based on [AudioCraft](https://github.com/facebookresearch/audiocraft), released under
the MIT license.

The weights for the speech-to-text models are released under the CC-BY 4.0 license.
