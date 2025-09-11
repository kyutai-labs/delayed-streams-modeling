# Delayed Streams Modeling: Kyutai STT & TTS

This repo contains instructions and examples of how to run
[Kyutai Speech-To-Text](#kyutai-speech-to-text)
and [Kyutai Text-To-Speech](#kyutai-text-to-speech) models.
See also [Unmute](https://github.com/kyutai-labs/unmute), an voice AI system built using Kyutai STT and Kyutai TTS.

But wait, what is "Delayed Streams Modeling"? It is a technique for solving many streaming X-to-Y tasks (with X, Y in `{speech, text}`)
that formalize the approach we had with Moshi and Hibiki. See our [pre-print about DSM](https://arxiv.org/abs/2509.08753).

## Kyutai Speech-To-Text

<a href="https://huggingface.co/collections/kyutai/speech-to-text-685403682cf8a23ab9466886" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KyutaiSTT-blue" style="display: inline-block; vertical-align: middle;"/>
</a>
<a target="_blank" href="https://colab.research.google.com/github/kyutai-labs/delayed-streams-modeling/blob/main/stt_pytorch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
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

We provide different implementations of Kyutai STT for different use cases.
Here is how to choose which one to use:

- **PyTorch: for research and tinkering.**
  If you want to call the model from Python for research or experimentation, use our PyTorch implementation.
- **Rust: for production.**
  If you want to serve Kyutai STT in a production setting, use our Rust server.
  Our robust Rust server provides streaming access to the model over websockets.
  We use this server to run [Unmute](https://unmute.sh/); on a L40S GPU, we can serve 64 simultaneous connections at a real-time factor of 3x.
- **MLX: for on-device inference on iPhone and Mac.**
  MLX is Apple's ML framework that allows you to use hardware acceleration on Apple silicon.
  If you want to run the model on a Mac or an iPhone, choose the MLX implementation.

<details>
<summary>PyTorch implementation</summary>
<a href="https://huggingface.co/kyutai/stt-2.6b-en" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>
<a target="_blank" href="https://colab.research.google.com/github/kyutai-labs/delayed-streams-modeling/blob/main/stt_pytorch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

For an example of how to use the model in a way where you can directly stream in PyTorch tensors,
[see our Colab notebook](https://colab.research.google.com/github/kyutai-labs/delayed-streams-modeling/blob/main/stt_pytorch.ipynb).

This requires the [moshi package](https://pypi.org/project/moshi/)
with version 0.2.6 or later, which can be installed via pip.

If you just want to run the model on a file, you can use `moshi.run_inference`.

```bash
python -m moshi.run_inference --hf-repo kyutai/stt-2.6b-en audio/bria.mp3
```

If you have [uv](https://docs.astral.sh/uv/) installed, you can skip the installation step
and just prefix the command above with `uvx --with moshi`.

Additionally, we provide two scripts that highlight different usage scenarios. The first script illustrates how to extract word-level timestamps from the model's outputs:

```bash
uv run \
  scripts/stt_from_file_pytorch.py \
  --hf-repo kyutai/stt-2.6b-en \
  audio/bria.mp3
```

The second script can be used to run a model on an existing Hugging Face dataset and calculate its performance metrics: 
```bash
uv run scripts/evaluate_on_dataset.py  \
  --dataset meanwhile  \
  --hf-repo kyutai/stt-2.6b-en
```

Another example shows how one can provide a text-, audio-, or text-audio prompt to our STT model:
```bash
uv run scripts/stt_from_file_pytorch_with_prompt.py \
  --hf-repo kyutai/stt-2.6b-en \
  --file bria.mp3 \
  --prompt_file ./audio/loonah.mp3 \
  --prompt_text "Loonah" \
  --cut-prompt-transcript
```
Produces the transcript of `bria.mp3` using the `Loonah` spelling for the name, instead of the `Luna` used without any prompt:
```
In the heart of an ancient forest, where the trees whispered secrets of the past, there lived a peculiar rabbit named Loonah (...)
```

Apart from nudging the model for a specific spelling of a word, other potential use-cases include speaker adaptation and steering the model towards a specific formatting style or even a language.
However, please bear in mind that is an experimental feature and its behavior is very sensitive to the prompt provided.
</details>

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

Then the server can be started via the following command using the config file
from this repository.
For `kyutai/stt-1b-en_fr`, use `configs/config-stt-en_fr.hf.toml`,
and for `kyutai/stt-2.6b-en`, use `configs/config-stt-en-hf.toml`,

```bash
moshi-server worker --config configs/config-stt-en_fr-hf.toml
```

Once the server has started you can transcribe audio from your microphone with the following script.
```bash
uv run scripts/stt_from_mic_rust_server.py
```

We also provide a script for transcribing from an audio file.
```bash
uv run scripts/stt_from_file_rust_server.py audio/bria.mp3
```

The script limits the decoding speed to simulates real-time processing of the audio. 
Faster processing can be triggered by setting 
the real-time factor, e.g. `--rtf 1000` will process
the data as fast as possible.
</details>

<details>
<summary>Rust standalone</summary>
<a href="https://huggingface.co/kyutai/stt-2.6b-en-candle" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

A standalone Rust example script is provided in the `stt-rs` directory in this repo.
This can be used as follows:
```bash
cd stt-rs
cargo run --features cuda -r -- ../audio/bria.mp3
```
You can get the timestamps by adding the `--timestamps` flag, and see the output
of the semantic VAD by adding the `--vad` flag.
</details>

<details>
<summary>MLX implementation</summary>
<a href="https://huggingface.co/kyutai/stt-2.6b-en-mlx" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" style="display: inline-block; vertical-align: middle;"/>
</a>

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is Apple's ML framework that allows you to use
hardware acceleration on Apple silicon.

This requires the [moshi-mlx package](https://pypi.org/project/moshi-mlx/)
with version 0.2.6 or later, which can be installed via pip.

If you just want to run the model on a file, you can use `moshi_mlx.run_inference`:

```bash
python -m moshi_mlx.run_inference --hf-repo kyutai/stt-2.6b-en-mlx audio/bria.mp3 --temp 0
```

If you have [uv](https://docs.astral.sh/uv/) installed, you can skip the installation step
and just prefix the command above with `uvx --with moshi-mlx`.

If you want to transcribe audio from your microphone, use:

```bash
python scripts/stt_from_mic_mlx.py
```

The MLX models can also be used in swift using the [moshi-swift
codebase](https://github.com/kyutai-labs/moshi-swift), the 1b model has been
tested to work fine on an iPhone 16 Pro.
</details>

## Kyutai Text-to-Speech

<a href="https://huggingface.co/collections/kyutai/text-to-speech-6866192e7e004ed04fd39e29" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-KyutaiTTS-blue" style="display: inline-block; vertical-align: middle;"/>
</a>
<a target="_blank" href="https://colab.research.google.com/github/kyutai-labs/delayed-streams-modeling/blob/main/tts_pytorch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**More details can be found on the [project page](https://kyutai.org/next/tts).**

We provide different implementations of Kyutai TTS for different use cases. Here is how to choose which one to use:

- PyTorch: for research and tinkering. If you want to call the model from Python for research or experimentation, use our PyTorch implementation.
- Rust: for production. If you want to serve Kyutai TTS in a production setting, use our Rust server. Our robust Rust server provides streaming access to the model over websockets. We use this server to run Unmute.
- MLX: for on-device inference on iPhone and Mac. MLX is Apple's ML framework that allows you to use hardware acceleration on Apple silicon. If you want to run the model on a Mac or an iPhone, choose the MLX implementation.

<details>
<summary>PyTorch implementation</summary>

<a target="_blank" href="https://colab.research.google.com/github/kyutai-labs/delayed-streams-modeling/blob/main/tts_pytorch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Check out our [Colab notebook](https://colab.research.google.com/github/kyutai-labs/delayed-streams-modeling/blob/main/tts_pytorch.ipynb) or use the script:

```bash
# From stdin, plays audio immediately
echo "Hey, how are you?" | python scripts/tts_pytorch.py - -

# From text file to audio file
python scripts/tts_pytorch.py text_to_say.txt audio_output.wav
```

The `tts_pytorch.py` script waits for all the text to be available before
starting the audio generation. A fully streaming implementation is available in
the `tts_pytorch_streaming.py` script, which can be used as follows:

```bash
echo "Hey, how are you?" | python scripts/tts_pytorch_streaming.py audio_output.wav
```

This requires the [moshi package](https://pypi.org/project/moshi/), which can be installed via pip.
If you have [uv](https://docs.astral.sh/uv/) installed, you can skip the installation step
and just prefix the command above with `uvx --with moshi`.
</details>

<details>
<summary>Rust server</summary>


The Rust implementation provides a server that can process multiple streaming
queries in parallel.

Installing the Rust server is a bit tricky because it uses our Python implementation under the hood,
which also requires installing the Python dependencies.
Use the [start_tts.sh](https://github.com/kyutai-labs/unmute/blob/main/dockerless/start_tts.sh) script to properly install the Rust server.
If you already installed the `moshi-server` crate before and it's not working, you might need to force a reinstall by running `cargo uninstall moshi-server` first.
Feel free to open an issue if the installation is still broken.

Once installed, the server can be started via the following command using the config file
from this repository.

```bash
moshi-server worker --config configs/config-tts.toml
```

Once the server has started you can connect to it using our script as follows:
```bash
# From stdin, plays audio immediately
echo "Hey, how are you?" | python scripts/tts_rust_server.py - -

# From text file to audio file
python scripts/tts_rust_server.py text_to_say.txt audio_output.wav
```
</details>

<details>
<summary>MLX implementation</summary>

[MLX](https://ml-explore.github.io/mlx/build/html/index.html) is Apple's ML framework that allows you to use
hardware acceleration on Apple silicon.

Use our example script to run Kyutai TTS on MLX.
The script takes text from stdin or a file and can output to a file or stream the resulting audio.
When streaming the output, if the model is not fast enough to keep with
real-time, you can use the `--quantize 8` or `--quantize 4` flags to quantize
the model resulting in faster inference.

```bash
# From stdin, plays audio immediately
echo "Hey, how are you?" | python scripts/tts_mlx.py - - --quantize 8

# From text file to audio file
python scripts/tts_mlx.py text_to_say.txt audio_output.wav
```

This requires the [moshi-mlx package](https://pypi.org/project/moshi-mlx/), which can be installed via pip.
If you have [uv](https://docs.astral.sh/uv/) installed, you can skip the installation step
and just prefix the command above with `uvx --with moshi-mlx`.
</details>

## FAQ

Checkout the [Frequently Asked Questions](FAQ.md) section before opening an issue.

## License

The present code is provided under the MIT license for the Python parts, and Apache license for the Rust backend.
The web client code is provided under the MIT license.
Note that parts of this code is based on [AudioCraft](https://github.com/facebookresearch/audiocraft), released under
the MIT license.

The weights for the speech-to-text models are released under the CC-BY 4.0 license.

## Developing

Install the [pre-commit hooks](https://pre-commit.com/) by running:

```bash
pip install pre-commit
pre-commit install
```

If you're using `uv`, you can replace the two commands with `uvx pre-commit install`.

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
