# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx==0.2.12",
#     "numpy",
#     "rustymimi",
#     "sentencepiece",
#     "sounddevice",
# ]
# ///

import argparse
import json
import queue

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
import sounddevice as sd
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", default=4096)
    parser.add_argument("--hf-repo", default="kyutai/stt-1b-en_fr-mlx")
    parser.add_argument(
        "--vad", action="store_true", help="Enable VAD (Voice Activity Detection)."
    )
    args = parser.parse_args()

    lm_config = hf_hub_download(args.hf_repo, "config.json")
    with open(lm_config, "r") as fobj:
        lm_config = json.load(fobj)
    mimi_weights = hf_hub_download(args.hf_repo, lm_config["mimi_name"])
    moshi_name = lm_config.get("moshi_name", "model.safetensors")
    moshi_weights = hf_hub_download(args.hf_repo, moshi_name)
    tokenizer = hf_hub_download(args.hf_repo, lm_config["tokenizer_name"])

    lm_config = models.LmConfig.from_config_dict(lm_config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if moshi_weights.endswith(".q4.safetensors"):
        nn.quantize(model, bits=4, group_size=32)
    elif moshi_weights.endswith(".q8.safetensors"):
        nn.quantize(model, bits=8, group_size=64)

    print(f"loading model weights from {moshi_weights}")
    if args.hf_repo.endswith("-candle"):
        model.load_pytorch_weights(moshi_weights, lm_config, strict=True)
    else:
        model.load_weights(moshi_weights, strict=True)

    print(f"loading the text tokenizer from {tokenizer}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)  # type: ignore

    print(f"loading the audio tokenizer {mimi_weights}")
    generated_codebooks = lm_config.generated_codebooks
    other_codebooks = lm_config.other_codebooks
    mimi_codebooks = max(generated_codebooks, other_codebooks)
    audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)  # type: ignore
    print("warming up the model")
    model.warmup()
    gen = models.LmGen(
        model=model,
        max_steps=args.max_steps,
        text_sampler=utils.Sampler(top_k=25, temp=0),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        check=False,
    )

    block_queue = queue.Queue()

    def audio_callback(indata, _frames, _time, _status):
        block_queue.put(indata.copy())

    print("recording audio from microphone, speak to get your words transcribed")
    last_print_was_vad = False
    with sd.InputStream(
        channels=1,
        dtype="float32",
        samplerate=24000,
        blocksize=1920,
        callback=audio_callback,
    ):
        while True:
            block = block_queue.get()
            block = block[None, :, 0]
            other_audio_tokens = audio_tokenizer.encode_step(block[None, 0:1])
            other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                :, :, :other_codebooks
            ]
            if args.vad:
                text_token, vad_heads = gen.step_with_extra_heads(other_audio_tokens[0])
                if vad_heads:
                    pr_vad = vad_heads[2][0, 0, 0].item()
                    if pr_vad > 0.5 and not last_print_was_vad:
                        print(" [end of turn detected]")
                        last_print_was_vad = True
            else:
                text_token = gen.step(other_audio_tokens[0])
            text_token = text_token[0].item()
            audio_tokens = gen.last_audio_tokens()
            _text = None
            if text_token not in (0, 3):
                _text = text_tokenizer.id_to_piece(text_token)  # type: ignore
                _text = _text.replace("▁", " ")
                print(_text, end="", flush=True)
                last_print_was_vad = False
