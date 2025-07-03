# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx @ git+https://git@github.com/kyutai-labs/moshi#egg=moshi_mlx&subdirectory=moshi_mlx",
#     "numpy",
#     "sounddevice",
# ]
# ///

import argparse
import json
import queue
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import sentencepiece
import sounddevice as sd
import sphn
from moshi_mlx import models
from moshi_mlx.client_utils import make_log
from moshi_mlx.models.tts import (
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
)
from moshi_mlx.utils.loaders import hf_get


def log(level: str, msg: str):
    print(make_log(level, msg))


def main():
    parser = argparse.ArgumentParser(
        description="Run Kyutai TTS using the PyTorch implementation"
    )
    parser.add_argument("inp", type=str, help="Input file, use - for stdin")
    parser.add_argument(
        "out", type=str, help="Output file to generate, use - for playing the audio"
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=DEFAULT_DSM_TTS_REPO,
        help="HF repo in which to look for the pretrained models.",
    )
    parser.add_argument(
        "--voice-repo",
        default=DEFAULT_DSM_TTS_VOICE_REPO,
        help="HF repo in which to look for pre-computed voice embeddings.",
    )
    parser.add_argument(
        "--voice", default="expresso/ex03-ex01_happy_001_channel1_334s.wav"
    )
    parser.add_argument(
        "--quantize",
        type=int,
        help="The quantization to be applied, e.g. 8 for 8 bits.",
    )
    args = parser.parse_args()

    mx.random.seed(299792458)

    log("info", "retrieving checkpoints")

    raw_config = hf_get("config.json", args.hf_repo)
    with open(hf_get(raw_config), "r") as fobj:
        raw_config = json.load(fobj)

    mimi_weights = hf_get(raw_config["mimi_name"], args.hf_repo)
    moshi_name = raw_config.get("moshi_name", "model.safetensors")
    moshi_weights = hf_get(moshi_name, args.hf_repo)
    tokenizer = hf_get(raw_config["tokenizer_name"], args.hf_repo)
    lm_config = models.LmConfig.from_config_dict(raw_config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)

    log("info", f"loading model weights from {moshi_weights}")
    model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

    if args.quantize is not None:
        log("info", f"quantizing model to {args.quantize} bits")
        nn.quantize(model.depformer, bits=args.quantize)
        for layer in model.transformer.layers:
            nn.quantize(layer.self_attn, bits=args.quantize)
            nn.quantize(layer.gating, bits=args.quantize)

    log("info", f"loading the text tokenizer from {tokenizer}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer))  # type: ignore

    log("info", f"loading the audio tokenizer {mimi_weights}")
    generated_codebooks = lm_config.generated_codebooks
    audio_tokenizer = models.mimi.Mimi(models.mimi_202407(generated_codebooks))
    audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

    cfg_coef_conditioning = None
    tts_model = TTSModel(
        model,
        audio_tokenizer,
        text_tokenizer,
        voice_repo=args.voice_repo,
        temp=0.6,
        cfg_coef=1,
        max_padding=8,
        initial_padding=2,
        final_padding=2,
        padding_bonus=0,
        raw_config=raw_config,
    )
    if tts_model.valid_cfg_conditionings:
        # Model was trained with CFG distillation.
        cfg_coef_conditioning = tts_model.cfg_coef
        tts_model.cfg_coef = 1.0
        cfg_is_no_text = False
        cfg_is_no_prefix = False
    else:
        cfg_is_no_text = True
        cfg_is_no_prefix = True
    mimi = tts_model.mimi

    log("info", f"reading input from {args.inp}")
    if args.inp == "-":
        if sys.stdin.isatty():  # Interactive
            print("Enter text to synthesize (Ctrl+D to end input):")
        text_to_tts = sys.stdin.read().strip()
    else:
        with open(args.inp, "r") as fobj:
            text_to_tts = fobj.read().strip()

    all_entries = [tts_model.prepare_script([text_to_tts])]
    if tts_model.multi_speaker:
        voices = [tts_model.get_voice_path(args.voice)]
    else:
        voices = []
    all_attributes = [
        tts_model.make_condition_attributes(voices, cfg_coef_conditioning)
    ]

    wav_frames = queue.Queue()

    def _on_frame(frame):
        if (frame == -1).any():
            return
        _pcm = tts_model.mimi.decode_step(frame[:, :, None])
        _pcm = np.array(mx.clip(_pcm[0, 0], -1, 1))
        wav_frames.put_nowait(_pcm)

    def run():
        log("info", "starting the inference loop")
        begin = time.time()
        result = tts_model.generate(
            all_entries,
            all_attributes,
            cfg_is_no_prefix=cfg_is_no_prefix,
            cfg_is_no_text=cfg_is_no_text,
            on_frame=_on_frame,
        )
        frames = mx.concat(result.frames, axis=-1)
        total_duration = frames.shape[0] * frames.shape[-1] / mimi.frame_rate
        time_taken = time.time() - begin
        total_speed = total_duration / time_taken
        log("info", f"[LM] took {time_taken:.2f}s, total speed {total_speed:.2f}x")
        return result

    if args.out == "-":

        def audio_callback(outdata, _a, _b, _c):
            try:
                pcm_data = wav_frames.get(block=False)
                outdata[:, 0] = pcm_data
            except queue.Empty:
                outdata[:] = 0

        with sd.OutputStream(
            samplerate=mimi.sample_rate,
            blocksize=1920,
            channels=1,
            callback=audio_callback,
        ):
            run()
            time.sleep(3)
            while True:
                if wav_frames.qsize() == 0:
                    break
                time.sleep(1)
    else:
        run()
        frames = []
        while True:
            try:
                frames.append(wav_frames.get_nowait())
            except queue.Empty:
                break
        wav = np.concat(frames, -1)
        sphn.write_wav(args.out, wav, mimi.sample_rate)


if __name__ == "__main__":
    main()
