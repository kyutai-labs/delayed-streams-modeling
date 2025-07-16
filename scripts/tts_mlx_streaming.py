# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface_hub",
#     "moshi_mlx==0.2.11",
#     "numpy",
#     "sounddevice",
# ]
# ///

import argparse
from dataclasses import dataclass
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
import typing as tp
from moshi_mlx import models
from moshi_mlx.models.generate import LmGen
from moshi_mlx.client_utils import make_log
from moshi_mlx.modules.conditioner import (
    ConditionAttributes,
    ConditionTensor,
    dropout_all_conditions,
)
from moshi_mlx.utils.sampling import Sampler
from moshi_mlx.models.tts import (
    Entry,
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
    script_to_entries,
)
from moshi_mlx.utils.loaders import hf_get


def prepare_script(model: TTSModel, script: str, first_turn: bool) -> list[Entry]:
    multi_speaker = first_turn and model.multi_speaker
    return script_to_entries(
        model.tokenizer,
        model.machine.token_ids,
        model.mimi.frame_rate,
        [script],
        multi_speaker=multi_speaker,
        padding_between=1,
    )


def _make_null(
    all_attributes: tp.Sequence[ConditionAttributes],
) -> list[ConditionAttributes]:
    # When using CFG, returns the null conditions.
    return dropout_all_conditions(all_attributes)


@dataclass
class TTSGen:
    tts_model: TTSModel
    attributes: tp.Sequence[ConditionAttributes]
    on_frame: tp.Optional[tp.Callable[[mx.array], None]] = None

    def __post_init__(self):
        tts_model = self.tts_model
        attributes = self.attributes
        self.offset = 0
        self.state = self.tts_model.machine.new_state([])

        if tts_model.cfg_coef != 1.0:
            if tts_model.valid_cfg_conditionings:
                raise ValueError(
                    "This model does not support direct CFG, but was trained with "
                    "CFG distillation. Pass instead `cfg_coef` to `make_condition_attributes`."
                )
            nulled = _make_null(attributes)
            attributes = list(attributes) + nulled

        assert tts_model.lm.condition_provider is not None
        self.ct = None
        self.cross_attention_src = None
        for _attr in attributes:
            for _key, _value in _attr.text.items():
                _ct = tts_model.lm.condition_provider.condition_tensor(_key, _value)
                if self.ct is None:
                    self.ct = _ct
                else:
                    self.ct = ConditionTensor(self.ct.tensor + _ct.tensor)
            for _key, _value in _attr.tensor.items():
                _conditioner = tts_model.lm.condition_provider.conditioners[_key]
                _ca_src = _conditioner.condition(_value)
                if self.cross_attention_src is None:
                    self.cross_attention_src = _ca_src
                else:
                    raise ValueError("multiple cross-attention conditioners")

        def _on_audio_hook(audio_tokens):
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[0]):
                delay = delays[q]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = mx.array(out_tokens, dtype=mx.int64)

        self.lm_gen = LmGen(
            tts_model.lm,
            max_steps=tts_model.max_gen_length,
            text_sampler=Sampler(temp=tts_model.temp),
            audio_sampler=Sampler(temp=tts_model.temp),
            cfg_coef=tts_model.cfg_coef,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            # TODO(laurent):
            # cfg_is_masked_until=cfg_is_masked_until,
            # cfg_is_no_text=cfg_is_no_text,
        )

    def process_last(self):
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            self._step()
        additional_steps = (
            self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        )
        for _ in range(additional_steps):
            self._step()

    def process(self):
        while len(self.state.entries) > self.tts_model.machine.second_stream_ahead:
            self._step()

    def _step(self):
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = (
            mx.ones((1, missing), dtype=mx.int64)
            * self.tts_model.machine.token_ids.zero
        )
        self.lm_gen.step(
            input_tokens, ct=self.ct, cross_attention_src=self.cross_attention_src
        )
        frame = self.lm_gen.last_audio_tokens()
        self.offset += 1
        if frame is not None:
            if self.on_frame is not None:
                self.on_frame(frame)

    def append_entry(self, entry):
        self.state.entries.append(entry)


def log(level: str, msg: str):
    print(make_log(level, msg))


def main():
    parser = argparse.ArgumentParser(
        description="Run Kyutai TTS using the MLX implementation"
    )
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
    mimi = tts_model.mimi

    log("info", "reading input from stdin")

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

    gen = TTSGen(tts_model, all_attributes, on_frame=_on_frame)

    def run():
        log("info", "starting the inference loop")
        first_turn = True
        for line in sys.stdin:
            entries = prepare_script(tts_model, line.strip(), first_turn=first_turn)
            first_turn = False
            for entry in entries:
                gen.append_entry(entry)
                gen.process()
        gen.process_last()

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
