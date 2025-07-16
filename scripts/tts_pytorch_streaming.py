# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "moshi==0.2.10",
#     "torch",
#     "sphn",
#     "sounddevice",
# ]
# ///
import argparse
from dataclasses import dataclass
import sys

import numpy as np
import queue
import sphn
import time
import torch
import typing as tp
from moshi.models.loaders import CheckpointInfo
from moshi.conditioners import dropout_all_conditions
from moshi.models.lm import LMGen
from moshi.models.tts import (
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    TTSModel,
    ConditionAttributes,
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
    on_frame: tp.Optional[tp.Callable[[torch.Tensor], None]] = None

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
        prepared = tts_model.lm.condition_provider.prepare(attributes)
        condition_tensors = tts_model.lm.condition_provider(prepared)

        def _on_text_logits_hook(text_logits):
            if tts_model.padding_bonus:
                text_logits[..., tts_model.machine.token_ids.pad] += (
                    tts_model.padding_bonus
                )
            return text_logits

        def _on_audio_hook(audio_tokens):
            audio_offset = tts_model.lm.audio_offset
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[1]):
                delay = delays[q + audio_offset]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[:, q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = torch.tensor(
                out_tokens, dtype=torch.long, device=text_tokens.device
            )

        tts_model.lm.dep_q = tts_model.n_q
        self.lm_gen = LMGen(
            tts_model.lm,
            temp=tts_model.temp,
            temp_text=tts_model.temp,
            cfg_coef=tts_model.cfg_coef,
            condition_tensors=condition_tensors,
            on_text_logits_hook=_on_text_logits_hook,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            cfg_is_masked_until=None,
            cfg_is_no_text=True,
        )
        self.lm_gen.streaming_forever(1)

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
        input_tokens = torch.full(
            (1, missing, 1),
            self.tts_model.machine.token_ids.zero,
            dtype=torch.long,
            device=self.tts_model.lm.device,
        )
        frame = self.lm_gen.step(input_tokens)
        self.offset += 1
        if frame is not None:
            if self.on_frame is not None:
                self.on_frame(frame)

    def append_entry(self, entry):
        self.state.entries.append(entry)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Run Kyutai TTS using the PyTorch implementation"
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
        "--voice",
        default="expresso/ex03-ex01_happy_001_channel1_334s.wav",
        help="The voice to use, relative to the voice repo root. "
        f"See {DEFAULT_DSM_TTS_VOICE_REPO}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    args = parser.parse_args()

    print("Loading model...")
    checkpoint_info = CheckpointInfo.from_hf_repo(args.hf_repo)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, n_q=32, temp=0.6, device=args.device
    )

    voice_path = tts_model.get_voice_path(args.voice)
    # CFG coef goes here because the model was trained with CFG distillation,
    # so it's not _actually_ doing CFG at inference time.
    # Also, if you are generating a dialog, you should have two voices in the list.
    condition_attributes = tts_model.make_condition_attributes(
        [voice_path], cfg_coef=2.0
    )

    if sys.stdin.isatty():  # Interactive
        print("Enter text to synthesize (Ctrl+D to end input):")

    if args.out == "-":
        # Stream the audio to the speakers using sounddevice.
        import sounddevice as sd

        pcms = queue.Queue()

        def _on_frame(frame):
            if (frame != -1).all():
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.put_nowait(np.clip(pcm[0, 0], -1, 1))

        def audio_callback(outdata, _a, _b, _c):
            try:
                pcm_data = pcms.get(block=False)
                outdata[:, 0] = pcm_data
            except queue.Empty:
                outdata[:] = 0

        gen = TTSGen(tts_model, [condition_attributes], on_frame=_on_frame)

        with sd.OutputStream(
            samplerate=tts_model.mimi.sample_rate,
            blocksize=1920,
            channels=1,
            callback=audio_callback,
        ) and tts_model.mimi.streaming(1):
            for line in sys.stdin:
                # TODO: Fix the following to only include bos on the first line.
                entries = tts_model.prepare_script([line.strip()], padding_between=1)
                for entry in entries:
                    gen.append_entry(entry)
                    gen.process()
            gen.process_last()
            while True:
                if pcms.qsize() == 0:
                    break
                time.sleep(1)
    else:
        pcms = []

        def _on_frame(frame: torch.Tensor):
            if (frame != -1).all():
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(np.clip(pcm[0, 0]))

        gen = TTSGen(tts_model, [condition_attributes], on_frame=_on_frame)
        with tts_model.mimi.streaming(1):
            for line in sys.stdin:
                # TODO: Fix the following to only include bos on the first line.
                entries = tts_model.prepare_script([line.strip()], padding_between=1)
                for entry in entries:
                    gen.append_entry(entry)
                    gen.process()
            gen.process_last()
        pcm = np.concatenate(pcms, axis=-1)
        sphn.write_wav(args.out, pcm, tts_model.mimi.sample_rate)


if __name__ == "__main__":
    main()
