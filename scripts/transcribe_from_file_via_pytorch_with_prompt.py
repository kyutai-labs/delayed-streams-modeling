"""An example script that illustrates how one can prompt Kyutai STT models."""

import argparse
import dataclasses
import itertools
import math
from collections import deque

import julius
import moshi.models
import sphn
import torch
import tqdm


class PromptHook:
    def __init__(
        self,
        tokenizer,
        prefix,
        padding_tokens=(
            0,
            3,
        ),
    ):
        self.tokenizer = tokenizer
        self.prefix_enforce = deque(self.tokenizer.encode(prefix))
        self.padding_tokens = padding_tokens

    def on_token(self, token):
        if not self.prefix_enforce:
            return

        token = token.item()

        if token in self.padding_tokens:
            pass
        elif token == self.prefix_enforce[0]:
            self.prefix_enforce.popleft()
        else:
            assert False

    def on_logits(self, logits):
        if not self.prefix_enforce:
            return

        mask = torch.zeros_like(logits, dtype=torch.bool)
        for t in self.padding_tokens:
            mask[..., t] = True
        mask[..., self.prefix_enforce[0]] = True

        logits[:] = torch.where(mask, logits, float("-inf"))


def main(args):
    info = moshi.models.loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weights=args.moshi_weight,
        mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer,
        config_path=args.config_path,
    )

    mimi = info.get_mimi(device=args.device)
    tokenizer = info.get_text_tokenizer()
    lm = info.get_moshi(
        device=args.device,
        dtype=torch.bfloat16,
    )

    if args.prompt_text:
        prompt_hook = PromptHook(tokenizer, args.prompt_text)
        lm_gen = moshi.models.LMGen(
            lm,
            temp=0,
            temp_text=0.0,
            on_text_hook=prompt_hook.on_token,
            on_text_logits_hook=prompt_hook.on_logits,
        )
    else:
        lm_gen = moshi.models.LMGen(lm, temp=0, temp_text=0.0)

    audio_silence_prefix_seconds = info.stt_config.get(
        "audio_silence_prefix_seconds", 1.0
    )
    audio_delay_seconds = info.stt_config.get("audio_delay_seconds", 5.0)
    padding_token_id = info.raw_config.get("text_padding_token_id", 3)

    def _load_and_process(path):
        audio, input_sample_rate = sphn.read(path)
        audio = torch.from_numpy(audio).to(args.device).mean(axis=0, keepdim=True)
        audio = julius.resample_frac(audio, input_sample_rate, mimi.sample_rate)
        if audio.shape[-1] % mimi.frame_size != 0:
            to_pad = mimi.frame_size - audio.shape[-1] % mimi.frame_size
            audio = torch.nn.functional.pad(audio, (0, to_pad))
        return audio

    n_prefix_chunks = math.ceil(audio_silence_prefix_seconds * mimi.frame_rate)
    n_suffix_chunks = math.ceil(audio_delay_seconds * mimi.frame_rate)
    silence_chunk = torch.zeros(
        (1, 1, mimi.frame_size), dtype=torch.float32, device=args.device
    )

    audio = _load_and_process(args.file)
    if args.prompt_file:
        audio_prompt = _load_and_process(args.prompt_file)
    else:
        audio_prompt = None

    chain = [itertools.repeat(silence_chunk, n_prefix_chunks)]

    if audio_prompt is not None:
        chain.append(torch.split(audio_prompt[:, None, :], mimi.frame_size, dim=-1))
        # adding a bit (0.8s) of silence to separate prompt and the actual audio
        chain.append(itertools.repeat(silence_chunk, 10))

    chain += [
        torch.split(audio[:, None, :], mimi.frame_size, dim=-1),
        itertools.repeat(silence_chunk, n_suffix_chunks),
    ]

    chunks = itertools.chain(*chain)

    text_tokens_accum = []
    with mimi.streaming(1), lm_gen.streaming(1):
        for audio_chunk in tqdm.tqdm(chunks):
            audio_tokens = mimi.encode(audio_chunk)
            text_tokens = lm_gen.step(audio_tokens)
            if text_tokens is not None:
                text_tokens_accum.append(text_tokens)

    utterance_tokens = torch.concat(text_tokens_accum, dim=-1)
    text_tokens = utterance_tokens.cpu().view(-1)

    # if we have an audio prompt and we don't want to have it in the transcript,
    # we should cut the corresponding number of frames from the output tokens.
    # However, there is also some amount of padding that happens before it
    # due to silence_prefix and audio_delay. Normally it is ignored in detokenization,
    # but now we should account for it to find the position of the prompt transcript.
    if args.cut_prompt_transcript and audio_prompt is not None:
        prompt_frames = audio_prompt.shape[1] // mimi.frame_size
        no_prompt_offset_seconds = audio_delay_seconds + audio_silence_prefix_seconds
        no_prompt_offset = int(no_prompt_offset_seconds * mimi.frame_rate)
        text_tokens = text_tokens[prompt_frames + no_prompt_offset:]

    text = tokenizer.decode(
        text_tokens[text_tokens > padding_token_id].numpy().tolist()
    )

    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example streaming STT w/ a prompt.")
    parser.add_argument(
        "--file",
        required=True,
        help="File to transcribe.",
    )
    parser.add_argument(
        "--prompt_file",
        required=False,
        help="Audio of the prompt.",
    )
    parser.add_argument(
        "--prompt_text",
        required=False,
        help="Text of the prompt.",
    )
    parser.add_argument(
        "--cut-prompt-transcript",
        action="store_true",
        help="Cut the prompt from the output transcript",
    )
    parser.add_argument(
        "--hf-repo", type=str, help="HF repo to load the STT model from. "
    )
    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument(
        "--moshi-weight", type=str, help="Path to a local checkpoint file."
    )
    parser.add_argument(
        "--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi."
    )
    parser.add_argument(
        "--config-path", type=str, help="Path to a local config file.", default=None
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device on which to run, defaults to 'cuda'.",
    )
    args = parser.parse_args()

    main(args)
