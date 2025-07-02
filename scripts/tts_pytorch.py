# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "moshi @ git+https://git@github.com/kyutai-labs/moshi#egg=moshi&subdirectory=moshi",
#     "torch",
#     "sphn",
# ]
# ///
import argparse
import sys

import sphn
import torch
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel


def main():
    parser = argparse.ArgumentParser(
        description="Run Kyutai TTS using the PyTorch implementation"
    )
    parser.add_argument("inp", type=str, help="Input file, use - for stdin.")
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
    args = parser.parse_args()

    print("Loading model...")
    checkpoint_info = CheckpointInfo.from_hf_repo(args.hf_repo)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, n_q=32, temp=0.6, device=torch.device("cuda"), dtype=torch.half
    )

    if args.inp == "-":
        print("Enter text to synthesize (Ctrl+D to end input):")
        text = sys.stdin.read().strip()
    else:
        with open(args.inp, "r") as fobj:
            text = fobj.read().strip()

    # You could also generate multiple audios at once by passing a list of texts.
    entries = tts_model.prepare_script([text], padding_between=1)
    voice_path = tts_model.get_voice_path(args.voice)
    # CFG coef goes here because the model was trained with CFG distillation,
    # so it's not _actually_ doing CFG at inference time.
    condition_attributes = tts_model.make_condition_attributes(
        [voice_path], cfg_coef=2.0
    )

    print("Generating audio...")
    # This doesn't do streaming generation,
    result = tts_model.generate([entries], [condition_attributes])

    frames = torch.cat(result.frames, dim=-1)
    audio_tokens = frames[:, tts_model.lm.audio_offset :, tts_model.delay_steps :]
    with torch.no_grad():
        audios = tts_model.mimi.decode(audio_tokens)

    sphn.write_wav(args.out, audios[0].cpu().numpy(), tts_model.mimi.sample_rate)
    print(f"Audio saved to {args.out}")


if __name__ == "__main__":
    main()
