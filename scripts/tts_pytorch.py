# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "moshi==0.2.7",
#     "torch",
#     "sphn",
#     "sounddevice",
# ]
# ///
import argparse
import sys

import numpy as np
import queue
import sphn
import time
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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU for inference",
    )
    args = parser.parse_args()

    print("Loading model...")
    checkpoint_info = CheckpointInfo.from_hf_repo(args.hf_repo)

    # Set device and precision
    if args.cpu:
        device = torch.device("cpu")
        dtype = torch.float32
    else:
        device = torch.device("cuda")
        dtype = torch.bfloat16

    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, n_q=32, temp=0.6, device=device, dtype=dtype
    )

    if args.inp == "-":
        if sys.stdin.isatty():  # Interactive
            print("Enter text to synthesize (Ctrl+D to end input):")
        text = sys.stdin.read().strip()
    else:
        with open(args.inp, "r") as fobj:
            text = fobj.read().strip()

    # If you want to make a dialog, you can pass more than one turn [text_speaker_1, text_speaker_2, text_2_speaker_1, ...]
    entries = tts_model.prepare_script([text], padding_between=1)
    voice_path = tts_model.get_voice_path(args.voice)
    # CFG coef goes here because the model was trained with CFG distillation,
    # so it's not _actually_ doing CFG at inference time.
    # Also, if you are generating a dialog, you should have two voices in the list.
    condition_attributes = tts_model.make_condition_attributes(
        [voice_path], cfg_coef=2.0
    )

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

        with sd.OutputStream(
            samplerate=tts_model.mimi.sample_rate,
            blocksize=1920,
            channels=1,
            callback=audio_callback,
        ):
            with tts_model.mimi.streaming(1):
                tts_model.generate([entries], [condition_attributes], on_frame=_on_frame)
            time.sleep(3)
            while True:
                if pcms.qsize() == 0:
                    break
                time.sleep(1)
    else:
        result = tts_model.generate([entries], [condition_attributes])
        with tts_model.mimi.streaming(1), torch.no_grad():
            pcms = []
            for frame in result.frames[tts_model.delay_steps :]:
                pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcms.append(np.clip(pcm[0, 0], -1, 1))
            pcm = np.concatenate(pcms, axis=-1)
        sphn.write_wav(args.out, pcm, tts_model.mimi.sample_rate)


if __name__ == "__main__":
    main()
