# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "msgpack",
#     "numpy",
#     "sphn",
#     "websockets",
# ]
# ///
import argparse
import asyncio
import time

import msgpack
import numpy as np
import sphn
import websockets

SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # Send data in chunks


def load_and_process_audio(file_path):
    """Load an MP3 file, resample to 24kHz, convert to mono, and extract PCM float32 data."""
    pcm_data, _ = sphn.read(file_path, sample_rate=SAMPLE_RATE)
    return pcm_data[0]


async def receive_messages(websocket):
    transcript = []

    async for message in websocket:
        data = msgpack.unpackb(message, raw=False)
        if data["type"] == "Step":
            # This message contains the signal from the semantic VAD, and tells us how
            # much audio the server has already processed. We don't use either here.
            continue
        if data["type"] == "Word":
            print(data["text"], end=" ", flush=True)
            transcript.append(
                {
                    "text": data["text"],
                    "timestamp": [data["start_time"], data["start_time"]],
                }
            )
        if data["type"] == "EndWord":
            if len(transcript) > 0:
                transcript[-1]["timestamp"][1] = data["stop_time"]
        if data["type"] == "Marker":
            # Received marker, stopping stream
            break

    return transcript


async def send_messages(websocket, rtf: float):
    audio_data = load_and_process_audio(args.in_file)

    async def send_audio(audio: np.ndarray):
        await websocket.send(
            msgpack.packb(
                {"type": "Audio", "pcm": [float(x) for x in audio]},
                use_single_float=True,
            )
        )

    # Start with a second of silence.
    # This is needed for the 2.6B model for technical reasons.
    await send_audio([0.0] * SAMPLE_RATE)

    start_time = time.time()
    for i in range(0, len(audio_data), FRAME_SIZE):
        await send_audio(audio_data[i : i + FRAME_SIZE])

        expected_send_time = start_time + (i + 1) / SAMPLE_RATE / rtf
        current_time = time.time()
        if current_time < expected_send_time:
            await asyncio.sleep(expected_send_time - current_time)
        else:
            await asyncio.sleep(0.001)

    for _ in range(5):
        await send_audio([0.0] * SAMPLE_RATE)

    # Send a marker to indicate the end of the stream.
    await websocket.send(
        msgpack.packb({"type": "Marker", "id": 0}, use_single_float=True)
    )

    # We'll get back the marker once the corresponding audio has been transcribed,
    # accounting for the delay of the model. That's why we need to send some silence
    # after the marker, because the model will not return the marker immediately.
    for _ in range(35):
        await send_audio([0.0] * SAMPLE_RATE)


async def stream_audio(url: str, api_key: str, rtf: float):
    """Stream audio data to a WebSocket server."""
    headers = {"kyutai-api-key": api_key}

    # Instead of using the header, you can authenticate by adding `?auth_id={api_key}` to the URL
    async with websockets.connect(url, additional_headers=headers) as websocket:
        send_task = asyncio.create_task(send_messages(websocket, rtf))
        receive_task = asyncio.create_task(receive_messages(websocket))
        _, transcript = await asyncio.gather(send_task, receive_task)

    return transcript


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument(
        "--url",
        help="The url of the server to which to send the audio",
        default="ws://127.0.0.1:8080",
    )
    parser.add_argument("--api-key", default="public_token")
    parser.add_argument(
        "--rtf",
        type=float,
        default=1.01,
        help="The real-time factor of how fast to feed in the audio.",
    )
    args = parser.parse_args()

    url = f"{args.url}/api/asr-streaming"
    transcript = asyncio.run(stream_audio(url, args.api_key, args.rtf))

    print()
    print()
    for word in transcript:
        print(
            f"{word['timestamp'][0]:7.2f} -{word['timestamp'][1]:7.2f}  {word['text']}"
        )
