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
import json
import msgpack
import sphn
import struct
import time

import numpy as np
import websockets

# Desired audio properties
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1  # Mono
all_text = []
transcript = []
finished = False


def load_and_process_audio(file_path):
    """Load an MP3 file, resample to 24kHz, convert to mono, and extract PCM float32 data."""
    pcm_data, _ = sphn.read(file_path, sample_rate=TARGET_SAMPLE_RATE)
    return pcm_data[0]


async def receive_messages(websocket):
    global all_text
    global transcript
    global finished
    try:
        async for message in websocket:
            data = msgpack.unpackb(message, raw=False)
            if data["type"] == "Step":
                continue
            print("received:", data)
            if data["type"] == "Word":
                all_text.append(data["text"])
                transcript.append({
                    "speaker": "SPEAKER_00",
                    "text": data["text"],
                    "timestamp": [data["start_time"], data["start_time"]],
                })
            if data["type"] == "EndWord":
                if len(transcript) > 0:
                    transcript[-1]["timestamp"][1] = data["stop_time"]
            if data["type"] == "Marker":
                print("Received marker, stopping stream.")
                break
    except websockets.ConnectionClosed:
        print("Connection closed while receiving messages.")
    finished = True


async def send_messages(websocket, rtf: float):
    global finished
    audio_data = load_and_process_audio(args.in_file)
    try:
        # Start with a second of silence
        chunk = { "type": "Audio", "pcm": [0.0] * 24000 }
        msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
        await websocket.send(msg)

        chunk_size = 1920  # Send data in chunks
        start_time = time.time()
        for i in range(0, len(audio_data), chunk_size):
            chunk = { "type": "Audio", "pcm": [float(x) for x in audio_data[i : i + chunk_size]] }
            msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
            await websocket.send(msg)
            expected_send_time = start_time + (i + 1) / 24000 / rtf
            current_time = time.time()
            if current_time < expected_send_time:
                await asyncio.sleep(expected_send_time - current_time)
            else:
                await asyncio.sleep(0.001)
        chunk = { "type": "Audio", "pcm": [0.0] * 1920 * 5 }
        msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
        await websocket.send(msg)
        msg = msgpack.packb({"type": "Marker", "id": 0}, use_bin_type=True, use_single_float=True)
        await websocket.send(msg)
        for _ in range(35):
            chunk = { "type": "Audio", "pcm": [0.0] * 1920 }
            msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
            await websocket.send(msg)
        while True:
            if finished:
                break
            await asyncio.sleep(1.0)
            # Keep the connection alive as there is a 20s timeout on the rust side.
            await websocket.ping()
    except websockets.ConnectionClosed:
        print("Connection closed while sending messages.")


async def stream_audio(url: str, rtf: float, api_key: str):
    """Stream audio data to a WebSocket server."""

    headers = {"kyutai-api-key": api_key}
    async with websockets.connect(url, additional_headers=headers) as websocket:
        send_task = asyncio.create_task(send_messages(websocket, rtf))
        receive_task = asyncio.create_task(receive_messages(websocket))
        await asyncio.gather(send_task, receive_task)
    print("exiting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file")
    parser.add_argument("--transcript")
    parser.add_argument("--api-key", default="open_key")
    parser.add_argument(
        "--url",
        help="The url of the server to which to send the audio",
        default="ws://127.0.0.1:8080",
    )
    parser.add_argument("--rtf", type=float, default=1.01)
    args = parser.parse_args()

    url = f"{args.url}/api/asr-streaming"
    asyncio.run(stream_audio(url, args.rtf, args.api_key))
    print(" ".join(all_text))
    if args.transcript is not None:
        with open(args.transcript, "w") as fobj:
            json.dump({"transcript": transcript}, fobj, indent=4)
