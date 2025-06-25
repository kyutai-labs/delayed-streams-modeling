# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "msgpack",
#     "numpy",
#     "sounddevice",
#     "websockets",
# ]
# ///
import argparse
import asyncio
import signal

import msgpack
import numpy as np
import sounddevice as sd
import websockets

# Desired audio properties
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1  # Mono


async def receive_messages(websocket):
    """Receive and process messages from the WebSocket server."""
    try:
        async for message in websocket:
            data = msgpack.unpackb(message, raw=False)
            if data["type"] == "Word":
                print(data["text"], end=" ", flush=True)
    except websockets.ConnectionClosed:
        print("Connection closed while receiving messages.")


async def send_messages(websocket, audio_queue):
    """Send audio data from microphone to WebSocket server."""
    try:
        # Start by draining the queue to avoid lags
        while not audio_queue.empty():
            await audio_queue.get()
        print("Starting the transcription")
        while True:
            audio_data = await audio_queue.get()
            chunk = {"type": "Audio", "pcm": [float(x) for x in audio_data]}
            msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
            await websocket.send(msg)
    except websockets.ConnectionClosed:
        print("Connection closed while sending messages.")


async def stream_audio(url: str, api_key: str):
    """Stream audio data to a WebSocket server."""
    print("Starting microphone recording...")
    print("Press Ctrl+C to stop recording")
    audio_queue = asyncio.Queue()

    loop = asyncio.get_event_loop()

    def audio_callback(indata, frames, time, status):
        loop.call_soon_threadsafe(
            audio_queue.put_nowait, indata[:, 0].astype(np.float32).copy()
        )

    # Start audio stream
    with sd.InputStream(
        samplerate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
        dtype="float32",
        callback=audio_callback,
        blocksize=1920,  # 80ms blocks
    ):
        headers = {"kyutai-api-key": api_key}
        # Instead of using the header, you can authenticate by adding `?auth_id={api_key}` to the URL
        async with websockets.connect(url, additional_headers=headers) as websocket:
            send_task = asyncio.create_task(send_messages(websocket, audio_queue))
            receive_task = asyncio.create_task(receive_messages(websocket))
            await asyncio.gather(send_task, receive_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time microphone transcription")
    parser.add_argument(
        "--url",
        help="The URL of the server to which to send the audio",
        default="ws://127.0.0.1:8080",
    )
    parser.add_argument("--api-key", default="public_token")
    parser.add_argument(
        "--list-devices", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--device", type=int, help="Input device ID (use --list-devices to see options)"
    )

    args = parser.parse_args()

    def handle_sigint(signum, frame):
        print("Interrupted by user")
        exit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        exit(0)

    if args.device is not None:
        sd.default.device[0] = args.device  # Set input device

    url = f"{args.url}/api/asr-streaming"
    asyncio.run(stream_audio(url, args.api_key))
