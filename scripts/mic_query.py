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
import json
import msgpack
import queue
import struct
import time
import threading

import numpy as np
import sounddevice as sd
import websockets

# Desired audio properties
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1  # Mono
HEADERS = {"kyutai-api-key": "open_token"}
all_text = []
transcript = []
finished = False
audio_queue = queue.Queue()
recording = True


def audio_callback(indata, frames, time, status):
    """Callback function for sounddevice to capture audio."""
    if status:
        print(f"Audio callback status: {status}")
    # Convert to float32 and flatten to mono
    audio_data = indata[:, 0].astype(np.float32)
    audio_queue.put(audio_data.copy())


async def receive_messages(websocket):
    """Receive and process messages from the WebSocket server."""
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
                # Print words in real-time
                print(f"Word: {data['text']}")
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
    """Send audio data from microphone to WebSocket server."""
    global finished
    global recording
    
    try:
        # Start with a second of silence
        chunk = {"type": "Audio", "pcm": [0.0] * 24000}
        msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
        await websocket.send(msg)

        chunk_size = 1920  # Send data in chunks (80ms at 24kHz)
        
        while recording and not finished:
            try:
                # Get audio data from queue with timeout
                audio_data = audio_queue.get(timeout=0.1)
                
                # Process audio in chunks
                for i in range(0, len(audio_data), chunk_size):
                    if not recording:
                        break
                    
                    chunk_data = audio_data[i:i + chunk_size]
                    # Pad with zeros if chunk is smaller than expected
                    if len(chunk_data) < chunk_size:
                        chunk_data = np.pad(chunk_data, (0, chunk_size - len(chunk_data)), 'constant')
                    
                    chunk = {"type": "Audio", "pcm": [float(x) for x in chunk_data]}
                    msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
                    await websocket.send(msg)
                    
                    # Small delay to avoid overwhelming the server
                    await asyncio.sleep(0.001)
                    
            except queue.Empty:
                # No audio data available, continue
                continue
                
        # Send final silence and marker
        chunk = {"type": "Audio", "pcm": [0.0] * 1920 * 5}
        msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
        await websocket.send(msg)
        
        msg = msgpack.packb({"type": "Marker", "id": 0}, use_bin_type=True, use_single_float=True)
        await websocket.send(msg)
        
        # Send additional silence chunks
        for _ in range(35):
            chunk = {"type": "Audio", "pcm": [0.0] * 1920}
            msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
            await websocket.send(msg)
            
        # Keep connection alive
        while not finished:
            await asyncio.sleep(1.0)
            await websocket.ping()
            
    except websockets.ConnectionClosed:
        print("Connection closed while sending messages.")


def start_recording():
    """Start recording audio from microphone."""
    global recording
    print("Starting microphone recording...")
    print("Press Ctrl+C to stop recording")
    
    # Start audio stream
    with sd.InputStream(
        samplerate=TARGET_SAMPLE_RATE,
        channels=TARGET_CHANNELS,
        dtype='float32',
        callback=audio_callback,
        blocksize=1920  # 80ms blocks
    ):
        try:
            while recording:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping recording...")
            recording = False


async def stream_audio(url: str, rtf: float):
    """Stream audio data to a WebSocket server."""
    global recording
    
    # Start recording in a separate thread
    recording_thread = threading.Thread(target=start_recording)
    recording_thread.daemon = True
    recording_thread.start()
    
    try:
        async with websockets.connect(url, additional_headers=HEADERS) as websocket:
            send_task = asyncio.create_task(send_messages(websocket, rtf))
            receive_task = asyncio.create_task(receive_messages(websocket))
            await asyncio.gather(send_task, receive_task)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        recording = False
    finally:
        recording = False
        
    print("Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time microphone transcription")
    parser.add_argument("--transcript", help="Output transcript file (JSON)")
    parser.add_argument(
        "--url",
        help="The URL of the server to which to send the audio",
        default="ws://5.9.97.57:8080",
    )
    parser.add_argument("--rtf", type=float, default=1.01, help="Real-time factor")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices")
    parser.add_argument("--device", type=int, help="Input device ID (use --list-devices to see options)")
    
    args = parser.parse_args()
    
    if args.list_devices:
        print("Available audio devices:")
        print(sd.query_devices())
        exit(0)
    
    if args.device is not None:
        sd.default.device[0] = args.device  # Set input device
    
    url = f"{args.url}/api/asr-streaming"
    
    try:
        asyncio.run(stream_audio(url, args.rtf))
        print("\nFinal transcript:")
        print(" ".join(all_text))
        
        if args.transcript is not None:
            with open(args.transcript, "w") as fobj:
                json.dump({"transcript": transcript}, fobj, indent=4)
            print(f"Transcript saved to {args.transcript}")
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        recording = False