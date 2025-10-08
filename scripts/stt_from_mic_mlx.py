"""
Real-time Speech-to-Text Transcription Application

This application performs real-time speech-to-text transcription using the Moshi model
from Kyutai Labs. It captures audio from the microphone, processes it through an audio
tokenizer (Mimi), and uses a language model to generate text transcriptions.

Features:
- Real-time audio capture and transcription
- Support for English and French languages
- Optional Voice Activity Detection (VAD) to detect end of speech
- Uses MLX framework for efficient inference on Apple Silicon
- Model quantization support (4-bit and 8-bit)

Requirements:
- Python 3.12+
- Apple Silicon Mac (for MLX support)
- Microphone for audio input
- ~2.5GB RAM for model inference
- ~2.1GB disk space for model cache

Usage:
    # Basic usage with MLX model
    python app.py
    
    # With Voice Activity Detection
    python app.py --vad
    
    # Custom max steps
    python app.py --max-steps 8192
    
    # Custom HuggingFace repository
    python app.py --hf-repo kyutai/stt-1b-en_fr-mlx
"""

import argparse
import json
import queue

# MLX framework for efficient Apple Silicon inference
import mlx.core as mx
import mlx.nn as nn
# Audio tokenizer (Mimi) for encoding audio into discrete tokens
import rustymimi
# Text tokenizer for converting token IDs to text pieces
import sentencepiece
# Audio I/O library for microphone input
import sounddevice as sd
# HuggingFace Hub for downloading model files
from huggingface_hub import hf_hub_download
# Moshi model and utilities
from moshi_mlx import models, utils

if __name__ == "__main__":
    # ============================================================================
    # ARGUMENT PARSING
    # ============================================================================
    print("Starting app.py...")
    parser = argparse.ArgumentParser()
    # Maximum number of generation steps (affects how long the model will run)
    parser.add_argument("--max-steps", default=4096)
    # HuggingFace repository containing the model files
    parser.add_argument("--hf-repo")
    # Enable Voice Activity Detection to detect when user stops speaking
    parser.add_argument(
        "--vad", action="store_true", help="Enable VAD (Voice Activity Detection)."
    )
    args = parser.parse_args()
    print(f"Arguments parsed: max_steps={args.max_steps}, vad={args.vad}, hf_repo={args.hf_repo}")

    # ============================================================================
    # MODEL FILE DOWNLOAD/LOCATION
    # ============================================================================
    # Select default repository based on VAD flag
    # - MLX models are optimized for Apple Silicon
    # - Candle models are PyTorch-based and used when VAD is enabled
    if args.hf_repo is None:
        if args.vad:
            args.hf_repo = "kyutai/stt-1b-en_fr-candle"
        else:
            args.hf_repo = "kyutai/stt-1b-en_fr-mlx"
    print(f"Using HuggingFace repo: {args.hf_repo}")
    
    # Download or locate model configuration file (~1.2KB)
    print("Downloading config.json...")
    lm_config = hf_hub_download(args.hf_repo, "config.json")
    print("Config downloaded, loading...")
    with open(lm_config, "r") as fobj:
        lm_config = json.load(fobj)
    
    # Download or locate Mimi audio tokenizer weights (~367MB)
    # Mimi encodes raw audio into discrete tokens
    print(f"Downloading mimi weights: {lm_config['mimi_name']}...")
    mimi_weights = hf_hub_download(args.hf_repo, lm_config["mimi_name"])
    
    # Download or locate Moshi language model weights (~1.8GB)
    # This is the main speech-to-text model
    moshi_name = lm_config.get("moshi_name", "model.safetensors")
    print(f"Downloading moshi weights: {moshi_name}...")
    moshi_weights = hf_hub_download(args.hf_repo, moshi_name)
    
    # Download or locate text tokenizer (~118KB)
    # Converts token IDs to text pieces
    print(f"Downloading tokenizer: {lm_config['tokenizer_name']}...")
    tokenizer = hf_hub_download(args.hf_repo, lm_config["tokenizer_name"])

    # ============================================================================
    # MODEL INITIALIZATION
    # ============================================================================
    # Create model configuration from downloaded JSON
    print("Creating model configuration...")
    lm_config = models.LmConfig.from_config_dict(lm_config)
    
    # Initialize the Moshi language model structure
    # This creates the neural network layers but doesn't load weights yet
    print("Initializing model...")
    model = models.Lm(lm_config)
    
    # Set model to use bfloat16 precision for efficient inference
    # bfloat16 reduces memory usage while maintaining good accuracy
    model.set_dtype(mx.bfloat16)
    
    # Apply quantization if the model file indicates it
    # Quantization reduces model size and speeds up inference
    if moshi_weights.endswith(".q4.safetensors"):
        # 4-bit quantization: smallest size, fastest inference, slight accuracy loss
        print("Quantizing model to 4-bit...")
        nn.quantize(model, bits=4, group_size=32)
    elif moshi_weights.endswith(".q8.safetensors"):
        # 8-bit quantization: balanced size/speed/accuracy
        print("Quantizing model to 8-bit...")
        nn.quantize(model, bits=8, group_size=64)

    # Load the actual model weights from disk into memory
    # This step is CPU/memory intensive and may take 30-60 seconds
    print(f"loading model weights from {moshi_weights}")
    if args.hf_repo.endswith("-candle"):
        # Load PyTorch-format weights (for Candle models)
        model.load_pytorch_weights(moshi_weights, lm_config, strict=True)
    else:
        # Load MLX-format weights (for MLX models)
        model.load_weights(moshi_weights, strict=True)

    # ============================================================================
    # TOKENIZER INITIALIZATION
    # ============================================================================
    # Initialize text tokenizer (SentencePiece) for decoding text tokens
    # Converts integer token IDs to readable text pieces
    print(f"loading the text tokenizer from {tokenizer}")
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)  # type: ignore

    # Initialize audio tokenizer (Mimi) for encoding audio
    # Converts raw audio waveforms into discrete tokens
    print(f"loading the audio tokenizer {mimi_weights}")
    generated_codebooks = lm_config.generated_codebooks
    other_codebooks = lm_config.other_codebooks
    mimi_codebooks = max(generated_codebooks, other_codebooks)
    audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)  # type: ignore
    
    # ============================================================================
    # MODEL WARMUP AND GENERATION SETUP
    # ============================================================================
    # Run warmup to compile/optimize the model for faster inference
    # This runs a few test inferences to optimize the computation graph
    print("warming up the model")
    model.warmup()
    
    # Create generation manager with sampling parameters
    gen = models.LmGen(
        model=model,
        max_steps=args.max_steps,
        # Text sampling: use top-k=25 with temperature=0 (deterministic)
        text_sampler=utils.Sampler(top_k=25, temp=0),
        # Audio sampling: use top-k=250 with temperature=0.8 (more diverse)
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        check=False,
    )

    # ============================================================================
    # AUDIO STREAMING SETUP
    # ============================================================================
    # Create a queue to pass audio data from the callback to the main loop
    # Thread-safe queue allows audio callback to run on separate thread
    block_queue = queue.Queue()

    def audio_callback(indata, _frames, _time, _status):
        """
        Audio callback function called by sounddevice for each audio block.
        
        Args:
            indata: Audio data as numpy array (frames x channels)
            _frames: Number of frames (unused)
            _time: Time information (unused)
            _status: Status flags (unused)
        """
        # Copy audio data to queue for processing in main loop
        # Copy is needed because buffer may be reused
        block_queue.put(indata.copy())

    # ============================================================================
    # REAL-TIME TRANSCRIPTION LOOP
    # ============================================================================
    print("recording audio from microphone, speak to get your words transcribed")
    last_print_was_vad = False
    
    # Open audio input stream with 24kHz mono audio
    # blocksize=1920 means 80ms of audio per block (1920/24000 = 0.08s)
    with sd.InputStream(
        channels=1,          # Mono audio
        dtype="float32",     # 32-bit floating point samples
        samplerate=24000,    # 24kHz sample rate (required by Mimi)
        blocksize=1920,      # 80ms blocks for low latency
        callback=audio_callback,
    ):
        # Main processing loop - runs indefinitely until interrupted (Ctrl+C)
        while True:
            # Get next audio block from queue (blocks if queue is empty)
            block = block_queue.get()
            
            # Reshape audio: add batch dimension and remove channel dimension
            # Shape: (frames,) -> (1, frames)
            block = block[None, :, 0]
            
            # Encode audio to tokens using Mimi audio tokenizer
            # This converts raw audio waveform to discrete tokens
            other_audio_tokens = audio_tokenizer.encode_step(block[None, 0:1])
            
            # Convert to MLX array and transpose to expected shape
            # Extract only the codebooks we need (other_codebooks)
            other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
                :, :, :other_codebooks
            ]
            
            # Run model inference to get text token
            if args.vad:
                # VAD mode: also get Voice Activity Detection heads
                text_token, vad_heads = gen.step_with_extra_heads(other_audio_tokens[0])
                if vad_heads:
                    # Check VAD probability (> 0.5 means end of speech detected)
                    pr_vad = vad_heads[2][0, 0, 0].item()
                    if pr_vad > 0.5 and not last_print_was_vad:
                        print(" [end of turn detected]")
                        last_print_was_vad = True
            else:
                # Normal mode: just get text token
                text_token = gen.step(other_audio_tokens[0])
            
            # Extract text token ID from tensor
            text_token = text_token[0].item()
            
            # Get generated audio tokens (for future use if needed)
            audio_tokens = gen.last_audio_tokens()
            
            # Decode and print text if token is not special token
            # Token 0: padding, Token 3: end-of-sequence
            _text = None
            if text_token not in (0, 3):
                # Convert token ID to text piece using SentencePiece
                _text = text_tokenizer.id_to_piece(text_token)  # type: ignore
                
                # Replace SentencePiece underscore with space
                # SentencePiece uses ▁ to represent spaces
                _text = _text.replace("▁", " ")
                
                # Print text without newline for streaming output
                print(_text, end="", flush=True)
                last_print_was_vad = False
