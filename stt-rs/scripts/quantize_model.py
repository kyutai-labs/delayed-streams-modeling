#!/usr/bin/env python3

import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
import safetensors.torch


def quantize_moshi_stt_1b():
    """
    Load the 1B Moshi STT model weights and apply int8 quantization.
    Saves the quantized weights to the current directory.
    """
    print("Loading Moshi STT 1B model weights...")

    model_name = "kyutai/stt-1b-en_fr"

    try:
        # Download the model weights
        model_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")

        # Load the state dict
        state_dict = safetensors.torch.load_file(model_path)
        print(f"Successfully loaded weights from: {model_name}")

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Apply int8 quantization to weights
    print("Applying int8 quantization to weights...")

    quantized_state_dict = {}
    original_size = 0
    quantized_size = 0

    for name, tensor in state_dict.items():
        original_size += tensor.numel() * tensor.element_size()

        # Quantize float tensors to int8
        if tensor.dtype in [torch.float32, torch.float16]:
            # Simple linear quantization
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / 255.0
            zero_point = min_val

            # Quantize
            quantized = (
                torch.round((tensor - zero_point) / scale).clamp(0, 255).to(torch.uint8)
            )

            # Store quantized tensor with scale and zero_point for dequantization
            quantized_state_dict[name] = {
                "quantized": quantized,
                "scale": scale,
                "zero_point": zero_point,
                "original_shape": tensor.shape,
                "original_dtype": tensor.dtype,
            }
            quantized_size += (
                quantized.numel() + 2 * 4
            )  # +8 bytes for scale and zero_point
        else:
            # Keep non-float tensors as-is
            quantized_state_dict[name] = tensor
            quantized_size += tensor.numel() * tensor.element_size()

    print("Quantization completed!")

    # Save the quantized weights to current directory
    output_dir = Path.cwd() / "moshi_stt_1b_int8"
    output_dir.mkdir(exist_ok=True)

    # Save the quantized weights
    torch.save(quantized_state_dict, output_dir / "quantized_weights.pth")

    # Save the original weights for reference
    torch.save(state_dict, output_dir / "original_weights.pth")

    print(f"Quantized weights saved to: {output_dir}")

    # Print size comparison
    print(f"Original size: {original_size / (1024 * 1024):.2f} MB")
    print(f"Quantized size: {quantized_size / (1024 * 1024):.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")


if __name__ == "__main__":
    quantize_moshi_stt_1b()
