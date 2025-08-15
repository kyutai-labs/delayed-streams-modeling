#!/usr/bin/env python3

import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
import safetensors.torch


def quantize_moshi_stt_1b():
    """
    Load the 1B Moshi STT model weights and apply per-tensor uint8 quantization.
    Saves the quantized weights to the current directory in safetensors format.
    """
    print("Loading Moshi STT 1B model weights...")

    model_repo = "kyutai/stt-1b-en_fr"
    model_filename = "model.safetensors"

    try:
        # Download the original safetensors (to compare sizes apples-to-apples)
        original_model_path = hf_hub_download(
            repo_id=model_repo, filename=model_filename
        )
        state_dict = safetensors.torch.load_file(original_model_path, device="cpu")
        print(f"Successfully loaded weights from: {model_repo}/{model_filename}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Quantize
    print("Applying uint8 per-tensor quantization to float/bfloat tensors...")

    quantized_safetensors_dict = {}
    original_num_bytes = 0
    quantized_num_bytes = 0

    # dtypes we will quantize
    QUANTIZABLE_DTYPES = {torch.float32, torch.float16, torch.bfloat16}

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            # Just skip non-tensors (rare in safetensors)
            continue

        t = tensor.detach().cpu()
        original_num_bytes += t.numel() * t.element_size()

        if t.dtype in QUANTIZABLE_DTYPES:
            # Compute per-tensor min/max in fp32 for stable numerics
            t_fp32 = t.to(torch.float32)
            t_min = t_fp32.amin()
            t_max = t_fp32.amax()

            # Handle degenerate range
            # If max == min, every value is the same: store zero scale and zero_point 0, and zeros as data
            same = (t_max - t_min) <= 0
            if same:
                scale = torch.tensor(0.0, dtype=torch.float32)
                zero_point = torch.tensor(0.0, dtype=torch.float32)
                q = torch.zeros_like(t_fp32, dtype=torch.uint8)
            else:
                # Asymmetric per-tensor quantization to [0, 255]
                # q = round((x - min) / scale), scale = (max - min) / 255
                scale = (t_max - t_min) / 255.0
                zero_point = t_min
                q = (
                    torch.round((t_fp32 - zero_point) / scale)
                    .clamp(0, 255)
                    .to(torch.uint8)
                )

            # Save quantized weights and metadata (float32!) into safetensors dict
            quantized_safetensors_dict[name] = q.contiguous()
            quantized_safetensors_dict[f"{name}.__scale__"] = scale.to(torch.float32)
            quantized_safetensors_dict[f"{name}.__zero_point__"] = zero_point.to(
                torch.float32
            )

            # Size accounting (weights + 8 bytes metadata ≈ negligible vs weight tensor; but we’ll count exactly)
            quantized_num_bytes += q.numel() * q.element_size()
            quantized_num_bytes += 4 + 4  # scale + zero_point as float32 scalars
        else:
            # Keep non-float tensors as-is
            quantized_safetensors_dict[name] = t
            quantized_num_bytes += t.numel() * t.element_size()

    print("Quantization completed!")

    # Prepare output directory
    output_dir = Path.cwd() / "moshi_stt_1b_uint8"
    output_dir.mkdir(exist_ok=True)

    # Save quantized model as safetensors
    safetensors_file = output_dir / "quantized_model.safetensors"
    safetensors.torch.save_file(quantized_safetensors_dict, str(safetensors_file))
    print(f"Quantized safetensors model saved to: {safetensors_file}")

    # Report sizes:
    # (a) original HF safetensors file size
    original_file_size_mb = Path(original_model_path).stat().st_size / (1024 * 1024)
    # (b) quantized safetensors file size
    quantized_file_size_mb = Path(safetensors_file).stat().st_size / (1024 * 1024)

    # (c) raw memory accounting we computed
    original_mem_mb = original_num_bytes / (1024 * 1024)
    quantized_mem_mb = quantized_num_bytes / (1024 * 1024)

    print("\n=== Size Report ===")
    print(f"Original safetensors file: {original_file_size_mb:.2f} MB")
    print(f"Quantized safetensors file: {quantized_file_size_mb:.2f} MB")
    print(
        f"Raw memory (calc): original ~{original_mem_mb:.2f} MB -> quantized ~{quantized_mem_mb:.2f} MB"
    )
    if quantized_mem_mb > 0:
        print(
            f"Estimated memory compression: {original_mem_mb / quantized_mem_mb:.2f}x"
        )
    if quantized_file_size_mb > 0:
        print(
            f"File compression: {original_file_size_mb / quantized_file_size_mb:.2f}x"
        )

    print("\nTo use with Rust, load tensors by name and dequantize as:")
    print("x ≈ uint8_tensor.float() * scale + zero_point")
    print(
        "(Using the matching '<name>.__scale__' and '<name>.__zero_point__' entries.)"
    )


if __name__ == "__main__":
    quantize_moshi_stt_1b()
