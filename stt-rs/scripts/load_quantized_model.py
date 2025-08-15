#!/usr/bin/env python3

import torch


def load_quantized_weights():
    """Load and dequantize the Moshi STT 1B weights."""

    # Load the quantized weights
    quantized_data = torch.load("./moshi_stt_1b_int8/quantized_weights.pth")

    # Dequantize the weights
    dequantized_state_dict = {}

    for name, data in quantized_data.items():
        if isinstance(data, dict) and "quantized" in data:
            # Dequantize
            quantized = data["quantized"]
            scale = data["scale"]
            zero_point = data["zero_point"]
            original_dtype = data["original_dtype"]

            # Dequantize: float_val = scale * (quantized - zero_point)
            dequantized = (quantized.to(torch.float32) * scale + zero_point).to(
                original_dtype
            )
            dequantized_state_dict[name] = dequantized
        else:
            # Non-quantized tensor
            dequantized_state_dict[name] = data

    print("Quantized weights loaded and dequantized successfully!")
    return dequantized_state_dict


if __name__ == "__main__":
    weights = load_quantized_weights()
    print(f"Loaded {len(weights)} weight tensors")
