# Walkthrough: Performance Optimizations (Final)

I have implemented and refined several performance optimizations across the Moshi server and core components.

## 1. Configurable Hardware Optimizations
- **CUDA Event Tracking**: Refined `--disable-cuda-events` flag in `moshi-server worker`. When enabled, it calls `d.disable_event_tracking()`, reducing kernel launch overhead by avoiding unnecessary event synchronizations.
- **TF32 Support**: Added/Refined `--enable-tf32` flag (defaults to true). This allows the GPU to use Tensor Float 32 for faster matrix multiplications on Ampere+ architectures.
- **Rotary Embedding Optimization**: Optimized `RotaryEmbedding::rope` in `transformer.rs` to use `broadcast_mul` instead of `matmul` for outer products, which is significantly more efficient in Candle.

## 2. Verified Flash Attention
- **Flash Attention Logging**: Added `tracing::debug!("using flash_attn")` in `transformer.rs` to allow easy verification of its usage.
- **Device Safety**: Ensured Flash Attention is only invoked on CUDA devices, preventing crashes on other hardware.

## 3. Optimized Device Transfers & Non-Blocking Logging
- **Asynchronous Logging**: In `asr.rs`, `tts.rs`, and `lm.rs`, moved all GPU->CPU transfers, tensor concatenations, and file I/O for token logging into dedicated background tasks. This ensures the main inference loop never blocks on telemetry or persistence.
- **Fast Tensor Creation**: Replaced `Tensor::new` with `Tensor::from_vec` where possible for faster CPU->GPU data transfers.
- **Improved Buffer Management**: Increased channel capacities (e.g., from 10 to 100 in ASR) to better handle bursts of data and prevent pipeline stalls.

## 4. Advanced Pipelining (Overlapping Stages)
- **3-Stage ASR Pipeline**: Successfully implemented a high-performance 3-stage pipeline in `batched_asr.rs` and `asr.rs`:
  1. **Stage 1 (Encoder)**: Mimi encoding and PCM pre-processing.
  2. **Stage 2 (Inference)**: LM token prediction (the most compute-intensive part).
  3. **Stage 3 (Post-process)**: Token decoding, word assembly, and result distribution.
  This architecture allows overlapping the heavy LM inference of step $N$ with the Mimi encoding of step $N+1$ and the post-processing of step $N-1$, significantly improving throughput and reducing pipeline bubbles.
- **Single-Stream ASR Pipelining**: Refactored single-stream `asr.rs` to follow the same 3-stage pattern, ensuring consistency across the codebase.
- **Exposed Core State**: Refactored `moshi::asr::State` in `moshi-core` to expose necessary fields (like `asr_delay_in_tokens`), enabling external control required for high-performance pipelining.

## 5. Algorithmic & Tensor Optimizations
- **Efficient Mask Generation**: Rewrote KV cache and transformer attention mask generation to use pure tensor operations on the device. This eliminates large CPU loops and host-to-device transfers, providing a significant speedup for long context windows.
- **Tensorized Repetition Penalty**: Optimized the repetition penalty logic in `lm_generate_multistream.rs` to use tensor operations on the device, avoiding expensive GPU-CPU synchronizations in the hot path.
- **F16 Matmuls for Quantization**: Updated `matmul_dtype` to use `F16` on CUDA devices during quantized execution. This leverages specialized hardware units (Tensor Cores) for much faster intermediate attention calculations compared to `F32`.

## 6. Code Quality & Reliability
- **Simplified Decoding**: Streamlined `TextDecoder::text` in `lm.rs` to reduce overhead and remove unused variables.
- **Clean Shutdown Logic**: Robustly updated socket handlers to ensure background tasks are properly aborted or awaited on connection close, preventing memory leaks and orphaned tasks.
- **Compilation Fixes**: Resolved all moved value and type mismatch issues introduced during optimization.
- **Optimized Logging**: Moved all `to_vec1` (GPU->CPU) transfers for token logging into background threads for both single-stream and batched ASR, ensuring zero impact on the critical path.

## 7. Build & Environment Fixes
- **Workspace Dependency Management**: Fixed workspace build errors by adding `mimalloc` to the root `Cargo.toml`.
- **Warning Cleanup**: Addressed several compiler warnings regarding unused variables and mismatched types in the hot paths.
- **Prerequisite Note**: Building the project requires `libsentencepiece-dev` to be installed on the system (provides `sentencepiece_processor.h`).
- **CUDA 13.1 Compatibility**: Added an environment setup script `ops/setup_env.sh` to handle CUDA 13.1 compatibility by spoofing `nvcc` version 12.9. This is necessary because `cudarc 0.16.6` (required by Candle 0.9.1) does not natively support CUDA 13.x version strings yet.
- **Build Command**: To build with GPU support, use: `source ops/setup_env.sh && cargo build -p moshi-server --features cuda`.

## 9. Pinned Memory for PCM Transfers
- **Asynchronous PCM Transfers**: Implemented pinned memory allocation for PCM buffers in both `asr.rs` and `batched_asr.rs`. Using `CudaContext::alloc_pinned` provides page-locked host memory that enables significantly faster and truly asynchronous host-to-device transfers.
- **Efficient Tensor Creation**: Replaced `Tensor::from_vec` (which involves a copy to a standard `Vec` first) with `Tensor::from_slice` directly on the pinned buffer, reducing CPU overhead and memory bandwidth pressure during streaming inference.
- **Graceful Fallback**: The implementation includes automatic fallback to standard heap-allocated `Vec` if CUDA is unavailable or if pinned memory allocation fails, ensuring robustness across different hardware environments.

## 10. CUDA Graph Feasibility Study
- **Current Status**: Researched CUDA Graph integration for the ASR inference loop. While `cudarc` supports graph capture and launch, Candle 0.9.1 does not yet provide a high-level abstraction for capturing its kernel launches into a reusable graph.
- **Recommendation**: Postponed full implementation until Candle provides a native `Capture` API or equivalent, as manual capture via `cudarc` is risky and could interfere with Candle's internal stream management. Pipelining already addresses much of the launch overhead by overlapping computation.
