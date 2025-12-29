# Implementation Plan: STT Server Performance Refinement

This plan covers refining the Speech-to-Text (STT) server with recent performance optimizations, focusing on pipelining, CUDA graphs, and memory efficiency.

## Master Issue: #99

### 1. Pipelining Mimi and LM in Batched ASR (#104)
- **Goal**: Reduce inter-token latency by overlapping Mimi encoding and LM inference in `batched_asr.rs`.
- **Status**: Implemented. Refactored `BatchedAsrInner::start_model_loop` to use a multi-stage pipeline. `mimi.encode_step` is now separated from `lm.forward_cond` and `post_process` stages, allowing for overlapping computation. This involved creating `PipelineMsg` and `PostProcessMsg` channels for inter-stage communication.
- **Next Steps**: Verification on GPU-enabled hardware to confirm performance gains.

### 2. CUDA Graph Integration for ASR (#105)
- **Goal**: Reduce kernel launch overhead in the streaming ASR inference loop.
- **Status**: Deferred. Research indicates Candle 0.9.1 lacks native capture APIs for complex transformer models. Manual capture via `cudarc` is high-risk. Pipelining already mitigates launch overhead.

### 3. Pinned Memory & Async Transfers (#106)
- **Goal**: Improve GPU-CPU transfer speeds for PCM input and token output.
- **Status**: Completed. Implemented pinned memory allocation for PCM buffers in `asr.rs` and `batched_asr.rs` using `CudaContext::alloc_pinned`. Added graceful fallbacks for CPU/non-CUDA environments.

### 4. Final Verification & Benchmarking
- **Goal**: Quantify improvements and ensure no regressions.
- **Status**: Initial verification with `cargo check` passed. Full performance benchmarking pending due to build environment issues (missing `libsentencepiece-dev`).
- **Next Steps**: Resolve `sentencepiece` dependency, then run `nsys profile` and compare with baselines.

## Verification Plan
- Run existing benchmarks: `cargo bench` or equivalent if available.
- Use `nsys` profiles to verify kernel launch overhead reduction and overlap.
- Document results in `walkthrough.md`.

## Prerequisites for Agent Handoff
- **System Dependency**: `libsentencepiece-dev` must be installed on the system to resolve the `sentencepiece_processor.h` build error.
