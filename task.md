# Task: Comprehensive Performance Optimizations for Moshi

## Objective
Optimize the Moshi Rust implementation (core and server) to reduce latency, increase throughput, and improve resource utilization.

## Status
- **Master Issue**: #99
- **Sub-Issues**:
  - #100: Infrastructure & Startup Optimizations (Done)
  - #101: Inference Hot Path Optimizations (Done)
  - #103: Pipeline & Transfer Optimizations (Done)
  - #104: Overlap Mimi and LM in ASR (Done)
  - #105: CUDA Graph Integration for ASR (Deferred)
  - #106: Pinned Memory for PCM Buffers (Done)

## Completed Tasks
- [x] Parallelize module loading in `moshi-server` (#100)
- [x] Parallelize file resolution and downloads during configuration load (#100)
- [x] Optimize KV cache mask generation to reduce CPU overhead (#101)
- [x] Remove redundant `contiguous()` calls in transformer hot paths (#101)
- [x] Optimize GPU-CPU transfers for logging (eager conversion to CPU vectors) (#103)
- [x] Integrate `mimalloc` high-performance allocator (#100)
- [x] Refine release profile for maximum performance (#100)
- [x] Increase TTS audio processing channel capacity (#103)
- [x] Create GitHub Master and Sub-Issues (#99)
- [x] Implement 3-stage pipeline (Encoder -> Inference -> Post-process) in `asr.rs` and `batched_asr.rs` (#104)
- [x] Use pinned memory for PCM buffers (#106)
- [x] Address `sentencepiece` system dependency in build environment

## In-Progress Tasks
- [ ] Final verification on GPU hardware (#104)

## Deferred Tasks
- [ ] Implement CUDA Graph Integration for ASR (#105) - Pending Candle native support.
