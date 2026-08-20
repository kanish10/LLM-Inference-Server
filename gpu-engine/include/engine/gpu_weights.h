#pragma once
#include <cuda_fp16.h>

#include <string>
#include <vector>

#include "engine/model.h"

struct GPT2LayerWeightsGPU {
    const half* ln1_g;
    const half* ln1_b;
    const half* qkv_w;      // [3*d_model, d_model]
    const half* qkv_b;      // [3*d_model]
    const half* proj_w;     // [d_model, d_model]
    const half* proj_b;     // [d_model]
    const half* ln2_g;
    const half* ln2_b;
    const half* fc_w;       // [4*d_model, d_model]
    const half* fc_b;       // [4*d_model]
    const half* fc_proj_w;  // [d_model, 4*d_model]
    const half* fc_proj_b;  // [d_model]
};

// All pointers below point into one contiguous cudaMalloc'd arena
// (device_blob). Weights are downcast from the fp32 weights.bin to fp16 at
// load time (accumulation during inference stays FP32; only storage is
// fp16, per the precision constraint in gpu-engine/CLAUDE.md).
struct GPT2WeightsGPU {
    const half* wte = nullptr;  // [vocab, d_model]; lm_head aliases this (weight tying)
    const half* wpe = nullptr;  // [n_ctx, d_model]
    const half* lnf_g = nullptr;
    const half* lnf_b = nullptr;
    std::vector<GPT2LayerWeightsGPU> layers;

    void* device_blob = nullptr;
    size_t device_blob_bytes = 0;
};

// Reads weights_dir/{weights.bin,manifest.json} (fp32 on disk, same format
// export_gpt2.py and the CPU oracle use), downcasts to fp16, and uploads
// into one contiguous device allocation. Cross-checks the same
// checksum_sum the CPU loader checks. Throws std::runtime_error on any
// mismatch.
GPT2WeightsGPU load_gpt2_weights_gpu(const std::string& weights_dir, const GPT2Config& config);

// Frees device_blob. Safe to call on a default-constructed (never-loaded)
// GPT2WeightsGPU.
void free_gpt2_weights_gpu(GPT2WeightsGPU& weights);
