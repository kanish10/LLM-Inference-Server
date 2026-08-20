#include "engine/gpu_weights.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "reference/weights_loader.h"

namespace {

void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

GPT2WeightsGPU load_gpt2_weights_gpu(const std::string& weights_dir, const GPT2Config& config) {
    // WeightsLoader validates weights.bin against manifest.json (size,
    // shapes, checksum_sum) exactly as the CPU oracle does -- same
    // guarantee, different consumer.
    WeightsLoader host_weights(weights_dir);

    std::vector<std::string> names;
    names.push_back("wte");
    names.push_back("wpe");
    for (int i = 0; i < config.n_layer; ++i) {
        const std::string p = "layer" + std::to_string(i) + ".";
        names.push_back(p + "ln1_g");
        names.push_back(p + "ln1_b");
        names.push_back(p + "qkv_w");
        names.push_back(p + "qkv_b");
        names.push_back(p + "proj_w");
        names.push_back(p + "proj_b");
        names.push_back(p + "ln2_g");
        names.push_back(p + "ln2_b");
        names.push_back(p + "fc_w");
        names.push_back(p + "fc_b");
        names.push_back(p + "fc_proj_w");
        names.push_back(p + "fc_proj_b");
    }
    names.push_back("lnf_g");
    names.push_back("lnf_b");

    // Downcast every tensor to fp16 in one host-side staging buffer, laid
    // out in the same order as `names`, and remember each tensor's element
    // offset so device pointers can be computed after a single upload.
    std::vector<half> staging;
    std::vector<size_t> elem_offsets;
    elem_offsets.reserve(names.size());

    for (const auto& name : names) {
        const auto& t = host_weights.get(name);
        elem_offsets.push_back(staging.size());
        const size_t n = t.num_elements();
        staging.resize(staging.size() + n);
        half* dst = staging.data() + (staging.size() - n);
        for (size_t i = 0; i < n; ++i) dst[i] = __float2half(t.data[i]);
    }

    GPT2WeightsGPU gpu;
    gpu.device_blob_bytes = staging.size() * sizeof(half);
    cuda_check(cudaMalloc(&gpu.device_blob, gpu.device_blob_bytes), "cudaMalloc weights arena");
    cuda_check(cudaMemcpy(gpu.device_blob, staging.data(), gpu.device_blob_bytes,
                           cudaMemcpyHostToDevice),
               "cudaMemcpy weights arena");

    const half* base = static_cast<const half*>(gpu.device_blob);
    size_t idx = 0;
    gpu.wte = base + elem_offsets[idx++];
    gpu.wpe = base + elem_offsets[idx++];

    gpu.layers.resize(config.n_layer);
    for (int i = 0; i < config.n_layer; ++i) {
        GPT2LayerWeightsGPU& lw = gpu.layers[i];
        lw.ln1_g = base + elem_offsets[idx++];
        lw.ln1_b = base + elem_offsets[idx++];
        lw.qkv_w = base + elem_offsets[idx++];
        lw.qkv_b = base + elem_offsets[idx++];
        lw.proj_w = base + elem_offsets[idx++];
        lw.proj_b = base + elem_offsets[idx++];
        lw.ln2_g = base + elem_offsets[idx++];
        lw.ln2_b = base + elem_offsets[idx++];
        lw.fc_w = base + elem_offsets[idx++];
        lw.fc_b = base + elem_offsets[idx++];
        lw.fc_proj_w = base + elem_offsets[idx++];
        lw.fc_proj_b = base + elem_offsets[idx++];
    }

    gpu.lnf_g = base + elem_offsets[idx++];
    gpu.lnf_b = base + elem_offsets[idx++];

    return gpu;
}

void free_gpt2_weights_gpu(GPT2WeightsGPU& weights) {
    if (weights.device_blob) {
        cudaFree(weights.device_blob);
        weights.device_blob = nullptr;
        weights.device_blob_bytes = 0;
    }
}
