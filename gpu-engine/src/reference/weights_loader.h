#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Host-only loader for weights.bin + manifest.json (as produced by
// export/export_gpt2.py). Used by the CPU oracle, which needs plain float
// host buffers -- no CUDA. The GPU engine's src/loader.cpp reads the same
// files but uploads to device memory and downcasts to fp16.
class WeightsLoader {
public:
    struct Tensor {
        std::vector<int64_t> shape;
        const float* data = nullptr;  // points into blob_
        size_t num_elements() const;
    };

    // Reads manifest.json + weights.bin from dir. Throws std::runtime_error
    // on any mismatch (missing file, truncated blob, checksum mismatch).
    explicit WeightsLoader(const std::string& dir);

    const Tensor& get(const std::string& name) const;
    bool has(const std::string& name) const;

    const std::string& checksum_sha256() const { return checksum_sha256_; }

private:
    std::vector<uint8_t> blob_;
    std::unordered_map<std::string, Tensor> tensors_;
    std::string checksum_sha256_;
};
