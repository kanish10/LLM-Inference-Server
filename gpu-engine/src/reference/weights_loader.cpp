#include "weights_loader.h"

#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

size_t WeightsLoader::Tensor::num_elements() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
}

WeightsLoader::WeightsLoader(const std::string& dir) {
    const std::string manifest_path = dir + "/manifest.json";
    const std::string weights_path = dir + "/weights.bin";

    std::ifstream mf(manifest_path);
    if (!mf) throw std::runtime_error("cannot open manifest: " + manifest_path);
    nlohmann::json manifest;
    mf >> manifest;

    std::ifstream wf(weights_path, std::ios::binary | std::ios::ate);
    if (!wf) throw std::runtime_error("cannot open weights: " + weights_path);
    const size_t file_bytes = static_cast<size_t>(wf.tellg());
    wf.seekg(0);

    const size_t expected_bytes = manifest.at("total_bytes").get<size_t>();
    if (file_bytes != expected_bytes) {
        throw std::runtime_error(
            "weights.bin size mismatch: file=" + std::to_string(file_bytes) +
            " manifest=" + std::to_string(expected_bytes));
    }

    blob_.resize(file_bytes);
    wf.read(reinterpret_cast<char*>(blob_.data()), static_cast<std::streamsize>(file_bytes));
    if (!wf) throw std::runtime_error("short read on weights.bin");

    checksum_sha256_ = manifest.value("checksum_sha256", "");

    // Recompute the double-precision element sum in the same order the
    // exporter wrote tensors, and cross-check against manifest checksum_sum.
    double recomputed_sum = 0.0;

    for (const auto& t : manifest.at("tensors")) {
        Tensor tensor;
        for (auto& d : t.at("shape")) tensor.shape.push_back(d.get<int64_t>());
        const size_t offset = t.at("byte_offset").get<size_t>();
        const size_t length = t.at("byte_length").get<size_t>();
        if (offset + length > blob_.size()) {
            throw std::runtime_error("tensor byte range out of bounds: " + t.at("name").get<std::string>());
        }
        tensor.data = reinterpret_cast<const float*>(blob_.data() + offset);

        const size_t n = tensor.num_elements();
        if (n * sizeof(float) != length) {
            throw std::runtime_error("tensor shape/byte_length mismatch: " + t.at("name").get<std::string>());
        }
        for (size_t i = 0; i < n; ++i) recomputed_sum += static_cast<double>(tensor.data[i]);

        tensors_.emplace(t.at("name").get<std::string>(), std::move(tensor));
    }

    const double manifest_sum = manifest.value("checksum_sum", 0.0);
    const double denom = std::max(1.0, std::abs(manifest_sum));
    const double rel_err = std::abs(recomputed_sum - manifest_sum) / denom;
    std::fprintf(stderr, "[WeightsLoader] checksum_sum recomputed=%.6f manifest=%.6f rel_err=%.3e\n",
                 recomputed_sum, manifest_sum, rel_err);
    if (rel_err > 1e-6) {
        throw std::runtime_error("weight checksum_sum mismatch vs manifest (possible corrupt export)");
    }
}

bool WeightsLoader::has(const std::string& name) const {
    return tensors_.find(name) != tensors_.end();
}

const WeightsLoader::Tensor& WeightsLoader::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("unknown tensor: " + name);
    return it->second;
}
