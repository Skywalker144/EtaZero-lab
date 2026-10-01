#pragma once
#include "inference.h"
#include <torch/script.h>
namespace etazero {
class TorchBackend final : public Backend {
    torch::Device device_;
    torch::jit::Module model_;
    int canvas_;
    torch::Tensor host_;
public:
    TorchBackend(const std::string& path, const std::string& device, int canvas, int max_batch);
    std::vector<Evaluation> evaluate(const std::vector<std::vector<float>>& inputs) override;
};
}
