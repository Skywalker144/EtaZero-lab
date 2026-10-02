#pragma once
#include "inference.h"
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
namespace etazero {
class TorchBackend final : public Backend {
public:
    struct LoadedModel {
        std::mutex mutex;
        torch::jit::Module module;
        bool ready=false;
    };
private:
    torch::Device device_;
    torch::jit::Module model_;
    std::string path_, precision_;
    std::shared_ptr<LoadedModel> shared_model_;
    std::unique_ptr<c10::cuda::CUDAStream> stream_;
    int canvas_;
    torch::Tensor host_, global_host_;
public:
    bool supports_auxiliary() const override { return true; }
    int max_batch_;
    TorchBackend(const std::string& path, const std::string& device, int canvas, int max_batch,
                 const std::string& precision = "float32", std::shared_ptr<LoadedModel> shared_model = nullptr);
    void initialize() override;
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override;
};
}
