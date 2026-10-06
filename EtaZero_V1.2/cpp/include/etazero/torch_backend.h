#pragma once
#include "inference.h"
#include "prepared_model.h"
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>
namespace etazero {
class TorchBackend final : public Backend {
public:
    using LoadedModel = PreparedModel;
private:
    struct Graph;
    torch::Device device_;
    torch::jit::Module model_;
    std::string path_, precision_;
    std::shared_ptr<LoadedModel> shared_model_;
    std::unique_ptr<c10::cuda::CUDAStream> stream_;
    int canvas_;
    bool precision_checked_=false;
    torch::Tensor host_, global_host_, input_, global_input_;
    std::vector<std::unique_ptr<Graph>> graphs_;
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& globals);
    void capture_graphs();
public:
    bool supports_auxiliary() const override { return true; }
    int max_batch_;
    TorchBackend(const std::string& path, const std::string& device, int canvas, int max_batch,
                 const std::string& precision = "float32", std::shared_ptr<LoadedModel> shared_model = nullptr);
    ~TorchBackend() override;
    void initialize() override;
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override;
};
}
