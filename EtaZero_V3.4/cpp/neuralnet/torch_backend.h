#pragma once
#include "neuralnet/inference.h"
#include "neuralnet/prepared_model.h"
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
    int input_planes_=5, global_features_=8, action_count_=0;
    int canvas_;
    int metadata_channels_=0;
    std::string requested_profile_;
    bool precision_checked_=false;
    torch::Tensor host_, global_host_, input_, global_input_;
    std::vector<std::unique_ptr<Graph>> graphs_;
    torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& globals);
    void capture_graphs();
public:
    bool supports_auxiliary() const override { return true; }
    int metadata_channels() const override {return metadata_channels_;}
    ModelDescription description() const override {ModelDescription d;d.name=path_;d.uses_human_profile=metadata_channels_>0;return d;}
    int max_batch_;
    TorchBackend(const std::string& path, const std::string& device, int canvas, int max_batch,
                 const std::string& precision = "float32", std::shared_ptr<LoadedModel> shared_model = nullptr, const std::string& profile="");
    ~TorchBackend() override;
    void initialize() override;
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override;
};
}
