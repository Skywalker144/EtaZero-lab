#pragma once
#include "inference.h"
#include "etazero/prepared_model.h"
#include <torch/script.h>
#include <c10/cuda/CUDAStream.h>

namespace etazero::muzero {
// Latents belong to one backend/model/device and one tree orientation. They
// cannot be placed in AlphaZero's board-keyed, orientation-free NN cache.
class TorchBackend : public Backend {
public:
    class Latent : public etazero::muzero::Latent {
        friend class TorchBackend;
        torch::Tensor tensor_;
        std::shared_ptr<const int> owner_;
        std::shared_ptr<const std::vector<uint8_t>> mask_;
    public:
        // Diagnostic copy only; production recurrent inference keeps tensors on device.
        std::vector<float> copy_to_cpu() const;
    };
    using Output = InferenceOutput;
    using Action = InferenceAction;
    TorchBackend(const std::string& path, const std::string& device, int canvas,
                 int max_batch, const std::string& precision = "float32",
                 std::shared_ptr<PreparedModel> shared_model = nullptr);
    TorchBackend(const TorchBackend&) = delete;
    TorchBackend& operator=(const TorchBackend&) = delete;
    ~TorchBackend() override;
    // Each backend owns a stream and reusable inputs. The batch service
    // serializes calls and routes recurrent requests to the latent owner.
    std::vector<Output> initial(const InferenceInputs& inputs);
    std::vector<Output> recurrent(const std::vector<Action>& inputs);
    int latent_channels() const { return latent_channels_; }
private:
    torch::Device device_;
    torch::jit::Module model_;
    std::shared_ptr<PreparedModel> shared_model_;
    std::unique_ptr<c10::cuda::CUDAStream> stream_;
    torch::Tensor observation_host_, globals_host_, actions_host_;
    struct Graph;
    std::vector<std::unique_ptr<Graph>> initial_graphs_, recurrent_graphs_;
    std::string precision_;
    int canvas_, max_batch_, latent_channels_;
    std::shared_ptr<const int> owner_ = std::make_shared<const int>(0);
    void check_batch(size_t n) const;
    void capture_graphs();
    Graph& graph_for(const std::vector<std::unique_ptr<Graph>>& graphs,size_t n);
    c10::IValue graph_outputs(Graph& graph,size_t n);
    c10::IValue replay(const std::vector<std::unique_ptr<Graph>>& graphs,
                      const torch::Tensor& input0,const torch::Tensor& input1);
    std::vector<Output> unpack(const c10::IValue& result,
        const std::vector<std::shared_ptr<const std::vector<uint8_t>>>& masks);
};
}
