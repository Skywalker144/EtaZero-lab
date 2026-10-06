#pragma once
#include "inference.h"
#include <torch/script.h>
#include <optional>

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
                 std::optional<torch::jit::Module> loaded_model = std::nullopt);
    TorchBackend(const TorchBackend&) = delete;
    TorchBackend& operator=(const TorchBackend&) = delete;
    // Synchronous, single-owner API. A future batch service must serialize each
    // backend and route recurrent requests back to the owner of their latent.
    std::vector<Output> initial(const InferenceInputs& inputs);
    std::vector<Output> recurrent(const std::vector<Action>& inputs);
    int latent_channels() const { return latent_channels_; }
private:
    torch::Device device_;
    torch::jit::Module model_;
    std::string precision_;
    int canvas_, max_batch_, latent_channels_;
    std::shared_ptr<const int> owner_ = std::make_shared<const int>(0);
    void check_batch(size_t n) const;
    std::vector<Output> unpack(const c10::IValue& result,
        const std::vector<std::shared_ptr<const std::vector<uint8_t>>>& masks);
};
}
