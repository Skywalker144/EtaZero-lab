#pragma once
#include <torch/script.h>
#include <optional>

namespace etazero {
// Shared immutable inference weights and JIT plans. Backends own CUDA graphs,
// streams and inputs; destroy them before offloading this model between rounds.
class PreparedModel {
    std::optional<torch::jit::Module> module_, supplied_;
    std::string layout_, precision_;
    bool ready_=false;
public:
    struct Timing {
        double load_seconds=0, weights_seconds=0, graph_seconds=0;
        bool reused_runtime=false;
    } timing;
    PreparedModel() = default;
    explicit PreparedModel(torch::jit::Module module):supplied_(std::move(module)) {}
    torch::jit::Module prepare(const std::string& path, const torch::Device& device,
                               const std::string& precision);
    void offload();
};
}
