#include "etazero/torch_backend.h"
#include "etazero/schema.h"
#include <c10/core/InferenceMode.h>
#include <cstring>
namespace etazero {
TorchBackend::TorchBackend(const std::string& path, const std::string& device, int canvas, int max_batch)
    : device_(device), model_(torch::jit::load(path, device_)), canvas_(canvas) {
    model_.eval();
    auto meta = model_.get_method("metadata")({}).toTuple();
    if (meta->elements().size() != 2 || meta->elements()[0].toInt() != canvas || meta->elements()[1].toStringRef() != CONTRACT_ID)
        throw std::runtime_error("Model contract/canvas mismatch");
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(device_.is_cuda());
    host_ = torch::empty({max_batch, INPUT_PLANES, canvas, canvas}, options);
}
std::vector<Evaluation> TorchBackend::evaluate(const std::vector<std::vector<float>>& inputs) {
    c10::InferenceMode guard;
    const int64_t n = inputs.size(), actions = canvas_ * canvas_, stride = INPUT_PLANES * actions;
    if (n < 1 || n > host_.size(0)) throw std::runtime_error("Invalid native batch size");
    for (int64_t i = 0; i < n; ++i) {
        if (inputs[i].size() != static_cast<size_t>(stride)) throw std::runtime_error("Native input shape mismatch");
        std::memcpy(host_.data_ptr<float>() + i * stride, inputs[i].data(), stride * sizeof(float));
    }
    auto input = host_.narrow(0, 0, n).to(device_, true);
    auto out = model_.forward({input}).toTuple();
    if (out->elements().size() != 2) throw std::runtime_error("Expected policy and value outputs");
    auto logits = out->elements()[0].toTensor(), values = out->elements()[1].toTensor();
    if (logits.dim() != 2 || logits.size(0) != n || logits.size(1) != actions || values.dim() != 1 || values.size(0) != n)
        throw std::runtime_error("Native model output shape mismatch");
    auto cpu = torch::cat({logits, values.unsqueeze(1)}, 1).to(torch::kCPU).to(torch::kFloat64).contiguous();
    std::vector<Evaluation> result(n);
    for (int64_t i = 0; i < n; ++i) {
        auto row = cpu.data_ptr<double>() + i * (actions+1);
        result[i].logits.assign(row, row+actions); result[i].value = row[actions];
    }
    return result;
}
}
