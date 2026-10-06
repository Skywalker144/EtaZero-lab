#include "etazero/torch_backend.h"
#include "etazero/schema.h"
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/autocast_mode.h>
#include <cstring>
namespace etazero {
TorchBackend::TorchBackend(const std::string& path, const std::string& device, int canvas, int max_batch, const std::string& precision,
                           std::shared_ptr<LoadedModel> shared_model)
    : device_(device), path_(path), precision_(precision), shared_model_(shared_model?std::move(shared_model):std::make_shared<LoadedModel>()),
      canvas_(canvas), max_batch_(max_batch) {
    if(precision_=="auto")precision_=device_.is_cuda()?"float16":"float32";
    if (precision_!="float32" && precision_!="float16") throw std::runtime_error("Invalid inference precision");
    if (precision_=="float16" && !device_.is_cuda()) throw std::runtime_error("FP16 inference requires CUDA");
}
void TorchBackend::initialize() {
    // Like KataGo's compute handle, each server owns its device binding, stream and buffers.
    c10::cuda::OptionalCUDAGuard device_guard;
    if (device_.is_cuda()) {
        device_guard.set_device(device_);
        stream_=std::make_unique<c10::cuda::CUDAStream>(c10::cuda::getStreamFromPool(false,device_.index()));
    }
    {
        std::lock_guard<std::mutex> lock(shared_model_->mutex);
        if (!shared_model_->ready) {
            shared_model_->module=torch::jit::load(path_,device_);
            shared_model_->module.eval();shared_model_->ready=true;
        }
        model_=shared_model_->module; // Shared immutable weights, independent server streams/buffers.
    }
    auto meta = model_.get_method("metadata")({}).toTuple();
    if (meta->elements().size() != 2 || meta->elements()[0].toInt() != canvas_ || meta->elements()[1].toStringRef() != CONTRACT_ID)
        throw std::runtime_error("Model contract/canvas mismatch");
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(device_.is_cuda());
    host_ = torch::empty({max_batch_, INPUT_PLANES, canvas_, canvas_}, options);
    global_host_ = torch::empty({max_batch_, GLOBAL_FEATURES}, options);
}
std::vector<Evaluation> TorchBackend::evaluate(const InferenceInputs& inputs) {
    c10::InferenceMode guard;
    c10::cuda::OptionalCUDAStreamGuard stream_guard;
    if (stream_) stream_guard.reset_stream(*stream_);
    struct AutocastGuard {
        bool enabled=at::autocast::is_autocast_enabled(at::kCUDA);
        at::ScalarType dtype=at::autocast::get_autocast_dtype(at::kCUDA);
        explicit AutocastGuard(bool use) {
            at::autocast::set_autocast_dtype(at::kCUDA,at::kHalf);
            at::autocast::set_autocast_enabled(at::kCUDA,use);
        }
        ~AutocastGuard() {
            at::autocast::clear_cache();at::autocast::set_autocast_enabled(at::kCUDA,enabled);
            at::autocast::set_autocast_dtype(at::kCUDA,dtype);
        }
    } autocast(precision_=="float16");
    const int64_t n = inputs.size(), actions = canvas_ * canvas_, stride = INPUT_PLANES * actions;
    if (n < 1 || n > host_.size(0)) throw std::runtime_error("Invalid native batch size");
    for (int64_t i = 0; i < n; ++i) {
        if (inputs[i]->size() != static_cast<size_t>(stride + GLOBAL_FEATURES)) throw std::runtime_error("Native input shape mismatch");
        std::memcpy(host_.data_ptr<float>() + i * stride, inputs[i]->data(), stride * sizeof(float));
        std::memcpy(global_host_.data_ptr<float>() + i * GLOBAL_FEATURES, inputs[i]->data()+stride, GLOBAL_FEATURES*sizeof(float));
    }
    auto input = host_.narrow(0, 0, n).to(device_, true);
    auto out = model_.forward({input, global_host_.narrow(0, 0, n).to(device_, true)}).toTuple();
    if (out->elements().size() != 4) throw std::runtime_error("Expected ordinary/WDL/optimistic/error outputs");
    auto logits = out->elements()[0].toTensor(), values = out->elements()[1].toTensor();
    auto optimistic=out->elements()[2].toTensor(), error=out->elements()[3].toTensor();
    if(optimistic.sizes()!=logits.sizes() || error.dim()!=1 || error.size(0)!=n)
        throw std::runtime_error("Native auxiliary output shape mismatch");
    if (precision_=="float16" && (logits.scalar_type()!=torch::kFloat16 || values.scalar_type()!=torch::kFloat16))
        throw std::runtime_error("FP16 inference did not execute half-precision output heads");
    if (logits.dim() != 2 || logits.size(0) != n || logits.size(1) != actions || values.dim() != 2 || values.size(0) != n || values.size(1) != 3)
        throw std::runtime_error("Native model output shape mismatch");
    auto cpu = torch::cat({logits.to(torch::kFloat32), torch::softmax(values.to(torch::kFloat32),1), optimistic.to(torch::kFloat32), error.to(torch::kFloat32).unsqueeze(1)}, 1).to(torch::kCPU).to(torch::kFloat32).contiguous();
    std::vector<Evaluation> result(n);
    for (int64_t i = 0; i < n; ++i) {
        auto row = cpu.data_ptr<float>() + i * (2*actions+4);
        result[i].logits.assign(row, row+actions); result[i].wdl = {row[actions],row[actions+1],row[actions+2]};
        result[i].optimistic_logits.assign(row+actions+3,row+2*actions+3);
        result[i].shortterm_value_stdev=row[2*actions+3];result[i].has_auxiliary=true;
    }
    return result;
}
}
