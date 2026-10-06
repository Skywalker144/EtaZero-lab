#include "etazero/torch_backend.h"
#include "etazero/schema.h"
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <algorithm>
#include <cstring>
namespace etazero {
namespace {
struct AutocastGuard {
    bool enabled=at::autocast::is_autocast_enabled(at::kCUDA);
    bool cache_enabled=at::autocast::is_autocast_cache_enabled();
    at::ScalarType dtype=at::autocast::get_autocast_dtype(at::kCUDA);
    explicit AutocastGuard(bool use) {
        at::autocast::set_autocast_dtype(at::kCUDA,at::kHalf);
        at::autocast::set_autocast_enabled(at::kCUDA,use);
        // Preloaded evaluation models may still have FP32 weights. Their
        // casts must belong to the captured graph, not a temporary warmup cache.
        at::autocast::set_autocast_cache_enabled(false);
    }
    ~AutocastGuard() {
        at::autocast::clear_cache();
        at::autocast::set_autocast_enabled(at::kCUDA,enabled);
        at::autocast::set_autocast_dtype(at::kCUDA,dtype);
        at::autocast::set_autocast_cache_enabled(cache_enabled);
    }
};
}
struct TorchBackend::Graph {
    at::cuda::CUDAGraph graph;
    torch::Tensor input, globals, output;
};
TorchBackend::~TorchBackend()=default;
TorchBackend::TorchBackend(const std::string& path, const std::string& device, int canvas, int max_batch, const std::string& precision,
                           std::shared_ptr<LoadedModel> shared_model)
    : device_(device), path_(path), precision_(precision), shared_model_(shared_model?std::move(shared_model):std::make_shared<LoadedModel>()),
      canvas_(canvas), max_batch_(max_batch) {
    if(canvas<5 || canvas>25 || max_batch<1)throw std::runtime_error("Invalid inference backend dimensions");
    if(precision_=="auto")precision_=device_.is_cuda()?"float16":"float32";
    if (precision_!="float32" && precision_!="float16") throw std::runtime_error("Invalid inference precision");
    if (!precision_checked_ && precision_=="float16" && !device_.is_cuda()) throw std::runtime_error("FP16 inference requires CUDA");
}
void TorchBackend::initialize() {
    // Preparation is serial; each service still owns its stream and buffers.
    c10::InferenceMode inference_guard;
    c10::cuda::OptionalCUDAGuard device_guard;
    if (device_.is_cuda()) {
        device_guard.set_device(device_);
        stream_=std::make_unique<c10::cuda::CUDAStream>(c10::cuda::getStreamFromPool(false,device_.index()));
    }
    model_=shared_model_->prepare(path_,device_,precision_);
    auto meta = model_.get_method("metadata")({}).toTuple();
    if (meta->elements().size() != 2 || meta->elements()[0].toInt() != canvas_ || meta->elements()[1].toStringRef() != CONTRACT_ID)
        throw std::runtime_error("Model contract/canvas mismatch");
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(device_.is_cuda());
    host_ = torch::empty({max_batch_, INPUT_PLANES, canvas_, canvas_}, options);
    global_host_ = torch::empty({max_batch_, GLOBAL_FEATURES}, options);
    if(stream_) {
        auto started=std::chrono::steady_clock::now();capture_graphs();
        shared_model_->timing.graph_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    }
    else {
        input_ = torch::empty(host_.sizes(), options.pinned_memory(false));
        global_input_ = torch::empty(global_host_.sizes(), options.pinned_memory(false));
    }
}
torch::Tensor TorchBackend::forward(const torch::Tensor& input,const torch::Tensor& globals) {
    const int64_t n=input.size(0),actions=canvas_*canvas_;
    auto out = model_.forward({input,globals}).toTuple();
    if (out->elements().size() != 4) throw std::runtime_error("Expected ordinary/WDL/optimistic/error outputs");
    auto logits = out->elements()[0].toTensor(), values = out->elements()[1].toTensor();
    auto optimistic=out->elements()[2].toTensor(), error=out->elements()[3].toTensor();
    if(optimistic.sizes()!=logits.sizes() || error.dim()!=1 || error.size(0)!=n)
        throw std::runtime_error("Native auxiliary output shape mismatch");
    if (!precision_checked_ && precision_=="float16" && (logits.scalar_type()!=torch::kFloat16 || values.scalar_type()!=torch::kFloat16))
        throw std::runtime_error("FP16 inference did not execute half-precision output heads");
    precision_checked_=true;
    if (logits.dim() != 2 || logits.size(0) != n || logits.size(1) != actions || values.dim() != 2 || values.size(0) != n || values.size(1) != 3)
        throw std::runtime_error("Native model output shape mismatch");
    return torch::cat({logits.to(torch::kFloat32), torch::softmax(values.to(torch::kFloat32),1), optimistic.to(torch::kFloat32), error.to(torch::kFloat32).unsqueeze(1)}, 1);
}
void TorchBackend::capture_graphs() {
    c10::cuda::CUDAStreamGuard stream_guard(*stream_);
    AutocastGuard autocast(precision_=="float16");
    for(int batch=1;;batch=std::min(max_batch_,batch*2)) {
        auto captured=std::make_unique<Graph>();
        auto options=torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        captured->input=torch::zeros({batch,INPUT_PLANES,canvas_,canvas_},options);
        captured->input.select(1,0).fill_(1); // A valid mask for normalization warmup.
        captured->globals=torch::zeros({batch,GLOBAL_FEATURES},options);
        for(int warmup=0;warmup<3;++warmup)forward(captured->input,captured->globals);
        stream_->synchronize();
        captured->graph.capture_begin();
        captured->output=forward(captured->input,captured->globals);
        captured->graph.capture_end();
        graphs_.push_back(std::move(captured));
        if(batch==max_batch_)break;
    }
    stream_->synchronize();
}
std::vector<Evaluation> TorchBackend::evaluate(const InferenceInputs& inputs) {
    c10::InferenceMode guard;
    c10::cuda::OptionalCUDAStreamGuard stream_guard;
    if(stream_)stream_guard.reset_stream(*stream_);
    const int64_t n=inputs.size(),actions=canvas_*canvas_,stride=INPUT_PLANES*actions;
    if(n<1 || n>host_.size(0))throw std::runtime_error("Invalid native batch size");
    for(int64_t i=0;i<n;++i) {
        if(inputs[i]->size()!=static_cast<size_t>(stride+GLOBAL_FEATURES))throw std::runtime_error("Native input shape mismatch");
        std::memcpy(host_.data_ptr<float>()+i*stride,inputs[i]->data(),stride*sizeof(float));
        std::memcpy(global_host_.data_ptr<float>()+i*GLOBAL_FEATURES,inputs[i]->data()+stride,GLOBAL_FEATURES*sizeof(float));
    }
    torch::Tensor output;
    if(stream_) {
        auto which=std::lower_bound(graphs_.begin(),graphs_.end(),n,
            [](const auto& graph,int64_t size){return graph->input.size(0)<size;});
        if(which==graphs_.end())throw std::runtime_error("Graph batch exceeds backend capacity");
        auto& captured=**which;const int64_t batch=captured.input.size(0);
        // Exported eval networks normalize each row independently. Repeating
        // a valid row fills the bucket; padding never becomes a search result.
        for(int64_t i=n;i<batch;++i) {
            std::memcpy(host_.data_ptr<float>()+i*stride,host_.data_ptr<float>(),stride*sizeof(float));
            std::memcpy(global_host_.data_ptr<float>()+i*GLOBAL_FEATURES,global_host_.data_ptr<float>(),GLOBAL_FEATURES*sizeof(float));
        }
        captured.input.copy_(host_.narrow(0,0,batch),true);
        captured.globals.copy_(global_host_.narrow(0,0,batch),true);
        captured.graph.replay();
        output=captured.output.narrow(0,0,n);
    } else {
        auto input=input_.narrow(0,0,n),globals=global_input_.narrow(0,0,n);
        input.copy_(host_.narrow(0,0,n));globals.copy_(global_host_.narrow(0,0,n));
        output=forward(input,globals);
    }
    // Blocking D2H finishes this replay before its buffers can be reused.
    auto cpu=output.to(torch::kCPU).contiguous();
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
