#include "neuralnet/prepared_model.h"
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <chrono>
#include <regex>
#include <sstream>

namespace etazero {
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
std::string layout(const torch::jit::Module& model) {
    // Parameters alone cannot identify an executable: activation, metadata and
    // non-tensor attributes may change with identical tensor names and shapes.
    std::ostringstream out;out.precision(17);
    for(const auto& named:model.named_modules()) {
        out << named.name << '\n';
        const auto& module=named.value;
        for(const auto& method:module.get_methods())
            out << method.name() << '\n' << *method.graph() << '\n';
        auto type=module.type();
        for(size_t i=0;i<type->numAttributes();++i) {
            const auto& name=type->getAttributeName(i);
            auto value=module.attr(name);
            out << name << ':' << type->getAttribute(i)->str() << ':';
            if(value.isTensor()) {
                auto tensor=value.toTensor();
                out << tensor.sizes() << ':' << tensor.strides() << ':' << tensor.scalar_type();
            } else if(!value.isObject()) out << value;
            out << '\n';
        }
    }
    // Import assigns different class suffixes to structurally identical files.
    static const std::regex mangled("\\.___torch_mangle_[0-9]+");
    return std::regex_replace(out.str(),mangled,"");
}
void copy_tensors(torch::jit::Module& target,const torch::jit::Module& source) {
    auto modules=source.named_modules();auto incoming=modules.begin();
    for(const auto& named:target.named_modules()) {
        auto module=named.value;
        auto source_module=(*incoming).value;
        auto type=module.type();
        for(size_t i=0;i<type->numAttributes();++i) {
            const auto& name=type->getAttributeName(i);
            auto value=module.attr(name);
            if(value.isTensor())value.toTensor().copy_(source_module.attr(name).toTensor());
        }
        ++incoming;
    }
}
}
PreparedModel::PreparedModel(torch::jit::Module module,bool preload_layout):supplied_(std::move(module)) {
    if(preload_layout)supplied_layout_=layout(*supplied_);
}
torch::jit::Module PreparedModel::prepare(const std::string& path,const torch::Device& device,
                                        const std::string& precision) {
    if(ready_)return *module_;
    c10::InferenceMode inference;
    c10::cuda::OptionalCUDAGuard guard;
    if(device.is_cuda())guard.set_device(device);
    timing={};auto started=Clock::now();
    auto incoming=supplied_?std::move(*supplied_):torch::jit::load(path,torch::kCPU);
    supplied_.reset();incoming.eval();
    auto signature=supplied_layout_?std::move(*supplied_layout_):layout(incoming);
    supplied_layout_.reset();
    timing.load_seconds=elapsed(started);started=Clock::now();
    timing.reused_runtime=module_.has_value() && signature==layout_ && precision==precision_;
    if(timing.reused_runtime)copy_tensors(*module_,incoming);
    else {
        module_=std::move(incoming);layout_=std::move(signature);precision_=precision;
        if(precision=="float16") {
            // Match CUDA autocast's Conv/Linear rounding once. Keep normalization
            // and other buffers in their original precision.
            for(const auto& named:module_->named_modules()) {
                auto type=named.value.type()->name();
                if(type && (type->name()=="Conv2d" || type->name()=="Linear"))
                    for(auto parameter:named.value.parameters(false))
                        parameter.set_data(parameter.to(torch::kFloat16));
            }
        }
    }
    module_->to(device);
    if(device.is_cuda())c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
    timing.weights_seconds=elapsed(started);ready_=true;
    return *module_;
}
std::shared_ptr<PreparedModel> PreparedModel::clone_runtime() const {
    if(!module_)throw std::runtime_error("Cannot clone an unprepared model runtime");
    c10::InferenceMode inference;
    auto copy=std::make_shared<PreparedModel>();
    copy->module_=module_->deepcopy(torch::kCPU);
    copy->layout_=layout_;copy->precision_=precision_;
    return copy;
}
bool PreparedModel::reuse_runtime(const PreparedModel& source,const std::string& precision) {
    if(ready_ || !supplied_layout_ || !source.module_ || *supplied_layout_!=source.layout_ ||
       precision!=source.precision_)return false;
    c10::InferenceMode inference;
    module_=source.module_->deepcopy(torch::kCPU);
    layout_=source.layout_;precision_=source.precision_;
    return true;
}
void PreparedModel::offload() {
    if(module_) {
        c10::InferenceMode inference;
        module_->to(torch::kCPU);
    }
    ready_=false;
}
}
