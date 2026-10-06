#include "etazero/muzero/torch_backend.h"
#include "etazero/schema.h"
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <thread>

namespace etazero::muzero {
namespace {
[[noreturn]] void latent_failure(const std::exception& error,const std::string& stage,
    const std::string& precision,torch::jit::Module& model,const c10::IValue& result,
    const torch::Tensor& input0,const torch::Tensor& input1) {
    std::string message=std::string(error.what())+" (stage="+stage+", precision="+precision+
        ", batch="+std::to_string(input0.size(0));
    const char* directory=std::getenv("ETAZERO_NAN_DIAGNOSTIC_DIR");
    if(directory && *directory) {
        try {
            std::filesystem::create_directories(directory);
            std::ostringstream name;
            name<<"latent_"<<stage<<"_"<<std::chrono::steady_clock::now().time_since_epoch().count()
                <<"_"<<std::this_thread::get_id()<<".pt";
            auto path=std::filesystem::path(directory)/name.str();
            torch::jit::Module dump("MuZeroLatentDiagnostic");
            dump.register_attribute("stage",c10::StringType::get(),stage);
            dump.register_attribute("precision",c10::StringType::get(),precision);
            dump.register_buffer("input0",input0.to(torch::kCPU));
            dump.register_buffer("input1",input1.to(torch::kCPU));
            dump.register_buffer("output_hidden",result.toTuple()->elements()[0].toTensor().to(torch::kCPU));
            // Preserve the actual loaded model together with the failing batch.
            dump.register_module("model",model);
            dump.save(path.string());
            message+=", diagnostic="+path.string();
        } catch(const std::exception& capture_error) {
            message+=", diagnostic_save_failed="+std::string(capture_error.what());
        }
    }
    throw std::runtime_error(message+")");
}
struct AutocastGuard {
    bool enabled=at::autocast::is_autocast_enabled(at::kCUDA);
    at::ScalarType dtype=at::autocast::get_autocast_dtype(at::kCUDA);
    explicit AutocastGuard(bool use) {
        at::autocast::set_autocast_dtype(at::kCUDA,at::kHalf);
        at::autocast::set_autocast_enabled(at::kCUDA,use);
    }
    ~AutocastGuard() {
        at::autocast::clear_cache();
        at::autocast::set_autocast_enabled(at::kCUDA,enabled);
        at::autocast::set_autocast_dtype(at::kCUDA,dtype);
    }
};
}
TorchBackend::TorchBackend(const std::string& path,const std::string& device,int canvas,
                           int max_batch,const std::string& precision,std::optional<torch::jit::Module> loaded_model)
    : device_(device),precision_(precision),canvas_(canvas),max_batch_(max_batch) {
    if(canvas<5 || canvas>25 || max_batch<1)throw std::runtime_error("Invalid MuZero backend dimensions");
    if(precision!="float32" && precision!="float16")throw std::runtime_error("Invalid MuZero inference precision");
    if(precision=="float16" && !device_.is_cuda())throw std::runtime_error("MuZero FP16 inference requires CUDA");
    c10::cuda::OptionalCUDAGuard device_guard;
    if(device_.is_cuda())device_guard.set_device(device_);
    model_=loaded_model?std::move(*loaded_model):torch::jit::load(path,device_);model_.eval();
    auto meta=model_.get_method("metadata")({}).toTuple();
    const auto& fields=meta->elements();
    if(fields.size()!=4 || fields[0].toInt()!=canvas || fields[1].toStringRef()!=CONTRACT_ID ||
       fields[2].toStringRef()!="muzero" || fields[3].toInt()<1)
        throw std::runtime_error("MuZero model algorithm/contract/canvas mismatch");
    latent_channels_=fields[3].toInt();
}
void TorchBackend::check_batch(size_t n) const {
    if(!n || n>static_cast<size_t>(max_batch_))throw std::runtime_error("Invalid MuZero batch size");
}
std::vector<float> TorchBackend::Latent::copy_to_cpu() const {
    auto cpu=tensor_.to(torch::kCPU).contiguous();
    return {cpu.data_ptr<float>(),cpu.data_ptr<float>()+cpu.numel()};
}
std::vector<TorchBackend::Output> TorchBackend::initial(const InferenceInputs& inputs) {
    check_batch(inputs.size());c10::InferenceMode inference_guard;
    c10::cuda::OptionalCUDAGuard device_guard;
    if(device_.is_cuda())device_guard.set_device(device_);
    AutocastGuard autocast(precision_=="float16");
    const int64_t n=inputs.size(),area=canvas_*canvas_,spatial=INPUT_PLANES*area;
    auto obs=torch::empty({n,INPUT_PLANES,canvas_,canvas_},torch::kFloat32);
    auto globals=torch::empty({n,GLOBAL_FEATURES},torch::kFloat32);
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
    for(int64_t i=0;i<n;++i) {
        if(!inputs[i] || inputs[i]->size()!=static_cast<size_t>(spatial+GLOBAL_FEATURES))
            throw std::runtime_error("MuZero observation shape mismatch");
        const auto& input=*inputs[i];
        auto mask=std::make_shared<std::vector<uint8_t>>(area);int count=0;
        for(int64_t j=0;j<spatial+GLOBAL_FEATURES;++j) {
            if(!std::isfinite(input[j]) || (j<spatial && input[j]!=0 && input[j]!=1))
                throw std::runtime_error("MuZero requires binary spatial and finite global features");
            if(j<area){(*mask)[j]=input[j];count+=(*mask)[j];}
        }
        if(!count)throw std::runtime_error("MuZero requires a nonempty board mask");
        masks.push_back(mask);
        std::memcpy(obs.data_ptr<float>()+i*spatial,input.data(),spatial*sizeof(float));
        std::memcpy(globals.data_ptr<float>()+i*GLOBAL_FEATURES,input.data()+spatial,GLOBAL_FEATURES*sizeof(float));
    }
    auto device_obs=obs.to(device_),device_globals=globals.to(device_);
    auto result=model_.get_method("initial")({device_obs,device_globals});
    try {return unpack(result,masks);}
    catch(const std::exception& error) {
        if(std::string(error.what())!="Nonfinite MuZero latent")throw;
        latent_failure(error,"initial",precision_,model_,result,device_obs,device_globals);
    }
}
std::vector<TorchBackend::Output> TorchBackend::recurrent(const std::vector<Action>& inputs) {
    check_batch(inputs.size());c10::InferenceMode inference_guard;
    c10::cuda::OptionalCUDAGuard device_guard;
    if(device_.is_cuda())device_guard.set_device(device_);
    AutocastGuard autocast(precision_=="float16");
    std::vector<torch::Tensor> latents;
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
    auto actions=torch::empty({static_cast<int64_t>(inputs.size())},torch::kInt64);
    for(size_t i=0;i<inputs.size();++i) {
        const auto& input=inputs[i];
        auto latent=std::dynamic_pointer_cast<const Latent>(input.latent);
        if(!latent || latent->owner_!=owner_)
            throw std::runtime_error("MuZero latent belongs to another backend/model/device");
        if(input.action<0 || input.action>=canvas_*canvas_ || !(*latent->mask_)[input.action])
            throw std::runtime_error("MuZero recurrent action outside board mask");
        latents.push_back(latent->tensor_);masks.push_back(latent->mask_);
        actions.data_ptr<int64_t>()[i]=input.action;
    }
    auto hidden=torch::cat(latents,0),device_actions=actions.to(device_);
    auto result=model_.get_method("recurrent")({hidden,device_actions});
    try {return unpack(result,masks);}
    catch(const std::exception& error) {
        if(std::string(error.what())!="Nonfinite MuZero latent")throw;
        latent_failure(error,"recurrent",precision_,model_,result,hidden,device_actions);
    }
}
std::vector<TorchBackend::Output> TorchBackend::unpack(const c10::IValue& result,
    const std::vector<std::shared_ptr<const std::vector<uint8_t>>>& masks) {
    auto tuple=result.toTuple();const auto& fields=tuple->elements();
    if(fields.size()!=5)throw std::runtime_error("MuZero requires latent/policy/WDL/optimistic/error outputs");
    auto hidden=fields[0].toTensor(),policy=fields[1].toTensor(),value=fields[2].toTensor();
    auto optimistic=fields[3].toTensor(),error=fields[4].toTensor();
    const int64_t n=masks.size(),area=canvas_*canvas_;
    if(hidden.sizes()!=torch::IntArrayRef({n,latent_channels_+1,canvas_,canvas_}) ||
       hidden.scalar_type()!=torch::kFloat32 || hidden.device()!=policy.device() ||
       policy.sizes()!=torch::IntArrayRef({n,area}) || optimistic.sizes()!=policy.sizes() ||
       value.sizes()!=torch::IntArrayRef({n,3}) || error.sizes()!=torch::IntArrayRef({n}))
        throw std::runtime_error("MuZero output shape/dtype mismatch");
    if(precision_=="float16" && (policy.scalar_type()!=torch::kFloat16 || value.scalar_type()!=torch::kFloat16))
        throw std::runtime_error("MuZero FP16 heads did not execute in half precision");
    if(!torch::isfinite(hidden).all().item<bool>())throw std::runtime_error("Nonfinite MuZero latent");
    auto actual_mask=hidden.select(1,latent_channels_).to(torch::kCPU).contiguous();
    auto cpu=torch::cat({policy.to(torch::kFloat32),torch::softmax(value.to(torch::kFloat32),1),
                        optimistic.to(torch::kFloat32),error.to(torch::kFloat32).unsqueeze(1)},1).to(torch::kCPU).contiguous();
    if(!torch::isfinite(cpu).all().item<bool>())throw std::runtime_error("Nonfinite MuZero prediction");
    std::vector<Output> outputs(n);
    for(int64_t i=0;i<n;++i) {
        for(int64_t a=0;a<area;++a)
            if(actual_mask.data_ptr<float>()[i*area+a]!=(*masks[i])[a])
                throw std::runtime_error("MuZero changed the immutable board mask");
        auto state=std::make_shared<Latent>();
        // Clone releases the whole batch allocation when only one node survives.
        state->tensor_=hidden.narrow(0,i,1).clone();state->owner_=owner_;state->mask_=masks[i];
        outputs[i].latent=std::move(state);
        auto row=cpu.data_ptr<float>()+i*(2*area+4);auto& e=outputs[i].evaluation;
        e.logits.assign(row,row+area);e.wdl={row[area],row[area+1],row[area+2]};
        e.optimistic_logits.assign(row+area+3,row+2*area+3);e.shortterm_value_stdev=row[2*area+3];e.has_auxiliary=true;
        if(e.shortterm_value_stdev<0)throw std::runtime_error("Negative MuZero error stdev");
    }
    // Cloned tensors are ready before a caller may hand them to another thread.
    if(device_.is_cuda())c10::cuda::getCurrentCUDAStream(device_.index()).synchronize();
    return outputs;
}
}
