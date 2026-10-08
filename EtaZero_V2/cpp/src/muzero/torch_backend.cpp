#include "etazero/muzero/torch_backend.h"
#include "etazero/schema.h"
#include <ATen/autocast_mode.h>
#include <ATen/ops/_foreach_copy.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace etazero::muzero {
namespace {
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
struct TorchBackend::Graph {
    at::cuda::CUDAGraph graph;
    torch::Tensor input0,input1;
    std::vector<torch::Tensor> rows;
    c10::IValue output;
};
TorchBackend::~TorchBackend()=default;
TorchBackend::TorchBackend(const std::string& path,const std::string& device,int canvas,
                           int max_batch,const std::string& precision,std::shared_ptr<PreparedModel> shared_model)
    : device_(device),shared_model_(shared_model?std::move(shared_model):std::make_shared<PreparedModel>()),
      precision_(precision),canvas_(canvas),max_batch_(max_batch) {
    if(canvas<5 || canvas>25 || max_batch<1)throw std::runtime_error("Invalid MuZero backend dimensions");
    if(precision!="float32" && precision!="float16")throw std::runtime_error("Invalid MuZero inference precision");
    if(precision=="float16" && !device_.is_cuda())throw std::runtime_error("MuZero FP16 inference requires CUDA");
    c10::cuda::OptionalCUDAGuard device_guard;
    if(device_.is_cuda())device_guard.set_device(device_);
    model_=shared_model_->prepare(path,device_,precision_);
    auto meta=model_.get_method("metadata")({}).toTuple();
    const auto& fields=meta->elements();
    if(fields.size()!=4 || fields[0].toInt()!=canvas || fields[1].toStringRef()!=CONTRACT_ID ||
       fields[2].toStringRef()!="muzero" || fields[3].toInt()<1)
        throw std::runtime_error("MuZero model algorithm/contract/canvas mismatch");
    latent_channels_=fields[3].toInt();
    if(device_.is_cuda()) {
        // Model loading completes on the constructor's stream before services
        // may use the immutable weights on their independent streams.
        c10::cuda::getCurrentCUDAStream(device_.index()).synchronize();
        stream_=std::make_unique<c10::cuda::CUDAStream>(c10::cuda::getStreamFromPool(false,device_.index()));
    }
    auto host=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(device_.is_cuda());
    observation_host_=torch::empty({max_batch_,INPUT_PLANES,canvas_,canvas_},host);
    globals_host_=torch::empty({max_batch_,GLOBAL_FEATURES},host);
    actions_host_=torch::empty({max_batch_},host.dtype(torch::kInt64));
    if(stream_) {
        auto started=std::chrono::steady_clock::now();capture_graphs();
        shared_model_->timing.graph_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    }
}
void TorchBackend::capture_graphs() {
    // Constructors run before batch-service threads start. Capture is therefore
    // isolated from other CUDA work in this worker, even with several services.
    c10::InferenceMode inference_guard;
    c10::cuda::CUDAStreamGuard stream_guard(*stream_);
    AutocastGuard autocast(precision_=="float16");
    for(int batch=1;;batch=std::min(max_batch_,batch*2)) {
        for(bool initial:{true,false}) {
            auto captured=std::make_unique<Graph>();
            captured->input0=torch::zeros({batch,initial?INPUT_PLANES:latent_channels_+1,canvas_,canvas_},
                torch::TensorOptions().device(device_).dtype(torch::kFloat32));
            captured->input0.select(1,initial?0:latent_channels_).fill_(1);
            for(int i=0;i<batch;++i)captured->rows.push_back(captured->input0.narrow(0,i,1));
            captured->input1=initial?torch::zeros({batch,GLOBAL_FEATURES},captured->input0.options()):
                torch::zeros({batch},captured->input0.options().dtype(torch::kInt64));
            auto method=model_.get_method(initial?"initial":"recurrent");
            for(int warmup=0;warmup<3;++warmup)method({captured->input0,captured->input1});
            stream_->synchronize();
            captured->graph.capture_begin();
            captured->output=method({captured->input0,captured->input1});
            captured->graph.capture_end();
            (initial?initial_graphs_:recurrent_graphs_).push_back(std::move(captured));
        }
        if(batch==max_batch_)break;
    }
}
TorchBackend::Graph& TorchBackend::graph_for(const std::vector<std::unique_ptr<Graph>>& graphs,size_t n) {
    auto which=std::lower_bound(graphs.begin(),graphs.end(),n,
        [](const auto& graph,int64_t size){return graph->input0.size(0)<size;});
    if(which==graphs.end())throw std::runtime_error("MuZero graph batch exceeds backend capacity");
    return **which;
}
c10::IValue TorchBackend::graph_outputs(Graph& captured,size_t n) {
    captured.graph.replay();
    std::vector<c10::IValue> result;
    for(const auto& field:captured.output.toTuple()->elements())result.emplace_back(field.toTensor().narrow(0,0,n));
    return c10::ivalue::Tuple::create(std::move(result));
}
c10::IValue TorchBackend::replay(const std::vector<std::unique_ptr<Graph>>& graphs,
                               const torch::Tensor& input0,const torch::Tensor& input1) {
    const int64_t n=input0.size(0);
    auto& captured=graph_for(graphs,n);const int64_t batch=captured.input0.size(0);
    captured.input0.narrow(0,0,n).copy_(input0,true);
    captured.input1.narrow(0,0,n).copy_(input1,true);
    if(n<batch) {
        // Exported NBT/ResNet inference has per-sample normalization. Repeating
        // a valid row fills the bucket without empty-mask normalization or any
        // extra search nodes/targets; only the real n outputs are consumed.
        auto shape0=captured.input0.sizes().vec(),shape1=captured.input1.sizes().vec();
        shape0[0]=shape1[0]=batch-n;
        captured.input0.narrow(0,n,batch-n).copy_(captured.input0.narrow(0,0,1).expand(shape0));
        captured.input1.narrow(0,n,batch-n).copy_(captured.input1.narrow(0,0,1).expand(shape1));
    }
    return graph_outputs(captured,n);
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
    c10::cuda::OptionalCUDAStreamGuard stream_guard;
    if(stream_)stream_guard.reset_stream(*stream_);
    AutocastGuard autocast(precision_=="float16");
    const int64_t n=inputs.size(),area=canvas_*canvas_,spatial=INPUT_PLANES*area;
    auto obs=observation_host_.narrow(0,0,n);
    auto globals=globals_host_.narrow(0,0,n);
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
    auto result=stream_?replay(initial_graphs_,obs,globals):model_.get_method("initial")({obs,globals});
    return unpack(result,masks);
}
std::vector<TorchBackend::Output> TorchBackend::recurrent(const std::vector<Action>& inputs) {
    check_batch(inputs.size());c10::InferenceMode inference_guard;
    c10::cuda::OptionalCUDAStreamGuard stream_guard;
    if(stream_)stream_guard.reset_stream(*stream_);
    AutocastGuard autocast(precision_=="float16");
    std::vector<torch::Tensor> latents;
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
    auto actions=actions_host_.narrow(0,0,inputs.size());
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
    c10::IValue result;
    if(stream_) {
        auto& captured=graph_for(recurrent_graphs_,inputs.size());
        const size_t batch=captured.rows.size();
        // Gather directly into stable graph inputs, without an intermediate
        // concatenated latent allocation and a second device-to-device copy.
        while(latents.size()<batch)latents.push_back(latents.front());
        at::_foreach_copy_(captured.rows,latents);
        for(size_t i=inputs.size();i<batch;++i)actions_host_.data_ptr<int64_t>()[i]=inputs.front().action;
        captured.input1.copy_(actions_host_.narrow(0,0,batch),true);
        result=graph_outputs(captured,inputs.size());
    } else result=model_.get_method("recurrent")({torch::cat(latents,0),actions});
    return unpack(result,masks);
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
    std::vector<Output> outputs(n);
    std::vector<torch::Tensor> sources,destinations;
    sources.reserve(n);destinations.reserve(n);
    for(int64_t i=0;i<n;++i) {
        auto state=std::make_shared<Latent>();
        // Independent node allocations release dead nodes without retaining
        // an entire inference batch; foreach copies them in grouped kernels.
        sources.push_back(hidden.narrow(0,i,1));
        state->tensor_=torch::empty_like(sources.back());
        destinations.push_back(state->tensor_);state->owner_=owner_;state->mask_=masks[i];
        outputs[i].latent=std::move(state);
    }
    at::_foreach_copy_(destinations,sources);
    // One blocking prediction return completes the preceding node copies on
    // this backend's stream. Numerical/mask parity is checked by development
    // tests, without extra GPU reductions or mask transfers in selfplay.
    auto cpu=torch::cat({policy.to(torch::kFloat32),torch::softmax(value.to(torch::kFloat32),1),
                        optimistic.to(torch::kFloat32),error.to(torch::kFloat32).unsqueeze(1)},1).to(torch::kCPU).contiguous();
    const int64_t stride=2*area+4;
    for(int64_t i=0;i<n;++i) {
        auto row=cpu.data_ptr<float>()+i*stride;auto& e=outputs[i].evaluation;
        e.logits.assign(row,row+area);e.wdl={row[area],row[area+1],row[area+2]};
        e.optimistic_logits.assign(row+area+3,row+2*area+3);e.shortterm_value_stdev=row[2*area+3];e.has_auxiliary=true;
    }
    return outputs;
}
}
