#include "etazero/muzero/cuda_backend.h"
#include "etazero/cuda_backend.h"
#include "katago/neuralnet/nninterface.h"
#include "katago/neuralnet/muzerohelpers.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <stdexcept>
namespace etazero::muzero {
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Allocation {
    float* data=nullptr;int device;
    Allocation(size_t count,int d):device(d){check(cudaSetDevice(device));check(cudaMalloc(&data,count*sizeof(float)));}
    ~Allocation(){cudaSetDevice(device);cudaFree(data);}
};
struct State : Latent {
    std::shared_ptr<Allocation> allocation;size_t offset,count;bool allow_no_result;
    std::shared_ptr<const int> owner;std::shared_ptr<const std::vector<uint8_t>> mask;
    const float* data() const {return allocation->data+offset;}
    std::vector<float> copy_to_cpu() const override {
        check(cudaSetDevice(allocation->device));std::vector<float> result(count);
        check(cudaMemcpy(result.data(),data(),count*sizeof(float),cudaMemcpyDeviceToHost));return result;
    }
};
struct Engine {
    LoadedModel* model=nullptr;ComputeContext* context=nullptr;ComputeHandle* handle=nullptr;InputBuffers* host=nullptr;
    ConfigParser config;cudaEvent_t ready=nullptr;int canvas,batch,device;bool fp16;
    Engine(const std::string& path,int c,int b,int d,bool half):canvas(c),batch(b),device(d),fp16(half){model=NeuralNet::loadModelFile(path,"");}
    ~Engine(){
        cudaSetDevice(device);if(ready)cudaEventDestroy(ready);
        if(host)NeuralNet::freeInputBuffers(host);if(handle)NeuralNet::freeComputeHandle(handle);
        if(context)NeuralNet::freeComputeContext(context);if(model)NeuralNet::freeLoadedModel(model);
    }
    const ModelDesc& desc() const {return NeuralNet::getModelDesc(model);}
    void initialize(){
        context=NeuralNet::createComputeContext({device},nullptr,canvas,canvas,"",fp16?enabled_t::True:enabled_t::False,model,config);
        handle=NeuralNet::createComputeHandle(context,model,nullptr,batch,false,true,device,0);
        if(!desc().numLatentChannels)host=NeuralNet::createInputBuffers(model,batch,canvas,canvas);
        check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
    }
    cudaStream_t stream(){return static_cast<cudaStream_t>(NeuralNet::computeStream(handle));}
    float* spatial(){return NeuralNet::inputDeviceBuffer(handle);}
    float* globals(){return NeuralNet::globalDeviceBuffer(handle);}
    void wait(Engine& other){check(cudaStreamWaitEvent(stream(),other.ready,0));}
    void latent(int n,float* output){
        std::vector<float> p,pass,v,aux;
        NeuralNet::getDeviceOutput(handle,nullptr,n,output,p,pass,v,aux);
        check(cudaEventRecord(ready,stream()));
    }
};
}
struct CudaBackend::Impl {
    int canvas,batch,device,channels=0,action_planes;GameSpec spec;
    std::unique_ptr<Engine> representation,dynamics,prediction;
    const float** pointers=nullptr;int* actions_from=nullptr;int* actions_to=nullptr;
    std::shared_ptr<const int> owner=std::make_shared<const int>(0);
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
    std::vector<uint8_t> allow_no_result;
    std::vector<float> spatial_host,global_host;std::vector<const float*> pointer_host;std::vector<int> from_host,to_host;
    Impl(const std::string& path,int c,int b,int d,const std::string& precision,GameSpec s)
      :canvas(c),batch(b),device(d),action_planes(s.kind==GameKind::XIANGQI?2:1),spec(s){
        if(precision!="float16" && precision!="float32")throw std::runtime_error("Invalid CUDA inference precision");
        auto root=std::filesystem::path(path).parent_path();bool half=precision=="float16";
        representation=std::make_unique<Engine>((root/"representation.bin").string(),c,b,d,half);
        dynamics=std::make_unique<Engine>((root/"dynamics.bin").string(),c,b,d,half);
        prediction=std::make_unique<Engine>(path,c,b,d,half);
        channels=representation->desc().numLatentChannels;
        if(channels<=0 || dynamics->desc().numLatentChannels!=channels || prediction->desc().numLatentChannels ||
           representation->desc().numInputChannels!=s.planes || representation->desc().numInputGlobalChannels!=s.globals ||
           dynamics->desc().numInputChannels!=channels+1+action_planes || dynamics->desc().numInputGlobalChannels!=1 ||
           prediction->desc().numInputChannels!=channels+1 || prediction->desc().numInputGlobalChannels!=1 ||
           prediction->desc().numValueChannels!=4)throw std::runtime_error("MuZero CUDA component shape mismatch");
    }
    ~Impl(){cudaSetDevice(device);cudaFree(pointers);cudaFree(actions_from);cudaFree(actions_to);}
    void initialize(){
        representation->initialize();dynamics->initialize();prediction->initialize();
        check(cudaMalloc(&pointers,batch*sizeof(float*)));check(cudaMalloc(&actions_from,batch*sizeof(int)));check(cudaMalloc(&actions_to,batch*sizeof(int)));
    }
    size_t stride()const{return size_t(channels+1)*canvas*canvas;}
    void pack(Engine& engine,int n,int actionPlanes){
        auto stream=engine.stream();
        check(cudaMemcpyAsync(pointers,pointer_host.data(),n*sizeof(float*),cudaMemcpyHostToDevice,stream));
        if(actionPlanes) {
            check(cudaMemcpyAsync(actions_from,from_host.data(),n*sizeof(int),cudaMemcpyHostToDevice,stream));
            check(cudaMemcpyAsync(actions_to,to_host.data(),n*sizeof(int),cudaMemcpyHostToDevice,stream));
        }
        etaPackLatent(pointers,actions_from,actions_to,engine.spatial(),n,channels,canvas*canvas,actionPlanes,stream);
        check(cudaMemsetAsync(engine.globals(),0,n*sizeof(float),stream));
    }
    std::vector<InferenceOutput> predict(std::shared_ptr<Allocation> memory,int n,Engine& previous){
        prediction->wait(previous);pointer_host.resize(n);
        for(int row=0;row<n;++row)pointer_host[row]=memory->data+row*stride();
        pack(*prediction,n,0);
        std::vector<float> policy,pass,value,aux;
        NeuralNet::getDeviceOutput(prediction->handle,prediction->host,n,nullptr,policy,pass,value,aux);
        const auto& desc=prediction->desc();
        auto eval=decode_cuda_outputs(policy,pass,value,aux,n,canvas,spec,desc.numPolicyChannels,desc.numValueChannels,
                                      desc.numScoreValueChannels,desc.postProcessParams.outputScaleMultiplier,allow_no_result);
        std::vector<InferenceOutput> output(n);
        for(int row=0;row<n;++row){auto state=std::make_shared<State>();state->allocation=memory;state->offset=row*stride();state->count=stride();state->owner=owner;state->mask=masks[row];state->allow_no_result=allow_no_result[row];output[row]={state,std::move(eval[row])};}
        return output;
    }
};
CudaBackend::CudaBackend(const std::string& p,int c,int b,int d,const std::string& precision,GameSpec s)
  :Backend(s),impl_(std::make_unique<Impl>(p,c,b,d,precision,s)){}
CudaBackend::~CudaBackend()=default;
void CudaBackend::initialize(){impl_->initialize();}
std::vector<InferenceOutput> CudaBackend::initial(const InferenceInputs& inputs){
    auto& p=*impl_;check(cudaSetDevice(p.device));int n=inputs.size(),area=p.canvas*p.canvas;
    if(n<=0||n>p.batch)throw std::runtime_error("Invalid initial CUDA batch");
    p.masks.resize(n);p.allow_no_result.resize(n);p.spatial_host.resize(n*spec_.planes*area);p.global_host.resize(n*spec_.globals);
    for(int row=0;row<n;++row){
        if(inputs[row]->size()!=size_t(spec_.planes*area+spec_.globals))throw std::runtime_error("MuZero observation shape mismatch");
        auto mask=std::make_shared<std::vector<uint8_t>>(area);int count=0;
        for(int i=0;i<area;++i){float v=(*inputs[row])[i];if(v!=0 && v!=1)throw std::runtime_error("Latent board mask must be binary");(*mask)[i]=v;count+=v;}
        if(!count)throw std::runtime_error("Latent board mask is empty");p.masks[row]=mask;p.allow_no_result[row]=no_result_possible(*inputs[row],spec_,p.canvas);
        for(int i=0;i<area;++i)for(int c=0;c<spec_.planes;++c)p.spatial_host[(row*area+i)*spec_.planes+c]=(*inputs[row])[c*area+i];
        std::copy(inputs[row]->end()-spec_.globals,inputs[row]->end(),p.global_host.begin()+row*spec_.globals);
    }
    auto& e=*p.representation;
    check(cudaMemcpyAsync(e.spatial(),p.spatial_host.data(),p.spatial_host.size()*sizeof(float),cudaMemcpyHostToDevice,e.stream()));
    check(cudaMemcpyAsync(e.globals(),p.global_host.data(),p.global_host.size()*sizeof(float),cudaMemcpyHostToDevice,e.stream()));
    auto memory=std::make_shared<Allocation>(n*p.stride(),p.device);e.latent(n,memory->data);
    return p.predict(memory,n,e);
}
std::vector<InferenceOutput> CudaBackend::recurrent(const std::vector<InferenceAction>& inputs){
    auto& p=*impl_;check(cudaSetDevice(p.device));int n=inputs.size();
    if(n<=0||n>p.batch)throw std::runtime_error("Invalid recurrent CUDA batch");
    p.masks.resize(n);p.allow_no_result.resize(n);p.pointer_host.resize(n);p.from_host.resize(n);p.to_host.resize(n);
    for(int row=0;row<n;++row){
        auto state=std::dynamic_pointer_cast<const State>(inputs[row].latent);
        if(!state||state->owner!=p.owner||state->allocation->device!=p.device||state->count!=p.stride())throw std::runtime_error("MuZero CUDA latent owner/layout mismatch");
        int action=inputs[row].action;if(action<0||action>=spec_.actions(p.canvas))throw std::runtime_error("Invalid latent action");
        p.masks[row]=state->mask;p.allow_no_result[row]=state->allow_no_result;p.pointer_host[row]=state->data();p.from_host[row]=action;p.to_host[row]=-1;
        if(spec_.kind==GameKind::XIANGQI){const auto& move=XIANGQI_MOVES[action];p.from_host[row]=move[1]*p.canvas+move[0];p.to_host[row]=move[3]*p.canvas+move[2];}
        if(p.from_host[row]<p.canvas*p.canvas && !(*state->mask)[p.from_host[row]])throw std::runtime_error("Latent action is padding");
        if(p.to_host[row]>=0 && !(*state->mask)[p.to_host[row]])throw std::runtime_error("Latent action destination is padding");
    }
    p.pack(*p.dynamics,n,p.action_planes);
    auto memory=std::make_shared<Allocation>(n*p.stride(),p.device);p.dynamics->latent(n,memory->data);
    return p.predict(memory,n,*p.dynamics);
}
}
