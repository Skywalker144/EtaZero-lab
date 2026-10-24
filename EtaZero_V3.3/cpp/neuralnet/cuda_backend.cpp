#include "neuralnet/cuda_backend.h"
#include "neuralnet/nninterface.h"
#include "neuralnet/sgfmetadata.h"
#include <cuda_runtime.h>
#include <cmath>
#include <numeric>
namespace etazero {
struct CudaBackend::Impl {
    LoadedModel* model=nullptr;
    ComputeContext* context=nullptr;
    ComputeHandle* handle=nullptr;
    InputBuffers* buffers=nullptr;
    ConfigParser config;
    SGFMetadata profile;
    int canvas,max_batch,device;
    bool fp16;
    Impl(const std::string& path,int size,int batch,int gpu,const std::string& precision,const std::string& human)
      :canvas(size),max_batch(batch),device(gpu),fp16(precision=="float16") {
        if(precision!="float16" && precision!="float32")throw std::runtime_error("Invalid CUDA inference precision");
        model=NeuralNet::loadModelFile(path,"");
        if(!human.empty())profile=SGFMetadata::getProfile(human);
    }
    ~Impl(){
        if(buffers)NeuralNet::freeInputBuffers(buffers);
        if(handle)NeuralNet::freeComputeHandle(handle);
        if(context)NeuralNet::freeComputeContext(context);
        if(model)NeuralNet::freeLoadedModel(model);
    }
};
CudaBackend::CudaBackend(const std::string& path,int canvas,int batch,int device,
                         const std::string& precision,GameSpec spec,const std::string& profile)
  :Backend(spec),impl_(std::make_unique<Impl>(path,canvas,batch,device,precision,profile)) {
    const auto& desc=NeuralNet::getModelDesc(impl_->model);
    if(desc.numInputChannels!=spec.planes || desc.numInputGlobalChannels!=spec.globals || (desc.numValueChannels!=4 && !(spec.kind==GameKind::GO && desc.numValueChannels==3)))
        throw std::runtime_error("CUDA model game input/outcome layout mismatch");
    if(!desc.numInputMetaChannels && !profile.empty())throw std::runtime_error("Human SL profile has no metadata encoder to condition");
}
CudaBackend::~CudaBackend()=default;
bool CudaBackend::available() const {return !metadata_channels()||impl_->profile.initialized;}
int CudaBackend::metadata_channels() const {return NeuralNet::getModelDesc(impl_->model).numInputMetaChannels;}
ModelDescription CudaBackend::description() const {
    const auto& d=NeuralNet::getModelDesc(impl_->model);
    return {d.name,d.modelVersion,d.numInputMetaChannels>0,d.preferPassAliveUnderSuicideRules,d.preferExcludeTerritoryAdjacentToAtari};
}
void CudaBackend::append_metadata(std::vector<float>& input,int player) const {
    if(!metadata_channels())return;
    if(!impl_->profile.initialized)throw std::runtime_error("Human SL profile is not set");
    if(player!=1 && player!=-1)throw std::runtime_error("Metadata requires the actual side to move");
    const int area=impl_->canvas*impl_->canvas;
    int board_area=std::lround(std::accumulate(input.begin(),input.begin()+area,0.0f));
    size_t begin=input.size();input.resize(begin+192);
    SGFMetadata::fillMetadataRow(&impl_->profile,input.data()+begin,player==1?P_BLACK:P_WHITE,board_area);
}
void CudaBackend::initialize(){
    auto& p=*impl_;if(p.handle)return;
    p.context=NeuralNet::createComputeContext({p.device},nullptr,p.canvas,p.canvas,"",
        p.fp16?enabled_t::True:enabled_t::False,p.model,p.config);
    p.handle=NeuralNet::createComputeHandle(p.context,p.model,nullptr,p.max_batch,false,true,p.device,0);
    p.buffers=NeuralNet::createInputBuffers(p.model,p.max_batch,p.canvas,p.canvas);
}
std::vector<Evaluation> CudaBackend::evaluate(const InferenceInputs& inputs){
    auto& p=*impl_;if(!p.handle)throw std::runtime_error("CUDA backend is not initialized");
    if(cudaSetDevice(p.device)!=cudaSuccess)throw std::runtime_error("Cannot select CUDA device");
    const int n=inputs.size(),area=p.canvas*p.canvas,spatial=spec_.planes*area;
    const auto& desc=NeuralNet::getModelDesc(p.model);
    std::vector<const float*> obs(n),globals(n),metas(n);
    for(int row=0;row<n;++row){
        if(inputs[row]->size()!=static_cast<size_t>(spatial+spec_.globals+desc.numInputMetaChannels))throw std::runtime_error("CUDA input shape mismatch");
        obs[row]=inputs[row]->data();globals[row]=obs[row]+spatial;
        if(desc.numInputMetaChannels)metas[row]=globals[row]+spec_.globals;
    }
    std::vector<float> policy,pass,value,auxiliary;
    NeuralNet::getRawOutput(p.handle,p.buffers,n,obs.data(),globals.data(),metas.data(),policy,pass,value,auxiliary);
    std::vector<uint8_t> allow(n);for(int row=0;row<n;++row)allow[row]=no_result_possible(*inputs[row],spec_,p.canvas);
    auto result=decode_cuda_outputs(policy,pass,value,auxiliary,n,p.canvas,spec_,desc.numPolicyChannels,
                               desc.numValueChannels,desc.numScoreValueChannels,desc.postProcessParams.outputScaleMultiplier,allow);
    for(auto& output:result)output.policy_optimism_allowed=desc.numInputMetaChannels==0;
    return result;
}
std::vector<Evaluation> decode_cuda_outputs(std::vector<float>& policy,std::vector<float>& pass,
    std::vector<float>& value,std::vector<float>& auxiliary,int n,int canvas,GameSpec spec,
    int policyChannels,int valueChannels,int scoreChannels,float scale,const std::vector<uint8_t>& allow_no_result) {
    int area=canvas*canvas;
    // Source ModelDesc may rescale activations for FP16. Policy scaling is
    // deferred until after optimism mixing and combined with NN temperature.
    for(auto* output:{&value,&auxiliary})for(auto& v:*output)v*=scale;
    std::vector<Evaluation> result(n);
    for(int row=0;row<n;++row){
        auto& out=result[row];out.policy_output_scale=scale;int channels=policyChannels;
        out.logits.assign(policy.begin()+row*channels*area,policy.begin()+(row*channels+1)*area);
        out.optimistic_logits.assign(policy.begin()+(row*channels+1)*area,policy.begin()+(row*channels+2)*area);
        if(spec.kind==GameKind::XIANGQI){
            out.logits.resize(spec.fixed_actions);out.optimistic_logits.resize(spec.fixed_actions);
            const int deltas=channels/2;
            for(int a=0;a<spec.fixed_actions;++a){
                const auto& m=XIANGQI_MOVES[a];int channel=-1;
                for(int d=0;d<deltas;++d)if(XIANGQI_DELTAS[d][0]==m[2]-m[0] && XIANGQI_DELTAS[d][1]==m[3]-m[1]){channel=d;break;}
                if(channel<0)throw std::runtime_error("CUDA Xiangqi action plane mismatch");
                int point=m[1]*canvas+m[0];
                out.logits[a]=policy[(row*channels+channel)*area+point];
                out.optimistic_logits[a]=policy[(row*channels+deltas+channel)*area+point];
            }
        }
        if(spec.kind==GameKind::GO){out.logits.push_back(pass[row*channels]);out.optimistic_logits.push_back(pass[row*channels+1]);}
        WDL logits;for(int i=0;i<4;++i){
            int source=i==0?0:i==1?3:i==2?1:2;
            logits[i]=source>=valueChannels?-std::numeric_limits<double>::infinity():value[row*valueChannels+source];
        }
        out.wdl=outcome_probabilities(logits,allow_no_result.at(row));
        float variance=auxiliary[row*scoreChannels+3];out.variance_time=40*(std::max(0.0f,variance)+std::log1p(std::exp(-std::abs(variance))));
        float raw=auxiliary[row*scoreChannels+4]*.5f;
        out.shortterm_value_stdev=.5*(std::max(0.0f,raw)+std::log1p(std::exp(-std::abs(raw))));out.has_auxiliary=true;
    }
    return result;
}
}

namespace etazero { void CudaBackend::set_human_profile(const std::string& value) {validate_human_profile(value);impl_->profile=SGFMetadata::getProfile(value);} }

namespace etazero {
void CudaBackend::validate_human_profile(const std::string& value) const {if(!metadata_channels())throw std::runtime_error("Metadata model requires a human profile");SGFMetadata::getProfile(value);}
void CudaBackend::append_profile(std::vector<float>& input,int player,const SGFMetadata& profile) const {

    int board_area=std::lround(std::accumulate(input.begin(),input.begin()+impl_->canvas*impl_->canvas,0.0f));size_t begin=input.size();input.resize(begin+192);
    SGFMetadata::fillMetadataRow(&profile,input.data()+begin,player==1?P_BLACK:P_WHITE,board_area);
}
}
