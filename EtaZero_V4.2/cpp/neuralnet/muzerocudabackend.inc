// Included after the source CUDA Model/ComputeHandle/InputBuffers definitions.
// The learned networks, scratch allocator, precision selection and all backend
// options remain source implementations. This adapter owns only the state boundary.
#include "muzerobundle.inc"

namespace MuZero {
struct DeviceAllocation {
  int device;
  void* data = nullptr;
  DeviceAllocation(int device_, size_t bytes) : device(device_) {
    CUDA_ERR("MuZero allocation",cudaMalloc(&data,bytes));
  }
  ~DeviceAllocation() {
    int previous = device;
    (void)cudaGetDevice(&previous);
    if(previous != device) (void)cudaSetDevice(device);
    (void)cudaFree(data);
    if(previous != device) (void)cudaSetDevice(previous);
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
};

struct StateData {
  std::shared_ptr<DeviceAllocation> allocation;
  std::shared_ptr<const int> owner;
  std::shared_ptr<const std::vector<uint8_t>> mask;
  size_t offset;
  const float* data() const { return (const float*)allocation->data + offset; }
};

const void* stateOwner(const State& state) { return state ? state->owner.get() : nullptr; }

struct Evaluator::Impl {
  const int maxBatch, device;
  const bool inputsUseNHWC;
  int canvas, channels, actionPlanes;
  std::string gameName;
  std::shared_ptr<const int> owner = std::make_shared<const int>(0);
  std::shared_ptr<const ModelBundle> bundle;
  const LoadedModel *representation, *dynamics, *prediction;
  std::unique_ptr<ComputeContext> context;
  // Destruction order keeps the source CUDA handles alive until all layers and
  // scratch buffers using them have been released.
  std::unique_ptr<ComputeHandle> handle;
  std::unique_ptr<Model> representationModel, dynamicsModel;
  std::unique_ptr<Buffers> representationBuffers, dynamicsBuffers;
  std::unique_ptr<InputBuffers> rootHost, predictionHost;
  std::shared_ptr<DeviceAllocation> deviceStates, deviceActions;
  std::vector<const float*> hostStates;
  std::vector<int> hostActionSquares;

  Impl(std::shared_ptr<const ModelBundle> bundle_, int maxBatch_, int device_, enabled_t fp16, ConfigParser& cfg, Logger* logger, bool inputsUseNHWC_)
    : maxBatch(maxBatch_), device(device_ < 0 ? 0 : device_), inputsUseNHWC(inputsUseNHWC_),
      canvas(bundle_->posLen()), channels(bundle_->latentChannels()), actionPlanes(bundle_->game() == "xiangqi" ? 2 : 1),
      gameName(bundle_->game()), bundle(std::move(bundle_)),
      representation(bundle->impl->representation.get()), dynamics(bundle->impl->dynamics.get()), prediction(bundle->impl->prediction.get()) {
    if(maxBatch <= 0) throw StringError("MuZero maxBatch must be positive");
    if((int64_t)channels+1+actionPlanes > (((int64_t)1<<31)-1)/maxBatch/(canvas*canvas))
      throw StringError("MuZero GPU tensor exceeds the supported 2^31 entry bound");
    context.reset(NeuralNet::createComputeContext({device},logger,canvas,canvas,"",fp16,prediction,cfg));
    // All three networks share a stream/layout. The source CUDA transformer
    // requirement applies even when only a body (not prediction) has attention.
    if(bundle->hasTransformer()) {
      if(context->useNHWCMode == enabled_t::False)
        throw StringError("CUDA backend: MuZero transformer components require NHWC");
      context->useNHWCMode = enabled_t::True;
    }
    handle.reset(NeuralNet::createComputeHandle(context.get(),prediction,logger,maxBatch,false,inputsUseNHWC,device,0));
    auto* cuda = handle->cudaHandles.get();
    representationModel = std::make_unique<Model>(cuda,&representation->modelDesc,maxBatch,canvas,canvas,
                                                  inputsUseNHWC,handle->usingFP16,handle->usingNHWC);
    dynamicsModel = std::make_unique<Model>(cuda,&dynamics->modelDesc,maxBatch,canvas,canvas,
                                           inputsUseNHWC,handle->usingFP16,handle->usingNHWC);
    representationBuffers = std::make_unique<Buffers>(cuda,*representationModel,*handle->scratch);
    dynamicsBuffers = std::make_unique<Buffers>(cuda,*dynamicsModel,*handle->scratch);
    rootHost = std::make_unique<InputBuffers>(representation,maxBatch,canvas,canvas);
    predictionHost = std::make_unique<InputBuffers>(prediction,maxBatch,canvas,canvas);
    deviceStates = std::make_shared<DeviceAllocation>(device,maxBatch*sizeof(const float*));
    deviceActions = std::make_shared<DeviceAllocation>(device,(size_t)maxBatch*actionPlanes*sizeof(int));
    hostStates.resize(maxBatch);
    if(gameName == "xiangqi") hostActionSquares.resize((size_t)maxBatch*actionPlanes);
    CUDA_ERR("MuZero initialize",cudaStreamSynchronize(cuda->stream));
  }
  ~Impl() {
    // The worker owns these handles, but state handles can be released on any
    // search thread (or after this evaluator), hence DeviceAllocation's device guard.
    if(handle) (void)cudaStreamSynchronize(handle->cudaHandles->stream);
  }
  size_t stateSize() const { return (size_t)(channels+1)*canvas*canvas; }
  void checkBatch(size_t batch) const {
    if(batch == 0 || batch > (size_t)maxBatch) throw StringError("MuZero batch is empty or exceeds maxBatch");
  }
  void checkState(const State& state) const {
    if(!state || state->owner != owner) throw StringError("MuZero state belongs to another evaluator/model");
  }
  cudaStream_t stream() const { return handle->cudaHandles->stream; }
  void upload(float* host, float* fp32, void* native, size_t count) {
    if(count == 0) return;
    CUDA_ERR("MuZero input upload",cudaMemcpyAsync(handle->usingFP16 ? fp32 : native,host,count*sizeof(float),cudaMemcpyHostToDevice,stream()));
    if(handle->usingFP16) customCudaCopyToHalf(fp32,(half*)native,count,stream());
    CUDA_ERR("MuZero input convert",cudaPeekAtLastError());
  }
  void apply(Model* model, Buffers* buffers, int batch, float* latent = nullptr) {
    model->apply(handle->cudaHandles.get(),handle->scratch.get(),batch,false,
      buffers->inputBuf,buffers->inputGlobalBuf,buffers->inputMetaBuf,
      buffers->policyPassBuf,buffers->policyBuf,buffers->valueBuf,buffers->auxValueBuf,
      buffers->workspaceBuf,buffers->workspaceBytes,latent);
  }
  void pack(Buffers* buffers, int batch, const float* contiguous, bool actions) {
    float* packed = handle->usingFP16 ? buffers->inputBufFloat : (float*)buffers->inputBuf;
    const int planes = actions ? actionPlanes : 0;
    muZeroPack((const float* const*)deviceStates->data,contiguous,
      actions ? (const int*)deviceActions->data : nullptr,packed,batch,channels,canvas*canvas,planes,inputsUseNHWC,stream());
    if(handle->usingFP16)
      customCudaCopyToHalf(packed,(half*)buffers->inputBuf,(size_t)batch*(channels+1+planes)*canvas*canvas,stream());
    CUDA_ERR("MuZero state pack",cudaPeekAtLastError());
  }
  std::vector<Output> predict(int batch, std::shared_ptr<DeviceAllocation> storage,
                             const std::vector<std::shared_ptr<const std::vector<uint8_t>>>& masks) {
    auto* buffers = handle->buffers.get();
    auto* host = predictionHost.get();
    pack(buffers,batch,(const float*)storage->data,false);
    apply(handle->model.get(),buffers,batch);
    if(gameName == "go")
      CUDA_ERR("MuZero pass download",cudaMemcpyAsync(host->policyPassResults,buffers->policyPassBuf,batch*host->singlePolicyPassResultBytes,cudaMemcpyDeviceToHost,stream()));
    CUDA_ERR("MuZero policy download",cudaMemcpyAsync(host->policyResults,buffers->policyBuf,batch*host->singlePolicyResultBytes,cudaMemcpyDeviceToHost,stream()));
    CUDA_ERR("MuZero value download",cudaMemcpyAsync(host->valueResults,buffers->valueBuf,batch*host->singleValueResultBytes,cudaMemcpyDeviceToHost,stream()));
    CUDA_ERR("MuZero auxiliary download",cudaMemcpyAsync(host->auxValueResults,buffers->auxValueBuf,batch*host->singleAuxValueResultBytes,cudaMemcpyDeviceToHost,stream()));
    CUDA_ERR("MuZero output completion",cudaStreamSynchronize(stream()));
    const auto& desc = prediction->modelDesc;
    const int area = canvas*canvas, actions = gameName == "xiangqi" ? Xiangqi::NUM_ACTIONS : area+(gameName == "go");
    const int policies = desc.numPolicyChannels;
    const float scale = desc.postProcessParams.outputScaleMultiplier;
    std::vector<Output> results(batch);
    for(int row = 0; row < batch; row++) {
      auto& result = results[row];
      result.state = std::make_shared<StateData>(StateData{storage,owner,masks[row],row*stateSize()});
      result.policy.resize(policies*actions);
      for(int channel = 0; channel < policies; channel++) {
        if(gameName == "xiangqi") {
          const int spatialChannels = policies*Xiangqi::NUM_DELTAS;
          for(int action = 0; action < actions; action++) {
            const int index = Xiangqi::policyIndex(action,canvas,canvas);
            const int point = index%area, spatialChannel = channel*Xiangqi::NUM_DELTAS+index/area;
            const int offset = row*spatialChannels*area +
              (handle->usingNHWC ? point*spatialChannels+spatialChannel : spatialChannel*area+point);
            result.policy[channel*actions+action] = host->policyResults[offset]*scale;
          }
        }
        else for(int point = 0; point < area; point++) {
          const int offset = row*policies*area + (handle->usingNHWC ? point*policies+channel : channel*area+point);
          // Same static spatial mask as the Python head. Root game legality and
          // recurrent occupancy are intentionally not part of this boundary.
          result.policy[channel*actions+point] = (*masks[row])[point] ? host->policyResults[offset]*scale : -5000.0f;
        }
        if(gameName == "go") result.policy[channel*actions+area] = host->policyPassResults[row*policies+channel]*scale;
      }
      for(int c = 0; c < 4; c++) result.value[c] = host->valueResults[row*4+c]*scale;
      for(int c = 0; c < 2; c++) result.auxiliary[c] = host->auxValueResults[row*2+c]*scale;
    }
    return results;
  }
};

Evaluator::Evaluator(const std::string& directory, int maxBatch, int device, enabled_t fp16, ConfigParser& cfg, Logger* logger, bool inputsUseNHWC_)
  : Evaluator(std::make_shared<ModelBundle>(directory),maxBatch,device,fp16,cfg,logger,inputsUseNHWC_) {}
Evaluator::Evaluator(std::shared_ptr<const ModelBundle> bundle, int maxBatch, int device, enabled_t fp16, ConfigParser& cfg, Logger* logger, bool inputsUseNHWC_)
{
  if(!bundle) throw StringError("MuZero model bundle must not be null");
  impl = std::make_unique<Impl>(std::move(bundle),maxBatch,device,fp16,cfg,logger,inputsUseNHWC_);
}
Evaluator::~Evaluator() = default;
const ModelDesc& Evaluator::representationDesc() const { return impl->representation->modelDesc; }
const ModelDesc& Evaluator::predictionDesc() const { return impl->prediction->modelDesc; }
const void* Evaluator::identity() const { return impl->owner.get(); }
bool Evaluator::isUsingFP16() const { return impl->handle->usingFP16; }
int Evaluator::posLen() const { return impl->canvas; }
int Evaluator::latentChannels() const { return impl->channels; }
const std::string& Evaluator::game() const { return impl->gameName; }

std::vector<Output> Evaluator::initial(int batch, const std::vector<float>& spatial,
                                      const std::vector<float>& global, const std::vector<float>& metadata) {
  impl->checkBatch(batch);
  auto* host = impl->rootHost.get();
  if(spatial.size() != batch*host->singleInputElts || global.size() != batch*host->singleInputGlobalElts ||
     metadata.size() != batch*host->singleInputMetaElts)
    throw StringError("MuZero initial observation dimensions differ");
  const int area = impl->canvas*impl->canvas;
  std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
  for(int row = 0; row < batch; row++) {
    auto mask = std::make_shared<std::vector<uint8_t>>(area);
    bool nonempty = false;
    for(int point = 0; point < area; point++) {
      float value = spatial[row*host->singleInputElts+point];
      if(value != 0 && value != 1) throw StringError("MuZero observation mask must be binary");
      if(impl->gameName == "xiangqi" &&
         (value != 0) != (point%impl->canvas < Xiangqi::WIDTH && point/impl->canvas < Xiangqi::HEIGHT))
        throw StringError("Xiangqi MuZero observation mask must cover exactly the 9 x 10 board");
      (*mask)[point] = value != 0; nonempty |= value != 0;
    }
    if(!nonempty) throw StringError("MuZero observation mask must be nonempty");
    masks.push_back(std::move(mask));
  }
  // The public observation contract is NCHW; adapt only at the source backend
  // boundary, which also handles conversion to the chosen compute layout.
  const int features = impl->representation->modelDesc.numInputChannels;
  if(impl->inputsUseNHWC) {
    for(int row = 0; row < batch; row++)
      for(int point = 0; point < area; point++)
        for(int channel = 0; channel < features; channel++)
          host->userInputBuffer[(row*area+point)*features+channel] = spatial[(row*features+channel)*area+point];
  }
  else std::copy(spatial.begin(),spatial.end(),host->userInputBuffer);
  std::copy(global.begin(),global.end(),host->userInputGlobalBuffer);
  if(!metadata.empty()) std::copy(metadata.begin(),metadata.end(),host->userInputMetaBuffer);
  auto storage = std::make_shared<DeviceAllocation>(impl->device,batch*impl->stateSize()*sizeof(float));
  auto* buffers = impl->representationBuffers.get();
  try {
    impl->upload(host->userInputBuffer,buffers->inputBufFloat,buffers->inputBuf,spatial.size());
    impl->upload(host->userInputGlobalBuffer,buffers->inputGlobalBufFloat,buffers->inputGlobalBuf,global.size());
    impl->upload(host->userInputMetaBuffer,buffers->inputMetaBufFloat,buffers->inputMetaBuf,metadata.size());
    impl->apply(impl->representationModel.get(),buffers,batch,(float*)storage->data);
    return impl->predict(batch,storage,masks);
  }
  catch(...) { (void)cudaStreamSynchronize(impl->stream()); throw; }
}

std::vector<Output> Evaluator::recurrent(const std::vector<State>& states, const std::vector<int>& actions) {
  impl->checkBatch(states.size());
  if(actions.size() != states.size()) throw StringError("MuZero recurrent action count differs");
  const int batch = states.size(), area = impl->canvas*impl->canvas;
  const bool xiangqi = impl->gameName == "xiangqi";
  std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
  for(int row = 0; row < batch; row++) {
    impl->checkState(states[row]);
    const int action = actions[row];
    if(action < 0 || action >= (xiangqi ? Xiangqi::NUM_ACTIONS : area+(impl->gameName == "go")) ||
       (!xiangqi && action < area && !(*states[row]->mask)[action]))
      throw StringError("MuZero recurrent action is outside the static game domain");
    if(xiangqi) {
      const auto& move = Xiangqi::MOVES[action];
      impl->hostActionSquares[2*row] = move[1]*impl->canvas+move[0];
      impl->hostActionSquares[2*row+1] = move[3]*impl->canvas+move[2];
    }
    masks.push_back(states[row]->mask);
    impl->hostStates[row] = states[row]->data();
  }
  auto storage = std::make_shared<DeviceAllocation>(impl->device,batch*impl->stateSize()*sizeof(float));
  try {
    CUDA_ERR("MuZero state pointers",cudaMemcpyAsync(impl->deviceStates->data,impl->hostStates.data(),batch*sizeof(const float*),cudaMemcpyHostToDevice,impl->stream()));
    CUDA_ERR("MuZero actions",cudaMemcpyAsync(impl->deviceActions->data,xiangqi ? impl->hostActionSquares.data() : actions.data(),
      (size_t)batch*impl->actionPlanes*sizeof(int),cudaMemcpyHostToDevice,impl->stream()));
    impl->pack(impl->dynamicsBuffers.get(),batch,nullptr,true);
    impl->apply(impl->dynamicsModel.get(),impl->dynamicsBuffers.get(),batch,(float*)storage->data);
    return impl->predict(batch,storage,masks);
  }
  catch(...) { (void)cudaStreamSynchronize(impl->stream()); throw; }
}

void Evaluator::warmup() {
  if(!impl->bundle->hasTransformer()) return;
  const bool previous = NeuralNet::setIsWarmup(impl->handle.get(),true);
  try {
    const auto& desc = representationDesc();
    const int area = posLen()*posLen();
    for(int count = 1; count <= impl->maxBatch; count++) {
      std::vector<float> spatial((size_t)count*desc.numInputChannels*area,0.0f);
      for(int row = 0; row < count; row++) {
        auto mask = spatial.begin()+(size_t)row*desc.numInputChannels*area;
        if(game() == "xiangqi") {
          for(int y = 0; y < Xiangqi::HEIGHT; y++)
            std::fill_n(mask+y*posLen(),Xiangqi::WIDTH,1.0f);
        }
        else std::fill_n(mask,area,1.0f);
      }
      auto output = initial(count,spatial,std::vector<float>((size_t)count*desc.numInputGlobalChannels),
                            std::vector<float>((size_t)count*desc.numInputMetaChannels));
      std::vector<State> states;
      for(const auto& row: output) states.push_back(row.state);
      recurrent(states,std::vector<int>(count,0));
    }
  }
  catch(...) { NeuralNet::setIsWarmup(impl->handle.get(),previous); throw; }
  NeuralNet::setIsWarmup(impl->handle.get(),previous);
}

std::vector<float> Evaluator::readStateForTesting(const State& state) const {
  impl->checkState(state);
  std::vector<float> result(impl->stateSize());
  CUDA_ERR("MuZero diagnostic state download",cudaMemcpyAsync(result.data(),state->data(),result.size()*sizeof(float),cudaMemcpyDeviceToHost,impl->stream()));
  CUDA_ERR("MuZero diagnostic completion",cudaStreamSynchronize(impl->stream()));
  return result;
}
}
