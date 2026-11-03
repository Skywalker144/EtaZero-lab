// Included after the source Eigen layers, Model and ComputeHandle definitions.
// Learned computation uses those implementations; only the latent boundary is
// specific to MuZero. Stored states are FP32 NCHW with the mask in the last plane.
#include "muzerobundle.inc"

namespace MuZero {
struct StateData {
  std::shared_ptr<const std::vector<float>> storage;
  std::shared_ptr<const int> owner;
  std::shared_ptr<const std::vector<uint8_t>> mask;
  size_t offset;
  const float* data() const { return storage->data()+offset; }
};
const void* stateOwner(const State& state) { return state ? state->owner.get() : nullptr; }

// A headless source trunk followed by the exported learned 1x1 projection.
// Ordinary Model and its heads keep their original execution path.
struct EigenBody {
  const ModelDesc& desc;
  Trunk trunk;
  ConvLayer projection;
  TENSOR4 trunkOutput, projected;
  TENSOR3 mask;
  std::vector<float> maskSum, workspace, input;

  EigenBody(const ModelDesc& d, int canvas, int maxBatch)
    : desc(d), trunk(d.trunk,canvas,canvas), projection(d.latentProjection,canvas,canvas),
      trunkOutput(d.trunk.trunkNumChannels,canvas,canvas,maxBatch),
      projected(d.numLatentChannels,canvas,canvas,maxBatch), mask(canvas,canvas,maxBatch),
      maskSum(maxBatch),
      workspace(std::max(trunk.requiredConvWorkspaceElts(maxBatch),projection.requiredConvWorkspaceElts(maxBatch))),
      input((size_t)maxBatch*d.numInputChannels*canvas*canvas)
  {}

  void apply(ComputeHandle* handle, int batch, const float* global, const float* metadata, float* output) {
    const int canvas = handle->context->nnXLen, area = canvas*canvas, channels = desc.numLatentChannels;
    CONSTTENSORMAP4 spatial(input.data(),desc.numInputChannels,canvas,canvas,batch);
    CONSTTENSORMAP2 globalInput((float*)global,desc.numInputGlobalChannels,batch);
    CONSTTENSORMAP2 metaInput((float*)metadata,desc.numInputMetaChannels,batch);
    TENSORMAP4 trunkBuf(trunkOutput.data(),desc.trunk.trunkNumChannels,canvas,canvas,batch);
    TENSORMAP4 projectedBuf(projected.data(),channels,canvas,canvas,batch);
    TENSORMAP3 maskBuf(mask.data(),canvas,canvas,batch);
    maskBuf = spatial.chip(0,0);
    computeMaskSum(&maskBuf,maskSum.data());
    trunk.apply(&handle->handleInternal,handle->scratch.get(),&spatial,&globalInput,
      desc.numInputMetaChannels > 0 ? &metaInput : nullptr,&trunkBuf,&maskBuf,maskSum.data(),workspace.data());
    projection.apply(&handle->handleInternal,&trunkBuf,&projectedBuf,workspace.data(),false);

    // Same masked min/max and small-span denominator as Python/CUDA. Preserve
    // nonfinite values rather than hiding a failed network evaluation.
    for(int row = 0; row < batch; row++) {
      const float* values = projected.data()+(size_t)row*channels*area;
      const float* rowMask = mask.data()+(size_t)row*area;
      float lo = std::numeric_limits<float>::infinity(), hi = -lo;
      for(int point = 0; point < area; point++) {
        if(rowMask[point] == 0) continue;
        for(int channel = 0; channel < channels; channel++) {
          const float v = values[point*channels+channel];
          if(std::isnan(v)) lo = hi = v;
          else if(!std::isnan(lo)) { lo = std::min(lo,v); hi = std::max(hi,v); }
        }
      }
      const float span = hi-lo, denominator = span < 1e-5f ? span+1e-5f : span;
      float* state = output+(size_t)row*(channels+1)*area;
      for(int channel = 0; channel < channels; channel++)
        for(int point = 0; point < area; point++)
          state[channel*area+point] = rowMask[point] == 0 ? 0 : (values[point*channels+channel]-lo)/denominator;
      std::copy(rowMask,rowMask+area,state+channels*area);
    }
  }
};

struct Evaluator::Impl {
  const int maxBatch, canvas, channels, actionPlanes;
  const std::string gameName;
  std::shared_ptr<const int> owner = std::make_shared<const int>(0);
  std::shared_ptr<const ModelBundle> bundle;
  const LoadedModel *representation, *dynamics, *prediction;
  std::unique_ptr<ComputeContext> context;
  std::unique_ptr<ComputeHandle> handle;
  std::unique_ptr<EigenBody> representationBody, dynamicsBody;
  std::vector<float> predictionInput;

  Impl(std::shared_ptr<const ModelBundle> bundle_, int maxBatch_, int device, enabled_t fp16,
       ConfigParser& config, Logger* logger, bool inputsUseNHWC)
    : maxBatch(maxBatch_), canvas(bundle_->posLen()), channels(bundle_->latentChannels()),
      actionPlanes(bundle_->game() == "xiangqi" ? 2 : 1), gameName(bundle_->game()), bundle(std::move(bundle_)),
      representation(bundle->impl->representation.get()), dynamics(bundle->impl->dynamics.get()),
      prediction(bundle->impl->prediction.get()) {
    if(maxBatch <= 0) throw StringError("MuZero maxBatch must be positive");
    if((int64_t)channels+1+actionPlanes > (((int64_t)1<<31)-1)/maxBatch/(canvas*canvas))
      throw StringError("MuZero tensor exceeds the supported 2^31 entry bound");
    context.reset(NeuralNet::createComputeContext({device},logger,canvas,canvas,"",fp16,prediction,config));
    handle.reset(NeuralNet::createComputeHandle(context.get(),prediction,logger,maxBatch,false,inputsUseNHWC,device,0));
    representationBody = std::make_unique<EigenBody>(representation->modelDesc,canvas,maxBatch);
    dynamicsBody = std::make_unique<EigenBody>(dynamics->modelDesc,canvas,maxBatch);
    predictionInput.resize((size_t)maxBatch*(channels+1)*canvas*canvas);
  }
  size_t stateSize() const { return (size_t)(channels+1)*canvas*canvas; }
  void checkBatch(size_t batch) const {
    if(batch == 0 || batch > (size_t)maxBatch) throw StringError("MuZero batch is empty or exceeds maxBatch");
  }
  void checkState(const State& state) const {
    if(!state || state->owner != owner) throw StringError("MuZero state belongs to another evaluator/model");
  }
  void pack(const float* state, float* destination, const int* actionSquares = nullptr) const {
    const int area = canvas*canvas, planes = actionSquares ? actionPlanes : 0;
    const int width = channels+1+planes;
    for(int point = 0; point < area; point++) {
      destination[point*width] = state[channels*area+point];
      for(int channel = 0; channel < channels; channel++)
        destination[point*width+channel+1] = state[channel*area+point];
      for(int plane = 0; plane < planes; plane++)
        destination[point*width+channels+1+plane] = actionSquares[plane] == point;
    }
  }
  std::vector<Output> predict(int batch, std::shared_ptr<const std::vector<float>> storage,
                            const std::vector<std::shared_ptr<const std::vector<uint8_t>>>& masks) {
    const int area = canvas*canvas;
    for(int row = 0; row < batch; row++)
      pack(storage->data()+row*stateSize(),predictionInput.data()+row*stateSize());
    const auto& desc = prediction->modelDesc;
    auto& buffers = *handle->buffers;
    CONSTTENSORMAP4 input(predictionInput.data(),desc.numInputChannels,canvas,canvas,batch);
    CONSTTENSORMAP2 global(nullptr,0,batch);
    TENSORMAP4 trunkBuf(buffers.trunk.data(),buffers.trunk.dimension(0),canvas,canvas,batch);
    TENSORMAP4 policy(buffers.policy.data(),buffers.policy.dimension(0),canvas,canvas,batch);
    TENSORMAP2 pass(buffers.policyPass.data(),desc.numPolicyChannels,batch);
    TENSORMAP2 value(buffers.value.data(),desc.numValueChannels,batch);
    TENSORMAP2 aux(buffers.auxValue.data(),desc.numAuxValueChannels,batch);
    TENSORMAP3 mask(buffers.mask.data(),canvas,canvas,batch);
    handle->model->apply(&handle->handleInternal,handle->scratch.get(),&input,&global,nullptr,
      &trunkBuf,&pass,&policy,&value,&aux,&mask,buffers.maskSum.data(),buffers.convWorkspace.data());
    const int actions = gameName == "xiangqi" ? Xiangqi::NUM_ACTIONS : area+(gameName == "go");
    const int policies = desc.numPolicyChannels, spatialChannels = desc.policyHead.p2Conv.outChannels;
    const float scale = desc.postProcessParams.outputScaleMultiplier;
    std::vector<Output> results(batch);
    for(int row = 0; row < batch; row++) {
      auto& result = results[row];
      result.state = std::make_shared<StateData>(StateData{storage,owner,masks[row],row*stateSize()});
      result.policy.resize(policies*actions);
      for(int channel = 0; channel < policies; channel++) {
        if(gameName == "xiangqi") {
          for(int action = 0; action < actions; action++) {
            const int index = Xiangqi::policyIndex(action,canvas,canvas);
            const int offset = (row*area+index%area)*spatialChannels+channel*Xiangqi::NUM_DELTAS+index/area;
            result.policy[channel*actions+action] = policy.data()[offset]*scale;
          }
        }
        else for(int point = 0; point < area; point++)
          result.policy[channel*actions+point] = (*masks[row])[point] ? policy.data()[(row*area+point)*policies+channel]*scale : -5000.0f;
        if(gameName == "go") result.policy[channel*actions+area] = pass.data()[row*policies+channel]*scale;
      }
      for(int c = 0; c < 4; c++) result.value[c] = value.data()[row*4+c]*scale;
      for(int c = 0; c < 2; c++) result.auxiliary[c] = aux.data()[row*2+c]*scale;
    }
    return results;
  }
};

Evaluator::Evaluator(const std::string& directory, int maxBatch, int device, enabled_t fp16,
                     ConfigParser& config, Logger* logger, bool inputsUseNHWC)
  : Evaluator(std::make_shared<ModelBundle>(directory),maxBatch,device,fp16,config,logger,inputsUseNHWC) {}
Evaluator::Evaluator(std::shared_ptr<const ModelBundle> bundle, int maxBatch, int device, enabled_t fp16,
                     ConfigParser& config, Logger* logger, bool inputsUseNHWC) {
  if(!bundle) throw StringError("MuZero model bundle must not be null");
  impl = std::make_unique<Impl>(std::move(bundle),maxBatch,device,fp16,config,logger,inputsUseNHWC);
}
Evaluator::~Evaluator() = default;
const ModelDesc& Evaluator::representationDesc() const { return impl->representation->modelDesc; }
const ModelDesc& Evaluator::predictionDesc() const { return impl->prediction->modelDesc; }
const void* Evaluator::identity() const { return impl->owner.get(); }
bool Evaluator::isUsingFP16() const { return false; }
int Evaluator::posLen() const { return impl->canvas; }
int Evaluator::latentChannels() const { return impl->channels; }
const std::string& Evaluator::game() const { return impl->gameName; }
// Eigen has no device/kernel autotuning, matching its ordinary warmup hook.
void Evaluator::warmup() {}

std::vector<Output> Evaluator::initial(int batch, const std::vector<float>& spatial,
                                      const std::vector<float>& global, const std::vector<float>& metadata) {
  impl->checkBatch(batch);
  const auto& desc = representationDesc();
  const int area = impl->canvas*impl->canvas, features = desc.numInputChannels;
  if(spatial.size() != (size_t)batch*features*area || global.size() != (size_t)batch*desc.numInputGlobalChannels ||
     metadata.size() != (size_t)batch*desc.numInputMetaChannels)
    throw StringError("MuZero initial observation dimensions differ");
  std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
  for(int row = 0; row < batch; row++) {
    auto mask = std::make_shared<std::vector<uint8_t>>(area);
    bool nonempty = false;
    for(int point = 0; point < area; point++) {
      const float value = spatial[(size_t)row*features*area+point];
      if(value != 0 && value != 1) throw StringError("MuZero observation mask must be binary");
      if(impl->gameName == "xiangqi" &&
         (value != 0) != (point%impl->canvas < Xiangqi::WIDTH && point/impl->canvas < Xiangqi::HEIGHT))
        throw StringError("Xiangqi MuZero observation mask must cover exactly the 9 x 10 board");
      (*mask)[point] = value != 0; nonempty |= value != 0;
    }
    if(!nonempty) throw StringError("MuZero observation mask must be nonempty");
    masks.push_back(std::move(mask));
  }
  auto& body = *impl->representationBody;
  for(int row = 0; row < batch; row++)
    for(int point = 0; point < area; point++)
      for(int channel = 0; channel < features; channel++)
        body.input[(row*area+point)*features+channel] = spatial[(row*features+channel)*area+point];
  auto storage = std::make_shared<std::vector<float>>(batch*impl->stateSize());
  body.apply(impl->handle.get(),batch,global.data(),metadata.data(),storage->data());
  return impl->predict(batch,std::move(storage),masks);
}

std::vector<Output> Evaluator::recurrent(const std::vector<State>& states, const std::vector<int>& actions) {
  impl->checkBatch(states.size());
  if(actions.size() != states.size()) throw StringError("MuZero recurrent action count differs");
  const int batch = states.size(), area = impl->canvas*impl->canvas;
  const bool xiangqi = impl->gameName == "xiangqi";
  auto& body = *impl->dynamicsBody;
  std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
  for(int row = 0; row < batch; row++) {
    impl->checkState(states[row]);
    const int action = actions[row];
    if(action < 0 || action >= (xiangqi ? Xiangqi::NUM_ACTIONS : area+(impl->gameName == "go")) ||
       (!xiangqi && action < area && !(*states[row]->mask)[action]))
      throw StringError("MuZero recurrent action is outside the static game domain");
    int squares[2] = {action,0}; // Go pass gives an all-zero action plane.
    if(xiangqi) {
      const auto& move = Xiangqi::MOVES[action];
      squares[0] = move[1]*impl->canvas+move[0];
      squares[1] = move[3]*impl->canvas+move[2];
    }
    impl->pack(states[row]->data(),body.input.data()+(size_t)row*body.desc.numInputChannels*area,squares);
    masks.push_back(states[row]->mask);
  }
  auto storage = std::make_shared<std::vector<float>>(batch*impl->stateSize());
  body.apply(impl->handle.get(),batch,nullptr,nullptr,storage->data());
  return impl->predict(batch,std::move(storage),masks);
}

std::vector<float> Evaluator::readStateForTesting(const State& state) const {
  impl->checkState(state);
  return std::vector<float>(state->data(),state->data()+impl->stateSize());
}
}
