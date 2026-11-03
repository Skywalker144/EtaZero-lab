// Included after the source OpenCL layers, Model and ComputeHandle. Learned
// computation and precision/tuning remain native; states stay on this device.
#include "muzerobundle.inc"

namespace MuZero {
using CLBuffer = WrappedWithDeleter<cl_mem,int,clReleaseMemObject>;

static const char* stateKernels = R"CLC(
// Source spatial buffers are NCHW with per-channel alignment padding. Public
// latent states are compact FP32 NCHW and retain their mask in the final plane.
__kernel void packState(__global const float* state, __global const int* actions,
                        __global realstore* input, int batch, int channels,
                        int area, int paddedArea, int planes) {
  int i = get_global_id(0), width = channels+1+planes;
  if(i >= batch*width*paddedArea) return;
  int point = i%paddedArea, channel = (i/paddedArea)%width, row = i/(paddedArea*width);
  float value = 0.0f;
  if(point < area) {
    if(channel <= channels)
      value = state[(row*(channels+1)+(channel == 0 ? channels : channel-1))*area+point];
    else
      value = actions[row*planes+channel-channels-1] == point;
  }
  STORE(input,i,value);
}

__kernel void normalizeState(__global const realstore* projection, __global const realstore* mask,
                             __global float* output, int channels, int area, int paddedArea) {
  int row = get_group_id(0), lane = get_local_id(0);
  __local float lows[64], highs[64];
  float lo = INFINITY, hi = -INFINITY;
  for(int i = lane; i < channels*area; i += 64) {
    int point = i%area;
    if(LOAD(mask,row*paddedArea+point) == 0) continue;
    float v = (float)LOAD(projection,(row*channels+i/area)*paddedArea+point);
    if(isnan(v)) lo = hi = v;
    else if(!isnan(lo)) { lo = fmin(lo,v); hi = fmax(hi,v); }
  }
  lows[lane] = lo; highs[lane] = hi;
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int stride = 32; stride > 0; stride /= 2) {
    if(lane < stride) {
      lows[lane] = isnan(lows[lane]) || isnan(lows[lane+stride]) ? NAN : fmin(lows[lane],lows[lane+stride]);
      highs[lane] = isnan(highs[lane]) || isnan(highs[lane+stride]) ? NAN : fmax(highs[lane],highs[lane+stride]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float span = highs[0]-lows[0], denominator = span < 1e-5f ? span+1e-5f : span;
  for(int i = lane; i < (channels+1)*area; i += 64) {
    int point = i%area;
    float m = (float)LOAD(mask,row*paddedArea+point), value;
    if(i >= channels*area) value = m;
    else value = m == 0 ? 0 : ((float)LOAD(projection,(row*channels+i/area)*paddedArea+point)-lows[0])/denominator;
    output[row*(channels+1)*area+i] = value;
  }
}
)CLC";

template<typename... Args>
static void launchStateKernel(ComputeHandleInternal* handle, cl_kernel kernel, const char* name, size_t count, const Args&... args) {
  int index = 0;
  auto set = [&](const auto& value) { CHECK_ERR(clSetKernelArg(kernel,index++,sizeof(value),&value)); };
  (set(args),...);
  const size_t local = 64, global = roundUpToMultiple(count,local);
  MAYBE_EVENT;
  const cl_int err = clEnqueueNDRangeKernel(handle->commandQueue,kernel,1,nullptr,&global,&local,0,nullptr,MAYBE_EVENTREF);
  CHECK_ERR(err);
  MAYBE_PROFILE(name);
  MAYBE_FREE_EVENT;
  (void)name;
}

struct StateData {
  std::shared_ptr<CLBuffer> storage;
  std::shared_ptr<const int> owner;
  std::shared_ptr<const std::vector<uint8_t>> mask;
  size_t offset;
};
const void* stateOwner(const State& state) { return state ? state->owner.get() : nullptr; }

struct OpenCLBody {
  const ModelDesc& desc;
  Trunk trunk;
  ConvLayer projection;
  CLBuffer input, global, metadata, trunkOutput, projected, mask, maskSum, workspace, workspace2;
  OpenCLBody(ComputeHandleInternal* handle, const ModelDesc& d, int canvas, int maxBatch)
    : desc(d), trunk(handle,&d.trunk,maxBatch,canvas,canvas,handle->usingFP16Storage),
      projection(handle,&d.latentProjection,canvas,canvas,handle->usingFP16Storage) {
    const size_t batchArea = (size_t)maxBatch*handle->paddedNNXYLen;
    const bool half = handle->usingFP16Storage;
    input = createReadWriteBuffer(handle,batchArea*d.numInputChannels,half);
    global = createReadWriteBuffer(handle,(size_t)maxBatch*d.numInputGlobalChannels,false);
    if(d.numInputMetaChannels > 0)
      metadata = createReadWriteBuffer(handle,(size_t)maxBatch*d.numInputMetaChannels,false);
    trunkOutput = createReadWriteBuffer(handle,batchArea*d.trunk.trunkNumChannels,half);
    projected = createReadWriteBuffer(handle,batchArea*d.numLatentChannels,half);
    mask = createReadWriteBuffer(handle,batchArea,half);
    maskSum = createReadWriteBuffer(handle,maxBatch,false);
    const auto sizes = ConvWorkspaceEltsNeeded::getMax(trunk.requiredConvWorkspaceElts(handle,maxBatch),
      projection.requiredConvWorkspaceElts(handle,maxBatch));
    workspace = createReadWriteBuffer(handle,sizes.size1,half);
    workspace2 = createReadWriteBuffer(handle,sizes.size2,half);
  }
  void apply(ComputeHandle* handle, int batch) {
    auto* h = handle->handle.get();
    // Each row's mask is exactly input channel zero, including alignment padding.
    const size_t bytes = h->paddedNNXYLen*(h->usingFP16Storage ? sizeof(half_t) : sizeof(float));
    for(int row = 0; row < batch; row++)
      CHECK_ERR(clEnqueueCopyBuffer(h->commandQueue,input,mask,row*desc.numInputChannels*bytes,row*bytes,bytes,0,nullptr,nullptr));
    computeMaskSums(h,mask,maskSum,batch,h->paddedNNXYLen);
    trunk.apply(h,handle->scratch.get(),batch,input,global,
      desc.numInputMetaChannels > 0 ? (cl_mem)metadata : nullptr,trunkOutput,mask,maskSum,workspace,workspace2);
    projection.apply(h,batch,trunkOutput,projected,workspace,workspace2);
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
  std::unique_ptr<OpenCLBody> representationBody, dynamicsBody;
  CLProgram program;
  CLKernel packKernel, normalizeKernel;
  CLBuffer gathered, actionBuffer;

  Impl(std::shared_ptr<const ModelBundle> bundle_, int maxBatch_, int device, enabled_t fp16,
       ConfigParser& config, Logger* logger, bool inputsUseNHWC)
    : maxBatch(maxBatch_), canvas(bundle_->posLen()), channels(bundle_->latentChannels()),
      actionPlanes(bundle_->game() == "xiangqi" ? 2 : 1), gameName(bundle_->game()), bundle(std::move(bundle_)),
      representation(bundle->impl->representation.get()), dynamics(bundle->impl->dynamics.get()), prediction(bundle->impl->prediction.get()) {
    if(maxBatch <= 0) throw StringError("MuZero maxBatch must be positive");
    context.reset(NeuralNet::createComputeContext({device},logger,canvas,canvas,"",fp16,prediction,config));
    handle.reset(NeuralNet::createComputeHandle(context.get(),prediction,logger,maxBatch,false,inputsUseNHWC,device,0));
    auto* h = handle->handle.get();
    if((int64_t)channels+1+actionPlanes > (((int64_t)1<<31)-1)/maxBatch/h->paddedNNXYLen)
      throw StringError("MuZero tensor exceeds the supported 2^31 entry bound");
    representationBody = std::make_unique<OpenCLBody>(h,representation->modelDesc,canvas,maxBatch);
    dynamicsBody = std::make_unique<OpenCLBody>(h,dynamics->modelDesc,canvas,maxBatch);
    program = compileProgram("MuZero state boundary",h->clContext,{h->deviceId},OpenCLKernels::common+stateKernels,h->usingFP16Storage ? OpenCLKernels::fp16StorageDefine : "");
    cl_int err;
    packKernel = clCreateKernel(program,"packState",&err); CHECK_ERR(err);
    normalizeKernel = clCreateKernel(program,"normalizeState",&err); CHECK_ERR(err);
    gathered = createReadWriteBuffer(h,maxBatch*stateSize(),false);
    actionBuffer = createReadWriteBuffer(h,(size_t)maxBatch*actionPlanes,false);
  }
  ~Impl() { if(handle) (void)clFinish(queue()); }
  cl_command_queue queue() const { return handle->handle->commandQueue; }
  size_t stateSize() const { return (size_t)(channels+1)*canvas*canvas; }
  void checkBatch(size_t batch) const {
    if(batch == 0 || batch > (size_t)maxBatch) throw StringError("MuZero batch is empty or exceeds maxBatch");
  }
  void checkState(const State& state) const {
    if(!state || state->owner != owner) throw StringError("MuZero state belongs to another evaluator/model");
  }
  void pack(cl_mem storage, cl_mem input, int batch, int planes) {
    const int padded = handle->handle->paddedNNXYLen;
    launchStateKernel(handle->handle.get(),packKernel,"MuZeroPack",(size_t)batch*(channels+1+planes)*padded,
      storage,(cl_mem)actionBuffer,input,batch,channels,canvas*canvas,padded,planes);
  }
  std::shared_ptr<CLBuffer> apply(OpenCLBody& body, int batch) {
    auto storage = std::make_shared<CLBuffer>(createReadWriteBuffer(handle->handle.get(),batch*stateSize(),false));
    body.apply(handle.get(),batch);
    launchStateKernel(handle->handle.get(),normalizeKernel,"MuZeroNormalize",(size_t)batch*64,(cl_mem)body.projected,(cl_mem)body.mask,(cl_mem)*storage,
      channels,canvas*canvas,handle->handle->paddedNNXYLen);
    return storage;
  }
  void upload(cl_mem destination, const std::vector<float>& values) {
    if(!values.empty()) CHECK_ERR(clEnqueueWriteBuffer(queue(),destination,CL_TRUE,0,values.size()*sizeof(float),values.data(),0,nullptr,nullptr));
  }
  void uploadSpatial(const std::vector<float>& values, int batch) {
    const int area = canvas*canvas, padded = handle->handle->paddedNNXYLen;
    const int totalChannels = batch*representation->modelDesc.numInputChannels;
    std::vector<float> aligned((size_t)totalChannels*padded,0.0f);
    for(int c = 0; c < totalChannels; c++) std::copy_n(values.data()+c*area,area,aligned.data()+c*padded);
    if(handle->handle->usingFP16Storage) {
      std::vector<half_t> half(aligned.size());
      for(size_t i = 0; i < aligned.size(); i++) half[i] = half_float::half_cast<half_t>(aligned[i]);
      CHECK_ERR(clEnqueueWriteBuffer(queue(),representationBody->input,CL_TRUE,0,half.size()*sizeof(half_t),half.data(),0,nullptr,nullptr));
    }
    else upload(representationBody->input,aligned);
  }
  std::vector<Output> predict(int batch, std::shared_ptr<CLBuffer> storage,
                             const std::vector<std::shared_ptr<const std::vector<uint8_t>>>& masks) {
    auto* h = handle->handle.get();
    auto& buffers = *handle->buffers;
    pack(*storage,buffers.input,batch,0);
    handle->model->apply(h,handle->scratch.get(),batch,buffers.input,buffers.inputGlobal,buffers.inputMeta,
      buffers.mask,buffers.maskSum,buffers.trunk,buffers.policyPass,buffers.policy,buffers.value,buffers.auxValue,
      buffers.convWorkspace,buffers.convWorkspace2);
    const auto& desc = prediction->modelDesc;
    const int area = canvas*canvas, padded = h->paddedNNXYLen;
    const int policies = desc.numPolicyChannels, spatialChannels = desc.policyHead.p2Conv.outChannels;
    const int actions = gameName == "xiangqi" ? Xiangqi::NUM_ACTIONS : area+(gameName == "go");
    std::vector<float> policy, pass, value, aux;
    blockingReadBuffer(queue(),buffers.policy,(size_t)batch*spatialChannels*padded,policy,h->usingFP16Storage);
    if(gameName == "go") blockingReadBuffer(queue(),buffers.policyPass,(size_t)batch*policies,pass);
    blockingReadBuffer(queue(),buffers.value,(size_t)batch*4,value);
    blockingReadBuffer(queue(),buffers.auxValue,(size_t)batch*2,aux);
    finishKernelProfiles(h);
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
            const int offset = (row*spatialChannels+channel*Xiangqi::NUM_DELTAS+index/area)*padded+index%area;
            result.policy[channel*actions+action] = policy[offset]*scale;
          }
        }
        else for(int point = 0; point < area; point++)
          result.policy[channel*actions+point] = (*masks[row])[point] ? policy[(row*policies+channel)*padded+point]*scale : -5000.0f;
        if(gameName == "go") result.policy[channel*actions+area] = pass[row*policies+channel]*scale;
      }
      for(int c = 0; c < 4; c++) result.value[c] = value[row*4+c]*scale;
      for(int c = 0; c < 2; c++) result.auxiliary[c] = aux[row*2+c]*scale;
    }
    return results;
  }
};

Evaluator::Evaluator(const std::string& directory, int maxBatch, int device, enabled_t fp16, ConfigParser& cfg, Logger* logger, bool inputsUseNHWC)
  : Evaluator(std::make_shared<ModelBundle>(directory),maxBatch,device,fp16,cfg,logger,inputsUseNHWC) {}
Evaluator::Evaluator(std::shared_ptr<const ModelBundle> bundle, int maxBatch, int device, enabled_t fp16, ConfigParser& cfg, Logger* logger, bool inputsUseNHWC) {
  if(!bundle) throw StringError("MuZero model bundle must not be null");
  impl = std::make_unique<Impl>(std::move(bundle),maxBatch,device,fp16,cfg,logger,inputsUseNHWC);
}
Evaluator::~Evaluator() = default;
const ModelDesc& Evaluator::representationDesc() const { return impl->representation->modelDesc; }
const ModelDesc& Evaluator::predictionDesc() const { return impl->prediction->modelDesc; }
const void* Evaluator::identity() const { return impl->owner.get(); }
bool Evaluator::isUsingFP16() const { return impl->handle->handle->usingFP16Storage; }
int Evaluator::posLen() const { return impl->canvas; }
int Evaluator::latentChannels() const { return impl->channels; }
const std::string& Evaluator::game() const { return impl->gameName; }

std::vector<Output> Evaluator::initial(int batch, const std::vector<float>& spatial,
                                    const std::vector<float>& global, const std::vector<float>& metadata) {
  impl->checkBatch(batch);
  const auto& desc = representationDesc();
  const int area = posLen()*posLen();
  if(spatial.size() != (size_t)batch*desc.numInputChannels*area ||
     global.size() != (size_t)batch*desc.numInputGlobalChannels || metadata.size() != (size_t)batch*desc.numInputMetaChannels)
    throw StringError("MuZero initial observation dimensions differ");
  std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
  for(int row = 0; row < batch; row++) {
    auto mask = std::make_shared<std::vector<uint8_t>>(area);
    bool nonempty = false;
    for(int point = 0; point < area; point++) {
      const float v = spatial[(size_t)row*desc.numInputChannels*area+point];
      if(v != 0 && v != 1) throw StringError("MuZero observation mask must be binary");
      if(game() == "xiangqi" && (v != 0) != (point%posLen() < Xiangqi::WIDTH && point/posLen() < Xiangqi::HEIGHT))
        throw StringError("Xiangqi MuZero observation mask must cover exactly the 9 x 10 board");
      (*mask)[point] = v != 0; nonempty |= v != 0;
    }
    if(!nonempty) throw StringError("MuZero observation mask must be nonempty");
    masks.push_back(std::move(mask));
  }
  try {
    impl->uploadSpatial(spatial,batch);
    impl->upload(impl->representationBody->global,global);
    if(!metadata.empty()) impl->upload(impl->representationBody->metadata,metadata);
    return impl->predict(batch,impl->apply(*impl->representationBody,batch),masks);
  }
  catch(...) { (void)clFinish(impl->queue()); throw; }
}

std::vector<Output> Evaluator::recurrent(const std::vector<State>& states, const std::vector<int>& actions) {
  impl->checkBatch(states.size());
  if(actions.size() != states.size()) throw StringError("MuZero recurrent action count differs");
  const int batch = states.size(), area = posLen()*posLen();
  std::vector<std::shared_ptr<const std::vector<uint8_t>>> masks;
  std::vector<int> squares((size_t)batch*impl->actionPlanes);
  for(int row = 0; row < batch; row++) {
    impl->checkState(states[row]);
    const int action = actions[row];
    if(action < 0 || action >= (game() == "xiangqi" ? Xiangqi::NUM_ACTIONS : area+(game() == "go")) ||
       (game() != "xiangqi" && action < area && !(*states[row]->mask)[action]))
      throw StringError("MuZero recurrent action is outside the static game domain");
    masks.push_back(states[row]->mask);
    if(game() == "xiangqi") {
      const auto& move = Xiangqi::MOVES[action];
      squares[2*row] = move[1]*posLen()+move[0]; squares[2*row+1] = move[3]*posLen()+move[2];
    }
    else squares[row] = action;
  }
  try {
    // OpenCL 1.2 buffers cannot be passed as a device pointer array. Gather only
    // on the device; parents from earlier batches remain immutable and retained.
    for(int row = 0; row < batch; row++)
      CHECK_ERR(clEnqueueCopyBuffer(impl->queue(),*states[row]->storage,impl->gathered,
        states[row]->offset*sizeof(float),row*impl->stateSize()*sizeof(float),impl->stateSize()*sizeof(float),0,nullptr,nullptr));
    CHECK_ERR(clEnqueueWriteBuffer(impl->queue(),impl->actionBuffer,CL_TRUE,0,squares.size()*sizeof(int),squares.data(),0,nullptr,nullptr));
    impl->pack(impl->gathered,impl->dynamicsBody->input,batch,impl->actionPlanes);
    return impl->predict(batch,impl->apply(*impl->dynamicsBody,batch),masks);
  }
  catch(...) { (void)clFinish(impl->queue()); throw; }
}

// Source OpenCL programs and tuning are ready at handle creation; there is no
// per-batch CUDA graph compilation to prepare here.
void Evaluator::warmup() {}
std::vector<float> Evaluator::readStateForTesting(const State& state) const {
  impl->checkState(state);
  std::vector<float> values(impl->stateSize());
  CHECK_ERR(clEnqueueReadBuffer(impl->queue(),*state->storage,CL_TRUE,state->offset*sizeof(float),
    values.size()*sizeof(float),values.data(),0,nullptr,nullptr));
  return values;
}
}
