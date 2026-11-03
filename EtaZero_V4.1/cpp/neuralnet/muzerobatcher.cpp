#include "muzerobatcher.h"
#include "../core/multithread.h"
#include <deque>
#include <unordered_map>

namespace MuZero {
struct Batcher::Impl {
  struct Request {
    const std::vector<float> *spatial = nullptr, *global = nullptr, *metadata = nullptr;
    State parent;
    int action = 0;
    uint64_t order = 0;
    std::mutex mutex;
    std::condition_variable ready;
    bool done = false;
    std::exception_ptr error;
    Output output;
  };
  struct ClearRequest {
    Request& request;
    ~ClearRequest() {
      request.spatial = request.global = request.metadata = nullptr;
      request.parent.reset();
      request.error = nullptr;
      request.output = Output{};
    }
  };
  std::shared_ptr<const ModelBundle> model;
  const int maxBatch;
  const int numEffectiveDevices;
  const std::vector<int> gpuIdxByServerThread;
  const enabled_t usingFP16Mode;
  mutable std::mutex mutex;
  std::condition_variable changed, completed;
  std::vector<std::thread> threads;
  std::deque<Request*> initialQueue;
  std::vector<std::deque<Request*>> recurrentQueues;
  std::unordered_map<const void*,size_t> affinity;
  bool closing = false;
  std::exception_ptr error;
  size_t starting = 0, ongoing = 0;
  uint64_t nextOrder = 0;
  int maxRows;
  BatchStats counters;

  Impl(std::shared_ptr<const ModelBundle> model_, int maxBatch_, const std::vector<int>& devices,
       enabled_t fp16, ConfigParser& config, Logger* logger, bool inputsUseNHWC)
    : model(std::move(model_)), maxBatch(maxBatch_),
      numEffectiveDevices(NeuralNet::getNumEffectiveDevices(config,devices)), gpuIdxByServerThread(devices), usingFP16Mode(fp16), recurrentQueues(devices.size()),
      starting(devices.size()), maxRows(maxBatch_) {
    if(!model || maxBatch < 1 || devices.empty()) throw StringError("Invalid MuZero batcher configuration");
    counters.rowsByWorker.resize(devices.size());
    threads.reserve(devices.size());
    try {
      for(size_t worker = 0; worker < devices.size(); worker++)
        threads.emplace_back([this,&config,worker,device=devices[worker],fp16,logger,inputsUseNHWC]() { serve(worker,device,fp16,config,logger,inputsUseNHWC); });
      std::unique_lock<std::mutex> lock(mutex);
      changed.wait(lock,[&]() { return starting == 0 || error; });
      if(error) std::rethrow_exception(error);
    }
    catch(...) { stopAndJoin(); throw; }
  }
  ~Impl() { stopAndJoin(); }

  static Request& threadRequest() {
    thread_local Request request;
    return request;
  }
  static void complete(Request* request, Output output, std::exception_ptr failure) {
    std::lock_guard<std::mutex> lock(request->mutex);
    request->output = std::move(output);
    request->error = failure;
    request->done = true;
    request->ready.notify_all();
  }
  void fail(std::exception_ptr failure, const std::vector<Request*>& active) {
    std::vector<Request*> pending;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if(!error) error = failure;
      failure = error;
      closing = true;
      pending.assign(initialQueue.begin(),initialQueue.end());
      initialQueue.clear();
      for(auto& queue: recurrentQueues) {
        pending.insert(pending.end(),queue.begin(),queue.end());
        queue.clear();
      }
      ongoing -= pending.size()+active.size();
    }
    // Only this worker's active batch is completed here. Other workers may
    // still read their callers' tensors and must finish/synchronize first.
    for(auto* request: pending) complete(request,{},failure);
    for(auto* request: active) complete(request,{},failure);
    changed.notify_all();
    completed.notify_all();
  }
  void stopAndJoin() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      closing = true;
    }
    changed.notify_all();
    completed.notify_all();
    for(auto& thread: threads) if(thread.joinable()) thread.join();
  }
  Output submit(Request& request) {
    ClearRequest clear{request};
    request.done = false;
    bool wake;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if(error) std::rethrow_exception(error);
      if(closing) throw StringError("MuZero inference service is closing");
      size_t worker = 0;
      if(!request.spatial) {
        const auto found = affinity.find(stateOwner(request.parent));
        if(found == affinity.end()) throw StringError("MuZero state belongs to another inference service");
        worker = found->second;
      }
      auto& queue = request.spatial ? initialQueue : recurrentQueues[worker];
      request.order = nextOrder++;
      wake = queue.empty();
      queue.push_back(&request);
      ongoing++;
    }
    if(wake) changed.notify_all();
    std::unique_lock<std::mutex> lock(request.mutex);
    request.ready.wait(lock,[&]() { return request.done; });
    if(request.error) std::rethrow_exception(request.error);
    return std::move(request.output);
  }
  void serve(size_t worker, int device, enabled_t fp16, ConfigParser& config, Logger* logger, bool inputsUseNHWC) {
    bool announced = false;
    std::vector<Request*> batch;
    try {
      // Config values are immutable during startup; ConfigParser protects its
      // used-key bookkeeping. No config reference is read after startup ends.
      Evaluator evaluator(model,maxBatch,device,fp16,config,logger,inputsUseNHWC);
      evaluator.warmup();
      {
        std::lock_guard<std::mutex> lock(mutex);
        affinity.emplace(evaluator.identity(),worker);
        starting--;
        announced = true;
      }
      changed.notify_all();
      batch.reserve(maxBatch);
      std::vector<float> spatial, global, metadata;
      std::vector<State> states;
      std::vector<int> actions;
      while(true) {
        bool initial;
        {
          std::unique_lock<std::mutex> lock(mutex);
          changed.wait(lock,[&]() { return error || closing || !initialQueue.empty() || !recurrentQueues[worker].empty(); });
          if(error) break;
          auto& recurrence = recurrentQueues[worker];
          if(initialQueue.empty() && recurrence.empty()) {
            if(closing) break;
            continue;
          }
          initial = recurrence.empty() || (!initialQueue.empty() && initialQueue.front()->order < recurrence.front()->order);
          auto& queue = initial ? initialQueue : recurrence;
          // Source waitPopUpToN semantics: take available rows immediately. The
          // two kinds cannot share one forward, so select the oldest kind first.
          while(!queue.empty() && batch.size() < (size_t)maxRows) {
            batch.push_back(queue.front()); queue.pop_front();
          }
        }
        std::vector<Output> output;
        if(initial) {
          spatial.clear(); global.clear(); metadata.clear();
          for(const auto* request: batch) {
            spatial.insert(spatial.end(),request->spatial->begin(),request->spatial->end());
            global.insert(global.end(),request->global->begin(),request->global->end());
            metadata.insert(metadata.end(),request->metadata->begin(),request->metadata->end());
          }
          output = evaluator.initial(batch.size(),spatial,global,metadata);
        }
        else {
          states.clear(); actions.clear();
          for(const auto* request: batch) { states.push_back(request->parent); actions.push_back(request->action); }
          output = evaluator.recurrent(states,actions);
          states.clear();
        }
        if(output.size() != batch.size()) throw StringError("MuZero batch result count differs");
        std::exception_ptr failure;
        {
          std::lock_guard<std::mutex> lock(mutex);
          counters.rows += batch.size(); counters.batches++;
          if(initial) counters.initialRows += batch.size();
          counters.rowsByWorker[worker] += batch.size();
          failure = error;
        }
        for(size_t row = 0; row < batch.size(); row++)
          complete(batch[row],failure ? Output{} : std::move(output[row]),failure);
        {
          std::lock_guard<std::mutex> lock(mutex);
          ongoing -= batch.size();
        }
        batch.clear();
        completed.notify_all();
      }
    }
    catch(...) {
      fail(std::current_exception(),batch);
      if(!announced) {
        std::lock_guard<std::mutex> lock(mutex);
        starting--;
      }
      changed.notify_all();
    }
  }
};

Batcher::Batcher(std::shared_ptr<const ModelBundle> model, int maxBatch, const std::vector<int>& devices,
                 enabled_t fp16, ConfigParser& config, Logger* logger, bool inputsUseNHWC)
  : impl(std::make_unique<Impl>(std::move(model),maxBatch,devices,fp16,config,logger,inputsUseNHWC)) {}
Batcher::~Batcher() = default;
const ModelBundle& Batcher::model() const { return *impl->model; }
Output Batcher::initial(const std::vector<float>& spatial, const std::vector<float>& global, const std::vector<float>& metadata) {
  const auto& desc = impl->model->representationDesc();
  const int canvas = impl->model->posLen();
  if(spatial.size() != (size_t)desc.numInputChannels*canvas*canvas || global.size() != (size_t)desc.numInputGlobalChannels ||
     metadata.size() != (size_t)desc.numInputMetaChannels)
    throw StringError("MuZero initial observation dimensions differ");
  auto& request = Impl::threadRequest();
  request.spatial = &spatial; request.global = &global; request.metadata = &metadata;
  return impl->submit(request);
}
Output Batcher::recurrent(State state, int action) {
  auto& request = Impl::threadRequest();
  request.parent = std::move(state); request.action = action;
  return impl->submit(request);
}
void Batcher::setMaxRowsToSendPerBatch(int rows) {
  if(rows < 1 || rows > impl->maxBatch) throw StringError("MuZero batch cap must be in [1,maxBatch]");
  std::lock_guard<std::mutex> lock(impl->mutex);
  impl->maxRows = rows;
}
int Batcher::getMaxRowsToSendPerBatch() const { std::lock_guard<std::mutex> lock(impl->mutex); return impl->maxRows; }
int Batcher::getMaxBatchSize() const { return impl->maxBatch; }
int Batcher::getNumGpus() const { return impl->numEffectiveDevices; }
std::vector<int> Batcher::getGpuIdxByServerThread() const { return impl->gpuIdxByServerThread; }
enabled_t Batcher::getUsingFP16Mode() const { return impl->usingFP16Mode; }
BatchStats Batcher::stats() const { std::lock_guard<std::mutex> lock(impl->mutex); return impl->counters; }
void Batcher::clearStats() {
  std::lock_guard<std::mutex> lock(impl->mutex);
  impl->counters.rows = impl->counters.initialRows = impl->counters.batches = 0;
  std::fill(impl->counters.rowsByWorker.begin(),impl->counters.rowsByWorker.end(),0);
}
void Batcher::drain() {
  std::unique_lock<std::mutex> lock(impl->mutex);
  impl->completed.wait(lock,[&]() { return impl->error || impl->ongoing == 0; });
  if(impl->error) std::rethrow_exception(impl->error);
}
void Batcher::waitForNextNNEvalIfAny() {
  std::unique_lock<std::mutex> lock(impl->mutex);
  const auto batches = impl->counters.batches;
  impl->completed.wait(lock,[&]() { return impl->error || impl->closing || impl->ongoing == 0 || impl->counters.batches != batches; });
  if(impl->error) std::rethrow_exception(impl->error);
}
void Batcher::finish() {
  impl->stopAndJoin();
  std::lock_guard<std::mutex> lock(impl->mutex);
  if(impl->error) std::rethrow_exception(impl->error);
}
}
