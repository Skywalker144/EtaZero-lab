#ifndef NEURALNET_MUZEROBATCHER_H_
#define NEURALNET_MUZEROBATCHER_H_

#include "muzero.h"

namespace MuZero {
struct BatchStats {
  uint64_t rows = 0;
  uint64_t initialRows = 0;
  uint64_t batches = 0;
  std::vector<uint64_t> rowsByWorker;
};

// Raw tensor service. Position featurization, root symmetry, caching and search
// postprocessing belong to the position evaluator, not this request queue.
class Batcher {
  struct Impl;
  std::unique_ptr<Impl> impl;
public:
  Batcher(std::shared_ptr<const ModelBundle> model, int maxBatch, const std::vector<int>& devices,
          enabled_t fp16, ConfigParser& config, Logger* logger = nullptr, bool inputsUseNHWC = true);
  ~Batcher();
  Batcher(const Batcher&) = delete;
  Batcher& operator=(const Batcher&) = delete;
  const ModelBundle& model() const;
  int getMaxBatchSize() const;
  int getNumGpus() const;
  std::vector<int> getGpuIdxByServerThread() const;
  enabled_t getUsingFP16Mode() const; // Source query_models reports the configured mode, including auto.

  // Thread-safe synchronous calls, like source NNEvaluator::evaluate. A waiting
  // caller keeps its input alive; a reusable per-thread result buffer is internal.
  Output initial(const std::vector<float>& spatial, const std::vector<float>& global,
                 const std::vector<float>& metadata = {});
  Output recurrent(State state, int action);
  void setMaxRowsToSendPerBatch(int rows);
  int getMaxRowsToSendPerBatch() const;
  BatchStats stats() const;
  void clearStats();
  void waitForNextNNEvalIfAny();
  void drain();
  // Call from one owner thread. Stop admission, drain accepted work, then join.
  // Clients may race with finish(), but must be joined before object destruction.
  // Failure is persistent: all pending callers wake, no retry or fallback occurs.
  void finish();
};
}
#endif
