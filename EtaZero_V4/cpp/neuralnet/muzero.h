#ifndef NEURALNET_MUZERO_H_
#define NEURALNET_MUZERO_H_

#include "nninterface.h"
#include <array>
#include <memory>

namespace MuZero {
struct StateData;
// Immutable backend latent (device memory or CPU). Copies share storage and may outlive the evaluator.
// A state can only be consumed by the evaluator/model that produced it.
using State = std::shared_ptr<const StateData>;
// Stable while the state lives; only used to route recurrence to its owning worker.
const void* stateOwner(const State& state);

class ModelBundle {
  struct Impl;
  std::unique_ptr<Impl> impl;
  friend class Evaluator;
public:
  explicit ModelBundle(const std::string& directory);
  ~ModelBundle();
  const ModelDesc& representationDesc() const;
  const ModelDesc& predictionDesc() const;
  int posLen() const;
  int latentChannels() const;
  const std::string& game() const;
  bool hasTransformer() const;
};

struct Output {
  State state;
  // Value logits use the current latent player's perspective. Policy is [C,A]
  // in the root's fixed coordinates (including Xiangqi's 2086 joint actions),
  // source inference channels (ordinary, optimistic, optional Q). A includes
  // pass only for Go. Value is W/D/L/NR; aux is variance-time/value-error.
  // Source scale8 is already undone. No softmax, legality or NR mask is applied.
  std::vector<float> policy;
  std::array<float,4> value;
  std::array<float,2> auxiliary;
};

// Like the source ComputeHandle, create/use/destroy on one backend worker; calls must
// not overlap. State references may be released by other threads.
// Root inputs are FP32 NCHW/NC in one fixed orientation; recurrence never reads a
// successor board. The caller batches requests and owns root symmetry/legality.
class Evaluator {
  struct Impl;
  std::unique_ptr<Impl> impl;
public:
  Evaluator(const std::string& directory, int maxBatch, int device,
            enabled_t fp16, ConfigParser& config, Logger* logger = nullptr, bool inputsUseNHWC = true);
  Evaluator(std::shared_ptr<const ModelBundle> bundle, int maxBatch, int device,
            enabled_t fp16, ConfigParser& config, Logger* logger = nullptr, bool inputsUseNHWC = true);
  ~Evaluator();
  Evaluator(const Evaluator&) = delete;
  Evaluator& operator=(const Evaluator&) = delete;

  const ModelDesc& representationDesc() const;
  const ModelDesc& predictionDesc() const;
  int posLen() const;
  int latentChannels() const;
  const std::string& game() const;
  const void* identity() const;
  bool isUsingFP16() const;
  void warmup();
  std::vector<Output> initial(int batch, const std::vector<float>& spatial,
                             const std::vector<float>& global, const std::vector<float>& metadata = {});
  std::vector<Output> recurrent(const std::vector<State>& states, const std::vector<int>& actions);
  // Explicit diagnostic copy; production CUDA initial/recurrent never download latent.
  std::vector<float> readStateForTesting(const State& state) const;
};
}
#endif
