#ifndef NEURALNET_MUZEROPOSITION_H_
#define NEURALNET_MUZEROPOSITION_H_

#include "muzerobatcher.h"
#include "nneval.h"

namespace MuZero {
// No real successor board or history is carried into the latent tree.
struct RootOrientation {
  int symmetry, xSize, ySize;
  bool suppressNoResult;
  Player rootPlayer; // Xiangqi coordinates retain the initial player's rotation.
};

struct Evaluation final : NNOutput {
  State state;
  std::shared_ptr<const RootOrientation> orientation;
  Player nextPlayer;
};

// One reusable buffer per calling/search thread, like source NNResultBuf.
struct PositionBuffer {
  std::vector<float> spatial, global, metadata, transformed;
  std::unique_ptr<bool[]> jointLegal; // Allocated only for Xiangqi, reused across calls.
  bool errorLogLockout = false;
  std::unique_ptr<Rand> symmetryRand;
  std::unique_ptr<Rand> evaluationRand;
};

// Batcher must outlive this adapter and all its clients. The root cache uses
// source NNCacheTable and its hashes; entries retain their latent orientation.
class PositionEvaluator {
  Batcher* batcher;
  std::shared_ptr<const ModelBundle> randomModel;
  const ModelBundle& model;
  const bool requireExactNNLen;
  const std::string symmetrySeed;
  std::atomic<bool> randomize;
  std::atomic<int> defaultSymmetry;
  std::unique_ptr<NNCacheTable> cache;
  std::atomic<uint64_t> cacheHits{0};
  Logger* logger;
  ModelPostProcessParams postprocess;
  mutable std::mutex randomSeedMutex;
  mutable Rand randomSeed;
  Output randomOutput(PositionBuffer&) const;

  std::shared_ptr<Evaluation> finish(Output output, Player nextPlayer,
    std::shared_ptr<const RootOrientation> orientation, const bool* legal,
    const MiscNNInputParams& params, PositionBuffer& buffer) const;
public:
  PositionEvaluator(Batcher&, bool requireExactNNLen, bool randomize, int defaultSymmetry,
                    int cacheSizePowerOfTwo, int mutexPoolSizePowerOfTwo, Logger* logger = nullptr, const std::string& symmetrySeed = "");
  PositionEvaluator(std::shared_ptr<const ModelBundle>, bool requireExactNNLen, bool randomize, int defaultSymmetry,
                    int cacheSizePowerOfTwo, int mutexPoolSizePowerOfTwo, Logger*, const std::string& seed);
  std::shared_ptr<Evaluation> initial(const Board&, const BoardHistory&, Player,
    const SGFMetadata*, const MiscNNInputParams&, PositionBuffer&, Rand&, bool skipCache);
  // Action is an absolute joint action (Xiangqi) or a policy index in the original
  // root canvas orientation, not Loc. Only the static game vocabulary applies,
  // regardless of real occupancy or termination.
  // Recurrent input params only supply policy temperature/optimism; observation
  // features and orientation are fixed by the retained root state.
  std::shared_ptr<Evaluation> recurrent(const Evaluation&, int action,
    const MiscNNInputParams&, PositionBuffer&);
  std::shared_ptr<Evaluation> averageMultipleSymmetries(const Board&, const BoardHistory&, Player,
    const SGFMetadata*, const MiscNNInputParams&, PositionBuffer&, Rand&, int numSymmetries);
  void clearCache();
  uint64_t numCacheHits() const;
  void clearStats() { cacheHits.store(0); }
  Logger* getLogger() const { return logger; }
  void setDoRandomize(bool enabled);
  void setDefaultSymmetry(int symmetry);
  const ModelBundle& getModel() const { return model; }
  bool getRequireExactNNLen() const { return requireExactNNLen; }
  void waitForNextNNEvalIfAny() { if(batcher) batcher->waitForNextNNEvalIfAny(); }
};
}
#endif
