#ifndef DATAIO_MUZEROWRITE_H_
#define DATAIO_MUZEROWRITE_H_

#include "../core/hash.h"
#include "numpywrite.h"
#include <memory>

struct TrainingWriteBuffers;

// Complete targets for one real trajectory, including its final observed state.
// Sampled roots remain ordinary TrainingWriteBuffers rows. Recurrent training
// needs targets and actions, never future network inputs.
struct MuZeroTrajectory {
  Hash128 gameHash;
  bool finished;
  // Canvas actions or Xiangqi joint actions in each row's player perspective;
  // shuffle converts continuations to the sampled root's coordinates.
  std::vector<int32_t> actions; // -1 at the final state.
  std::vector<float> weights; // Unrounded source sampling weights; 1 at the final state.
  std::vector<int16_t> policy;
  std::vector<float> global;
  std::vector<int8_t> future;
  std::vector<int16_t> q;

  void append(const TrainingWriteBuffers& row, int action, float weight);
};

// A shard may contain a subset of a game's sampled roots, but each referenced
// game is included in full. This preserves continuation across file boundaries.
class MuZeroWriteBuffers {
 public:
  void addRoot(const std::shared_ptr<const MuZeroTrajectory>& trajectory, int turn);
  void clear();
  void writeToZipFile(ZipFile& zip, int numRoots, int xLen, int yLen, int policySize);

 private:
  std::vector<std::shared_ptr<const MuZeroTrajectory>> games;
  std::vector<int64_t> roots; // [N,2]: local game index, turn (-1 for side positions).
  // Source ZipFile borrows buffers until close(), after this writer returns.
  std::vector<std::shared_ptr<void>> numpyBuffers;
};

#endif
