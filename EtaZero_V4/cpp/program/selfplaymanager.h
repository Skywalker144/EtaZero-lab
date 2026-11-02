#ifndef PROGRAM_SELFPLAYMANAGER_H_
#define PROGRAM_SELFPLAYMANAGER_H_

#include <atomic>

#include "../core/threadsafequeue.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../dataio/trainingwrite.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/muzeroposition.h"

// Own one board evaluator or one complete MuZero service for selfplay, matches
// and protocols. Queue draining and manager reference counts stay in SelfplayManager.
class SelfplayModel {
  std::unique_ptr<NNEvaluator> board;
  std::unique_ptr<MuZero::Batcher> batcher;
  std::unique_ptr<MuZero::PositionEvaluator> position;
  std::string name, file;
public:
  explicit SelfplayModel(NNEvaluator* evaluator);
  SelfplayModel(const std::string& name, const std::string& directory,
                std::unique_ptr<MuZero::Batcher>, std::unique_ptr<MuZero::PositionEvaluator>);
  NNEvaluator* boardEvaluator() const { return board.get(); }
  MuZero::PositionEvaluator* muZeroEvaluator() const { return position.get(); }
  const std::string& getModelName() const { return name; }
  const std::string& getModelFileName() const { return file; }
  std::string getInternalModelName() const;
  int getModelVersion() const;
  int getNNXLen() const;
  int getNNYLen() const;
  bool getRequireExactNNLen() const;
  int getMaxBatchSize() const;
  int getNumGpus() const;
  std::vector<int> getGpuIdxByServerThread() const;
  void setMaxRowsToSendPerBatch(int rows);
  enabled_t getUsingFP16Mode() const;
  bool requiresSGFMetadata() const;
  Rules getSupportedRules(const Rules&, bool& supported) const;
  uint64_t numRowsProcessed() const;
  bool isNeuralNetEnabled() const { return board ? board->isNeuralNetEnabled() : batcher != nullptr; }
  uint64_t numBatchesProcessed() const;
  double averageProcessedBatchSize() const;
  uint64_t numCacheHits() const;
  Logger* getLogger() const;
  void clearStats();
  void clearCache();
};

class SelfplayManager {
 public:
  SelfplayManager(
    int maxDataQueueSize,
    Logger* logger,
    int64_t logGamesEvery,
    bool autoCleanupAllButLatestIfUnused
  );
  ~SelfplayManager();

  SelfplayManager(const SelfplayManager& other);
  SelfplayManager& operator=(const SelfplayManager& other);
  SelfplayManager(SelfplayManager&& other);
  SelfplayManager& operator=(SelfplayManager&& other);

  //All below functions are internally synchronized and thread-safe.

  //SelfplayManager takes responsibility for deleting the data writers and closing and deleting sgfOut.
  //loadModelNoDataWritingLoop is for the manual writing interface
  void loadModelAndStartDataWriting(
    SelfplayModel* nnEval,
    TrainingDataWriter* tdataWriter,
    std::ofstream* sgfOut
  );
  void loadModelNoDataWritingLoop(
    SelfplayModel* nnEval,
    TrainingDataWriter* tdataWriter,
    std::ofstream* sgfOut
  );

  //NN queries summed across all the models managed by this manager over all time.
  uint64_t getTotalNumRowsProcessed() const;

  //For all of the below, model names are simply from nnEval->getModelName().

  //Models that aren't cleaned up yet are in the order from earliest to latest
  std::vector<std::string> modelNames() const;
  std::string getLatestModelName() const;
  bool hasModel(const std::string& modelName) const;
  size_t numModels() const;

  //Returns NULL if acquire failed (such as if that model was scheduled to be cleaned up or already cleaned up,).
  //Must call release when done, and cease using the model after that.
  SelfplayModel* acquireModel(const std::string& modelName);
  SelfplayModel* acquireLatest();
  //Release a model either by name or by the nnEval object that was returned.
  void release(const std::string& modelName);
  void release(SelfplayModel* nnEval);

  //Clean up any currently-unused models if their last usage was older than this many seconds ago.
  void cleanupUnusedModelsOlderThan(double seconds);
  //Clear the evaluation caches of any models that are currently unused.
  void clearUnusedModelCaches();

  //====================================================================================
  //These should only be called by a thread that has currently acquired the model.

  //Increment a counter and maybe log some stats
  void countOneGameStarted(SelfplayModel* nnEval);

  //SelfplayManager takes responsibility for deleting the gameData once written.
  //Use these only if loadModelAndStartDataWriting was used to start the model.
  void enqueueDataToWrite(const std::string& modelName, FinishedGameData* gameData);
  void enqueueDataToWrite(SelfplayModel* nnEval, FinishedGameData* gameData);

  //Use these if loadModelNoDataWritingLoop was used to start the model.
  void withDataWriters(
    SelfplayModel* nnEval,
    const std::function<void(TrainingDataWriter* tdataWriter, std::ofstream* sgfOut)>& f
  );

  //====================================================================================

  //For internal use
  struct ModelData {
    std::string modelName;
    SelfplayModel* nnEval;
    int64_t gameStartedCount;
    // Counted at game-finish in the data write loop (lock-free), read cross-thread for logging.
    std::atomic<int64_t> gamesFinishedCount;
    std::atomic<int64_t> movesPlayedCount;
    double lastReleaseTime;
    bool hasDataWriteLoop;

    ThreadSafeQueue<FinishedGameData*> finishedGameQueue;
    int acquireCount;

    TrainingDataWriter* tdataWriter;
    std::ofstream* sgfOut;

    ModelData(
      const std::string& name, SelfplayModel* neval, int maxDataQueueSize,
      TrainingDataWriter* tdWriter, std::ofstream* sOut,
      double initialLastReleaseTime,
      bool hasDataWriteLoop
    );
    ~ModelData();
  };

 private:
  const int maxDataQueueSize;
  Logger* logger;
  const int64_t logGamesEvery;
  const bool autoCleanupAllButLatestIfUnused;

  const ClockTimer timer;

  mutable std::mutex managerMutex;
  std::vector<ModelData*> modelDatas;
  int numDataWriteLoopsActive;
  std::condition_variable dataWriteLoopsAreDone;

  uint64_t totalNumRowsProcessed;

  SelfplayModel* acquireModelAlreadyLocked(SelfplayManager::ModelData* foundData);
  void releaseAlreadyLocked(SelfplayManager::ModelData* foundData);
  void maybeAutoCleanupAlreadyLocked();
  void runDataWriteLoopImpl(ModelData* modelData);

 public:
  //For internal use
  void runDataWriteLoop(ModelData* modelData);

};

#endif //PROGRAM_SELFPLAYMANAGER_H_
