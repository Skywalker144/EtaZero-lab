#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "nnpostprocess.h"
#include "../core/test.h"
#include "../game/xiangqi.h"

#include <algorithm>
#include <chrono>

using namespace std;

//-------------------------------------------------------------------------------------

NNResultBuf::NNResultBuf()
  : clientWaitingForResult(),
    resultMutex(),
    hasResult(false),
    boardXSizeForServer(0),
    boardYSizeForServer(0),
    rowSpatialBuf(),
    rowGlobalBuf(),
    rowMetaBuf(),
    hasRowMeta(false),
    result(nullptr),
    errorLogLockout(false),
    // If no symmetry is specified, it will use default or random based on config.
    symmetry(NNInputs::SYMMETRY_NOTSPECIFIED),
    policyOptimism(0.0)
{}

NNResultBuf::~NNResultBuf() {
}

//-------------------------------------------------------------------------------------

NNServerBuf::NNServerBuf(const NNEvaluator& nnEval, const LoadedModel* model)
  :inputBuffers(NULL)
{
  int maxBatchSize = nnEval.getMaxBatchSize();
  if(model != NULL)
    inputBuffers = NeuralNet::createInputBuffers(model,maxBatchSize,nnEval.getNNXLen(),nnEval.getNNYLen());
}

NNServerBuf::~NNServerBuf() {
  if(inputBuffers != NULL)
    NeuralNet::freeInputBuffers(inputBuffers);
  inputBuffers = NULL;
}

//-------------------------------------------------------------------------------------

NNEvaluator::NNEvaluator(
  const string& mName,
  const string& mFileName,
  const string& expectedSha256,
  Logger* lg,
  int maxBatchSz,
  int xLen,
  int yLen,
  bool rExactNNLen,
  bool iUseNHWC,
  int nnCacheSizePowerOfTwo,
  int nnMutexPoolSizePowerofTwo,
  bool skipNeuralNet,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  int numThr,
  const vector<int>& gpuIdxByServerThr,
  const string& rSeed,
  bool doRandomize,
  int defaultSymmetry,
  ConfigParser& cfg
)
  :modelName(mName),
   modelFileName(mFileName),
   nnXLen(xLen),
   nnYLen(yLen),
   requireExactNNLen(rExactNNLen),
   policySize(NNPos::getPolicySize(xLen,yLen)),
   inputsUseNHWC(iUseNHWC),
   usingFP16Mode(useFP16Mode),
   numThreads(numThr),
   gpuIdxByServerThread(gpuIdxByServerThr),
   randSeed(rSeed),
   debugSkipNeuralNet(skipNeuralNet),
   computeContext(NULL),
   loadedModel(NULL),
   nnCacheTable(NULL),
   logger(lg),
   internalModelName(),
   modelVersion(-1),
   inputsVersion(-1),
   numInputMetaChannels(0),
   postProcessParams(),
   numServerThreadsEverSpawned(0),
   serverThreads(),
   maxBatchSize(maxBatchSz),
   numEffectiveDevices(NeuralNet::getNumEffectiveDevices(cfg, gpuIdxByServerThr)),
   m_numRowsProcessed(0),
   m_numBatchesProcessed(0),
   m_numCacheHits(0),
   bufferMutex(),
   isKilled(false),
   numServerThreadsStartingUp(0),
   mainThreadWaitingForSpawn(),
   numOngoingEvals(0),
   numWaitingEvals(0),
   numEvalsToAwaken(0),
   waitingForFinish(),
   currentDoRandomize(doRandomize),
   currentDefaultSymmetry(defaultSymmetry),
   maxRowsToSendPerBatch(maxBatchSz),
   queryQueue()
{
  if(nnXLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(nnYLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(maxBatchSize <= 0)
    throw StringError("maxBatchSize is negative: " + Global::intToString(maxBatchSize));
  if(gpuIdxByServerThread.size() != numThreads)
    throw StringError("gpuIdxByServerThread.size() != numThreads");

  if(logger != NULL) {
    logger->write(
      "Initializing neural net buffer to be size " +
      Global::intToString(nnXLen) + " * " + Global::intToString(nnYLen) +
      (requireExactNNLen ? " exactly" : " allowing smaller boards")
    );
  }

  if(nnCacheSizePowerOfTwo >= 0)
    nnCacheTable = new NNCacheTable(nnCacheSizePowerOfTwo, nnMutexPoolSizePowerofTwo);

  if(!debugSkipNeuralNet) {
    vector<int> gpuIdxs = gpuIdxByServerThread;
    std::sort(gpuIdxs.begin(), gpuIdxs.end());
    auto last = std::unique(gpuIdxs.begin(), gpuIdxs.end());
    gpuIdxs.erase(last,gpuIdxs.end());
    loadedModel = NeuralNet::loadModelFile(modelFileName,expectedSha256);
    const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
    if(desc.modelVersion == 19 && (nnXLen < Xiangqi::WIDTH || nnYLen < Xiangqi::HEIGHT ||
       (requireExactNNLen && (nnXLen != Xiangqi::WIDTH || nnYLen != Xiangqi::HEIGHT)))) {
      NeuralNet::freeLoadedModel(loadedModel);
      loadedModel = nullptr;
      delete nnCacheTable;
      nnCacheTable = nullptr;
      throw StringError("Xiangqi needs a 9 x 10 board and a canvas containing it; padded canvases cannot requireExactNNLen");
    }
    internalModelName = desc.name;
    modelVersion = desc.modelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
    numInputMetaChannels = desc.numInputMetaChannels;
    computeContext = NeuralNet::createComputeContext(
      gpuIdxs,logger,nnXLen,nnYLen,
      homeDataDirOverride,
      usingFP16Mode,loadedModel,cfg
    );
    // Snapshot postProcessParams only after createComputeContext: backends may apply
    // config-dependent transforms to the model desc there (e.g. the ONNX backend's
    // scale8 workaround multiplies outputScaleMultiplier by 8).
    postProcessParams = desc.postProcessParams;
  }
  else {
    internalModelName = "random";
    modelVersion = NNModelVersion::defaultModelVersion;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  }

  // Reserve a decent amount above the batch size so that allocation is unlikely.
  queryQueue.reserve(maxBatchSize * 4 * gpuIdxByServerThread.size());
  // Starts readonly. Becomes writable once we spawn server threads
  queryQueue.setReadOnly();
}

NNEvaluator::~NNEvaluator() {
  killServerThreads();

  if(computeContext != NULL)
    NeuralNet::freeComputeContext(computeContext);
  computeContext = NULL;

  if(loadedModel != NULL)
    NeuralNet::freeLoadedModel(loadedModel);
  loadedModel = NULL;

  delete nnCacheTable;
}

string NNEvaluator::getModelName() const {
  return modelName;
}
string NNEvaluator::getModelFileName() const {
  return modelFileName;
}
string NNEvaluator::getInternalModelName() const {
  return internalModelName;
}

static bool tryAbbreviateStepString(const string& input, string& buf) {
  size_t i = 0;
  while(i < input.length() && !Global::isDigit(input[i]))
    i++;
  if(i > 1)
    return false;

  string prefix = input.substr(0, i);
  int64_t number;
  bool suc = Global::tryStringToInt64(input.substr(i),number);
  if(!suc)
    return false;

  if(number >= 10000000000LL)
    buf = prefix + std::to_string(number / 1000000000LL) + "G";
  if(number >= 10000000)
    buf = prefix + std::to_string(number / 1000000) + "M";
  else if(number >= 10000)
    buf = prefix + std::to_string(number / 1000) + "K";
  else
    buf = input;
  return true;
}

string NNEvaluator::getAbbrevInternalModelName() const {
  string name = getInternalModelName();
  std::vector<string> pieces = Global::split(name,'-');
  std::vector<string> newPieces;
  for(const string& piece: pieces) {
    string buf;
    if(piece == "kata1") {
      // skip
    }
    else if(piece.size() > 1 && piece[0] == 's' && tryAbbreviateStepString(piece,buf)) {
      newPieces.push_back(buf);
    }
    else if(piece.size() > 1 && piece[0] == 'd' && tryAbbreviateStepString(piece,buf)) {
      // skip
    }
    else {
      newPieces.push_back(piece);
    }
  }
  return Global::concat(newPieces,"-");
}

Logger* NNEvaluator::getLogger() {
  return logger;
}
bool NNEvaluator::isNeuralNetLess() const {
  return debugSkipNeuralNet;
}
int NNEvaluator::getMaxBatchSize() const {
  return maxBatchSize;
}
int NNEvaluator::getMaxRowsToSendPerBatch() const {
  return maxRowsToSendPerBatch.load(std::memory_order_acquire);
}
void NNEvaluator::setMaxRowsToSendPerBatch(int maxRows) {
  if(maxRows <= 0 || maxRows > maxBatchSize)
    throw StringError("Invalid setting for max rows to send per batch");
  maxRowsToSendPerBatch.store(maxRows,std::memory_order_release);
}
bool NNEvaluator::requiresSGFMetadata() const {
  return numInputMetaChannels > 0;
}

int NNEvaluator::getNumGpus() const {
  return numEffectiveDevices;
}
int NNEvaluator::getNumServerThreads() const {
  return (int)gpuIdxByServerThread.size();
}
const std::vector<int>& NNEvaluator::getGpuIdxByServerThread() const {
  return gpuIdxByServerThread;
}
std::set<int> NNEvaluator::getGpuIdxs() const {
  std::set<int> gpuIdxs;
#ifdef USE_EIGEN_BACKEND
  gpuIdxs.insert(0);
#else
  for(int i = 0; i<gpuIdxByServerThread.size(); i++) {
    gpuIdxs.insert(gpuIdxByServerThread[i]);
  }
#endif
  return gpuIdxs;
}

int NNEvaluator::getNNXLen() const {
  return nnXLen;
}
int NNEvaluator::getNNYLen() const {
  return nnYLen;
}
bool NNEvaluator::getRequireExactNNLen() const {
  return requireExactNNLen;
}
int NNEvaluator::getModelVersion() const {
  return modelVersion;
}
double NNEvaluator::getTrunkSpatialConvDepth() const {
  return NeuralNet::getModelDesc(loadedModel).getTrunkSpatialConvDepth();
}

int64_t NNEvaluator::getNumModelParameters() const {
  return NeuralNet::getModelDesc(loadedModel).getNumParameters();
}

bool NNEvaluator::modelHasAnyTransformerBlocks() const {
  return NeuralNet::getModelDesc(loadedModel).hasAnyTransformerBlocks();
}

bool NNEvaluator::modelHasAnyNestedBottleneckBlocks() const {
  return NeuralNet::getModelDesc(loadedModel).hasAnyNestedBottleneckBlocks();
}

enabled_t NNEvaluator::getUsingFP16Mode() const {
  return usingFP16Mode;
}

bool NNEvaluator::supportsShorttermError() const {
  return modelVersion >= 9;
}

bool NNEvaluator::modelPreferPassAliveUnderSuicideRules() const {
  if(loadedModel == NULL)
    return false;
  return NeuralNet::getModelDesc(loadedModel).preferPassAliveUnderSuicideRules;
}

bool NNEvaluator::modelPreferExcludeTerritoryAdjacentToAtari() const {
  if(loadedModel == NULL)
    return false;
  return NeuralNet::getModelDesc(loadedModel).preferExcludeTerritoryAdjacentToAtari;
}

bool NNEvaluator::getDoRandomize() const {
  return currentDoRandomize.load(std::memory_order_acquire);
}
int NNEvaluator::getDefaultSymmetry() const {
  return currentDefaultSymmetry.load(std::memory_order_acquire);
}
void NNEvaluator::setDoRandomize(bool b) {
  currentDoRandomize.store(b, std::memory_order_release);
}
void NNEvaluator::setDefaultSymmetry(int s) {
  if(inputsVersion == 9 && s != 0 && s != 2)
    throw StringError("Xiangqi only supports identity and horizontal reflection");
  currentDefaultSymmetry.store(s, std::memory_order_release);
}

Rules NNEvaluator::getSupportedRules(const Rules& desiredRules, bool& supported) const {
  if(loadedModel == NULL) {
    supported = true;
    return desiredRules;
  }
  return NeuralNet::getModelDesc(loadedModel).getSupportedRules(desiredRules, supported);
}

uint64_t NNEvaluator::numRowsProcessed() const {
  return m_numRowsProcessed.load(std::memory_order_relaxed);
}
uint64_t NNEvaluator::numBatchesProcessed() const {
  return m_numBatchesProcessed.load(std::memory_order_relaxed);
}
double NNEvaluator::averageProcessedBatchSize() const {
  return (double)numRowsProcessed() / (double)numBatchesProcessed();
}
uint64_t NNEvaluator::numCacheHits() const {
  return m_numCacheHits.load(std::memory_order_relaxed);
}

void NNEvaluator::clearStats() {
  m_numRowsProcessed.store(0);
  m_numBatchesProcessed.store(0);
  m_numCacheHits.store(0);
}

void NNEvaluator::clearCache() {
  if(nnCacheTable != NULL)
    nnCacheTable->clear();
}


bool NNEvaluator::isAnyThreadUsingFP16() const {
  lock_guard<std::mutex> lock(bufferMutex);
  for(const int& isUsingFP16: serverThreadsIsUsingFP16) {
    if(isUsingFP16)
      return true;
  }
  return false;
}

static void serveEvals(
  string randSeedThisThread,
  NNEvaluator* nnEval,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  Rand rand(randSeedThisThread);

  // We're in big trouble if this raises an exception (e.g. out of GPU memory) and causes possibly
  // the only nnEval thread to die, so let the exception escape and terminate the process. But log
  // and print the error first, since on some platforms an exception escaping a thread aborts
  // without printing anything. cerr is used unconditionally rather than only when there is no
  // logger, since the logger may be temporarily disabled (e.g. the benchmark suppresses logging
  // while respawning server threads).
  try {
    nnEval->serve(rand,gpuIdxForThisThread,serverThreadIdx);
  }
  catch(const std::exception& e) {
    Logger* logger = nnEval->getLogger();
    if(logger != NULL)
      logger->write(string("ERROR: NN server thread failed: ") + e.what());
    cerr << (string("ERROR: NN server thread failed: ") + e.what()) << endl;
    throw;
  }
  catch(...) {
    Logger* logger = nnEval->getLogger();
    if(logger != NULL)
      logger->write("ERROR: NN server thread failed with an exception that is not a std::exception");
    cerr << "ERROR: NN server thread failed with an exception that is not a std::exception" << endl;
    throw;
  }
}

void NNEvaluator::setNumThreads(const vector<int>& gpuIdxByServerThr) {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::setNumThreads called when threads were already running!");
  numThreads = (int)gpuIdxByServerThr.size();
  gpuIdxByServerThread = gpuIdxByServerThr;
}

void NNEvaluator::spawnServerThreads() {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::spawnServerThreads called when threads were already running!");

  {
    lock_guard<std::mutex> lock(bufferMutex);
    serverThreadsIsUsingFP16.resize(numThreads,0);
  }

  queryQueue.unsetReadOnly();

  numServerThreadsStartingUp = numThreads;
  for(int i = 0; i<numThreads; i++) {
    int gpuIdxForThisThread = gpuIdxByServerThread[i];
    string randSeedThisThread = randSeed + ":NNEvalServerThread:" + Global::intToString(numServerThreadsEverSpawned);
    numServerThreadsEverSpawned++;
    std::thread* thread = new std::thread(
      &serveEvals,randSeedThisThread,this,gpuIdxForThisThread,i
    );
    serverThreads.push_back(thread);
  }

  unique_lock<std::mutex> lock(bufferMutex);
  while(numServerThreadsStartingUp > 0)
    mainThreadWaitingForSpawn.wait(lock);
}

void NNEvaluator::killServerThreads() {
  unique_lock<std::mutex> lock(bufferMutex);
  isKilled = true;
  lock.unlock();
  queryQueue.setReadOnly();

  waitingForFinish.notify_all();

  for(size_t i = 0; i<serverThreads.size(); i++)
    serverThreads[i]->join();
  for(size_t i = 0; i<serverThreads.size(); i++)
    delete serverThreads[i];
  serverThreads.clear();
  serverThreadsIsUsingFP16.clear();

  // Can unset now that threads are dead
  isKilled = false;

  testAssert(numOngoingEvals == 0);
  testAssert(numWaitingEvals == 0);
  testAssert(numEvalsToAwaken == 0);
}

void NNEvaluator::fillRowBufs(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf
) const {
  const int rowSpatialLen = NNModelVersion::getNumSpatialFeatures(modelVersion) * nnXLen * nnYLen;
  if(buf.rowSpatialBuf.size() < rowSpatialLen)
    buf.rowSpatialBuf.resize(rowSpatialLen);
  const int rowGlobalLen = NNModelVersion::getNumGlobalFeatures(modelVersion);
  if(buf.rowGlobalBuf.size() < rowGlobalLen)
    buf.rowGlobalBuf.resize(rowGlobalLen);
  const int rowMetaLen = numInputMetaChannels;
  if(buf.rowMetaBuf.size() < rowMetaLen)
    buf.rowMetaBuf.resize(rowMetaLen);

  NNInputs::fillRow(inputsVersion,board,history,nextPlayer,nnInputParams,nnXLen,nnYLen,
                    inputsUseNHWC,buf.rowSpatialBuf.data(),buf.rowGlobalBuf.data());

  if(rowMetaLen > 0) {
    if(sgfMeta == NULL)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not provided");
    if(!sgfMeta->initialized)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
    SGFMetadata::fillMetadataRow(
      sgfMeta,
      buf.rowMetaBuf.data(),
      nextPlayer,
      board.x_size*board.y_size
    );
    buf.hasRowMeta = true;
  }
  else {
    buf.hasRowMeta = false;
  }
}

void NNEvaluator::fillRowBufs(const Xiangqi::State& state, const MiscNNInputParams& params, NNResultBuf& buf) const {
  buf.rowSpatialBuf.resize(Xiangqi::SPATIAL_FEATURES*nnXLen*nnYLen);
  buf.rowGlobalBuf.resize(Xiangqi::GLOBAL_FEATURES);
  state.fillFeatures(state.nextPlayer(),params.playoutDoublingAdvantage,false,nnXLen,nnYLen,
                     inputsUseNHWC,buf.rowSpatialBuf.data(),buf.rowGlobalBuf.data(),params.maxHistory);
  buf.hasRowMeta = false;
}

void NNEvaluator::maybeWarmupComputeHandle(ComputeHandle* gpuHandle, int serverThreadIdx) {
  if(gpuHandle == NULL || debugSkipNeuralNet || loadedModel == NULL)
    return;
  // Warmup currently only matters on CUDA, where cuDNN lazily compiles an SDPA execution plan per
  // batch size on first use. Other backends: nothing to warm up for now.
#if !defined(USE_CUDA_BACKEND)
  (void)serverThreadIdx;
  return;
#else
  // Only transformer models build the lazy SDPA graphs; skip the (otherwise harmless but wasteful)
  // warmup passes for plain convnets.
  if(!NeuralNet::getModelDesc(loadedModel).hasAnyTransformerBlocks())
    return;

  if(logger != NULL) {
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) +
      ": warming up transformer graphs for batch sizes 1.." + Global::intToString(maxBatchSize)
    );
  }

  // Empty board of the configured size, default rules/params. Outputs are discarded; we only want
  // the forward passes to trigger graph compilation for every batch size that will be seen.
  Board board(nnXLen, nnYLen);
  //Featurize the way this model expects (a no-op under Tromp-Taylorish rules, but robust if the
  //warmup rules ever change).
  BoardHistory history(
    board, P_BLACK, (inputsVersion == 8 ? Rules::forGame(Rules::Game::FREESTYLE) : Rules::getTrompTaylorish()), 0,
    BoardHistoryModes(modelPreferPassAliveUnderSuicideRules(), modelPreferExcludeTerritoryAdjacentToAtari())
  );
  MiscNNInputParams nnInputParams;
  SGFMetadata sgfMeta;
  const SGFMetadata* sgfMetaPtr = NULL;
  if(numInputMetaChannels > 0) {
    sgfMeta = SGFMetadata::makeDummyWarmupProfile();
    sgfMetaPtr = &sgfMeta;
  }

  // Mark the handle as warming up so the backend treats lazy-graph-compilation failures (e.g. cudnn
  // SDPA) leniently, falling back to a custom kernel instead of failing hard. Restored when done.
  bool prevIsWarmup = NeuralNet::setIsWarmup(gpuHandle, true);

  InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);

  // Reusable per-row input; identical for every row since it's an empty board.
  std::vector<std::unique_ptr<NNResultBuf>> ownedBufs;
  std::vector<NNResultBuf*> resultBufs;
  ownedBufs.reserve(maxBatchSize);
  resultBufs.reserve(maxBatchSize);
  for(int i = 0; i < maxBatchSize; i++) {
    ownedBufs.push_back(std::make_unique<NNResultBuf>());
    NNResultBuf* buf = ownedBufs.back().get();
    if(inputsVersion == 9) fillRowBufs(Xiangqi::State(),nnInputParams,*buf);
    else fillRowBufs(board, history, P_BLACK, sgfMetaPtr, nnInputParams, *buf);
    buf->symmetry = 0;
    buf->policyOptimism = nnInputParams.policyOptimism;
    resultBufs.push_back(buf);
  }

  for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
    std::vector<NNOutput*> outputs;
    outputs.reserve(batchSize);
    for(int row = 0; row < batchSize; row++) {
      NNOutput* out = new NNOutput();
      out->nnXLen = nnXLen;
      out->nnYLen = nnYLen;
      outputs.push_back(out);
    }
    NeuralNet::getOutput(gpuHandle, inputBuffers, batchSize, resultBufs.data(), outputs);
    for(NNOutput* out : outputs)
      delete out;
  }

  NeuralNet::freeInputBuffers(inputBuffers);
  NeuralNet::setIsWarmup(gpuHandle, prevIsWarmup);
#endif
}

NNEvalBenchmarkResult NNEvaluator::benchmarkPureForward(
  int numWarmups,
  int numIterations,
  const std::vector<std::pair<int,int>>& boardSizes,
  const Rules& rules
) {
  if(numIterations <= 0)
    throw StringError("benchmarkPureForward: numIterations must be positive");
  if(numWarmups < 0)
    throw StringError("benchmarkPureForward: numWarmups must be nonnegative");
  if(boardSizes.size() <= 0)
    throw StringError("benchmarkPureForward: no board sizes specified");
  if(debugSkipNeuralNet || loadedModel == NULL || computeContext == NULL)
    throw StringError("benchmarkPureForward requires a real neural net model");
  if(serverThreads.size() > 0)
    throw StringError("benchmarkPureForward: server threads must not be spawned");
  bool rulesSupported;
  getSupportedRules(rules,rulesSupported);
  if(!rulesSupported) throw StringError("benchmarkPureForward: model and game rules differ");
  for(const auto& size: boardSizes) {
    const int xSize = size.first, ySize = size.second;
    if(xSize < 2 || ySize < 2 || xSize > nnXLen || ySize > nnYLen)
      throw StringError("benchmarkPureForward: board size out of range for NN buffer");
    if(rules.isXiangqi() && (xSize != 9 || ySize != 10))
      throw StringError("benchmarkPureForward: Xiangqi requires a 9x10 board");
    if(!rules.isGo() && !rules.isXiangqi() && xSize != ySize)
      throw StringError("benchmarkPureForward: placement games require square boards");
    if(requireExactNNLen && (xSize != nnXLen || ySize != nnYLen))
      throw StringError("benchmarkPureForward: requireExactNNLen is set but board size does not fill the NN buffer");
  }

  const int numThreadsToUse = (int)gpuIdxByServerThread.size();
  const int batchSize = maxBatchSize;
  testAssert(numThreadsToUse > 0);

  // Left populated after the benchmark returns, deliberately: callers query
  // isAnyThreadUsingFP16() afterward to report what precision was measured.
  {
    std::lock_guard<std::mutex> lock(bufferMutex);
    if(serverThreadsIsUsingFP16.size() < (size_t)numThreadsToUse)
      serverThreadsIsUsingFP16.resize(numThreadsToUse,0);
  }

  struct ThreadContext {
    ComputeHandle* gpuHandle = nullptr;
    std::unique_ptr<NNServerBuf> serverBuf;
    std::vector<std::unique_ptr<NNResultBuf>> ownedBufs;
    std::vector<NNResultBuf*> resultBufs;
    std::vector<std::unique_ptr<NNOutput>> ownedOutputs;
    std::vector<NNOutput*> outputs;
    int batchSize;
    ~ThreadContext() {
      if(gpuHandle) NeuralNet::freeComputeHandle(gpuHandle);
    }
    void forward() {
      NeuralNet::getOutput(gpuHandle,serverBuf->inputBuffers,batchSize,resultBufs.data(),outputs);
    }
  };
  return runNNBenchmark(numWarmups,numIterations,batchSize,numThreadsToUse,[&](int threadIdx) {
    auto context = std::make_unique<ThreadContext>();
    context->batchSize = batchSize;
    context->gpuHandle = NeuralNet::createComputeHandle(
      computeContext,
      loadedModel,
      logger,
      maxBatchSize,
      requireExactNNLen,
      inputsUseNHWC,
      gpuIdxByServerThread[threadIdx],
      threadIdx
    );
    auto* gpuHandle = context->gpuHandle;
    auto& serverBuf = context->serverBuf;
    auto& ownedBufs = context->ownedBufs;
    auto& resultBufs = context->resultBufs;
    auto& ownedOutputs = context->ownedOutputs;
    auto& outputs = context->outputs;
    // Mirror serve(): one compute handle and one set of input buffers per server thread,
    // with the buffers allocated only after createComputeHandle has bound this thread to
    // its GPU (see serve() for why the order matters).
    serverBuf = std::make_unique<NNServerBuf>(*this, loadedModel);
    maybeWarmupComputeHandle(gpuHandle, threadIdx);
    {
      std::lock_guard<std::mutex> lock(bufferMutex);
      if((size_t)threadIdx < serverThreadsIsUsingFP16.size())
        serverThreadsIsUsingFP16[threadIdx] = NeuralNet::isUsingFP16(gpuHandle) ? 1 : 0;
    }

    MiscNNInputParams nnInputParams;
    SGFMetadata sgfMeta;
    const SGFMetadata* sgfMetaPtr = NULL;
    if(numInputMetaChannels > 0) {
      sgfMeta = SGFMetadata::makeDummyWarmupProfile();
      sgfMetaPtr = &sgfMeta;
    }

    // Empty boards, cycling through the requested sizes across the rows of the batch.
    // Position content doesn't affect speed, but board size does when masking is active.
    ownedBufs.reserve(batchSize);
    resultBufs.reserve(batchSize);
    for(int row = 0; row < batchSize; row++) {
      const auto size = boardSizes[row % boardSizes.size()];
      Board board(size.first, size.second);
      const BoardHistoryModes modes(modelPreferPassAliveUnderSuicideRules(), modelPreferExcludeTerritoryAdjacentToAtari());
      BoardHistory history = rules.isXiangqi() ? BoardHistory(Xiangqi::State(),modes) :
        BoardHistory(board,P_BLACK,rules,0,modes);
      if(rules.isXiangqi()) board = history.getRecentBoard(0);
      ownedBufs.push_back(std::make_unique<NNResultBuf>());
      NNResultBuf* buf = ownedBufs.back().get();
      fillRowBufs(board, history, P_BLACK, sgfMetaPtr, nnInputParams, *buf);
      buf->symmetry = 0;
      buf->policyOptimism = nnInputParams.policyOptimism;
      resultBufs.push_back(buf);
    }

    ownedOutputs.reserve(batchSize);
    outputs.reserve(batchSize);
    for(int row = 0; row < batchSize; row++) {
      ownedOutputs.push_back(std::make_unique<NNOutput>());
      NNOutput* out = ownedOutputs.back().get();
      out->nnXLen = nnXLen;
      out->nnYLen = nnYLen;
      outputs.push_back(out);
    }

    return context;
  });
}

void NNEvaluator::serve(
  Rand& rand,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  int64_t numBatchesHandledThisThread = 0;
  int64_t numRowsHandledThisThread = 0;

  ComputeHandle* gpuHandle = NULL;
  if(loadedModel != NULL) {
    gpuHandle = NeuralNet::createComputeHandle(
      computeContext,
      loadedModel,
      logger,
      maxBatchSize,
      requireExactNNLen,
      inputsUseNHWC,
      gpuIdxForThisThread,
      serverThreadIdx
    );

    // Warm up lazily-compiled backend graphs before reporting this thread as started.
    maybeWarmupComputeHandle(gpuHandle, serverThreadIdx);
  }

  // Allocate input buffers only after createComputeHandle, which binds this thread to its GPU.
  // On the CUDA backend the pinned host allocation in the buffers initializes a context on the
  // thread's current device, so doing it before the GPU binding would leave a stray context
  // (and some wasted VRAM) on GPU 0 even when only other GPUs are configured.
  NNServerBuf buf(*this, loadedModel);

  {
    lock_guard<std::mutex> lock(bufferMutex);
    testAssert(serverThreadIdx < serverThreadsIsUsingFP16.size());
    serverThreadsIsUsingFP16[serverThreadIdx] = gpuHandle == NULL ? 0 : NeuralNet::isUsingFP16(gpuHandle) ? 1 : 0;
    numServerThreadsStartingUp--;
    if(numServerThreadsStartingUp <= 0)
      mainThreadWaitingForSpawn.notify_all();
  }

  vector<NNResultBuf*> resultBufs;
  resultBufs.reserve(maxBatchSize);

  vector<NNOutput*> outputBuf;

  unique_lock<std::mutex> lock(bufferMutex,std::defer_lock);
  while(true) {
    resultBufs.clear();
    int desiredBatchSize = std::min(maxBatchSize, maxRowsToSendPerBatch.load(std::memory_order_acquire));
    bool gotAnything = queryQueue.waitPopUpToN(resultBufs,desiredBatchSize);
    // Queue being closed is a signal that we're done.
    if(!gotAnything)
      break;

    int numRows = (int)resultBufs.size();
    testAssert(numRows > 0);

    bool doRandomize = currentDoRandomize.load(std::memory_order_acquire);
    int defaultSymmetry = currentDefaultSymmetry.load(std::memory_order_acquire);

    if(debugSkipNeuralNet) {
      for(int row = 0; row < numRows; row++) {
        testAssert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        int boardXSize = resultBuf->boardXSizeForServer;
        int boardYSize = resultBuf->boardYSizeForServer;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        testAssert(resultBuf->hasResult == false);
        resultBuf->result = std::make_shared<NNOutput>();

        if(resultBuf->xiangqiSymmetries) {
          resultBuf->result->resizePolicy(Xiangqi::NUM_ACTIONS);
          for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++)
            resultBuf->result->policyProbs[action] = (float)rand.nextGaussian();
        }
        else {
          float* policyProbs = resultBuf->result->policyProbs;
          for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
            policyProbs[i] = 0;

          // At this point, these aren't probabilities, since this is before the postprocessing
          // that happens for each result. These just need to be unnormalized log probabilities.
          // Illegal move filtering happens later.
          for(int y = 0; y<boardYSize; y++) {
            for(int x = 0; x<boardXSize; x++) {
              int pos = NNPos::xyToPos(x,y,nnXLen);
              policyProbs[pos] = (float)rand.nextGaussian();
            }
          }
          policyProbs[NNPos::locToPos(Board::PASS_LOC,boardXSize,nnXLen,nnYLen)] = (float)rand.nextGaussian();
        }

        resultBuf->result->nnXLen = nnXLen;
        resultBuf->result->nnYLen = nnYLen;


        // These aren't really probabilities. Win/Loss/NoResult will get softmaxed later
        double whiteWinProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteLossProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteDrawProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteNoResultProb = 0.0 + rand.nextGaussian() * 0.20;
        double varTimeLeft = 0.5 * boardXSize * boardYSize;
        resultBuf->result->whiteWinProb = (float)whiteWinProb;
        resultBuf->result->whiteLossProb = (float)whiteLossProb;
        resultBuf->result->whiteDrawProb = (float)whiteDrawProb;
        resultBuf->result->whiteNoResultProb = (float)whiteNoResultProb;
        resultBuf->result->varTimeLeft = (float)varTimeLeft;
        resultBuf->result->shorttermWinlossError = 0.0f;
        resultBuf->result->policyOptimismUsed = (float)resultBuf->policyOptimism;
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
    }
    else {
      outputBuf.clear();
      for(int row = 0; row<numRows; row++) {
        NNOutput* emptyOutput = new NNOutput();
        testAssert(resultBufs[row] != NULL);
        emptyOutput->nnXLen = nnXLen;
        emptyOutput->nnYLen = nnYLen;
        outputBuf.push_back(emptyOutput);
      }

      for(int row = 0; row<numRows; row++) {
        if(resultBufs[row]->symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
          if(doRandomize) {
            if(resultBufs[row]->xiangqiSymmetries)
              resultBufs[row]->symmetry = (int)rand.nextUInt(2)*2;
            else
              resultBufs[row]->symmetry = resultBufs[row]->hexSymmetries ? (int)rand.nextUInt(2)*3 : rand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES);
          }
          else {
            testAssert(defaultSymmetry >= 0 && defaultSymmetry <= SymmetryHelpers::NUM_SYMMETRIES-1);
            resultBufs[row]->symmetry = defaultSymmetry;
          }
        }
      }

      NeuralNet::getOutput(gpuHandle, buf.inputBuffers, numRows, resultBufs.data(), outputBuf);
      testAssert(outputBuf.size() == numRows);

      m_numRowsProcessed.fetch_add(numRows, std::memory_order_relaxed);
      m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
      numRowsHandledThisThread += numRows;
      numBatchesHandledThisThread += 1;

      for(int row = 0; row < numRows; row++) {
        testAssert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        testAssert(resultBuf->hasResult == false);
        resultBuf->result = std::shared_ptr<NNOutput>(outputBuf[row]);
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
    }

    // Lock and update stats before looping again
    lock.lock();
    numOngoingEvals -= numRows;

    if(numWaitingEvals > 0) {
      numEvalsToAwaken += numWaitingEvals;
      numWaitingEvals = 0;
      waitingForFinish.notify_all();
    }
    lock.unlock();
    continue;
  }

  NeuralNet::freeComputeHandle(gpuHandle);
  if(logger != NULL) {
    logger->write(
      "GPU " + Global::intToString(gpuIdxForThisThread) + " finishing, processed " +
      Global::int64ToString(numRowsHandledThisThread) + " rows " +
      Global::int64ToString(numBatchesHandledThisThread) + " batches"
    );
  }
}

void NNEvaluator::waitForNextNNEvalIfAny() {
  unique_lock<std::mutex> lock(bufferMutex);
  if(numOngoingEvals <= 0)
    return;

  numWaitingEvals++;
  while(numEvalsToAwaken <= 0 && !isKilled)
    waitingForFinish.wait(lock);
  numEvalsToAwaken--;
}


static double softPlus(double x) {
  // Avoid blowup
  if(x > 40.0)
    return x;
  else
    return log(1.0 + exp(x));
}

void NNPostprocess::policy(NNOutput& output, const bool* isLegal, int policySize,
                           float outputScale, float temperature, double optimism,
                           bool& errorLogLockout, Logger* logger, const std::string& modelName) {
  float* policy = output.policyProbs;
  float policyOutputScaling = outputScale / temperature;
  float maxPolicy = -1e25f;
  int legalCount = 0;
  for(int i = 0; i<policySize; i++) {
    float policyValue;
    if(isLegal[i]) {
      legalCount += 1;
      policyValue = policy[i] * policyOutputScaling;
    }
    else
      policyValue = -1e30f;

    policy[i] = policyValue;
    if(policyValue > maxPolicy)
      maxPolicy = policyValue;
  }

  testAssert(legalCount > 0);

  float policySum = 0.0f;

  for(int i = 0; i<policySize; i++) {
    policy[i] = exp(policy[i] - maxPolicy);
    policySum += policy[i];
  }

  if(!isfinite(policySum)) {
    cout << "Got nonfinite for policy sum" << endl;
    throw StringError("Got nonfinite for policy sum");
  }

  // Somehow all legal moves rounded to 0 probability
  if(policySum <= 0.0) {
    if(!errorLogLockout && logger != NULL) {
      errorLogLockout = true;
      logger->write("Warning: all legal moves rounded to 0 probability for " + string(modelName));
    }
    float uniform = 1.0f / legalCount;
    for(int i = 0; i<policySize; i++) {
      policy[i] = isLegal[i] ? uniform : -1.0f;
    }
  }
  // Normal case
  else {
    for(int i = 0; i<policySize; i++)
      policy[i] = isLegal[i] ? (policy[i] / policySum) : -1.0f;
  }

  // Fill everything out-of-bounds too, for robustness.
  for(int i = policySize; i<output.getPolicyStorageSize(); i++)
    policy[i] = -1.0f;

  output.policyOptimismUsed = (float)optimism;

}

void NNPostprocess::value(NNOutput& output, Player nextPlayer, bool suppressNoResult,
                          int modelVersion, const ModelPostProcessParams& params) {
  // Fix up the value as well. Note that the neural net gives us back the value from the perspective
  // of the player so we need to negate that to make it the white value.
  if(modelVersion >= 3) {
    double winProb;
    double lossProb;
    double drawProb;
    double noResultProb;
    double varTimeLeft;
    double shorttermWinlossError;
    {
      double winLogits = output.whiteWinProb * params.outputScaleMultiplier;
      double lossLogits = output.whiteLossProb * params.outputScaleMultiplier;
      double drawLogits = output.whiteDrawProb * params.outputScaleMultiplier;
      double noResultLogits = output.whiteNoResultProb * params.outputScaleMultiplier;
      double varTimeLeftPreSoftplus = output.varTimeLeft * params.outputScaleMultiplier;
      double shorttermWinlossErrorPreSoftplus = output.shorttermWinlossError * params.outputScaleMultiplier;

      if(suppressNoResult)
        noResultLogits -= 100000.0;

      // Softmax
      double maxLogits = std::max(std::max(winLogits,lossLogits),std::max(drawLogits,noResultLogits));
      winProb = exp(winLogits - maxLogits);
      lossProb = exp(lossLogits - maxLogits);
      drawProb = exp(drawLogits - maxLogits);
      noResultProb = exp(noResultLogits - maxLogits);

      if(suppressNoResult)
        noResultProb = 0.0;

      double probSum = winProb + drawProb + lossProb + noResultProb;
      winProb /= probSum;
      lossProb /= probSum;
      drawProb /= probSum;
      noResultProb /= probSum;

      varTimeLeft = softPlus(varTimeLeftPreSoftplus) * params.varianceTimeMultiplier;

      if(modelVersion >= 14) {
        {
          double s = softPlus(shorttermWinlossErrorPreSoftplus * 0.5);
          shorttermWinlossError = sqrt(s * s * params.shorttermValueErrorMultiplier);
        }
      }
      else if(modelVersion >= 10) {
        shorttermWinlossError = sqrt(softPlus(shorttermWinlossErrorPreSoftplus) * params.shorttermValueErrorMultiplier);
      }
      else {
        shorttermWinlossError = softPlus(shorttermWinlossErrorPreSoftplus);
      }

      if(
        !isfinite(probSum) ||
        !isfinite(varTimeLeft) ||
        !isfinite(shorttermWinlossError)
      ) {
        cout << "Got nonfinite for nneval value" << endl;
        cout << winLogits << " " << lossLogits << " " << noResultLogits
             << " " << varTimeLeft << " " << shorttermWinlossError
             << endl;
        throw StringError("Got nonfinite for nneval value");
      }
    }

    if(nextPlayer == P_WHITE) {
      output.whiteWinProb = (float)winProb;
      output.whiteLossProb = (float)lossProb;
      output.whiteDrawProb = (float)drawProb;
      output.whiteNoResultProb = (float)noResultProb;
    }
    else {
      output.whiteWinProb = (float)lossProb;
      output.whiteLossProb = (float)winProb;
      output.whiteDrawProb = (float)drawProb;
      output.whiteNoResultProb = (float)noResultProb;
    }

    if(modelVersion >= 9) {
      output.varTimeLeft = (float)varTimeLeft;
      output.shorttermWinlossError = (float)shorttermWinlossError;
    }
    else {
      output.varTimeLeft = -1;
      output.shorttermWinlossError = -1;
    }
  }
  else {
    throw StringError("NNEval value postprocessing not implemented for model version");
  }
}

template<typename Evaluate>
static std::shared_ptr<NNOutput>* averageSymmetries(
  std::array<int,SymmetryHelpers::NUM_SYMMETRIES> symmetryIndexes, int numAllowedSymmetries,
  int numSymmetriesToSample, NNResultBuf& buf, Rand& rand, const Evaluate& evaluate
) {
  vector<std::shared_ptr<NNOutput>> ptrs;
  numSymmetriesToSample = std::min(numSymmetriesToSample,numAllowedSymmetries);
  for(int i = 0; i<numSymmetriesToSample; i++) {
    std::swap(symmetryIndexes[i], symmetryIndexes[rand.nextInt(i,numAllowedSymmetries-1)]);
    evaluate(symmetryIndexes[i]);
    ptrs.push_back(std::move(buf.result));
  }
  return new std::shared_ptr<NNOutput>(new NNOutput(ptrs));
}

std::shared_ptr<NNOutput>* NNEvaluator::averageMultipleSymmetries(
  const Board& board, const BoardHistory& history, Player nextPlayer, const SGFMetadata* sgfMeta,
  const MiscNNInputParams& baseNNInputParams, NNResultBuf& buf, Rand& rand, int numSymmetriesToSample
) {
  MiscNNInputParams params = baseNNInputParams;
  std::array<int,SymmetryHelpers::NUM_SYMMETRIES> indexes{};
  int count = 0;
  for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++)
    if(history.rules.allowsSymmetry(symmetry)) indexes[count++] = symmetry;
  return averageSymmetries(indexes,count,numSymmetriesToSample,buf,rand,[&](int symmetry) {
    params.symmetry = symmetry;
    // Skip cache since there's no guarantee which symmetry is in the cache.
    evaluate(board,history,nextPlayer,sgfMeta,params,buf,true);
  });
}

std::shared_ptr<NNOutput>* NNEvaluator::averageMultipleSymmetries(
  const Xiangqi::State& state, const MiscNNInputParams& baseParams, NNResultBuf& buf,
  Rand& rand, int numSymmetriesToSample
) {
  MiscNNInputParams params = baseParams;
  std::array<int,SymmetryHelpers::NUM_SYMMETRIES> indexes{0,2};
  return averageSymmetries(indexes,2,numSymmetriesToSample,buf,rand,[&](int symmetry) {
    params.symmetry = symmetry;
    evaluate(state,params,buf,true);
  });
}

void NNEvaluator::evaluate(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  bool skipCache) {
  evaluate(
    board,
    history,
    nextPlayer,
    NULL,
    nnInputParams,
    buf,
    skipCache);
}

void NNEvaluator::evaluate(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParamsArg,
  NNResultBuf& buf,
  bool skipCache) {
  testAssert(!isKilled);
  buf.hasResult = false;

  if(history.rules.isXiangqi()) {
    if(history.isGameFinished || nextPlayer != history.getXiangqiState().nextPlayer())
      throw StringError("Xiangqi evaluation requires an ongoing state and its next player");
    evaluate(history.getXiangqiState(),nnInputParamsArg,buf,skipCache);
    return;
  }

  if(board.x_size > nnXLen || board.y_size > nnYLen)
    throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                      " nnYLen = " + Global::intToString(nnYLen) +
                      " but was asked to evaluate board with larger x or y size");
  if(requireExactNNLen) {
    if(board.x_size != nnXLen || board.y_size != nnYLen)
      throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                        " nnYLen = " + Global::intToString(nnYLen) +
                        " and requireExactNNLen, but was asked to evaluate board with different x or y size");
  }

  if(!debugSkipNeuralNet && (inputsVersion == 9 || ((inputsVersion == 8) == history.rules.isGo())))
    throw StringError("Model input features do not match the requested game");
  const bool hex = history.rules.game == Rules::Game::HEX;
  if(nnInputParamsArg.symmetry >= 0 && !history.rules.allowsSymmetry(nnInputParamsArg.symmetry))
    throw StringError("Requested neural-net symmetry is invalid for the game");
  if(nnInputParamsArg.symmetry == NNInputs::SYMMETRY_NOTSPECIFIED && !getDoRandomize() && !history.rules.allowsSymmetry(getDefaultSymmetry()))
    throw StringError("Configured neural-net symmetry is invalid for the game");

  // Avoid using policy optimism for humanSL
  MiscNNInputParams nnInputParams = nnInputParamsArg;
  if(numInputMetaChannels > 0)
    nnInputParams.policyOptimism = 0.0;

  Hash128 nnHash = NNInputs::getHash(board, history, nextPlayer, nnInputParams);
  if(numInputMetaChannels > 0) {
    if(sgfMeta == NULL)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not provided");
    if(!sgfMeta->initialized)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
    nnHash ^= sgfMeta->getHash(nextPlayer);
  }

  if(nnCacheTable != NULL && !skipCache && nnCacheTable->get(nnHash,buf.result)) {
    m_numCacheHits.fetch_add(1, std::memory_order_relaxed);
    buf.hasResult = true;
    return;
  }

  buf.boardXSizeForServer = board.x_size;
  buf.boardYSizeForServer = board.y_size;

  if(!debugSkipNeuralNet) {
    fillRowBufs(board, history, nextPlayer, sgfMeta, nnInputParams, buf);
  }

  buf.symmetry = nnInputParams.symmetry;
  buf.hexSymmetries = hex;
  buf.xiangqiSymmetries = false;
  buf.policyOptimism = nnInputParams.policyOptimism;

  submitAndWait(buf);

  bool isLegal[NNPos::MAX_NN_POLICY_SIZE];
  testAssert(nextPlayer == history.presumedNextMovePla);
  for(int i = 0; i<policySize; i++) {
    Loc loc = NNPos::posToLoc(i,board.x_size,board.y_size,nnXLen,nnYLen);
    isLegal[i] = history.isLegal(board,loc,nextPlayer);
  }
  try {
    NNPostprocess::policy(*buf.result,isLegal,policySize,postProcessParams.outputScaleMultiplier,
                          nnInputParams.nnPolicyTemperature,nnInputParams.policyOptimism,
                          buf.errorLogLockout,logger,modelFileName);
  }
  catch(...) { history.printDebugInfo(cout,board); throw; }
  const bool suppressNoResult = !history.rules.isGo() ||
    (history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY);
  NNPostprocess::value(*buf.result,nextPlayer,suppressNoResult,modelVersion,postProcessParams);

  // And record the nnHash in the result and put it into the table
  buf.result->nnHash = nnHash;
  if(nnCacheTable != NULL)
    nnCacheTable->set(buf.result);

}

void NNEvaluator::submitAndWait(NNResultBuf& buf) {
  unique_lock<std::mutex> lock(bufferMutex);
  numOngoingEvals += 1;
  lock.unlock();

  bool suc = queryQueue.forcePush(&buf);
  testAssert(suc);

  unique_lock<std::mutex> resultLock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(resultLock);
  resultLock.unlock();

}

void NNEvaluator::evaluate(const Xiangqi::State& state, const MiscNNInputParams& params,
                           NNResultBuf& buf, bool skipCache) {
  testAssert(!isKilled);
  buf.hasResult = false;
  if(!debugSkipNeuralNet && inputsVersion != 9)
    throw StringError("Xiangqi requires a joint-policy model (version 19)");
  if(nnXLen < Xiangqi::WIDTH || nnYLen < Xiangqi::HEIGHT ||
     (requireExactNNLen && (nnXLen != Xiangqi::WIDTH || nnYLen != Xiangqi::HEIGHT)))
    throw StringError("Xiangqi board does not match the evaluator canvas");
  if(state.result().finished)
    throw StringError("Cannot evaluate a terminal Xiangqi position with no legal policy");
  auto allowed = [](int symmetry) { return symmetry == 0 || symmetry == 2; };
  if(params.symmetry != NNInputs::SYMMETRY_NOTSPECIFIED && !allowed(params.symmetry))
    throw StringError("Xiangqi only supports identity and horizontal reflection");
  if(params.symmetry == NNInputs::SYMMETRY_NOTSPECIFIED && !getDoRandomize() && !allowed(getDefaultSymmetry()))
    throw StringError("Configured neural-net symmetry is invalid for Xiangqi");

  const Hash128 nnHash = NNInputs::getHash(state,params);
  if(nnCacheTable != NULL && !skipCache && nnCacheTable->get(nnHash,buf.result)) {
    m_numCacheHits.fetch_add(1, std::memory_order_relaxed);
    buf.hasResult = true;
    return;
  }
  buf.boardXSizeForServer = Xiangqi::WIDTH;
  buf.boardYSizeForServer = Xiangqi::HEIGHT;
  if(!debugSkipNeuralNet) fillRowBufs(state,params,buf);
  buf.symmetry = params.symmetry;
  buf.hexSymmetries = false;
  buf.xiangqiSymmetries = true;
  buf.policyOptimism = params.policyOptimism;
  submitAndWait(buf);

  // The backend has already reversed reflection. Its remaining action frame is
  // the current player's; the source cache/search use absolute board actions.
  if(state.nextPlayer() == P_WHITE) {
    float* policy = buf.result->policyProbs;
    for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) {
      int rotated = Xiangqi::transformAction(action,P_WHITE,false);
      if(rotated > action) std::swap(policy[action],policy[rotated]);
    }
  }
  bool legal[Xiangqi::NUM_ACTIONS] = {};
  for(int action : state.legalActions()) legal[action] = true;
  NNPostprocess::policy(*buf.result,legal,Xiangqi::NUM_ACTIONS,postProcessParams.outputScaleMultiplier,
                        params.nnPolicyTemperature,params.policyOptimism,
                        buf.errorLogLockout,logger,modelFileName);
  NNPostprocess::value(*buf.result,state.nextPlayer(),true,modelVersion,postProcessParams);
  buf.result->nnHash = nnHash;
  if(nnCacheTable != NULL) nnCacheTable->set(buf.result);
}

// Uncomment this to lower the effective hash size down to one where we get true collisions
// #define SIMULATE_TRUE_HASH_COLLISIONS

NNCacheTable::Entry::Entry()
  :ptr(nullptr)
{}
NNCacheTable::Entry::~Entry()
{}

NNCacheTable::NNCacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
    throw StringError("NNCacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  sizePowerOfTwo = sizePowerOfTwo > 12 ? 12 : sizePowerOfTwo;
#endif
  if(mutexPoolSizePowerOfTwo > sizePowerOfTwo)
    mutexPoolSizePowerOfTwo = sizePowerOfTwo;

  tableSize = ((uint64_t)1) << sizePowerOfTwo;
  tableMask = tableSize-1;
  entries = new Entry[tableSize];
  uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
  mutexPoolMask = mutexPoolSize-1;
  mutexPool = new MutexPool(mutexPoolSize);
}
NNCacheTable::~NNCacheTable() {
  delete[] entries;
  delete mutexPool;
}

bool NNCacheTable::get(Hash128 nnHash, shared_ptr<NNOutput>& ret) {
  // Free ret BEFORE locking, to avoid any expensive operations while locked.
  if(ret != nullptr)
    ret.reset();

  uint64_t idx = nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  std::lock_guard<std::mutex> lock(mutex);

  bool found = false;
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  if(entry.ptr != nullptr && ((entry.ptr->nnHash.hash0 ^ nnHash.hash0) & 0xFFF) == 0) {
    ret = entry.ptr;
    found = true;
  }
#else
  if(entry.ptr != nullptr && entry.ptr->nnHash == nnHash) {
    ret = entry.ptr;
    found = true;
  }
#endif
  return found;
}

void NNCacheTable::set(const shared_ptr<NNOutput>& p) {
  // Immediately copy p right now, before locking, to avoid any expensive operations while locked.
  shared_ptr<NNOutput> buf(p);

  uint64_t idx = p->nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  {
    std::lock_guard<std::mutex> lock(mutex);
    // Perform a swap, to avoid any expensive free under the mutex.
    entry.ptr.swap(buf);
  }

  // No longer locked, allow buf to fall out of scope now, will free whatever used to be present in the table.
}

void NNCacheTable::clear() {
  shared_ptr<NNOutput> buf;
  for(size_t idx = 0; idx<tableSize; idx++) {
    Entry& entry = entries[idx];
    uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
    std::mutex& mutex = mutexPool->getMutex(mutexIdx);
    {
      std::lock_guard<std::mutex> lock(mutex);
      entry.ptr.swap(buf);
    }
    buf.reset();
  }
}
