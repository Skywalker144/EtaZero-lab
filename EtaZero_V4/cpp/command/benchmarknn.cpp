#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../game/xiangqi.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/muzero.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../external/nlohmann_json/json.hpp"
#include "../main.h"

#include <iomanip>
#include <sstream>

using namespace std;
using json = nlohmann::json;

static json benchmarkMetrics(const NNEvalBenchmarkResult& result) {
  vector<double> milliseconds;
  for(double seconds : result.perThreadMedianSeconds) milliseconds.push_back(seconds*1000.0);
  return json{{"batchSize",result.batchSize}, {"numThreads",result.numThreads}, {"numIterations",result.numIterations},
    {"perThreadMedianMs",milliseconds}, {"perThreadNNEvalsPerSec",result.perThreadNNEvalsPerSec},
    {"sumMedianNNEvalsPerSec",result.sumMedianNNEvalsPerSec}, {"actualWallSeconds",result.actualWallSeconds},
    {"actualWallNNEvalsPerSec",result.actualWallNNEvalsPerSec}};
}

static void printBenchmark(const json& info, const json& metrics, const string& operation) {
  cout << "=== benchmarknn" << (operation.empty() ? "" : " " + operation) << " ===" << endl;
  cout << "model: " << info.at("modelFile").get<string>() << endl;
  cout << "internal model: " << info.at("modelName").get<string>() << endl;
  cout << "revision/backend: " << info.at("revision").get<string>() << endl;
  cout << "game: " << info.at("game").get<string>() << endl;
  cout << "board sizes:";
  for(const auto& shape : info.at("boardShapes")) cout << " " << shape[0] << "x" << shape[1];
  cout << " (NN buffer " << info.at("nnBufferSize")[0] << "x" << info.at("nnBufferSize")[1]
       << (info.at("requireExactNNLen").get<bool>() ? ", requireExactNNLen" : "") << ")" << endl;
  cout << "FP16: " << (metrics.at("usingFP16").get<bool>() ? "true" : "false") << endl;
  cout << "batch size per thread: " << metrics.at("batchSize") << endl;
  cout << "NN server threads: " << metrics.at("numThreads") << endl;
  cout << "GPU indices:";
  for(const auto& gpu : info.at("gpuIdxs")) cout << " " << gpu;
  cout << endl;
  cout << "timed iterations per thread: " << metrics.at("numIterations") << endl;
  for(int i = 0; i < metrics.at("numThreads").get<int>(); i++)
    cout << "thread " << i << ": " << setprecision(6) << metrics.at("perThreadMedianMs")[i].get<double>()
         << " ms/batch median, " << setprecision(8) << metrics.at("perThreadNNEvalsPerSec")[i].get<double>() << " nnEval/s" << endl;
  cout << "sum of per-thread median rates: " << setprecision(8) << metrics.at("sumMedianNNEvalsPerSec").get<double>() << " nnEval/s" << endl;
  cout << "wall time of timed region: " << setprecision(6) << metrics.at("actualWallSeconds").get<double>() << " s" << endl;
  cout << "overall throughput (wall): " << setprecision(8) << metrics.at("actualWallNNEvalsPerSec").get<double>() << " nnEval/s" << endl;
}

static NNEvalBenchmarkResult benchmarkMuZero(
  const shared_ptr<const MuZero::ModelBundle>& bundle, ConfigParser& cfg, Logger& logger,
  const Setup::EvaluatorOptions& options, const vector<pair<int,int>>& boardSizes,
  int numWarmups, int numIterations, bool recurrent, vector<int>& usingFP16
) {
  struct ThreadContext {
    // States are destroyed before their evaluator, on the worker that owns it.
    unique_ptr<MuZero::Evaluator> evaluator;
    vector<float> spatial, global, metadata;
    vector<MuZero::State> parents;
    vector<int> actions;
    int batchSize;
    bool recurrent;
    void forward() {
      // Output/latent allocation and release are part of the real native API.
      // Recurrence reuses a fixed parent batch, rather than downloading latent
      // states or silently timing representation again on each iteration.
      if(recurrent) evaluator->recurrent(parents,actions);
      else evaluator->initial(batchSize,spatial,global,metadata);
    }
  };
  const int threads = (int)options.gpuIdxByServerThread.size(), batch = options.nnMaxBatchSize;
  usingFP16.assign(threads,0);
  return runNNBenchmark(numWarmups,numIterations,batch,threads,[&](int index) {
    auto context = make_unique<ThreadContext>();
    context->batchSize = batch;
    context->recurrent = recurrent;
    context->evaluator = make_unique<MuZero::Evaluator>(bundle,batch,options.gpuIdxByServerThread[index],
      options.useFP16Mode,cfg,&logger,options.inputsUseNHWC);
    context->evaluator->warmup();
    usingFP16[index] = context->evaluator->isUsingFP16();
    const auto& desc = bundle->representationDesc();
    const int canvas = bundle->posLen(), area = canvas*canvas, spatialSize = desc.numInputChannels*area;
    context->spatial.resize((size_t)batch*spatialSize);
    context->global.resize((size_t)batch*desc.numInputGlobalChannels);
    context->metadata.resize((size_t)batch*desc.numInputMetaChannels);
    const Rules rules = Rules::forGame(Rules::parseGame(bundle->game()));
    const BoardHistoryModes modes(desc.preferPassAliveUnderSuicideRules,desc.preferExcludeTerritoryAdjacentToAtari);
    const SGFMetadata metadata = SGFMetadata::makeDummyWarmupProfile();
    for(int row = 0; row < batch; row++) {
      const auto size = boardSizes[row%boardSizes.size()];
      Board board(size.first,size.second);
      BoardHistory history = rules.isXiangqi() ? BoardHistory(Xiangqi::State(),modes) :
        BoardHistory(board,P_BLACK,rules,0,modes);
      if(rules.isXiangqi()) board = history.getRecentBoard(0);
      NNInputs::fillRow(NNModelVersion::getInputsVersion(desc.modelVersion),board,history,P_BLACK,MiscNNInputParams(),
        canvas,canvas,false,context->spatial.data()+(size_t)row*spatialSize,
        context->global.data()+(size_t)row*desc.numInputGlobalChannels);
      if(desc.numInputMetaChannels > 0)
        SGFMetadata::fillMetadataRow(&metadata,context->metadata.data()+(size_t)row*desc.numInputMetaChannels,P_BLACK,board.x_size*board.y_size);
      context->actions.push_back(rules.isXiangqi() ? history.getXiangqiState().legalActions()[row%history.getXiangqiState().legalActions().size()] :
        (row%(size.first*size.second))/size.first*canvas + row%size.first);
    }
    if(recurrent) {
      auto initial = context->evaluator->initial(batch,context->spatial,context->global,context->metadata);
      for(const auto& output : initial) context->parents.push_back(output.state);
    }
    return context;
  });
}

int MainCmds::benchmarknn(const vector<string>& args) {
  Board::initHash();
  Rand seedRand;
  ConfigParser cfg;
  string modelFile, boardSizesStr, operation;
  int numIterations, numWarmups, batchSizeOverride;
  bool requireExactNNLen, jsonOut, operationSpecified;
  try {
    KataGoCommandLine cmd(
      "Benchmark raw neural net forward throughput, without search. Uses numNNServerThreadsPerModel "
      "and per-thread GPU assignment from the config. Timing includes host-device transfers and "
      "output postprocessing, excludes input feature generation and search."
    );
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<int> iterationsArg("","iterations","Number of timed forward passes per NN server thread (default 200)",false,200,"N");
    TCLAP::ValueArg<int> warmupArg("","warmup","Untimed forward passes per NN server thread before timing (default 20)",false,20,"N");
    TCLAP::ValueArg<int> batchSizeArg("","batch-size","Batch size per NN server thread (default: nnMaxBatchSize from config, or 16)",false,-1,"N");
    TCLAP::ValueArg<string> boardSizesArg("","boardsize",
      "Board sizes cycled across batch rows, e.g. 9,13,19 or 5x7,9x9. Default 19, Xiangqi 9x10, MuZero its model canvas.",false,"","SIZES");
    TCLAP::ValueArg<string> operationArg("","muzero-operation","MuZero forward calls to measure: initial, recurrent, or both (default both)",false,"both","OP");
    TCLAP::SwitchArg requireExactNNLenArg("","require-exact-nnlen","Run with requireExactNNLen (backend may skip mask handling). Needs a single board size.");
    TCLAP::SwitchArg jsonArg("","json","Print results as JSON",false);
    cmd.add(iterationsArg); cmd.add(warmupArg); cmd.add(batchSizeArg); cmd.add(boardSizesArg);
    cmd.add(requireExactNNLenArg); cmd.add(jsonArg); cmd.add(operationArg);
    cmd.setShortUsageArgLimit(); cmd.addOverrideConfigArg(); cmd.parseArgs(args);
    modelFile = cmd.getModelFile(); numIterations = iterationsArg.getValue(); numWarmups = warmupArg.getValue();
    batchSizeOverride = batchSizeArg.getValue(); boardSizesStr = boardSizesArg.getValue();
    requireExactNNLen = requireExactNNLenArg.getValue(); jsonOut = jsonArg.getValue();
    operation = operationArg.getValue(); operationSpecified = operationArg.isSet(); cmd.getConfig(cfg);
    if(numIterations <= 0) throw StringError("benchmarknn: iterations must be positive");
    if(numWarmups < 0) throw StringError("benchmarknn: warmup must be nonnegative");
    if(batchSizeArg.isSet() && batchSizeOverride <= 0) throw StringError("benchmarknn: batch-size must be positive");
    if(operation != "both" && operation != "initial" && operation != "recurrent") throw StringError("benchmarknn: invalid muzero-operation");
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  // Keep stdout machine-readable with -json, including backend warnings.
  Logger logger(NULL,!jsonOut,jsonOut,false);
  logger.write("Version " + Version::getGitRevisionWithBackend());
  const bool muzero = cfg.contains("useMuZero") && cfg.getBool("useMuZero");
  if(!muzero && operationSpecified) throw StringError("benchmarknn: muzero-operation requires useMuZero=true");
  shared_ptr<const MuZero::ModelBundle> bundle;
  if(muzero) bundle = make_shared<MuZero::ModelBundle>(modelFile);
  const bool gameSpecified = cfg.contains("game") || cfg.contains("rules");
  Rules::Game game = cfg.contains("rules") ? Rules::parseRules(cfg.getString("rules")).game :
    bundle ? Rules::parseGame(bundle->game()) : Rules::Game::GO;
  if(cfg.contains("game")) {
    Rules::Game declared = Rules::parseGame(cfg.getString("game"));
    if(cfg.contains("rules") && declared != game) throw StringError("benchmarknn: config game and rules differ");
    game = declared;
  }
  if(bundle && game != Rules::parseGame(bundle->game())) throw StringError("benchmarknn: MuZero model and game differ");
  if(!cfg.contains("game")) cfg.overrideKey("game",Rules::gameName(game));
  if(boardSizesStr.empty()) boardSizesStr = game == Rules::Game::XIANGQI ? "9x10" : bundle ? Global::intToString(bundle->posLen()) : "19";
  vector<pair<int,int>> boardSizes;
  int nnXLen = 0, nnYLen = 0;
  for(const string& piece : Global::split(boardSizesStr,',')) {
    auto shape = Global::split(Global::toLower(Global::trim(piece)),'x');
    if(shape.size() < 1 || shape.size() > 2) throw StringError("benchmarknn: invalid board size " + piece);
    int x = Global::stringToInt(shape[0]), y = shape.size() == 1 ? x : Global::stringToInt(shape[1]);
    if(x < 2 || y < 2 || x > Board::MAX_LEN || y > Board::MAX_LEN) throw StringError("benchmarknn: invalid board size " + piece);
    if(game == Rules::Game::XIANGQI && (x != 9 || y != 10)) throw StringError("benchmarknn: Xiangqi requires a 9x10 board");
    if(game != Rules::Game::GO && game != Rules::Game::XIANGQI && x != y) throw StringError("benchmarknn: placement games require square boards");
    boardSizes.emplace_back(x,y); nnXLen = std::max(nnXLen,x); nnYLen = std::max(nnYLen,y);
  }
  if(boardSizes.empty()) throw StringError("benchmarknn: no board sizes specified");
  if(requireExactNNLen && boardSizes.size() > 1) throw StringError("benchmarknn: require-exact-nnlen needs a single board size");
  const int maxBatchSize = batchSizeOverride > 0 ? batchSizeOverride :
    cfg.contains("nnMaxBatchSize") ? cfg.getInt("nnMaxBatchSize",1,65536) : 16;
  json info{{"modelFile",modelFile}, {"revision",Version::getGitRevisionWithBackend()}, {"game",Rules::gameName(game)}, {"boardShapes",boardSizes}};
  vector<int> squares;
  for(auto size : boardSizes) if(size.first == size.second) squares.push_back(size.first);
  if(squares.size() == boardSizes.size()) info["boardSizes"] = squares;
  unique_ptr<NNEvaluator> nnEval;
  try {
    if(!muzero) {
      nnEval.reset(Setup::initializeNNEvaluator(modelFile,modelFile,"",cfg,logger,seedRand,maxBatchSize,
        nnXLen,nnYLen,Setup::MaxBatchSizeRequest::explicitSize(maxBatchSize),requireExactNNLen,false,Setup::SETUP_FOR_BENCHMARKNN));
      // The source used Freestyle inputs for an otherwise unspecified placement
      // model. An explicit game/rules always selects the actual variant.
      if(!gameSpecified && nnEval->getModelVersion() == 18) game = Rules::Game::FREESTYLE;
      auto result = nnEval->benchmarkPureForward(numWarmups,numIterations,boardSizes,Rules::forGame(game));
      info["game"] = Rules::gameName(game);
      info["modelName"] = nnEval->getInternalModelName();
      info["nnBufferSize"] = {nnEval->getNNXLen(),nnEval->getNNYLen()};
      info["requireExactNNLen"] = nnEval->getRequireExactNNLen();
      info["gpuIdxs"] = nnEval->getGpuIdxs();
      json metrics = benchmarkMetrics(result);
      metrics["usingFP16"] = nnEval->isAnyThreadUsingFP16();
      if(jsonOut) { info.update(metrics); cout << info.dump() << endl; }
      else printBenchmark(info,metrics,"");
    }
    else {
      const int canvas = bundle->posLen();
      auto options = Setup::readEvaluatorOptions(modelFile,0,cfg,logger,seedRand,maxBatchSize,canvas,canvas,
        Setup::MaxBatchSizeRequest::explicitSize(maxBatchSize),requireExactNNLen,false,Setup::SETUP_FOR_BENCHMARKNN);
      if(options.debugSkipNeuralNet) throw StringError("benchmarknn: MuZero requires a real model");
      if(options.nnXLen != canvas || options.nnYLen != canvas) throw StringError("benchmarknn: MuZero buffer must match model canvas");
      for(auto size : boardSizes) {
        if(size.first > canvas || size.second > canvas) throw StringError("benchmarknn: board size exceeds MuZero canvas");
        if(options.requireExactNNLen && (size.first != canvas || size.second != canvas))
          throw StringError("benchmarknn: requireExactNNLen is set but board size does not fill the NN buffer");
      }
      info["modelName"] = bundle->predictionDesc().name;
      info["nnBufferSize"] = {canvas,canvas};
      info["requireExactNNLen"] = options.requireExactNNLen;
      info["gpuIdxs"] = set<int>(options.gpuIdxByServerThread.begin(),options.gpuIdxByServerThread.end());
      info["operations"] = json::object();
      for(bool recurrent : {false,true}) {
        const string name = recurrent ? "recurrent" : "initial";
        if(operation != "both" && operation != name) continue;
        vector<int> usingFP16;
        auto result = benchmarkMuZero(bundle,cfg,logger,options,boardSizes,numWarmups,numIterations,recurrent,usingFP16);
        json metrics = benchmarkMetrics(result);
        metrics["usingFP16"] = std::any_of(usingFP16.begin(),usingFP16.end(),[](int value) { return value != 0; });
        info["operations"][name] = metrics;
        if(!jsonOut) printBenchmark(info,metrics,name);
      }
      if(jsonOut) cout << info.dump() << endl;
    }
  }
  catch(...) {
    nnEval.reset(); bundle.reset(); NeuralNet::globalCleanup();
    throw;
  }
  nnEval.reset(); bundle.reset(); NeuralNet::globalCleanup();
  return 0;
}
