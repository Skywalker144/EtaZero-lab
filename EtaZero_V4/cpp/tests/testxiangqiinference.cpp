#include "tests.h"
#include "../game/xiangqi.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../search/search.h"
#include "../search/searchnode.h"
#include "../external/nlohmann_json/json.hpp"
#include <fstream>

namespace {
void runEvaluator(const std::string& model, const nlohmann::json& input, const std::string& response);
}

// Exercise the production backend directly while the game/Search adapter is
// still separate. No postprocessing, softmax, or legality filtering is hidden.
void Tests::runXiangqiInference(const std::string& model, const std::string& request, const std::string& response) {
  using json = nlohmann::json;
  std::ifstream in(request);
  const json input = json::parse(in);
  if(input.value("evaluate",false)) { runEvaluator(model,input,response); return; }
  const int nx = input.at("width"), ny = input.at("height");
  const int batch = (int)input.at("rows").size();
  const bool nhwc = input.at("inputs_nhwc");
  const auto precision = input.at("fp16").get<bool>() ? enabled_t::True : enabled_t::False;
  ConfigParser config(input.value("backend_config",std::map<std::string,std::string>{}));
  NeuralNet::globalInitialize();
  std::unique_ptr<LoadedModel,decltype(&NeuralNet::freeLoadedModel)> loaded(
    NeuralNet::loadModelFile(model,""),NeuralNet::freeLoadedModel);
  const auto& desc = NeuralNet::getModelDesc(loaded.get());
  testAssert(desc.modelVersion == 19);
  std::unique_ptr<ComputeContext,decltype(&NeuralNet::freeComputeContext)> context(
    NeuralNet::createComputeContext({0},nullptr,nx,ny,"",precision,loaded.get(),config),NeuralNet::freeComputeContext);
  std::unique_ptr<ComputeHandle,decltype(&NeuralNet::freeComputeHandle)> handle(
    NeuralNet::createComputeHandle(context.get(),loaded.get(),nullptr,batch,input.value("require_exact",false),nhwc,0,0),NeuralNet::freeComputeHandle);
  std::unique_ptr<InputBuffers,decltype(&NeuralNet::freeInputBuffers)> buffers(
    NeuralNet::createInputBuffers(loaded.get(),batch,nx,ny),NeuralNet::freeInputBuffers);
  std::vector<std::unique_ptr<NNResultBuf>> inputs;
  std::vector<std::unique_ptr<NNOutput>> outputs;
  std::vector<NNResultBuf*> rawInputs;
  for(const auto& row : input.at("rows")) {
    auto buf = std::make_unique<NNResultBuf>();
    buf->rowSpatialBuf = row.at("spatial").get<std::vector<float>>();
    buf->rowGlobalBuf = row.at("global").get<std::vector<float>>();
    testAssert(buf->rowSpatialBuf.size() == (size_t)(nx*ny*Xiangqi::SPATIAL_FEATURES));
    testAssert(buf->rowGlobalBuf.size() == Xiangqi::GLOBAL_FEATURES);
    buf->hasRowMeta = false;
    buf->symmetry = row.at("symmetry");
    buf->policyOptimism = row.at("optimism");
    rawInputs.push_back(buf.get());
    inputs.push_back(std::move(buf));
    auto output = std::make_unique<NNOutput>();
    output->nnXLen = nx; output->nnYLen = ny;
    outputs.push_back(std::move(output));
  }
  json result = {{"scale",desc.postProcessParams.outputScaleMultiplier}, {"fp16",NeuralNet::isUsingFP16(handle.get())},
    {"policy_channels",desc.numPolicyChannels}, {"spatial_policy_channels",desc.policyHead.p2Conv.outChannels},
    {"batches",json::array()}};
  for(int count : input.at("batches").get<std::vector<int>>()) {
    testAssert(count > 0 && count <= batch);
    std::vector<NNOutput*> rawOutputs;
    for(int row = 0; row < count; row++) rawOutputs.push_back(outputs[row].get());
    NeuralNet::getOutput(handle.get(),buffers.get(),count,rawInputs.data(),rawOutputs);
    json rows = json::array();
    for(const auto* output : rawOutputs) {
      testAssert(output->getPolicyStorageSize() == Xiangqi::NUM_ACTIONS);
      rows.push_back({{"policy",std::vector<float>(output->policyProbs,output->policyProbs+Xiangqi::NUM_ACTIONS)},
        {"value",{output->whiteWinProb,output->whiteDrawProb,output->whiteLossProb,output->whiteNoResultProb}},
        {"auxiliary",{output->varTimeLeft,output->shorttermWinlossError}}});
    }
    result["batches"].push_back(rows);
  }
  std::ofstream out(response);
  out << result.dump() << std::endl;
}

namespace {
void runSearch(NNEvaluator& evaluator, NNEvaluator* human, const nlohmann::json& input, const std::string& response) {
  using json = nlohmann::json;
  Logger logger;
  SearchParams params;
  params.maxVisits = params.maxPlayouts = 96;
  params.numThreads = input.at("threads");
  params.useGumbel = input.at("gumbel");
  params.useGumbelNonroot = params.useGumbel;
  params.useGraphSearch = !params.useGumbel;
  params.useEvalCache = !params.useGumbel;
  params.rootSymmetryPruning = !params.useGumbel;
  params.rootNumSymmetriesToSample = 2;
  params.rootNoiseEnabled = true;
  params.rootPolicyTemperature = 0.85;
  params.rootPolicyTemperatureEarly = 0.85;
  params.useLcbForSelection = true;
  params.useNoisePruning = true;
  params.useUncertainty = true;
  params.cpuctUtilityStdevScale = 0.5;
  params.subtreeValueBiasFactor = 0.4;
  params.avoidRepeatedPatternUtility = 0.03;
  if(human != nullptr) {
    params.rootSymmetryPruning = false;
    params.humanSLChosenMoveProp = 1;
    params.humanSLChosenMovePiklLambda = 1e9;
    bool weightless = input.at("human_mode") == "weightless";
    params.humanSLRootExploreProbWeightless = params.humanSLPlaExploreProbWeightless =
      params.humanSLOppExploreProbWeightless = weightless ? 1 : 0;
    params.humanSLRootExploreProbWeightful = params.humanSLPlaExploreProbWeightful =
      params.humanSLOppExploreProbWeightful = weightless ? 0 : 1;
  }
  Search search(params,&evaluator,human,&logger,"xiangqi-cuda-search");
  Xiangqi::State state;
  BoardHistory history(state,BoardHistoryModes());
  search.setPosition(P_BLACK,history.getRecentBoard(0),history);
  json result = {{"turns",json::array()}};
  for(int turn = 0; turn < 3; turn++) {
    search.runWholeSearch(state.nextPlayer());
    json analysis;
    testAssert(search.getAnalysisJson(state.nextPlayer(),8,false,true,true,true,analysis));
    std::vector<Loc> locs;
    std::vector<double> target;
    testAssert(search.getPolicyTargetValues(*search.getRootNode(),locs,target));
    std::vector<int> actions;
    for(Loc loc : locs) {
      const int action = Xiangqi::locToAction(loc);
      testAssert(state.isLegal(action)); actions.push_back(action);
    }
    for(const auto& entry : analysis["moveInfos"]) {
      Xiangqi::State replay(state);
      for(const auto& move : entry["pv"]) replay.play(Xiangqi::parseAction(move.get<std::string>()));
    }
    const Loc chosen = search.getChosenMoveLoc();
    testAssert(state.isLegal(Xiangqi::locToAction(chosen)));
    result["turns"].push_back({{"analysis",analysis},{"actions",actions},{"target",target},
      {"legal",state.legalActions()},{"chosen",Xiangqi::locToAction(chosen)},
      {"symmetries",search.rootSymmetries},{"player",state.nextPlayer()}});
    if(human != nullptr) {
      testAssert(search.getPlaySelectionValues(locs,target,1.0));
      actions.clear();
      for(Loc loc : locs) actions.push_back(Xiangqi::locToAction(loc));
      result["turns"].back()["play_actions"] = actions;
      result["turns"].back()["play_values"] = target;
    }
    testAssert(search.makeMove(chosen,state.nextPlayer()));
    state.play(Xiangqi::locToAction(chosen));
    testAssert(state.historyHash() == search.getRootHist().getXiangqiState().historyHash());
    if(params.useGumbel) testAssert(search.getRootVisits() == 0);
  }
  result["rows"] = evaluator.numRowsProcessed();
  result["batches"] = evaluator.numBatchesProcessed();
  if(human != nullptr) {
    result["human_rows"] = human->numRowsProcessed();
    result["human_batches"] = human->numBatchesProcessed();
  }
  std::ofstream out(response);
  out << result.dump() << std::endl;
}

void runEvaluator(const std::string& model, const nlohmann::json& input, const std::string& response) {
  using json = nlohmann::json;
  Board::initHash();
  const int side = input.at("side");
  const bool nhwc = input.at("inputs_nhwc");
  ConfigParser config(input.value("backend_config",std::map<std::string,std::string>{}));
  NNEvaluator evaluator("xiangqi",model,"",nullptr,4,side,side,false,nhwc,8,4,false,"",
    input.at("fp16").get<bool>() ? enabled_t::True : enabled_t::False,
    2,{0,0},"xiangqi-evaluator",false,0,config);
  evaluator.spawnServerThreads();
  if(input.value("search",false)) {
    std::unique_ptr<NNEvaluator> human;
    if(input.contains("human_model")) {
      human.reset(new NNEvaluator("xiangqi-human",input.at("human_model"),"",nullptr,4,side,side,false,nhwc,8,4,false,"",
        input.at("fp16").get<bool>() ? enabled_t::True : enabled_t::False,
        2,{0,0},"xiangqi-human-evaluator",false,0,config));
      human->spawnServerThreads();
    }
    runSearch(evaluator,human.get(),input,response);
    if(human != nullptr) human->killServerThreads();
    evaluator.killServerThreads();
    return;
  }
  std::vector<Xiangqi::State> states;
  std::vector<BoardHistory> histories;
  std::vector<MiscNNInputParams> parameters;
  for(const auto& request : input.at("cases")) {
    Xiangqi::State state(request.value("fen",Xiangqi::State::START_FEN));
    BoardHistory history(state,BoardHistoryModes());
    Board board = history.getRecentBoard(0);
    for(const auto& move : request.at("moves")) {
      const int action = Xiangqi::parseAction(move.get<std::string>());
      testAssert(history.makeBoardMoveTolerant(board,Xiangqi::actionToLoc(action),state.nextPlayer()));
      state.play(action);
    }
    testAssert(history.getXiangqiState().historyHash() == state.historyHash());
    states.push_back(std::move(state));
    histories.push_back(std::move(history));
    MiscNNInputParams params;
    params.symmetry = request.value("symmetry",0);
    params.playoutDoublingAdvantage = request.value("pda",0.0);
    params.policyOptimism = request.value("optimism",0.0);
    params.nnPolicyTemperature = request.value("temperature",1.0f);
    params.maxHistory = request.value("max_history",1000);
    parameters.push_back(params);
  }
  auto capture = [&](int index, int symmetry, const NNOutput& out) {
    const auto& state = states[index];
    const auto& params = parameters[index];
    std::vector<float> spatial(Xiangqi::SPATIAL_FEATURES*side*side), globals(Xiangqi::GLOBAL_FEATURES);
    state.fillFeatures(state.nextPlayer(),params.playoutDoublingAdvantage,false,side,side,false,
                       spatial.data(),globals.data(),params.maxHistory);
    return json{{"case",index},{"spatial",spatial},{"global",globals},{"player",state.nextPlayer()},
      {"legal",state.legalActions()},{"symmetry",symmetry},{"hash",{out.nnHash.hash0,out.nnHash.hash1}},
      {"policy",std::vector<float>(out.policyProbs,out.policyProbs+Xiangqi::NUM_ACTIONS)},
      {"value",{out.whiteWinProb,out.whiteDrawProb,out.whiteLossProb,out.whiteNoResultProb}},
      {"auxiliary",{out.varTimeLeft,out.shorttermWinlossError}}};
  };
  json result = {{"cases",json::array()},{"averages",json::array()},{"parallel",json::array()}};
  NNResultBuf buf;
  for(int i = 0; i < (int)states.size(); i++) {
    const auto& history = histories[i];
    evaluator.evaluate(history.getRecentBoard(0),history,states[i].nextPlayer(),parameters[i],buf,true);
    result["cases"].push_back(capture(i,buf.symmetry,*buf.result));
    auto cached = buf.result;
    const auto rows = evaluator.numRowsProcessed(), hits = evaluator.numCacheHits();
    evaluator.evaluate(states[i],parameters[i],buf,false);
    testAssert(buf.result == cached && evaluator.numRowsProcessed() == rows && evaluator.numCacheHits() == hits+1);
    Rand rand("xiangqi-symmetry-average");
    std::unique_ptr<std::shared_ptr<NNOutput>> average(
      evaluator.averageMultipleSymmetries(history.getRecentBoard(0),history,states[i].nextPlayer(),nullptr,parameters[i],buf,rand,8));
    testAssert((*average)->nnHash == cached->nnHash);
    result["averages"].push_back(capture(i,-2,**average));
  }
  evaluator.clearCache();
  auto before = evaluator.numRowsProcessed();
  evaluator.evaluate(states[0],parameters[0],buf,false);
  testAssert(evaluator.numRowsProcessed() == before+1);
  bool rejected = false;
  try { evaluator.setDefaultSymmetry(1); } catch(const StringError&) { rejected = true; }
  testAssert(rejected);

  evaluator.setDoRandomize(true);
  const int count = 16;
  const auto batches = evaluator.numBatchesProcessed();
  before = evaluator.numRowsProcessed();
  std::vector<json> parallel(count);
  std::vector<std::exception_ptr> errors(count);
  std::vector<std::thread> threads;
  std::mutex mutex;
  std::condition_variable ready;
  int waiting = 0;
  bool start = false;
  for(int client = 0; client < count; client++) threads.emplace_back([&,client]() {
    {
      std::unique_lock<std::mutex> lock(mutex);
      waiting++; ready.notify_all();
      ready.wait(lock,[&] { return start; });
    }
    try {
      const int index = client % (int)states.size();
      auto params = parameters[index]; params.symmetry = NNInputs::SYMMETRY_NOTSPECIFIED;
      NNResultBuf local;
      if(client % 2 == 0) {
        const auto& history = histories[index];
        evaluator.evaluate(history.getRecentBoard(0),history,states[index].nextPlayer(),params,local,true);
      }
      else evaluator.evaluate(states[index],params,local,true);
      testAssert(local.symmetry == 0 || local.symmetry == 2);
      parallel[client] = capture(index,local.symmetry,*local.result);
    }
    catch(...) { errors[client] = std::current_exception(); }
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    ready.wait(lock,[&] { return waiting == count; });
    start = true;
  }
  ready.notify_all();
  for(auto& thread : threads) thread.join();
  for(const auto& error : errors) if(error) std::rethrow_exception(error);
  testAssert(evaluator.numRowsProcessed() == before+count);
  result["parallel"] = parallel;
  result["parallel_rows"] = evaluator.numRowsProcessed()-before;
  result["parallel_batches"] = evaluator.numBatchesProcessed()-batches;
  result["cache_hits"] = evaluator.numCacheHits();
  evaluator.evaluate(states[0],parameters[0],buf,true);
  const auto retained = buf.result;
  const auto rowsBeforeRestart = evaluator.numRowsProcessed(), hitsBeforeRestart = evaluator.numCacheHits();
  evaluator.killServerThreads();
  // Recreating server resources preserves the source NN cache and result lifetime.
  evaluator.spawnServerThreads();
  evaluator.evaluate(states[0],parameters[0],buf,false);
  testAssert(buf.result == retained && evaluator.numRowsProcessed() == rowsBeforeRestart);
  testAssert(evaluator.numCacheHits() == hitsBeforeRestart+1);
  evaluator.evaluate(states[0],parameters[0],buf,true);
  testAssert(buf.result != retained && buf.result->nnHash == retained->nnHash);
  result["restarted"] = capture(0,buf.symmetry,*buf.result);
  evaluator.killServerThreads();
  std::ofstream out(response);
  out << result.dump() << std::endl;
}
}
