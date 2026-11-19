#include "tests.h"
#include "testvisitcap.h"
#include "testgumbelsymmetry.h"
#include "testsearchcommon.h"
#include "../search/search.h"
#include "../search/searchnode.h"
#include "../game/xiangqi.h"
#include "../external/nlohmann_json/json.hpp"
#include <fstream>

namespace {
struct Totals {
  int nodes = 0, depth = 0;
  int64_t visits = 1, edgeCatchups = 0;
  double winLoss = 0, draw = 0, noResult = 0;
};
void close(double actual, double expected, double tolerance = 1e-9) {
  if(std::abs(actual-expected) > tolerance)
    throw StringError("MuZero search mismatch: " + Global::doubleToString(actual) + " vs " + Global::doubleToString(expected));
}
Totals checkTree(const Search& search, const SearchNode& node, MuZero::PositionEvaluator& evaluator,
                 MuZero::PositionBuffer& buffer, bool uniformWeights, double tolerance) {
  const auto* output = static_cast<const MuZero::Evaluation*>(node.getNNOutput());
  testAssert(output != nullptr && output->state && output->orientation && output->nextPlayer == node.nextPla);
  testAssert(output->getPolicyStorageSize() == search.getPolicyBufferSize());
  testAssert(node.virtualLosses.load() == 0 && node.dirtyCounter.load() == 0);
  Totals total{1,0,1,0,output->whiteWinProb-(double)output->whiteLossProb,output->whiteDrawProb,output->whiteNoResultProb};
  auto children = node.getChildren();
  int64_t edgeSum = 0;
  for(int i = 0; i < children.getCapacity(); i++) {
    const auto& edge = children[i];
    const SearchNode* child = edge.getIfAllocated();
    if(child == nullptr) break;
    testAssert(child->virtualLosses.load() == 0);
    const int64_t edgeVisits = edge.getEdgeVisits();
    edgeSum += edgeVisits;
    if(child->stats.visits.load() == 0) continue;
    const auto* childOutput = static_cast<const MuZero::Evaluation*>(child->getNNOutput());
    testAssert(childOutput != nullptr && childOutput->orientation == output->orientation);
    MiscNNInputParams params;
    params.nnPolicyTemperature = search.searchParams.nnPolicyTemperature;
    params.policyOptimism = search.searchParams.policyOptimism;
    auto reference = evaluator.recurrent(*output,search.getPos(edge.getMoveLoc()),params,buffer);
    close(childOutput->whiteWinProb,reference->whiteWinProb,tolerance);
    close(childOutput->whiteLossProb,reference->whiteLossProb,tolerance);
    close(childOutput->whiteDrawProb,reference->whiteDrawProb,tolerance);
    close(childOutput->whiteNoResultProb,reference->whiteNoResultProb,tolerance);
    for(int pos = 0; pos < search.policySize; pos++) close(childOutput->policyProbs[pos],reference->policyProbs[pos],tolerance);
    auto below = checkTree(search,*child,evaluator,buffer,uniformWeights,tolerance);
    total.nodes += below.nodes;
    total.depth = std::max(total.depth,below.depth+1);
    total.edgeCatchups += below.edgeCatchups + edgeVisits - below.visits;
    total.winLoss += edgeVisits * below.winLoss;
    total.draw += edgeVisits * below.draw;
    total.noResult += edgeVisits * below.noResult;
    if(uniformWeights) testAssert(edgeVisits >= below.visits);
  }
  total.visits += edgeSum;
  total.winLoss /= total.visits;
  total.draw /= total.visits;
  total.noResult /= total.visits;
  if(uniformWeights) {
    // Source edge catch-up can count an in-flight child's observation before
    // its original thread publishes the edge visit, even without graph search.
    // Each edge visit weights the child's mean; the node's own NN has weight 1.
    // In a single-thread tree this reduces to the flat mean of unique nodes.
    testAssert(node.stats.visits.load() == total.visits);
    testAssert(total.visits == total.nodes + total.edgeCatchups);
    close(node.stats.weightSum.load(),total.visits);
    close(node.stats.winLossValueAvg.load(),total.winLoss);
    close(node.stats.drawValueAvg.load(),total.draw);
    close(node.stats.noResultValueAvg.load(),total.noResult);
    close(node.stats.utilityAvg.load(),search.searchParams.winLossUtilityFactor*total.winLoss);
  }
  return total;
}

int checkHumanTree(const Search& search, const SearchNode& node, MuZero::PositionEvaluator* evaluator,
                   MuZero::PositionBuffer& buffer, double tolerance) {
  const auto* output = evaluator ? static_cast<const MuZero::Evaluation*>(node.getHumanOutput()) : nullptr;
  if(evaluator) {
    testAssert(output && output->state && output->orientation && output->nextPlayer == node.nextPla);
    const auto* primary = static_cast<const MuZero::Evaluation*>(node.getNNOutput());
    testAssert(MuZero::stateOwner(output->state) != MuZero::stateOwner(primary->state));
  }
  int depth = 0;
  auto children = node.getChildren();
  for(int i = 0; i < children.getCapacity(); i++) {
    const auto* child = children[i].getIfAllocated();
    if(!child) break;
    if(!child->getNNOutput()) continue;
    if(evaluator) {
      const auto* actual = static_cast<const MuZero::Evaluation*>(child->getHumanOutput());
      testAssert(actual && actual->orientation == output->orientation);
      MiscNNInputParams params;
      params.nnPolicyTemperature = search.searchParams.nnPolicyTemperature;
      params.policyOptimism = search.searchParams.policyOptimism;
      const auto expected = evaluator->recurrent(*output,search.getPos(children[i].getMoveLoc()),params,buffer);
      close(actual->whiteWinProb,expected->whiteWinProb,tolerance);
      close(actual->whiteLossProb,expected->whiteLossProb,tolerance);
      close(actual->whiteDrawProb,expected->whiteDrawProb,tolerance);
      close(actual->whiteNoResultProb,expected->whiteNoResultProb,tolerance);
      for(int p = 0; p < search.policySize; p++) close(actual->policyProbs[p],expected->policyProbs[p],tolerance);
    }
    else testAssert(child->getHumanOutput() == nullptr);
    depth = std::max(depth,1+checkHumanTree(search,*child,evaluator,buffer,tolerance));
  }
  return depth;
}

void checkHumanChoice(Search& search) {
  std::vector<Loc> targetLocs; std::vector<double> target;
  testAssert(search.getPolicyTargetValues(*search.rootNode,targetLocs,target));
  search.searchParams.humanSLChosenMoveProp = 1;
  search.searchParams.humanSLChosenMovePiklLambda = 1e9;
  std::vector<Loc> locs; std::vector<double> weights;
  testAssert(search.getPlaySelectionValues(locs,weights,0));
  const auto* human = search.rootNode->getHumanOutput();
  testAssert(human);
  double total = std::accumulate(weights.begin(),weights.end(),0.0), prior = 0;
  for(Loc loc : locs) prior += human->policyProbs[search.getPos(loc)];
  for(size_t i = 0; i < locs.size(); i++)
    close(weights[i]/total,human->policyProbs[search.getPos(locs[i])]/prior,1e-8);
  if(search.searchParams.useGumbel) {
    std::vector<Loc> afterLocs; std::vector<double> after;
    testAssert(search.getPolicyTargetValues(*search.rootNode,afterLocs,after));
    testAssert(targetLocs == afterLocs && target == after);
    const auto candidates = search.getGumbelChildren(*search.rootNode);
    int64_t edges = 0;
    for(size_t i = 0; i < candidates.size(); i++) {
      testAssert(candidates[i].visits == search.gumbelRoot->visits()[i] && candidates[i].pending == 0);
      edges += candidates[i].visits;
    }
    testAssert(search.getRootVisits() == edges+1 && search.gumbelRoot->pending() == 0);
  }
}

int checkBoardHumanTree(const Search& search, const SearchNode& node, Board board, BoardHistory history,
                       MuZero::PositionEvaluator& evaluator, MuZero::PositionBuffer& buffer, Rand& rand, double tolerance) {
  if(!node.getNNOutput()) return 0; // Source real terminal nodes need no prediction.
  const auto* actual = static_cast<const MuZero::Evaluation*>(node.getHumanOutput());
  testAssert(actual && actual->state && actual->nextPlayer == node.nextPla);
  MiscNNInputParams params;
  params.symmetry = actual->orientation->symmetry;
  params.nnPolicyTemperature = search.searchParams.nnPolicyTemperature;
  params.policyOptimism = &node == search.rootNode ? search.searchParams.rootPolicyOptimism : search.searchParams.policyOptimism;
  const auto modes = Search::resolveHistoryModes(search.searchParams,evaluator);
  params.passAliveSuicideRulesOverride = modes.alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
  params.excludeTerritoryAdjAtariOverride = modes.excludeTerritoryAdjacentToAtari ? 1 : 0;
  // A secondary MuZero in ordinary search predicts each actual node from its
  // real observation. Compare with replayed rules, not learned recurrence.
  auto expected = evaluator.initial(board,history,node.nextPla,&search.searchParams.humanSLProfile,params,buffer,rand,true);
  close(actual->whiteWinProb,expected->whiteWinProb,tolerance);
  close(actual->whiteLossProb,expected->whiteLossProb,tolerance);
  for(int p = 0; p < search.policySize; p++) close(actual->policyProbs[p],expected->policyProbs[p],tolerance);
  int count = 1;
  auto children = node.getChildren();
  for(int i = 0; i < children.getCapacity(); i++) {
    const auto* child = children[i].getIfAllocated();
    if(!child) break;
    Board nextBoard(board); BoardHistory nextHistory(history);
    testAssert(nextHistory.isLegal(nextBoard,children[i].getMoveLoc(),node.nextPla));
    nextHistory.makeBoardMoveAssumeLegal(nextBoard,children[i].getMoveLoc(),node.nextPla,nullptr);
    count += checkBoardHumanTree(search,*child,nextBoard,nextHistory,evaluator,buffer,rand,tolerance);
  }
  return count;
}
}

void Tests::runMuZeroSearch(const std::string& directory, const std::string& request, const std::string& response) {
  using json = nlohmann::json;
  Board::initHash();
  std::ifstream in(request);
  const json input = json::parse(in);
  const bool fp16 = input.at("fp16").get<bool>();
  ConfigParser config(input.value("cuda_config",std::map<std::string,std::string>{}));
  Logger logger(nullptr,false,false,false);
  auto model = std::make_shared<MuZero::ModelBundle>(directory);
  MuZero::Batcher batcher(model,8,{0,0},fp16 ? enabled_t::True : enabled_t::False,config,nullptr,input.value("inputs_nhwc",true));
  MuZero::PositionEvaluator evaluator(batcher,false,true,0,4,2);
  MuZero::PositionBuffer buffer;
  const bool go = model->game() == "go";
  const bool xiangqi = model->game() == "xiangqi";
  Board board(xiangqi ? 9 : 4,xiangqi ? 10 : 4);
  Rules rules = Rules::forGame(Rules::parseGame(model->game()));
  if(go) { rules.koRule = Rules::KO_SIMPLE; rules.scoringRule = Rules::SCORING_TERRITORY; }
  BoardHistory history = xiangqi ? BoardHistory(Xiangqi::State(),BoardHistoryModes(false,false)) :
    BoardHistory(board,P_BLACK,rules,0,BoardHistoryModes(false,false));
  if(xiangqi) board = history.getRecentBoard(0);
  history.makeBoardMoveAssumeLegal(board,xiangqi ? Xiangqi::actionToLoc(Xiangqi::parseAction("a3a4")) : Location::getLoc(0,0,4),P_BLACK,nullptr);
  history.makeBoardMoveAssumeLegal(board,xiangqi ? Xiangqi::actionToLoc(Xiangqi::parseAction("a6a5")) : Location::getLoc(1,0,4),P_WHITE,nullptr);
  const Loc repeat = xiangqi ? Xiangqi::actionToLoc(Xiangqi::parseAction("b0c2")) : go ? Board::PASS_LOC : Location::getLoc(1,1,4);
  const auto boardHash = board.pos_hash;
  json results = json::array();
  if(input.value("symmetry",false)) {
    Board empty(xiangqi ? 9 : 4,xiangqi ? 10 : 4);
    BoardHistory emptyHistory = xiangqi ? BoardHistory(Xiangqi::State(),BoardHistoryModes()) :
      BoardHistory(empty,P_BLACK,rules,0,BoardHistoryModes());
    if(xiangqi) empty = emptyHistory.getRecentBoard(0);
    for(int threads : {1,4}) for(bool interior : {false,true}) {
      SearchParams params; params.useGumbel = true; params.useGumbelNonroot = interior;
      params.numThreads = threads; params.gumbel.max_num_considered_actions = 4;
      Search search(params,evaluator,evaluator,&logger,"muzero-gumbel-symmetry");
      TestGumbelSymmetry::exercise(search,empty,emptyHistory);
    }
    std::ofstream out(response); out << "true";
    return;
  }
  if(input.contains("humanBundle")) {
    auto humanModel = std::make_shared<MuZero::ModelBundle>(input.at("humanBundle").get<std::string>());
    MuZero::Batcher humanBatcher(humanModel,8,{0,0},fp16 ? enabled_t::True : enabled_t::False,config,nullptr,input.value("inputs_nhwc",true));
    const int humanSymmetry = xiangqi ? 2 : model->game() == "hex" ? 3 : 1;
    MuZero::PositionEvaluator human(humanBatcher,false,false,humanSymmetry,4,2);
    MuZero::PositionBuffer humanBuffer;
    const double tolerance = fp16 ? 4e-3 : 5e-5;
    evaluator.setDoRandomize(false);
    evaluator.setDefaultSymmetry(0);
    const bool metadata = humanModel->representationDesc().numInputMetaChannels > 0;
    if(metadata) {
      SearchParams params; params.maxVisits = params.maxPlayouts = 8;
      params.humanSLPlaExploreProbWeightful = params.humanSLOppExploreProbWeightful = 1;
      Search search(params,evaluator,human,&logger,"inactive-human-profile");
      search.setPosition(P_BLACK,board,history);
      search.runWholeSearch(P_BLACK);
      testAssert(search.rootNode->getHumanOutput() == nullptr && humanBatcher.stats().rows == 0);
    }
    for(int algorithm : {0,1,2}) for(int threads : {1,4}) for(int mode : {0,1,2,3,4,5}) {
      evaluator.clearCache(); human.clearCache();
      SearchParams params;
      params.useGumbel = algorithm > 0; params.useGumbelNonroot = algorithm == 2;
      params.numThreads = threads; params.maxVisits = params.maxPlayouts = 32;
      params.useUncertainty = params.useNoisePruning = false; params.valueWeightExponent = 0;
      params.gumbel.max_num_considered_actions = 4;
      params.nnPolicyTemperature = .8; params.rootPolicyOptimism = .4; params.policyOptimism = .2;
      params.rootNumSymmetriesToSample = mode == 4 ? 2 : 1;
      params.humanSLRootExploreProbWeightless = mode == 1 ? 1 : mode == 4 ? .3 : 0;
      params.humanSLPlaExploreProbWeightless = params.humanSLOppExploreProbWeightless = mode == 2 ? 1 : mode == 4 ? .2 : 0;
      params.humanSLPlaExploreProbWeightful = params.humanSLOppExploreProbWeightful = mode == 3 || mode == 5 ? 1 : mode == 4 ? .3 : 0;
      if(metadata) params.humanSLProfile = SGFMetadata::getProfile("rank_5d");
      Search search(params,evaluator,human,&logger,"independent-human-latent");
      search.setPosition(P_BLACK,board,history);
      if(mode == 5) {
        // Repeat the same point/pass/absolute Xiangqi action beyond real legality.
        std::vector<int> avoid(search.getMoveLocBufferSize(),100); avoid[repeat] = 0;
        search.setAvoidMoveUntilByLoc(avoid,avoid);
      }
      search.runWholeSearch(P_BLACK);
      testAssert(search.getRootBoard().pos_hash == boardHash && search.getRootHist().moveHistory.size() == history.moveHistory.size());
      testAssert(search.lastSearchNumPlayouts >= 32 && search.lastSearchNumPlayouts < 32+threads);
      if(mode == 1) testAssert(search.getRootVisits() == 1);
      if(mode == 0 || mode == 3 || mode == 5) testAssert(search.getRootVisits() >= 32);
      checkTree(search,*search.rootNode,evaluator,buffer,mode == 0 || mode == 3 || mode == 5,tolerance);
      const int depth = checkHumanTree(search,*search.rootNode,mode >= 2 ? &human : nullptr,humanBuffer,tolerance);
      if(mode == 5) testAssert(depth > 2);
      if(mode != 4) {
        const auto* actual = static_cast<const MuZero::Evaluation*>(search.rootNode->getHumanOutput());
        const auto* primary = static_cast<const MuZero::Evaluation*>(search.rootNode->getNNOutput());
        testAssert(actual && actual->orientation->symmetry == humanSymmetry && primary->orientation->symmetry == 0);
        MiscNNInputParams inputParams;
        inputParams.nnPolicyTemperature = params.nnPolicyTemperature; inputParams.policyOptimism = params.rootPolicyOptimism;
        const auto modes = Search::resolveHistoryModes(params,human);
        inputParams.passAliveSuicideRulesOverride = modes.alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
        inputParams.excludeTerritoryAdjAtariOverride = modes.excludeTerritoryAdjacentToAtari ? 1 : 0;
        Rand rand("human-root-reference");
        auto expected = human.initial(board,search.getRootHist(),P_BLACK,&params.humanSLProfile,
          inputParams,humanBuffer,rand,true);
        for(int p = 0; p < search.policySize; p++) close(actual->policyProbs[p],expected->policyProbs[p],tolerance);
      }
      checkHumanChoice(search);
      results.push_back({{"algorithm",algorithm},{"threads",threads},{"mode",mode},{"depth",depth}});
      if(mode == 3) {
        testAssert(search.makeMove(search.getChosenMoveLoc(),P_BLACK));
        testAssert(search.rootNode == nullptr);
        search.runWholeSearch(P_WHITE);
        checkHumanTree(search,*search.rootNode,&human,humanBuffer,tolerance);
      }
    }
    NeuralNet::globalInitialize();
    auto ordinary = std::unique_ptr<NNEvaluator>(TestSearchCommon::startNNEval(input.at("humanBoardModel").get<std::string>(),
      logger,"human-board",model->posLen(),model->posLen(),humanSymmetry,true,true,fp16,false,false));
    for(int algorithm : {0,1,2}) for(int threads : {1,4}) {
      SearchParams params; params.useGumbel = algorithm > 0; params.useGumbelNonroot = algorithm == 2;
      params.numThreads = threads; params.maxVisits = params.maxPlayouts = 24;
      params.humanSLPlaExploreProbWeightful = params.humanSLOppExploreProbWeightful = 1;
      params.nnPolicyTemperature = .8; params.rootPolicyOptimism = .4; params.policyOptimism = .2;
      if(metadata) params.humanSLProfile = SGFMetadata::getProfile("rank_5d");
      Search search(params,ordinary.get(),human,&logger,"ordinary-tree-human-muzero");
      search.setPosition(P_BLACK,board,history);
      search.runWholeSearch(P_BLACK);
      Rand rand("real-secondary-reference");
      testAssert(checkBoardHumanTree(search,*search.rootNode,search.getRootBoard(),search.getRootHist(),human,humanBuffer,rand,tolerance) > 2);
      testAssert(search.makeMove(search.getChosenMoveLoc(),P_BLACK));
      search.runWholeSearch(P_WHITE);
      testAssert(checkBoardHumanTree(search,*search.rootNode,search.getRootBoard(),search.getRootHist(),human,humanBuffer,rand,tolerance) > 2);
    }
    for(int algorithm : {0,1,2}) for(int mode : {0,1}) {
      SearchParams params; params.useGumbel = algorithm > 0; params.useGumbelNonroot = algorithm == 2;
      params.maxVisits = params.maxPlayouts = 32; params.numThreads = 4;
      params.humanSLRootExploreProbWeightless = mode;
      if(metadata) params.humanSLProfile = SGFMetadata::getProfile("rank_5d");
      AsyncBot bot(params,evaluator,ordinary.get(),&logger,"ordinary-root-human");
      bot.setPosition(P_BLACK,board,history);
      testAssert(history.isLegal(board,bot.genMoveSynchronous(P_BLACK,TimeControls()),P_BLACK));
      Search* search = bot.getSearchStopAndWait();
      checkHumanTree(*search,*search->rootNode,nullptr,humanBuffer,tolerance);
      checkHumanChoice(*search);
      for(int forbidden = 0; forbidden < 4; forbidden++) {
        auto invalid = params;
        if(forbidden == 0) invalid.humanSLPlaExploreProbWeightless = .1;
        if(forbidden == 1) invalid.humanSLPlaExploreProbWeightful = .1;
        if(forbidden == 2) invalid.humanSLOppExploreProbWeightless = .1;
        if(forbidden == 3) invalid.humanSLOppExploreProbWeightful = .1;
        bool rejected = false;
        try { search->setParams(invalid); }
        catch(const StringError& e) { rejected = std::string(e.what()).find("ordinary HumanSL is root-only") != std::string::npos; }
        testAssert(rejected);
      }
    }
    {
      SearchParams params; params.maxVisits = params.maxPlayouts = 8;
      if(metadata) params.humanSLProfile = SGFMetadata::getProfile("rank_5d");
      params.humanSLRootExploreProbWeightful = 1;
      AsyncBot bot(params,evaluator,human,&logger,"async-human-latent");
      bot.setPosition(P_BLACK,board,history);
      testAssert(history.isLegal(board,bot.genMoveSynchronous(P_BLACK,TimeControls()),P_BLACK));
      Search* search = bot.getSearchStopAndWait();
      testAssert(search->getRootVisits() == 8);
      params.useGumbel = true;
      search->setParams(params);
      bool rejected = false;
      try { search->beginSearch(false); }
      catch(const StringError& e) { rejected = std::string(e.what()).find("root weightful HumanSL") != std::string::npos; }
      testAssert(rejected);
    }
    std::ofstream out(response); out << results.dump(2);
    return;
  }
  for(int mode : {0,1,2}) for(int threads : {1,4}) {
    SearchParams params;
    params.useGumbel = mode > 0; params.useGumbelNonroot = mode == 2;
    params.numThreads = threads; params.maxVisits = params.maxPlayouts = 96;
    params.visitCapContempt = 4; params.gumbel.max_num_considered_actions = 2;
    params.valueWeightExponent = 0; params.useUncertainty = params.useNoisePruning = false;
    Search capped(params,evaluator,&logger,"muzero-visit-cap");
    capped.setPosition(P_BLACK,board,history);
    //Concentrate root work so even a nearly uniform untrained policy reaches
    //the cap well inside this small budget. Interior action domains stay full.
    std::vector<int> rootAvoid(capped.getMoveLocBufferSize(),1);
    rootAvoid[repeat] = 0;
    capped.setAvoidMoveUntilByLoc(rootAvoid,{});
    capped.runWholeSearch(P_BLACK);
    testAssert(capped.getRootVisits() >= 96 && capped.getRootVisits() < 96+threads);
    testAssert(TestVisitCap::checkTree(capped) > 0);
    checkTree(capped,*capped.rootNode,evaluator,buffer,true,fp16 ? 4e-3 : 2e-5);
  }
  for(int mode = 0; mode < 3; mode++) for(int threads: {1,4}) for(bool chain: {false,true}) {
    SearchParams params;
    params.useGumbel = mode > 0; params.useGumbelNonroot = mode == 2;
    params.numThreads = threads; params.maxVisits = 32; params.maxPlayouts = 32;
    params.useUncertainty = false; params.useNoisePruning = false; params.valueWeightExponent = 0;
    params.rootNoiseEnabled = false; params.chosenMoveSubtract = 0; params.chosenMovePrune = 0;
    params.nnPolicyTemperature = .8f; params.rootPolicyOptimism = .4; params.policyOptimism = .2;
    params.gumbel.max_num_considered_actions = 5;
    Search search(params,evaluator,&logger,"muzero-tree");
    search.setPosition(P_BLACK,board,history);
    if(chain) {
      std::vector<int> avoid(search.getMoveLocBufferSize(),100);
      avoid[repeat] = 0;
      search.setAvoidMoveUntilByLoc(avoid,avoid);
    }
    search.runWholeSearch(P_BLACK);
    testAssert(search.getRootBoard().pos_hash == boardHash && search.getRootHist().moveHistory.size() == history.moveHistory.size());
    testAssert(search.getRootVisits() >= 32 && search.getRootVisits() <= 32+threads);
    if(params.useGumbel) testAssert(search.getRootVisits() == 32 && search.gumbelRoot->pending() == 0);
    auto totals = checkTree(search,*search.rootNode,evaluator,buffer,true,fp16 ? .003 : 5e-5);
    if(threads == 1) testAssert(totals.nodes == 32 && totals.edgeCatchups == 0);
    // Pass/pass and repeated occupied points never stop latent search. Parallel
    // catch-up visits do not require expanding another unique latent node.
    if(chain) testAssert(totals.depth == totals.nodes-1 && totals.depth > 2);
    Loc chosen = search.getChosenMoveLoc();
    testAssert(history.isLegal(board,chosen,P_BLACK));
    if(chain) testAssert(chosen == repeat);
    std::vector<Loc> locs, pv;
    std::vector<double> values;
    std::vector<int64_t> pvVisits,pvEdges;
    testAssert(search.getPolicyTargetValues(*search.rootNode,locs,values));
    for(Loc loc: locs) testAssert(history.isLegal(board,loc,P_BLACK));
    search.appendPV(pv,pvVisits,pvEdges,locs,values,search.rootNode,40);
    testAssert(!pv.empty());
    if(chain) testAssert(pv.size() == (size_t)totals.depth);
    std::vector<AnalysisData> analysis;
    search.getAnalysisData(*search.rootNode,analysis,0,false,40,false);
    testAssert(!analysis.empty());
    json analysisJson;
    testAssert(search.getAnalysisJson(P_WHITE,40,true,true,true,true,analysisJson));
    if(chain) testAssert(analysisJson.at("moveInfos").at(0).at("pv").size() == (size_t)totals.depth);
    results.push_back({{"mode",mode},{"threads",threads},{"chain",chain},{"nodes",totals.nodes},
                       {"depth",totals.depth},{"visits",totals.visits},{"edgeCatchups",totals.edgeCatchups}});
    search.runWholeSearch(P_BLACK);
    testAssert(search.getRootVisits() >= 32 && search.getRootVisits() <= 32+threads);
    testAssert(search.makeMove(search.getChosenMoveLoc(),P_BLACK));
    testAssert(search.rootNode == nullptr && search.getRootVisits() == 0);
    if(xiangqi) search.setAvoidMoveUntilByLoc({},{});
    search.runWholeSearch(P_WHITE);
    testAssert(search.getRootVisits() >= 32);
  }
  {
    // Deterministically replay the source catch-up interleaving: A has evaluated
    // a child but not published its edge; B catches up, then A publishes too.
    SearchParams params;
    params.useUncertainty = false; params.useNoisePruning = false; params.valueWeightExponent = 0;
    params.graphSearchCatchUpLeakProb = 0;
    Search search(params,evaluator,&logger,"muzero-edge-catchup");
    search.setPosition(P_BLACK,board,history);
    std::vector<int> avoid(search.getMoveLocBufferSize(),100);
    avoid[repeat] = 0;
    search.setAvoidMoveUntilByLoc(avoid,avoid);
    search.beginSearch(false);
    SearchThread thread(0,search);
    search.runSinglePlayout(thread,100);
    search.runSinglePlayout(thread,100);
    SearchNode& root = *search.rootNode;
    auto children = root.getChildren();
    SearchChildPointer& edge = children[0];
    const SearchNode* child = edge.getIfAllocated();
    testAssert(root.stats.visits.load() == 2 && child != nullptr && child->stats.visits.load() == 1);
    edge.setEdgeVisits(0);
    root.stats.visits.store(1);
    search.runSinglePlayout(thread,100);
    testAssert(root.stats.visits.load() == 2 && edge.getEdgeVisits() == 1 && child->stats.visits.load() == 1);
    // Publish A's pending visit, then use a normal playout to recompute backup.
    edge.addEdgeVisits(1);
    root.stats.visits.fetch_add(1);
    search.runSinglePlayout(thread,100);
    auto totals = checkTree(search,root,evaluator,buffer,true,fp16 ? .003 : 5e-5);
    testAssert(totals.nodes == 3 && totals.visits == 4 && totals.edgeCatchups == 1);
    const auto* own = root.getNNOutput();
    const auto* other = child->getNNOutput();
    const auto* grandchild = child->getChildren()[0].getIfAllocated()->getNNOutput();
    close(root.stats.winLossValueAvg.load(),
          ((double)own->whiteWinProb-own->whiteLossProb + 1.5*((double)other->whiteWinProb-other->whiteLossProb +
           (double)grandchild->whiteWinProb-grandchild->whiteLossProb))/4);
  }
  {
    // Exercise the source root-output copying path and adaptive value weights.
    SearchParams params;
    params.numThreads = 4; params.maxVisits = 48; params.maxPlayouts = 48;
    params.rootNoiseEnabled = true; params.rootPolicyTemperature = .7; params.rootPolicyTemperatureEarly = .7;
    params.rootNumSymmetriesToSample = 2; params.rootSymmetryPruning = true;
    params.rootDesiredPerChildVisitsCoeff = 2; params.useUncertainty = true; params.useNoisePruning = true;
    params.useLcbForSelection = true;
    Search search(params,evaluator,&logger,"muzero-options");
    search.setPosition(P_BLACK,board,history);
    search.setRootHintLoc(xiangqi ? repeat : Location::getLoc(1,1,4));
    search.setRootFocus({xiangqi ? Xiangqi::actionToLoc(Xiangqi::parseAction("c3c4")) : Location::getLoc(1,2,4)},{1.0},.15);
    search.runWholeSearch(P_BLACK);
    testAssert(search.rootNode->getNNOutput()->noisedPolicyProbs != nullptr);
    checkTree(search,*search.rootNode,evaluator,buffer,false,fp16 ? .003 : 5e-5);
    testAssert(history.isLegal(board,search.getChosenMoveLoc(),P_BLACK));
  }
  for(int excluded = 0; excluded < 3; excluded++) {
    SearchParams params;
    if(excluded == 0) params.useGraphSearch = true;
    if(excluded == 1) params.subtreeValueBiasFactor = .5;
    if(excluded == 2) params.avoidRepeatedPatternUtility = .1;
    bool rejected = false;
    try { Search invalid(params,evaluator,&logger,"muzero-exclusions"); }
    catch(const StringError&) { rejected = true; }
    testAssert(rejected);
    Search valid(SearchParams(),evaluator,&logger,"muzero-setter-exclusions");
    for(bool clearing: {false,true}) {
      rejected = false;
      try {
        if(clearing) valid.setParams(params);
        else valid.setParamsNoClearing(params);
      } catch(const StringError&) { rejected = true; }
      testAssert(rejected);
      testAssert(!valid.searchParams.useGraphSearch && valid.searchParams.subtreeValueBiasFactor == 0 &&
                 valid.searchParams.avoidRepeatedPatternUtility == 0);
    }
  }
  {
    SearchParams params;
    Search search(params,evaluator,&logger,"muzero-cache-legality");
    search.setPosition(P_BLACK,board,history);
    search.beginSearch(false);
    SearchThread thread(0,search);
    search.runSinglePlayout(thread,100);
    // Simulate the source's rare cache/legality mismatch at an occupied point.
    // It must not publish a new latent into the existing search tree.
    auto* output = search.rootNode->getNNOutput();
    std::fill(output->policyProbs,output->policyProbs+output->getPolicyStorageSize(),-1);
    output->policyProbs[0] = 1;
    bool rejected = false;
    try { search.runSinglePlayout(thread,100); }
    catch(const StringError& e) { rejected = std::string(e.what()).find("replace an active root latent") != std::string::npos; }
    testAssert(rejected);
    evaluator.clearCache();
  }
  if(xiangqi) {
    for(const char* fen : {"3Rk4/3R5/9/9/4P4/9/9/9/9/4K4 b - - 0 1",
                           "4k4/3R1R3/9/9/4P4/9/9/9/9/4K4 b - - 0 1",
                           "5k3/9/9/9/9/9/9/9/9/4K4 w - - 0 1"}) {
      BoardHistory terminal(Xiangqi::State(fen),BoardHistoryModes(false,false));
      testAssert(terminal.isGameFinished);
      SearchParams params; params.maxVisits = 2; params.maxPlayouts = 2;
      Search search(params,evaluator,&logger,"muzero-xiangqi-real-terminal");
      search.setPosition(terminal.presumedNextMovePla,terminal.getRecentBoard(0),terminal);
      const auto rows = batcher.stats().rows;
      search.runWholeSearch(terminal.presumedNextMovePla);
      testAssert(batcher.stats().rows == rows && search.rootNode->getNNOutput() == nullptr);
      close(search.rootNode->stats.drawValueAvg.load(),terminal.winner == C_EMPTY ? 1.0 : 0.0);
      close(search.rootNode->stats.winLossValueAvg.load(),terminal.winner == P_WHITE ? 1.0 : terminal.winner == P_BLACK ? -1.0 : 0.0);
      close(search.rootNode->stats.noResultValueAvg.load(),0.0);
    }
  }
  else if(!go) {
    Board terminalBoard(4,4);
    BoardHistory terminal(terminalBoard,P_BLACK,rules,0,BoardHistoryModes(false,false));
    for(int pos = 0; pos < 16 && !terminal.isGameFinished; pos++)
      terminal.makeBoardMoveAssumeLegal(terminalBoard,Location::getLoc(pos%4,pos/4,4),terminal.presumedNextMovePla,nullptr);
    testAssert(terminal.isGameFinished);
    SearchParams params; params.maxVisits = 2; params.maxPlayouts = 2;
    Search search(params,evaluator,&logger,"muzero-real-terminal");
    search.setPosition(terminal.presumedNextMovePla,terminalBoard,terminal);
    const auto rows = batcher.stats().rows;
    search.runWholeSearch(terminal.presumedNextMovePla);
    testAssert(batcher.stats().rows == rows && search.rootNode->getNNOutput() == nullptr);
    close(search.rootNode->stats.drawValueAvg.load(),terminal.winner == C_EMPTY ? 1.0 : 0.0);
  }
  batcher.finish();
  std::ofstream out(response);
  out << results.dump() << '\n';
}
