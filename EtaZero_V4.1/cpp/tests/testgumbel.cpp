#include "tests.h"
#include "testsearchcommon.h"
#include "testvisitcap.h"
#include "testgumbelsymmetry.h"
#include "../game/xiangqi.h"
#include "../search/gumbel.h"
#include "../search/searchnode.h"
#include "../program/play.h"
#include "../core/test.h"
#include "../search/analysisdata.h"

using namespace Gumbel;
namespace {
void near(double a,double b) { testAssert(std::fabs(a-b) < 1e-11); }
void mathAndScheduling() {
  GumbelSettings s; s.rescale_q_values = false; s.c_visit = 8; s.c_scale = 0.1; s.noise_scale = 0;
  std::vector<GumbelChild> children{{0.5,0.7,2,0},{0.25,0,0,0},{0.25,0,0,0}};
  auto q = gumbel_completed_q(children,0.1,s);
  near(q[0],0.7); near(q[1],0.5); near(q[2],0.5);
  auto pi = gumbel_policy(children,0.1,s);
  near(pi[0],1/(1+std::exp(-0.2))); near(pi[1],pi[2]);
  testAssert(gumbel_interior_selection({{0.6,0,1,0},{0.4,0,0,0}},0,s) == 1);
  testAssert(gumbel_interior_selection({{0.6,0,0,1},{0.4,0,0,0}},0,s) == 1);
  testAssert(gumbel_interior_selection({{-1,10,100,5},{0.6,0,1,0},{0.4,0,0,0}},0,s) == 2);
  testAssert(gumbel_interior_selection({{-1,0,0,0}},0,s) == -1);
  auto beforePending = gumbel_policy(children,0.1,s);
  children[0].pending = 7;
  testAssert(beforePending == gumbel_policy(children,0.1,s));
  for(int actions : {1,2,3,5,16}) for(int budget : {0,1,2,3,5,7,16,31,64,129}) {
    Rand rand("gumbel");
    std::vector<double> prior(actions,1.0/actions), scores(actions,0);
    GumbelRoot scheduler(prior,budget,s,rand);
    int completed = 0;
    while(!scheduler.done()) {
      std::vector<int> batch;
      for(int i = 0; i < 4; i++) {
        int a = scheduler.reserve(scores); if(a < 0) break; batch.push_back(a);
      }
      testAssert(!batch.empty());
      // Cancel a reservation at a phase boundary and reissue it without changing the budget.
      if(completed == 0) { scheduler.cancel(batch.back(),true); batch.pop_back(); int a = scheduler.reserve(scores); testAssert(a >= 0); batch.push_back(a); }
      for(auto it = batch.rbegin(); it != batch.rend(); ++it) { scheduler.commit(*it); completed++; }
    }
    testAssert(completed == budget && scheduler.pending() == 0);
    testAssert(scheduler.winner(scores) >= 0);
    testAssert(std::accumulate(scheduler.visits().begin(),scheduler.visits().end(),int64_t(0)) == budget);
  }
}
void sourceSearch() {
  NeuralNet::globalInitialize();
  Logger logger(nullptr,false,false,false);
  auto nn = std::unique_ptr<NNEvaluator>(TestSearchCommon::startNNEval("/dev/null",logger,"gumbel",5,5,0,true,false,false,true,false));
  for(auto game : {Rules::Game::GO,Rules::Game::RENJU,Rules::Game::HEX}) for(int threads : {1,4}) for(int budget : {1,2,7,32}) for(bool interior : {false,true}) {
    SearchParams params; params.useGumbel = true; params.useGraphSearch = false; params.numThreads = threads;
    params.useGumbelNonroot = interior;
    params.maxVisits = budget; params.maxPlayouts = 100; params.gumbel.noise_scale = 0;
    params.gumbel.max_num_considered_actions = 5;
    params.rootNoiseEnabled = true; params.rootDesiredPerChildVisitsCoeff = 10;
    params.useLcbForSelection = true; params.chosenMoveSubtract = 1000; params.chosenMovePrune = 1000;
    params.rootPolicyTemperature = 10; params.rootPolicyTemperatureEarly = 10;
    Search search(params,nn.get(),&logger,"gumbel-search");
    Board board(5,5); BoardHistory hist(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    search.setPosition(P_BLACK,board,hist);
    search.runWholeSearch(P_BLACK);
    testAssert(search.getRootVisits() == budget && search.lastSearchNumPlayouts == budget);
    testAssert(search.gumbelRoot && search.gumbelRoot->pending() == 0);
    testAssert(hist.isLegal(board,search.getChosenMoveLoc(),P_BLACK));
    int64_t edgeSum = 0;
    auto children = search.getGumbelChildren(*search.rootNode);
    for(const auto& c : children) { edgeSum += c.visits; testAssert(c.pending == 0); }
    testAssert(edgeSum == budget-1);
    std::vector<Loc> reportLocs, pv;
    std::vector<double> reportValues, reportVisits;
    std::vector<int64_t> pvVisits, pvEdgeVisits;
    testAssert(search.getPlaySelectionValues(reportLocs,reportValues,&reportVisits,0.0));
    if(budget > 1) {
      testAssert(reportValues == reportVisits);
      near(std::accumulate(reportVisits.begin(),reportVisits.end(),0.0),budget-1);
      std::vector<AnalysisData> analysis;
      search.getAnalysisData(*search.rootNode,analysis,0,false,5,false);
      testAssert(!analysis.empty() && analysis.front().move == search.getChosenMoveLoc());
    }
    search.appendPV(pv,pvVisits,pvEdgeVisits,reportLocs,reportValues,search.rootNode,5);
    testAssert(!pv.empty() && pv.front() == search.getChosenMoveLoc());
    std::vector<Loc> locs; std::vector<double> target;
    testAssert(search.getPolicyTargetValues(*search.rootNode,locs,target));
    near(std::accumulate(target.begin(),target.end(),0.0),1.0);
    testAssert(locs.size() == (game == Rules::Game::GO ? 26 : 25));
    std::vector<PolicyTargetMove> packed;
    Play::extractPolicyTarget(packed,&search,search.rootNode,locs,target);
    int maxPacked = 0; for(const auto& p : packed) maxPacked = std::max(maxPacked,(int)p.policyTarget);
    testAssert(maxPacked == 30000);
    search.runWholeSearch(P_BLACK); // Repeated analysis uses a fresh budget, not reused root counts.
    testAssert(search.getRootVisits() == budget);
    Loc selected = search.getChosenMoveLoc();
    testAssert(search.makeMove(selected,P_BLACK));
    testAssert(search.getRootVisits() == 0 && !search.gumbelRoot && search.getChosenMoveLoc() == Board::NULL_LOC);
    search.runWholeSearch(P_WHITE);
    testAssert(search.getRootVisits() == budget && search.gumbelRoot->pending() == 0);
  }
  {
    // A controlled nonroot position tests the actual selection hook, including depth-dependent avoids.
    SearchParams params; params.useGumbel = true; params.useGumbelNonroot = true; params.useGraphSearch = false;
    params.gumbel.rescale_q_values = false;
    Search search(params,nn.get(),&logger,"interior-hook");
    Board board(5,5); BoardHistory hist(board,P_BLACK,Rules::forGame(Rules::Game::HEX),0,BoardHistoryModes());
    search.setPosition(P_BLACK,board,hist);
    SearchThread thread(0,search);
    SearchNode node(P_BLACK,false,0,Hash128());
    SearchNode child(P_WHITE,false,0,Hash128());
    node.initializeChildren(); node.state.store(SearchNode::STATE_EXPANDED0);
    auto output = std::make_shared<NNOutput>();
    output->whiteWinProb = output->whiteLossProb = 0.5;
    output->whiteDrawProb = output->whiteNoResultProb = 0;
    std::fill(output->policyProbs,output->policyProbs+NNPos::MAX_NN_POLICY_SIZE,-1);
    output->policyProbs[0] = 0.6f; output->policyProbs[1] = 0.4f;
    node.storeNNOutputIfNull(new std::shared_ptr<NNOutput>(output));
    Loc first = Location::getLoc(0,0,5), second = Location::getLoc(1,0,5);
    auto edges = node.getChildren(); edges[0].setMoveLoc(first); edges[0].store(&child); edges[0].setEdgeVisits(1);
    child.stats.utilityAvg.store(0); child.stats.visits.store(1); child.stats.weightSum.store(1);
    auto select = [&]() {
      return search.getGumbelInteriorMove(thread,node);
    };
    testAssert(select() == second); // .6 - 1/2 < .4 - 0/2
    std::vector<int> avoid(Board::MAX_ARR_SIZE,0); avoid[second] = 2;
    search.setAvoidMoveUntilByLoc(avoid,{});
    testAssert(select() == first);
    avoid[first] = 2; search.setAvoidMoveUntilByLoc(avoid,{});
    testAssert(select() == Board::NULL_LOC);
    thread.history.makeBoardMoveAssumeLegal(thread.board,Location::getLoc(4,4,5),P_BLACK,nullptr);
    thread.history.makeBoardMoveAssumeLegal(thread.board,Location::getLoc(3,4,5),P_WHITE,nullptr);
    testAssert(select() == second); // Both avoids expire after two moves.
    std::vector<Loc> locs; std::vector<double> values;
    testAssert(search.getPlaySelectionValues(node,locs,values,nullptr,0.0,false));
    testAssert(locs.size() == 1 && locs[0] == first);
    testAssert(std::fabs(values[0] - 0.6) < 1e-7); // Policy, without PUCT/LCB target pruning.
    search.clearSearch();
  }
  // Source root focus adds child work without spending an SH reservation or
  // contributing a root visit. Illegal/avoided targets retain zero weight.
  for(auto game : {Rules::Game::GO,Rules::Game::FREESTYLE,Rules::Game::STANDARD,Rules::Game::RENJU,Rules::Game::HEX})
  for(int threads : {1,4}) for(bool interior : {false,true}) {
    SearchParams params;
    params.useGumbel = true; params.useGumbelNonroot = interior; params.useGraphSearch = false;
    params.numThreads = threads; params.maxVisits = 32; params.maxPlayouts = 32;
    params.gumbel.max_num_considered_actions = 1; params.gumbel.noise_scale = 0;
    Search focus(params,nn.get(),&logger,"gumbel-focus");
    Board board(5,5);
    Loc target = Location::getLoc(0,0,5), avoided = Location::getLoc(1,0,5), occupied = Location::getLoc(2,2,5);
    board.setStone(occupied,P_WHITE);
    BoardHistory hist(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    focus.setPosition(P_BLACK,board,hist);
    std::vector<int> avoid(Board::MAX_ARR_SIZE,0); avoid[avoided] = 1;
    focus.setAvoidMoveUntilByLoc(avoid,{});
    focus.setRootFocus({target,avoided,occupied},{1.0,1000.0,1000.0},0.8);
    focus.runWholeSearch(P_BLACK);
    testAssert(focus.getRootVisits() == 32 && focus.lastSearchNumPlayouts == 32);
    testAssert(focus.gumbelRoot && focus.gumbelRoot->pending() == 0);
    auto entries = focus.getGumbelChildren(*focus.rootNode);
    int64_t edgeVisits = 0;
    for(size_t i = 0; i < entries.size(); i++) {
      testAssert(entries[i].visits == focus.gumbelRoot->visits()[i] && entries[i].pending == 0);
      edgeVisits += entries[i].visits;
    }
    testAssert(edgeVisits == 31);
    bool extraFocusWork = false;
    auto edges = focus.rootNode->getChildren();
    for(int i = 0; i < edges.iterateAndCountChildren(); i++) {
      Loc loc = edges[i].getMoveLocRelaxed();
      testAssert(loc != avoided && loc != occupied);
      if(loc == target)
        extraFocusWork = edges[i].getIfAllocated()->stats.visits.load() > edges[i].getEdgeVisits();
    }
    testAssert(extraFocusWork);
    std::vector<Loc> locs; std::vector<double> policy;
    testAssert(focus.getPolicyTargetValues(*focus.rootNode,locs,policy));
    near(std::accumulate(policy.begin(),policy.end(),0.0),1.0);
    testAssert(std::find(locs.begin(),locs.end(),target) != locs.end());
    focus.setRootFocus({}, {}, 0.0);
    focus.runWholeSearch(P_BLACK);
    testAssert(focus.getRootVisits() == 32 && focus.gumbelRoot->pending() == 0);
  }
  auto human = std::unique_ptr<NNEvaluator>(TestSearchCommon::startNNEval("/dev/null",logger,"gumbel-human",5,5,0,true,false,false,true,false));
  {
    auto joint = std::unique_ptr<NNEvaluator>(TestSearchCommon::startNNEval("/dev/null",logger,"gumbel-cap-xiangqi",10,10,0,true,false,false,true,false));
    for(auto game : {Rules::Game::GO,Rules::Game::FREESTYLE,Rules::Game::STANDARD,Rules::Game::RENJU,Rules::Game::HEX,Rules::Game::XIANGQI})
    for(int mode : {0,1,2}) for(int threads : {1,4}) for(bool withHuman : {false,true}) {
      SearchParams params;
      params.useGumbel = mode > 0; params.useGumbelNonroot = mode == 2;
      params.numThreads = threads; params.maxVisits = params.maxPlayouts = 192;
      params.visitCapContempt = 4; params.gumbel.max_num_considered_actions = 2;
      params.valueWeightExponent = 0; params.useUncertainty = params.useNoisePruning = false;
      params.humanSLPlaExploreProbWeightful = params.humanSLOppExploreProbWeightful = withHuman ? 1.0 : 0.0;
      NNEvaluator* primary = game == Rules::Game::XIANGQI ? joint.get() : nn.get();
      NNEvaluator* secondary = withHuman ? (game == Rules::Game::XIANGQI ? joint.get() : human.get()) : nullptr;
      Search capped(params,primary,secondary,&logger,"gumbel-visit-cap");
      Board board(5,5);
      BoardHistory hist = game == Rules::Game::XIANGQI ? BoardHistory(Xiangqi::State(),BoardHistoryModes()) :
        BoardHistory(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
      board = hist.getRecentBoard(0);
      capped.setPosition(P_BLACK,board,hist);
      capped.runWholeSearch(P_BLACK);
      testAssert(capped.getRootVisits() >= 192 && capped.getRootVisits() < 192+threads);
      testAssert(capped.getVisitCappedPla() == P_WHITE && TestVisitCap::checkTree(capped) > 0);
      if(mode == 0 || threads != 1 || withHuman) continue;
      params.visitCapContemptPla = P_BLACK;
      capped.setParams(params);
      bool rejected = false;
      try { capped.runWholeSearch(P_BLACK); }
      catch(const StringError& error) { rejected = std::string(error.what()).find("Gumbel root visit-cap") != std::string::npos; }
      testAssert(rejected);
      params.visitCapContemptPla = C_EMPTY;
      capped.setParams(params);
      capped.runWholeSearch(P_BLACK);
      testAssert(capped.getRootVisits() == 192);
      testAssert(capped.makeMove(capped.getChosenMoveLoc(),P_BLACK));
      rejected = false;
      try { capped.runWholeSearch(P_WHITE,true); }
      catch(const StringError& error) { rejected = std::string(error.what()).find("Gumbel root visit-cap") != std::string::npos; }
      testAssert(rejected); // Ponder still searches for black, so the white root is capped.
      params.visitCapContemptPla = P_BLACK; params.maxVisitsPondering = params.maxPlayoutsPondering = 192;
      capped.setParams(params);
      capped.runWholeSearch(P_WHITE,true);
      testAssert(capped.getRootVisits() == 192 && TestVisitCap::checkTree(capped) > 0);
    }
  }
  for(auto game : {Rules::Game::GO,Rules::Game::HEX}) for(int threads : {1,4})
  for(bool interior : {false,true}) for(int mode : {0,1,2,3,4}) {
    SearchParams params;
    params.useGumbel = true; params.useGumbelNonroot = interior; params.useGraphSearch = false;
    params.numThreads = threads; params.maxVisits = params.maxPlayouts = 64;
    params.humanSLRootExploreProbWeightless = mode == 1 ? 1.0 : mode == 4 ? 0.3 : 0.0;
    params.humanSLPlaExploreProbWeightless = params.humanSLOppExploreProbWeightless = mode == 2 ? 1.0 : mode == 4 ? 0.3 : 0.0;
    params.humanSLPlaExploreProbWeightful = params.humanSLOppExploreProbWeightful = mode == 3 ? 1.0 : mode == 4 ? 0.3 : 0.0;
    params.humanSLChosenMovePiklLambda = 1e9;
    params.chosenMoveTemperature = params.chosenMoveTemperatureEarly = 0;
    Search search(params,nn.get(),human.get(),&logger,"gumbel-human-search");
    Board board(5,5); BoardHistory hist(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    search.setPosition(P_BLACK,board,hist);
    if(mode == 4) search.setRootFocus({Location::getLoc(0,0,5)},{1.0},0.4);
    search.runWholeSearch(P_BLACK);
    testAssert(search.lastSearchNumPlayouts >= 64 && search.lastSearchNumPlayouts < 64+threads);
    testAssert(search.gumbelRoot && search.gumbelRoot->pending() == 0);
    auto candidates = search.getGumbelChildren(*search.rootNode);
    int64_t edges = 0;
    for(size_t i = 0; i < candidates.size(); i++) {
      testAssert(candidates[i].visits == search.gumbelRoot->visits()[i] && candidates[i].pending == 0);
      edges += candidates[i].visits;
    }
    testAssert(search.getRootVisits() == edges+1);
    if(mode == 1) testAssert(edges == 0 && search.getRootVisits() == 1);
    if(mode == 0 || mode == 3) testAssert(edges == 63);
    std::vector<Loc> targetLocs; std::vector<double> target;
    testAssert(search.getPolicyTargetValues(*search.rootNode,targetLocs,target));
    const NNOutput* humanOutput = search.rootNode->getHumanOutput();
    testAssert(humanOutput != nullptr);
    double humanSum = 0, priorSum = 0;
    Loc bestHuman = Board::NULL_LOC; double bestHumanProb = -1;
    for(Loc loc : targetLocs) {
      double prob = humanOutput->policyProbs[search.getPos(loc)];
      humanSum += prob; priorSum += candidates[search.getPos(loc)].prior;
      if(prob > bestHumanProb) { bestHumanProb = prob; bestHuman = loc; }
    }
    Loc shWinner = search.getGumbelMove();
    for(bool sampled : {false,true}) for(double proportion : {0.25,1.0}) {
      search.searchParams.gumbel.sample_visits = sampled;
      search.searchParams.humanSLChosenMoveProp = proportion;
      std::vector<Loc> locs; std::vector<double> values, visits;
      testAssert(search.getPlaySelectionValues(locs,values,&visits,0.0));
      testAssert(locs.size() == targetLocs.size() && values.size() == visits.size());
      double sum = std::accumulate(values.begin(),values.end(),0.0);
      for(size_t i = 0; i < locs.size(); i++) {
        const auto& c = candidates[search.getPos(locs[i])];
        double base = !sampled ? (locs[i] == shWinner ? 1.0 : 0.0) : edges > 0 ? (double)c.visits/edges : c.prior/priorSum;
        double expected = (1-proportion)*base + proportion*humanOutput->policyProbs[search.getPos(locs[i])]/humanSum;
        testAssert(std::abs(values[i]/sum-expected) < 1e-8);
        testAssert(visits[i] == c.visits);
      }
      if(proportion == 1.0) testAssert(search.getChosenMoveLoc() == bestHuman);
      if(!sampled) {
        // Reports may omit the unallocated SH winner. Its base mass must not
        // become a prior-weighted choice among unrelated allocated moves.
        std::vector<Loc> reportLocs; std::vector<double> reportValues, reportVisits;
        testAssert(search.getPlaySelectionValues(*search.rootNode,reportLocs,reportValues,&reportVisits,0.0,false));
        testAssert(reportLocs.size() == (size_t)search.rootNode->getChildren().iterateAndCountChildren());
        bool containsWinner = std::find(reportLocs.begin(),reportLocs.end(),shWinner) != reportLocs.end();
        double reportHumanSum = 0;
        for(Loc loc : reportLocs) reportHumanSum += humanOutput->policyProbs[search.getPos(loc)];
        double reportSum = std::accumulate(reportValues.begin(),reportValues.end(),0.0);
        for(size_t i = 0; i < reportLocs.size(); i++) {
          int pos = search.getPos(reportLocs[i]);
          double humanProb = humanOutput->policyProbs[pos]/reportHumanSum;
          double expected = containsWinner ? (1-proportion)*(reportLocs[i] == shWinner ? 1.0 : 0.0)+proportion*humanProb : humanProb;
          testAssert(std::abs(reportValues[i]/reportSum-expected) < 1e-8);
          testAssert(reportVisits[i] == candidates[pos].visits);
        }
      }
      std::vector<Loc> afterLocs; std::vector<double> after;
      testAssert(search.getPolicyTargetValues(*search.rootNode,afterLocs,after));
      testAssert(targetLocs == afterLocs && target == after); // Chosen-move style never replaces pi'.
    }
  }
  {
    // Independent two-action PIKL calculation: human odds 1:4, utility gap 1,
    // lambda 1/2, mixed into a deterministic A choice with proportion 1/4.
    SearchParams params; params.humanSLChosenMoveProp = 0.25; params.humanSLChosenMovePiklLambda = 0.5;
    params.useLcbForSelection = false; params.chosenMoveSubtract = params.chosenMovePrune = 0;
    Search search(params,nn.get(),human.get(),&logger,"human-pikl-reference");
    Board board(5,5); BoardHistory hist(board,P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes());
    search.setPosition(P_BLACK,board,hist);
    SearchNode node(P_BLACK,false,0,Hash128()), a(P_WHITE,false,0,Hash128()), b(P_WHITE,false,0,Hash128());
    node.initializeChildren(); node.state.store(SearchNode::STATE_EXPANDED0);
    const Loc first = Location::getLoc(0,0,5), second = Location::getLoc(1,0,5);
    auto output = std::make_shared<NNOutput>();
    output->policyProbs[0] = 0.2f; output->policyProbs[1] = 0.8f;
    node.storeNNOutputIfNull(new std::shared_ptr<NNOutput>(output));
    node.storeHumanOutputIfNull(new std::shared_ptr<NNOutput>(output));
    auto children = node.getChildren();
    children[0].setMoveLoc(first); children[0].store(&a); a.stats.utilityAvg.store(-0.5);
    children[1].setMoveLoc(second); children[1].store(&b); b.stats.utilityAvg.store(0.5);
    children[0].setEdgeVisits(1);
    a.stats.visits.store(1); a.stats.weightSum.store(1);
    b.stats.visits.store(1); b.stats.weightSum.store(1);
    std::vector<Loc> locs; std::vector<double> values;
    testAssert(search.getPlaySelectionValues(node,locs,values,nullptr,0.0,false));
    double expected = 0.75+0.25/(1+4*std::exp(-2.0));
    near(values[0],expected); near(values[1],1-expected);
  }
  human.reset();
  // Budget caps, early stop, time stop and exceptions must drain every in-flight reservation.
  SearchParams params; params.useGumbel = true; params.useGraphSearch = false; params.numThreads = 4;
  params.useGumbelNonroot = true;
  params.maxVisits = 10000; params.maxPlayouts = 9;
  Search search(params,nn.get(),&logger,"gumbel-stop");
  Board board(5,5); BoardHistory hist(board,P_BLACK,Rules::forGame(Rules::Game::HEX),0,BoardHistoryModes());
  search.setPosition(P_BLACK,board,hist);
  search.setRootFocus({Location::getLoc(0,0,5)},{1.0},0.8);
  search.runWholeSearch(P_BLACK);
  testAssert(search.getRootVisits() == 9 && search.gumbelRoot->pending() == 0);
  params.maxPlayouts = 10000;
  search.setParams(params);
  std::function<bool()> stop = [&]() { return search.getRootVisits() >= 9; };
  search.runWholeSearch(P_BLACK,&stop);
  testAssert(search.getRootVisits() >= 9 && search.getRootVisits() < 20 && search.gumbelRoot->pending() == 0);
  testAssert(hist.isLegal(board,search.getChosenMoveLoc(),P_BLACK));
  std::function<bool()> fail = [&]() -> bool {
    if(search.getRootVisits() >= 9) throw StringError("injected stop callback failure");
    return false;
  };
  bool threw = false;
  try { search.runWholeSearch(P_BLACK,&fail); }
  catch(const StringError& error) { threw = std::string(error.what()) == "injected stop callback failure"; }
  testAssert(threw && search.gumbelRoot->pending() == 0);
  for(const auto& child : search.getGumbelChildren(*search.rootNode)) testAssert(child.pending == 0);
  params.maxTime = 0;
  search.setParams(params);
  search.runWholeSearch(P_BLACK);
  testAssert(search.getRootVisits() >= 2 && search.getRootVisits() < 20 && search.gumbelRoot->pending() == 0);
  params.maxTime = 1e20;
  params.maxVisits = 7;
  search.setParams(params);
  search.runWholeSearch(nullptr,nullptr,false,TimeControls(),0.5);
  testAssert(search.getRootVisits() == 4);
  std::function<bool()> stopImmediately = []() noexcept { return true; };
  search.runWholeSearch(P_BLACK,&stopImmediately);
  testAssert(search.getRootVisits() == 0 && search.getChosenMoveLoc() == Board::NULL_LOC);
  std::vector<int> avoid(Board::MAX_ARR_SIZE,1);
  search.setAvoidMoveUntilByLoc(avoid,avoid);
  search.runWholeSearch(P_BLACK);
  testAssert(search.getRootVisits() == 1 && search.getChosenMoveLoc() == Board::NULL_LOC);
  search.setAvoidMoveUntilByLoc({},{});
  for(int x = 0; x < 5; x++) board.setStone(Location::getLoc(x,0,5),P_BLACK);
  BoardHistory terminal(board,P_WHITE,Rules::forGame(Rules::Game::STANDARD),0,BoardHistoryModes());
  testAssert(terminal.isGameFinished);
  search.setPosition(P_WHITE,board,terminal);
  search.runWholeSearch(P_WHITE);
  testAssert(search.getRootVisits() == 1 && search.getChosenMoveLoc() == Board::NULL_LOC);
  nn.reset(); NeuralNet::globalCleanup();
}
}
void Tests::runGumbelTests() {
  std::cout << "Running Gumbel scheduling and source search tests" << std::endl;
  mathAndScheduling(); sourceSearch();
  NeuralNet::globalInitialize();
  Logger logger(nullptr,false,false,false);
  for(auto game : {Rules::Game::GO,Rules::Game::FREESTYLE,Rules::Game::STANDARD,Rules::Game::RENJU,Rules::Game::HEX,Rules::Game::XIANGQI}) {
    const bool xiangqi = game == Rules::Game::XIANGQI;
    auto nn = std::unique_ptr<NNEvaluator>(TestSearchCommon::startNNEval("/dev/null",logger,"gumbel-symmetry",xiangqi ? 10 : 5,xiangqi ? 10 : 5,0,true,false,false,true,false));
    for(int threads : {1,4}) for(bool interior : {false,true}) {
      SearchParams params; params.useGumbel = true; params.useGumbelNonroot = interior;
      params.numThreads = threads; params.gumbel.max_num_considered_actions = 4;
      Search search(params,nn.get(),nn.get(),&logger,"gumbel-symmetry");
      Board board(5,5);
      BoardHistory history = xiangqi ? BoardHistory(Xiangqi::State(),BoardHistoryModes()) :
        BoardHistory(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
      if(xiangqi) board = history.getRecentBoard(0);
      TestGumbelSymmetry::exercise(search,board,history);
      if(game == Rules::Game::GO) {
        Board rectangle(5,4);
        BoardHistory white(rectangle,P_WHITE,Rules::forGame(game),0,BoardHistoryModes());
        TestGumbelSymmetry::exercise(search,rectangle,white);
      }
    }
  }
  NeuralNet::globalCleanup();
}

void Tests::runGumbelReference() {
  nlohmann::json output = nlohmann::json::array();
  for(int budget : {0,1,2,3,4,5,6,7,8,9,10,16,17,31,32,63,64,127,256}) {
    GumbelSettings settings; settings.noise_scale = 0;
    std::vector<GumbelChild> children{{0.5,0.8,0,0},{0.3,-0.2,0,0},{0.2,-0.6,0,0}};
    Rand rand("reference"); GumbelRoot root({0.5,0.3,0.2},budget,settings,rand);
    for(int i = 0; i < budget; i++) {
      int action = root.reserve(gumbel_completed_q(children,0.1,settings));
      root.commit(action); children[action].visits++;
    }
    auto interiorChildren = std::vector<GumbelChild>{{0.5,0.8,0,0},{0.3,-0.2,0,0},{0.2,-0.6,0,0}};
    std::vector<int> interiorActions;
    for(int i = 0; i < budget; i++) {
      int action = gumbel_interior_selection(interiorChildren,0.1,settings);
      interiorActions.push_back(action); interiorChildren[action].visits++;
    }
    output.push_back({{"interiorActions",interiorActions},{"budget",budget},{"action",root.winner(gumbel_completed_q(children,0.1,settings))},
                      {"visits",root.visits()},{"policy",gumbel_policy(children,0.1,settings)}});
  }
  std::cout << output.dump() << std::endl;
}
