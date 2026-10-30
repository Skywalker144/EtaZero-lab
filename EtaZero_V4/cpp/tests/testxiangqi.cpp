#include "tests.h"
#include "../game/xiangqi.h"
#include "../game/graphhash.h"
#include "../dataio/sgf.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../program/playutils.h"
#include "../search/search.h"
#include "../search/searchnode.h"
#include <set>

namespace {
using Xiangqi::State;
using Xiangqi::Reason;

void play(State& state, const std::string& move) { state.play(Xiangqi::parseAction(move)); }

uint64_t perft(const State& state, int depth) {
  if(depth == 0) return 1;
  uint64_t total = 0;
  for(int action : state.legalActions()) {
    State next(state); next.play(action);
    total += perft(next,depth-1);
  }
  return total;
}

void cycle(State& state) {
  for(const char* move : {"b0c2","b9c7","c2b0","c7b9"}) play(state,move);
}

void vocabulary() {
  std::set<int> indices;
  for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) {
    testAssert(Xiangqi::locToAction(Xiangqi::actionToLoc(action)) == action);
    const auto& move = Xiangqi::MOVES[action];
    testAssert(Xiangqi::actionOf(move[0],move[1],move[2],move[3]) == action);
    testAssert(Xiangqi::parseAction(Xiangqi::actionToString(action)) == action);
    const int index = Xiangqi::policyIndex(action,10,10);
    testAssert(index >= 0 && index < 5000);
    testAssert(indices.insert(index).second);
    testAssert(index % 100 == move[1]*10+move[0]);
    testAssert(Xiangqi::DELTAS[index/100][0] == move[2]-move[0]);
    testAssert(Xiangqi::DELTAS[index/100][1] == move[3]-move[1]);
    for(Player perspective : {P_BLACK,P_WHITE}) for(bool mirror : {false,true}) {
      const int other = Xiangqi::transformAction(action,perspective,mirror);
      testAssert(other >= 0 && other < Xiangqi::NUM_ACTIONS);
      testAssert(Xiangqi::transformAction(other,perspective,mirror) == action);
    }
  }
  testAssert(Xiangqi::NUM_ACTIONS == 2086 && Xiangqi::NUM_DELTAS == 50);
  for(const char* invalid : {"pass","a0a0","a0j0","a0a10","c0d1","c0e3"})
    testAssert(Xiangqi::parseAction(invalid) == -1);
  for(Loc invalid : {Loc(-1),Board::NULL_LOC,Board::PASS_LOC,Loc(Xiangqi::NUM_ACTIONS+2),Loc(32767)})
    testAssert(Xiangqi::locToAction(invalid) == -1);
}

void nativeHistory() {
  const Rules rules = Rules::forGame(Rules::Game::XIANGQI);
  testAssert(rules.isXiangqi() && !rules.isGo() && !rules.isPlacement() && rules.komi == 0);
  testAssert(Rules::parseGame(" XIANGQI ") == rules.game);
  testAssert(Rules::parseRules(rules.toJsonString()) == rules && Rules::parseRules("xiangqi") == rules);
  for(int sym = 0; sym < 8; sym++) testAssert(rules.allowsSymmetry(sym) == (sym == 0 || sym == 2));
  for(int version : {17,18,19}) {
    ModelDesc model; model.modelVersion = version;
    for(Rules::Game game : {Rules::Game::GO,Rules::Game::RENJU,Rules::Game::HEX,Rules::Game::XIANGQI}) {
      bool supported;
      model.getSupportedRules(Rules::forGame(game),supported);
      testAssert(supported == (version == 19 ? game == Rules::Game::XIANGQI :
        version == 17 ? game == Rules::Game::GO : game == Rules::Game::RENJU || game == Rules::Game::HEX));
    }
  }
  auto step = [](BoardHistory& hist, Board& board, const char* move) {
    const Loc loc = Xiangqi::actionToLoc(Xiangqi::parseAction(move));
    const Player pla = hist.presumedNextMovePla;
    testAssert(hist.isLegal(board,loc,pla) && !hist.isLegal(board,loc,getOpp(pla)));
    testAssert(hist.makeBoardMoveTolerant(board,loc,pla,true));
    board.checkConsistency(true);
    testAssert(board.pos_hash == hist.getRecentBoard(0).pos_hash);
    const auto& pieces = hist.getXiangqiState().pieces();
    for(int y = 0; y < 10; y++) for(int x = 0; x < 9; x++) {
      const int piece = pieces[y*9+x];
      testAssert(board.colors[Location::getLoc(x,y,9)] == (piece > 0 ? P_BLACK : piece < 0 ? P_WHITE : C_EMPTY));
    }
    testAssert(!hist.isNoResult && !hist.isScored && !hist.isResignation && hist.finalWhiteMinusBlackScore == 0);
  };
  BoardHistory hist;
  Hash128 initialHash;
  {
    State prior; cycle(prior);
    initialHash = prior.historyHash();
    BoardHistory donor(prior,BoardHistoryModes(true,true));
    hist = donor;
  }
  testAssert(hist.getCurrentTurnNumber() == 4 && hist.moveHistory.empty());
  testAssert(hist.getXiangqiState().repetitions() == 2 && hist.getXiangqiState().historyHash() == initialHash);
  Board board = hist.getRecentBoard(0);
  testAssert(board.x_size == 9 && board.y_size == 10);
  testAssert(!hist.isLegal(board,Board::PASS_LOC,P_BLACK) && !hist.isLegal(board,Board::NULL_LOC,P_BLACK));
  testAssert(!hist.passWouldEndPhase(board,P_BLACK) && !hist.passWouldEndGame(board,P_BLACK));
  auto graphHash = GraphHash::getGraphHashFromScratch(hist,P_BLACK,1,0.5);
  State fresh(hist.getXiangqiState().fen());
  BoardHistory reloaded(fresh,BoardHistoryModes());
  testAssert(reloaded.getRecentBoard(0).pos_hash == board.pos_hash);
  testAssert(GraphHash::getGraphHashFromScratch(reloaded,P_BLACK,1,0.5) != graphHash);
  for(const char* move : {"b0c2","b9c7","c2b0","c7b9"}) {
    step(hist,board,move);
    graphHash = GraphHash::getGraphHash(graphHash,hist,hist.presumedNextMovePla,1,0.5);
    testAssert(graphHash == GraphHash::getGraphHashFromScratch(hist,hist.presumedNextMovePla,100,0.5));
  }
  testAssert(hist.isGameFinished && hist.winner == C_EMPTY && hist.getCurrentTurnNumber() == 8);
  testAssert(hist.getXiangqiState().result().reason == Reason::REPETITION);
  testAssert(hist.numTurnsThisPhase == 4 && hist.numConsecValidTurnsThisGame == 4 && hist.whiteHasMoved);
  testAssert(!hist.makeBoardMoveTolerant(board,Xiangqi::actionToLoc(Xiangqi::parseAction("b0c2")),P_BLACK));
  testAssert(hist.koHashHistory.front() == hist.koHashHistory.back());
  testAssert(hist.numberOfKoHashOccurrencesInHistory(hist.koHashHistory.back(),nullptr) == 2);
  KoHashTable table; table.recompute(hist);
  testAssert(hist.numberOfKoHashOccurrencesInHistory(hist.koHashHistory.back(),&table) == 2);
  BoardHistory rewound = hist.copyToInitial();
  testAssert(!rewound.isGameFinished && rewound.getCurrentTurnNumber() == 4 && rewound.modes == hist.modes);
  testAssert(rewound.getXiangqiState().historyHash() == initialHash && rewound.getXiangqiState().repetitions() == 2);
  Board replayBoard = rewound.getRecentBoard(0);
  for(size_t i = 0; i < hist.moveHistory.size(); i++)
    testAssert(rewound.makeBoardMoveTolerant(replayBoard,hist.moveHistory[i].loc,hist.moveHistory[i].pla,hist.preventEncoreHistory[i]));
  testAssert(rewound.getXiangqiState().historyHash() == hist.getXiangqiState().historyHash());
  testAssert(rewound.isGameFinished && rewound.winner == hist.winner);
  BoardHistory copied(hist), moved(std::move(copied)), assigned;
  assigned = std::move(moved);
  assigned = assigned;
  testAssert(assigned.getXiangqiState().historyHash() == hist.getXiangqiState().historyHash());
  testAssert(assigned.copyToInitial().getXiangqiState().historyHash() == initialHash);
  // Clearing from an aliased state must retain all earlier repetition history.
  assigned.clear(assigned.getXiangqiState());
  testAssert(assigned.moveHistory.empty() && assigned.isGameFinished && assigned.getCurrentTurnNumber() == 8);
  assigned.clear(Board(7,7),P_BLACK,Rules::getTrompTaylorish(),0);
  testAssert(assigned.modes == hist.modes && assigned.rules.isGo());
  bool rejected = false;
  try { assigned.getXiangqiState(); } catch(const StringError&) { rejected = true; }
  testAssert(rejected);
  assigned.clear(State());
  Board captureBoard = assigned.getRecentBoard(0);
  step(assigned,captureBoard,"b2b9");
  testAssert(captureBoard.colors[Location::getLoc(1,2,9)] == C_EMPTY);
  testAssert(captureBoard.colors[Location::getLoc(1,9,9)] == P_BLACK);
  testAssert(assigned.getXiangqiState().pieces()[9*9+1] == 3);
  assigned.setOverrideNumHandicapStones(5);
  testAssert(assigned.computeNumHandicapStones() == 0 && assigned.whiteHandicapBonusScore == 0);
  const auto hashBeforeResign = GraphHash::getGraphHashFromScratch(assigned,P_WHITE,1,0.5);
  assigned.setWinnerByResignation(P_BLACK);
  testAssert(GraphHash::getGraphHashFromScratch(assigned,P_WHITE,1,0.5) != hashBeforeResign);
  testAssert(!assigned.isLegal(captureBoard,Xiangqi::actionToLoc(Xiangqi::parseAction("h9g7")),P_WHITE));
  for(const char* fen : {"5k3/9/9/9/9/9/9/9/9/4K4 w - - 0 1", "4k4/3R1R3/9/9/4P4/9/9/9/9/4K4 b - - 0 1"}) {
    State terminal(fen);
    BoardHistory finished(terminal,BoardHistoryModes());
    testAssert(finished.isGameFinished && finished.winner == terminal.result().winner);
    testAssert(!finished.isNoResult && !finished.isScored);
  }
  // Equal color occupancy must not alias different piece identities.
  State swapped("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/NRBAKABNR w - - 0 1");
  BoardHistory a{State(),BoardHistoryModes()}, b(swapped,BoardHistoryModes());
  testAssert(a.getRecentBoard(0).pos_hash == b.getRecentBoard(0).pos_hash);
  testAssert(GraphHash::getStateHash(a,P_BLACK,0.5) != GraphHash::getStateHash(b,P_BLACK,0.5));
  rejected = false;
  try { BoardHistory invalid(board,P_BLACK,rules,0,BoardHistoryModes()); } catch(const StringError&) { rejected = true; }
  testAssert(rejected);
  std::ostringstream debug;
  hist.printDebugInfo(debug,board);
  testAssert(debug.str().find("b0c2") != std::string::npos);
  std::ostringstream sgfText;
  WriteSgf::writeSgf(sgfText,"red","black",hist,nullptr,false,false);
  CompactSgf sgf(*Sgf::parse(sgfText.str()));
  Board restoredBoard;
  BoardHistory restoredHist;
  Player restoredPla;
  sgf.setupBoardAndHistTolerant(rules,restoredBoard,restoredPla,restoredHist,sgf.moves.size(),false,hist.modes);
  testAssert(restoredHist.getXiangqiState().historyHash() == hist.getXiangqiState().historyHash());
}

void outputStorage() {
  auto makeOutput = [](int count) {
    auto output = std::make_shared<NNOutput>();
    output->resizePolicy(count);
    output->nnHash = Hash128(0,0);
    output->nnXLen = output->nnYLen = 10;
    output->whiteWinProb = 0.4f; output->whiteDrawProb = 0.2f;
    output->whiteLossProb = 0.4f; output->whiteNoResultProb = 0;
    output->varTimeLeft = 1; output->shorttermWinlossError = 2;
    output->policyOptimismUsed = 0.25f;
    for(int a = 0; a < output->getPolicyStorageSize(); a++) output->policyProbs[a] = (float)a;
    output->policyProbs[7] = -1;
    return output;
  };
  for(int count : {NNPos::MAX_NN_POLICY_SIZE, Xiangqi::NUM_ACTIONS}) {
    auto first = makeOutput(count);
    first->noisedPolicyProbs = new float[count];
    for(int a = 0; a < count; a++) first->noisedPolicyProbs[a] = (float)(a+1);
    auto copied = std::make_shared<NNOutput>(*first);
    testAssert(copied->policyProbs != first->policyProbs && copied->noisedPolicyProbs != first->noisedPolicyProbs);
    NNOutput assigned;
    assigned.resizePolicy(count == Xiangqi::NUM_ACTIONS ? NNPos::MAX_NN_POLICY_SIZE : Xiangqi::NUM_ACTIONS);
    assigned = *copied;
    assigned = assigned;
    for(int a = 0; a < count; a++) {
      testAssert(copied->policyProbs[a] == first->policyProbs[a] && assigned.policyProbs[a] == first->policyProbs[a]);
      testAssert(assigned.noisedPolicyProbs[a] == a+1);
    }
    copied->policyProbs[count-1] += 2;
    NNOutput average(std::vector<std::shared_ptr<NNOutput>>{first,copied});
    testAssert(average.getPolicyStorageSize() == count && average.policyProbs[count-1] == count);
    testAssert(average.policyProbs[7] == -1 && average.noisedPolicyProbs == nullptr);
    copied->policyProbs[7] = 1; // source legality mismatch falls back to the first result
    NNOutput mismatch(std::vector<std::shared_ptr<NNOutput>>{first,copied});
    testAssert(mismatch.policyProbs[7] == -1 && mismatch.policyProbs[count-1] == count-1);
    first.reset(); copied.reset();
    testAssert(assigned.policyProbs[count-1] == count-1 && assigned.noisedPolicyProbs[count-1] == count);
  }
}

void searchStorage() {
  SearchNode leaf(P_WHITE,false,0,Hash128());
  SearchNode node(P_BLACK,false,0,Hash128(),Xiangqi::NUM_ACTIONS);
  node.initializeChildren();
  node.state.store(SearchNode::STATE_EXPANDED0);
  for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) {
    auto state = node.state.load();
    testAssert(node.maybeExpandChildrenCapacityForNewChild(state,action+1));
    auto children = node.getChildren(state);
    children[action].setMoveLoc(Xiangqi::actionToLoc(action));
    children[action].setEdgeVisits(action+1);
    children[action].store(&leaf);
  }
  testAssert(node.getChildren().getCapacity() == Xiangqi::NUM_ACTIONS);
  testAssert(node.getChildren().iterateAndCountChildren() == Xiangqi::NUM_ACTIONS);
  SearchNode copied(node,false,false);
  for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) {
    const auto& child = copied.getChildren()[action];
    testAssert(child.getIfAllocated() == &leaf && child.getMoveLoc() == Xiangqi::actionToLoc(action));
    testAssert(child.getEdgeVisits() == action+1);
    if(action > 0) copied.getChildren()[action].store(nullptr);
  }
  copied.collapseChildrenCapacity(1);
  testAssert(copied.getChildren().getCapacity() == 8 && copied.getChildren().iterateAndCountChildren() == 1);
  SearchNode source(P_BLACK,false,0,Hash128());
  testAssert(source.maxChildren == NNPos::MAX_NN_POLICY_SIZE);

  Rand rand("joint-policy-storage");
  std::vector<double> probabilities(Xiangqi::NUM_ACTIONS,0), processed(Xiangqi::NUM_ACTIONS);
  probabilities.back() = 1;
  testAssert(Search::chooseIndexWithTemperature(rand,probabilities.data(),Xiangqi::NUM_ACTIONS,1,1,nullptr) == Xiangqi::NUM_ACTIONS-1);
  testAssert(Search::chooseIndexWithTemperature(rand,probabilities.data(),Xiangqi::NUM_ACTIONS,0,1,processed.data()) == Xiangqi::NUM_ACTIONS-1);
  testAssert(processed.back() == 1);
  std::vector<float> policy(Xiangqi::NUM_ACTIONS,-1);
  policy[0] = 0.25f; policy.back() = 0.75f;
  SearchParams params;
  Search::addDirichletNoise(params,rand,Xiangqi::NUM_ACTIONS,policy.data());
  testAssert(std::abs(policy[0]+policy.back()-1) < 1e-6);
  for(int i = 1; i < Xiangqi::NUM_ACTIONS-1; i++) testAssert(policy[i] == -1);
}

void nativeSearch() {
  ConfigParser config(std::map<std::string,std::string>{});
  Logger logger;
  NNEvaluator evaluator("random","","",&logger,8,10,10,false,false,10,4,true,"",enabled_t::False,
                       1,{0},"xiangqi-search",false,0,config);
  evaluator.spawnServerThreads();
  for(bool gumbel : {false,true}) for(int threads : {1,4}) {
    SearchParams params;
    params.maxVisits = params.maxPlayouts = 96;
    params.numThreads = threads;
    params.useGumbel = gumbel; params.useGumbelNonroot = gumbel;
    params.useGraphSearch = !gumbel;
    params.useEvalCache = !gumbel;
    params.rootNoiseEnabled = true;
    params.rootNumSymmetriesToSample = 2;
    params.useNoisePruning = true;
    params.useUncertainty = true;
    params.subtreeValueBiasFactor = 0.4;
    params.avoidRepeatedPatternUtility = 0.03;
    params.rootSymmetryPruning = !gumbel;
    Search search(params,&evaluator,&logger,"xiangqi-native-search");
    State state;
    BoardHistory hist(state,BoardHistoryModes());
    search.setPosition(P_BLACK,hist.getRecentBoard(0),hist);
    const int hint = state.legalActions().back();
    search.setRootHintLoc(Xiangqi::actionToLoc(hint));
    const Loc focused = Xiangqi::actionToLoc(Xiangqi::parseAction("b0c2"));
    if(!gumbel) search.setRootFocus({focused,Board::PASS_LOC,Loc(32767)},{1,1,1},0.5);
    for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) testAssert(search.getPos(search.getLoc(action)) == action);
    search.runWholeSearch(P_BLACK);
    {
      std::ostringstream log;
      const Loc chosen = search.getChosenMoveLoc();
      PlayUtils::printGenmoveLog(log,&search,&evaluator,chosen,0.01,P_BLACK,true,PrintTreeOptions());
      testAssert(log.str().find("Xiangqi FEN: " + state.fen()) != std::string::npos);
      testAssert(log.str().find(Xiangqi::actionToString(Xiangqi::locToAction(chosen))) != std::string::npos);
    }
    const auto* root = search.getRootNode();
    testAssert(root->getNNOutput()->getPolicyStorageSize() == Xiangqi::NUM_ACTIONS);
    testAssert(search.getRootVisits() >= 96 && search.getRootVisits() <= 96+threads);
    testAssert(root->stats.noResultValueAvg.load() == 0);
    testAssert(search.rootSymmetries.size() == (gumbel ? 1 : 2));
    std::set<Loc> visited;
    auto children = root->getChildren();
    for(int i = 0; i < children.iterateAndCountChildren(); i++) {
      Loc loc = children[i].getMoveLoc();
      testAssert(state.isLegal(Xiangqi::locToAction(loc)));
      if(!gumbel && search.getSymMove(loc,2) != loc) testAssert(!visited.count(search.getSymMove(loc,2)));
      visited.insert(loc);
    }
    std::vector<Loc> locs;
    std::vector<double> target;
    testAssert(search.getPolicyTargetValues(*root,locs,target));
    if(gumbel) testAssert(locs.size() == 44);
    for(size_t i = 0; i < locs.size(); i++) testAssert(state.isLegal(Xiangqi::locToAction(locs[i])) && std::isfinite(target[i]) && target[i] >= 0);
    double surprise, searchEntropy, policyEntropy;
    testAssert(search.getPolicySurpriseAndEntropy(surprise,searchEntropy,policyEntropy));
    testAssert(std::isfinite(surprise) && std::isfinite(searchEntropy) && std::isfinite(policyEntropy));
    nlohmann::json analysis;
    testAssert(search.getAnalysisJson(P_BLACK,6,false,true,true,true,analysis));
    testAssert(analysis["policy"].size() == Xiangqi::NUM_ACTIONS);
    bool reportedFocus = false;
    for(const auto& info : analysis["moveInfos"]) {
      State replay(state);
      testAssert(replay.isLegal(Xiangqi::parseAction(info["move"].get<std::string>())));
      for(const auto& text : info["pv"]) play(replay,text.get<std::string>());
      if(!gumbel && info["move"] == "b0c2") {
        // Source focus reaches the searched mirror representative and adds
        // child visits without charging those visits to the root's edge.
        testAssert(info["visits"].get<int64_t>() > info["edgeVisits"].get<int64_t>());
        reportedFocus = true;
      }
    }
    if(!gumbel) testAssert(reportedFocus);
    const Loc chosen = search.getChosenMoveLoc();
    const auto* child = search.getChildForMove(root,chosen);
    const auto childVisits = child ? child->stats.visits.load() : 0;
    const bool reusable = child && child->getNNOutput();
    testAssert(search.makeMove(chosen,P_BLACK));
    state.play(Xiangqi::locToAction(chosen));
    testAssert(search.getRootHist().getXiangqiState().historyHash() == state.historyHash());
    if(gumbel) testAssert(search.getRootVisits() == 0);
    else if(reusable) testAssert(search.getRootVisits() == childVisits);
    search.runWholeSearch(P_WHITE);
    testAssert(state.isLegal(Xiangqi::locToAction(search.getChosenMoveLoc())));

    // The last move in this known long-check cycle awards the black side a win.
    State checker("5k3/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1");
    for(const char* m : {"e8f8","f9e9","f8e8","e9f9","e8f8","f9e9","f8e8"}) play(checker,m);
    BoardHistory checked(checker,BoardHistoryModes());
    search.setPosition(P_WHITE,checked.getRecentBoard(0),checked);
    Loc winning = Xiangqi::actionToLoc(Xiangqi::parseAction("e9f9"));
    std::vector<int> avoid(search.getMoveLocBufferSize(),1); avoid[winning] = 0;
    search.setAvoidMoveUntilByLoc({},avoid);
    search.setAvoidMoveUntilRescaleRoot(true);
    search.runWholeSearch(P_WHITE);
    testAssert(search.getChosenMoveLoc() == winning);
    testAssert(search.getRootNode()->stats.winLossValueAvg.load() > 0.9);
    testAssert(search.makeMove(winning,P_WHITE));
    testAssert(search.getRootHist().isGameFinished && search.getRootHist().winner == P_WHITE);
    search.runWholeSearch(P_BLACK);
    testAssert(search.getChosenMoveLoc() == Board::NULL_LOC);
    testAssert(search.getRootNode()->stats.winLossValueAvg.load() == 1 && search.getRootNode()->stats.noResultValueAvg.load() == 0);
  }
  evaluator.killServerThreads();
}

void secondaryHumanSearch() {
  ConfigParser config(std::map<std::string,std::string>{});
  Logger logger;
  NNEvaluator evaluator("main","","",&logger,8,10,10,false,false,10,4,true,"",enabled_t::False,
                       1,{0},"xiangqi-main",false,0,config);
  NNEvaluator human("human","","",&logger,8,10,10,false,false,10,4,true,"",enabled_t::False,
                   1,{0},"xiangqi-human",false,0,config);
  evaluator.spawnServerThreads(); human.spawnServerThreads();
  for(bool gumbel : {false,true}) for(int mode : {0,1,2}) for(int threads : {1,3}) for(int symmetries : {1,2}) {
    SearchParams params;
    params.maxVisits = params.maxPlayouts = 96;
    params.numThreads = threads;
    params.rootNumSymmetriesToSample = symmetries;
    params.useGraphSearch = !gumbel;
    params.useGumbel = params.useGumbelNonroot = gumbel;
    params.humanSLChosenMoveProp = 1;
    // A large lambda leaves the normalized human policy unchanged to 1e-8.
    params.humanSLChosenMovePiklLambda = 1e9;
    params.humanSLRootExploreProbWeightful = params.humanSLPlaExploreProbWeightful =
      params.humanSLOppExploreProbWeightful = mode == 1 ? 1 : 0;
    params.humanSLRootExploreProbWeightless = params.humanSLPlaExploreProbWeightless =
      params.humanSLOppExploreProbWeightless = mode == 2 ? 1 : 0;
    if(gumbel) params.humanSLRootExploreProbWeightful = 0;
    Search search(params,&evaluator,&human,&logger,"xiangqi-secondary");
    State state;
    BoardHistory hist(state,BoardHistoryModes());
    search.setPosition(P_BLACK,hist.getRecentBoard(0),hist);
    for(int turn = 0; turn < 2; turn++) {
      search.runWholeSearch(state.nextPlayer());
      if(gumbel) {
        testAssert(search.lastSearchNumPlayouts >= 96 && search.lastSearchNumPlayouts < 96+threads);
        testAssert(search.gumbelRoot && search.gumbelRoot->pending() == 0);
        auto entries = search.getGumbelChildren(*search.getRootNode());
        for(size_t i = 0; i < entries.size(); i++)
          testAssert(entries[i].visits == search.gumbelRoot->visits()[i] && entries[i].pending == 0);
      }
      const SearchNode* root = search.getRootNode();
      const NNOutput* output = root->getHumanOutput();
      testAssert(output != nullptr && output->getPolicyStorageSize() == Xiangqi::NUM_ACTIONS);
      std::vector<Loc> locs;
      std::vector<double> values;
      testAssert(search.getPlaySelectionValues(locs,values,1.0));
      testAssert(locs.size() == state.legalActions().size());
      double sum = 0, policySum = 0;
      for(size_t i = 0; i < locs.size(); i++) {
        const int action = Xiangqi::locToAction(locs[i]);
        testAssert(state.isLegal(action));
        sum += values[i]; policySum += output->policyProbs[action];
      }
      for(size_t i = 0; i < locs.size(); i++)
        testAssert(std::abs(values[i]/sum-output->policyProbs[Xiangqi::locToAction(locs[i])]/policySum) < 1e-8);
      auto children = root->getChildren();
      int64_t edges = 0;
      for(int i = 0; i < children.iterateAndCountChildren(); i++) {
        const SearchNode* child = children[i].getIfAllocated();
        testAssert(child->getNNOutput() != nullptr);
        testAssert((child->getHumanOutput() != nullptr) == (mode != 0));
        edges += children[i].getEdgeVisits();
      }
      if(mode == 2) {
        // Human weightless exploration expands nodes without changing edge
        // weights or replacing the main model's value with the secondary's.
        testAssert(edges == 0);
        const NNOutput* main = root->getNNOutput();
        testAssert(root->stats.visits.load() == 1 && root->stats.weightSum.load() == 1);
        // A retained root can refresh its NN average without a counted visit
        // to recompute aggregate stats. Check equality at the fresh root.
        if(turn == 0)
          testAssert(std::abs(root->stats.utilityAvg.load()-(double(main->whiteWinProb)-main->whiteLossProb)) < 1e-6);
      }
      else testAssert(edges > 0);
      const Loc chosen = search.getChosenMoveLoc();
      testAssert(state.isLegal(Xiangqi::locToAction(chosen)));
      testAssert(search.makeMove(chosen,state.nextPlayer()));
      state.play(Xiangqi::locToAction(chosen));
    }
  }
  human.killServerThreads(); evaluator.killServerThreads();
}

void evaluatorInputsAndCache() {
  State state;
  play(state,"b0c2"); play(state,"h9g7");
  MiscNNInputParams params;
  const auto fullHash = NNInputs::getHash(state,params);
  params.maxHistory = 1;
  testAssert(NNInputs::getHash(state,params) != fullHash);
  const auto oneHash = NNInputs::getHash(state,params);
  params.maxHistory = 0;
  testAssert(NNInputs::getHash(state,params) != oneHash);
  for(int cap : {0,1,7}) {
    float global[8];
    std::vector<float> spatial(113*100);
    state.fillFeatures(P_WHITE,0,false,10,10,false,spatial.data(),global,cap);
    const int frames = std::min(cap+1,3);
    testAssert(global[5] == frames/8.0f);
    testAssert(global[1] == 2/120.0f && global[7] == 2/200.0f);
    for(int c = 1+frames*14; c < 113; c++)
      for(int pos = 0; pos < 100; pos++) testAssert(spatial[c*100+pos] == 0);
  }
  State repeat; cycle(repeat);
  State reload(repeat.fen());
  testAssert(repeat.positionKey() == reload.positionKey());
  testAssert(NNInputs::getHash(repeat,params) != NNInputs::getHash(reload,params));
  params = MiscNNInputParams();
  const auto plainHash = NNInputs::getHash(state,params);
  params.playoutDoublingAdvantage = 1.5;
  testAssert(NNInputs::getHash(state,params) != plainHash);
  params = MiscNNInputParams(); params.nnPolicyTemperature = 0.8f;
  testAssert(NNInputs::getHash(state,params) != plainHash);
  params = MiscNNInputParams(); params.policyOptimism = 0.6;
  testAssert(NNInputs::getHash(state,params) != plainHash);
  params = MiscNNInputParams(); params.symmetry = 2;
  testAssert(NNInputs::getHash(state,params) == plainHash); // source cache intentionally ignores symmetry

  ConfigParser config(std::map<std::string,std::string>{});
  NNEvaluator evaluator("random","","",nullptr,4,10,10,false,false,8,4,true,"",enabled_t::False,
                         1,{0},"xiangqi-cache-test",false,0,config);
  evaluator.spawnServerThreads();
  Logger logger;
  NNResultBuf buf;
  params = MiscNNInputParams();
  for(int turn = 0; turn < 2; turn++) {
    evaluator.evaluate(state,params,buf,false);
    const auto result = buf.result;
    BoardHistory history(state,BoardHistoryModes());
    const Board& board = history.getRecentBoard(0);
    testAssert(NNInputs::getHash(board,history,state.nextPlayer(),params) == NNInputs::getHash(state,params));
    evaluator.evaluate(board,history,state.nextPlayer(),params,buf,false);
    testAssert(buf.result == result);
    bool rejected = false;
    try { evaluator.evaluate(board,history,getOpp(state.nextPlayer()),params,buf,false); }
    catch(const StringError&) { rejected = true; }
    testAssert(rejected);
    for(bool nhwc : {false,true}) for(int cap : {0,1,7}) {
      auto inputParams = params; inputParams.maxHistory = cap; inputParams.playoutDoublingAdvantage = -1.5;
      std::vector<float> actual(113*100), expected(113*100);
      float ag[8], eg[8];
      NNInputs::fillRow(9,board,history,state.nextPlayer(),inputParams,10,10,nhwc,actual.data(),ag);
      state.fillFeatures(state.nextPlayer(),-1.5,false,10,10,nhwc,expected.data(),eg,cap);
      testAssert(actual == expected && std::equal(ag,ag+8,eg));
      rejected = false;
      try { NNInputs::fillRow(8,board,history,state.nextPlayer(),inputParams,10,10,nhwc,actual.data(),ag); }
      catch(const StringError&) { rejected = true; }
      testAssert(rejected);
    }
    Search search(SearchParams(),&evaluator,&logger,"xiangqi-search-boundary");
    search.setPosition(state.nextPlayer(),board,history);
    rejected = false;
    try { search.setPlayerAndClearHistory(getOpp(state.nextPlayer())); } catch(const StringError&) { rejected = true; }
    testAssert(rejected);
    history.setWinnerByResignation(P_BLACK);
    rejected = false;
    try { evaluator.evaluate(board,history,state.nextPlayer(),params,buf,false); } catch(const StringError&) { rejected = true; }
    testAssert(rejected);
    testAssert(result->getPolicyStorageSize() == Xiangqi::NUM_ACTIONS);
    testAssert(result->whiteNoResultProb == 0);
    double sum = 0;
    for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) {
      if(state.isLegal(action)) { testAssert(result->policyProbs[action] >= 0); sum += result->policyProbs[action]; }
      else testAssert(result->policyProbs[action] == -1);
    }
    testAssert(std::abs(sum-1) < 1e-6);
    const auto hits = evaluator.numCacheHits();
    evaluator.evaluate(state,params,buf,false);
    testAssert(buf.result == result && evaluator.numCacheHits() == hits+1);
    params.maxHistory = 0;
    evaluator.evaluate(state,params,buf,false);
    testAssert(buf.result != result);
    params.maxHistory = 1000;
    state.play(state.legalActions()[0]);
  }
  State terminal("5k3/9/9/9/9/9/9/9/9/4K4 w - - 0 1");
  bool rejected = false;
  try { evaluator.evaluate(terminal,params,buf,false); } catch(const StringError&) { rejected = true; }
  testAssert(rejected);
  evaluator.killServerThreads();
}

void rulesAndHistory() {
  State start;
  testAssert(start.historyHash() == start.mirroredHistoryHash());
  State unreflected, mirrored;
  for(const char* move : {"b0c2","b9c7","c2b0","c7b9"}) {
    int action = Xiangqi::parseAction(move);
    unreflected.play(action);
    mirrored.play(Xiangqi::transformAction(action,P_BLACK,true));
    testAssert(unreflected.mirroredHistoryHash() == mirrored.historyHash());
    testAssert(mirrored.mirroredHistoryHash() == unreflected.historyHash());
  }
  testAssert(unreflected.historyHash() != unreflected.mirroredHistoryHash()); // Same final board, distinct ordered histories.
  testAssert(start.nextPlayer() == P_BLACK && !start.result().finished);
  testAssert(start.legalActions().size() == 44);
  testAssert(perft(start,2) == 1920);
  testAssert(start.isLegal(Xiangqi::parseAction("b2b9"))); // one cannon screen
  testAssert(!start.isLegal(Xiangqi::parseAction("b2b7"))); // no cannon screen
  testAssert(!start.isLegal(Xiangqi::parseAction("e0e2"))); // king cannot jump
  testAssert(!start.isLegal(Xiangqi::parseAction("a3b3"))); // pawn before river
  testAssert(!start.isLegal(-1) && !start.isLegal(Xiangqi::NUM_ACTIONS));
  const auto original = start.fen();
  bool rejected = false;
  try { play(start,"pass"); } catch(const StringError&) { rejected = true; }
  testAssert(rejected && start.fen() == original);

  State copied;
  {
    State donor;
    cycle(donor);
    testAssert(donor.repetitions() == 2 && !donor.result().finished);
    testAssert(donor.positionKey() == start.positionKey());
    testAssert(donor.historyHash() != start.historyHash());
    copied = donor;
    testAssert(copied.historyHash() == donor.historyHash());
  } // copied must own all StateInfo links after donor destruction.
  State fresh(copied.fen());
  testAssert(fresh.positionKey() == copied.positionKey());
  testAssert(fresh.repetitions() == 1 && copied.repetitions() == 2);
  testAssert(fresh.historyHash() != copied.historyHash());
  cycle(copied);
  testAssert(copied.result().finished && copied.result().winner == C_EMPTY);
  testAssert(copied.result().reason == Reason::REPETITION && copied.legalActions().empty());
  cycle(fresh);
  testAssert(!fresh.result().finished);
  State moved(std::move(fresh));
  cycle(moved);
  testAssert(moved.result().reason == Reason::REPETITION);

  State checker("5k3/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1");
  for(int i = 0; i < 2; i++) for(const char* m : {"e8f8","f9e9","f8e8","e9f9"}) play(checker,m);
  testAssert(checker.result().reason == Reason::REPETITION && checker.result().winner == P_WHITE);
  State chaser("5k3/4n4/9/1R7/4P4/9/9/9/9/4K4 w - - 0 1");
  for(int i = 0; i < 2; i++) for(const char* m : {"b6b8","e8d6","b8b6","d6e8"}) play(chaser,m);
  testAssert(chaser.result().reason == Reason::REPETITION && chaser.result().winner == P_WHITE);

  State material("5k3/9/9/9/9/9/9/9/9/4K4 w - - 0 1");
  testAssert(material.result().reason == Reason::INSUFFICIENT_MATERIAL && material.result().winner == C_EMPTY);
  State stalemate("4k4/3R1R3/9/9/4P4/9/9/9/9/4K4 b - - 0 1");
  testAssert(!stalemate.isCheck() && stalemate.result().reason == Reason::MATE && stalemate.result().winner == P_BLACK);
  State mate("3Rk4/3R5/9/9/4P4/9/9/9/9/4K4 b - - 0 1");
  testAssert(mate.isCheck() && mate.result().reason == Reason::MATE && mate.result().winner == P_BLACK);
  std::string limitFen = start.fen();
  limitFen.replace(limitFen.rfind("0 1"),3,"119 61");
  State limit(limitFen), capture(limitFen);
  play(limit,"b0c2");
  testAssert(limit.result().reason == Reason::MOVE_LIMIT && limit.result().winner == C_EMPTY);
  play(capture,"b2b9");
  testAssert(capture.rule60Count() == 0 && !capture.result().finished);
  limitFen = start.fen(); limitFen.replace(limitFen.rfind("0 1"),3,"112 57");
  State simultaneous(limitFen); cycle(simultaneous); cycle(simultaneous);
  testAssert(simultaneous.result().reason == Reason::REPETITION);

  // Export augmentation must preserve the complete rule state, not just its
  // final FEN. The fixtures include long check, chase and the 120-ply draw.
  for(const State* state : {&start,&unreflected,&copied,&checker,&chaser,&limit,&simultaneous}) {
    State reflected = state->mirrored();
    testAssert(reflected.historyHash() == state->mirroredHistoryHash());
    testAssert(reflected.mirroredHistoryHash() == state->historyHash());
    testAssert(reflected.mirrored().record().initialFen == state->record().initialFen);
    testAssert(reflected.mirrored().record().actions == state->record().actions);
    testAssert(reflected.nextPlayer() == state->nextPlayer());
    testAssert(reflected.gamePly() == state->gamePly() && reflected.rule60Count() == state->rule60Count());
    testAssert(reflected.repetitions() == state->repetitions());
    testAssert(reflected.result().finished == state->result().finished);
    testAssert(reflected.result().reason == state->result().reason && reflected.result().winner == state->result().winner);
    for(int action : state->legalActions()) testAssert(reflected.isLegal(Xiangqi::transformAction(action,P_BLACK,true)));
    for(int i = 0; i < state->numRecentBoards(); i++)
      for(int y = 0; y < Xiangqi::HEIGHT; y++) for(int x = 0; x < Xiangqi::WIDTH; x++)
        testAssert(reflected.pieces(i)[y*Xiangqi::WIDTH+8-x] == state->pieces(i)[y*Xiangqi::WIDTH+x]);
  }

  rejected = false;
  try { State invalid("9/9/9/9/9/9/9/9/9/9 w - - 0 1"); } catch(const StringError&) { rejected = true; }
  testAssert(rejected);
}

void features() {
  State state;
  const auto initial = state.pieces();
  play(state,"b0c2"); play(state,"h9g7");
  testAssert(state.numRecentBoards() == 3 && state.pieces(2) == initial);
  for(Player perspective : {P_BLACK,P_WHITE}) for(bool mirror : {false,true}) {
    const int nx = 12, ny = 11, area = nx*ny;
    std::vector<float> nchw(Xiangqi::SPATIAL_FEATURES*area,-1), nhwc(nchw.size(),-1);
    float global[8], otherGlobal[8];
    state.fillFeatures(perspective,-2,mirror,nx,ny,false,nchw.data(),global);
    state.fillFeatures(perspective,-2,mirror,nx,ny,true,nhwc.data(),otherGlobal);
    for(bool layout : {false,true}) {
      std::vector<float> base(nchw.size()), reflected(nchw.size());
      state.fillFeatures(perspective,-2,false,nx,ny,layout,base.data(),otherGlobal);
      Xiangqi::copyInputsWithMirror(base.data(),reflected.data(),nx,ny,Xiangqi::SPATIAL_FEATURES,layout,mirror);
      testAssert(reflected == (layout ? nhwc : nchw));
    }
    for(int i = 0; i < 8; i++) testAssert(global[i] == otherGlobal[i]);
    testAssert(global[0] == (perspective == P_BLACK) && global[1] == 2/120.0f && global[2] == 0);
    testAssert(global[3] == 1 && global[4] == -1 && global[5] == 3/8.0f && global[7] == 2/200.0f);
    for(int y = 0; y < ny; y++) for(int x = 0; x < nx; x++) {
      const int pos = y*nx+x;
      for(int c = 0; c < Xiangqi::SPATIAL_FEATURES; c++) testAssert(nchw[c*area+pos] == nhwc[pos*Xiangqi::SPATIAL_FEATURES+c]);
      const bool inside = x < 9 && y < 10;
      testAssert(nchw[pos] == inside);
      if(!inside) for(int c = 1; c < Xiangqi::SPATIAL_FEATURES; c++) testAssert(nchw[c*area+pos] == 0);
    }
    // The red horse moved b0 -> c2. It is in plane 5 for red, plane 12 for black.
    int x = perspective == P_BLACK ? 2 : 6, y = perspective == P_BLACK ? 2 : 7;
    if(mirror) x = 8-x;
    testAssert(nchw[(perspective == P_BLACK ? 5 : 12)*area+y*nx+x] == 1);
    const int transformed = Xiangqi::transformAction(Xiangqi::parseAction("b0c2"),perspective,mirror);
    testAssert(Xiangqi::MOVES[transformed][2] == x && Xiangqi::MOVES[transformed][3] == y);
    for(int c = 1+3*14; c < Xiangqi::SPATIAL_FEATURES; c++)
      for(int i = 0; i < area; i++) testAssert(nchw[c*area+i] == 0);
  }
}
}

void Tests::runXiangqiTests() {
  std::cout << "Running Xiangqi state, action and feature tests" << std::endl;
  vocabulary(); rulesAndHistory(); nativeHistory(); features(); outputStorage(); searchStorage(); evaluatorInputsAndCache(); nativeSearch(); secondaryHumanSearch();
}
