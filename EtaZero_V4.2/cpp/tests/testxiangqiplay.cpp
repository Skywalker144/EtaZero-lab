#include "tests.h"
#include "../program/play.h"
#include "../program/playutils.h"
#include "../dataio/sgf.h"
#include "../game/xiangqi.h"
#include "../core/makedir.h"
#include "../external/nlohmann_json/json.hpp"
#include <fstream>
#include <set>

namespace {
using json = nlohmann::json;
using Xiangqi::State;

void helpers() {
  State state;
  BoardHistory hist(state,BoardHistoryModes());
  Board board = state.colorBoard();
  Rand random("xiangqi-play-helpers");
  std::set<Loc> seen, expected;
  for(int action : state.legalActions()) expected.insert(Xiangqi::actionToLoc(action));
  std::vector<Loc> choices(4096);
  testAssert(PlayUtils::chooseRandomLegalMoves(board,hist,P_BLACK,random,choices.data(),choices.size()) == 4096);
  for(Loc loc : choices) { testAssert(hist.isLegal(board,loc,P_BLACK)); seen.insert(loc); }
  testAssert(seen == expected && *seen.rbegin() > Board::MAX_ARR_SIZE);
  Loc banned = *expected.rbegin();
  seen.clear();
  for(int i = 0; i < 4096; i++) {
    Loc loc = PlayUtils::chooseRandomLegalMove(board,hist,P_BLACK,random,banned);
    testAssert(loc != banned && hist.isLegal(board,loc,P_BLACK)); seen.insert(loc);
  }
  expected.erase(banned); testAssert(seen == expected);
  NNOutput output;
  output.nnXLen = output.nnYLen = 10;
  output.resizePolicy(Xiangqi::NUM_ACTIONS);
  std::fill(output.policyProbs,output.policyProbs+Xiangqi::NUM_ACTIONS,0.0f);
  output.policyProbs[Xiangqi::locToAction(banned)] = 1.0f;
  for(double temperature : {0.0,0.5,1.0,2.0}) {
    testAssert(PlayUtils::chooseRandomPolicyMove(&output,board,hist,P_BLACK,random,temperature,false,Board::NULL_LOC) == banned);
    testAssert(PlayUtils::chooseRandomPolicyMove(&output,board,hist,P_BLACK,random,temperature,false,banned) == Board::NULL_LOC);
  }
  // Resignation ends BoardHistory without altering the underlying rule state.
  BoardHistory resigned = hist;
  resigned.setWinnerByResignation(P_WHITE);
  testAssert(!resigned.getXiangqiState().legalActions().empty());
  testAssert(PlayUtils::chooseRandomLegalMove(board,resigned,P_BLACK,random,Board::NULL_LOC) == Board::NULL_LOC);
  testAssert(PlayUtils::chooseRandomLegalMoves(board,resigned,P_BLACK,random,choices.data(),choices.size()) == 0);
  testAssert(PlayUtils::chooseRandomPolicyMove(&output,board,resigned,P_BLACK,random,1.0,false,Board::NULL_LOC) == Board::NULL_LOC);

  // Hint fork must restore the hidden rule history preceding BoardHistory's
  // initial snapshot. Reconstructing from a FEN or the color board loses it.
  State prior("5k3/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1");
  for(const char* m : {"e8f8","f9e9"}) prior.play(Xiangqi::parseAction(m));
  hist = BoardHistory(prior,BoardHistoryModes()); board = prior.colorBoard();
  hist.makeBoardMoveAssumeLegal(board,Xiangqi::actionToLoc(Xiangqi::parseAction("f8e8")),P_BLACK,nullptr);
  FinishedGameData game;
  game.startBoard = board; game.startHist = hist; game.startPla = P_WHITE; game.trainingWeight = 0.75;
  const Loc hint = Xiangqi::actionToLoc(Xiangqi::parseAction("e9f9"));
  Loc alternate = PlayUtils::chooseRandomLegalMove(board,hist,P_WHITE,random,hint);
  testAssert(alternate != Board::NULL_LOC);
  hist.makeBoardMoveAssumeLegal(board,alternate,P_WHITE,nullptr); game.endHist = hist;
  OtherGameProperties props;
  props.hintLoc = hint; props.hintTurn = game.startHist.moveHistory.size();
  props.hintPosHash = game.startHist.getXiangqiState().historyHash();
  ForkData forks;
  Play::maybeHintForkGame(&game,&forks,props);
  testAssert(forks.forks.size() == 1 && forks.forks[0]->isHintFork && forks.forks[0]->trainingWeight == 0.75);
  State continued(prior);
  for(const char* m : {"f8e8","e9f9"}) continued.play(Xiangqi::parseAction(m));
  testAssert(forks.forks[0]->hist.getXiangqiState().historyHash() == continued.historyHash());
  State forked(forks.forks[0]->hist.getXiangqiState());
  for(const char* m : {"e8f8","f9e9","f8e8","e9f9"}) forked.play(Xiangqi::parseAction(m));
  testAssert(forked.result().finished && forked.result().winner == P_WHITE);
  // The same occupancy hash is not a hint identity for Xiangqi.
  props.hintPosHash = game.startBoard.pos_hash;
  Play::maybeHintForkGame(&game,&forks,props);
  testAssert(forks.forks.size() == 1);
}

json record(const BoardHistory& hist, const std::vector<PolicyTargetMove>& policy, const NNRawStats& stats) {
  const auto& state = hist.getXiangqiState();
  json frames = json::array(), targets = json::array();
  for(int i = 0; i < state.numRecentBoards(); i++) frames.push_back(state.pieces(i));
  for(const auto& target : policy) {
    testAssert(state.isLegal(Xiangqi::locToAction(target.loc)));
    targets.push_back({Xiangqi::locToAction(target.loc),target.policyTarget});
  }
  return {{"frames",frames},{"player",state.nextPlayer()},{"rule60",state.rule60Count()},
    {"repetitions",state.repetitions()},{"check",state.isCheck()},{"ply",state.gamePly()},
    {"fen",state.fen()},{"policy",targets},{"entropy",stats.policyEntropy}};
}
}

void Tests::runXiangqiPlay(const std::string& model, const std::string& configFile, const std::string& output) {
  Board::initHash();
  helpers();
  ConfigParser config(configFile);
  MakeDir::make(output);
  Logger logger(nullptr,false,false,false);
  logger.addFile(output + "/play.log");
  const int canvas = config.getInt("dataBoardLen",10,19);
  const bool random = model == "-";
  NNEvaluator evaluator("xiangqi-play",random ? "" : model,"",&logger,8,canvas,canvas,false,false,10,4,random,"",
    config.getBool("testFP16") ? enabled_t::True : enabled_t::False,2,{0,0},"xiangqi-play",false,0,config);
  evaluator.spawnServerThreads();
  SearchParams params;
  params.maxVisits = params.maxPlayouts = 32;
  params.numThreads = 2;
  params.rootNoiseEnabled = true;
  // Fix the evaluated symmetry so post-game entropy checks can re-evaluate
  // the same logits even after eviction from the source symmetry-agnostic cache.
  params.rootNumSymmetriesToSample = 1;
  params.rootSymmetryPruning = false;
  params.useNoisePruning = true;
  params.useUncertainty = true;
  params.useLcbForSelection = true;
  params.gumbel.max_num_considered_actions = 4;
  MatchPairer::BotSpec bot;
  bot.botIdx = 0; bot.botName = "xiangqi-play"; bot.nnEval = &evaluator;
  PlaySettings settings;
  settings.forSelfPlay = true;
  settings.initGamesWithPolicy = true; settings.policyInitAreaProp = 0.05;
  settings.policyInitGammaShape = 2.0;
  settings.cheapSearchProb = 1; settings.cheapSearchVisits = 8; settings.cheapSearchTargetWeight = 0;
  settings.useReanalyze = true; settings.reanalyzeProp = 1; settings.reanalyzeSurpriseExponent = 0;
  settings.sidePositionProb = 1;
  settings.recordTreePositions = true; settings.recordTreeThreshold = 2; settings.recordTreeTargetWeight = 0.125f;
  settings.policySurpriseDataWeight = 0.25; settings.valueSurpriseDataWeight = 0.25;
  // Force both source fork branches in the fixture, independently of sampling probability.
  settings.earlyForkGameProb = 1; settings.earlyForkGameExpectedMoveProp = 0.05;
  settings.forkGameMinChoices = 2; settings.earlyForkGameMaxChoices = settings.forkGameMaxChoices = 8;
  json report = json::array();
  for(int mode = 0; mode < 3; mode++) {
    params.useGumbel = mode > 0; params.useGumbelNonroot = mode == 2;
    params.useGraphSearch = params.useEvalCache = mode == 0;
    bot.baseParams = params;
    settings.reanalyzeUseOutcomeTargets = mode != 1;
    settings.earlyForkGameProb = mode == 1 ? 0 : 1; settings.forkGameProb = 1;
    GameRunner runner(config,"xiangqi-game-init",settings,logger);
    ForkData forks;
    for(int game = 0; game < 3; game++) {
      if(game == 2) {
        for(const auto* fork : forks.forks) delete fork;
        forks.forks.clear();
        State terminalNext("3aka3/9/9/9/4P4/9/9/9/9/R3K4 w - - 119 60");
        testAssert(!terminalNext.result().finished);
        for(int action : terminalNext.legalActions()) {
          State next(terminalNext); next.play(action);
          testAssert(next.result().finished && next.result().winner == C_EMPTY);
        }
        BoardHistory history(terminalNext,BoardHistoryModes());
        forks.add(new InitialPosition(terminalNext.colorBoard(),history,P_BLACK,true,false,0.75));
      }
      int roots = 0;
      auto onMove = [&](const Board& board, const BoardHistory& hist, Player pla, Loc move,
                        const std::vector<double>&, const Search* search) {
        testAssert(hist.isLegal(board,move,pla));
        testAssert(search->getRootHist().getXiangqiState().historyHash() == hist.getXiangqiState().historyHash());
        testAssert(search->getRootVisits() > 0);
        roots++;
      };
      std::unique_ptr<FinishedGameData> data(runner.runGame("xiangqi-play-"+std::to_string(mode)+"-"+std::to_string(game),
        bot,bot,&forks,nullptr,logger,[]() noexcept {return false;},nullptr,nullptr,nullptr,onMove));
      testAssert(data && data->hasFullData && roots > 0);
      testAssert(data->policyTargetsByTurn.size() == (size_t)roots);
      testAssert(data->whiteValueTargetsByTurn.size() == (size_t)roots+1);
      testAssert(data->handicapForSgf == 0 && !data->endHist.isNoResult);
      testAssert((game == 2) == data->endHist.isGameFinished);
      testAssert((game != 2) == data->hitTurnLimit);
      if(game == 2) testAssert(data->whiteValueTargetsByTurn.back().draw == 1);
      Board board = data->startBoard;
      BoardHistory hist = data->startHist;
      json positions = json::array(), sides = json::array();
      int reanalyzed = 0, treeSides = 0;
      auto checkEntropy = [&](const Board& b, const BoardHistory& h, const NNRawStats& stats) {
        testAssert(std::isfinite(stats.policyEntropy) && stats.policyEntropy >= 0);
        testAssert(stats.policyEntropy <= log(h.getXiangqiState().legalActions().size())+1e-5);
        // Source random evaluation draws fresh logits after cache eviction;
        // numerical re-evaluation is meaningful only with actual model weights.
        if(random) return;
        NNResultBuf buf;
        evaluator.evaluate(b,h,h.presumedNextMovePla,MiscNNInputParams(),buf,false);
        double entropy = 0;
        for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) {
          double p = buf.result->policyProbs[action];
          if(p >= 1e-30) entropy -= p*log(p);
        }
        testAssert(std::abs(entropy-stats.policyEntropy) < 1e-5);
      };
      for(int turn = 0; turn < roots; turn++) {
        const auto& policy = *data->policyTargetsByTurn[turn].policyTargets;
        positions.push_back(record(hist,policy,data->nnRawStatsByTurn[turn]));
        checkEntropy(board,hist,data->nnRawStatsByTurn[turn]);
        const auto& reanalysis = data->reanalysisByTurn[turn];
        testAssert(reanalysis.wasReanalyzed && reanalysis.usedOutcomeTargets == settings.reanalyzeUseOutcomeTargets);
        testAssert(reanalysis.originalNumVisits <= 8+params.numThreads);
        reanalyzed++;
        Move move = data->endHist.moveHistory[data->startHist.moveHistory.size()+turn];
        testAssert(hist.isLegal(board,move.loc,move.pla));
        hist.makeBoardMoveAssumeLegal(board,move.loc,move.pla,nullptr);
      }
      testAssert(hist.getXiangqiState().historyHash() == data->endHist.getXiangqiState().historyHash());
      for(const auto* side : data->sidePositions) {
        testAssert(!side->hist.isGameFinished);
        for(const auto& q : side->whiteQValueTargets.targets) testAssert(side->hist.isLegal(side->board,q.loc,side->pla));
        if(side->targetWeightUnrounded == settings.recordTreeTargetWeight) treeSides++;
        sides.push_back(record(side->hist,side->policyTarget,side->nnRawStats));
        checkEntropy(side->board,side->hist,side->nnRawStats);
      }
      std::string path = output+"/"+std::to_string(mode)+"-"+std::to_string(game);
      MakeDir::make(path);
      std::ostringstream sgfText;
      WriteSgf::writeSgf(sgfText,"red","black",data->endHist,data.get(),false,false);
      std::ofstream sgfFile(path+"/game.sgf"); sgfFile << sgfText.str();
      CompactSgf sgf(*Sgf::parse(sgfText.str()));
      Board replayBoard;
      BoardHistory replayHist;
      Player replayPla;
      sgf.setupBoardAndHistTolerant(data->endHist.rules,replayBoard,replayPla,replayHist,sgf.moves.size(),false,data->endHist.modes);
      testAssert(replayHist.getXiangqiState().historyHash() == data->endHist.getXiangqiState().historyHash());
      TrainingDataWriter writer(path,9,128,1.,canvas,canvas,"xiangqi-play-writer");
      writer.writeGame(*data); writer.flushIfNonempty();
      json row = {{"mode",mode},{"game",game},{"roots",roots},{"reanalyzed",reanalyzed},{"tree_side",treeSides},
        {"used_fork",data->usedInitialPosition},{"opening",data->startHist.moveHistory.size()},
        {"truncated",data->hitTurnLimit},{"outcome_targets",settings.reanalyzeUseOutcomeTargets},
        {"positions",positions},{"side",sides},{"rows",writer.numRowsWritten()}};
      report.push_back(row);
    }
  }
  std::ofstream out(output+"/report.json"); out << report.dump();
  evaluator.killServerThreads();
}
