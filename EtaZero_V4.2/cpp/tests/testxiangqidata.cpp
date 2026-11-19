#include "tests.h"
#include "../core/makedir.h"
#include "../dataio/trainingwrite.h"
#include "../game/xiangqi.h"
#include "../external/nlohmann_json/json.hpp"
#include <fstream>

namespace {
using json = nlohmann::json;
using Xiangqi::State;

ValueTargets estimate(int turn) {
  ValueTargets v;
  v.win = 0.2f + 0.005f*turn; v.draw = 0.1f; v.loss = 1-v.win-v.draw;
  return v;
}

json position(const State& state) {
  json frames = json::array();
  for(int i = 0; i < state.numRecentBoards(); i++) frames.push_back(state.pieces(i));
  return {{"frames",frames},{"player",state.nextPlayer()},{"rule60",state.rule60Count()},
          {"repetitions",state.repetitions()},{"check",state.isCheck()},{"ply",state.gamePly()},{"fen",state.fen()}};
}

void targets(const State& state, std::vector<PolicyTargetMove>& policy, QValueTargets& q, json& expected) {
  expected["policy"] = json::array(); expected["q"] = json::array();
  int i = 0;
  for(int action : state.legalActions()) {
    policy.emplace_back(Xiangqi::actionToLoc(action),i+1);
    q.targets.emplace_back(Xiangqi::actionToLoc(action),0.25f,i+5);
    expected["policy"].push_back({action,i+1});
    expected["q"].push_back({action,0.25,i+5});
    i++;
  }
}

void generate(const std::string& directory, int canvas, bool truncated) {
  const char* checking[] = {"e8f8","f9e9","f8e8","e9f9","e8f8","f9e9","f8e8","e9f9"};
  State prior(truncated ? State::START_FEN : "5k3/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1");
  prior.play(Xiangqi::parseAction(truncated ? "b0c2" : checking[0]));
  prior.play(Xiangqi::parseAction(truncated ? "h9g7" : checking[1]));
  BoardHistory hist(prior,BoardHistoryModes());
  Board board = prior.colorBoard();
  hist.makeBoardMoveAssumeLegal(board,Xiangqi::actionToLoc(Xiangqi::parseAction(truncated ? "h0g2" : checking[2])),P_BLACK,nullptr);
  FinishedGameData game;
  game.startBoard = board; game.startHist = hist; game.startPla = P_WHITE;
  game.gameHash = Hash128(100+canvas,200+truncated);
  game.drawEquivalentWinsForWhite = 0.5;
  game.playoutDoublingAdvantagePla = P_BLACK; game.playoutDoublingAdvantage = 0.5;
  game.trainingWeight = 0.75; game.hasFullData = true;
  Rand random("xiangqi-writer-moves");
  json positions = json::array(), sideExpected;
  for(int turn = 0; turn < (truncated ? 36 : 5); turn++) {
    const State& state = hist.getXiangqiState();
    testAssert(!state.result().finished);
    json expected = position(state);
    auto* policy = new std::vector<PolicyTargetMove>();
    QValueTargets q;
    targets(state,*policy,q,expected);
    game.policyTargetsByTurn.emplace_back(policy,64+turn);
    game.whiteQValueTargetsByTurn.push_back(q);
    game.whiteValueTargetsByTurn.push_back(estimate(turn));
    game.targetWeightByTurn.push_back(turn == 0 ? 2 : turn == 1 ? 0 : 1);
    game.targetWeightByTurnUnrounded.push_back(game.targetWeightByTurn.back());
    game.policySurpriseByTurn.push_back(0.25); game.policyEntropyByTurn.push_back(0.5); game.searchEntropyByTurn.push_back(0.75);
    game.nnRawStatsByTurn.push_back(NNRawStats{0.125,0.625});
    ReanalysisData reanalysis;
    if(turn == 2 || turn == 3) {
      reanalysis.wasReanalyzed = true; reanalysis.usedOutcomeTargets = turn == 3;
      reanalysis.originalNumVisits = 8;
    }
    game.reanalysisByTurn.push_back(reanalysis);
    int action = truncated ? -1 : Xiangqi::parseAction(checking[turn+3]);
    if(truncated) {
      const auto& legal = state.legalActions();
      int start = random.nextUInt((uint32_t)legal.size());
      for(int offset = 0; offset < legal.size(); offset++) {
        State candidate(state);
        int proposed = legal[(start+offset)%legal.size()];
        candidate.play(proposed);
        if(!candidate.result().finished) { action = proposed; break; }
      }
      testAssert(action >= 0);
    }
    if(turn == 0) {
      // Actual alternative continuation, with complete opening rule history.
      for(int other : state.legalActions()) {
        if(other == action) continue;
        auto sideHist = hist; auto sideBoard = board;
        sideHist.makeBoardMoveAssumeLegal(sideBoard,Xiangqi::actionToLoc(other),hist.presumedNextMovePla,nullptr);
        if(sideHist.isGameFinished) continue;
        auto* side = new SidePosition(sideBoard,sideHist,sideHist.presumedNextMovePla,0);
        side->unreducedNumVisits = 10000;
        sideExpected = position(sideHist.getXiangqiState());
        targets(sideHist.getXiangqiState(),side->policyTarget,side->whiteQValueTargets,sideExpected);
        side->whiteValueTargets = estimate(0); side->nnRawStats = NNRawStats{0.125,0.625};
        game.sidePositions.push_back(side);
        break;
      }
      testAssert(game.sidePositions.size() == 1);
    }
    positions.push_back(expected);
    hist.makeBoardMoveAssumeLegal(board,Xiangqi::actionToLoc(action),hist.presumedNextMovePla,nullptr);
  }
  positions.push_back(position(hist.getXiangqiState()));
  game.endHist = hist; game.hitTurnLimit = truncated;
  auto finalValue = estimate(36);
  if(!truncated) {
    testAssert(hist.isGameFinished && hist.winner == P_WHITE);
    finalValue = ValueTargets(); finalValue.win = 1;
  }
  game.whiteValueTargetsByTurn.push_back(finalValue);
  TrainingDataWriter writer(directory,9,64,1.0,canvas,canvas,"xiangqi-training-data");
  writer.writeGame(game); writer.flushIfNonempty();
  std::ofstream out(directory + "/expected.json");
  out << json{{"positions",positions},{"side",sideExpected},{"truncated",truncated},
              {"canvas",canvas},{"rows",writer.numRowsWritten()}}.dump();
  std::ostringstream debug;
  game.printDebug(debug);
  testAssert(debug.str().find(Xiangqi::actionToString(Xiangqi::locToAction(game.policyTargetsByTurn[0].policyTargets->at(0).loc))) != std::string::npos);
}
}

void Tests::runXiangqiData(const std::string& directory) {
  Board::initHash();
  MakeDir::make(directory);
  for(int canvas : {10,12}) for(bool truncated : {false,true}) {
    std::string path = directory + "/" + std::to_string(canvas) + (truncated ? "-truncated" : "-terminal");
    MakeDir::make(path);
    generate(path,canvas,truncated);
  }
  bool rejected = false;
  try { TrainingWriteBuffers buffers(9,1,113,8,10,10,true); }
  catch(const StringError&) { rejected = true; }
  testAssert(rejected);
}
