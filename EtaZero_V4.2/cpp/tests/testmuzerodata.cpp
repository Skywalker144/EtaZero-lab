#include "tests.h"
#include "../core/makedir.h"
#include "../dataio/trainingwrite.h"
#include "../game/xiangqi.h"

namespace {
ValueTargets searchValue(int turn) {
  ValueTargets v;
  v.win = 0.1f + 0.02f * turn;
  v.draw = 0.2f;
  v.noResult = 0.0f;
  v.loss = 1.0f - v.win - v.draw;
  return v;
}

void generate(TrainingDataWriter& ordinary, TrainingDataWriter& muZero, Rules::Game game,
              bool truncate, bool specialEnding, int identity) {
  FinishedGameData data;
  const bool xiangqi = game == Rules::Game::XIANGQI;
  const bool noResult = specialEnding && !xiangqi;
  const bool longCheck = specialEnding && xiangqi;
  Board board(xiangqi ? 9 : 3,xiangqi ? 10 : 3);
  Rules rules = Rules::forGame(game);
  rules.komi = 0.0f;
  BoardHistory hist = xiangqi ? BoardHistory(Xiangqi::State(longCheck ?
    "5k3/3RR4/9/9/4P4/9/9/9/9/4K4 w - - 0 1" : Xiangqi::State::START_FEN),BoardHistoryModes(false,false)) :
    BoardHistory(board,P_BLACK,rules,0,BoardHistoryModes(false,false));
  if(xiangqi) board = hist.getRecentBoard(0);
  // A real opening prefix is not a sampled training position.
  hist.makeBoardMoveAssumeLegal(board,xiangqi ? Xiangqi::actionToLoc(Xiangqi::parseAction(longCheck ? "e8f8" : "b0c2")) :
    Location::getLoc(2,2,3),P_BLACK,NULL);
  data.startBoard = board;
  data.startHist = hist;
  data.startPla = P_WHITE;
  data.gameHash = Hash128(101 + identity,301 + identity);
  data.drawEquivalentWinsForWhite = 0.5;
  data.playoutDoublingAdvantagePla = C_EMPTY;
  data.playoutDoublingAdvantage = 0.0;
  data.trainingWeight = 0.75;
  data.hasFullData = true;
  data.changedNeuralNets.push_back(new ChangedNeuralNet("later-model",3));

  const float weights[] = {2.4f,0.0f,0.5f,1.6f,0.0f,2.0f,0.25f,1.25f};
  int turn = 0;
  const char* xiangqiDraw[] = {"b9c7","c2b0","c7b9","b0c2","b9c7","c2b0","c7b9"};
  const char* xiangqiWin[] = {"f9e9","f8e8","e9f9","e8f8","f9e9","f8e8","e9f9"};
  while(!hist.isGameFinished && !(truncate && turn == 3)) {
    Player pla = hist.presumedNextMovePla;
    testAssert(!xiangqi || turn < 7);
    Loc move = xiangqi ? Xiangqi::actionToLoc(Xiangqi::parseAction((longCheck ? xiangqiWin : xiangqiDraw)[turn])) :
      game == Rules::Game::GO && turn >= 4 ? Board::PASS_LOC : Location::getLoc(turn%3,turn/3,3);
    testAssert(turn < 8 && hist.isLegal(board,move,pla));
    auto* policy = new std::vector<PolicyTargetMove>{PolicyTargetMove(move,1000+turn)};
    data.policyTargetsByTurn.emplace_back(policy,64+turn);
    data.targetWeightByTurn.push_back(weights[turn]);
    // The zero root at turn 1 models a low-weight position rounded away by play.cpp.
    data.targetWeightByTurnUnrounded.push_back(turn == 1 ? 0.125f : weights[turn]);
    data.policySurpriseByTurn.push_back(0.25+0.01*turn);
    data.policyEntropyByTurn.push_back(0.5+0.01*turn);
    data.searchEntropyByTurn.push_back(0.75+0.01*turn);
    data.whiteValueTargetsByTurn.push_back(searchValue(turn));
    QValueTargets q;
    q.targets.emplace_back(move,0.123456f,23+turn);
    data.whiteQValueTargetsByTurn.push_back(q);
    NNRawStats raw;
    raw.whiteWinLoss = 0.02*turn;
    raw.policyEntropy = 0.1*turn;
    data.nnRawStatsByTurn.push_back(raw);
    ReanalysisData reanalysis;
    if(turn == 1 || turn == 2) {
      reanalysis.wasReanalyzed = true;
      reanalysis.usedOutcomeTargets = turn == 2;
      reanalysis.selectionPolicySurprise = 0.5;
      reanalysis.selectionValueSurprise = 0.25;
      reanalysis.originalNumVisits = 8;
      reanalysis.numNeuralNetChangesSoFar = 1;
    }
    data.reanalysisByTurn.push_back(reanalysis);
    if(turn == 0) {
      // A searched fork with no played continuation remains root-only.
      auto* side = new SidePosition(board,hist,pla,0);
      side->policyTarget = *policy;
      side->whiteValueTargets = searchValue(0);
      side->whiteQValueTargets = q;
      side->targetWeight = 1.4f;
      side->targetWeightUnrounded = 1.4f;
      side->nnRawStats = raw;
      data.sidePositions.push_back(side);
    }
    hist.makeBoardMoveAssumeLegal(board,move,pla,NULL);
    turn++;
  }
  data.endHist = hist;
  data.hitTurnLimit = truncate;
  if(noResult) {
    testAssert(hist.isGameFinished);
    data.endHist.isNoResult = true;
    data.endHist.winner = C_EMPTY;
  }
  ValueTargets last;
  if(truncate) {
    if(game == Rules::Game::GO) last.draw = 1.0f; // Source move-limit adjudication.
    else last = searchValue(turn); // Placement endpoint bootstrap.
  }
  else if(noResult) last.noResult = 1.0f;
  else if(hist.winner == P_WHITE) last.win = 1.0f;
  else if(hist.winner == P_BLACK) last.loss = 1.0f;
  else last.draw = 1.0f;
  data.whiteValueTargetsByTurn.push_back(last);
  ordinary.writeGame(data);
  muZero.writeGame(data);
  testAssert(ordinary.numRowsWritten() == muZero.numRowsWritten());
  testAssert(ordinary.numRowsInBuffer() == muZero.numRowsInBuffer());
}
}

void Tests::runMuZeroData(const std::string& directory) {
  Board::initHash();
  MakeDir::make(directory);
  int identity = 0;
  for(auto game : {Rules::Game::GO,Rules::Game::RENJU,Rules::Game::HEX,Rules::Game::XIANGQI}) {
    for(int ending = 0; ending < (game == Rules::Game::GO || game == Rules::Game::XIANGQI ? 3 : 2); ending++) {
      std::string name = directory + "/" + std::to_string(identity);
      MakeDir::make(name);
      MakeDir::make(name + "/az");
      MakeDir::make(name + "/mu");
      int version = game == Rules::Game::GO ? 7 : game == Rules::Game::XIANGQI ? 9 : 8;
      int canvas = version == 9 ? (ending == 2 ? 12 : 10) : 5;
      TrainingDataWriter ordinary(name + "/az",version,4,0.5,canvas,canvas,"paired-writer");
      TrainingDataWriter muZero(name + "/mu",version,4,0.5,canvas,canvas,"paired-writer",true);
      generate(ordinary,muZero,game,ending == 1,ending == 2,identity*2);
      generate(ordinary,muZero,game,ending == 1,ending == 2,identity*2+1);
      ordinary.flushIfNonempty();
      muZero.flushIfNonempty();
      identity++;
    }
  }
}
