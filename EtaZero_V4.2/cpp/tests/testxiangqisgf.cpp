#include "tests.h"
#include "../dataio/sgf.h"
#include "../game/xiangqi.h"
#include "../program/play.h"
#include "../core/makedir.h"
#include "../external/nlohmann_json/json.hpp"
#include <fstream>

namespace {
using Xiangqi::State;
void rejects(const std::function<void()>& f) {
  bool rejected = false;
  try { f(); } catch(const std::exception&) { rejected = true; }
  testAssert(rejected);
}
void same(const State& a, const State& b) {
  testAssert(a.fen() == b.fen() && a.historyHash() == b.historyHash() && a.mirroredHistoryHash() == b.mirroredHistoryHash());
  testAssert(a.positionKey() == b.positionKey() && a.repetitions() == b.repetitions());
  testAssert(a.legalActions() == b.legalActions() && a.numRecentBoards() == b.numRecentBoards());
  testAssert(a.result().finished == b.result().finished && a.result().winner == b.result().winner);
  for(int i = 0; i < a.numRecentBoards(); i++) testAssert(a.pieces(i) == b.pieces(i));
}
}

void Tests::runXiangqiSgf(const std::string& output) {
  using json = nlohmann::json;
  Board::initHash(); MakeDir::make(output);
  const std::string fen = "5k3/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1";
  const std::vector<std::string> checking = {"e8f8","f9e9","f8e8","e9f9","e8f8","f9e9","f8e8","e9f9"};
  State prior(fen);
  for(int i = 0; i < 2; i++) prior.play(Xiangqi::parseAction(checking[i]));
  BoardHistory hist(prior,BoardHistoryModes(true,true));
  hist.setInitialTurnNumber(97);
  Board board = prior.colorBoard();
  for(int i = 2; i < 7; i++) hist.makeBoardMoveAssumeLegal(board,Xiangqi::actionToLoc(Xiangqi::parseAction(checking[i])),hist.presumedNextMovePla,nullptr);
  const State& current = hist.getXiangqiState();
  for(int ago = 0; ago <= 7; ago++) {
    auto record = current.record(ago);
    testAssert(record.initialFen == fen && record.actions.size() == (size_t)(7-ago));
    State expected(fen);
    for(int i = 0; i < 7-ago; i++) {
      testAssert(Xiangqi::actionToString(record.actions[i]) == checking[i]);
      expected.play(record.actions[i]);
    }
    same(State(record),expected);
  }
  // Undoing a capture must restore the origin piece identities too.
  State captured; captured.play(Xiangqi::parseAction("b2b9"));
  auto captureRecord = captured.record();
  testAssert(captureRecord.initialFen == State::START_FEN);
  same(State(captureRecord),captured);
  same(State(captured.record(1)),State());
  rejects([&] { current.record(8); });

  Sgf::PositionSample sample;
  Sgf::PositionSample::writePosOfHist(sample,hist,P_WHITE);
  sample.hintLoc = Xiangqi::actionToLoc(Xiangqi::parseAction(checking.back()));
  sample.weight = 2.5; sample.trainingWeight = 0.375; sample.metadata = "xiangqi full-history hint";
  testAssert(sample.moves.size() == 5 && sample.initialTurnNumber == 97 && sample.getCurrentTurnNumber() == 102);
  same(*sample.xiangqiState,prior);
  const auto line = Sgf::PositionSample::toJsonLine(sample);
  std::ofstream hint(output+"/position.hintposes.txt"); hint << line << '\n'; hint.close();
  auto loaded = Sgf::PositionSample::ofJsonLine(line);
  testAssert(sample.isEqualForTesting(loaded,true,true));
  testAssert(loaded.trainingWeight == sample.trainingWeight && loaded.metadata == sample.metadata);
  BoardHistory restored;
  Player pla;
  testAssert(loaded.tryGetCurrentBoardHistory(hist.rules,pla,restored,hist.modes));
  testAssert(pla == P_WHITE && restored.getCurrentTurnNumber() == 102);
  same(restored.getXiangqiState(),current);
  State finish(restored.getXiangqiState()); finish.play(Xiangqi::locToAction(sample.hintLoc));
  testAssert(finish.result().finished && finish.result().winner == P_WHITE);
  same(State(finish.record()),finish);
  State lostHistory(prior.fen());
  for(int i = 2; i < 8; i++) lostHistory.play(Xiangqi::parseAction(checking[i]));
  testAssert(!lostHistory.result().finished);
  auto previous = loaded.previousPosition(0.5);
  testAssert(previous.hintLoc == Board::NULL_LOC && previous.weight == 0.5 && previous.hasPreviousPositions(4));
  testAssert(previous.tryGetCurrentBoardHistory(hist.rules,pla,restored,hist.modes));
  same(restored.getXiangqiState(),State(current.record(1)));
  rejects([&] { loaded.getColorFlipped(); });
  for(const auto& change : std::vector<json>{{{"nextPla","W"}},{{"xSize",10}},{{"xiangqiHistory",{"e8e9"}}},{{"moveLocs",{"pass"}},{"movePlas",{"B"}}}}) {
    json bad = json::parse(line); bad.update(change);
    rejects([&] { Sgf::PositionSample::ofJsonLine(bad.dump()); });
  }

  std::ostringstream sgfText;
  WriteSgf::writeSgf(sgfText,"red","black",hist,nullptr,false,false);
  testAssert(sgfText.str().find("GM[7]SZ[9:10]") != std::string::npos);
  testAssert(sgfText.str().find("XQHIST[e8f8][f9e9]") != std::string::npos);
  testAssert(sgfText.str().find("AB[") == std::string::npos && sgfText.str().find("KM[") == std::string::npos);
  std::ofstream sgfFile(output+"/position.sgf"); sgfFile << sgfText.str(); sgfFile.close();
  auto sgf = Sgf::parse(sgfText.str());
  CompactSgf compact(*sgf);
  testAssert(compact.moves.size() == 5 && compact.getRulesOrFail() == hist.rules);
  compact.setupBoardAndHistTolerant(hist.rules,board,pla,restored,5,false,hist.modes);
  same(restored.getXiangqiState(),current);
  testAssert(restored.getCurrentTurnNumber() == 102);
  int count = 0;
  std::set<Hash128> unique;
  auto checkSample = [&](Sgf::PositionSample& pose, const BoardHistory& position, const std::string&) {
    Player p; BoardHistory h;
    testAssert(pose.tryGetCurrentBoardHistory(hist.rules,p,h,position.modes));
    same(h.getXiangqiState(),position.getXiangqiState());
    testAssert(pose.getCurrentTurnNumber() == position.getCurrentTurnNumber()); count++;
  };
  sgf->iterAllUniquePositions(unique,false,true,true,false,nullptr,checkSample);
  testAssert(count == 6);
  sgf->iterAllUniquePositions(unique,false,true,true,false,nullptr,checkSample);
  testAssert(count == 6);
  auto branch = Sgf::parse("(;GM[7];B[b0c2](;W[h9g7])(;W[b9c7]))");
  count = 0; branch->iterAllPositions(true,false,nullptr,checkSample); testAssert(count == 4);
  auto setup = Sgf::parse("(;GM[7];B[b0c2];XQFEN["+fen+"];B[e8f8])");
  count = 0; setup->iterAllPositions(true,false,nullptr,checkSample); testAssert(count == 4);
  rejects([&] { CompactSgf invalid(*setup); });
  for(const char* bad : {"(;GM[1]RU[xiangqi])", "(;GM[7]SZ[10])", "(;GM[7]AB[aa])",
      "(;GM[7]PL[W])", "(;GM[7];B[aa])", "(;GM[7];B[])", "(;GM[7];W[b0c2])", "(;GM[7];B[b0b2])"}) {
    rejects([&] { Sgf::parse(bad)->iterAllPositions(false,false,nullptr,checkSample,true); });
    rejects([&] {
      CompactSgf invalid(*Sgf::parse(bad));
      invalid.setupBoardAndHistTolerant(hist.rules,board,pla,restored,invalid.moves.size(),false,hist.modes);
    });
  }

  Logger logger(nullptr,false,false,false);
  for(bool hints : {true,false}) {
    std::map<std::string,std::string> values{{"game","xiangqi"},{"bSizesXY","9-10"},{"bSizeRelProbs","1"}};
    if(hints) { values["hintPosesDir"] = output; values["hintPosesProb"] = "1"; }
    else {
      values["startPosesFromSgfDir"] = output; values["startPosesProb"] = "1";
      values["startPosesLoadProb"] = "1"; values["startPosesTurnWeightLambda"] = "0";
    }
    ConfigParser config(values);
    GameInitializer init(config,logger,"xiangqi-files");
    OtherGameProperties props; PlaySettings settings; int extra;
    init.createGame(board,pla,restored,extra,nullptr,settings,props,nullptr,hist.modes);
    testAssert(restored.rules.isXiangqi() && !restored.isGameFinished && extra == 0);
    if(hints) {
      same(restored.getXiangqiState(),current);
      testAssert(props.isHintPos && !props.allowPolicyInit && props.trainingWeight == 0.375);
      testAssert(props.hintLoc == sample.hintLoc && props.hintPosHash == current.historyHash());
    }
    else {
      testAssert(props.isSgfPos && props.allowPolicyInit);
      auto record = restored.getXiangqiState().record();
      testAssert(record.initialFen == fen && record.actions.size() >= 2 && record.actions.size() <= 7);
      for(size_t i = 0; i < record.actions.size(); i++) testAssert(Xiangqi::actionToString(record.actions[i]) == checking[i]);
    }
  }
}
