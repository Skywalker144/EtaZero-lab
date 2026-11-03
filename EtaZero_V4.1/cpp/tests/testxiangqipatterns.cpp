#include "tests.h"
#include "../core/makedir.h"
#include "../dataio/sgf.h"
#include "../game/xiangqi.h"
#include "../neuralnet/nneval.h"
#include "../search/localpattern.h"
#include "../search/patternbonustable.h"
#include "../search/search.h"
#include "../search/searchnode.h"
#include <fstream>
#include <map>
#include <set>

namespace {
Loc move(const char* text) { return Xiangqi::actionToLoc(Xiangqi::parseAction(text)); }
void play(BoardHistory& hist, const char* text) {
  Board board = hist.getRecentBoard(0);
  testAssert(hist.makeBoardMoveTolerant(board,move(text),hist.presumedNextMovePla));
}
bool close(double a, double b) { return std::abs(a-b) < 1e-9 * std::max(1.0,std::abs(a)); }

void hashGeometry() {
  Xiangqi::State state;
  state.play(Xiangqi::parseAction("b0c2")); state.play(Xiangqi::parseAction("h9g7"));
  for(int size : {5,9}) {
    Rand rand("xiangqi-pattern-reference");
    XiangqiLocalPatternHasher hasher;
    hasher.init(size,size,rand);
    for(bool flip : {false,true}) for(bool mirror : {false,true}) {
      std::array<int8_t,90> transformed{};
      // Transform an entire physical board, independently of the local windows.
      for(int y = 0; y < 10; y++) for(int x = 0; x < 9; x++) {
        int sx = flip ? 8-x : x, sy = flip ? 9-y : y;
        if(mirror) sx = 8-sx;
        transformed[sy*9+sx] = state.pieces()[y*9+x] * (flip ? -1 : 1);
      }
      for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) {
        Loc loc = Xiangqi::actionToLoc(action);
        Loc symLoc = Xiangqi::actionToLoc(Xiangqi::transformAction(action,flip ? P_WHITE : P_BLACK,mirror));
        for(Player pla : {P_BLACK,P_WHITE})
          testAssert(hasher.getHash(state.pieces(),loc,pla,mirror,flip) ==
                     hasher.getHash(transformed,symLoc,flip ? getOpp(pla) : pla));
      }
    }
    auto pieces = Xiangqi::State().pieces();
    auto original = hasher.getHash(pieces,move("b0c2"),P_BLACK);
    pieces[1] = pieces[0]; // Horse -> rook, with identical color occupancy.
    testAssert(hasher.getHash(pieces,move("b0c2"),P_BLACK) != original);
    pieces = Xiangqi::State().pieces();
    original = hasher.getHash(pieces,move("b2b9"),P_BLACK);
    pieces[9*9+1] = pieces[9*9]; // Destination piece identity also matters.
    testAssert(hasher.getHash(pieces,move("b2b9"),P_BLACK) != original);
    if(size == 5) {
      pieces = Xiangqi::State().pieces();
      original = hasher.getHash(pieces,move("b0c2"),P_BLACK);
      pieces[9*9+7] = 0; // Outside both 5x5 neighborhoods.
      testAssert(hasher.getHash(pieces,move("b0c2"),P_BLACK) == original);
    }
  }
}

void tablesAndReplay() {
  BoardHistory hist{Xiangqi::State(),BoardHistoryModes()};
  const auto initial = hist;
  PatternBonusTable bonus;
  SubtreeValueBiasTable bias(16);
  std::set<Hash128> keys;
  for(int a = 0; a < Xiangqi::NUM_ACTIONS; a++) {
    const Loc loc = Xiangqi::actionToLoc(a);
    testAssert(keys.insert(bonus.getHash(P_BLACK,loc,hist)).second);
    testAssert(bias.get(P_BLACK,loc,loc,hist,0) != nullptr);
  }
  testAssert(bonus.getHash(P_BLACK,Board::NULL_LOC,hist) == Hash128());
  testAssert(bonus.getHash(P_BLACK,Board::PASS_LOC,hist) == Hash128());
  auto held = bias.get(P_BLACK,move("b9c7"),move("b0c2"),hist,0);
  testAssert(held == bias.get(P_BLACK,move("b9c7"),move("b0c2"),hist,0));
  testAssert(held != bias.get(P_BLACK,move("h9g7"),move("b0c2"),hist,0));
  bias.clearUnusedSynchronous();
  size_t count = 0; for(const auto& shard : bias.entries) count += shard.size();
  testAssert(count == 1);
  held.reset(); bias.clearUnusedSynchronous();
  for(const auto& shard : bias.entries) testAssert(shard.empty());

  play(hist,"b0c2");
  testAssert(bonus.getHash(P_BLACK,move("b0c2"),hist,1) == bonus.getHash(P_BLACK,move("b0c2"),initial));
  testAssert(bias.get(P_BLACK,move("b9c7"),move("b0c2"),hist,1) ==
             bias.get(P_BLACK,move("b9c7"),move("b0c2"),initial,0));
  for(const char* m : {"b9c7","c2b0","c7b9","b0c2"}) play(hist,m);
  bonus.addBonusForGameMoves(hist,0.125,P_BLACK);
  // The same pattern on the second occurrence is counted once per game.
  testAssert(close(bonus.get(bonus.getHash(P_BLACK,move("b0c2"),initial)).utilityBonus,0.125));
  // Replaying a history whose origin already owns earlier moves uses that full origin.
  BoardHistory continued(hist.getXiangqiState(),BoardHistoryModes());
  testAssert(continued.moveHistory.empty());
  testAssert(continued.getXiangqiState().previousAction() == Xiangqi::parseAction("b0c2"));
  testAssert(continued.getXiangqiState().previousAction(4) == Xiangqi::parseAction("b0c2"));
  testAssert(continued.getXiangqiState().previousAction(5) == -1);
  PatternBonusTable imported;
  imported.addBonusForGameMoves(continued,0.125,P_BLACK);
  testAssert(close(imported.get(imported.getHash(P_BLACK,move("b0c2"),initial)).utilityBonus,0.125));
  const auto before = continued;
  play(continued,"b9c7");
  PatternBonusTable resumed;
  resumed.addBonusForGameMoves(continued,0.25,P_WHITE);
  testAssert(close(resumed.get(resumed.getHash(P_WHITE,move("b9c7"),before)).utilityBonus,0.25));

  // Both valid geometric symmetries and rotation+color swap map whole moves.
  for(bool flip : {false,true}) for(bool mirror : {false,true}) {
    std::string fen = Xiangqi::State::START_FEN;
    if(flip) fen.replace(fen.find(" w "),3," b ");
    BoardHistory sym{Xiangqi::State(fen),BoardHistoryModes()};
    int a = Xiangqi::transformAction(Xiangqi::parseAction("b0c2"),flip ? P_WHITE : P_BLACK,mirror);
    const auto hash = bonus.getHash(flip ? P_WHITE : P_BLACK,Xiangqi::actionToLoc(a),sym);
    testAssert(close(bonus.get(hash).utilityBonus,0.125));
  }

  // Passing the source BoardHistory overload leaves Go's original hashes/slots unchanged.
  Board board(9,9);
  BoardHistory go(board,P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes());
  const Loc loc = Location::getLoc(3,3,9);
  testAssert(bonus.getHash(P_BLACK,loc,board) == bonus.getHash(P_BLACK,loc,go));
  testAssert(bias.get(P_BLACK,Board::PASS_LOC,loc,board) == bias.get(P_BLACK,Board::PASS_LOC,loc,go,0));
}

void files(const std::string& output, Logger& logger) {
  MakeDir::make(output);
  // The cycle revisits the opening pattern. Names and %SKIP% retain source filtering.
  const std::string text = "(;GM[7]SZ[9:10]PB[red]PW[black];B[b0c2];W[b9c7];B[c2b0];W[c7b9];B[b0c2])";
  for(const char* name : {"first.sgf","second.sgf"}) {
    std::ofstream out(output+"/"+name); out << text;
  }
  BoardHistory initial{Xiangqi::State(),BoardHistoryModes()};
  PatternBonusTable sgfs;
  sgfs.avoidRepeatedSgfMoves({output+"/first.sgf",output+"/second.sgf"},.4,.5,0,2,{"red"},logger,"test");
  testAssert(close(sgfs.get(sgfs.getHash(P_BLACK,move("b0c2"),initial)).utilityBonus,.6));
  // Mirrored opening and color-swapped opponent get the source white-utility sign.
  testAssert(close(sgfs.get(sgfs.getHash(P_BLACK,move("h0g2"),initial)).utilityBonus,.6));
  std::string blackFen = Xiangqi::State::START_FEN;
  blackFen.replace(blackFen.find(" w "),3," b ");
  BoardHistory black{Xiangqi::State(blackFen),BoardHistoryModes()};
  testAssert(close(sgfs.get(sgfs.getHash(P_WHITE,move("h9g7"),black)).utilityBonus,-.6));
  PatternBonusTable filtered;
  filtered.avoidRepeatedSgfMoves({output+"/first.sgf"},.4,1.,6,1,{},logger,"min-turn");
  testAssert(filtered.get(filtered.getHash(P_BLACK,move("b0c2"),initial)).utilityBonus == 0);
  {
    std::ofstream out(output+"/skip.sgf"); out << "(;GM[7]SZ[9:10];B[b0c2]C[%SKIP%])";
  }
  filtered.avoidRepeatedSgfMoves({output+"/skip.sgf"},.4,1.,0,1,{},logger,"skip-comment");
  testAssert(filtered.get(filtered.getHash(P_BLACK,move("b0c2"),initial)).utilityBonus == 0);

  MakeDir::make(output+"/poses");
  Sgf::PositionSample sample;
  Sgf::PositionSample::writePosOfHist(sample,initial,P_BLACK);
  sample.hintLoc = move("b0c2"); sample.initialTurnNumber = 7;
  for(const char* name : {"first.hintposes.txt","second.hintposes.txt"}) {
    std::ofstream out(output+"/poses/"+name);
    out << Sgf::PositionSample::toJsonLine(sample) << '\n';
  }
  PatternBonusTable poses;
  poses.avoidRepeatedPosMovesAndDeleteExcessFiles({output+"/poses"},.4,.5,7,7,10,logger,"poses");
  testAssert(close(poses.get(poses.getHash(P_BLACK,move("b0c2"),initial)).utilityBonus,.6));
  testAssert(close(poses.get(poses.getHash(P_WHITE,move("h9g7"),black)).utilityBonus,-.6));
  PatternBonusTable pruned;
  pruned.avoidRepeatedPosMovesAndDeleteExcessFiles({output+"/poses"},.4,.5,7,7,1,logger,"prune");
  testAssert(close(pruned.get(pruned.getHash(P_BLACK,move("b0c2"),initial)).utilityBonus,.4));
}

void searchUpdates(Logger& logger) {
  ConfigParser config(std::map<std::string,std::string>{});
  NNEvaluator evaluator("random","","",&logger,8,10,10,false,false,10,4,true,"",enabled_t::False,
                       1,{0},"xiangqi-pattern-search",false,0,config);
  evaluator.spawnServerThreads();
  for(int threads : {1,4}) {
    SearchParams params;
    params.numThreads = threads; params.maxVisits = params.maxPlayouts = 160;
    params.subtreeValueBiasFactor = 0.4; params.subtreeValueBiasFreeProp = 0.6;
    params.avoidRepeatedPatternUtility = 0.05;
    Search search(params,&evaluator,&logger,"xiangqi-pattern-updates");
    BoardHistory hist{Xiangqi::State(),BoardHistoryModes()};
    play(hist,"b0c2"); play(hist,"h9g7");
    if(threads == 4) hist = BoardHistory(hist.getXiangqiState(),BoardHistoryModes());
    search.setPosition(P_BLACK,hist.getRecentBoard(0),hist);
    search.runWholeSearch(P_BLACK);
    auto rootChildren = search.getRootNode()->getChildren();
    for(int i = 0; i < rootChildren.iterateAndCountChildren(); i++)
      testAssert(rootChildren[i].getIfAllocated()->subtreeValueBiasTableEntry != nullptr);
    const Loc chosen = search.getChosenMoveLoc();
    const auto* promoted = search.getChildForMove(search.getRootNode(),chosen);
    testAssert(promoted != nullptr && promoted->getNNOutput() != nullptr);
    std::set<const SearchNode*> preserved;
    std::function<void(const SearchNode*)> keep = [&](const SearchNode* node) {
      if(!preserved.insert(node).second) return;
      auto children = node->getChildren();
      for(int i = 0; i < children.iterateAndCountChildren(); i++) keep(children[i].getIfAllocated());
    };
    keep(promoted);
    preserved.erase(promoted); // The source promotes a copy without the old bias entry.
    struct Tracked { std::shared_ptr<SubtreeValueBiasEntry> entry; double delta = 0, weight = 0, beforeDelta = 0, beforeWeight = 0; };
    std::map<SubtreeValueBiasEntry*,Tracked> tracked;
    std::set<const SearchNode*> visited;
    std::function<void(const SearchNode*)> visit = [&](const SearchNode* node) {
      if(!visited.insert(node).second) return;
      if(node->subtreeValueBiasTableEntry) {
        auto entry = node->subtreeValueBiasTableEntry;
        auto& row = tracked[entry.get()]; row.entry = entry;
        if(!preserved.count(node)) {
          row.delta += node->lastSubtreeValueBiasDeltaSum;
          row.weight += node->lastSubtreeValueBiasWeight;
        }
        row.beforeDelta = entry->deltaUtilitySum; row.beforeWeight = entry->weightSum;
        testAssert(std::isfinite(row.beforeDelta) && row.beforeWeight >= 0);
      }
      auto children = node->getChildren();
      for(int i = 0; i < children.iterateAndCountChildren(); i++) visit(children[i].getIfAllocated());
    };
    visit(search.getRootNode());
    testAssert(!tracked.empty());
    double weight = 0; for(const auto& pair : tracked) weight += pair.second.beforeWeight;
    testAssert(weight > 0);
    testAssert(search.makeMove(chosen,P_BLACK));
    for(const auto& pair : tracked) {
      const auto& row = pair.second;
      testAssert(close(row.entry->deltaUtilitySum,row.beforeDelta-.6*row.delta));
      testAssert(close(row.entry->weightSum,row.beforeWeight-.6*row.weight));
    }
    // Unlike partial pruning, source clearSearch discards nodes without
    // subtracting their contributions. Retain this source lifecycle distinction.
    search.clearSearch();
    for(const auto& pair : tracked) {
      const auto& row = pair.second;
      testAssert(close(row.entry->deltaUtilitySum,row.beforeDelta-.6*row.delta));
      testAssert(close(row.entry->weightSum,row.beforeWeight-.6*row.weight));
    }
  }
  // A single nonterminal leaf gets exactly the source white-utility bonus.
  SearchParams params;
  params.maxVisits = params.maxPlayouts = 2; params.numThreads = 1;
  Search search(params,&evaluator,&logger,"xiangqi-pattern-leaf");
  BoardHistory hist{Xiangqi::State(),BoardHistoryModes()};
  search.setPosition(P_BLACK,hist.getRecentBoard(0),hist);
  auto table = std::make_unique<PatternBonusTable>();
  std::set<Hash128> hashes;
  for(int a : hist.getXiangqiState().legalActions())
    table->addBonus(P_BLACK,Xiangqi::actionToLoc(a),hist,0,.125,0,false,hashes);
  search.setExternalPatternBonusTable(std::move(table));
  search.runWholeSearch(P_BLACK);
  auto children = search.getRootNode()->getChildren();
  testAssert(children.iterateAndCountChildren() == 1);
  const auto* child = children[0].getIfAllocated();
  const auto* nn = child->getNNOutput();
  testAssert(nn != nullptr && close(child->stats.utilityAvg.load(),double(nn->whiteWinProb)-nn->whiteLossProb+.125));
  evaluator.killServerThreads();
}
}

void Tests::runXiangqiPatternTests(const std::string& output) {
  Board::initHash();
  Logger logger(nullptr,false,false,false);
  hashGeometry(); tablesAndReplay(); files(output,logger); searchUpdates(logger);
  std::cout << "Xiangqi local patterns, source SVB updates and pattern files passed" << std::endl;
}
