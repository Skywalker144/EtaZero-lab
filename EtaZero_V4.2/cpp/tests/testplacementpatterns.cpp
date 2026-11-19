#include "tests.h"
#include "../core/makedir.h"
#include "../dataio/sgf.h"
#include "../game/placement.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../search/localpattern.h"
#include "../search/patternbonustable.h"
#include "../search/search.h"
#include "../search/searchnode.h"
#include <fstream>
#include <set>

namespace {
using G = Rules::Game;
const std::array<G,4> games{{G::FREESTYLE,G::STANDARD,G::RENJU,G::HEX}};
Loc point(const Board& board, int x, int y) { return Location::getLoc(x,y,board.x_size); }
bool close(double a, double b) { return std::abs(a-b) < 1e-9; }

Board transformed(const Board& board, int symmetry, bool flip) {
  Board result(board.x_size,board.y_size);
  for(int y = 0; y < board.y_size; y++) for(int x = 0; x < board.x_size; x++) {
    Color color = board.colors[point(board,x,y)];
    if(flip && (color == P_BLACK || color == P_WHITE)) color = getOpp(color);
    result.colors[SymmetryHelpers::getSymLoc(x,y,board,symmetry)] = color;
  }
  result.regenChainsFromColors(); // No Go capture/zero-liberty cleanup.
  return result;
}

Board asymmetricBoard() {
  Board board(7,7);
  board.colors[point(board,1,0)] = P_BLACK;
  board.colors[point(board,5,2)] = P_BLACK;
  board.colors[point(board,0,0)] = P_WHITE;
  board.colors[point(board,3,4)] = P_WHITE;
  board.regenChainsFromColors();
  return board;
}

void localHash() {
  for(int window : {5,9}) for(int size : {3,7,11}) {
    Rand rand("placement local pattern geometry");
    LocalPatternHasher hasher; hasher.init(window,window,rand);
    Board board(size,size);
    for(int y = 0; y < size; y++) for(int x = 0; x < size; x++)
      board.colors[point(board,x,y)] = (Color)((x*7+y*11+x*y)%3);
    board.regenChainsFromColors();
    for(int symmetry = 0; symmetry < 8; symmetry++) for(bool flip : {false,true}) {
      const Board other = transformed(board,symmetry,flip);
      for(int y = 0; y < size; y++) for(int x = 0; x < size; x++) {
        Loc loc = point(board,x,y), symLoc = SymmetryHelpers::getSymLoc(loc,board,symmetry);
        testAssert(hasher.getPlacementHash(board,loc,P_BLACK,G::FREESTYLE,symmetry,flip) ==
                   hasher.getPlacementHash(other,symLoc,flip ? P_WHITE : P_BLACK,G::FREESTYLE));
      }
    }
  }
  // The remote end of a Go chain changes its atari state, but no colors
  // in this 5x5 window. Placement patterns must depend only on those colors.
  Board board(11,11);
  for(int x = 0; x < 10; x++) {
    board.colors[point(board,x,4)] = board.colors[point(board,x,6)] = P_BLACK;
    board.colors[point(board,x,5)] = P_WHITE;
  }
  board.regenChainsFromColors();
  testAssert(board.getNumLiberties(point(board,1,5)) == 1);
  Board remote = board; remote.colors[point(remote,9,4)] = C_EMPTY; remote.regenChainsFromColors();
  testAssert(remote.getNumLiberties(point(remote,1,5)) == 2);
  Rand rand("placement atari exclusion");
  LocalPatternHasher hasher; hasher.init(5,5,rand);
  Loc loc = point(board,1,3);
  testAssert(hasher.getHash(board,loc,P_BLACK) != hasher.getHash(remote,loc,P_BLACK));
  for(G game : games)
    testAssert(hasher.getPlacementHash(board,loc,P_BLACK,game) == hasher.getPlacementHash(remote,loc,P_BLACK,game));
}

void symmetryAndIdentity() {
  Board board = asymmetricBoard();
  const Loc loc = point(board,2,3), previous = point(board,3,4);
  PatternBonusTable table;
  SubtreeValueBiasTable bias(8);
  std::set<Hash128> identities;
  std::set<std::shared_ptr<SubtreeValueBiasEntry>> biasEntries;
  for(G game : games) {
    BoardHistory initial(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    testAssert(identities.insert(table.getHash(P_BLACK,loc,initial)).second);
    testAssert(biasEntries.insert(bias.get(P_BLACK,previous,loc,initial,0)).second);
    auto hist = initial; Board after = board;
    testAssert(hist.makeBoardMoveTolerant(after,loc,P_BLACK));
    PatternBonusTable repeated;
    repeated.addBonusForGameMoves(hist,.25,P_BLACK);
    int count = 0;
    for(bool flip : {false,true}) for(int symmetry = 0; symmetry < 8; symmetry++) {
      Board other = transformed(board,symmetry,flip);
      const Player pla = flip ? P_WHITE : P_BLACK;
      BoardHistory otherHist(other,pla,Rules::forGame(game),0,BoardHistoryModes());
      const Loc otherLoc = SymmetryHelpers::getSymLoc(loc,board,symmetry);
      const auto bonus = repeated.get(repeated.getHash(pla,otherLoc,otherHist)).utilityBonus;
      const std::set<int> hexNoFlip{0,3}, hexFlip{4,7};
      const bool valid = game == G::HEX ? (flip ? hexFlip : hexNoFlip).count(symmetry) : game != G::RENJU || !flip;
      testAssert(close(bonus,valid ? .25 : 0));
      count += valid;
      for(G otherGame : games) if(otherGame != game) {
        BoardHistory unrelated(other,pla,Rules::forGame(otherGame),0,BoardHistoryModes());
        testAssert(repeated.get(repeated.getHash(pla,otherLoc,unrelated)).utilityBonus == 0);
      }
    }
    testAssert(count == (game == G::HEX ? 4 : game == G::RENJU ? 8 : 16));
    testAssert(table.getHash(P_BLACK,Board::PASS_LOC,initial) == Hash128());
    testAssert(table.getHash(P_BLACK,Board::NULL_LOC,initial) == Hash128());
    testAssert(table.getHash(P_BLACK,loc,hist,1) == table.getHash(P_BLACK,loc,initial));
    testAssert(bias.get(P_BLACK,previous,loc,hist,1) == bias.get(P_BLACK,previous,loc,initial,0));
  }
  BoardHistory go(board,P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes());
  testAssert(identities.insert(table.getHash(P_BLACK,loc,go)).second);
  testAssert(table.getHash(P_BLACK,loc,go) == table.getHash(P_BLACK,loc,board));
  testAssert(bias.get(P_BLACK,previous,loc,go,0) == bias.get(P_BLACK,previous,loc,board));

  // Independent rule check for the combined Hex transforms: top/bottom for
  // black maps to left/right for white only when transpose accompanies color swap.
  Board won(7,7);
  for(int y = 0; y < 7; y++) won.colors[point(won,2,y)] = P_BLACK;
  won.regenChainsFromColors();
  for(int symmetry : {0,3,4,7}) {
    bool flip = symmetry >= 4;
    Board other = transformed(won,symmetry,flip);
    testAssert(Placement::resultFromSetup(other,G::HEX).winner == (flip ? P_WHITE : P_BLACK));
  }
  testAssert(!Placement::resultFromSetup(transformed(won,4,false),G::HEX).finished);
  testAssert(!Placement::resultFromSetup(transformed(won,0,true),G::HEX).finished);
}

void koAndSuicide() {
  Board board(7,7);
  for(auto xy : {std::pair<int,int>{2,3},{4,3},{3,2},{3,4}}) board.colors[point(board,xy.first,xy.second)] = P_WHITE;
  for(auto xy : {std::pair<int,int>{2,4},{4,4},{3,5}}) board.colors[point(board,xy.first,xy.second)] = P_BLACK;
  board.regenChainsFromColors();
  const Loc loc = point(board,3,3);
  testAssert(board.wouldBeKoCapture(loc,P_BLACK));
  PatternBonusTable table;
  testAssert(table.getHash(P_BLACK,loc,board) == Hash128()); // Source Go exception remains.
  for(G game : games) {
    BoardHistory hist(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    testAssert(hist.isLegal(board,loc,P_BLACK));
    auto hash = table.getHash(P_BLACK,loc,hist);
    testAssert(hash != Hash128());
    std::set<Hash128> seen;
    table.addBonus(P_BLACK,loc,hist,0,.125,0,false,seen);
    testAssert(close(table.get(hash).utilityBonus,.125));
    SubtreeValueBiasTable bias(4);
    auto entry = bias.get(P_BLACK,point(board,3,2),loc,hist,0);
    Board withKo = board; withKo.setSimpleKoLoc(point(withKo,0,0));
    BoardHistory other(withKo,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    testAssert(entry == bias.get(P_BLACK,point(board,3,2),loc,other,0));
    Board after = board; testAssert(hist.makeBoardMoveTolerant(after,loc,P_BLACK));
    testAssert(after.colors[point(after,3,4)] == P_WHITE); // Placement did not capture the Go ko stone.
  }
}

void files(const std::string& output, Logger& logger) {
  MakeDir::make(output); MakeDir::make(output+"/sgfs"); MakeDir::make(output+"/poses");
  Board board = asymmetricBoard(); const Loc loc = point(board,2,3);
  for(G game : games) {
    BoardHistory hist(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    Board after = board; testAssert(hist.makeBoardMoveTolerant(after,loc,P_BLACK));
    std::ofstream out(output+"/sgfs/"+Rules::gameName(game)+".sgf");
    WriteSgf::writeSgf(out,"first","second",hist,{});
    Board enclosed(7,7);
    for(auto xy : {std::pair<int,int>{2,3},{4,3},{3,2},{3,4}}) enclosed.colors[point(enclosed,xy.first,xy.second)] = P_WHITE;
    enclosed.regenChainsFromColors();
    testAssert(!enclosed.isLegal(point(enclosed,3,3),P_BLACK,true)); // Illegal single-stone Go suicide.
    BoardHistory position(enclosed,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    testAssert(position.isLegal(enclosed,point(enclosed,3,3),P_BLACK));
    Sgf::PositionSample sample;
    Sgf::PositionSample::writePosOfHist(sample,position,P_BLACK);
    sample.hintLoc = point(enclosed,3,3); sample.initialTurnNumber = 7;
    std::ofstream poses(output+"/poses/"+Rules::gameName(game)+".hintposes.txt");
    poses << Sgf::PositionSample::toJsonLine(sample) << '\n';
    // Actual game boundaries, including positions that Go's raw Board accepts.
    sample.hintLoc = Board::PASS_LOC; poses << Sgf::PositionSample::toJsonLine(sample) << '\n';
    sample.hintLoc = point(enclosed,2,3); poses << Sgf::PositionSample::toJsonLine(sample) << '\n';
    if(game == G::RENJU) {
      Board forbidden = transformed(enclosed,0,true);
      BoardHistory forbiddenHist(forbidden,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
      testAssert(forbiddenHist.isLegal(forbidden,point(forbidden,3,3),P_BLACK));
      sample.board = forbidden; sample.hintLoc = point(forbidden,3,3);
      poses << Sgf::PositionSample::toJsonLine(sample) << '\n';
      forbiddenHist.makeBoardMoveAssumeLegal(forbidden,sample.hintLoc,P_BLACK,nullptr);
      testAssert(forbiddenHist.isGameFinished && forbiddenHist.winner == P_WHITE);
    }
    Board won(7,7);
    for(int i = 0; i < (game == G::HEX ? 7 : 5); i++)
      won.colors[game == G::HEX ? point(won,1,i) : point(won,i,0)] = P_BLACK;
    won.regenChainsFromColors(); sample.board = won; sample.hintLoc = point(won,6,6);
    poses << Sgf::PositionSample::toJsonLine(sample) << '\n';
    if(game != G::HEX) {
      for(int i = 0; i < 5; i++) sample.board.colors[point(sample.board,i,2)] = P_WHITE;
      sample.board.regenChainsFromColors();
      poses << Sgf::PositionSample::toJsonLine(sample) << '\n'; // conflicting winners
    }
  }
  PatternBonusTable sgfs;
  sgfs.avoidRepeatedSgfMoves({output+"/sgfs"},.2,1,0,10,{"first"},logger,"mixed-games");
  for(G game : games) {
    BoardHistory hist(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    testAssert(close(sgfs.get(sgfs.getHash(P_BLACK,loc,hist)).utilityBonus,.2));
    Board flipped = transformed(board,4,true);
    BoardHistory other(flipped,P_WHITE,Rules::forGame(game),0,BoardHistoryModes());
    testAssert(close(sgfs.get(sgfs.getHash(P_WHITE,SymmetryHelpers::getSymLoc(loc,board,4),other)).utilityBonus,
                     game == G::RENJU ? 0 : -.2));
  }
  PatternBonusTable poses;
  poses.avoidRepeatedPosMovesAndDeleteExcessFiles({output+"/poses"},.3,1,7,7,20,logger,"placement-poses");
  size_t patternCount = 0; for(const auto& shard : poses.entries) patternCount += shard.size();
  testAssert(patternCount == 8); // 2 freestyle, 2 standard, 2 Renju (including forbidden), 2 Hex
  for(G game : games) {
    Board enclosed(7,7);
    for(auto xy : {std::pair<int,int>{2,3},{4,3},{3,2},{3,4}}) enclosed.colors[point(enclosed,xy.first,xy.second)] = P_WHITE;
    enclosed.regenChainsFromColors();
    BoardHistory hist(enclosed,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    testAssert(close(poses.get(poses.getHash(P_BLACK,point(enclosed,3,3),hist)).utilityBonus,.3));
    if(game == G::RENJU) {
      Board forbidden = transformed(enclosed,0,true);
      BoardHistory forbiddenHist(forbidden,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
      testAssert(close(poses.get(poses.getHash(P_BLACK,point(forbidden,3,3),forbiddenHist)).utilityBonus,.3));
    }
  }
}

void searchUpdates(Logger& logger) {
  ConfigParser cfg(std::map<std::string,std::string>{});
  NNEvaluator evaluator("random","","",&logger,8,7,7,false,false,10,4,true,"",enabled_t::False,
                       1,{0},"placement-patterns",false,0,cfg);
  evaluator.spawnServerThreads();
  for(G game : games) {
    Board board = asymmetricBoard();
    BoardHistory hist(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    auto table = std::make_unique<PatternBonusTable>();
    std::set<Hash128> hashes;
    for(int y = 0; y < 7; y++) for(int x = 0; x < 7; x++) {
      Loc loc = point(board,x,y);
      if(hist.isLegal(board,loc,P_BLACK)) table->addBonus(P_BLACK,loc,hist,0,.125,0,false,hashes);
    }
    SearchParams params; params.maxVisits = params.maxPlayouts = 2; params.numThreads = 1;
    Search search(params,&evaluator,&logger,"placement-pattern-leaf");
    search.setPosition(P_BLACK,board,hist); search.setExternalPatternBonusTable(std::move(table));
    search.runWholeSearch(P_BLACK);
    auto children = search.getRootNode()->getChildren();
    testAssert(children.iterateAndCountChildren() == 1);
    const auto* child = children[0].getIfAllocated(); const auto* nn = child->getNNOutput();
    testAssert(nn != nullptr && close(child->stats.utilityAvg.load(),double(nn->whiteWinProb)-nn->whiteLossProb+.125));
  }
  for(G game : games) for(bool gumbel : {false,true}) for(int threads : {1,3}) {
    SearchParams params;
    params.maxVisits = params.maxPlayouts = 64; params.numThreads = threads;
    params.useGumbel = gumbel; params.useGumbelNonroot = gumbel;
    params.useGraphSearch = !gumbel; params.rootSymmetryPruning = !gumbel;
    params.subtreeValueBiasFactor = .4; params.avoidRepeatedPatternUtility = .03;
    Board board = asymmetricBoard();
    BoardHistory hist(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    for(Loc loc : {point(board,2,3),point(board,6,6)})
      testAssert(hist.makeBoardMoveTolerant(board,loc,hist.presumedNextMovePla));
    Search search(params,&evaluator,&logger,"placement-pattern-search");
    search.setPosition(P_BLACK,board,hist); search.runWholeSearch(P_BLACK);
    auto children = search.getRootNode()->getChildren();
    testAssert(children.iterateAndCountChildren() > 0);
    for(int i = 0; i < children.iterateAndCountChildren(); i++) {
      const auto* child = children[i].getIfAllocated();
      testAssert(child->subtreeValueBiasTableEntry != nullptr);
      testAssert(std::isfinite(child->subtreeValueBiasTableEntry->deltaUtilitySum));
      testAssert(hist.isLegal(board,children[i].getMoveLoc(),P_BLACK));
    }
    const Loc chosen = search.getChosenMoveLoc();
    testAssert(hist.isLegal(board,chosen,P_BLACK) && search.makeMove(chosen,P_BLACK));
    if(!search.getRootHist().isGameFinished) {
      search.runWholeSearch(P_WHITE);
      testAssert(search.isLegalStrict(search.getChosenMoveLoc(),P_WHITE));
    }
  }
  evaluator.killServerThreads();
}
}

void Tests::runPlacementPatternTests(const std::string& output) {
  Board::initHash();
  MakeDir::make(output);
  Logger logger(nullptr,false,false,false);
  logger.addFile(output+"/native.log");
  localHash(); symmetryAndIdentity(); koAndSuicide(); files(output,logger); searchUpdates(logger);
  std::cout << "Placement pattern identity, valid symmetries, files and source searches passed" << std::endl;
}
