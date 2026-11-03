#include "tests.h"
#include "testsearchcommon.h"
#include "../game/placement.h"
#include "../dataio/trainingwrite.h"
#include "../program/play.h"
#include "../game/graphhash.h"
#include "../core/test.h"
#include "../search/searchnode.h"

#include <memory>

namespace {
using G = Rules::Game;
using F = Placement::Forbidden;
Loc point(const Board& board, int x, int y) { return Location::getLoc(x,y,board.x_size); }

Board setup(int size, Player pla, std::initializer_list<std::pair<int,int>> points) {
  Board board(size,size);
  for(auto p : points) board.colors[point(board,p.first,p.second)] = pla;
  board.regenChainsFromColors();
  return board;
}

void checkRenju(int size, int x, int y, std::initializer_list<std::pair<int,int>> offsets, F expected) {
  Board board(size,size);
  for(auto p : offsets) board.colors[point(board,x+p.first,y+p.second)] = P_BLACK;
  board.regenChainsFromColors();
  Loc loc = point(board,x,y);
  testAssert(Placement::renjuForbidden(board,loc) == expected);
  BoardHistory hist(board,P_BLACK,Rules::forGame(G::RENJU),0,BoardHistoryModes(false,false));
  testAssert(hist.isLegal(board,loc,P_BLACK));
  hist.makeBoardMoveAssumeLegal(board,loc,P_BLACK,NULL);
  if(expected != F::NONE) testAssert(hist.isGameFinished && hist.winner == P_WHITE && !hist.isNoResult);
  board.checkConsistency(true);
}

void rulesAndTransitions() {
  for(G game : {G::FREESTYLE,G::STANDARD,G::RENJU,G::HEX}) {
    Rules rules = Rules::forGame(game);
    testAssert(Rules::parseRules(rules.toString()) == rules);
    testAssert(Rules::parseRules(rules.toJsonString()) == rules);
    testAssert(Rules::parseRulesWithoutKomi(rules.toString(),0) == rules);
    Rules ignored;
    testAssert(!Rules::tryParseRulesWithoutKomi(rules.toString(),ignored,7.5f));
    testAssert(rules != Rules::getTrompTaylorish());
    Board board = setup(7,P_WHITE,{{2,3},{3,2},{4,3},{3,4}});
    BoardHistory hist(board,P_BLACK,rules,0,BoardHistoryModes(false,false));
    Loc loc = point(board,3,3);
    testAssert(!hist.isLegal(board,Board::PASS_LOC,P_BLACK));
    testAssert(!hist.isLegal(board,loc,P_WHITE));
    testAssert(hist.isLegal(board,loc,P_BLACK));
    hist.makeBoardMoveAssumeLegal(board,loc,P_BLACK,NULL);
    testAssert(board.colors[loc] == P_BLACK && board.getNumLiberties(loc) == 0);
    testAssert(board.numStonesOnBoard() == 5 && !hist.isGameFinished);
    testAssert(board.numBlackCaptures == 0 && board.numWhiteCaptures == 0);
    testAssert(hist.getRecentBoard(0).pos_hash == board.pos_hash);
    testAssert(hist.getRecentBoard(1).colors[loc] == C_EMPTY);
    board.checkConsistency(true);
    auto copied = hist;
    testAssert(copied.rules == rules && copied.moveHistory.size() == 1);
    testAssert(copied.moveHistory[0].loc == loc && copied.moveHistory[0].pla == P_BLACK);
    testAssert(copied.copyToInitial().getRecentBoard(0).colors[loc] == C_EMPTY);
  }
  for(G game : {G::FREESTYLE,G::STANDARD,G::RENJU}) {
    for(Player pla : {P_BLACK,P_WHITE}) {
      Board board = setup(9,pla,{{0,4},{1,4},{2,4},{3,4}});
      BoardHistory hist(board,pla,Rules::forGame(game),0,BoardHistoryModes(false,false));
      hist.makeBoardMoveAssumeLegal(board,point(board,4,4),pla,NULL);
      testAssert(hist.isGameFinished && hist.winner == pla);
      testAssert(!hist.isLegal(board,point(board,8,8),getOpp(pla)));
      board = setup(9,pla,{{0,4},{1,4},{2,4},{4,4},{5,4}});
      hist.clear(board,pla,Rules::forGame(game),0);
      hist.makeBoardMoveAssumeLegal(board,point(board,3,4),pla,NULL);
      if(game == G::STANDARD) testAssert(!hist.isGameFinished);
      else testAssert(hist.isGameFinished && hist.winner == (game == G::RENJU && pla == P_BLACK ? P_WHITE : pla));
    }
    // A filled board with no five is an ordinary draw, never no-result.
    Board board(3,3);
    BoardHistory hist(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes(false,false));
    for(int y = 0; y < 3; y++)
      for(int x = 0; x < 3; x++)
        hist.makeBoardMoveAssumeLegal(board,point(board,x,y),hist.presumedNextMovePla,NULL);
    testAssert(hist.isGameFinished && hist.winner == C_EMPTY && !hist.isNoResult);
    board.checkConsistency(true);
  }
  for(int size : {11,15,19}) {
    int c = size/2;
    checkRenju(size,c,c,{{-2,0},{-1,0},{1,0},{2,0},{0,-3},{0,-2},{0,-1},{0,1},{0,2}},F::NONE);
    checkRenju(size,c,c,{{-3,0},{-2,0},{-1,0},{1,0},{2,0}},F::OVERLINE);
    checkRenju(size,c,c,{{-1,0},{1,0},{2,0},{0,-1},{0,1},{0,2}},F::DOUBLE_FOUR);
    checkRenju(size,c,c,{{-3,0},{-1,0},{1,0},{3,0}},F::DOUBLE_FOUR);
    checkRenju(size,c,c,{{-1,0},{1,0},{0,-1},{0,1}},F::DOUBLE_THREE);
    checkRenju(size,1,1,{{-1,0},{1,0},{0,-1},{0,1}},F::NONE);
    // Apparent threes whose extensions are overlines are not real threes.
    checkRenju(size,c,c,{{-1,0},{1,0},{0,-1},{0,1},{-2,-3},{-2,-2},{-2,-1},{-2,1},{-2,2},
                       {2,-3},{2,-2},{2,-1},{2,1},{2,2}},F::NONE);
  }
}

void hexAndIdentity() {
  Board slash = setup(3,P_BLACK,{{2,0},{1,1},{0,2}});
  Board backslash = setup(3,P_BLACK,{{0,0},{1,1},{2,2}});
  testAssert(Placement::hexConnected(slash,P_BLACK));
  testAssert(!Placement::hexConnected(backslash,P_BLACK));
  for(int bits = 0; bits < 512; bits++) {
    Board board(3,3);
    for(int i = 0; i < 9; i++) board.colors[point(board,i%3,i/3)] = (bits & (1<<i)) ? P_BLACK : P_WHITE;
    // Hex's no-draw theorem: exactly one color connects on every full board.
    testAssert(Placement::hexConnected(board,P_BLACK) != Placement::hexConnected(board,P_WHITE));
  }
  Board board(7,7);
  BoardHistory hex(board,P_BLACK,Rules::forGame(G::HEX),0,BoardHistoryModes(false,false));
  bool duplicate[Board::MAX_ARR_SIZE];
  std::vector<int> symmetries;
  SymmetryHelpers::markDuplicateMoveLocs(board,hex,NULL,{},duplicate,symmetries);
  testAssert(symmetries == std::vector<int>({0,3}));
  BoardHistory freestyle(board,P_BLACK,Rules::forGame(G::FREESTYLE),0,BoardHistoryModes(false,false));
  SymmetryHelpers::markDuplicateMoveLocs(board,freestyle,NULL,{},duplicate,symmetries);
  testAssert(symmetries.size() == 8);
  testAssert(GraphHash::getStateHash(hex,P_BLACK,0.5) != GraphHash::getStateHash(freestyle,P_BLACK,0.5));

  Board a(7,7), b(7,7);
  BoardHistory ha(a,P_BLACK,Rules::forGame(G::STANDARD),0,BoardHistoryModes(false,false)), hb = ha;
  for(auto xy : {std::pair<int,int>{0,0},{4,4},{1,0},{5,4}})
    ha.makeBoardMoveAssumeLegal(a,point(a,xy.first,xy.second),ha.presumedNextMovePla,NULL);
  for(auto xy : {std::pair<int,int>{1,0},{5,4},{0,0},{4,4}})
    hb.makeBoardMoveAssumeLegal(b,point(b,xy.first,xy.second),hb.presumedNextMovePla,NULL);
  testAssert(GraphHash::getGraphHashFromScratch(ha,P_BLACK,5,0.5) == GraphHash::getGraphHashFromScratch(hb,P_BLACK,5,0.5));
}

void placementFeatures() {
  const int width = 11, height = 10;
  // The center is a black double-three, legal to play but losing in Renju.
  Board board = setup(9,P_BLACK,{{3,4},{5,4},{4,3},{4,5}});
  board.colors[point(board,1,2)] = P_WHITE;
  board.regenChainsFromColors();
  for(G game : {G::FREESTYLE,G::STANDARD,G::RENJU,G::HEX}) for(Player pla : {P_BLACK,P_WHITE}) {
    BoardHistory hist(board,pla,Rules::forGame(game),0,BoardHistoryModes(false,false));
    MiscNNInputParams params;
    params.playoutDoublingAdvantage = -2;
    std::vector<float> nchw(5*width*height,99), nhwc(nchw.size(),99);
    float global[8], otherGlobal[8];
    NNInputs::fillRowV8(board,hist,pla,params,width,height,false,nchw.data(),global);
    NNInputs::fillRowV8(board,hist,pla,params,width,height,true,nhwc.data(),otherGlobal);
    for(int c = 0; c < 8; c++) testAssert(global[c] == otherGlobal[c]);
    testAssert(global[0] == (game == G::STANDARD));
    testAssert(global[1] == (game == G::RENJU));
    testAssert(global[2] == (game == G::RENJU ? (pla == P_BLACK ? -1 : 1) : 0));
    testAssert(global[3] == (game == G::RENJU));
    testAssert(global[4] == 1 && global[5] == -1);
    testAssert(global[6] == (game == G::HEX));
    testAssert(global[7] == (game == G::HEX && pla == P_WHITE));
    for(int y = 0; y < height; y++) for(int x = 0; x < width; x++) {
      const int pos = y*width+x;
      for(int c = 0; c < 5; c++) testAssert(nchw[c*width*height+pos] == nhwc[pos*5+c]);
      const bool inside = x < 9 && y < 9;
      testAssert(nchw[pos] == inside);
      if(!inside) {
        for(int c = 1; c < 5; c++) testAssert(nchw[c*width*height+pos] == 0);
      }
      else {
        Color color = board.colors[point(board,x,y)];
        testAssert(nchw[width*height+pos] == (color == pla));
        testAssert(nchw[2*width*height+pos] == (color == getOpp(pla)));
        if(game != G::RENJU || color != C_EMPTY) {
          testAssert(nchw[3*width*height+pos] == 0 && nchw[4*width*height+pos] == 0);
        }
      }
    }
    int forbiddenChannel = pla == P_BLACK ? 3 : 4;
    testAssert(nchw[forbiddenChannel*width*height+4*width+4] == (game == G::RENJU));
    testAssert(nchw[(7-forbiddenChannel)*width*height+4*width+4] == 0);
    testAssert(hist.isLegal(board,point(board,4,4),pla));
  }
  BoardHistory go(board,P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes(false,false));
  float spatial[5*width*height], global[8];
  bool rejected = false;
  try { NNInputs::fillRowV8(board,go,P_BLACK,MiscNNInputParams(),width,height,false,spatial,global); }
  catch(const StringError&) { rejected = true; }
  testAssert(rejected);
}

void placementTrainingRows() {
  TrainingWriteBuffers buffers(8,2,5,8,7,7,false);
  testAssert(buffers.policyTargetsNCMove.shape[2] == 49 && buffers.qValueTargetsNCMove.shape[2] == 49);
  Board board(5,5);
  board.setStone(point(board,1,1),P_BLACK);
  std::vector<PolicyTargetMove> policy{PolicyTargetMove(point(board,4,4),10)};
  std::vector<QValueTargets> q(2);
  q[0].targets.emplace_back(point(board,4,4),0.25f,9);
  std::vector<ValueTargets> values(2);
  values[0].win = 0.3f; values[0].draw = 0.2f; values[0].loss = 0.5f;
  values[1].win = 1.0f;
  NNRawStats raw{};
  Rand rand("placement-writer");
  for(Player pla : {P_BLACK,P_WHITE}) {
    BoardHistory hist(board,pla,Rules::forGame(G::HEX),0,BoardHistoryModes(false,false));
    buffers.addRow(board,hist,pla,hist,hist,0,1.0f,100,&policy,NULL,0.5,1.0,1.0,
                   values,q,0,1.0f,1.0f,raw,NULL,NULL,false,0,0.5,
                   C_EMPTY,0.0,Hash128(),{},false,0,FinishedGameData::MODE_NORMAL,NULL,rand,ReanalysisData());
    const int row = buffers.curRows-1;
    const float* global = buffers.globalInputNC.data+row*8;
    testAssert(global[6] == 1 && global[7] == (pla == P_WHITE));
    const uint8_t* packed = buffers.binaryInputNCHWPacked.data+row*5*7;
    for(int channel = 0; channel < 5; channel++) for(int pos = 0; pos < 49; pos++) {
      bool actual = packed[channel*7+pos/8] & (1 << (7-pos%8));
      bool expected = channel == 0 ? pos%7 < 5 && pos/7 < 5 :
        channel == (pla == P_BLACK ? 1 : 2) && pos == 8;
      testAssert(actual == expected);
    }
    const int16_t* p = buffers.policyTargetsNCMove.data+row*2*49;
    const int16_t* qt = buffers.qValueTargetsNCMove.data+row*2*49;
    for(int pos = 0; pos < 49; pos++) {
      testAssert(p[pos] == (pos == 32 ? 10 : 0));
      testAssert(qt[pos] == (pos == 32 ? (pla == P_WHITE ? 8000 : -8000) : 0));
      testAssert(qt[49+pos] == (pos == 32 ? 9 : 0));
    }
    const float* target = buffers.globalTargetsNC.data+row*80;
    testAssert(target[0] == (pla == P_WHITE) && target[1] == (pla == P_BLACK));
    testAssert(target[2] == 0 && target[70] == 0);
  }
}

void placementTruncatedFutureRows() {
  TrainingWriteBuffers buffers(8,2,5,8,7,7,false);
  Board board(5,5);
  BoardHistory hist(board,P_BLACK,Rules::forGame(G::HEX),0,BoardHistoryModes(false,false));
  std::vector<ValueTargets> values(10);
  std::vector<QValueTargets> q(10);
  std::vector<Board> future(10,board);
  NNRawStats raw{};
  Rand rand("placement-truncated-future");
  for(int index : {0,2}) {
    buffers.addRow(board,hist,P_BLACK,hist,hist,index,1.0f,100,NULL,NULL,0.5,1.0,1.0,
                   values,q,index,0.0f,1.0f,raw,&board,&future,false,0,0.5,
                   C_EMPTY,0.0,Hash128(),{},true,0,FinishedGameData::MODE_NORMAL,NULL,rand,ReanalysisData());
    const float* target = buffers.globalTargetsNC.data+(buffers.curRows-1)*80;
    testAssert(target[75] == (index == 2) && target[76] == 1);
    testAssert(target[33] == 1 && target[27] == 0 && target[35] == 1);
  }
}

void placementImports() {
  auto rejects = [](const std::function<void()>& f) {
    bool rejected = false;
    try { f(); } catch(const StringError&) { rejected = true; }
    testAssert(rejected);
  };
  const std::string setupNodes = "SZ[7]AB[dd]AW[cd][dc][ed][de]PL[W];W[aa];B[bb])";
  for(G game : {G::FREESTYLE,G::STANDARD,G::RENJU,G::HEX}) {
    std::string text = "(;GM[" + std::string(game == G::HEX ? "11" : "4") + "]RU[" + Rules::gameName(game) + "]" + setupNodes;
    auto sgf = Sgf::parse(text);
    auto compact = CompactSgf::parse(text);
    Rules rules = sgf->getRulesOrFail();
    testAssert(rules == Rules::forGame(game) && compact->getRulesOrFail() == rules);
    Board board;
    BoardHistory hist;
    Player pla;
    compact->setupBoardAndHistTolerant(rules,board,pla,hist,2,false,BoardHistoryModes());
    testAssert(board.numStonesOnBoard() == 7 && board.colors[point(board,3,3)] == P_BLACK);
    testAssert(board.getNumLiberties(point(board,3,3)) == 0 && !hist.isGameFinished);
    board.checkConsistency(true);
    std::set<Hash128> hashes;
    std::vector<Sgf::PositionSample> positions;
    sgf->loadAllUniquePositions(hashes,false,false,true,false,NULL,positions);
    testAssert(positions.size() == 3);
    for(const auto& pos : positions) {
      testAssert(pos.game == game);
      auto decoded = Sgf::PositionSample::ofJsonLine(Sgf::PositionSample::toJsonLine(pos));
      testAssert(pos.isEqualForTesting(decoded,true,true));
      testAssert(decoded.tryGetCurrentBoardHistory(rules,pla,hist,BoardHistoryModes()));
      testAssert(hist.getCurrentTurnNumber() == pos.getCurrentTurnNumber());
      testAssert(!decoded.tryGetCurrentBoardHistory(Rules::getTrompTaylorish(),pla,hist,BoardHistoryModes()));
      if(game == G::RENJU || game == G::HEX) {
        testAssert(pos.nextPla == P_WHITE && pos.board.colors[point(pos.board,3,3)] == P_BLACK);
        rejects([&] { pos.getColorFlipped(); });
      }
      else {
        auto flipped = pos.getColorFlipped();
        testAssert(flipped.board.numStonesOnBoard() == pos.board.numStonesOnBoard());
        testAssert(flipped.getColorFlipped().isEqualForTesting(pos,true,true));
      }
    }
    // Tolerant import must not remove zero-liberty stones either.
    int count = 0;
    sgf->iterAllPositions(false,false,NULL,[&](Sgf::PositionSample& pos,const BoardHistory& h,const std::string&) {
      testAssert(pos.game == game && h.getRecentBoard(0).colors[point(board,3,3)] == P_BLACK);
      count++;
    },true);
    testAssert(count == 3);
    compact->setupInitialBoardAndHist(rules,board,pla,hist,BoardHistoryModes());
    std::ostringstream out;
    WriteSgf::writeSgf(out,"B","W",hist,{});
    testAssert(out.str().find("KM[") == std::string::npos && out.str().find("HA[") == std::string::npos);
    auto roundtrip = CompactSgf::parse(out.str());
    roundtrip->setupInitialBoardAndHist(roundtrip->getRulesOrFail(),board,pla,hist,BoardHistoryModes());
    testAssert(board.numStonesOnBoard() == 5 && pla == P_WHITE);

    ConfigParser cfg(std::map<std::string,std::string>{{"game",Rules::gameName(game)},{"bSizes","7"},{"bSizeRelProbs","1"}});
    Logger logger(nullptr,false,false,false);
    GameInitializer init(cfg,logger,"placement-import");
    PlaySettings settings;
    OtherGameProperties props;
    int extraBlack;
    auto sample = positions.back();
    sample.hintLoc = point(sample.board,6,6);
    sample.trainingWeight = 0.375;
    init.createGame(board,pla,hist,extraBlack,NULL,settings,props,&sample,BoardHistoryModes());
    testAssert(hist.rules.game == game && board.numStonesOnBoard() == 7);
    testAssert(props.isHintPos && props.trainingWeight == 0.375 && props.hintLoc == sample.hintLoc);
    sample.game = G::GO;
    rejects([&] { init.createGame(board,pla,hist,extraBlack,NULL,settings,props,&sample,BoardHistoryModes()); });
  }
  // Same stones under distinct rules are distinct sampling positions.
  std::set<Hash128> hashes;
  std::vector<Sgf::PositionSample> positions;
  for(G game : {G::FREESTYLE,G::STANDARD,G::RENJU,G::HEX}) {
    auto sgf = Sgf::parse("(;RU["+Rules::gameName(game)+"]SZ[7])");
    sgf->loadAllUniquePositions(hashes,false,false,false,false,NULL,positions);
  }
  testAssert(positions.size() == 4);
  for(const std::string text : {"(;GM[4]SZ[7])", "(;GM[1]RU[renju]SZ[7])", "(;GM[11]RU[standard]SZ[7])",
                                "(;GM[11]RU[unknown]SZ[7])", "(;GM[2]SZ[7])", "(;GM[4]RU[renju]KM[7.5]SZ[7])"})
    rejects([&] { CompactSgf::parse(text)->getRulesOrFail(); });
  testAssert(CompactSgf::parse("(;GM[11]SZ[7])")->getRulesOrFail().game == G::HEX);
  Board board;
  BoardHistory hist;
  Player pla;
  rejects([&] { CompactSgf::parse("(;GM[4]RU[standard]SZ[7])")->setupInitialBoardAndHist(
    Rules::getTrompTaylorish(),board,pla,hist,BoardHistoryModes()); });
  rejects([&] { CompactSgf::parse("(;GM[4]RU[standard]SZ[7]PL[W];B[aa])")->setupInitialBoardAndHist(
    Rules::forGame(G::STANDARD),board,pla,hist,BoardHistoryModes()); });
  auto blackSetup = CompactSgf::parse("(;RU[renju]SZ[7]AB[aa][bb])");
  blackSetup->setupInitialBoardAndHist(Rules::forGame(G::RENJU),board,pla,hist,BoardHistoryModes());
  testAssert(pla == P_BLACK); // Black placements are not Go handicap stones.
  // A recorded forbidden transition has an unambiguous loss even without a line of five.
  auto forbidden = Sgf::parse("(;RU[renju]SZ[7]AB[cd][ed][dc][de];B[dd])");
  int forbiddenPositions = 0;
  forbidden->iterAllPositions(false,true,NULL,[&](Sgf::PositionSample& pos,const BoardHistory& h,const std::string&) {
    forbiddenPositions++;
    if(h.moveHistory.size() == 1) {
      testAssert(h.isGameFinished && h.winner == P_WHITE);
      testAssert(pos.tryGetCurrentBoardHistory(h.rules,pla,hist,BoardHistoryModes()));
      testAssert(hist.isGameFinished && hist.winner == P_WHITE);
    }
  });
  testAssert(forbiddenPositions == 2);
  // Full non-winning setups are draws; simultaneous winners cannot identify a valid game result.
  Board small = Board::parseBoard(3,3,"XOX/OXO/XOX/",'/',true);
  auto result = Placement::resultFromSetup(small,G::STANDARD);
  testAssert(result.finished && result.winner == C_EMPTY);
  Board conflict = Board::parseBoard(5,5,"XXXXX/OOOOO/...../...../...../",'/',true);
  rejects([&] { Placement::resultFromSetup(conflict,G::FREESTYLE); });
  // Terminal setups must not feed an all-illegal policy into selfplay.
  for(G game : {G::FREESTYLE,G::STANDARD,G::RENJU,G::HEX}) {
    const std::string stones = game == G::HEX ? "[aa][ab][ac][ad][ae]" : "[aa][ba][ca][da][ea]";
    auto sgf = Sgf::parse("(;RU["+Rules::gameName(game)+"]SZ[5]AB"+stones+")");
    int count = 0;
    sgf->iterAllPositions(false,false,NULL,[&](auto&,const auto&,const auto&) { count++; });
    testAssert(count == 0);
    sgf->iterAllPositions(false,true,NULL,[&](auto&,const BoardHistory& hist,const auto&) {
      testAssert(hist.isGameFinished && hist.winner == P_BLACK);
      count++;
    });
    testAssert(count == 1);
  }
  // Source Go implicit identity and zero-liberty rejection are retained.
  auto go = Sgf::parse("(;SZ[7]KM[7.5]RU[Chinese];B[aa])");
  go->iterAllPositions(false,false,NULL,[&](Sgf::PositionSample& pos,const auto&,const auto&) {
    testAssert(pos.game == G::GO && Sgf::PositionSample::toJsonLine(pos).find("\"game\"") == std::string::npos);
  });
  rejects([&] { Sgf::parse("(;SZ[7]AB[dd]AW[cd][dc][ed][de])")->iterAllPositions(false,false,NULL,[](auto&,const auto&,const auto&) {}); });
}

void sourceSearch() {
  NeuralNet::globalInitialize();
  Logger logger(nullptr,false,false,false);
  auto nn = std::unique_ptr<NNEvaluator>(TestSearchCommon::startNNEval("/dev/null",logger,"placement",7,7,0,true,false,false,true,false));
  SearchParams params;
  params.maxVisits = 128;
  params.chosenMoveTemperature = 0;
  params.chosenMoveTemperatureEarly = 0;
  params.subtreeValueBiasFactor = 0.3;
  params.rootSymmetryPruning = true;
  for(bool graph : {false,true}) for(int threads : {1,3}) for(G game : {G::STANDARD,G::HEX}) {
    params.useGraphSearch = graph;
    params.numThreads = threads;
    Search search(params,nn.get(),&logger,"placement-search");
    Board board = game == G::HEX ? setup(7,P_BLACK,{{0,0},{0,1},{0,2},{0,3},{0,4},{0,5}}) :
                                 setup(7,P_BLACK,{{0,3},{1,3},{2,3},{3,3}});
    Loc winning = game == G::HEX ? point(board,0,6) : point(board,4,3);
    BoardHistory hist(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes(false,false));
    // Only one legal root action remains in this controlled tactical position.
    std::vector<int> avoid(Board::MAX_ARR_SIZE,1);
    avoid[winning] = 0;
    search.setPosition(P_BLACK,board,hist);
    search.setAvoidMoveUntilByLoc(avoid,{});
    search.runWholeSearch(P_BLACK);
    testAssert(search.getChosenMoveLoc() == winning);
    testAssert(search.rootNode->stats.winLossValueAvg.load() < -0.9);
    testAssert(search.rootNode->stats.noResultValueAvg.load() == 0.0);
    testAssert(search.makeMove(winning,P_BLACK));
    testAssert(search.rootHistory.isGameFinished && search.rootHistory.winner == P_BLACK);
    search.runWholeSearch(P_WHITE);
    testAssert(search.getChosenMoveLoc() == Board::NULL_LOC);
    testAssert(search.rootNode->stats.winLossValueAvg.load() == -1.0);
  }
  // A fork candidate can end the game. It must use the rule result without asking for a policy.
  Board endBoard(2,2);
  endBoard.colors[point(endBoard,0,0)] = P_BLACK;
  endBoard.colors[point(endBoard,1,0)] = P_WHITE;
  endBoard.colors[point(endBoard,1,1)] = P_WHITE;
  endBoard.regenChainsFromColors();
  BoardHistory endHist(endBoard,P_BLACK,Rules::forGame(G::HEX),0,BoardHistoryModes(false,false));
  FinishedGameData data;
  data.startBoard = endBoard;
  data.startHist = endHist;
  data.endHist = endHist;
  data.startPla = P_BLACK;
  PlaySettings settings;
  settings.earlyForkGameProb = 1;
  ForkData forks;
  Rand forkRand("terminal-fork");
  Search search(params,nn.get(),&logger,"terminal-fork");
  Play::maybeForkGame(&data,&forks,settings,forkRand,&search);
  testAssert(forks.forks.empty());
  nn.reset();
  NeuralNet::globalCleanup();
}
}

void Tests::runPlacementTests() {
  std::cout << "Running placement game rules, identity, and source PUCT tests" << std::endl;
  rulesAndTransitions();
  hexAndIdentity();
  placementFeatures();
  placementTrainingRows();
  placementTruncatedFutureRows();
  placementImports();
  sourceSearch();
}
