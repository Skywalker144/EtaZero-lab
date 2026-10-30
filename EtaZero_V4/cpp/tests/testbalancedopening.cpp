#include "tests.h"
#include "../program/balancedopening.h"
#include "../core/test.h"
#include <limits>

namespace {
using namespace BalancedOpening;
using G = Rules::Game;
BoardHistory history(const Board& b, G game) {
  return BoardHistory(b,P_BLACK,Rules::forGame(game),0,BoardHistoryModes(false,false));
}
void configuration() {
  ConfigParser empty;
  testAssert(Settings::load(empty,true).probability == 0);
  for(bool self : {false,true}) for(G game : {G::RENJU,G::HEX}) {
    ConfigParser cfg(std::map<std::string,std::string>{{"game",Rules::gameName(game)},{"balanceOpeningProb","1"}});
    auto s = Settings::load(cfg,self);
    testAssert(s.probability == 1);
    testAssert(s.dropPow == (game == G::HEX ? (self ? 6 : 20) : (self ? 4 : 10)));
    testAssert(s.minAcceptRate > 0);
  }
  for(auto values : {
      std::map<std::string,std::string>{{"game","go"},{"balanceOpeningProb","1"}},
      {{"game","xiangqi"},{"balanceOpeningProb","1"}},
      {{"game","hex"},{"minAcceptRate","0"}},
      {{"game","renju"},{"makeOpeningFairRate","1"}},
      {{"game","hex"},{"balanceOpeningProb","nan"}}}) {
    ConfigParser cfg(values);
    bool rejected = false;
    try { Settings::load(cfg,false); } catch(const StringError&) { rejected = true; }
    testAssert(rejected);
  }
}
void disabledAndInterrupted() {
  Board board(7,7); auto hist = history(board,G::RENJU); Player pla = P_BLACK;
  Rand rand("opening no side effects"), untouched("opening no side effects");
  Settings settings;
  Evaluate unavailable = [](const Board&,const BoardHistory&,Player,int)->Value { ASSERT_UNREACHABLE; };
  testAssert(initialize(board,hist,pla,rand,settings,unavailable,{}) == Result::SKIPPED);
  testAssert(rand.nextUInt64() == untouched.nextUInt64());
  settings.probability = 1;
  bool cancel = false;
  Evaluate interrupt = [&](const Board&,const BoardHistory&,Player,int) { cancel = true; return Value{0.5,0.5}; };
  testAssert(initialize(board,hist,pla,rand,settings,interrupt,[&](){return cancel;}) == Result::INTERRUPTED);
  testAssert(board.numStonesOnBoard() == 0 && hist.moveHistory.empty() && pla == P_BLACK);
  Evaluate invalid = [](const Board&,const BoardHistory&,Player,int) { return Value{std::numeric_limits<double>::quiet_NaN(),0}; };
  bool rejected = false;
  try { initialize(board,hist,pla,rand,settings,invalid,{}); } catch(const StringError&) { rejected = true; }
  testAssert(rejected && board.numStonesOnBoard() == 0 && hist.moveHistory.empty());
  Loc loc = Location::getLoc(3,3,7);
  hist.makeBoardMoveAssumeLegal(board,loc,P_BLACK,nullptr); pla = P_WHITE;
  testAssert(initialize(board,hist,pla,rand,settings,unavailable,{}) == Result::SKIPPED);
  testAssert(hist.moveHistory.size() == 1 && board.numStonesOnBoard() == 1);
}
void hexAcceptance() {
  Rand rand("Hex white probability reference");
  Settings settings; settings.probability = 1; settings.dropPow = 2; settings.minAcceptRate = 0.001;
  int left = 0, calls = 0;
  for(int i = 0; i < 2000; i++) {
    Board board(4,4); auto hist = history(board,G::HEX); Player pla = P_BLACK;
    Evaluate evaluate = [&](const Board& b,const BoardHistory& h,Player p,int reference) {
      testAssert(reference == 1 && p == P_WHITE && h.presumedNextMovePla == p);
      testAssert(!h.isGameFinished && h.moveHistory.size() == 1 && b.numStonesOnBoard() == 1);
      calls++;
      bool isLeft = Location::getX(h.moveHistory.back().loc,b.x_size) < 2;
      // Draw mass deliberately makes whiteWin differ from (1+W-L)/2.
      return Value{isLeft ? 0.5 : 0.75,0.0};
    };
    testAssert(initialize(board,hist,pla,rand,settings,evaluate,{}) == Result::APPLIED);
    testAssert(pla == P_WHITE && hist.moveHistory.size() == 1);
    left += Location::getX(hist.moveHistory[0].loc,board.x_size) < 2;
  }
  // Uniform proposals, acceptance 1 and (3/4)^2: left probability 16/25.
  testAssert(left > 1180 && left < 1380 && calls > 2000);
}
void gomokuCandidates() {
  Settings settings; settings.probability = 1;
  for(G game : {G::FREESTYLE,G::STANDARD,G::RENJU}) {
    Rand rand("Gomoku candidates " + Rules::gameName(game));
    for(int i = 0; i < 40; i++) {
      Board board(11,11); auto hist = history(board,game); Player pla = P_BLACK;
      int calls = 0, reference = -1, baseMoves = -1;
      Player rootPla = C_EMPTY;
      Hash128 rootHash;
      Board rootBoard;
      Evaluate evaluate = [&](const Board& b,const BoardHistory& h,Player p,int ref) {
        testAssert(!h.isGameFinished && h.presumedNextMovePla == p);
        if(calls == 0) {
          reference = ref; rootHash = b.pos_hash; rootPla = p; baseMoves = h.moveHistory.size(); rootBoard = b;
        }
        testAssert(reference == ref);
        if(calls == 1) testAssert(b.pos_hash == rootHash && p == getOpp(rootPla));
        if(calls >= 2) {
          testAssert((int)h.moveHistory.size() == baseMoves+1 && p == getOpp(rootPla));
          if(i % 2 && baseMoves > 0) {
            Loc loc = h.moveHistory.back().loc;
            int x = Location::getX(loc,b.x_size), y = Location::getY(loc,b.x_size);
            bool nearby = false;
            for(int xx = 0; xx < b.x_size; xx++) for(int yy = 0; yy < b.y_size; yy++)
              if(rootBoard.colors[Location::getLoc(xx,yy,b.x_size)] != C_EMPTY && std::abs(x-xx) <= 3 && std::abs(y-yy) <= 3)
                nearby = true;
            testAssert(nearby);
          }
        }
        calls++;
        if(calls == 2 && i % 2)
          return p == P_WHITE ? Value{0.6,0.4} : Value{0.4,0.6};
        return Value{0.5,0.5};
      };
      testAssert(initialize(board,hist,pla,rand,settings,evaluate,{}) == Result::APPLIED);
      testAssert(calls > 2 && !hist.isGameFinished && hist.moveHistory.size() >= 1 && hist.moveHistory.size() <= 10);
      Board replay(11,11); auto replayHist = history(replay,game); Player next = P_BLACK;
      for(const auto move : hist.moveHistory) {
        testAssert(move.pla == next && replayHist.isLegal(replay,move.loc,next));
        replayHist.makeBoardMoveAssumeLegal(replay,move.loc,next,nullptr);
        testAssert(!replayHist.isGameFinished);
        next = getOpp(next);
      }
      testAssert(next == pla && replay.pos_hash == board.pos_hash);
    }
  }
  // No zero-weight candidate may be selected.
  Rand rand("Gomoku zero mass");
  for(int i = 0; i < 20; i++) {
    Board board(9,9); auto hist = history(board,G::RENJU); Player pla = P_BLACK;
    int calls = 0;
    Evaluate evaluate = [&](const Board& b,const BoardHistory& h,Player,int) {
      calls++;
      if(calls <= 2) return Value{0.5,0.5};
      bool allowed = Location::getX(h.moveHistory.back().loc,b.x_size) == 1;
      return allowed ? Value{0.5,0.5} : Value{1,0};
    };
    auto result = initialize(board,hist,pla,rand,settings,evaluate,[&](){return calls > 10000;});
    testAssert(result == Result::APPLIED && calls > 2);
    testAssert(Location::getX(hist.moveHistory.back().loc,board.x_size) == 1);
  }
}
}

void Tests::runBalancedOpeningTests() {
  configuration(); disabledAndInterrupted(); hexAcceptance(); gomokuCandidates();
  std::cout << "Balanced opening tests passed" << std::endl;
}
