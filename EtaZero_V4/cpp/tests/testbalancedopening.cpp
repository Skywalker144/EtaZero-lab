#include "tests.h"
#include "../program/balancedopening.h"
#include "../program/playsettings.h"
#include "../program/playutils.h"
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
  for(bool self : {false,true}) for(G game : {G::FREESTYLE,G::STANDARD,G::RENJU}) {
    ConfigParser cfg(std::map<std::string,std::string>{{"game",Rules::gameName(game)}});
    testAssert(Settings::load(cfg,self).probability == (self ? 0.99 : 1.0));
    cfg.overrideKey("balanceOpeningProb","0");
    testAssert(Settings::load(cfg,self).probability == 0);
  }
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
void policyLength() {
  // A draw representing 10.75 requested policy moves leaves 10,8,2,0,0
  // moves after existing played prefixes 0,1,4,6,20. Board area and gamma
  // have no role in KataGomo; imported and locally played turns both count.
  for(G game : {G::FREESTYLE,G::STANDARD,G::RENJU}) for(int size : {9,15}) {
    for(auto example : {std::pair<int,int>{0,10},{1,8},{4,2},{6,0},{20,0}}) {
      Board board(size,size); auto hist = history(board,game);
      if(example.first > 0) hist.makeBoardMoveAssumeLegal(board,Location::getLoc(0,0,size),P_BLACK,nullptr);
      hist.setInitialTurnNumber(example.first > 0 ? example.first-1 : 0);
      Rand rand("KataGomo policy draw"), reference("KataGomo policy draw");
      double mean = 10.75 / reference.nextExponential();
      testAssert(PlayUtils::samplePolicyInitMoveCount(hist,rand,mean,7) == example.second);
      testAssert(rand.nextUInt64() == reference.nextUInt64());
    }
  }
  for(G game : {G::GO,G::HEX}) for(double gamma : {1.0,3.0}) {
    Board board(9,9); auto hist = history(board,game);
    hist.setInitialTurnNumber(20);
    Rand rand("other policy draw"), reference("other policy draw");
    int expected = (int)std::floor(gamma == 1 ? reference.nextExponential()*6 : reference.nextGamma(gamma)*2);
    testAssert(PlayUtils::samplePolicyInitMoveCount(hist,rand,6,gamma) == expected);
    testAssert(rand.nextUInt64() == reference.nextUInt64());
  }
}
void policyConfiguration() {
  ConfigParser cfg(std::map<std::string,std::string>{
    {"game","renju"},{"initGamesWithPolicy","true"},{"sidePositionProb","0"},
    {"earlyForkGameProb","0"},{"earlyForkGameExpectedMoveProp","0"},{"forkGameProb","0"},
    {"forkGameMinChoices","1"},{"earlyForkGameMaxChoices","1"},{"forkGameMaxChoices","1"},
    {"cheapSearchProb","0"},{"cheapSearchVisits","1"},{"cheapSearchTargetWeight","1"},
    {"reduceVisits","false"},{"reduceVisitsThreshold","0.9"},{"reduceVisitsThresholdLookback","1"},
    {"reducedVisitsMin","1"},{"reducedVisitsWeight","1"},
    {"policySurpriseDataWeight","0"},{"valueSurpriseDataWeight","0"},
    {"handicapAsymmetricPlayoutProb","0"},{"normalAsymmetricPlayoutProb","0"},{"maxAsymmetricRatio","2"},
    {"allowResignation","false"},{"resignThreshold","-0.9"},{"resignConsecTurns","1"}});
  auto self = PlaySettings::loadForSelfplay(cfg,false);
  testAssert(self.policyInitAvgMoveNum == 12 && self.startPosesPolicyInitAvgMoveNum == 0);
  testAssert(self.policyInitAreaTemperature == 1 && self.balancedOpening.probability == 0.99);
  bool missingMean = false;
  try { PlaySettings::loadForMatch(cfg); } catch(const StringError&) { missingMean = true; }
  testAssert(missingMean); // KataGomo match requires a mean when policy init is enabled.
  cfg.overrideKey("policyInitAvgMoveNum","6");
  cfg.overrideKey("startPosesPolicyInitAvgMoveNum","3");
  cfg.overrideKey("policyInitAreaTemperature","1.6");
  for(auto settings : {PlaySettings::loadForSelfplay(cfg,false),PlaySettings::loadForMatch(cfg)}) {
    testAssert(settings.policyInitAvgMoveNum == 6 && settings.startPosesPolicyInitAvgMoveNum == 3);
    testAssert(settings.policyInitAreaTemperature == 1.6);
  }
  auto gate = PlaySettings::loadForGatekeeper(cfg);
  testAssert(gate.balancedOpening.probability == 1 && !gate.initGamesWithPolicy);
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
  Rand gated("nonempty KataGomo gate"), source("nonempty KataGomo gate");
  source.nextBool(settings.probability);
  testAssert(initialize(board,hist,pla,gated,settings,unavailable,{}) == Result::SKIPPED);
  testAssert(hist.moveHistory.size() == 1 && board.numStonesOnBoard() == 1);
  testAssert(gated.nextUInt64() == source.nextUInt64());
  board = Board(7,7);
  hist = BoardHistory(board,P_WHITE,Rules::forGame(G::RENJU),0,BoardHistoryModes(false,false));
  pla = P_WHITE;
  Evaluate even = [](const Board&,const BoardHistory&,Player,int) { return Value{0.5,0.5}; };
  testAssert(initialize(board,hist,pla,rand,settings,even,{}) == Result::APPLIED);
  testAssert(hist.moveHistory.front().pla == P_WHITE);
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
        if(calls == 1) {
          testAssert(b.pos_hash == rootHash && p == getOpp(rootPla));
          testAssert(h.moveHistory.size() == baseMoves);
        }
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
  configuration(); policyLength(); policyConfiguration(); disabledAndInterrupted(); hexAcceptance(); gomokuCandidates();
  std::cout << "Balanced opening tests passed" << std::endl;
}
