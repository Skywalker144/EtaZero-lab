// Adapted from pinned KataGomo sources; see reference_sources.json and
// licenses/KataGomo.txt. Only no-VC Gomoku and Hex first-move balancing apply.
#include "balancedopening.h"
#include "../search/search.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace BalancedOpening {
Settings Settings::load(ConfigParser& cfg, bool forSelfPlay) {
  Settings s;
  const auto game = cfg.contains("game") ? Rules::parseGame(cfg.getString("game")) : Rules::Game::GO;
  const bool hex = game == Rules::Game::HEX;
  s.probability = cfg.contains("balanceOpeningProb") ? cfg.getDouble("balanceOpeningProb",0.0,1.0) : 0.0;
  if(s.probability > 0.0 && !Rules::forGame(game).isPlacement())
    throw StringError("balanceOpeningProb is supported only for Gomoku and Hex");
  s.dropPow = cfg.contains("balanceOpeningDropPow") ?
    cfg.getDouble("balanceOpeningDropPow",0.0,std::numeric_limits<double>::max()) :
    (hex ? (forSelfPlay ? 6.0 : 20.0) : (forSelfPlay ? 4.0 : 10.0));
  s.makeOpeningFairRate = cfg.contains("makeOpeningFairRate") ? cfg.getDouble("makeOpeningFairRate",0.0,1.0) : (forSelfPlay ? 0.98 : 1.0);
  s.minAcceptRate = cfg.contains("minAcceptRate") ? cfg.getDouble("minAcceptRate",0.0,1.0) : (forSelfPlay ? 0.005 : 0.001);
  if(s.minAcceptRate <= 0.0)
    throw StringError("minAcceptRate must be positive");
  if(!hex && (cfg.contains("makeOpeningFairRate") || cfg.contains("minAcceptRate")))
    throw StringError("makeOpeningFairRate and minAcceptRate are Hex opening settings");
  return s;
}

namespace {
// Source visits points in x-major order. Unlike its negative epsilon, strict
// positive sampling cannot select an occupied/terminal zero-weight point.
int sample(const std::vector<double>& weights, Rand& rand) {
  double total = 0.0;
  for(double w : weights) total += w;
  if(total == 0.0) return -1;
  double draw = rand.nextDouble() * total;
  int last = -1;
  for(size_t i = 0; i < weights.size(); i++) {
    if(weights[i] <= 0.0) continue;
    last = (int)i;
    if(draw < weights[i]) return last;
    draw -= weights[i];
  }
  return last; // Rounding at the positive cumulative endpoint.
}
Loc point(int index, const Board& b) { return Location::getLoc(index / b.y_size,index % b.y_size,b.x_size); }
Loc nearby(const Board& b, Rand& rand, double avgDist) {
  if(b.numStonesOnBoard() == 0) {
    double xd = rand.nextGaussianTruncated(1.5 * 0.999) / 3.0;
    double yd = rand.nextGaussianTruncated(1.5 * 0.999) / 3.0;
    int x = (int)std::round(xd * b.x_size + 0.5 * (b.x_size-1));
    int y = (int)std::round(yd * b.y_size + 0.5 * (b.y_size-1));
    return Location::getLoc(x,y,b.x_size);
  }
  std::vector<double> weights(b.x_size*b.y_size,0.0);
  const double hx = 0.5 * (b.x_size-1), hy = 0.5 * (b.y_size-1), half = std::max(hx,hy);
  for(int x = 0; x < b.x_size; x++) for(int y = 0; y < b.y_size; y++) {
    if(b.colors[Location::getLoc(x,y,b.x_size)] == C_EMPTY) continue;
    for(int xx = 0; xx < b.x_size; xx++) for(int yy = 0; yy < b.y_size; yy++) {
      if(b.colors[Location::getLoc(xx,yy,b.x_size)] != C_EMPTY) continue;
      double center = 1.5 * (half - std::max(std::abs(xx-hx),std::abs(yy-hy))) / half;
      double distance = (xx-x)*(xx-x) + (yy-y)*(yy-y) + avgDist*avgDist;
      weights[xx*b.y_size+yy] += (1.0+center) * std::pow(distance,-2);
    }
  }
  int index = sample(weights,rand);
  return index < 0 ? Board::NULL_LOC : point(index,b);
}
bool nearStone(const Board& b, int x, int y) {
  for(int xx = std::max(0,x-3); xx <= std::min(b.x_size-1,x+3); xx++)
    for(int yy = std::max(0,y-3); yy <= std::min(b.y_size-1,y+3); yy++)
      if(b.colors[Location::getLoc(xx,yy,b.x_size)] != C_EMPTY) return true;
  return false;
}
struct Interrupted {};
}

Result initialize(Board& board, BoardHistory& hist, Player& pla, Rand& rand,
                  const Settings& settings, const Evaluate& evaluate, const std::function<bool()>& shouldStop) {
  if(settings.probability == 0.0) return Result::SKIPPED;
  if(!hist.rules.isPlacement()) throw StringError("Balanced openings require Gomoku or Hex");
  if(hist.isGameFinished || pla != P_BLACK || !hist.moveHistory.empty() || board.numStonesOnBoard() != 0)
    return Result::SKIPPED;
  if(board.x_size < 2 || board.y_size < 2)
    throw StringError("Balanced openings require both board dimensions to be at least 2");
  if(!rand.nextBool(settings.probability)) return Result::SKIPPED;
  const auto stop = [&]() { if(shouldStop && shouldStop()) throw Interrupted(); };
  const auto value = [&](const Board& b,const BoardHistory& h,Player p,int reference) {
    stop();
    // Placement legality follows presumedNextMovePla. The counterfactual root
    // uses the same stones and rules, with the opposite player to move.
    Value v;
    if(h.presumedNextMovePla != p) {
      BoardHistory input = h;
      input.clear(b,p,h.rules,h.encorePhase);
      v = evaluate(b,input,p,reference);
    }
    else v = evaluate(b,h,p,reference);
    stop();
    if(!std::isfinite(v.whiteWin) || !std::isfinite(v.whiteLoss) ||
       v.whiteWin < 0 || v.whiteWin > 1 || v.whiteLoss < 0 || v.whiteLoss > 1 ||
       v.whiteWin+v.whiteLoss > 1.000001)
      throw StringError("Balanced opening evaluator returned invalid W/L probabilities");
    return v;
  };
  const auto utility = [&](const Board& b,const BoardHistory& h,Player p,int reference) {
    Value v = value(b,h,p,reference);
    return p == P_WHITE ? v.whiteWin-v.whiteLoss : v.whiteLoss-v.whiteWin;
  };
  try {
    stop();
    if(hist.rules.game == Rules::Game::HEX) {
      if(!rand.nextBool(settings.makeOpeningFairRate)) return Result::SKIPPED;
      for(;;) {
        stop();
        int x = rand.nextUInt(board.x_size), y = rand.nextUInt(board.y_size);
        Loc loc = Location::getLoc(x,y,board.x_size);
        Board b = board;
        BoardHistory h = hist;
        h.makeBoardMoveAssumeLegal(b,loc,P_BLACK,nullptr);
        double p = value(b,h,P_WHITE,1).whiteWin;
        double bias = 2*p-1;
        double accept = std::max(std::pow(1-bias*bias,settings.dropPow),settings.minAcceptRate);
        if(!rand.nextBool(accept)) continue;
        board = b; hist = h; pla = P_WHITE;
        return Result::APPLIED;
      }
    }
    const std::vector<double> counts{10,30,50,80,60,40,20,10,5,1,0,0};
    int failures = 0;
    for(;;) {
      stop();
      // The source relaxes rejection after more than 20 failed attempts.
      const double rejectProb = failures > 20 ? 0.8 : 0.995;
      if(failures <= 20) failures++;
      Board b = board;
      BoardHistory h = hist;
      Player p = pla;
      int length = sample(counts,rand);
      double avgDist = rand.nextExponential() * 0.8;
      for(int i = 0; i < length; i++) {
        stop();
        Loc loc = nearby(b,rand,avgDist);
        if(loc == Board::NULL_LOC) break;
        h.makeBoardMoveAssumeLegal(b,loc,p,nullptr);
        p = getOpp(p);
        if(h.isGameFinished) break;
      }
      if(h.isGameFinished) continue;
      const int reference = rand.nextBool(0.5) ? 0 : 1;
      const double own = utility(b,h,p,reference);
      if(own < 0 && rand.nextBool(1-std::exp(-3*own*own)) && rand.nextBool(rejectProb)) continue;
      const double opp = utility(b,h,getOpp(p),reference);
      if(opp < 0 && rand.nextBool(1-std::exp(-3*opp*opp)) && rand.nextBool(rejectProb)) continue;
      const bool local = opp > 0 && b.numStonesOnBoard() > 0;
      double maxWeight = 0;
      std::vector<double> weights(b.x_size*b.y_size,0.0);
      for(int x = 0; x < b.x_size; x++) for(int y = 0; y < b.y_size; y++) {
        stop();
        Loc loc = Location::getLoc(x,y,b.x_size);
        if(!h.isLegal(b,loc,p) || (local && !nearStone(b,x,y))) continue;
        Board candidate = b;
        BoardHistory candidateHist = h;
        candidateHist.makeBoardMoveAssumeLegal(candidate,loc,p,nullptr);
        if(candidateHist.isGameFinished) continue;
        double v = utility(candidate,candidateHist,getOpp(p),reference);
        double w = std::pow(1-v*v,settings.dropPow);
        weights[x*b.y_size+y] = w;
        maxWeight = std::max(maxWeight,w);
      }
      if(rand.nextBool(1-maxWeight) && rand.nextBool(rejectProb)) continue;
      int index = sample(weights,rand);
      if(index < 0) continue;
      h.makeBoardMoveAssumeLegal(b,point(index,b),p,nullptr);
      board = b; hist = h; pla = getOpp(p);
      return Result::APPLIED;
    }
  }
  catch(const Interrupted&) { return Result::INTERRUPTED; }
}

Result initialize(Search* black, Search* white, Board& board, BoardHistory& hist, Player& pla, Rand& rand,
                  const Settings& settings, const std::function<bool()>& shouldStop) {
  // Source AlphaZero cold start has no model with which to balance positions.
  if((!black->muZeroEvaluator && !black->nnEvaluator->isNeuralNetEnabled()) ||
     (!white->muZeroEvaluator && !white->nnEvaluator->isNeuralNetEnabled())) return Result::SKIPPED;
  NNResultBuf buffer;
  Evaluate evaluate = [&](const Board& b,const BoardHistory& h,Player p,int reference) {
    Search* bot = reference == 0 ? black : white;
    MiscNNInputParams params;
    const BoardHistoryModes modes = bot->mainEvaluatorHistoryModes();
    params.passAliveSuicideRulesOverride = modes.alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
    params.excludeTerritoryAdjAtariOverride = modes.excludeTerritoryAdjacentToAtari ? 1 : 0;
    bot->evaluatePosition(b,h,p,params,buffer,false);
    return Value{buffer.result->whiteWinProb,buffer.result->whiteLossProb};
  };
  return initialize(board,hist,pla,rand,settings,evaluate,shouldStop);
}
}
