#ifndef PROGRAM_BALANCEDOPENING_H_
#define PROGRAM_BALANCEDOPENING_H_

#include "../core/config_parser.h"
#include "../core/rand.h"
#include "../game/boardhistory.h"
#include <functional>

class Search;

namespace BalancedOpening {
struct Settings {
  double probability = 0.0;
  double dropPow = 10.0;
  double makeOpeningFairRate = 1.0;
  double minAcceptRate = 0.001;
  static Settings load(ConfigParser& cfg, bool forSelfPlay);
};

struct Value { double whiteWin, whiteLoss; };
// reference is 0 for the black model and 1 for the white model. All positions
// are real boards, including MuZero's representation/initial evaluation.
using Evaluate = std::function<Value(const Board&,const BoardHistory&,Player,int reference)>;
enum class Result { SKIPPED, APPLIED, INTERRUPTED };

// Only empty, ongoing placement positions with Black to move are eligible.
// The caller excludes explicit SGF/hint/fork starts. Rejected and interrupted
// attempts never modify the caller's board or history.
Result initialize(Board&,BoardHistory&,Player&,Rand&,const Settings&,
                  const Evaluate&,const std::function<bool()>& shouldStop);
Result initialize(Search* black,Search* white,Board&,BoardHistory&,Player&,Rand&,
                  const Settings&,const std::function<bool()>& shouldStop);
}
#endif
