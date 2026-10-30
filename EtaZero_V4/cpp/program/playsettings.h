#ifndef PROGRAM_PLAYSETTINGS_H_
#define PROGRAM_PLAYSETTINGS_H_

#include "../core/config_parser.h"
#include "../search/searchprint.h"
#include "balancedopening.h"

struct PlaySettings {
  BalancedOpening::Settings balancedOpening;
  //Play a bunch of mostly policy-distributed moves at the start to initialize a game.
  bool initGamesWithPolicy;
  double policyInitAreaProp; //Avg number of moves is this * board area
  double startPosesPolicyInitAreaProp; //Avg number of moves when using a starting position from sgf
  double policyInitGammaShape; //Controls the shape of policy init
  //Occasionally try some alternative moves and search the responses to them.
  double sidePositionProb;

  //Temperature to use for placing handicap stones and for initializing the board position
  double policyInitAreaTemperature;
  double handicapTemperature;



  //Occasionally fork an entire new game to try out an experimental move in the opening
  double earlyForkGameProb; //Expected number of early forked games per game
  double earlyForkGameExpectedMoveProp; //Fork on average within the first board area * this prop moves
  double forkGameProb; //Expected number of forked games per game
  int forkGameMinChoices; //Fork between the favorite of this many random legal moves, at minimum
  int earlyForkGameMaxChoices; //Fork between the favorite of this many random legal moves, at maximum
  int forkGameMaxChoices; //Fork between the favorite of this many random legal moves, at maximum


  //With this probability, use only this many visits for a move, and record it with only this weight
  double cheapSearchProb;
  int cheapSearchVisits;
  float cheapSearchTargetWeight;

  //Attenuate the number of visits used in positions where one player or the other is extremely winning
  bool reduceVisits;
  double reduceVisitsThreshold; //When mcts value is more extreme than this
  int reduceVisitsThresholdLookback; //Value must be more extreme over the last this many turns
  int reducedVisitsMin; //Number of visits at the most extreme winrate
  float reducedVisitsWeight; //Amount of weight to put on the training sample at minimum visits winrate.

  //Probabilistically favor samples that had high policy surprise (kl divergence).
  double policySurpriseDataWeight;
  //Probabilistically favor samples that had high winLossValue surprise (kl divergence).
  double valueSurpriseDataWeight;
  //If true, value surprise (both for valueSurpriseDataWeight and for reanalyze selection) is the KL divergence
  //of this turn's own search value result from the raw neural net value prediction, instead of the KL divergence
  //of the smoothed forward-looking game result from that prediction. The search value surprise depends only on
  //information available at the time of the search, so it does not condition data weighting/selection on the
  //game's realized outcome, at the cost of not noticing surprises that only the actual continuation of the game
  //would reveal. Its typical magnitude is also smaller, so valueSurpriseDataWeight and
  //reanalyzeValueSurpriseWeight may need retuning when enabling this.
  bool useSearchValueSurprise;
  //Scale frequency weights for writing data by this
  double scaleDataWeight;

  //After the game ends, select some cheap-search positions and redo them with a full search to record
  //as training data, favoring positions whose cheap search was surprising. Replaces the behavior where
  //sufficiently policy-surprising cheap searches would be recorded (at reduced weight) based on the cheap
  //search itself - note that this replaced behavior is disabled for cheap searches whenever useReanalyze is
  //true, even if reanalyzeProp is 0. It remains active for reduced-weight rows that are not cheap searches,
  //i.e. rows whose visits were reduced due to extreme winrates, including reanalyzed rows reduced that way.
  bool useReanalyze;
  //Number of positions reanalyzed is binomial with this probability on the number of cheap-search positions.
  double reanalyzeProp;
  //Positions are drawn without replacement with probability proportional to
  //(reanalyzePolicySurpriseWeight * policySurprise + reanalyzeValueSurpriseWeight * valueSurprise) ** reanalyzeSurpriseExponent
  //where policySurprise and valueSurprise are those of the original cheap search.
  double reanalyzePolicySurpriseWeight;
  double reanalyzeValueSurpriseWeight;
  double reanalyzeSurpriseExponent;
  //If true, reanalyzed positions are recorded exactly like normal full-search positions. If false, they omit the
  //targets derived from the final board and the actual game continuation (final score, future board
  //positions), but still record policy, lead, and value targets (which as usual blend the searched values of this
  //and later turns with the game outcome).
  bool reanalyzeUseOutcomeTargets;

  //Record positions from within the search tree that had at least this many visits, recording only with this weight.
  bool recordTreePositions;
  int recordTreeThreshold;
  float recordTreeTargetWeight;

  //Don't stochastically integerify target weights
  bool noResolveTargetWeights;

  //Resign conditions
  bool allowResignation;
  double resignThreshold; //Require that mcts win value is less than this
  double resignConsecTurns; //Require that both players have agreed on it continuously for this many turns

  //Enable full data recording and a variety of other minor tweaks applying only for self-play training.
  bool forSelfPlay;

  //Asymmetric playouts training
  double handicapAsymmetricPlayoutProb; //Probability of asymmetric playouts on handicap games
  double normalAsymmetricPlayoutProb; //Probability of asymmetric playouts on normal games
  double maxAsymmetricRatio;


  //Record time taken per move
  bool recordTimePerMove;

  //How much of the search tree to write to the log after each move, when logSearchInfo is on.
  PrintTreeOptions logSearchTreeOptions;

  PlaySettings();
  ~PlaySettings();

  static PlaySettings loadForMatch(ConfigParser& cfg);
  static PlaySettings loadForGatekeeper(ConfigParser& cfg);
  static PlaySettings loadForSelfplay(ConfigParser& cfg, bool isDistributed);
};

#endif // PROGRAM_PLAYSETTINGS_H_
