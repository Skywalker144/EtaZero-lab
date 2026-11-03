#ifndef NEURALNET_NNINPUTS_H_
#define NEURALNET_NNINPUTS_H_

#include <memory>

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"

namespace Xiangqi { class State; }

namespace NNPos {
  constexpr int MAX_BOARD_LEN = Board::MAX_LEN;
  constexpr int MAX_BOARD_AREA = MAX_BOARD_LEN * MAX_BOARD_LEN;
  //Policy output adds +1 for the pass move
  constexpr int MAX_NN_POLICY_SIZE = MAX_BOARD_AREA + 1;
  //Extra score distribution radius, used for writing score in data rows and for the neural net score belief output
  //Used various places we clip komi beyond board area.
  constexpr float KOMI_CLIP_RADIUS = 20.0f;

  inline int xyToPos(int x, int y, int nnXLen) { return y * nnXLen + x; }
  inline int locToPos(Loc loc, int boardXSize, int nnXLen, int nnYLen) {
    if(loc == Board::PASS_LOC)
      return nnXLen * nnYLen;
    else if(loc == Board::NULL_LOC)
      return nnXLen * (nnYLen + 1);
    return Location::getY(loc,boardXSize) * nnXLen + Location::getX(loc,boardXSize);
  }
  inline Loc posToLoc(int pos, int boardXSize, int boardYSize, int nnXLen, int nnYLen) {
    if(pos == nnXLen * nnYLen)
      return Board::PASS_LOC;
    int x = pos % nnXLen;
    int y = pos / nnXLen;
    if(x < 0 || x >= boardXSize || y < 0 || y >= boardYSize)
      return Board::NULL_LOC;
    return Location::getLoc(x,y,boardXSize);
  }
  inline int getPassPos(int nnXLen, int nnYLen) { return nnXLen * nnYLen; }
  inline bool isPassPos(int pos, int nnXLen, int nnYLen) { return pos == nnXLen * nnYLen; }
  inline int getPolicySize(int nnXLen, int nnYLen) { return nnXLen * nnYLen + 1; }
}

namespace NNInputs {
  constexpr int SYMMETRY_NOTSPECIFIED = -1;
  constexpr int SYMMETRY_ALL = -2;
}

struct MiscNNInputParams {
  double drawEquivalentWinsForWhite = 0.5;
  double playoutDoublingAdvantage = 0.0;
  float nnPolicyTemperature = 1.0f;
  // If no symmetry is specified, it will use default or random based on config, unless node is already cached.
  int symmetry = NNInputs::SYMMETRY_NOTSPECIFIED;
  double policyOptimism = 0.0;
  int maxHistory = 1000;
  // -1 = no override: pass-alive featurization follows hist.modes.alwaysComputePassAliveUnderSuicideRules.
  // 0/1 = override that flag for featurization (and its contribution to the nn cache hash). Used when
  // evaluating with a secondary net (e.g. a human SL profile net) whose declared featurization mode
  // differs from the mode the search itself is using.
  int passAliveSuicideRulesOverride = -1;
  // Same, but for hist.modes.excludeTerritoryAdjacentToAtari.
  int excludeTerritoryAdjAtariOverride = -1;

  // The effective value of modes.alwaysComputePassAliveUnderSuicideRules for featurization: the
  // override if one is set, else hist's own flag.
  bool getAlwaysComputePassAliveUnderSuicideRules(const BoardHistory& hist) const;
  // The effective value of modes.excludeTerritoryAdjacentToAtari for featurization: the override
  // if one is set, else hist's own flag.
  bool getExcludeTerritoryAdjacentToAtari(const BoardHistory& hist) const;
  // The effective full modes for featurization, taking both overrides into account.
  BoardHistoryModes getModes(const BoardHistory& hist) const;
  // The suicide legality to use for pass-alive area computations in featurization, taking the
  // override and hist.modes.alwaysComputePassAliveUnderSuicideRules into account.
  bool getSuicideLegalForPassAlive(const BoardHistory& hist) const;

  static const Hash128 ZOBRIST_PLAYOUT_DOUBLINGS;
  static const Hash128 ZOBRIST_NN_POLICY_TEMP;
  static const Hash128 ZOBRIST_POLICY_OPTIMISM;
  static const Hash128 ZOBRIST_ZERO_HISTORY;
};

namespace NNInputs {
  const int NUM_FEATURES_SPATIAL_V3 = 22;
  const int NUM_FEATURES_GLOBAL_V3 = 14;

  const int NUM_FEATURES_SPATIAL_V4 = 22;
  const int NUM_FEATURES_GLOBAL_V4 = 14;

  const int NUM_FEATURES_SPATIAL_V5 = 13;
  const int NUM_FEATURES_GLOBAL_V5 = 12;

  const int NUM_FEATURES_SPATIAL_V6 = 22;
  const int NUM_FEATURES_GLOBAL_V6 = 16;

  const int NUM_FEATURES_SPATIAL_V7 = 22;
  const int NUM_FEATURES_GLOBAL_V7 = 19;

  // Placement inputs: mask, own/opponent stones, black forbidden points by perspective.
  // Globals: standard, renju, renju color, forbidden enabled, PDA enabled/value, hex, hex color.
  const int NUM_FEATURES_SPATIAL_V8 = 5;
  const int NUM_FEATURES_GLOBAL_V8 = 8;

  void fillRowV8(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );

  Hash128 getHash(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams
  );

  Hash128 getHash(const Xiangqi::State& state, const MiscNNInputParams& nnInputParams);

  void fillRow(int inputsVersion, const Board& board, const BoardHistory& hist, Player nextPlayer,
               const MiscNNInputParams& params, int nnXLen, int nnYLen, bool useNHWC,
               float* rowBin, float* rowGlobal);

  void fillRowV3(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV4(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV5(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV6(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );
  void fillRowV7(
    const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
    const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
  );

  //If groupTax is specified, for each color region of area, reduce weight on empty spaces equally to reduce the total sum by 2.
  //(but should handle seki correctly)
  void fillScoring(
    const Board& board,
    const Color* area,
    bool groupTax,
    float* scoring
  );
}

struct NNOutput {
  Hash128 nnHash; //NNInputs - getHash

  //Initially from the perspective of the player to move at the time of the eval, fixed up later in nnEval.cpp
  //to be the value from white's perspective.
  //These four are categorical probabilities; D and NR are perspective-independent.
  float whiteWinProb;
  float whiteLossProb;
  float whiteDrawProb = 0.0f;
  float whiteNoResultProb;

  //Expected arrival time of remaining game variance, in turns, weighted by variance, only when modelVersion >= 9
  float varTimeLeft;
  //A metric indicating the "typical" error in the winloss value that the net expects, relative to the
  //short-term future MCTS value.
  float shorttermWinlossError;

  //Indexed by pos rather than loc
  //Values in here will be set to negative for illegal moves, including superko
  float* policyProbs;
  //The optimism value used for this neural net evaluation.
  float policyOptimismUsed;

  int nnXLen;
  int nnYLen;

  //If not NULL, then contains policy with dirichlet noise or any other noise adjustments for this node
  float* noisedPolicyProbs;

  NNOutput(); //Does NOT initialize values
  NNOutput(const NNOutput& other);
  ~NNOutput();

  //Averages the others. Others must be nonempty and share the same nnHash (i.e. be for the same board, except if somehow the hash collides)
  //Does NOT carry over noisedPolicyProbs.
  NNOutput(const std::vector<std::shared_ptr<NNOutput>>& others);

  NNOutput& operator=(const NNOutput&);

  // Point policies keep the source inline storage. Joint action spaces allocate
  // only for that output; the Go cache footprint does not grow to 2086 floats.
  void resizePolicy(int size);
  int getPolicyStorageSize() const { return policyStorageSize; }

  inline float* getPolicyProbsMaybeNoised() { return noisedPolicyProbs != NULL ? noisedPolicyProbs : policyProbs; }
  inline const float* getPolicyProbsMaybeNoised() const { return noisedPolicyProbs != NULL ? noisedPolicyProbs : policyProbs; }
  void debugPrint(std::ostream& out, const Board& board) const;
  inline int getPos(Loc loc, const Board& board) const { return NNPos::locToPos(loc, board.x_size, nnXLen, nnYLen ); }

 private:
  float pointPolicyProbs[NNPos::MAX_NN_POLICY_SIZE];
  std::unique_ptr<float[]> extendedPolicyProbs;
  int policyStorageSize = NNPos::MAX_NN_POLICY_SIZE;
};

namespace SymmetryHelpers {
  //A symmetry is 3 bits flipY(bit 0), flipX(bit 1), transpose(bit 2). They are applied in that order.
  //The first four symmetries only reflect, and do not transpose X and Y.
  constexpr int NUM_SYMMETRIES = 8;
  constexpr int NUM_SYMMETRIES_WITHOUT_TRANSPOSE = 4;

  //These two IGNORE transpose if hSize and wSize do not match. So non-square transposes are disallowed.
  //copyOutputsWithSymmetry performs the inverse of symmetry.
  void copyInputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry);
  void copyOutputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int symmetry);

  //Applies a symmetry to a location
  Loc getSymLoc(int x, int y, const Board& board, int symmetry);
  Loc getSymLoc(Loc loc, const Board& board, int symmetry);
  Loc getSymLoc(int x, int y, int xSize, int ySize, int symmetry);
  Loc getSymLoc(Loc loc, int xSize, int ySize, int symmetry);

  //Applies a symmetry to a board
  Board getSymBoard(const Board& board, int symmetry);

  //Get the inverse of the specified symmetry
  int invert(int symmetry);
  //Get the symmetry equivalent to first applying firstSymmetry and then applying nextSymmetry.
  int compose(int firstSymmetry, int nextSymmetry);
  int compose(int firstSymmetry, int nextSymmetry, int nextNextSymmetry);

  inline bool isTranspose(int symmetry) { return (symmetry & 0x4) != 0; }
  inline bool isFlipX(int symmetry) { return (symmetry & 0x2) != 0; }
  inline bool isFlipY(int symmetry) { return (symmetry & 0x1) != 0; }

  //Fill isSymDupLoc with true on all but one copy of each symmetrically equivalent move, and false everywhere else.
  //isSymDupLocs should be an array of size Board::MAX_ARR_SIZE
  //If onlySymmetries is not NULL, will only consider the symmetries specified there.
  //validSymmetries will be filled with all symmetries of the current board, including using history for checking ko/superko and some encore-related state.
  //This implementation is dependent on specific order of the symmetries (i.e. transpose is coded as 0x4)
  //Will pretend moves that have a nonzero value in avoidMoves do not exist.
  void markDuplicateMoveLocs(
    const Board& board,
    const BoardHistory& hist,
    const std::vector<int>* onlySymmetries,
    const std::vector<int>& avoidMoves,
    bool* isSymDupLoc,
    std::vector<int>& validSymmetries
  );

// For each symmetry, return a metric about the "amount" of difference that board would have with other
// if symmetry were applied to board.
  void getSymmetryDifferences(
    const Board& board, const Board& other, double maxDifferenceToReport, double symmetryDifferences[NUM_SYMMETRIES]
  );
}

//Score transforms used by source model formats, score reports and rule target encoding.


#endif  // NEURALNET_NNINPUTS_H_
