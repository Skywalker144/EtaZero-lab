#ifndef GAME_XIANGQI_H_
#define GAME_XIANGQI_H_

#include "board.h"
#include "xiangqilayout.h"
#include <memory>

namespace Xiangqi {

// Fixed geometric vocabulary, independent of a position's actual legal moves.
int actionOf(int fromX, int fromY, int toX, int toY);
int transformAction(int action, Player perspective, bool mirror);
int policyIndex(int action, int nnXLen, int nnYLen);
std::string actionToString(int action);
int parseAction(const std::string& text);
// Joint moves use the existing Move/Loc storage, reserving NULL_LOC and PASS_LOC.
// These values are action identifiers, never indices into Board's point arrays.
Loc actionToLoc(int action);
int locToAction(Loc loc); // -1 for sentinels or values outside the vocabulary

// Inputs already use the requested player's perspective. Reflection is about
// the physical nine files, keeping padding on the right of the NN canvas.
void copyInputsWithMirror(const float* src, float* dst, int nnXLen, int nnYLen,
                          int channels, bool useNHWC, bool mirror);

enum class Reason { NONE, MATE, REPETITION, MOVE_LIMIT, INSUFFICIENT_MATERIAL };
struct Result {
  bool finished = false;
  Player winner = C_EMPTY;
  Reason reason = Reason::NONE;
};

// P_BLACK denotes the first player (red); P_WHITE denotes black. Piece signs
// follow the same convention; absolute piece types are Pikafish's 1..7.
// The implementation owns every StateInfo link. Copying never starts from FEN,
// since FEN cannot restore the repetition/check/chase history.
class State {
 public:
  struct Record {
    std::string initialFen;
    std::vector<int> actions;
  };
  static const std::string START_FEN;
  explicit State(const std::string& fen = START_FEN);
  explicit State(const Record& record);
  ~State();
  State(const State& other);
  State& operator=(const State& other);
  State(State&& other) noexcept;
  State& operator=(State&& other) noexcept;

  Player nextPlayer() const;
  Result result() const;
  bool isLegal(int action) const;
  const std::vector<int>& legalActions() const;
  void play(int action);
  std::string fen() const;
  const std::array<int8_t,WIDTH*HEIGHT>& pieces(int movesAgo = 0) const;
  int numRecentBoards() const;
  int previousAction(int movesAgo = 0) const; // -1 before the owned history's origin
  int rule60Count() const;
  int repetitions() const;
  int gamePly() const;
  bool isCheck() const;
  uint64_t positionKey() const;
  // Includes the normalized initial FEN and every played action; no bounded Go
  // graph-history reset is applied to Xiangqi's history-dependent adjudication.
  Hash128 historyHash() const;
  Hash128 mirroredHistoryHash() const;
  // File boundaries only: recover the origin and complete legal history from
  // the owned StateInfo chain. Search copies do not store a duplicate move log.
  Record record(int movesAgo = 0) const;
  // File augmentation: reflect the origin and replay the complete history so
  // repetition, check/chase and no-progress adjudication remain intact.
  State mirrored() const;

  // Color-only view for source interfaces. Piece identity and rule history stay
  // in State; Board::pos_hash remains the hash of this occupancy view.
  Board colorBoard() const;

  // Float features in NCHW/NHWC and the fixed global order. Rotate physical
  // 9x10 coordinates for black's perspective, then optionally mirror files.
  // Padding stays zero, including the extra column of a 10x10 NN canvas.
  void fillFeatures(Player perspective, double signedPda, bool mirror,
                    int nnXLen, int nnYLen, bool useNHWC, float* spatial, float* global,
                    int maxHistory = HISTORY_FRAMES-1) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}
#endif
