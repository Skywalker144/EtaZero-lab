#ifndef SEARCH_LOCALPATTERN_H
#define SEARCH_LOCALPATTERN_H

#include "../core/global.h"
#include "../core/hash.h"
#include "../core/rand.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../game/xiangqilayout.h"

struct LocalPatternHasher {
  int xSize;
  int ySize;
  std::vector<Hash128> zobristLocalPattern;
  std::vector<Hash128> zobristPla;
  std::vector<Hash128> zobristAtari;

  LocalPatternHasher();
  ~LocalPatternHasher();

  void init(int xSize, int ySize, Rand& rand);

  Hash128 getHash(const Board& board, Loc loc, Player pla) const;

  //Returns the hash that would occur if symmetry were applied to both board and loc.
  //So basically, the only thing that changes is the zobrist indexing.
  Hash128 getHashWithSym(const Board& board, Loc loc, Player pla, int symmetry, bool flipColors) const;
  // Placement games have no Go liberty/atari state. Keep the source window,
  // colors and player, with a game-specific identity for mixed pattern files.
  Hash128 getPlacementHash(const Board& board, Loc loc, Player pla, Rules::Game game,
                           int symmetry = 0, bool flipColors = false) const;
};

// A joint move changes two neighborhoods. Keep their piece-type hashes in
// separate banks so overlapping origin/destination windows cannot cancel.
struct XiangqiLocalPatternHasher {
  int xSize = 0;
  int ySize = 0;
  std::vector<Hash128> zobristPieces;
  std::array<Hash128,NUM_BOARD_COLORS> zobristPla;
  Hash128 domain;

  void init(int xSize, int ySize, Rand& rand);
  // flipColors also rotates 180 degrees; a bare color swap is not a Xiangqi symmetry.
  Hash128 getHash(const std::array<int8_t,Xiangqi::WIDTH*Xiangqi::HEIGHT>& pieces,
                  Loc moveLoc, Player pla, bool mirror = false, bool flipColors = false) const;
};

#endif //SEARCH_LOCALPATTERN_H
