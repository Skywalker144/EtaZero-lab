#ifndef GAME_PLACEMENT_H_
#define GAME_PLACEMENT_H_

#include "board.h"
#include "rules.h"

namespace Placement {
enum class Forbidden { NONE, OVERLINE, DOUBLE_FOUR, DOUBLE_THREE };
// Renju forbidden placements are legal actions that lose, not masked actions.
Forbidden renjuForbidden(const Board& board, Loc loc);
bool hexConnected(const Board& board, Player pla);
struct Result { bool finished; Player winner; };
// Setup nodes have no last move. Detect geometric terminal states; historical
// Renju double-three/double-four losses require a recorded transition.
Result resultFromSetup(const Board& board, Rules::Game game);
Result play(Board& board, Loc loc, Player pla, Rules::Game game);
}

#endif
