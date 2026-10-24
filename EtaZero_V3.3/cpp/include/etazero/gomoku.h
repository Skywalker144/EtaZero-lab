#pragma once
#include "rules.h"
#include <map>
namespace etazero {
struct GomokuBoard : Board {
    using Board::Board;
    int offset(int origin,int direction,int distance) const;
    std::array<int,4> lengths(int action,int player) const;
};
enum class Forbidden { NONE, OVERLINE, DOUBLE_FOUR, DOUBLE_THREE };
class RenjuAnalyzer {
    std::map<std::pair<std::vector<int>,int>,Forbidden> cache_;
public:
    Forbidden analyze(const GomokuBoard& board,int action);
    bool forbidden(const GomokuBoard& board,int action) { return analyze(board,action)!=Forbidden::NONE; }
};
}
