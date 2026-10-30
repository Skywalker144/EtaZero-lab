#include "placement.h"

#include <array>
#include <map>
#include <set>

// Renju continuation analysis follows EtaZero V3.4's gomoku.cpp, adapted to the
// source Board/Player representation. Recursive legality matters for real threes.
namespace {
struct RenjuPosition {
  int size;
  std::vector<Color> cells;
  explicit RenjuPosition(const Board& board) : size(board.x_size), cells(size*size) {
    for(int y = 0; y < size; y++)
      for(int x = 0; x < size; x++)
        cells[y*size+x] = board.colors[Location::getLoc(x,y,size)];
  }
  int offset(int origin, int direction, int distance) const {
    constexpr int dx[] = {1,0,1,1}, dy[] = {0,1,1,-1};
    int x = origin % size + dx[direction]*distance;
    int y = origin / size + dy[direction]*distance;
    return x >= 0 && x < size && y >= 0 && y < size ? y*size+x : -1;
  }
  std::array<int,4> lengths(int action, Player pla) const {
    std::array<int,4> lengths{};
    for(int d = 0; d < 4; d++) {
      lengths[d] = 1;
      for(int sign : {-1,1})
        for(int distance = sign; ; distance += sign) {
          int loc = offset(action,d,distance);
          if(loc < 0 || cells[loc] != pla) break;
          lengths[d]++;
        }
    }
    return lengths;
  }
  std::vector<int> run(int anchor, int d) const {
    std::vector<int> stones;
    for(int distance = -1; ; distance--) {
      int loc = offset(anchor,d,distance);
      if(loc < 0 || cells[loc] != P_BLACK) break;
      stones.push_back(loc);
    }
    std::reverse(stones.begin(), stones.end());
    stones.push_back(anchor);
    for(int distance = 1; ; distance++) {
      int loc = offset(anchor,d,distance);
      if(loc < 0 || cells[loc] != P_BLACK) break;
      stones.push_back(loc);
    }
    return stones;
  }
};

class RenjuAnalyzer {
  std::map<std::pair<std::vector<Color>,int>,Placement::Forbidden> cache;
public:
  Placement::Forbidden analyze(const RenjuPosition& board, int action) {
    using F = Placement::Forbidden;
    int nearby = 0;
    for(int d = 0; d < 4; d++)
      for(int distance : {-2,-1,1,2}) {
        int loc = board.offset(action,d,distance);
        nearby += loc >= 0 && board.cells[loc] == P_BLACK;
      }
    if(nearby < 2) return F::NONE;
    auto key = std::make_pair(board.cells,action);
    auto found = cache.find(key);
    if(found != cache.end()) return found->second;
    RenjuPosition after = board;
    after.cells[action] = P_BLACK;
    auto lengths = after.lengths(action,P_BLACK);
    if(std::find(lengths.begin(), lengths.end(), 5) != lengths.end()) return cache[key] = F::NONE;
    if(*std::max_element(lengths.begin(), lengths.end()) > 5) return cache[key] = F::OVERLINE;
    std::set<std::vector<int>> fours, threes;
    for(int target : {5,4}) {
      for(int d = 0; d < 4; d++) {
        for(int distance = -4; distance <= 4; distance++) {
          int extension = after.offset(action,d,distance);
          if(distance == 0 || extension < 0 || after.cells[extension] != C_EMPTY) continue;
          RenjuPosition extended = after;
          extended.cells[extension] = P_BLACK;
          auto stones = extended.run(extension,d);
          if((int)stones.size() != target || std::find(stones.begin(),stones.end(),action) == stones.end()) continue;
          if(target == 4) {
            auto extensionLengths = extended.lengths(extension,P_BLACK);
            if(std::find(extensionLengths.begin(),extensionLengths.end(),5) != extensionLengths.end()) continue;
            int left = extended.offset(stones.front(),d,-1), right = extended.offset(stones.back(),d,1);
            if(left < 0 || right < 0 || extended.cells[left] != C_EMPTY || extended.cells[right] != C_EMPTY ||
               extended.lengths(left,P_BLACK)[d] != 5 || extended.lengths(right,P_BLACK)[d] != 5 ||
               analyze(after,extension) != F::NONE) continue;
          }
          stones.erase(std::find(stones.begin(),stones.end(),extension));
          std::sort(stones.begin(),stones.end());
          (target == 5 ? fours : threes).insert(stones);
        }
      }
      if(target == 5 && fours.size() >= 2) return cache[key] = F::DOUBLE_FOUR;
    }
    return cache[key] = threes.size() >= 2 ? F::DOUBLE_THREE : F::NONE;
  }
};
}

Placement::Forbidden Placement::renjuForbidden(const Board& board, Loc loc) {
  assert(board.x_size == board.y_size && board.isOnBoard(loc) && board.colors[loc] == C_EMPTY);
  RenjuAnalyzer analyzer;
  return analyzer.analyze(RenjuPosition(board),Location::getY(loc,board.x_size)*board.x_size + Location::getX(loc,board.x_size));
}

bool Placement::hexConnected(const Board& board, Player pla) {
  const int size = board.x_size;
  constexpr int dx[] = {0,1,1,0,-1,-1}, dy[] = {-1,-1,0,1,1,0};
  std::array<bool,Board::MAX_ARR_SIZE> seen{};
  std::vector<Loc> pending;
  for(int i = 0; i < size; i++) {
    Loc loc = pla == P_BLACK ? Location::getLoc(i,0,size) : Location::getLoc(0,i,size);
    if(board.colors[loc] == pla) { seen[loc] = true; pending.push_back(loc); }
  }
  while(!pending.empty()) {
    Loc loc = pending.back();
    pending.pop_back();
    int x = Location::getX(loc,size), y = Location::getY(loc,size);
    if((pla == P_BLACK ? y : x) == size-1) return true;
    for(int d = 0; d < 6; d++) {
      int u = x+dx[d], v = y+dy[d];
      if(u < 0 || u >= size || v < 0 || v >= size) continue;
      Loc next = Location::getLoc(u,v,size);
      if(!seen[next] && board.colors[next] == pla) { seen[next] = true; pending.push_back(next); }
    }
  }
  return false;
}

Placement::Result Placement::resultFromSetup(const Board& board, Rules::Game game) {
  assert(game != Rules::Game::GO && board.x_size == board.y_size);
  if(board.numStonesOnBoard() == 0) return {false,C_EMPTY};
  bool blackWins = false, whiteWins = false, blackOverline = false;
  if(game == Rules::Game::HEX) {
    blackWins = hexConnected(board,P_BLACK);
    whiteWins = hexConnected(board,P_WHITE);
  }
  else {
    RenjuPosition position(board);
    for(int action = 0; action < (int)position.cells.size(); action++) {
      Player pla = position.cells[action];
      if(pla != P_BLACK && pla != P_WHITE) continue;
      for(int length : position.lengths(action,pla)) {
        if(length == 5 || (length > 5 && (game == Rules::Game::FREESTYLE || (game == Rules::Game::RENJU && pla == P_WHITE)))) {
          if(pla == P_BLACK) blackWins = true;
          else whiteWins = true;
        }
        if(game == Rules::Game::RENJU && pla == P_BLACK && length > 5) blackOverline = true;
      }
    }
    // As in renjuForbidden, exact five has precedence over simultaneous threats.
    if(blackOverline && !blackWins) whiteWins = true;
  }
  if(blackWins && whiteWins) throw StringError("Placement setup has conflicting winners");
  if(blackWins || whiteWins) return {true,blackWins ? P_BLACK : P_WHITE};
  return {board.numStonesOnBoard() == board.x_size*board.y_size,C_EMPTY};
}

Placement::Result Placement::play(Board& board, Loc loc, Player pla, Rules::Game game) {
  assert(game != Rules::Game::GO && board.colors[loc] == C_EMPTY);
  const bool forbidden = game == Rules::Game::RENJU && pla == P_BLACK && renjuForbidden(board,loc) != Forbidden::NONE;
  // Rebuild without captures. The source utility also supports zero-liberty
  // groups, preserving Board consistency for generic IO and search diagnostics.
  board.colors[loc] = pla;
  board.clearSimpleKoLoc();
  board.regenChainsFromColors();
  if(forbidden) return {true,P_WHITE};
  if(game == Rules::Game::HEX) {
    if(hexConnected(board,pla)) return {true,pla};
    if(board.numStonesOnBoard() == board.x_size*board.y_size)
      throw StringError("Full Hex board reached without the mover connecting; invalid starting state");
  }
  else {
    RenjuPosition position(board);
    int action = Location::getY(loc,board.x_size)*board.x_size + Location::getX(loc,board.x_size);
    for(int length : position.lengths(action,pla))
      if(length == 5 || (length > 5 && (game == Rules::Game::FREESTYLE || (game == Rules::Game::RENJU && pla == P_WHITE))))
        return {true,pla};
  }
  return {board.numStonesOnBoard() == board.x_size*board.y_size,C_EMPTY};
}
