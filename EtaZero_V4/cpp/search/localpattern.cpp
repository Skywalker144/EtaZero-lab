#include "../search/localpattern.h"

#include "../core/test.h"
#include "../neuralnet/nninputs.h"
#include "../game/xiangqi.h"

using namespace std;

LocalPatternHasher::LocalPatternHasher()
  : xSize(),
    ySize(),
    zobristLocalPattern(),
    zobristPla(),
    zobristAtari()
{}


void LocalPatternHasher::init(int x, int y, Rand& rand) {
  xSize = x;
  ySize = y;
  testAssert(xSize > 0 && xSize % 2 == 1);
  testAssert(ySize > 0 && ySize % 2 == 1);
  zobristLocalPattern.resize(NUM_BOARD_COLORS * xSize * ySize);
  zobristPla.resize(NUM_BOARD_COLORS);
  zobristAtari.resize(xSize * ySize);

  for(int i = 0; i<NUM_BOARD_COLORS; i++) {
    for(int dy = 0; dy<ySize; dy++) {
      for(int dx = 0; dx<xSize; dx++) {
        uint64_t h0 = rand.nextUInt64();
        uint64_t h1 = rand.nextUInt64();
        zobristLocalPattern[i * ySize*xSize + dy*xSize + dx] = Hash128(h0,h1);
      }
    }
  }
  for(int i = 0; i<NUM_BOARD_COLORS; i++) {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    zobristPla[i] = Hash128(h0,h1);
  }
  for(int dy = 0; dy<ySize; dy++) {
    for(int dx = 0; dx<xSize; dx++) {
      uint64_t h0 = rand.nextUInt64();
      uint64_t h1 = rand.nextUInt64();
      zobristAtari[dy*xSize + dx] = Hash128(h0,h1);
    }
  }
}

LocalPatternHasher::~LocalPatternHasher() {
}


Hash128 LocalPatternHasher::getHash(const Board& board, Loc loc, Player pla) const {
  Hash128 hash = zobristPla[pla];

  if(loc != Board::PASS_LOC && loc != Board::NULL_LOC) {
    const int dxi = board.adj_offsets[2];
    const int dyi = board.adj_offsets[3];
    assert(dxi == 1);
    assert(dyi == board.x_size+1);

    int xRadius = xSize/2;
    int yRadius = ySize/2;
    int xCenter = xSize/2;
    int yCenter = ySize/2;

    int x = Location::getX(loc,board.x_size);
    int y = Location::getY(loc,board.x_size);
    int dxMin = -xRadius, dxMax = xRadius, dyMin = -yRadius, dyMax = yRadius;
    if(x < xRadius) { dxMin = -x; } else if(x >= board.x_size-xRadius) { dxMax = board.x_size-1-x; }
    if(y < yRadius) { dyMin = -y; } else if(y >= board.y_size-yRadius) { dyMax = board.y_size-1-y; }
    for(int dy = dyMin; dy <= dyMax; dy++) {
      for(int dx = dxMin; dx <= dxMax; dx++) {
        Loc loc2 = loc + dx * dxi + dy * dyi;
        int y2 = dy + yCenter;
        int x2 = dx + xCenter;
        int xy2 = y2 * xSize + x2;
        hash ^= zobristLocalPattern[(int)board.colors[loc2] * xSize * ySize + xy2];
        if((board.colors[loc2] == P_BLACK || board.colors[loc2] == P_WHITE) && board.getNumLiberties(loc2) == 1)
          hash ^= zobristAtari[xy2];
      }
    }
  }

  return hash;
}

Hash128 LocalPatternHasher::getHashWithSym(const Board& board, Loc loc, Player pla, int symmetry, bool flipColors) const {
  Player symPla = flipColors ? getOpp(pla) : pla;
  Hash128 hash = zobristPla[symPla];

  if(loc != Board::PASS_LOC && loc != Board::NULL_LOC) {
    const int dxi = board.adj_offsets[2];
    const int dyi = board.adj_offsets[3];
    assert(dxi == 1);
    assert(dyi == board.x_size+1);

    int xRadius = xSize/2;
    int yRadius = ySize/2;
    int xCenter = xSize/2;
    int yCenter = ySize/2;

    bool transpose = SymmetryHelpers::isTranspose(symmetry);
    bool flipX = SymmetryHelpers::isFlipX(symmetry);
    bool flipY = SymmetryHelpers::isFlipY(symmetry);

    int x = Location::getX(loc,board.x_size);
    int y = Location::getY(loc,board.x_size);
    int dxMin = -xRadius, dxMax = xRadius, dyMin = -yRadius, dyMax = yRadius;
    if(x < xRadius) { dxMin = -x; } else if(x >= board.x_size-xRadius) { dxMax = board.x_size-1-x; }
    if(y < yRadius) { dyMin = -y; } else if(y >= board.y_size-yRadius) { dyMax = board.y_size-1-y; }
    for(int dy = dyMin; dy <= dyMax; dy++) {
      for(int dx = dxMin; dx <= dxMax; dx++) {
        Loc loc2 = loc + dx * dxi + dy * dyi;
        int y2 = dy + yCenter;
        int x2 = dx + xCenter;

        int symXY2;
        int symX2 = flipX ? xSize - x2 - 1 : x2;
        int symY2 = flipY ? ySize - y2 - 1 : y2;
        if(transpose) {
          std::swap(symX2,symY2);
          symXY2 = symY2 * ySize + symX2;
        }
        else {
          symXY2 = symY2 * xSize + symX2;
        }

        int symColor;
        if(board.colors[loc2] == P_BLACK || board.colors[loc2] == P_WHITE)
          symColor = (int)(flipColors ? getOpp(board.colors[loc2]) : board.colors[loc2]);
        else
          symColor = (int)board.colors[loc2];

        hash ^= zobristLocalPattern[symColor * xSize * ySize + symXY2];
        if((board.colors[loc2] == P_BLACK || board.colors[loc2] == P_WHITE) && board.getNumLiberties(loc2) == 1)
          hash ^= zobristAtari[symXY2];
      }
    }
  }

  return hash;
}

Hash128 LocalPatternHasher::getPlacementHash(const Board& board, Loc loc, Player pla, Rules::Game game,
                                           int symmetry, bool flipColors) const {
  const uint64_t gameId = (uint64_t)game;
  Hash128 hash = zobristPla[flipColors ? getOpp(pla) : pla] ^
    Hash128(Hash::splitMix64(0x6e4f2de6739dc169ULL+gameId),Hash::splitMix64(0x2d992c1fb38a7ac5ULL+gameId));
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC) return hash;
  const int x = Location::getX(loc,board.x_size), y = Location::getY(loc,board.x_size);
  const bool transpose = SymmetryHelpers::isTranspose(symmetry);
  const bool flipX = SymmetryHelpers::isFlipX(symmetry), flipY = SymmetryHelpers::isFlipY(symmetry);
  // Clip both ends, including boards smaller than the 5x5/9x9 pattern window.
  for(int dy = std::max(-ySize/2,-y); dy <= std::min(ySize/2,board.y_size-1-y); dy++) {
    for(int dx = std::max(-xSize/2,-x); dx <= std::min(xSize/2,board.x_size-1-x); dx++) {
      int sx = xSize/2 + (flipX ? -dx : dx), sy = ySize/2 + (flipY ? -dy : dy);
      if(transpose) std::swap(sx,sy);
      const int index = sy*(transpose ? ySize : xSize)+sx;
      Color color = board.colors[Location::getLoc(x+dx,y+dy,board.x_size)];
      if(flipColors && (color == P_BLACK || color == P_WHITE)) color = getOpp(color);
      hash ^= zobristLocalPattern[color*xSize*ySize+index];
    }
  }
  return hash;
}

void XiangqiLocalPatternHasher::init(int x, int y, Rand& rand) {
  testAssert(x > 0 && x % 2 == 1 && y > 0 && y % 2 == 1);
  xSize = x;
  ySize = y;
  auto nextHash = [&]() { uint64_t lo = rand.nextUInt64(); return Hash128(lo,rand.nextUInt64()); };
  domain = nextHash();
  for(auto& hash : zobristPla) hash = nextHash();
  zobristPieces.resize(2*15*x*y); // empty and seven piece types for each side
  for(auto& hash : zobristPieces) hash = nextHash();
}

Hash128 XiangqiLocalPatternHasher::getHash(
  const std::array<int8_t,Xiangqi::WIDTH*Xiangqi::HEIGHT>& pieces,
  Loc moveLoc, Player pla, bool mirror, bool flipColors
) const {
  const int action = Xiangqi::locToAction(moveLoc);
  assert(action >= 0);
  const auto& move = Xiangqi::MOVES[action];
  Hash128 hash = domain ^ zobristPla[flipColors ? getOpp(pla) : pla];
  const int area = xSize*ySize;
  for(int endpoint = 0; endpoint < 2; endpoint++) {
    const int x = move[2*endpoint], y = move[2*endpoint+1];
    for(int dy = std::max(-ySize/2,-y); dy <= std::min(ySize/2,Xiangqi::HEIGHT-1-y); dy++) {
      for(int dx = std::max(-xSize/2,-x); dx <= std::min(xSize/2,Xiangqi::WIDTH-1-x); dx++) {
        const int piece = pieces[(y+dy)*Xiangqi::WIDTH+x+dx];
        const int symPiece = flipColors ? -piece : piece;
        const int symDx = (mirror != flipColors) ? -dx : dx;
        const int symDy = flipColors ? -dy : dy;
        hash ^= zobristPieces[(endpoint*15+symPiece+7)*area + (symDy+ySize/2)*xSize + symDx+xSize/2];
      }
    }
  }
  return hash;
}
