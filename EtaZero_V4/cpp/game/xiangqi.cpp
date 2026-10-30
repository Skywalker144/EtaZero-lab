#include "xiangqi.h"
#include "../core/sha2.h"
#include "../external/pikafish/position.h"
#include "../external/pikafish/movegen.h"
#include <bitset>
#include <deque>
#include <mutex>

namespace Xiangqi {
namespace sf = Stockfish;

namespace {
struct ActionTable {
  std::array<std::array<int,WIDTH*HEIGHT>,WIDTH*HEIGHT> index;
  std::array<int,NUM_ACTIONS> delta;
  ActionTable() {
    for(auto& row : index) row.fill(-1);
    for(int a = 0; a < NUM_ACTIONS; a++) {
      const auto& m = MOVES[a];
      index[m[1]*WIDTH+m[0]][m[3]*WIDTH+m[2]] = a;
      const std::array<int,2> d{{m[2]-m[0],m[3]-m[1]}};
      delta[a] = (int)(std::find(DELTAS.begin(),DELTAS.end(),d)-DELTAS.begin());
    }
  }
};
const ActionTable& actionTable() { static const ActionTable table; return table; }

std::string mirrorFen(std::string fen) {
  size_t begin = 0;
  for(size_t end = 0; end < fen.size(); end++) {
    if(fen[end] != '/' && fen[end] != ' ') continue;
    std::reverse(fen.begin()+begin,fen.begin()+end);
    begin = end+1;
    if(fen[end] == ' ') break;
  }
  return fen;
}

void checkAction(int action) {
  if(action < 0 || action >= NUM_ACTIONS) throw StringError("Xiangqi action outside its fixed vocabulary");
}

sf::Move nativeMove(int action) {
  const auto& m = MOVES[action];
  return sf::Move(sf::Square(m[1]*WIDTH+m[0]),sf::Square(m[3]*WIDTH+m[2]));
}

std::array<int8_t,WIDTH*HEIGHT> piecesOf(const sf::Position& position) {
  std::array<int8_t,WIDTH*HEIGHT> pieces{};
  for(int i = 0; i < WIDTH*HEIGHT; i++) {
    const auto piece = position.piece_on(sf::Square(i));
    if(piece != sf::NO_PIECE)
      pieces[i] = (int8_t)(sf::type_of(piece)*(sf::color_of(piece) == sf::WHITE ? 1 : -1));
  }
  return pieces;
}
}

void copyInputsWithMirror(const float* src, float* dst, int nx, int ny,
                          int channels, bool useNHWC, bool mirror) {
  const int area = nx*ny;
  std::copy(src,src+channels*area,dst);
  if(!mirror) return;
  for(int c = 0; c < channels; c++) for(int y = 0; y < HEIGHT; y++) for(int x = 0; x < WIDTH; x++) {
    const int from = y*nx+x, to = y*nx+(WIDTH-1-x);
    dst[useNHWC ? to*channels+c : c*area+to] = src[useNHWC ? from*channels+c : c*area+from];
  }
}

int actionOf(int x, int y, int u, int v) {
  if(x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT || u < 0 || u >= WIDTH || v < 0 || v >= HEIGHT)
    return -1;
  return actionTable().index[y*WIDTH+x][v*WIDTH+u];
}

Loc actionToLoc(int action) {
  checkAction(action);
  static_assert(Board::NULL_LOC == 0 && Board::PASS_LOC == 1, "Joint moves reserve the source sentinels");
  static_assert(NUM_ACTIONS+1 <= std::numeric_limits<Loc>::max(), "Joint moves must fit in Loc");
  return (Loc)(action+2);
}

int locToAction(Loc loc) {
  return loc >= 2 && loc < NUM_ACTIONS+2 ? (int)loc-2 : -1;
}

Board State::colorBoard() const {
  Board board(WIDTH,HEIGHT);
  const auto& current = pieces();
  for(int y = 0; y < HEIGHT; y++) for(int x = 0; x < WIDTH; x++) {
    int piece = current[y*WIDTH+x];
    board.colors[Location::getLoc(x,y,WIDTH)] = piece > 0 ? P_BLACK : piece < 0 ? P_WHITE : C_EMPTY;
  }
  // setStone would apply Go captures. Rebuilding chains preserves every piece,
  // including occupied groups with no liberties in the color-only view.
  board.regenChainsFromColors();
  return board;
}

int transformAction(int action, Player perspective, bool mirror) {
  checkAction(action);
  if(perspective != P_BLACK && perspective != P_WHITE) throw StringError("Invalid Xiangqi perspective");
  auto m = MOVES[action];
  if(perspective == P_WHITE) {
    m[0] = WIDTH-1-m[0]; m[2] = WIDTH-1-m[2];
    m[1] = HEIGHT-1-m[1]; m[3] = HEIGHT-1-m[3];
  }
  if(mirror) { m[0] = WIDTH-1-m[0]; m[2] = WIDTH-1-m[2]; }
  return actionOf(m[0],m[1],m[2],m[3]);
}

int policyIndex(int action, int nnXLen, int nnYLen) {
  checkAction(action);
  if(nnXLen < WIDTH || nnYLen < HEIGHT) throw StringError("Xiangqi NN canvas is smaller than 9 x 10");
  const auto& m = MOVES[action];
  return actionTable().delta[action]*nnXLen*nnYLen+m[1]*nnXLen+m[0];
}

std::string actionToString(int action) {
  checkAction(action);
  const auto& m = MOVES[action];
  return std::string{(char)('a'+m[0]),(char)('0'+m[1]),(char)('a'+m[2]),(char)('0'+m[3])};
}

int parseAction(const std::string& text) {
  if(text.size() != 4) return -1;
  return actionOf(text[0]-'a',text[1]-'0',text[2]-'a',text[3]-'0');
}

const std::string State::START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

struct State::Impl {
  sf::Position position;
  std::deque<sf::StateInfo> history;
  std::deque<std::array<int8_t,WIDTH*HEIGHT>> recent;
  std::vector<int> legal;
  std::bitset<NUM_ACTIONS> mask;
  Result outcome;
  int repeats = 1;
  Hash128 hash;
  Hash128 mirrorHash;

  explicit Impl(const std::string& fen) {
    static std::once_flag initialized;
    std::call_once(initialized,[] { sf::Attacks::init(); sf::Position::init(); });
    history.emplace_back();
    const auto error = position.set(fen,&history.back());
    if(error) throw StringError("Invalid Xiangqi FEN: " + std::string(error->what()));
    recent.push_front(piecesOf(position));
    uint64_t digest[4];
    SHA2::get256(position.fen().c_str(),digest);
    hash = Hash128(digest[0],digest[1]);
    const std::string mirroredFen = mirrorFen(position.fen());
    SHA2::get256(mirroredFen.c_str(),digest);
    mirrorHash = Hash128(digest[0],digest[1]);
    update();
  }

  Impl(const Impl& other)
    : history(other.history), recent(other.recent), legal(other.legal), mask(other.mask),
      outcome(other.outcome), repeats(other.repeats), hash(other.hash), mirrorHash(other.mirrorHash) {
    for(size_t i = 0; i < history.size(); i++) history[i].previous = i ? &history[i-1] : nullptr;
    position.copy_game_state(other.position,&history.back());
  }

  Player nextPlayer() const { return position.side_to_move() == sf::WHITE ? P_BLACK : P_WHITE; }

  void update() {
    legal.clear(); mask.reset();
    const sf::MoveList<sf::ALL> moves(position);
    if(moves.size() == 0) { outcome = {true,getOpp(nextPlayer()),Reason::MATE}; return; }
    sf::Value value;
    sf::Position::RuleReason reason;
    if(position.rule_judge(value,0,&reason)) {
      outcome.finished = true;
      outcome.winner = value == sf::VALUE_DRAW ? C_EMPTY : value > 0 ? nextPlayer() : getOpp(nextPlayer());
      switch(reason) {
        case sf::Position::RuleReason::REPETITION: outcome.reason = Reason::REPETITION; break;
        case sf::Position::RuleReason::MOVE_LIMIT: outcome.reason = Reason::MOVE_LIMIT; break;
        case sf::Position::RuleReason::INSUFFICIENT_MATERIAL: outcome.reason = Reason::INSUFFICIENT_MATERIAL; break;
        case sf::Position::RuleReason::MATE: outcome.reason = Reason::MATE; break;
        default: throw StringError("Missing Pikafish adjudication reason");
      }
      return;
    }
    for(const auto move : moves) {
      const int from = (int)move.from_sq(), to = (int)move.to_sq();
      const int action = actionOf(from%WIDTH,from/WIDTH,to%WIDTH,to/WIDTH);
      if(action < 0) throw StringError("Legal Pikafish move missing from the Xiangqi vocabulary");
      mask.set(action); legal.push_back(action);
    }
  }
};

State::State(const std::string& fen) : impl(new Impl(fen)) {}
State::State(const Record& record) : State(record.initialFen) {
  for(int action : record.actions) play(action);
}
State::~State() = default;
State::State(const State& other) : impl(new Impl(*other.impl)) {}
State& State::operator=(const State& other) {
  if(this != &other) impl.reset(new Impl(*other.impl));
  return *this;
}
State::State(State&& other) noexcept = default;
State& State::operator=(State&& other) noexcept = default;
Player State::nextPlayer() const { return impl->nextPlayer(); }
Result State::result() const { return impl->outcome; }
bool State::isLegal(int action) const { return action >= 0 && action < NUM_ACTIONS && impl->mask[action]; }
const std::vector<int>& State::legalActions() const { return impl->legal; }
std::string State::fen() const { return impl->position.fen(); }
const std::array<int8_t,WIDTH*HEIGHT>& State::pieces(int movesAgo) const { return impl->recent.at(movesAgo); }
int State::numRecentBoards() const { return (int)impl->recent.size(); }
int State::previousAction(int movesAgo) const {
  if(movesAgo < 0 || (size_t)movesAgo >= impl->history.size()-1) return -1;
  const auto move = impl->history[impl->history.size()-1-movesAgo].move;
  const int from = (int)move.from_sq(), to = (int)move.to_sq();
  return actionOf(from%WIDTH,from/WIDTH,to%WIDTH,to/WIDTH);
}
int State::rule60Count() const { return impl->history.back().rule60; }
int State::repetitions() const { return impl->repeats; }
int State::gamePly() const { return impl->position.game_ply(); }
bool State::isCheck() const { return bool(impl->position.checkers()); }
uint64_t State::positionKey() const { return impl->history.back().key; }
Hash128 State::historyHash() const { return impl->hash; }
Hash128 State::mirroredHistoryHash() const { return impl->mirrorHash; }

State::Record State::record(int movesAgo) const {
  if(movesAgo < 0 || (size_t)movesAgo >= impl->history.size())
    throw StringError("Xiangqi record requests unavailable history");
  Impl copy(*impl);
  Record result;
  result.actions.resize(copy.history.size()-1-movesAgo);
  for(size_t i = copy.history.size()-1; i > 0; i--) {
    const auto move = copy.history[i].move;
    if(i <= result.actions.size()) {
      int from = (int)move.from_sq(), to = (int)move.to_sq();
      result.actions[i-1] = actionOf(from%WIDTH,from/WIDTH,to%WIDTH,to/WIDTH);
    }
    copy.position.undo_move(move);
  }
  result.initialFen = copy.position.fen();
  return result;
}

State State::mirrored() const {
  Record reflected = record();
  reflected.initialFen = mirrorFen(reflected.initialFen);
  for(int& action : reflected.actions) action = transformAction(action,P_BLACK,true);
  return State(reflected);
}

void State::play(int action) {
  if(!isLegal(action)) throw StringError("Illegal Xiangqi move");
  impl->history.emplace_back();
  impl->position.do_move(nativeMove(action),impl->history.back(),nullptr);
  const auto& current = impl->history.back();
  impl->repeats = 1;
  if(current.capturedPiece == sf::NO_PIECE) {
    for(auto* prior = current.previous; prior; prior = prior->previous) {
      if(prior->key == current.key) impl->repeats++;
      if(prior->capturedPiece != sf::NO_PIECE) break;
    }
  }
  impl->hash = Hash128(Hash::splitMix64(impl->hash.hash0 ^ (uint64_t)(action+1)),
                      Hash::splitMix64(impl->hash.hash1 + 0x9e3779b97f4a7c15ULL + action));
  const int mirroredAction = transformAction(action,P_BLACK,true);
  impl->mirrorHash = Hash128(Hash::splitMix64(impl->mirrorHash.hash0 ^ (uint64_t)(mirroredAction+1)),
                            Hash::splitMix64(impl->mirrorHash.hash1 + 0x9e3779b97f4a7c15ULL + mirroredAction));
  impl->recent.push_front(piecesOf(impl->position));
  if(impl->recent.size() > HISTORY_FRAMES) impl->recent.pop_back();
  impl->update();
}

void State::fillFeatures(Player perspective, double signedPda, bool mirror,
                        int nnXLen, int nnYLen, bool useNHWC, float* spatial, float* global, int maxHistory) const {
  if(perspective != P_BLACK && perspective != P_WHITE) throw StringError("Invalid Xiangqi feature perspective");
  if(nnXLen < WIDTH || nnYLen < HEIGHT || nnXLen > Board::MAX_LEN || nnYLen > Board::MAX_LEN)
    throw StringError("Xiangqi NN canvas must contain 9 x 10 within the compiled board limit");
  const int frames = std::min(numRecentBoards(), 1 + std::clamp(maxHistory,0,HISTORY_FRAMES-1));
  const int area = nnXLen*nnYLen;
  std::fill(spatial,spatial+SPATIAL_FEATURES*area,0.0f);
  std::fill(global,global+GLOBAL_FEATURES,0.0f);
  for(int y = 0; y < HEIGHT; y++) for(int x = 0; x < WIDTH; x++) {
    int u = perspective == P_WHITE ? WIDTH-1-x : x;
    const int v = perspective == P_WHITE ? HEIGHT-1-y : y;
    if(mirror) u = WIDTH-1-u;
    const int pos = v*nnXLen+u;
    auto set = [&](int c) { spatial[useNHWC ? pos*SPATIAL_FEATURES+c : c*area+pos] = 1.0f; };
    set(0);
    for(int frame = 0; frame < frames; frame++) {
      const int piece = impl->recent[frame][y*WIDTH+x];
      if(piece != 0) {
        const bool mine = (piece > 0) == (perspective == P_BLACK);
        set(1+(int)frame*14+(mine ? 0 : 7)+std::abs(piece)-1);
      }
    }
  }
  global[0] = perspective == P_BLACK;
  global[1] = rule60Count()/120.0f;
  global[2] = (repetitions()-1)/2.0f;
  global[3] = signedPda != 0.0;
  global[4] = (float)(signedPda*0.5);
  global[5] = frames/8.0f;
  global[6] = isCheck();
  global[7] = gamePly()/200.0f;
}
}
