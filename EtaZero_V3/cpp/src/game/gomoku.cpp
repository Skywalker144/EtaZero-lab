// Renju semantics adapted from MuZero_V2; see THIRD_PARTY.md.
#include "etazero/gomoku.h"
#include "etazero/rule_state.h"
#include "placement_bias.h"
#include <set>
#include <cstdint>
#include <stdexcept>

namespace etazero {

int GomokuBoard::offset(int origin, int direction, int distance) const {
    constexpr int dx[] = {1, 0, 1, 1}, dy[] = {0, 1, 1, -1};
    int x = origin % width + dx[direction] * distance;
    int y = origin / width + dy[direction] * distance;
    return x >= 0 && x < width && y >= 0 && y < width ? y * width + x : -1;
}

std::array<int, 4> GomokuBoard::lengths(int action, int player) const {
    std::array<int, 4> result{};
    for (int d = 0; d < 4; ++d) {
        result[d] = 1;
        for (int sign : {-1, 1}) {
            for (int distance = sign; ; distance += sign) {
                int loc = offset(action, d, distance);
                if (loc < 0 || cells[loc] != player) break;
                ++result[d];
            }
        }
    }
    return result;
}


namespace {
std::vector<int> run(const GomokuBoard& board, int anchor, int direction) {
    std::vector<int> negative;
    for (int distance = -1; ; --distance) {
        int loc = board.offset(anchor, direction, distance);
        if (loc < 0 || board.cells[loc] != 1) break;
        negative.push_back(loc);
    }
    std::reverse(negative.begin(), negative.end());
    negative.push_back(anchor);
    for (int distance = 1; ; ++distance) {
        int loc = board.offset(anchor, direction, distance);
        if (loc < 0 || board.cells[loc] != 1) break;
        negative.push_back(loc);
    }
    return negative;
}
}

Forbidden RenjuAnalyzer::analyze(const GomokuBoard& board, int action) {
    if (action < 0 || action >= board.width * board.width || board.cells[action] != 0)
        return Forbidden::NONE;
    int nearby = 0;
    for (int d = 0; d < 4; ++d)
        for (int distance : {-2, -1, 1, 2}) {
            int loc = board.offset(action, d, distance);
            nearby += loc >= 0 && board.cells[loc] == 1;
        }
    if (nearby < 2) return Forbidden::NONE;
    auto key = std::make_pair(board.cells, action);
    auto found = cache_.find(key);
    if (found != cache_.end()) return found->second;
    GomokuBoard after = board;
    after.cells[action] = 1;
    auto lengths = after.lengths(action, 1);
    if (std::find(lengths.begin(), lengths.end(), 5) != lengths.end()) return cache_[key] = Forbidden::NONE;
    if (*std::max_element(lengths.begin(), lengths.end()) > 5) return cache_[key] = Forbidden::OVERLINE;
    std::set<std::vector<int>> fours, threes;
    for (int target : {5, 4}) {
        for (int d = 0; d < 4; ++d) {
            for (int distance = -4; distance <= 4; ++distance) {
                int extension = after.offset(action, d, distance);
                if (distance == 0 || extension < 0 || after.cells[extension] != 0) continue;
                GomokuBoard extended = after;
                extended.cells[extension] = 1;
                auto stones = run(extended, extension, d);
                if (static_cast<int>(stones.size()) != target ||
                    std::find(stones.begin(), stones.end(), action) == stones.end()) continue;
                if (target == 4) {
                    auto extension_lengths = extended.lengths(extension, 1);
                    if (std::find(extension_lengths.begin(), extension_lengths.end(), 5) != extension_lengths.end()) continue;
                    int left = extended.offset(stones.front(), d, -1), right = extended.offset(stones.back(), d, 1);
                    if (left < 0 || right < 0 || extended.cells[left] != 0 || extended.cells[right] != 0 ||
                        extended.lengths(left, 1)[d] != 5 || extended.lengths(right, 1)[d] != 5 || forbidden(after, extension)) continue;
                }
                stones.erase(std::find(stones.begin(), stones.end(), extension));
                std::sort(stones.begin(), stones.end());
                (target == 5 ? fours : threes).insert(stones);
            }
        }
        if (target == 5 && fours.size() >= 2) return cache_[key] = Forbidden::DOUBLE_FOUR;
    }
    return cache_[key] = threes.size() >= 2 ? Forbidden::DOUBLE_THREE : Forbidden::NONE;
}

}

namespace etazero {
namespace {
class GomokuState final : public RuleState {
    GomokuBoard board_;
    Rule rule_;
    int canvas_,player_=1,turn_=0,winner_=0;
    Reason reason_=Reason::DRAW;
    bool finished_=false;
public:
    GomokuState(Rule rule,int size,int canvas):board_(size),rule_(rule),canvas_(canvas) {}
    std::unique_ptr<RuleState> clone() const override { return std::make_unique<GomokuState>(*this); }
    void reset_from(const RuleState& other) override { *this=static_cast<const GomokuState&>(other); }
    int player() const override { return player_; }
    Outcome outcome() const override { return {winner_,reason_}; }
    bool finished() const override { return finished_; }
    const Board& board() const override { return board_; }
    bool subtree_bias_key(int previous,int action,SubtreeValueBiasKey& key) const override {
        return placement_bias_key(board_,canvas_,player_,previous,action,key);
    }
    bool legal(int a) const override {
        return !finished_ && a>=0 && a<canvas_*canvas_ && a/canvas_<board_.width && a%canvas_<board_.width &&
               board_.cells[a/canvas_*board_.width+a%canvas_]==0;
    }
    void play(int a) override {
        int local=a/canvas_*board_.width+a%canvas_;
        if(rule_==Rule::RENJU && player_==1) {
            RenjuAnalyzer analyzer;
            if(analyzer.forbidden(board_,local)){winner_=-1;reason_=Reason::FORBIDDEN;}
        }
        board_.cells[local]=player_;++turn_;
        if(!winner_)for(int n:board_.lengths(local,player_))
            if(n==5 || (n>5 && (rule_==Rule::FREESTYLE || (rule_==Rule::RENJU && player_==-1)))) {
                winner_=player_;reason_=Reason::FIVE;
            }
        finished_=winner_ || turn_==board_.width*board_.width;player_=-player_;
    }
    std::vector<float> observation(int perspective,double pda,bool forbidden_feature) const override {
        auto spec=game_spec(rule_);int area=canvas_*canvas_,spatial=spec.planes*area;
        std::vector<float> out(spatial+spec.globals,0);
        RenjuAnalyzer analyzer;bool enabled=rule_==Rule::RENJU && forbidden_feature;
        for(int y=0;y<board_.width;++y)for(int x=0;x<board_.width;++x) {
            int a=y*canvas_+x,local=y*board_.width+x,cell=board_.cells[local];
            out[a]=1;out[area+a]=cell==perspective;out[2*area+a]=cell==-perspective;
            if(enabled && !cell && analyzer.forbidden(board_,local))out[(perspective==1?3:4)*area+a]=1;
        }
        out[spatial]=rule_==Rule::STANDARD;out[spatial+1]=rule_==Rule::RENJU;
        out[spatial+2]=rule_==Rule::RENJU?-perspective:0;out[spatial+3]=enabled;
        out[spatial+4]=pda!=0;out[spatial+5]=static_cast<float>(.5*pda);
        return out;
    }
};
}
std::unique_ptr<RuleState> make_gomoku_state(Rule rule,int size,int canvas) {
    return std::make_unique<GomokuState>(rule,size,canvas);
}
}
