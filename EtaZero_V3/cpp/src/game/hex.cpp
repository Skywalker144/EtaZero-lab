#include "etazero/hex.h"
#include "etazero/rule_state.h"
#include "placement_bias.h"
#include <cstdint>
namespace etazero {
bool hex_connected(const Board& board,int player) {
    const int size=board.width;const auto& cells=board.cells;
    if (player != 1 && player != -1) throw std::runtime_error("Invalid Hex player");
    constexpr int dx[] = {0,1,1,0,-1,-1}, dy[] = {-1,-1,0,1,1,0};
    std::vector<uint8_t> seen(size*size,0);
    std::vector<int> pending;
    for (int i=0;i<size;++i) {
        int a=player==1?i:i*size;
        if (cells[a]==player) {seen[a]=1;pending.push_back(a);}
    }
    while (!pending.empty()) {
        int a=pending.back();pending.pop_back();int x=a%size,y=a/size;
        if ((player==1?y:x)==size-1) return true;
        for (int d=0;d<6;++d) {
            int xx=x+dx[d],yy=y+dy[d];
            if (xx<0 || xx>=size || yy<0 || yy>=size) continue;
            int b=yy*size+xx;
            if (!seen[b] && cells[b]==player) {seen[b]=1;pending.push_back(b);}
        }
    }
    return false;
}

}

namespace etazero {
namespace {
class HexState final : public RuleState {
    Board board_;
    int canvas_,player_=1,turn_=0,winner_=0;
public:
    HexState(int size,int canvas):board_(size),canvas_(canvas) {}
    std::unique_ptr<RuleState> clone() const override { return std::make_unique<HexState>(*this); }
    void reset_from(const RuleState& other) override { *this=static_cast<const HexState&>(other); }
    int player() const override { return player_; }
    Outcome outcome() const override { return {winner_,winner_?Reason::CONNECTION:Reason::DRAW}; }
    bool finished() const override { return winner_!=0; }
    const Board& board() const override { return board_; }
    bool subtree_bias_key(int previous,int action,SubtreeValueBiasKey& key) const override {
        return placement_bias_key(board_,canvas_,player_,previous,action,key);
    }
    bool legal(int a) const override {
        return !finished() && a>=0 && a<canvas_*canvas_ && a/canvas_<board_.width && a%canvas_<board_.width &&
               board_.cells[a/canvas_*board_.width+a%canvas_]==0;
    }
    void play(int a) override {
        board_.cells[a/canvas_*board_.width+a%canvas_]=player_;++turn_;
        if(hex_connected(board_,player_))winner_=player_;
        if(turn_==board_.width*board_.width && !winner_)throw std::runtime_error("Full Hex board has no connection");
        player_=-player_;
    }
    std::vector<float> observation(int perspective,double pda,bool) const override {
        auto spec=game_spec(GameKind::HEX);int area=canvas_*canvas_,spatial=spec.planes*area;
        std::vector<float> out(spatial+spec.globals,0);
        for(int y=0;y<board_.width;++y)for(int x=0;x<board_.width;++x) {
            int a=y*canvas_+x,cell=board_.cells[y*board_.width+x];
            out[a]=1;out[area+a]=cell==perspective;out[2*area+a]=cell==-perspective;
        }
        out[spatial+4]=pda!=0;out[spatial+5]=static_cast<float>(.5*pda);
        out[spatial+HEX_GLOBAL]=1;out[spatial+HEX_WHITE_GLOBAL]=perspective==-1;
        return out;
    }
};
}
std::unique_ptr<RuleState> make_hex_state(int size,int canvas) { return std::make_unique<HexState>(size,canvas); }
}
