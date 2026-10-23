#include "etazero/rule_state.h"
#include "../../external/pikafish/position.h"
#include "../../external/pikafish/movegen.h"
#include <deque>
#include <mutex>
namespace etazero {
namespace {
namespace sf=Stockfish;
const std::string start_fen="rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";
// Stateless SplitMix64 finalizer for two independently salted pattern hashes.
uint64_t mix_bias_token(uint64_t x) {
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
class XiangqiState final: public RuleState {
    sf::Position position_;
    std::deque<sf::StateInfo> history_;
    std::deque<std::vector<int>> recent_;
    std::vector<uint8_t> legal_;
    std::string initial_, identity_;
    Board board_{9,10,10};
    int canvas_,winner_=0;
    Reason reason_=Reason::DRAW;
    bool finished_=false;
    // Cache the NN repetition feature and SVB guard once per transition.
    int repeats_=1;
    bool repeated_since_capture_=false;
    sf::Move move(int a) const {
        auto m=XIANGQI_MOVES[a];return sf::Move(sf::Square(m[1]*9+m[0]),sf::Square(m[3]*9+m[2]));
    }
    void update() {
        legal_.assign(game_spec(GameKind::XIANGQI).fixed_actions,0);
        sf::MoveList<sf::ALL> moves(position_);
        if(!moves.size()){finished_=true;winner_=-player();reason_=Reason::MATE;return;}
        sf::Value result;sf::Position::RuleReason reason;
        if(position_.rule_judge(result,0,&reason)) {
            finished_=true;winner_=result==sf::VALUE_DRAW?0:(result>0?player():-player());
            switch(reason) {
                case sf::Position::RuleReason::REPETITION:reason_=Reason::REPETITION;break;
                case sf::Position::RuleReason::MOVE_LIMIT:reason_=Reason::MOVE_LIMIT;break;
                case sf::Position::RuleReason::INSUFFICIENT_MATERIAL:reason_=Reason::INSUFFICIENT_MATERIAL;break;
                case sf::Position::RuleReason::MATE:reason_=Reason::MATE;break;
                default:throw std::runtime_error("Missing Xiangqi adjudication reason");
            }
            return;
        }
        for(auto m:moves) {
            auto from=m.from_sq(),to=m.to_sq();int a=xiangqi_action(int(from)%9,int(from)/9,int(to)%9,int(to)/9);
            if(a<0)throw std::runtime_error("Xiangqi move missing from action vocabulary");
            legal_[a]=1;
        }
    }
    void update_repetitions() {
        const auto& current=history_.back();repeats_=1;
        if(current.capturedPiece!=sf::NO_PIECE){repeated_since_capture_=false;return;}
        for(auto* h=current.previous;h;h=h->previous) {
            if(h->key==current.key)++repeats_;
            // Material only decreases: no board can recur across a capture.
            if(h->capturedPiece!=sf::NO_PIECE)break;
        }
        repeated_since_capture_|=repeats_>1;
    }
public:
    XiangqiState(int canvas,const RuleOptions& options):initial_(options.fen.empty()?start_fen:options.fen),identity_(initial_),canvas_(canvas) {
        history_.emplace_back();auto error=position_.set(initial_,&history_.back());
        if(error)throw std::runtime_error(error->what());
        update_board();recent_.push_back(board_.cells);update();
    }
    XiangqiState(const XiangqiState& other) { reset_from(other); }
    void reset_from(const RuleState& state) override {
        const auto& other=static_cast<const XiangqiState&>(state);
        history_=other.history_;recent_=other.recent_;legal_=other.legal_;board_=other.board_;
        initial_=other.initial_;identity_=other.identity_;canvas_=other.canvas_;
        winner_=other.winner_;reason_=other.reason_;finished_=other.finished_;
        repeats_=other.repeats_;repeated_since_capture_=other.repeated_since_capture_;
        for(size_t i=0;i<history_.size();++i)history_[i].previous=i?&history_[i-1]:nullptr;
        position_.copy_game_state(other.position_,&history_.back());
    }
    std::unique_ptr<RuleState> clone() const override {return std::make_unique<XiangqiState>(*this);}
    int player() const override {return position_.side_to_move()==sf::WHITE?1:-1;}
    Outcome outcome() const override { return {winner_,reason_}; }
    bool finished() const override{return finished_;}
    bool legal(int a) const override{return !finished_&&a>=0&&a<int(legal_.size())&&legal_[a];}
    void play(int a) override {
        history_.emplace_back();position_.do_move(move(a),history_.back(),nullptr);
        update_repetitions();
        identity_.append(reinterpret_cast<const char*>(&a),sizeof(a));
        update_board();recent_.push_front(board_.cells);if(recent_.size()>8)recent_.pop_back();update();
    }
    const Board& board() const override { return board_; }
    bool subtree_bias_key(int previous,int action,SubtreeValueBiasKey& key) const override {
        if(previous<0 || !legal(action) || repeated_since_capture_)return false;
        key={};uint64_t index=0;
        auto add=[&](uint32_t value) {
            // Feature index prevents permutations of identical pieces/values
            // from aliasing. This is a statistical pattern, not a state ID.
            uint64_t token=(index++<<32)|value;
            key.pattern^=mix_bias_token(token+0x9e3779b97f4a7c15ULL);
            key.moves^=mix_bias_token(token+0xd1b54a32d192ed03ULL);
        };
        add(previous);add(action);add(position_.side_to_move());
        const auto& state=history_.back();
        add(state.rule60);add(state.check10[sf::WHITE]);add(state.check10[sf::BLACK]);
        add(bool(position_.checkers()));
        auto piece=[&](int x,int y) {
            if(x<0 || x>=9 || y<0 || y>=10)return 15;
            int p=board_.cells[board_.index(x,y)];
            return p<0?7-p:p; // empty=0, red=1..7, black=8..14, wall=15
        };
        auto m=XIANGQI_MOVES[action];
        for(int end:{0,2}) {
            int x=m[end],y=m[end+1];
            for(int dy=-1;dy<=1;++dy)for(int dx=-1;dx<=1;++dx)add(piece(x+dx,y+dy));
            // Rook/cannon context beyond the 3x3: first two blockers on each
            // ray, with absolute distance. Zero slots denote an empty ray.
            for(auto direction:{std::array<int,2>{1,0},{-1,0},{0,1},{0,-1}}) {
                int found=0;
                for(int distance=1;;++distance) {
                    int p=piece(x+distance*direction[0],y+distance*direction[1]);
                    if(p==15)break;
                    if(p){add(p|(distance<<4));if(++found==2)break;}
                }
                while(found++<2)add(0);
            }
        }
        return true;
    }
    void update_board() {
        auto& out=board_.cells;std::fill(out.begin(),out.end(),0);
        for(int y=0;y<board_.height;++y)for(int x=0;x<board_.width;++x){auto p=position_.piece_on(sf::Square(y*9+x));if(p!=sf::NO_PIECE)out[board_.index(x,y)]=int(sf::type_of(p))*(sf::color_of(p)==sf::WHITE?1:-1);}
    }
    std::vector<float> observation(int perspective,double pda,bool) const override {
        auto spec=game_spec(GameKind::XIANGQI);
        int area=canvas_*canvas_;std::vector<float> out(spec.planes*area+spec.globals,0);
        for(int y=0;y<board_.height;++y)for(int x=0;x<board_.width;++x) {
            int a=y*canvas_+x;out[a]=1;
            for(size_t t=0;t<recent_.size();++t) {
                int pc=recent_[t][board_.index(x,y)];if(!pc)continue;
                int plane=1+int(t)*14+(pc*perspective>0?0:7)+std::abs(pc)-1;
                out[plane*area+a]=1;
            }
        }
        auto g=out.data()+spec.planes*area;
        g[0]=perspective==1;g[1]=position_.rule60_count()/120.f;
        g[2]=(repeats_-1)/2.f;g[3]=pda!=0;g[4]=pda*.5;
        g[5]=recent_.size()/8.f;g[6]=bool(position_.checkers());g[7]=position_.game_ply()/200.f;
        return out;
    }
    bool symmetry_history_ok(const std::vector<int>& mapping) const override {
        for(size_t i=initial_.size();i<identity_.size();i+=sizeof(int)) {
            int action;std::memcpy(&action,identity_.data()+i,sizeof(action));if(mapping[action]!=action)return false;
        }
        return true;
    }
    std::string identity() const override{return identity_;}
    std::string fen() const override{return position_.fen();}
};
}
std::unique_ptr<RuleState> make_xiangqi_state(int,int canvas,const RuleOptions& options) {
    static std::once_flag init;std::call_once(init,[]{sf::Attacks::init();sf::Position::init();});
    return std::make_unique<XiangqiState>(canvas,options);
}
}
