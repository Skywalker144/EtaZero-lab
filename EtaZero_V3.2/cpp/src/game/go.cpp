#include "etazero/rule_state.h"
#include "../../external/katago/neuralnet/nninputs.h"
#include "../../external/katago/game/graphhash.h"
#include <mutex>
#include "../../external/katago/search/localpattern.h"
namespace etazero {
namespace {
class GoState final: public RuleState {
    ::Board board_;
    ::BoardHistory history_;
    mutable Board display_;
    mutable bool display_dirty_=true;
    int canvas_;
    RuleOptions options_;
    mutable int graph_rep_bound_=-1;
    mutable Hash128 graph_hash_;
    ::Loc location(int a) const { return a==canvas_*canvas_?::Board::PASS_LOC:Location::getLoc(a%canvas_,a/canvas_,board_.x_size); }
public:
    GoState(int size,int canvas,const RuleOptions& options):board_(size,size),display_(size),canvas_(canvas),options_(options) {
        if(!std::isfinite(options.komi)||options.komi<::Rules::MIN_USER_KOMI||options.komi>::Rules::MAX_USER_KOMI||options.komi*2!=std::round(options.komi*2))
            throw std::runtime_error("Go komi must be an integer or half-integer in [-400,400]");
        auto rules=::Rules::parseRulesWithoutKomi(options.go_rules,options.komi);
        history_=::BoardHistory(board_,P_BLACK,rules,0,::BoardHistoryModes(true,true));
    }
    std::unique_ptr<RuleState> clone() const override { return std::make_unique<GoState>(*this); }
    void reset_from(const RuleState& other) override { *this=static_cast<const GoState&>(other); }
    int player() const override { return history_.presumedNextMovePla==P_BLACK?1:-1; }
    Outcome outcome() const override {
        return {history_.winner==C_EMPTY?0:history_.winner==P_BLACK?1:-1,
                history_.isNoResult?Reason::NO_RESULT:Reason::SCORE};
    }
    bool finished() const override { return history_.isGameFinished; }
    bool legal(int a) const override {
        if(finished()||a<0||a>canvas_*canvas_)return false;
        if(a!=canvas_*canvas_&&(a%canvas_>=board_.x_size||a/canvas_>=board_.y_size))return false;
        return history_.isLegal(board_,location(a),history_.presumedNextMovePla);
    }
    void play(int a) override {
        history_.makeBoardMoveAssumeLegal(board_,location(a),history_.presumedNextMovePla,nullptr);display_dirty_=true;
        if(graph_rep_bound_>=0)
            graph_hash_=GraphHash::getGraphHash(graph_hash_,history_,history_.presumedNextMovePla,graph_rep_bound_,.5);
    }
    bool subtree_bias_key(int,int action,SubtreeValueBiasKey& key) const override {
        // KataGo associates a new child with the board BEFORE its move. A
        // preceding pass is allowed; the selected pass and absent history are not.
        if(history_.moveHistory.empty() || action==canvas_*canvas_)return false;
        ::Loc previous=history_.moveHistory.back().loc;
        if(previous==::Board::NULL_LOC)return false;
        key={};uint64_t atari=0;int x=action%canvas_,y=action/canvas_;
        for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
            int xx=x+dx,yy=y+dy,color=C_WALL;
            if(xx>=0 && yy>=0 && xx<board_.x_size && yy<board_.y_size) {
                ::Loc loc=Location::getLoc(xx,yy,board_.x_size);color=board_.colors[loc];
                if((color==P_BLACK || color==P_WHITE) && board_.getNumLiberties(loc)==1)
                    atari|=uint64_t{1}<<((dy+2)*5+dx+2);
            }
            key.pattern=(key.pattern<<2)|static_cast<uint64_t>(color);
        }
        // Exact encoding of LocalPatternHasher + move locations + ko ban:
        // 50 color bits, 10 ko bits; 16+16 location bits, 2 player bits, 25 atari bits.
        // Atari uses the full chain's liberties, even beyond the 5x5 window.
        static_assert(::Board::MAX_ARR_SIZE<=1024);
        key.pattern|=static_cast<uint64_t>(board_.ko_loc)<<50;
        key.moves=static_cast<uint64_t>(previous) | (static_cast<uint64_t>(location(action))<<16) |
                  (static_cast<uint64_t>(history_.presumedNextMovePla)<<32) | (atari<<34);
        return true;
    }
    std::string pattern_key(int action,int symmetry,bool flip) const override {
        ::Loc loc=location(action);auto pla=history_.presumedNextMovePla;
        if(loc==::Board::PASS_LOC||loc==::Board::NULL_LOC||board_.wouldBeKoCapture(loc,pla))return {};
        struct Hasher {
            LocalPatternHasher local;Hash128 moves[::Board::MAX_ARR_SIZE];
            Hasher(){Rand rand("PatternBonusTable ZOBRIST STUFF");local.init(9,9,rand);
                rand.init("Reseed PatternBonusTable zobrist so that zobrists don't change when Board::MAX_ARR_SIZE changes");
                for(auto& h:moves){auto a=rand.nextUInt64();auto b=rand.nextUInt64();h=Hash128(a,b);}}
        };
        static const Hasher hasher;
        auto hash=hasher.local.getHashWithSym(board_,loc,pla,symmetry,flip);
        hash^=hasher.moves[SymmetryHelpers::getSymLoc(loc,board_,symmetry)];
        hash^=::Board::ZOBRIST_SIZE_X_HASH[board_.x_size];hash^=::Board::ZOBRIST_SIZE_Y_HASH[board_.y_size];
        return hash==Hash128()?std::string():"go:"+hash.toString();
    }
    const Board& board() const override {
        if(!display_dirty_)return display_;
        auto& out=display_.cells;
        for(int y=0;y<board_.y_size;++y)for(int x=0;x<board_.x_size;++x){auto c=board_.colors[Location::getLoc(x,y,board_.x_size)];out[y*board_.x_size+x]=c==C_EMPTY?0:c==P_BLACK?1:-1;}
        display_dirty_=false;return display_;
    }
    std::vector<float> observation(int perspective,double pda,bool) const override {
        auto spec=game_spec(GameKind::GO);
        std::vector<float> out(spec.planes*canvas_*canvas_+spec.globals);
        MiscNNInputParams params;params.playoutDoublingAdvantage=pda;
        NNInputs::fillRowV7(board_,history_,perspective==1?P_BLACK:P_WHITE,params,canvas_,canvas_,false,out.data(),out.data()+spec.planes*canvas_*canvas_);
        return out;
    }
    std::vector<int> root_symmetry_representatives(const std::vector<int>& avoid) const override {
        std::vector<int> avoids(::Board::MAX_ARR_SIZE,0),valid;bool duplicate[::Board::MAX_ARR_SIZE];
        if(!avoid.empty())for(int y=0;y<board_.y_size;++y)for(int x=0;x<board_.x_size;++x)avoids[Location::getLoc(x,y,board_.x_size)]=avoid[y*canvas_+x];
        SymmetryHelpers::markDuplicateMoveLocs(board_,history_,nullptr,avoids,duplicate,valid);
        std::vector<int> result(canvas_*canvas_+1);for(size_t a=0;a<result.size();++a)result[a]=a;
        for(int y=0;y<board_.y_size;++y)for(int x=0;x<board_.x_size;++x) {
            auto loc=Location::getLoc(x,y,board_.x_size);if(!duplicate[loc])continue;
            for(int symmetry:valid){auto mapped=SymmetryHelpers::getSymLoc(loc,board_,symmetry);if(!duplicate[mapped]) {
                result[y*canvas_+x]=Location::getY(mapped,board_.x_size)*canvas_+Location::getX(mapped,board_.x_size);break;}}
        }
        return result;
    }
    std::string identity() const override {
        std::string key=history_.rules.toJsonString();
        for(auto m:history_.moveHistory){key.append(reinterpret_cast<const char*>(&m.loc),sizeof(m.loc));key.push_back(char(m.pla));}
        return key;
    }
    std::string graph_identity(int rep_bound) const override {
        if(rep_bound<3 || rep_bound>50)throw std::runtime_error("Go graph repetition bound must be in [3,50]");
        // Initialize once at the root; cloned search states update incrementally.
        // Full BoardHistory remains intact for legality, scoring and NN features.
        if(graph_rep_bound_!=rep_bound) {
            graph_hash_=GraphHash::getGraphHashFromScratch(history_,history_.presumedNextMovePla,rep_bound,.5);
            graph_rep_bound_=rep_bound;
        }
        return graph_hash_.toString();
    }
};
}
std::unique_ptr<RuleState> make_go_state(int size,int canvas,const RuleOptions& options) {
    static std::once_flag init;std::call_once(init,[]{::Board::initHash();});
    return std::make_unique<GoState>(size,canvas,options);
}
}
