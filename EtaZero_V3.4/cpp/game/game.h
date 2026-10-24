#pragma once
#include "game/rule_state.h"
#include <cmath>
namespace etazero {
// Common state ownership and NN conditioning. Rules live in RuleState implementations.
class Game {
    std::unique_ptr<RuleState> state_;
    RuleOptions options_;
    int size_,canvas_,turn_=0,last_action_=-1;
    Rule rule_;
    std::vector<int> moves_;
    double pda_doublings_=0;
    int pda_player_=0;
public:
    explicit Game(int size,int canvas,Rule rule,const RuleOptions& options={});
    Game(int size,int canvas,Rule rule,const RuleOptions& options,std::unique_ptr<RuleState> state,std::vector<int> moves,int turn)
        :state_(std::move(state)),options_(options),size_(size),canvas_(canvas),turn_(turn),last_action_(moves.empty()?-1:moves.back()),rule_(rule),moves_(std::move(moves)) {
        if(rule_==Rule::GO) {options_.go_width=width();options_.go_height=height();}
    }
    Game(const Game& other);
    Game& operator=(const Game& other);
    Game(Game&&)=default;
    Game& operator=(Game&&)=default;
    GameSpec spec() const { return game_spec(rule_); }
    const RuleOptions& options() const { return options_; }
    const std::vector<int>& moves() const {return moves_;}
    std::string pattern_key(int action,int symmetry=0,bool flip=false) const;
    std::string history_identity() const { return state_->identity(); }
    std::string graph_identity(int rep_bound) const { return state_->graph_identity(rep_bound); }
    std::array<float,5> training_metadata() const { return state_->training_metadata(); }
    std::string position_json(int hint=-1) const { return state_->position_json(hint); }
    std::string fen() const { return state_->fen(); }
    int size() const { return size_; }
    int width() const { return rule_==Rule::GO?state_->geometry().width:spec().width(size_); }
    int height() const { return rule_==Rule::GO?state_->geometry().height:spec().height(size_); }
    BoardGeometry geometry() const { return rule_==Rule::GO?BoardGeometry{width(),height(),width()}:spec().geometry(size_); }
    const ::BoardHistory& go_history() const { auto p=state_->go_history();if(!p)throw std::runtime_error("Command requires Go");return *p; }
    int area() const { return width()*height(); }
    int canvas_area() const { return canvas_*canvas_; }
    int canvas() const { return canvas_; }
    int planes() const { return spec().planes; }
    int globals() const { return spec().globals; }
    int actions() const { return spec().actions(canvas_); }
    std::vector<int> root_symmetry_representatives(const std::vector<int>& avoid={}) const;
    int local_actions() const { return rule_==Rule::GO?area()+1:spec().actions(size_); }
    int local_action(int a) const { return rule_!=Rule::GO?spec().local_action(a,size_,canvas_):a<0?-1:a==canvas_area()?area():(a/canvas_)*width()+a%canvas_; }
    int native_action(int a) const { return rule_!=Rule::GO?spec().native_action(a,size_,canvas_):a<0?-1:a==area()?canvas_area():(a/width())*canvas_+a%width(); }
    std::vector<int> static_actions() const { if(rule_!=Rule::GO)return spec().static_actions(size_,canvas_);std::vector<int> out;for(int a=0;a<local_actions();++a)out.push_back(native_action(a));return out; }
    double pda_doublings() const { return pda_doublings_; }
    int pda_player() const { return pda_player_; }
    void set_pda(double doublings,int player) {
        if(!std::isfinite(doublings) || doublings<0 || doublings>std::log2(100.0) ||
           (player!=1 && player!=-1 && player!=0) || (doublings!=0 && player==0))
            throw std::runtime_error("Invalid PDA conditioning");
        pda_doublings_=doublings;pda_player_=doublings==0?0:player;
    }
    Rule rule() const { return rule_; }
    int player() const { return state_->player(); }
    int turn() const { return turn_; }
    int last_action() const { return last_action_; }
    Outcome outcome() const { return state_->outcome(); }
    int winner() const { return outcome().winner; }
    Reason reason() const { return outcome().reason; }
    bool finished() const { return state_->finished(); }
    const Board& board() const { return state_->board(); }
    bool legal(int a) const { return state_->legal(a); }
    bool raw_legal(int a) const {return state_->raw_legal(a); }
    bool subtree_bias_key(int action,SubtreeValueBiasKey& key) const {
        return state_->subtree_bias_key(last_action_,action,key);
    }
    std::vector<float> observation() const { return observation(player()); }
    std::vector<float> observation(int perspective,bool forbidden_feature=true) const {
        if(perspective!=1 && perspective!=-1)throw std::runtime_error("Invalid observation perspective");
        return state_->observation(perspective,pda_doublings_==0?0.0:pda_doublings_*(perspective==pda_player_?1:-1),forbidden_feature);
    }
    double terminal_value() const {
        if(!finished())throw std::runtime_error("State is not terminal");
        return outcome().value(player());
    }
    double terminal_no_result_probability() const { if(!finished())throw std::runtime_error("State is not terminal");return outcome().no_result_probability(); }
    double terminal_draw_probability() const { if(!finished())throw std::runtime_error("State is not terminal");return outcome().draw_probability(); }
    void play(int a) {
        if(!legal(a))throw std::runtime_error("Illegal move: "+std::to_string(a));
        state_->play(a);moves_.push_back(a);last_action_=a;++turn_;
    }
};
}
