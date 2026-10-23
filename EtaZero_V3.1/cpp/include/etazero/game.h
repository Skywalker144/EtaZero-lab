#pragma once
#include "rule_state.h"
#include <cmath>
namespace etazero {
// Common state ownership and NN conditioning. Rules live in RuleState implementations.
class Game {
    std::unique_ptr<RuleState> state_;
    RuleOptions options_;
    int size_,canvas_,turn_=0,last_action_=-1;
    Rule rule_;
    double pda_doublings_=0;
    int pda_player_=0;
public:
    explicit Game(int size,int canvas,Rule rule,const RuleOptions& options={});
    Game(const Game& other);
    Game& operator=(const Game& other);
    Game(Game&&)=default;
    Game& operator=(Game&&)=default;
    GameSpec spec() const { return game_spec(rule_); }
    const RuleOptions& options() const { return options_; }
    std::string history_identity() const { return state_->identity(); }
    std::string graph_identity(int rep_bound) const { return state_->graph_identity(rep_bound); }
    std::string fen() const { return state_->fen(); }
    int size() const { return size_; }
    int width() const { return spec().width(size_); }
    int height() const { return spec().height(size_); }
    BoardGeometry geometry() const { return spec().geometry(size_); }
    int area() const { return spec().area(size_); }
    int canvas_area() const { return canvas_*canvas_; }
    int canvas() const { return canvas_; }
    int planes() const { return spec().planes; }
    int globals() const { return spec().globals; }
    int actions() const { return spec().actions(canvas_); }
    int local_actions() const { return spec().actions(size_); }
    int local_action(int a) const { return spec().local_action(a,size_,canvas_); }
    int native_action(int a) const { return spec().native_action(a,size_,canvas_); }
    std::vector<int> static_actions() const { return spec().static_actions(size_,canvas_); }
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
    double terminal_draw_probability() const { if(!finished())throw std::runtime_error("State is not terminal");return outcome().draw_probability(); }
    void play(int a) {
        if(!legal(a))throw std::runtime_error("Illegal move: "+std::to_string(a));
        state_->play(a);last_action_=a;++turn_;
    }
};
}
