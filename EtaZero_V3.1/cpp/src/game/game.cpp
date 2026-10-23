#include "etazero/game.h"
namespace etazero {
std::unique_ptr<RuleState> make_gomoku_state(Rule,int,int);
std::unique_ptr<RuleState> make_hex_state(int,int);
std::unique_ptr<RuleState> make_go_state(int,int,const RuleOptions&);
std::unique_ptr<RuleState> make_xiangqi_state(int,int,const RuleOptions&);
std::unique_ptr<RuleState> make_rule_state(Rule rule,int size,int canvas,const RuleOptions& options) {
    auto spec=game_spec(rule);spec.validate_dimensions(size,canvas);
    switch(spec.kind) {
        case GameKind::GOMOKU:return make_gomoku_state(rule,size,canvas);
        case GameKind::HEX:return make_hex_state(size,canvas);
        case GameKind::GO:return make_go_state(size,canvas,options);
        case GameKind::XIANGQI:return make_xiangqi_state(size,canvas,options);
    }
    throw std::runtime_error("Unknown game kind");
}
Game::Game(int size,int canvas,Rule rule,const RuleOptions& options)
    :state_(make_rule_state(rule,size,canvas,options)),options_(options),size_(size),canvas_(canvas),rule_(rule) {}
Game::Game(const Game& other)
    :state_(other.state_->clone()),options_(other.options_),size_(other.size_),canvas_(other.canvas_),
     turn_(other.turn_),last_action_(other.last_action_),rule_(other.rule_),
     pda_doublings_(other.pda_doublings_),pda_player_(other.pda_player_) {}
Game& Game::operator=(const Game& other) {
    if(this==&other)return *this;
    if(state_ && rule_==other.rule_ && size_==other.size_ && canvas_==other.canvas_)
        state_->reset_from(*other.state_);
    else state_=other.state_->clone();
    options_=other.options_;size_=other.size_;canvas_=other.canvas_;turn_=other.turn_;
    last_action_=other.last_action_;rule_=other.rule_;
    pda_doublings_=other.pda_doublings_;pda_player_=other.pda_player_;
    return *this;
}
}
