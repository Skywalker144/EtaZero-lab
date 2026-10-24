#include "etazero/game.h"
#include "etazero/symmetry.h"
#include <numeric>
#include <algorithm>
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
    :state_(make_rule_state(rule,size,canvas,options)),options_(options),size_(size),canvas_(canvas),rule_(rule) {
    turn_=state_->initial_turn();
    // SGF/PositionSample dimensions take precedence over optional size hints.
    if(rule_==Rule::GO) {options_.go_width=width();options_.go_height=height();}
}
Game::Game(const Game& other)
    :state_(other.state_->clone()),options_(other.options_),size_(other.size_),canvas_(other.canvas_),
     turn_(other.turn_),last_action_(other.last_action_),rule_(other.rule_),moves_(other.moves_),
     pda_doublings_(other.pda_doublings_),pda_player_(other.pda_player_) {}
Game& Game::operator=(const Game& other) {
    if(this==&other)return *this;
    if(state_ && rule_==other.rule_ && size_==other.size_ && canvas_==other.canvas_)
        state_->reset_from(*other.state_);
    else state_=other.state_->clone();
    options_=other.options_;size_=other.size_;canvas_=other.canvas_;turn_=other.turn_;
    last_action_=other.last_action_;rule_=other.rule_;moves_=other.moves_;
    pda_doublings_=other.pda_doublings_;pda_player_=other.pda_player_;
    return *this;
}
std::vector<int> Game::root_symmetry_representatives(const std::vector<int>& avoid) const {
    auto specialized=state_->root_symmetry_representatives(avoid);if(!specialized.empty())return specialized;
    std::vector<int> result(actions());std::iota(result.begin(),result.end(),0);
    std::vector<std::vector<int>> symmetries;
    for(int symmetry=1;symmetry<spec().symmetries;++symmetry) {
        int transform=spec().symmetry==SymmetryKind::HEX?2*symmetry:spec().symmetry==SymmetryKind::XIANGQI?8+symmetry:symmetry;
        auto spatial=spatial_mapping(size_,transform);
        if(rule_==Rule::XIANGQI)spatial=spatial_mapping(canvas_,transform);
        bool valid=true;
        const auto& b=board();
        for(int y=0;y<height() && valid;++y)for(int x=0;x<width();++x) {
            int index=y*(rule_==Rule::XIANGQI?canvas_:size_)+x,mapped=spatial[index];
            int mx=mapped%(rule_==Rule::XIANGQI?canvas_:size_),my=mapped/(rule_==Rule::XIANGQI?canvas_:size_);
            if(b.cells[y*b.stride+x]!=b.cells[my*b.stride+mx]){valid=false;break;}
        }
        if(!valid)continue;
        auto local_map=policy_mapping(spatial,spec());std::vector<int> mapping(actions());std::iota(mapping.begin(),mapping.end(),0);
        for(int action:static_actions())mapping[action]=rule_==Rule::XIANGQI?local_map[action]:native_action(local_map[local_action(action)]);
        if(!state_->symmetry_history_ok(mapping))continue;
        for(int action:static_actions())if(legal(action)!=legal(mapping[action])){valid=false;break;}
        if(valid)symmetries.push_back(std::move(mapping));
    }
    std::vector<int> order=static_actions();
    auto coordinate=[&](int action){if(rule_==Rule::XIANGQI){auto m=XIANGQI_MOVES[action];return std::pair<int,int>{m[0],m[1]};}return std::pair<int,int>{action%canvas_,action/canvas_};};
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){auto x=coordinate(a),y=coordinate(b);return player()==1?(x.first!=y.first?x.first>y.first:x.second<y.second):(x.first!=y.first?x.first<y.first:x.second>y.second);});
    for(int action:order) {
        if((!avoid.empty()&&avoid[action]>0)||result[action]!=action)continue;
        for(const auto& mapping:symmetries)if(mapping[action]!=action)result[mapping[action]]=action;
    }
    return result;
}

}

namespace etazero {
std::string Game::pattern_key(int action,int symmetry,bool flip) const {
    if(rule_==Rule::GO)return state_->pattern_key(action,symmetry,flip);
    if(action<0 || action>=actions())return {};
    int w=width(),h=height();
    auto transform=[&](int x,int y){
        if(rule_==Rule::XIANGQI){if(flip){x=w-1-x;y=h-1-y;}if(symmetry)x=w-1-x;}
        else if(rule_==Rule::HEX){if(symmetry){x=w-1-x;y=h-1-y;}if(flip)std::swap(x,y);}
        else {if(symmetry&1)y=h-1-y;if(symmetry&2)x=w-1-x;if(symmetry&4)std::swap(x,y);}
        return std::pair<int,int>{x,y};};
    std::vector<int> transformed(w*h,0);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){auto [sx,sy]=transform(x,y);transformed[sy*w+sx]=board().cells[y*board().stride+x]*(flip?-1:1);}
    std::string key="game:"+std::to_string(int(rule_))+":"+std::to_string(w)+":"+std::to_string(h)+":"+std::to_string(player()*(flip?-1:1))+":";
    auto append=[&](int x,int y){auto [sx,sy]=transform(x,y);key+=std::to_string(sx)+","+std::to_string(sy)+":";
        for(int dy=-4;dy<=4;++dy)for(int dx=-4;dx<=4;++dx){int xx=sx+dx,yy=sy+dy;key.push_back(xx>=0&&xx<w&&yy>=0&&yy<h?char(transformed[yy*w+xx]+32):char(127));}};
    if(rule_==Rule::XIANGQI){auto move=XIANGQI_MOVES[action];append(move[0],move[1]);append(move[2],move[3]);}
    else append(action%canvas_,action/canvas_);
    return key;
}
}
