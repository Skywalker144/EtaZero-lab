#pragma once
#include "etazero/schema.h"
#include "geometry.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace etazero {
struct GameSpec : GameLayout {
    constexpr explicit GameSpec(GameLayout layout) : GameLayout(layout) {}
    int actions(int canvas) const { return fixed_actions ? fixed_actions : canvas*canvas+(action_layout==ActionLayout::POINTS_PASS); }
    int width(int size) const { return board_width ? board_width : size; }
    int height(int size) const { return board_height ? board_height : size; }
    BoardGeometry geometry(int size) const { return {width(size),height(size),size}; }
    int area(int size) const { return width(size)*height(size); }
    int local_action(int action,int size,int canvas) const {
        if(action<0 || action_layout==ActionLayout::MOVES)return action;
        return action==canvas*canvas ? size*size : action/canvas*size+action%canvas;
    }
    int native_action(int action,int size,int canvas) const {
        if(action<0 || action>=actions(size))throw std::runtime_error("Action outside vocabulary");
        if(action_layout==ActionLayout::MOVES)return action;
        return action==size*size ? canvas*canvas : action/size*canvas+action%size;
    }
    std::vector<int> static_actions(int size,int canvas) const {
        std::vector<int> result;
        for(int a=0;a<actions(canvas);++a)
            if(action_layout==ActionLayout::MOVES || a==canvas*canvas || (a/canvas<size && a%canvas<width(size)))result.push_back(a);
        return result;
    }
    void validate_dimensions(int size,int canvas) const {
        if(size<(kind==GameKind::GO?2:5) || size>canvas || canvas>25 || (board_height && (size!=board_height || canvas!=board_height)))
            throw std::runtime_error(std::string("Invalid board/canvas dimensions for ")+name);
    }
};
inline constexpr GameSpec game_spec(GameKind kind) { return GameSpec(GAME_LAYOUTS[static_cast<int>(kind)]); }
inline GameSpec game_spec(Rule rule) { return game_spec(RULE_GAME_KINDS[static_cast<int>(rule)]); }
inline GameSpec game_spec(const std::string& name) {
    for(const auto& layout:GAME_LAYOUTS)if(name==layout.name)return GameSpec(layout);
    throw std::runtime_error("Unknown game kind: "+name);
}
inline bool same_layout(GameSpec a,GameSpec b) {
    return a.planes==b.planes && a.globals==b.globals && a.action_layout==b.action_layout && a.fixed_actions==b.fixed_actions;
}
inline int xiangqi_action(int x,int y,int u,int v) {
    static const auto table=[] {
        std::vector<int> result(8100,-1);
        for(int a=0;a<game_spec(GameKind::XIANGQI).fixed_actions;++a){auto m=XIANGQI_MOVES[a];result[(m[1]*9+m[0])*90+m[3]*9+m[2]]=a;}
        return result;
    }();
    if(x<0||x>=9||u<0||u>=9||y<0||y>=10||v<0||v>=10)return -1;
    return table[(y*9+x)*90+v*9+u];
}
}
