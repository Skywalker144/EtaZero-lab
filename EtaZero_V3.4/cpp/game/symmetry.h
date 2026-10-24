#pragma once
#include "game/game_spec.h"
#include <stdexcept>
#include <vector>

namespace etazero {
// Gomoku and Hex can share a model layout; its declared Hex indicator selects
// the per-row group. Other models have a fixed group, supplied explicitly.
inline GameSpec input_spec(const std::vector<float>& obs,GameSpec model) {
    if(obs.size()<static_cast<size_t>(model.globals))throw std::runtime_error("Invalid observation shape");
    if(model.kind==GameKind::GOMOKU || model.kind==GameKind::HEX)
        return game_spec(obs[obs.size()-model.globals+HEX_GLOBAL]!=0?GameKind::HEX:GameKind::GOMOKU);
    return model;
}
inline int input_symmetry(const std::vector<float>& obs,GameSpec model,int symmetry) {
    auto spec=input_spec(obs,model);
    if(symmetry<0 || symmetry>=spec.symmetries)throw std::runtime_error("Symmetry outside game group");
    if(spec.symmetry==SymmetryKind::XIANGQI) {
        bool red=obs[obs.size()-spec.globals]!=0;
        return 8+symmetry+(!red?2:0);
    }
    if(spec.symmetry==SymmetryKind::HEX) {
        bool white=obs[obs.size()-spec.globals+HEX_WHITE_GLOBAL]!=0;
        return 2*symmetry+(white?4:0);
    }
    return symmetry;
}
inline int symmetry_count(const std::vector<float>& obs,GameSpec model) { return input_spec(obs,model).symmetries; }
inline std::vector<int> spatial_mapping(int n,int s) {
    std::vector<int> map(n*n);
    for(int a=0;a<n*n;++a) {
        int x=a%n,y=a/n,tx=x,ty=y;
        if(s>=8) {
            if(x<9&&y<10){if((s-8)&2){tx=8-x;ty=9-y;}if((s-8)&1)tx=8-tx;}
        } else switch(s){case 1:tx=y;ty=n-1-x;break;case 2:tx=n-1-x;ty=n-1-y;break;
            case 3:tx=n-1-y;ty=x;break;case 4:tx=y;ty=x;break;case 5:tx=n-1-x;break;
            case 6:tx=n-1-y;ty=n-1-x;break;case 7:ty=n-1-y;break;}
        map[a]=ty*n+tx;
    }
    return map;
}
inline std::vector<int> policy_mapping(const std::vector<int>& spatial,GameSpec spec) {
    if(spec.action_layout==ActionLayout::POINTS)return spatial;
    if(spec.action_layout==ActionLayout::POINTS_PASS){auto map=spatial;map.push_back(spatial.size());return map;}
    int n=static_cast<int>(std::sqrt(spatial.size()));std::vector<int> map(spec.fixed_actions);
    for(int a=0;a<spec.fixed_actions;++a){auto m=XIANGQI_MOVES[a];int f=spatial[m[1]*n+m[0]],t=spatial[m[3]*n+m[2]];
        map[a]=xiangqi_action(f%n,f/n,t%n,t/n);if(map[a]<0)throw std::runtime_error("Invalid Xiangqi symmetry");}
    return map;
}
}
