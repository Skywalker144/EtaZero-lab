#pragma once
#include "game_spec.h"
#include <algorithm>
#include <vector>

namespace etazero {
// NN-only prefix restriction. RuleState, repetition and move legality retain
// their full history. Same feature semantics as learner random-history masking.
inline void limit_history_features(std::vector<float>& obs,int canvas,GameSpec spec,int maximum) {
    if(maximum<0)return;
    const int area=canvas*canvas;
    if(spec.kind==GameKind::GO) {
        for(int i=std::min(5,maximum);i<5;++i) {
            std::fill_n(obs.begin()+(9+i)*area,area,0);
            obs[spec.planes*area+i]=0;
        }
        if(maximum<1)std::copy_n(obs.begin()+14*area,area,obs.begin()+15*area);
        if(maximum<2)std::copy_n(obs.begin()+15*area,area,obs.begin()+16*area);
    } else if(spec.kind==GameKind::XIANGQI) {
        int retained=std::min(8,maximum+1);
        std::fill(obs.begin()+(1+14*retained)*area,obs.begin()+spec.planes*area,0);
        obs[spec.planes*area+5]=std::min(obs[spec.planes*area+5],retained/8.0f);
    }
}
}
