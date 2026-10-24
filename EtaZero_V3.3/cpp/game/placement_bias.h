#pragma once
#include "game/rules.h"
#include "search/subtree_value_bias_key.h"

namespace etazero {
// Gomoku/Hex share a three-color placement pattern, without atari or ko.
inline bool placement_bias_key(const Board& board,int canvas,int player,int previous,int action,SubtreeValueBiasKey& key) {
    if(previous<0)return false;
    key={};int x=action%canvas,y=action/canvas;
    for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
        int xx=x+dx,yy=y+dy,color=3;
        if(xx>=0 && yy>=0 && xx<board.width && yy<board.height) {
            int cell=board.cells[yy*board.stride+xx];color=cell==0?0:cell==1?1:2;
        }
        key.pattern=(key.pattern<<2)|static_cast<uint64_t>(color);
    }
    // Absolute board locations, independent of the padded NN canvas.
    auto location=[&](int a){return (a/canvas+1)*(board.width+1)+a%canvas+1;};
    key.moves=static_cast<uint64_t>(location(previous)) | (static_cast<uint64_t>(location(action))<<16) |
              (static_cast<uint64_t>(player==1?1:2)<<32);
    return true;
}
}
