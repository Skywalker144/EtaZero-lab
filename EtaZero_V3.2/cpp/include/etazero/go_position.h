#pragma once
#include "game.h"
#include "inference.h"
#include "../../external/katago/dataio/sgf.h"
namespace etazero {
// Source samples retain setup stones, player, effective turn and recent history.
Game game_from_go_sample(const Sgf::PositionSample&,int canvas,const RuleOptions&);
Game game_from_go_history(const ::BoardHistory&,int canvas);
inline ::BoardHistoryModes resolve_go_modes(std::string pass_alive,std::string atari,const ModelDescription& model) {
    auto enabled=[](std::string value,bool fallback){for(auto& c:value)c=std::tolower(c);if(value=="auto")return fallback;if(value=="true")return true;if(value=="false")return false;throw std::runtime_error("Go history mode must be true, false or auto");};
    return {enabled(pass_alive,model.prefer_pass_alive),enabled(atari,model.prefer_exclude_atari)};
}
int go_action(Loc loc,int board_width,int canvas);
}
