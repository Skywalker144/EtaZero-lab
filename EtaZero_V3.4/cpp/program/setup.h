#pragma once
#include "core/config.h"
#include "search/search_common.h"
namespace etazero {
int playout_advantage_player(std::string,bool allow_unspecified=true);
void set_playout_advantage(Game&,double,int);
void set_playout_advantage(Game&,const Config&,const std::string& mode);
SearchSettings settings(const Config&,const std::string& mode);
double move_temperature(const Config&,const std::string& mode,const Game&);
std::unique_ptr<GameSearch> search_for(InferenceService&,SearchSettings,uint64_t seed,const Config&,const std::string& mode);
}
