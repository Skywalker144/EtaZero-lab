#pragma once
#include "core/config.h"
#include "search/search_common.h"
namespace etazero {
SearchSettings settings(const Config&,const std::string& mode);
double move_temperature(const Config&,const std::string& mode,const Game&);
std::unique_ptr<GameSearch> search_for(InferenceService&,SearchSettings,uint64_t seed,const Config&,const std::string& mode);
}
