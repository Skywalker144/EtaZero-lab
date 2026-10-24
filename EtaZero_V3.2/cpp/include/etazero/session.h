#pragma once
#include "opening.h"
#include <nlohmann/json.hpp>
namespace etazero {
using Json=nlohmann::json;
// All protocols use the same board, history, search and model. The reader remains
// responsive while one worker owns board mutation and the entire search lifetime.
int engine_session(std::unique_ptr<GameSearch>,InferenceService&,const Config&,const Json& models,
                   const std::function<std::unique_ptr<GameSearch>(const Json&)>& make_search,uint64_t,const std::string& protocol,
                   const std::function<Json(const Game&)>& raw_nn,
                   const std::function<double(const Game&,const Config&)>& temperature,
                   const std::function<bool()>& interrupted);
int source_analysis_session(InferenceService&,const Config&,const Json& model_info,
    const std::function<std::unique_ptr<GameSearch>(const Json&)>& make_search,
    const std::function<bool()>& interrupted);
Json analysis_json(const SearchResult&,const Game&,uint64_t requests,uint64_t batches,bool final);
}
