#pragma once
#include "opening.h"
#include <iosfwd>

namespace etazero {
// Values use absolute Black/first-player W-L; history begins after the opening.
bool should_resign(const std::vector<double>& history,int player,int area,double threshold,int consecutive);
struct MatchModel { std::string id, path; uint64_t seed; };
struct MatchTask {
    std::string pair;
    size_t a, b;
    int id, generator;
    uint64_t seed;
    bool generated;
    std::vector<int> moves;
};
using MatchEventSink = std::function<void(const std::string&)>;
using MatchEvaluatorFactory = std::function<std::unique_ptr<InferenceService>(const MatchModel&,const MatchEventSink&)>;
using MatchSearchFactory = std::function<std::unique_ptr<GameSearch>(InferenceService&,uint64_t)>;

// Pair identity is only for persistence. Arena openings and games share one pool.
// Each game has an independently initialized opening and fixed assigned color.
// Model residency changes happen with no searches in flight (CUDA graph capture).
int run_matches(const Config& config, int size, Rule rule, const std::string& precision,
                const std::vector<MatchModel>& models, std::vector<MatchTask>& tasks,
                int max_models, const MatchEvaluatorFactory& evaluator,
                const MatchSearchFactory& search, const std::function<double(const Game&)>& temperature,
                const std::function<bool()>& stopped, std::ostream& output, bool single_pair = false,
                int model_load_workers = 1);
}
