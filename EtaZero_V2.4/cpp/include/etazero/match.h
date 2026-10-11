#pragma once
#include "opening.h"
#include <iosfwd>

namespace etazero {
struct MatchModel { std::string id, path; uint64_t seed; };
struct MatchTask {
    std::string pair;
    size_t a, b;
    int id, generator, mask;
    uint64_t seed;
    bool generated;
    std::vector<int> moves;
};
using MatchEvaluatorFactory = std::function<std::unique_ptr<InferenceService>(const MatchModel&)>;
using MatchSearchFactory = std::function<std::unique_ptr<GameSearch>(InferenceService&,uint64_t)>;

// Pair identity is only for persistence. Arena openings and games share one pool.
// A single pair retains its opening-generation barrier and per-side statistics.
// Model residency changes happen with no searches in flight (CUDA graph capture).
int run_matches(const Config& config, int size, Rule rule, const std::string& precision,
                const std::vector<MatchModel>& models, std::vector<MatchTask>& tasks,
                int max_models, const MatchEvaluatorFactory& evaluator,
                const MatchSearchFactory& search, const std::function<double(const Game&)>& temperature,
                const std::function<bool()>& stopped, std::ostream& output, bool single_pair = false);
}
