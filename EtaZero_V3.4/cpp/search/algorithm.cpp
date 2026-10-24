#include "search/algorithm.h"
#include "neuralnet/metadata.h"
namespace etazero {
Evaluation AlphaZeroState::evaluate_with(Evaluator& evaluator,SearchEvaluation options) const {
    auto observation=game_.observation();limit_history_features(observation,game_.canvas(),game_.spec(),options.max_history);
    if(evaluator.uses_selfplay_metadata())return evaluator.evaluate_profile(observation,game_.player(),options.symmetry,options.skip_cache,
        options.policy_temperature,options.randomize,options.optimism,selfplay_metadata(game_.date()));
    return evaluator.evaluate_position(observation,game_.player(),options.symmetry,options.skip_cache,
                                       options.policy_temperature,options.randomize,options.optimism);
}
}
