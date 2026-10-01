#pragma once
#include "config.h"
#include "search.h"

namespace etazero {
struct SelfplaySearchConfig {
    int full_visits, cheap_visits;
    double cheap_probability, cheap_target_weight;
    bool clear_before_search, reduce_visits;
    double reduce_threshold;
    int reduce_lookback, reduced_visits_min;
    double reduced_visits_weight;
};
struct SelfplaySearchLimits {
    SearchRun search;
    bool cheap_search;
    double target_weight;
};
SelfplaySearchConfig selfplay_search_config(const Config& config);
// History contains completed root W-L values from one fixed (Black) perspective,
// including cheap searches, excluding the opening prefix and the current turn.
SelfplaySearchLimits selfplay_search_limits(const SelfplaySearchConfig& config,
                                          const std::vector<double>& history, bool cheap_selected);
}
