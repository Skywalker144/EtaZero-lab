#pragma once
#include "config.h"
#include "search.h"

namespace etazero {
struct PlayoutAdvantage { double doublings=0; int player=0; };
PlayoutAdvantage sample_playout_advantage(double probability,double max_ratio,std::mt19937_64& rng);
double playout_budget_factor(PlayoutAdvantage advantage,int player);
struct SelfplaySearchConfig {
    int full_visits, cheap_visits;
    double cheap_probability, cheap_target_weight;
    bool clear_before_search, reduce_visits;
    double reduce_threshold;
    int reduce_lookback, reduced_visits_min;
    double reduced_visits_weight;
    int max_playouts=std::numeric_limits<int>::max();
};
struct SelfplaySearchLimits {
    SearchRun search;
    bool cheap_search;
    double target_weight;
};
struct SelfplaySearchContext {
    int hint_action=-1, hint_turn=-1, current_turn=0;
    bool exact_hint=false, hint_fork=false, force_full=false;
};
double cheap_search_probability(const SelfplaySearchConfig& config,const SelfplaySearchContext& context);
SelfplaySearchConfig selfplay_search_config(const Config& config);
// History contains completed root W-L values from one fixed (Black) perspective,
// including cheap searches, excluding the opening prefix and the current turn.
SelfplaySearchLimits selfplay_search_limits(const SelfplaySearchConfig& config,
                                          const std::vector<double>& history, bool cheap_selected,
                                          PlayoutAdvantage advantage={},int player=1,
                                          SelfplaySearchContext context={});
}
