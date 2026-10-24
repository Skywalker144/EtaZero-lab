#include "search/search_common.h"
#include <algorithm>
#include <cmath>
namespace etazero {
EffectiveSearchLimits effective_search_limits(const SearchSettings& settings,const SearchRun& run) {
    return {run.max_visits ? run.max_visits : (run.pondering ? settings.max_visits_pondering : settings.max_visits),
            run.max_playouts < 0 ? (run.pondering ? settings.max_playouts_pondering : settings.max_playouts) : run.max_playouts,
            run.max_time < 0 ? (run.pondering ? settings.max_time_pondering : settings.max_time) : run.max_time};
}
int active_search_threads(const SearchSettings& settings,int64_t playouts) {
    return settings.min_playouts_per_thread > 0 ? std::max(1,static_cast<int>(std::min(double(settings.threads),
        std::floor(playouts/settings.min_playouts_per_thread)))) : settings.threads;
}
SearchSettings gumbel_search_settings(SearchSettings s) {
    if(s.graph_search||s.reuse_tree)throw std::runtime_error("Gumbel requires graph_search=false and reuse_tree=false");
    // These are PUCT root mechanisms. Non-root PUCT and value aggregation remain configurable.
    s.noise_fraction=0;s.forced_playouts=0;s.root_policy_temperature=s.root_policy_temperature_early=1;
    return s;
}

}
