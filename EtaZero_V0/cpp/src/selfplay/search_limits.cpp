#include "etazero/search_limits.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace etazero {
namespace {
void validate(const SelfplaySearchConfig& c) {
    if(c.full_visits<2 || c.cheap_visits<2 || c.cheap_visits>c.full_visits ||
       !std::isfinite(c.cheap_probability) || c.cheap_probability<0 || c.cheap_probability>1 ||
       !std::isfinite(c.cheap_target_weight) || c.cheap_target_weight<0 || c.cheap_target_weight>1 ||
       !std::isfinite(c.reduce_threshold) || c.reduce_threshold<0 || c.reduce_threshold>0.999999 ||
       c.reduce_lookback<1 || c.reduce_lookback>1000 || c.reduced_visits_min<2 || c.reduced_visits_min>c.full_visits ||
       !std::isfinite(c.reduced_visits_weight) || c.reduced_visits_weight<0 || c.reduced_visits_weight>1)
        throw std::runtime_error("Invalid selfplay PCR / Reduce Visits settings");
}
}
SelfplaySearchConfig selfplay_search_config(const Config& c) {
    int simulations=c.integer("search.simulations");
    if(simulations<1 || simulations==std::numeric_limits<int>::max())
        throw std::runtime_error("Invalid full selfplay search budget");
    SelfplaySearchConfig result{simulations+1,c.integer("search.cheap_search_visits"),
        c.number("search.cheap_search_probability"),c.number("search.cheap_search_target_weight"),
        c.boolean("search.clear_before_search"),c.boolean("search.reduce_visits"),
        c.number("search.reduce_visits_threshold"),c.integer("search.reduce_visits_threshold_lookback"),
        c.integer("search.reduced_visits_min"),c.number("search.reduced_visits_weight")};
    validate(result);return result;
}
SelfplaySearchLimits selfplay_search_limits(const SelfplaySearchConfig& c,
                                          const std::vector<double>& history,bool cheap_selected) {
    validate(c);
    SelfplaySearchLimits limits{{c.full_visits,true,c.clear_before_search,false},cheap_selected,1};
    // KataGo play.cpp: PCR and Reduce Visits are mutually exclusive branches.
    if(cheap_selected) {
        limits.search.max_visits=c.cheap_visits;
        limits.target_weight=c.cheap_target_weight;
        if(c.cheap_target_weight==0) {
            limits.search.clear_before_search=false;
            limits.search.remove_root_noise=true;
        }
    } else if(c.reduce_visits && history.size()>=static_cast<size_t>(c.reduce_lookback)) {
        double minimum=1e20,maximum=-1e20;
        for(int j=0;j<c.reduce_lookback;++j) {
            double q=history[history.size()-1-j];
            if(!std::isfinite(q) || q < -1.000001 || q > 1.000001)
                throw std::runtime_error("Invalid Reduce Visits historical W-L value");
            minimum=std::min(minimum,q);maximum=std::max(maximum,q);
        }
        double extreme=std::min(1.0,std::max(minimum,-maximum));
        if(extreme>c.reduce_threshold) {
            double proportion=(extreme-c.reduce_threshold)/(1-c.reduce_threshold);
            double reduction=proportion*proportion;
            limits.search.max_visits=std::max(c.reduced_visits_min,static_cast<int>(std::round(
                c.full_visits+reduction*(c.reduced_visits_min-c.full_visits))));
            limits.target_weight=1+reduction*(c.reduced_visits_weight-1);
            // Reduced full searches retain full-search tree and exploration settings,
            // even when their target weight reaches zero.
        }
    }
    return limits;
}
}
