#include "etazero/search_limits.h"
#include <algorithm>
#include <cmath>

namespace etazero {
namespace {
void validate(const SelfplaySearchConfig& c) {
    if(c.max_playouts<0 || c.full_visits<2 ||
       !std::isfinite(c.cheap_probability) || c.cheap_probability<0 || c.cheap_probability>1 ||
       (c.cheap_probability>0 && (c.cheap_visits<2 || c.cheap_visits>c.full_visits ||
        !std::isfinite(c.cheap_target_weight) || c.cheap_target_weight<0 || c.cheap_target_weight>1)) ||
       (c.reduce_visits && (!std::isfinite(c.reduce_threshold) || c.reduce_threshold<0 || c.reduce_threshold>0.999999 ||
        c.reduce_lookback<1 || c.reduce_lookback>1000 || c.reduced_visits_min<2 || c.reduced_visits_min>c.full_visits ||
        !std::isfinite(c.reduced_visits_weight) || c.reduced_visits_weight<0 || c.reduced_visits_weight>1)))
        throw std::runtime_error("Invalid selfplay PCR / Reduce Visits settings");
}
}
PlayoutAdvantage sample_playout_advantage(double probability,double max_ratio,std::mt19937_64& rng) {
    if(!std::isfinite(probability) || probability<0 || probability>1 ||
       !std::isfinite(max_ratio) || max_ratio<1 || max_ratio>100)throw std::runtime_error("Invalid PDA sampling parameters");
    if(probability<=0 || !std::bernoulli_distribution(probability)(rng))return {};
    double d=std::uniform_real_distribution<double>(0,std::log2(max_ratio))(rng);
    return {d,std::bernoulli_distribution(.5)(rng)?-1:1};
}
double playout_budget_factor(PlayoutAdvantage a,int player) {
    if(!std::isfinite(a.doublings) || a.doublings<0 || a.doublings>std::log2(100.0) ||
       (a.player!=0 && a.player!=1 && a.player!=-1) || (a.doublings!=0 && a.player==0) ||
       (player!=1 && player!=-1))throw std::runtime_error("Invalid PDA budget condition");
    if(a.doublings==0)return 1;
    double ratio=std::pow(2.0,a.doublings);return 2*(player==a.player?ratio:1)/(1+ratio);
}
SelfplaySearchConfig selfplay_search_config(const Config& c) {
    SelfplaySearchConfig result{c.integer("search.full_search_visits"),c.integer("search.cheap_search_visits"),
        c.number("search.cheap_search_probs"),c.number("search.cheap_search_target_weight"),
        c.boolean("search.clear_before_search"),c.boolean("reduce_visits.reduce_visits"),
        c.number("reduce_visits.reduce_visits_threshold"),c.integer("reduce_visits.reduce_visits_threshold_lookback"),
        c.integer("reduce_visits.reduced_visits_min"),c.number("reduce_visits.reduced_visits_weight")};
    result.max_playouts=c.integer("search.max_playouts");validate(result);return result;
}
double cheap_search_probability(const SelfplaySearchConfig& c,const SelfplaySearchContext& context) {
    if(context.force_full || context.exact_hint)return 0;
    double probability=c.cheap_probability;
    if((context.hint_action>=0 || context.hint_fork) && context.hint_turn+6>context.current_turn)probability*=.5;
    return probability;
}
SelfplaySearchLimits selfplay_search_limits(const SelfplaySearchConfig& c,
                                          const std::vector<double>& history,bool cheap_selected,PlayoutAdvantage advantage,int player,SelfplaySearchContext context) {
    bool hinted=!context.force_full && context.exact_hint;
    if(context.force_full || hinted)cheap_selected=false;
    if(hinted && context.hint_action<0)throw std::runtime_error("Exact hint lacks action");
    validate(c);
    if((cheap_selected && c.cheap_visits>c.max_playouts) ||
       (!cheap_selected && !hinted && c.reduce_visits && c.reduced_visits_min>c.max_playouts))
        throw std::runtime_error("PCR / Reduce Visits cap exceeds explicit playout ceiling");
    double reduction=0;SelfplaySearchLimits limits{};limits.search.max_visits=c.full_visits;limits.search.training=true;
    limits.search.clear_before_search=c.clear_before_search;limits.cheap_search=cheap_selected;limits.target_weight=1;
    // KataGo play.cpp: PCR and Reduce Visits are mutually exclusive branches.
    if(hinted) {
        if(c.full_visits>std::numeric_limits<int>::max()/4 ||
           (c.max_playouts!=std::numeric_limits<int>::max() && c.max_playouts>std::numeric_limits<int>::max()/4))
            throw std::runtime_error("Hint fourfold budget exceeds int32 ceiling");
        limits.search.max_visits=c.full_visits*4;limits.search.clear_before_search=true;
        limits.search.hint_action=context.hint_action;
    } else if(cheap_selected) {
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
            reduction=proportion*proportion;
            limits.search.max_visits=std::max(c.reduced_visits_min,static_cast<int>(std::round(
                c.full_visits+reduction*(c.reduced_visits_min-c.full_visits))));
            limits.target_weight=1+reduction*(c.reduced_visits_weight-1);
            // Reduced full searches retain full-search tree and exploration settings,
            // even when their target weight reaches zero.
        }
    }
    limits.search.max_playouts=hinted && c.max_playouts!=std::numeric_limits<int>::max()?c.max_playouts*4:c.max_playouts;
    if(cheap_selected)limits.search.max_playouts=std::min(c.max_playouts,c.cheap_visits);
    else if(reduction>0 && c.max_playouts!=std::numeric_limits<int>::max()) {
        if(c.max_playouts<c.reduced_visits_min)throw std::runtime_error("Reduced minimum exceeds playout cap");
        double r=reduction;
        limits.search.max_playouts=static_cast<int>(std::round(c.max_playouts+r*(double(c.reduced_visits_min)-c.max_playouts)));
    }
    double factor=playout_budget_factor(advantage,player);
    if(advantage.doublings!=0) {
        limits.search.clear_before_search=true;
        double visits=std::round(limits.search.max_visits*factor);
        if(visits<5 || visits>std::numeric_limits<int>::max())throw std::runtime_error("PDA budget outside [5, int32 max]");
        limits.search.max_visits=static_cast<int>(visits);
        // int32 max denotes the inherited unlimited ceiling in EtaZero.
        if(c.max_playouts!=std::numeric_limits<int>::max() || cheap_selected || limits.search.max_playouts!=c.max_playouts) {
            double playouts=std::round(limits.search.max_playouts*factor);
            if(playouts<5 || playouts>std::numeric_limits<int>::max())throw std::runtime_error("PDA playout budget outside [5, int32 max]");
            limits.search.max_playouts=static_cast<int>(playouts);
        }
    }
    return limits;
}
}
