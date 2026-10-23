// KataGo searchtimehelpers.cpp, adapted to pure WDL and multigame board geometry.
#include "etazero/time_budget.h"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace etazero {
SearchTimeBudget::SearchTimeBudget(const SearchSettings& settings,const SearchRun& run,double carried)
    :settings_(settings),area_(run.board_area),turn_(run.turn),occupied_(run.occupied_points<0?run.turn:run.occupied_points),
     pondering_(run.pondering),carried_(carried),cap_(run.max_time<0?(run.pondering?settings.max_time_pondering:settings.max_time):run.max_time),limit_(cap_) {
    if(run.time_controls && !run.time_controls->isEffectivelyUnlimitedTime() && !pondering_)clock_=*run.time_controls;
}
double SearchTimeBudget::estimate_visits_left(int64_t visits,double elapsed,double limit,double carried,int threads) {
    if(visits<=1 || carried+elapsed<.1)return 1e30;
    return std::ceil((limit-elapsed)/(carried+elapsed)*visits+threads-1);
}
void SearchTimeBudget::update(double elapsed,int64_t root_visits,const std::vector<RootChildStats>& children,
    const std::vector<double>& policy,ValueStats parent,double temperature,const SearchResult* override) {
    if(pondering_ || (last_update_>=0 && elapsed<last_update_+.1))return;
    last_update_=elapsed;
    if(!clock_)return; // Source only uses the visits/time bound with an actual clock.
    const auto& tc=*clock_;const auto& s=settings_;
    double minimum,recommended,maximum;tc.getTime(area_,occupied_,s.lag_buffer,minimum,recommended,maximum);
    recommended*=s.overallocate_time_factor;
    if(s.midgame_time_factor!=1) {
        double scale=area_/361.0,turn=std::max(0,turn_),peak=s.midgame_turn_peak_time*scale;
        double mid=turn<peak?turn/peak:(s.endgame_turn_time_decay>0?std::exp(-(turn-peak)/(s.endgame_turn_time_decay*scale)):(turn==peak?1:0));
        mid=std::clamp(mid,0.0,1.0);recommended*=1+mid*(s.midgame_time_factor-1);
    }
    std::vector<double> selections;
    if(override)selections=override->move_policy;
    else if(!children.empty())selections=root_selection_weights(children,s,s.use_lcb,parent);
    auto visits=[&](size_t i){return override?override->visits[i]:children[i].visits;};
    if(s.obvious_moves_time_factor<1) {
        double entropy=0,surprise=0;
        for(double p:policy)if(p>0)entropy-=p*std::log(p);
        for(size_t i=0;i<selections.size();++i)if(selections[i]>0)surprise+=selections[i]*std::log(selections[i]/std::max(1e-100,(override?override->search_policy[i]:children[i].prior)));
        double obvious=std::min(std::exp(-entropy/s.obvious_moves_policy_entropy_tolerance),std::exp(-std::max(0.0,surprise)/s.obvious_moves_policy_surprise_tolerance));
        recommended*=1+obvious*(s.obvious_moves_time_factor-1);
    }
    if(recommended>1e-20) {
        double fraction=(recommended-carried_)/recommended;
        recommended*=std::min(1.0,std::log1p(std::exp(fraction*6))/6);
    }
    recommended=std::min(maximum,tc.roundUpTimeLimitIfNeeded(s.lag_buffer,elapsed,recommended));
    if(s.futile_visits_threshold>0) {
        double remaining=estimate_visits_left(root_visits,elapsed,recommended,carried_,s.threads);
        if(remaining<s.futile_visits_threshold*root_visits && !selections.empty()) {
            size_t best=std::max_element(selections.begin(),selections.end())-selections.begin(),most=0;
            for(size_t i=1;i<selections.size();++i)if(visits(i)>visits(most))most=i;
            if(best==most) {
                double required=s.futile_visits_threshold*visits(most);
                if(temperature>=1e-3)required=std::min(required,visits(most)*std::pow(.01,temperature));
                bool alternative=false;for(size_t i=0;i<selections.size();++i)if(i!=best && visits(i)+remaining>=required)alternative=true;
                if(!alternative)recommended=elapsed*(1-1e-10);
            }
        }
    }
    recommended=std::min(maximum,tc.roundUpTimeLimitIfNeeded(s.lag_buffer,elapsed,recommended));
    recommended=std::min(maximum,std::max(minimum,recommended));
    limit_.store(std::min(cap_,recommended),std::memory_order_release);
    visits_left_.store(estimate_visits_left(root_visits,elapsed,std::min(cap_,recommended),carried_,s.threads),std::memory_order_release);
}
}
