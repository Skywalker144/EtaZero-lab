#include "search/gumbel.h"
#include <algorithm>
#include <numeric>
namespace etazero {
void normalize_selection_weights(std::vector<double>& weights) {
    double mass=std::accumulate(weights.begin(),weights.end(),0.0);
    if(!std::isfinite(mass) || mass<=0)throw std::runtime_error("Empty selection policy");
    for(auto& x:weights)x/=mass;
}
// KataGo searchhelpers.cpp: half uniform, half positive centered clipped log-policy.
std::vector<double> noise_alpha_distribution(const std::vector<double>& policy) {
    if(policy.empty())throw std::runtime_error("Empty noise policy");
    std::vector<double> alpha;double mean=0;
    for(auto p:policy) {
        if(!std::isfinite(p)||p<0)throw std::runtime_error("Invalid noise policy");
        alpha.push_back(std::log(std::min(0.01,p)+1e-20));mean+=alpha.back();
    }
    mean/=policy.size();double mass=0;
    for(auto& a:alpha){a=std::max(0.0,a-mean);mass+=a;}
    for(auto& a:alpha)a=mass>0?0.5*(1.0/policy.size()+a/mass):1.0/policy.size();
    return alpha;
}
double fpu_value(double nn_value,double parent_value,double visited_mass,double power,double reduction,bool by_visited_policy,double nn_weight) {
    double mass=std::clamp(visited_mass,0.0,1.0),mix=by_visited_policy?std::pow(mass,power):1-nn_weight;
    return mix*parent_value+(1-mix)*nn_value-reduction*std::sqrt(mass);
}
std::vector<double> root_selection_weights(const std::vector<RootChildStats>& children,
                                         const SearchSettings& s,bool use_lcb,ValueStats parent,bool normalize_output,
                                         std::vector<RootSelectionDiagnostic>* diagnostics) {
    if(diagnostics)diagnostics->assign(children.size(),{});
    std::vector<double> weights;double total=0,best_goodness=-1e30;size_t stable=0;
    for(size_t i=0;i<children.size();++i) {
        const auto& c=children[i];weights.push_back(c.weight);total+=c.weight;
        double goodness=c.weight*std::max(0.0,double(c.visits)-1)/std::max(1.0,double(c.visits))+2*c.prior;
        if(c.visits>0 && goodness>best_goodness){best_goodness=goodness;stable=i;}
    }
    if(total<=0)throw std::runtime_error("No completed root edges");
    const auto& best=children[stable];
    double explore=explore_scaling(total,parent,s);
    if(s.policy_target_pruning) {
        double best_score=best.value_sum/best.weight+explore*best.prior/(best.weight+1);
        for(size_t i=0;i<children.size();++i) {
            const auto& c=children[i];if(i==stable || c.visits<=0 || c.weight<=0)continue;
            double gap=best_score-c.value_sum/c.weight;
            if(gap>0)weights[i]=std::min(c.weight,std::max(0.0,explore*c.prior/gap-1));
            // Source rounds every non-stable edge, even when inverse PUCT retains its weight.
            weights[i]=std::ceil(weights[i]);
        }
    }
    if(use_lcb) {
        std::vector<double> radius(children.size(),2*s.lcb_stdevs),lcb(children.size(),-2*s.lcb_stdevs);
        int candidate=-1;
        for(size_t i=0;i<children.size();++i) {
            const auto& c=children[i];
            if(c.visits>0 && c.weight>0 && c.weight_sq>0) {
                double n=c.weight,mean=c.value_sum/n,ess=n*n/c.weight_sq,prior=n/(ess*ess*ess);
                double variance=std::max(1e-8,c.value_sq_sum/n-mean*mean)+prior/(n+prior);
                ess=(n+prior)*(n+prior)/(c.weight_sq+prior*prior);
                radius[i]=s.lcb_stdevs*std::sqrt(variance/ess);lcb[i]=mean-radius[i];
                if(diagnostics)(*diagnostics)[i]={0,lcb[i],radius[i],true};
            }
            if(weights[i]>0 && weights[i]>=s.min_lcb_visit_prop*best.weight &&
               (candidate<0 || lcb[i]>lcb[candidate]))candidate=i;
        }
        if(candidate<0)throw std::runtime_error("No eligible LCB move");
        double adjusted=weights[candidate];
        for(size_t i=0;i<children.size();++i) {
            double excess=lcb[candidate]-lcb[i];
            if(int(i)==candidate || excess<0)continue;
            double factor=(radius[i]+excess)/(radius[i]+0.20*excess);
            adjusted=std::max(adjusted,factor*factor*weights[i]);
        }
        if(s.use_non_buggy_lcb||candidate>0)weights[candidate]=adjusted;
    }
    double maximum=*std::max_element(weights.begin(),weights.end());
    double cutoff=std::min(s.chosen_move_prune,maximum/64),subtract=std::min(s.chosen_move_subtract,maximum/64);
    for(auto& w:weights)w=w<cutoff?0:std::max(0.0,w-subtract);
    if(diagnostics)for(size_t i=0;i<weights.size();++i)(*diagnostics)[i].weight=weights[i];
    if(normalize_output)normalize_selection_weights(weights);
    return weights;
}

std::vector<int16_t> quantize_policy(const std::vector<double>& weights) {
    double maximum=0;
    for(double w:weights) {
        if(!std::isfinite(w)||w<0)throw std::runtime_error("Invalid policy selection weight");
        maximum=std::max(maximum,w);
    }
    if(maximum<=0)throw std::runtime_error("Empty policy selection weight");
    double factor=maximum<10?10/maximum:(maximum>30000?30000/maximum:1);
    std::vector<int16_t> result;result.reserve(weights.size());
    for(double w:weights)result.push_back(static_cast<int16_t>(std::round(w*factor)));
    return result;
}

std::vector<double> direct_policy_selection(std::vector<double> policy,const SearchSettings& s) {
    double maximum=*std::max_element(policy.begin(),policy.end());
    if(maximum<=0)return policy;
    double cutoff=std::min(s.chosen_move_prune,maximum/64),subtract=std::min(s.chosen_move_subtract,maximum/64);
    for(auto& p:policy)p=p<cutoff?0:std::max(0.0,p-subtract);
    normalize_selection_weights(policy);return policy;
}

// No sampling, tree mutation or training targets are produced by a snapshot.
void analysis_policy(SearchResult& result,const std::vector<int>& actions,const std::vector<RootChildStats>& stats,
                     const SearchSettings& settings,ValueStats parent,const GumbelSettings* gumbel) {
    result.gumbel=gumbel!=nullptr;
    result.pruned_value=result.value;result.pruned_utility=result.utility;result.pruned_wdl=result.search_wdl;
    if(!gumbel&&!stats.empty()) {
        // Source getPrunedNodeValues uses move-selection weights without LCB,
        // then includes the root NN evaluation at its uncertainty weight.
        auto weights=root_selection_weights(stats,settings,false,parent,false);
        double mass=result.network_sample_weight,winloss=mass*(result.network_wdl[0]-result.network_wdl[2]);
        double utility=winloss,draw=mass*result.network_wdl[1],nr=mass*result.network_wdl[3];
        for(size_t i=0;i<stats.size();++i)if(stats[i].weight>0) {
            double scale=weights[i]/stats[i].weight;mass+=weights[i];
            winloss+=scale*stats[i].win_loss_sum;utility+=scale*stats[i].value_sum;
            draw+=scale*stats[i].draw_sum;nr+=scale*stats[i].no_result_sum;
        }
        if(mass>0){result.pruned_value=winloss/mass;result.pruned_utility=utility/mass;result.pruned_wdl=wdl_from_value(winloss/mass,draw/mass,nr/mass);}
    }
    if(gumbel) {
        std::vector<GumbelChild> children;
        for(const auto& c:result.analysis)children.push_back({result.search_policy[c.action],c.utility,result.visits[c.action],0});
        auto policy=gumbel_policy(children,result.network_wdl[0]-result.network_wdl[2],*gumbel);
        for(size_t i=0;i<policy.size();++i)result.policy[result.analysis[i].action]=policy[i];
        result.move_policy=result.policy;
    } else if(!stats.empty()) {
        std::vector<RootSelectionDiagnostic> diagnostics;
        auto weights=root_selection_weights(stats,settings,settings.use_lcb,parent,false,&diagnostics);
        if(!settings.use_lcb){root_selection_weights(stats,settings,true,parent,false,&diagnostics);for(size_t i=0;i<weights.size();++i)diagnostics[i].weight=weights[i];}
        double mass=std::accumulate(weights.begin(),weights.end(),0.0);
        for(size_t i=0;i<actions.size();++i) {
            int action=actions[i];result.move_policy[action]=result.policy[action]=weights[i]/mass;
            for(auto& c:result.analysis)if(c.action==action){c.selection=diagnostics[i];break;}
        }
    } else {
        auto policy=result.search_policy;
        bool allocated=std::any_of(result.analysis.begin(),result.analysis.end(),[](const auto& c){return c.has_value;});
        if(allocated)for(const auto& c:result.analysis)if(!c.has_value)policy[c.action]=0;
        result.move_policy=result.policy=direct_policy_selection(std::move(policy),settings);
        for(auto& c:result.analysis)c.selection.weight=result.policy[c.action];
    }
    double visited_mass=0;for(const auto& c:result.analysis)if(c.has_value)visited_mass+=result.search_policy[c.action];
    double fpu=settings.use_fpu?fpu_value(result.network_wdl[0]-result.network_wdl[2],parent.value,visited_mass,settings.fpu_parent_power,settings.root_fpu_reduction_max,settings.fpu_parent_weight_by_visited_policy,settings.fpu_parent_weight):0;
    if(settings.use_fpu)fpu+=(-1-fpu)*settings.root_fpu_loss_prop;
    for(auto& c:result.analysis)if(!c.has_value){c.utility=fpu;c.win_loss=std::clamp(parent.win_loss+fpu-parent.value,-1.0,1.0);}
    auto& ranking=gumbel?result.policy:result.move_policy;
    result.action=int(std::max_element(ranking.begin(),ranking.end())-ranking.begin());
}

}
