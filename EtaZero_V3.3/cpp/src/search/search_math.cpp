#include "etazero/search.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace etazero {
WDL wdl_from_value(double value,double draw,double no_result) {
    if(!std::isfinite(value) || !std::isfinite(draw) || !std::isfinite(no_result))throw std::runtime_error("Nonfinite search WDL statistics");
    // KataGo ReportedSearchValues: roundoff in separately accumulated W-L
    // and draw statistics must not create negative probability targets.
    value=std::clamp(value,-1.0,1.0);
    draw=std::max(0.0,draw);no_result=std::max(0.0,no_result);
    double neutral=draw+no_result,available=1.0-std::abs(value);
    if(neutral>available){double scale=available/neutral;draw*=scale;no_result*=scale;}
    return {std::clamp((1.0-draw-no_result+value)/2,0.0,1.0),draw,
            std::clamp((1.0-draw-no_result-value)/2,0.0,1.0),no_result};
}
std::vector<double> policy_temperature_distribution(const std::vector<double>& weights,double temperature) {
    if(weights.empty() || !std::isfinite(temperature) || temperature<=0)throw std::runtime_error("Invalid policy temperature");
    double maximum=0;
    for(double w:weights) {
        if(!std::isfinite(w)||w<0)throw std::runtime_error("Invalid policy weight");
        maximum=std::max(maximum,w);
    }
    if(maximum<=0)throw std::runtime_error("Empty policy weights");
    std::vector<double> result(weights.size());double sum=0;
    for(size_t i=0;i<weights.size();++i) {
        result[i]=weights[i]>0?static_cast<float>(std::exp((std::log(weights[i])-std::log(maximum))*(1.0/temperature))):0;sum+=result[i];
    }
    for(auto& w:result)w=static_cast<float>(w/sum);
    return result;
}
double temperature_at_turn(double early,double late,double halflife,int turn,int board_area) {
    if(halflife<=0 || turn<0 || board_area<=0)throw std::runtime_error("Invalid temperature schedule");
    return late+(early-late)*std::pow(0.5,turn/halflife*19/std::sqrt(double(board_area)));
}
std::vector<double> temperature_distribution(const std::vector<double>& weights,double temperature,double only_below_prob) {
    if(weights.empty() || !std::isfinite(temperature) || temperature<0 || !std::isfinite(only_below_prob) || only_below_prob<0 || only_below_prob>1)
        throw std::runtime_error("Invalid move temperature");
    double maximum=0,sum=0;
    for(double w:weights) {
        if(!std::isfinite(w) || w<0)throw std::runtime_error("Invalid move weight");
        maximum=std::max(maximum,w);sum+=w;
    }
    if(maximum<=0 || !std::isfinite(sum))throw std::runtime_error("Empty move weights");
    std::vector<double> result(weights.size(),0);
    if(temperature<=1e-4 && only_below_prob>=1) {
        result[std::max_element(weights.begin(),weights.end())-weights.begin()]=1;
        return result;
    }
    double threshold=std::min(0.0,std::log(std::max(1e-50,only_below_prob))+std::log(sum)-std::log(maximum));
    for(size_t i=0;i<weights.size();++i)if(weights[i]>0) {
        double logw=std::log(weights[i])-std::log(maximum);
        // The limit at T=0 is well defined even with a protected high-probability prefix.
        result[i]=logw>threshold?std::exp(logw):temperature==0?(logw==threshold?std::exp(threshold):0):
                  std::exp((logw-threshold)/temperature+threshold);
    }
    sum=std::accumulate(result.begin(),result.end(),0.0);
    for(auto& w:result)w/=sum;
    return result;
}
double value_weight_cdf(double z) {
    // KataGo DistributionTable: t(3), 2000 points on [-50,50], linear interpolation.
    // The closed-form t(3) CDF avoids an additional incomplete-beta dependency.
    static const auto table=[] {
        std::array<double,2000> values{};values.back()=1;
        const double pi=std::acos(-1.0),root3=std::sqrt(3.0);
        for(size_t i=1;i+1<values.size();++i) {
            double x=-50+i*100.0/1999;
            values[i]=0.5+(std::atan(x/root3)+x*root3/(x*x+3))/pi;
        }
        return values;
    }();
    double d=1999*(z+50)/100;
    if(d<=0)return 0;
    if(d>=1999)return 1;
    int index=int(d);return table[index]+(d-index)*(table[index+1]-table[index]);
}
double mixed_policy_logit(double ordinary,double optimistic,double optimism) {
    // The fixed source CUDA/ROCm adapter mixes its float output buffers on the host.
    float p=static_cast<float>(ordinary),pOpt=static_cast<float>(optimistic),o=static_cast<float>(optimism);
    return p+(pOpt-p)*o;
}
std::vector<double> source_policy_probabilities(const Evaluation& e,const std::vector<int>& actions,double temperature,double optimism) {
    std::vector<float> policy(e.logits.size(),0);
    float maximum=-1e25f,scale=e.policy_output_scale/temperature,sum=0;
    if(!e.policy_optimism_allowed)optimism=0;
    for(int a:actions) {
        float value=e.has_auxiliary?mixed_policy_logit(e.logits.at(a),e.optimistic_logits.at(a),optimism):e.logits.at(a);
        if(!std::isfinite(value))throw std::runtime_error("Nonfinite policy logit");
        policy[a]=value*scale;maximum=std::max(maximum,policy[a]);
    }
    // KataGo stores each exp and division in float and accumulates a float sum.
    for(int a:actions){policy[a]=std::exp(policy[a]-maximum);sum+=policy[a];}
    if(!std::isfinite(sum)||actions.empty())throw std::runtime_error("Invalid policy mass");
    std::vector<double> result(policy.size(),0);
    for(int a:actions)result[a]=sum<=0?1.0f/actions.size():policy[a]/sum;
    return result;
}
double uncertainty_weight(double stdev,bool supported,const SearchSettings& s) {
    if(!s.use_uncertainty || !supported)return 1;
    if(!std::isfinite(stdev) || stdev<0)throw std::runtime_error("Invalid shortterm error standard deviation");
    double powered=s.uncertainty_exponent==1?stdev:s.uncertainty_exponent==0.5?std::sqrt(stdev):std::pow(stdev,s.uncertainty_exponent);
    double weight=s.uncertainty_coeff/(powered+s.uncertainty_coeff/s.uncertainty_max_weight);
    if(!std::isfinite(weight) || weight<=0)throw std::runtime_error("Nonpositive uncertainty sample weight");
    return weight;
}
std::vector<double> noise_pruned_weights(const std::vector<RootChildStats>& children,const SearchSettings& s) {
    std::vector<double> weights;double total=0;
    for(const auto& c:children){weights.push_back(c.weight);total+=c.weight;}
    if(!s.use_noise_pruning || children.size()<=1 || total<=0.00001)return weights;
    double utility_sum=0,weight_sum=0,policy_sum=0;
    for(size_t i=0;i<children.size();++i) {
        const auto& c=children[i];if(c.weight<=0)continue;double utility=c.value_sum/c.weight,policy=std::max(1e-30,c.prior);
        double next=weights[i];
        if(weight_sum>0 && policy_sum>0) {
            double gap=utility_sum/weight_sum-utility;
            if(gap>0) {
                double lenient=2*weight_sum*policy/policy_sum;
                if(next>lenient)next-=std::min((next-lenient)*(1-std::exp(-gap/s.noise_prune_utility_scale)),s.noise_pruning_cap);
            }
        }
        weights[i]=next;utility_sum+=utility*next;weight_sum+=next;policy_sum+=policy;
    }
    return weights;
}
ValueStats aggregate_values(const WDL& initial,const std::vector<RootChildStats>& children,const SearchSettings& s,bool noisy_root,double initial_weight,SubtreeValueBias* bias) {
    auto adjusted=noise_pruned_weights(children,s);double total=std::accumulate(adjusted.begin(),adjusted.end(),0.0),maximum=0,simple_value=0;
    int64_t visits=1;
    for(size_t i=0;i<children.size();++i) {
        const auto& c=children[i];visits+=c.visits;maximum=std::max(maximum,adjusted[i]);
        if(c.weight>0)simple_value+=c.value_sum/c.weight*adjusted[i];
    }
    // Noise pruning takes precedence over the root-only chosen prune/subtract backup branch.
    double subtract=noisy_root && !s.use_noise_pruning?std::min(s.chosen_move_subtract,maximum/64):0;
    double prune=noisy_root && !s.use_noise_pruning?std::min(s.chosen_move_prune,maximum/64):0;
    double adjusted_total=0;
    if(total>0) {
        simple_value/=total;
        for(size_t i=0;i<children.size();++i) {
            const auto& c=children[i];if(c.weight<=0 || c.visits<=0)continue;
            double stdev=std::sqrt(1e-8+1/(1.5*std::sqrt(adjusted[i])));
            if(adjusted[i]<prune)adjusted[i]=0;
            else adjusted[i]=std::max(0.0,adjusted[i]-subtract);
            if(s.value_weight_exponent>0) {
                double z=(c.value_sum/c.weight-simple_value)/stdev;
                adjusted[i]*=std::pow(value_weight_cdf(z)+0.0001,s.value_weight_exponent);
            }
            adjusted_total+=adjusted[i];
        }
        if(adjusted_total<=0)throw std::runtime_error("Value weighting removed every child");
    }
    double value=0,value_sq=0,win_loss=0,draw=0,no_result=0,weight_sq=0;
    for(size_t i=0;i<children.size();++i) {
        const auto& c=children[i];if(c.weight<=0 || total<=0)continue;
        double desired=adjusted[i]*total/adjusted_total,scale=desired/c.weight;
        value+=scale*c.value_sum;value_sq+=scale*c.value_sq_sum;draw+=scale*c.draw_sum;no_result+=scale*c.no_result_sum;
        win_loss+=scale*c.win_loss_sum;
        weight_sq+=scale*scale*c.weight_sq;
    }
    // KataGo learns from the weighted CHILD utility before adding this node's
    // raw NN sample. Confidence uses child weight BEFORE noise pruning.
    double own=initial[0]-initial[2],utility=own;
    if(bias) {
        double original=0;for(const auto& c:children)original+=c.weight;
        utility=bias->update(own,total>0?value/total:0,original,total,
                             s.subtree_value_bias_factor,s.subtree_value_bias_weight_exponent);
    }
    value+=initial_weight*utility;value_sq+=initial_weight*utility*utility;
    win_loss+=initial_weight*own;draw+=initial_weight*initial[1];no_result+=initial_weight*initial[3];weight_sq+=initial_weight*initial_weight;
    double weight=initial_weight+total;
    return {visits,value/weight,value_sq/weight,draw/weight,weight,weight_sq,win_loss/weight,no_result/weight};
}
double explore_scaling(double total,const ValueStats& parent,const SearchSettings& s) {
    double stdev=s.c_puct_stdev_prior;
    if(parent.visits>0 && parent.weight>1) {
        double variance_prior=s.c_puct_stdev_prior*s.c_puct_stdev_prior;
        double square=parent.value*parent.value;
        stdev=std::sqrt(std::max(0.0,((square+variance_prior)*s.c_puct_stdev_prior_weight+
              std::max(square,parent.value_sq)*parent.weight)/(s.c_puct_stdev_prior_weight+parent.weight-1)-square));
    }
    return (s.c_puct+s.c_puct_log*std::log((total+s.c_puct_base)/s.c_puct_base))*std::sqrt(total+0.01)*
           (1+s.c_puct_stdev_scale*(stdev/s.c_puct_stdev_prior-1));
}
double child_selection_score(double prior,double value,double weight,int pending,double total,
                             const ValueStats& parent,const SearchSettings& s,bool force) {
    double virtual_weight=pending*s.virtual_loss;
    if(virtual_weight>0)value+=(-1-value)*virtual_weight/(virtual_weight+std::max(0.25,weight));
    weight+=virtual_weight;
    if(force && prior>0 && weight<std::sqrt(prior*total*s.forced_playouts))return 1e20;
    return value+explore_scaling(total,parent,s)*prior/(1+weight);
}
double root_explore_score(double prior,double utility,double weight,int pending,int64_t edge_visits,
    int64_t child_visits,double total,double max_weight,const ValueStats& parent,
    const SearchSettings& s,bool force,bool hint,double remaining,std::mt19937_64& random) {
    double virtual_weight=pending*s.virtual_loss;
    if(virtual_weight>0)utility+=(-1-utility)*virtual_weight/(virtual_weight+std::max(.25,weight));
    weight+=virtual_weight;
    double per_visit=parent.weight/std::max<int64_t>(1,parent.visits);
    if(s.futile_visits_threshold>0 && child_visits+remaining<
       s.futile_visits_threshold*max_weight*(edge_visits+1.0)/(weight+per_visit))return -1e40;
    if(force && prior>0 && weight<std::sqrt(prior*total*s.forced_playouts))return 1e20;
    if(hint && weight+(weight+per_visit)/(child_visits+1.0)<max_weight*.8)return 1e20;
    if(s.wide_root_noise>0) {
        prior=static_cast<float>(std::pow(static_cast<float>(prior),1/(4*s.wide_root_noise+1)));
        if(std::bernoulli_distribution(.5)(random))utility+=s.wide_root_noise*std::abs(std::normal_distribution<double>()(random));
    }
    return utility+explore_scaling(total,parent,s)*prior/(1+weight);
}

}

namespace etazero {
void mix_secondary_policy(SearchResult& result,const std::vector<double>& policy,const std::vector<double>& utilities,
    const std::vector<bool>& known,const SearchSettings& settings,int pass,double target_scale) {
    double proportion=settings.human_sl_chosen_move_proportion;if(proportion<=0||policy.empty())return;
    auto mix=[&](std::vector<double>& destination,bool direct){
        double sum=0,maximum=-1e100;int count=0;
        for(size_t i=0;i<policy.size();++i)if(known[i] && policy[i]>=0){
            sum+=utilities[i];maximum=std::max(maximum,utilities[i]);++count;
        }
        double average=sum/std::max(1,count);maximum=std::max(maximum,average);
        auto shifted=policy;double mass=0,lambda=settings.human_sl_chosen_move_pikl_lambda;
        for(size_t i=0;i<shifted.size();++i){double utility=known[i]?utilities[i]:average;
            shifted[i]=(direct || known[i])?std::max(0.0,shifted[i])*
                (lambda==0?(utility==maximum?1:0):std::exp((utility-maximum)/lambda)):0;
            mass+=shifted[i];
        }
        if(mass<=0)return;
        for(auto& p:shifted)p/=mass;
        if(settings.human_sl_chosen_move_ignore_pass && pass>=0){double nonpass=1-shifted[pass];
            if(nonpass<=0)return;
            for(size_t i=0;i<shifted.size();++i)if(int(i)!=pass)shifted[i]*=(1-destination[pass])/nonpass;
            shifted[pass]=destination[pass];}
        for(size_t i=0;i<destination.size();++i)destination[i]+=proportion*(shifted[i]-destination[i]);};
    // Source extractPolicyTarget disallows direct unsearched moves. Actual
    // move selection may include them, so normalize each domain independently.
    result.statistics_policy=result.policy;mix(result.statistics_policy,true);
    mix(result.policy,false);mix(result.move_policy,true);auto target=result.policy;
    for(auto& p:target)p*=target_scale;
    result.policy_target=quantize_policy(target);
    result.policy_surprise=0;
    for(size_t i=0;i<result.statistics_policy.size();++i)if(result.statistics_policy[i]>0)result.policy_surprise+=result.statistics_policy[i]*std::log(result.statistics_policy[i]/std::max(1e-100,result.search_policy[i]));
    result.policy_surprise=std::max(0.0,result.policy_surprise);
}
}

namespace etazero {
void validate_search_controls(const SearchSettings& s) {
    auto bound=[](double x,double low,double high,const char* key){if(!std::isfinite(x)||x<low||x>high)throw std::runtime_error(std::string("Invalid search control: ")+key);};
    for(auto cap:{s.simulations,s.max_visits,s.max_playouts})if(cap<0||cap>SEARCH_CONFIG_LIMIT)throw std::runtime_error("Search configuration budget must be in [0, 2^50]");
    bound(s.c_puct,0,1000,"c_puct");bound(s.root_policy_temperature,.01,100,"root_policy_temperature");bound(s.root_policy_temperature_early,.01,100,"root_policy_temperature_early");
    bound(s.min_playouts_per_thread,0,1e20,"min_playouts_per_thread");
    bound(s.max_visits_pondering,1,SEARCH_CONFIG_LIMIT,"max_visits_pondering");bound(s.max_playouts_pondering,1,SEARCH_CONFIG_LIMIT,"max_playouts_pondering");
    bound(s.max_time_pondering,0,1e20,"max_time_pondering");bound(s.lag_buffer,0,3600,"lag_buffer");
    bound(s.tree_reuse_carry_over_time_factor,0,1,"tree_reuse_carry_over_time_factor");
    bound(s.overallocate_time_factor,.01,100,"overallocate_time_factor");bound(s.midgame_time_factor,.01,100,"midgame_time_factor");
    bound(s.midgame_turn_peak_time,0,1000,"midgame_turn_peak_time");bound(s.endgame_turn_time_decay,0,1000,"endgame_turn_time_decay");
    bound(s.obvious_moves_time_factor,.01,1,"obvious_moves_time_factor");bound(s.obvious_moves_policy_entropy_tolerance,.001,2,"obvious_moves_policy_entropy_tolerance");bound(s.obvious_moves_policy_surprise_tolerance,.001,2,"obvious_moves_policy_surprise_tolerance");
    bound(s.human_sl_cpuct_exploration,0,1000,"human_sl_cpuct_exploration");bound(s.human_sl_cpuct_permanent,0,1000,"human_sl_cpuct_permanent");
    for(double p:{s.human_sl_root_weightless,s.human_sl_root_weightful,s.human_sl_player_weightless,s.human_sl_player_weightful,s.human_sl_opponent_weightless,s.human_sl_opponent_weightful,s.human_sl_chosen_move_proportion})bound(p,0,1,"human_sl_probability");
    bound(s.human_sl_chosen_move_pikl_lambda,0,1e9,"human_sl_chosen_move_pikl_lambda");bound(s.avoid_repeated_pattern_utility,-3,3,"avoid_repeated_pattern_utility");
}
}
