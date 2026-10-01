#include "etazero/search.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace etazero {
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
        result[i]=weights[i]>0?std::exp((std::log(weights[i])-std::log(maximum))/temperature):0;sum+=result[i];
    }
    for(auto& w:result)w/=sum;
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
ValueStats aggregate_values(const WDL& initial,const std::vector<RootChildStats>& children,const SearchSettings& s,bool noisy_root) {
    std::vector<double> adjusted(children.size());double total=0,maximum=0,simple_value=0;
    int64_t visits=1;
    for(size_t i=0;i<children.size();++i) {
        const auto& c=children[i];visits+=c.visits;
        adjusted[i]=c.weight;total+=c.weight;maximum=std::max(maximum,c.weight);simple_value+=c.value_sum;
    }
    double subtract=noisy_root?std::min(s.chosen_move_subtract,maximum/64):0;
    double prune=noisy_root?std::min(s.chosen_move_prune,maximum/64):0;
    double adjusted_total=0;
    if(total>0) {
        simple_value/=total;
        for(size_t i=0;i<children.size();++i) {
            const auto& c=children[i];if(c.weight<=0 || c.visits<=0)continue;
            if(c.weight<prune)adjusted[i]=0;
            else adjusted[i]=std::max(0.0,c.weight-subtract);
            if(s.value_weight_exponent>0) {
                double stdev=std::sqrt(1e-8+1/(1.5*std::sqrt(c.weight)));
                double z=(c.value_sum/c.weight-simple_value)/stdev;
                adjusted[i]*=std::pow(value_weight_cdf(z)+0.0001,s.value_weight_exponent);
            }
            adjusted_total+=adjusted[i];
        }
        if(adjusted_total<=0)throw std::runtime_error("Value weighting removed every child");
    }
    double value=initial[0]-initial[2],value_sq=value*value,draw=initial[1],weight_sq=1;
    for(size_t i=0;i<children.size();++i) {
        const auto& c=children[i];if(c.weight<=0)continue;
        double desired=adjusted[i]*total/adjusted_total,scale=desired/c.weight;
        value+=scale*c.value_sum;value_sq+=scale*c.value_sq_sum;draw+=scale*c.draw_sum;
        weight_sq+=scale*scale*c.weight_sq;
    }
    return {visits,value/(1+total),value_sq/(1+total),draw/(1+total),1+total,weight_sq};
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
}
