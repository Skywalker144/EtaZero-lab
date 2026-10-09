// Algorithm adapted from google-deepmind/mctx, Apache-2.0, commit 428bfb7.
// See THIRD_PARTY.md and reference_sources.json for the fixed sources.
#include "etazero/gumbel.h"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace etazero {
void GumbelSettings::validate() const {
    if(max_num_considered_actions<1 || max_num_considered_actions>625 ||
       !std::isfinite(c_visit) || c_visit<0 || !std::isfinite(c_scale) || c_scale<=0 ||
       !std::isfinite(noise_scale) || noise_scale<0)
        throw std::runtime_error("Invalid Gumbel settings");
}
std::vector<double> gumbel_completed_q(const std::vector<GumbelChild>& children,double raw,const GumbelSettings& s) {
    double total=0,maxvisit=0,mass=0,weighted=0;
    for(const auto& c:children) {
        total+=c.visits;maxvisit=std::max(maxvisit,double(c.visits));
        if(c.visits>0){double p=std::max(c.prior,std::numeric_limits<double>::min());mass+=p;weighted+=p*c.q;}
    }
    double mix=mass>0?(raw+total*(weighted/mass))/(1+total):raw;
    std::vector<double> q;q.reserve(children.size());
    for(const auto& c:children)q.push_back(c.visits>0?c.q:mix);
    if(s.rescale_q_values) {
        auto bounds=std::minmax_element(q.begin(),q.end());double low=*bounds.first,gap=std::max(1e-8,*bounds.second-low);
        for(auto& value:q)value=(value-low)/gap;
    }
    double scale=(s.c_visit+maxvisit)*s.c_scale;
    for(auto& value:q)value*=scale;
    return q;
}
std::vector<double> gumbel_policy(const std::vector<GumbelChild>& children,double raw,const GumbelSettings& s) {
    auto policy=gumbel_completed_q(children,raw,s);double maximum=-std::numeric_limits<double>::infinity();
    for(size_t i=0;i<policy.size();++i){policy[i]+=std::log(std::max(children[i].prior,std::numeric_limits<double>::min()));maximum=std::max(maximum,policy[i]);}
    double mass=0;for(auto& p:policy){p=std::exp(p-maximum);mass+=p;}
    for(auto& p:policy)p/=mass;
    return policy;
}
int gumbel_interior_selection(const std::vector<GumbelChild>& children,double raw,const GumbelSettings& s) {
    auto policy=gumbel_policy(children,raw,s);double total=1;
    for(const auto& c:children)total+=c.visits+c.pending;
    int best=0;double score=-std::numeric_limits<double>::infinity();
    for(size_t i=0;i<children.size();++i){double x=policy[i]-(children[i].visits+children[i].pending)/total;
        if(x>score){score=x;best=int(i);}}
    return best;
}
std::vector<int16_t> quantize_gumbel_policy(const std::vector<double>& policy) {
    double maximum=0;
    for(double p:policy)maximum=std::max(maximum,p);
    std::vector<int16_t> out;out.reserve(policy.size());
    for(double p:policy)out.push_back(static_cast<int16_t>(std::round(30000*(p/maximum))));
    return out;
}
GumbelRoot::GumbelRoot(const std::vector<double>& priors,int budget,const GumbelSettings& s,std::mt19937_64& rng)
    :visits_(priors.size(),0),pending_(priors.size(),0),budget_(budget) {
    s.validate();if(priors.empty()||budget<0)throw std::runtime_error("Invalid Gumbel root budget/domain");
    std::extreme_value_distribution<double> draw(0,1);
    for(double p:priors){if(!std::isfinite(p)||p<0)throw std::runtime_error("Invalid Gumbel prior");
        logits_.push_back(std::log(std::max(p,std::numeric_limits<double>::min())));
        noise_.push_back(s.noise_scale>0?s.noise_scale*draw(rng):0);}
    int m=std::min(s.max_num_considered_actions,int(priors.size()));
    if(m<=1){phases_.push_back({0,budget,1,0});return;}
    int log2m=int(std::ceil(std::log2(m))),alive=m,first_visit=0;int64_t begin=0;
    // Exact mctx visit sequence, represented by O(log(m)) runs rather than an O(budget) table.
    while(begin<budget) {
        int rounds=std::max(1,budget/(log2m*alive));int64_t length=int64_t(rounds)*alive;
        phases_.push_back({begin,length,alive,first_visit});begin+=length;first_visit+=rounds;
        if(alive==2)break;
        alive=std::max(2,alive/2);
    }
}
int GumbelRoot::considered_visit(int simulation) const {
    if(simulation<0||simulation>=budget_)throw std::runtime_error("Gumbel simulation index outside budget");
    if(phases_[0].actions==1)return simulation;
    for(const auto& phase:phases_)if(simulation<phase.begin+phase.length)
        return phase.first_visit+int((simulation-phase.begin)/phase.actions);
    const auto& phase=phases_.back();
    return phase.first_visit+int((simulation-phase.begin)/phase.actions);
}
int GumbelRoot::reserve(const std::vector<double>& q) {
    if(q.size()!=visits_.size())throw std::runtime_error("Gumbel root Q shape mismatch");
    if(done())return -1;
    int required=considered_visit(issued_);
    // Wait at visit-level boundaries. Elimination never uses in-flight estimates.
    if(required!=considered_&&pending_total_)return -1;
    considered_=required;int best=-1;double score=-std::numeric_limits<double>::infinity();
    for(size_t i=0;i<q.size();++i)if(visits_[i]+pending_[i]==required) {
        double x=logits_[i]+noise_[i]+q[i];if(x>score){score=x;best=int(i);}}
    if(best<0)throw std::runtime_error("Gumbel scheduler has no eligible action");
    ++pending_[best];++pending_total_;++issued_;return best;
}
void GumbelRoot::commit(int action) {
    if(action<0||size_t(action)>=pending_.size()||pending_[action]<=0)throw std::runtime_error("Invalid Gumbel completion");
    --pending_[action];--pending_total_;++visits_[action];
}
void GumbelRoot::cancel(int action) {
    if(action<0||size_t(action)>=pending_.size()||pending_[action]<=0)throw std::runtime_error("Invalid Gumbel cancellation");
    --pending_[action];--pending_total_;
}
int GumbelRoot::winner(const std::vector<double>& q) const {
    if(pending_total_||q.size()!=visits_.size())throw std::runtime_error("Unfinished Gumbel result");
    auto maxvisit=*std::max_element(visits_.begin(),visits_.end());int best=-1;double score=-std::numeric_limits<double>::infinity();
    for(size_t i=0;i<q.size();++i)if(visits_[i]==maxvisit){double x=logits_[i]+noise_[i]+q[i];if(x>score){score=x;best=int(i);}}
    return best;
}
void finish_gumbel_result(SearchResult& result,const std::vector<int>& actions,
                          const std::vector<GumbelChild>& children,double raw,
                          const GumbelRoot& root,const GumbelSettings& s,double temperature,std::mt19937_64& rng) {
    auto target=gumbel_policy(children,raw,s);auto q=gumbel_completed_q(children,raw,s);
    result.policy.assign(result.network_policy.size(),0);result.move_policy=result.policy;
    result.policy_target.assign(result.policy.size(),0);auto packed=quantize_gumbel_policy(target);
    std::vector<double> behavior(actions.size(),0);
    if(s.sample_visits) {
        for(size_t i=0;i<actions.size();++i)behavior[i]=double(root.visits()[i]);
        if(std::accumulate(behavior.begin(),behavior.end(),0.0)==0)for(size_t i=0;i<actions.size();++i)behavior[i]=children[i].prior;
        behavior=temperature_distribution(behavior,temperature);
        result.action=actions[std::discrete_distribution<size_t>(behavior.begin(),behavior.end())(rng)];
    } else {int winner=root.winner(q);result.action=actions[winner];behavior[winner]=1;}
    for(size_t i=0;i<actions.size();++i) {
        int action=actions[i];result.policy[action]=target[i];result.policy_target[action]=packed[i];result.move_policy[action]=behavior[i];
        if(target[i]>0)result.policy_surprise+=target[i]*std::log(target[i]/std::max(children[i].prior,std::numeric_limits<double>::min()));
    }
    result.policy_surprise=std::max(0.0,result.policy_surprise);
}
}
