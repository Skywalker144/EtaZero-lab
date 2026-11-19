// Copyright 2021 DeepMind Technologies Limited.
// Adapted from Mctx 428bfb7 (Apache-2.0); see licenses/Mctx.txt.
#include "gumbel.h"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace Gumbel {
void GumbelSettings::validate() const {
    if(max_num_considered_actions<1 ||
       !std::isfinite(c_visit) || c_visit<0 || !std::isfinite(c_scale) || c_scale<=0 ||
       !std::isfinite(noise_scale) || noise_scale<0)
        throw StringError("Invalid Gumbel settings");
}
std::vector<double> gumbel_completed_q(const std::vector<GumbelChild>& children,double raw,const GumbelSettings& s) {
    double total=0,maxvisit=0,mass=0,weighted=0;
    for(const auto& c:children) {
        if(c.prior<0)continue;
        total+=c.visits;maxvisit=std::max(maxvisit,double(c.visits));
        if(c.visits>0){double p=std::max(c.prior,std::numeric_limits<double>::min());mass+=p;weighted+=p*c.q;}
    }
    double mix=mass>0?(raw+total*(weighted/mass))/(1+total):raw;
    std::vector<double> q;q.reserve(children.size());
    for(const auto& c:children)q.push_back(c.visits>0?c.q:mix);
    if(s.rescale_q_values) {
        double low=std::numeric_limits<double>::infinity(),high=-low;
        for(size_t i=0;i<q.size();++i)if(children[i].prior>=0){low=std::min(low,q[i]);high=std::max(high,q[i]);}
        double gap=std::max(1e-8,high-low);
        for(auto& value:q)value=(value-low)/gap;
    }
    double scale=(s.c_visit+maxvisit)*s.c_scale;
    for(auto& value:q)value*=scale;
    return q;
}
std::vector<double> gumbel_policy(const std::vector<GumbelChild>& children,double raw,const GumbelSettings& s) {
    auto policy=gumbel_completed_q(children,raw,s);double maximum=-std::numeric_limits<double>::infinity();
    for(size_t i=0;i<policy.size();++i){if(children[i].prior<0)policy[i]=-std::numeric_limits<double>::infinity();else policy[i]+=std::log(std::max(children[i].prior,std::numeric_limits<double>::min()));maximum=std::max(maximum,policy[i]);}
    double mass=0;for(auto& p:policy){p=std::exp(p-maximum);mass+=p;}
    for(auto& p:policy)p/=mass;
    return policy;
}
int gumbel_interior_selection(const std::vector<GumbelChild>& children,double raw,const GumbelSettings& s) {
    double total=1;
    bool any=false;
    for(const auto& c:children)if(c.prior>=0){total+=c.visits+c.pending;any=true;}
    if(!any)return -1;
    // Completed visits alone enter Q. In-flight work only reduces the allocation deficit.
    auto policy=gumbel_policy(children,raw,s);
    int best=-1;double score=-std::numeric_limits<double>::infinity();
    for(size_t i=0;i<children.size();++i)if(children[i].prior>=0){double x=policy[i]-(children[i].visits+children[i].pending)/total;
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
GumbelRoot::GumbelRoot(const std::vector<double>& priors,int64_t budget,const GumbelSettings& s,Rand& rng)
    :visits_(priors.size(),0),pending_(priors.size(),0),budget_(budget) {
    s.validate();if(priors.empty()||budget<0)throw StringError("Invalid Gumbel root budget/domain");
    for(double p:priors){if(!std::isfinite(p))throw StringError("Invalid Gumbel prior");
        logits_.push_back(p<0?-std::numeric_limits<double>::infinity():std::log(std::max(p,std::numeric_limits<double>::min())));
        noise_.push_back(s.noise_scale>0?-s.noise_scale*std::log(rng.nextExponential()):0);}
    int m=std::min(s.max_num_considered_actions,int(std::count_if(priors.begin(),priors.end(),[](double p){return p>=0;})));
    if(m==0)throw StringError("No allowed Gumbel root action");
    if(m<=1){phases_.push_back({0,budget,1,0});return;}
    int log2m=int(std::ceil(std::log2(m))),alive=m;int64_t begin=0,first_visit=0;
    // Exact mctx visit sequence, represented by O(log(m)) runs rather than an O(budget) table.
    while(begin<budget) {
        int64_t rounds=std::max<int64_t>(1,budget/(log2m*alive));int64_t length=rounds*alive;
        phases_.push_back({begin,length,alive,first_visit});begin+=length;first_visit+=rounds;
        if(alive==2)break;
        alive=std::max(2,alive/2);
    }
}
int64_t GumbelRoot::considered_visit(int64_t simulation) const {
    if(simulation<0||simulation>=budget_)throw StringError("Gumbel simulation index outside budget");
    if(phases_[0].actions==1)return simulation;
    for(const auto& phase:phases_)if(simulation<phase.begin+phase.length)
        return phase.first_visit+(simulation-phase.begin)/phase.actions;
    const auto& phase=phases_.back();
    return phase.first_visit+(simulation-phase.begin)/phase.actions;
}
int GumbelRoot::reserve(const std::vector<double>& q) {
    if(q.size()!=visits_.size())throw StringError("Gumbel root Q shape mismatch");
    if(done())return -1;
    int64_t required=considered_visit(issued_);
    // Wait at visit-level boundaries. Elimination never uses in-flight estimates.
    if(required!=considered_&&pending_total_)return -1;
    considered_=required;int best=-1;double score=-std::numeric_limits<double>::infinity();
    for(size_t i=0;i<q.size();++i)if(visits_[i]+pending_[i]==required) {
        double x=logits_[i]+noise_[i]+q[i];if(x>score){score=x;best=int(i);}}
    if(best<0)throw StringError("Gumbel scheduler has no eligible action");
    ++pending_[best];++pending_total_;++issued_;return best;
}
void GumbelRoot::commit(int action) {
    if(action<0||size_t(action)>=pending_.size()||pending_[action]<=0)throw StringError("Invalid Gumbel completion");
    --pending_[action];--pending_total_;++visits_[action];
}
void GumbelRoot::cancel(int action,bool retry) {
    if(action<0||size_t(action)>=pending_.size()||pending_[action]<=0)throw StringError("Invalid Gumbel cancellation");
    --pending_[action];--pending_total_;if(retry)--issued_;
}
int GumbelRoot::winner(const std::vector<double>& q, bool allow_pending) const {
    if((pending_total_ && !allow_pending)||q.size()!=visits_.size())throw StringError("Unfinished Gumbel result");
    auto maxvisit=*std::max_element(visits_.begin(),visits_.end());int best=-1;double score=-std::numeric_limits<double>::infinity();
    for(size_t i=0;i<q.size();++i)if(visits_[i]==maxvisit){double x=logits_[i]+noise_[i]+q[i];if(x>score){score=x;best=int(i);}}
    return best;
}
}
