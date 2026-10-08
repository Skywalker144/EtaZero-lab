#include "etazero/gumbel_search.h"
#include "etazero/symmetry.h"
#include <algorithm>
#include <cmath>
namespace etazero::muzero {
template<bool Full> std::vector<GumbelChild> GumbelSearch<Full>::node_children(const Node& node) const {
    std::vector<GumbelChild> out;out.reserve(node.edges.size());
    for(const auto& edge:node.edges) {
        GumbelChild c;c.prior=edge.search_prior;c.pending=edge.pending;
        if(edge.child&&edge.child->ready){c.visits=edge.child->stats.visits;c.q=-edge.child->stats.value;}
        out.push_back(c);
    }
    return out;
}
template<bool Full> SearchResult GumbelSearch<Full>::run(const Game& game,double temperature,SearchRun options) {
    if(!std::isfinite(temperature)||temperature<0||options.max_visits<0||options.max_playouts<-1||options.max_time<-1||!std::isfinite(options.max_time))
        throw std::runtime_error("Invalid Gumbel MuZero search caps/temperature");
    if(options.hint_action<-1||(options.hint_action>=0&&!game.legal(options.hint_action)))throw std::runtime_error("Illegal Gumbel MuZero root hint");
    advance(0);auto start=std::chrono::steady_clock::now();SearchResult result;int area=game.actions();
    result.policy.resize(area);result.move_policy.resize(area);result.network_policy.resize(area);result.search_policy.resize(area);
    result.visits.resize(area);result.q_values.resize(area);result.q_visits.resize(area);
    if(game.finished()){double v=game.terminal_value();result.value=v;result.network_wdl=result.search_wdl={(v+std::abs(v))/2,1-std::abs(v),(std::abs(v)-v)/2};return result;}
    int max_playouts=options.max_playouts<0?settings_.max_playouts:options.max_playouts;
    if(!max_playouts||(options.should_stop&&options.should_stop())){result.stopped_early=max_playouts!=0;return result;}
    auto obs=game.observation();
    int symmetry=settings_.nn_randomize?std::uniform_int_distribution<int>(0,game.rule()==Rule::HEX?1:7)(random_):settings_.nn_symmetry;
    mapping_=symmetry_mapping(game.canvas(),input_symmetry(obs,symmetry));board_actions_.clear();std::vector<int> root_actions;
    for(int a=0;a<area;++a){if(obs[a])board_actions_.push_back(a);if(game.legal(a))root_actions.push_back(a);}
    root_=node();remove_noise_=options.remove_root_noise;hint_=options.hint_action;
    auto initial=evaluator_.initial(transform_observation(obs,game.canvas(),mapping_));
    if(options.collect_root_policy_invalid_mass) {
        const auto& logits=initial.evaluation.logits;double maximum=-std::numeric_limits<double>::infinity();
        for(int a:board_actions_)maximum=std::max(maximum,logits.at(mapping_[a]));
        double mass=0,invalid=0;
        for(int a:board_actions_){double p=std::exp(logits.at(mapping_[a])-maximum);mass+=p;if(obs[area+a]||obs[2*area+a])invalid+=p;}
        if(!std::isfinite(mass)||mass<=0||!std::isfinite(invalid))throw std::runtime_error("Invalid Gumbel root policy diagnostic");
        result.root_policy_invalid_mass=invalid/mass;
    }
    expand(*root_,std::move(initial),root_actions,true);
    std::vector<double> priors;
    for(auto& edge:root_->edges) {
        result.network_policy[edge.action]=edge.prior;
        edge.search_prior=hint_>=0?.98*edge.prior+(edge.action==hint_?.02:0):edge.prior;
        priors.push_back(edge.search_prior);result.search_policy[edge.action]=edge.search_prior;
    }
    int cap=options.max_visits?options.max_visits:settings_.max_visits;
    int budget=std::min(cap?std::max(0,cap-1):settings_.simulations,std::max(0,max_playouts-1));
    double max_time=options.max_time<0?settings_.max_time:options.max_time;
    auto effective=gumbel_;if(options.remove_root_noise)effective.noise_scale=0;
    GumbelRoot schedule(priors,budget,effective,random_);
    std::mutex mutex;std::condition_variable changed;int completed=0;int64_t pending=0;bool stopped=false;std::exception_ptr error;
    auto worker=[&] {
        int reserved=-1;bool path_reserved=false;std::vector<std::pair<Node*,size_t>> path;
        try {
            for(;;) {
                Node* leaf=nullptr;Node* parent=nullptr;int action=-1;
                {std::unique_lock<std::mutex> lock(mutex);
                    for(;;) {
                        if(error||stopped||schedule.done())return;
                        if((options.should_stop&&options.should_stop())||(completed+1>=2&&max_time<1e12&&
                            std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>=max_time)) {
                            stopped=true;changed.notify_all();return;
                        }
                        reserved=schedule.reserve(gumbel_completed_q(node_children(*root_),root_->initial[0]-root_->initial[2],gumbel_));
                        if(reserved>=0)break;
                        changed.wait_for(lock,std::chrono::milliseconds(2));
                    }
                    // This root reservation survives waits for an in-flight latent expansion.
                    for(;;) {
                        if(error){schedule.cancel(reserved);reserved=-1;return;}
                        path.clear();Node* current=root_;size_t which=size_t(reserved);
                        for(;;) {
                            auto& edge=current->edges[which];
                            if(edge.child&&!edge.child->ready)break;
                            path.push_back({current,which});
                            if(!edge.child){edge.child=node();current->expansion_order.push_back(which);leaf=edge.child;parent=current;action=edge.action;break;}
                            current=edge.child;
                            if constexpr(Full)which=size_t(gumbel_interior_selection(node_children(*current),current->initial[0]-current->initial[2],gumbel_));
                            else which=select(*current,false);
                            if(which==current->edges.size())break;
                        }
                        if(leaf){for(auto [p,i]:path){++p->edges[i].pending;++pending;}path_reserved=true;break;}
                        changed.wait_for(lock,std::chrono::milliseconds(2));
                    }
                }
                auto output=evaluator_.recurrent(parent->latent,mapping_[action]);
                {std::lock_guard<std::mutex> lock(mutex);
                    expand(*leaf,std::move(output),board_actions_,false);
                    for(auto it=path.rbegin();it!=path.rend();++it)recompute(*it->first);
                    for(auto [p,i]:path){--p->edges[i].pending;--pending;}
                    path.clear();path_reserved=false;schedule.commit(reserved);reserved=-1;++completed;
                }
                changed.notify_all();
            }
        } catch(...) {
            std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();
            if(reserved>=0){if(path_reserved)for(auto [p,i]:path){--p->edges[i].pending;--pending;}schedule.cancel(reserved);}
            changed.notify_all();
        }
    };
    std::vector<std::thread> threads;
    try{for(int i=1;i<std::min(settings_.threads,budget);++i)threads.emplace_back(worker);worker();}
    catch(...){std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();changed.notify_all();}
    for(auto& thread:threads)thread.join();
    if(pending||schedule.pending())throw std::runtime_error("Gumbel MuZero leaked pending visits");
    if(error)std::rethrow_exception(error);
    if(!stopped&&completed!=budget)throw std::runtime_error("Gumbel MuZero budget mismatch");
    result.simulations=completed;result.new_playouts=completed+1;result.root_visits=root_->stats.visits;result.stopped_early=stopped;
    result.network_wdl=root_->initial;result.network_sample_weight=root_->initial_weight;result.network_value_stdev=root_->stdev;
    auto p=root_->stats;result.value=p.value;result.search_weight=p.weight;result.search_weight_sq=p.weight_sq;
    result.search_wdl={(1-p.draw+p.value)/2,p.draw,(1-p.draw-p.value)/2};auto children=node_children(*root_);
    for(size_t i=0;i<root_actions.size();++i) {
        int a=root_actions[i];result.visits[a]=result.q_visits[a]=children[i].visits;
        if(children[i].visits>0)result.q_values[a]=float(children[i].q);
    }
    finish_gumbel_result(result,root_actions,children,root_->initial[0]-root_->initial[2],schedule,gumbel_,temperature,random_);
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return result;
}
template class GumbelSearch<false>;
template class GumbelSearch<true>;
}
