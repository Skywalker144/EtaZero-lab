#include "etazero/gumbel_search.h"
#include <algorithm>
#include <cmath>
namespace etazero {
SearchSettings gumbel_search_settings(SearchSettings s) {
    if(s.graph_search||s.reuse_tree)throw std::runtime_error("Gumbel requires graph_search=false and reuse_tree=false");
    // These are PUCT root mechanisms. Non-root PUCT and value aggregation remain configurable.
    s.noise_fraction=0;s.forced_playouts=0;s.root_policy_temperature=s.root_policy_temperature_early=1;
    return s;
}
template<bool Full>
GumbelSearch<Full>::GumbelSearch(Evaluator& e,SearchSettings s,GumbelSettings g,uint64_t seed)
    :Search(e,gumbel_search_settings(s),seed,false),gumbel_(g) {
    gumbel_.validate();
    try{for(int i=1;i<s.threads;++i)workers_.emplace_back(&GumbelSearch::gumbel_worker_loop,this,i);}
    catch(...){ {std::lock_guard<std::mutex> lock(work_mutex_);closing_=true;}work_changed_.notify_all();for(auto& t:workers_)t.join();workers_.clear();throw; }
}
template<bool Full> GumbelSearch<Full>::~GumbelSearch() {
    // Join before the derived scheduler/locks are destroyed; the base owns node cleanup.
    {std::lock_guard<std::mutex> lock(work_mutex_);closing_=true;}
    work_changed_.notify_all();root_changed_.notify_all();for(auto& t:workers_)t.join();workers_.clear();
}
template<bool Full> std::vector<GumbelChild> GumbelSearch<Full>::root_children() {
    std::vector<GumbelChild> children(root_->move_count);
    for(size_t i=0;i<root_->move_count;++i) {
        auto& out=children[i];const auto& move=root_->moves[i];out.prior=move.search_prior;
        int index=move.child_index.load(std::memory_order_acquire);
        if(index>=0){auto& edge=root_->edge(index);out.visits=edge.n.load();
            if(Node* child=edge.child.load()){auto q=snapshot(*child);if(q.weight>0)out.q=edge.perspective.load()*q.value;}}
    }
    return children;
}
template<bool Full> void GumbelSearch<Full>::simulate_gumbel(int worker) {
    int reserved=-1;
    try {
        for(;;) {
            {std::unique_lock<std::mutex> lock(root_mutex_);
                for(;;) {
                    if(failed_||stopped_||schedule_->done())return;
                    if((should_stop_&&should_stop_()) || (completed_.load()+fresh_playouts_>=2 && max_time_<1e12 &&
                        std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count()>=max_time_)) {
                        stopped_=true;root_changed_.notify_all();return;
                    }
                    reserved=schedule_->reserve(gumbel_completed_q(root_children(),root_->initial_value,gumbel_));
                    if(reserved>=0)break;
                    root_changed_.wait_for(lock,std::chrono::milliseconds(2));
                }
            }
            simulation<true,Full>(worker,reserved,&gumbel_);
            {std::lock_guard<std::mutex> lock(root_mutex_);schedule_->commit(reserved);reserved=-1;}
            root_changed_.notify_all();
        }
    } catch(...) {
        {std::lock_guard<std::mutex> lock(root_mutex_);if(reserved>=0)schedule_->cancel(reserved);}
        {std::lock_guard<std::mutex> lock(error_mutex_);if(!error_)error_=std::current_exception();}
        failed_=true;root_changed_.notify_all();
    }
}
template<bool Full> void GumbelSearch<Full>::gumbel_worker_loop(int worker) {
    uint64_t seen=0;
    for(;;) {
        std::unique_lock<std::mutex> lock(work_mutex_);
        work_changed_.wait(lock,[&]{return closing_||generation_!=seen;});if(closing_)return;
        seen=generation_;bool cleaning=cleaning_;lock.unlock();
        if(cleaning)TreeDeleter{}(cleanup_roots_[worker]);else simulate_gumbel(worker);
        lock.lock();++workers_done_;lock.unlock();work_changed_.notify_all();
    }
}
template<bool Full> SearchResult GumbelSearch<Full>::run(const Game& game,double temperature,SearchRun options) {
    options.turn=game.turn();options.board_area=game.size()*game.size();
    AlphaZeroState state(game,*evaluator_);return run(state,temperature,options);
}
template<bool Full> SearchResult GumbelSearch<Full>::run(const SearchState& state,double temperature,SearchRun options) {
    if(!std::isfinite(temperature)||temperature<0||options.max_visits<0||options.max_playouts<-1||options.max_time<-1||!std::isfinite(options.max_time))
        throw std::runtime_error("Invalid Gumbel search caps/temperature");
    if(options.hint_action<-1||(options.hint_action>=0&&!state.legal(options.hint_action)))throw std::runtime_error("Illegal Gumbel root hint");
    start_=std::chrono::steady_clock::now();advance(0);SearchResult result;
    result.policy.assign(state.actions(),0);result.move_policy=result.policy;result.network_policy=result.policy;result.search_policy=result.policy;
    result.visits.assign(state.actions(),0);result.q_values.assign(state.actions(),0);result.q_visits.assign(state.actions(),0);
    if(state.terminal()) {double v=state.terminal_value();result.value=v;result.network_wdl=result.search_wdl={(v+std::abs(v))/2,1-std::abs(v),(std::abs(v)-v)/2};return result;}
    int cap=options.max_playouts<0?settings_.max_playouts:options.max_playouts;
    if(cap==0||(options.should_stop&&options.should_stop())){result.stopped_early=cap!=0;return result;}
    remove_root_noise_=options.remove_root_noise;root_symmetries_=remove_root_noise_?1:settings_.root_symmetries;
    expand(*root_,state,0);fresh_playouts_=1;
    int visit_cap=options.max_visits?options.max_visits:settings_.max_visits;
    budget_=std::min(visit_cap?std::max(0,visit_cap-1):settings_.simulations,std::max(0,cap-1));
    max_time_=options.max_time<0?settings_.max_time:options.max_time;should_stop_=options.should_stop;
    std::vector<double> priors;std::vector<int> actions;
    for(size_t i=0;i<root_->move_count;++i) {
        auto& move=root_->moves[i];actions.push_back(move.action);result.network_policy[move.action]=move.prior;
        move.search_prior=options.hint_action>=0?.98*move.prior+(move.action==options.hint_action?.02:0):move.prior;
        priors.push_back(move.search_prior);result.search_policy[move.action]=move.search_prior;
    }
    auto effective=gumbel_;if(options.remove_root_noise)effective.noise_scale=0;
    schedule_=std::make_unique<GumbelRoot>(priors,budget_,effective,random_[0]);
    {std::lock_guard<std::mutex> lock(work_mutex_);position_=&state;issued_=0;completed_=0;failed_=false;stopped_=false;error_=nullptr;workers_done_=0;cleaning_=false;++generation_;}
    work_changed_.notify_all();simulate_gumbel(0);
    {std::unique_lock<std::mutex> lock(work_mutex_);work_changed_.wait(lock,[&]{return workers_done_==int(workers_.size());});}
    if(pending()!=0||schedule_->pending()!=0)throw std::runtime_error("Gumbel leaked pending visits");
    if(error_)std::rethrow_exception(error_);
    if(!stopped_&&completed_!=budget_)throw std::runtime_error("Gumbel budget mismatch");
    auto children=root_children();auto parent=snapshot(*root_);
    result.simulations=completed_;result.new_playouts=completed_+1;result.root_visits=root_->visits.load()+1;result.stopped_early=stopped_;
    result.network_wdl=root_->initial_wdl;result.network_sample_weight=root_->initial_weight;result.network_value_stdev=root_->initial_stdev;
    result.value=parent.value;result.search_weight=parent.weight;result.search_weight_sq=parent.weight_sq;
    result.search_wdl={(1-parent.draw+parent.value)/2,parent.draw,(1-parent.draw-parent.value)/2};
    for(size_t i=0;i<actions.size();++i) {
        result.visits[actions[i]]=children[i].visits;
        int index=root_->moves[i].child_index.load();
        if(index>=0){auto& edge=root_->edge(index);if(Node* child=edge.child.load()){auto q=snapshot(*child);
            if(q.visits>0){result.q_visits[actions[i]]=q.visits;result.q_values[actions[i]]=float(edge.perspective.load()*q.value);}}}
    }
    finish_gumbel_result(result,actions,children,root_->initial_value,*schedule_,gumbel_,temperature,random_[0]);
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count();return result;
}
template class GumbelSearch<false>;
template class GumbelSearch<true>;
}
