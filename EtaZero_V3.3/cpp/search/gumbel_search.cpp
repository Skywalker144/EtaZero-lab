#include "search/gumbel_search.h"
#include "search/time_budget.h"
#include <algorithm>
#include <cmath>
namespace etazero {
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
    auto eval=root_->evaluated();
    std::vector<GumbelChild> children(root_->move_count);
    for(size_t i=0;i<root_->move_count;++i) {
        auto& out=children[i];const auto& move=root_->moves[i];out.prior=eval->search_prior[i];
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
                    if(worker==0)update_time_budget(std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count());
                    if((should_stop_&&should_stop_()) || (completed_.load()+fresh_playouts_>=2 && time_budget_->limit()<1e12 &&
                        std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count()>=time_budget_->limit())) {
                        stopped_=true;root_changed_.notify_all();return;
                    }
                    reserved=schedule_->reserve(gumbel_completed_q(root_children(),root_->evaluated()->initial_value,gumbel_));
                    if(reserved>=0)break;
                    root_changed_.wait_for(lock,std::chrono::milliseconds(2));
                }
            }
            simulation<true,Full>(worker,reserved,&gumbel_);
            {std::lock_guard<std::mutex> lock(root_mutex_);if(thread_states_[worker].counted_playout)schedule_->commit(reserved);else schedule_->cancel(reserved,true);reserved=-1;}
            root_changed_.notify_all();
            if(worker==0 && report_)report_();
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
        if(cleaning)TreeDeleter{settings_.subtree_value_bias_free_prop}(cleanup_roots_[worker]);else if(worker<active_threads_)simulate_gumbel(worker);
        lock.lock();++workers_done_;lock.unlock();work_changed_.notify_all();
    }
}
template<bool Full> SearchResult GumbelSearch<Full>::run(const Game& game,double temperature,SearchRun options) {
    options.turn=game.turn();options.board_area=game.area();
    options.occupied_points=std::count_if(game.board().cells.begin(),game.board().cells.end(),[](int c){return c!=0;});
    AlphaZeroState state(game,*evaluator_);return run(state,temperature,options);
}
template<bool Full> SearchResult GumbelSearch<Full>::run(const SearchState& state,double temperature,SearchRun options) {
    analysis_pv_length_=options.analysis_pv_length;
    if(!std::isfinite(temperature)||temperature<0||options.max_visits<0||options.max_playouts<-1||options.max_time<-1||!std::isfinite(options.max_time))
        throw std::runtime_error("Invalid Gumbel search caps/temperature");
    if(options.hint_action<-1||(options.hint_action>=0&&!state.legal(options.hint_action)))throw std::runtime_error("Illegal Gumbel root hint");
    root_player_=state.player();
    capped_player_=settings_.visit_cap_contempt==0?0:settings_.visit_cap_contempt_player;
    if(settings_.visit_cap_contempt>0 && capped_player_==0)capped_player_=-(options.search_for_player?options.search_for_player:state.player());
    start_=std::chrono::steady_clock::now();auto focus=focus_snapshot();advance(0);std::atomic_store(&focus_,focus);SearchResult result;result.gumbel=true;
    prepare_restrictions(state,options);
    prepare_patterns(state,options);
    prepare_subtree_bias(state);
    result.policy.assign(state.actions(),0);result.move_policy=result.policy;result.network_policy=result.policy;result.search_policy=result.policy;
    result.visits.assign(state.actions(),0);result.q_values.assign(state.actions(),0);result.q_visits.assign(state.actions(),0);
    if(state.terminal()) {double v=state.terminal_value();result.value=v;result.network_wdl=result.search_wdl=wdl_from_value(v,state.terminal_draw_probability(),state.terminal_no_result_probability());return result;}
    int64_t cap=options.max_playouts<0?(options.pondering?settings_.max_playouts_pondering:settings_.max_playouts):options.max_playouts;
    if(cap==0||(options.should_stop&&options.should_stop())){result.stopped_early=cap!=0;return result;}
    remove_root_noise_=options.remove_root_noise;root_symmetries_=remove_root_noise_?1:settings_.root_symmetries;
    expand(*root_,state,0);fresh_playouts_=1;
    int64_t visit_cap=options.max_visits?options.max_visits:(options.pondering?settings_.max_visits_pondering:settings_.max_visits);
    budget_=std::min(visit_cap?std::max<int64_t>(0,visit_cap-1):settings_.simulations,std::max<int64_t>(0,cap-1));
    active_threads_=settings_.min_playouts_per_thread>0 ? std::max(1,static_cast<int>(std::min(double(settings_.threads),
        std::floor((budget_+1)/settings_.min_playouts_per_thread)))) : settings_.threads;
    max_time_=options.max_time<0?(options.pondering?settings_.max_time_pondering:settings_.max_time):options.max_time;should_stop_=options.should_stop;
    auto eval=std::make_shared<NodeEvaluation>(*root_->evaluated());
    std::vector<double> priors;std::vector<int> actions;
    for(size_t i=0;i<root_->move_count;++i) {
        auto& move=root_->moves[i];actions.push_back(move.action);result.network_policy[move.action]=eval->prior[i];
        eval->search_prior[i]=options.hint_action>=0?.98*eval->prior[i]+(move.action==options.hint_action?.02:0):eval->prior[i];
        if(!move_allowed(move.action,state.player(),0))eval->search_prior[i]=-1;
        priors.push_back(eval->search_prior[i]);result.search_policy[move.action]=std::max(0.0,eval->search_prior[i]);
    }
    root_->publish(eval);
    auto effective=gumbel_;if(options.remove_root_noise)effective.noise_scale=0;
    schedule_=std::make_unique<GumbelRoot>(priors,budget_,effective,random_[0]);

    auto next_report=start_;
    report_=options.on_report?std::function<void()>([&]{
        auto now=std::chrono::steady_clock::now();if(now<next_report)return;
        next_report=now+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(options.report_interval));
        options.on_report(analysis_snapshot(state.actions(),&gumbel_));
    }):std::function<void()>{};
    position_=&state;current_temperature_=temperature;time_budget_=std::make_unique<SearchTimeBudget>(settings_,options,carried_time_);
    update_time_budget(std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count());
    auto actual_search_start=std::chrono::steady_clock::now();
    {std::lock_guard<std::mutex> lock(work_mutex_);position_=&state;issued_=0;completed_=0;failed_=false;stopped_=false;error_=nullptr;workers_done_=0;cleaning_=false;++generation_;}
    work_changed_.notify_all();simulate_gumbel(0);
    {std::unique_lock<std::mutex> lock(work_mutex_);work_changed_.wait(lock,[&]{return workers_done_==int(workers_.size());});}
    if(schedule_->pending()!=0)throw std::runtime_error("Gumbel leaked pending visits");
    if(error_)std::rethrow_exception(error_);
    carried_time_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-actual_search_start).count();
    report_={};
    if(options.collect_analysis){auto report=analysis_snapshot(state.actions(),&gumbel_);result.analysis=std::move(report.analysis);result.human_policy=std::move(report.human_policy);result.pruned_value=report.pruned_value;result.pruned_utility=report.pruned_utility;result.pruned_wdl=report.pruned_wdl;result.gumbel=true;}
    if(!stopped_&&completed_!=budget_)throw std::runtime_error("Gumbel budget mismatch");
    auto children=root_children();auto parent=snapshot(*root_);
    result.simulations=completed_;result.new_playouts=completed_+1;result.root_visits=root_->visits.load()+1;result.stopped_early=stopped_;
    result.network_wdl=eval->initial_wdl;result.network_sample_weight=eval->initial_weight;result.network_value_stdev=eval->initial_stdev;result.network_variance_time=eval->initial_variance_time;
    result.value=parent.win_loss;result.utility=parent.value;result.search_weight=parent.weight;result.search_weight_sq=parent.weight_sq;
    result.search_wdl=wdl_from_value(parent.win_loss,parent.draw,parent.no_result);
    for(size_t i=0;i<actions.size();++i) {
        result.visits[actions[i]]=children[i].visits;
        int index=root_->moves[i].child_index.load();
        if(index>=0){auto& edge=root_->edge(index);if(Node* child=edge.child.load()){auto q=snapshot(*child);
            if(q.visits>0){result.q_visits[actions[i]]=q.visits;result.q_values[actions[i]]=float(edge.perspective.load()*q.win_loss);}}}
    }
    finish_gumbel_result(result,actions,children,eval->initial_value,*schedule_,gumbel_,temperature,random_[0]);
    mix_human_policy(result,state,temperature,options.training);
    if(human_evaluator()&&settings_.human_sl_chosen_move_proportion>0){auto behavior=temperature_distribution(result.move_policy,temperature,settings_.chosen_move_temperature_only_below_prob);result.action=std::discrete_distribution<int>(behavior.begin(),behavior.end())(random_[0]);}
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count();return result;
}
template class GumbelSearch<false>;
template class GumbelSearch<true>;
}
