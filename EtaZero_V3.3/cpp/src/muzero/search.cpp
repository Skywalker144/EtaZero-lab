#include "etazero/muzero/search.h"
#include "etazero/symmetry.h"
#include "etazero/time_budget.h"
#include "etazero/gumbel.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace etazero::muzero {
Search::Search(Evaluator& evaluator,SearchSettings settings,uint64_t seed)
    :evaluator_(evaluator),settings_(settings),random_(seed) {
    validate_search_controls(settings);
    if(settings.avoid_repeated_pattern_utility!=0 || settings.external_patterns)throw std::runtime_error("Pattern bonuses require a real-board AlphaZero tree; MuZero latent search has no local board patterns");
    if(settings.use_eval_cache)throw std::runtime_error("MuZero has no real-board graph evaluation cache");
    if(settings.subtree_value_bias_factor!=0)
        throw std::runtime_error("MuZero latent search does not support real-board subtree value bias; set subtree_value_bias_factor=0");
    if(settings.graph_search||settings.reuse_tree||settings.root_symmetries!=1)
        throw std::runtime_error("MuZero requires graph_search=false, reuse_tree=false, root_symmetries=1");
    if(!std::isfinite(settings.wide_root_noise)||settings.wide_root_noise<0||settings.wide_root_noise>5||
       !std::isfinite(settings.futile_visits_threshold)||settings.futile_visits_threshold<0||settings.futile_visits_threshold>1||
       settings.visit_cap_contempt<0||settings.visit_cap_contempt==1||settings.visit_cap_contempt>SEARCH_CONFIG_LIMIT||
       std::abs(settings.visit_cap_contempt_player)>1)throw std::runtime_error("Invalid root exploration or visit cap settings");
    if(settings.threads<1)throw std::runtime_error("MuZero requires positive search threads");
    for(int i=0;i<settings.threads;++i)worker_random_.emplace_back(seed+i);
    try {for(int i=1;i<settings.threads;++i)workers_.emplace_back(&Search::worker_loop,this,i);}
    catch(...) {{std::lock_guard<std::mutex> lock(work_mutex_);shutdown_=true;}work_changed_.notify_all();for(auto& t:workers_)t.join();throw;}
}
Search::~Search() {
    {std::lock_guard<std::mutex> lock(work_mutex_);shutdown_=true;}
    work_changed_.notify_all();for(auto& thread:workers_)thread.join();
}
void Search::worker_loop(int index) {
    uint64_t generation=0;std::unique_lock<std::mutex> lock(work_mutex_);
    for(;;) {
        work_changed_.wait(lock,[&]{return shutdown_||work_generation_!=generation;});
        if(shutdown_)return;generation=work_generation_;auto work=work_;
        lock.unlock();work(index);lock.lock();if(--work_pending_==0)work_done_.notify_one();
    }
}
void Search::dispatch(const std::function<void(int)>& work,int active) {
    {std::lock_guard<std::mutex> lock(work_mutex_);work_=[&,active](int i){if(i<active)work(i);};work_pending_=workers_.size();++work_generation_;}
    work_changed_.notify_all();work(0);
    std::unique_lock<std::mutex> lock(work_mutex_);work_done_.wait(lock,[&]{return work_pending_==0;});work_={};
}
Search::Node* Search::node(){std::lock_guard<std::mutex> lock(allocation_mutex_);nodes_.push_back(std::make_unique<Node>());return nodes_.back().get();}
void Search::expand(Node& node,InferenceOutput output,const std::vector<int>& actions,bool root) {
    auto& e=output.evaluation;restore_evaluation(e,mapping_);
    if(!output.latent||actions.empty())throw std::runtime_error("MuZero expansion requires latent and board actions");
    double optimism=root?settings_.root_policy_optimism:settings_.policy_optimism;
    auto policy=source_policy_probabilities(e,actions,settings_.nn_policy_temperature,optimism);
    node.edges.reserve(actions.size());
    for(int a:actions)node.edges.push_back({a,policy[a],policy[a]});
    node.initial=e.wdl;node.initial_weight=uncertainty_weight(e.shortterm_value_stdev,e.has_auxiliary,settings_);
    node.variance_time=output.evaluation.variance_time;node.stdev=e.shortterm_value_stdev;node.latent=std::move(output.latent);
    double value=e.value();node.stats={1,value,value*value,e.wdl[1],node.initial_weight,node.initial_weight*node.initial_weight,value,e.wdl[3]};
}
std::vector<RootChildStats> Search::children(const Node& node,bool aggregation) const {
    std::lock_guard<std::recursive_mutex> lock(node.mutex);
    std::vector<RootChildStats> result;
    for(size_t i:node.expansion_order){const auto& edge=node.edges[i];if(!edge.child||!edge.child->ready)continue;const auto& s=edge.child->snapshot();
        double factor=double(edge.visits)/std::max<int64_t>(1,s.visits);if(edge.visits<=0)continue;
        result.push_back({edge.search_prior,-s.value*s.weight*factor,s.value_sq*s.weight*factor,edge.visits,s.weight*factor,s.weight_sq*factor*(aggregation?factor:1),s.draw*s.weight*factor,-s.win_loss*s.weight*factor,s.no_result*s.weight*factor});}
    return result;
}
void Search::recompute(Node& node) {
    std::lock_guard<std::recursive_mutex> lock(node.mutex);
    if(settings_.visit_cap_contempt>0 && node.player==capped_player_ && node.capped_distribution.empty()) {
        int64_t visits=1;double sum=0;std::vector<double> fractions(node.edges.size(),0);
        for(size_t i=0;i<node.edges.size();++i)if(auto* child=node.edges[i].child;child && child->ready){visits+=node.edges[i].visits;fractions[i]=child->snapshot().weight*double(node.edges[i].visits)/std::max<int64_t>(1,child->snapshot().visits);sum+=fractions[i];}
        if(visits>=settings_.visit_cap_contempt && sum>0){for(auto& f:fractions)f/=sum;node.capped_distribution=std::move(fractions);}
    }
    node.publish(aggregate_values(node.initial,children(node,true),settings_,&node==root_&&!remove_noise_&&settings_.noise_fraction>0,node.initial_weight));
}
void Search::update_after_playout(Node& node) {
    if(node.dirty_count.fetch_add(1,std::memory_order_acq_rel)>0)return;
    int count=1;
    for(;;) {recompute(node);int remaining=node.dirty_count.fetch_sub(count,std::memory_order_acq_rel)-count;if(!remaining)return;count=remaining;}
}
bool Search::allowed(const Node& node,int action,int depth) const {
    const auto& avoid=avoid_[node.player==1?0:1];
    return (avoid.empty()||avoid[action]<=depth) && (depth!=0||representatives_.empty()||representatives_[action]==action);
}
void Search::expand_human(Node& node,Evaluation e) {
    restore_evaluation(e,mapping_);
    double optimism=&node==root_?settings_.root_policy_optimism:settings_.policy_optimism;
    std::vector<int> actions;for(const auto& edge:node.edges)actions.push_back(edge.action);
    auto policy=source_policy_probabilities(e,actions,settings_.nn_policy_temperature,optimism);
    for(auto& edge:node.edges)edge.human_prior=policy[edge.action];
}
std::vector<GumbelChild> Search::gumbel_children(const Node& node,int depth) const {
    std::lock_guard<std::recursive_mutex> lock(node.mutex);
    std::vector<GumbelChild> result;
    for(const auto& edge:node.edges){GumbelChild c;c.prior=allowed(node,edge.action,depth)?edge.search_prior:-1;c.pending=edge.pending;c.visits=edge.visits;
        if(edge.child&&edge.child->ready)c.q=-edge.child->snapshot().value;result.push_back(c);}
    return result;
}
Search::Selection Search::select(Node& node,int depth,std::mt19937_64& random,double remaining,const GumbelSettings* gumbel,bool full,int forced) {
    bool root=depth==0;
    if(root)if(auto focus=focus_snapshot();focus && std::uniform_real_distribution<double>()(random)<focus->probability){
        std::vector<double> weights(node.edges.size(),0);double sum=0;
        for(size_t j=0;j<focus->moves.size();++j){int action=focus->moves[j];if(action>=int(representatives_.size()))continue;
            action=representatives_[action];for(size_t i=0;i<node.edges.size();++i)if(node.edges[i].action==action&&allowed(node,action,depth)){weights[i]+=focus->weights[j];sum+=focus->weights[j];}}
        if(sum>0)return {std::discrete_distribution<size_t>(weights.begin(),weights.end())(random),true,true};
    }
    if(root&&forced>=0)return {size_t(forced)};
    double total=0,visited=0,max_weight=0;bool human=false,weightless=false;
    if(node.capped_distribution.empty()&&!node.edges.empty()&&node.edges[0].human_prior>=0){
        double wl=root?settings_.human_sl_root_weightless:node.player==root_player_?settings_.human_sl_player_weightless:settings_.human_sl_opponent_weightless;
        double wf=root?settings_.human_sl_root_weightful:node.player==root_player_?settings_.human_sl_player_weightful:settings_.human_sl_opponent_weightful;
        if(wl+wf>0){double r=std::uniform_real_distribution<double>()(random);human=r<wl+wf;weightless=r<wl;}}
    if(full && !root && !human && node.capped_distribution.empty()) {
        auto cs=gumbel_children(node,depth);
        if(std::none_of(cs.begin(),cs.end(),[](const auto& c){return c.prior>=0;}))return {node.edges.size()};
        return {size_t(gumbel_interior_selection(cs,node.initial[0]-node.initial[2],*gumbel))};
    }
    auto weight=[&](const Edge& e){return e.child&&e.child->ready?e.child->snapshot().weight*(weightless?1:double(e.visits)/std::max<int64_t>(1,e.child->snapshot().visits)):0;};
    auto prior=[&](const Edge& e){return human?e.human_prior:e.search_prior;};
    for(const auto& e:node.edges)if(e.child&&e.child->ready){double w=weight(e);total+=w;max_weight=std::max(max_weight,w);visited+=prior(e);}
    if(!node.capped_distribution.empty()) {
        double total_weight=total,fraction=0,best=-1e100;size_t choice=node.edges.size();
        for(size_t i:node.expansion_order){total_weight+=node.edges[i].pending*settings_.virtual_loss;if(allowed(node,node.edges[i].action,depth))fraction+=node.capped_distribution[i];}
        for(size_t i:node.expansion_order)if(node.capped_distribution[i]>0&&allowed(node,node.edges[i].action,depth)){auto& e=node.edges[i];
            double deficit=node.capped_distribution[i]/fraction*(total_weight+node.stats.weight/std::max<int64_t>(1,node.stats.visits))-weight(e)-e.pending*settings_.virtual_loss;
            if(deficit>best){best=deficit;choice=i;}}
        if(choice<node.edges.size())return {choice};
    }
    auto settings=settings_;
    if(human){settings.c_puct=settings_.human_sl_cpuct_exploration+settings_.human_sl_cpuct_permanent*std::sqrt(total);settings.c_puct_log=0;settings.c_puct_stdev_scale=0;}
    double reduction=root&&!remove_noise_?settings_.root_fpu_reduction_max:settings_.fpu_reduction_max;
    double fpu=settings_.use_fpu?fpu_value(node.initial[0]-node.initial[2],node.stats.value,visited,settings_.fpu_parent_power,reduction,settings_.fpu_parent_weight_by_visited_policy,settings_.fpu_parent_weight):0;
    if(settings_.use_fpu)fpu+=(-1-fpu)*(root&&!remove_noise_?settings_.root_fpu_loss_prop:settings_.fpu_loss_prop);
    double best=-std::numeric_limits<double>::infinity();size_t choice=node.edges.size();
    auto consider=[&](size_t i){const auto& e=node.edges[i];if(!allowed(node,e.action,depth))return;
        const ValueStats q=e.child&&e.child->ready?e.child->snapshot():ValueStats{};double w=weight(e);
        double score=root&&!weightless?root_explore_score(prior(e),w>0?-q.value:fpu,w,e.pending,e.visits,q.visits,total,max_weight,node.stats,settings,!remove_noise_&&e.child,e.action==hint_,remaining,random):
            child_selection_score(prior(e),w>0?-q.value:fpu,w,e.pending,total,node.stats,settings,false);
        if(score>best){best=score;choice=i;}};
    for(size_t i:node.expansion_order)consider(i);
    size_t unseen=node.edges.size();double max_prior=-1;
    for(size_t i=0;i<node.edges.size();++i)if(!node.edges[i].child&&allowed(node,node.edges[i].action,depth)){
        if(root&&!weightless&&node.edges[i].action==hint_)consider(i);
        if(prior(node.edges[i])>max_prior){max_prior=prior(node.edges[i]);unseen=i;}}
    if(unseen<node.edges.size())consider(unseen);
    return {choice,weightless,false};
}
void Search::mix_human(SearchResult& result) const {
    std::lock_guard<std::recursive_mutex> lock(root_->mutex);
    if(!human_evaluator()||settings_.human_sl_chosen_move_proportion<=0)return;
    std::vector<double> human(result.policy.size(),-1),utilities(result.policy.size(),0);std::vector<bool> known(result.policy.size(),false);
    for(const auto& edge:root_->edges)if(allowed(*root_,edge.action,0)){human[edge.action]=edge.human_prior;if(edge.child&&edge.child->ready){utilities[edge.action]=-edge.child->snapshot().value;known[edge.action]=true;}}
    auto cs=children(*root_);double scale=1;if(!cs.empty()){auto weights=root_selection_weights(cs,settings_,settings_.use_lcb,root_->stats,false);scale=std::accumulate(weights.begin(),weights.end(),0.0);}
    mix_secondary_policy(result,human,utilities,known,settings_,pass_,scale);
}
SearchResult Search::analysis_snapshot(int actions,const GumbelSettings* gumbel) const {
    std::lock_guard<std::recursive_mutex> lock(root_->mutex);
    SearchResult out;out.policy.assign(actions,0);out.move_policy=out.network_policy=out.search_policy=out.policy;out.visits.assign(actions,0);
    auto p=root_->stats;out.value=p.win_loss;out.utility=p.value;out.search_weight=p.weight;out.search_wdl=wdl_from_value(p.value,p.draw,p.no_result);out.network_wdl=root_->initial;out.root_visits=p.visits;
    out.network_sample_weight=root_->initial_weight;out.network_value_stdev=root_->stdev;out.network_variance_time=root_->variance_time;
    if(human_evaluator()){out.human_policy.assign(actions,-1);for(const auto& e:root_->edges)out.human_policy[e.action]=e.human_prior;}
    std::vector<int> visited;auto stats=children(*root_);
    for(size_t i:root_->expansion_order){const auto& e=root_->edges[i];if(e.child&&e.child->ready&&e.visits>0)visited.push_back(e.action);}
    for(const auto& e:root_->edges) {
        if(!allowed(*root_,e.action,0))continue;
        AnalysisCandidate c;c.action=e.action;c.pv.push_back(e.action);out.network_policy[e.action]=e.prior;out.search_policy[e.action]=e.search_prior;
        if(e.child&&e.child->ready) {
            auto q=e.child->snapshot();c.node_visits=q.visits;c.node_weight=q.weight;c.pv_visits.push_back(q.visits);c.pv_edge_visits.push_back(e.visits);c.has_value=q.visits>0&&q.weight>0;c.weight=q.weight;c.utility=c.win_loss=-q.value;c.wdl=wdl_from_value(-q.value,q.draw,q.no_result);out.visits[e.action]=e.visits;
            const Node* current=e.child;
            while(c.pv.size()<size_t(analysis_pv_length_)) {
                std::lock_guard<std::recursive_mutex> child_lock(current->mutex);
                const Edge* best=nullptr;int64_t count=0;
                for(const auto& next:current->edges)if(next.child&&next.child->ready&&next.child->snapshot().visits>count){count=next.child->snapshot().visits;best=&next;}
                if(!best)break;
                c.pv.push_back(best->action);c.pv_visits.push_back(count);c.pv_edge_visits.push_back(best->visits);current=best->child;
            }
        }
        out.analysis.push_back(std::move(c));
    }
    analysis_policy(out,visited,stats,settings_,p,gumbel);mix_human(out);return out;
}
SearchResult Search::run(const Game& game,double temperature,SearchRun options) {
    return run_impl(game,temperature,std::move(options),nullptr,false);
}
SearchResult Search::run_impl(const Game& game,double temperature,SearchRun options,const GumbelSettings* gumbel,bool full) {
    analysis_pv_length_=options.analysis_pv_length;
    if(!std::isfinite(temperature)||temperature<0||options.max_visits<0||options.max_playouts<-1||options.max_time<-1||!std::isfinite(options.max_time))throw std::runtime_error("Invalid MuZero search caps/temperature");
    if(options.hint_action<-1||(options.hint_action>=0&&!game.legal(options.hint_action)))throw std::runtime_error("Illegal MuZero root hint");
    auto focus=focus_snapshot();advance(0);std::atomic_store(&focus_,focus);
    auto start=std::chrono::steady_clock::now();SearchResult result;int area=game.actions();result.gumbel=gumbel!=nullptr;
    result.policy.resize(area);result.move_policy.resize(area);result.network_policy.resize(area);result.search_policy.resize(area);result.visits.resize(area);result.q_values.resize(area);result.q_visits.resize(area);
    if(game.finished()){double v=game.terminal_value();result.value=v;result.network_wdl=result.search_wdl=wdl_from_value(v,game.terminal_draw_probability(),game.terminal_no_result_probability());return result;}
    avoid_=options.avoid_until;for(auto& a:avoid_){if(a.empty())a.assign(area,0);if(int(a.size())!=area)throw std::runtime_error("Avoid action count differs");for(int d:a)if(d<0)throw std::runtime_error("Negative avoid depth");}
    root_player_=game.player();pass_=game.rule()==Rule::GO?game.canvas_area():-1;
    representatives_.resize(area);std::iota(representatives_.begin(),representatives_.end(),0);
    if(settings_.root_symmetry_pruning)representatives_=game.root_symmetry_representatives(avoid_[root_player_==1?0:1]);
    int64_t max_playouts=options.max_playouts<0?(options.pondering?settings_.max_playouts_pondering:settings_.max_playouts):options.max_playouts;
    if(!max_playouts||(options.should_stop&&options.should_stop())){result.stopped_early=max_playouts!=0;return result;}
    auto obs=game.observation();if(settings_.ignore_all_history||settings_.ignore_pre_root_history)limit_history_features(obs,game.canvas(),game.spec(),0);
    int symmetry=settings_.nn_randomize?std::uniform_int_distribution<int>(0,game.spec().symmetries-1)(random_):settings_.nn_symmetry;
    auto spatial=symmetry_mapping(game.canvas(),input_symmetry(obs,game.spec(),symmetry));mapping_=policy_mapping(spatial,game.spec());
    board_actions_=game.static_actions();std::vector<int> root_actions;for(int a:board_actions_)if(game.legal(a))root_actions.push_back(a);
    root_=node();root_->player=game.player();remove_noise_=options.remove_root_noise;hint_=options.hint_action;
    capped_player_=settings_.visit_cap_contempt==0?0:settings_.visit_cap_contempt_player;
    if(settings_.visit_cap_contempt>0&&capped_player_==0)capped_player_=-(options.search_for_player?options.search_for_player:game.player());
    auto input=transform_observation(obs,game.canvas(),spatial,game.spec());auto initial=evaluator_.initial(input);
    if(options.collect_root_policy_invalid_mass){const auto& logits=initial.evaluation.logits;double maximum=-1e100,total=0,invalid=0;
        for(int a:board_actions_)maximum=std::max(maximum,logits.at(mapping_[a]));
        for(int a:board_actions_){double p=std::exp((logits.at(mapping_[a])-maximum)*initial.evaluation.policy_output_scale);total+=p;if(!game.legal(a))invalid+=p;}
        if(!std::isfinite(total)||total<=0)throw std::runtime_error("Invalid root policy mass");result.root_policy_invalid_mass=invalid/total;}
    expand(*root_,std::move(initial),root_actions,true);
    auto* human_latent=dynamic_cast<Evaluator*>(human_evaluator().get());
    bool human_nonroot=settings_.human_sl_player_weightless+settings_.human_sl_player_weightful+settings_.human_sl_opponent_weightless+settings_.human_sl_opponent_weightful>0;
    if(human_nonroot && !human_latent)throw std::runtime_error("MuZero secondary nonroot exploration requires a secondary MuZero latent model");
    if(human_evaluator()){if(human_latent){auto out=human_latent->initial(input);root_->human_latent=std::move(out.latent);expand_human(*root_,std::move(out.evaluation));}
        else expand_human(*root_,human_evaluator()->evaluate_position(input,game.player()));}
    root_->ready.store(true,std::memory_order_release);
    std::vector<double> policy;for(const auto& edge:root_->edges)policy.push_back(edge.prior);
    if(!gumbel){double temp=remove_noise_?1:temperature_at_turn(settings_.root_policy_temperature_early,settings_.root_policy_temperature,settings_.temperature_halflife,game.turn(),game.area());if(temp!=1)policy=policy_temperature_distribution(policy,temp);
        double fraction=remove_noise_?0:settings_.noise_fraction;
        if(fraction>0){auto alpha=settings_.shaped_noise?noise_alpha_distribution(policy):std::vector<double>(policy.size(),1.0/policy.size());std::vector<double> noise(policy.size());double mass;
            do{mass=0;for(size_t i=0;i<noise.size();++i){noise[i]=std::gamma_distribution<double>(settings_.dirichlet_total_concentration*alpha[i],1)(random_);mass+=noise[i];}}while(mass==0);
            for(size_t i=0;i<policy.size();++i)policy[i]=static_cast<float>((1-fraction)*policy[i]+fraction*noise[i]/mass);}}
    if(hint_>=0){float mass=0;for(auto& p:policy){float f=p;mass+=f*.02f;p=float(f*.98f);}for(size_t i=0;i<root_->edges.size();++i)if(root_->edges[i].action==hint_)policy[i]=float(float(policy[i])+mass);}
    if(options.avoid_rescale_root){double mass=0;for(size_t i=0;i<policy.size();++i)if(allowed(*root_,root_->edges[i].action,0))mass+=policy[i];if(mass>0)for(auto& p:policy)p=static_cast<float>(p/mass);}
    bool has_allowed=false;
    for(size_t i=0;i<policy.size();++i){auto& edge=root_->edges[i];edge.search_prior=policy[i];int a=edge.action;result.network_policy[a]=edge.prior;result.search_policy[a]=policy[i];has_allowed|=allowed(*root_,a,0);}
    if(!has_allowed){result.new_playouts=result.root_visits=1;return result;}
    int64_t cap=options.max_visits?options.max_visits:(options.pondering?settings_.max_visits_pondering:settings_.max_visits);
    int64_t budget=std::min(cap?std::max<int64_t>(0,cap-1):settings_.simulations,std::max<int64_t>(0,max_playouts-1));
    std::unique_ptr<GumbelRoot> schedule;
    if(gumbel){auto effective=*gumbel;if(remove_noise_)effective.noise_scale=0;auto priors=policy;for(size_t i=0;i<priors.size();++i)if(!allowed(*root_,root_->edges[i].action,0))priors[i]=-1;schedule=std::make_unique<GumbelRoot>(priors,budget,effective,random_);}
    options.turn=game.turn();options.board_area=game.area();options.occupied_points=std::count_if(game.board().cells.begin(),game.board().cells.end(),[](int c){return c!=0;});
    SearchTimeBudget time_budget(settings_,options,0);std::mutex control;std::condition_variable changed;
    int64_t completed=0;std::atomic<int64_t> pending{0};bool stopped=false;std::exception_ptr error;auto next_report=start;
    auto update_clock=[&]{double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();if(time_budget.needs_update(elapsed)){auto stats=children(*root_);auto root_stats=root_->snapshot();if(settings_.human_sl_chosen_move_proportion>0||stats.empty()){auto selection=analysis_snapshot(area);time_budget.update(elapsed,root_stats.visits,stats,policy,root_stats,temperature,&selection);}else time_budget.update(elapsed,root_stats.visits,stats,policy,root_stats,temperature);}};
    struct Visit{Node* parent;size_t index;bool weightless;};
    auto worker=[&](int worker_index){int reserved=-1;std::vector<Visit> path;auto& random=worker_random_[worker_index];
        auto release=[&]{for(auto visit:path){std::lock_guard<std::recursive_mutex> lock(visit.parent->mutex);--visit.parent->edges[visit.index].pending;--pending;}path.clear();};
        try {for(;;) {
            double remaining;
            {std::unique_lock<std::mutex> lock(control);
                for(;;) {
                    update_clock();if(error||stopped||(schedule?schedule->done():completed>=budget))return;
                    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                    if((options.should_stop&&options.should_stop())||(completed+1>=2&&elapsed>=time_budget.limit())){stopped=true;changed.notify_all();return;}
                    if(schedule){reserved=schedule->reserve(gumbel_completed_q(gumbel_children(*root_,0),root_->initial[0]-root_->initial[2],*gumbel));if(reserved<0){changed.wait_for(lock,std::chrono::milliseconds(2));continue;}}
                    remaining=std::min(double(budget-completed),time_budget.visits_left());break;
                }
            }
            Node* leaf=nullptr;Node* parent=nullptr;int action=-1;bool blocked=false,focused=false;
            Node* current=root_;
            for(int depth=0;;++depth) {
                if(options.should_stop&&options.should_stop())break;
                std::lock_guard<std::recursive_mutex> lock(current->mutex);
                auto selection=select(*current,depth,random,remaining,gumbel,full,depth==0?reserved:-1);
                if(selection.index==current->edges.size()) {
                    bool allowed_move=false;for(const auto& edge:current->edges)allowed_move|=allowed(*current,edge.action,depth);
                    if(!allowed_move){leaf=current;blocked=true;}break;
                }
                auto& edge=current->edges[selection.index];
                if(edge.child&&!edge.child->ready.load(std::memory_order_acquire))break;
                focused|=selection.focus;path.push_back({current,selection.index,selection.weightless});++edge.pending;++pending;
                if(!edge.child){edge.child=node();edge.child->player=-current->player;current->expansion_order.push_back(selection.index);leaf=edge.child;parent=current;action=edge.action;break;}
                current=edge.child;
            }
            if(!leaf) {
                release();{std::lock_guard<std::mutex> lock(control);if(reserved>=0){schedule->cancel(reserved,true);reserved=-1;}}
                changed.notify_all();std::this_thread::yield();continue;
            }
            if(blocked) {
                std::lock_guard<std::recursive_mutex> lock(leaf->mutex);auto q=leaf->snapshot();++q.visits;q.weight+=leaf->initial_weight;q.weight_sq+=leaf->initial_weight*leaf->initial_weight;leaf->publish(q);
            } else {
                auto output=evaluator_.recurrent(parent->latent,mapping_[action]);
                InferenceOutput human_output;if(human_nonroot)human_output=human_latent->recurrent(parent->human_latent,mapping_[action]);
                expand(*leaf,std::move(output),board_actions_,false);
                if(human_nonroot){leaf->human_latent=std::move(human_output.latent);expand_human(*leaf,std::move(human_output.evaluation));}
                leaf->ready.store(true,std::memory_order_release);
            }
            bool update=true;
            for(auto it=path.rbegin();it!=path.rend();++it) {
                if(it->weightless)update=false;
                if(update){{std::lock_guard<std::recursive_mutex> lock(it->parent->mutex);++it->parent->edges[it->index].visits;}update_after_playout(*it->parent);}
            }
            release();
            {std::lock_guard<std::mutex> lock(control);
                if(schedule){if(focused)schedule->cancel(reserved,true);else schedule->commit(reserved);reserved=-1;}
                if(!focused)++completed;
                if(options.on_report){auto now=std::chrono::steady_clock::now();if(now>=next_report){next_report=now+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(options.report_interval));auto report=analysis_snapshot(area,gumbel);report.seconds=std::chrono::duration<double>(now-start).count();options.on_report(report);}}
            }changed.notify_all();
        }} catch(...) {
            release();std::lock_guard<std::mutex> lock(control);if(!error)error=std::current_exception();if(reserved>=0)schedule->cancel(reserved);changed.notify_all();
        }
    };
    int active_threads=settings_.min_playouts_per_thread>0?std::max(1,static_cast<int>(std::min(double(settings_.threads),std::floor((budget+1)/settings_.min_playouts_per_thread)))):settings_.threads;
    dispatch(worker,active_threads);
    if(pending||(schedule&&schedule->pending()))throw std::runtime_error("MuZero leaked pending visits");if(error)std::rethrow_exception(error);
    if(schedule&&!stopped&&completed!=budget)throw std::runtime_error("MuZero search budget mismatch");
    result.simulations=0;result.new_playouts=completed+1;result.root_visits=root_->stats.visits;result.stopped_early=stopped;result.network_wdl=root_->initial;result.network_sample_weight=root_->initial_weight;result.network_value_stdev=root_->stdev;result.network_variance_time=root_->variance_time;
    auto p=root_->stats;result.value=p.win_loss;result.utility=p.value;result.search_weight=p.weight;result.search_weight_sq=p.weight_sq;result.search_wdl=wdl_from_value(p.win_loss,p.draw,p.no_result);
    std::vector<int> actions;auto stats=children(*root_);
    for(size_t i:root_->expansion_order){auto& edge=root_->edges[i];if(!edge.child||!edge.child->ready)continue;result.visits[edge.action]=edge.visits;result.simulations+=edge.visits;result.q_visits[edge.action]=edge.child->snapshot().visits;result.q_values[edge.action]=-edge.child->snapshot().win_loss;if(edge.visits>0)actions.push_back(edge.action);}
    if(gumbel)finish_gumbel_result(result,root_actions,gumbel_children(*root_,0),root_->initial[0]-root_->initial[2],*schedule,*gumbel,temperature,random_);
    else if(actions.empty()){result.policy=result.search_policy;bool allocated=std::any_of(result.q_visits.begin(),result.q_visits.end(),[](auto n){return n>0;});for(int a=0;a<area;++a)if(!allowed(*root_,a,0)||(allocated&&result.q_visits[a]==0))result.policy[a]=0;result.policy=direct_policy_selection(std::move(result.policy),settings_);result.move_policy=result.policy;result.policy_target=quantize_policy(result.policy);}
    else {auto move=root_selection_weights(stats,settings_,settings_.use_lcb&&!options.training,p);auto target=root_selection_weights(stats,settings_,settings_.use_lcb,p,false);auto quantized=quantize_policy(target);double mass=std::accumulate(target.begin(),target.end(),0.0);result.policy_target.resize(area);
        for(size_t i=0;i<actions.size();++i){int a=actions[i];result.policy[a]=target[i]/mass;result.move_policy[a]=move[i];result.policy_target[a]=quantized[i];if(result.policy[a]>0)result.policy_surprise+=result.policy[a]*std::log(result.policy[a]/std::max(1e-100,stats[i].prior));}result.policy_surprise=std::max(0.0,result.policy_surprise);}
    mix_human(result);
    if(!gumbel||settings_.human_sl_chosen_move_proportion>0){auto behavior=temperature_distribution(result.move_policy,temperature,settings_.chosen_move_temperature_only_below_prob);result.action=std::discrete_distribution<int>(behavior.begin(),behavior.end())(random_);}
    if(options.collect_analysis){auto report=analysis_snapshot(area,gumbel);result.analysis=std::move(report.analysis);result.human_policy=std::move(report.human_policy);result.pruned_value=report.pruned_value;result.pruned_utility=report.pruned_utility;result.pruned_wdl=report.pruned_wdl;}
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return result;
}
}

namespace etazero::muzero {
SearchResult Search::inspect(const Game& game,const std::vector<int>& branch) {
    Node* node=root_;
    for(int a:branch){if(!node)break;Node* next=nullptr;for(const auto& edge:node->edges)if(edge.action==a){next=edge.child;break;}node=next;}
    if(!node||!node->ready)throw std::runtime_error("No searched node for branch");
    auto old=root_;int player=root_player_;root_=node;root_player_=game.player();
    try {auto out=analysis_snapshot(game.actions());root_=old;root_player_=player;return out;}
    catch(...){root_=old;root_player_=player;throw;}
}
}
