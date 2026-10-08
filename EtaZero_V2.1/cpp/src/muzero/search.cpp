#include "etazero/muzero/search.h"
#include "etazero/symmetry.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace etazero::muzero {
Search::Search(Evaluator& evaluator,SearchSettings settings,uint64_t seed)
    :evaluator_(evaluator),settings_(settings),random_(seed) {
    if(settings.graph_search||settings.reuse_tree||settings.root_symmetries!=1)
        throw std::runtime_error("MuZero requires graph_search=false, reuse_tree=false, root_symmetries=1");
    if(settings.threads<1)throw std::runtime_error("MuZero requires positive search threads");
}
Search::Node* Search::node(){nodes_.push_back(std::make_unique<Node>());return nodes_.back().get();}
void Search::expand(Node& node,InferenceOutput output,const std::vector<int>& actions,bool root) {
    auto& e=output.evaluation;restore_evaluation(e,mapping_);
    if(!output.latent||actions.empty())throw std::runtime_error("MuZero expansion requires latent and board actions");
    double optimism=root?settings_.root_policy_optimism:settings_.policy_optimism;
    std::vector<double> logits;double maximum=-std::numeric_limits<double>::infinity();
    for(int a:actions){double p=e.has_auxiliary?mixed_policy_logit(e.logits.at(a),e.optimistic_logits.at(a),optimism):e.logits.at(a);logits.push_back(p);maximum=std::max(maximum,p);}
    double mass=0;for(auto& p:logits){p=std::exp((p-maximum)/settings_.nn_policy_temperature);mass+=p;}
    if(!std::isfinite(mass)||mass<=0)throw std::runtime_error("Invalid MuZero policy mass");
    for(size_t i=0;i<actions.size();++i)node.edges.push_back({actions[i],logits[i]/mass,logits[i]/mass});
    node.initial=e.wdl;node.initial_weight=uncertainty_weight(e.shortterm_value_stdev,e.has_auxiliary,settings_);
    node.stdev=e.shortterm_value_stdev;node.latent=std::move(output.latent);
    double value=e.value();node.stats={1,value,value*value,e.wdl[1],node.initial_weight,node.initial_weight*node.initial_weight};node.ready=true;
}
std::vector<RootChildStats> Search::children(const Node& node) const {
    std::vector<RootChildStats> result;
    for(size_t i:node.expansion_order){const auto& edge=node.edges[i];if(!edge.child||!edge.child->ready)continue;const auto& s=edge.child->stats;
        result.push_back({edge.search_prior,-s.value*s.weight,s.value_sq*s.weight,s.visits,s.weight,s.weight_sq,s.draw*s.weight});}
    return result;
}
void Search::recompute(Node& node) {
    node.stats=aggregate_values(node.initial,children(node),settings_,&node==root_&&!remove_noise_&&settings_.noise_fraction>0,node.initial_weight);
}
size_t Search::select(Node& node,bool root) {
    double total=0,visited=0;
    for(const auto& e:node.edges)if(e.child&&e.child->ready){total+=e.child->stats.weight;visited+=e.search_prior;}
    double reduction=root&&!remove_noise_?settings_.root_fpu_reduction_max:settings_.fpu_reduction_max;
    double fpu=settings_.use_fpu?fpu_value(node.initial[0]-node.initial[2],node.stats.value,visited,settings_.fpu_parent_power,reduction,settings_.fpu_parent_weight_by_visited_policy,settings_.fpu_parent_weight):0;
    if(settings_.use_fpu)fpu+=(-1-fpu)*(root&&!remove_noise_?settings_.root_fpu_loss_prop:settings_.fpu_loss_prop);
    double best=-std::numeric_limits<double>::infinity();std::vector<size_t> choices;
    for(size_t i=0;i<node.edges.size();++i) {
        const auto& e=node.edges[i];if(e.child&&!e.child->ready)continue;
        const ValueStats s=e.child?e.child->stats:ValueStats{};
        double score=child_selection_score(e.search_prior,s.weight>0?-s.value:fpu,s.weight,e.pending,total,node.stats,settings_,root&&!remove_noise_&&e.child!=nullptr);
        if(root&&e.action==hint_) {
            double weight=s.weight+e.pending*settings_.virtual_loss;
            double next=(weight+node.stats.weight/std::max<int64_t>(1,node.stats.visits))/(s.visits+1.0);
            for(const auto& other:node.edges)if(other.child&&other.child->ready&&weight+next<other.child->stats.weight*.8)score=1e20;
        }
        if(score>best){best=score;choices.clear();}if(score==best)choices.push_back(i);
    }
    return choices.empty()?node.edges.size():choices[std::uniform_int_distribution<size_t>(0,choices.size()-1)(random_)];
}
SearchResult Search::run(const Game& game,double temperature,SearchRun options) {
    if(!std::isfinite(temperature)||temperature<0||options.max_visits<0||options.max_playouts<-1||options.max_time<-1||!std::isfinite(options.max_time))throw std::runtime_error("Invalid MuZero search caps/temperature");
    if(options.hint_action<-1||(options.hint_action>=0&&!game.legal(options.hint_action)))throw std::runtime_error("Illegal MuZero root hint");
    advance(0);auto start=std::chrono::steady_clock::now();SearchResult result;int area=game.actions();
    result.policy.resize(area);result.move_policy.resize(area);result.network_policy.resize(area);result.search_policy.resize(area);result.visits.resize(area);result.q_values.resize(area);result.q_visits.resize(area);
    if(game.finished()){double v=game.terminal_value();result.value=v;result.network_wdl=result.search_wdl={(v+std::abs(v))/2,1-std::abs(v),(std::abs(v)-v)/2};return result;}
    int max_playouts=options.max_playouts<0?settings_.max_playouts:options.max_playouts;
    if(!max_playouts||(options.should_stop&&options.should_stop())){result.stopped_early=bool(options.should_stop)&&options.should_stop();return result;}
    auto obs=game.observation();
    int symmetry=settings_.nn_randomize?std::uniform_int_distribution<int>(0,game.rule()==Rule::HEX?1:7)(random_):settings_.nn_symmetry;
    mapping_=symmetry_mapping(game.canvas(),input_symmetry(obs,symmetry));board_actions_.clear();std::vector<int> root_actions;
    for(int a=0;a<area;++a){if(obs[a])board_actions_.push_back(a);if(game.legal(a))root_actions.push_back(a);}
    root_=node();remove_noise_=options.remove_root_noise;hint_=options.hint_action;
    auto initial=evaluator_.initial(transform_observation(obs,game.canvas(),mapping_));
    if(options.collect_root_policy_invalid_mass) {
        const auto& logits=initial.evaluation.logits;
        double maximum=-std::numeric_limits<double>::infinity();
        for(int a:board_actions_)maximum=std::max(maximum,logits.at(mapping_[a]));
        double total=0,invalid=0;
        for(int a:board_actions_) {
            double mass=std::exp(logits.at(mapping_[a])-maximum);
            total+=mass;
            if(obs[area+a] || obs[2*area+a])invalid+=mass;
        }
        if(!std::isfinite(total)||total<=0||!std::isfinite(invalid))
            throw std::runtime_error("Invalid root policy diagnostic mass");
        result.root_policy_invalid_mass=invalid/total;
    }
    expand(*root_,std::move(initial),root_actions,true);
    std::vector<double> policy;for(const auto& e:root_->edges)policy.push_back(e.prior);
    double root_temp=remove_noise_?1:temperature_at_turn(settings_.root_policy_temperature_early,settings_.root_policy_temperature,settings_.temperature_halflife,game.turn(),game.size()*game.size());
    if(root_temp!=1)policy=policy_temperature_distribution(policy,root_temp);
    double fraction=remove_noise_?0:settings_.noise_fraction;
    std::vector<double> noise(policy.size());double mass=1;
    if(fraction>0){auto alpha=settings_.shaped_noise?noise_alpha_distribution(policy):std::vector<double>(policy.size(),1.0/policy.size());do{mass=0;for(size_t i=0;i<noise.size();++i){noise[i]=std::gamma_distribution<double>(settings_.dirichlet_total_concentration*alpha[i],1)(random_);mass+=noise[i];}}while(mass==0);}
    for(size_t i=0;i<policy.size();++i)root_->edges[i].search_prior=(1-fraction)*policy[i]+fraction*noise[i]/mass;
    if(hint_>=0){float proportion=.02f;double moved=0;for(auto& e:root_->edges){float p=e.search_prior;moved+=p*proportion;p*=1-proportion;e.search_prior=p;}for(auto& e:root_->edges)if(e.action==hint_)e.search_prior=float(float(e.search_prior)+float(moved));}
    int cap=options.max_visits?options.max_visits:settings_.max_visits;
    int budget=std::min(cap?std::max(0,cap-1):settings_.simulations,std::max(0,max_playouts-1));
    double max_time=options.max_time<0?settings_.max_time:options.max_time;
    std::mutex mutex;std::condition_variable changed;int issued=0,completed=0;int64_t pending=0;bool stopped=false;std::exception_ptr error;
    auto worker=[&]{
      try {
        for(;;) {
            Node* leaf=nullptr;Node* parent=nullptr;int action=-1;
            std::vector<std::pair<Node*,size_t>> path;
            {std::unique_lock<std::mutex> lock(mutex);
                for(;;) {
                    if(error||issued>=budget)return;
                    // Time limits allow a root prediction plus at least one
                    // recurrent playout. Explicit stops and visit caps still win.
                    if((options.should_stop&&options.should_stop()) || (completed+1>=2 && max_time<1e12 &&
                       std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>=max_time)){stopped=true;return;}
                    path.clear();Node* current=root_;
                    for(;;){size_t which=select(*current,current==root_);if(which==current->edges.size())break;
                        path.push_back({current,which});auto& edge=current->edges[which];
                        if(!edge.child){edge.child=node();current->expansion_order.push_back(which);leaf=edge.child;parent=current;action=edge.action;break;}current=edge.child;}
                    if(leaf){for(auto [p,i]:path){++p->edges[i].pending;++pending;}++issued;break;}
                    changed.wait_for(lock,std::chrono::milliseconds(2));
                }
            }
            try {
                auto output=evaluator_.recurrent(parent->latent,mapping_[action]);
                std::lock_guard<std::mutex> lock(mutex);expand(*leaf,std::move(output),board_actions_,false);
                for(auto it=path.rbegin();it!=path.rend();++it)recompute(*it->first);
                for(auto [p,i]:path){--p->edges[i].pending;--pending;}++completed;
            } catch(...) {
                std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();
                for(auto [p,i]:path){--p->edges[i].pending;--pending;}
            }
            changed.notify_all();
        }
      } catch(...) {std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();changed.notify_all();}
    };
    std::vector<std::thread> threads;
    try{for(int i=1;i<std::min(settings_.threads,budget);++i)threads.emplace_back(worker);worker();}
    catch(...){std::lock_guard<std::mutex> lock(mutex);error=std::current_exception();changed.notify_all();}
    for(auto& thread:threads)thread.join();
    if(pending)throw std::runtime_error("MuZero leaked pending visits");
    if(error)std::rethrow_exception(error);
    if(!stopped&&completed!=budget)throw std::runtime_error("MuZero search budget mismatch");
    result.simulations=completed;result.new_playouts=completed+1;result.root_visits=root_->stats.visits;result.stopped_early=stopped;
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    result.network_wdl=root_->initial;result.network_sample_weight=root_->initial_weight;result.network_value_stdev=root_->stdev;
    auto p=root_->stats;result.value=p.value;result.search_weight=p.weight;result.search_weight_sq=p.weight_sq;result.search_wdl={(1-p.draw+p.value)/2,p.draw,(1-p.draw-p.value)/2};
    for(const auto& e:root_->edges){result.network_policy[e.action]=e.prior;result.search_policy[e.action]=e.search_prior;}
    auto stats=children(*root_);std::vector<int> actions;
    for(size_t i:root_->expansion_order){const auto& e=root_->edges[i];if(e.child&&e.child->ready){actions.push_back(e.action);result.visits[e.action]=result.q_visits[e.action]=e.child->stats.visits;result.q_values[e.action]=-e.child->stats.value;}}
    if(!completed){result.policy=result.move_policy=result.search_policy;result.policy_target=quantize_policy(result.policy);auto behavior=temperature_distribution(result.policy,temperature,settings_.chosen_move_temperature_only_below_prob);result.action=std::discrete_distribution<int>(behavior.begin(),behavior.end())(random_);return result;}
    auto move=root_selection_weights(stats,settings_,settings_.use_lcb&&!options.training,p);
    auto target=root_selection_weights(stats,settings_,settings_.use_lcb,p,false);auto quantized=quantize_policy(target);mass=std::accumulate(target.begin(),target.end(),0.0);
    result.policy_target.resize(area);auto behavior=temperature_distribution(move,temperature,settings_.chosen_move_temperature_only_below_prob);
    for(size_t i=0;i<actions.size();++i){double probability=target[i]/mass;result.policy[actions[i]]=probability;result.move_policy[actions[i]]=move[i];result.policy_target[actions[i]]=quantized[i];if(probability>0)result.policy_surprise+=probability*std::log(probability/std::max(1e-100,stats[i].prior));}
    result.policy_surprise=std::max(0.0,result.policy_surprise);result.action=actions[std::discrete_distribution<size_t>(behavior.begin(),behavior.end())(random_)];return result;
}
}
