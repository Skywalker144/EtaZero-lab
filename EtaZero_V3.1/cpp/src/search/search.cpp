#include "etazero/search.h"
#include "etazero/gumbel.h"
#include "etazero/time_budget.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream>
namespace etazero {
namespace {
void normalize(std::vector<double>& weights) {
    double mass=std::accumulate(weights.begin(),weights.end(),0.0);
    if(!std::isfinite(mass) || mass<=0)throw std::runtime_error("Empty selection policy");
    for(auto& x:weights)x/=mass;
}
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
        weights[candidate]=adjusted;
    }
    double maximum=*std::max_element(weights.begin(),weights.end());
    double cutoff=std::min(s.chosen_move_prune,maximum/64),subtract=std::min(s.chosen_move_subtract,maximum/64);
    for(auto& w:weights)w=w<cutoff?0:std::max(0.0,w-subtract);
    if(diagnostics)for(size_t i=0;i<weights.size();++i)(*diagnostics)[i].weight=weights[i];
    if(normalize_output)normalize(weights);
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
    normalize(policy);return policy;
}

void Search::TreeDeleter::operator()(Node* node) const noexcept {
    // Each unmarked table-owned node is transferred into exactly one work list.
    while(node) {Node* next=node->cleanup_next;node->subtree_bias.release(free_prop);delete node;node=next;}
}
Search::Node* Search::new_node() {
    size_t id=next_node_.fetch_add(1);
    std::string key="private:"+std::to_string(id);
    auto& shard=node_table_[id%node_table_.size()];
    auto owner=std::make_unique<Node>(id);Node* node=owner.get();
    std::lock_guard<std::mutex> lock(shard.mutex);shard.nodes.emplace(std::move(key),std::move(owner));
    return node;
}
Search::Node* Search::find_node(const SearchState& state,const SubtreeValueBiasKey* bias_key,const std::string& pattern_key) {
    if(!settings_.graph_search) {
        Node* node=new_node();node->pattern_key=pattern_key;node->player=state.player();if(bias_key)node->subtree_bias.entry=subtree_bias_table_->get(*bias_key);return node;
    }
    std::string key="graph:"+state.graph_key();
    auto& shard=node_table_[std::hash<std::string>{}(key)%node_table_.size()];
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto found=shard.nodes.find(key);
    if(found!=shard.nodes.end()){++graph_hits_;return found->second.get();}
    auto owner=std::make_unique<Node>(next_node_.fetch_add(1));Node* node=owner.get();
    node->graph_identity=state.graph_key();node->pattern_key=pattern_key;node->player=state.player();
    node->cached_evaluation=cached_evaluation(node->graph_identity);
    // A transposition keeps the bias association from the first creating path.
    if(bias_key)node->subtree_bias.entry=subtree_bias_table_->get(*bias_key);
    shard.nodes.emplace(std::move(key),std::move(owner));return node;
}
std::shared_ptr<const Search::CachedEvaluation> Search::cached_evaluation(const std::string& identity) const {
    if(eval_cache_.empty())return nullptr;
    const auto& shard=eval_cache_[std::hash<std::string>{}(identity)%eval_cache_.size()];
    auto found=shard.find(identity);return found==shard.end()?nullptr:found->second;
}
void Search::apply_cached_evaluation(Node& node,ValueStats& stats,int64_t visits) const {
    const auto& entry=node.cached_evaluation;
    if(&node==root_ || !entry || entry->weight<=0)return;
    double weight=entry->weight;
    if(weight>settings_.eval_cache_min_visits)weight=std::sqrt(settings_.eval_cache_min_visits*weight);
    double ratio=visits/weight,fraction=1/(1+3*ratio*(1+2*ratio*ratio)),old=stats.value;
    stats.value+=fraction*(entry->value-stats.value);stats.win_loss+=fraction*(entry->value-stats.win_loss);
    stats.draw+=fraction*(entry->draw-stats.draw);
    stats.value_sq+=stats.value*stats.value-old*old;
}
void Search::record_evaluations() {
    if(eval_cache_.empty())return;
    std::unordered_set<Node*> seen;std::vector<Node*> stack{root_};
    while(!stack.empty()) {
        auto* node=stack.back();stack.pop_back();if(!seen.insert(node).second)continue;
        for(int i=0;i<node->child_count.load();++i)if(auto* child=node->edge(i).child.load())stack.push_back(child);
        auto stats=snapshot(*node);if(stats.visits<settings_.eval_cache_min_visits)continue;
        auto old=cached_evaluation(node->graph_identity);float weight=stats.visits;
        if(old && weight<old->weight*.75f)continue;
        auto replacement=old?std::make_shared<CachedEvaluation>(*old):std::make_shared<CachedEvaluation>();
        for(int i=0;i<node->child_count.load();++i) {
            auto& edge=node->edge(i);auto* child=edge.child.load();if(!child)continue;auto child_stats=snapshot(*child);
            if(child_stats.visits<settings_.eval_cache_min_visits)continue;
            auto& first=replacement->first_explore[node->moves[edge.move_index].action];
            float child_weight=child_stats.visits;
            if(child_weight>=first.weight)first={float(edge.perspective.load()*child_stats.win_loss),child_weight};
        }
        // Pure WDL search has no special root passing utility to exclude here.
        replacement->value=stats.win_loss;replacement->draw=stats.draw;replacement->weight=weight;
        eval_cache_[std::hash<std::string>{}(node->graph_identity)%eval_cache_.size()][node->graph_identity]=std::move(replacement);
    }
}
double Search::pattern_bonus(const Node& node) const {
    if(!patterns_ || -node.player!=search_for_player_)return 0;
    return patterns_->get(node.pattern_key)*(node.player==-1?1:-1);
}
void Search::prepare_patterns(const SearchState& state,const SearchRun& options) {
    search_for_player_=options.search_for_player?options.search_for_player:state.player();
    patterns_.reset();
    if(settings_.external_patterns)patterns_=std::make_unique<PatternBonusTable>(*settings_.external_patterns);
    if(settings_.avoid_repeated_pattern_utility!=0){if(!patterns_)patterns_=std::make_unique<PatternBonusTable>();
        state.repeated_patterns(*patterns_,settings_.avoid_repeated_pattern_utility*(search_for_player_==-1?-1:1),search_for_player_);}
    root_->pattern_key.clear();
}
void Search::prepare_subtree_bias(const SearchState& state) {
    if(settings_.subtree_value_bias_factor==0 && !patterns_)return;
    if(settings_.subtree_value_bias_factor!=0 && !state.supports_subtree_value_bias())throw std::runtime_error("Subtree value bias requires a real-board state adapter");
    if(settings_.subtree_value_bias_factor!=0 && !subtree_bias_table_)subtree_bias_table_=std::make_unique<SubtreeValueBiasTable>(settings_.subtree_value_bias_table_num_shards);
    // Refresh the retained graph in postorder after promotion/deletion. KataGo
    // resets childless utilities to raw W-L here; backups reapply table bias.
    std::unordered_set<Node*> seen;std::vector<std::pair<Node*,bool>> stack{{root_,false}};
    while(!stack.empty()) {
        auto [node,after]=stack.back();stack.pop_back();
        if(!after) {
            if(!seen.insert(node).second)continue;
            stack.push_back({node,true});
            for(int i=0;i<node->child_count.load();++i)if(auto child=node->edge(i).child.load())stack.push_back({child,false});
        } else if(node->child_count.load()>0 && node->status.load()==Node::READY)recompute(*node);
        else {
            std::lock_guard<std::mutex> lock(node->stats_mutex);
            node->stats.value=node->stats.win_loss+pattern_bonus(*node);node->stats.value_sq=node->stats.value*node->stats.value;
        }
    }
    if(subtree_bias_table_)subtree_bias_table_->clear_unused();
}
size_t Search::node_count() const {
    size_t count=0;for(const auto& shard:node_table_)count+=shard.nodes.size();return count;
}
bool Search::catch_up(Edge& edge,Node& child,int worker) {
    int64_t visits=snapshot(child).visits,edge_visits=edge.n.load();
    if(settings_.graph_catch_up_leak_prob>0 && edge_visits<visits &&
       std::bernoulli_distribution(settings_.graph_catch_up_leak_prob)(random_[worker]))return false;
    do {
        if(edge_visits>=visits)return false;
    } while(!edge.n.compare_exchange_weak(edge_visits,edge_visits+1));
    ++graph_catch_ups_;return true;
}
Search::Search(Evaluator& e,SearchSettings s,uint64_t seed,bool start_workers) : Search(s,seed,start_workers) {evaluator_=&e;}
Search::Search(SearchSettings s,uint64_t seed,bool start_workers) : settings_(s) {
    validate_search_controls(s);
    if(s.simulations<0 || s.threads<1 || s.c_puct<0 || s.virtual_loss<0 ||
       s.noise_fraction<0 || s.noise_fraction>1 || s.dirichlet_total_concentration<=0 || s.max_visits<0 || (s.use_fpu && (s.fpu_reduction_max<0 || s.root_fpu_reduction_max<0 ||
       s.fpu_parent_power<0 || s.fpu_parent_weight<0 || s.fpu_parent_weight>1)) || s.max_playouts<0 || s.max_time<0 || !std::isfinite(s.max_time) ||
       s.nn_symmetry<0 || s.nn_symmetry>=8 || s.forced_playouts<0 || s.lcb_stdevs<=0 || s.min_lcb_visit_prop<0 || s.min_lcb_visit_prop>1 ||
       s.value_weight_exponent<0 || s.chosen_move_subtract<0 || s.chosen_move_prune<0 || (s.use_fpu && (s.fpu_loss_prop<0 || s.fpu_loss_prop>1 ||
       s.root_fpu_loss_prop<0 || s.root_fpu_loss_prop>1)) || s.c_puct_log<0 || s.c_puct_base<=0 || s.c_puct_stdev_prior<=0 ||
       s.c_puct_stdev_prior_weight<0 || s.c_puct_stdev_scale<0 || s.c_puct_stdev_scale>1 || s.root_symmetries<1 || s.root_symmetries>8 ||
       s.nn_policy_temperature<=0 || s.root_policy_temperature<.01 || s.root_policy_temperature>100 || s.root_policy_temperature_early<.01 || s.root_policy_temperature_early>100 || s.temperature_halflife<=0 ||
       s.chosen_move_temperature_only_below_prob<0 || s.chosen_move_temperature_only_below_prob>1 ||
       s.graph_rep_bound<3 || s.graph_rep_bound>50 || !std::isfinite(s.graph_catch_up_leak_prob) || s.graph_catch_up_leak_prob<0 || s.graph_catch_up_leak_prob>1 ||
       (s.use_uncertainty && (!std::isfinite(s.uncertainty_coeff) || s.uncertainty_coeff<0.0001 || s.uncertainty_coeff>1 ||
       !std::isfinite(s.uncertainty_exponent) || s.uncertainty_exponent<0 || s.uncertainty_exponent>2 ||
       !std::isfinite(s.uncertainty_max_weight) || s.uncertainty_max_weight<1 || s.uncertainty_max_weight>100)) ||
       !std::isfinite(s.policy_optimism) || s.policy_optimism<0 || s.policy_optimism>1 ||
       !std::isfinite(s.root_policy_optimism) || s.root_policy_optimism<0 || s.root_policy_optimism>1 ||
       (s.use_noise_pruning && (!std::isfinite(s.noise_prune_utility_scale) || s.noise_prune_utility_scale<0.001 || s.noise_prune_utility_scale>10 ||
       !std::isfinite(s.noise_pruning_cap) || s.noise_pruning_cap<0 || s.noise_pruning_cap>1e50)) ||
       !std::isfinite(s.subtree_value_bias_factor) || s.subtree_value_bias_factor<0 || s.subtree_value_bias_factor>1 ||
       !std::isfinite(s.subtree_value_bias_free_prop) || s.subtree_value_bias_free_prop<0 || s.subtree_value_bias_free_prop>1 ||
       !std::isfinite(s.subtree_value_bias_weight_exponent) || s.subtree_value_bias_weight_exponent<0 || s.subtree_value_bias_weight_exponent>1 ||
       s.subtree_value_bias_table_num_shards<1 || s.node_table_shards_power_of_two<8 || s.node_table_shards_power_of_two>24 ||
       !std::isfinite(s.min_playouts_per_thread) || s.min_playouts_per_thread<0 || s.min_playouts_per_thread>1e20)
        throw std::runtime_error("Invalid search settings");
    if(!std::isfinite(s.wide_root_noise)||s.wide_root_noise<0||s.wide_root_noise>5||
       !std::isfinite(s.futile_visits_threshold)||s.futile_visits_threshold<0||s.futile_visits_threshold>1||
       s.visit_cap_contempt<0||s.visit_cap_contempt==1||s.visit_cap_contempt>SEARCH_CONFIG_LIMIT||
       std::abs(s.visit_cap_contempt_player)>1||s.eval_cache_min_visits<1||s.eval_cache_min_visits>SEARCH_CONFIG_LIMIT)
        throw std::runtime_error("Invalid root exploration, visit cap, or eval cache settings");
    if(s.use_eval_cache && s.graph_search)eval_cache_.resize(s.subtree_value_bias_table_num_shards);
    node_table_.resize(size_t(1)<<s.node_table_shards_power_of_two);
    root_=new_node();thread_states_.resize(s.threads);cleanup_roots_.resize(s.threads,nullptr);
    for(int i=0;i<s.threads;++i) {
        random_.emplace_back(seed+0x9e3779b97f4a7c15ULL*i);
        inference_random_.emplace_back((seed^0xd1b54a32d192ed03ULL)+0x9e3779b97f4a7c15ULL*i);
    }
    try {if(start_workers)for(int i=1;i<s.threads;++i)workers_.emplace_back(&Search::worker_loop,this,i);}
    catch(...) {
        {std::lock_guard<std::mutex> lock(work_mutex_);closing_=true;}
        work_changed_.notify_all();for(auto& t:workers_)t.join();throw;
    }
}
Search::~Search() {
    root_=nullptr;collect_nodes(nullptr);
    {std::lock_guard<std::mutex> lock(work_mutex_);closing_=true;}
    work_changed_.notify_all();for(auto& t:workers_)t.join();
}
void Search::collect_nodes(Node* keep) {
    std::unordered_set<Node*> marked;std::vector<Node*> queue;
    if(keep){marked.insert(keep);queue.push_back(keep);}
    for(size_t i=0;i<queue.size();++i) {
        Node* node=queue[i];
        for(int j=0;j<node->child_count.load();++j) {
            Node* child=node->edge(j).child.load();
            if(child && marked.insert(child).second)queue.push_back(child);
        }
    }
    std::fill(cleanup_roots_.begin(),cleanup_roots_.end(),nullptr);
    size_t count=0;
    for(auto& shard:node_table_) {
        for(auto it=shard.nodes.begin();it!=shard.nodes.end();) {
            if(marked.count(it->second.get())){++it;continue;}
            Node* node=it->second.release();it=shard.nodes.erase(it);
            auto& list=cleanup_roots_[count++%cleanup_roots_.size()];node->cleanup_next=list;list=node;
        }
    }
    if(count<4096 || workers_.empty()) {
        for(auto node:cleanup_roots_)TreeDeleter{settings_.subtree_value_bias_free_prop}(node);
    } else {
        {std::lock_guard<std::mutex> lock(work_mutex_);cleaning_=true;workers_done_=0;++generation_;}
        work_changed_.notify_all();TreeDeleter{settings_.subtree_value_bias_free_prop}(cleanup_roots_[0]);
        std::unique_lock<std::mutex> lock(work_mutex_);
        work_changed_.wait(lock,[&]{return workers_done_==static_cast<int>(workers_.size());});cleaning_=false;
    }
    std::fill(cleanup_roots_.begin(),cleanup_roots_.end(),nullptr);
}
void Search::reset(uint64_t seed) {
    carried_time_=0;set_root_focus({},{},0);avoid_until_={};
    for(auto& shard:eval_cache_)shard.clear();
    root_=nullptr;collect_nodes(nullptr);root_=new_node();
    if(subtree_bias_table_)subtree_bias_table_->clear_unused();
    for(int i=0;i<settings_.threads;++i) {
        random_[i].seed(seed+0x9e3779b97f4a7c15ULL*i);
        inference_random_[i].seed((seed^0xd1b54a32d192ed03ULL)+0x9e3779b97f4a7c15ULL*i);
    }
}
void Search::set_evaluator(Evaluator& evaluator) {
    carried_time_=0;set_root_focus({},{},0);avoid_until_={};
    for(auto& shard:eval_cache_)shard.clear();
    root_=nullptr;collect_nodes(nullptr);root_=new_node();evaluator_=&evaluator;
    if(subtree_bias_table_)subtree_bias_table_->clear_unused();
    for(auto& thread:thread_states_)thread.state.reset();
}
void Search::evaluate_node(Node& node,const SearchState& state,int worker,int symmetries,bool skip_cache,int depth) {
    const int actions=state.actions(),symmetry_count=state.symmetry_count();
    const bool need_human=evaluator_ && evaluator_->secondary && (&node==root_ || settings_.human_sl_player_weightless>0 || settings_.human_sl_player_weightful>0 || settings_.human_sl_opponent_weightless>0 || settings_.human_sl_opponent_weightful>0);
    if(symmetries>symmetry_count)throw std::runtime_error("Root symmetry count exceeds state support");
    auto& legal=thread_states_[worker].legal_actions;legal.clear();
    for(int a=0;a<actions;++a)if(state.legal(a))legal.push_back(a);
    bool full_domain=settings_.graph_search && state.graph_approximates_history();
    if(!node.moves) {
        node.move_count=full_domain?size_t(actions):legal.size();
        node.moves=std::make_unique<Move[]>(node.move_count);
        for(size_t i=0;i<node.move_count;++i)node.moves[i].action=full_domain?int(i):legal[i];
    }
    auto eval=std::make_shared<NodeEvaluation>();
    eval->prior.assign(node.move_count,-1);eval->search_prior=eval->prior;
    if(full_domain)eval->prior_order=legal;
    else {eval->prior_order.resize(node.move_count);std::iota(eval->prior_order.begin(),eval->prior_order.end(),0);}
    std::vector<int> symmetry_indices(symmetry_count);std::iota(symmetry_indices.begin(),symmetry_indices.end(),0);
    std::vector<double> policy(actions,0);WDL averaged{0,0,0};double stdev=0;
    bool auxiliary=false;double optimism=&node==root_?settings_.root_policy_optimism:settings_.policy_optimism;
    for(int i=0;i<symmetries;++i) {
        if(symmetries>1)std::swap(symmetry_indices[i],symmetry_indices[std::uniform_int_distribution<int>(i,symmetry_count-1)(random_[worker])]);
        int symmetry=symmetry_indices[i];
        if(symmetries==1) {
            symmetry=settings_.nn_randomize && symmetry_count>1?
                std::uniform_int_distribution<int>(0,symmetry_count-1)(inference_random_[worker]):settings_.nn_symmetry;
        }
        if(symmetry>=symmetry_count)throw std::runtime_error("NN symmetry exceeds state support");
        auto output=state.evaluate({symmetry,skip_cache || symmetries>1,settings_.nn_policy_temperature,
                                    symmetries==1 && settings_.nn_randomize,optimism,
                                    settings_.ignore_all_history?0:settings_.ignore_pre_root_history?depth:-1});
        if(output.logits.size()!=policy.size())throw std::runtime_error("Bad leaf policy shape");
        if(i==0)auxiliary=output.has_auxiliary;
        if(auxiliary!=output.has_auxiliary)throw std::runtime_error("D4 auxiliary capabilities disagree");
        if(auxiliary) {
            if(output.optimistic_logits.size()!=policy.size() || !std::isfinite(output.shortterm_value_stdev) || output.shortterm_value_stdev<0)
                throw std::runtime_error("Invalid auxiliary leaf output");
            stdev+=output.shortterm_value_stdev/symmetries;
            for(size_t a=0;a<output.logits.size();++a) {
                if(!std::isfinite(output.optimistic_logits[a]))throw std::runtime_error("Nonfinite optimistic policy");
                // Fixed source v15/v17 backend mixes short-optimistic and ordinary LOGITS before Tnn/softmax.
                output.logits[a]=mixed_policy_logit(output.logits[a],output.optimistic_logits[a],optimism);
            }
        }
        double mass=0;
        for(int j=0;j<3;++j) {
            double p=output.wdl[j];if(!std::isfinite(p)||p<0||p>1)throw std::runtime_error("Invalid leaf WDL");
            mass+=p;averaged[j]+=p/symmetries;
        }
        if(std::abs(mass-1)>1e-5)throw std::runtime_error("Unnormalized leaf WDL");
        double maximum=-std::numeric_limits<double>::infinity(),sum=0;
        for(int j:eval->prior_order) {
            int a=node.moves[j].action;
            if(!std::isfinite(output.logits[a]))throw std::runtime_error("Nonfinite leaf policy");
            maximum=std::max(maximum,output.logits[a]);
        }
        std::vector<double> probabilities(actions,0);
        for(int j:eval->prior_order) {
            int a=node.moves[j].action;
            probabilities[a]=std::exp((output.logits[a]-maximum)/settings_.nn_policy_temperature);sum+=probabilities[a];
        }
        if(!std::isfinite(sum)||sum<=0)throw std::runtime_error("No valid leaf policy");
        for(int a=0;a<actions;++a)policy[a]+=probabilities[a]/sum/symmetries;
    }
    if(need_human) {
        eval->human_policy.assign(node.move_count,-1);
        for(int index:eval->prior_order)eval->human_policy[index]=0;
        std::iota(symmetry_indices.begin(),symmetry_indices.end(),0);
        for(int i=0;i<symmetries;++i) {
            if(symmetries>1)std::swap(symmetry_indices[i],symmetry_indices[std::uniform_int_distribution<int>(i,symmetry_count-1)(random_[worker])]);
            int symmetry=symmetries>1?symmetry_indices[i]:settings_.nn_symmetry;
            auto output=state.evaluate_with(*evaluator_->secondary,{symmetry,skip_cache||symmetries>1,settings_.nn_policy_temperature,symmetries==1&&settings_.nn_randomize,optimism,settings_.ignore_all_history?0:settings_.ignore_pre_root_history?depth:-1});
            if(output.logits.size()!=size_t(actions))throw std::runtime_error("Secondary policy action shape mismatch");
            double maximum=-std::numeric_limits<double>::infinity();
            for(int index:eval->prior_order){int action=node.moves[index].action;
                if(output.has_auxiliary)output.logits[action]=mixed_policy_logit(output.logits[action],output.optimistic_logits.at(action),optimism);
                if(!std::isfinite(output.logits[action]))throw std::runtime_error("Nonfinite secondary policy");
                maximum=std::max(maximum,output.logits[action]);}
            double sum=0;for(int index:eval->prior_order){int action=node.moves[index].action;output.logits[action]=std::exp((output.logits[action]-maximum)/settings_.nn_policy_temperature);sum+=output.logits[action];}
            for(int index:eval->prior_order)eval->human_policy[index]+=output.logits[node.moves[index].action]/sum/symmetries;
        }
    }
    for(int i:eval->prior_order)eval->search_prior[i]=eval->prior[i]=policy[node.moves[i].action];
    std::sort(eval->prior_order.begin(),eval->prior_order.end(),[&](int a,int b){return eval->prior[a]!=eval->prior[b]?eval->prior[a]>eval->prior[b]:node.moves[a].action<node.moves[b].action;});
    eval->initial_wdl=averaged;eval->initial_value=averaged[0]-averaged[2];eval->initial_stdev=stdev;eval->has_auxiliary=auxiliary;
    eval->initial_weight=uncertainty_weight(stdev,auxiliary,settings_);
    {
        std::lock_guard<std::mutex> lock(node.stats_mutex);
        if(node.stats.visits==0) {
            ValueStats initial{1,eval->initial_value,eval->initial_value*eval->initial_value,averaged[1],eval->initial_weight,eval->initial_weight*eval->initial_weight,eval->initial_value};
            apply_cached_evaluation(node,initial,1);
            initial.value=node.subtree_bias.apply(initial.value,settings_.subtree_value_bias_factor)+pattern_bonus(node);
            initial.value_sq=initial.value*initial.value;node.stats=initial;
        }
    }
    // As in KataGo's isReInit path, replacing NN output adds no visit and leaves
    // aggregate statistics to be refreshed on the next ordinary backup.
    node.publish(std::move(eval));
}

ValueStats Search::snapshot(Node& node) {std::lock_guard<std::mutex> lock(node.stats_mutex);return node.stats;}
std::vector<RootChildStats> Search::child_stats(Node& node,const NodeEvaluation& eval,bool aggregation) {
    std::vector<RootChildStats> children;children.reserve(node.child_count.load());
    int count=node.child_count.load(std::memory_order_acquire);
    for(int i=0;i<count;++i) {
        auto& edge=node.edge(i);Node* child=edge.child.load(std::memory_order_acquire);auto n=edge.n.load();
        if(!child || n<=0)continue;
        auto stats=snapshot(*child);if(stats.visits<=0 || stats.weight<=0)continue;
        double scale=double(n)/stats.visits,weight=stats.weight*scale;
        double perspective=edge.perspective.load();
        children.push_back({eval.search_prior[edge.move_index],weight*perspective*stats.value,
                            weight*stats.value_sq,n,weight,stats.weight_sq*scale*(aggregation?scale:1),weight*stats.draw,weight*perspective*stats.win_loss});
    }
    return children;
}
void Search::recompute(Node& node,int visits_to_add) {
    std::lock_guard<std::mutex> update(node.update_mutex);
    // LCB uses linear edge sampling weightSq; parent aggregation scales a weighted mean quadratically.
    auto eval=node.evaluated();
    auto children=child_stats(node,*eval,true);
    auto result=aggregate_values(eval->initial_wdl,children,settings_,&node==root_ && !remove_root_noise_ && settings_.noise_fraction>0,eval->initial_weight,&node.subtree_bias);
    double old_utility=result.value;result.value+=pattern_bonus(node);result.value_sq+=result.value*result.value-old_utility*old_utility;
    apply_cached_evaluation(node,result,result.visits);
    auto previous=snapshot(node);
    std::vector<double> distribution;
    if(settings_.visit_cap_contempt>0 && node.player==capped_player_ &&
       previous.visits+visits_to_add>=settings_.visit_cap_contempt && node.capped_distribution.empty()) {
        double sum=0;distribution.assign(node.move_count,0);
        for(int i=0;i<node.child_count.load();++i) {
            auto& e=node.edge(i);auto* child=e.child.load();if(!child || e.n<=0)continue;
            auto stats=snapshot(*child);
            double w=stats.weight*double(e.n.load())/std::max<int64_t>(1,stats.visits);
            distribution[e.move_index]=w;sum+=w;
        }
        if(sum>0)for(auto& w:distribution)w/=sum;else distribution.clear();
    }
    std::lock_guard<std::mutex> lock(node.stats_mutex);
    if(!distribution.empty())node.capped_distribution=std::move(distribution);
    result.visits=node.stats.visits+visits_to_add;node.stats=result;
}
bool Search::expand(Node& node,const SearchState& state,int worker,int depth) {
    auto status=node.status.load(std::memory_order_acquire);
    if(status==Node::READY)return false;
    auto& sync=locks_[node.lock_index];std::unique_lock<std::mutex> lock(sync.mutex);
    sync.changed.wait(lock,[&]{return node.status.load(std::memory_order_acquire)!=Node::EXPANDING;});
    status=node.status.load(std::memory_order_acquire);
    if(status==Node::FAILED)std::rethrow_exception(node.error);
    if(status==Node::READY)return false;
    node.player=state.player();
    node.status.store(Node::EXPANDING,std::memory_order_release);lock.unlock();
    try {
        evaluate_node(node,state,worker,&node==root_?root_symmetries_:1,false,depth);
        lock.lock();
        node.status.store(Node::READY,std::memory_order_release);lock.unlock();sync.changed.notify_all();return true;
    } catch(...) {
        if(!lock.owns_lock())lock.lock();
        node.error=std::current_exception();node.status.store(Node::FAILED,std::memory_order_release);
        lock.unlock();sync.changed.notify_all();throw;
    }
}
Search::Edge& Search::child_edge(Node& node,int move_index) {
    auto& move=node.moves[move_index];int index=move.child_index.load(std::memory_order_acquire);
    if(index>=0)return node.edge(index);
    std::lock_guard<std::mutex> lock(locks_[node.lock_index].mutex);
    index=move.child_index.load(std::memory_order_relaxed);
    if(index<0) {
        index=node.child_count.load(std::memory_order_relaxed);
        int tier=index<8?0:index<64?1:2;
        if(!node.storage[tier]) {
            size_t start=tier==0?0:tier==1?8:64;
            size_t capacity=std::min(node.move_count-start,tier==0?size_t(8):tier==1?size_t(56):node.move_count-start);
            node.storage[tier]=std::make_unique<Edge[]>(capacity);
            node.published[tier].store(node.storage[tier].get(),std::memory_order_release);
        }
        node.edge(index).move_index=move_index;
        node.child_count.store(index+1,std::memory_order_release);
        move.child_index.store(index,std::memory_order_release);
    }
    return node.edge(index);
}
bool Search::move_allowed(int action,int player,int depth) const {
    const auto& avoid=avoid_until_[player==1?0:1];
    if(!avoid.empty() && depth<avoid[action])return false;
    return depth!=0 || root_representative_.empty() || root_representative_[action]==action;
}
bool Search::prepare_restrictions(const SearchState& state,const SearchRun& options) {
    for(const auto& avoid:options.avoid_until)if(!avoid.empty()) {
        if(avoid.size()!=size_t(state.actions()))throw std::runtime_error("Avoid-move shape mismatch");
        for(int until:avoid)if(until<0)throw std::runtime_error("Avoid-move depth must be nonnegative");
    }
    bool changed=avoid_until_!=options.avoid_until;avoid_until_=options.avoid_until;
    root_representative_=settings_.root_symmetry_pruning?state.root_symmetry_representatives(avoid_until_[state.player()==1?0:1]):std::vector<int>();
    return changed;
}
bool Search::prune_root_children(const SearchState& state) {
    // The root is private and workers are idle. Retain legal edges in activation
    // order, including their visits. Recompute retained bias before the caller
    // collects unreachable subgraphs, matching KataGo's root preparation order.
    struct Retained {int move,perspective;int64_t visits;Node* child;};
    std::vector<Retained> retained;
    int count=root_->child_count.load();bool filtered=false;
    for(int i=0;i<count;++i) {
        auto& edge=root_->edge(i);
        if(state.legal(root_->moves[edge.move_index].action) && move_allowed(root_->moves[edge.move_index].action,state.player(),0))
            retained.push_back({edge.move_index,edge.perspective.load(),edge.n.load(),edge.child.load()});
        else filtered=true;
    }
    if(!filtered)return false;
    for(int i=0;i<count;++i) {
        auto& edge=root_->edge(i);root_->moves[edge.move_index].child_index=-1;
        edge.n=0;edge.child=nullptr;edge.perspective=0;edge.move_index=-1;
    }
    root_->child_count=0;int64_t visits=0;
    for(auto kept:retained) {
        auto& edge=child_edge(*root_,kept.move);
        edge.n=kept.visits;edge.perspective=kept.perspective;edge.child=kept.child;visits+=kept.visits;
    }
    root_->visits=visits;
    {std::lock_guard<std::mutex> lock(root_->stats_mutex);root_->stats.visits=visits+1;}
    recompute(*root_);return true;
}
template<bool ForcedRoot,bool GumbelInterior>
void Search::simulation(int worker,int root_move,const GumbelSettings* gumbel) {
    auto& scratch=thread_states_[worker];
    if(!scratch.state || !scratch.state->reset_from(*position_))scratch.state=position_->clone();
    auto& state=*scratch.state;Node* node=root_;auto& path=scratch.path;path.clear();scratch.graph_path.clear();
    auto release=[&]{for(auto& p:path){--p.edge->pending;--p.edge->child.load()->pending;}path.clear();};
    try {
        double value,draw;bool count_playout=true;
        for(;;) {
            if(state.terminal()) {
                value=state.terminal_value();draw=state.terminal_draw_probability();
                if(!std::isfinite(value) || std::abs(value)>1)throw std::runtime_error("Invalid terminal value");
                std::lock_guard<std::mutex> lock(node->stats_mutex);
                auto& stats=node->stats;++stats.visits;stats.win_loss=value;stats.value=value+pattern_bonus(*node);stats.value_sq=stats.value*stats.value;stats.draw=draw;
                double terminal_weight=settings_.use_uncertainty && root_->evaluated()->has_auxiliary?settings_.uncertainty_max_weight:1;
                stats.weight=stats.visits*terminal_weight;stats.weight_sq=stats.visits*terminal_weight*terminal_weight;break;
            }
            if(expand(*node,state,worker,scratch.path.size())){auto eval=node->evaluated();value=eval->initial_value;draw=eval->initial_wdl[1];break;}
            auto eval=node->evaluated();
            int choice=-1;bool weightless=false;
            if(node==root_)if(auto focus=focus_snapshot();focus && std::uniform_real_distribution<double>()(random_[worker])<focus->probability) {
                std::vector<double> weights(focus->moves.size(),0);std::vector<int> targets(focus->moves.size(),-1);double sum=0;
                for(size_t i=0;i<weights.size();++i) {
                    int action=focus->moves[i];if(action>=state.actions() || !state.legal(action))continue;
                    const auto& avoid=avoid_until_[state.player()==1?0:1];if(!avoid.empty()&&avoid[action]>0)continue;
                    int target=root_representative_.empty()?action:root_representative_[action];
                    if(!move_allowed(target,state.player(),0))continue;
                    for(size_t j=0;j<node->move_count;++j)if(node->moves[j].action==target && eval->search_prior[j]>=0) {
                        targets[i]=j;weights[i]=focus->weights[i];sum+=weights[i];break;
                    }
                }
                if(sum>0){choice=targets[std::discrete_distribution<int>(weights.begin(),weights.end())(random_[worker])];weightless=true;count_playout=false;}
            }
            double human_draw=-1;
            if constexpr(ForcedRoot) {
                if(choice<0){
                    if(node==root_)choice=root_move;
                    else if constexpr(GumbelInterior){
                        bool capped;{std::lock_guard<std::mutex> lock(node->stats_mutex);capped=!node->capped_distribution.empty();}
                        double probability=state.player()==root_player_?settings_.human_sl_player_weightless+settings_.human_sl_player_weightful:settings_.human_sl_opponent_weightless+settings_.human_sl_opponent_weightful;
                        if(!capped&&!eval->human_policy.empty()&&probability>0)human_draw=std::uniform_real_distribution<double>()(random_[worker]);
                        if(!capped&&(human_draw<0||human_draw>=probability))choice=gumbel_selection(*node,*gumbel,path.size());
                    }
                }
            }
            if(choice<0) {
            std::vector<double> capped;
            {std::lock_guard<std::mutex> lock(node->stats_mutex);capped=node->capped_distribution;}
            bool use_human=false;
            if(capped.empty() && !eval->human_policy.empty()) {
                double weightless_probability=node==root_?settings_.human_sl_root_weightless:state.player()==root_player_?settings_.human_sl_player_weightless:settings_.human_sl_opponent_weightless;
                double weightful_probability=node==root_?settings_.human_sl_root_weightful:state.player()==root_player_?settings_.human_sl_player_weightful:settings_.human_sl_opponent_weightful;
                if(weightless_probability+weightful_probability>0){double r=human_draw>=0?human_draw:std::uniform_real_distribution<double>()(random_[worker]);
                    use_human=r<weightless_probability+weightful_probability;if(r<weightless_probability)weightless=true;}
            }
            const auto& selected_policy=use_human?eval->human_policy:eval->search_prior;
            auto& candidates=scratch.candidates;
            do {
                candidates.clear();
                double best=-std::numeric_limits<double>::infinity();
                // KataGo keeps the first allocated child on an equal score.
                auto consider=[&](int i,double score){if(score>best){best=score;candidates.assign(1,i);}};
                int count=node->child_count.load(std::memory_order_acquire);
                double visited_mass=0,total=0,max_weight=0;auto parent=snapshot(*node);
                std::vector<ValueStats> child_values(count);
                for(int i=0;i<count;++i) {
                    auto& e=node->edge(i);Node* child=e.child.load(std::memory_order_acquire);
                    if(child) {
                        visited_mass+=std::max(0.0,selected_policy[e.move_index]);
                        child_values[i]=snapshot(*child);auto& stats=child_values[i];
                        if(!weightless)stats.weight*=double(e.n.load())/std::max<int64_t>(1,stats.visits);
                        total+=stats.weight;max_weight=std::max(max_weight,stats.weight);
                    }
                }
                bool is_root=node==root_;
                auto selection_settings=settings_;
                if(use_human){selection_settings.c_puct=settings_.human_sl_cpuct_exploration+settings_.human_sl_cpuct_permanent*std::sqrt(total);selection_settings.c_puct_log=0;selection_settings.c_puct_stdev_scale=0;}
                std::vector<int> human_order;
                if(use_human){human_order=eval->prior_order;std::stable_sort(human_order.begin(),human_order.end(),[&](int a,int b){return selected_policy[a]!=selected_policy[b]?selected_policy[a]>selected_policy[b]:node->moves[a].action<node->moves[b].action;});}
                const auto& selected_order=use_human?human_order:eval->prior_order;
                if(!capped.empty()) {
                    double total_weight=0,total_fraction=0;
                    for(int i=0;i<count;++i){auto& e=node->edge(i);total_weight+=child_values[i].weight+(e.child.load()?e.child.load()->pending.load():e.pending.load())*settings_.virtual_loss;
                        if(selected_policy[e.move_index]>=0 && move_allowed(node->moves[e.move_index].action,state.player(),path.size()))total_fraction+=capped[e.move_index];}
                    if(total_fraction>0)for(int i=0;i<count;++i){auto& e=node->edge(i);
                        if(selected_policy[e.move_index]>=0 && capped[e.move_index]>0 && move_allowed(node->moves[e.move_index].action,state.player(),path.size()))
                            consider(e.move_index,capped[e.move_index]/total_fraction*(total_weight+parent.weight/std::max<int64_t>(1,parent.visits))-
                                child_values[i].weight-(e.child.load()?e.child.load()->pending.load():e.pending.load())*settings_.virtual_loss);}
                    if(!candidates.empty())break;
                }
                double reduction=is_root&&!remove_root_noise_?settings_.root_fpu_reduction_max:settings_.fpu_reduction_max;
                double fpu=settings_.use_fpu?fpu_value(eval->initial_value,parent.value,visited_mass,settings_.fpu_parent_power,reduction,
                    settings_.fpu_parent_weight_by_visited_policy,settings_.fpu_parent_weight):0;
                if(settings_.use_fpu) {
                    double loss_prop=is_root&&!remove_root_noise_?settings_.root_fpu_loss_prop:settings_.fpu_loss_prop;
                    fpu+=(-1-fpu)*loss_prop;
                }
                for(int i=0;i<count;++i) {
                    auto& e=node->edge(i);auto& m=node->moves[e.move_index];int pending=settings_.graph_search && e.child.load()?e.child.load()->pending.load(std::memory_order_relaxed):e.pending.load(std::memory_order_relaxed);
                    const auto& stats=child_values[i];double q=stats.weight>0?e.perspective.load()*stats.value:fpu;
                    if(selected_policy[e.move_index]<0 || !move_allowed(m.action,state.player(),path.size()))continue;
                    double score=is_root&&!weightless?root_explore_score(selected_policy[e.move_index],q,stats.weight,pending,e.n.load(),stats.visits,
                        total,max_weight,parent,selection_settings,!remove_root_noise_,m.action==root_hint_action_,std::min(double(budget_-completed_.load()),time_budget_->visits_left()),random_[worker]):
                        child_selection_score(selected_policy[e.move_index],q,stats.weight,pending,total,parent,selection_settings,false);
                    consider(e.move_index,score);
                }
                if(is_root && !weightless && root_hint_action_>=0) {
                    for(size_t i=0;i<node->move_count;++i)if(node->moves[i].action==root_hint_action_ && node->moves[i].child_index.load()<0 && move_allowed(node->moves[i].action,state.player(),path.size())) {
                        double parent_per_visit=parent.weight/std::max<int64_t>(1,parent.visits);
                        for(const auto& other:child_values)if(parent_per_visit<other.weight*.8) {consider(i,1e20);break;}
                    }
                }
                if(node->cached_evaluation)for(const auto& [action,cached]:node->cached_evaluation->first_explore) {
                    for(size_t i=0;i<node->move_count;++i)if(node->moves[i].action==action && node->moves[i].child_index.load()<0 && selected_policy[i]>=0 && move_allowed(action,state.player(),path.size())) {
                        double score=is_root&&!weightless?root_explore_score(selected_policy[i],cached.value,0,0,0,0,total,max_weight,parent,selection_settings,false,
                            action==root_hint_action_,std::min(double(budget_-completed_.load()),time_budget_->visits_left()),random_[worker]):
                            cached.value+explore_scaling(total,parent,selection_settings)*selected_policy[i];
                        consider(i,score);break;
                    }
                }
                // Only the highest-prior unexpanded action competes with children;
                // equal priors follow action order and a tie cannot displace a child.
                for(int i:selected_order)if(node->moves[i].child_index.load(std::memory_order_acquire)<0 && move_allowed(node->moves[i].action,state.player(),path.size())) {
                    double score=is_root&&!weightless?root_explore_score(selected_policy[i],fpu,0,0,0,0,total,max_weight,parent,selection_settings,false,
                        node->moves[i].action==root_hint_action_,std::min(double(budget_-completed_.load()),time_budget_->visits_left()),random_[worker]):
                        fpu+explore_scaling(total,parent,selection_settings)*selected_policy[i];
                    consider(i,score);
                    break;
                }
                // A concurrent allocation can move the last candidate between the
                // two snapshots. Retry instead of losing or dereferencing it.
                if(candidates.empty()) {
                    bool allowed=false;for(size_t i=0;i<node->move_count;++i)if(selected_policy[i]>=0 && move_allowed(node->moves[i].action,state.player(),path.size())){allowed=true;break;}
                    if(!allowed)break;
                }
            } while(candidates.empty());
            if(!candidates.empty())choice=candidates.front();
            }
            if(choice<0) {
                ValueStats sample{1,eval->initial_value,eval->initial_value*eval->initial_value,eval->initial_wdl[1],eval->initial_weight,eval->initial_weight*eval->initial_weight,eval->initial_value};
                apply_cached_evaluation(*node,sample,1);sample.value=node->subtree_bias.apply(sample.value,settings_.subtree_value_bias_factor)+pattern_bonus(*node);sample.value_sq=sample.value*sample.value;
                std::lock_guard<std::mutex> lock(node->stats_mutex);auto& s=node->stats;double total=s.weight+sample.weight;
                s.value=(s.value*s.weight+sample.value*sample.weight)/total;s.value_sq=(s.value_sq*s.weight+sample.value_sq*sample.weight)/total;
                s.draw=(s.draw*s.weight+sample.draw*sample.weight)/total;s.win_loss=(s.win_loss*s.weight+sample.win_loss*sample.weight)/total;
                s.weight=total;s.weight_sq+=sample.weight_sq;++s.visits;++node->visits;value=s.value;draw=s.draw;break;
            }
            if(settings_.graph_search && state.graph_approximates_history() && !state.legal(node->moves[choice].action)) {
                if(scratch.illegal_graph_nodes.insert(node->id).second) {
                    std::lock_guard<std::mutex> lock(error_mutex_);
                    std::cerr<<"Graph move is illegal on the current history; refreshing NN output at node "
                             <<node->id<<", action "<<node->moves[choice].action<<'\n';
                }
                evaluate_node(*node,state,worker,node==root_?root_symmetries_:1,true,scratch.path.size());
                // Count the incoming edges, but add neither a visit nor a new
                // leaf value to this node, matching KataGo's illegal-move repair.
                auto previous=snapshot(*node);value=previous.value;draw=previous.draw;break;
            }
            Edge* edge=&child_edge(*node,choice);
            Node* child=edge->child.load(std::memory_order_acquire);
            SubtreeValueBiasKey bias_key;
            bool has_bias_key=!child && settings_.subtree_value_bias_factor!=0 && state.subtree_bias_key(node->moves[choice].action,bias_key);
            std::string pattern_key=!child&&patterns_?state.pattern_key(node->moves[choice].action):std::string();
            int perspective=state.move(node->moves[choice].action);
            if(perspective!=1 && perspective!=-1)throw std::runtime_error("WDL search perspective must be +1 or -1");
            if(!child) {
                Node* candidate=find_node(state,has_bias_key?&bias_key:nullptr,pattern_key);Node* expected=nullptr;
                if(edge->child.compare_exchange_strong(expected,candidate,std::memory_order_release,std::memory_order_acquire))child=candidate;
                else child=expected; // Unused racing candidates are collected with the table.
            }
            path.push_back({node,edge,perspective,true,weightless});++edge->pending;++child->pending;
            int expected=0;
            edge->perspective.compare_exchange_strong(expected,perspective);
            if(expected!=0 && expected!=perspective)throw std::runtime_error("Inconsistent edge perspective");
            if(!weightless && settings_.graph_search && catch_up(*edge,*child,worker)) {
                path.back().increment_edge=false;value=snapshot(*child).value;draw=0;break;
            }
            if(settings_.graph_search && !scratch.graph_path.insert(child).second) {
                ++graph_cycles_;value=snapshot(*child).value;draw=0;break;
            }
            node=child;
        }
        bool update_ancestors=true;
        while(!path.empty()) {
            auto p=path.back();value=p.perspective*value;update_ancestors&=!p.weightless;
            if(!std::isfinite(value))throw std::runtime_error("Nonfinite search backup");
            if(update_ancestors){if(p.increment_edge)++p.edge->n;
                ++p.parent->visits;recompute(*p.parent,1);}
            --p.edge->pending;--p.edge->child.load()->pending;path.pop_back();
        }
        scratch.counted_playout=count_playout;
        if(count_playout)++completed_;else if constexpr(!ForcedRoot)--issued_;
    } catch(...){release();throw;}
}
// Only Gumbel search uses these instantiations. The PUCT specialization has no strategy dispatch.
template void Search::simulation<true,false>(int,int,const GumbelSettings*);
template void Search::simulation<true,true>(int,int,const GumbelSettings*);
int Search::gumbel_selection(Node& node,const GumbelSettings& settings,int depth) {
    auto eval=node.evaluated();
    std::vector<GumbelChild> stats(node.move_count);
    for(size_t i=0;i<node.move_count;++i) {
        auto& move=node.moves[i];auto& out=stats[i];out.prior=move_allowed(move.action,node.player,depth)?eval->prior[i]:-1;
        int index=move.child_index.load(std::memory_order_acquire);
        if(index>=0) {
            auto& edge=node.edge(index);out.visits=edge.n.load();out.pending=edge.pending.load();
            Node* child=edge.child.load(std::memory_order_acquire);
            if(child){auto q=snapshot(*child);if(q.weight>0)out.q=edge.perspective.load()*q.value;}
        }
    }
    if(std::none_of(stats.begin(),stats.end(),[](const auto& c){return c.prior>=0;}))return -1;
    return gumbel_interior_selection(stats,eval->initial_value,settings);
}
void Search::mix_human_policy(SearchResult& result,const SearchState& state,double temperature,bool training) {
    auto eval=root_->evaluated();double proportion=settings_.human_sl_chosen_move_proportion;
    if(proportion<=0 || !eval || eval->human_policy.empty())return;
    std::vector<double> utilities(state.actions(),0),human(state.actions(),-1);std::vector<bool> known(state.actions(),false);
    for(int i=0;i<root_->child_count.load();++i){auto& edge=root_->edge(i);auto* child=edge.child.load();if(!child)continue;
        int action=root_->moves[edge.move_index].action;utilities[action]=edge.perspective.load()*snapshot(*child).value;known[action]=true;}
    for(size_t i=0;i<root_->move_count;++i){int action=root_->moves[i].action;if(state.legal(action)&&move_allowed(action,state.player(),0))human[action]=eval->human_policy[i];}
    auto children=child_stats(*root_,*eval);double target_scale=1;
    if(!children.empty()){auto weights=root_selection_weights(children,settings_,settings_.use_lcb,snapshot(*root_),false);target_scale=std::accumulate(weights.begin(),weights.end(),0.0);}
    mix_secondary_policy(result,human,utilities,known,settings_,state.pass_action(),target_scale);(void)temperature;(void)training;
}
void Search::update_time_budget(double elapsed) {
    if(!time_budget_->needs_update(elapsed))return;
    auto eval=root_->evaluated();auto stats=child_stats(*root_,*eval);auto parent=snapshot(*root_);
    if((settings_.human_sl_chosen_move_proportion>0||stats.empty())&&position_){auto selection=analysis_snapshot(position_->actions());
        time_budget_->update(elapsed,parent.visits,stats,eval->search_prior,parent,current_temperature_,&selection);}
    else time_budget_->update(elapsed,parent.visits,stats,eval->search_prior,parent,current_temperature_);
}
void Search::simulate_many(int worker) {
    try {while(!failed_.load() && !stopped_.load()) {
        if(worker==0)update_time_budget(std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count());
        if((should_stop_ && should_stop_()) || (completed_.load()+fresh_playouts_>=2 && time_budget_->limit()<1e12 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count()>=time_budget_->limit())) {
            stopped_=true;break;
        }
        int64_t issued=issued_.load();
        do {if(issued>=budget_)break;} while(!issued_.compare_exchange_weak(issued,issued+1));
        if(issued>=budget_)break;
        simulation(worker);
        if(worker==0 && report_)report_();
    }}
    catch(...){{std::lock_guard<std::mutex> lock(error_mutex_);if(!error_)error_=std::current_exception();}failed_=true;}
}
void Search::worker_loop(int worker) {
    uint64_t seen=0;
    for(;;) {
        std::unique_lock<std::mutex> lock(work_mutex_);
        work_changed_.wait(lock,[&]{return closing_ || generation_!=seen;});if(closing_)return;
        seen=generation_;bool cleaning=cleaning_;lock.unlock();
        if(cleaning)TreeDeleter{settings_.subtree_value_bias_free_prop}(cleanup_roots_[worker]);else if(worker<active_threads_)simulate_many(worker);
        lock.lock();++workers_done_;lock.unlock();work_changed_.notify_all();
    }
}
SearchResult Search::run(const Game& game,double temperature,SearchRun options) {
    if(!evaluator_)throw std::runtime_error("Game search requires an AlphaZero evaluator");
    options.turn=game.turn();options.board_area=game.area();
    options.occupied_points=std::count_if(game.board().cells.begin(),game.board().cells.end(),[](int c){return c!=0;});
    AlphaZeroState state(game,*evaluator_);return run(state,temperature,options);
}
SearchResult Search::run(const SearchState& state,double temperature,SearchRun options) {
    if(temperature<0 || !std::isfinite(temperature))throw std::runtime_error("Invalid search temperature");
    if(options.max_visits<0 || options.max_playouts<-1 || options.max_time<-1 || !std::isfinite(options.max_time))
        throw std::runtime_error("Invalid search caps");
    if(options.hint_action < -1 || (options.hint_action>=0 && !state.legal(options.hint_action)))
        throw std::runtime_error("Illegal root hint action");
    root_player_=state.player();
    int capped=settings_.visit_cap_contempt==0?0:settings_.visit_cap_contempt_player;
    if(settings_.visit_cap_contempt>0 && capped==0)capped=-(options.search_for_player?options.search_for_player:state.player());
    bool cap_changed=capped!=capped_player_;capped_player_=capped;
    bool restrictions_changed=prepare_restrictions(state,options);
    bool hint_changed=options.hint_action!=root_hint_action_;root_hint_action_=options.hint_action;
    if(settings_.graph_search)state.prepare_graph(settings_.graph_rep_bound);
    std::string identity=settings_.graph_search?state.graph_key():std::string();
    graph_hits_=0;graph_catch_ups_=0;graph_cycles_=0;
    for(auto& thread:thread_states_)thread.illegal_graph_nodes.clear();
    start_=std::chrono::steady_clock::now();
    if(options.clear_before_search || restrictions_changed || cap_changed || hint_changed || (settings_.graph_search && !root_->graph_identity.empty() && root_->graph_identity!=identity)) {
        carried_time_=0;root_=nullptr;collect_nodes(nullptr);root_=new_node();
    }
    if(settings_.graph_search)root_->graph_identity=identity;
    if(!eval_cache_.empty())root_->cached_evaluation=cached_evaluation(root_->graph_identity);
    remove_root_noise_=options.remove_root_noise;
    bool filtered=(settings_.root_symmetry_pruning || (settings_.graph_search && state.graph_approximates_history())) &&
        root_->status.load()==Node::READY && prune_root_children(state);
    prepare_patterns(state,options);
    prepare_subtree_bias(state);
    if(filtered) {
        collect_nodes(root_);
        if(subtree_bias_table_)subtree_bias_table_->clear_unused();
    }
    SearchResult result;result.policy.assign(state.actions(),0);result.move_policy=result.policy;
    result.visits.assign(state.actions(),0);result.network_policy=result.policy;result.search_policy=result.policy;
    result.initial_visits=root_->status.load()==Node::READY?root_->visits.load()+1:0;
    int64_t playout_cap=options.max_playouts<0?(options.pondering?settings_.max_playouts_pondering:settings_.max_playouts):options.max_playouts;
    if(state.terminal()) {
        double value=state.terminal_value();
        if(!std::isfinite(value) || std::abs(value)>1)throw std::runtime_error("Invalid terminal value");
        result.value=value;result.network_wdl=result.search_wdl=wdl_from_value(value,state.terminal_draw_probability());
        return result; // EtaZero keeps rule-terminal roots final; no KataGo forceNonTerminal root mode.
    }
    if(result.initial_visits==0 && (playout_cap==0 || (options.should_stop && options.should_stop()))) {
        result.stopped_early=static_cast<bool>(options.should_stop) && options.should_stop();return result;
    }
    remove_root_noise_=options.remove_root_noise;
    root_symmetries_=remove_root_noise_?1:settings_.root_symmetries;
    bool fresh=expand(*root_,state,0);
    fresh_playouts_=fresh?1:0;
    bool changed_legality=false;
    if(!fresh && settings_.graph_search && state.graph_approximates_history()) {
        auto eval=root_->evaluated();
        for(size_t i=0;i<root_->move_count;++i)
            changed_legality|=(eval->prior[i]>=0)!=state.legal(root_->moves[i].action);
    }
    if(!fresh && (changed_legality || (evaluator_&&evaluator_->secondary&&root_->evaluated()->human_policy.empty()) || root_symmetries_>1 || settings_.root_policy_optimism!=settings_.policy_optimism ||
                  (settings_.ignore_pre_root_history && !settings_.ignore_all_history)))
        evaluate_node(*root_,state,0,root_symmetries_,changed_legality);
    int64_t cap=options.max_visits?options.max_visits:(options.pondering?settings_.max_visits_pondering:settings_.max_visits);
    budget_=cap ? std::max<int64_t>(0,cap-1-root_->visits.load()) : settings_.simulations;
    budget_=std::min(budget_,std::max<int64_t>(0,playout_cap-fresh_playouts_));
    active_threads_=settings_.min_playouts_per_thread>0 ? std::max(1,static_cast<int>(std::min(double(settings_.threads),
        std::floor((budget_+fresh_playouts_)/settings_.min_playouts_per_thread)))) : settings_.threads;
    max_time_=options.max_time<0?(options.pondering?settings_.max_time_pondering:settings_.max_time):options.max_time;should_stop_=options.should_stop;stopped_=false;
    auto root_eval=std::make_shared<NodeEvaluation>(*root_->evaluated());
    std::vector<double> policy;std::vector<int> legal_indices;
    for(size_t i=0;i<root_->move_count;++i)if(root_eval->prior[i]>=0) {
        legal_indices.push_back(i);policy.push_back(root_eval->prior[i]);
    }
    double root_temperature=remove_root_noise_?1:temperature_at_turn(settings_.root_policy_temperature_early,
        settings_.root_policy_temperature,settings_.temperature_halflife,options.turn,options.board_area);
    if(root_temperature!=1)policy=policy_temperature_distribution(policy,root_temperature);
    auto proportions=settings_.shaped_noise?noise_alpha_distribution(policy):std::vector<double>(policy.size(),1.0/policy.size());
    double fraction=remove_root_noise_?0:settings_.noise_fraction;
    std::vector<double> noise(policy.size(),0);double sum=1;
    if(fraction>0) {
        do {
            sum=0;
            for(size_t i=0;i<noise.size();++i){noise[i]=std::gamma_distribution<double>(settings_.dirichlet_total_concentration*proportions[i],1)(random_[0]);sum+=noise[i];}
        } while(sum==0);
        if(!std::isfinite(sum))throw std::runtime_error("Invalid Dirichlet draw");
    }
    for(size_t i=0;i<legal_indices.size();++i)
        root_eval->search_prior[legal_indices[i]]=(1-fraction)*policy[i]+fraction*noise[i]/sum;
    if(options.avoid_rescale_root && !avoid_until_[state.player()==1?0:1].empty()) {
        const auto& avoid=avoid_until_[state.player()==1?0:1];double mass=0;
        for(int i:legal_indices)if(avoid[root_->moves[i].action]<=0)mass+=root_eval->search_prior[i];
        if(mass>0)for(int i:legal_indices)root_eval->search_prior[i]=float(root_eval->search_prior[i]/mass);
    }
    if(root_hint_action_>=0) {
        // Source uses float policy and float 0.02, accumulating moved mass in double.
        float proportion=.02f;double moved=0;
        for(int i:legal_indices) {
            float prior=static_cast<float>(root_eval->search_prior[i]);
            moved+=prior*proportion;prior*=1.0f-proportion;root_eval->search_prior[i]=prior;
        }
        for(int i:legal_indices)if(root_->moves[i].action==root_hint_action_)
            root_eval->search_prior[i]=static_cast<float>(static_cast<float>(root_eval->search_prior[i])+static_cast<float>(moved));
    }
    std::sort(root_eval->prior_order.begin(),root_eval->prior_order.end(),[&](int a,int b){return root_eval->search_prior[a]!=root_eval->search_prior[b]?root_eval->search_prior[a]>root_eval->search_prior[b]:root_->moves[a].action<root_->moves[b].action;});
    root_->publish(root_eval);
    if(!fresh)recompute(*root_);

    auto next_report=start_;
    report_=options.on_report?std::function<void()>([&]{
        auto now=std::chrono::steady_clock::now();
        if(now<next_report)return;
        next_report=now+std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(options.report_interval));
        options.on_report(analysis_snapshot(state.actions()));
    }):std::function<void()>{};
    position_=&state;current_temperature_=temperature;time_budget_=std::make_unique<SearchTimeBudget>(settings_,options,carried_time_);
    update_time_budget(std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count());
    auto actual_search_start=std::chrono::steady_clock::now();
    {std::lock_guard<std::mutex> lock(work_mutex_);position_=&state;issued_=0;completed_=0;failed_=false;error_=nullptr;workers_done_=0;cleaning_=false;++generation_;}
    work_changed_.notify_all();simulate_many(0);
    {std::unique_lock<std::mutex> lock(work_mutex_);work_changed_.wait(lock,[&]{return workers_done_==static_cast<int>(workers_.size());});}
    if(error_)std::rethrow_exception(error_);
    carried_time_+=std::chrono::duration<double>(std::chrono::steady_clock::now()-actual_search_start).count();
    record_evaluations();
    report_={};
    if(options.collect_analysis)result.analysis=analysis_snapshot(state.actions()).analysis;
    if(!stopped_ && completed_!=budget_)throw std::runtime_error("Search budget mismatch");
    result.simulations=completed_;result.new_playouts=completed_+fresh_playouts_;result.root_visits=root_->visits.load()+1;
    result.graph_hits=graph_hits_.load();result.graph_catch_ups=graph_catch_ups_.load();
    result.graph_cycles=graph_cycles_.load();result.graph_nodes=node_count();
    result.stopped_early=stopped_;result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count();
    for(size_t i=0;i<root_->move_count;++i) {
        const auto& move=root_->moves[i];
        result.network_policy[move.action]=std::max(0.0,root_eval->prior[i]);result.search_policy[move.action]=std::max(0.0,root_eval->search_prior[i]);
    }
    int64_t total=0;std::vector<int> actions;
    auto stats=child_stats(*root_,*root_eval);
    result.q_values.assign(state.actions(),0);result.q_visits.assign(state.actions(),0);
    for(int i=0;i<root_->child_count.load();++i) {
        auto& e=root_->edge(i);
        int action=root_->moves[e.move_index].action;
        Node* child=e.child.load();
        if(child) {
            auto q=snapshot(*child);
            if(q.visits>0 && q.weight>0) {
                result.q_visits[action]=q.visits;
                result.q_values[action]=static_cast<float>(e.perspective.load()*q.win_loss);
            }
        }
        auto n=e.n.load();if(n<=0)continue;
        actions.push_back(action);total+=n;result.visits[action]=n;
    }
    if(actions.size()!=stats.size())throw std::runtime_error("Root child stats are incomplete");
    if(total>root_->visits.load())throw std::runtime_error("Search has inconsistent completed root visits");
    result.network_wdl=root_eval->initial_wdl;result.network_sample_weight=root_eval->initial_weight;result.network_value_stdev=root_eval->initial_stdev;
    auto parent=snapshot(*root_);result.value=parent.win_loss;result.search_weight=parent.weight;result.search_weight_sq=parent.weight_sq;
    result.search_wdl=wdl_from_value(parent.win_loss,parent.draw);
    if(total==0) {
        result.policy=result.search_policy;
        bool allocated=std::any_of(result.q_visits.begin(),result.q_visits.end(),[](auto n){return n>0;});
        for(size_t action=0;action<result.policy.size();++action)if(!move_allowed(action,state.player(),0)||(allocated&&result.q_visits[action]==0))result.policy[action]=0;
        if(std::accumulate(result.policy.begin(),result.policy.end(),0.0)<=0)return result;
        result.policy=direct_policy_selection(std::move(result.policy),settings_);result.move_policy=result.policy;
        result.policy_target=quantize_policy(result.policy);
        mix_human_policy(result,state,temperature,options.training);
        auto behavior=temperature_distribution(result.move_policy,temperature,settings_.chosen_move_temperature_only_below_prob);
        result.action=std::discrete_distribution<int>(behavior.begin(),behavior.end())(random_[0]);return result;
    }
    auto move_weights=root_selection_weights(stats,settings_,settings_.use_lcb&&!options.training,parent);
    auto target_weights=root_selection_weights(stats,settings_,settings_.use_lcb,parent,false);
    auto quantized=quantize_policy(target_weights);
    normalize(target_weights);
    result.policy_target.assign(state.actions(),0);
    result.move_policy.assign(state.actions(),0);

    for(size_t i=0;i<actions.size();++i) {
        double p=target_weights[i],weight=move_weights[i];
        result.policy[actions[i]]=p;result.policy_target[actions[i]]=quantized[i];result.move_policy[actions[i]]=weight;
        if(p>0)result.policy_surprise+=p*std::log(p/std::max(1e-100,stats[i].prior));
    }
    result.policy_surprise=std::max(0.0,result.policy_surprise);
    mix_human_policy(result,state,temperature,options.training);
    auto behavior=temperature_distribution(result.move_policy,temperature,settings_.chosen_move_temperature_only_below_prob);
    result.action=std::discrete_distribution<int>(behavior.begin(),behavior.end())(random_[0]);return result;
}
void Search::advance(int action) {
    Node* next=nullptr;
    if(settings_.reuse_tree)for(size_t i=0;i<root_->move_count;++i) {
        auto& m=root_->moves[i];int index=m.child_index.load();
        if(m.action==action && index>=0)next=root_->edge(index).child.load();
    }
    set_root_focus({},{},0);avoid_until_={};
    if(next && next->status.load()==Node::READY) {
        carried_time_*=std::min(1.0,double(snapshot(*next).visits)/std::max<int64_t>(1,snapshot(*root_).visits))*settings_.tree_reuse_carry_over_time_factor;
        // KataGo makeMove copies the promoted child: the root never aliases a table-keyed node.
        Node* copy=new_node();copy->graph_identity=next->graph_identity;copy->pattern_key=next->pattern_key;copy->player=next->player;copy->capped_distribution=next->capped_distribution;copy->cached_evaluation=next->cached_evaluation;
        copy->publish(next->evaluated());copy->stats=snapshot(*next);
        copy->visits=copy->stats.visits-1;copy->status=Node::READY;
        copy->move_count=next->move_count;copy->moves=std::make_unique<Move[]>(copy->move_count);
        for(size_t i=0;i<copy->move_count;++i)copy->moves[i].action=next->moves[i].action;
        for(int i=0;i<next->child_count.load();++i) {
            auto& source=next->edge(i);auto& dest=child_edge(*copy,source.move_index);
            dest.n=source.n.load();dest.perspective=source.perspective.load();dest.child=source.child.load();
        }
        root_=copy;
    } else {carried_time_=0;root_=new_node();}
    collect_nodes(root_);
    if(auto eval=root_->evaluated()) {
        auto unnoised=std::make_shared<NodeEvaluation>(*eval);unnoised->search_prior=unnoised->prior;
        root_->publish(std::move(unnoised));
    }
}
std::vector<SearchNodeSnapshot> Search::inspect_graph() {
    std::vector<SearchNodeSnapshot> result;
    for(auto& shard:node_table_)for(auto& item:shard.nodes) {
        Node& node=*item.second;
        SearchNodeSnapshot view{node.id,&node==root_,node.status.load()==Node::READY,node.pending.load(),node.graph_identity,snapshot(node),{}};
        for(int j=0;j<node.child_count.load();++j) {
            auto& edge=node.edge(j);Node* child=edge.child.load();
            if(child)view.edges.push_back({node.moves[edge.move_index].action,edge.perspective.load(),edge.pending.load(),edge.n.load(),child->id});
        }
        result.push_back(std::move(view));
    }
    return result;
}
// Diagnostic only: callers must wait for search workers to finish.
int Search::pending() const {
    int total=0;
    for(const auto& shard:node_table_)for(const auto& item:shard.nodes) {
        const Node& node=*item.second;
        total+=node.pending.load();
        for(int i=0;i<node.child_count.load();++i)total+=node.edge(i).pending.load();
    }
    return total;
}
}
