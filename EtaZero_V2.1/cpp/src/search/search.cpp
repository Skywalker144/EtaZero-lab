#include "etazero/search.h"
#include "etazero/gumbel.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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
                                         const SearchSettings& s,bool use_lcb,ValueStats parent,bool normalize_output) {
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

void Search::TreeDeleter::operator()(Node* node) const noexcept {
    // Each unmarked table-owned node is transferred into exactly one work list.
    while(node) {Node* next=node->cleanup_next;delete node;node=next;}
}
Search::Node* Search::new_node() {
    size_t id=next_node_.fetch_add(1);
    std::string key="private:"+std::to_string(id);
    auto& shard=node_table_[id%node_table_.size()];
    auto owner=std::make_unique<Node>(id);Node* node=owner.get();
    std::lock_guard<std::mutex> lock(shard.mutex);shard.nodes.emplace(std::move(key),std::move(owner));
    return node;
}
Search::Node* Search::find_node(const SearchState& state) {
    if(!settings_.graph_search)return new_node();
    std::string key="graph:"+state.graph_key();
    auto& shard=node_table_[std::hash<std::string>{}(key)%node_table_.size()];
    std::lock_guard<std::mutex> lock(shard.mutex);
    auto found=shard.nodes.find(key);
    if(found!=shard.nodes.end()){++graph_hits_;return found->second.get();}
    auto owner=std::make_unique<Node>(next_node_.fetch_add(1));Node* node=owner.get();
    node->graph_identity=state.graph_key();
    shard.nodes.emplace(std::move(key),std::move(owner));return node;
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
    if(s.simulations<1 || s.threads<1 || s.c_puct<=0 || s.virtual_loss<0 ||
       s.noise_fraction<0 || s.noise_fraction>1 || s.dirichlet_total_concentration<=0 || s.max_visits<0 || (s.use_fpu && (s.fpu_reduction_max<0 || s.root_fpu_reduction_max<0 ||
       s.fpu_parent_power<0 || s.fpu_parent_weight<0 || s.fpu_parent_weight>1)) || s.max_playouts<0 || s.max_time<0 || !std::isfinite(s.max_time) ||
       s.nn_symmetry<0 || s.nn_symmetry>=8 || s.forced_playouts<0 || s.lcb_stdevs<=0 || s.min_lcb_visit_prop<0 || s.min_lcb_visit_prop>1 ||
       s.value_weight_exponent<0 || s.chosen_move_subtract<0 || s.chosen_move_prune<0 || (s.use_fpu && (s.fpu_loss_prop<0 || s.fpu_loss_prop>1 ||
       s.root_fpu_loss_prop<0 || s.root_fpu_loss_prop>1)) || s.c_puct_log<0 || s.c_puct_base<=0 || s.c_puct_stdev_prior<=0 ||
       s.c_puct_stdev_prior_weight<0 || s.c_puct_stdev_scale<0 || s.c_puct_stdev_scale>1 || s.root_symmetries<1 || s.root_symmetries>8 ||
       s.nn_policy_temperature<=0 || s.root_policy_temperature<=0 || s.root_policy_temperature_early<=0 || s.temperature_halflife<=0 ||
       s.chosen_move_temperature_only_below_prob<0 || s.chosen_move_temperature_only_below_prob>1 ||
       !std::isfinite(s.graph_catch_up_leak_prob) || s.graph_catch_up_leak_prob<0 || s.graph_catch_up_leak_prob>1 ||
       (s.use_uncertainty && (!std::isfinite(s.uncertainty_coeff) || s.uncertainty_coeff<0.0001 || s.uncertainty_coeff>1 ||
       !std::isfinite(s.uncertainty_exponent) || s.uncertainty_exponent<0 || s.uncertainty_exponent>2 ||
       !std::isfinite(s.uncertainty_max_weight) || s.uncertainty_max_weight<1 || s.uncertainty_max_weight>100)) ||
       !std::isfinite(s.policy_optimism) || s.policy_optimism<0 || s.policy_optimism>1 ||
       !std::isfinite(s.root_policy_optimism) || s.root_policy_optimism<0 || s.root_policy_optimism>1 ||
       (s.use_noise_pruning && (!std::isfinite(s.noise_prune_utility_scale) || s.noise_prune_utility_scale<0.001 || s.noise_prune_utility_scale>10 ||
       !std::isfinite(s.noise_pruning_cap) || s.noise_pruning_cap<0 || s.noise_pruning_cap>1e50)))
        throw std::runtime_error("Invalid search settings");
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
    if(pending()!=0)throw std::runtime_error("Cannot collect nodes with pending visits");
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
        for(auto node:cleanup_roots_)TreeDeleter{}(node);
    } else {
        {std::lock_guard<std::mutex> lock(work_mutex_);cleaning_=true;workers_done_=0;++generation_;}
        work_changed_.notify_all();TreeDeleter{}(cleanup_roots_[0]);
        std::unique_lock<std::mutex> lock(work_mutex_);
        work_changed_.wait(lock,[&]{return workers_done_==static_cast<int>(workers_.size());});cleaning_=false;
    }
    std::fill(cleanup_roots_.begin(),cleanup_roots_.end(),nullptr);
}
void Search::reset(uint64_t seed) {
    root_=nullptr;collect_nodes(nullptr);root_=new_node();
    for(int i=0;i<settings_.threads;++i) {
        random_[i].seed(seed+0x9e3779b97f4a7c15ULL*i);
        inference_random_[i].seed((seed^0xd1b54a32d192ed03ULL)+0x9e3779b97f4a7c15ULL*i);
    }
}
void Search::set_evaluator(Evaluator& evaluator) {
    root_=nullptr;collect_nodes(nullptr);root_=new_node();evaluator_=&evaluator;
    for(auto& thread:thread_states_)thread.state.reset();
}
void Search::evaluate_node(Node& node,const SearchState& state,int worker,int symmetries) {
    if(symmetries>state.symmetry_count())throw std::runtime_error("Root symmetry count exceeds state support");
    std::vector<int> symmetry_indices(state.symmetry_count());std::iota(symmetry_indices.begin(),symmetry_indices.end(),0);
    std::vector<double> policy(state.actions(),0);WDL averaged{0,0,0};double stdev=0;
    bool auxiliary=false;double optimism=&node==root_?settings_.root_policy_optimism:settings_.policy_optimism;
    for(int i=0;i<symmetries;++i) {
        if(symmetries>1)std::swap(symmetry_indices[i],symmetry_indices[std::uniform_int_distribution<int>(i,state.symmetry_count()-1)(random_[worker])]);
        int symmetry=symmetry_indices[i];
        if(symmetries==1) {
            symmetry=settings_.nn_randomize && state.symmetry_count()>1?
                std::uniform_int_distribution<int>(0,state.symmetry_count()-1)(inference_random_[worker]):settings_.nn_symmetry;
        }
        if(symmetry>=state.symmetry_count())throw std::runtime_error("NN symmetry exceeds state support");
        auto output=state.evaluate_symmetry(symmetry,symmetries>1,settings_.nn_policy_temperature,
                                           symmetries==1 && settings_.nn_randomize,optimism);
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
        for(int a=0;a<state.actions();++a)if(state.legal(a)) {
            if(!std::isfinite(output.logits[a]))throw std::runtime_error("Nonfinite leaf policy");
            maximum=std::max(maximum,output.logits[a]);
        }
        std::vector<double> probabilities(state.actions(),0);
        for(int a=0;a<state.actions();++a)if(state.legal(a)) {
            probabilities[a]=std::exp((output.logits[a]-maximum)/settings_.nn_policy_temperature);sum+=probabilities[a];
        }
        if(!std::isfinite(sum)||sum<=0)throw std::runtime_error("No valid leaf policy");
        for(int a=0;a<state.actions();++a)policy[a]+=probabilities[a]/sum/symmetries;
    }
    if(!node.moves) {
        size_t count=0;for(int a=0;a<state.actions();++a)if(state.legal(a))++count;
        node.moves=std::make_unique<Move[]>(count);node.move_count=count;
        size_t index=0;for(int a=0;a<state.actions();++a)if(state.legal(a))node.moves[index++].action=a;
        node.prior_order.resize(count);std::iota(node.prior_order.begin(),node.prior_order.end(),0);
    }
    for(size_t i=0;i<node.move_count;++i)node.moves[i].search_prior=node.moves[i].prior=policy[node.moves[i].action];
    std::sort(node.prior_order.begin(),node.prior_order.end(),[&](int a,int b){return node.moves[a].prior>node.moves[b].prior;});
    std::lock_guard<std::mutex> lock(node.stats_mutex);
    node.initial_wdl=averaged;node.initial_value=averaged[0]-averaged[2];node.initial_stdev=stdev;node.has_auxiliary=auxiliary;
    node.initial_weight=uncertainty_weight(stdev,auxiliary,settings_);
    if(node.stats.visits==0)node.stats={1,node.initial_value,node.initial_value*node.initial_value,averaged[1],node.initial_weight,node.initial_weight*node.initial_weight};
}
ValueStats Search::snapshot(Node& node) {std::lock_guard<std::mutex> lock(node.stats_mutex);return node.stats;}
std::vector<RootChildStats> Search::child_stats(Node& node,bool aggregation) {
    std::vector<RootChildStats> children;children.reserve(node.child_count.load());
    int count=node.child_count.load(std::memory_order_acquire);
    for(int i=0;i<count;++i) {
        auto& edge=node.edge(i);Node* child=edge.child.load(std::memory_order_acquire);auto n=edge.n.load();
        if(!child || n<=0)continue;
        auto stats=snapshot(*child);if(stats.visits<=0 || stats.weight<=0)continue;
        double scale=double(n)/stats.visits,weight=stats.weight*scale;
        double perspective=edge.perspective.load();
        children.push_back({node.moves[edge.move_index].search_prior,weight*perspective*stats.value,
                            weight*stats.value_sq,n,weight,stats.weight_sq*scale*(aggregation?scale:1),weight*stats.draw});
    }
    return children;
}
void Search::recompute(Node& node,int visits_to_add) {
    std::lock_guard<std::mutex> update(node.update_mutex);
    // LCB uses linear edge sampling weightSq; parent aggregation scales a weighted mean quadratically.
    auto children=child_stats(node,true);
    auto result=aggregate_values(node.initial_wdl,children,settings_,&node==root_ && !remove_root_noise_ && settings_.noise_fraction>0,node.initial_weight);
    std::lock_guard<std::mutex> lock(node.stats_mutex);
    result.visits=node.stats.visits+visits_to_add;node.stats=result;
}
bool Search::expand(Node& node,const SearchState& state,int worker) {
    auto status=node.status.load(std::memory_order_acquire);
    if(status==Node::READY)return false;
    auto& sync=locks_[node.lock_index];std::unique_lock<std::mutex> lock(sync.mutex);
    sync.changed.wait(lock,[&]{return node.status.load(std::memory_order_acquire)!=Node::EXPANDING;});
    status=node.status.load(std::memory_order_acquire);
    if(status==Node::FAILED)std::rethrow_exception(node.error);
    if(status==Node::READY)return false;
    node.status.store(Node::EXPANDING,std::memory_order_release);lock.unlock();
    try {
        evaluate_node(node,state,worker,&node==root_?root_symmetries_:1);
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
template<bool ForcedRoot,bool GumbelInterior>
void Search::simulation(int worker,int root_move,const GumbelSettings* gumbel) {
    auto& scratch=thread_states_[worker];auto& rng=random_[worker];
    if(!scratch.state || !scratch.state->reset_from(*position_))scratch.state=position_->clone();
    auto& state=*scratch.state;Node* node=root_;auto& path=scratch.path;path.clear();scratch.graph_path.clear();
    auto release=[&]{for(auto& p:path){--p.edge->pending;--p.edge->child.load()->pending;--pending_count_;}path.clear();};
    try {
        double value,draw;
        for(;;) {
            if(state.terminal()) {
                value=state.terminal_value();draw=value==0?1:0;
                if(!std::isfinite(value) || std::abs(value)>1)throw std::runtime_error("Invalid terminal value");
                std::lock_guard<std::mutex> lock(node->stats_mutex);
                auto& stats=node->stats;++stats.visits;stats.value=value;stats.value_sq=value*value;stats.draw=draw;
                double terminal_weight=settings_.use_uncertainty && root_->has_auxiliary?settings_.uncertainty_max_weight:1;
                stats.weight=stats.visits*terminal_weight;stats.weight_sq=stats.visits*terminal_weight*terminal_weight;break;
            }
            if(expand(*node,state,worker)){value=node->initial_value;draw=node->initial_wdl[1];break;}
            int choice=-1;
            if constexpr(ForcedRoot) {
                if(node==root_)choice=root_move;
                else if constexpr(GumbelInterior)choice=gumbel_selection(*node,*gumbel);
            }
            if(choice<0) {
            auto& candidates=scratch.candidates;
            do {
                candidates.clear();
                double best=-std::numeric_limits<double>::infinity();
                auto consider=[&](int i,double score){if(score>best){best=score;candidates.clear();}if(score==best)candidates.push_back(i);};
                int count=node->child_count.load(std::memory_order_acquire);
                double visited_mass=0,total=0;auto parent=snapshot(*node);
                std::vector<ValueStats> child_values(count);
                for(int i=0;i<count;++i) {
                    auto& e=node->edge(i);Node* child=e.child.load(std::memory_order_acquire);
                    if(child) {
                        visited_mass+=node->moves[e.move_index].search_prior;
                        child_values[i]=snapshot(*child);auto& stats=child_values[i];
                        stats.weight*=double(e.n.load())/std::max<int64_t>(1,stats.visits);total+=stats.weight;
                    }
                }
                bool is_root=node==root_;
                double reduction=is_root&&!remove_root_noise_?settings_.root_fpu_reduction_max:settings_.fpu_reduction_max;
                double fpu=settings_.use_fpu?fpu_value(node->initial_value,parent.value,visited_mass,settings_.fpu_parent_power,reduction,
                    settings_.fpu_parent_weight_by_visited_policy,settings_.fpu_parent_weight):0;
                if(settings_.use_fpu) {
                    double loss_prop=is_root&&!remove_root_noise_?settings_.root_fpu_loss_prop:settings_.fpu_loss_prop;
                    fpu+=(-1-fpu)*loss_prop;
                }
                for(int i=0;i<count;++i) {
                    auto& e=node->edge(i);auto& m=node->moves[e.move_index];int pending=settings_.graph_search && e.child.load()?e.child.load()->pending.load(std::memory_order_relaxed):e.pending.load(std::memory_order_relaxed);
                    const auto& stats=child_values[i];double q=stats.weight>0?e.perspective.load()*stats.value:fpu;
                    double score=child_selection_score(m.search_prior,q,stats.weight,pending,total,parent,settings_,is_root&&!remove_root_noise_);
                    if(is_root && m.action==root_hint_action_) {
                        double weight=stats.weight+pending*settings_.virtual_loss;
                        double parent_per_visit=parent.weight/std::max<int64_t>(1,parent.visits);
                        double next_weight=(weight+parent_per_visit)/(stats.visits+1.0);
                        for(const auto& other:child_values)if(weight+next_weight < other.weight*.8) {score=1e20;break;}
                    }
                    consider(e.move_index,score);
                }
                if(is_root && root_hint_action_>=0) {
                    for(size_t i=0;i<node->move_count;++i)if(node->moves[i].action==root_hint_action_ && node->moves[i].child_index.load()<0) {
                        double parent_per_visit=parent.weight/std::max<int64_t>(1,parent.visits);
                        for(const auto& other:child_values)if(parent_per_visit<other.weight*.8) {consider(i,1e20);break;}
                    }
                }
                // Every unvisited legal action is still in the action domain. Its Q=FPU
                // and denominator=1, so a sorted prior list supplies all best ties.
                for(int i:node->prior_order)if(node->moves[i].child_index.load(std::memory_order_acquire)<0) {
                    double score=fpu+explore_scaling(total,parent,settings_)*node->moves[i].search_prior;
                    if(score<best)break;
                    consider(i,score);
                }
                // A concurrent allocation can move the last candidate between the
                // two snapshots. Retry instead of losing or dereferencing it.
            } while(candidates.empty());
            std::sort(candidates.begin(),candidates.end());
            // An activation can be visible in child_count before its move index;
            // both snapshots may then include the same action. Keep tie sampling
            // uniform over actions even across that publication interval.
            candidates.erase(std::unique(candidates.begin(),candidates.end()),candidates.end());
            choice=candidates[std::uniform_int_distribution<size_t>(0,candidates.size()-1)(rng)];
            }
            Edge* edge=&child_edge(*node,choice);
            auto transition=state.move(node->moves[choice].action);
            if(transition.reward!=0 || transition.discount!=1 ||
               (transition.perspective!=1 && transition.perspective!=-1))throw std::runtime_error("WDL AlphaZero search requires zero reward and unit discount");
            Node* child=edge->child.load(std::memory_order_acquire);
            if(!child) {
                Node* candidate=find_node(state);Node* expected=nullptr;
                if(edge->child.compare_exchange_strong(expected,candidate,std::memory_order_release,std::memory_order_acquire))child=candidate;
                else child=expected; // Unused racing candidates are collected with the table.
            }
            path.push_back({node,edge,transition,true});++edge->pending;++child->pending;++pending_count_;
            int expected=0,perspective=int(transition.perspective);
            edge->perspective.compare_exchange_strong(expected,perspective);
            if(expected!=0 && expected!=perspective)throw std::runtime_error("Inconsistent edge perspective");
            if(settings_.graph_search && catch_up(*edge,*child,worker)) {
                path.back().increment_edge=false;value=snapshot(*child).value;draw=0;break;
            }
            if(settings_.graph_search && !scratch.graph_path.insert(child).second) {
                ++graph_cycles_;value=snapshot(*child).value;draw=0;break;
            }
            node=child;
        }
        while(!path.empty()) {
            auto p=path.back();value=p.transition.reward+p.transition.discount*p.transition.perspective*value;
            if(!std::isfinite(value))throw std::runtime_error("Nonfinite search backup");
            if(p.increment_edge)++p.edge->n;
            ++p.parent->visits;recompute(*p.parent,1);
            --p.edge->pending;--p.edge->child.load()->pending;--pending_count_;path.pop_back();
        }
        ++completed_;
    } catch(...){release();throw;}
}
// Only Gumbel search uses these instantiations. The PUCT specialization has no strategy dispatch.
template void Search::simulation<true,false>(int,int,const GumbelSettings*);
template void Search::simulation<true,true>(int,int,const GumbelSettings*);
int Search::gumbel_selection(Node& node,const GumbelSettings& settings) {
    std::vector<GumbelChild> stats(node.move_count);
    for(size_t i=0;i<node.move_count;++i) {
        auto& move=node.moves[i];auto& out=stats[i];out.prior=move.prior;
        int index=move.child_index.load(std::memory_order_acquire);
        if(index>=0) {
            auto& edge=node.edge(index);out.visits=edge.n.load();out.pending=edge.pending.load();
            Node* child=edge.child.load(std::memory_order_acquire);
            if(child){auto q=snapshot(*child);if(q.weight>0)out.q=edge.perspective.load()*q.value;}
        }
    }
    return gumbel_interior_selection(stats,node.initial_value,settings);
}
void Search::simulate_many(int worker) {
    try {while(!failed_.load() && !stopped_.load()) {
        if((should_stop_ && should_stop_()) || (completed_.load()+fresh_playouts_>=2 && max_time_<1e12 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count()>=max_time_)) {
            stopped_=true;break;
        }
        if(issued_.fetch_add(1)>=budget_)break;
        simulation(worker);
    }}
    catch(...){{std::lock_guard<std::mutex> lock(error_mutex_);if(!error_)error_=std::current_exception();}failed_=true;}
}
void Search::worker_loop(int worker) {
    uint64_t seen=0;
    for(;;) {
        std::unique_lock<std::mutex> lock(work_mutex_);
        work_changed_.wait(lock,[&]{return closing_ || generation_!=seen;});if(closing_)return;
        seen=generation_;bool cleaning=cleaning_;lock.unlock();
        if(cleaning)TreeDeleter{}(cleanup_roots_[worker]);else simulate_many(worker);
        lock.lock();++workers_done_;lock.unlock();work_changed_.notify_all();
    }
}
SearchResult Search::run(const Game& game,double temperature,SearchRun options) {
    if(!evaluator_)throw std::runtime_error("Game search requires an AlphaZero evaluator");
    options.turn=game.turn();options.board_area=game.size()*game.size();
    AlphaZeroState state(game,*evaluator_);return run(state,temperature,options);
}
SearchResult Search::run(const SearchState& state,double temperature,SearchRun options) {
    if(temperature<0 || !std::isfinite(temperature))throw std::runtime_error("Invalid search temperature");
    if(options.max_visits<0 || options.max_playouts<-1 || options.max_time<-1 || !std::isfinite(options.max_time))
        throw std::runtime_error("Invalid search caps");
    if(options.hint_action < -1 || (options.hint_action>=0 && !state.legal(options.hint_action)))
        throw std::runtime_error("Illegal root hint action");
    bool hint_changed=options.hint_action!=root_hint_action_;root_hint_action_=options.hint_action;
    std::string identity=settings_.graph_search?state.graph_key():std::string();
    graph_hits_=0;graph_catch_ups_=0;graph_cycles_=0;
    start_=std::chrono::steady_clock::now();
    if(options.clear_before_search || hint_changed || (settings_.graph_search && !root_->graph_identity.empty() && root_->graph_identity!=identity)) {
        root_=nullptr;collect_nodes(nullptr);root_=new_node();
    }
    if(settings_.graph_search)root_->graph_identity=identity;
    SearchResult result;result.policy.assign(state.actions(),0);result.move_policy=result.policy;
    result.visits.assign(state.actions(),0);result.network_policy=result.policy;result.search_policy=result.policy;
    result.initial_visits=root_->status.load()==Node::READY?root_->visits.load()+1:0;
    int playout_cap=options.max_playouts<0?settings_.max_playouts:options.max_playouts;
    if(state.terminal()) {
        double value=state.terminal_value();
        if(!std::isfinite(value) || std::abs(value)>1)throw std::runtime_error("Invalid terminal value");
        result.value=value;result.network_wdl=result.search_wdl={(value+std::abs(value))/2,1-std::abs(value),(std::abs(value)-value)/2};
        return result; // Finished Gomoku games cannot be extended (unlike Go's forceNonTerminal root).
    }
    if(result.initial_visits==0 && (playout_cap==0 || (options.should_stop && options.should_stop()))) {
        result.stopped_early=static_cast<bool>(options.should_stop) && options.should_stop();return result;
    }
    remove_root_noise_=options.remove_root_noise;
    root_symmetries_=remove_root_noise_?1:settings_.root_symmetries;
    bool fresh=expand(*root_,state,0);
    fresh_playouts_=fresh?1:0;
    if(!fresh && (root_symmetries_>1 || settings_.root_policy_optimism!=settings_.policy_optimism))evaluate_node(*root_,state,0,root_symmetries_);
    int cap=options.max_visits?options.max_visits:settings_.max_visits;
    budget_=cap ? static_cast<int>(std::max<int64_t>(0,cap-1-root_->visits.load())) : settings_.simulations;
    budget_=std::min(budget_,std::max(0,playout_cap-fresh_playouts_));
    max_time_=options.max_time<0?settings_.max_time:options.max_time;should_stop_=options.should_stop;stopped_=false;
    std::vector<double> policy;for(size_t i=0;i<root_->move_count;++i)policy.push_back(root_->moves[i].prior);
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
    for(size_t i=0;i<root_->move_count;++i) {
        auto& m=root_->moves[i];m.search_prior=(1-fraction)*policy[i]+fraction*noise[i]/sum;
    }
    if(root_hint_action_>=0) {
        // Source uses float policy and float 0.02, accumulating moved mass in double.
        float proportion=.02f;double moved=0;
        for(size_t i=0;i<root_->move_count;++i) {
            auto& m=root_->moves[i];float prior=static_cast<float>(m.search_prior);
            moved+=prior*proportion;prior*=1.0f-proportion;m.search_prior=prior;
        }
        for(size_t i=0;i<root_->move_count;++i)if(root_->moves[i].action==root_hint_action_)
            root_->moves[i].search_prior=static_cast<float>(static_cast<float>(root_->moves[i].search_prior)+static_cast<float>(moved));
    }
    std::sort(root_->prior_order.begin(),root_->prior_order.end(),[&](int a,int b){return root_->moves[a].search_prior>root_->moves[b].search_prior;});
    if(!fresh)recompute(*root_);
    {std::lock_guard<std::mutex> lock(work_mutex_);position_=&state;issued_=0;completed_=0;failed_=false;error_=nullptr;workers_done_=0;cleaning_=false;++generation_;}
    work_changed_.notify_all();simulate_many(0);
    {std::unique_lock<std::mutex> lock(work_mutex_);work_changed_.wait(lock,[&]{return workers_done_==static_cast<int>(workers_.size());});}
    if(pending()!=0)throw std::runtime_error("Search leaked pending visits");
    if(error_)std::rethrow_exception(error_);
    if(!stopped_ && completed_!=budget_)throw std::runtime_error("Search budget mismatch");
    result.simulations=completed_;result.new_playouts=completed_+fresh_playouts_;result.root_visits=root_->visits.load()+1;
    result.graph_hits=graph_hits_.load();result.graph_catch_ups=graph_catch_ups_.load();
    result.graph_cycles=graph_cycles_.load();result.graph_nodes=node_count();
    result.stopped_early=stopped_;result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count();
    for(size_t i=0;i<root_->move_count;++i) {
        const auto& move=root_->moves[i];result.network_policy[move.action]=move.prior;result.search_policy[move.action]=move.search_prior;
    }
    int64_t total=0;std::vector<int> actions;
    auto stats=child_stats(*root_);
    result.q_values.assign(state.actions(),0);result.q_visits.assign(state.actions(),0);
    for(int i=0;i<root_->child_count.load();++i) {
        auto& e=root_->edge(i);
        int action=root_->moves[e.move_index].action;
        Node* child=e.child.load();
        if(child) {
            auto q=snapshot(*child);
            if(q.visits>0 && q.weight>0) {
                result.q_visits[action]=q.visits;
                result.q_values[action]=static_cast<float>(e.perspective.load()*q.value);
            }
        }
        auto n=e.n.load();if(n<=0)continue;
        actions.push_back(action);total+=n;result.visits[action]=n;
    }
    if(actions.size()!=stats.size())throw std::runtime_error("Root child stats are incomplete");
    if(total!=root_->visits.load())throw std::runtime_error("Search has inconsistent completed root visits");
    result.network_wdl=root_->initial_wdl;result.network_sample_weight=root_->initial_weight;result.network_value_stdev=root_->initial_stdev;
    auto parent=snapshot(*root_);result.value=parent.value;result.search_weight=parent.weight;result.search_weight_sq=parent.weight_sq;double d=parent.draw;
    result.search_wdl={(1-d+result.value)/2,d,(1-d-result.value)/2};
    if(total==0) {
        result.policy=result.move_policy=result.search_policy;
        result.policy_target=quantize_policy(result.policy);
        auto behavior=temperature_distribution(result.move_policy,temperature,settings_.chosen_move_temperature_only_below_prob);
        result.action=std::discrete_distribution<int>(behavior.begin(),behavior.end())(random_[0]);return result;
    }
    auto move_weights=root_selection_weights(stats,settings_,settings_.use_lcb&&!options.training,parent);
    auto target_weights=root_selection_weights(stats,settings_,settings_.use_lcb,parent,false);
    auto quantized=quantize_policy(target_weights);
    normalize(target_weights);
    result.policy_target.assign(state.actions(),0);
    result.move_policy.assign(state.actions(),0);
    auto behavior=temperature_distribution(move_weights,temperature,settings_.chosen_move_temperature_only_below_prob);
    for(size_t i=0;i<actions.size();++i) {
        double p=target_weights[i],weight=move_weights[i];
        result.policy[actions[i]]=p;result.policy_target[actions[i]]=quantized[i];result.move_policy[actions[i]]=weight;
        if(p>0)result.policy_surprise+=p*std::log(p/std::max(1e-100,stats[i].prior));
    }
    result.policy_surprise=std::max(0.0,result.policy_surprise);
    result.action=actions[std::discrete_distribution<size_t>(behavior.begin(),behavior.end())(random_[0])];
    return result;
}
void Search::advance(int action) {
    Node* next=nullptr;
    if(settings_.reuse_tree)for(size_t i=0;i<root_->move_count;++i) {
        auto& m=root_->moves[i];int index=m.child_index.load();
        if(m.action==action && index>=0)next=root_->edge(index).child.load();
    }
    if(next && next->status.load()==Node::READY) {
        // KataGo makeMove copies the promoted child: the root never aliases a table-keyed node.
        Node* copy=new_node();copy->graph_identity=next->graph_identity;
        copy->initial_value=next->initial_value;copy->initial_wdl=next->initial_wdl;copy->stats=snapshot(*next);
        copy->initial_weight=next->initial_weight;copy->initial_stdev=next->initial_stdev;copy->has_auxiliary=next->has_auxiliary;
        copy->visits=copy->stats.visits-1;copy->status=Node::READY;
        copy->move_count=next->move_count;copy->moves=std::make_unique<Move[]>(copy->move_count);
        copy->prior_order=next->prior_order;
        for(size_t i=0;i<copy->move_count;++i) {
            copy->moves[i].action=next->moves[i].action;copy->moves[i].prior=next->moves[i].prior;
            copy->moves[i].search_prior=next->moves[i].search_prior;
        }
        for(int i=0;i<next->child_count.load();++i) {
            auto& source=next->edge(i);auto& dest=child_edge(*copy,source.move_index);
            dest.n=source.n.load();dest.perspective=source.perspective.load();dest.child=source.child.load();
        }
        root_=copy;
    } else root_=new_node();
    collect_nodes(root_);
    for(size_t i=0;i<root_->move_count;++i)root_->moves[i].search_prior=root_->moves[i].prior;
}
std::vector<SearchNodeSnapshot> Search::inspect_graph() {
    if(pending()!=0)throw std::runtime_error("Cannot inspect graph with pending visits");
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
int Search::pending() const {return pending_count_.load();}
}
