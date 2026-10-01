#include "etazero/search.h"
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
double fpu_value(double nn_value,double parent_value,double visited_mass,double power,double reduction) {
    double mass=std::clamp(visited_mass,0.0,1.0),mix=std::pow(mass,power);
    return mix*parent_value+(1-mix)*nn_value-reduction*std::sqrt(mass);
}
std::vector<double> root_selection_weights(const std::vector<RootChildStats>& children,
                                         const SearchSettings& s,bool use_lcb,ValueStats parent) {
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
            if(gap>0)weights[i]=std::ceil(std::min(c.weight,std::max(0.0,explore*c.prior/gap-1)));
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
    normalize(weights);return weights;
}

void Search::TreeDeleter::operator()(Node* node) const noexcept {
    // Intrusive work list: no recursion and no allocation during destruction.
    while(node) {
        Node* next=node->cleanup_next;
        for(int i=0;i<node->child_count.load(std::memory_order_relaxed);++i) {
            Node* child=node->edge(i).child.load(std::memory_order_relaxed);
            if(child) {child->cleanup_next=next;next=child;}
        }
        delete node;node=next;
    }
}
Search::Node* Search::new_node() {return new Node(next_node_.fetch_add(1)%locks_.size());}
Search::Search(Evaluator& e,SearchSettings s,uint64_t seed) : Search(s,seed) {evaluator_=&e;}
Search::Search(SearchSettings s,uint64_t seed) : settings_(s) {
    if(s.simulations<1 || s.threads<1 || s.c_puct<=0 || s.virtual_loss<0 ||
       s.noise_fraction<0 || s.noise_fraction>1 || s.dirichlet_total_concentration<=0 || s.max_visits<0 || s.fpu_reduction_max<0 || s.root_fpu_reduction_max<0 ||
       s.fpu_parent_power<=0 || s.forced_playouts<0 || s.lcb_stdevs<=0 || s.min_lcb_visit_prop<0 || s.min_lcb_visit_prop>1 ||
       s.value_weight_exponent<0 || s.chosen_move_subtract<0 || s.chosen_move_prune<0 || s.fpu_loss_prop<0 || s.fpu_loss_prop>1 ||
       s.root_fpu_loss_prop<0 || s.root_fpu_loss_prop>1 || s.c_puct_log<0 || s.c_puct_base<=0 || s.c_puct_stdev_prior<=0 ||
       s.c_puct_stdev_prior_weight<0 || s.c_puct_stdev_scale<0 || s.c_puct_stdev_scale>1 || s.root_symmetries<1 || s.root_symmetries>8 ||
       s.nn_policy_temperature<=0 || s.root_policy_temperature<=0 || s.root_policy_temperature_early<=0 || s.temperature_halflife<=0 ||
       s.chosen_move_temperature_only_below_prob<0 || s.chosen_move_temperature_only_below_prob>1)
        throw std::runtime_error("Invalid search settings");
    root_.reset(new_node());thread_states_.resize(s.threads);cleanup_roots_.resize(s.threads,nullptr);
    for(int i=0;i<s.threads;++i)random_.emplace_back(seed+0x9e3779b97f4a7c15ULL*i);
    try {for(int i=1;i<s.threads;++i)workers_.emplace_back(&Search::worker_loop,this,i);}
    catch(...) {
        {std::lock_guard<std::mutex> lock(work_mutex_);closing_=true;}
        work_changed_.notify_all();for(auto& t:workers_)t.join();throw;
    }
}
Search::~Search() {
    clear_tree(root_.release());
    {std::lock_guard<std::mutex> lock(work_mutex_);closing_=true;}
    work_changed_.notify_all();for(auto& t:workers_)t.join();
}
void Search::clear_tree(Node* node) {
    if(!node)return;
    int branches=0;int64_t discarded_visits=0;
    for(int i=0;i<node->child_count.load();++i) {
        auto& edge=node->edge(i);
        if(edge.child.load()) {++branches;discarded_visits+=edge.n.load();}
    }
    if(workers_.empty() || branches<2 || discarded_visits<4096) {TreeDeleter{}(node);return;}
    std::fill(cleanup_roots_.begin(),cleanup_roots_.end(),nullptr);
    int index=0;
    for(int i=0;i<node->child_count.load();++i) {
        Node* child=node->edge(i).child.exchange(nullptr);
        if(child) {
            auto& list=cleanup_roots_[index++%cleanup_roots_.size()];child->cleanup_next=list;list=child;
        }
    }
    delete node;
    {std::lock_guard<std::mutex> lock(work_mutex_);cleaning_=true;workers_done_=0;++generation_;}
    work_changed_.notify_all();TreeDeleter{}(cleanup_roots_[0]);
    {std::unique_lock<std::mutex> lock(work_mutex_);
     work_changed_.wait(lock,[&]{return workers_done_==static_cast<int>(workers_.size());});cleaning_=false;}
}
void Search::reset(uint64_t seed) {
    clear_tree(root_.release());root_.reset(new_node());
    for(int i=0;i<settings_.threads;++i)random_[i].seed(seed+0x9e3779b97f4a7c15ULL*i);
}
void Search::evaluate_node(Node& node,const SearchState& state,int worker,int symmetries) {
    if(symmetries>state.symmetry_count())throw std::runtime_error("Root symmetry count exceeds state support");
    std::vector<int> symmetry_indices(state.symmetry_count());std::iota(symmetry_indices.begin(),symmetry_indices.end(),0);
    std::vector<double> policy(state.actions(),0);WDL averaged{0,0,0};
    for(int i=0;i<symmetries;++i) {
        if(symmetries>1)std::swap(symmetry_indices[i],symmetry_indices[std::uniform_int_distribution<int>(i,state.symmetry_count()-1)(random_[worker])]);
        auto output=state.evaluate_symmetry(symmetry_indices[i]);
        if(output.logits.size()!=policy.size())throw std::runtime_error("Bad leaf policy shape");
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
    node.initial_wdl=averaged;node.initial_value=averaged[0]-averaged[2];
    if(node.stats.visits==0)node.stats={1,node.initial_value,node.initial_value*node.initial_value,averaged[1],1,1};
}
ValueStats Search::snapshot(Node& node) {std::lock_guard<std::mutex> lock(node.stats_mutex);return node.stats;}
std::vector<RootChildStats> Search::child_stats(Node& node) {
    std::vector<RootChildStats> children;children.reserve(node.child_count.load());
    int count=node.child_count.load(std::memory_order_acquire);
    for(int i=0;i<count;++i) {
        auto& edge=node.edge(i);Node* child=edge.child.load(std::memory_order_acquire);auto n=edge.n.load();
        if(!child || n<=0)continue;
        auto stats=snapshot(*child);if(stats.visits<=0 || stats.weight<=0)continue;
        double scale=double(n)/stats.visits,weight=stats.weight*scale;
        double perspective=edge.perspective.load();
        children.push_back({node.moves[edge.move_index].search_prior,weight*perspective*stats.value,
                            weight*stats.value_sq,n,weight,stats.weight_sq*scale,weight*stats.draw});
    }
    return children;
}
void Search::recompute(Node& node) {
    node.stats=aggregate_values(node.initial_wdl,child_stats(node),settings_,&node==root_.get() && !remove_root_noise_ && settings_.noise_fraction>0);
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
        evaluate_node(node,state,worker,&node==root_.get()?root_symmetries_:1);
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
void Search::simulation(int worker) {
    auto& scratch=thread_states_[worker];auto& rng=random_[worker];
    if(!scratch.state || !scratch.state->reset_from(*position_))scratch.state=position_->clone();
    auto& state=*scratch.state;Node* node=root_.get();auto& path=scratch.path;path.clear();
    auto release=[&]{for(auto& p:path){--p.edge->pending;--p.parent->pending;--pending_count_;}path.clear();};
    try {
        double value,draw;
        for(;;) {
            if(state.terminal()) {
                value=state.terminal_value();draw=value==0?1:0;
                if(!std::isfinite(value) || std::abs(value)>1)throw std::runtime_error("Invalid terminal value");
                std::lock_guard<std::mutex> lock(node->stats_mutex);
                auto& stats=node->stats;++stats.visits;stats.value=value;stats.value_sq=value*value;stats.draw=draw;
                stats.weight=stats.weight_sq=stats.visits;break;
            }
            if(expand(*node,state,worker)){value=node->initial_value;draw=node->initial_wdl[1];break;}
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
                bool is_root=node==root_.get();
                double reduction=is_root&&!remove_root_noise_?settings_.root_fpu_reduction_max:settings_.fpu_reduction_max;
                double fpu=settings_.use_fpu?fpu_value(node->initial_value,parent.value,visited_mass,settings_.fpu_parent_power,reduction):0;
                if(settings_.use_fpu) {
                    double loss_prop=is_root&&!remove_root_noise_?settings_.root_fpu_loss_prop:settings_.fpu_loss_prop;
                    fpu+=(-1-fpu)*loss_prop;
                }
                for(int i=0;i<count;++i) {
                    auto& e=node->edge(i);auto& m=node->moves[e.move_index];int pending=e.pending.load(std::memory_order_relaxed);
                    const auto& stats=child_values[i];double q=stats.weight>0?e.perspective.load()*stats.value:fpu;
                    double score=child_selection_score(m.search_prior,q,stats.weight,pending,total,parent,settings_,is_root&&!remove_root_noise_);
                    consider(e.move_index,score);
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
            int choice=candidates[std::uniform_int_distribution<size_t>(0,candidates.size()-1)(rng)];
            Edge* edge=&child_edge(*node,choice);Node* child=edge->child.load(std::memory_order_acquire);
            if(!child) {
                auto candidate=std::unique_ptr<Node,TreeDeleter>(new_node());Node* expected=nullptr;
                if(edge->child.compare_exchange_strong(expected,candidate.get(),std::memory_order_release,std::memory_order_acquire))child=candidate.release();
                else child=expected;
            }
            path.push_back({node,edge,{}});++edge->pending;++node->pending;++pending_count_;node=child;
            auto transition=state.move(path.back().parent->moves[choice].action);
            if(transition.reward!=0 || transition.discount!=1 ||
               (transition.perspective!=1 && transition.perspective!=-1))throw std::runtime_error("WDL AlphaZero search requires zero reward and unit discount");
            path.back().transition=transition;
            int expected=0,perspective=int(transition.perspective);
            path.back().edge->perspective.compare_exchange_strong(expected,perspective);
            if(expected!=0 && expected!=perspective)throw std::runtime_error("Inconsistent edge perspective");
        }
        while(!path.empty()) {
            auto p=path.back();value=p.transition.reward+p.transition.discount*p.transition.perspective*value;
            if(!std::isfinite(value))throw std::runtime_error("Nonfinite search backup");
            {
                std::lock_guard<std::mutex> lock(p.parent->stats_mutex);
                ++p.edge->n;++p.parent->visits;recompute(*p.parent);
            }
            --p.edge->pending;--p.parent->pending;--pending_count_;path.pop_back();
        }
        ++completed_;
    } catch(...){release();throw;}
}
void Search::simulate_many(int worker) {
    try {while(!failed_.load() && issued_.fetch_add(1)<budget_)simulation(worker);}
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
    if(state.terminal() || temperature<0 || !std::isfinite(temperature))throw std::runtime_error("Invalid search position/temperature");
    if(options.max_visits<0 || options.max_visits==1)throw std::runtime_error("Search cap must be >= 2");
    if(options.clear_before_search){clear_tree(root_.release());root_.reset(new_node());}
    remove_root_noise_=options.remove_root_noise;
    root_symmetries_=remove_root_noise_?1:settings_.root_symmetries;
    bool fresh=expand(*root_,state,0);
    if(!fresh && root_symmetries_>1)evaluate_node(*root_,state,0,root_symmetries_);
    int cap=options.max_visits?options.max_visits:settings_.max_visits;
    budget_=cap ? static_cast<int>(std::max<int64_t>(0,cap-1-root_->visits.load())) : settings_.simulations;
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
    std::sort(root_->prior_order.begin(),root_->prior_order.end(),[&](int a,int b){return root_->moves[a].search_prior>root_->moves[b].search_prior;});
    if(!fresh) {std::lock_guard<std::mutex> lock(root_->stats_mutex);recompute(*root_);}
    {std::lock_guard<std::mutex> lock(work_mutex_);position_=&state;issued_=0;completed_=0;failed_=false;error_=nullptr;workers_done_=0;cleaning_=false;++generation_;}
    work_changed_.notify_all();simulate_many(0);
    {std::unique_lock<std::mutex> lock(work_mutex_);work_changed_.wait(lock,[&]{return workers_done_==static_cast<int>(workers_.size());});}
    if(pending()!=0)throw std::runtime_error("Search leaked pending visits");
    if(error_)std::rethrow_exception(error_);
    if(completed_!=budget_)throw std::runtime_error("Search budget mismatch");
    SearchResult result;result.simulations=completed_;result.root_visits=root_->visits.load()+1;result.policy.assign(state.actions(),0);result.visits.assign(state.actions(),0);
    result.network_policy.assign(state.actions(),0);result.search_policy.assign(state.actions(),0);
    for(size_t i=0;i<root_->move_count;++i) {
        const auto& move=root_->moves[i];result.network_policy[move.action]=move.prior;result.search_policy[move.action]=move.search_prior;
    }
    int64_t total=0;std::vector<int> actions;
    auto stats=child_stats(*root_);
    for(int i=0;i<root_->child_count.load();++i) {
        auto& e=root_->edge(i);auto n=e.n.load();if(n<=0)continue;
        int action=root_->moves[e.move_index].action;
        actions.push_back(action);total+=n;result.visits[action]=n;
    }
    if(actions.size()!=stats.size())throw std::runtime_error("Root child stats are incomplete");
    if(total<1 || total!=root_->visits.load())throw std::runtime_error("Search has inconsistent completed root visits");
    result.network_wdl=root_->initial_wdl;
    auto parent=snapshot(*root_);result.value=parent.value;double d=parent.draw;
    result.search_wdl={(1-d+result.value)/2,d,(1-d-result.value)/2};
    auto move_weights=root_selection_weights(stats,settings_,settings_.use_lcb&&!options.training,parent);
    auto target_weights=options.training&&settings_.use_lcb?root_selection_weights(stats,settings_,true,parent):move_weights;
    result.move_policy.assign(state.actions(),0);
    auto behavior=temperature_distribution(move_weights,temperature,settings_.chosen_move_temperature_only_below_prob);
    for(size_t i=0;i<actions.size();++i) {
        double p=target_weights[i],weight=move_weights[i];
        result.policy[actions[i]]=p;result.move_policy[actions[i]]=weight;
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
        if(m.action==action && index>=0)next=root_->edge(index).child.exchange(nullptr);
    }
    clear_tree(root_.release());root_.reset(next?next:new_node());
    for(size_t i=0;i<root_->move_count;++i)root_->moves[i].search_prior=root_->moves[i].prior;
}
int Search::pending() const {return pending_count_.load();}
}
