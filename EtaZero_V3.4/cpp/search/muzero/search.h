#pragma once
#include "search/search_common.h"
#include "neuralnet/muzero/inference.h"
#include "search/gumbel.h"
namespace etazero::muzero {
class Search : public GameSearch {
    std::shared_ptr<etazero::Evaluator> human_evaluator() const {auto e=settings_.human_evaluator?settings_.human_evaluator:evaluator_.secondary;return e&&e->available()?e:nullptr;}
protected:
    struct Node;
    struct Edge {int action;double prior,search_prior;int pending=0;Node* child=nullptr;int64_t visits=0;double human_prior=-1;};
    struct Node {
        std::atomic<bool> ready{false};
        mutable std::recursive_mutex mutex;
        mutable std::mutex stats_mutex;
        std::atomic<int> dirty_count{0};
        ValueStats snapshot() const { std::lock_guard<std::mutex> lock(stats_mutex);return stats; }
        void publish(ValueStats value) { std::lock_guard<std::mutex> lock(stats_mutex);stats=value; }
        int player=1;std::vector<double> capped_distribution;
        std::shared_ptr<const Latent> latent,human_latent;
        WDL initial{0,1,0};double initial_weight=1,stdev=0,variance_time=-1;
        ValueStats stats;
        std::vector<Edge> edges;
        std::vector<size_t> expansion_order;
    };
    Evaluator& evaluator_;SearchSettings settings_;std::mt19937_64 random_;
    std::vector<std::unique_ptr<Node>> nodes_;Node* root_=nullptr;
    std::vector<int> mapping_,board_actions_;
    std::array<std::vector<int>,2> avoid_;std::vector<int> representatives_;
    int root_player_=1,pass_=-1;
    bool remove_noise_=false;int hint_=-1,capped_player_=0;
    std::mutex allocation_mutex_,work_mutex_;
    std::condition_variable work_changed_,work_done_;
    std::vector<std::thread> workers_;
    std::vector<std::mt19937_64> worker_random_;
    std::function<void(int)> work_;
    uint64_t work_generation_=0;
    int work_pending_=0;
    bool shutdown_=false;
    void worker_loop(int);
    void dispatch(const std::function<void(int)>&,int);
    Node* node();
    void expand(Node&,InferenceOutput,const std::vector<int>& actions,bool root);
    std::vector<RootChildStats> children(const Node&,bool aggregation=false) const;
    void recompute(Node&);
    void update_after_playout(Node&);
    struct Selection {size_t index;bool weightless=false,focus=false;};
    bool allowed(const Node&,int action,int depth) const;
    Selection select(Node&,int depth,std::mt19937_64&,double remaining,const GumbelSettings* gumbel=nullptr,bool full=false,int forced=-1);
    void expand_human(Node&,Evaluation,bool tree_coordinates);
    void mix_human(SearchResult&) const;
    SearchResult run_impl(const Game&,double,SearchRun,const GumbelSettings*,bool);
    std::vector<GumbelChild> gumbel_children(const Node&,int depth) const;
    SearchResult analysis_snapshot(int actions,const GumbelSettings* gumbel=nullptr) const;
public:
    Search(Evaluator&,SearchSettings,uint64_t seed);
    ~Search() override;
    SearchResult inspect(const Game&,const std::vector<int>&) override;
    SearchResult run(const Game&,double temperature,SearchRun options={}) override;
    void advance(int) override { nodes_.clear();root_=nullptr;set_root_focus({}, {},0); }
    void reset(uint64_t seed) override { advance(0);random_.seed(seed);for(size_t i=0;i<worker_random_.size();++i)worker_random_[i].seed(seed+i); }
};
}
