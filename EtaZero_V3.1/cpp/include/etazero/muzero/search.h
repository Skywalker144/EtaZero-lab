#pragma once
#include "etazero/search.h"
#include "inference.h"
#include "etazero/gumbel.h"
namespace etazero::muzero {
class Search : public GameSearch {
protected:
    struct Node;
    struct Edge {int action;double prior,search_prior;int pending=0;Node* child=nullptr;int64_t visits=0;double human_prior=-1;};
    struct Node {
        bool ready=false;
        int player=1;std::vector<double> capped_distribution;
        std::shared_ptr<const Latent> latent,human_latent;
        WDL initial{0,1,0};double initial_weight=1,stdev=0;
        ValueStats stats;
        std::vector<Edge> edges;
        std::vector<size_t> expansion_order;
    };
    Evaluator& evaluator_;SearchSettings settings_;std::mt19937_64 random_;
    std::vector<std::unique_ptr<Node>> nodes_;Node* root_=nullptr;
    std::vector<int> mapping_,board_actions_;
    std::array<std::vector<int>,2> avoid_;std::vector<int> representatives_;
    int root_player_=1,pass_=-1;
    bool remove_noise_=false;int hint_=-1,capped_player_=0;double remaining_=1e30;
    Node* node();
    void expand(Node&,InferenceOutput,const std::vector<int>& actions,bool root);
    std::vector<RootChildStats> children(const Node&,bool aggregation=false) const;
    void recompute(Node&);
    struct Selection {size_t index;bool weightless=false,focus=false;};
    bool allowed(const Node&,int action,int depth) const;
    Selection select(Node&,int depth,const GumbelSettings* gumbel=nullptr,bool full=false,int forced=-1);
    void expand_human(Node&,Evaluation);
    void mix_human(SearchResult&) const;
    SearchResult run_impl(const Game&,double,SearchRun,const GumbelSettings*,bool);
    std::vector<GumbelChild> gumbel_children(const Node&,int depth) const;
    SearchResult analysis_snapshot(int actions,const GumbelSettings* gumbel=nullptr) const;
public:
    Search(Evaluator&,SearchSettings,uint64_t seed);
    SearchResult run(const Game&,double temperature,SearchRun options={}) override;
    void advance(int) override { nodes_.clear();root_=nullptr;set_root_focus({}, {},0); }
    void reset(uint64_t seed) override { advance(0);random_.seed(seed); }
};
}
