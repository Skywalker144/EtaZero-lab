#pragma once
#include "etazero/search.h"
#include "inference.h"
namespace etazero::muzero {
class Search : public GameSearch {
protected:
    struct Node;
    struct Edge {int action;double prior,search_prior;int pending=0;Node* child=nullptr;};
    struct Node {
        bool ready=false;
        std::shared_ptr<const Latent> latent;
        WDL initial{0,1,0};double initial_weight=1,stdev=0;
        ValueStats stats;
        std::vector<Edge> edges;
        std::vector<size_t> expansion_order;
    };
    Evaluator& evaluator_;SearchSettings settings_;std::mt19937_64 random_;
    std::vector<std::unique_ptr<Node>> nodes_;Node* root_=nullptr;
    std::vector<int> mapping_,board_actions_;
    bool remove_noise_=false;int hint_=-1;
    Node* node();
    void expand(Node&,InferenceOutput,const std::vector<int>& actions,bool root);
    std::vector<RootChildStats> children(const Node&) const;
    void recompute(Node&);
    size_t select(Node&,bool root);
public:
    Search(Evaluator&,SearchSettings,uint64_t seed);
    SearchResult run(const Game&,double temperature,SearchRun options={}) override;
    void advance(int) override { nodes_.clear();root_=nullptr; }
    void reset(uint64_t seed) override { advance(0);random_.seed(seed); }
};
}
