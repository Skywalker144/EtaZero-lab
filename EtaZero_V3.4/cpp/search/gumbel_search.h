#pragma once
#include "search/gumbel.h"
#include "search/search.h"
namespace etazero {
template<bool Full> class GumbelSearch final : public Search {
    GumbelSettings gumbel_;
    std::unique_ptr<GumbelRoot> schedule_;
    std::mutex root_mutex_;
    std::condition_variable root_changed_;
    std::vector<GumbelChild> root_children();
    void simulate_gumbel(int worker);
    void gumbel_worker_loop(int worker);
public:
    GumbelSearch(Evaluator&,SearchSettings,GumbelSettings,uint64_t);
    ~GumbelSearch() override;
    SearchResult run(const Game&,double,SearchRun={}) override;
    SearchResult run(const SearchState&,double,SearchRun={});
    void advance(int) override { root_=nullptr;collect_nodes(nullptr);root_=new_node();set_root_focus({}, {},0); }
};
}
