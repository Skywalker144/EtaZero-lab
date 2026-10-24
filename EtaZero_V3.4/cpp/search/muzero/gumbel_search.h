#pragma once
#include "search/muzero/search.h"
#include "search/gumbel.h"
namespace etazero::muzero {
template<bool Full> class GumbelSearch final : public Search {
    GumbelSettings gumbel_;
    std::vector<GumbelChild> node_children(const Node&) const;
public:
    GumbelSearch(Evaluator& e,SearchSettings s,GumbelSettings g,uint64_t seed)
        :Search(e,gumbel_search_settings(s),seed),gumbel_(g){gumbel_.validate();}
    SearchResult run(const Game&,double,SearchRun={}) override;
};
}
