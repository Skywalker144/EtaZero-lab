#include "etazero/gumbel_search.h"
namespace etazero::muzero {
template<bool Full> std::vector<GumbelChild> GumbelSearch<Full>::node_children(const Node& node) const {
    return gumbel_children(node,&node==root_?0:1);
}
template<bool Full> SearchResult GumbelSearch<Full>::run(const Game& game,double temperature,SearchRun options) {
    return run_impl(game,temperature,std::move(options),&gumbel_,Full);
}
template class GumbelSearch<false>;
template class GumbelSearch<true>;
}
