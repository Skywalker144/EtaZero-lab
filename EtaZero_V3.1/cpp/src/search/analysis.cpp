#include "etazero/gumbel.h"
#include <algorithm>
#include <numeric>
#include <map>
namespace etazero {
// No sampling, tree mutation or training targets are produced by a snapshot.
void analysis_policy(SearchResult& result,const std::vector<int>& actions,const std::vector<RootChildStats>& stats,
                     const SearchSettings& settings,ValueStats parent,const GumbelSettings* gumbel) {
    result.gumbel=gumbel!=nullptr;
    if(gumbel) {
        std::vector<GumbelChild> children;
        for(const auto& c:result.analysis)children.push_back({result.search_policy[c.action],c.utility,result.visits[c.action],0});
        auto policy=gumbel_policy(children,result.network_wdl[0]-result.network_wdl[2],*gumbel);
        for(size_t i=0;i<policy.size();++i)result.policy[result.analysis[i].action]=policy[i];
        result.move_policy=result.policy;
    } else if(!stats.empty()) {
        std::vector<RootSelectionDiagnostic> diagnostics;
        auto weights=root_selection_weights(stats,settings,settings.use_lcb,parent,false,&diagnostics);
        double mass=std::accumulate(weights.begin(),weights.end(),0.0);
        for(size_t i=0;i<actions.size();++i) {
            int action=actions[i];result.move_policy[action]=result.policy[action]=weights[i]/mass;
            for(auto& c:result.analysis)if(c.action==action){c.selection=diagnostics[i];break;}
        }
    } else {
        auto policy=result.search_policy;
        bool allocated=std::any_of(result.analysis.begin(),result.analysis.end(),[](const auto& c){return c.has_value;});
        if(allocated)for(const auto& c:result.analysis)if(!c.has_value)policy[c.action]=0;
        result.move_policy=result.policy=direct_policy_selection(std::move(policy),settings);
        for(auto& c:result.analysis)c.selection.weight=result.policy[c.action];
    }
    auto& ranking=gumbel?result.policy:result.move_policy;
    result.action=int(std::max_element(ranking.begin(),ranking.end())-ranking.begin());
}
SearchResult Search::analysis_snapshot(int actions,const GumbelSettings* gumbel) {
    SearchResult out;
    out.policy.assign(actions,0);out.move_policy=out.network_policy=out.search_policy=out.policy;out.visits.assign(actions,0);
    auto parent=snapshot(*root_);out.value=parent.win_loss;
    auto eval=root_->evaluated();
    out.network_wdl=eval->initial_wdl;out.search_wdl=wdl_from_value(parent.win_loss,parent.draw);
    std::vector<int> visited;std::vector<RootChildStats> stats;
    std::map<int,std::pair<int,RootChildStats>> ordered;
    // Root moves are immutable once workers start. Acquire publication before reading a child.
    for(size_t i=0;i<root_->move_count;++i) {
        if(eval->prior[i]<0 || !move_allowed(root_->moves[i].action,root_player_,0))continue;
        const auto& move=root_->moves[i];AnalysisCandidate c;c.action=move.action;c.pv.push_back(c.action);
        out.network_policy[c.action]=eval->prior[i];out.search_policy[c.action]=eval->search_prior[i];
        int index=move.child_index.load(std::memory_order_acquire);
        if(index>=0) {
            auto& edge=root_->edge(index);auto n=edge.n.load();Node* child=edge.child.load(std::memory_order_acquire);
            if(child) {
                auto q=snapshot(*child);
                if(q.visits>0 && q.weight>0) {
                    double scale=double(n)/q.visits,perspective=edge.perspective.load();
                    c.has_value=true;c.weight=q.weight*scale;c.utility=perspective*q.value;c.win_loss=perspective*q.win_loss;
                    c.wdl=wdl_from_value(c.win_loss,q.draw);out.visits[c.action]=n;
                    if(n>0)ordered.emplace(index,std::make_pair(c.action,RootChildStats{eval->search_prior[i],c.weight*c.utility,q.value_sq*c.weight,n,
                        c.weight,q.weight_sq*scale,q.draw*c.weight,c.win_loss*c.weight}));
                    std::unordered_set<Node*> seen{root_};Node* current=child;
                    while(current && c.pv.size()<32 && seen.insert(current).second && current->status.load(std::memory_order_acquire)==Node::READY) {
                        auto current_eval=current->evaluated();
                        int best=-1;int64_t count=0;
                        for(int j=0;j<current->child_count.load(std::memory_order_acquire);++j) {
                            if(current_eval->prior[current->edge(j).move_index]<0)continue;
                            auto visits=current->edge(j).n.load();if(visits>count){count=visits;best=j;}
                        }
                        if(best<0)break;
                        auto& next=current->edge(best);c.pv.push_back(current->moves[next.move_index].action);
                        current=next.child.load(std::memory_order_acquire);
                    }
                }
            }
        }
        out.analysis.push_back(std::move(c));
    }
    // Concurrent samples may be slightly staggered; totals describe the included completed edges.
    out.root_visits=1+std::accumulate(out.visits.begin(),out.visits.end(),int64_t(0));
    for(const auto& item:ordered){visited.push_back(item.second.first);stats.push_back(item.second.second);}
    out.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count();
    analysis_policy(out,visited,stats,settings_,parent,gumbel);
    if(position_&&settings_.human_sl_chosen_move_proportion>0){mix_human_policy(out,*position_,0,false);out.action=int(std::max_element(out.move_policy.begin(),out.move_policy.end())-out.move_policy.begin());}
    return out;
}
}
