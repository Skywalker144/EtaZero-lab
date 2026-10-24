#include "search/search.h"
#include "search/gumbel.h"
#include <algorithm>
#include <numeric>
#include <map>
namespace etazero {
SearchResult Search::analysis_snapshot(int actions,const GumbelSettings* gumbel) {
    SearchResult out;
    out.policy.assign(actions,0);out.move_policy=out.network_policy=out.search_policy=out.policy;out.visits.assign(actions,0);
    auto parent=snapshot(*root_);out.value=parent.win_loss;out.utility=parent.value;out.search_weight=parent.weight;
    auto eval=root_->evaluated();
    if(!eval->human_policy.empty()){out.human_policy.assign(actions,-1);for(size_t i=0;i<root_->move_count;++i)out.human_policy[root_->moves[i].action]=eval->human_policy[i];}
    out.network_sample_weight=eval->initial_weight;out.network_value_stdev=eval->initial_stdev;out.network_variance_time=eval->initial_variance_time;
    out.network_wdl=eval->initial_wdl;out.search_wdl=wdl_from_value(parent.win_loss,parent.draw,parent.no_result);
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
                    c.node_visits=q.visits;c.node_weight=q.weight;c.pv_visits.push_back(q.visits);c.pv_edge_visits.push_back(n);
                    c.has_value=true;c.weight=q.weight*scale;c.utility=perspective*q.value;c.win_loss=perspective*q.win_loss;
                    c.wdl=wdl_from_value(c.win_loss,q.draw,q.no_result);out.visits[c.action]=n;
                    if(n>0)ordered.emplace(index,std::make_pair(c.action,RootChildStats{eval->search_prior[i],c.weight*c.utility,q.value_sq*c.weight,n,
                        c.weight,q.weight_sq*scale,q.draw*c.weight,c.win_loss*c.weight,q.no_result*c.weight}));
                    std::unordered_set<Node*> seen{root_};Node* current=child;
                    while(current && c.pv.size()<size_t(analysis_pv_length_) && seen.insert(current).second && current->status.load(std::memory_order_acquire)==Node::READY) {
                        auto current_eval=current->evaluated();
                        int best=-1;int64_t count=0;
                        for(int j=0;j<current->child_count.load(std::memory_order_acquire);++j) {
                            if(current_eval->prior[current->edge(j).move_index]<0)continue;
                            auto visits=current->edge(j).n.load();if(visits>count){count=visits;best=j;}
                        }
                        if(best<0)break;
                        auto& next=current->edge(best);c.pv.push_back(current->moves[next.move_index].action);
                        current=next.child.load(std::memory_order_acquire);c.pv_edge_visits.push_back(count);c.pv_visits.push_back(current?snapshot(*current).visits:0);
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

namespace etazero {
SearchResult Search::inspect(const Game& game,const std::vector<int>& branch) {
    Node* node=root_;
    for(int a:branch){if(!node)break;Node* next=nullptr;for(size_t i=0;i<node->move_count;++i)if(node->moves[i].action==a){int index=node->moves[i].child_index.load();if(index>=0)next=node->edge(index).child.load();break;}node=next;}
    if(!node||node->status.load()!=Node::READY)throw std::runtime_error("No searched node for branch");
    auto old=root_;int player=root_player_;auto position=position_;root_=node;root_player_=game.player();position_=nullptr;
    try {auto out=analysis_snapshot(game.actions());root_=old;root_player_=player;position_=position;return out;}
    catch(...){root_=old;root_player_=player;position_=position;throw;}
}
}
