#include "search/search.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
using namespace etazero;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct GoEvaluator : Evaluator {
    GoEvaluator(){spec_=game_spec(GameKind::GO);}
    Evaluation evaluate(const std::vector<float>& obs) override {
        int area=(obs.size()-spec_.globals)/spec_.planes;
        double own=std::accumulate(obs.begin()+area,obs.begin()+2*area,0.0);
        double other=std::accumulate(obs.begin()+2*area,obs.begin()+3*area,0.0);
        std::vector<double> logits(area+1,-1000);logits[area]=-10;
        for(int a:own==other?std::vector<int>{0,2}:std::vector<int>{area-3,area-1})logits[a]=0;
        return {logits,{.5,0,.5}};
    }
};
SearchSettings settings(int threads=1) {
    SearchSettings s{300,threads,1.5,1,0,1,true};s.graph_search=true;s.graph_catch_up_leak_prob=1;
    s.value_weight_exponent=0;s.chosen_move_prune=0;return s;
}
void quiescent(Search& search) {
    check(search.pending()==0,"Leaked pending work");
    auto nodes=search.inspect_graph();
    for(auto& node:nodes) {
        check(node.pending==0,"Leaked pending node");
        for(auto& edge:node.edges) {
            check(edge.pending==0,"Leaked pending edge");
            check(std::any_of(nodes.begin(),nodes.end(),[&](auto& n){return n.id==edge.child_id;}),"Dangling child after pruning");
        }
    }
}
// Deliberately collide histories with disjoint legal moves, forcing the rare
// superko/collision recovery branch without relying on an accidental hash hit.
struct HistoryCollision : SearchState {
    int phase=0,branch=0;
    std::atomic<int>& repairs;std::atomic<bool>& fail;
    HistoryCollision(std::atomic<int>& r,std::atomic<bool>& f):repairs(r),fail(f){}
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<HistoryCollision>(*this);}
    int actions() const override{return 2;}
    bool legal(int a) const override{return a>=0 && a<2 && !terminal() && (phase==0 || a==branch);}
    bool terminal() const override{return phase==2;}
    double terminal_value() const override{return 0;}
    Evaluation evaluate(SearchEvaluation options={}) const override {
        if(options.skip_cache){++repairs;if(fail)throw std::runtime_error("repair inference failed");}
        return {{0,0},{.5,0,.5}};
    }
    bool graph_approximates_history() const override{return true;}
    std::string graph_key() const override{return phase==0?"root":phase==1?"shared":branch?"terminal1":"terminal0";}
    int move(int a) override{check(legal(a),"Traversed history-illegal move");if(!phase)branch=a;++phase;return -1;}
};
struct PartialHistoryCollision : HistoryCollision {
    using HistoryCollision::HistoryCollision;
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<PartialHistoryCollision>(*this);}
    int actions() const override{return 3;}
    bool legal(int a) const override{return !terminal() && (a==branch || a==2);}
    Evaluation evaluate(SearchEvaluation options={}) const override {
        auto output=HistoryCollision::evaluate(options);output.logits.push_back(0);return output;
    }
};
}
int main(int argc,char**) {
    try {
        if(argc>1) {
            std::string line;
            while(std::getline(std::cin,line)) {
                std::istringstream in(line);int size,canvas,bound,count,action;RuleOptions options;
                in>>size>>canvas>>bound>>options.go_rules>>options.komi>>count;
                Game incremental(size,canvas,Rule::GO,options),replayed=incremental;
                incremental.graph_identity(bound);
                for(int i=0;i<count;++i){in>>action;incremental.play(incremental.native_action(action));replayed.play(replayed.native_action(action));}
                auto hash=incremental.graph_identity(bound);
                check(hash==replayed.graph_identity(bound),"Incremental and replayed graph identity differ");
                Game copied=incremental,reset(size,canvas,Rule::GO);reset=copied;
                check(reset.graph_identity(bound)==hash,"Copy/reset lost graph history");
                std::cout<<hash<<'\n';
            }
            return 0;
        }
        GoEvaluator evaluator;
        for(int size:{5,9}) {
            Game a(size,size,Rule::GO),b=a;
            for(int move:{0,size*size-1,2})a.play(move);
            for(int move:{2,size*size-1,0})b.play(move);
            check(a.board().cells==b.board().cells,"Transposition fixture board differs");
            check(a.history_identity()!=b.history_identity() && a.observation()!=b.observation(),"Fixture must retain different real/NN histories");
            check(AlphaZeroState(a,evaluator).graph_key()==AlphaZeroState(b,evaluator).graph_key(),"Go move orders did not transpose");
            auto original=a.graph_identity(11);
            if(size==5)check(a.graph_identity(50)!=b.graph_identity(50),"Large repetition bound failed to retain history");
            check(a.graph_identity(11)==original,"Changing repetition bound failed to replay history");
            Game copied=a;copied.play(size*size);
            a.graph_identity(50);a.play(size*size);
            check(a.graph_identity(11)==copied.graph_identity(11),"Incremental update after bound switch differs");
        }
        Game base(5,5,Rule::GO),padded(5,7,Rule::GO);
        check(base.graph_identity(11)==padded.graph_identity(11),"Rule identity depends on NN padding");
        check(AlphaZeroState(base,evaluator).graph_key()!=AlphaZeroState(padded,evaluator).graph_key(),"NN canvases were merged");
        auto conditioned=base;conditioned.set_pda(.5,1);
        check(AlphaZeroState(base,evaluator).graph_key()!=AlphaZeroState(conditioned,evaluator).graph_key(),"PDA conditions were merged");
        RuleOptions changed;changed.komi=.5;
        check(base.graph_identity(11)!=Game(5,5,Rule::GO,changed).graph_identity(11),"Different komi merged");
        changed.go_rules="japanese";changed.komi=7.5;
        check(base.graph_identity(11)!=Game(5,5,Rule::GO,changed).graph_identity(11),"Different rules merged");
        auto empty=base.graph_identity(11);base.play(25);auto passed=base.graph_identity(11);base.play(25);
        check(base.finished() && empty!=passed && passed!=base.graph_identity(11),"Pass/terminal context merged");
        for(int threads:{1,4}) {
            Game game(9,9,Rule::GO);Search search(evaluator,settings(threads),83);
            auto result=search.run(game,1);quiescent(search);
            check(result.graph_hits>0,"Real Go search did not share nodes");
            std::map<size_t,int> incoming;
            for(auto& n:search.inspect_graph())for(auto& e:n.edges)++incoming[e.child_id];
            check(std::any_of(incoming.begin(),incoming.end(),[](auto p){return p.second>1;}),"No Go node has multiple parents");
            check(game.legal(result.action),"Search returned illegal Go action");
            game.play(result.action);search.advance(result.action);
            auto reused=search.run(game,1);check(reused.initial_visits>0,"Go graph root promotion lost visits");quiescent(search);
            for(int a=0;a<game.actions();++a)if(!game.legal(a))
                check(reused.policy[a]==0 && reused.visits[a]==0 && reused.q_visits[a]==0,"Illegal Go training target");
            auto tree_settings=settings(threads);tree_settings.graph_search=false;
            Search tree(evaluator,tree_settings,83);check(tree.run(game,1).graph_hits==0,"Tree-off still shares Go nodes");
            std::atomic<int> repairs{0};std::atomic<bool> fail{false};HistoryCollision collision(repairs,fail);
            Search recovery(settings(threads),7);auto recovered=recovery.run(collision,1);quiescent(recovery);
            check(repairs>0 && recovered.root_visits>=301,"History collision was not repaired as a counted playout");
        }
        {
            std::atomic<int> repairs{0};std::atomic<bool> fail{false};HistoryCollision state(repairs,fail);state.phase=1;
            Search search(settings(),5);search.run(state,1);state.branch=1;
            SearchRun no_budget;no_budget.max_playouts=0;auto result=search.run(state,1,no_budget);
            check(repairs==1 && result.action==1 && result.visits[0]==0 && result.q_visits[0]==0 && result.policy[0]==0,
                  "Reused root retained history-illegal edges or omitted newly legal moves");
            check(result.root_visits==1 && result.initial_visits==1,"Root pruning did not repair visit counts");quiescent(search);
            state.branch=0;result=search.run(state,1,no_budget);check(result.action==0,"Repeated root legality changes failed");
        }
        {
            std::atomic<int> repairs{0};std::atomic<bool> fail{false};PartialHistoryCollision state(repairs,fail);state.phase=1;
            Search search(settings(),5);auto before=search.run(state,1);state.branch=1;
            SearchRun no_budget;no_budget.max_playouts=0;auto result=search.run(state,1,no_budget);
            check(before.visits[0]>0 && before.visits[2]>0,"Partial-pruning fixture did not explore both edges");
            check(result.visits[0]==0 && result.visits[2]==before.visits[2] && result.network_policy[1]>0,
                  "Root pruning lost a legal edge or omitted a newly legal action");
            check(result.root_visits==before.visits[2]+1 && result.initial_visits==result.root_visits,
                  "Partial root pruning has inconsistent visit counts");quiescent(search);
        }
        {
            std::atomic<int> repairs{0};std::atomic<bool> fail{true};HistoryCollision state(repairs,fail);Search search(settings(4),9);
            bool rejected=false;try{search.run(state,1);}catch(const std::exception& e){rejected=std::string(e.what())=="repair inference failed";}
            check(rejected,"Repair inference failure was hidden");quiescent(search);
            fail=false;search.reset(9);search.run(state,1);quiescent(search);
        }
        std::cout<<"Go graph tests passed\n";return 0;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
