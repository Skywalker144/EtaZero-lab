#include "etazero/search.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
using namespace etazero;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Diamond : SearchState {
    int phase=0,branch=0;std::array<std::atomic<int>,4>* calls;std::atomic<bool>* fail;
    Diamond(std::array<std::atomic<int>,4>& c,std::atomic<bool>& f):calls(&c),fail(&f){}
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<Diamond>(*this);}
    int actions() const override{return 2;}
    bool legal(int a) const override{return !terminal() && a>=0 && a<(phase==0?2:1);}
    bool terminal() const override{return phase==3;}
    double terminal_value() const override{return 1;}
    Evaluation evaluate() const override{++(*calls)[phase];if(phase==2 && fail->load())throw std::runtime_error("graph leaf failure");return {{0,0},{0.6,0.2,0.2}};}
    std::string graph_key() const override{return phase==1?(branch?"right":"left"):phase==0?"root":phase==2?"shared":"terminal";}
    Transition move(int a) override{check(legal(a),"Diamond illegal move");if(!phase)branch=a;++phase;return {0,1,1};}
};
struct Cycle : SearchState {
    bool at_root=true;
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<Cycle>(*this);}
    int actions() const override{return 1;}
    bool legal(int a) const override{return a==0;}
    bool terminal() const override{return false;}
    double terminal_value() const override{return 0;}
    Evaluation evaluate() const override{return {{0},{0.6,0.2,0.2}};}
    std::string graph_key() const override{return at_root?"cycle_root":"cycle";}
    Transition move(int) override{at_root=false;return {0,1,-1};}
};
struct Wide : Cycle {
    std::string path;
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<Wide>(*this);}
    int actions() const override{return 8;}
    bool legal(int a) const override{return a>=0 && a<8;}
    Evaluation evaluate() const override{return {std::vector<double>(8,0),{0.5,0,0.5}};}
    std::string graph_key() const override{return path;}
    Transition move(int a) override{path.push_back(char('0'+a));return {0,1,-1};}
};
struct Unsupported : Cycle {
    std::string graph_key() const override{return SearchState::graph_key();}
};
struct OrderedEvaluator : Evaluator {
    Evaluation evaluate(const std::vector<float>& obs) override {
        size_t n=(obs.size()-GLOBAL_FEATURES)/INPUT_PLANES;int canvas=int(std::sqrt(n));
        double own=std::accumulate(obs.begin()+n,obs.begin()+2*n,0.0),other=std::accumulate(obs.begin()+2*n,obs.begin()+3*n,0.0);
        std::vector<double> logits(n,-1000);bool black=own==other;
        for(int a:black?std::vector<int>{0,1}:std::vector<int>{canvas+1,canvas+2})logits[a]=0;
        return {logits,{0.6,0.2,0.2}};
    }
};
SearchSettings settings(bool graph=true,int threads=1,double leak=0) {
    SearchSettings s{200,threads,1.5,1,0,1,true};s.graph_search=graph;s.graph_catch_up_leak_prob=leak;
    s.value_weight_exponent=0;s.chosen_move_prune=0;return s;
}
void verify_quiescent(Search& search) {
    check(search.pending()==0,"Graph global pending leaked");
    auto nodes=search.inspect_graph();std::set<size_t> ids;
    for(const auto& n:nodes){check(ids.insert(n.id).second,"Duplicate graph node ownership");check(!n.pending,"Graph node pending leaked");}
    for(const auto& n:nodes)for(const auto& e:n.edges){check(!e.pending,"Graph edge pending leaked");check(ids.count(e.child_id),"Dangling graph edge");}
}
void verify_q(Search& search,const SearchResult& result) {
    auto nodes=search.inspect_graph();
    for(const auto& root:nodes)if(root.root)for(const auto& e:root.edges) {
        const auto child=std::find_if(nodes.begin(),nodes.end(),[&](const auto& n){return n.id==e.child_id;});
        check(child!=nodes.end(),"Q dangling child");
        if(child->stats.visits>0 && child->stats.weight>0) {
            check(result.q_visits[e.action]==child->stats.visits,"Q used edge instead of child NODE visits");
            check(result.q_values[e.action]==float(e.perspective*child->stats.value),"Q current-player pure W-L perspective");
        }
    }
}
void trace(Search& search,const SearchResult& result,int step) {
    std::cout<<"{\"step\":"<<step<<",\"catch_ups\":"<<result.graph_catch_ups<<",\"nodes\":[";
    bool first=true;for(const auto& n:search.inspect_graph()) {
        if(!first)std::cout<<',';first=false;
        std::cout<<"{\"id\":"<<n.id<<",\"key\":\""<<n.identity<<"\",\"root\":"<<(n.root?"true":"false")
            <<",\"visits\":"<<n.stats.visits<<",\"value\":"<<n.stats.value<<",\"value_sq\":"<<n.stats.value_sq
            <<",\"draw\":"<<n.stats.draw<<",\"weight\":"<<n.stats.weight<<",\"weight_sq\":"<<n.stats.weight_sq<<",\"edges\":[";
        bool edge_first=true;for(const auto& e:n.edges){if(!edge_first)std::cout<<',';edge_first=false;std::cout<<"["<<e.action<<','<<e.child_id<<','<<e.visits<<','<<e.perspective<<"]";}
        std::cout<<"]}";
    }std::cout<<"]}\n";
}
}
int main(int argc,char** argv) {
    try {
        std::cout.precision(17);
        if(argc>1) {
            std::array<std::atomic<int>,4> calls{};std::atomic<bool> fail{false};Diamond state(calls,fail);Search search(settings(true,1,argc>2?std::stod(argv[2]):0),53);
            SearchRun opts;opts.max_playouts=1;
            for(int i=0;i<100;++i){auto r=search.run(state,1,opts);trace(search,r,i);}
            return 0;
        }
        for(bool graph:{false,true})for(int threads:{1,8})for(double leak:{0.0,0.4,1.0}) {
            std::array<std::atomic<int>,4> calls{};std::atomic<bool> fail{false};Diamond state(calls,fail);Search search(settings(graph,threads,leak),13);
            auto result=search.run(state,1);verify_quiescent(search);verify_q(search,result);
            check(result.root_visits==201 && result.new_playouts==201,"Graph changed visit budget");
            check(std::accumulate(result.visits.begin(),result.visits.end(),int64_t(0))==200,"Graph root edge counts");
            if(graph) {
                check(calls[2]==1,"Shared graph node evaluated more than once");check(result.graph_hits>0,"Diamond transposition was not shared");
                check(leak!=0 || result.graph_catch_ups>0,"Catch-up branch not reached");check(leak!=1 || result.graph_catch_ups==0,"Leak-one must descend every visit");
                std::map<size_t,int> incoming;size_t shared=0;
                for(const auto& node:search.inspect_graph()){if(node.identity=="shared")shared=node.id;for(const auto& e:node.edges)++incoming[e.child_id];}
                check(incoming[shared]==2,"Diamond does not have two parents");
            } else {check(calls[2]==2,"Tree-off must evaluate independent child nodes");check(!result.graph_hits && !result.graph_catch_ups,"Tree-off executed graph search");}
            search.advance(0);state.move(0);auto before=search.inspect_graph().size();auto reused=search.run(state,1);
            verify_quiescent(search);verify_q(search,reused);
            if(graph && threads==1 && leak==1) {
                const auto nodes=search.inspect_graph();
                for(const auto& n:nodes)if(n.root)for(const auto& e:n.edges)
                    check(reused.q_visits[e.action]>e.visits,"Shared child NODE visits must exceed promoted parent edge");
            }
            check(reused.initial_visits>0 && before>0,"Promoted graph root lost statistics");
            search.reset(13);verify_quiescent(search);check(search.inspect_graph().size()==1,"Reset retained obsolete nodes");
        }
        for(int threads:{1,8})for(double leak:{0.0,1.0}) {
            Cycle state;Search search(settings(true,threads,leak),12);auto result=search.run(state,1);
            verify_q(search,result);check(result.root_visits==201 && (leak==0?result.graph_catch_ups>0:result.graph_cycles>0),"Cycle did not terminate counted playouts");verify_quiescent(search);
            search.advance(0);state.move(0);auto reused=search.run(state,1);check(reused.initial_visits>0,"Cycle root reuse");verify_quiescent(search);
            search.advance(0);verify_quiescent(search);search.reset(14);check(search.inspect_graph().size()==1,"Cyclic graph reset leaked");
        }
        {
            std::array<std::atomic<int>,4> calls{};std::atomic<bool> fail{true};Diamond state(calls,fail);Search search(settings(true,8),22);
            bool rejected=false;try{search.run(state,1);}catch(const std::exception& e){rejected=std::string(e.what())=="graph leaf failure";}
            check(rejected,"Shared graph failure did not propagate");verify_quiescent(search);
            fail=false;search.reset(22);check(search.run(state,1).root_visits==201,"Graph did not recover after explicit reset");verify_quiescent(search);
        }
        {
            Search search(settings(),0);Unsupported state;bool rejected=false;try{search.run(state,1);}catch(const std::exception&){rejected=true;}
            check(rejected,"Unsupported graph state silently became a tree");
        }
        for(double leak:{-0.1,1.1,std::numeric_limits<double>::quiet_NaN()}) {
            bool rejected=false;try{Search search(settings(true,1,leak),0);}catch(const std::exception&){rejected=true;}check(rejected,"Invalid graph leak accepted");
        }
        OrderedEvaluator eval;
        for(Rule rule:{Rule::FREESTYLE,Rule::STANDARD,Rule::RENJU}) {
            Game a(5,5,rule),b(5,5,rule);for(int m:{0,6,1,7})a.play(m);for(int m:{1,7,0,6})b.play(m);
            check(AlphaZeroState(a,eval).graph_key()==AlphaZeroState(b,eval).graph_key(),"Move order changed NOVC graph identity");
            Search search(eval,settings(),83);Game empty(5,5,rule);auto result=search.run(empty,1);check(result.graph_hits>0,"Real Gomoku did not transpose");
            verify_quiescent(search);search.advance(result.action);empty.play(result.action);check(search.run(empty,1).initial_visits>0,"Real Gomoku promotion lost subtree");
            // A rule/size/board switch must not consume a previously promoted root's policy/statistics.
            check(search.run(Game(6,6,rule),1).initial_visits==0,"Graph reused incompatible board conditions");verify_quiescent(search);
            search.set_evaluator(eval);check(search.inspect_graph().size()==1,"Model change retained graph values");
        }
        check(AlphaZeroState(Game(5,5,Rule::STANDARD),eval).graph_key()!=AlphaZeroState(Game(5,5,Rule::RENJU),eval).graph_key(),"Rules collide in graph identity");
        check(AlphaZeroState(Game(5,5,Rule::RENJU),eval).graph_key()!=AlphaZeroState(Game(5,6,Rule::RENJU),eval).graph_key(),"NN canvas collides in graph identity");
        check(AlphaZeroState(Game(5,6,Rule::RENJU),eval).graph_key()!=AlphaZeroState(Game(6,6,Rule::RENJU),eval).graph_key(),"Actual size collides in graph identity");
        {
            auto s=settings(true,8);s.simulations=6000;Wide state;Search search(s,44);auto result=search.run(state,1);
            check(result.graph_nodes>4096,"Large graph did not exercise parallel collection");verify_quiescent(search);
            search.advance(result.action);verify_quiescent(search);search.reset(45);check(search.inspect_graph().size()==1,"Large graph collection leaked ownership");
        }
        std::cout<<"graph tests passed\n";return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
