#include "etazero/gumbel_search.h"
#include "etazero/random_evaluator.h"
#include <cmath>
#include <iostream>
#include <numeric>
using namespace etazero;
void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
struct Bandit : SearchState {
    int action=-1;
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<Bandit>(*this);}
    int actions() const override{return 3;}
    bool legal(int a) const override{return action<0&&a>=0&&a<3;}
    bool terminal() const override{return action>=0;}
    double terminal_value() const override{return action==0?-.8:action==1?.2:.6;}
    Evaluation evaluate() const override{return {{std::log(.5),std::log(.3),std::log(.2)},{.55,0,.45}};}
    Transition move(int a) override{action=a;return {0,1,-1};}
};
struct State : muzero::Latent {int depth=0;};
struct TestMuBackend : muzero::Backend {
    bool fail=false;
    muzero::InferenceOutput output(int depth) {
        auto state=std::make_shared<State>();state->depth=depth;Evaluation e;e.logits.assign(36,0);e.wdl=depth%2?WDL{.2,0,.8}:WDL{.8,0,.2};return {state,e};
    }
    std::vector<muzero::InferenceOutput> initial(const InferenceInputs& inputs) override {
        std::vector<muzero::InferenceOutput> out;for(size_t i=0;i<inputs.size();++i)out.push_back(output(0));return out;
    }
    std::vector<muzero::InferenceOutput> recurrent(const std::vector<muzero::InferenceAction>& inputs) override {
        if(fail)throw std::runtime_error("injected Gumbel recurrent failure");
        std::vector<muzero::InferenceOutput> out;for(const auto& in:inputs)out.push_back(output(std::dynamic_pointer_cast<const State>(in.latent)->depth+1));return out;
    }
};
template<bool Full> void alpha() {
    struct Uniform : Evaluator {Evaluation evaluate(const std::vector<float>&) override{return {std::vector<double>(36,0),{0,1,0}};}} evaluator;
    SearchSettings s{36,4,1,1,0,1,false};s.value_weight_exponent=0;
    GumbelSettings g;g.noise_scale=0;g.max_num_considered_actions=3;g.rescale_q_values=false;g.c_scale=.1;
    GumbelSearch<Full> search(evaluator,s,g,17);Bandit state;
    for(int budget:{0,1,2,3,7,17,31,200}) {
        SearchRun o;o.max_visits=budget+1;auto result=search.run(state,0,o);
        check(result.simulations==budget&&result.root_visits==budget+1,"AZ exact edge/root budget");
        check(std::accumulate(result.visits.begin(),result.visits.end(),int64_t(0))==budget,"AZ completed visits");
        check(result.action>=0&&result.action<3,"AZ legal action");
        if(budget>=3)check(result.action==0,"SH identifies best evaluated action");
        if(budget==1)check(result.visits[1]==0&&result.policy[1]>0&&result.policy_target[1]>0,"unvisited target survives packing");
        check(search.pending()==0,"AZ drains reservations");
    }
    SearchRun stopped;stopped.should_stop=[]{return true;};check(search.run(state,0,stopped).action==-1,"stop before root inference");
    std::atomic<int> stop_checks{0};SearchRun interrupted;interrupted.max_visits=100001;
    interrupted.should_stop=[&]{return ++stop_checks>10;};auto partial=search.run(state,0,interrupted);
    check(partial.stopped_early&&partial.simulations<100000&&partial.root_visits==partial.simulations+1&&search.pending()==0,"AZ mid-search stop drains work");
    SearchRun timed;timed.max_visits=100001;timed.max_time=0;auto limited=search.run(state,0,timed);
    check(limited.stopped_early&&limited.simulations>0&&limited.simulations<100000&&search.pending()==0,"AZ time cap drains work");
    for(Rule rule:{Rule::FREESTYLE,Rule::RENJU,Rule::HEX}) {
        Game game(5,6,rule);game.play(0);auto r=search.run(game,0,SearchRun{18});
        check(game.legal(r.action)&&r.policy[0]==0&&r.policy[5]==0,"AZ occupied/padding mask");
        search.advance(r.action);check(search.run(game,0,SearchRun{3}).initial_visits==0,"Gumbel fresh roots");
    }
    struct Failure : Evaluator {
        std::atomic<int> calls{0};Evaluation evaluate(const std::vector<float>&) override {
            if(calls++>0)throw std::runtime_error("injected Gumbel AZ failure");
            return {std::vector<double>(36,0),{0,1,0}};
        }
    } failure;
    GumbelSearch<Full> broken(failure,s,g,17);bool raised=false;
    try{broken.run(Game(5,6,Rule::FREESTYLE),0);}catch(const std::runtime_error& e){raised=std::string(e.what()).find("injected")!=std::string::npos;}
    check(raised&&broken.pending()==0,"AZ inference failure drains work and propagates");
}
template<bool Full> void muzero_case() {
    std::vector<std::unique_ptr<muzero::Backend>> backends;backends.push_back(std::make_unique<TestMuBackend>());
    muzero::BatchEvaluator evaluator(std::move(backends),6,8,0,0,false,0,17);
    SearchSettings s{36,4,1,1,0,1,false};s.value_weight_exponent=0;GumbelSettings g;g.noise_scale=0;
    muzero::GumbelSearch<Full> search(evaluator,s,g,17);
    for(Rule rule:{Rule::FREESTYLE,Rule::RENJU,Rule::HEX})for(int budget:{0,1,2,7,17,31}) {
        Game game(5,6,rule);game.play(0);auto result=search.run(game,0,SearchRun{budget+1});
        check(result.simulations==budget&&result.root_visits==budget+1,"MZ exact root budget");
        check(game.legal(result.action)&&result.policy[0]==0&&result.policy[5]==0,"MZ root legal mask");
        if(budget==1)check(std::count(result.visits.begin(),result.visits.end(),0)>1&&result.policy[2]>0,"MZ dense completed target");
    }
    std::atomic<int> stop_checks{0};SearchRun interrupted;interrupted.max_visits=100001;
    interrupted.should_stop=[&]{return ++stop_checks>10;};auto partial=search.run(Game(5,6,Rule::FREESTYLE),0,interrupted);
    check(partial.stopped_early&&partial.simulations<100000&&partial.root_visits==partial.simulations+1,"MZ mid-search stop drains work");
    SearchRun timed;timed.max_visits=100001;timed.max_time=0;auto limited=search.run(Game(5,6,Rule::FREESTYLE),0,timed);
    check(limited.stopped_early&&limited.simulations>0&&limited.simulations<100000,"MZ time cap drains work");
    evaluator.finish();
    auto backend=std::make_unique<TestMuBackend>();backend->fail=true;backends.push_back(std::move(backend));
    muzero::BatchEvaluator failure(std::move(backends),6,8,0,0,false,0,17);
    muzero::GumbelSearch<Full> broken(failure,s,g,17);bool raised=false;
    try{broken.run(Game(5,6,Rule::FREESTYLE),0);}catch(const std::runtime_error& e){raised=std::string(e.what()).find("injected")!=std::string::npos;}
    check(raised,"recurrent failure propagates without deadlock");
    try{failure.finish();}catch(const std::runtime_error& e){check(std::string(e.what()).find("injected")!=std::string::npos,"service preserves failure");}
}
int main(int argc,char**) {
    if(argc>1) {
        std::cout<<'[';bool first=true;
        for(int budget:{1,2,3,7,17,31,200}) {
            GumbelSettings s;s.noise_scale=0;std::mt19937_64 rng(7);
            std::vector<GumbelChild> children{{.5,.8,0,0},{.3,-.2,0,0},{.2,-.6,0,0}};
            GumbelRoot root({.5,.3,.2},budget,s,rng);
            for(int i=0;i<budget;++i){int a=root.reserve(gumbel_completed_q(children,.1,s));root.commit(a);++children[a].visits;}
            auto pi=gumbel_policy(children,.1,s);auto q=gumbel_completed_q(children,.1,s);
            if(!first)std::cout<<',';first=false;
            std::cout.precision(17);std::cout<<"{\"budget\":"<<budget<<",\"action\":"<<root.winner(q)<<",\"visits\":[";
            for(size_t i=0;i<children.size();++i){if(i)std::cout<<',';std::cout<<children[i].visits;}
            std::cout<<"],\"policy\":[";for(size_t i=0;i<pi.size();++i){if(i)std::cout<<',';std::cout<<pi[i];}std::cout<<"]}";
        }
        std::cout<<"]\n";return 0;
    }
    GumbelSettings g;g.rescale_q_values=false;g.c_visit=8;g.c_scale=.1;
    std::vector<GumbelChild> children{{.5,.7,2,0},{.25,0,0,0},{.25,0,0,0}};
    auto q=gumbel_completed_q(children,.1,g);check(std::abs(q[0]-.7)<1e-12&&std::abs(q[1]-.5)<1e-12,"hand-computed mixed Q");
    auto pi=gumbel_policy(children,.1,g);check(std::abs(pi[0]-1/(1+std::exp(-.2)))<1e-12,"hand-computed policy");
    check(gumbel_interior_selection({{.6,0,1,0},{.4,0,0,0}},0,g)==1,"Eq14 correct visitation deficit");
    check(quantize_gumbel_policy({.9,.04,.03,.02,.01})[4]>0,"dense target precision");
    std::mt19937_64 rng(7);
    GumbelRoot odd({.5,.3,.2},10,g,rng);
    std::vector<int> expected{0,0,0,1,1,2,2,3,3,4};
    for(int i=0;i<10;++i)check(odd.considered_visit(i)==expected[i],"mctx odd-candidate schedule");
    for(int m:{1,2,3,5,16})for(int budget:{1,2,17,31,200}) {
        g.max_num_considered_actions=m;g.noise_scale=0;GumbelRoot root(std::vector<double>(m,1.0/m),budget,g,rng);
        for(int i=0;i<budget;++i){int a=root.reserve(std::vector<double>(m,0));check(a>=0,"budget remainder issued");root.commit(a);}
        check(root.done()&&std::accumulate(root.visits().begin(),root.visits().end(),int64_t(0))==budget,"complete schedule budget");
    }
    alpha<false>();alpha<true>();muzero_case<false>();muzero_case<true>();std::cout<<"Gumbel math and all four search combinations passed\n";
}
