#include "etazero/record.h"
#include <iostream>
#include <filesystem>
#include <limits>
using namespace etazero;
WDL wdl(double v) { return {(1+v)/2,0,(1-v)/2}; }
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct KnownEvaluator : Evaluator {
    int preferred; double v; std::atomic<int> calls{0};
    KnownEvaluator(int action, double value) : preferred(action), v(value) {}
    Evaluation evaluate(const std::vector<float>& obs) override {
        ++calls; Evaluation e; e.logits.assign((obs.size()-GLOBAL_FEATURES)/INPUT_PLANES,-100);e.logits.at(preferred)=100;e.wdl=wdl(v);return e;
    }
};
struct BrokenBackend : Backend {
    std::vector<Evaluation> evaluate(const InferenceInputs&) override { throw std::runtime_error("injected backend failure"); }
};
struct IdentityBackend : Backend {
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override {
        std::vector<Evaluation> result;
        for(auto obs:inputs)result.push_back({std::vector<double>((obs->size()-GLOBAL_FEATURES)/INPUT_PLANES,(*obs)[0]),wdl((*obs)[0])});
        return result;
    }
};
struct ConcurrentBackend : IdentityBackend {
    std::atomic<int>& entered;
    explicit ConcurrentBackend(std::atomic<int>& count) : entered(count) {}
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override {
        ++entered;auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
        while(entered<2) {
            if(std::chrono::steady_clock::now()>deadline)throw std::runtime_error("Services did not execute concurrently");
            std::this_thread::yield();
        }
        return IdentityBackend::evaluate(inputs);
    }
};
struct ArithmeticState : SearchState {
    int depth=0;
    std::unique_ptr<SearchState> clone() const override { return std::make_unique<ArithmeticState>(*this); }
    int actions() const override { return 1; }
    bool legal(int a) const override { return a==0&&!terminal(); }
    bool terminal() const override { return depth==2; }
    double terminal_value() const override { return -1; }
    Evaluation evaluate() const override { return {{0},wdl(0.6)}; }
    Transition move(int) override { ++depth;return {0.2,0.5,1}; }
};
struct FailedLeaf : Evaluator {
    std::atomic<int> calls{0};
    Evaluation evaluate(const std::vector<float>& obs) override {
        if(calls.fetch_add(1)>0)throw std::runtime_error("injected leaf failure");
        return {std::vector<double>((obs.size()-GLOBAL_FEATURES)/INPUT_PLANES,0),wdl(0)};
    }
};
struct BanditState : SearchState {
    int depth=0,action=0;bool heterogeneous;
    explicit BanditState(bool h) : heterogeneous(h) {}
    std::unique_ptr<SearchState> clone() const override {return std::make_unique<BanditState>(*this);}
    int actions() const override {return 100;}
    bool legal(int a) const override {return !terminal() && a%3!=0;}
    bool terminal() const override {return depth==1;}
    double terminal_value() const override {return heterogeneous?action/100.0:0;}
    Evaluation evaluate() const override {return {std::vector<double>(100,0),wdl(0)};}
    Transition move(int a) override {++depth;action=a;return {0,1,1};}
};
struct BroadState : SearchState {
    int depth=0;
    std::unique_ptr<SearchState> clone() const override {return std::make_unique<BroadState>(*this);}
    int actions() const override {return 96;}
    bool legal(int a) const override {return a>=0 && a<96 && !terminal();}
    bool terminal() const override {return depth==5;}
    double terminal_value() const override {return 0;}
    Evaluation evaluate() const override {return {std::vector<double>(96,0),wdl(0)};}
    Transition move(int) override {++depth;return {0,1,1};}
};
void compare_exhaustive_bandit(bool heterogeneous) {
    BanditState state(heterogeneous);std::vector<int64_t> expected(100,0);std::mt19937_64 rng(42);
    const int budget=523,legal=66;
    for(int t=0;t<budget;++t) {
        double best=-std::numeric_limits<double>::infinity();std::vector<int> choices;
        for(int a=0;a<100;++a)if(a%3!=0) {
            double q=expected[a] && heterogeneous?a/100.0:0;
            double score=q+1.5*(1.0/legal)*std::sqrt(static_cast<double>(t)+0.01)/(1+expected[a]);
            if(score>best){best=score;choices.clear();}if(score==best)choices.push_back(a);
        }
        ++expected[choices[std::uniform_int_distribution<size_t>(0,choices.size()-1)(rng)]];
    }
    Search search({budget,1,1.5,1,0,0.3,true},42);auto result=search.run(state,0);
    check(result.visits==expected,"Sparse selection matches exhaustive PUCT including ties and masks");
    search.reset(42);check(search.run(state,0).visits==expected,"Search reset preserves independent game RNG");
}
void forbidden_case(std::initializer_list<std::pair<int,int>> points, int x,int y,bool expected) {
    Board b(15);for(auto p:points)b.cells[p.second*15+p.first]=1;
    RenjuAnalyzer a;check(a.forbidden(b,y*15+x)==expected,"Renju reference case");
}
int main() {
    try {
        for(int threads : {1,4}) {
            BroadState position;
            Search capped({99,threads,1.5,1,0,1,true,100},42);
            auto first=capped.run(position,0);
            check(first.root_visits==100 && first.simulations==99,"100v includes initial root visit");
            auto again=capped.run(position,0);
            check(again.root_visits==100 && again.simulations==0,"Reused visits consume the cap");
            capped.advance(first.action);position.move(first.action);
            auto next=capped.run(position,0);
            check(next.root_visits==100 && next.simulations<=99,"Child reuse never adds 100 visits on top");
        }
        for(Rule rule:{Rule::FREESTYLE,Rule::STANDARD,Rule::RENJU}) {
            Game g(6,9,rule);
            for(int a:{0,6,1,8,3,10,4,12,5,14,2})g.play(a/6*9+a%6);
            check(g.finished()==(rule!=Rule::STANDARD),"Overline terminal rule");
            check(g.winner()==(rule==Rule::FREESTYLE?1:rule==Rule::RENJU?-1:0),"Overline winner");
            Game white(6,6,rule);for(int a:{6,0,8,1,10,3,12,4,14,5,16,2})white.play(a);
            check(white.winner()==(rule==Rule::STANDARD?0:-1),"White overline");
            Game draw(5,7,rule);
            for(int a:{0,1,2,3,4,5,6,7,8,9,11,10,13,12,15,14,17,16,19,18,20,21,22,23,24})draw.play(a/5*7+a%5);
            check(draw.finished()&&draw.winner()==0,"Full-board draw");
        }
        forbidden_case({{6,7},{8,7},{7,6},{7,8}},7,7,true);
        forbidden_case({{5,7},{6,7},{8,7},{7,5},{7,6},{7,8}},7,7,true);
        forbidden_case({{1,0},{2,0},{0,1},{0,2}},0,0,false);
        forbidden_case({{3,7},{4,7},{5,7},{6,7},{7,6},{7,8},{6,6},{8,8}},7,7,false);
        forbidden_case({{5,7},{6,7},{8,7},{9,7},{7,4},{7,5},{7,6},{7,8},{7,9}},7,7,false);
        forbidden_case({{6,1},{8,1},{6,3},{7,3},{9,3},{10,3},{9,4}},7,2,false);
        SearchSettings one{1,1,1.5,1,0,0.3,true};
        Game win(5,7,Rule::FREESTYLE);for(int a:{0,7,1,8,2,9,3,10})win.play(a);
        KnownEvaluator e(4,0.25);Search search(e,one,1);auto r=search.run(win,0);
        check(r.action==4&&r.value==0.625&&r.visits[4]==1&&e.calls==1,"One-ply win and terminal inference");
        Game renju(15,15,Rule::RENJU);for(int a:{111,0,113,2,97,4,127,6})renju.play(a);
        for(int perspective:{1,-1}) {
            auto full=renju.observation(perspective), dropped=renju.observation(perspective,false);
            int area=225,g=INPUT_PLANES*area,plane=perspective==1?3:4;
            check(full[plane*area+112]==1 && full[(7-plane)*area+112]==0,"Known double-three is in the mover-specific black forbidden plane");
            check(full[g+2]==-perspective && full[g+3]==1 && dropped[g+3]==0,"Renju global signs and feature availability");
            for(int i=0;i<g;++i) {
                if(i<3*area)check(full[i]==dropped[i],"Feature dropout preserves mask and stones");
                else check(dropped[i]==0,"Feature dropout zeros both forbidden planes");
            }
            for(auto rule:{Rule::FREESTYLE,Rule::STANDARD}) {
                Game other(15,15,rule);for(int a:{111,0,113,2,97,4,127,6})other.play(a);
                auto input=other.observation(perspective);
                check(input[g]==(rule==Rule::STANDARD) && input[g+1]==0 && input[g+2]==0 && input[g+3]==0,"Non-Renju globals");
                for(int i=3*area;i<g;++i)check(input[i]==0,"Non-Renju has no forbidden feature");
            }
        }
        KnownEvaluator re(112,0);Search rs(re,one,2);auto rr=rs.run(renju,0);
        check(rr.action==112&&rr.value==(-1+re.v)/2&&re.calls==1,"Forbidden loss perspective");
        Game empty(5,5,Rule::FREESTYLE);KnownEvaluator ve(0,0.5);Search vs(ve,one,3);auto vr=vs.run(empty,1);
        check(vr.value==0&&ve.calls==2,"Leaf value changes player perspective");
        ArithmeticState state;Search generic(one,9);bool unsupported=false;
        try {generic.run(state,0);}catch(const std::runtime_error&){unsupported=true;}
        check(unsupported && generic.pending()==0,"WDL search rejects unsupported reward/discount adapters");
        SearchSettings many{31,4,1.5,1,0.25,0.3,true};KnownEvaluator pe(0,0);
        Search ps(pe,many,4);auto pr=ps.run(empty,1);int64_t sum=0;for(auto n:pr.visits)sum+=n;
        check(sum==31&&pr.simulations==31&&ps.pending()==0,"Parallel exact budget and rollback");
        empty.play(pr.action);ps.advance(pr.action);auto next=ps.run(empty,0);
        check(next.simulations==31&&ps.pending()==0,"Persistent threads and subtree promotion");
        FailedLeaf fe;Search fs(fe,many,3);
        bool leaf_failed=false;
        try{fs.run(Game(5,5,Rule::FREESTYLE),0);}catch(const std::runtime_error&){leaf_failed=true;}
        check(leaf_failed,"Leaf error must propagate");
        check(fs.pending()==0,"Failed search releases all in-flight reservations");
        compare_exhaustive_bandit(false);compare_exhaustive_bandit(true);
        BroadState broad;Search wide({8192,4,1.5,1,0,0.3,true},7);
        auto broad_result=wide.run(broad,0);int64_t broad_visits=0;
        for(auto n:broad_result.visits){check(n>0,"All legal children remain reachable across storage tiers");broad_visits+=n;}
        check(broad_visits==8192 && wide.pending()==0,"Large parallel search budget");
        broad.move(broad_result.action);wide.advance(broad_result.action);
        check(wide.run(broad,0).simulations==8192,"Parallel reclamation preserves the promoted subtree");
        wide.reset(7);check(wide.pending()==0,"Reset after parallel reclamation");
        std::atomic<int> entered{0};std::vector<std::unique_ptr<Backend>> concurrent;
        for(int i=0;i<2;++i)concurrent.push_back(std::make_unique<ConcurrentBackend>(entered));
        BatchEvaluator services(std::move(concurrent),"m",5,1,1,0,0);
        std::vector<std::thread> service_callers;std::atomic<int> serviced{0};
        for(int i=0;i<2;++i)service_callers.emplace_back([&]{services.evaluate(empty.observation());++serviced;});
        for(auto& t:service_callers)t.join();services.finish();
        check(serviced==2 && services.rows_by_server[0]==1 && services.rows_by_server[1]==1,"Each service owns a concurrent backend");
        std::vector<std::unique_ptr<Backend>> identity;identity.push_back(std::make_unique<IdentityBackend>());
        BatchEvaluator cache(std::move(identity),"m",5,1,2,0,1);
        auto obs=empty.observation();cache.evaluate(obs);cache.evaluate(obs);
        check(cache.requests==1 && cache.cache_hits==1,"Identical network inputs hit the cache");
        obs[0]=0;check(cache.evaluate(obs).value()==0,"Collision must compare full input key");
        obs[0]=1;check(cache.evaluate(obs).value()==1,"Eviction must not return stale output");
        obs[INPUT_PLANES*25+2]=-1;cache.evaluate(obs);
        check(cache.requests==4,"Rule feature belongs to cache key");cache.finish();
        std::vector<std::unique_ptr<Backend>> broken_backends;
        for(int i=0;i<2;++i)broken_backends.push_back(std::make_unique<BrokenBackend>());
        BatchEvaluator broken(std::move(broken_backends),"test",5,8,2,1000,0);
        std::vector<std::thread> callers;std::atomic<int> errors{0};
        for(int i=0;i<12;++i)callers.emplace_back([&]{try{broken.evaluate(empty.observation());}catch(...){++errors;}});
        for(auto& t:callers)t.join();check(errors==12,"All inference waiters are awakened on failure");
        bool finish_failed=false;
        try{broken.finish();}catch(const std::runtime_error&){finish_failed=true;}
        check(finish_failed,"Failure must propagate at finish");
        auto failed_dir=std::filesystem::temp_directory_path()/(
            "etazero_writer_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(!std::filesystem::exists(failed_dir),"Test directory must be new");
        RecordWriter writer(failed_dir,{"r","a","m","c","s",1,0},1,1,0.01);
        std::filesystem::remove(failed_dir);
        Game wg(5,7,Rule::FREESTYLE);FinishedGame record{};
        record.id=0;record.seed=0;record.size=5;record.canvas=7;record.rule=Rule::FREESTYLE;
        for(int action:{0,7,1,8,2,9,3,10,4}) {
            Step s{};s.player=wg.player();s.action=action;s.simulations=1;s.temperature=1;s.observation=wg.observation();
            s.policy.assign(49,0);s.visits.assign(49,0);s.policy[action]=1;s.visits[action]=1;
            wg.play(action);s.reward=wg.finished()?wg.winner()*s.player:0;record.steps.push_back(std::move(s));
        }
        record.winner=wg.winner();record.reason=wg.reason();record.final_player=wg.player();record.final_observation=wg.observation();
        writer.enqueue(record);std::atomic<int> writer_errors{0};std::vector<std::thread> producers;
        for(int i=0;i<3;++i)producers.emplace_back([&]{try{writer.enqueue(record);}catch(...){++writer_errors;}});
        for(auto& t:producers)t.join();bool writer_failed=false;
        try{writer.finish();}catch(const std::exception&){writer_failed=true;}
        check(writer_failed&&writer_errors>0,"Writer failure propagates and wakes blocked producers");
        std::cout<<"Rules, perspectives, search budgets, persistent workers and failure propagation passed\n";
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
