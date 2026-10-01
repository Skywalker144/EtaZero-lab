#include "etazero/record.h"
#include <iostream>
#include <filesystem>
using namespace etazero;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct KnownEvaluator : Evaluator {
    int preferred; double v; std::atomic<int> calls{0};
    KnownEvaluator(int action, double value) : preferred(action), v(value) {}
    Evaluation evaluate(const std::vector<float>& obs) override {
        ++calls; Evaluation e; e.logits.assign(obs.size()/INPUT_PLANES,-100);e.logits.at(preferred)=100;e.value=v;return e;
    }
};
struct BrokenBackend : Backend {
    std::vector<Evaluation> evaluate(const std::vector<std::vector<float>>&) override { throw std::runtime_error("injected backend failure"); }
};
struct ArithmeticState : SearchState {
    int depth=0;
    std::unique_ptr<SearchState> clone() const override { return std::make_unique<ArithmeticState>(*this); }
    int actions() const override { return 1; }
    bool legal(int a) const override { return a==0&&!terminal(); }
    bool terminal() const override { return depth==2; }
    double terminal_value() const override { return -1; }
    Evaluation evaluate() const override { return {{0},0.6}; }
    Transition move(int) override { ++depth;return {0.2,0.5,1}; }
};
struct FailedLeaf : Evaluator {
    std::atomic<int> calls{0};
    Evaluation evaluate(const std::vector<float>& obs) override {
        if(calls.fetch_add(1)>0)throw std::runtime_error("injected leaf failure");
        return {std::vector<double>(obs.size()/INPUT_PLANES,0),0};
    }
};
void forbidden_case(std::initializer_list<std::pair<int,int>> points, int x,int y,bool expected) {
    Board b(15);for(auto p:points)b.cells[p.second*15+p.first]=1;
    RenjuAnalyzer a;check(a.forbidden(b,y*15+x)==expected,"Renju reference case");
}
int main() {
    try {
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
        check(r.action==4&&r.value==1&&r.visits[4]==1&&e.calls==1,"One-ply win and terminal inference");
        Game renju(15,15,Rule::RENJU);for(int a:{111,0,113,2,97,4,127,6})renju.play(a);
        KnownEvaluator re(112,0);Search rs(re,one,2);auto rr=rs.run(renju,0);
        check(rr.action==112&&rr.value==-1&&re.calls==1,"Forbidden loss perspective");
        Game empty(5,5,Rule::FREESTYLE);KnownEvaluator ve(0,0.5);Search vs(ve,one,3);auto vr=vs.run(empty,1);
        check(vr.value==-0.5&&ve.calls==2,"Leaf value changes player perspective");
        ArithmeticState state;Search generic(one,9);auto gr=generic.run(state,0);
        check(std::abs(gr.value-0.5)<1e-12,"Algorithm adapter supplies reward, discount and perspective");
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
        BatchEvaluator broken(std::make_unique<BrokenBackend>(),"test",5,8,2,1000);
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
