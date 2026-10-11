#include "etazero/game.h"
#include "etazero/opening.h"
#include "etazero/algorithm.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
using namespace etazero;
static void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct White : Evaluator {
    int calls=0;
    Evaluation evaluate(const std::vector<float>& obs) override {
        ++calls;int area=(obs.size()-GLOBAL_FEATURES)/INPUT_PLANES;
        check(obs[INPUT_PLANES*area+HEX_GLOBAL]==1 && obs[INPUT_PLANES*area+HEX_WHITE_GLOBAL]==1,"Hex opening evaluates White");
        check(std::accumulate(obs.begin()+area,obs.begin()+2*area,0.0)==0,"White has no stones at opening");
        check(std::accumulate(obs.begin()+2*area,obs.begin()+3*area,0.0)==1,"Opening adds one Black stone");
        return {std::vector<double>(area,0),{.5,.4,.1}};
    }
};
struct Unused : Evaluator {Evaluation evaluate(const std::vector<float>&)override{throw std::runtime_error("Black opening evaluator used");}};
struct Coordinates : Backend {
    std::vector<std::vector<float>> seen;
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override {
        std::vector<Evaluation> result;
        for(auto p:inputs){seen.push_back(*p);int area=(p->size()-GLOBAL_FEATURES)/INPUT_PLANES;std::vector<double> logits(area);std::iota(logits.begin(),logits.end(),0);result.push_back({logits,{.5,0,.5}});}
        return result;
    }
};
int main(){
    try {
        Board b(5);
        for(int y=0;y<5;++y)b.cells[y*5+4-y]=1;
        check(b.hex_connected(1),"Anti-diagonal is a six-neighbor Black connection");
        std::fill(b.cells.begin(),b.cells.end(),0);
        for(int y=0;y<5;++y)b.cells[y*5+y]=1;
        check(!b.hex_connected(1),"Virtual bridges are not actual connections");
        std::fill(b.cells.begin(),b.cells.end(),0);
        for(int x=0;x<5;++x)b.cells[10+x]=-1;
        check(b.hex_connected(-1) && !b.hex_connected(1),"White connects left/right");
        std::mt19937_64 random(19);
        for(int seed=0;seed<1000;++seed){for(auto& cell:b.cells)cell=(random()&1)?1:-1;check(b.hex_connected(1)!=b.hex_connected(-1),"Every full Hex coloring has exactly one winner");}
        Game game(5,7,Rule::HEX);
        for(int a:{2,0,9,1,16,3,23,4,30})game.play(a);
        check(game.finished() && game.winner()==1 && game.reason()==3 && game.terminal_value()==-1,"Black terminal perspective and reason");
        check(!game.legal(5),"No moves after connection");
        check(hex_opening_accept_rate(.5,20,.001)==1,"Fair first move always accepted");
        check(std::abs(hex_opening_accept_rate(.75,2,.001)-.5625)<1e-12,"Acceptance formula hand calculation");
        check(hex_opening_accept_rate(0,6,.005)==.005 && hex_opening_accept_rate(1,20,.001)==.001,"Acceptance floor at extremes");
        bool refused=false;try{hex_opening_accept_rate(.5,6,0);}catch(const std::runtime_error&){refused=true;}check(refused,"Zero acceptance floor is rejected");
        auto path=std::filesystem::temp_directory_path()/"etazero_hex_opening_test.cfg";
        {std::ofstream f(path);f<<"[opening]\nprobability=1\navg_dist_factor=.8\nbalance_exponent=4\nrejection_probability=.995\nrejection_probability_fallback=.8\nmax_tries=20\npolicy_init=false\n[hex_opening]\nprobability=1\nmake_fair_probability=1\nbalance_exponent=20\nmin_accept_rate=.001\n";}
        Config config(path.string());OpeningConfig opening(config);std::filesystem::remove(path);
        Unused black;White white;
        std::array<int,25> frequencies{};
        for(int seed=0;seed<1000;++seed){Game g(5,7,Rule::HEX);std::mt19937_64 rng(seed);auto result=initialize_opening(g,opening,black,white,rng);check(result.status==OpeningStatus::Success && result.attempts==1 && result.actions.size()==1 && g.player()==-1,"One-stone acceptance uses W even when D is nonzero");check(result.balance_evaluators==std::vector<int>{1},"White evaluator recorded");int a=result.actions[0];++frequencies[a/7*5+a%7];}
        for(int n:frequencies)check(n>15 && n<70,"Uniform first-move proposals cover edges and center");
        opening.hex_make_fair_probability=0;Game empty(5,7,Rule::HEX);auto none=initialize_opening(empty,opening,black,white,random);check(none.status==OpeningStatus::NotAttempted && none.attempts==0 && empty.turn()==0,"Fair gate may leave the board empty");
        opening.hex_make_fair_probability=1;auto interrupted=initialize_opening(empty,opening,black,white,random,[]{return true;});check(interrupted.status==OpeningStatus::Interrupted,"Hex opening is cancellable");
        Game position(5,7,Rule::HEX);position.play(1);auto obs=position.observation();
        auto backend=std::make_unique<Coordinates>();auto* probe=backend.get();BatchEvaluator evaluator(std::move(backend),"hex",7,8,32,0);
        auto result=evaluator.evaluate_symmetry(obs,0,true);
        check(probe->seen.back()[2*49+7]==1 && probe->seen.back()[2*49+1]==0,"White input is transposed before model");
        for(int a=0;a<49;++a)check(result.logits[a]==(a%7)*7+a/7,"White output is restored to physical coordinates");
        result=evaluator.evaluate_symmetry(obs,1,true);
        for(int a=0;a<49;++a)check(result.logits[a]==(6-a%7)*7+6-a/7,"Hex 180 and White transpose compose");
        refused=false;try{evaluator.evaluate_symmetry(obs,2,true);}catch(const std::runtime_error&){refused=true;}check(refused,"Hex refuses invalid symmetries");
        AlphaZeroState state(position,evaluator);check(state.symmetry_count()==2,"Hex root ensemble has two members");
        evaluator.finish();std::cout<<"Hex rules, opening and model coordinates passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
