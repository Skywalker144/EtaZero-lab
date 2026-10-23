#include "etazero/game.h"
#include "etazero/game_config.h"
#include "etazero/algorithm.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <unistd.h>

// Search resets the same scratch state once per simulation. Count only that
// operation, after constructors and third-party rule tables have initialized.
namespace { bool count_allocations=false; size_t allocations=0; }
void* operator new(std::size_t size) {
    if(count_allocations)++allocations;
    if(void* p=std::malloc(size?size:1))return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
void operator delete[](void* p,std::size_t) noexcept { std::free(p); }

using namespace etazero;
void check(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
int first_move(const Game& g) {
    for(int a=0;a<g.actions();++a)if(g.legal(a))return a;
    throw std::runtime_error("Expected a legal move");
}
void same_state(const Game& a,const Game& b) {
    check(a.rule()==b.rule()&&a.size()==b.size()&&a.canvas()==b.canvas(),"Copied game dimensions/rule");
    check(a.player()==b.player()&&a.turn()==b.turn()&&a.last_action()==b.last_action(),"Copied move state");
    check(a.finished()==b.finished()&&a.winner()==b.winner()&&a.reason()==b.reason(),"Copied outcome");
    check(a.board().cells==b.board().cells&&a.observation()==b.observation(),"Copied board/observation");
    check(a.history_identity()==b.history_identity()&&a.fen()==b.fen(),"Copied history");
    for(int m=0;m<a.actions();++m)check(a.legal(m)==b.legal(m),"Copied legal actions");
}
struct NoEvaluation : Evaluator {
    Evaluation evaluate(const std::vector<float>&) override {throw std::runtime_error("Graph identity needs no inference");}
};
void history_identity() {
    NoEvaluation evaluator;
    auto distinct=[&](const Game& a,const Game& b) {
        check(a.turn()==b.turn() && a.player()==b.player() && a.board().cells==b.board().cells,"History fixture has identical visible state and turn");
        check(a.observation()!=b.observation(),"Neural input must retain recent history");
        AlphaZeroState first(a,evaluator),second(b,evaluator);
        check(a.history_identity()!=b.history_identity(),"Rule state must retain different histories");
        if(a.rule()==Rule::GO)check(first.graph_key()==second.graph_key(),"Go should merge safe move-order transpositions");
        else check(first.graph_key()!=second.graph_key(),"Exact graph identity must retain history");
        auto clone=first.clone();check(clone->graph_key()==first.graph_key(),"Search clone preserves history identity");
        check(clone->reset_from(second)&&clone->graph_key()==second.graph_key(),"Search reset replaces all history");
    };
    Game ga(5,5,Rule::GO),gb(5,5,Rule::GO);
    for(int a:{0,1,2,3})ga.play(a);
    for(int a:{2,3,0,1})gb.play(a);
    distinct(ga,gb);
    Game xa(10,10,Rule::XIANGQI),xb=xa;
    for(auto m:{std::array<int,4>{1,0,2,2},{1,9,2,7},{2,2,1,0},{2,7,1,9}}) {
        xa.play(xiangqi_action(m[0],m[1],m[2],m[3]));
        xb.play(xiangqi_action(8-m[0],m[1],8-m[2],m[3]));
    }
    distinct(xa,xb);
    RuleOptions options;options.go_rules="japanese";
    Game phases(5,5,Rule::GO,options),scratch(5,5,Rule::GO);
    for(int i=0;i<6;++i) {
        check(!phases.finished(),"Territory scoring must preserve both encore phases");
        phases.play(25);scratch=phases;same_state(scratch,phases);
        Game cloned=scratch;same_state(cloned,phases);
    }
    check(phases.finished(),"Territory scoring must end after the second encore");
    options.go_rules="tromp-taylor";Game superko(5,5,Rule::GO,options);
    for(int a:{0,5,24,6,23,2})superko.play(a);
    check(superko.legal(1),"First two-stone suicide is legal under Tromp-Taylor");
    for(int a:{1,25})superko.play(a);
    Game before(5,5,Rule::GO,options);for(int a:{25,5,24,6,23,2})before.play(a);
    check(superko.board().cells==before.board().cells && superko.player()==before.player(),"Superko fixture returns to the same board/player");
    check(before.legal(0)&&!superko.legal(0),"Positional superko forbids recreating the position before suicide");
    scratch=superko;Game cloned=superko;same_state(scratch,superko);same_state(cloned,superko);
    check(!scratch.legal(0)&&!cloned.legal(0),"Clone/reset must retain superko beyond the current board");
}
int main() {
    history_identity();
    for(auto rule:{Rule::FREESTYLE,Rule::STANDARD,Rule::RENJU,Rule::HEX,Rule::GO,Rule::XIANGQI}) {
        int size=rule==Rule::XIANGQI?10:5,canvas=rule==Rule::XIANGQI?10:7;
        Game source(size,canvas,rule);source.set_pda(.5,-1);
        source.play(first_move(source));source.play(first_move(source));
        auto initial=source.observation();
        Game cloned=source;same_state(cloned,source);
        Game scratch(size,canvas,rule);scratch=source;same_state(scratch,source);
        check(scratch.pda_doublings()==.5&&scratch.pda_player()==-1,"Copied conditioning");
        scratch.play(first_move(scratch));cloned.play(first_move(cloned));
        same_state(scratch,cloned);check(source.observation()==initial,"Independent rule state/history");
        scratch=source;same_state(scratch,source);
        scratch=scratch;same_state(scratch,source);
        Game replaced(6,6,Rule::STANDARD);replaced=source;same_state(replaced,source);
    }
    for(auto rule:{Rule::FREESTYLE,Rule::STANDARD,Rule::RENJU,Rule::HEX}) {
        Game source(15,19,rule);source.play(0);source.play(20);
        for(int perspective:{1,-1}) {
            auto observation=source.observation(perspective);
            check(!std::signbit(observation[5*19*19+5]),"Disabled PDA must preserve positive zero for random bootstrap hashing");
        }
        Game scratch=source;
        allocations=0;count_allocations=true;
        for(int i=0;i<1000;++i)scratch=source;
        count_allocations=false;
        check(allocations==0,"Same-rule scratch reset must reuse storage");
    }
    Game ko(5,5,Rule::GO);
    for(int a:{5,6,1,10,7,12,24,16,11})ko.play(a);
    Game reset(5,5,Rule::GO);reset.board();reset=ko;
    check(!reset.legal(6),"Reset preserves ko restriction");
    check(reset.board().cells==ko.board().cells,"Reset invalidates stale display cache");
    reset.play(4);reset.play(9);check(reset.legal(6)&&!ko.legal(6),"Independent ko history after reset");
    ko.play(25);reset=ko;reset.play(25);
    check(reset.finished()&&!ko.finished(),"Reset preserves pass history");

    Game repetition(10,10,Rule::XIANGQI);
    {
        Game source(10,10,Rule::XIANGQI);
        for(auto m:{std::array<int,4>{1,0,2,2},{1,9,2,7},{2,2,1,0},{2,7,1,9}})
            source.play(xiangqi_action(m[0],m[1],m[2],m[3]));
        repetition=source;
    }
    for(auto m:{std::array<int,4>{1,0,2,2},{1,9,2,7},{2,2,1,0},{2,7,1,9}})
        repetition.play(xiangqi_action(m[0],m[1],m[2],m[3]));
    check(repetition.finished()&&repetition.winner()==0&&repetition.reason()==Reason::REPETITION,"Reset owns its repetition links");

    auto go=game_spec(Rule::GO),xq=game_spec(Rule::XIANGQI);
    auto geometry=repetition.board();
    check(geometry.width==9&&geometry.height==10&&geometry.stride==10&&geometry.area()==90&&geometry.cells.size()==100,"Board geometry is independent of storage padding");
    check(repetition.area()==90&&repetition.canvas_area()==100&&repetition.actions()==2086,"Board, canvas and action domains are distinct");
    check((Outcome{0,Reason::NO_RESULT}).wdl(1)==std::array<double,3>{.5,0,.5},"No-result WDL is neutral");
    check(!(Outcome{0,Reason::NO_RESULT}).has_result()&&(Outcome{0,Reason::SCORE}).draw_probability()==1,"No-result and scored draw differ");
    auto path=std::filesystem::temp_directory_path()/("etazero-game-config-"+std::to_string(getpid()));
    {std::ofstream out(path);out<<"[go]\nrules=chinese\nkomi=7.5\n[analysis]\ngo_rules=japanese\nkomi=6.5\n[match]\ngo_rules=tromp-taylor\nkomi=0.5\n";}
    Config config(path.string());std::filesystem::remove(path);
    auto train=rule_options(config,GameConfigMode::SELFPLAY),analysis=rule_options(config,GameConfigMode::ANALYSIS),match=rule_options(config,GameConfigMode::MATCH);
    check(train.go_rules=="chinese"&&train.komi==7.5,"Selfplay ignores engine-profile overrides");
    check(analysis.go_rules=="japanese"&&analysis.komi==6.5&&match.go_rules=="tromp-taylor"&&match.komi==.5,"Each engine mode selects only its own override");
    check(go.native_action(25,5,7)==49&&go.local_action(49,5,7)==25,"Padded pass mapping");
    check(go.static_actions(5,7).size()==26,"Padded Go action mask");
    check(xq.width(10)==9&&xq.static_actions(10,10).size()==2086,"Xiangqi rectangular vocabulary");
    check(same_layout(game_spec(Rule::HEX),game_spec(Rule::RENJU))&&!same_layout(go,xq),"Shared layout boundary");
    bool rejected=false;try{Game bad(9,10,Rule::XIANGQI);}catch(const std::runtime_error&){rejected=true;}
    check(rejected,"Xiangqi rejects an invalid board size");
    std::cout<<"All rule states clone/reset independently; Gomoku/Hex reset allocations: 0\n";
}
