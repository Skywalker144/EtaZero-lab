#include "etazero/search.h"
#include "etazero/gumbel_search.h"
#include "etazero/time_budget.h"
#include "etazero/random_evaluator.h"
#include <iostream>
#include <numeric>
#include <set>
#include <fstream>
#include <filesystem>
using namespace etazero;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void near(double a,double b,const char* message){check(std::abs(a-b)<1e-9,message);}
struct Branch : SearchState {
    int depth=0;std::string path;std::vector<std::pair<int,int>>* history=nullptr;
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<Branch>(*this);}
    int player() const override{return depth%2?-1:1;}
    int actions() const override{return 3;}
    bool legal(int a) const override{return a>=0&&a<3&&!terminal();}
    bool terminal() const override{return depth==4;}
    double terminal_value() const override{return 0;}
    Evaluation evaluate(SearchEvaluation options={}) const override {
        if(history)history->push_back({depth,options.max_history});
        return {{0,0,0},{.25,.5,.25}};
    }
    int move(int action) override{check(legal(action),"illegal branch move");path+=char('0'+action);++depth;return -1;}
    std::string graph_key() const override{return path;}
};
struct Flat : Evaluator {
    int preferred=-1;
    Evaluation evaluate(const std::vector<float>&) override {Evaluation e;e.logits.assign(36,0);if(preferred>=0)e.logits[preferred]=20;e.wdl={.25,.5,.25};return e;}
};
struct Probe : Search {
    using Search::Search;
    size_t cached() const {size_t n=0;for(const auto& shard:eval_cache_)n+=shard.size();return n;}
    double cached_root_value() const {return root_->cached_evaluation?root_->cached_evaluation->value:100;}
};
SearchSettings settings(){SearchSettings s{90,1,1.5,1,0,1,true};s.value_weight_exponent=0;s.chosen_move_prune=0;return s;}
}
int main(){try {
    // Equal scores retain allocation order and an unexpanded tie follows action order.
    Branch state;auto s=settings();s.simulations=1;Search first(s,90);check(first.run(state,0).visits==std::vector<int64_t>({1,0,0}),"source deterministic tie");
    // FPU cache survives clearing the tree, is refreshed between searches, and reset clears it.
    s=settings();s.graph_search=true;s.use_eval_cache=true;s.eval_cache_min_visits=2;Probe cache(s,4);
    cache.run(state,0);check(cache.cached()>1,"cache records searched subtrees");
    SearchRun fresh;fresh.clear_before_search=true;fresh.max_playouts=1;cache.run(state,0,fresh);
    near(cache.cached_root_value(),0,"root retrieves previous immutable eval entry");cache.reset(4);check(cache.cached()==0,"cache clear on new game");
    // Cap freezes the original child fractions, while completed visits continue to grow.
    s=settings();s.visit_cap_contempt=4;s.visit_cap_contempt_player=1;Probe capped(s,7);
    SearchRun limits;limits.max_visits=4;auto at_cap=capped.run(state,0,limits);limits.max_visits=91;auto later=capped.run(state,0,limits);
    for(int i=0;i<3;++i)check(std::abs(later.visits[i]-30*at_cap.visits[i])<=1,"visit cap follows frozen fractions");
    // Futility uses node visits, edge visits, virtual weight, and remaining budget.
    ValueStats parent{101,0,0,.5,101,101};std::mt19937_64 random(1);s=settings();s.futile_visits_threshold=.5;
    near(root_explore_score(.2,.1,5,0,5,5,100,80,parent,s,false,false,10,random),-1e40,"futile existing child");
    check(root_explore_score(.2,.1,5,0,5,50,100,80,parent,s,false,false,10,random)>-1e30,"graph node visits can catch up instantly");
    s.futile_visits_threshold=0;s.wide_root_noise=.4;s.c_puct=0;int changed=0;double sum=0;
    for(int i=0;i<30000;++i){double q=root_explore_score(.2,0,0,0,0,0,100,80,parent,s,false,false,100,random);changed+=q>0;sum+=q;}
    check(std::abs(changed/30000.0-.5)<.015,"wide root noise Bernoulli gate");check(std::abs(sum/30000.0-.2*std::sqrt(2/std::acos(-1)))<.008,"wide root half-normal mean");
    // Search-visible history increases with depth; game state and legal actions are unchanged.
    std::vector<std::pair<int,int>> history;state.history=&history;s=settings();s.ignore_pre_root_history=true;s.simulations=10;
    Search masked(s,7);masked.run(state,0);for(auto [depth,visible]:history)check(depth==visible,"pre-root history limit");
    history.clear();s.ignore_all_history=true;Search none(s,7);none.run(state,0);for(auto [depth,visible]:history)check(visible==0,"ignore all history");state.history=nullptr;
    // Restrictions affect the player/depth requested, and an all-forbidden node terminates each playout.
    s=settings();Search constrained(s,7);SearchRun avoid;avoid.avoid_until[0]={1,0,1};auto only=constrained.run(state,0,avoid);
    check(only.visits[0]==0&&only.visits[2]==0&&only.visits[1]==90,"root allow subset");
    avoid.avoid_until[0]={1,1,1};auto blocked=constrained.run(state,0,avoid);check(blocked.action==-1&&blocked.new_playouts==91,"all forbidden root is finite");
    // Focus adds child work without root edge weight or counted playouts.
    Search focused(settings(),17);focused.set_root_focus({2},{1},.8);auto result=focused.run(state,0);
    check(result.new_playouts==91&&result.root_visits==91,"focus doesn't consume counted playout budget");
    check(result.q_visits[2]>result.visits[2],"focus is weightless at the root");check(focused.pending()==0,"focus releases virtual losses");
    // Source Go symmetry representatives use the upper-right point for Black; pass is its own orbit.
    Game go(5,7,Rule::GO);auto reps=go.root_symmetry_representatives();check(reps[0]==4&&reps[4]==4&&reps[49]==49,"source Go root representatives");
    Game gomoku(5,7,Rule::FREESTYLE);auto gomoku_reps=gomoku.root_symmetry_representatives();check(gomoku_reps[0]==4,"Gomoku canvas-independent symmetry");
    Game hex(5,7,Rule::HEX);auto hex_reps=hex.root_symmetry_representatives();check(hex_reps[0]==32&&hex_reps[4]==4,"Hex excludes transpose from same-color group");
    Game xiangqi(10,10,Rule::XIANGQI);auto xiangqi_reps=xiangqi.root_symmetry_representatives();check(std::count_if(xiangqi_reps.begin(),xiangqi_reps.end(),[i=0](int x)mutable{return x!=i++;})>0,"Xiangqi reflection group");
    // Secondary exploration updates its descendants while weightless edges stay out of the root.
    Game empty(5,6,Rule::FREESTYLE);Flat primary;auto human=std::make_shared<Flat>();human->preferred=2;primary.secondary=human;
    s=settings();s.reuse_tree=false;s.human_sl_root_weightless=1;s.human_sl_chosen_move_proportion=1;
    Search human_search(primary,s,22);auto human_result=human_search.run(empty,0);
    check(human_result.new_playouts==91&&human_result.root_visits==1,"HumanSL weightless still consumes playout budget");
    check(human_result.visits[2]==0&&human_result.q_visits[2]>0&&human_result.action==2,"HumanSL descendants and final policy mixture");
    s.human_sl_chosen_move_proportion=0;Search weightless_only(primary,s,22);auto raw_choice=weightless_only.run(empty,0);
    check(raw_choice.policy[2]==1&&raw_choice.action==2,"all weightless children fall back to policy over allocated children");
    s.human_sl_root_weightless=0;s.human_sl_root_weightful=1;Search weighted(primary,s,22);
    auto weighted_result=weighted.run(empty,0);check(weighted_result.root_visits==91&&weighted_result.visits[2]>80,"HumanSL weightful normal backup");
    // Pattern penalties change utility while WDL remains the network's pure outcome estimate.
    auto table=std::make_shared<PatternBonusTable>();std::unordered_set<std::string> seen;
    table->add(empty,0,.5,true,seen);check(table->get(empty.pattern_key(0))==.5,"external Black penalty favors White");
    Flat neutral;s=settings();s.simulations=1;s.external_patterns=table;Search patterned(neutral,s,2);
    auto pr=patterned.run(empty,0);check(pr.value==0&&pr.q_values[0]==0,"pattern utility cannot contaminate WDL targets");
    bool penalized=false;for(const auto& n:patterned.inspect_graph())if(!n.root&&n.stats.value>.49&&n.stats.win_loss==0)penalized=true;
    check(penalized,"pattern applied on child from parent board");
    check(go.pattern_key(go.canvas_area()).empty(),"Go pass has no local pattern bonus");
    auto file=std::filesystem::temp_directory_path()/"etazero-alignment.patterns";
    {std::ofstream out(file);out<<R"({"size":5,"rule":"freestyle","black":"model","white":"other","moves":[0,1],"comments":["","%SKIP%"]})"<<'\n';}
    auto corpus=PatternBonusTable::load("[{\"paths\":[\""+file.string()+"\"],\"penalty\":0.5,\"players\":[\"model\"]}]",6,{});
    near(corpus->get(empty.pattern_key(0)),.5,"external corpus player filter and dedup");std::filesystem::remove(file);
    // Full Gumbel focus must not replace its focused choice with nonroot selection.
    for(bool full:{false,true}){s=settings();s.reuse_tree=false;GumbelSettings gs;std::unique_ptr<Search> focused_gumbel;
        if(full)focused_gumbel=std::make_unique<GumbelSearch<true>>(neutral,s,gs,7);else focused_gumbel=std::make_unique<GumbelSearch<false>>(neutral,s,gs,7);
        focused_gumbel->set_root_focus({2},{1},.8);auto r=full?static_cast<GumbelSearch<true>*>(focused_gumbel.get())->run(state,0):static_cast<GumbelSearch<false>*>(focused_gumbel.get())->run(state,0);
        check(r.new_playouts==91&&r.q_visits[2]>r.visits[2],"Gumbel root focus is weightless and budget-neutral");
        focused_gumbel->advance(r.action);r=full?static_cast<GumbelSearch<true>*>(focused_gumbel.get())->run(state,0):static_cast<GumbelSearch<false>*>(focused_gumbel.get())->run(state,0);
        for(size_t i=0;i<r.visits.size();++i)check(r.q_visits[i]==r.visits[i],"real move clears Gumbel root focus");}
    // Large caps retain their exact int64 values; cancellation avoids performing large work.
    s=settings();s.max_visits=int64_t(1)<<40;s.max_playouts=int64_t(1)<<40;Search large(s,7);int polls=0;SearchRun cancel;cancel.should_stop=[&]{return ++polls>10;};auto huge=large.run(state,0,cancel);check(huge.stopped_early&&huge.new_playouts<20,"64-bit cap cancellation");
    // Hand-computed absolute/Fischer allocation and last-period lag buffering.
    double minimum,recommended,maximum;auto absolute=TimeControls::absoluteTime(1000);absolute.getTime(361,0,0,minimum,recommended,maximum);
    near(recommended,1000/((.95*361+20)*.5),"absolute allocation");near(maximum,200,"absolute time ceiling");
    auto fischer=TimeControls::fischerTime(1000,2);fischer.getTime(361,0,0,minimum,recommended,maximum);
    near(recommended,2+998/((.75*361+15)*.5),"Fischer allocation");
    auto byo=TimeControls::canadianOrByoYomiTime(0,10,1,1);byo.inOvertime=true;byo.numStonesLeftInPeriod=1;byo.timeLeftInPeriod=10;
    byo.getTime(361,120,.5,minimum,recommended,maximum);near(minimum,9,"last period doubles lag buffer");near(maximum,9,"last period cap");
    near(SearchTimeBudget::estimate_visits_left(100,.5,1,.5,4),53,"carried time visits estimate");
    s=settings();s.chosen_move_subtract=2;s.chosen_move_prune=2;auto raw_policy=direct_policy_selection({.9,.1},s);
    near(raw_policy[0],(.9-.9/64)/(1-2*.9/64),"raw NN fallback uses source selection subtract/prune");
    std::cout<<"Alignment search branches passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
