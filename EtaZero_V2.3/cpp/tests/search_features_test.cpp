#include "etazero/record.h"
#include "etazero/search_limits.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
using namespace etazero;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void near(double a,double b,const char* message){check(std::abs(a-b)<1e-10,message);}
struct TwoMoveState : SearchState {
    int action=-1;bool lcb_case=false;
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<TwoMoveState>(*this);}
    int actions() const override{return 2;}
    bool legal(int a) const override{return action<0 && a>=0 && a<2;}
    bool terminal() const override{return action>=0;}
    double terminal_value() const override{return lcb_case?(action==0?0:1):(action==0?0.8:-1);}
    Evaluation evaluate() const override{return {{std::log(lcb_case?0.98:0.8),std::log(lcb_case?0.02:0.2)},{0.2,0.3,0.5}};}
    Transition move(int a) override{action=a;return {0,1,1};}
};
struct LineState : SearchState {
    int depth=0;mutable int* calls;
    explicit LineState(int& c):calls(&c){}
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<LineState>(*this);}
    int actions() const override{return 1;}
    bool legal(int a) const override{return a==0 && !terminal();}
    bool terminal() const override{return depth>=20;}
    double terminal_value() const override{return 0;}
    Evaluation evaluate() const override{++*calls;return {{0},{0.7,0.2,0.1}};}
    Transition move(int) override{++depth;return {0,1,-1};}
};
struct SymmetryState : TwoMoveState {
    std::vector<int>* seen;
    explicit SymmetryState(std::vector<int>& s):seen(&s){}
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<SymmetryState>(*this);}
    int symmetry_count() const override{return 8;}
    Evaluation evaluate_symmetry(int symmetry) const override {
        seen->push_back(symmetry);
        return {symmetry%2?std::vector<double>{0,std::log(3)}:std::vector<double>{std::log(9),0},
                {0.1+0.1*symmetry,0.1,0.8-0.1*symmetry}};
    }
};
struct SymmetryLineState : LineState {
    std::vector<std::pair<int,int>>* seen;
    SymmetryLineState(int& count,std::vector<std::pair<int,int>>& s):LineState(count),seen(&s){}
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<SymmetryLineState>(*this);}
    int symmetry_count() const override{return 8;}
    Evaluation evaluate_symmetry(int symmetry) const override {
        ++*calls;seen->push_back({depth,symmetry});
        return {{0},{0.1+0.1*symmetry,0.1,0.8-0.1*symmetry}};
    }
};
struct CoordinateEvaluator : Evaluator {
    std::vector<float> input;
    Evaluation evaluate(const std::vector<float>& obs) override {
        input=obs;std::vector<double> logits((obs.size()-GLOBAL_FEATURES)/INPUT_PLANES);
        for(size_t i=0;i<logits.size();++i)logits[i]=i/100.0;
        return {logits,{0.2,0.3,0.5}};
    }
};
struct CoordinateBackend : Backend {
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override {
        std::vector<Evaluation> result;
        for(const auto* input:inputs) {
            std::vector<double> logits((input->size()-GLOBAL_FEATURES)/INPUT_PLANES);
            std::iota(logits.begin(),logits.end(),0.0);
            result.push_back({std::move(logits),{0.2,0.3,0.5}});
        }
        return result;
    }
};
struct DeepTemperatureState : SearchState {
    int depth=0;std::vector<int>* chosen;
    explicit DeepTemperatureState(std::vector<int>& c):chosen(&c){}
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<DeepTemperatureState>(*this);}
    int actions() const override{return 2;}
    bool legal(int a) const override{return !terminal() && a>=0 && a<(depth==0?1:2);}
    bool terminal() const override{return depth==2;}
    double terminal_value() const override{return 0;}
    Evaluation evaluate() const override{return {{std::log(9),0},{0,1,0}};}
    Transition move(int a) override{if(depth==1)chosen->push_back(a);++depth;return {0,1,-1};}
};
}
int main(){
    try {
        SelfplaySearchConfig caps{101,50,0.75,0,true,true,0.9,3,20,0.1};
        auto full_limits=selfplay_search_limits(caps,{1,1},false);
        check(full_limits.search.max_visits==101 && full_limits.target_weight==1,
              "Reduce Visits waits for the complete historical window");
        auto boundary_limits=selfplay_search_limits(caps,{0.9,1,1},false);
        check(boundary_limits.search.max_visits==101 && boundary_limits.target_weight==1,
              "The least extreme value must exceed the threshold strictly");
        auto reduced_limits=selfplay_search_limits(caps,{0.99,0.95,0.98},false);
        check(reduced_limits.search.max_visits==81,"Quarter reduction interpolates 101v towards 20v then rounds");
        near(reduced_limits.target_weight,0.775,"Reduce Visits interpolates base sample weight using the same squared fraction");
        check(!reduced_limits.cheap_search && reduced_limits.search.clear_before_search && !reduced_limits.search.remove_root_noise,
              "Reduced full searches keep full-search tree and exploration settings");
        auto negative_limits=selfplay_search_limits(caps,{-0.99,-0.95,-0.98},false);
        check(negative_limits.search.max_visits==81,"Stable wins and stable losses get the same reduction");
        for(const auto& history:std::vector<std::vector<double>>{{1,-1,1},{1,0.8,1},{1,1,0}}) {
            auto limits=selfplay_search_limits(caps,history,false);
            check(limits.search.max_visits==101 && limits.target_weight==1,
                  "Alternating winners, one uncertain move or a draw prevents reduction");
        }
        auto extreme_limits=selfplay_search_limits(caps,{-0.2,1,1,1},false);
        check(extreme_limits.search.max_visits==20,"Only the most recent lookback values affect reduction");
        near(extreme_limits.target_weight,0.1,"Extreme certainty reaches configured minimum weight");
        auto cheap_limits=selfplay_search_limits(caps,{1,1,1},true);
        check(cheap_limits.cheap_search && cheap_limits.search.max_visits==50 && cheap_limits.target_weight==0 &&
              !cheap_limits.search.clear_before_search && cheap_limits.search.remove_root_noise,
              "PCR takes priority over Reduce Visits and preserves zero-weight cheap trees");
        caps.cheap_target_weight=0.4;cheap_limits=selfplay_search_limits(caps,{1,1,1},true);
        near(cheap_limits.target_weight,0.4,"PCR weight is not multiplied by Reduce Visits weight");
        check(cheap_limits.search.max_visits==50 && cheap_limits.search.clear_before_search && !cheap_limits.search.remove_root_noise,
              "Positive-weight cheap searches keep ordinary root exploration and clearing");
        caps.reduced_visits_weight=0;auto zero_reduced=selfplay_search_limits(caps,{1,1,1},false);
        check(zero_reduced.target_weight==0 && !zero_reduced.cheap_search && zero_reduced.search.clear_before_search &&
              !zero_reduced.search.remove_root_noise,"Zero-weight reduced full search never becomes an unrecorded cheap search");
        caps.clear_before_search=false;
        check(!selfplay_search_limits(caps,{1,1,1},false).search.clear_before_search,
              "Reduce Visits respects an explicitly disabled full-search clear");
        caps.reduce_visits=false;
        check(selfplay_search_limits(caps,{1,1,1},false).search.max_visits==101,
              "Reduce Visits can be disabled independently of PCR");
        caps.reduce_visits=true;caps.reduce_threshold=0;caps.full_visits=20;caps.reduced_visits_min=2;caps.cheap_visits=5;
        check(selfplay_search_limits(caps,{0.5,0.5,0.5},false).search.max_visits==16,
              "Positive half-integer budgets round away from zero as KataGo");
        caps.reduced_visits_min=10;
        check(selfplay_search_limits(caps,{1,1,1},true).search.max_visits==5,
              "Reduced minimum need not be below the independent PCR cheap cap");
        near(temperature_at_turn(0.75,0.15,19,0,225),0.75,"Move schedule starts at early temperature");
        near(temperature_at_turn(0.75,0.15,19,15,225),0.45,"Temperature half-life scales with board width");
        near(temperature_at_turn(0.75,0.15,19,19,361),0.45,"19x19 temperature half-life");
        auto tempered=temperature_distribution({9,1,0},2);
        near(tempered[0],0.75,"Stable power temperature");near(tempered[1],0.25,"Zero weights stay zero");
        check(temperature_distribution({1,1},0)==std::vector<double>({1,0}),"Zero temperature chooses first maximum as KataGo");
        near(policy_temperature_distribution({1,1},1e-8)[0],0.5,"Tiny policy temperature preserves equal priors");
        auto protected_moves=temperature_distribution({8,6,1},2,0.2);
        near(protected_moves[0]/protected_moves[1],4.0/3,"Temperature preserves ratios above protected probability");
        near(value_weight_cdf(-3),0.028841595095129614,"Independent scipy t(3) table CDF");
        near(value_weight_cdf(1),0.8044342587133926,"Value weighting retains source interpolation grid");
        SearchSettings math{8,1,1.5,3,0,6.75,false};
        std::vector<RootChildStats> weighted{{0.5,2,1,4,4,4,0.8},{0.5,-2,1,4,4,4,0.8}};
        auto aggregate=aggregate_values({0.1,0.2,0.7},weighted,math,false);
        near(aggregate.value,0.06642401486707425,"Independent weighted-subtree mean");
        near(aggregate.weight_sq,9.717381745200182,"Weight square tracks normalized subtree reweighting");
        near(aggregate.weight,9,"Value reweighting preserves total sample weight");near(aggregate.draw,0.2,"Draw mass uses same weights as value");
        math.value_weight_exponent=0;auto unit=aggregate_values({0.1,0.2,0.7},weighted,math,false);
        near(unit.value,-0.6/9,"Exponent zero is unit-weight ablation");near(unit.weight_sq,9,"Unit-weight ESS");
        ValueStats parent{11,0,0,1,11,11};
        near(child_selection_score(0.2,0.4,2,1,10,parent,math,false),-0.44+1.5*std::sqrt(10.01)*0.2/6,
             "Virtual loss is sample weight, including PUCT denominator");
        math.forced_playouts=2;
        check(child_selection_score(0.5,0,1,0,9,parent,math,true)==1e20,"Forced playout catches deficient completed weight");
        check(child_selection_score(0.5,0,1,1,9,parent,math,true)<1e20,"Virtual-loss weight participates in forced quota");
        SearchSettings variance{8,1,1,1,0,1,false};variance.c_puct_log=0.45;
        variance.c_puct_stdev_prior=0.4;variance.c_puct_stdev_prior_weight=2;variance.c_puct_stdev_scale=0.85;
        near(explore_scaling(100,{1,0,0,1,1,1},variance),(1+0.45*std::log(1.2))*std::sqrt(100.01),
             "Low-weight parent uses the exact prior stdev, leaving scale factor one");
        // Constant Q=1/2, nine samples: ((.25+.16)*2+.25*9)/10-.25 = .057.
        near(explore_scaling(100,{9,0.5,0.25,0,9,9},variance),
             (1+0.45*std::log(1.2))*std::sqrt(100.01)*(0.15+0.85*std::sqrt(0.057)/0.4),
             "Match variance uses prior weight two and the weight-minus-one denominator");
        near(explore_scaling(100,{9,0.5,0.20,0,9,9},variance),explore_scaling(100,{9,0.5,0.25,0,9,9},variance),
             "Inconsistent concurrent second moment clamps to mean square");
        // This boundary previously pruned to 3; source's +0.01 must retain 4.
        math.policy_target_pruning=true;
        double old_scale=1.5*std::sqrt(20.0),q1=0.5+old_scale*0.5/11-old_scale*0.5/(4-1e-5);
        auto boundary=root_selection_weights({{0.5,5,2.5,10,10,10},{0.5,10*q1,10*q1*q1,10,10,10}},math,false);
        near(boundary[1]/boundary[0],4.0/10,"KataGo exploration offset survives inverse PUCT ceil boundary");
        // Symmetry averaging is over individually normalized probabilities, after NN temperature.
        std::vector<int> seen;SymmetryState sym_state(seen);SearchSettings ensemble{1,1,1.5,1,0,6.75,false};
        ensemble.root_symmetries=8;ensemble.nn_policy_temperature=2;
        ensemble.root_policy_temperature=ensemble.root_policy_temperature_early=2;
        Search sym_search(ensemble,17);auto sym_result=sym_search.run(sym_state,0,{2,true,true,false});
        auto sorted=seen;std::sort(sorted.begin(),sorted.end());check(sorted==std::vector<int>({0,1,2,3,4,5,6,7}),"Root samples D4 without replacement");
        near(sym_result.network_wdl[0],0.45,"Root averages WDL probabilities");near(sym_result.network_wdl[1],0.1,"Root retains draw probability");
        double average=(0.75+1/(1+std::sqrt(3.0)))/2;
        double root_p=std::sqrt(average)/(std::sqrt(average)+std::sqrt(1-average));
        near(sym_result.policy_surprise,-std::log(root_p),"NN temperature then probability ensemble then root temperature");
        seen.clear();auto sym_cheap=sym_search.run(sym_state,0,{2,true,true,true});
        check(seen==std::vector<int>({0}),"Cheap search disables multi-symmetry evaluation");
        near(sym_cheap.policy_surprise,-std::log(0.75),"Cheap search retains NN temperature but disables root temperature");
        seen.clear();ensemble.root_symmetries=4;Search sampled(ensemble,17);sampled.run(sym_state,0,{2,true,true,false});
        sorted=seen;std::sort(sorted.begin(),sorted.end());check(sorted.size()==4 && std::adjacent_find(sorted.begin(),sorted.end())==sorted.end(),"Partial ensemble uses distinct symmetries");
        CoordinateEvaluator coordinates;Game padded(5,6,Rule::RENJU);padded.play(1);padded.play(6);
        AlphaZeroState real_state(padded,coordinates);
        auto reflected=real_state.evaluate_symmetry(5);
        near(reflected.logits[0],0.05,"Policy restored from reflected canvas coordinates");
        check(coordinates.input[36+4]==1 && coordinates.input[72+11]==1 && coordinates.input[0]==0 && coordinates.input[1]==1,
              "D4 moves stones and board mask across full padded canvas");
        auto rotated=real_state.evaluate_symmetry(1);near(rotated.logits[1],0.24,"Inverse 90-degree action mapping");
        check(coordinates.input[5*36+2]==-1 && coordinates.input[5*36+3]==1,"D4 preserves rule globals");
        std::vector<int> cold_actions,hot_actions;DeepTemperatureState cold_state(cold_actions),hot_state(hot_actions);
        SearchSettings deep{8,1,1.5,1,0,6.75,false};Search cold(deep,42);cold.run(cold_state,0);
        deep.nn_policy_temperature=2;Search hot(deep,42);hot.run(hot_state,0);
        check(std::count(cold_actions.begin(),cold_actions.end(),1)==0 && std::count(hot_actions.begin(),hot_actions.end(),1)>0,
              "NN policy temperature applies below root");
        int sym_calls=0;std::vector<std::pair<int,int>> orientations;SymmetryLineState sym_line(sym_calls,orientations);
        SearchSettings reuse_ensemble{8,1,1.5,1,0,6.75,true};reuse_ensemble.root_symmetries=8;
        Search promoted(reuse_ensemble,7);promoted.run(sym_line,0,{9,true,true,false});
        promoted.advance(0);sym_line.move(0);int before_calls=sym_calls;
        auto kept=promoted.run(sym_line,0,{3,true,false,true});
        check(sym_calls==before_calls && kept.simulations==0 && kept.root_visits==8,"Cheap retained root adds neither visits nor symmetry evaluations");
        near(kept.network_wdl[0],0.1,"Cheap promoted root retains its non-root NN evaluation");
        auto refreshed=promoted.run(sym_line,0,{3,true,false,false});
        check(sym_calls==before_calls+8 && refreshed.simulations==0 && refreshed.root_visits==8,"Normal promoted root refreshes ensemble without consuming visits");
        near(refreshed.network_wdl[0],0.45,"Promoted ensemble replaces single-orientation root WDL");
        near(fpu_value(0.6,-0.2,0.25,2,0.2),0.45,"FPU visited-policy interpolation and reduction");
        near(fpu_value(0.6,-0.2,0,2,0.2),0.6,"FPU starts at network value");
        near(fpu_value(0.6,-0.2,1,2,0.2),-0.4,"FPU ends at parent minus reduction");
        near(fpu_value(0.6,-0.2,0.25,2,0.2,false,0),-0.3,"Alternative FPU starts at parent, including reduction");
        near(fpu_value(0.6,-0.2,0.25,2,0.2,false,0.75),0.3,"Source fpuParentWeight is the NN coefficient");
        near(fpu_value(0.6,-0.2,0,0,0),-0.2,"Zero policy power selects parent even at zero visited mass");
        // Explicit orientations restore coordinates; a cache hit reuses the first
        // canonical output, and ensemble requests neither read nor replace it.
        std::vector<std::unique_ptr<Backend>> backends;backends.push_back(std::make_unique<CoordinateBackend>());
        BatchEvaluator cached(std::move(backends),"coordinate",6,8,32,0,1024);
        auto canonical=padded.observation();
        auto first=cached.evaluate_symmetry(canonical,5);
        auto hit=cached.evaluate_symmetry(canonical,1);
        near(first.logits[0],5,"Cache miss inverse-maps horizontal reflection");
        check(hit.logits==first.logits && cached.requests==1 && cached.cache_hits==1,"Orientation does not split canonical cache keys");
        auto bypass=cached.evaluate_symmetry(canonical,1,true);
        near(bypass.logits[0],30,"Bypass evaluates requested rotation independently");
        check(cached.evaluate_symmetry(canonical,2).logits==first.logits && cached.requests==2,"Bypass leaves the first cached orientation intact");
        cached.evaluate_symmetry(canonical,1,false,2);
        check(cached.requests==3,"NN policy temperature is an input condition in the cache key");
        auto changed_globals=canonical;changed_globals.back()+=1;
        cached.evaluate_symmetry(changed_globals,1);
        check(cached.requests==4,"Rule/PDA globals split cache input conditions");
        SearchSettings ensemble_cache{1,1,1,1,0,1,true};ensemble_cache.root_symmetries=8;
        Search cached_search(cached,ensemble_cache,1);SearchRun root_only;root_only.max_visits=1;
        auto cache_ensemble=cached_search.run(padded,0,root_only);
        check(cache_ensemble.root_visits==1 && cache_ensemble.new_playouts==1 && cached.requests==12,"Eight root NN rows consume one fresh-root playout");
        auto ensemble_refresh=cached_search.run(padded,0,root_only);
        check(ensemble_refresh.initial_visits==1 && ensemble_refresh.new_playouts==0 && cached.requests==20,"Reused root ensemble consumes NN rows but no playouts");
        check(cached.evaluate_symmetry(canonical,3).logits==first.logits && cached.requests==20,"Root ensemble never overwrites single-output cache");
        cached.finish();
        std::vector<std::unique_ptr<Backend>> r1,r2;
        r1.push_back(std::make_unique<CoordinateBackend>());r2.push_back(std::make_unique<CoordinateBackend>());
        BatchEvaluator random_cached(std::move(r1),"random",6,8,32,0,1024,true,0,47);
        BatchEvaluator random_control(std::move(r2),"random",6,8,32,0,1024,true,0,47);
        auto random_first=random_cached.evaluate(canonical);
        check(random_first.logits==random_control.evaluate(canonical).logits,"NN RNG is reproducible with fixed request order");
        for(int i=0;i<10;++i)check(random_cached.evaluate(canonical).logits==random_first.logits,"Random cache hits reuse original output");
        check(random_cached.evaluate(changed_globals).logits==random_control.evaluate(changed_globals).logits,
              "Cache hits do not consume the NN orientation random stream");
        auto specified_output=random_cached.evaluate_symmetry(canonical,5,true);
        near(specified_output.logits[0],5,"Explicit orientation overrides evaluator randomization");
        random_cached.finish();random_control.finish();
        seen.clear();SearchSettings reset_priors{1,1,1,1,0,1,true};
        reset_priors.root_policy_temperature_early=2;reset_priors.root_policy_temperature=1;
        Search prior_reset(reset_priors,1);auto early_root=prior_reset.run(sym_state,0,root_only);
        near(early_root.search_policy[0],0.75,"Early root temperature applies to raw 9:1 policy");
        root_only.turn=19;root_only.board_area=361;
        auto later_root=prior_reset.run(sym_state,0,root_only);
        near(later_root.network_policy[0],0.9,"Reused root preserves original NN prior");
        near(later_root.search_policy[0],std::pow(9,2.0/3)/(std::pow(9,2.0/3)+1),
             "Root temperature restarts from raw prior rather than compounding prior exploration");
        // Random single orientations reach both cheap roots and non-root leaves.
        orientations.clear();sym_calls=0;SymmetryLineState random_line(sym_calls,orientations);
        SearchSettings randomized{6,1,1,1,0,1,false};randomized.nn_randomize=true;randomized.root_symmetries=4;
        Search random_search(randomized,42);random_search.run(random_line,0,{7,true,true,true});
        check(orientations.size()==7 && orientations.front().first==0,"Cheap root samples one orientation, leaves also sample one");
        check(std::any_of(orientations.begin()+1,orientations.end(),[](auto x){return x.second!=0;}),"Leaf randomization is active");
        std::vector<int> sampled_roots;
        for(uint64_t seed=0;seed<64;++seed) {
            random_search.reset(seed);orientations.clear();SearchRun one;one.max_visits=1;one.remove_root_noise=true;
            random_search.run(random_line,0,one);sampled_roots.push_back(orientations.front().second);
        }
        std::sort(sampled_roots.begin(),sampled_roots.end());
        check(std::unique(sampled_roots.begin(),sampled_roots.end())-sampled_roots.begin()==8,"Random cheap roots cover all eight D4 orientations");
        randomized.nn_randomize=false;randomized.nn_symmetry=6;randomized.root_symmetries=1;
        Search specified(randomized,42);orientations.clear();specified.run(random_line,0,{7,true,true,true});
        check(std::all_of(orientations.begin(),orientations.end(),[](auto x){return x.second==6;}),"Explicit symmetry applies to root and leaves");
        // Independent visit and playout caps, including root initialization.
        int budget_calls=0;LineState budget_line(budget_calls);SearchSettings limits{100,1,1,1,0,1,true};
        Search budget_search(limits,1);SearchRun limited;limited.max_playouts=3;
        auto fresh_budget=budget_search.run(budget_line,0,limited);
        check(fresh_budget.initial_visits==0 && fresh_budget.root_visits==3 && fresh_budget.simulations==2 && fresh_budget.new_playouts==3,
              "Fresh root initialization counts against maxPlayouts");
        limited.max_playouts=2;auto reused_budget=budget_search.run(budget_line,0,limited);
        check(reused_budget.initial_visits==3 && reused_budget.root_visits==5 && reused_budget.new_playouts==2,"Reused root playout cap only limits new visits");
        limited.max_visits=6;limited.max_playouts=8;
        check(budget_search.run(budget_line,0,limited).new_playouts==1,"Visit cap includes retained visits and dominates larger playout cap");
        limited.max_visits=2;
        check(budget_search.run(budget_line,0,limited).new_playouts==0,"Already exceeded visit cap adds no visits");
        budget_search.reset(1);limited={};limited.max_playouts=0;budget_calls=0;
        auto no_search=budget_search.run(budget_line,0,limited);
        check(no_search.action==-1 && no_search.root_visits==0 && budget_calls==0,"Zero new-playout cap does not evaluate a fresh root");
        limited.max_playouts=1;auto prior_only=budget_search.run(budget_line,0,limited);
        check(prior_only.action==0 && prior_only.new_playouts==1 && prior_only.simulations==0,"Root-only search selects from NN prior without LCB");
        budget_search.reset(1);limited={};limited.max_time=0;
        auto fresh_time=budget_search.run(budget_line,0,limited);
        check(fresh_time.new_playouts==2 && fresh_time.root_visits==2 && fresh_time.stopped_early,"Fresh time limit allows two total playouts");
        auto reused_time=budget_search.run(budget_line,0,limited);
        check(reused_time.new_playouts==2 && reused_time.initial_visits==2,"Reused time limit allows two additional playouts");
        budget_search.reset(1);limited={};limited.should_stop=[] {return true;};budget_calls=0;
        auto interrupted=budget_search.run(budget_line,0,limited);
        check(interrupted.new_playouts==0 && interrupted.action==-1 && budget_calls==0,"Explicit stop can prevent all search");
        limits.threads=4;Search concurrent_stop(limits,1);TwoMoveState terminal_edges;
        std::atomic<int> checks{0};limited={};limited.should_stop=[&]{return checks.fetch_add(1)>=8;};
        auto parallel_stop=concurrent_stop.run(terminal_edges,0,limited);
        check(parallel_stop.stopped_early && parallel_stop.new_playouts<100 && concurrent_stop.pending()==0,
              "Explicit parallel stop completes in-flight paths and releases virtual loss");
        terminal_edges.move(1);budget_calls=0;
        auto terminal_root=budget_search.run(terminal_edges,0);
        check(terminal_root.action==-1 && terminal_root.root_visits==0 && terminal_root.new_playouts==0 && terminal_root.value==-1,"Finished Gomoku root returns exact terminal value without NN requests");
        CoordinateEvaluator replacement;CoordinateEvaluator original;
        Search model_change(original,ensemble_cache,1);model_change.run(padded,0,root_only);
        model_change.set_evaluator(replacement);
        check(model_change.run(padded,0,root_only).initial_visits==0 && !replacement.input.empty(),"Model replacement clears root stats, priors and per-thread state");
        // Exercise the new limits through real tree promotion: a reduced full
        // search clears a preceding PCR tree rather than inheriting cheap flags.
        caps={101,50,0.75,0,true,true,0.9,3,20,0.1};
        int cap_calls=0;LineState capped_line(cap_calls);
        SearchSettings cap_search_settings{100,1,1.5,1,0,6.75,true};
        Search cap_search(cap_search_settings,7);
        auto cap_full=cap_search.run(capped_line,0,selfplay_search_limits(caps,{},false).search);
        cap_search.advance(0);capped_line.move(0);
        int saved_calls=cap_calls;
        auto cap_cheap=cap_search.run(capped_line,0,selfplay_search_limits(caps,{1,1,1},true).search);
        check(cap_full.root_visits==101 && cap_cheap.root_visits==100 && cap_cheap.simulations==0 && cap_calls==saved_calls,
              "PCR retained visits can exceed the cheap cap without extra simulations");
        cap_search.advance(0);capped_line.move(0);
        auto cap_reduced=cap_search.run(capped_line,0,selfplay_search_limits(caps,{1,1,1},false).search);
        check(cap_reduced.root_visits==20 && cap_reduced.simulations==19 && cap_calls>saved_calls,
              "Reduced full search clears PCR tree and uses its own smaller cap");
        auto uniform=noise_alpha_distribution({0.3,0.3,0.3});
        for(auto a:uniform)near(a,1.0/3,"Flat shaped noise is uniform");
        auto shaped=noise_alpha_distribution({0.01,0.001,0});
        near(std::accumulate(shaped.begin(),shaped.end(),0.0),1,"Noise alpha proportions sum to one");
        near(shaped[2],1.0/6,"Shaped noise preserves half uniform mass even at zero policy");
        check(shaped[0]>shaped[1] && shaped[1]>shaped[2],"Clipped centered log-policy ordering");
        SearchSettings settings{256,1,1.5,1,0,6.75,true};settings.policy_target_pruning=true;
        auto pruned=root_selection_weights({{0.5,50,25,100,100,100},{0.5,-50,25,100,100,100}},settings,false);
        near(pruned[0],100.0/109,"Retrospective inverse PUCT pruning");near(pruned[1],9.0/109,"Ceil reduced visits");
        {
            SearchSettings fractional;fractional.c_puct=1;fractional.policy_target_pruning=true;
            // Nonpositive inverse-PUCT gaps retain the raw weight before ceil.
            // The stable edge keeps its fractional weight; only other edges round.
            for(bool zero_gap:{false,true}) {
                double prior=zero_gap?0:0.9,q=zero_gap?0:1;
                std::vector<RootChildStats> edges{{prior,0,0,10,10.2,10.404},
                                                {1-prior,1.2*q,1.2*q*q,2,1.2,0.72}};
                auto raw=root_selection_weights(edges,fractional,false,{},false);
                near(raw[0],10.2,"Pruning preserves fractional stable-edge weight");
                near(raw[1],2,"Nonpositive gap still rounds other-edge weight upward");
                auto normalized=root_selection_weights(edges,fractional,false);
                near(normalized[1],2.0/12.2,"Normalize only after rounding other edges");
                fractional.policy_target_pruning=false;
                near(root_selection_weights(edges,fractional,false,{},false)[1],1.2,
                     "Pruning ablation preserves fractional other-edge weight");
                fractional.policy_target_pruning=true;
            }
            fractional.lcb_stdevs=1;
            auto lcb=root_selection_weights({{0.9,0,0,10,10.2,10.404},{0.1,1.2,1.2,2,1.2,0.72}},fractional,true,{},false);
            check(lcb[1]>lcb[0],"Rounding before LCB makes the higher-value edge eligible");
        }
        settings.policy_target_pruning=false;settings.use_lcb=true;
        std::vector<RootChildStats> children{{0.5,20,100,100,100,100},{0.5,24,14.4,40,40,40}};
        auto before=root_selection_weights(children,settings,false),after=root_selection_weights(children,settings,true);
        check(before[0]>before[1] && after[1]>after[0],"LCB favors reliable higher-value child over raw visits");
        auto correlated=children;correlated[1].weight_sq=1600;
        auto low_ess=root_selection_weights(correlated,settings,true);
        check(low_ess[0]>low_ess[1],"LCB uses weighted ESS rather than raw visit count");
        settings.min_lcb_visit_prop=0.5;
        auto excluded=root_selection_weights(children,settings,true);
        near(excluded[0],before[0],"LCB eligibility uses pre-pruning stable-child visits");
        settings.use_lcb=false;settings.min_lcb_visit_prop=0.15;
        TwoMoveState bandit;Search ordinary(settings,42);auto plain=ordinary.run(bandit,0,{257,true,false,false});
        settings.forced_playouts=2;Search forced(settings,42);auto extra=forced.run(bandit,0,{257,true,false,false});
        check(extra.visits[1]>plain.visits[1],"Forced playouts explore weak prior-positive root move");
        Search removed(settings,42);auto cheap=removed.run(bandit,0,{257,true,false,true});
        check(cheap.visits==plain.visits,"Unrecorded cheap search disables forced exploration");
        settings.policy_target_pruning=true;Search target(settings,42);auto reduced=target.run(bandit,0,{257,true,false,false});
        check(reduced.policy[1]<double(reduced.visits[1])/256,"Target pruning removes forced exploration bias");
        settings.forced_playouts=0;settings.policy_target_pruning=false;settings.use_lcb=true;
        TwoMoveState lcb_position;lcb_position.lcb_case=true;
        Search train_lcb(settings,42),eval_lcb(settings,42);
        auto training=train_lcb.run(lcb_position,0,{61,true,false,false});
        auto evaluation=eval_lcb.run(lcb_position,0,{61,false,false,false});
        check(training.visits==evaluation.visits,"LCB changes final selection, not PUCT visits");
        check(training.action==0 && training.policy[1]>training.policy[0],"Selfplay move omits LCB while policy target includes it");
        check(evaluation.action==1 && evaluation.move_policy==evaluation.policy,"Evaluation move includes LCB");
        // Root NN and leaves swap W/L across every alternating-player edge; draws stay fixed.
        int calls=0;LineState line(calls);Search line_search({8,1,1.5,1,0,6.75,true},7);
        auto full=line_search.run(line,0,{9,true,true,false});
        near(full.search_wdl[1],0.2,"Draw mass survives backup");
        near(full.search_wdl[0],(0.7*5+0.1*4)/9,"WDL backs up alternating player perspectives");
        line_search.advance(0);line.move(0);int previous=calls;
        auto low=line_search.run(line,0,{3,true,false,true});
        check(low.root_visits==8 && low.simulations==0 && calls==previous,"Cheap cap counts retained subtree and issues no redundant visits");
        auto cleared=line_search.run(line,0,{9,true,true,false});
        check(cleared.root_visits==9 && cleared.simulations==8 && calls>previous,"Next full search clears retained cheap-search tree");
        // Policy-only weighting: one zero-weight shallow surprise joins the sample pool.
        FinishedGame game{};game.size=5;game.winner=1;
        for(double surprise:{1.0,3.0,4.0}) {
            Step s{};s.player=1;s.policy_surprise=surprise;s.target_weight=game.steps.size()==2?0:1;
            s.network_wdl={1,0,0};s.search_wdl={1,0,0};game.steps.push_back(s);
        }
        Step prefix{};prefix.trainable=false;game.steps.insert(game.steps.begin(),prefix);
        std::mt19937_64 rng(3);apply_training_weights(game,0.5,0,rng);
        near(game.steps[1].target_weight,0.7,"Per-game policy surprise redistribution");
        near(game.steps[2].target_weight,1.1,"Weights can exceed one");
        near(game.steps[3].target_weight,0.2,"Surprising cheap rows regain training weight");
        check(game.steps[0].row_repeats==0 && game.steps[0].target_weight==0,"Opening prefix never gains surprise weight");
        double weight=0;for(auto& s:game.steps)weight+=s.target_weight;near(weight,2,"Redistribution conserves expected rows");
        FinishedGame reduced_game{};reduced_game.size=5;reduced_game.winner=1;
        for(double base_weight:{1.0,0.1,0.0}) {
            Step s{};s.player=1;s.target_weight=base_weight;s.policy_surprise=base_weight==0.1?4:1;
            s.network_wdl=s.search_wdl={1,0,0};reduced_game.steps.push_back(s);
        }
        apply_training_weights(reduced_game,0.5,0,rng);
        weight=0;for(auto& s:reduced_game.steps)weight+=s.target_weight;
        near(weight,1.1,"Surprise redistribution uses reduced base weights, not a full weight for every non-cheap row");
        check(reduced_game.steps[1].target_weight>0.1,"Excess policy surprise can restore weight on a reduced full search");
        FinishedGame value_game{};value_game.size=5;value_game.winner=-1;
        Step v{};v.player=-1;v.network_wdl={0.1,0.2,0.7};v.search_wdl={0.7,0.2,0.1};value_game.steps.push_back(v);
        apply_training_weights(value_game,0,0.1,rng);
        check(value_game.steps[0].value_surprise>0.5 && value_game.steps[0].value_surprise<=1,"Value surprise uses smoothed future WDL in correct player perspective");
        int copies=0;for(int i=0;i<20000;++i) {auto sampled=game;sampled.steps.erase(sampled.steps.begin());
            sampled.steps[0].target_weight=0.25;sampled.steps[1].target_weight=0;sampled.steps[2].target_weight=0;
            apply_training_weights(sampled,0,0,rng);copies+=sampled.steps[0].row_repeats;}
        check(std::abs(copies/20000.0-0.25)<0.015,"Randomized rounding matches expected sample frequency");
        std::cout<<"WDL, FPU, shaped noise, forced playouts, pruning, LCB, reuse and surprise sampling passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
