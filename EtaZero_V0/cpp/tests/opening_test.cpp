#include "etazero/opening.h"
#include "etazero/random_evaluator.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
using namespace etazero;
WDL wdl(double v) { return {(1+v)/2,0,(1-v)/2}; }
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct Constant : Evaluator {
    double v;std::vector<std::vector<float>> inputs;
    explicit Constant(double value):v(value){}
    Evaluation evaluate(const std::vector<float>& observation) override {
        inputs.push_back(observation);return {std::vector<double>((observation.size()-GLOBAL_FEATURES)/INPUT_PLANES,0),wdl(v)};
    }
};
struct Asymmetric : Evaluator {
    int calls=0,left=0,right=0;
    std::vector<float> root;
    Evaluation evaluate(const std::vector<float>& obs) override {
        double value=0;
        if(calls==0)root=obs;
        else if(calls>=2) {
            // After a real move, the mover becomes the opponent in the input.
            int action=-1;
            for(int a=0;a<49;++a)if(obs[2*49+a] && !root[49+a]) {
                check(action<0,"Candidate adds exactly one mover stone");action=a;
            }
            check(action>=0,"Candidate has a new mover stone");
            if(action%7<2)++left;
            else {++right;value=0.8;}
        }
        ++calls;return {std::vector<double>(49,0),wdl(value)};
    }
};
}
int main() {
    auto path=std::filesystem::temp_directory_path()/("etazero_opening_"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".cfg");
    try {
        {
            std::ofstream file(path);
            file<<"[agent]\nalgorithm=alphazero\nroot_search_algo=puct\nnonroot_search_algo=puct\n"
                  "[opening]\nprobability=1\navg_dist_factor=0.8\nbalance_exponent=4\n"
                  "rejection_probability=0.995\nrejection_probability_fallback=0.8\nmax_tries=20\n"
                  "policy_init=false\npolicy_after=true\npolicy_on_failure=true\npolicy_init_mean=5\npolicy_temperature=1\n";
        }
        Config config(path.string());OpeningConfig settings(config);
        for(Rule rule:{Rule::FREESTYLE,Rule::STANDARD,Rule::RENJU}) {
            for(int size:{5,6}) for(int seed=0;seed<20;++seed) {
                Game game(size,7,rule);Constant evaluator(0);std::mt19937_64 random(seed);
                auto result=initialize_opening(game,settings,evaluator,random);
                check(result.status==OpeningStatus::Success && !game.finished(),"Nonterminal balanced opening");
                check(result.balanced_moves>=1 && result.balanced_moves<=10 && result.policy_moves==0,"KataGomo no-VC skeleton count");
                Game replay(size,7,rule);
                for(int action:result.actions){check(replay.legal(action),"Prefix action is submittable");replay.play(action);}
                check(replay.observation()==game.observation(),"Prefix reconstructs initialized position");
                check(evaluator.inputs.size()>=3,"Both root perspectives and candidates are evaluated");
                const auto& own=evaluator.inputs[0];const auto& opposite=evaluator.inputs[1];
                for(int i=0;i<49;++i) {
                    check(own[49+i]==opposite[2*49+i] && own[2*49+i]==opposite[49+i],"Counterfactual perspective swaps stone planes");
                    check(own[i]==opposite[i] && own[i]==(i/7<size && i%7<size),"Counterfactual preserves sampled size");
                    check(own[3*49+i]==opposite[4*49+i] && own[4*49+i]==opposite[3*49+i],"Counterfactual swaps forbidden perspective");
                }
                const int g=INPUT_PLANES*49;
                check(own[g]==(rule==Rule::STANDARD) && own[g+1]==(rule==Rule::RENJU),"Opening uses sampled rule");
                check(own[g+2]==-opposite[g+2] && own[g+3]==(rule==Rule::RENJU),"Opening color and full forbidden flag");
                check(result.start_value==0,"Candidate value is from next player's perspective");
            }
        }
        // Independent sampling-odds check: left candidates have v=0, right v=.8.
        // At exponent 4 their weights are exactly 1 and .36^4 = .01679616.
        auto weighted=settings;weighted.rejection_probability=0;weighted.rejection_probability_fallback=0;
        double observed=0,expected=0,variance=0;
        for(int seed=0;seed<600;++seed) {
            Game game(5,7,Rule::FREESTYLE);Asymmetric evaluator;std::mt19937_64 rng(seed);
            auto prefix=initialize_opening(game,weighted,evaluator,rng);
            double p=evaluator.left/(evaluator.left+evaluator.right*0.01679616);
            observed+=prefix.actions.back()%7<2;expected+=p;variance+=p*(1-p);
        }
        check(std::abs(observed-expected)<5*std::sqrt(variance),"Balance sampling follows fourth-power weights, not uniform policy");
        settings.max_tries=1;settings.rejection_probability=1;settings.rejection_probability_fallback=0;
        Game fallback(5,5,Rule::FREESTYLE);Constant near_one(0.9);std::mt19937_64 random(1);
        auto result=initialize_opening(fallback,settings,near_one,random);
        check(result.status==OpeningStatus::Success && result.attempts>settings.max_tries+1,"KataGomo switches rejection after > max_tries");
        check(std::abs(result.start_value-0.9)<1e-12,"Balance records unnegated successor value");
        // With saturated candidates there is no admissible probability mass.
        Game stalled(5,5,Rule::FREESTYLE);Constant saturated(1);int checks=0;
        auto interrupted=initialize_opening(stalled,settings,saturated,random,[&]{return ++checks>400;});
        check(interrupted.status==OpeningStatus::Interrupted && stalled.turn()==0,"Cancelled retry leaves original game untouched");
        Game broken(5,5,Rule::FREESTYLE);Constant nan(std::numeric_limits<double>::quiet_NaN());
        auto failed=initialize_opening(broken,settings,nan,random);
        check(failed.status==OpeningStatus::Failed && !failed.failure.empty() && broken.turn()==0,"Nonfinite opening evaluation is explicit");
        settings.probability=0;
        Game disabled(5,5,Rule::FREESTYLE);Constant zero(0);
        check(initialize_opening(disabled,settings,zero,random).actions.empty() && zero.inputs.empty(),"Disabled opening makes no inference requests");
        settings.policy_init=true;settings.policy_init_mean=100;
        bool policy_terminal=false;
        for(int seed=0;seed<30 && !policy_terminal;++seed) {
            Game policy(5,5,Rule::FREESTYLE);std::mt19937_64 rng(seed);
            auto prefix=initialize_opening(policy,settings,zero,rng);
            check(prefix.balanced_moves==0 && prefix.policy_moves==policy.turn(),"Independent policy init count");
            policy_terminal=policy.finished();
        }
        check(policy_terminal,"Policy initialization can finish a complete game without search rows");
        Game position(5,7,Rule::STANDARD);auto input=position.observation();
        RandomBackend a(7,7),b(7,7),other(7,8);
        auto batch=a.evaluate({&input,&input});auto single=b.evaluate({&input})[0];
        check(batch[0].logits==batch[1].logits && batch[0].logits==single.logits && batch[0].wdl==single.wdl,"Random evaluator independent of batching");
        check(batch[0].logits!=other.evaluate({&input})[0].logits,"Random evaluator seed changes output");
        check(std::isfinite(single.value()) && std::abs(single.value())<=1,"Random value is bounded");
        std::filesystem::remove(path);
        std::cout<<"Random evaluator, KataGomo opening, perspectives, retries and policy init passed\n";
    } catch(const std::exception& error){std::filesystem::remove(path);std::cerr<<error.what()<<'\n';return 1;}
}
