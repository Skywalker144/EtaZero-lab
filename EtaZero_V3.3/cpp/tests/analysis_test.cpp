#include "search/muzero/gumbel_search.h"
#include "search/gumbel_search.h"
#include <cmath>
#include <iostream>
using namespace etazero;
void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
struct Outcomes : SearchState {
    int chosen=-1;
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<Outcomes>(*this);}
    int actions() const override{return 3;}
    bool legal(int a) const override{return chosen<0&&a>=0&&a<3;}
    bool terminal() const override{return chosen>=0;}
    double terminal_value() const override{return chosen==0?-1:chosen==1?0:1;}
    Evaluation evaluate(SearchEvaluation={}) const override{return {{0,-.2,-.7},{.6,.2,.2}};}
    int move(int a) override{chosen=a;return -1;}
};
void compare(const SearchResult& a,const SearchResult& b) {
    check(a.action==b.action&&a.visits==b.visits&&a.policy==b.policy&&a.move_policy==b.move_policy&&a.policy_target==b.policy_target&&a.search_wdl==b.search_wdl,"Diagnostics changed sampling/targets");
    check(a.analysis.empty()&&!b.analysis.empty(),"Diagnostics must be opt-in");
    double raw=0;for(const auto& c:b.analysis)raw+=c.selection.weight;
    for(const auto& c:b.analysis) {
        check(c.pv.size()>0&&c.pv[0]==c.action,"PV starts at candidate");
        if(!c.has_value)continue;
        check(std::abs(c.wdl[0]+c.wdl[1]+c.wdl[2]-1)<1e-12,"WDL normalized");
        check(std::abs(c.win_loss-(c.wdl[0]-c.wdl[2]))<1e-12,"Candidate value perspective");
        if(!b.gumbel){check(c.selection.has_lcb,"PUCT LCB available");check(std::abs(c.selection.weight/raw-b.move_policy[c.action])<1e-12,"Raw weight matches actual move policy");}
        else check(!c.selection.has_lcb,"Gumbel does not use LCB");
    }
}
struct Uniform : Evaluator {Evaluation evaluate(const std::vector<float>&) override{return {std::vector<double>(36,0),{.6,.2,.2}};}};
struct Latent : muzero::Latent {int depth=0;};
struct AnalysisBackend : muzero::Backend {
    muzero::InferenceOutput output(int depth) {
        auto state=std::make_shared<Latent>();state->depth=depth;
        return {state,{std::vector<double>(36,0),depth%2?WDL{.2,.3,.5}:WDL{.5,.3,.2}}};
    }
    std::vector<muzero::InferenceOutput> initial(const InferenceInputs& in) override{return std::vector<muzero::InferenceOutput>(in.size(),output(0));}
    std::vector<muzero::InferenceOutput> recurrent(const std::vector<muzero::InferenceAction>& in) override {
        std::vector<muzero::InferenceOutput> out;for(const auto& x:in)out.push_back(output(std::dynamic_pointer_cast<const Latent>(x.latent)->depth+1));return out;
    }
};
int main() {
    SearchSettings s{49,1,1,1,0,1,false};s.use_lcb=true;s.value_weight_exponent=0;s.chosen_move_prune=0;
    std::vector<RootChildStats> children{{.5,2,3,4,4,4,0},{.5,0,2,4,4,4,0}};
    std::vector<RootSelectionDiagnostic> d;auto weights=root_selection_weights(children,s,true,{},false,&d);
    double radius=s.lcb_stdevs*std::sqrt((67.0/130)/(169.0/41));
    check(std::abs(d[0].radius-radius)<1e-12&&std::abs(d[0].lcb-(.5-radius))<1e-12,"Independent weighted LCB calculation");
    check(weights==root_selection_weights(children,s,true,{},false),"Selection diagnostics are observational");
    Uniform evaluator;Outcomes position;GumbelSettings g;g.noise_scale=.4;
    Search p1(s,17),p2(s,17);GumbelSearch<false> gp1(evaluator,s,g,17),gp2(evaluator,s,g,17);
    GumbelSearch<true> gg1(evaluator,s,g,17),gg2(evaluator,s,g,17);
    for(auto pair:{std::make_pair(static_cast<Search*>(&p1),static_cast<Search*>(&p2)),std::make_pair(static_cast<Search*>(&gp1),static_cast<Search*>(&gp2)),std::make_pair(static_cast<Search*>(&gg1),static_cast<Search*>(&gg2))}) {
        int reports=0;int64_t last=0;SearchRun options;options.collect_analysis=true;options.report_interval=0;
        options.on_report=[&](const SearchResult& r){check(r.root_visits>=last,"Monotone completed visits");last=r.root_visits;++reports;};
        // SearchState overloads are intentionally not virtual.
        SearchResult a,b;
        if(pair.first==&p1){a=p1.run(position,.7);b=p2.run(position,.7,options);}
        else if(pair.first==&gp1){a=gp1.run(position,.7);b=gp2.run(position,.7,options);}
        else {a=gg1.run(position,.7);b=gg2.run(position,.7,options);}
        compare(a,b);check(reports>1,"Same search emits multiple snapshots");
    }
    std::vector<std::unique_ptr<muzero::Backend>> backends;backends.push_back(std::make_unique<AnalysisBackend>());
    muzero::BatchEvaluator mu(std::move(backends),6,8,0,0,false,0,1);Game game(5,6,Rule::HEX);game.play(0);
    muzero::Search mp1(mu,s,17),mp2(mu,s,17);muzero::GumbelSearch<false> mg1(mu,s,g,17),mg2(mu,s,g,17);
    muzero::GumbelSearch<true> mf1(mu,s,g,17),mf2(mu,s,g,17);
    for(auto pair:{std::make_pair(static_cast<GameSearch*>(&mp1),static_cast<GameSearch*>(&mp2)),std::make_pair(static_cast<GameSearch*>(&mg1),static_cast<GameSearch*>(&mg2)),std::make_pair(static_cast<GameSearch*>(&mf1),static_cast<GameSearch*>(&mf2))}) {
        int reports=0;SearchRun options;options.collect_analysis=true;options.report_interval=0;options.on_report=[&](const SearchResult&){++reports;};
        auto a=pair.first->run(game,.7),b=pair.second->run(game,.7,options);compare(a,b);check(reports>1,"MuZero live snapshots");
    }
    mu.finish();std::cout<<"Analysis diagnostics preserve all six search paths\n";
}
