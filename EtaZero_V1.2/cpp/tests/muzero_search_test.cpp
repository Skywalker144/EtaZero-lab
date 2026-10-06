#include "etazero/muzero/search.h"
#include <cmath>
#include <iostream>
using namespace etazero;
namespace mu = etazero::muzero;
void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
struct State : mu::Latent { int depth=0,owner=0; };
struct Probe { std::atomic<int> initials{0},recurrents{0},deepest{0};bool fail=false; };
struct ProbeBackend : mu::Backend {
    Probe& probe;int owner;
    ProbeBackend(Probe& p,int o):probe(p),owner(o){}
    mu::InferenceOutput output(int depth) {
        auto state=std::make_shared<State>();state->depth=depth;state->owner=owner;
        Evaluation e;e.logits.assign(36,-1000);e.logits[1]=1000;e.wdl={.7,0,.3};
        return {state,e};
    }
    std::vector<mu::InferenceOutput> initial(const InferenceInputs& inputs) override {
        std::vector<mu::InferenceOutput> out;
        for(auto input:inputs){check((*input)[0]==1,"real board mask");++probe.initials;out.push_back(output(0));}
        return out;
    }
    std::vector<mu::InferenceOutput> recurrent(const std::vector<mu::InferenceAction>& inputs) override {
        if(probe.fail)throw std::runtime_error("injected recurrent failure");
        std::vector<mu::InferenceOutput> out;
        for(const auto& input:inputs){auto state=std::dynamic_pointer_cast<const State>(input.latent);
            check(state&&state->owner==owner,"latent routed to wrong backend");
            check(input.action==1,"preferred action must remain available at every latent depth");
            ++probe.recurrents;probe.deepest=std::max(probe.deepest.load(),state->depth+1);out.push_back(output(state->depth+1));}
        return out;
    }
};
std::unique_ptr<mu::BatchEvaluator> service(Probe& p,size_t cache=0) {
    std::vector<std::unique_ptr<mu::Backend>> backends;
    for(int i=0;i<2;++i)backends.push_back(std::make_unique<ProbeBackend>(p,i));
    return std::make_unique<mu::BatchEvaluator>(std::move(backends),6,8,0,cache,false,0,1);
}
struct QuotaState : mu::Latent { int depth=0,root_action=-1; };
struct QuotaBackend : mu::Backend {
    mu::InferenceOutput output(int depth,int root_action) {
        auto state=std::make_shared<QuotaState>();state->depth=depth;state->root_action=root_action;
        Evaluation e;e.logits.assign(36,-100);
        if(depth==0){e.logits[1]=std::log(.75);e.logits[2]=std::log(.25);e.wdl={0,1,0};}
        else {
            e.logits[1]=1000;
            bool win=(root_action==1)==(depth%2==0);e.wdl=win?WDL{1,0,0}:WDL{0,0,1};
        }
        return {state,e};
    }
    std::vector<mu::InferenceOutput> initial(const InferenceInputs& inputs) override {
        return std::vector<mu::InferenceOutput>(inputs.size(),output(0,-1));
    }
    std::vector<mu::InferenceOutput> recurrent(const std::vector<mu::InferenceAction>& inputs) override {
        std::vector<mu::InferenceOutput> out;
        for(const auto& input:inputs){auto state=std::dynamic_pointer_cast<const QuotaState>(input.latent);
            check(bool(state),"quota latent type");
            out.push_back(output(state->depth+1,state->depth==0?input.action:state->root_action));}
        return out;
    }
};
int main(){
    Game game(5,6,Rule::FREESTYLE);game.play(0);
    SearchSettings settings{79,1,1000,1,0,.3,false};settings.value_weight_exponent=0;
    Probe probe;auto evaluator=service(probe);mu::Search search(*evaluator,settings,12);
    auto result=search.run(game,0);
    check(result.root_visits==80&&result.new_playouts==80&&result.visits[1]==79,"exact visit budget");
    check(probe.initials==1&&probe.recurrents==79&&probe.deepest==79,"no real-board termination or recurrent masking");
    check(std::abs(result.value)<1e-12,"alternating perspective arithmetic mean");
    check(result.policy[0]==0&&result.policy[5]==0&&result.policy[1]==1,"true root legal mask");
    search.advance(1);search.run(game,0,SearchRun{2});check(probe.initials==2,"new actual move reencodes root");
    evaluator->finish();
    {
        std::vector<std::unique_ptr<mu::Backend>> backends;backends.push_back(std::make_unique<QuotaBackend>());
        mu::BatchEvaluator quota(std::move(backends),6,8,0,0,false,0,1);
        SearchSettings limits{64,1,2,1,0,.3,false};limits.value_weight_exponent=0;
        limits.forced_playouts=2;
        mu::Search forced(quota,limits,12);auto fp=forced.run(game,0);
        check(fp.root_visits==65&&fp.simulations==64,"forced playout preserves search budget");
        for(int a=0;a<game.actions();++a)if(game.legal(a)&&a!=1&&a!=2) {
            check(fp.search_policy[a]>0,"unsearched candidates retain positive priors");
            check(fp.visits[a]==0,"forced playout must not force first visits of unsearched actions");
        }
        limits.forced_playouts=0;mu::Search ordinary(quota,limits,12);auto nofp=ordinary.run(game,0);
        check(nofp.visits[2]>0,"ordinary PUCT activates the secondary action");
        check(fp.visits[2]>=5&&fp.visits[2]>nofp.visits[2],"evaluated children still receive forced quota");
        check(fp.action==1&&fp.q_values[1]==1&&fp.q_values[2]==-1,"forced exploration preserves winning action and Q perspective");
        SearchRun clean;clean.remove_root_noise=true;auto evaluation=forced.run(game,0,clean);
        check(evaluation.visits==nofp.visits,"remove-root-noise search disables forced playout");
        limits.forced_playouts=2;limits.threads=4;mu::Search parallel(quota,limits,12);
        auto concurrent=parallel.run(game,0);int64_t completed=0;
        for(auto visits:concurrent.visits)completed+=visits;
        check(concurrent.root_visits==65&&completed==64,"parallel forced playout completes every reserved visit");
        check(concurrent.action==1&&concurrent.visits[1]>concurrent.visits[2],"parallel forced exploration retains preferred action");
    }
    {
        Probe timed;auto service_owner=service(timed);auto limits=settings;limits.max_time=0;
        mu::Search limited(*service_owner,limits,12);
        for(double seconds:{0.0,1e-12}) {
            SearchRun options;options.max_time=seconds;
            int before=timed.recurrents;
            auto r=limited.run(game,0,options);
            check(r.new_playouts==2&&r.root_visits==2&&timed.recurrents==before+1&&r.stopped_early,
                  "time cap must allow root plus one recurrent playout");
        }
        for(bool visit_cap:{false,true}) {
            SearchRun options;if(visit_cap)options.max_visits=1;else options.max_playouts=1;
            auto r=limited.run(game,0,options);
            check(r.new_playouts==1&&r.simulations==0&&!r.stopped_early,"hard caps take precedence over time minimum");
        }
        SearchRun options;options.max_playouts=0;int before=timed.initials;
        auto r=limited.run(game,0,options);
        check(r.new_playouts==0&&r.action==-1&&timed.initials==before,"zero playout cap issues no inference");
        options={};options.should_stop=[] {return true;};
        r=limited.run(game,0,options);
        check(r.new_playouts==0&&r.stopped_early&&timed.initials==before,"immediate stop precedes root prediction");
        options.should_stop=[&] {return timed.initials>before;};
        r=limited.run(game,0,options);
        check(r.new_playouts==1&&r.simulations==0&&r.stopped_early,"explicit stop overrides time minimum after root");
    }
    Probe cached;auto cache=service(cached,8);auto a=cache->initial(game.observation());auto b=cache->initial(game.observation());
    check(a.latent==b.latent&&cached.initials==1&&cache->cache_hits==1,"cache retains latent and prediction");
    cache->recurrent(b.latent,1);bool rejected=false;
    Probe foreign;auto other=service(foreign);
    try{other->recurrent(a.latent,1);}catch(const std::exception&){rejected=true;}
    check(rejected,"foreign service latent rejected");
    for(int threads:{1,4}){
        Probe broken;broken.fail=true;auto failed=service(broken);settings.threads=threads;mu::Search s(*failed,settings,1);
        rejected=false;try{s.run(game,0);}catch(const std::exception& e){rejected=std::string(e.what()).find("injected recurrent")!=std::string::npos;}
        check(rejected,"parallel recurrent failure must propagate without deadlock");
    }
    for(int symmetry=0;symmetry<8;++symmetry){
        auto map=mu::symmetry_mapping(6,symmetry);auto obs=game.observation();auto transformed=mu::transform_observation(obs,6,map);
        for(int plane=0;plane<5;++plane)for(int action=0;action<36;++action)
            check(obs[plane*36+action]==transformed[plane*36+map[action]],"fixed orientation input/action consistency");
        Evaluation e;e.logits.resize(36);for(int action=0;action<36;++action)e.logits[map[action]]=action;
        mu::restore_evaluation(e,map);for(int action=0;action<36;++action)check(e.logits[action]==action,"restore root coordinates");
    }
    std::cout<<"MuZero search, masking, perspective, routing, cache and failures passed\n";
}
