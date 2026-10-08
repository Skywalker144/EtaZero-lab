#include "etazero/search.h"
#include <cmath>
#include <iostream>
using namespace etazero;
namespace {
void check(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
void near(double a,double b,const char* msg){check(std::abs(a-b)<1e-11,msg);}
struct AuxState : SearchState {
    int depth=0;double error=.5;bool auxiliary=true,draw=false;std::vector<std::pair<int,double>>* seen=nullptr;
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<AuxState>(*this);}
    int actions() const override{return 2;}
    bool legal(int a) const override{return !terminal() && a>=0 && a<2;}
    bool terminal() const override{return depth==2;}
    double terminal_value() const override{return draw?0:1;}
    Evaluation evaluate() const override{return {{std::log(9),0},{.6,.2,.2},{0,std::log(9)},error,auxiliary};}
    Evaluation evaluate_symmetry(int,bool,double,bool,double optimism) const override {
        if(seen)seen->push_back({depth,optimism});return evaluate();
    }
    std::string graph_key() const override{return std::to_string(depth);}
    Transition move(int a) override{check(legal(a),"Illegal aux action");++depth;return {0,1,1};}
};
struct SymmetricAux : AuxState {
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<SymmetricAux>(*this);}
    int symmetry_count() const override{return 8;}
    Evaluation evaluate_symmetry(int symmetry,bool,double,bool,double) const override {
        auto result=evaluate();result.shortterm_value_stdev=(symmetry+1)/8.0;
        result.optimistic_logits={0,double(symmetry)/2};return result;
    }
};
struct AuxBackend : Backend {
    bool supports_auxiliary() const override{return true;}
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override {
        std::vector<Evaluation> result;for(auto obs:inputs){size_t n=(obs->size()-GLOBAL_FEATURES)/INPUT_PLANES;
            Evaluation e{std::vector<double>(n),{.6,.2,.2},std::vector<double>(n),.5,true};
            for(size_t i=0;i<n;++i){e.logits[i]=i;e.optimistic_logits[i]=-double(i);}result.push_back(std::move(e));}return result;
    }
};
SearchSettings settings() {SearchSettings s{30,1,1.5,1,0,1,true};s.value_weight_exponent=0;s.chosen_move_prune=0;return s;}
}
int main(int argc,char**) {
    try {
        std::cout.precision(17);
        if(argc>1) {
            char kind;while(std::cin>>kind) {
                auto s=settings();s.use_uncertainty=true;s.use_noise_pruning=true;
                if(kind=='u') {double u;int support;std::cin>>u>>support>>s.uncertainty_coeff>>s.uncertainty_exponent>>s.uncertainty_max_weight;std::cout<<uncertainty_weight(u,support,s)<<'\n';}
                else if(kind=='m') {double p,o,a;std::cin>>p>>o>>a;std::cout<<mixed_policy_logit(p,o,a)<<'\n';}
                else if(kind=='n') {int count;std::cin>>count>>s.noise_prune_utility_scale>>s.noise_pruning_cap;
                    std::vector<RootChildStats> children;for(int i=0;i<count;++i){double p,w,q;std::cin>>p>>w>>q;children.push_back({p,w*q,w*q*q,1,w,w,0});}
                    for(double w:noise_pruned_weights(children,s))std::cout<<w<<' ';std::cout<<'\n';}
                else return 1;
            }return 0;
        }
        auto s=settings();s.use_uncertainty=true;
        near(uncertainty_weight(0,true,s),8,"Zero error cap");near(uncertainty_weight(.5,true,s),8.0/17,"Error standard deviation used directly");
        near(uncertainty_weight(.5,false,s),1,"Absent error capability fallback");
        s.uncertainty_exponent=.5;near(uncertainty_weight(.25,true,s),8.0/17,"Square-root uncertainty exponent");
        s.uncertainty_exponent=0;near(uncertainty_weight(0,true,s),8.0/33,"Exponent-zero source pow(0,0)");
        s.uncertainty_exponent=2;near(uncertainty_weight(.5,true,s),8.0/9,"Power-two uncertainty");s.uncertainty_exponent=1;
        AuxState state;Search search(s,32);SearchRun first;first.max_playouts=1;auto initial=search.run(state,1,first);
        auto own=8.0/17;near(search.inspect_graph()[0].stats.weight,own,"NN sample weight");
        first.max_playouts=2;search.run(state,1,first);
        bool terminal=false;for(const auto& n:search.inspect_graph()) {
            if(!n.root && !n.ready && n.stats.visits>0){terminal=true;near(n.stats.weight,8*n.stats.visits,"Terminal uncertainty max weight");near(n.stats.weight_sq,64*n.stats.visits,"Terminal squared sample weight accumulates per visit");}
        }check(terminal,"Did not visit a terminal leaf");
        // D4 averages stdev first; the nonlinear reciprocal cannot be averaged instead.
        SymmetricAux symmetric;auto eight=s;eight.root_symmetries=8;Search ensemble(eight,1);SearchRun one;one.max_playouts=1;
        ensemble.run(symmetric,1,one);near(ensemble.inspect_graph()[0].stats.weight,8.0/19,"Average D4 error before weighting");
        double expected=0;for(int k=0;k<8;++k){float main=std::log(9);double x=main+(float(0)-main)*float(.2),y=float(k)/2*float(.2);expected+=std::exp(x)/(std::exp(x)+std::exp(y))/8;}
        eight.root_policy_optimism=.2;Search optimism_ensemble(eight,1);near(optimism_ensemble.run(symmetric,1,one).network_policy[0],expected,"Mix D4 logits before each probability average");
        auto opt=settings();opt.root_policy_optimism=.2;opt.policy_optimism=1;opt.nn_policy_temperature=2;opt.graph_search=true;
        std::vector<std::pair<int,double>> seen;state.seen=&seen;Search optimistic(opt,1);
        auto r=optimistic.run(state,1,one);float main=std::log(9),mixed=main+(float(0)-main)*float(.2);double odds=std::exp((double(mixed)-float(main*float(.2)))/2);near(r.network_policy[0],odds/(1+odds),"Logit mixture precedes Tnn softmax");
        optimistic.run(state,1,one);check(seen.back()==std::pair<int,double>{1,1},"Leaf optimism condition");
        optimistic.advance(r.action);state.move(r.action);one.max_playouts=0;auto promoted=optimistic.run(state,1,one);
        check(seen.back()==std::pair<int,double>{1,.2},"Promoted root must re-evaluate with root optimism");check(promoted.new_playouts==0,"Root optimism refresh counted as a new playout");
        near(promoted.network_policy[0],odds/(1+odds),"Promoted leaf policy did not become a root policy");
        state.depth=0;state.auxiliary=false;Search unsupported(opt,1);one.max_playouts=1;
        near(unsupported.run(state,1,one).network_policy[0],.75,"Absent optimism capability uses ordinary logits at Tnn2");
        auto noise=settings();noise.use_noise_pruning=true;
        double q=.6-.15*std::log(2);std::vector<RootChildStats> children{{.5,2.4,1.44,4,4,4,0},{.25,20*q,20*q*q,20,20,20,0}};
        auto weights=noise_pruned_weights(children,noise);near(weights[0],4,"First noise prefix kept");near(weights[1],12,"Half excess pruned at log2 utility gap");
        auto combined=aggregate_values({0,1,0},children,noise,false,2);
        near(combined.weight,18,"Pruned total is not normalized back to original total");near(combined.weight_sq,15.2,"Pruned aggregation second moments");near(combined.value,(2.4+12*q)/18,"Pruned weighted mean");near(combined.draw,1.0/9,"Pruned draw mass");check(combined.visits==25,"Noise pruning changed visits");
        noise.noise_pruning_cap=1;weights=noise_pruned_weights(children,noise);near(weights[1],19,"Noise pruning cap");
        noise.noise_pruning_cap=0;noise.chosen_move_subtract=100;noise.chosen_move_prune=100;
        near(aggregate_values({0,1,0},children,noise,true).weight,25,"Noise pruning takes precedence over chosen backup pruning even at cap zero");
        noise.use_noise_pruning=false;near(noise_pruned_weights(children,noise)[1],20,"Noise pruning disabled path");
        {
            std::vector<std::unique_ptr<Backend>> backends;backends.push_back(std::make_unique<AuxBackend>());
            BatchEvaluator batch(std::move(backends),"aux",5,4,8,0,1024);Game game(5,5,Rule::RENJU);auto obs=game.observation();
            auto a=batch.evaluate_symmetry(obs,5,false,1,false,.2);auto hit=batch.evaluate_symmetry(obs,1,false,1,false,.2);
            check(a.logits==hit.logits && batch.requests==1 && batch.cache_hits==1,"Optimism cache hit changed canonical output");
            auto b=batch.evaluate_symmetry(obs,1,false,1,false,1);check(batch.requests==2 && b.logits!=a.logits,"Optimism condition missing from cache key");
            check(b.has_auxiliary && b.shortterm_value_stdev==.5,"Auxiliary cache output lost");
            batch.evaluate_symmetry(obs,5,true,1,false,.2);check(batch.requests==3,"Root ensemble bypass did not infer");batch.finish();
        }
        std::cout<<"search corrections passed\n";return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
