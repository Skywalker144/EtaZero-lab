#include "etazero/gumbel_search.h"
#include <cmath>
#include <iostream>
#include <numeric>
using namespace etazero;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void near(double a,double b,const char* message){check(std::abs(a-b)<1e-11,message);}
struct Constant : Evaluator {
    Evaluation evaluate(const std::vector<float>& obs) override {
        size_t area=(obs.size()-GLOBAL_FEATURES)/INPUT_PLANES;
        return {std::vector<double>(area,0),{.65,.1,.25}};
    }
};
struct Probe : Search {
    using Search::Search;
    size_t entries() const {return subtree_bias_table_?subtree_bias_table_->size():0;}
    void verify(const SearchResult& result) {
        bool corrected=false;size_t attached=0;
        for(auto& shard:node_table_)for(auto& item:shard.nodes) {
            auto& n=*item.second;auto s=snapshot(n);
            check(std::isfinite(s.value)&&std::isfinite(s.value_sq)&&std::isfinite(s.win_loss),"Nonfinite utility/W-L");
            check(std::abs(s.win_loss)<=1+1e-12,"Bias corrupted raw W-L");
            if(n.status.load()!=Node::READY && s.visits>0)near(s.value,s.win_loss,"Terminal received bias");
            if(n.subtree_bias.entry)++attached;
            if(std::abs(s.value-s.win_loss)>1e-8)corrected=true;
        }
        check(corrected&&attached>0&&entries()>0,"Search did not apply subtree value bias");
        check(!root_->subtree_bias.entry,"Root carries a bias association");
        near(result.value,snapshot(*root_).win_loss,"Reported value uses biased utility");
        near(result.search_wdl[0]-result.search_wdl[2],result.value,"Reported WDL differs from W-L");
        for(int i=0;i<root_->child_count;++i) {
            auto& e=root_->edge(i);auto child=e.child.load();if(!child)continue;
            int a=root_->moves[e.move_index].action;
            near(result.q_values[a],float(e.perspective.load()*snapshot(*child).win_loss),"Q target uses biased utility");
        }
        check(pending()==0,"Bias search leaked pending visits");
    }
    double root_utility(){return snapshot(*root_).value;}
    void association(const SearchState& state,const SubtreeValueBiasKey& key) {
        prepare_subtree_bias(state);
        Node* first=find_node(state,&key);
        auto different=key;different.pattern^=1;
        Node* transposed=find_node(state,&different);
        check(first==transposed&&first->subtree_bias.entry==subtree_bias_table_->get(key),"Transposition rebound the first-path bias");
        reset(3);check(entries()==0,"Unreachable graph bias survived reset");
    }
};
SearchSettings settings(int threads=1) {
    SearchSettings s{300,threads,1,1,0,1,true};s.graph_search=true;s.use_fpu=true;
    s.subtree_value_bias_factor=.45;s.subtree_value_bias_table_num_shards=16;
    s.value_weight_exponent=0;s.chosen_move_prune=0;return s;
}
template<bool Full> void gumbel(Constant& evaluator) {
    auto s=settings(4);s.graph_search=false;s.reuse_tree=false;
    GumbelSettings g;g.noise_scale=0;g.max_num_considered_actions=4;
    GumbelSearch<Full> search(evaluator,s,g,44);Game game(5,6,Rule::HEX);game.play(0);
    for(int i=0;i<2;++i) {
        auto r=search.run(game,0,SearchRun{128});check(game.legal(r.action)&&search.pending()==0,"Gumbel bias search failed");
        for(double q:r.q_values)check(std::isfinite(q)&&std::abs(q)<=1,"Gumbel bias corrupted Q");
        check(std::abs(r.value-(r.search_wdl[0]-r.search_wdl[2]))<1e-12,"Gumbel WDL corrupted");
        game.play(r.action);search.advance(r.action);
    }
}
}
int main(int argc,char**) {
    try {
        std::cout.precision(17);
        if(argc>1) {
            char op;SubtreeValueBias nodes[2];auto shared=std::make_shared<SubtreeValueBiasEntry>();
            nodes[0].entry=nodes[1].entry=shared;
            while(std::cin>>op) {
                int n;double utility,factor;std::cin>>n;
                if(op=='r') {shared=std::make_shared<SubtreeValueBiasEntry>();nodes[0]={shared,0,0};nodes[1]={shared,0,0};continue;}
                if(op=='u') {double child,original,current,exponent;std::cin>>utility>>child>>original>>current>>factor>>exponent;
                    // The source oracle accepts a child sum; normalize the
                    // identical input sum here before entering the mean API.
                    double child_mean=current>1e-10?(child*current)/current:0;
                    std::cout<<nodes[n].update(utility,child_mean,original,current,factor,exponent);}
                else if(op=='l') {std::cin>>utility>>factor;std::cout<<nodes[n].apply(utility,factor);}
                else if(op=='d') {double prop;std::cin>>prop;nodes[n].release(prop);std::cout<<0;}
                else return 1;
                std::cout<<' '<<shared->delta_sum<<' '<<shared->weight_sum<<' '<<nodes[n].last_delta<<' '<<nodes[n].last_weight<<'\n';
            }
            return 0;
        }
        SubtreeValueBiasTable table(1);Game game(9,10,Rule::FREESTYLE);game.play(0);
        auto key=subtree_value_bias_key(game,44);auto entry=table.get(key);
        check(entry==table.get(key),"Equal patterns failed to share");
        // Restore the same immediately preceding move using a different legal
        // history, while changing only stones outside the target 5x5.
        Game other(9,10,Rule::FREESTYLE);other.play(88);other.play(8);other.play(0);
        check(key==subtree_value_bias_key(other,44),"Remote stones split local pattern");
        Game local(9,10,Rule::FREESTYLE);local.play(33);local.play(8);local.play(0);
        check(!(key==subtree_value_bias_key(local,44)),"Local stones failed to split pattern");
        Game padded(9,11,Rule::FREESTYLE);padded.play(0);
        check(key==subtree_value_bias_key(padded,48),"NN canvas changed board-pattern key");
        Game previous(9,10,Rule::FREESTYLE);previous.play(1);
        check(!(key==subtree_value_bias_key(previous,44)),"Previous absolute location was omitted");
        check(!(key==subtree_value_bias_key(game,45)),"Current absolute location was omitted");
        Game next_board=game;next_board.play(44);
        check(!(key==subtree_value_bias_key(next_board,45)),"Post-move pattern was substituted for pre-move pattern");
        SubtreeValueBias a{entry,0,0},b{entry,0,0};
        near(a.update(.2,.6,16,16,.5,.5),.4,"Hand-computed first correction");
        near(entry->delta_sum,1.6,"Contribution delta");near(entry->weight_sum,4,"Contribution weight");
        near(a.update(.2,.6,16,16,.5,.5),.4,"Repeated backup changed contribution");near(entry->weight_sum,4,"Double-counted subtree visits");
        near(b.update(.4,-.2,4,4,.5,.5),.4+(.4/6)*.5,"Two subtrees shared empirical bias incorrectly");
        near(a.apply(.2,.5),.2+(.4/6)*.5,"Leaf failed to read shared bias");
        a.release(.8);near(entry->delta_sum,-.88,"Wrong partial free delta");near(entry->weight_sum,2.8,"Wrong partial free weight");
        a.release(.8);near(entry->weight_sum,2.8,"Double removal");
        b.release(1);near(entry->weight_sum,.8,"Retained residual lost");
        table.clear_unused();check(table.size()==1,"Live entry removed");entry.reset();table.clear_unused();check(table.size()==0,"Unused residual entry leaked");
        // Bias affects only the own utility and its second moment. W-L/draw
        // aggregation uses the same weights but stays a probability statistic.
        auto settings0=settings();settings0.subtree_value_bias_factor=.5;
        SubtreeValueBias bias{std::make_shared<SubtreeValueBiasEntry>(),0,0};
        auto aggregate=aggregate_values({.6,.2,.2},{{1,1.6,1.28,2,2,2,.2,1.0}},settings0,false,1,&bias);
        near(aggregate.value,(1.6+.6)/3,"Own utility not corrected after child aggregation");
        near(aggregate.value_sq,(1.28+.36)/3,"Biased utility square missing");
        near(aggregate.win_loss,1.4/3,"Raw W-L changed");near(aggregate.draw,.4/3,"Draw mass changed");
        auto pruned=settings0;pruned.use_noise_pruning=true;
        SubtreeValueBias pruning_bias{std::make_shared<SubtreeValueBiasEntry>(),0,0};
        auto pruned_stats=aggregate_values({.5,0,.5},{{.9,.5,.25,1,1,1,0},{.1,-4.5,2.25,9,9,9,0}},pruned,false,1,&pruning_bias);
        check(pruned_stats.weight<11,"Noise pruning fixture did not prune");
        near(pruning_bias.last_weight,std::pow(10,.85),"Bias confidence used pruned child weights");
        SubtreeValueBias tiny{std::make_shared<SubtreeValueBiasEntry>(),0,0};
        tiny.entry->delta_sum=1;tiny.entry->weight_sum=.001;near(tiny.apply(.4,1),.4,"Threshold is not strictly greater than .001");
        tiny.update(.4,.9,1,1e-10,.5,.5);near(tiny.last_weight,0,"Child threshold is not strictly greater than 1e-10");
        Constant evaluator;
        {AlphaZeroState state(Game(5,6,Rule::FREESTYLE),evaluator);SubtreeValueBiasKey k;check(!state.subtree_bias_key(0,k),"First move received a history pattern");}
        for(bool graph:{false,true})for(int threads:{1,4})for(Rule rule:{Rule::FREESTYLE,Rule::STANDARD,Rule::RENJU,Rule::HEX}) {
            auto s=settings(threads);s.graph_search=graph;s.use_noise_pruning=true;s.use_uncertainty=true;
            Probe search(evaluator,s,33);Game position(5,6,rule);position.play(0);
            auto r=search.run(position,0);search.verify(r);
            search.advance(r.action);position.play(r.action);r=search.run(position,0);search.verify(r);
            search.reset(34);check(search.entries()==0,"Reset retained pattern statistics");
            r=search.run(position,0);search.verify(r);search.set_evaluator(evaluator);check(search.entries()==0,"Model replacement retained bias");
        }
        gumbel<false>(evaluator);gumbel<true>(evaluator);
        {Probe search(evaluator,settings(),1);AlphaZeroState state(game,evaluator);search.association(state,key);}
        {
            auto s=settings(4);s.simulations=6500;Probe large(evaluator,s,3);
            auto r=large.run(Game(19,19,Rule::FREESTYLE),0);large.verify(r);
            check(large.inspect_graph().size()>4096,"Bias cleanup did not exercise parallel deletion");
            large.reset(4);check(large.entries()==0,"Parallel cleanup leaked bias entries");
        }
        {
            std::vector<std::unique_ptr<muzero::Backend>> backends;
            backends.push_back(std::make_unique<muzero::RandomBackend>(6,1));
            muzero::BatchEvaluator mu(std::move(backends),6,1,0,0,false,0,1);
            auto s=settings();s.graph_search=false;s.reuse_tree=false;
            bool rejected=false;try{muzero::Search latent(mu,s,1);}catch(const std::runtime_error& e){rejected=std::string(e.what()).find("subtree_value_bias_factor=0")!=std::string::npos;}
            check(rejected,"MuZero silently ignored subtree value bias");
        }
        auto off=settings();off.subtree_value_bias_factor=0;Probe unbiased(evaluator,off,5);auto r=unbiased.run(Game(5,6,Rule::HEX),0);
        check(unbiased.entries()==0,"Disabled bias allocated a table");near(unbiased.root_utility(),r.value,"Disabled utility differs from W-L");
        std::cout<<"subtree value bias tests passed; fixed bytes per 65536 shards="<<SubtreeValueBiasTable::fixed_bytes(65536)<<'\n';return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
