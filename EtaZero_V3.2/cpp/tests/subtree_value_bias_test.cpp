#include "etazero/gumbel_search.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>
using namespace etazero;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void near(double a,double b,const char* message){check(std::abs(a-b)<1e-11,message);}
SubtreeValueBiasKey subtree_value_bias_key(const Game& game,int action) {
    SubtreeValueBiasKey key;check(game.legal(action)&&game.subtree_bias_key(action,key),"Expected a shareable legal move");return key;
}
struct Constant : Evaluator {
    void set_rule(Rule rule) {spec_=game_spec(rule);}
    Evaluation evaluate(const std::vector<float>& obs) override {
        int canvas=static_cast<int>(std::sqrt((obs.size()-spec_.globals)/spec_.planes));
        return {std::vector<double>(spec_.actions(canvas),0),{.65,.1,.25}};
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
void go_patterns() {
    Game empty(9,9,Rule::GO);SubtreeValueBiasKey key;
    check(!empty.subtree_bias_key(0,key),"Go first move has no previous location");
    empty.play(81);
    check(empty.subtree_bias_key(0,key),"Go previous pass must allow bias sharing");
    check(!empty.subtree_bias_key(81,key),"Go selected pass must not share bias");
    Game padded(9,11,Rule::GO);padded.play(121);
    check(key==subtree_value_bias_key(padded,0),"Go padding changed the previous-pass key");
    Game ko(5,5,Rule::GO);for(int a:{5,6,1,10,7,12,24,16,11})ko.play(a);
    check((subtree_value_bias_key(ko,4).pattern>>50)==14,"Go key omitted the absolute simple-ko location");
    ko.play(25);check((subtree_value_bias_key(ko,4).pattern>>50)==0,"Go key retained an expired ko ban");
    auto chain=[](bool atari) {
        Game game(9,9,Rule::GO);
        for(int x=2;x<=7;++x){game.play(4*9+x);game.play(3*9+x);}
        for(int x=3;x<=7;++x){game.play(x-3);game.play(5*9+x);}
        for(int a:{5,4*9+1,6,atari?4*9+8:8*9+8,8*9})game.play(a);
        return game;
    };
    auto atari=chain(true),free=chain(false);
    auto a=subtree_value_bias_key(atari,5*9+2),b=subtree_value_bias_key(free,5*9+2);
    check(a.pattern==b.pattern && (a.moves&((uint64_t{1}<<34)-1))==(b.moves&((uint64_t{1}<<34)-1)),"Atari fixture must have identical colors and move locations");
    check((a.moves^b.moves)==((uint64_t{7}<<7)<<34),"Remote liberty must split all local stones of the same chain");
}
// FEN fixtures differ only in the feature under test. The same legal black
// rook move supplies real previous-action history before the red candidate.
Game xiangqi_fixture(std::initializer_list<std::tuple<int,int,char>> changes={},int clock=0,
                     int previous_x=8,int previous_y=8) {
    std::array<std::array<char,9>,10> board{};
    for(auto [x,y,p]:{std::tuple<int,int,char>{4,0,'K'},{5,9,'k'},{0,0,'R'},
                      {1,0,'N'},{4,3,'P'},{4,6,'p'}})board[y][x]=p;
    board[previous_y][previous_x]='r';
    for(auto [x,y,p]:changes)board[y][x]=p;
    std::string fen;
    for(int y=9;y>=0;--y) {
        int empty=0;
        for(int x=0;x<9;++x) {
            if(!board[y][x]){++empty;continue;}
            if(empty){fen+=char('0'+empty);empty=0;}
            fen+=board[y][x];
        }
        if(empty)fen+=char('0'+empty);
        if(y)fen+='/';
    }
    RuleOptions options;options.fen=fen+" b - - "+std::to_string(clock)+" 1";
    Game game(10,10,Rule::XIANGQI,options);game.play(xiangqi_action(previous_x,previous_y,8,9));return game;
}
void xiangqi_patterns() {
    int action=xiangqi_action(1,0,2,2);SubtreeValueBiasKey unused;
    Game start(10,10,Rule::XIANGQI);
    check(start.spec().subtree_value_bias && !start.subtree_bias_key(action,unused),"Xiangqi first move must lack a previous action");
    auto game=xiangqi_fixture();auto key=subtree_value_bias_key(game,action);
    auto remote=xiangqi_fixture({{6,4,'P'}});
    check(key==subtree_value_bias_key(remote,action),"Irrelevant remote Xiangqi pieces must share residuals");
    check(game.history_identity()!=remote.history_identity(),"SVB sharing must not require graph identity");
    // Reach shareable patterns through two legal branches of the same start,
    // not just independently constructed FENs.
    auto left=start,right=start;
    for(auto [g,file]:{std::pair<Game*,int>{&left,6},{&right,8}}) {
        g->play(xiangqi_action(file,3,file,4));g->play(xiangqi_action(file,6,file,5));g->play(action);
    }
    check(left.history_identity()!=right.history_identity() &&
          subtree_value_bias_key(left,xiangqi_action(1,9,2,7))==subtree_value_bias_key(right,xiangqi_action(1,9,2,7)),
          "Different legal search branches must share unchanged local patterns");
    SubtreeValueBiasTable table(2);auto entry=table.get(key);
    SubtreeValueBias bias{entry,0,0},shared{table.get(subtree_value_bias_key(remote,action)),0,0};
    near(bias.update(.2,.6,16,16,.5,.5),.4,"Xiangqi residual update");
    near(shared.apply(.1,.5),.3,"Remote Xiangqi position did not read the shared residual");
    check(table.size()==1,"Equivalent Xiangqi patterns allocated separate entries");
    for(auto changed:{xiangqi_fixture({{0,1,'C'}}),xiangqi_fixture({{3,3,'C'}}),
                      xiangqi_fixture({{2,2,'p'}}),xiangqi_fixture({},37),xiangqi_fixture({},0,7,9)})
        check(!(key==subtree_value_bias_key(changed,action)),"Local piece, capture, clock or previous move was omitted");
    check(!(key==subtree_value_bias_key(game,xiangqi_action(1,0,3,1))),"Xiangqi current full move was omitted");
    auto horse_leg=xiangqi_fixture({{1,1,'C'}});
    check(!horse_leg.subtree_bias_key(action,unused),"Blocked horse leg must not produce a legal-move key");
    auto rook=xiangqi_fixture(),cannon=xiangqi_fixture({{0,0,'C'}});
    int straight=xiangqi_action(0,0,0,1);
    check(!(subtree_value_bias_key(rook,straight)==subtree_value_bias_key(cannon,straight)),"Moving piece type was omitted");
    auto ray=xiangqi_fixture({{4,2,'C'},{6,2,'R'}});
    for(auto changed:{xiangqi_fixture({{5,2,'C'},{6,2,'R'}}),xiangqi_fixture({{4,2,'C'},{6,2,'C'}}),
                      xiangqi_fixture({{4,2,'C'},{7,2,'R'}}),xiangqi_fixture({{4,2,'p'},{6,2,'R'}})})
        check(!(subtree_value_bias_key(ray,action)==subtree_value_bias_key(changed,action)),"Remote blocker color, type or distance was omitted");
    Game clone=game,reset=remote;reset=game;
    check(key==subtree_value_bias_key(clone,action) && key==subtree_value_bias_key(reset,action),"Xiangqi clone/reset changed the key");
    // FEN preserves rule60 but not check10. One common reply leaves the same
    // board, side, previous action and rule60, isolating the checking history.
    RuleOptions checks;checks.fen="5k3/4R4/9/9/9/9/4P4/9/9/4K4 w - - 0 1";
    Game checking(10,10,Rule::XIANGQI,checks);checking.play(xiangqi_action(4,8,5,8));
    checks.fen=checking.fen();Game no_checks(10,10,Rule::XIANGQI,checks);
    for(auto* g:{&checking,&no_checks})g->play(xiangqi_action(5,9,4,9));
    check(checking.fen()==no_checks.fen(),"Check-count fixture must have identical FENs");
    check(!(subtree_value_bias_key(checking,xiangqi_action(5,8,6,8))==subtree_value_bias_key(no_checks,xiangqi_action(5,8,6,8))),"Checking counters were omitted");
    auto repeated=start;
    for(auto m:{std::array<int,4>{1,0,2,2},{1,9,2,7},{2,2,1,0},{2,7,1,9}})
        repeated.play(xiangqi_action(m[0],m[1],m[2],m[3]));
    check(!repeated.finished() && !repeated.subtree_bias_key(action,unused),"Twofold history must disable Xiangqi bias sharing");
    near(repeated.observation()[repeated.planes()*100+2],.5,"Twofold NN repetition feature");
    auto escaped=repeated;escaped.play(xiangqi_action(7,0,6,2));
    check(!escaped.subtree_bias_key(xiangqi_action(7,9,6,7),unused),"Leaving a repeated board must retain the history guard");
    near(escaped.observation()[escaped.planes()*100+2],0,"SVB history guard must not replace the current-board NN repetition count");
    repeated.play(xiangqi_action(1,2,1,9));
    check(repeated.subtree_bias_key(xiangqi_action(7,9,6,7),unused),"Capture must start a fresh repetition segment");
    near(repeated.observation()[repeated.planes()*100+2],0,"Capture must reset the NN repetition feature");
    escaped=game;check(key==subtree_value_bias_key(escaped,action),"Reset retained a stale repetition guard");
    auto limit=xiangqi_fixture({},118);limit.play(action);
    check(limit.finished() && limit.reason()==Reason::MOVE_LIMIT && !limit.subtree_bias_key(straight,unused),"Terminal Xiangqi state must not share bias");
}
template<bool Full> void gumbel(Constant& evaluator,Rule rule) {
    auto s=settings(4);s.graph_search=false;s.reuse_tree=false;
    GumbelSettings g;g.noise_scale=0;g.max_num_considered_actions=4;
    GumbelSearch<Full> search(evaluator,s,g,44);
    Game game(rule==Rule::XIANGQI?10:5,rule==Rule::XIANGQI?10:6,rule);
    game.play(rule==Rule::XIANGQI?xiangqi_action(1,0,2,2):0);
    for(int i=0;i<2;++i) {
        auto r=search.run(game,0,SearchRun{128});check(game.legal(r.action)&&search.pending()==0,"Gumbel bias search failed");
        for(double q:r.q_values)check(std::isfinite(q)&&std::abs(q)<=1,"Gumbel bias corrupted Q");
        check(std::abs(r.value-(r.search_wdl[0]-r.search_wdl[2]))<1e-12,"Gumbel WDL corrupted");
        game.play(r.action);search.advance(r.action);
    }
}
}
int main(int argc,char** argv) {
    try {
        std::cout.precision(17);
        if(argc>1 && std::string(argv[1])=="--patterns") {
            int size,canvas,count,action;std::string rules;
            while(std::cin>>size>>canvas>>rules>>count) {
                RuleOptions options;options.go_rules=rules;Game game(size,canvas,Rule::GO,options);
                for(int i=0;i<count;++i){std::cin>>action;game.play(game.native_action(action));}
                for(int a=0;a<game.local_actions();++a) {
                    int native=game.native_action(a);SubtreeValueBiasKey key;
                    if(game.legal(native) && game.subtree_bias_key(native,key))std::cout<<a<<' '<<key.pattern<<' '<<key.moves<<' ';
                }
                std::cout<<'\n';
            }
            return 0;
        }
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
        go_patterns();xiangqi_patterns();
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
        for(bool graph:{false,true})for(int threads:{1,4})for(Rule rule:{Rule::FREESTYLE,Rule::STANDARD,Rule::RENJU,Rule::HEX,Rule::GO,Rule::XIANGQI}) {
            evaluator.set_rule(rule);
            auto s=settings(threads);s.graph_search=graph;s.use_noise_pruning=true;s.use_uncertainty=true;
            Probe search(evaluator,s,33);Game position(rule==Rule::XIANGQI?10:5,rule==Rule::XIANGQI?10:6,rule);
            position.play(rule==Rule::XIANGQI?xiangqi_action(1,0,2,2):0);
            auto r=search.run(position,0);search.verify(r);
            search.advance(r.action);position.play(r.action);r=search.run(position,0);search.verify(r);
            search.reset(34);check(search.entries()==0,"Reset retained pattern statistics");
            r=search.run(position,0);search.verify(r);search.set_evaluator(evaluator);check(search.entries()==0,"Model replacement retained bias");
        }
        for(auto rule:{Rule::HEX,Rule::GO,Rule::XIANGQI}){evaluator.set_rule(rule);gumbel<false>(evaluator,rule);gumbel<true>(evaluator,rule);}
        evaluator.set_rule(Rule::FREESTYLE);
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
