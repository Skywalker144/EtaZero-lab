#include "etazero/game.h"
#include "etazero/record.h"
#include "etazero/random_evaluator.h"
#include "etazero/muzero/search.h"
#include "etazero/gumbel_search.h"
#include "etazero/sampling.h"
#include <iostream>
using namespace etazero;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
std::vector<int> legal(const Game& g){std::vector<int> out;for(int a=0;a<g.actions();++a)if(g.legal(a))out.push_back(a);return out;}
uint64_t perft(const Game& g,int depth){if(!depth)return 1;uint64_t n=0;for(int a:legal(g)){Game next=g;next.play(a);n+=perft(next,depth-1);}return n;}
struct NoResult : SearchState {
    std::unique_ptr<SearchState> clone() const override{return std::make_unique<NoResult>(*this);}
    int actions() const override{return 1;}
    bool legal(int) const override{return false;}
    bool terminal() const override{return true;}
    double terminal_value() const override{return 0;}
    double terminal_draw_probability() const override{return 0;}
    double terminal_no_result_probability() const override{return 1;}
    Evaluation evaluate(SearchEvaluation={}) const override{throw std::runtime_error("Terminal state evaluated");}
    int move(int) override{throw std::runtime_error("Terminal state moved");}
};
struct DenseSearch : muzero::Search {
    using muzero::Search::Search;
    void verify_storage(const Game& game) const {
        size_t count=0,capacity=0;
        for(const auto& node:nodes_) {
            count+=node->edges.size();capacity+=node->edges.capacity();
            check(node->edges.size()==(&*node==root_?legal(game).size():game.static_actions().size()),"MuZero must retain the full latent action domain");
        }
        check(capacity<=count*1.1,"Dense action storage must avoid geometric over-allocation");
    }
};
void reanalysis_history(Rule rule,Evaluator& evaluator) {
    int size=rule==Rule::GO?5:10;Game game(size,size,rule);game.set_pda(.5,1);
    FinishedGame record{};record.size=size;record.canvas=size;record.rule=rule;record.options=game.options();
    std::vector<int> moves;
    if(rule==Rule::GO)moves={5,6,1,10,7,12,24,16,11,25,25};
    else for(int i=0;i<2;++i)for(auto m:{std::array<int,4>{1,0,2,2},{1,9,2,7},{2,2,1,0},{2,7,1,9}})
        moves.push_back(xiangqi_action(m[0],m[1],m[2],m[3]));
    record.opening.actions.assign(moves.begin(),moves.begin()+2);
    std::vector<Game> positions;std::vector<double> history;
    for(int action:moves) {
        Step step{};step.player=game.player();step.action=action;step.observation=game.observation();
        step.trainable=game.turn()>=2;step.cheap_search=step.trainable;step.target_weight=0;
        step.network_wdl=step.search_wdl={.5,0,.5};step.visits.assign(game.actions(),0);step.visits[action]=1;
        if(step.trainable){positions.push_back(game);history.push_back(.25);}
        game.play(action);record.steps.push_back(std::move(step));
    }
    check(game.finished(),"Reanalysis fixture must complete a real game");
    record.winner=game.winner();record.reason=game.reason();
    SearchSettings settings{12,2,1.1,1,0,1,false,12};settings.graph_search=true;
    settings.subtree_value_bias_factor=.45;settings.subtree_value_bias_table_num_shards=4;
    Search search(evaluator,settings,13);SelfplaySearchConfig limits{12,6,1,0,true,false,.9,3,6,1};
    ReanalysisConfig config;config.enabled=true;config.proportion=1;std::mt19937_64 rng(14);
    size_t calls=0;
    reanalyze_positions(record,positions,history,limits,{.5,1},config,search,[&](const Game& position) {
        Game replay(size,size,rule);replay.set_pda(.5,1);
        for(int i=0;i<position.turn();++i)replay.play(moves[i]);
        check(position.observation()==replay.observation() && position.history_identity()==replay.history_identity(),"Reanalysis must restore full rule and NN history");
        for(int a=0;a<position.actions();++a)check(position.legal(a)==replay.legal(a),"Reanalysis history changed move legality");
        check(AlphaZeroState(position,evaluator).graph_key()==AlphaZeroState(replay,evaluator).graph_key(),"Reanalysis history changed graph identity");
        ++calls;return 0;
    },rng,[]{return false;});
    check(calls==positions.size(),"All cheap positions must be reanalyzed, excluding the opening");
    for(size_t i=0;i<positions.size();++i) {
        const auto& step=record.steps[i+2];check(step.reanalyzed&&step.action==moves[i+2],"Reanalysis must preserve real actions");
        for(int a=0;a<positions[i].actions();++a)if(!positions[i].legal(a))
            check(step.policy[a]==0 && step.visits[a]==0 && step.q_visits[a]==0,"Reanalysis trained on a history-illegal move");
    }
}
int main(int argc,char** argv) {
    Game x(10,10,Rule::XIANGQI);check(legal(x).size()==44,"Xiangqi initial legal count");
    check(perft(x,2)==1920,"Xiangqi initial depth-two perft");
    Game repeat=x;for(int i=0;i<2;++i)for(auto m:{std::array<int,4>{1,0,2,2},{1,9,2,7},{2,2,1,0},{2,7,1,9}})repeat.play(xiangqi_action(m[0],m[1],m[2],m[3]));
    check(repeat.finished()&&repeat.winner()==0&&repeat.reason()==Reason::REPETITION,"Neutral repetition draw");
    RuleOptions material;material.fen="5k3/9/9/9/9/9/9/9/9/4K4 w - - 0 1";
    Game bare_kings(10,10,Rule::XIANGQI,material);
    check(bare_kings.finished()&&bare_kings.winner()==0&&bare_kings.reason()==Reason::INSUFFICIENT_MATERIAL,"Bare kings have no repetition history; material ends the game");
    RuleOptions limit;limit.fen=x.fen();limit.fen.replace(limit.fen.rfind("0 1"),3,"119 61");
    Game move_limit(10,10,Rule::XIANGQI,limit);move_limit.play(xiangqi_action(1,0,2,2));
    check(move_limit.finished()&&move_limit.reason()==Reason::MOVE_LIMIT,"60 move limit has a distinct reason");
    limit.fen=x.fen();limit.fen.replace(limit.fen.rfind("0 1"),3,"112 57");
    Game simultaneous(10,10,Rule::XIANGQI,limit);
    for(int i=0;i<2;++i)for(auto m:{std::array<int,4>{1,0,2,2},{1,9,2,7},{2,2,1,0},{2,7,1,9}})simultaneous.play(xiangqi_action(m[0],m[1],m[2],m[3]));
    check(simultaneous.finished()&&simultaneous.reason()==Reason::REPETITION,"Repetition retains upstream priority when move limit is reached too");
    RuleOptions check_options;check_options.fen="5k3/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1";
    Game checker(10,10,Rule::XIANGQI,check_options);
    for(int i=0;i<2;++i)for(auto m:{std::array<int,4>{4,8,5,8},{5,9,4,9},{5,8,4,8},{4,9,5,9}})checker.play(xiangqi_action(m[0],m[1],m[2],m[3]));
    check(checker.finished()&&checker.winner()==-1&&checker.reason()==Reason::REPETITION,"Perpetual checker loses");
    Game go(5,5,Rule::GO);for(int a:{1,0,5})go.play(a);
    check(go.board().cells[0]==0,"Go captures corner stone");
    check(!go.legal(0),"Go suicide rejected");
    go.play(25);Game copied=go;copied.play(25);
    check(copied.finished()&&!go.finished(),"Go copied pass history independent");
    Game ko(5,5,Rule::GO);for(int a:{5,6,1,10,7,12,24,16,11})ko.play(a);
    check(!ko.legal(6),"Immediate ko recapture rejected");
    ko.play(4);ko.play(9);check(ko.legal(6),"Ko recapture after intervening moves");
    for(const char* rules:{"japanese","tromp-taylor"}) {
        RuleOptions options;options.go_rules=rules;Game game(5,6,Rule::GO,options);auto obs=game.observation();
        RandomBackend az(6,19,game.spec());muzero::RandomBackend mz(6,19,game.spec());
        auto a=az.evaluate({&obs})[0];auto m=mz.initial({&obs})[0];
        check(a.wdl==m.evaluation.wdl && a.logits==m.evaluation.logits,"AZ/MZ random outputs share source distribution");
        bool nr=std::string(rules)=="japanese";
        check((a.wdl[3]>0)==nr && a.wdl[1]>0,"Random Go separates ordinary draw and rule-dependent NR");
        for(int action:{0,1,36}) {
            auto next=mz.recurrent({{m.latent,action}})[0];
            check((next.evaluation.wdl[3]>0)==nr && next.evaluation.wdl[1]>0,"Random latent steps retain the root NR rule mask");
            check(next.evaluation.logits!=m.evaluation.logits,"Random recurrent requests advance server RNG");m=std::move(next);
        }
    }
    for(auto rule:{Rule::GO,Rule::XIANGQI}) {
        int size=rule==Rule::GO?5:10;Game game(size,size,rule);
        auto obs=game.observation();check(obs.size()==size_t(game.planes()*size*size+game.globals()),"Observation layout");
        std::vector<std::unique_ptr<Backend>> backends;backends.push_back(std::make_unique<RandomBackend>(size,7,game.spec()));
        BatchEvaluator eval(std::move(backends),"random",size,8,0,32,false,0,7);
        bool rejected=false;try{eval.evaluate(Game(6,6,Rule::GO).observation());}catch(const std::runtime_error&){rejected=true;}
        check(rejected,"AZ rejects another input canvas before symmetry mapping");
        SearchSettings settings{7,1,1.1,1,0,1,false,8};
        Search az(eval,settings,7);NoResult no_result;
        auto neutral=az.run(no_result,0);check(neutral.search_wdl==WDL{0,0,0,1},"No-result is neutral, not ordinary draw");auto result=az.run(game,0);check(game.legal(result.action),"AZ move legality");
        GumbelSettings gs;gs.max_num_considered_actions=4;
        GumbelSearch<true> gaz(eval,settings,gs,7);result=gaz.run(game,0);check(game.legal(result.action),"Gumbel move legality");
        std::vector<std::unique_ptr<muzero::Backend>> models;models.push_back(std::make_unique<muzero::RandomBackend>(size,7,game.spec()));
        muzero::BatchEvaluator meval(std::move(models),size,8,0,32,false,0,7,32);
        rejected=false;try{meval.initial(Game(6,6,Rule::GO).observation());}catch(const std::runtime_error&){rejected=true;}
        check(rejected,"MuZero rejects another input canvas");
        DenseSearch mz(meval,settings,7);result=mz.run(game,0);check(game.legal(result.action),"MuZero move legality");mz.verify_storage(game);
        muzero::GumbelSearch<true> gmz(meval,settings,gs,7);result=gmz.run(game,0);check(game.legal(result.action),"MuZero Gumbel legality");
        reanalysis_history(rule,eval);
        eval.finish();meval.finish();
        if(argc>1) {
            FinishedGame record{};record.id=0;record.seed=9;record.size=size;record.canvas=size;record.rule=rule;record.options=game.options();
            std::mt19937_64 random(9);
            while(!game.finished()) {
                check(game.turn()<10000,"Random rule trajectory did not terminate");
                auto moves=legal(game);int action=moves[random()%moves.size()];
                Step step{};step.player=game.player();step.action=action;step.simulations=1;step.temperature=1;step.observation=game.observation();
                step.policy.assign(game.actions(),0);step.policy[action]=1;step.visits.assign(game.actions(),0);step.visits[action]=1;
                step.q_values.assign(game.actions(),0);step.q_visits.assign(game.actions(),0);
                step.policy_target=quantize_policy(step.policy);step.network_wdl=step.search_wdl={.5,0,.5};
                game.play(action);step.reward=game.finished()?game.winner()*step.player:0;record.steps.push_back(std::move(step));
            }
            check(game.reason()!=Reason::NO_RESULT,"Fixture must have ordinary terminal labels");
            record.winner=game.winner();record.reason=game.reason();record.final_player=game.player();record.final_observation=game.observation();
            const auto& first=record.steps.front();
            record.side_positions.push_back({first.player,first.observation,first.policy_target,first.visits,first.search_wdl});
            record.side_positions.back().q_values=first.q_values;
            record.side_positions.back().q_visits=first.q_visits;
            for(int corruption=0;corruption<3;++corruption) {
                auto invalid=record;auto& obs=invalid.side_positions.front().observation;
                int globals=game.planes()*size*size,pda=rule==Rule::GO?15:3;
                obs[globals+(corruption==2?0:pda+corruption)]=corruption==2?std::numeric_limits<float>::quiet_NaN():.25f;
                auto dir=std::filesystem::path(argv[1])/(std::string(RULE_NAMES[int(rule)])+"-invalid-"+std::to_string(corruption));
                bool rejected=false;
                try {
                    RecordWriter writer(dir,{"test","invalid","model","config","source",1,0},1000,1,1,9);
                    writer.enqueue(invalid);writer.finish();
                } catch(const std::runtime_error& e) {
                    rejected=std::string(e.what())==(corruption==2?"Nonfinite side globals":"Side rows must clear PDA conditioning");
                }
                check(rejected,"Writer accepted invalid Go/Xiangqi side globals");
            }
            for(int k:{0,3}) {
                auto dir=std::filesystem::path(argv[1])/(std::string(RULE_NAMES[int(rule)])+(k?"-mz":"-az"));
                RecordWriter writer(dir,{"test","2","model","config","source",1,0,k,false,false,true},64,1,1,9);writer.enqueue(record);writer.finish();
            }
            std::cout<<RULE_NAMES[int(rule)]<<" plies "<<game.turn()<<'\n';
        }
    }
}
