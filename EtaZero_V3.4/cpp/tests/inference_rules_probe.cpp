#include "neuralnet/backend_factory.h"
#include "search/search.h"
#include <iostream>
#include <iomanip>
using namespace etazero;
void report(const Evaluation& e,const Game& game) {
    std::vector<int> legal;for(int a=0;a<game.actions();++a)if(game.legal(a))legal.push_back(a);
    auto ordinary=source_policy_probabilities(e,legal,1,0),optimistic=source_policy_probabilities(e,legal,1,1);
    std::cout<<std::setprecision(17)<<"{\"wdl\":[";
    for(int i=0;i<4;++i){if(i)std::cout<<',';std::cout<<e.wdl[i];}
    std::cout<<"],\"optimism_allowed\":"<<(e.policy_optimism_allowed?"true":"false")
             <<",\"same_policy\":"<<(ordinary==optimistic?"true":"false")<<"}\n";
}
int main(int argc,char** argv){try{
    if(argc<6||argc>7)throw std::runtime_error("MODEL ALGORITHM RULES DEVICE PRECISION [HUMAN_PROFILE]");
    RuleOptions rules;rules.go_rules=argv[3];Game game(5,6,Rule::GO,rules);
    if(std::string(argv[2])=="alphazero") {
        auto backend=make_backend(argv[1],argv[4],6,2,argv[5],nullptr,argc==7?argv[6]:"");
        BatchEvaluator evaluator(std::move(backend),argv[1],6,2,0);
        report(evaluator.evaluate_position(game.observation(),game.player()),game);
    } else {
        auto backend=make_muzero_backend(argv[1],argv[4],6,2,argv[5]);backend->initialize();
        auto obs=game.observation();auto out=backend->initial({&obs}).at(0);report(out.evaluation,game);
        for(int action:{0,1,2}){out=backend->recurrent({{out.latent,action}}).at(0);report(out.evaluation,game);}
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
