#include "etazero/record.h"
#include "etazero/search.h"
#include <iostream>
#include <limits>
using namespace etazero;
int main(int argc,char** argv) {
    if(argc>1 && std::string(argv[1])=="--first-limit") {
        size_t rows;double minimum,uniform;
        while(std::cin>>rows>>minimum>>uniform)std::cout<<first_file_row_limit(rows,minimum,uniform)<<'\n';
        return 0;
    }
    std::mt19937_64 rng(123);
    for(float x:{-1.f,-.25f,0.f,.25f,1.f})
        if(quantize_q_value(x,rng)!=int(x*32000))throw std::runtime_error("Q exact/cap rounding");
    bool low=false,high=false;
    for(int i=0;i<256;++i) {
        int x=quantize_q_value(-.2500125f,rng);
        if(x==-8001)low=true;else if(x==-8000)high=true;else throw std::runtime_error("Q fractional bounds");
    }
    if(!low||!high)throw std::runtime_error("Q stochastic rows collapsed");
    bool rejected=false;try{quantize_q_value(std::numeric_limits<float>::quiet_NaN(),rng);}catch(const std::exception&){rejected=true;}
    if(!rejected)throw std::runtime_error("Q nonfinite accepted");
    auto path=argc>1?std::filesystem::path(argv[1]):std::filesystem::temp_directory_path()/
        ("etazero_record_contract_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Game game(5,6,Rule::RENJU);FinishedGame record{};
    record.id=0;record.seed=7;record.size=5;record.canvas=6;record.rule=Rule::RENJU;
    for(int action:{0,6,1,7,2,8,3,9,4}) {
        Step step{};step.player=game.player();step.action=action;step.simulations=1;step.temperature=1;
        step.observation=game.observation();step.policy.assign(36,0);step.policy[action]=1;
        step.policy_target=quantize_policy(step.policy);step.visits.assign(36,0);step.visits[action]=1;
        step.q_values.assign(36,0);step.q_values[action]=.25f;
        step.q_visits.assign(36,0);step.q_visits[action]=2;
        step.search_wdl={.2,.3,.5};
        if(game.turn()==2) {
            SidePosition side{game.player(),game.observation(),step.policy_target,step.visits,{.2,.3,.5},2,2};
            side.q_values=step.q_values;side.q_visits=step.q_visits;
            record.side_positions.push_back(side);
        }
        game.play(action);step.reward=game.finished()?game.winner()*step.player:0;record.steps.push_back(std::move(step));
    }
    record.winner=game.winner();record.reason=game.reason();record.final_player=game.player();record.final_observation=game.observation();
    bool shards=argc>2 && std::string(argv[2])=="shards";
    if(argc==3 && !shards)record.side_positions.clear();
    if(argc>3 || shards) {
        auto& first=record.steps.front();first.row_repeats=64;first.target_weight=64;
        first.q_values[first.action]=-.2500125f;first.q_visits[first.action]=40000;
        auto& side=record.side_positions.front();side.row_repeats=64;side.target_weight=64;
        for(size_t a=0;a<side.q_values.size();++a)if(side.q_visits[a]) {side.q_values[a]=-.2500125f;side.q_visits[a]=40000;}
    }
    if(shards) {
        record.steps.front().row_repeats=7;record.steps.front().target_weight=7;
        record.steps[3].row_repeats=0;record.steps[3].target_weight=0;
        record.side_positions.front().row_repeats=5;record.side_positions.front().target_weight=5;
    }
    size_t limit=shards && argc>3?std::stoull(argv[3]):1000;
    double first_min=shards && argc>4?std::stod(argv[4]):1;
    RecordWriter writer(path,{"test","2","model","config","source",1,0},limit,1,first_min,123);
    int games=shards && argc>5?std::stoi(argv[5]):1;
    for(int g=0;g<games;++g) {record.id=g;record.seed=7+13*g;writer.enqueue(record);}
    writer.finish();
    if(argc==1)std::filesystem::remove_all(path);
}
