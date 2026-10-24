#include "neuralnet/muzero/cuda_backend.h"
#include "game/game.h"
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace etazero;
int main(int argc,char** argv){try{
    if(argc!=7)throw std::runtime_error("Usage: muzero_cuda_probe MODEL GAME CANVAS BATCH PRECISION INPUT_FILE");
    auto spec=game_spec(argv[2]);int canvas=std::stoi(argv[3]),batch=std::stoi(argv[4]);
    muzero::CudaBackend backend(argv[1],canvas,batch,0,argv[5],spec);backend.initialize();
    std::vector<std::vector<float>> rows;InferenceInputs inputs;
    for(int i=0;i<batch;++i){Game game(spec.kind==GameKind::XIANGQI?10:canvas-i%2,canvas,parse_rule(spec.kind==GameKind::GOMOKU?"freestyle":spec.name));rows.push_back(game.observation());}
    std::ofstream file(argv[6],std::ios::binary);for(auto& row:rows){inputs.push_back(&row);file.write(reinterpret_cast<const char*>(row.data()),row.size()*sizeof(float));}file.close();
    auto initial=backend.initial(inputs),current=initial;std::cout<<std::setprecision(17);
    auto output=[&]{for(auto& out:current){for(float x:out.latent->copy_to_cpu())std::cout<<x<<' ';auto& e=out.evaluation;
        for(auto x:e.scaled_policy_logits())std::cout<<x<<' ';for(auto x:e.wdl)std::cout<<x<<' ';for(auto x:e.scaled_policy_logits(true))std::cout<<x<<' ';std::cout<<e.shortterm_value_stdev<<'\n';}};
    output();
    for(int step=0;step<4;++step){
        std::vector<muzero::InferenceAction> actions;
        for(int i=0;i<batch;++i){int action=spec.kind==GameKind::GO && step==2?canvas*canvas:(step==3?0:step);
            actions.push_back({(step==3?initial:current)[i].latent,action});}
        current=backend.recurrent(actions);output();
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
