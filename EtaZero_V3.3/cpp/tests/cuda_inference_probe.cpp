#include "etazero/cuda_backend.h"
#include "etazero/game.h"
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace etazero;
int main(int argc,char** argv){try{
    if(argc<6 || argc>8)throw std::runtime_error("Usage: cuda_inference_probe MODEL GAME CANVAS BATCH PRECISION [INPUTS_FILE [HUMAN_PROFILE]]");
    auto spec=game_spec(argv[2]);int canvas=std::stoi(argv[3]),batch=std::stoi(argv[4]);
    CudaBackend backend(argv[1],canvas,batch,0,argv[5],spec,argc==8?argv[7]:"");backend.initialize();
    std::vector<std::vector<float>> rows;InferenceInputs inputs;
    for(int i=0;i<batch;++i){
        Game game(spec.kind==GameKind::XIANGQI?10:canvas-1,canvas,parse_rule(spec.kind==GameKind::GOMOKU?"freestyle":spec.name));
        rows.push_back(game.observation());
        backend.append_metadata(rows.back(),i%2?-1:1);
    }
    if(argc>=7){std::ofstream file(argv[6],std::ios::binary);for(const auto& row:rows)file.write(reinterpret_cast<const char*>(row.data()),row.size()*sizeof(float));}
    for(auto& row:rows)inputs.push_back(&row);
    auto outputs=backend.evaluate(inputs);std::cout<<std::setprecision(17);
    for(auto& out:outputs){for(auto x:out.scaled_policy_logits())std::cout<<x<<' ';for(auto x:out.wdl)std::cout<<x<<' ';
        for(auto x:out.scaled_policy_logits(true))std::cout<<x<<' ';std::cout<<out.shortterm_value_stdev<<'\n';}
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
