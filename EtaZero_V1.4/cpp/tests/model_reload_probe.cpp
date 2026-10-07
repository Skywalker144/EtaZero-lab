#include "etazero/torch_backend.h"
#include "etazero/muzero/torch_backend.h"
#include <ATen/Parallel.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <iomanip>
#include <iostream>

namespace {
void append(std::vector<double>& result,const etazero::Evaluation& e) {
    result.insert(result.end(),e.logits.begin(),e.logits.end());
    result.insert(result.end(),e.wdl.begin(),e.wdl.end());
    result.insert(result.end(),e.optimistic_logits.begin(),e.optimistic_logits.end());
    result.push_back(e.shortterm_value_stdev);
}
void append(std::vector<double>& result,const etazero::muzero::InferenceOutput& e) {
    append(result,e.evaluation);auto latent=e.latent->copy_to_cpu();
    result.insert(result.end(),latent.begin(),latent.end());
}
}
int main(int argc,char** argv) {
    try {
        if(argc<6)throw std::runtime_error("Usage: model_reload_probe ALGORITHM DEVICE PRECISION MODELS...");
        at::set_num_threads(1);std::cout<<std::setprecision(10);
        auto prepared=std::make_shared<etazero::PreparedModel>();
        std::vector<std::vector<float>> obs(5,std::vector<float>(5*36+6));
        etazero::InferenceInputs inputs;
        for(int i=0;i<5;++i) {
            for(int y=0;y<6;++y)for(int x=0;x<6;++x)obs[i][y*6+x]=1;
            obs[i][36+i]=1;obs[i][72+6+i]=1;inputs.push_back(&obs[i]);
        }
        std::cout<<'[';
        for(int round=4;round<argc;++round) {
            std::vector<double> values;
            // Both services use the same prepared weights but independent graphs.
            if(std::string(argv[1])=="alphazero") {
                etazero::TorchBackend first(argv[round],argv[2],6,5,argv[3],prepared);
                etazero::TorchBackend second("unused",argv[2],6,5,argv[3],prepared);
                first.initialize();second.initialize();
                for(const auto& e:first.evaluate(inputs))append(values,e);
                for(const auto& e:second.evaluate({inputs[0]}))append(values,e);
            } else {
                etazero::muzero::TorchBackend first(argv[round],argv[2],6,5,argv[3],prepared);
                etazero::muzero::TorchBackend second("unused",argv[2],6,5,argv[3],prepared);
                auto roots=first.initial(inputs);
                for(const auto& e:roots)append(values,e);
                std::vector<etazero::muzero::InferenceAction> actions;
                for(int i=0;i<5;++i)actions.push_back({roots[i].latent,i+1});
                for(const auto& e:first.recurrent(actions))append(values,e);
                for(const auto& e:second.initial({inputs[0]}))append(values,e);
            }
            prepared->offload();c10::cuda::CUDACachingAllocator::emptyCache();
            auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(torch::Device(argv[2]).index());
            if(round>4)std::cout<<',';
            std::cout<<"{\"reused\":"<<(prepared->timing.reused_runtime?"true":"false")
                     <<",\"graphs_seconds\":"<<prepared->timing.graph_seconds
                     <<",\"allocated_after_offload\":"<<stats.allocated_bytes[0].current<<",\"values\":[";
            for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}
            std::cout<<"]}";
        }
        std::cout<<"]\n";return 0;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
