// Development parity probe; deliberately not an application search/runtime path.
#include "etazero/muzero/torch_backend.h"
#include "etazero/schema.h"
#include <ATen/Parallel.h>
#include <iostream>
#include <iomanip>
using etazero::muzero::TorchBackend;
template<class T> void array(const std::vector<T>& values) {
    std::cout<<'[';
    for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}
    std::cout<<']';
}
void output(const std::vector<TorchBackend::Output>& values) {
    std::cout<<'[';
    for(size_t i=0;i<values.size();++i) {
        if(i)std::cout<<',';const auto& e=values[i].evaluation;
        std::cout<<"{\"latent\":";array(values[i].latent->copy_to_cpu());
        std::cout<<",\"policy\":";array(e.logits);
        std::cout<<",\"wdl\":";array(std::vector<double>(e.wdl.begin(),e.wdl.end()));
        std::cout<<",\"optimistic\":";array(e.optimistic_logits);
        std::cout<<",\"stdev\":"<<e.shortterm_value_stdev<<'}';
    }
    std::cout<<']';
}
template<class F> void rejects(F f) {
    try{f();}catch(const std::exception&){return;}
    throw std::runtime_error("MuZero invalid input was accepted");
}
int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("Usage: muzero_inference_probe MODEL DEVICE PRECISION");
        at::set_num_threads(1);
        TorchBackend backend(argv[1],argv[2],6,2,argv[3]);
        // A detected module must work without reopening its source file.
        auto detected=torch::jit::load(argv[1],torch::Device(argv[2]));
        TorchBackend other(std::string(argv[1])+".must-not-be-opened",argv[2],6,2,argv[3],std::move(detected));
        std::vector<std::vector<float>> obs(2,std::vector<float>(5*36+6,0));
        for(int i=0;i<2;++i) {
            for(int y=0;y<5+i;++y)for(int x=0;x<5+i;++x)obs[i][y*6+x]=1;
            obs[i][36+0]=1;obs[i][72+6]=1;
            obs[i][5*36+1]=1;obs[i][5*36+2]=-1;obs[i][5*36+3]=1;
            obs[i][5*36+4]=1;obs[i][5*36+5]=0.5;
        }
        auto initial=backend.initial({&obs[0],&obs[1]});
        auto reused=other.initial({&obs[0],&obs[1]});
        for(size_t i=0;i<initial.size();++i)
            if(initial[i].evaluation.logits!=reused[i].evaluation.logits || initial[i].evaluation.wdl!=reused[i].evaluation.wdl)
                throw std::runtime_error("Preloaded MuZero inference differs from file-loaded inference");
        rejects([&]{backend.initial({});});
        rejects([&]{backend.initial({&obs[0],&obs[1],&obs[0]});});
        auto empty=obs[0];std::fill(empty.begin(),empty.begin()+36,0);
        rejects([&]{backend.initial({&empty});});
        rejects([&]{backend.recurrent({{initial[0].latent,5}});}); // padding
        rejects([&]{backend.recurrent({{initial[0].latent,-1}});});
        rejects([&]{other.recurrent({{initial[0].latent,0}});});
        rejects([&]{backend.recurrent({{nullptr,0}});});
        // Occupied actions remain in the latent action space. A backend must
        // not reconstruct/check the real board at recurrent depths.
        auto repeated=backend.recurrent({{initial[0].latent,0},{initial[1].latent,0}});
        (void)repeated;
        std::cout<<std::setprecision(9)<<'[';output(initial);
        auto current=initial;
        for(int step=0;step<3;++step) {
            current=backend.recurrent({{current[0].latent,step+1},{current[1].latent,step+7}});
            std::cout<<',';output(current);
        }
        // Sibling expansion must not mutate an already cached parent latent.
        auto repeat=backend.recurrent({{initial[0].latent,1},{initial[1].latent,7}});
        std::cout<<',';output(repeat);std::cout<<"]\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
