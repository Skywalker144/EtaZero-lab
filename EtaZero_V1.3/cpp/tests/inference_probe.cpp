// Native parity/lifetime probe; no search or benchmark path depends on this.
#include "etazero/torch_backend.h"
#include "etazero/schema.h"
#include <ATen/Parallel.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>
using namespace etazero;
void output(const std::vector<Evaluation>& values) {
    std::cout<<'[';
    for(size_t i=0;i<values.size();++i) {
        if(i)std::cout<<',';
        const auto& e=values[i];std::cout<<'[';
        for(auto x:e.logits)std::cout<<x<<',';
        for(auto x:e.wdl)std::cout<<x<<',';
        for(auto x:e.optimistic_logits)std::cout<<x<<',';
        std::cout<<e.shortterm_value_stdev<<']';
    }
    std::cout<<']';
}
void close(const std::vector<Evaluation>& a,const std::vector<Evaluation>& b) {
    if(a.size()!=b.size())throw std::runtime_error("Native parity batch mismatch");
    auto compare=[](double x,double y) {
        if(!std::isfinite(x) || !std::isfinite(y) || std::abs(x-y)>3e-3+3e-3*std::abs(y))
            throw std::runtime_error("Native variable-batch/server parity mismatch");
    };
    for(size_t i=0;i<a.size();++i) {
        if(!a[i].has_auxiliary)throw std::runtime_error("Native auxiliary output lost");
        for(size_t j=0;j<a[i].logits.size();++j) {
            compare(a[i].logits[j],b[i].logits[j]);compare(a[i].optimistic_logits[j],b[i].optimistic_logits[j]);
        }
        for(int j=0;j<3;++j)compare(a[i].wdl[j],b[i].wdl[j]);
        compare(a[i].shortterm_value_stdev,b[i].shortterm_value_stdev);
    }
}
int main(int argc,char** argv) {
    try {
        if(argc!=4)throw std::runtime_error("Usage: inference_probe MODEL DEVICE PRECISION");
        at::set_num_threads(1);
        auto shared=std::make_shared<TorchBackend::LoadedModel>();
        TorchBackend first(argv[1],argv[2],6,5,argv[3],shared);
        TorchBackend second(std::string(argv[1])+".must-not-be-opened",argv[2],6,5,argv[3],shared);
        first.initialize();second.initialize();
        std::vector<std::vector<float>> obs(5,std::vector<float>(INPUT_PLANES*36+GLOBAL_FEATURES,0));
        for(int i=0;i<5;++i) {
            int size=5+i%2;
            for(int y=0;y<size;++y)for(int x=0;x<size;++x)obs[i][y*6+x]=1;
            obs[i][36+i]=1;obs[i][72+6+i]=1;
            auto globals=obs[i].data()+INPUT_PLANES*36;
            globals[0]=i%2;globals[1]=i%3==0;globals[2]=i%3==0?-1:0;
            globals[3]=i%3==0;globals[4]=i%2;globals[5]=.25f*i;
        }
        InferenceInputs all;for(auto& row:obs)all.push_back(&row);
        auto original=first.evaluate(all);
        auto saved=original;
        std::cout<<std::setprecision(17)<<'[';
        int turn=0;
        for(int n:{1,3,5,2,1,4,3}) {
            InferenceInputs inputs;
            for(int i=0;i<n;++i)inputs.push_back(&obs[(turn+i)%5]);
            auto actual=first.evaluate(inputs);close(actual,second.evaluate(inputs));
            if(turn++)std::cout<<',';
            output(actual);
        }
        std::cout<<"]\n";
        close(original,saved);close(first.evaluate(all),saved);
        for(auto invalid:{InferenceInputs{},InferenceInputs{&obs[0],&obs[0],&obs[0],&obs[0],&obs[0],&obs[0]}}) {
            bool rejected=false;try{first.evaluate(invalid);}catch(const std::runtime_error&){rejected=true;}
            if(!rejected)throw std::runtime_error("Native invalid batch accepted");
        }
        auto malformed=obs[0];malformed.pop_back();
        bool rejected=false;try{first.evaluate({&malformed});}catch(const std::runtime_error&){rejected=true;}
        if(!rejected)throw std::runtime_error("Native invalid input shape accepted");
        std::exception_ptr failures[2];
        auto run=[&](int owner,TorchBackend& backend) {
            try {
                for(int iteration=0;iteration<4;++iteration) {
                    for(int n:{3,1,5,2}) {
                        InferenceInputs inputs(all.begin(),all.begin()+n);
                        close(backend.evaluate(inputs),std::vector<Evaluation>(saved.begin(),saved.begin()+n));
                    }
                }
            } catch(...) {failures[owner]=std::current_exception();}
        };
        std::thread a(run,0,std::ref(first)),b(run,1,std::ref(second));a.join();b.join();
        for(auto error:failures)if(error)std::rethrow_exception(error);
        return 0;
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
