#include "game/game.h"
// Time the production inference path, including packing, transfers and outputs.
#include "neuralnet/backend_factory.h"
#include "etazero/schema.h"
#include <ATen/Parallel.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace etazero;
using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}

double validate(const std::vector<Evaluation>& values, int batch, int actions) {
    if(values.size()!=static_cast<size_t>(batch))
        throw std::runtime_error("Benchmark output batch mismatch");
    double checksum=0;
    auto check=[&](double x) {
        if(!std::isfinite(x))throw std::runtime_error("Non-finite benchmark output");
        checksum+=x;
    };
    for(const auto& value:values) {
        if(!value.has_auxiliary || value.logits.size()!=static_cast<size_t>(actions) ||
           value.optimistic_logits.size()!=static_cast<size_t>(actions))
            throw std::runtime_error("Benchmark output contract mismatch");
        for(double x:value.logits)check(x);
        for(double x:value.optimistic_logits)check(x);
        for(double x:value.wdl)check(x);
        check(value.shortterm_value_stdev);
    }
    return checksum;
}

int main(int argc, char** argv) {
    try {
        if(argc<9)throw std::runtime_error(
            "Usage: inference_benchmark DEVICE PRECISION CANVAS BATCHES REPEATS SECONDS WARMUP MODEL...");
        at::set_num_threads(1);
        const std::string device=argv[1],precision=argv[2];
        const int canvas=std::stoi(argv[3]),repeats=std::stoi(argv[5]),warmup=std::stoi(argv[7]);
        const double duration=std::stod(argv[6]);
        if(device.rfind("cuda:",0)!=0 || repeats<1 || warmup<1 || !std::isfinite(duration) || duration<=0)
            throw std::runtime_error("Use CUDA, positive repeats/seconds/warmup");
        std::vector<int> batches;
        std::istringstream list(argv[4]);std::string item;
        while(std::getline(list,item,',')) {
            int n=std::stoi(item);
            if(n<1)throw std::runtime_error("Batch sizes must be positive");
            batches.push_back(n);
        }
        if(batches.empty())throw std::runtime_error("No batch sizes supplied");
        const int max_batch=*std::max_element(batches.begin(),batches.end());
        std::vector<std::unique_ptr<Backend>> backends;
        std::cout<<std::setprecision(17);
        for(int i=8;i<argc;++i) {
            auto shared=std::make_shared<TorchBackend::LoadedModel>();
            auto backend=make_backend(argv[i],device,canvas,max_batch,precision,shared);
            auto started=Clock::now();backend->initialize();
            std::cout<<"{\"event\":\"initialize\",\"model\":"<<i-8
                     <<",\"seconds\":"<<seconds(started)
                     <<",\"load_seconds\":"<<shared->timing.load_seconds
                     <<",\"weights_seconds\":"<<shared->timing.weights_seconds
                     <<",\"graph_seconds\":"<<shared->timing.graph_seconds<<"}"<<std::endl;
            backends.push_back(std::move(backend));
        }
        std::vector<std::vector<std::vector<float>>> observations(backends.size());
        for(size_t model=0;model<backends.size();++model) {
            auto spec=backends[model]->spec();
            Rule rule=parse_rule(spec.kind==GameKind::GOMOKU?"freestyle":spec.name);
            for(int i=0;i<max_batch;++i) {
                Game game(canvas,canvas,rule);
                // Fixed legal prefixes, outside the measured interval.
                for(int ply=0;ply<2 && !game.finished();++ply) {
                    std::vector<int> legal;
                    for(int a:game.static_actions())if(game.legal(a))legal.push_back(a);
                    game.play(legal[(i+ply*(canvas*canvas/2))%legal.size()]);
                }
                observations[model].push_back(game.observation());
            }
        }
        // Reverse both sweep and model order between repeats to balance drift.
        for(int repeat=0;repeat<repeats;++repeat) {
            for(size_t step=0;step<batches.size();++step) {
                int batch=batches[repeat%2?batches.size()-1-step:step];
                int bucket=1;while(bucket<batch)bucket=std::min(max_batch,bucket*2);
                for(size_t turn=0;turn<backends.size();++turn) {
                    size_t index=repeat%2?backends.size()-1-turn:turn;
                    auto& backend=*backends[index];
                    int actions=backend.spec().actions(canvas);
                    InferenceInputs inputs;
                    for(int i=0;i<batch;++i)inputs.push_back(&observations[index][i]);
                    auto warmed=Clock::now();int warmups=0;
                    do {
                        validate(backend.evaluate(inputs),batch,actions);
                        ++warmups;
                    } while(warmups<warmup || seconds(warmed)<0.2);
                    uint64_t iterations=0;
                    std::vector<Evaluation> last;
                    auto started=Clock::now();double elapsed;
                    do {
                        last=backend.evaluate(inputs);
                        ++iterations;
                        elapsed=seconds(started);
                    } while(elapsed<duration);
                    const double checksum=validate(last,batch,actions);
                    std::cout<<"{\"event\":\"measure\",\"model\":"<<index
                             <<",\"repeat\":"<<repeat<<",\"batch\":"<<batch
                             <<",\"executed_batch\":"<<(is_cuda_model(argv[8+index])?batch:bucket)<<",\"iterations\":"<<iterations
                             <<",\"seconds\":"<<elapsed<<",\"batch_ms\":"<<1000*elapsed/iterations
                             <<",\"positions_per_second\":"<<batch*iterations/elapsed
                             <<",\"checksum\":"<<checksum<<"}"<<std::endl;
                }
            }
        }
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
