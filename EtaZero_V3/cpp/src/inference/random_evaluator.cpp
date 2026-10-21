#include "etazero/random_evaluator.h"
#include "etazero/game_spec.h"
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace etazero {
namespace {
uint64_t mix(uint64_t n) {
    n += 0x9e3779b97f4a7c15ULL; n = (n^(n>>30))*0xbf58476d1ce4e5b9ULL;
    n = (n^(n>>27))*0x94d049bb133111ebULL; return n^(n>>31);
}
}
RandomBackend::RandomBackend(int canvas,uint64_t seed,GameSpec spec) : Backend(spec),canvas_(canvas),seed_(mix(seed)) {
    if (canvas < 5 || canvas > 25) throw std::runtime_error("Invalid random evaluator canvas");
}
std::vector<Evaluation> RandomBackend::evaluate(const InferenceInputs& inputs) {
    std::vector<Evaluation> result;
    result.reserve(inputs.size());
    for (auto observation : inputs) {
        auto spec=spec_;
        if (observation->size() != static_cast<size_t>(spec.planes*canvas_*canvas_ + spec.globals))
            throw std::runtime_error("Random observation shape mismatch");
        uint64_t key = seed_;
        const size_t spatial = spec.planes*canvas_*canvas_;
        for (size_t i=0;i<spatial;++i) {
            float value=(*observation)[i];
            key = mix(key ^ static_cast<uint64_t>(static_cast<int>(value)+1));
        }
        // Global features include continuous, signed PDA conditioning. Hash the
        // exact float bits so fractional advantages remain distinct and seeded
        // outputs stay independent of batch/thread scheduling.
        for (size_t i=spatial;i<observation->size();++i) {
            float value=(*observation)[i];
            uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
            key = mix(key ^ bits);
        }
        std::mt19937_64 random(key); std::normal_distribution<double> gaussian;
        Evaluation e; e.logits.resize(spec.actions(canvas_));
        for (auto& logit : e.logits) logit = gaussian(random);
        double w=std::exp(0.2*gaussian(random)), d=std::exp(0.2*gaussian(random)), l=std::exp(0.2*gaussian(random));
        double sum=w+d+l;e.wdl={w/sum,d/sum,l/sum};
        result.push_back(std::move(e));
    }
    return result;
}
}
