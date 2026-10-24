#include "neuralnet/random_evaluator.h"
#include "game/game_spec.h"
#include "core/rand.h"
#include <cmath>
#include <stdexcept>
namespace etazero {
Evaluation random_evaluation(Rand& random,const std::vector<uint8_t>& mask,bool allow_no_result) {
    Evaluation e;e.logits.resize(mask.size());
    for(size_t i=0;i<mask.size();++i)if(mask[i])e.logits[i]=static_cast<float>(random.nextGaussian());
    // nneval.cpp debugSkipNeuralNet, extended with a separate ordinary draw.
    WDL logits{};for(int i:{0,2,3,1})logits[i]=static_cast<float>(0.20*random.nextGaussian());
    e.wdl=outcome_probabilities(logits,allow_no_result);
    e.has_auxiliary=true;e.optimistic_logits=e.logits;e.policy_optimism_allowed=false;
    // Source v17 postprocessing of raw shorttermWinlossError=0, multiplier=.25.
    e.shortterm_value_stdev=static_cast<float>(0.5*std::log(2.0));
    return e;
}
RandomBackend::RandomBackend(int canvas,uint64_t seed,GameSpec spec) : Backend(spec),canvas_(canvas),random_(std::make_unique<Rand>(seed)) {
    if (canvas < 5 || canvas > 25) throw std::runtime_error("Invalid random evaluator canvas");
}
RandomBackend::~RandomBackend()=default;
std::vector<Evaluation> RandomBackend::evaluate(const InferenceInputs& inputs) {
    std::vector<Evaluation> result;
    result.reserve(inputs.size());
    for (auto observation : inputs) {
        auto spec=spec_;
        if (observation->size() != static_cast<size_t>(spec.planes*canvas_*canvas_ + spec.globals))
            throw std::runtime_error("Random observation shape mismatch");
        std::vector<uint8_t> mask(spec.actions(canvas_),1);
        if(spec.action_layout!=ActionLayout::MOVES)
            for(int i=0;i<canvas_*canvas_;++i)mask[i]=(*observation)[i]!=0;
        result.push_back(random_evaluation(*random_,mask,no_result_possible(*observation,spec,canvas_)));
    }
    return result;
}
}
