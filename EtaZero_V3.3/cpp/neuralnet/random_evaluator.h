#pragma once
#include "neuralnet/inference.h"
class Rand;
namespace etazero {
Evaluation random_evaluation(Rand&,const std::vector<uint8_t>& action_mask,bool allow_no_result);
// Each inference server advances its own source RNG on cache misses.
class RandomBackend final : public Backend {
    int canvas_;
    std::unique_ptr<Rand> random_;
public:
    RandomBackend(int canvas,uint64_t seed,GameSpec spec=game_spec(GameKind::GOMOKU));
    ~RandomBackend() override;
    bool supports_auxiliary() const override {return true;}
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override;
};
}
