#pragma once
#include "inference.h"
#include <random>
namespace etazero {
// Stateless per-observation outputs: reproducible across batch/thread scheduling.
class RandomBackend final : public Backend {
    int canvas_;
    uint64_t seed_;
public:
    RandomBackend(int canvas,uint64_t seed,GameSpec spec=game_spec(GameKind::GOMOKU));
    std::vector<Evaluation> evaluate(const InferenceInputs& inputs) override;
};
}
