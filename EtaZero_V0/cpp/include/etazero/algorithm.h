#pragma once
#include "game.h"
#include "inference.h"
namespace etazero {
// Search consumes these operations without assuming a real board, zero reward,
// or a particular network method. First-round support is only AlphaZeroState.
struct Transition {
    double reward = 0, discount = 1, perspective = -1;
};
class SearchState {
public:
    virtual ~SearchState() = default;
    virtual std::unique_ptr<SearchState> clone() const = 0;
    virtual int actions() const = 0;
    virtual bool legal(int action) const = 0;
    virtual bool terminal() const = 0;
    virtual double terminal_value() const = 0;
    virtual Evaluation evaluate() const = 0;
    virtual Transition move(int action) = 0;
};
class AlphaZeroState final : public SearchState {
    Game game_;
    Evaluator& evaluator_;
public:
    AlphaZeroState(const Game& game, Evaluator& evaluator) : game_(game), evaluator_(evaluator) {}
    std::unique_ptr<SearchState> clone() const override { return std::make_unique<AlphaZeroState>(game_,evaluator_); }
    int actions() const override { return game_.actions(); }
    bool legal(int action) const override { return game_.legal(action); }
    bool terminal() const override { return game_.finished(); }
    double terminal_value() const override { return game_.terminal_value(); }
    Evaluation evaluate() const override { return evaluator_.evaluate(game_.observation()); }
    Transition move(int action) override {
        int parent=game_.player();game_.play(action);
        // Exact terminal values supply the outcome; no duplicated terminal reward.
        return {0,1,parent==game_.player()?1.0:-1.0};
    }
};
}
