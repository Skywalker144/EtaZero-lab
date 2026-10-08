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
    // Adapters with cheap in-place reset can reuse per-thread state storage.
    virtual bool reset_from(const SearchState&) { return false; }
    virtual int actions() const = 0;
    virtual bool legal(int action) const = 0;
    virtual bool terminal() const = 0;
    virtual double terminal_value() const = 0;
    virtual Evaluation evaluate() const = 0;
    virtual int symmetry_count() const { return 1; }
    // Policy logits are returned in the original action coordinates.
    virtual Evaluation evaluate_symmetry(int symmetry) const {
        if(symmetry!=0)throw std::runtime_error("State does not support D4 evaluation");
        return evaluate();
    }
    virtual Evaluation evaluate_symmetry(int symmetry, bool skip_cache, double policy_temperature) const {
        (void)skip_cache; (void)policy_temperature;
        return evaluate_symmetry(symmetry);
    }
    virtual Evaluation evaluate_symmetry(int symmetry, bool skip_cache, double policy_temperature, bool randomize) const {
        (void)randomize;return evaluate_symmetry(symmetry,skip_cache,policy_temperature);
    }
    virtual Evaluation evaluate_symmetry(int symmetry,bool skip_cache,double temperature,bool randomize,double optimism) const {
        (void)optimism;return evaluate_symmetry(symmetry,skip_cache,temperature,randomize);
    }
    // Exact situation identity includes all rule/history/input conditions that affect continuations.
    // Unsupported state adapters must refuse graph search explicitly.
    virtual std::string graph_key() const { throw std::runtime_error("State does not support graph search"); }
    virtual Transition move(int action) = 0;
};
class AlphaZeroState final : public SearchState {
    Game game_;
    Evaluator& evaluator_;
public:
    AlphaZeroState(const Game& game, Evaluator& evaluator) : game_(game), evaluator_(evaluator) {}
    std::unique_ptr<SearchState> clone() const override { return std::make_unique<AlphaZeroState>(game_,evaluator_); }
    bool reset_from(const SearchState& state) override {
        auto other=dynamic_cast<const AlphaZeroState*>(&state);
        if (!other || &evaluator_!=&other->evaluator_) return false;
        game_=other->game_;return true;
    }
    int actions() const override { return game_.actions(); }
    bool legal(int action) const override { return game_.legal(action); }
    bool terminal() const override { return game_.finished(); }
    double terminal_value() const override { return game_.terminal_value(); }
    Evaluation evaluate() const override { return evaluator_.evaluate(game_.observation()); }
    int symmetry_count() const override { return game_.rule()==Rule::HEX?2:8; }
    Evaluation evaluate_symmetry(int symmetry) const override {
        return evaluate_symmetry(symmetry,false,1);
    }
    Evaluation evaluate_symmetry(int symmetry, bool skip_cache, double policy_temperature) const override {
        return evaluator_.evaluate_symmetry(game_.observation(),symmetry,skip_cache,policy_temperature);
    }
    Evaluation evaluate_symmetry(int symmetry, bool skip_cache, double policy_temperature, bool randomize) const override {
        return evaluator_.evaluate_symmetry(game_.observation(),symmetry,skip_cache,policy_temperature,randomize);
    }
    Evaluation evaluate_symmetry(int symmetry,bool skip_cache,double temperature,bool randomize,double optimism) const override {
        return evaluator_.evaluate_symmetry(game_.observation(),symmetry,skip_cache,temperature,randomize,optimism);
    }
    std::string graph_key() const override {
        // NOVC Gomoku only adds stones: repetitions/ko/pass/history-based adjudication do not occur.
        // The terminal flag/result distinguish an ended path from a playable board of the same layout.
        std::string key;
        for(int n:{game_.size(),game_.canvas(),game_.player(),game_.turn(),int(game_.rule()),
                   int(game_.finished()),game_.winner(),game_.reason()}) {
            uint32_t bits=static_cast<uint32_t>(n);
            for(int i=0;i<4;++i)key.push_back(static_cast<char>((bits>>(8*i))&255));
        }
        // Conditioning changes NN input at every depth and must split transpositions.
        double d=game_.pda_doublings();
        key.append(reinterpret_cast<const char*>(&d),sizeof(d));key.push_back(static_cast<char>(game_.pda_player()));
        for(int8_t cell:game_.board().cells)key.push_back(static_cast<char>(cell));
        return key;
    }
    Transition move(int action) override {
        int parent=game_.player();game_.play(action);
        // Exact terminal values supply the outcome; no duplicated terminal reward.
        return {0,1,parent==game_.player()?1.0:-1.0};
    }
};
}
