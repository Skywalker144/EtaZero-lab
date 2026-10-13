#pragma once
#include "game.h"
#include "inference.h"
namespace etazero {
// WDL search over terminal-outcome games: no intermediate reward or discount.
struct SearchEvaluation {
    int symmetry = 0;
    bool skip_cache = false;
    double policy_temperature = 1;
    bool randomize = false;
    double optimism = 0;
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
    // Policy logits are returned in the original action coordinates.
    virtual Evaluation evaluate(SearchEvaluation options = {}) const = 0;
    virtual int symmetry_count() const { return 1; }
    // Exact situation identity includes all rule/history/input conditions that affect continuations.
    // Unsupported state adapters must refuse graph search explicitly.
    virtual std::string graph_key() const { throw std::runtime_error("State does not support graph search"); }
    // Return +1 for the same value perspective, -1 when the player changes.
    virtual int move(int action) = 0;
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
    int symmetry_count() const override { return game_.rule()==Rule::HEX?2:8; }
    Evaluation evaluate(SearchEvaluation options = {}) const override {
        return evaluator_.evaluate_symmetry(game_.observation(),options.symmetry,options.skip_cache,
                                           options.policy_temperature,options.randomize,options.optimism);
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
    int move(int action) override {
        int parent=game_.player();game_.play(action);
        // Exact terminal values supply the outcome; no duplicated terminal reward.
        return parent==game_.player()?1:-1;
    }
};
}
