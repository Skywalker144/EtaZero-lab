#pragma once
#include "game/game.h"
#include "neuralnet/inference.h"
#include "search/subtree_value_bias.h"
#include "game/history_features.h"
#include "search/pattern_bonus.h"
namespace etazero {
// WDL search over terminal-outcome games: no intermediate reward or discount.
struct SearchEvaluation {
    int symmetry = 0;
    bool skip_cache = false;
    double policy_temperature = 1;
    bool randomize = false;
    double optimism = 0;
    int max_history = -1;
};
class SearchState {
public:
    virtual ~SearchState() = default;
    virtual std::unique_ptr<SearchState> clone() const = 0;
    // Adapters with cheap in-place reset can reuse per-thread state storage.
    virtual bool reset_from(const SearchState&) { return false; }
    virtual int actions() const = 0;
    virtual int pass_action() const {return -1;}
    virtual int player() const { return 1; }
    virtual bool legal(int action) const = 0;
    virtual bool terminal() const = 0;
    virtual double terminal_value() const = 0;
    virtual double terminal_no_result_probability() const {return 0;}
    virtual double terminal_draw_probability() const {return terminal_value()==0?1:0;}
    // Policy logits are returned in the original action coordinates.
    virtual Evaluation evaluate(SearchEvaluation options = {}) const = 0;
    virtual Evaluation evaluate_with(Evaluator&,SearchEvaluation) const {throw std::runtime_error("State has no secondary network adapter");}
    virtual std::vector<int> root_symmetry_representatives(const std::vector<int>&) const {
        std::vector<int> result(actions());for(int i=0;i<actions();++i)result[i]=i;return result;
    }
    virtual std::string pattern_key(int) const {throw std::runtime_error("State has no real-board pattern adapter");}
    virtual void repeated_patterns(PatternBonusTable&,double,int) const {throw std::runtime_error("State has no pattern history adapter");}
    virtual int symmetry_count() const { return 1; }
    // Exact identity by default. Go opts into KataGo's bounded-history sharing;
    // the actual path remains authoritative for legality and terminal outcomes.
    virtual void prepare_graph(int) const {}
    virtual bool graph_approximates_history() const { return false; }
    virtual std::string graph_key() const { throw std::runtime_error("State does not support graph search"); }
    virtual bool supports_subtree_value_bias() const { return false; }
    // Called BEFORE the selected legal move. The game decides whether to share.
    virtual bool subtree_bias_key(int,SubtreeValueBiasKey&) const {
        throw std::runtime_error("State does not support real-board subtree value bias");
    }
    // Return +1 for the same value perspective, -1 when the player changes.
    virtual int move(int action) = 0;
};
class AlphaZeroState final : public SearchState {
    Game game_;
    Evaluator& evaluator_;
    mutable int graph_rep_bound_=11;
public:
    AlphaZeroState(const Game& game, Evaluator& evaluator) : game_(game), evaluator_(evaluator) {}
    std::unique_ptr<SearchState> clone() const override { return std::make_unique<AlphaZeroState>(*this); }
    bool reset_from(const SearchState& state) override {
        auto other=dynamic_cast<const AlphaZeroState*>(&state);
        if (!other || &evaluator_!=&other->evaluator_) return false;
        game_=other->game_;graph_rep_bound_=other->graph_rep_bound_;return true;
    }
    int actions() const override { return game_.actions(); }
    int pass_action() const override {return game_.rule()==Rule::GO?game_.canvas_area():-1;}
    int player() const override { return game_.player(); }
    bool legal(int action) const override { return game_.legal(action); }
    bool terminal() const override { return game_.finished(); }
    double terminal_value() const override { return game_.terminal_value(); }
    double terminal_no_result_probability() const override {return game_.terminal_no_result_probability();}
    double terminal_draw_probability() const override {return game_.terminal_draw_probability();}
    std::vector<int> root_symmetry_representatives(const std::vector<int>& avoid) const override {return game_.root_symmetry_representatives(avoid);}
    std::string pattern_key(int action) const override {return game_.pattern_key(action);}
    void repeated_patterns(PatternBonusTable& table,double bonus,int player) const override {table.add_game(game_,bonus,player);}
    int symmetry_count() const override { return game_.spec().symmetries; }
    void prepare_graph(int rep_bound) const override { graph_rep_bound_=rep_bound; }
    bool graph_approximates_history() const override { return game_.rule()==Rule::GO; }
    bool supports_subtree_value_bias() const override { return game_.spec().subtree_value_bias; }
    bool subtree_bias_key(int action,SubtreeValueBiasKey& key) const override {
        return game_.subtree_bias_key(action,key);
    }
    Evaluation evaluate(SearchEvaluation options = {}) const override {
        return evaluate_with(evaluator_,options);
    }
    Evaluation evaluate_with(Evaluator& evaluator,SearchEvaluation options) const override {
        auto observation=game_.observation();limit_history_features(observation,game_.canvas(),game_.spec(),options.max_history);
        return evaluator.evaluate_position(observation,game_.player(),options.symmetry,options.skip_cache,
                                           options.policy_temperature,options.randomize,options.optimism);
    }
    std::string graph_key() const override {
        std::string key;
        auto append=[&](int n) {
            uint32_t bits=static_cast<uint32_t>(n);
            for(int i=0;i<4;++i)key.push_back(static_cast<char>((bits>>(8*i))&255));
        };
        for(int n:{game_.size(),game_.canvas(),int(game_.rule())})append(n);
        // Conditioning changes NN input at every depth and must split transpositions.
        double d=game_.pda_doublings();
        key.append(reinterpret_cast<const char*>(&d),sizeof(d));key.push_back(static_cast<char>(game_.pda_player()));
        if(graph_approximates_history()) {
            // Do not append turn, recent NN history or the full move sequence:
            // those would prevent the transpositions GraphHash deliberately permits.
            key+=game_.graph_identity(graph_rep_bound_);
        } else {
            for(int n:{game_.player(),game_.turn(),int(game_.finished()),game_.winner(),int(game_.reason())})append(n);
            for(int cell:game_.board().cells)key.push_back(static_cast<char>(cell));
            key+=game_.history_identity();
        }
        return key;
    }
    int move(int action) override {
        int parent=game_.player();game_.play(action);
        // Exact terminal values supply the outcome; no duplicated terminal reward.
        return parent==game_.player()?1:-1;
    }
};
}
