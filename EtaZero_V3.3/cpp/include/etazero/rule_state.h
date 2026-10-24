#pragma once
#include "outcome.h"
#include "game_spec.h"
#include "rules.h"
#include "subtree_value_bias_key.h"
#include <memory>
class BoardHistory;
namespace etazero {
struct RuleOptions {
    std::string go_rules="chinese", fen, go_position;
    double komi=7.5;
    int go_width=0,go_height=0;
    bool go_pass_alive=true,go_exclude_atari=true;
};
class RuleState {
public:
    virtual ~RuleState()=default;
    virtual std::unique_ptr<RuleState> clone() const=0;
    // Game only resets states with the same rule and dimensions. Reuse storage.
    virtual void reset_from(const RuleState& other)=0;
    virtual int player() const=0;
    virtual BoardGeometry geometry() const {return board();}
    virtual const ::BoardHistory* go_history() const { return nullptr; }
    virtual int initial_turn() const { return 0; }
    virtual Outcome outcome() const=0;
    virtual bool finished() const=0;
    virtual bool legal(int action) const=0;
    virtual bool raw_legal(int action) const {return legal(action); }
    virtual void play(int action)=0;
    virtual const Board& board() const=0;
    virtual std::vector<float> observation(int perspective,double pda,bool forbidden_feature) const=0;
    // Before a legal move. Game owns the actual previous action (-1 at start).
    // False means this game/transition must not share a bias estimate.
    virtual bool subtree_bias_key(int,int,SubtreeValueBiasKey&) const { return false; }
    // Go supplies its exact ko-aware source implementation. Other games use their rule symmetry group.
    virtual std::vector<int> root_symmetry_representatives(const std::vector<int>&) const { return {}; }
    virtual bool symmetry_history_ok(const std::vector<int>&) const { return true; }
    virtual std::string pattern_key(int,int,bool) const {return {};}
    virtual std::string identity() const { return {}; }
    virtual std::string graph_identity(int) const { return identity(); }
    // Self komi, area-scoring phase, initial turn and the two Go adjudication modes.
    virtual std::array<float,5> training_metadata() const { return {}; }
    virtual std::string fen() const { return {}; }
    virtual std::string position_json(int) const { throw std::runtime_error("PositionSample requires Go"); }
};
std::unique_ptr<RuleState> make_rule_state(Rule,int,int,const RuleOptions&);
}
