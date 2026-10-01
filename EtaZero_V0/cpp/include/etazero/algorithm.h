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
    int symmetry_count() const override { return 8; }
    Evaluation evaluate_symmetry(int symmetry) const override {
        if(symmetry<0 || symmetry>=8)throw std::runtime_error("Invalid D4 symmetry");
        auto input=game_.observation(),transformed=input;
        int n=game_.canvas(),area=game_.actions();
        std::vector<int> mapping(area);
        for(int a=0;a<area;++a) {
            int x=a%n,y=a/n,tx=x,ty=y;
            switch(symmetry) {
                case 1:tx=y;ty=n-1-x;break;
                case 2:tx=n-1-x;ty=n-1-y;break;
                case 3:tx=n-1-y;ty=x;break;
                case 4:tx=y;ty=x;break;
                case 5:tx=n-1-x;break;
                case 6:tx=n-1-y;ty=n-1-x;break;
                case 7:ty=n-1-y;break;
            }
            mapping[a]=ty*n+tx;
            for(int p=0;p<INPUT_PLANES;++p)transformed[p*area+mapping[a]]=input[p*area+a];
        }
        auto result=evaluator_.evaluate(transformed);
        if(result.logits.size()!=mapping.size())throw std::runtime_error("Invalid symmetry policy shape");
        auto logits=result.logits;
        for(int a=0;a<area;++a)result.logits[a]=logits[mapping[a]];
        return result;
    }
    Transition move(int action) override {
        int parent=game_.player();game_.play(action);
        // Exact terminal values supply the outcome; no duplicated terminal reward.
        return {0,1,parent==game_.player()?1.0:-1.0};
    }
};
}
