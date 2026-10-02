#pragma once
#include "record.h"
#include "search_limits.h"
#include <optional>
namespace etazero {
enum class InitialKind { Ordinary, Hint, EarlyFork, GameFork, HintFork };
struct InitialPosition {
    Game game;
    std::vector<int> actions;
    InitialKind kind=InitialKind::Ordinary;
    int hint_action=-1;
    double sampling_weight=1;
};
class ForkPool {
    std::mutex mutex_;
    std::vector<InitialPosition> positions_;
public:
    void add(InitialPosition position);
    std::optional<InitialPosition> take(std::mt19937_64& rng);
};
struct GameForkConfig {
    double early_probability,late_probability,early_expected_move_proportion;
    int minimum_choices,early_maximum_choices,late_maximum_choices;
    explicit GameForkConfig(const Config& config);
    GameForkConfig(double early,double late,double proportion,int minimum,int early_maximum,int late_maximum);
};
std::vector<InitialPosition> load_hint_positions(const Config& config,int canvas);
std::optional<InitialPosition> sample_hint_position(const std::vector<InitialPosition>& positions,double probability,std::mt19937_64& rng);
SelfplaySearchContext hint_context(const Game& current,const Game& start,int hint,InitialKind kind,bool force_full=false);
std::optional<InitialPosition> make_game_fork(const FinishedGame& record,const GameForkConfig& config,Evaluator& evaluator,
    bool randomize,int symmetry,std::mt19937_64& rng,const std::function<bool()>& should_stop);
std::optional<InitialPosition> make_hint_fork(const FinishedGame& record);
// Canonical, legal-domain NN policy; excludes the actual action before sampling.
int sample_fork_move(const Game& game,const std::vector<double>& policy,int banned,std::mt19937_64& rng);
std::vector<double> evaluate_policy(const Game& game,Evaluator& evaluator,double temperature,bool randomize,int symmetry);
struct ReanalysisConfig {
    bool enabled=false, direct_value_surprise=false, use_outcome_targets=true;
    double proportion=0,policy_surprise_weight=1,value_surprise_weight=1,surprise_exponent=1;
};
ReanalysisConfig reanalysis_config(const Config& config);
std::vector<size_t> select_reanalysis_turns(const FinishedGame& record,const ReanalysisConfig& config,std::mt19937_64& rng);
void reanalyze_positions(FinishedGame& record,const std::vector<Game>& positions,const std::vector<double>& history,
                        const SelfplaySearchConfig& search_config,PlayoutAdvantage advantage,const ReanalysisConfig& config,
                        Search& search,const std::function<double(const Game&)>& temperature,std::mt19937_64& rng,
                        const std::function<bool()>& should_stop);
void search_side_positions(std::vector<Game>& positions,FinishedGame& record,Search& search,Evaluator& evaluator,
                           int full_visits,bool randomize,int symmetry,
                           const std::function<double(const Game&)>& move_temperature,std::mt19937_64& rng,
                           const std::function<bool()>& should_stop);
}
