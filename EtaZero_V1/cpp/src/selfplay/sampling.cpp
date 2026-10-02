#include "etazero/sampling.h"
#include <cmath>
#include <iterator>
#include <openssl/evp.h>
namespace etazero {
void ForkPool::add(InitialPosition position) {
    if(position.game.finished())throw std::runtime_error("Cannot enqueue terminal fork");
    position.game.set_pda(0,0);std::lock_guard<std::mutex> lock(mutex_);positions_.push_back(std::move(position));
}
std::optional<InitialPosition> ForkPool::take(std::mt19937_64& rng) {
    std::lock_guard<std::mutex> lock(mutex_);if(positions_.empty())return {};
    size_t index=std::uniform_int_distribution<size_t>(0,positions_.size()-1)(rng);
    InitialPosition result=std::move(positions_[index]);
    if(index+1!=positions_.size())positions_[index]=std::move(positions_.back());
    positions_.pop_back();return result;
}
GameForkConfig::GameForkConfig(const Config& c)
    :GameForkConfig(c.number("game_forks.early_fork_game_prob"),c.number("game_forks.fork_game_prob"),c.number("game_forks.early_fork_game_expected_move_prop"),
                    c.integer("game_forks.fork_game_min_choices"),c.integer("game_forks.early_fork_game_max_choices"),c.integer("game_forks.fork_game_max_choices")) {}
GameForkConfig::GameForkConfig(double early,double late,double proportion,int minimum,int early_maximum,int late_maximum)
    :early_probability(early),late_probability(late),early_expected_move_proportion(proportion),
     minimum_choices(minimum),early_maximum_choices(early_maximum),late_maximum_choices(late_maximum) {
    if(!std::isfinite(early_probability)||!std::isfinite(late_probability)||!std::isfinite(early_expected_move_proportion)||
       early_probability<0||early_probability>1||late_probability<0||late_probability>1||early_expected_move_proportion<0||early_expected_move_proportion>1||
       minimum_choices<1||early_maximum_choices<minimum_choices||late_maximum_choices<minimum_choices||early_maximum_choices>100||late_maximum_choices>100)
        throw std::runtime_error("Invalid early/game fork configuration");
}
std::vector<InitialPosition> load_hint_positions(const Config& c,int canvas) {
    std::vector<InitialPosition> positions;
    if(c.number("hint_positions.hint_positions_prob")==0)return positions;
    const auto path=c.text("hint_positions.positions_file");
    std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("Cannot open Gomoku hint positions: "+path);
    std::string contents{std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
    if(file.bad())throw std::runtime_error("Cannot read Gomoku hint positions: "+path);
    unsigned char digest[EVP_MAX_MD_SIZE];unsigned int length=0;
    if(EVP_Digest(contents.data(),contents.size(),digest,&length,EVP_sha256(),nullptr)!=1)
        throw std::runtime_error("Cannot hash Gomoku hint positions: "+path);
    std::string checksum;checksum.reserve(2*length);
    for(unsigned int i=0;i<length;++i){checksum.push_back("0123456789abcdef"[digest[i]>>4]);checksum.push_back("0123456789abcdef"[digest[i]&15]);}
    if(checksum!=c.text("hint_positions.positions_sha256"))throw std::runtime_error("Hint positions checksum mismatch: "+path);
    // Parse the checked bytes so a concurrent file replacement cannot change the input.
    std::istringstream input(contents);
    for(std::string line;std::getline(input,line);) {
        line=trim(line);if(line.empty() || line[0]=='#')continue;
        std::istringstream row(line);int size,hint,count;std::string rule;double weight;
        if(!(row>>size>>rule>>weight>>hint>>count) || !std::isfinite(weight)||weight<0||size>canvas||size<5||count<0||count>=size*size||hint<0||hint>=size*size)
            throw std::runtime_error("Invalid Gomoku hint row: "+line);
        Game game(size,canvas,parse_rule(rule));InitialPosition position{game,{},InitialKind::Hint,hint/size*canvas+hint%size,weight};
        for(int i=0;i<count;++i) {
            int local;if(!(row>>local)||local<0||local>=size*size)throw std::runtime_error("Invalid Gomoku hint prefix");
            int action=local/size*canvas+local%size;position.game.play(action);position.actions.push_back(action);
        }
        std::string extra;if(row>>extra)throw std::runtime_error("Trailing Gomoku hint fields");
        if(position.game.finished()||!position.game.legal(position.hint_action))throw std::runtime_error("Terminal hint position or occupied hint move");
        positions.push_back(std::move(position));
    }
    double mass=0;for(const auto& p:positions)mass+=p.sampling_weight;
    if(positions.empty()||!std::isfinite(mass)||mass<=0)throw std::runtime_error("Enabled hint input has no positive sampling mass");
    return positions;
}
std::optional<InitialPosition> sample_hint_position(const std::vector<InitialPosition>& positions,double probability,std::mt19937_64& rng) {
    if(!std::isfinite(probability)||probability<0||probability>1)throw std::runtime_error("Invalid hint probability");
    if(probability==0 || !std::bernoulli_distribution(probability)(rng))return {};
    if(positions.empty())throw std::runtime_error("Enabled hint sampling lacks positions");
    std::vector<double> weights;for(const auto& p:positions)weights.push_back(p.sampling_weight);
    return positions[std::discrete_distribution<size_t>(weights.begin(),weights.end())(rng)];
}
SelfplaySearchContext hint_context(const Game& current,const Game& start,int hint,InitialKind kind,bool force_full) {
    SelfplaySearchContext context;context.hint_action=hint;context.current_turn=current.turn();context.force_full=force_full;
    context.hint_fork=kind==InitialKind::HintFork;context.hint_turn=(hint>=0||context.hint_fork)?start.turn():-1;
    context.exact_hint=hint>=0 && current.turn()==start.turn() && current.size()==start.size() &&
        current.rule()==start.rule() && current.player()==start.player() && current.board().cells==start.board().cells;
    return context;
}
std::optional<InitialPosition> make_game_fork(const FinishedGame& record,const GameForkConfig& c,Evaluator& evaluator,
    bool randomize,int symmetry,std::mt19937_64& rng,const std::function<bool()>& should_stop) {
    bool early=c.early_probability>0 && std::bernoulli_distribution(c.early_probability)(rng);
    bool late=!early && c.late_probability>0 && std::bernoulli_distribution(c.late_probability)(rng);
    if(!early && !late)return {};
    size_t move_index=early?static_cast<size_t>(std::floor(std::exponential_distribution<double>(1)(rng)*c.early_expected_move_proportion*record.size*record.size)):
        (record.steps.empty()?0:std::uniform_int_distribution<size_t>(0,record.steps.size()-1)(rng));
    // Source replayGameUpToMove always stops before the final recorded move,
    // including the unbounded exponential tail of early forks.
    if(!record.steps.empty())move_index=std::min(move_index,record.steps.size()-1);
    Game game(record.size,record.canvas,record.rule);std::vector<int> actions;
    for(size_t i=0;i<move_index && i<record.steps.size() && !game.finished();++i) {int a=record.steps[i].action;game.play(a);actions.push_back(a);}
    if(game.finished())return {};
    int choices=std::uniform_int_distribution<int>(c.minimum_choices,early?c.early_maximum_choices:c.late_maximum_choices)(rng);
    std::vector<int> legal;for(int a=0;a<game.actions();++a)if(game.legal(a))legal.push_back(a);
    if(legal.empty())return {};
    // Source chooseRandomLegalMoves samples with replacement, including when
    // there are fewer legal actions than requested candidates.
    std::vector<int> candidates; candidates.reserve(choices);
    for(int i=0;i<choices;++i)candidates.push_back(legal[std::uniform_int_distribution<size_t>(0,legal.size()-1)(rng)]);
    double best=-1e100;int chosen=-1;
    for(int i=0;i<choices;++i) {
        if(should_stop && should_stop())return {};
        Game next=game;next.play(candidates[i]);
        auto nn=evaluator.evaluate_symmetry(next.observation(),symmetry,false,1,randomize,0);
        if(!std::isfinite(nn.wdl[0])||!std::isfinite(nn.wdl[2]))throw std::runtime_error("Nonfinite fork value");
        // Pure W-L adaptation of source's value-net score ranking, from the parent perspective.
        double value=nn.wdl[2]-nn.wdl[0];if(chosen<0 || value>best){chosen=candidates[i];best=value;}
    }
    game.play(chosen);if(game.finished())return {};actions.push_back(chosen);
    return InitialPosition{game,std::move(actions),early?InitialKind::EarlyFork:InitialKind::GameFork};
}
std::optional<InitialPosition> make_hint_fork(const FinishedGame& record) {
    size_t prefix=record.opening.actions.size();int hint=record.opening.hint_action;
    if(record.opening.initial_position_kind!=static_cast<int>(InitialKind::Hint) || hint<0 || prefix>=record.steps.size() || record.steps[prefix].action==hint)return {};
    Game game(record.size,record.canvas,record.rule);std::vector<int> actions;
    for(size_t i=0;i<prefix;++i){int action=record.steps[i].action;game.play(action);actions.push_back(action);}
    if(game.finished()||!game.legal(hint))return {};
    game.play(hint);if(game.finished())return {};actions.push_back(hint);
    return InitialPosition{game,std::move(actions),InitialKind::HintFork};
}
int sample_fork_move(const Game& game,const std::vector<double>& policy,int banned,std::mt19937_64& rng) {
    if(policy.size()!=static_cast<size_t>(game.actions()))throw std::runtime_error("Fork policy shape mismatch");
    double r=std::uniform_real_distribution<double>(0,1)(rng);
    std::vector<int> actions;std::vector<double> weights;
    for(int action=0;action<game.actions();++action) {
        double p=policy[action];if(!std::isfinite(p)||p<0)throw std::runtime_error("Invalid fork policy");
        if(action!=banned && game.legal(action)) {
            actions.push_back(action);weights.push_back(r<.70?p:r<.95?std::sqrt(p):1);
        }
    }
    if(actions.empty())return -1;
    double mass=0;for(double w:weights)mass+=w;if(mass<=0)return -1;
    return actions[std::discrete_distribution<size_t>(weights.begin(),weights.end())(rng)];
}
std::vector<double> evaluate_policy(const Game& game,Evaluator& evaluator,double temperature,bool randomize,int symmetry) {
    if(!std::isfinite(temperature)||temperature<=0)throw std::runtime_error("Invalid fork NN temperature");
    auto output=evaluator.evaluate_symmetry(game.observation(),symmetry,false,temperature,randomize,0);
    if(output.logits.size()!=static_cast<size_t>(game.actions()))throw std::runtime_error("Fork NN output mismatch");
    std::vector<double> policy(game.actions(),0);double maximum=-1e100,mass=0;
    for(int a=0;a<game.actions();++a)if(game.legal(a)) {
        if(!std::isfinite(output.logits[a]))throw std::runtime_error("Nonfinite fork logits");
        maximum=std::max(maximum,output.logits[a]/temperature);
    }
    for(int a=0;a<game.actions();++a)if(game.legal(a)) {policy[a]=std::exp(output.logits[a]/temperature-maximum);mass+=policy[a];}
    if(mass<=0 || !std::isfinite(mass))throw std::runtime_error("Empty fork NN policy");
    for(auto& p:policy)p/=mass;
    return policy;
}
void search_side_positions(std::vector<Game>& positions,FinishedGame& record,Search& search,Evaluator& evaluator,
                           int full_visits,bool randomize,int symmetry,
                           const std::function<double(const Game&)>& temperature,std::mt19937_64& rng,
                           const std::function<bool()>& should_stop) {
    // Source processes the growing queue after main-game reanalysis/weight computation.
    // Every recursion adds two stones, so finite NOVC boards need no artificial depth cap.
    for(size_t i=0;i<positions.size();++i) {
        if(should_stop && should_stop())return;
        Game game=positions[i];game.set_pda(0,0);
        if(game.finished())throw std::runtime_error("Terminal side position cannot be searched");
        SearchRun options;options.training=false;options.clear_before_search=true;
        options.max_visits=full_visits;options.should_stop=should_stop;
        auto result=search.run(game,temperature(game),options);
        if(should_stop && should_stop())return;
        if(result.action<0)throw std::runtime_error("Side search did not complete");
        record.side_positions.push_back({game.player(),game.observation(),std::move(result.policy_target),std::move(result.visits),result.search_wdl});
        record.side_positions.back().q_values=std::move(result.q_values);
        record.side_positions.back().q_visits=std::move(result.q_visits);
        // Source uses the response chosen by this independent full search, then a new
        // ordinary-policy fork. No actual-main-game move is available as a ban here.
        if(std::bernoulli_distribution(.25)(rng)) {
            game.play(result.action);if(game.finished())continue;
            auto policy=evaluate_policy(game,evaluator,1,randomize,symmetry);
            int action=sample_fork_move(game,policy,-1,rng);if(action<0)continue;
            game.play(action);if(!game.finished())positions.push_back(std::move(game));
        }
    }
}
ReanalysisConfig reanalysis_config(const Config& c) {
    ReanalysisConfig result;result.enabled=c.boolean("reanalysis.use_reanalyze");
    result.direct_value_surprise=c.boolean("surprise_weighting.use_search_value_surprise");
    if(result.enabled) {
        result.proportion=c.number("reanalysis.reanalyze_prop");
        result.policy_surprise_weight=c.number("reanalysis.reanalyze_policy_surprise_weight");
        result.value_surprise_weight=c.number("reanalysis.reanalyze_value_surprise_weight");
        result.surprise_exponent=c.number("reanalysis.reanalyze_surprise_exponent");
        result.use_outcome_targets=c.boolean("reanalysis.reanalyze_use_outcome_targets");
    }
    return result;
}
std::vector<size_t> select_reanalysis_turns(const FinishedGame& record,const ReanalysisConfig& c,std::mt19937_64& rng) {
    if(!std::isfinite(c.proportion) || c.proportion<0 || c.proportion>1 ||
       !std::isfinite(c.policy_surprise_weight) || c.policy_surprise_weight<0 || c.policy_surprise_weight>100 ||
       !std::isfinite(c.value_surprise_weight) || c.value_surprise_weight<0 || c.value_surprise_weight>100 ||
       !std::isfinite(c.surprise_exponent) || c.surprise_exponent<0 || c.surprise_exponent>10)
        throw std::runtime_error("Invalid reanalysis sampling configuration");
    if(!c.enabled || c.proportion<=0)return {};
    std::vector<size_t> candidates;std::vector<double> weights;size_t count=0;
    for(size_t i=0;i<record.steps.size();++i) {
        const auto& step=record.steps[i];if(!step.trainable || !step.cheap_search)continue;
        candidates.push_back(i);if(std::bernoulli_distribution(c.proportion)(rng))++count;
        double surprise=c.policy_surprise_weight*step.policy_surprise+c.value_surprise_weight*step.value_surprise;
        if(!std::isfinite(surprise)||surprise<0)throw std::runtime_error("Invalid reanalysis surprise");
        double weight=std::pow(surprise,c.surprise_exponent);
        if(!std::isfinite(weight))throw std::runtime_error("Nonfinite reanalysis selection weight");
        weights.push_back(weight);
    }
    std::vector<size_t> selected;
    for(size_t k=0;k<count;++k) {
        double mass=0;for(double w:weights)mass+=w;
        size_t which=mass>1e-30?std::discrete_distribution<size_t>(weights.begin(),weights.end())(rng):
            std::uniform_int_distribution<size_t>(0,candidates.size()-1)(rng);
        selected.push_back(candidates[which]);candidates.erase(candidates.begin()+which);weights.erase(weights.begin()+which);
    }
    std::sort(selected.begin(),selected.end());return selected;
}
void reanalyze_positions(FinishedGame& record,const std::vector<Game>& positions,const std::vector<double>& history,
                        const SelfplaySearchConfig& search_config,PlayoutAdvantage advantage,const ReanalysisConfig& config,
                        Search& search,const std::function<double(const Game&)>& temperature,std::mt19937_64& rng,
                        const std::function<bool()>& should_stop) {
    compute_value_surprises(record,config.direct_value_surprise);
    auto selected=select_reanalysis_turns(record,config,rng);size_t prefix=record.opening.actions.size();
    if(positions.size()+prefix!=record.steps.size())throw std::runtime_error("Reanalysis positions/trajectory mismatch");
    for(size_t turn:selected) {
        if(should_stop && should_stop())return;
        size_t index=turn-prefix;const Game& game=positions.at(index);
        // History is the ORIGINAL completed searches up to this turn, not later
        // reanalyses, so reduced limits reproduce the counterfactual full turn.
        std::vector<double> past(history.begin(),history.begin()+std::min(index,history.size()));
        auto limits=selfplay_search_limits(search_config,past,false,advantage,game.player(),SelfplaySearchContext{-1,-1,game.turn(),false,false,true});
        limits.search.should_stop=should_stop;
        limits.search.clear_before_search=true;auto result=search.run(game,temperature(game),limits.search);
        if(should_stop && should_stop())return;
        if(result.action<0)throw std::runtime_error("Reanalysis search did not complete");
        auto& step=record.steps[turn];step.reanalyzed=true;step.reanalysis_used_outcome=config.use_outcome_targets;
        step.reanalysis_policy_surprise=step.policy_surprise;step.reanalysis_value_surprise=step.value_surprise;
        step.reanalysis_original_visits=1;for(auto n:step.visits)step.reanalysis_original_visits+=n;
        step.policy=std::move(result.policy);step.visits=std::move(result.visits);step.policy_target=std::move(result.policy_target);
        step.q_values=std::move(result.q_values);step.q_visits=std::move(result.q_visits);
        step.network_wdl=result.network_wdl;step.search_wdl=result.search_wdl;step.policy_surprise=result.policy_surprise;
        step.target_weight=limits.target_weight;
    }
    compute_value_surprises(record,config.direct_value_surprise);
}

}
