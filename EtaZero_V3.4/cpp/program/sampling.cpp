#include "program/game_config.h"
#include "program/sampling.h"
#include <cmath>
#include <iterator>
#include <filesystem>
#include <set>
#include <iostream>
#include "game/go_position.h"
#include "dataio/files.h"
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
                    c.integer("game_forks.fork_game_min_choices"),c.integer("game_forks.early_fork_game_max_choices"),c.integer("game_forks.fork_game_max_choices")) {
    if(early_probability>.5 || late_probability>.5)throw std::runtime_error("Configured game-fork probabilities must be <= 0.5");
}
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
    const auto options=rule_options(c,GameConfigMode::SELFPLAY);
    const auto configured=c.contains("environment.rules")?parse_rule(c.list("environment.rules").front()):Rule::FREESTYLE;
    if(configured==Rule::GO)Game(std::min(5,canvas),canvas,Rule::GO,options); // Initialize source board hashes before parsing.
    std::vector<std::filesystem::path> files;
    for(const auto& path:c.list("hint_positions.positions_file")) {
        if(std::filesystem::is_directory(path)) {
            for(const auto& e:std::filesystem::recursive_directory_iterator(path)) {
                auto name=e.path().filename().string();
                if(e.is_regular_file()&&(Global::isSuffix(name,".hintposes.txt")||Global::isSuffix(name,".startposes.txt")||Global::isSuffix(name,".bookposes.txt")))files.push_back(e.path());
            }
        } else files.emplace_back(path);
    }
    std::sort(files.begin(),files.end());std::string contents;
    for(const auto& path:files){std::ifstream file(path);if(!file)throw std::runtime_error("Cannot open hint positions: "+path.string());
        contents.append(std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>());contents+='\n';}
    std::istringstream input(contents);
    for(std::string line;std::getline(input,line);) {
        line=trim(line);if(line.empty() || line[0]=='#')continue;
        if(line[0]=='{') {
            if(configured!=Rule::GO)throw std::runtime_error("Source Go hints require a Go model");
            try {
                auto sample=Sgf::PositionSample::ofJsonLine(line);
                auto game=game_from_go_sample(sample,canvas,options);
                int hint=go_action(sample.hintLoc,sample.board.x_size,canvas);
                if(game.finished()||(hint>=0&&!game.legal(hint)))continue;
                auto moves=game.moves();positions.push_back({std::move(game),std::move(moves),InitialKind::Hint,hint,sample.weight,sample.trainingWeight});
            } catch(const std::exception& e){std::cerr<<"Invalid hint position: "<<e.what()<<'\n';}
            continue;
        }
        std::istringstream row(line);int size,hint,count;std::string rule;double weight;
        if(!(row>>size>>rule>>weight>>hint>>count) || !std::isfinite(weight)||weight<0||size>canvas||size<5||count<0||hint<0)
            throw std::runtime_error("Invalid game hint row: "+line);
        Game game(size,canvas,parse_rule(rule),options);
        if(c.contains("environment.rules") && !same_layout(game_spec(parse_rule(c.list("environment.rules").front())),game.spec()))
            throw std::runtime_error("Hint input layout differs from configured environment");
        InitialPosition position{game,{},InitialKind::Hint,game.native_action(hint),weight};
        for(int i=0;i<count;++i) {
            int local;if(!(row>>local)||local<0||local>=game.local_actions())throw std::runtime_error("Invalid game hint prefix");
            int action=game.native_action(local);position.game.play(action);position.actions.push_back(action);
        }
        std::string extra;
        if(row>>extra) {
            const std::string prefix="training_weight=";
            if(extra.rfind(prefix,0)!=0)throw std::runtime_error("Trailing game hint fields");
            size_t consumed=0;auto value=extra.substr(prefix.size());position.training_weight=std::stod(value,&consumed);
            if(consumed!=value.size()||!std::isfinite(position.training_weight)||position.training_weight<0||row>>extra)
                throw std::runtime_error("Invalid hint training weight");
        }
        if(position.game.finished()||!position.game.legal(position.hint_action))throw std::runtime_error("Terminal hint position or occupied hint move");
        positions.push_back(std::move(position));
    }
    double mass=0;for(const auto& p:positions)mass+=p.sampling_weight;
    if(!std::isfinite(mass)||(!positions.empty()&&mass<=0))throw std::runtime_error("Enabled hint input has no positive sampling mass");
    return positions;
}
std::optional<InitialPosition> sample_hint_position(const std::vector<InitialPosition>& positions,double probability,std::mt19937_64& rng) {
    if(!std::isfinite(probability)||probability<0||probability>1)throw std::runtime_error("Invalid hint probability");
    if(probability==0 || !std::bernoulli_distribution(probability)(rng))return {};
    if(positions.empty())return {};
    std::vector<double> weights;for(const auto& p:positions)weights.push_back(p.sampling_weight);
    return positions[std::discrete_distribution<size_t>(weights.begin(),weights.end())(rng)];
}

std::vector<InitialPosition> load_start_positions(const Config& c,int canvas,std::mt19937_64& rng) {
    std::vector<InitialPosition> positions;
    if(!c.contains("start_positions.probability") || c.number("start_positions.probability")==0)return positions;
    const double load=c.number("start_positions.load_probability"),lambda=c.number("start_positions.turn_weight_lambda");
    if(load<0||load>1||lambda< -10||lambda>10)throw std::runtime_error("Invalid start-position sampling parameters");
    std::set<std::string> excludes,unique;
    const bool is_go=parse_rule(c.list("environment.rules").front())==Rule::GO;
    auto source_excludes=is_go?Sgf::readExcludes(c.list("start_positions.excludes")):std::set<Hash128>{};
    for(const auto& path:is_go?std::vector<std::string>{}:c.list("start_positions.excludes")) {
        std::ifstream file(path);if(!file)throw std::runtime_error("Cannot open start-position exclusions: "+path);
        for(std::string line;std::getline(file,line);) {line=trim(line);if(!line.empty()&&line[0]!='#')excludes.insert(line);}
    }
    std::vector<std::filesystem::path> files;
    for(const auto& path:c.list("start_positions.files")) {
        if(std::filesystem::is_directory(path)) {
            for(const auto& entry:std::filesystem::recursive_directory_iterator(path))
                if(entry.is_regular_file() && (entry.path().extension()==".startpos" || entry.path().extension()==".txt" || entry.path().extension()==".sgf"))files.push_back(entry.path());
        } else files.emplace_back(path);
    }
    std::sort(files.begin(),files.end());
    const auto options=rule_options(c,GameConfigMode::SELFPLAY);
    std::set<Hash128> source_unique;
    if(parse_rule(c.list("environment.rules").front())==Rule::GO)Game(std::min(5,canvas),canvas,Rule::GO,options);
    for(const auto& path:files) {
        if(path.extension()==".sgf") {
            if(parse_rule(c.list("environment.rules").front())!=Rule::GO)throw std::runtime_error("SGF starts require a Go model");
            try {
                std::unique_ptr<Sgf> sgf(Sgf::loadFile(path.string()));
                if(source_excludes.count(sgf->hash))continue;
                sgf->iterAllUniquePositions(source_unique,false,false,true,false,nullptr,[&](Sgf::PositionSample& sample,const BoardHistory&,const std::string&){
                    if(!std::bernoulli_distribution(load)(rng))return;
                    auto game=game_from_go_sample(sample,canvas,options);if(game.finished())return;
                    auto moves=game.moves();positions.push_back({std::move(game),std::move(moves),InitialKind::External,-1,sample.weight,sample.trainingWeight});
                });
            }catch(const std::exception& e){std::cerr<<"Invalid SGF "<<path<<": "<<e.what()<<'\n';}
            continue;
        }
        std::ifstream file(path);if(!file)throw std::runtime_error("Cannot open start positions: "+path.string());
        for(std::string line;std::getline(file,line);) {
            line=trim(line);if(line.empty()||line[0]=='#')continue;
            if(line[0]=='{') {
                if(parse_rule(c.list("environment.rules").front())!=Rule::GO)throw std::runtime_error("Source Go starts require a Go model");
                auto sample=Sgf::PositionSample::ofJsonLine(line);auto game=game_from_go_sample(sample,canvas,options);
                if(!game.finished()&&std::bernoulli_distribution(load)(rng)) {auto moves=game.moves();positions.push_back({std::move(game),std::move(moves),InitialKind::External,-1,sample.weight,sample.trainingWeight});}
                continue;
            }
            // Corpus id, board size, rule, sampling weight, loss weight, move count,
            // then moves in physical coordinates. Rows explicitly enumerate starts.
            std::istringstream row(line);std::string id,rule;int size,count;double sampling,training;
            if(!(row>>id>>size>>rule>>sampling>>training>>count)||size<5||size>canvas||count<0||
               !std::isfinite(sampling)||sampling<0||!std::isfinite(training)||training<0)
                throw std::runtime_error("Invalid start position row: "+line);
            if(excludes.count(id))continue;
            InitialPosition position{Game(size,canvas,parse_rule(rule),options),{},InitialKind::External,-1,sampling,training};
            if(!same_layout(game_spec(parse_rule(c.list("environment.rules").front())),position.game.spec()))
                throw std::runtime_error("Start position layout differs from environment");
            for(int i=0;i<count;++i) {
                int local;if(!(row>>local)||local<0||local>=position.game.local_actions())throw std::runtime_error("Invalid start prefix");
                int action=position.game.native_action(local);
                if(!position.game.legal(action)||position.game.finished())throw std::runtime_error("Illegal start prefix");
                position.game.play(action);position.actions.push_back(action);
            }
            std::string extra;if(row>>extra)throw std::runtime_error("Trailing start position fields");
            if(position.game.finished())continue;
            // Parent history is intentionally excluded from corpus deduplication,
            // as in iterAllUniquePositions(hashParent=false). The chosen row still
            // retains its entire legal history for actual play and NN features.
            std::string key=std::to_string(size)+":"+rule+":"+std::to_string(position.game.player())+":";
            for(auto cell:position.game.board().cells)key+=std::to_string(cell)+",";
            if(!unique.insert(key).second || !std::bernoulli_distribution(load)(rng))continue;
            positions.push_back(std::move(position));
        }
    }
    int minimum_turn=0;for(const auto& p:positions)minimum_turn=std::min(minimum_turn,int(p.game.training_metadata()[2]));
    double total_weight=0;
    for(auto& p:positions) {
        p.sampling_weight*=std::exp(-(std::max(0,p.game.turn())-minimum_turn)*lambda);
        if(!std::isfinite(p.sampling_weight)||p.sampling_weight<0||p.sampling_weight>=1e200)throw std::runtime_error("Invalid start position sampling mass");
        total_weight+=p.sampling_weight;
    }
    if(!positions.empty()&&!(total_weight>0))throw std::runtime_error("Start corpus has zero sampling weight");
    return positions;
}

void initialize_start_policy(InitialPosition& position,const Config& c,Evaluator& evaluator,
                             std::mt19937_64& rng,const std::function<bool()>& should_stop) {
    if(!c.boolean("start_positions.policy_init"))return;
    double proportion=c.number("start_positions.policy_area_proportion");if(proportion<=0)return;
    double shape=c.number("start_positions.policy_gamma_shape"),mean=position.game.area()*proportion;
    int count=static_cast<int>(std::floor(shape==1 ? std::exponential_distribution<double>(1)(rng)*mean :
                                       std::gamma_distribution<double>(shape,mean/shape)(rng)));
    for(int i=0;i<count && !position.game.finished();++i) {
        if(should_stop && should_stop())return;
        auto probabilities=evaluate_policy(position.game,evaluator,c.number("start_positions.policy_temperature"),
                                           c.boolean("symmetry.nn_randomize"),c.integer("symmetry.nn_symmetry"));
        std::vector<int> legal;for(int a=0;a<position.game.actions();++a)if(position.game.legal(a)&&probabilities[a]>0)legal.push_back(a);
        int action=std::bernoulli_distribution(.0002)(rng)?legal[std::uniform_int_distribution<size_t>(0,legal.size()-1)(rng)]:
                   static_cast<int>(std::discrete_distribution<size_t>(probabilities.begin(),probabilities.end())(rng));
        position.game.play(action);position.actions.push_back(action);
    }
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
    Game game(record.size,record.canvas,record.rule,record.options);
    size_t move_index=early?static_cast<size_t>(std::floor(std::exponential_distribution<double>(1)(rng)*c.early_expected_move_proportion*game.area())):
        (record.steps.empty()?0:std::uniform_int_distribution<size_t>(0,record.steps.size()-1)(rng));
    // Source replayGameUpToMove always stops before the final recorded move,
    // including the unbounded exponential tail of early forks.
    if(!record.steps.empty())move_index=std::min(move_index,record.steps.size()-1);
    std::vector<int> actions;
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
        auto nn=evaluator.evaluate_position(next.observation(),next.player(),symmetry,false,1,randomize,0);
        if(!std::isfinite(nn.wdl[0])||!std::isfinite(nn.wdl[2]))throw std::runtime_error("Nonfinite fork value");
        // Pure W-L adaptation of source's value-net score ranking, from the parent perspective.
        double value=nn.wdl[2]-nn.wdl[0];if(chosen<0 || value>best){chosen=candidates[i];best=value;}
    }
    game.play(chosen);if(game.finished())return {};actions.push_back(chosen);
    return InitialPosition{game,std::move(actions),early?InitialKind::EarlyFork:InitialKind::GameFork,-1,1,record.training_weight};
}
std::optional<InitialPosition> make_hint_fork(const FinishedGame& record) {
    size_t prefix=record.opening.actions.size();int hint=record.opening.hint_action;
    if(record.opening.initial_position_kind!=static_cast<int>(InitialKind::Hint) || hint<0 || prefix>=record.steps.size() || record.steps[prefix].action==hint)return {};
    Game game(record.size,record.canvas,record.rule,record.options);std::vector<int> actions;
    for(size_t i=0;i<prefix;++i){int action=record.steps[i].action;game.play(action);actions.push_back(action);}
    if(game.finished()||!game.legal(hint))return {};
    game.play(hint);if(game.finished())return {};actions.push_back(hint);
    return InitialPosition{game,std::move(actions),InitialKind::HintFork,-1,1,record.training_weight};
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
    auto output=evaluator.evaluate_position(game.observation(),game.player(),symmetry,false,temperature,randomize,0);
    if(output.logits.size()!=static_cast<size_t>(game.actions()))throw std::runtime_error("Fork NN output mismatch");
    std::vector<int> legal;for(int a=0;a<game.actions();++a)if(game.legal(a))legal.push_back(a);
    return source_policy_probabilities(output,legal,temperature,0);
}
void search_side_positions(std::vector<Game>& positions,FinishedGame& record,GameSearch& search,Evaluator& evaluator,
                           int64_t full_visits,bool randomize,int symmetry,
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
        record.side_positions.back().metadata=search_metadata(result,game);
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
                        GameSearch& search,const std::function<double(const Game&)>& temperature,std::mt19937_64& rng,
                        const std::function<bool()>& should_stop) {
    compute_value_surprises(record,config.direct_value_surprise);
    auto selected=select_reanalysis_turns(record,config,rng);size_t prefix=record.opening.actions.size();
    if(selected.empty())return;
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
        step.metadata=search_metadata(result,game);
        step.policy=std::move(result.policy);step.visits=std::move(result.visits);step.policy_target=std::move(result.policy_target);
        step.q_values=std::move(result.q_values);step.q_visits=std::move(result.q_visits);
        step.network_wdl=result.network_wdl;step.search_wdl=result.search_wdl;step.policy_surprise=result.policy_surprise;
        step.target_weight=limits.target_weight;
    }
    compute_value_surprises(record,config.direct_value_surprise);
}

}
