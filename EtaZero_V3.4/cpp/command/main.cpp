#include "program/setup.h"
#include "program/game_config.h"
#include "core/config.h"
#include "dataio/record.h"
#include "neuralnet/backend_factory.h"
#include "neuralnet/random_evaluator.h"
#include "program/search_limits.h"
#include "program/sampling.h"
#include "search/muzero/search.h"
#include "search/gumbel_search.h"
#include "neuralnet/muzero/torch_backend.h"
#include "program/match.h"
#include "command/session.h"
#include "etazero/source_config.h"
#include "game/symmetry.h"
#include <csignal>
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/InferenceMode.h>
#include <ATen/autocast_mode.h>
using namespace etazero;
namespace {
static_assert(std::atomic<bool>::is_always_lock_free, "Signal stop flag must be lock-free");
std::atomic<bool> stop_requested{false};
void stop_handler(int) { stop_requested.store(true,std::memory_order_relaxed); }
struct Args {
    std::map<std::string, std::string> args;
    Args(int argc, char** argv) {
        const std::set<std::string> allowed{"config","model","model-id","device","games","output","run-id","attempt-id",
                                          "config-id","source-id","iteration","worker","seed","size","rule","moves","model-b","model-b-id","evaluator","tasks","stream","models","max-models","model-load-workers","protocol"};
        for (int i = 2; i < argc; i += 2) {
            std::string name = argv[i];
            if (name.rfind("--",0) != 0 || !allowed.count(name.substr(2)) || i+1 >= argc || !args.emplace(name.substr(2), argv[i+1]).second)
                throw std::runtime_error("Invalid native command argument: " + name);
        }
    }
    std::string get(const std::string& key) const { return args.at(key); }
    std::string get(const std::string& key, std::string value) const { auto it = args.find(key); return it == args.end() ? value : it->second; }
    int integer(const std::string& key) const { return parse_integer(get(key),"--"+key); }
};
uint64_t mix(uint64_t n) {
    n += 0x9e3779b97f4a7c15ULL; n = (n^(n>>30))*0xbf58476d1ce4e5b9ULL;
    n = (n^(n>>27))*0x94d049bb133111ebULL; return n^(n>>31);
}
template<class T> void array(std::ostream& out, const std::vector<T>& values) {
    out << '['; bool first = true;
    for (auto x : values) { if (!first) out << ','; out << x; first = false; } out << ']';
}
std::unique_ptr<InferenceService> evaluator(const Args& a, const Config& c, const std::string& mode,
                                          torch::jit::Module* web_model = nullptr,
                                          std::shared_ptr<PreparedModel> prepared = nullptr,
                                          torch::jit::Module* supplied_model = nullptr) {
    std::string prefix = (mode == "infer" || mode == "selfplay") ? "inference" : mode == "analysis" ? "analysis" : mode;
    int batch = c.integer(prefix+".max_batch"), canvas = c.integer("network.canvas");
    std::vector<std::unique_ptr<Backend>> backends;
    std::string kind=a.get("evaluator","network");
    if(kind!="random" && kind!="network")throw std::runtime_error("Unknown evaluator: "+kind);
    if(kind=="random" && mode!="selfplay")throw std::runtime_error("Random evaluator is for cold-start selfplay only");
    std::string algorithm=c.contains("agent.algorithm")?c.text("agent.algorithm"):"alphazero";
    if(kind=="network" && is_cuda_model(a.get("model")))algorithm=cuda_model_algorithm(a.get("model"));
    std::optional<torch::jit::Module> detected_model;
    if(kind=="network" && mode!="selfplay" && mode!="infer" && (!is_cuda_model(a.get("model")) || web_model)) {
        // Detect from the actual inference model, then hand it to its backend.
        // Selfplay already knows the algorithm and loads during backend preparation.
        torch::Device device(a.get("device"));c10::cuda::OptionalCUDAGuard guard;
        if(device.is_cuda())guard.set_device(device);
        detected_model=supplied_model?*supplied_model:torch::jit::load(reference_model_path(a.get("model")),device);
        detected_model->eval();
        if(device.is_cuda())c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
        auto metadata=detected_model->get_method("metadata")({}).toTuple();
        if(metadata->elements().size()==3 || metadata->elements().size()==2)
            algorithm=metadata->elements()[1].toStringRef();
        else throw std::runtime_error("Unknown model metadata");
        if(web_model)*web_model=*detected_model; // Share immutable weights with Web-only diagnostics.
    }
    if(!prepared)prepared=detected_model?std::make_shared<PreparedModel>(std::move(*detected_model)):
                                        std::make_shared<PreparedModel>();
    auto random_spec=kind=="random"?game_spec(parse_rule(c.list("environment.rules").front())):game_spec(GameKind::GOMOKU);
    auto attach_secondary=[&](std::unique_ptr<InferenceService> primary) {
        std::string key=(mode=="infer"||mode=="selfplay"?"human_sl":prefix)+".human_model";
        if(!c.contains(key)||c.text(key).empty())return primary;
        std::string path=c.text(key),device=a.get("device"),precision=c.text(prefix+".inference_precision");
        if(precision=="auto")precision=device.rfind("cuda",0)==0?"float16":"float32";
        std::string secondary_algorithm=is_cuda_model(path) && std::filesystem::exists(std::filesystem::path(path).parent_path()/"manifest.json")?cuda_model_algorithm(path):"alphazero";
        auto prepared_secondary=std::make_shared<PreparedModel>();
        if(!is_cuda_model(path)) {
            auto model=torch::jit::load(path,torch::Device(device));model.eval();
            secondary_algorithm=model.get_method("metadata")({}).toTuple()->elements().at(1).toStringRef();
            prepared_secondary=std::make_shared<PreparedModel>(std::move(model));
        }
        std::string profile_key=(mode=="infer"||mode=="selfplay"?"human_sl":prefix)+".human_sl_profile";
        std::string human_profile=c.contains(profile_key)?c.text(profile_key):"";
        if(!human_profile.empty() && !is_cuda_model(path))throw std::runtime_error("Human SL profiles require a source CUDA metadata model");
        uint64_t seed=std::stoull(a.get("seed",prefix=="inference"?c.text("run.seed"):c.text(prefix+".seed")));
        if(secondary_algorithm=="alphazero") {
            std::vector<std::unique_ptr<Backend>> models;for(int i=0;i<c.integer(prefix+".server_threads");++i)models.push_back(make_backend(path,device,canvas,batch,precision,prepared_secondary,human_profile));
            primary->secondary=std::make_shared<BatchEvaluator>(std::move(models),path,canvas,batch,c.integer(prefix+".batch_wait_us"),c.integer64(prefix+".cache_entries"),false,0,seed,
                c.contains(prefix+".nn_mutex_pool_power_of_two")?c.integer(prefix+".nn_mutex_pool_power_of_two"):16);
        } else if(secondary_algorithm=="muzero") {
            std::vector<std::unique_ptr<muzero::Backend>> models;for(int i=0;i<c.integer(prefix+".server_threads");++i)models.push_back(make_muzero_backend(path,device,canvas,batch,precision,prepared_secondary));
            primary->secondary=std::make_shared<muzero::BatchEvaluator>(std::move(models),canvas,batch,c.integer(prefix+".batch_wait_us"),c.integer64(prefix+".cache_entries"),false,0,seed,c.integer(prefix+".muzero_queue_capacity"),c.contains(prefix+".nn_mutex_pool_power_of_two")?c.integer(prefix+".nn_mutex_pool_power_of_two"):16);
        } else throw std::runtime_error("Unknown secondary network algorithm");
        if(!same_layout(primary->spec(),primary->secondary->spec()))throw std::runtime_error("Secondary policy game layout differs");
        return primary;
    };
    if(algorithm=="muzero") {
        std::vector<std::unique_ptr<muzero::Backend>> models;
        for(int i=0;i<c.integer(prefix+".server_threads");++i) {
            if(kind=="random")models.push_back(std::make_unique<muzero::RandomBackend>(canvas,std::stoull(a.get("seed"))+i,random_spec));
            else {
                auto precision=c.text(prefix+".inference_precision");
                if(precision=="auto")precision=a.get("device").rfind("cuda:",0)==0?"float16":"float32";
                models.push_back(make_muzero_backend(a.get("model"),a.get("device"),canvas,batch,precision,
                    prepared));
            }
        }
        return attach_secondary(std::make_unique<muzero::BatchEvaluator>(std::move(models),canvas,batch,c.integer(prefix+".batch_wait_us"),
            c.integer64(prefix+".cache_entries"),c.boolean((prefix=="inference"?"symmetry":prefix)+".nn_randomize"),
            c.integer((prefix=="inference"?"symmetry":prefix)+".nn_symmetry"),
            std::stoull(a.get("seed",prefix=="inference"?c.text("run.seed"):c.text(prefix+".seed"))),c.integer(prefix+".muzero_queue_capacity"),c.integer(prefix+".nn_mutex_pool_power_of_two")));
    }
    if(algorithm!="alphazero")throw std::runtime_error("Unknown model algorithm");
    for (int i=0;i<c.integer(prefix+".server_threads");++i) {
        if(kind=="random")backends.push_back(std::make_unique<RandomBackend>(canvas,std::stoull(a.get("seed"))+i,random_spec));
        else backends.push_back(make_backend(a.get("model"),a.get("device"),canvas,batch,
                                                         c.text(prefix+".inference_precision"),prepared));
    }
    return attach_secondary(std::make_unique<BatchEvaluator>(std::move(backends), a.get("model-id"), canvas,
                                          batch, c.integer(prefix+".batch_wait_us"),
                                          c.integer64(prefix+".cache_entries"),c.boolean((prefix=="inference"?"symmetry":prefix)+".nn_randomize"),
                                          c.integer((prefix=="inference"?"symmetry":prefix)+".nn_symmetry"),
                                          std::stoull(a.get("seed",prefix=="inference"?c.text("run.seed"):c.text(prefix+".seed"))),
                                          c.contains(prefix+".nn_mutex_pool_power_of_two")?c.integer(prefix+".nn_mutex_pool_power_of_two"):16));
}
SearchResult analyze_position(GameSearch& search,const Config& config,const Game& game,SearchRun options={}) {
    options.should_stop=[]{return stop_requested.load();};
    return search.run(game,move_temperature(config,"analysis",game),options);
}
void moves(Game& g, const std::string& text) {
    if (text.empty()) return;
    std::istringstream in(text); std::string item;
    while (std::getline(in,item,',')) g.play(std::stoi(item));
}
void stats(InferenceService& e) {
    std::cout << "{\"event\":\"inference\",\"requests\":" << e.requests << ",\"batches\":" << e.batches
              << ",\"max_batch\":" << e.max_observed_batch
              << ",\"submitted\":" << e.submitted << ",\"cache_hits\":" << e.cache_hits << ",\"rows_by_server\":";
    array(std::cout,e.rows_by_server);std::cout<<"}"<<std::endl;
}
int selfplay(const Args& a,const Config& c,InferenceService& service,ForkPool& forks) {
    int count = a.integer("games"), canvas = c.integer("network.canvas");
    if (count < 1) throw std::runtime_error("Selfplay requires positive game count");
    const auto game_options=rule_options(c,GameConfigMode::SELFPLAY);
    auto* eval=&service;eval->reset_stats();
    OpeningConfig opening(c,"policy_init"); auto search_config=selfplay_search_config(c);auto reanalysis=reanalysis_config(c);
    GameForkConfig fork_config(c);auto hints=load_hint_positions(c,canvas);
    std::mt19937_64 corpus_rng(std::stoull(a.get("seed"))^0xE7037ED1A0B428DBULL);
    auto starts=load_start_positions(c,canvas,corpus_rng);
    bool random=a.get("evaluator","network")=="random";
    Source source{a.get("run-id"),a.get("attempt-id"),a.get("model-id"),a.get("config-id"),a.get("source-id"),a.integer("iteration"),a.integer("worker")};
    if(c.text("agent.algorithm")=="muzero")source.unroll_steps=c.integer("unroll.steps");
    source.gumbel=c.text("agent.root_search_algo")=="gumbel";source.full_gumbel=c.text("agent.nonroot_search_algo")=="gumbel";
    source.research_records=c.contains("writer.research_records")&&c.boolean("writer.research_records");
    RecordWriter writer(a.get("output"),source,c.integer("writer.shard_rows"),c.integer("writer.writer_queue"),
                        c.number("writer.first_file_min_random_proportion"),std::stoull(a.get("seed"))^0xA0761D6478BD642FULL);
    std::vector<int> sizes; std::vector<Rule> rules; std::vector<double> sw, rw;
    for (auto x : c.list("environment.sizes")) sizes.push_back(parse_integer(x,"environment.sizes"));
    for (auto x : c.list("environment.rules")) rules.push_back(parse_rule(x));
    for (auto x : c.list("environment.size_weights")) sw.push_back(std::stod(x));
    for (auto x : c.list("environment.rule_weights")) rw.push_back(std::stod(x));
    uint64_t seed = std::stoull(a.get("seed"));
    std::atomic<int> next{0}, finished{0}; std::atomic<bool> failure{false};
    std::mutex error_mutex; std::exception_ptr error;
    auto loop = [&] {
        try {
            auto search_owner=search_for(*eval,settings(c,"selfplay"),0,c,"selfplay");auto& search=*search_owner;
            while (!stop_requested && !failure) {
                int id = next.fetch_add(1); if (id >= count) break;
                uint64_t game_seed = mix(seed+id); std::mt19937_64 rng(game_seed);
                int size = sizes[std::discrete_distribution<size_t>(sw.begin(),sw.end())(rng)];
                Rule rule = rules[std::discrete_distribution<size_t>(rw.begin(),rw.end())(rng)];
                auto initial=forks.take(rng);
                if(!initial && !starts.empty())initial=sample_hint_position(starts,c.number("start_positions.probability"),rng);
                if(!initial)initial=sample_hint_position(hints,c.number("hint_positions.hint_positions_prob"),rng);
                int external_prefix=initial?static_cast<int>(initial->actions.size()):0;
                if(initial && initial->kind==InitialKind::External && !random)
                    initialize_start_policy(*initial,c,*eval,rng,[&]{return stop_requested.load()||failure.load();});
                if(stop_requested || failure)break;
                if(initial){size=initial->game.size();rule=initial->game.rule();}
                Game game=initial?initial->game:Game(size,canvas,rule,game_options);search.reset(rng());
                FinishedGame record{}; record.id=id; record.seed=game_seed; record.size=size; record.canvas=canvas; record.rule=rule;record.options=game.options();
                record.forbidden_feature_dropout_prob=c.number("environment.forbidden_feature_dropout_prob");
                record.training_weight=initial?initial->training_weight:1.0;
                InitialKind kind=initial?initial->kind:InitialKind::Ordinary;
                bool fork_start=kind==InitialKind::EarlyFork || kind==InitialKind::GameFork || kind==InitialKind::HintFork;
                auto advantage=fork_start?PlayoutAdvantage{}:sample_playout_advantage(c.number("pda.normal_asymmetric_playout_prob"),c.number("pda.max_asymmetric_ratio"),rng);
                std::vector<Game> side_positions,positions;
                std::vector<double> historical_values;
                if(initial) {
                    record.opening.actions=initial->actions;record.opening.initial_position_moves=initial->actions.size();
                    record.opening.initial_position_kind=static_cast<int>(kind);record.opening.hint_action=initial->hint_action;
                    if(kind==InitialKind::External) {
                        record.opening.initial_position_moves=external_prefix;
                        record.opening.policy_moves=initial->actions.size()-external_prefix;
                    }
                } else if(!random) {
                    record.opening=initialize_opening(game,opening,*eval,rng,[&]{return stop_requested || failure;});
                    if(record.opening.status==OpeningStatus::Interrupted)break;
                }
                Game prefix(size,canvas,rule,game.options());prefix.set_pda(advantage.doublings,advantage.player);
                for(int action:record.opening.actions) {
                    Step step{prefix.player(),action,0,0,0,prefix.observation(),
                              std::vector<double>(game.actions(),0),std::vector<int64_t>(game.actions(),0),false};
                    prefix.play(action);if(prefix.finished())step.reward=prefix.winner()*step.player;
                    record.steps.push_back(std::move(step));
                }
                const Game start=game;
                game.set_pda(advantage.doublings,advantage.player);
                while (!game.finished() && !stop_requested && !failure) {
                    if(reanalysis.enabled && reanalysis.proportion>0)positions.push_back(game);
                    double temp = move_temperature(c,"selfplay",game);
                    auto context=hint_context(game,start,record.opening.hint_action,kind);
                    double cheap_probability=cheap_search_probability(search_config,context);
                    bool cheap=cheap_probability>0 && std::bernoulli_distribution(cheap_probability)(rng);
                    auto limits=selfplay_search_limits(search_config,historical_values,cheap,advantage,game.player(),context);
                    limits.search.should_stop=[&]{return stop_requested.load() || failure.load();};
                    auto result = search.run(game,temp,limits.search);
                    if(stop_requested || failure)break;
                    if(result.action<0)throw std::runtime_error("Selfplay search has no move: zero playout budget");
                    auto metadata=search_metadata(result,game);
                    Step step{game.player(),result.action,result.simulations,static_cast<float>(temp),0,
                              game.observation(),std::move(result.policy),std::move(result.visits)};
                    step.metadata=metadata;step.cheap_search=limits.cheap_search;step.target_weight=limits.target_weight;
                    step.policy_target=std::move(result.policy_target);step.policy_surprise=result.policy_surprise;step.network_wdl=result.network_wdl;step.search_wdl=result.search_wdl;
                    step.q_values=std::move(result.q_values);step.q_visits=std::move(result.q_visits);
                    historical_values.push_back(game.player()*result.value);
                    if(c.number("side_positions.side_position_prob")>0 &&
                       std::bernoulli_distribution(c.number("side_positions.side_position_prob"))(rng)) {
                        int fork=sample_fork_move(game,result.network_policy,result.action,rng);
                        if(fork>=0) {Game side=game;side.play(fork);side.set_pda(0,0);if(!side.finished())side_positions.push_back(std::move(side));}
                    }
                    game.play(result.action); if (game.finished()) step.reward = game.winner() * step.player;
                    record.steps.push_back(std::move(step)); search.advance(result.action);
                }
                if (!game.finished()) break; // Interrupted trajectories never receive fabricated targets.
                record.winner=game.winner(); record.reason=game.reason(); record.final_player=game.player(); record.final_observation=game.observation();
                reanalyze_positions(record,positions,historical_values,search_config,advantage,reanalysis,search,
                    [&](const Game& position){return move_temperature(c,"selfplay",position);},rng,
                    [&]{return stop_requested.load() || failure.load();});
                if(stop_requested || failure)break;
                search_side_positions(side_positions,record,search,*eval,search_config.full_visits,
                    c.boolean("symmetry.nn_randomize"),c.integer("symmetry.nn_symmetry"),
                    [&](const Game& position){return move_temperature(c,"selfplay",position);},rng,
                    [&]{return stop_requested.load() || failure.load();});
                if(stop_requested || failure)break;
                apply_training_weights(record,c.number("surprise_weighting.policy_surprise_data_weight"),c.number("surprise_weighting.value_surprise_data_weight"),rng,reanalysis.direct_value_surprise,reanalysis.enabled,
                                       c.contains("data_weight.scale_data_weight")?c.number("data_weight.scale_data_weight"):1.0);
                if(auto position=make_game_fork(record,fork_config,*eval,c.boolean("symmetry.nn_randomize"),c.integer("symmetry.nn_symmetry"),rng,
                                               [&]{return stop_requested.load() || failure.load();}))forks.add(std::move(*position));
                if(stop_requested || failure)break;
                if(auto position=make_hint_fork(record))forks.add(std::move(*position));
                writer.enqueue(std::move(record)); finished.fetch_add(1);
            }
        } catch (...) { std::lock_guard<std::mutex> lock(error_mutex); if (!error) error=std::current_exception(); failure=true; }
    };
    std::vector<std::thread> threads;
    try {
        for (int i=0;i<std::min(count,c.integer("parallelism.game_threads"));++i) threads.emplace_back(loop);
    } catch (...) { failure=true;for(auto& t:threads)t.join();throw; }
    for (auto& t:threads) t.join();
    writer.finish();eval->drain();if(error)std::rethrow_exception(error);stats(*eval);
    std::cout << "{\"event\":\"selfplay_complete\",\"games\":" << finished << "}" << std::endl;
    return stop_requested ? 2 : 0;
}
std::string read_field() {
    std::string header;if(!std::getline(std::cin,header))throw std::runtime_error("Incomplete worker field length");
    size_t consumed=0;size_t bytes=std::stoull(header,&consumed);
    if(consumed!=header.size() || bytes>1024*1024)throw std::runtime_error("Invalid worker field length");
    std::string value(bytes,'\0');
    if(!std::cin.read(value.data(),bytes) || std::cin.get()!='\n')throw std::runtime_error("Incomplete worker field");
    return value;
}
int worker(const Args& base,const Config& config) {
    ForkPool forks;std::unique_ptr<InferenceService> cached;
    auto prepared=std::make_shared<PreparedModel>();
    std::string model,path,operation;
    while(!stop_requested && std::getline(std::cin,operation)) {
        if(operation=="release") {
            bool allocated=static_cast<bool>(cached);cached.reset();prepared->offload();model.clear();path.clear();
            if(allocated && base.get("device").rfind("cuda:",0)==0)c10::cuda::CUDACachingAllocator::emptyCache();
            std::cout<<"{\"event\":\"worker_released\"}"<<std::endl;continue;
        }
        if(operation!="selfplay")throw std::runtime_error("Unknown worker operation");
        Args request=base;
        for(const char* key:{"model","model-id","games","output","attempt-id","iteration","seed","evaluator"})request.args[key]=read_field();
        if(!cached || model!=request.get("model-id") || path!=request.get("model") || request.get("evaluator")=="random") {
            cached.reset();prepared->offload();auto started=std::chrono::steady_clock::now();
            cached=evaluator(request,config,"selfplay",nullptr,prepared);
            model=request.get("model-id");path=request.get("model");
            if(request.get("evaluator")=="network") {
                const auto& timing=prepared->timing;
                std::cout<<"{\"event\":\"worker_prepared\",\"seconds\":"
                         <<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()
                         <<",\"load_seconds\":"<<timing.load_seconds<<",\"weights_seconds\":"<<timing.weights_seconds
                         <<",\"graph_seconds\":"<<timing.graph_seconds<<",\"reused_runtime\":"
                         <<(timing.reused_runtime?"true":"false")<<"}"<<std::endl;
            }
        } else if(path!=request.get("model"))throw std::runtime_error("Worker model identity collision");
        int code=selfplay(request,config,*cached,forks);
        std::cout<<"{\"event\":\"worker_complete\",\"attempt_id\":"<<quote(request.get("attempt-id"))<<",\"code\":"<<code<<"}"<<std::endl;
        if(code)return code;
    }
    return stop_requested?2:0;
}
int analyze(const Args& a, const Config& c, bool raw) {
    Game game(a.integer("size"),c.integer("network.canvas"),parse_rule(a.get("rule")),rule_options(c,GameConfigMode::ANALYSIS));
    moves(game,a.get("moves",""));
    if(!raw)game.set_pda(c.number("analysis.playout_doubling_advantage"),c.text("analysis.playout_doubling_advantage_player")=="black"?1:-1);
    if (game.finished() && !raw) {
        std::cout << "{\"terminal\":true,\"winner\":" << game.winner() << ",\"value\":" << game.terminal_value() << ",\"reason\":" << int(game.reason()) << "}\n";
        return 0;
    }
    auto eval=evaluator(a,c,raw?"infer":"analysis");
    // Raw diagnostics must not seed the search cache with an identity-orientation output.
    auto network=eval->evaluate_position(game.observation(),game.player(),0,true);
    std::cout.precision(12);
    std::cout << "{\"terminal\":" << (game.finished()?"true":"false") << ",\"raw_value\":" << network.value() << ",\"raw_logits\":";
    array(std::cout,network.scaled_policy_logits());std::cout << ",\"raw_wdl\":";array(std::cout,std::vector<double>(network.wdl.begin(),network.wdl.end()));
    std::cout << ",\"raw_optimistic_logits\":";array(std::cout,network.scaled_policy_logits(true));
    std::cout << ",\"raw_shortterm_value_stdev\":" << network.shortterm_value_stdev;
    if(raw)if(auto mu=dynamic_cast<muzero::Evaluator*>(eval.get())) {
        auto obs=game.observation();
        auto spatial=spatial_mapping(game.canvas(),input_symmetry(obs,game.spec(),0));
        auto mapping=policy_mapping(spatial,game.spec());
        auto output=mu->initial(muzero::transform_observation(obs,game.canvas(),spatial,game.spec()));
        // Latent diagnostics use the static action domain, including repeated
        // actions, in the same fixed root coordinates as MuZero search.
        auto actions=game.static_actions();std::cout<<",\"recurrent\":[";
        int index=0;
        for(int action:{actions[0],actions[1],actions[0]}) {
            output=mu->recurrent(output.latent,mapping[action]);
            muzero::restore_evaluation(output.evaluation,mapping);const auto& e=output.evaluation;
            if(index++)std::cout<<',';
            std::cout<<"{\"action\":"<<action<<",\"logits\":";array(std::cout,e.scaled_policy_logits());
            std::cout<<",\"wdl\":";array(std::cout,std::vector<double>(e.wdl.begin(),e.wdl.end()));
            std::cout<<",\"optimistic_logits\":";array(std::cout,e.scaled_policy_logits(true));
            std::cout<<",\"stdev\":"<<e.shortterm_value_stdev<<'}';
        }
        std::cout<<']';
    }
    if (!raw) {
        auto search_owner=search_for(*eval,settings(c,"analysis"),std::stoull(a.get("seed","0")),c,"analysis");auto& search=*search_owner;
        auto result=analyze_position(search,c,game);
        std::cout << ",\"action\":" << result.action << ",\"value\":" << result.value << ",\"simulations\":" << result.simulations << ",\"root_visits\":" << result.root_visits
                  << ",\"initial_visits\":" << result.initial_visits << ",\"new_playouts\":" << result.new_playouts << ",\"seconds\":" << result.seconds
                  << ",\"network_sample_weight\":" << result.network_sample_weight << ",\"network_value_stdev\":" << result.network_value_stdev
                  << ",\"search_weight\":" << result.search_weight << ",\"search_weight_sq\":" << result.search_weight_sq
                  << ",\"graph_hits\":" << result.graph_hits << ",\"graph_catch_ups\":" << result.graph_catch_ups
                  << ",\"graph_cycles\":" << result.graph_cycles << ",\"graph_nodes\":" << result.graph_nodes
                  << ",\"stopped_early\":" << (result.stopped_early?"true":"false") << ",\"policy\":";
        array(std::cout,result.policy); std::cout << ",\"wdl\":";
        array(std::cout,std::vector<double>(result.search_wdl.begin(),result.search_wdl.end()));
        std::cout << ",\"network_wdl\":";array(std::cout,std::vector<double>(result.network_wdl.begin(),result.network_wdl.end()));
        std::cout << ",\"network_policy\":";array(std::cout,result.network_policy);
        std::cout << ",\"search_policy\":";array(std::cout,result.search_policy);
        std::cout << ",\"visits\":"; array(std::cout,result.visits);
    }
    eval->finish();
    auto precision=c.text(raw?"inference.inference_precision":"analysis.inference_precision");
    if(precision=="auto")precision=a.get("device").rfind("cuda:",0)==0?"float16":"float32";
    std::cout << ",\"inference_precision\":" << quote(precision) << ",\"nn_requests\":" << eval->requests << ",\"cache_hits\":" << eval->cache_hits << "}\n"; return stop_requested?130:0;
}
struct WebPolicyPlanes {
    torch::Tensor logits, probabilities, value_logits, wdl;
    std::string precision;
};
WebPolicyPlanes web_policy_planes(torch::jit::Module& model,const Game& game,const std::string& device_name,std::string precision) {
    c10::InferenceMode inference_guard;
    torch::Device device(device_name);c10::cuda::OptionalCUDAGuard device_guard;
    if(device.is_cuda())device_guard.set_device(device);
    if(precision=="auto")precision=device.is_cuda()?"float16":"float32";
    struct AutocastGuard {
        bool enabled=at::autocast::is_autocast_enabled(at::kCUDA);
        at::ScalarType dtype=at::autocast::get_autocast_dtype(at::kCUDA);
        explicit AutocastGuard(bool use) {
            at::autocast::set_autocast_dtype(at::kCUDA,at::kHalf);
            at::autocast::set_autocast_enabled(at::kCUDA,use);
        }
        ~AutocastGuard() {
            at::autocast::clear_cache();at::autocast::set_autocast_enabled(at::kCUDA,enabled);
            at::autocast::set_autocast_dtype(at::kCUDA,dtype);
        }
    } autocast(precision=="float16");
    auto observation=game.observation();const int canvas=game.canvas(),area=game.actions();
    auto spatial=spatial_mapping(canvas,input_symmetry(observation,game.spec(),0));
    observation=muzero::transform_observation(observation,canvas,spatial,game.spec());
    auto obs=torch::from_blob(observation.data(),{1,game.planes(),canvas,canvas},torch::kFloat32).to(device);
    auto globals=torch::from_blob(observation.data()+game.planes()*canvas*canvas,{1,game.globals()},torch::kFloat32).to(device);
    auto output=model.get_method("raw")({obs,globals});
    auto logits=output.toTuple()->elements()[0].toTensor();
    if(logits.dim()!=3 || logits.size(0)!=1 || logits.size(1)<6 || logits.size(2)!=area)
        throw std::runtime_error("Web policy output shape mismatch");
    auto mapping=policy_mapping(spatial,game.spec());std::vector<int64_t> selected;
    for(int a:game.static_actions())selected.push_back(mapping[a]);
    logits=logits[0].narrow(0,0,6).index_select(1,torch::tensor(selected,torch::kInt64).to(device)).to(torch::kFloat32);
    auto probabilities=torch::softmax(logits,1).to(torch::kCPU).contiguous();
    auto value=output.toTuple()->elements()[1].toTensor().to(torch::kFloat32);
    if(value.dim()!=2 || value.size(0)!=1 || value.size(1)!=4)
        throw std::runtime_error("Invalid raw NN value output");
    auto wdl=torch::softmax(value,1).to(torch::kCPU).contiguous();
    logits=logits.to(torch::kCPU).contiguous();value=value.to(torch::kCPU).contiguous();
    for(const auto& tensor:{logits,value})for(int64_t i=0;i<tensor.numel();++i)
        if(!std::isfinite(tensor.data_ptr<float>()[i]))throw std::runtime_error("Nonfinite raw NN output");
    return {logits,probabilities,value,wdl,precision};
}
int analysis_session(const Args& a,const Config& c,const std::string& protocol) {
    torch::jit::Module model;
    auto eval=evaluator(a,c,"analysis",protocol=="katago"||protocol=="gtp"?nullptr:&model);
    auto factory=[&](const Json& overrides) {
            Config effective=c;
            for(auto it=overrides.begin();it!=overrides.end();++it) {
                if(it.key()=="reportAnalysisWinratesAs"||it.key()=="humanSLProfile"||it.key()=="alwaysComputePassAliveUnderSuicideRules"||it.key()=="excludeTerritoryAdjacentToAtari"||it.key()=="always_compute_pass_alive_under_suicide_rules"||it.key()=="exclude_territory_adjacent_to_atari")continue;
                auto key=SOURCE_SEARCH_KEYS.find(it.key());if(key==SOURCE_SEARCH_KEYS.end())throw std::runtime_error("Unsupported overrideSettings key: "+it.key());
                std::string text=it.value().is_string()?it.value().get<std::string>():it.value().dump();
                if(key->second=="visit_cap_contempt_player"||key->second=="playout_doubling_advantage_player"){for(auto& c:text)c=std::tolower(c);if(key->second=="visit_cap_contempt_player"){if(text=="b"||text=="black")text="1";else if(text=="w"||text=="white")text="-1";else if(text=="unspecified")text="0";}else {if(text=="b")text="black";else if(text=="w")text="white";if(text!="black"&&text!="white")throw std::runtime_error("Invalid playout doubling advantage player");}}
                effective.override_field("analysis."+key->second,text);
                auto name="analysis."+key->second;char type=SOURCE_SEARCH_TYPES.at(key->second);
                if(type=='b')effective.boolean(name);
                else if(type=='i'||type=='f'){
                    double value=type=='i'?double(effective.integer64(name)):effective.number(name);
                    auto range=SOURCE_SEARCH_RANGES.find(key->second);
                    if(range!=SOURCE_SEARCH_RANGES.end()&&(value<range->second.first||value>range->second.second))throw std::runtime_error("overrideSettings value outside source range: "+it.key());
                }
            }
            auto parameters=settings(effective,"analysis");
            if(overrides.contains("humanSLProfile")){if(!eval->secondary)throw std::runtime_error("humanSLProfile requires a human model");parameters.human_evaluator=std::make_shared<ProfileEvaluator>(*eval->secondary,overrides["humanSLProfile"].get<std::string>());}
            return search_for(*eval,parameters,std::stoull(a.get("seed","0")),effective,"analysis");
        };
    auto precision=c.text("analysis.inference_precision");
        if(precision=="auto")precision=a.get("device").rfind("cuda:",0)==0?"float16":"float32";
        Json models=Json::array();
        auto add_model=[&](InferenceService& service,const std::string& path) {
            auto d=service.description();
            Json info={{"name",path},{"internalName",d.name.empty()?path:d.name},
                {"maxBatchSize",c.integer("analysis.max_batch")},{"usesHumanSLProfile",d.uses_human_profile},{"usingFP16",precision=="float16"?"true":"false"}};
            // TorchScript and MuZero containers do not have a KataGo ModelDesc version.
            info["version"]=d.version<0?Json(nullptr):Json(d.version);models.push_back(info);
        };
        add_model(*eval,a.get("model"));
        if(auto secondary=dynamic_cast<InferenceService*>(eval->secondary.get()))add_model(*secondary,c.text("analysis.human_model"));
    if(protocol=="katago") {int code=source_analysis_session(*eval,c,models,factory,[]{return stop_requested.load();});eval->finish();return code;}
    auto search=search_for(*eval,settings(c,"analysis"),std::stoull(a.get("seed","0")),c,"analysis");
    auto raw=[&](const Game& game) {
        auto planes=web_policy_planes(model,game,a.get("device"),c.text("analysis.inference_precision"));
        Json result={{"precision",planes.precision},{"heads",Json::array()}};
        result["value_logits"]=std::vector<float>(planes.value_logits.data_ptr<float>(),planes.value_logits.data_ptr<float>()+4);
        result["wdl"]=std::vector<float>(planes.wdl.data_ptr<float>(),planes.wdl.data_ptr<float>()+4);
        const int count=game.local_actions();
        const std::array<int,4> heads{0,1,4,5};
        const std::array<std::string,4> names{"policy","opponent_policy","long_optimistic_policy","short_optimistic_policy"};
        for(size_t i=0;i<heads.size();++i) {
            auto logits=planes.logits.data_ptr<float>()+heads[i]*count;
            auto probabilities=planes.probabilities.data_ptr<float>()+heads[i]*count;
            result["heads"].push_back({{"name",names[i]},{"logits",std::vector<float>(logits,logits+count)},
                                      {"probabilities",std::vector<float>(probabilities,probabilities+count)}});
        }
        return result;
    };
    int code=engine_session(std::move(search),*eval,c,models,factory,std::stoull(a.get("seed","0")),protocol,raw,
                           [&](const Game& game,const Config& effective){return move_temperature(effective,"analysis",game);},[]{return stop_requested.load();});
    eval->finish();return code;
}
int arena(const Args& a,const Config& c) {
    std::vector<MatchModel> models;std::set<std::string> ids;
    std::ifstream model_file(a.get("models"));
    if(!model_file)throw std::runtime_error("Cannot read arena models");
    for(std::string line;std::getline(model_file,line);) {
        std::istringstream row(line);MatchModel m;
        if(!(row>>std::quoted(m.id)>>std::quoted(m.path)>>m.seed) || !(row>>std::ws).eof() ||
           m.id.empty() || !ids.insert(m.id).second)throw std::runtime_error("Invalid arena model");
        models.push_back(std::move(m));
    }
    const int size=a.integer("size"),canvas=c.integer("network.canvas");
    std::vector<MatchTask> tasks;std::ifstream task_file(a.get("tasks"));
    if(!task_file)throw std::runtime_error("Cannot read arena tasks");
    for(std::string line;std::getline(task_file,line);) {
        std::istringstream row(line);MatchTask t{};int count;
        if(!(row>>t.pair>>t.a>>t.b>>t.id>>t.seed>>t.generator>>count) || count < -1)
            throw std::runtime_error("Invalid arena task");
        t.generated=count>=0;Game task_game(size,canvas,parse_rule(a.get("rule")),rule_options(c,GameConfigMode::MATCH));
        for(int i=0;i<count;++i) {
            int move;if(!(row>>move)||move<0||move>=task_game.local_actions())throw std::runtime_error("Invalid arena move");
            t.moves.push_back(task_game.native_action(move));
        }
        if(!(row>>std::ws).eof())throw std::runtime_error("Trailing arena task data");
        tasks.push_back(std::move(t));
    }
    auto precision=c.text("match.inference_precision");
    if(precision=="auto")precision=a.get("device").rfind("cuda:",0)==0?"float16":"float32";
    const int load_workers=parse_integer(a.get("model-load-workers","4"),"--model-load-workers");
    if(load_workers<1)throw std::runtime_error("Model load workers must be positive");
    std::mutex cuda_preparation;
    std::vector<std::shared_ptr<PreparedModel>> runtimes;
    return run_matches(c,size,parse_rule(a.get("rule")),precision,models,tasks,a.integer("max-models"),
        [&](const MatchModel& m,const MatchEventSink& emit){auto args=a;args.args["model"]=m.path;args.args["model-id"]=m.id;
            args.args["seed"]=std::to_string(m.seed);
            if(is_cuda_model(m.path)){
                std::lock_guard<std::mutex> lock(cuda_preparation);
                return evaluator(args,c,"match");
            }
            // Deserialization and layout inspection are CPU-only and independent.
            // CUDA allocations, warmup and capture stay isolated from other models.
            auto started=std::chrono::steady_clock::now();
            c10::InferenceMode inference;
            auto model=torch::jit::load(m.path,torch::kCPU);model.eval();
            auto prepared=std::make_shared<PreparedModel>(model,true);
            double cpu_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            std::lock_guard<std::mutex> lock(cuda_preparation);
            bool reused=false;
            for(const auto& runtime:runtimes)if(prepared->reuse_runtime(*runtime,precision)){reused=true;break;}
            auto service=evaluator(args,c,"match",nullptr,prepared,&model);
            if(!reused)runtimes.push_back(prepared->clone_runtime());
            std::ostringstream row;row<<std::setprecision(17)<<"{\"type\":\"model_prepared\",\"model\":"<<quote(m.id)
                <<",\"cpu_seconds\":"<<cpu_seconds<<",\"weights_seconds\":"<<prepared->timing.weights_seconds
                <<",\"graph_seconds\":"<<prepared->timing.graph_seconds<<",\"reused_runtime\":"<<(reused?"true":"false")<<'}';
            emit(row.str());return service;},
        [&](InferenceService& e,uint64_t seed){return search_for(e,settings(c,"match"),seed,c,"match");},
        [&](const Game& game){return move_temperature(c,"match",game);},
        []{return stop_requested.load();},std::cout,false,load_workers);
}
int match(const Args& a,const Config& c) {
    if(a.get("model-id")==a.get("model-b-id") && a.get("model")!=a.get("model-b"))
        throw std::runtime_error("Match model identity collision");
    const uint64_t seed=std::stoull(a.get("seed",c.text("match.seed")));
    std::vector<MatchModel> models{{a.get("model-id"),a.get("model"),seed},
        {a.get("model-b-id"),a.get("model-b"),seed^0xd1b54a32d192ed03ULL}};
    const int size=a.integer("size"),canvas=c.integer("network.canvas");
    std::vector<MatchTask> tasks;std::set<int> ids;
    std::ifstream input(a.get("tasks"));
    if(!input)throw std::runtime_error("Cannot read match tasks");
    for(std::string line;std::getline(input,line);) {
        std::istringstream row(line);MatchTask t{};t.a=0;t.b=1;int count;
        if(!(row>>t.id>>t.seed>>t.generator>>count) || t.id<0 || (t.generator<0 || t.generator>1) || count < -1 || !ids.insert(t.id).second)
            throw std::runtime_error("Invalid match task");
        Game game(size,canvas,parse_rule(a.get("rule")),rule_options(c,GameConfigMode::MATCH));t.generated=count>=0;
        for(int i=0;i<count;++i) {
            int local;if(!(row>>local) || local<0 || local>=game.local_actions())throw std::runtime_error("Invalid opening action");
            int action=game.native_action(local);game.play(action);t.moves.push_back(action);
        }
        if(!(row>>std::ws).eof() || game.finished())throw std::runtime_error("Invalid opening task");
        tasks.push_back(std::move(t));
    }
    auto precision=c.text("match.inference_precision");
    if(precision=="auto")precision=a.get("device").rfind("cuda:",0)==0?"float16":"float32";
    return run_matches(c,size,parse_rule(a.get("rule")),precision,models,tasks,2,
        [&](const MatchModel& m,const MatchEventSink&){auto args=a;args.args["model"]=m.path;args.args["model-id"]=m.id;
            args.args["seed"]=std::to_string(m.seed);return evaluator(args,c,"match");},
        [&](InferenceService& e,uint64_t search_seed){return search_for(e,settings(c,"match"),search_seed,c,"match");},
        [&](const Game& game){return move_temperature(c,"match",game);},
        []{return stop_requested.load();},std::cout,true);
}

}
int main(int argc,char** argv) {
    try {
        if(argc<2)throw std::runtime_error("Expected selfplay, worker, analysis, gtp, infer, match or arena");
        std::string mode=argv[1]; Args args(argc,argv); Config config(args.get("config"));
        if(mode=="selfplay" || mode=="worker" || mode=="infer")
            if((config.text("agent.algorithm")!="alphazero" && config.text("agent.algorithm")!="muzero") ||
               (config.text("agent.root_search_algo")!="puct" && config.text("agent.root_search_algo")!="gumbel") ||
               (config.text("agent.nonroot_search_algo")!="puct" && config.text("agent.nonroot_search_algo")!="gumbel") ||
               (config.text("agent.root_search_algo")=="puct" && config.text("agent.nonroot_search_algo")!="puct"))
                throw std::runtime_error("Invalid native algorithm/search combination");
        torch::set_num_threads(config.integer((mode=="analysis"||mode=="gtp")?"analysis.cpu_threads":(mode=="match"||mode=="arena")?"match.cpu_threads":"run.cpu_threads")); torch::set_num_interop_threads(1);
        std::signal(SIGINT,stop_handler); std::signal(SIGTERM,stop_handler);
        if(mode=="selfplay") {ForkPool forks;auto service=evaluator(args,config,"selfplay");int code=selfplay(args,config,*service,forks);service->finish();return code;}
        if(mode=="worker")return worker(args,config);
        if(mode=="analysis") {
            const auto stream=args.get("stream","false");
            if(stream!="true" && stream!="false")throw std::runtime_error("--stream must be true or false");
            return stream=="true"?analysis_session(args,config,args.get("protocol","text")):analyze(args,config,false);
        }
        if(mode=="gtp")return analysis_session(args,config,"gtp");
        if(mode=="infer")return analyze(args,config,true);
        if(mode=="match")return match(args,config);
        if(mode=="arena")return arena(args,config);
        throw std::runtime_error("Unknown native command: "+mode);
    } catch(const std::exception& error) {std::cerr<<"EtaZero: "<<error.what()<<'\n';return 1;}
}
