#include "etazero/config.h"
#include "etazero/record.h"
#include "etazero/torch_backend.h"
#include "etazero/random_evaluator.h"
#include "etazero/search_limits.h"
#include "etazero/sampling.h"
#include "etazero/muzero/search.h"
#include "etazero/gumbel_search.h"
#include "etazero/muzero/torch_backend.h"
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
                                          "config-id","source-id","iteration","worker","seed","size","rule","moves","model-b","model-b-id","evaluator","tasks","stream"};
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
SearchSettings settings(const Config& c, const std::string& mode) {
    bool selfplay = mode == "selfplay";
    std::string section = selfplay ? "search" : mode == "match" ? "match" : "analysis";
    auto key = [&](const std::string& group, const std::string& name) {
        return (selfplay ? group : section) + "." + name;
    };
    SearchSettings s{selfplay?c.integer("search.full_search_visits")-1:std::max(1,c.integer(section+".visits")-1),
                     c.integer(key("parallelism","search_threads")),
                     c.number(key("puct","c_puct")),c.number(key("puct","virtual_loss")),
                     selfplay?c.number("dirichlet_noise.noise_fraction"):0,
                     selfplay?c.number("dirichlet_noise.dirichlet_total_concentration"):1,c.boolean(key("search","reuse_tree")),
                     selfplay?0:c.integer(section+".visits")};
    s.graph_search=c.boolean(key("graph_search","use_graph_search"));
    s.graph_catch_up_leak_prob=c.number(key("graph_search","graph_search_catch_up_leak_prob"));
    s.use_uncertainty=c.boolean(key("uncertainty","use_uncertainty"));
    s.uncertainty_coeff=c.number(key("uncertainty","uncertainty_coeff"));
    s.uncertainty_exponent=c.number(key("uncertainty","uncertainty_exponent"));
    s.uncertainty_max_weight=c.number(key("uncertainty","uncertainty_max_weight"));
    s.policy_optimism=c.number(key("optimistic_policy","policy_optimism"));
    s.root_policy_optimism=c.number(key("optimistic_policy","root_policy_optimism"));
    s.use_noise_pruning=c.boolean(key("noise_pruning","use_noise_pruning"));
    s.noise_prune_utility_scale=c.number(key("noise_pruning","noise_prune_utility_scale"));
    s.noise_pruning_cap=c.number(key("noise_pruning","noise_pruning_cap"));
    s.use_fpu=c.boolean(key("fpu","use_fpu"));s.fpu_reduction_max=c.number(key("fpu","fpu_reduction_max"));
    s.root_fpu_reduction_max=c.number(key("fpu","root_fpu_reduction_max"));
    s.fpu_parent_power=c.number(key("fpu","fpu_parent_weight_by_visited_policy_pow"));
    s.fpu_parent_weight_by_visited_policy=c.boolean(key("fpu","fpu_parent_weight_by_visited_policy"));
    s.fpu_parent_weight=c.number(key("fpu","fpu_parent_weight"));
    s.max_playouts=c.integer(key("search","max_playouts"));s.max_time=c.number(key("search","max_time"));
    s.use_lcb=c.boolean(key("lcb","use_lcb"));s.lcb_stdevs=c.number(key("lcb","lcb_stdevs"));
    s.min_lcb_visit_prop=c.number(key("lcb","min_visit_prop_for_lcb"));
    s.policy_target_pruning=c.boolean(key("policy_target","policy_target_pruning"));
    s.value_weight_exponent=c.number(key("value_weighting","value_weight_exponent"));
    s.chosen_move_subtract=c.number(key("policy_target","chosen_move_subtract"));s.chosen_move_prune=c.number(key("policy_target","chosen_move_prune"));
    s.fpu_loss_prop=c.number(key("fpu","fpu_loss_prop"));s.root_fpu_loss_prop=c.number(key("fpu","root_fpu_loss_prop"));
    s.c_puct_log=c.number(key("puct","c_puct_log"));s.c_puct_base=c.number(key("puct","c_puct_base"));
    s.c_puct_stdev_prior=c.number(key("puct","c_puct_stdev_prior"));s.c_puct_stdev_prior_weight=c.number(key("puct","c_puct_stdev_prior_weight"));
    s.c_puct_stdev_scale=c.number(key("puct","c_puct_stdev_scale"));
    s.root_symmetries=c.integer(key("symmetry","root_num_symmetries_to_sample"));
    s.nn_randomize=c.boolean(key("symmetry","nn_randomize"));s.nn_symmetry=c.integer(key("symmetry","nn_symmetry"));
    s.nn_policy_temperature=c.number(key("temperature","nn_policy_temperature"));
    s.root_policy_temperature=c.number(key("temperature","root_policy_temperature"));
    s.root_policy_temperature_early=c.number(key("temperature","root_policy_temperature_early"));
    s.temperature_halflife=c.number(key("temperature","temperature_halflife"));
    s.chosen_move_temperature_only_below_prob=c.number(key("temperature","temperature_only_below_prob"));
    if(selfplay){s.shaped_noise=c.boolean("dirichlet_noise.shaped_dirichlet_noise");s.forced_playouts=c.number("forced_playouts.root_desired_per_child_visits_coeff");}
    return s;
}
double move_temperature(const Config& c,const std::string& mode,const Game& game) {
    bool selfplay=mode=="selfplay";
    std::string section=selfplay?"temperature":mode=="match"?"match":"analysis";
    return temperature_at_turn(c.number(section+(selfplay?".temperature":".temperature_early")),
        c.number(section+(selfplay?".final_temperature":".temperature")),
        c.number(section+".temperature_halflife"),game.turn(),game.size()*game.size());
}
template<class T> void array(std::ostream& out, const std::vector<T>& values) {
    out << '['; bool first = true;
    for (auto x : values) { if (!first) out << ','; out << x; first = false; } out << ']';
}
std::unique_ptr<InferenceService> evaluator(const Args& a, const Config& c, const std::string& mode, bool second = false,
                                          torch::jit::Module* web_model = nullptr,
                                          std::shared_ptr<PreparedModel> prepared = nullptr) {
    std::string prefix = (mode == "infer" || mode == "selfplay") ? "inference" : mode == "analysis" ? "analysis" : mode;
    int batch = c.integer(prefix+".max_batch"), canvas = c.integer("network.canvas");
    std::vector<std::unique_ptr<Backend>> backends;
    std::string kind=a.get("evaluator","network");
    if(kind!="random" && kind!="network")throw std::runtime_error("Unknown evaluator: "+kind);
    if(kind=="random" && mode!="selfplay")throw std::runtime_error("Random evaluator is for cold-start selfplay only");
    std::string algorithm=c.contains("agent.algorithm")?c.text("agent.algorithm"):"alphazero";
    std::optional<torch::jit::Module> detected_model;
    if(kind=="network" && mode!="selfplay" && mode!="infer") {
        // Detect from the actual inference model, then hand it to its backend.
        // Selfplay already knows the algorithm and loads during backend preparation.
        torch::Device device(a.get("device"));c10::cuda::OptionalCUDAGuard guard;
        if(device.is_cuda())guard.set_device(device);
        detected_model=torch::jit::load(a.get(second?"model-b":"model"),device);
        detected_model->eval();
        if(device.is_cuda())c10::cuda::getCurrentCUDAStream(device.index()).synchronize();
        auto metadata=detected_model->get_method("metadata")({}).toTuple();
        if(metadata->elements().size()==4)algorithm=metadata->elements()[2].toStringRef();
        else if(metadata->elements().size()==2)algorithm="alphazero";
        else throw std::runtime_error("Unknown model metadata");
        if(web_model)*web_model=*detected_model; // Share immutable weights with Web-only diagnostics.
    }
    if(!prepared)prepared=detected_model?std::make_shared<PreparedModel>(std::move(*detected_model)):
                                        std::make_shared<PreparedModel>();
    if(algorithm=="muzero") {
        std::vector<std::unique_ptr<muzero::Backend>> models;
        for(int i=0;i<c.integer(prefix+".server_threads");++i) {
            if(kind=="random")models.push_back(std::make_unique<muzero::RandomBackend>(canvas,std::stoull(a.get("seed"))));
            else {
                auto precision=c.text(prefix+".inference_precision");
                if(precision=="auto")precision=a.get("device").rfind("cuda:",0)==0?"float16":"float32";
                models.push_back(std::make_unique<muzero::TorchBackend>(a.get(second?"model-b":"model"),a.get("device"),canvas,batch,precision,
                    prepared));
            }
        }
        return std::make_unique<muzero::BatchEvaluator>(std::move(models),canvas,batch,c.integer(prefix+".batch_wait_us"),
            c.integer(prefix+".cache_entries"),c.boolean((prefix=="inference"?"symmetry":prefix)+".nn_randomize"),
            c.integer((prefix=="inference"?"symmetry":prefix)+".nn_symmetry"),
            std::stoull(a.get("seed",prefix=="inference"?c.text("run.seed"):c.text(prefix+".seed")))^(second?0xd1b54a32d192ed03ULL:0),c.integer(prefix+".queue_capacity"));
    }
    if(algorithm!="alphazero")throw std::runtime_error("Unknown model algorithm");
    for (int i=0;i<c.integer(prefix+".server_threads");++i) {
        if(kind=="random")backends.push_back(std::make_unique<RandomBackend>(canvas,std::stoull(a.get("seed"))));
        else backends.push_back(std::make_unique<TorchBackend>(a.get(second ? "model-b" : "model"),a.get("device"),canvas,batch,
                                                         c.text(prefix+".inference_precision"),prepared));
    }
    return std::make_unique<BatchEvaluator>(std::move(backends), a.get(second ? "model-b-id" : "model-id"), canvas,
                                          batch, c.integer(prefix+".queue_capacity"), c.integer(prefix+".batch_wait_us"),
                                          c.integer(prefix+".cache_entries"),c.boolean((prefix=="inference"?"symmetry":prefix)+".nn_randomize"),
                                          c.integer((prefix=="inference"?"symmetry":prefix)+".nn_symmetry"),
                                          std::stoull(a.get("seed",prefix=="inference"?c.text("run.seed"):c.text(prefix+".seed")))^(second?0xd1b54a32d192ed03ULL:0));
}
std::unique_ptr<GameSearch> search_for(InferenceService& evaluator,SearchSettings settings,uint64_t seed,const Config& c,const std::string& mode) {
    std::string prefix=mode=="selfplay"?"agent":mode;
    std::string root=c.contains(prefix+".root_search_algo")?c.text(prefix+".root_search_algo"):"puct";
    std::string nonroot=c.contains(prefix+".nonroot_search_algo")?c.text(prefix+".nonroot_search_algo"):"puct";
    if((root!="puct"&&root!="gumbel")||(nonroot!="puct"&&nonroot!="gumbel")||(root=="puct"&&nonroot!="puct"))
        throw std::runtime_error("Invalid root/nonroot search combination");
    if(root=="gumbel") {
        GumbelSettings g;auto key=[&](const std::string& name){return mode=="selfplay"?"gumbel."+name:mode+".gumbel_"+name;};
        if(c.contains(key("max_num_considered_actions")))g.max_num_considered_actions=c.integer(key("max_num_considered_actions"));
        if(c.contains(key("c_visit")))g.c_visit=c.number(key("c_visit"));
        if(c.contains(key("c_scale")))g.c_scale=c.number(key("c_scale"));
        g.noise_scale=mode=="selfplay"?1:0;
        if(c.contains(key("noise_scale")))g.noise_scale=c.number(key("noise_scale"));
        if(c.contains(key("rescale_q_values")))g.rescale_q_values=c.boolean(key("rescale_q_values"));
        if(c.contains(key("action_selection"))) {
            auto behavior=c.text(key("action_selection"));
            if(behavior!="gumbel"&&behavior!="visit")throw std::runtime_error("Unknown Gumbel action selection");
            g.sample_visits=behavior=="visit";
        }
        if(auto mu=dynamic_cast<muzero::Evaluator*>(&evaluator)) {
            if(nonroot=="gumbel")return std::make_unique<muzero::GumbelSearch<true>>(*mu,settings,g,seed);
            return std::make_unique<muzero::GumbelSearch<false>>(*mu,settings,g,seed);
        }
        if(nonroot=="gumbel")return std::make_unique<GumbelSearch<true>>(evaluator,settings,g,seed);
        return std::make_unique<GumbelSearch<false>>(evaluator,settings,g,seed);
    }
    if(auto mu=dynamic_cast<muzero::Evaluator*>(&evaluator))return std::make_unique<muzero::Search>(*mu,settings,seed);
    return std::make_unique<Search>(evaluator,settings,seed);
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
              << ",\"max_batch\":" << e.max_observed_batch << ",\"queue_wait_us\":" << e.wait_microseconds
              << ",\"submitted\":" << e.submitted << ",\"cache_hits\":" << e.cache_hits << ",\"rows_by_server\":";
    array(std::cout,e.rows_by_server);std::cout<<"}"<<std::endl;
}
int selfplay(const Args& a,const Config& c,InferenceService& service,ForkPool& forks) {
    int count = a.integer("games"), canvas = c.integer("network.canvas");
    if (count < 1) throw std::runtime_error("Selfplay requires positive game count");
    auto* eval=&service;eval->reset_stats();
    OpeningConfig opening(c,"policy_init"); auto search_config=selfplay_search_config(c);auto reanalysis=reanalysis_config(c);
    GameForkConfig fork_config(c);auto hints=load_hint_positions(c,canvas);
    bool random=a.get("evaluator","network")=="random";
    Source source{a.get("run-id"),a.get("attempt-id"),a.get("model-id"),a.get("config-id"),a.get("source-id"),a.integer("iteration"),a.integer("worker")};
    if(c.text("agent.algorithm")=="muzero")source.unroll_steps=c.integer("unroll.steps");
    source.gumbel=c.text("agent.root_search_algo")=="gumbel";source.full_gumbel=c.text("agent.nonroot_search_algo")=="gumbel";
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
                if(!initial)initial=sample_hint_position(hints,c.number("hint_positions.hint_positions_prob"),rng);
                if(initial){size=initial->game.size();rule=initial->game.rule();}
                Game game=initial?initial->game:Game(size,canvas,rule);search.reset(rng());
                FinishedGame record{}; record.id=id; record.seed=game_seed; record.size=size; record.canvas=canvas; record.rule=rule;
                record.forbidden_feature_dropout_prob=c.number("environment.forbidden_feature_dropout_prob");
                InitialKind kind=initial?initial->kind:InitialKind::Ordinary;
                bool fork_start=kind==InitialKind::EarlyFork || kind==InitialKind::GameFork || kind==InitialKind::HintFork;
                auto advantage=fork_start?PlayoutAdvantage{}:sample_playout_advantage(c.number("pda.normal_asymmetric_playout_prob"),c.number("pda.max_asymmetric_ratio"),rng);
                std::vector<Game> side_positions,positions;
                std::vector<double> historical_values;
                if(initial) {
                    record.opening.actions=initial->actions;record.opening.initial_position_moves=initial->actions.size();
                    record.opening.initial_position_kind=static_cast<int>(kind);record.opening.hint_action=initial->hint_action;
                } else if(!random) {
                    record.opening=initialize_opening(game,opening,*eval,rng,[&]{return stop_requested || failure;});
                    if(record.opening.status==OpeningStatus::Interrupted)break;
                }
                Game prefix(size,canvas,rule);prefix.set_pda(advantage.doublings,advantage.player);
                for(int action:record.opening.actions) {
                    Step step{prefix.player(),action,0,0,0,prefix.observation(),
                              std::vector<double>(canvas*canvas,0),std::vector<int64_t>(canvas*canvas,0),false};
                    prefix.play(action);if(prefix.finished())step.reward=prefix.winner()*step.player;
                    record.steps.push_back(std::move(step));
                }
                const Game start=game;
                game.set_pda(advantage.doublings,advantage.player);
                while (!game.finished() && !stop_requested && !failure) {
                    positions.push_back(game);
                    double temp = move_temperature(c,"selfplay",game);
                    auto context=hint_context(game,start,record.opening.hint_action,kind);
                    double cheap_probability=cheap_search_probability(search_config,context);
                    bool cheap=cheap_probability>0 && std::bernoulli_distribution(cheap_probability)(rng);
                    auto limits=selfplay_search_limits(search_config,historical_values,cheap,advantage,game.player(),context);
                    limits.search.should_stop=[&]{return stop_requested.load() || failure.load();};
                    limits.search.collect_root_policy_invalid_mass=!random && source.unroll_steps>0;
                    auto result = search.run(game,temp,limits.search);
                    if(stop_requested || failure)break;
                    if(result.action<0)throw std::runtime_error("Selfplay search has no move: zero playout budget");
                    if(limits.search.collect_root_policy_invalid_mass) {
                        record.root_policy_invalid_mass_sum+=result.root_policy_invalid_mass;
                        ++record.root_policy_invalid_mass_count;
                    }
                    Step step{game.player(),result.action,result.simulations,static_cast<float>(temp),0,
                              game.observation(),std::move(result.policy),std::move(result.visits)};
                    step.cheap_search=limits.cheap_search;step.target_weight=limits.target_weight;
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
                apply_training_weights(record,c.number("surprise_weighting.policy_surprise_data_weight"),c.number("surprise_weighting.value_surprise_data_weight"),rng,reanalysis.direct_value_surprise,reanalysis.enabled);
                search_side_positions(side_positions,record,search,*eval,search_config.full_visits,
                    c.boolean("symmetry.nn_randomize"),c.integer("symmetry.nn_symmetry"),
                    [&](const Game& position){return move_temperature(c,"selfplay",position);},rng,
                    [&]{return stop_requested.load() || failure.load();});
                if(stop_requested || failure)break;
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
            cached=evaluator(request,config,"selfplay",false,nullptr,prepared);
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
    Game game(a.integer("size"),c.integer("network.canvas"),parse_rule(a.get("rule")));
    moves(game,a.get("moves",""));
    if(!raw)game.set_pda(c.number("analysis.playout_doubling_advantage"),c.text("analysis.playout_doubling_advantage_player")=="black"?1:-1);
    if (game.finished() && !raw) {
        std::cout << "{\"terminal\":true,\"winner\":" << game.winner() << ",\"value\":" << game.terminal_value() << ",\"reason\":" << game.reason() << "}\n";
        return 0;
    }
    auto eval=evaluator(a,c,raw?"infer":"analysis");
    // Raw diagnostics must not seed the search cache with an identity-orientation output.
    auto network=eval->evaluate_symmetry(game.observation(),0,true);
    std::cout.precision(12);
    std::cout << "{\"terminal\":" << (game.finished()?"true":"false") << ",\"raw_value\":" << network.value() << ",\"raw_logits\":";
    array(std::cout,network.logits);std::cout << ",\"raw_wdl\":";array(std::cout,std::vector<double>(network.wdl.begin(),network.wdl.end()));
    std::cout << ",\"raw_optimistic_logits\":";array(std::cout,network.optimistic_logits);
    std::cout << ",\"raw_shortterm_value_stdev\":" << network.shortterm_value_stdev;
    if(raw)if(auto mu=dynamic_cast<muzero::Evaluator*>(eval.get())) {
        auto output=mu->initial(game.observation());std::cout<<",\"recurrent\":[";
        int index=0;
        for(int action:{0,game.canvas(),0}) {
            output=mu->recurrent(output.latent,action);const auto& e=output.evaluation;
            if(index++)std::cout<<',';
            std::cout<<"{\"logits\":";array(std::cout,e.logits);
            std::cout<<",\"wdl\":";array(std::cout,std::vector<double>(e.wdl.begin(),e.wdl.end()));
            std::cout<<",\"optimistic_logits\":";array(std::cout,e.optimistic_logits);
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
// Interactive actions use the actual board width, unlike search's canvas indices.
int web_integer(const std::string& text) {
    int value;
    auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())
        throw std::runtime_error("Expected integer argument");
    return value;
}
uint64_t web_seed(const std::string& text) {
    uint64_t value;
    auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())
        throw std::runtime_error("Expected uint64 seed");
    return value;
}
struct WebOpening {
    OpeningResult result;
    uint64_t seed;
    double seconds;
};
void web_state(const Game& game,const std::vector<int>& moves,const std::optional<WebOpening>& opening) {
    std::cout<<"{\"board_size\":"<<game.size()<<",\"canvas_size\":"<<game.canvas()
             <<",\"player\":"<<game.player()<<",\"turn\":"<<game.turn()
             <<",\"finished\":"<<(game.finished()?"true":"false")<<",\"winner\":"<<game.winner()
             <<",\"reason\":"<<game.reason()<<",\"board\":";
    array(std::cout,game.board().cells);std::cout<<",\"moves\":";array(std::cout,moves);
    std::cout<<",\"opening\":";
    if(opening) {
        const auto& result=opening->result;
        // Seeds are strings in JSON so browsers retain all 64 bits.
        std::cout<<"{\"seed\":"<<quote(std::to_string(opening->seed))<<",\"attempts\":"<<result.attempts
                 <<",\"balanced_moves\":"<<result.balanced_moves<<",\"policy_moves\":"<<result.policy_moves
                 <<",\"value\":"<<result.start_value<<",\"value_player\":"<<(result.balanced_moves%2==0?1:-1)
                 <<",\"seconds\":"<<opening->seconds<<",\"moves\":";
        std::vector<int> local;
        for(int action:result.actions)local.push_back(action/game.canvas()*game.size()+action%game.canvas());
        array(std::cout,local);std::cout<<'}';
    } else std::cout<<"null";
    std::cout<<'}';
}
struct WebPolicyPlanes {
    torch::Tensor logits, probabilities;
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
    auto observation=game.observation();const int canvas=game.canvas(),area=game.actions(),size=game.size();
    auto obs=torch::from_blob(observation.data(),{1,INPUT_PLANES,canvas,canvas},torch::kFloat32).to(device);
    const bool hex_white=game.rule()==Rule::HEX && game.player()==-1;
    if(hex_white)obs=obs.transpose(2,3).contiguous();
    auto globals=torch::from_blob(observation.data()+INPUT_PLANES*area,{1,GLOBAL_FEATURES},torch::kFloat32).to(device);
    auto network=model.attr("model").toModule();
    auto metadata=model.get_method("metadata")({}).toTuple();
    c10::IValue output;
    if(metadata->elements().size()==4) {
        auto hidden=network.attr("representation").toModule().forward({obs,globals});
        output=network.attr("prediction").toModule().forward({hidden});
    } else output=network.get_method("forward_all")({obs,globals});
    auto logits=output.toTuple()->elements()[0].toTensor();
    if(logits.dim()!=3 || logits.size(0)!=1 || logits.size(1)<6 || logits.size(2)!=area)
        throw std::runtime_error("Web policy output shape mismatch");
    if(hex_white)logits=logits.reshape({1,logits.size(1),canvas,canvas}).transpose(2,3).contiguous().reshape({1,logits.size(1),area});
    // Match the training domain: all on-board points, INCLUDING occupied cells.
    // Canonical orientation, temperature 1, without search optimism/noise/ensemble.
    logits=logits[0].narrow(0,0,6).reshape({6,canvas,canvas}).narrow(1,0,size).narrow(2,0,size)
        .to(torch::kFloat32).contiguous().reshape({6,size*size});
    if(!torch::isfinite(logits).all().item<bool>())throw std::runtime_error("Nonfinite Web policy logits");
    auto probabilities=torch::softmax(logits,1).to(torch::kCPU).contiguous();
    return {logits.to(torch::kCPU).contiguous(),probabilities,precision};
}
void web_analysis(const SearchResult& result,const Game& game,double seconds,uint64_t requests,uint64_t batches,
                  const WebPolicyPlanes& planes) {
    std::vector<int> candidates;
    int64_t total=0;
    for(int a=0;a<game.actions();++a)if(game.legal(a)){candidates.push_back(a);total+=result.visits[a];}
    std::stable_sort(candidates.begin(),candidates.end(),[&](int a,int b){return result.move_policy[a]>result.move_policy[b];});
    std::cout<<"{\"action\":"<<result.action/game.canvas()*game.size()+result.action%game.canvas()
             <<",\"board_size\":"<<game.size()<<",\"canvas_size\":"<<game.canvas()
             <<",\"turn\":"<<game.turn()<<",\"player\":"<<game.player()<<",\"root_value\":"<<result.value
             <<",\"completed_visits\":"<<result.root_visits<<",\"seconds\":"<<seconds
             <<",\"requests\":"<<requests<<",\"batches\":"<<batches<<",\"wdl\":";
    array(std::cout,std::vector<double>(result.search_wdl.begin(),result.search_wdl.end()));
    std::cout<<",\"network_wdl\":";
    array(std::cout,std::vector<double>(result.network_wdl.begin(),result.network_wdl.end()));
    std::cout<<",\"board\":";array(std::cout,game.board().cells);std::cout<<",\"candidates\":[";
    for(size_t i=0;i<candidates.size();++i) {
        int a=candidates[i];if(i)std::cout<<',';
        std::cout<<"{\"action\":"<<a/game.canvas()*game.size()+a%game.canvas()<<",\"visits\":"<<result.visits[a]
                 <<",\"network_prior\":"<<result.network_policy[a]<<",\"visit_policy\":"<<double(result.visits[a])/total
                 <<",\"selection_weight\":"<<result.move_policy[a]<<'}';
    }
    std::cout<<"],\"network_planes\":{\"precision\":"<<quote(planes.precision)<<",\"heads\":[";
    const int count=game.size()*game.size();
    const std::array<int,4> displayed_heads{0,1,4,5};
    const std::array<std::string,4> names{"policy","opponent_policy","long_optimistic_policy","short_optimistic_policy"};
    for(size_t i=0;i<displayed_heads.size();++i) {
        int head=displayed_heads[i];if(i)std::cout<<',';
        auto logits=planes.logits.data_ptr<float>()+head*count;
        auto probabilities=planes.probabilities.data_ptr<float>()+head*count;
        std::cout<<"{\"name\":"<<quote(names[i])<<",\"logits\":";array(std::cout,std::vector<float>(logits,logits+count));
        std::cout<<",\"probabilities\":";array(std::cout,std::vector<float>(probabilities,probabilities+count));std::cout<<'}';
    }
    std::cout<<"]}}";
}
int analysis_session(const Args& a,const Config& c) {
    torch::jit::Module web_model;
    auto eval=evaluator(a,c,"analysis",false,&web_model);
    auto search_owner=search_for(*eval,settings(c,"analysis"),std::stoull(a.get("seed","0")),c,"analysis");auto& search=*search_owner;
    std::optional<Game> game;
    std::optional<WebOpening> opening;
    std::vector<int> played;
    const int canvas=c.integer("network.canvas");
    std::cout<<std::setprecision(17)<<"{\"ok\":true,\"canvas_size\":"<<canvas<<"}\n"<<std::flush;
    std::string line;
    while(!stop_requested && std::getline(std::cin,line)) {
        try {
            std::istringstream input(line);std::vector<std::string> words;
            for(std::string word;input>>word;)words.push_back(word);
            if(words.empty())throw std::runtime_error("Empty command");
            if(words[0]=="quit" && words.size()==1)break;
            if(words[0]=="new" && (words.size()==3 || words.size()==6)) {
                Game replacement(web_integer(words[1]),canvas,parse_rule(words[2]));
                std::optional<WebOpening> generated;
                std::vector<int> moves;
                if(words.size()==6) {
                    if(words[3]!="balanced")throw std::runtime_error("Unknown opening kind");
                    uint64_t seed=web_seed(words[4]);
                    int timeout_ms=web_integer(words[5]);
                    if(timeout_ms<1 || timeout_ms>120000)throw std::runtime_error("Opening timeout must be in [1, 120000] ms");
                    OpeningConfig config(c);
                    if(replacement.rule()==Rule::HEX ? (config.hex_probability!=1 || config.hex_make_fair_probability!=1 || config.hex_min_accept_rate<=0) : (config.probability!=1 || config.rejection_probability_fallback>=1))
                        throw std::runtime_error("Balanced openings require probability one and a rejection fallback below one");
                    std::mt19937_64 random(seed);
                    auto started=std::chrono::steady_clock::now();
                    auto deadline=started+std::chrono::milliseconds(timeout_ms);
                    for(int retry=0;retry<100;++retry) {
                        replacement=Game(replacement.size(),canvas,replacement.rule());
                        auto result=initialize_opening(replacement,config,*eval,random,[&]{
                            return stop_requested.load() || std::chrono::steady_clock::now()>=deadline;
                        });
                        if(result.status==OpeningStatus::Interrupted || stop_requested.load() || std::chrono::steady_clock::now()>=deadline)
                            throw std::runtime_error("平衡开局生成已取消或超时，请重新创建棋局");
                        if(result.status!=OpeningStatus::Success)
                            throw std::runtime_error("Balanced opening failed: "+result.failure);
                        if(replacement.finished())continue;
                        for(int action:result.actions)moves.push_back(action/canvas*replacement.size()+action%canvas);
                        generated=WebOpening{std::move(result),seed,
                            std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()};
                        break;
                    }
                    if(!generated)throw std::runtime_error("100 consecutive terminal openings");
                }
                game=std::move(replacement);played=std::move(moves);opening=std::move(generated);
                search.reset(std::stoull(a.get("seed","0")));
            } else if(!game)throw std::runtime_error("Start a new game first");
            else if(words[0]=="play" && words.size()==2) {
                int local=web_integer(words[1]);
                if(local<0 || local>=game->size()*game->size())throw std::runtime_error("Move outside board");
                game->play(local/game->size()*canvas+local%game->size());played.push_back(local);
            } else if(words[0]=="undo" && words.size()==2) {
                int count=web_integer(words[1]);
                if(count<1 || count>static_cast<int>(played.size()))throw std::runtime_error("Invalid undo count");
                Game replacement(game->size(),canvas,game->rule());
                auto retained=played;retained.resize(retained.size()-count);
                for(int local:retained)replacement.play(local/replacement.size()*canvas+local%replacement.size());
                game=std::move(replacement);played=std::move(retained);
                // Manual research may alter the generated prefix; its old value then no longer applies.
                if(opening && played.size()<opening->result.actions.size())opening.reset();
            } else if((words[0]=="genmove" || words[0]=="analyze") && words.size()==2) {
                int visits=web_integer(words[1]);
                if(visits<2 || visits>100000)throw std::runtime_error("Visits must be in [2, 100000]");
                Game before=*game;
                SearchRun options;options.max_visits=visits;options.clear_before_search=true;
                options.should_stop=[]{return stop_requested.load();};
                auto requests=eval->requests.load(),batches=eval->batches.load();
                auto start=std::chrono::steady_clock::now();
                auto result=analyze_position(search,c,before,options);
                if(stop_requested)break;
                if(result.action<0)throw std::runtime_error("Search has no move: zero playout budget");
                double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                auto planes=web_policy_planes(web_model,before,a.get("device"),c.text("analysis.inference_precision"));
                if(words[0]=="genmove") {
                    game->play(result.action);played.push_back(result.action/canvas*game->size()+result.action%canvas);
                }
                std::cout<<"{\"ok\":true,\"state\":";web_state(*game,played,opening);std::cout<<",\"analysis\":";
                web_analysis(result,before,seconds,eval->requests.load()-requests,eval->batches.load()-batches,planes);
                std::cout<<"}\n"<<std::flush;continue;
            } else if(!(words[0]=="state" && words.size()==1))throw std::runtime_error("Unknown command or arguments");
            std::cout<<"{\"ok\":true,\"state\":";web_state(*game,played,opening);std::cout<<"}\n"<<std::flush;
        } catch(const std::exception& error) {
            std::cout<<"{\"ok\":false,\"error\":"<<quote(error.what())<<"}\n"<<std::flush;
        }
    }
    eval->finish();return stop_requested?130:0;
}
int match(const Args& a,const Config& c) {
    auto match_start=std::chrono::steady_clock::now();
    std::atomic<int> completed_games{0};
    const bool same_bot=a.get("model-id")==a.get("model-b-id");
    if(same_bot && a.get("model")!=a.get("model-b"))throw std::runtime_error("Match model identity collision");
    auto ea=evaluator(a,c,"match"),eb=evaluator(a,c,"match",true);
    const int size=a.integer("size"),canvas=c.integer("network.canvas");
    OpeningConfig opening(c);
    if(parse_rule(a.get("rule"))==Rule::HEX ? (opening.hex_probability!=1 || opening.hex_make_fair_probability!=1 || opening.hex_min_accept_rate<=0) : (opening.probability!=1 || opening.rejection_probability_fallback>=1))
        throw std::runtime_error("Matches require balanced openings and a rejection fallback below one");
    struct Task {int id,generator,mask;uint64_t seed;bool generated;std::vector<int> moves;};
    std::vector<Task> tasks;std::set<int> ids;
    std::ifstream input(a.get("tasks"));
    if(!input)throw std::runtime_error("Cannot read match tasks");
    for(std::string line;std::getline(input,line);) {
        std::istringstream row(line);Task t{};int count;
        if(!(row>>t.id>>t.seed>>t.generator>>t.mask>>count) || t.id<0 || t.generator!=t.id%2 ||
           t.mask<1 || t.mask>3 || count < -1 || count>=size*size || !ids.insert(t.id).second)
            throw std::runtime_error("Invalid match task");
        Game game(size,canvas,parse_rule(a.get("rule")));t.generated=count>=0;
        for(int i=0;i<count;++i) {
            int local;if(!(row>>local) || local<0 || local>=size*size)throw std::runtime_error("Invalid opening action");
            int action=local/size*canvas+local%size;game.play(action);t.moves.push_back(action);
        }
        if(!(row>>std::ws).eof() || game.finished())throw std::runtime_error("Invalid opening task");
        tasks.push_back(std::move(t));
    }
    std::mutex output;
    auto write_moves=[&](std::ostream& out,const std::vector<int>& moves) {
        std::vector<int> local;for(int move:moves)local.push_back(move/canvas*size+move%canvas);array(out,local);
    };
    auto workers=[&](int count,const std::function<void(int)>& work) {
        std::atomic<int> next{0};std::atomic<bool> failed{false};std::mutex mutex;std::exception_ptr error;
        std::vector<std::thread> threads;
        auto loop=[&]{try{while(!stop_requested && !failed){int i=next++;if(i>=count)break;work(i);}}
            catch(...){failed=true;std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();}};
        try{for(int i=0;i<std::min(count,c.integer("match.game_threads"));++i)threads.emplace_back(loop);}
        catch(...){failed=true;for(auto& t:threads)t.join();throw;}
        for(auto& t:threads)t.join();if(error)std::rethrow_exception(error);
    };
    workers(tasks.size(),[&](int index){
        auto& task=tasks[index];if(task.generated)return;
        std::mt19937_64 random(task.seed);
        for(int retry=0;retry<100;++retry) {
            if(stop_requested)return;
            Game game(size,canvas,parse_rule(a.get("rule")));
            auto result=initialize_opening(game,opening,task.generator==0?*ea:*eb,task.generator==0?*eb:*ea,random,[]{return stop_requested.load();});
            if(result.status==OpeningStatus::Interrupted)return;
            if(result.status!=OpeningStatus::Success)throw std::runtime_error("Balanced match opening failed: "+result.failure);
            if(game.finished())continue;
            task.moves=result.actions;task.generated=true;
            std::ostringstream row;row<<std::setprecision(17)<<"{\"type\":\"opening\",\"id\":"<<task.id
                <<",\"generator\":"<<task.generator<<",\"seed\":"<<task.seed<<",\"attempts\":"<<result.attempts
                <<",\"balanced_moves\":"<<result.balanced_moves<<",\"policy_moves\":"<<result.policy_moves
                <<",\"value\":"<<result.start_value<<",\"moves\":";
            write_moves(row,task.moves);row<<",\"reference_black_a\":"<<(task.generator==0?"true":"false")
                <<",\"balance_evaluators\":";array(row,result.balance_evaluators);
            row<<",\"policy_evaluators\":";array(row,result.policy_evaluators);row<<'}';
            std::lock_guard<std::mutex> lock(output);std::cout<<row.str()<<std::endl;return;
        }
        throw std::runtime_error("100 consecutive terminal openings");
    });
    std::vector<std::pair<size_t,int>> games;
    for(size_t i=0;i<tasks.size();++i)if(tasks[i].generated)
        for(int color=0;color<2;++color)if(tasks[i].mask&(1<<color))games.emplace_back(i,color);
    workers(games.size(),[&](int index){
        auto [task_index,color]=games[index];const auto& task=tasks[task_index];
        Game game(size,canvas,parse_rule(a.get("rule")));for(int action:task.moves)game.play(action);
        game.set_pda(c.number("match.playout_doubling_advantage"),c.text("match.playout_doubling_advantage_player")=="black"?1:-1);
        auto start=std::chrono::steady_clock::now();uint64_t seed=task.seed^(0xd1b54a32d192ed03ULL*(color+1));
        auto sa_owner=search_for(*ea,settings(c,"match"),mix(seed),c,"match");auto& sa=*sa_owner;
        std::unique_ptr<GameSearch> sb;
        if(!same_bot)sb=search_for(*eb,settings(c,"match"),mix(seed^0x9e3779b97f4a7c15ULL),c,"match");
        std::vector<int> actions=task.moves;std::vector<int64_t> visits,initial_visits,new_playouts;
        while(!game.finished()) {
            if(stop_requested)return;
            SearchRun options;options.clear_before_search=same_bot;options.should_stop=[]{return stop_requested.load();};
            auto& bot=same_bot || ((game.player()==1)==(color==0))?sa:*sb;
            auto result=bot.run(game,move_temperature(c,"match",game),options);
            if(stop_requested)return;
            if(result.action<0)throw std::runtime_error("Match search has no move: zero playout budget");
            game.play(result.action);actions.push_back(result.action);visits.push_back(result.root_visits);
            initial_visits.push_back(result.initial_visits);new_playouts.push_back(result.new_playouts);
            sa.advance(result.action);if(sb)sb->advance(result.action);
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ostringstream row;row<<std::setprecision(17)<<"{\"type\":\"game\",\"id\":"<<2*task.id+color
            <<",\"opening_id\":"<<task.id<<",\"black_a\":"<<(color==0?"true":"false")
            <<",\"winner\":"<<game.winner()<<",\"seconds\":"<<seconds<<",\"root_visits\":";
        array(row,visits);row<<",\"initial_visits\":";array(row,initial_visits);row<<",\"new_playouts\":";array(row,new_playouts);
        auto precision=c.text("match.inference_precision");
        if(precision=="auto")precision=a.get("device").rfind("cuda:",0)==0?"float16":"float32";
        row<<",\"same_bot\":"<<(same_bot?"true":"false")<<",\"inference_precision\":"<<quote(precision)<<",\"moves\":";write_moves(row,actions);row<<'}';
        std::lock_guard<std::mutex> lock(output);std::cout<<row.str()<<std::endl;
        ++completed_games;
    });
    ea->finish();eb->finish();
    auto inference_stats=[](InferenceService& service) {
        std::cout<<"{\"requests\":"<<service.requests.load()<<",\"batches\":"<<service.batches.load()
                 <<",\"max_batch\":"<<service.max_observed_batch.load()
                 <<",\"queue_wait_us\":"<<service.wait_microseconds.load()
                 <<",\"submitted\":"<<service.submitted.load()<<",\"cache_hits\":"<<service.cache_hits.load()<<'}';
    };
    std::cout<<std::setprecision(17)<<"{\"type\":\"match_stats\",\"games\":"<<completed_games.load()
             <<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-match_start).count()
             <<",\"inference_a\":";inference_stats(*ea);
    std::cout<<",\"inference_b\":";inference_stats(*eb);std::cout<<'}'<<std::endl;
    return stop_requested?130:0;
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2)throw std::runtime_error("Expected selfplay, worker, analysis, infer or match");
        std::string mode=argv[1]; Args args(argc,argv); Config config(args.get("config"));
        if(mode=="selfplay" || mode=="worker" || mode=="infer")
            if((config.text("agent.algorithm")!="alphazero" && config.text("agent.algorithm")!="muzero") ||
               (config.text("agent.root_search_algo")!="puct" && config.text("agent.root_search_algo")!="gumbel") ||
               (config.text("agent.nonroot_search_algo")!="puct" && config.text("agent.nonroot_search_algo")!="gumbel") ||
               (config.text("agent.root_search_algo")=="puct" && config.text("agent.nonroot_search_algo")!="puct"))
                throw std::runtime_error("Invalid native algorithm/search combination");
        torch::set_num_threads(config.integer(mode=="analysis"?"analysis.cpu_threads":mode=="match"?"match.cpu_threads":"run.cpu_threads")); torch::set_num_interop_threads(1);
        std::signal(SIGINT,stop_handler); std::signal(SIGTERM,stop_handler);
        if(mode=="selfplay") {ForkPool forks;auto service=evaluator(args,config,"selfplay");int code=selfplay(args,config,*service,forks);service->finish();return code;}
        if(mode=="worker")return worker(args,config);
        if(mode=="analysis") {
            const auto stream=args.get("stream","false");
            if(stream!="true" && stream!="false")throw std::runtime_error("--stream must be true or false");
            return stream=="true"?analysis_session(args,config):analyze(args,config,false);
        }
        if(mode=="infer")return analyze(args,config,true);
        if(mode=="match")return match(args,config);
        throw std::runtime_error("Unknown native command: "+mode);
    } catch(const std::exception& error) {std::cerr<<"EtaZero: "<<error.what()<<'\n';return 1;}
}
