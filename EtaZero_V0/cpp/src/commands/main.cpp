#include "etazero/config.h"
#include "etazero/record.h"
#include "etazero/torch_backend.h"
#include "etazero/random_evaluator.h"
#include "etazero/search_limits.h"
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
using namespace etazero;
namespace {
static_assert(std::atomic<bool>::is_always_lock_free, "Signal stop flag must be lock-free");
std::atomic<bool> stop_requested{false};
void stop_handler(int) { stop_requested.store(true,std::memory_order_relaxed); }
struct Args {
    std::map<std::string, std::string> args;
    Args(int argc, char** argv) {
        const std::set<std::string> allowed{"config","model","model-id","device","games","output","run-id","attempt-id",
                                          "config-id","source-id","iteration","worker","seed","size","rule","moves","model-b","model-b-id","evaluator","tasks"};
        for (int i = 2; i < argc; i += 2) {
            std::string name = argv[i];
            if (name.rfind("--",0) != 0 || !allowed.count(name.substr(2)) || i+1 >= argc || !args.emplace(name.substr(2), argv[i+1]).second)
                throw std::runtime_error("Invalid native command argument: " + name);
        }
    }
    std::string get(const std::string& key) const { return args.at(key); }
    std::string get(const std::string& key, std::string value) const { auto it = args.find(key); return it == args.end() ? value : it->second; }
    int integer(const std::string& key) const { return std::stoi(get(key)); }
};
uint64_t mix(uint64_t n) {
    n += 0x9e3779b97f4a7c15ULL; n = (n^(n>>30))*0xbf58476d1ce4e5b9ULL;
    n = (n^(n>>27))*0x94d049bb133111ebULL; return n^(n>>31);
}
SearchSettings settings(const Config& c, const std::string& mode) {
    bool selfplay = mode == "selfplay";
    std::string section = selfplay ? "search" : mode == "match" ? "match" : "evaluation";
    auto key = [&](const std::string& group, const std::string& name) {
        return (selfplay ? group : section) + "." + name;
    };
    SearchSettings s{selfplay?c.integer("search.full_search_visits")-1:c.integer(section+".visits")-1,
                     c.integer(key("parallelism","search_threads")),
                     c.number(key("puct","c_puct")),c.number(key("puct","virtual_loss")),
                     selfplay?c.number("dirichlet_noise.noise_fraction"):0,
                     selfplay?c.number("dirichlet_noise.dirichlet_total_concentration"):1,c.boolean(key("search","reuse_tree")),
                     selfplay?0:c.integer(section+".visits")};
    s.use_fpu=c.boolean(key("fpu","use_fpu"));s.fpu_reduction_max=c.number(key("fpu","fpu_reduction_max"));
    s.root_fpu_reduction_max=c.number(key("fpu","root_fpu_reduction_max"));
    s.fpu_parent_power=c.number(key("fpu","fpu_parent_weight_by_visited_policy_pow"));
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
    std::string section=selfplay?"temperature":mode=="match"?"match":"evaluation";
    return temperature_at_turn(c.number(section+(selfplay?".temperature":".temperature_early")),
        c.number(section+(selfplay?".final_temperature":".temperature")),
        c.number(section+".temperature_halflife"),game.turn(),game.size()*game.size());
}
template<class T> void array(std::ostream& out, const std::vector<T>& values) {
    out << '['; bool first = true;
    for (auto x : values) { if (!first) out << ','; out << x; first = false; } out << ']';
}
std::unique_ptr<BatchEvaluator> evaluator(const Args& a, const Config& c, const std::string& mode, bool second = false) {
    std::string prefix = (mode == "infer" || mode == "selfplay") ? "inference" : mode == "evaluate" ? "evaluation" : mode;
    int batch = c.integer(prefix+".max_batch"), canvas = c.integer("network.canvas");
    std::vector<std::unique_ptr<Backend>> backends;
    std::string kind=a.get("evaluator","network");
    if(kind!="random" && kind!="network")throw std::runtime_error("Unknown evaluator: "+kind);
    if(kind=="random" && mode!="selfplay")throw std::runtime_error("Random evaluator is for cold-start selfplay only");
    auto loaded_model=std::make_shared<TorchBackend::LoadedModel>();
    for (int i=0;i<c.integer(prefix+".server_threads");++i) {
        if(kind=="random")backends.push_back(std::make_unique<RandomBackend>(canvas,std::stoull(a.get("seed"))));
        else backends.push_back(std::make_unique<TorchBackend>(a.get(second ? "model-b" : "model"),a.get("device"),canvas,batch,
                                                         c.text(prefix+".inference_precision"),loaded_model));
    }
    return std::make_unique<BatchEvaluator>(std::move(backends), a.get(second ? "model-b-id" : "model-id"), canvas,
                                          batch, c.integer(prefix+".queue_capacity"), c.integer(prefix+".batch_wait_us"),
                                          c.integer(prefix+".cache_entries"));
}
void moves(Game& g, const std::string& text) {
    if (text.empty()) return;
    std::istringstream in(text); std::string item;
    while (std::getline(in,item,',')) g.play(std::stoi(item));
}
void stats(BatchEvaluator& e) {
    std::cout << "{\"event\":\"inference\",\"requests\":" << e.requests << ",\"batches\":" << e.batches
              << ",\"max_batch\":" << e.max_observed_batch << ",\"queue_wait_us\":" << e.wait_microseconds
              << ",\"submitted\":" << e.submitted << ",\"cache_hits\":" << e.cache_hits << ",\"rows_by_server\":";
    array(std::cout,e.rows_by_server);std::cout<<"}"<<std::endl;
}
int selfplay(const Args& a,const Config& c,BatchEvaluator& service) {
    int count = a.integer("games"), canvas = c.integer("network.canvas");
    if (count < 1) throw std::runtime_error("Selfplay requires positive game count");
    auto* eval=&service;eval->reset_stats();
    OpeningConfig opening(c,"policy_init"); auto search_config=selfplay_search_config(c);
    bool random=a.get("evaluator","network")=="random";
    Source source{a.get("run-id"),a.get("attempt-id"),a.get("model-id"),a.get("config-id"),a.get("source-id"),a.integer("iteration"),a.integer("worker")};
    RecordWriter writer(a.get("output"),source,c.integer("writer.shard_rows"),c.integer("writer.writer_queue"),c.number("writer.flush_seconds"));
    std::vector<int> sizes; std::vector<Rule> rules; std::vector<double> sw, rw;
    for (auto x : c.list("environment.sizes")) sizes.push_back(std::stoi(x));
    for (auto x : c.list("environment.rules")) rules.push_back(parse_rule(x));
    for (auto x : c.list("environment.size_weights")) sw.push_back(std::stod(x));
    for (auto x : c.list("environment.rule_weights")) rw.push_back(std::stod(x));
    uint64_t seed = std::stoull(a.get("seed"));
    std::atomic<int> next{0}, finished{0}; std::atomic<bool> failure{false};
    std::mutex error_mutex; std::exception_ptr error;
    auto loop = [&] {
        try {
            Search search(*eval,settings(c,"selfplay"),0);
            while (!stop_requested && !failure) {
                int id = next.fetch_add(1); if (id >= count) break;
                uint64_t game_seed = mix(seed+id); std::mt19937_64 rng(game_seed);
                std::mt19937_64 feature_rng(game_seed ^ 0xD1B54A32D192ED03ULL);
                std::bernoulli_distribution drop_feature(c.number("environment.forbidden_feature_dropout_prob"));
                int size = sizes[std::discrete_distribution<size_t>(sw.begin(),sw.end())(rng)];
                Rule rule = rules[std::discrete_distribution<size_t>(rw.begin(),rw.end())(rng)];
                Game game(size,canvas,rule);search.reset(rng());
                FinishedGame record{}; record.id=id; record.seed=game_seed; record.size=size; record.canvas=canvas; record.rule=rule;
                std::vector<double> historical_values;
                if(!random) {
                    record.opening=initialize_opening(game,opening,*eval,rng,[&]{return stop_requested || failure;});
                    if(record.opening.status==OpeningStatus::Interrupted)break;
                    Game prefix(size,canvas,rule);
                    for(int action:record.opening.actions) {
                        Step step{prefix.player(),action,0,0,0,prefix.observation(),
                                  std::vector<double>(canvas*canvas,0),std::vector<int64_t>(canvas*canvas,0),false};
                        prefix.play(action);if(prefix.finished())step.reward=prefix.winner()*step.player;
                        record.steps.push_back(std::move(step));
                    }
                }
                while (!game.finished() && !stop_requested && !failure) {
                    double temp = move_temperature(c,"selfplay",game);
                    bool cheap=std::bernoulli_distribution(search_config.cheap_probability)(rng);
                    auto limits=selfplay_search_limits(search_config,historical_values,cheap);
                    auto result = search.run(game,temp,limits.search);
                    Step step{game.player(),result.action,result.simulations,static_cast<float>(temp),0,
                              game.observation(game.player(), !(rule == Rule::RENJU && drop_feature(feature_rng))),std::move(result.policy),std::move(result.visits)};
                    step.cheap_search=limits.cheap_search;step.target_weight=limits.target_weight;
                    step.policy_surprise=result.policy_surprise;step.network_wdl=result.network_wdl;step.search_wdl=result.search_wdl;
                    if(search_config.reduce_visits)historical_values.push_back(game.player()*result.value);
                    game.play(result.action); if (game.finished()) step.reward = game.winner() * step.player;
                    record.steps.push_back(std::move(step)); search.advance(result.action);
                }
                if (!game.finished()) break; // Interrupted trajectories never receive fabricated targets.
                record.winner=game.winner(); record.reason=game.reason(); record.final_player=game.player(); record.final_observation=game.observation();
                apply_training_weights(record,c.number("surprise_weighting.policy_surprise_data_weight"),c.number("surprise_weighting.value_surprise_data_weight"),rng);
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
    std::unique_ptr<BatchEvaluator> cached;
    std::string model,path,operation;
    while(!stop_requested && std::getline(std::cin,operation)) {
        if(operation=="release") {
            bool allocated=static_cast<bool>(cached);cached.reset();model.clear();path.clear();
            if(allocated && base.get("device").rfind("cuda:",0)==0)c10::cuda::CUDACachingAllocator::emptyCache();
            std::cout<<"{\"event\":\"worker_released\"}"<<std::endl;continue;
        }
        if(operation!="selfplay")throw std::runtime_error("Unknown worker operation");
        Args request=base;
        for(const char* key:{"model","model-id","games","output","attempt-id","iteration","seed","evaluator"})request.args[key]=read_field();
        if(!cached || model!=request.get("model-id") || request.get("evaluator")=="random") {
            cached.reset();cached=evaluator(request,config,"selfplay");model=request.get("model-id");path=request.get("model");
        } else if(path!=request.get("model"))throw std::runtime_error("Worker model identity collision");
        int code=selfplay(request,config,*cached);
        std::cout<<"{\"event\":\"worker_complete\",\"attempt_id\":"<<quote(request.get("attempt-id"))<<",\"code\":"<<code<<"}"<<std::endl;
        if(code)return code;
    }
    return stop_requested?2:0;
}
int evaluate(const Args& a, const Config& c, bool raw) {
    Game game(a.integer("size"),c.integer("network.canvas"),parse_rule(a.get("rule")));
    moves(game,a.get("moves",""));
    if (game.finished() && !raw) {
        std::cout << "{\"terminal\":true,\"winner\":" << game.winner() << ",\"value\":" << game.terminal_value() << ",\"reason\":" << game.reason() << "}\n";
        return 0;
    }
    auto eval=evaluator(a,c,raw?"infer":"evaluate"); auto network=eval->evaluate(game.observation());
    std::cout.precision(12);
    std::cout << "{\"terminal\":" << (game.finished()?"true":"false") << ",\"raw_value\":" << network.value() << ",\"raw_logits\":";
    array(std::cout,network.logits);std::cout << ",\"raw_wdl\":";array(std::cout,std::vector<double>(network.wdl.begin(),network.wdl.end()));
    if (!raw) {
        Search search(*eval,settings(c,"evaluate"),std::stoull(a.get("seed","0"))); auto result=search.run(game,move_temperature(c,"evaluate",game));
        std::cout << ",\"action\":" << result.action << ",\"value\":" << result.value << ",\"simulations\":" << result.simulations << ",\"root_visits\":" << result.root_visits << ",\"policy\":";
        array(std::cout,result.policy); std::cout << ",\"wdl\":";
        array(std::cout,std::vector<double>(result.search_wdl.begin(),result.search_wdl.end()));
        std::cout << ",\"network_wdl\":";array(std::cout,std::vector<double>(result.network_wdl.begin(),result.network_wdl.end()));
        std::cout << ",\"network_policy\":";array(std::cout,result.network_policy);
        std::cout << ",\"search_policy\":";array(std::cout,result.search_policy);
        std::cout << ",\"visits\":"; array(std::cout,result.visits);
    }
    eval->finish();
    std::cout << ",\"nn_requests\":" << eval->requests << ",\"cache_hits\":" << eval->cache_hits << "}\n"; return 0;
}
// Interactive actions use the actual board width, unlike search's canvas indices.
int web_integer(const std::string& text) {
    int value;
    auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size())
        throw std::runtime_error("Expected integer argument");
    return value;
}
void web_state(const Game& game,const std::vector<int>& moves) {
    std::cout<<"{\"board_size\":"<<game.size()<<",\"canvas_size\":"<<game.canvas()
             <<",\"player\":"<<game.player()<<",\"turn\":"<<game.turn()
             <<",\"finished\":"<<(game.finished()?"true":"false")<<",\"winner\":"<<game.winner()
             <<",\"reason\":"<<game.reason()<<",\"board\":";
    array(std::cout,game.board().cells);std::cout<<",\"moves\":";array(std::cout,moves);std::cout<<'}';
}
void web_analysis(const SearchResult& result,const Game& game,double seconds,uint64_t requests,uint64_t batches) {
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
    std::cout<<"]}";
}
int serve(const Args& a,const Config& c) {
    auto eval=evaluator(a,c,"evaluate");
    Search search(*eval,settings(c,"evaluate"),std::stoull(a.get("seed","0")));
    std::optional<Game> game;
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
            if(words[0]=="new" && words.size()==3) {
                Game replacement(web_integer(words[1]),canvas,parse_rule(words[2]));
                game=std::move(replacement);played.clear();search.reset(std::stoull(a.get("seed","0")));
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
            } else if((words[0]=="genmove" || words[0]=="analyze") && words.size()==2) {
                int visits=web_integer(words[1]);
                if(visits<2 || visits>100000)throw std::runtime_error("Visits must be in [2, 100000]");
                Game before=*game;
                SearchRun options;options.max_visits=visits;options.clear_before_search=true;
                auto requests=eval->requests.load(),batches=eval->batches.load();
                auto start=std::chrono::steady_clock::now();
                auto result=search.run(before,move_temperature(c,"evaluate",before),options);
                double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                if(words[0]=="genmove") {
                    game->play(result.action);played.push_back(result.action/canvas*game->size()+result.action%canvas);
                }
                std::cout<<"{\"ok\":true,\"state\":";web_state(*game,played);std::cout<<",\"analysis\":";
                web_analysis(result,before,seconds,eval->requests.load()-requests,eval->batches.load()-batches);
                std::cout<<"}\n"<<std::flush;continue;
            } else if(!(words[0]=="state" && words.size()==1))throw std::runtime_error("Unknown command or arguments");
            std::cout<<"{\"ok\":true,\"state\":";web_state(*game,played);std::cout<<"}\n"<<std::flush;
        } catch(const std::exception& error) {
            std::cout<<"{\"ok\":false,\"error\":"<<quote(error.what())<<"}\n"<<std::flush;
        }
    }
    eval->finish();return 0;
}
int match(const Args& a,const Config& c) {
    auto ea=evaluator(a,c,"match"),eb=evaluator(a,c,"match",true);
    const int size=a.integer("size"),canvas=c.integer("network.canvas");
    OpeningConfig opening(c);
    if(opening.probability!=1 || opening.rejection_probability_fallback>=1)
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
            auto result=initialize_opening(game,opening,task.generator==0?*ea:*eb,random,[]{return stop_requested.load();});
            if(result.status==OpeningStatus::Interrupted)return;
            if(result.status!=OpeningStatus::Success)throw std::runtime_error("Balanced match opening failed: "+result.failure);
            if(game.finished())continue;
            task.moves=result.actions;task.generated=true;
            std::ostringstream row;row<<std::setprecision(17)<<"{\"type\":\"opening\",\"id\":"<<task.id
                <<",\"generator\":"<<task.generator<<",\"seed\":"<<task.seed<<",\"attempts\":"<<result.attempts
                <<",\"value\":"<<result.start_value<<",\"moves\":";
            write_moves(row,task.moves);row<<'}';
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
        auto start=std::chrono::steady_clock::now();uint64_t seed=task.seed^(0xd1b54a32d192ed03ULL*(color+1));
        Search sa(*ea,settings(c,"match"),seed),sb(*eb,settings(c,"match"),seed);
        std::vector<int> actions=task.moves;std::vector<int64_t> visits;
        while(!game.finished()) {
            if(stop_requested)return;
            auto result=((game.player()==1)==(color==0)?sa:sb).run(game,move_temperature(c,"match",game));
            game.play(result.action);actions.push_back(result.action);visits.push_back(result.root_visits);
            sa.advance(result.action);sb.advance(result.action);
        }
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ostringstream row;row<<std::setprecision(17)<<"{\"type\":\"game\",\"id\":"<<2*task.id+color
            <<",\"opening_id\":"<<task.id<<",\"black_a\":"<<(color==0?"true":"false")
            <<",\"winner\":"<<game.winner()<<",\"seconds\":"<<seconds<<",\"root_visits\":";
        array(row,visits);row<<",\"moves\":";write_moves(row,actions);row<<'}';
        std::lock_guard<std::mutex> lock(output);std::cout<<row.str()<<std::endl;
    });
    ea->finish();eb->finish();return stop_requested?130:0;
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2)throw std::runtime_error("Expected selfplay, worker, evaluate, infer, match or serve");
        std::string mode=argv[1]; Args args(argc,argv); Config config(args.get("config"));
        if(mode=="selfplay" || mode=="worker" || mode=="infer")
            if(config.text("agent.algorithm")!="alphazero" || config.text("agent.root_search_algo")!="puct" || config.text("agent.nonroot_search_algo")!="puct")
                throw std::runtime_error("Native executable only supports AlphaZero/PUCT/PUCT");
        torch::set_num_threads(config.integer((mode=="evaluate" || mode=="serve")?"evaluation.cpu_threads":mode=="match"?"match.cpu_threads":"run.cpu_threads")); torch::set_num_interop_threads(1);
        std::signal(SIGINT,stop_handler); std::signal(SIGTERM,stop_handler);
        if(mode=="selfplay") {auto service=evaluator(args,config,"selfplay");int code=selfplay(args,config,*service);service->finish();return code;}
        if(mode=="worker")return worker(args,config);
        if(mode=="evaluate"||mode=="infer")return evaluate(args,config,mode=="infer");
        if(mode=="serve")return serve(args,config);
        if(mode=="match")return match(args,config);
        throw std::runtime_error("Unknown native command: "+mode);
    } catch(const std::exception& error) {std::cerr<<"EtaZero: "<<error.what()<<'\n';return 1;}
}
