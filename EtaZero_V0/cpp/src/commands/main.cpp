#include "etazero/config.h"
#include "etazero/record.h"
#include "etazero/torch_backend.h"
#include <csignal>
#include <iostream>
#include <map>
#include <set>
#include <torch/torch.h>
using namespace etazero;
namespace {
static_assert(std::atomic<bool>::is_always_lock_free, "Signal stop flag must be lock-free");
std::atomic<bool> stop_requested{false};
void stop_handler(int) { stop_requested.store(true,std::memory_order_relaxed); }
struct Args {
    std::map<std::string, std::string> args;
    Args(int argc, char** argv) {
        const std::set<std::string> allowed{"config","model","model-id","device","games","output","run-id","attempt-id",
                                          "config-id","source-id","cycle","worker","seed","size","rule","moves","model-b","model-b-id"};
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
    return {c.integer(section+".simulations"), c.integer(selfplay ? "selfplay.search_threads" : section+".search_threads"),
            c.number("search.c_puct"), c.number("search.virtual_loss"),
            selfplay ? c.number("exploration.noise_fraction") : 0, c.number("exploration.dirichlet_alpha"), c.boolean("search.reuse_tree")};
}
template<class T> void array(std::ostream& out, const std::vector<T>& values) {
    out << '['; bool first = true;
    for (auto x : values) { if (!first) out << ','; out << x; first = false; } out << ']';
}
std::unique_ptr<BatchEvaluator> evaluator(const Args& a, const Config& c, const std::string& mode, bool second = false) {
    std::string prefix = mode == "infer" || mode == "evaluate" ? "evaluation" : mode;
    int batch = c.integer(prefix+".max_batch"), canvas = c.integer("network.canvas");
    auto backend = std::make_unique<TorchBackend>(a.get(second ? "model-b" : "model"), a.get("device"), canvas, batch);
    return std::make_unique<BatchEvaluator>(std::move(backend), a.get(second ? "model-b-id" : "model-id"), canvas,
                                          batch, c.integer("selfplay.queue_capacity"), c.integer("selfplay.batch_wait_us"));
}
void moves(Game& g, const std::string& text) {
    if (text.empty()) return;
    std::istringstream in(text); std::string item;
    while (std::getline(in,item,',')) g.play(std::stoi(item));
}
void stats(BatchEvaluator& e) {
    std::cout << "{\"event\":\"inference\",\"requests\":" << e.requests << ",\"batches\":" << e.batches
              << ",\"max_batch\":" << e.max_observed_batch << ",\"queue_wait_us\":" << e.wait_microseconds << "}" << std::endl;
}
int selfplay(const Args& a, const Config& c) {
    int count = a.integer("games"), canvas = c.integer("network.canvas");
    if (count < 1) throw std::runtime_error("Selfplay requires positive game count");
    auto eval = evaluator(a,c,"selfplay");
    Source source{a.get("run-id"),a.get("attempt-id"),a.get("model-id"),a.get("config-id"),a.get("source-id"),a.integer("cycle"),a.integer("worker")};
    RecordWriter writer(a.get("output"),source,c.integer("selfplay.shard_rows"),c.integer("selfplay.writer_queue"),c.number("selfplay.flush_seconds"));
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
            while (!stop_requested && !failure) {
                int id = next.fetch_add(1); if (id >= count) break;
                uint64_t game_seed = mix(seed+id); std::mt19937_64 rng(game_seed);
                int size = sizes[std::discrete_distribution<size_t>(sw.begin(),sw.end())(rng)];
                Rule rule = rules[std::discrete_distribution<size_t>(rw.begin(),rw.end())(rng)];
                Game game(size,canvas,rule); Search search(*eval,settings(c,"selfplay"),rng());
                FinishedGame record{}; record.id=id; record.seed=game_seed; record.size=size; record.canvas=canvas; record.rule=rule;
                while (!game.finished() && !stop_requested && !failure) {
                    double temp = c.number(game.turn() < c.integer("exploration.temperature_moves") ? "exploration.temperature" : "exploration.final_temperature");
                    auto result = search.run(game,temp);
                    Step step{game.player(),result.action,result.simulations,static_cast<float>(temp),0,
                              game.observation(),std::move(result.policy),std::move(result.visits)};
                    game.play(result.action); if (game.finished()) step.reward = game.winner() * step.player;
                    record.steps.push_back(std::move(step)); search.advance(result.action);
                }
                if (!game.finished()) break; // Interrupted trajectories never receive fabricated targets.
                record.winner=game.winner(); record.reason=game.reason(); record.final_player=game.player(); record.final_observation=game.observation();
                writer.enqueue(std::move(record)); finished.fetch_add(1);
            }
        } catch (...) { std::lock_guard<std::mutex> lock(error_mutex); if (!error) error=std::current_exception(); failure=true; }
    };
    std::vector<std::thread> threads;
    try {
        for (int i=0;i<std::min(count,c.integer("selfplay.game_threads"));++i) threads.emplace_back(loop);
    } catch (...) { failure=true;for(auto& t:threads)t.join();throw; }
    for (auto& t:threads) t.join();
    writer.finish(); eval->finish(); if (error) std::rethrow_exception(error); stats(*eval);
    std::cout << "{\"event\":\"selfplay_complete\",\"games\":" << finished << "}" << std::endl;
    return stop_requested ? 2 : 0;
}
int evaluate(const Args& a, const Config& c, bool raw) {
    Game game(a.integer("size"),c.integer("network.canvas"),parse_rule(a.get("rule")));
    moves(game,a.get("moves",""));
    if (game.finished() && !raw) {
        std::cout << "{\"terminal\":true,\"winner\":" << game.winner() << ",\"value\":" << game.terminal_value() << ",\"reason\":" << game.reason() << "}\n";
        return 0;
    }
    auto eval=evaluator(a,c,"evaluate"); auto network=eval->evaluate(game.observation());
    std::cout.precision(12);
    std::cout << "{\"terminal\":" << (game.finished()?"true":"false") << ",\"raw_value\":" << network.value << ",\"raw_logits\":";
    array(std::cout,network.logits);
    if (!raw) {
        Search search(*eval,settings(c,"evaluate"),std::stoull(a.get("seed","0"))); auto result=search.run(game,0);
        std::cout << ",\"action\":" << result.action << ",\"value\":" << result.value << ",\"simulations\":" << result.simulations << ",\"policy\":";
        array(std::cout,result.policy); std::cout << ",\"visits\":"; array(std::cout,result.visits);
    }
    std::cout << "}\n"; eval->finish(); return 0;
}
int match(const Args& a,const Config& c) {
    auto ea=evaluator(a,c,"match"),eb=evaluator(a,c,"match",true);
    const int games=a.integer("games"),size=a.integer("size"),canvas=c.integer("network.canvas");
    if (games<2 || games%2) throw std::runtime_error("Match game count must be positive and even for balanced colors");
    std::vector<int> outcomes(games,9),winners(games),plies(games); std::atomic<int> next{0}; std::atomic<bool> failure{false};
    std::mutex mutex; std::exception_ptr error;
    auto loop=[&] {
        try {
            while (!stop_requested && !failure) {
                int i=next.fetch_add(1); if (i>=games) return;
                Game game(size,canvas,parse_rule(a.get("rule"))); moves(game,a.get("moves",""));
                if (game.finished()) throw std::runtime_error("Match opening must be nonterminal");
                uint64_t seed=mix(std::stoull(a.get("seed"))+i/2); int color_a=i%2? -1:1;
                Search sa(*ea,settings(c,"match"),seed),sb(*eb,settings(c,"match"),seed);
                while (!game.finished() && !stop_requested && !failure) {
                    double temp=game.turn()<c.integer("match.temperature_moves")?c.number("match.temperature"):0;
                    auto result=(game.player()==color_a?sa:sb).run(game,temp); game.play(result.action);
                    sa.advance(result.action); sb.advance(result.action);
                }
                if (game.finished()) { winners[i]=game.winner(); outcomes[i]=game.winner()*color_a; plies[i]=game.turn(); }
            }
        } catch (...) { std::lock_guard<std::mutex> lock(mutex); if (!error) error=std::current_exception(); failure=true; }
    };
    std::vector<std::thread> workers;
    try {
        for (int i=0;i<std::min(games,c.integer("match.game_threads"));++i) workers.emplace_back(loop);
    } catch (...) { failure=true;for(auto& t:workers)t.join();throw; }
    for (auto& t:workers)t.join();
    ea->finish();eb->finish();if(error)std::rethrow_exception(error);
    std::cout << "{\"outcomes_a\":";array(std::cout,outcomes);std::cout<<",\"winners\":";array(std::cout,winners);
    std::cout<<",\"plies\":";array(std::cout,plies);std::cout<<",\"complete\":"<<(stop_requested?"false":"true")<<"}\n";
    return stop_requested?2:0;
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2)throw std::runtime_error("Expected selfplay, evaluate, infer or match");
        std::string mode=argv[1]; Args args(argc,argv); Config config(args.get("config"));
        torch::set_num_threads(config.integer("run.cpu_threads")); torch::set_num_interop_threads(1);
        std::signal(SIGINT,stop_handler); std::signal(SIGTERM,stop_handler);
        if(mode=="selfplay")return selfplay(args,config);
        if(mode=="evaluate"||mode=="infer")return evaluate(args,config,mode=="infer");
        if(mode=="match")return match(args,config);
        throw std::runtime_error("Unknown native command: "+mode);
    } catch(const std::exception& error) {std::cerr<<"EtaZero: "<<error.what()<<'\n';return 1;}
}
