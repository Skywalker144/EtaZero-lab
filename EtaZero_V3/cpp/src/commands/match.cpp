#include "etazero/game_config.h"
#include "etazero/match.h"
#include "etazero/record.h"
#include <algorithm>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>

namespace etazero {
namespace {
uint64_t mix(uint64_t n) {
    n+=0x9e3779b97f4a7c15ULL;n=(n^(n>>30))*0xbf58476d1ce4e5b9ULL;
    n=(n^(n>>27))*0x94d049bb133111ebULL;return n^(n>>31);
}
template<class T> void array(std::ostream& out,const std::vector<T>& values) {
    out<<'[';for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<values[i];}out<<']';
}
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}
}
int run_matches(const Config& c,int size,Rule rule,const std::string& precision,
                const std::vector<MatchModel>& models,std::vector<MatchTask>& tasks,int max_models,
                const MatchEvaluatorFactory& evaluator,const MatchSearchFactory& search,
                const std::function<double(const Game&)>& temperature,const std::function<bool()>& stopped,
                std::ostream& output,bool single_pair,int model_load_workers) {
    const auto started=Clock::now();const int canvas=c.integer("network.canvas");
    const int threads=c.integer("match.game_threads");
    if(threads<1 || max_models<2 || model_load_workers<1)throw std::runtime_error("Invalid arena concurrency/model limit");
    const auto options=rule_options(c,GameConfigMode::MATCH);
    OpeningConfig opening(c);
    if(rule!=Rule::GO && rule!=Rule::XIANGQI && (rule==Rule::HEX ? (opening.hex_probability!=1 || opening.hex_make_fair_probability!=1 || opening.hex_min_accept_rate<=0)
                      : (opening.probability!=1 || opening.rejection_probability_fallback>=1)))
        throw std::runtime_error("Matches require balanced openings and a rejection fallback below one");
    std::set<std::pair<std::string,int>> identities;
    std::map<std::string,std::pair<size_t,size_t>> pairs;
    for(const auto& t:tasks) {
        if(t.a>=models.size() || t.b>=models.size() || t.id<0 || t.generator!=t.id%2 || t.mask<1 || t.mask>3 ||
           !identities.emplace(t.pair,t.id).second)throw std::runtime_error("Invalid arena task identity");
        auto inserted=pairs.emplace(t.pair,std::make_pair(t.a,t.b));
        if(!inserted.second && inserted.first->second!=std::make_pair(t.a,t.b))throw std::runtime_error("Conflicting arena pair");
        Game game(size,canvas,rule,options);for(int move:t.moves)game.play(move);
        if(game.finished() || (!t.generated && !t.moves.empty()))throw std::runtime_error("Invalid arena opening");
    }
    std::mutex output_mutex;
    auto emit=[&](const std::string& row){std::lock_guard<std::mutex> lock(output_mutex);output<<row<<std::endl;};
    auto moves=[&](std::ostream& out,const std::vector<int>& actions) {
        std::vector<int> local;for(int a:actions)local.push_back(game_spec(rule).local_action(a,size,canvas));array(out,local);
    };
    std::map<size_t,std::unique_ptr<InferenceService>> resident;
    std::map<size_t,std::string> pair_stats;
    std::vector<bool> assigned(tasks.size(),false);
    int completed=0,windows=0,loads=0;double load_seconds=0,play_seconds=0;
    auto release=[&](size_t index) {
        auto& service=*resident.at(index);service.finish();
        std::ostringstream row;row<<'{';
        if(!single_pair)row<<"\"type\":\"model_stats\",\"model\":"<<quote(models[index].id)<<',';
        row<<"\"requests\":"<<service.requests.load()<<",\"batches\":"<<service.batches.load()
            <<",\"max_batch\":"<<service.max_observed_batch.load()
            <<",\"submitted\":"<<service.submitted.load()<<",\"cache_hits\":"<<service.cache_hits.load()<<'}';
        if(single_pair)pair_stats[index]=row.str();else emit(row.str());
        resident.erase(index);
    };
    while(!stopped()) {
        // Fill a residency window, favoring pairs sharing already selected models.
        // No pair-completion barrier exists inside a window.
        std::set<size_t> selected;
        std::vector<size_t> pending;
        for(size_t i=0;i<tasks.size();++i)if(!assigned[i])pending.push_back(i);
        if(pending.empty())break;
        while(true) {
            size_t best=tasks.size();int best_score=-1;
            for(size_t i:pending) {
                const auto& t=tasks[i];std::set<size_t> added{t.a,t.b};
                for(size_t m:selected)added.erase(m);
                if(added.empty() || selected.size()+added.size()>static_cast<size_t>(max_models))continue;
                int score=4*int(selected.count(t.a)+selected.count(t.b))+int(resident.count(t.a)+resident.count(t.b));
                if(score>best_score){best=i;best_score=score;}
            }
            if(best==tasks.size())break;
            selected.insert(tasks[best].a);selected.insert(tasks[best].b);
        }
        for(auto it=resident.begin();it!=resident.end();) {
            auto index=it++->first;if(!selected.count(index))release(index);
        }
        const auto loading=Clock::now();
        std::vector<size_t> missing;
        for(size_t index:selected)if(!resident.count(index))missing.push_back(index);
        const auto load_workers=std::min<size_t>(missing.size(),single_pair?1:model_load_workers);
        auto progress=[&](const char* stage,size_t loaded) {
            if(single_pair)return;
            std::ostringstream row;row<<"{\"type\":\"arena_progress\",\"stage\":"<<quote(stage)
                <<",\"window\":"<<windows+1<<",\"loaded\":"<<loaded<<",\"models\":"<<missing.size()
                <<",\"workers\":"<<load_workers<<",\"seconds\":"<<elapsed(loading)<<'}';
            emit(row.str());
        };
        progress("loading",0);
        std::vector<std::unique_ptr<InferenceService>> loaded(missing.size());
        std::atomic<size_t> next{0};std::atomic<bool> load_failed{false};
        std::mutex load_mutex;std::exception_ptr load_error;size_t ready_models=0;
        auto load=[&] {
            try {
                while(!stopped() && !load_failed) {
                    size_t slot=next.fetch_add(1);if(slot>=missing.size())return;
                    loaded[slot]=evaluator(models[missing[slot]],emit);
                    std::lock_guard<std::mutex> lock(load_mutex);progress("model_ready",++ready_models);
                }
            } catch(...) {
                std::lock_guard<std::mutex> lock(load_mutex);load_failed=true;
                if(!load_error)load_error=std::current_exception();
            }
        };
        if(load_workers<=1)load();
        else {
            std::vector<std::thread> loaders;
            try {for(size_t i=0;i<load_workers;++i)loaders.emplace_back(load);}
            catch(...) {load_failed=true;for(auto& worker:loaders)worker.join();throw;}
            for(auto& worker:loaders)worker.join();
        }
        if(load_error)std::rethrow_exception(load_error);
        for(size_t slot=0;slot<missing.size();++slot)if(loaded[slot]) {
            resident.emplace(missing[slot],std::move(loaded[slot]));++loads;
        }
        load_seconds+=elapsed(loading);if(stopped())break;
        progress("playing",ready_models);
        ++windows;
        struct Job {size_t task;int color;}; // color=-1 generates an opening.
        std::deque<Job> ready;
        for(size_t i:pending) {
            const auto& t=tasks[i];if(!selected.count(t.a)||!selected.count(t.b))continue;
            assigned[i]=true;
            if(!t.generated)ready.push_back({i,-1});
            else if(!single_pair)for(int color=0;color<2;++color)if(t.mask&(1<<color))ready.push_back({i,color});
        }
        std::mutex mutex;std::condition_variable changed;
        size_t active=0;std::atomic<bool> failed{false};std::exception_ptr error;
        auto stop=[&]{return stopped()||failed.load();};
        auto generate=[&](MatchTask& t,InferenceService& ea,InferenceService& eb) {
            std::mt19937_64 random(t.seed);
            for(int retry=0;retry<100;++retry) {
                if(stop())return;
                Game game(size,canvas,rule,options);
                auto result=initialize_opening(game,opening,t.generator==0?ea:eb,t.generator==0?eb:ea,random,stop);
                if(result.status==OpeningStatus::Interrupted)return;
                if(rule!=Rule::GO && rule!=Rule::XIANGQI && result.status!=OpeningStatus::Success)throw std::runtime_error("Balanced match opening failed: "+result.failure);
                if(game.finished())continue;
                t.moves=result.actions;t.generated=true;
                std::ostringstream row;row<<std::setprecision(17)<<"{\"type\":\"opening\"";
                if(!single_pair)row<<",\"pair\":"<<quote(t.pair);
                row<<",\"id\":"<<t.id<<",\"generator\":"<<t.generator<<",\"seed\":"<<t.seed
                    <<",\"attempts\":"<<result.attempts<<",\"balanced_moves\":"<<result.balanced_moves
                    <<",\"policy_moves\":"<<result.policy_moves<<",\"value\":"<<result.start_value<<",\"moves\":";
                moves(row,t.moves);row<<",\"reference_black_a\":"<<(t.generator==0?"true":"false")
                    <<",\"balance_evaluators\":";array(row,result.balance_evaluators);
                row<<",\"policy_evaluators\":";array(row,result.policy_evaluators);row<<'}';
                // Publish the opening before either game can publish its result.
                emit(row.str());return;
            }
            throw std::runtime_error("100 consecutive terminal openings");
        };
        auto play=[&](const MatchTask& t,int color,InferenceService& ea,InferenceService& eb) {
            Game game(size,canvas,rule,options);for(int a:t.moves)game.play(a);
            game.set_pda(c.number("match.playout_doubling_advantage"),c.text("match.playout_doubling_advantage_player")=="black"?1:-1);
            auto start=Clock::now();uint64_t seed=t.seed^(0xd1b54a32d192ed03ULL*(color+1));
            bool same=models[t.a].id==models[t.b].id;auto sa=search(ea,mix(seed));
            auto sb=same?std::unique_ptr<GameSearch>{}:search(eb,mix(seed^0x9e3779b97f4a7c15ULL));
            std::vector<int> actions=t.moves;std::vector<int64_t> visits,initial,new_playouts;
            while(!game.finished()) {
                if(stop())return;
                SearchRun options;options.clear_before_search=same;options.should_stop=stop;
                auto& bot=same||((game.player()==1)==(color==0))?sa:sb;
                auto result=bot->run(game,temperature(game),options);if(stop())return;
                if(result.action<0)throw std::runtime_error("Match search has no move: zero playout budget");
                game.play(result.action);actions.push_back(result.action);visits.push_back(result.root_visits);
                initial.push_back(result.initial_visits);new_playouts.push_back(result.new_playouts);
                sa->advance(result.action);if(sb)sb->advance(result.action);
            }
            std::ostringstream row;row<<std::setprecision(17)<<"{\"type\":\"game\"";
            if(!single_pair)row<<",\"pair\":"<<quote(t.pair);
            row<<",\"id\":"<<2*t.id+color<<",\"opening_id\":"<<t.id<<",\"black_a\":"<<(color==0?"true":"false")
                <<",\"reason\":"<<int(game.reason())<<",\"winner\":"<<game.winner()<<",\"seconds\":"<<elapsed(start)<<",\"root_visits\":";
            array(row,visits);row<<",\"initial_visits\":";array(row,initial);row<<",\"new_playouts\":";array(row,new_playouts);
            row<<",\"same_bot\":"<<(same?"true":"false")<<",\"inference_precision\":"<<quote(precision)<<",\"moves\":";
            moves(row,actions);row<<'}';emit(row.str());
            std::lock_guard<std::mutex> lock(mutex);++completed;
        };
        auto loop=[&] {
            try {
                while(true) {
                    Job job;
                    {std::unique_lock<std::mutex> lock(mutex);
                        changed.wait(lock,[&]{return stop()||!ready.empty()||active==0;});
                        if(stop()||ready.empty())return;
                        job=ready.front();ready.pop_front();++active;
                    }
                    auto& t=tasks[job.task];auto& ea=*resident.at(t.a);auto& eb=*resident.at(t.b);
                    if(job.color<0)generate(t,ea,eb);else play(t,job.color,ea,eb);
                    {std::lock_guard<std::mutex> lock(mutex);
                        if(!single_pair && job.color<0 && t.generated && !stop())
                            for(int color=1;color>=0;--color)if(t.mask&(1<<color))ready.push_front({job.task,color});
                        --active;
                    }
                    changed.notify_all();
                }
            } catch(...) {
                {std::lock_guard<std::mutex> lock(mutex);failed=true;if(!error)error=std::current_exception();}
                changed.notify_all();
            }
        };
        auto execute_jobs=[&] {
            auto playing=Clock::now();std::vector<std::thread> workers;
            const auto worker_count=std::min<size_t>(threads,(single_pair?1:2)*ready.size());
            try {for(size_t i=0;i<worker_count;++i)workers.emplace_back(loop);}
            catch(...) {{std::lock_guard<std::mutex> lock(mutex);failed=true;}changed.notify_all();for(auto& w:workers)w.join();throw;}
            for(auto& w:workers)w.join();play_seconds+=elapsed(playing);
            if(error)std::rethrow_exception(error);
        };
        execute_jobs();
        if(single_pair && !stop()) {
            // Preserve match's barrier and game order, including resumed openings.
            for(size_t i:pending)if(assigned[i] && tasks[i].generated)
                for(int color=0;color<2;++color)if(tasks[i].mask&(1<<color))ready.push_back({i,color});
            execute_jobs();
        }
        for(auto& item:resident)item.second->drain();
    }
    while(!resident.empty())release(resident.begin()->first);
    std::ostringstream row;row<<std::setprecision(17);
    if(single_pair)row<<"{\"type\":\"match_stats\",\"games\":"<<completed
        <<",\"seconds\":"<<elapsed(started)<<",\"inference_a\":"<<(pair_stats.count(0)?pair_stats.at(0):"{}")
        <<",\"inference_b\":"<<(pair_stats.count(1)?pair_stats.at(1):"{}")<<'}';
    else row<<"{\"type\":\"arena_stats\",\"games\":"<<completed
        <<",\"seconds\":"<<elapsed(started)<<",\"load_seconds\":"<<load_seconds<<",\"play_seconds\":"<<play_seconds
        <<",\"model_loads\":"<<loads<<",\"model_windows\":"<<windows<<'}';
    emit(row.str());
    return stopped()?130:0;
}
}
