#include "program/match.h"
#include "etazero/schema.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace etazero;
namespace {
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct State {
    std::atomic<int> live{0},peak{0},loads{0},searches{0};
    int loading=0,peak_loading=0;
    std::mutex load_mutex;std::condition_variable load_changed;
    bool block_opening=false,fail_search=false,block_load=false,fail_load=false,losing_black=false;
};
struct Service : InferenceService {
    State& state;std::string id;
    Service(State& s,std::string name):state(s),id(std::move(name)) {
        int count=++state.live;++state.loads;
        state.peak=std::max(state.peak.load(),count);
    }
    ~Service(){--state.live;}
    Evaluation evaluate(const std::vector<float>& obs) override {
        if(state.block_opening && id=="slow") {
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
            while(!state.searches && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
            check(state.searches>0,"A ready game must start before another pair's opening finishes");
        }
        ++requests;++batches;
        return {std::vector<double>((obs.size()-GLOBAL_FEATURES)/INPUT_PLANES),{0.5,0,0.5,0}};
    }
    void finish() override {}
    void drain() override {}
    void reset_stats() override {}
};
struct FirstLegal : GameSearch {
    State& state;explicit FirstLegal(State& s):state(s){}
    SearchResult run(const Game& game,double,SearchRun) override {
        ++state.searches;
        if(state.fail_search)throw std::runtime_error("Injected search failure");
        SearchResult result;
        for(int a=0;a<game.actions();++a)if(game.legal(a)){result.action=a;break;}
        if(state.losing_black)result.search_wdl=game.player()==1?WDL{.005,0,.995,0}:WDL{.995,0,.005,0};
        result.root_visits=3;result.new_playouts=3;return result;
    }
    void advance(int) override {}
    void reset(uint64_t) override {}
};
size_t occurrences(const std::string& text,const std::string& needle) {
    size_t n=0,pos=0;while((pos=text.find(needle,pos))!=std::string::npos){++n;pos+=needle.size();}return n;
}
}
int main() {
    auto path=std::filesystem::temp_directory_path()/("etazero_match_"+
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".cfg");
    try {
        std::vector<double> values(6,-.99);
        check(!should_resign(values,1,25,-.95,3),"Minimum turn blocks resignation");
        values.push_back(-.99);check(should_resign(values,1,25,-.95,3),"Consecutive Black losses resign after minimum turn");
        check(!should_resign(values,-1,25,-.95,3),"Winning player cannot resign");
        values[5]=-.95;check(!should_resign(values,1,25,-.95,3),"Threshold equality breaks consecutive streak");
        values.assign(7,.99);check(should_resign(values,-1,25,-.95,6),"White uses absolute-value reversal");
        check(!should_resign(values,-1,25,-1,6),"Minus-one threshold never resigns");
        {std::ofstream file(path);file<<"[network]\ncanvas=5\n[match]\ngame_threads=2\n"
            "playout_doubling_advantage=0\nplayout_doubling_advantage_player=black\n"
            "[opening]\nprobability=1\navg_dist_factor=0.8\nbalance_exponent=10\n"
            "rejection_probability=0.995\nrejection_probability_fallback=0.8\nmax_tries=20\n";}
        Config c(path.string());
        std::vector<MatchModel> models{{"slow","",1},{"fast","",2},{"third","",3}};
        auto run=[&](State& state,std::vector<MatchTask> tasks,int capacity,std::ostream& out,bool single_pair=false,int load_workers=1) {
            return run_matches(c,5,Rule::FREESTYLE,"float32",models,tasks,capacity,
                [&](const MatchModel& m,const MatchEventSink&){
                    if(state.block_load) {
                        std::unique_lock<std::mutex> lock(state.load_mutex);
                        state.peak_loading=std::max(state.peak_loading,++state.loading);
                        state.load_changed.notify_all();
                        check(state.load_changed.wait_for(lock,std::chrono::seconds(2),[&]{return state.peak_loading>=2;}),
                              "Model preparation must run concurrently");
                        --state.loading;
                    }
                    if(state.fail_load && m.id=="fast")throw std::runtime_error("Injected model load failure");
                    return std::make_unique<Service>(state,m.id);},
                [&](InferenceService&,uint64_t){return std::make_unique<FirstLegal>(state);},
                [](const Game&){return 0.;},[]{return false;},out,single_pair,load_workers);
        };
        State pipeline;pipeline.block_opening=true;pipeline.block_load=true;std::ostringstream out;
        run(pipeline,{{"slow",0,0,0,0,10,false,{}},{"fast",1,1,0,0,11,false,{}},
                      {"shared",1,2,0,0,12,false,{}}},3,out,false,2);
        check(pipeline.loads==3 && pipeline.live==0,"Models shared and released");
        check(occurrences(out.str(),"\"type\":\"opening\"")==3,"Each opening generated once");
        check(occurrences(out.str(),"\"type\":\"game\"")==3,"Each independently initialized game finishes once");
        check(out.str().find("\"model_windows\":1")!=std::string::npos,"All resident pairs share a window");
        check(pipeline.peak_loading==2,"Model loading respects its worker limit");
        check(occurrences(out.str(),"\"stage\":\"model_ready\"")==3,"Preparation progress counts every model");
        check(out.str().find("\"stage\":\"playing\"")<out.str().find("\"type\":\"opening\""),
              "Playing starts only after all models are prepared");
        State bounded;std::ostringstream resumed;
        run(bounded,{{"ab",0,1,0,0,10,true,{12}},{"bc",1,2,0,0,11,true,{12}}},2,resumed);
        check(bounded.peak==2 && bounded.loads==3 && bounded.live==0,"Bounded residency retains the common model");
        check(occurrences(resumed.str(),"\"type\":\"opening\"")==0,"Saved openings never regenerated");
        check(occurrences(resumed.str(),"\"type\":\"game\"")==2,"Resume runs only missing games");
        State pair;std::ostringstream pair_output;
        // Duplicate model identity still gets two separately seeded services.
        models[1].id=models[0].id;
        run(pair,{{"",0,1,0,0,10,false,{}},{"",0,1,1,1,11,false,{}},
                  {"",0,1,2,0,12,true,{12}}},2,pair_output,true);
        const auto rows=pair_output.str();
        check(pair.loads==2 && pair.live==0,"Single pair retains two inference services");
        check(occurrences(rows,"\"type\":\"opening\"")==2,"Single pair only generates missing openings");
        check(occurrences(rows,"\"same_bot\":true")==3,"Same model identity selects one search for both colors");
        check(rows.find("\"pair\"")==std::string::npos && rows.find("\"model_stats\"")==std::string::npos,
              "Single pair keeps its event format");
        check(rows.find("\"type\":\"match_stats\",\"games\":3")!=std::string::npos &&
              rows.find("\"inference_a\":{")!=std::string::npos && rows.find("\"inference_b\":{")!=std::string::npos,
              "Single pair records per-side inference statistics");
        models[1].id="fast";
        State failure;failure.fail_search=true;std::ostringstream partial;bool threw=false;
        try{run(failure,{{"ab",0,1,0,0,10,true,{12}}},2,partial);}
        catch(const std::runtime_error& e){threw=std::string(e.what())=="Injected search failure";}
        check(threw && failure.live==0,"Worker failure propagates and releases all models");
        check(occurrences(partial.str(),"\"type\":\"game\"")==0,"Failed half-games are not published");
        State load_failure;load_failure.fail_load=true;std::ostringstream load_partial;threw=false;
        try{run(load_failure,{{"ab",0,1,0,0,10,true,{12}}},2,load_partial,false,2);}
        catch(const std::runtime_error& e){threw=std::string(e.what())=="Injected model load failure";}
        check(threw && load_failure.live==0,"Parallel loading failures release all prepared models");
        check(occurrences(load_partial.str(),"\"type\":\"game\"")==0,"Loading failures never start games");
        {std::ofstream file(path,std::ios::app);file<<"[match]\nallow_resignation=true\nresign_threshold=-0.95\nresign_consec_turns=6\n";}
        Config resign_config(path.string());State losing;losing.losing_black=true;
        std::vector<MatchTask> resign_tasks{{"ab",0,1,0,0,10,true,{}}};std::ostringstream resignation;
        run_matches(resign_config,5,Rule::FREESTYLE,"float32",models,resign_tasks,2,
            [&](const MatchModel& m,const MatchEventSink&){return std::make_unique<Service>(losing,m.id);},
            [&](InferenceService&,uint64_t){return std::make_unique<FirstLegal>(losing);},
            [](const Game&){return 0.;},[]{return false;},resignation,true,1);
        check(losing.searches==7,"Resignation stops after seven searched moves on 5x5");
        check(resignation.str().find("\"reason\":10,\"winner\":-1")!=std::string::npos,"Match publishes resignation winner/reason");
        std::filesystem::remove(path);std::cout<<"match scheduler passed\n";return 0;
    } catch(const std::exception& e){std::filesystem::remove(path);std::cerr<<e.what()<<'\n';return 1;}
}
