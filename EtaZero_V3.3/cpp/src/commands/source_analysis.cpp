#include "etazero/session.h"
#include "etazero/go_position.h"
#include "../../external/katago/neuralnet/nninputs.h"
#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <poll.h>
#include <unistd.h>

namespace etazero {
namespace {
int player(const Json& value) {
    auto s=value.get<std::string>();for(auto& c:s)c=std::tolower(c);
    if(s=="b"||s=="black")return P_BLACK;if(s=="w"||s=="white")return P_WHITE;
    throw std::runtime_error("Expected black or white");
}
Loc location(const Json& value,const ::Board& board,bool pass=true) {
    Loc loc;if(!Location::tryOfString(value.get<std::string>(),board.x_size,board.y_size,loc)||loc==::Board::NULL_LOC||(!pass&&loc==::Board::PASS_LOC))throw std::runtime_error("Invalid board vertex");return loc;
}
std::string vertex(int action,const Game& game) {
    Loc loc=action==game.canvas_area()?::Board::PASS_LOC:Location::getLoc(action%game.canvas(),action/game.canvas(),game.width());
    return Location::toString(loc,game.width(),game.height());
}
int64_t integer(const Json& input,const char* key,int64_t fallback,int64_t low,int64_t high) {
    if(!input.contains(key))return fallback;
    const auto& v=input[key];if(!v.is_number_integer())throw std::runtime_error(std::string(key)+" must be an integer");
    auto n=v.get<int64_t>();if(n<low||n>high)throw std::runtime_error(std::string(key)+" is outside its range");return n;
}
double number(const Json& input,const char* key,double fallback,double low,double high) {
    if(!input.contains(key))return fallback;
    double v=input[key].get<double>();if(!std::isfinite(v)||v<low||v>high)throw std::runtime_error(std::string(key)+" is outside its range");return v;
}
struct Job {
    Json request,overrides,focus_settings;std::string this_hash,sym_hash;std::vector<int> symmetries{0};Game game;int turn;int64_t priority;uint64_t sequence;
    std::atomic<bool> cancelled{false};GameSearch* search=nullptr;
    Job(Json request,Game game,int turn,int64_t priority,uint64_t sequence):request(std::move(request)),game(std::move(game)),turn(turn),priority(priority),sequence(sequence){}
};
struct Earlier {bool operator()(const std::shared_ptr<Job>& a,const std::shared_ptr<Job>& b) const {return a->priority!=b->priority?a->priority<b->priority:a->sequence>b->sequence;}};
class AnalysisServer {
    InferenceService& evaluator_;const Config& config_;Json models_;
    std::function<std::unique_ptr<GameSearch>(const Json&)> factory_;std::function<bool()> interrupted_;
    std::mutex mutex_,output_mutex_;std::condition_variable changed_;bool done_=false;uint64_t sequence_=0;
    std::priority_queue<std::shared_ptr<Job>,std::vector<std::shared_ptr<Job>>,Earlier> queue_;
    std::map<uint64_t,std::shared_ptr<Job>> open_;
    void emit(const Json& value){std::lock_guard<std::mutex> lock(output_mutex_);std::cout<<value.dump()<<'\n'<<std::flush;}
    void focus(Job& job,const Json& input) {
        auto vertices=input.value("focusMoves",std::vector<std::string>{});std::vector<int> moves;
        ::Board board(job.game.width(),job.game.height());for(const auto& v:vertices)moves.push_back(go_action(location(v,board),board.x_size,job.game.canvas()));
        auto weights=input.value("focusWeights",std::vector<double>(moves.size(),1));
        double probability=number(input,"focusProb",.5,0,1);
        if(weights.size()!=moves.size())throw std::runtime_error("focusWeights must match focusMoves");
        for(double w:weights)if(!std::isfinite(w)||w<=0)throw std::runtime_error("focusWeights must be positive");
        if(job.search)job.search->set_root_focus(moves,weights,probability);
        job.focus_settings["focusMoves"]=vertices;job.focus_settings["focusWeights"]=weights;job.focus_settings["focusProb"]=probability;
    }
    Json report(const Job& job,const SearchResult& result,bool during) {
        const auto& g=job.game;const auto& input=job.request;
        Json output={{"id",input["id"]},{"turnNumber",job.turn},{"isDuringSearch",during}};
        if(result.root_visits==0){output["noResults"]=true;return output;}
        int perspective=g.player();auto perspective_setting=job.overrides.value("reportAnalysisWinratesAs",config_.contains("analysis.report_analysis_winrates_as")?config_.text("analysis.report_analysis_winrates_as"):std::string("SIDETOMOVE"));
        for(auto& c:perspective_setting)c=std::toupper(c);
        if(perspective_setting!="SIDETOMOVE")perspective=player(perspective_setting)==P_BLACK?1:-1;
        double sign=perspective==g.player()?1:-1;
        output["rootInfo"]={{"visits",result.root_visits},{"weight",result.search_weight},{"winrate",.5+.5*sign*result.pruned_value},{"utility",sign*result.pruned_utility},
            {"rawWinrate",.5+.5*sign*(result.network_wdl[0]-result.network_wdl[2])},{"rawNoResultProb",result.network_wdl[3]},
            {"currentPlayer",g.player()==1?"B":"W"},{"drawRate",result.pruned_wdl[1]},
            {"rawStWrError",.5*result.network_value_stdev},{"thisHash",job.this_hash},{"symHash",job.sym_hash}};
        if(result.network_variance_time>=0)output["rootInfo"]["rawVarTimeLeft"]=result.network_variance_time;
        if(input.value("includeNoResultValue",false))output["rootInfo"]["noResultValue"]=result.pruned_wdl[3];
        auto candidates=result.analysis;
        std::stable_sort(candidates.begin(),candidates.end(),[&](const auto& a,const auto& b){double x=result.move_policy[a.action],y=result.move_policy[b.action];return x!=y?x>y:result.visits[a.action]>result.visits[b.action];});
        Json infos=Json::array();std::set<int> reported;::Board board(g.width(),g.height());
        auto transform=[&](int action,int symmetry){if(action==g.canvas_area())return action;return go_action(SymmetryHelpers::getSymLoc(Location::getLoc(action%g.canvas(),action/g.canvas(),g.width()),board,symmetry),g.width(),g.canvas());};
        for(const auto& c:candidates) {
            if(!c.has_value)continue;
            Json move={{"move",vertex(c.action,g)},{"visits",c.node_visits},{"weight",c.node_weight},{"edgeVisits",result.visits[c.action]},{"edgeWeight",c.weight},
                {"utility",sign*c.utility},{"winrate",.5+.5*sign*c.win_loss},{"drawRate",c.wdl[1]},{"prior",result.network_policy[c.action]},
                {"order",infos.size()},{"playSelectionValue",c.selection.weight},{"pv",Json::array()}};
            if(c.selection.has_lcb){move["utilityLcb"]=sign*c.selection.lcb;move["lcb"]=.5+.5*sign*(c.win_loss-c.selection.radius);}
            if(input.value("includeNoResultValue",false))move["noResultValue"]=c.wdl[3];
            Game position=g;size_t length=0;
            for(int action:c.pv){if(!position.legal(action))break;move["pv"].push_back(vertex(action,g));position.play(action);++length;}
            if(input.value("includePVVisits",false)) {
                move["pvVisits"]=std::vector<int64_t>(c.pv_visits.begin(),c.pv_visits.begin()+std::min(length,c.pv_visits.size()));
                move["pvEdgeVisits"]=std::vector<int64_t>(c.pv_edge_visits.begin(),c.pv_edge_visits.begin()+std::min(length,c.pv_edge_visits.size()));
            }
            for(int symmetry:job.symmetries) {
                int action=transform(c.action,symmetry);if(!reported.insert(action).second)continue;
                Json equivalent=move;equivalent["move"]=vertex(action,g);equivalent["order"]=infos.size();
                if(symmetry!=0)equivalent["isSymmetryOf"]=vertex(c.action,g);
                for(size_t i=0;i<length;++i)equivalent["pv"][i]=vertex(transform(c.pv[i],symmetry),g);
                infos.push_back(std::move(equivalent));
            }
        }
        output["moveInfos"]=std::move(infos);
        if(input.value("includePolicy",false)){std::vector<double> policy;for(int a=0;a<g.local_actions();++a){int native=g.native_action(a);policy.push_back(g.legal(native)?result.network_policy[native]:-1);}output["policy"]=std::move(policy);if(!result.human_policy.empty()){std::vector<double> human;for(int a=0;a<g.local_actions();++a){int native=g.native_action(a);human.push_back(g.legal(native)?result.human_policy[native]:-1);}output["humanPolicy"]=std::move(human);}}
        return output;
    }
    void worker() {
        std::unique_ptr<GameSearch> search;Json last_overrides;
        for(;;) {
            std::shared_ptr<Job> job;
            {std::unique_lock<std::mutex> lock(mutex_);changed_.wait(lock,[&]{return done_||!queue_.empty();});if(queue_.empty())return;job=queue_.top();queue_.pop();}
            try {
                if(job->cancelled||interrupted_())emit({{"id",job->request["id"]},{"turnNumber",job->turn},{"isDuringSearch",false},{"noResults",true}});
                else {
                    if(!search||last_overrides!=job->overrides){search=factory_(job->overrides);last_overrides=job->overrides;}
                    search->reset(job->sequence+config_.integer64("analysis.seed"));
                    {std::lock_guard<std::mutex> lock(mutex_);job->search=search.get();focus(*job,job->focus_settings);}
                    SearchRun options;options.collect_analysis=true;options.clear_before_search=true;
                    options.max_visits=integer(job->request,"maxVisits",0,1,SEARCH_CONFIG_LIMIT);
                    options.analysis_pv_length=integer(job->request,"analysisPVLen",config_.contains("analysis.analysis_pv_len")?config_.integer("analysis.analysis_pv_len"):15,1,1000);
                    options.should_stop=[&]{return job->cancelled.load()||interrupted_();};
                    bool allow=job->request.contains("allowMoves");auto key=allow?"allowMoves":"avoidMoves";
                    if(job->request.contains(key)) {
                        bool seen[2]={false,false};::Board board(job->game.width(),job->game.height());
                        for(const auto& entry:job->request[key]) {
                            int index=player(entry.at("player"))==P_BLACK?0:1;int depth=integer(entry,"untilDepth",-1,1,1000000000);
                            if(depth<0||(allow&&seen[index]))throw std::runtime_error("Invalid move restrictions");seen[index]=true;
                            auto& avoid=options.avoid_until[index];avoid.resize(job->game.actions(),0);if(allow)std::fill(avoid.begin(),avoid.end(),depth);
                            for(const auto& v:entry.at("moves"))avoid[go_action(location(v,board),board.x_size,job->game.canvas())]=allow?0:depth;
                        }
                    }
                    if(job->request.contains("reportDuringSearchEvery")||job->request.contains("firstReportDuringSearchAfter")) {
                        double interval=number(job->request,"reportDuringSearchEvery",1e30,.001,1e6);
                        double first=number(job->request,"firstReportDuringSearchAfter",interval,.001,1e6);double next=first;
                        options.report_interval=std::min(first,interval);
                        auto start=std::chrono::steady_clock::now();options.on_report=[&,start,interval,next](const SearchResult& r) mutable {
                            double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();if(elapsed<next)return;next=elapsed+interval;emit(report(*job,r,true));};
                    }
                    auto result=search->run(job->game,0,options);emit(report(*job,result,false));
                }
            }catch(const std::exception& e){emit({{"id",job->request["id"]},{"turnNumber",job->turn},{"error",e.what()}});}
            {std::lock_guard<std::mutex> lock(mutex_);job->search=nullptr;open_.erase(job->sequence);}
        }
    }
    void request(Json input) {
        if(!input.is_object()||!input.contains("id")||!input["id"].is_string())throw std::runtime_error("Request requires a string id");
        if(input.contains("action")) {
            auto action=input["action"].get<std::string>();
            if(action=="query_version"){input["version"]="EtaZero_V3.2";input["git_hash"]="unknown";input["referenceCommit"]="d91ea855110dae533f0aada947b2b7d78cc8a4e1";}
            else if(action=="query_models")input["models"]=models_;
            else if(action=="clear_cache")evaluator_.clear_cache();
            else if(action=="terminate"||action=="terminate_all"||action=="set_focus") {
                std::string target;if(action!="terminate_all")target=input.at(action=="set_focus"?"targetId":"terminateId").get<std::string>();
                auto turns=input.value("turnNumbers",std::vector<int>{});
                std::lock_guard<std::mutex> lock(mutex_);
                for(auto& entry:open_) {auto& job=*entry.second;if((action=="terminate_all"||job.request["id"]==target)&&(!input.contains("turnNumbers")||std::find(turns.begin(),turns.end(),job.turn)!=turns.end())) {
                    if(action=="set_focus")focus(job,input);else job.cancelled=true;
                }}
            }else throw std::runtime_error("Unknown analysis action");
            emit(input);changed_.notify_all();return;
        }
        if(evaluator_.spec().kind!=GameKind::GO)throw std::runtime_error("KataGo board protocol requires a Go model; other games use the session protocol");
        static const std::set<std::string> fields={"id","boardXSize","boardYSize","initialStones","initialPlayer","moves","analyzeTurns","priorities","rules","komi","whiteHandicapBonus","overrideSettings","maxVisits","analysisPVLen","rootFpuReductionMax","rootPolicyTemperature","includeMovesOwnership","includeMovesOwnershipStdev","includeOwnership","includeOwnershipStdev","includePolicy","includePVVisits","includeNoResultValue","reportDuringSearchEvery","firstReportDuringSearchAfter","priority","allowMoves","avoidMoves","focusMoves","focusWeights","focusProb"};
        for(auto it=input.begin();it!=input.end();++it)if(!fields.count(it.key()))emit({{"id",input["id"]},{"field",it.key()},{"warning","Unexpected or unused field"}});
        for(const auto* key:{"includeOwnership","includeOwnershipStdev","includeMovesOwnership","includeMovesOwnershipStdev"})if(input.value(key,false))throw std::runtime_error(std::string(key)+" is outside the W/D/L/no-result model");
        if(input.contains("allowMoves")&&input.contains("avoidMoves"))throw std::runtime_error("Cannot combine allowMoves and avoidMoves");
        if(input.contains("priority")&&input.contains("priorities"))throw std::runtime_error("Cannot combine priority and priorities");
        int canvas=config_.integer("network.canvas"),width=integer(input,"boardXSize",-1,2,canvas),height=integer(input,"boardYSize",-1,2,canvas);
        if(width<2||height<2)throw std::runtime_error("Missing board dimensions");
        Game(std::max(width,height),canvas,Rule::GO);::Board board(width,height);
        for(const auto& placement:input.value("initialStones",Json::array())){if(!placement.is_array()||placement.size()!=2)throw std::runtime_error("initialStones requires [player,vertex] pairs");board.setStone(location(placement[1],board,false),player(placement[0]));}
        auto rules_value=input.at("rules");auto rules=::Rules::parseRules(rules_value.is_string()?rules_value.get<std::string>():rules_value.dump());
        rules.komi=number(input,"komi",rules.komi,-400,400);if(!::Rules::komiIsIntOrHalfInt(rules.komi))throw std::runtime_error("komi must be an integer or half-integer");
        if(input.contains("whiteHandicapBonus"))rules.whiteHandicapBonusRule=::Rules::parseWhiteHandicapBonusRule(input["whiteHandicapBonus"].get<std::string>());
        const auto& moves=input.at("moves");if(!moves.is_array())throw std::runtime_error("moves must be an array");
        int next=input.contains("initialPlayer")?player(input["initialPlayer"]):!moves.empty()?player(moves[0][0]) : ::BoardHistory::numHandicapStonesOnBoard(board)>0?P_WHITE:P_BLACK;
        
        auto turns=input.value("analyzeTurns",std::vector<int>{int(moves.size())});std::map<int,int64_t> priorities;
        auto priorities_list=input.value("priorities",std::vector<int64_t>(turns.size(),integer(input,"priority",0,-0x3fffffffffffffffLL,0x3fffffffffffffffLL)));
        if(priorities_list.size()!=turns.size()||(input.contains("priorities")&&!input.contains("analyzeTurns")))throw std::runtime_error("priorities must match analyzeTurns");
        for(size_t i=0;i<turns.size();++i){if(turns[i]<0||size_t(turns[i])>moves.size()||priorities_list[i]<-0x3fffffffffffffffLL||priorities_list[i]>0x3fffffffffffffffLL)throw std::runtime_error("Invalid analysis turn/priority");priorities[turns[i]]=priorities_list[i];}
        Json overrides=input.value("overrideSettings",Json::object());if(!overrides.is_object())throw std::runtime_error("overrideSettings must be an object");
        for(auto key:{"rootFpuReductionMax","rootPolicyTemperature"})if(input.contains(key))overrides[key]=input[key];
        if(input.contains("rootPolicyTemperature"))overrides["rootPolicyTemperatureEarly"]=input["rootPolicyTemperature"];
        auto mode=[&](const std::string& alias,const std::string& key){auto value=config_.contains("analysis."+key)?config_.text("analysis."+key):"auto";for(const auto& name:{alias,key})if(overrides.contains(name)){auto v=overrides[name];value=v.is_string()?v.get<std::string>():v.dump();}return value;};
        auto modes=resolve_go_modes(mode("alwaysComputePassAliveUnderSuicideRules","always_compute_pass_alive_under_suicide_rules"),mode("excludeTerritoryAdjacentToAtari","exclude_territory_adjacent_to_atari"),evaluator_.description());
        ::BoardHistory history(board,next,rules,0,modes);history.setAssumeMultipleStartingBlackMovesAreHandicap(true);
        std::vector<std::shared_ptr<Job>> jobs;
        for(size_t turn=0;turn<=moves.size();++turn) {
            if(priorities.count(turn)){
                auto game=game_from_go_history(history,canvas);
                double pda=config_.number("analysis.playout_doubling_advantage");std::string pda_player=config_.text("analysis.playout_doubling_advantage_player");
                for(auto key:{"playoutDoublingAdvantage","playout_doubling_advantage"})if(overrides.contains(key)){auto v=overrides[key];pda=v.is_string()?std::stod(v.get<std::string>()):v.get<double>();}
                for(auto key:{"playoutDoublingAdvantagePla","playout_doubling_advantage_player"})if(overrides.contains(key))pda_player=overrides[key].get<std::string>();
                game.set_pda(pda,player(pda_player)==P_BLACK?1:-1);
                auto job=std::make_shared<Job>(input,std::move(game),turn,priorities[turn],sequence_++);job->overrides=overrides;focus(*job,input);
                bool prune=config_.boolean("analysis.root_symmetry_pruning");
                for(auto key:{"rootSymmetryPruning","root_symmetry_pruning"})if(overrides.contains(key)) {
                    auto value=overrides[key];std::string text=value.is_string()?value.get<std::string>():value.dump();for(auto& c:text)c=std::tolower(c);
                    if(text!="true"&&text!="false")throw std::runtime_error("Invalid rootSymmetryPruning");prune=text=="true";
                }
                if(prune) {
                    std::vector<int> avoid(::Board::MAX_ARR_SIZE,0);bool duplicate[::Board::MAX_ARR_SIZE];
                    for(auto key:{"allowMoves","avoidMoves"})if(input.contains(key))for(const auto& entry:input[key])if(player(entry.at("player"))==history.presumedNextMovePla) {
                        int depth=integer(entry,"untilDepth",-1,1,1000000000);if(depth<0)throw std::runtime_error("Missing untilDepth");
                        bool allow=std::string(key)=="allowMoves";if(allow)std::fill(avoid.begin(),avoid.end(),depth);
                        for(const auto& v:entry.at("moves"))avoid[location(v,board)]=allow?0:depth;
                    }
                    job->symmetries.clear();SymmetryHelpers::markDuplicateMoveLocs(board,history,nullptr,avoid,duplicate,job->symmetries);
                }
                Hash128 first=board.getSitHashWithSimpleKo(history.presumedNextMovePla),minimum=first;
                for(int s=1;s<8;++s){auto hash=SymmetryHelpers::getSymBoard(board,s).getSitHashWithSimpleKo(history.presumedNextMovePla);if(hash<minimum)minimum=hash;}
                job->this_hash=Global::uint64ToHexString(first.hash1)+Global::uint64ToHexString(first.hash0);
                job->sym_hash=Global::uint64ToHexString(minimum.hash1)+Global::uint64ToHexString(minimum.hash0);jobs.push_back(std::move(job));
            }
            if(turn==moves.size())break;const auto& move=moves[turn];if(!move.is_array()||move.size()!=2)throw std::runtime_error("moves requires [player,vertex] pairs");
            int pla=player(move[0]);auto loc=location(move[1],board);
            if(pla!=next){board.clearSimpleKoLoc();history.clear(board,pla,rules,history.encorePhase);history.setAssumeMultipleStartingBlackMovesAreHandicap(true);}
            if(!history.makeBoardMoveTolerant(board,loc,pla,true))throw std::runtime_error("Illegal move at turn "+std::to_string(turn));next=getOpp(pla);
        }
        {std::lock_guard<std::mutex> lock(mutex_);for(auto& job:jobs){open_[job->sequence]=job;queue_.push(job);}}changed_.notify_all();
    }
public:
    AnalysisServer(InferenceService& e,const Config& c,Json models,std::function<std::unique_ptr<GameSearch>(const Json&)> f,std::function<bool()> stop)
        :evaluator_(e),config_(c),models_(std::move(models)),factory_(std::move(f)),interrupted_(std::move(stop)){}
    int run() {
        int count=config_.contains("analysis.num_analysis_threads")?config_.integer("analysis.num_analysis_threads"):1;
        if(count<1||count>16384)throw std::runtime_error("numAnalysisThreads must be in [1,16384]");
        std::vector<std::thread> workers;
        try{for(int i=0;i<count;++i)workers.emplace_back(&AnalysisServer::worker,this);}
        catch(...){{std::lock_guard<std::mutex> lock(mutex_);done_=true;}changed_.notify_all();for(auto& t:workers)t.join();throw;}
        std::string buffer;bool eof=false;
        while(!eof&&!interrupted_()) {
            pollfd fd{STDIN_FILENO,POLLIN,0};int ready=::poll(&fd,1,100);if(ready<0){if(errno==EINTR)continue;break;}if(!ready)continue;
            char bytes[16384];auto n=::read(STDIN_FILENO,bytes,sizeof(bytes));if(n>0)buffer.append(bytes,n);else if(n==0)eof=true;else if(errno!=EINTR&&errno!=EAGAIN)break;
            for(;;){auto end=buffer.find('\n');if(end==std::string::npos&&!eof)break;if(buffer.empty())break;
                std::string line=end==std::string::npos?std::move(buffer):buffer.substr(0,end);if(end==std::string::npos)buffer.clear();else buffer.erase(0,end+1);
                if(trim(line).empty())continue;Json input;
                try{input=Json::parse(line);request(input);}catch(const std::exception& e){Json error={{"error",e.what()}};if(input.is_object()&&input.contains("id"))error["id"]=input["id"];emit(error);}
            }
        }
        {std::lock_guard<std::mutex> lock(mutex_);done_=true;if(interrupted_())for(auto& job:open_)job.second->cancelled=true;}
        changed_.notify_all();for(auto& thread:workers)thread.join();return interrupted_()?130:0;
    }
};
}
int source_analysis_session(InferenceService& e,const Config& c,const Json& models,const std::function<std::unique_ptr<GameSearch>(const Json&)>& factory,const std::function<bool()>& stop) {
    return AnalysisServer(e,c,models,factory,stop).run();
}
}
