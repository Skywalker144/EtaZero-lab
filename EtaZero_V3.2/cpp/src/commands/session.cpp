#include "etazero/game_config.h"
#include "etazero/session.h"
#include <algorithm>
#include <charconv>
#include <deque>
#include <iostream>
#include <numeric>
#include <optional>
#include <set>
#include <poll.h>
#include <unistd.h>
#include <cerrno>
namespace etazero {
namespace {
int64_t integer(const Json& object,const char* key,int64_t fallback,int64_t minimum,int64_t maximum) {
    if(!object.contains(key))return fallback;
    const auto& v=object.at(key);
    if(!v.is_number_integer() || v.get<int64_t>()<minimum || v.get<int64_t>()>maximum)
        throw std::runtime_error(std::string(key)+" is outside its integer range");
    return v.get<int64_t>();
}
int number(const std::string& s) {return parse_integer(s,"command");}
const std::string columns="ABCDEFGHJKLMNOPQRSTUVWXYZ";
std::string vertex(int a,const Game& g) {
    if(g.rule()==Rule::XIANGQI){auto m=XIANGQI_MOVES[a];return std::string(1,char('a'+m[0]))+std::to_string(m[1])+char('a'+m[2])+std::to_string(m[3]);}
    if(a==g.size()*g.size())return "pass";
    return std::string(1,columns.at(a%g.size()))+std::to_string(g.size()-a/g.size());
}
int action(const std::string& v,const Game& g) {
    if(g.rule()==Rule::GO && (v=="pass"||v=="PASS"))return g.size()*g.size();
    if(g.rule()==Rule::XIANGQI){
        if(v.size()!=4)throw std::runtime_error("Expected Xiangqi move such as h2e2");
        int a=xiangqi_action(v[0]-'a',v[1]-'0',v[2]-'a',v[3]-'0');
        if(a<0)throw std::runtime_error("Unknown Xiangqi move");return a;
    }
    if(v.size()<2)throw std::runtime_error("Expected board coordinate");
    int size=g.size();auto x=columns.find(std::toupper(static_cast<unsigned char>(v[0])));int y=number(v.substr(1));
    if(x>=size_t(size)||y<1||y>size)throw std::runtime_error("Move outside board");
    return (size-y)*size+int(x);
}
int color(std::string v) {
    for(char& c:v)c=char(std::tolower(static_cast<unsigned char>(c)));
    if(v=="b"||v=="black")return 1;if(v=="w"||v=="white")return -1;
    throw std::runtime_error("Expected black or white");
}
std::vector<std::string> words(std::string line) {
    auto comment=line.find('#');if(comment!=std::string::npos)line.resize(comment);
    std::istringstream in(line);std::vector<std::string> result;
    for(std::string word;in>>word;)result.push_back(word);return result;
}
struct Request {
    Json data=Json::object(),id=nullptr;
    std::string command,gtp_id;
    std::vector<std::string> args;
    std::shared_ptr<std::atomic<bool>> cancelled=std::make_shared<std::atomic<bool>>(false);
    bool continuous=false,streaming=false,internal=false;
};
const std::vector<std::string> commands={"protocol_version","name","version","list_commands","known_command",
    "boardsize","clear_board","play","undo","genmove","quit","kata-analyze","kata-genmove_analyze",
    "stop","kata-raw-nn","etazero-rule","komi","showboard","time_settings","time_left","kgs-time_settings","kata-time_settings","kata-list_time_settings"};
class Session {
    GameSearch& search_;InferenceService& evaluator_;const Config& config_;uint64_t seed_;
    std::string protocol_;int canvas_;std::optional<Game> game_;std::vector<int> history_;Json opening_=nullptr;
    std::vector<std::vector<int>> boards_;std::vector<int> players_;
    std::function<Json(const Game&)> raw_;std::function<double(const Game&)> temperature_;std::function<bool()> interrupted_;
    std::mutex mutex_,output_mutex_;std::condition_variable changed_;std::deque<Request> queue_;std::optional<Request> active_;bool eof_=false;
    std::array<TimeControls,2> clocks_;bool want_ponder_=false;int search_for_player_=0;
    std::set<std::string> pending_ids_;
    std::string input_buffer_;bool input_closed_=false;
    bool read_line(std::string& line) {
        while(!interrupted_()) {
            auto end=input_buffer_.find('\n');
            if(end!=std::string::npos){line=input_buffer_.substr(0,end);input_buffer_.erase(0,end+1);return true;}
            if(input_closed_){if(input_buffer_.empty())return false;line=std::move(input_buffer_);input_buffer_.clear();return true;}
            pollfd input{STDIN_FILENO,POLLIN,0};int ready=::poll(&input,1,100);
            if(ready<0){if(errno==EINTR)continue;return false;}
            if(!ready)continue;
            char block[16384];auto count=::read(STDIN_FILENO,block,sizeof(block));
            if(count>0)input_buffer_.append(block,size_t(count));else if(count==0)input_closed_=true;
            else if(errno!=EINTR&&errno!=EAGAIN)return false;
        }
        return false;
    }
    int native(int local) const {return game_->native_action(local);}
    void emit(const Json& value) {std::lock_guard<std::mutex> lock(output_mutex_);std::cout<<value.dump()<<'\n'<<std::flush;}
    void output(const std::string& value) {std::lock_guard<std::mutex> lock(output_mutex_);std::cout<<value<<std::flush;}
    void reply(const Request& r,Json result,bool final=true) {
        result["ok"]=true;if(protocol_=="json") {
            result["id"]=r.id;result["event"]=final?"result":"analysis";result["final"]=final;
            if(final){std::lock_guard<std::mutex> lock(mutex_);pending_ids_.erase(r.id.dump());}
        }
        emit(result);
    }
    Json state() const {
        if(!game_)return nullptr;const auto& g=*game_;
        std::vector<int> legal;for(int a=0;a<g.actions();++a)if(g.legal(a))legal.push_back(g.local_action(a));
        return {{"board_width",g.width()},{"board_height",g.height()},{"board_stride",g.geometry().stride},{"legal_actions",legal},{"fen",g.fen()},{"komi",g.options().komi},{"go_rules",g.options().go_rules},{"board_size",g.size()},{"canvas_size",canvas_},{"rule",int(g.rule())},{"player",g.player()},
            {"turn",g.turn()},{"finished",g.finished()},{"winner",g.winner()},{"reason",int(g.reason())},
            {"board",g.board().cells},{"moves",history_},{"opening",opening_},{"boards",boards_},{"players",players_}};
    }
    void create(const Json& data,const Request& r) {
        int size=integer(data,"size",config_.integer("analysis.board_size"),5,canvas_);
        auto options=rule_options(config_,GameConfigMode::ANALYSIS);options.fen=data.value("fen",std::string());
        options.komi=data.value("komi",options.komi);options.go_rules=data.value("go_rules",options.go_rules);
        Game replacement(size,canvas_,parse_rule(data.value("rule",config_.text("analysis.rule"))),options);
        if(!same_layout(replacement.spec(),evaluator_.spec()))
            throw std::runtime_error("Requested game layout does not match the loaded model");
        std::vector<int> moves;Json opening=nullptr;
        if(data.value("opening",std::string("empty"))=="balanced") {
            OpeningConfig settings(config_);uint64_t seed=seed_;
            if(data.contains("seed")){auto s=data.at("seed").get<std::string>();size_t used=0;seed=std::stoull(s,&used);if(used!=s.size()||s.empty()||s[0]=='-')throw std::runtime_error("Invalid opening seed");}
            int timeout=integer(data,"timeout_ms",120000,1,120000);std::mt19937_64 random(seed);
            if(replacement.rule()!=Rule::GO && replacement.rule()!=Rule::XIANGQI && (replacement.rule()==Rule::HEX?(settings.hex_probability!=1||settings.hex_make_fair_probability!=1||settings.hex_min_accept_rate<=0):
                (settings.probability!=1||settings.rejection_probability_fallback>=1)))throw std::runtime_error("Balanced opening requires probability one and rejection fallback below one");
            auto start=std::chrono::steady_clock::now();auto cancelled=[&]{return r.cancelled->load()||interrupted_()||std::chrono::steady_clock::now()-start>=std::chrono::milliseconds(timeout);};
            for(int retry=0;retry<100;++retry) {
                replacement=Game(size,canvas_,replacement.rule(),options);auto result=initialize_opening(replacement,settings,evaluator_,random,cancelled);
                if(cancelled()||result.status==OpeningStatus::Interrupted)throw std::runtime_error("平衡开局生成已取消或超时，请重新创建棋局");
                if(replacement.rule()!=Rule::GO && replacement.rule()!=Rule::XIANGQI && result.status!=OpeningStatus::Success)throw std::runtime_error("Balanced opening failed: "+result.failure);
                if(replacement.finished())continue;
                for(int a:result.actions)moves.push_back(replacement.local_action(a));
                opening={{"seed",std::to_string(seed)},{"attempts",result.attempts},{"balanced_moves",result.balanced_moves},
                    {"policy_moves",result.policy_moves},{"value",result.start_value},{"value_player",result.balanced_moves%2==0?1:-1},
                    {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},{"moves",moves}};break;
            }
            if(opening.is_null())throw std::runtime_error("100 consecutive terminal openings");
        } else if(data.value("opening",std::string("empty"))!="empty")throw std::runtime_error("Unknown opening kind");
        if(data.contains("moves")) {
            if(!opening.is_null())throw std::runtime_error("moves cannot be combined with balanced opening");
            if(!data.at("moves").is_array())throw std::runtime_error("moves must be an array");
            for(const auto& move:data.at("moves")) {
                if(!move.is_number_integer())throw std::runtime_error("moves must contain integer board actions");
                int64_t a=move.get<int64_t>();if(a<0||a>=replacement.local_actions())throw std::runtime_error("Move outside board");
                replacement.play(replacement.native_action(int(a)));moves.push_back(int(a));
            }
        }
        game_=std::move(replacement);history_=std::move(moves);opening_=std::move(opening);search_.reset(seed_);
        boards_.clear();players_.clear();
        if(game_->rule()==Rule::GO||game_->rule()==Rule::XIANGQI){Game replay(size,canvas_,game_->rule(),options);
            boards_.push_back(replay.board().cells);players_.push_back(replay.player());
            for(int a:history_){replay.play(replay.native_action(a));boards_.push_back(replay.board().cells);players_.push_back(replay.player());}}
    }
    void play(int local) {if(local<0||local>=game_->local_actions())throw std::runtime_error("Move outside board");int a=native(local);game_->play(a);history_.push_back(local);search_.advance(a);
        if(!boards_.empty()){boards_.push_back(game_->board().cells);players_.push_back(game_->player());}}
    void undo(int count) {
        if(count<1||count>int(history_.size()))throw std::runtime_error("Invalid undo count");
        Game replacement(game_->size(),canvas_,game_->rule(),game_->options());auto retained=history_;retained.resize(retained.size()-count);
        for(int a:retained)replacement.play(replacement.native_action(a));
        game_=std::move(replacement);history_=std::move(retained);search_.reset(seed_);
        if(!boards_.empty()){boards_.resize(history_.size()+1);players_.resize(history_.size()+1);}
        if(!opening_.is_null()&&history_.size()<opening_.at("moves").size())opening_=nullptr;
    }
    std::string infos(const Json& analysis,const Request& r) {
        int maxmoves=integer(r.data,"maxmoves",100000,1,100000),count=0;std::ostringstream out;out.precision(17);
        for(const auto& c:analysis.at("candidates")) {
            if(c.at("visits").get<int64_t>()<=0||count>=maxmoves)continue;if(count++)out<<' ';
            out<<"info move "<<vertex(c.at("action"),*game_)<<" visits "<<c.at("visits")
               <<" utility "<<c.at("utility")<<" winrate "<<c.at("winrate")<<" drawRate "<<c.at("draw_rate")
               <<" prior "<<c.at("network_prior")<<" weight "<<c.at("sample_weight")<<" order "<<count-1;
            if(!c.at("lcb").is_null())out<<" utilityLcb "<<c.at("lcb")<<" lcb "<<(1+c.at("lcb").get<double>())*.5<<" radius "<<c.at("radius");
            if(!c.at("raw_selection_weight").is_null())out<<" selectionWeight "<<c.at("raw_selection_weight");
            out<<" policy "<<c.at("improved_policy")<<" pv";
            for(const auto& a:c.at("pv"))out<<' '<<vertex(a,*game_);
        }
        return out.str()+"\n";
    }
    std::vector<int> move_list(const Json& moves) const {
        if(!moves.is_array())throw std::runtime_error("moves must be an array");
        std::vector<int> result;
        for(const auto& item:moves) {
            if(!item.is_number_integer() || item.get<int64_t>()<0 || item.get<int64_t>()>=game_->local_actions())throw std::runtime_error("Move outside board");
            result.push_back(native(item.get<int>()));
        }
        return result;
    }
    void set_focus(const Json& data) {
        auto moves=move_list(data.value("focus_moves",Json::array()));
        auto weights=data.contains("focus_weights")?data.at("focus_weights").get<std::vector<double>>():std::vector<double>(moves.size(),1);
        double probability=data.value("focus_probability",.5);
        if(!std::isfinite(probability) || probability<0 || probability>1)throw std::runtime_error("Focus probability must be in [0,1]");
        search_.set_root_focus(std::move(moves),std::move(weights),probability);
    }
    void restrictions(const Json& data,SearchRun& options) {
        if(data.contains("avoid_moves") && data.contains("allow_moves"))throw std::runtime_error("Cannot combine allow_moves and avoid_moves");
        bool allow=data.contains("allow_moves");const char* key=allow?"allow_moves":"avoid_moves";
        if(data.contains(key)) {
            const auto& entries=data.at(key);if(!entries.is_array())throw std::runtime_error("Move restrictions must be an array");
            bool seen[2]={false,false};
            for(const auto& entry:entries) {
                int player=entry.at("player").is_string()?color(entry.at("player").get<std::string>()):entry.at("player").get<int>();
                if(player!=1 && player!=-1)throw std::runtime_error("Restriction player must be black or white");
                int index=player==1?0:1,depth=integer(entry,"until_depth",-1,1,1000000000);
                if(depth<1)throw std::runtime_error("Missing until_depth");
                if(allow && seen[index])throw std::runtime_error("Only one allow_moves entry per player is permitted");seen[index]=true;
                auto moves=move_list(entry.at("moves"));auto& avoid=options.avoid_until[index];avoid.resize(game_->actions(),0);
                if(allow)std::fill(avoid.begin(),avoid.end(),depth);
                for(int action:moves)avoid[action]=allow?0:depth;
            }
        }
        options.avoid_rescale_root=data.value("avoid_rescale_root",false);
    }
    void set_clock(const std::vector<std::string>& args,bool standard) {
        auto time=[&](size_t i,bool negative=false){if(i>=args.size())throw std::runtime_error("Missing time argument");size_t n=0;double value=std::stod(args[i],&n);
            if(n!=args[i].size()||!std::isfinite(value)||value>(TimeControls::MAX_USER_INPUT_TIME)||(!negative&&value<0))throw std::runtime_error("Invalid time");
            return negative&&value<0?TimeControls::MAX_USER_INPUT_TIME:value;};
        auto count=[&](size_t i){if(i>=args.size())throw std::runtime_error("Missing period count");int value=number(args[i]);if(value<0||value>100000)throw std::runtime_error("Invalid period count");return value;};
        TimeControls clock;std::string kind=standard?"canadian":args.empty()?"":args[0];size_t first=standard?0:1;
        size_t expected=kind=="none"?1:kind=="absolute"?2:kind=="fischer"?3:kind=="fischer-capped"?5:first+3;
        if(args.size()!=expected)throw std::runtime_error("Wrong number of clock arguments");
        if(kind=="none"){}
        else if(kind=="absolute")clock=TimeControls::absoluteTime(time(1));
        else if(kind=="fischer")clock=TimeControls::fischerTime(time(1),time(2));
        else if(kind=="fischer-capped")clock=TimeControls::fischerCappedTime(time(1),time(2),time(3,true),time(4,true));
        else if(kind=="canadian"||kind=="byoyomi") {
            double main=time(first),period=time(first+1);int n=count(first+2);
            if(n==0)clock=kind=="canadian"&&period>0?TimeControls():TimeControls::absoluteTime(main);
            else clock=TimeControls::canadianOrByoYomiTime(main,period,kind=="byoyomi"?n:1,kind=="canadian"?n:1);
        } else throw std::runtime_error("Unknown clock type");
        clocks_.fill(clock);
    }
    void clock_left(const std::vector<std::string>& args) {
        if(args.size()!=3)throw std::runtime_error("Expected player, time, and stones");
        int player=color(args[0]),stones=number(args[2]);size_t used=0;double time=std::stod(args[1],&used);
        if(used!=args[1].size() || !std::isfinite(time)||time< -10||time>TimeControls::MAX_USER_INPUT_TIME || stones<0||stones>100000)throw std::runtime_error("Invalid time left");
        auto clock=clocks_[player==1?0:1];
        if(stones>0 && clock.originalNumPeriods<=0)throw std::runtime_error("Clock has no overtime periods");
        if(stones==0){clock.mainTimeLeft=time;clock.inOvertime=false;clock.numPeriodsLeftIncludingCurrent=clock.originalNumPeriods;clock.numStonesLeftInPeriod=0;clock.timeLeftInPeriod=0;}
        else {clock.mainTimeLeft=0;clock.inOvertime=true;clock.timeLeftInPeriod=time;
            bool japanese=clock.originalNumPeriods>1&&clock.numStonesPerPeriod==1;
            clock.numPeriodsLeftIncludingCurrent=japanese?std::min(stones,clock.originalNumPeriods):1;
            clock.numStonesLeftInPeriod=japanese?1:std::min(stones,clock.numStonesPerPeriod);}
        clocks_[player==1?0:1]=clock;
    }
    void ponder(Request& r) {
        if(!game_ || game_->finished())return;
        SearchRun options;options.pondering=true;options.search_for_player=search_for_player_?search_for_player_:-game_->player();
        options.should_stop=[&]{return r.cancelled->load()||interrupted_();};
        auto result=search_.run(*game_,temperature_(*game_),options);
        if(!r.internal)reply(r,{{"state",state()},{"analysis",analysis_json(result,*game_,0,0,true)}});
    }
    Json analyze(Request& r) {
        if(game_->finished())throw std::runtime_error("Game has finished");
        int64_t visits=integer(r.data,"visits",config_.integer64("analysis.visits"),1,SEARCH_CONFIG_LIMIT);
        int interval=integer(r.data,"report_interval_ms",protocol_=="text"?0:250,0,60000);
        SearchRun options;restrictions(r.data,options);{std::lock_guard<std::mutex> lock(mutex_);set_focus(active_->data);}
        options.max_visits=r.continuous?SEARCH_CONFIG_LIMIT:visits;
        const bool generate_move=r.command=="genmove"||r.command=="kata-genmove_analyze";
        if(generate_move){options.time_controls=&clocks_[game_->player()==1?0:1];options.search_for_player=game_->player();}
        options.clear_before_search=r.data.value("fresh",true);options.collect_analysis=true;
        options.should_stop=[&]{return r.cancelled->load()||interrupted_();};
        if(r.continuous){options.max_playouts=SEARCH_CONFIG_LIMIT;options.max_time=1e20;}
        auto requests=evaluator_.requests.load(),batches=evaluator_.batches.load();Game before=*game_;
        const bool gtp_stream=protocol_=="gtp"&&(r.command=="kata-analyze"||r.command=="kata-genmove_analyze");
        if(gtp_stream){output("="+r.gtp_id+"\n");r.streaming=true;}
        if(interval>0) {
            options.report_interval=interval*.001;options.on_report=[&](const SearchResult& result) {
                auto report=analysis_json(result,before,evaluator_.requests.load()-requests,evaluator_.batches.load()-batches,false);
                if(protocol_=="gtp")output(infos(report,r));else reply(r,{{"analysis",report},{"state",state()}},false);
            };
        }
        auto result=search_.run(before,temperature_(before),options);
        auto report=analysis_json(result,before,evaluator_.requests.load()-requests,evaluator_.batches.load()-batches,true);
        if(result.action<0) {
            if(!r.cancelled->load())throw std::runtime_error("Search has no move: zero playout budget");
        } else if(!r.cancelled->load()&&!interrupted_())report["network_planes"]=raw_(before);
        if(gtp_stream)output(infos(report,r));
        bool generate=r.command=="genmove"||r.command=="kata-genmove_analyze";
        bool moved=false;
        {
            // Cancellation and committing a generated move have one ordering point.
            std::lock_guard<std::mutex> lock(mutex_);
            moved=generate&&result.action>=0&&!r.cancelled->load()&&!interrupted_();
            if(moved){search_for_player_=before.player();play(before.local_action(result.action));
                want_ponder_=config_.contains("analysis.pondering_enabled")&&config_.boolean("analysis.pondering_enabled")&&!game_->finished();}
            active_.reset();
        }
        if(protocol_=="gtp") {
            if(gtp_stream) {
                output((generate?(moved?"play "+vertex(report.at("action"),before):"error search cancelled"):"")+"\n");
                r.streaming=false;
            } else if(moved)output("="+r.gtp_id+" "+vertex(report.at("action"),before)+"\n\n");
            else output("?"+r.gtp_id+" search cancelled\n\n");
        }
        return {{"state",state()},{"analysis",report},{"cancelled",r.cancelled->load()||interrupted_()},{"played",moved}};
    }
    void execute(Request& r) {
        if(((r.cancelled->load()&&!r.continuous)||interrupted_())&&protocol_=="json") {reply(r,{{"cancelled",true},{"state",state()},{"played",false}});return;}
        const auto& cmd=r.command;
        if(cmd=="ponder"){ponder(r);return;}
        bool gtp=protocol_=="gtp";std::string value;bool search_reply=false;
        if(gtp) {
            if(std::find(commands.begin(),commands.end(),cmd)==commands.end())throw std::runtime_error("Unknown command");
            size_t expected=(cmd=="known_command"||cmd=="boardsize"||cmd=="etazero-rule"||cmd=="komi"||cmd=="genmove")?1:cmd=="play"?2:0;
            if(cmd!="kata-analyze"&&cmd!="kata-genmove_analyze"&&cmd!="kata-raw-nn"&&cmd!="time_settings"&&cmd!="time_left"&&cmd!="kata-time_settings"&&cmd!="kgs-time_settings"&&r.args.size()!=expected)throw std::runtime_error("Unexpected command arguments");
            if(cmd=="protocol_version")value="2";else if(cmd=="name")value="EtaZero";else if(cmd=="version")value="2.9";
            else if(cmd=="known_command")value=std::find(commands.begin(),commands.end(),r.args[0])!=commands.end()?"true":"false";
            else if(cmd=="list_commands"){for(const auto& c:commands){if(!value.empty())value+='\n';value+=c;}}
            else if(cmd=="boardsize"||cmd=="clear_board"||cmd=="etazero-rule") {
                Json data={{"size",cmd=="boardsize"?number(r.args[0]):game_->size()},
                    {"rule",cmd=="etazero-rule"?r.args[0]:rule_name()},{"komi",game_->options().komi},{"go_rules",game_->options().go_rules}};create(data,r);
            } else if(cmd=="komi") {
                if(game_->rule()!=Rule::GO)throw std::runtime_error("komi requires Go");
                size_t used=0;double komi=std::stod(r.args[0],&used);
                if(used!=r.args[0].size()||!std::isfinite(komi))throw std::runtime_error("Invalid komi");
                create({{"size",game_->size()},{"rule","go"},{"komi",komi},{"go_rules",game_->options().go_rules},{"moves",history_}},r);
            } else if(cmd=="play") {if(color(r.args[0])!=game_->player())throw std::runtime_error("Color is not side to move");play(action(r.args[1],*game_));}
            else if(cmd=="undo")undo(1);
            else if(cmd=="time_settings"||cmd=="kata-time_settings"||cmd=="kgs-time_settings"){if(cmd=="kgs-time_settings"&&!r.args.empty()&&r.args[0].rfind("fischer",0)==0)throw std::runtime_error("Use kata-time_settings for Fischer clocks");set_clock(r.args,cmd=="time_settings");}
            else if(cmd=="time_left")clock_left(r.args);
            else if(cmd=="kata-list_time_settings")value="none absolute byoyomi canadian fischer fischer-capped";
            else if(cmd=="genmove"||cmd=="kata-analyze"||cmd=="kata-genmove_analyze") {
                size_t i=0;bool needs_color=cmd!="kata-analyze";
                if(i<r.args.size()&&(r.args[i]=="b"||r.args[i]=="w"||r.args[i]=="B"||r.args[i]=="W"||r.args[i]=="black"||r.args[i]=="white")) {
                    if(color(r.args[i++])!=game_->player())throw std::runtime_error("Color is not side to move");needs_color=false;
                }
                if(needs_color)throw std::runtime_error("Expected side to move");
                r.data["report_interval_ms"]=cmd=="genmove"?0:100;
                r.data["fresh"]=false;
                if(i<r.args.size()&&std::isdigit(static_cast<unsigned char>(r.args[i][0])))r.data["report_interval_ms"]=number(r.args[i++])*10;
                while(i<r.args.size()) {
                    std::string key=r.args[i++];if(i==r.args.size())throw std::runtime_error("Missing analysis option value");
                    if(key=="interval")r.data["report_interval_ms"]=number(r.args[i++])*10;
                    else if(key=="maxmoves"||key=="visits"){auto value=r.args[i++];size_t used=0;int64_t count=std::stoll(value,&used);if(used!=value.size())throw std::runtime_error("Invalid visit count");r.data[key]=count;}
                    else if(key=="avoid"||key=="allow") {
                        if(i+2>=r.args.size())throw std::runtime_error("Expected player, comma-separated moves, depth");
                        int player=color(r.args[i++]);std::string list=r.args[i++];Json moves=Json::array();
                        std::istringstream stream(list);for(std::string v;std::getline(stream,v,',');)moves.push_back(game_->local_action(action(v,*game_)));
                        std::string field=key=="allow"?"allow_moves":"avoid_moves";
                        if(!r.data.contains(field))r.data[field]=Json::array();
                        r.data[field].push_back({{"player",player},{"moves",moves},{"until_depth",number(r.args[i++])}});
                    }
                    else throw std::runtime_error("Unsupported analysis option: "+key);
                }
                if(r.data.contains("visits"))r.continuous=false;
                integer(r.data,"maxmoves",100000,1,100000);
                analyze(r);return;
            } else if(cmd=="kata-raw-nn") {
                if(r.args.size()>1||(r.args.size()==1&&r.args[0]!="0"))throw std::runtime_error("kata-raw-nn supports symmetry 0 only");
                value=raw_(*game_).dump();
            } else if(cmd=="showboard")value=state().dump();
            output("="+r.gtp_id+(value.empty()?"":" "+value)+"\n\n");return;
        }
        if(cmd=="time_settings") {
            std::vector<std::string> args{r.data.at("clock_type").get<std::string>()};
            for(const auto& value:r.data.value("clock_arguments",Json::array()))args.push_back(value.is_string()?value.get<std::string>():value.dump());
            set_clock(args,false);
        } else if(cmd=="time_left"){auto player=r.data.at("player");if(!player.is_string()&&(!player.is_number_integer()||std::abs(player.get<int>())!=1))throw std::runtime_error("Clock player must be black/white or +1/-1");clock_left({player.is_string()?player.get<std::string>():player.get<int>()==1?"b":"w",r.data.at("time").dump(),r.data.at("stones").dump()});}
        else if(cmd=="new")create(r.data,r);
        else if(cmd=="quit"||cmd=="stop"){if(protocol_=="json")reply(r,{{"state",state()}});return;}
        else if(!game_&&!(r.data.contains("moves")&&(cmd=="analyze"||cmd=="genmove")))throw std::runtime_error("Start a new game first");
        else if(cmd=="play")play(integer(r.data,"action",-1,0,game_->local_actions()-1));
        else if(cmd=="undo")undo(integer(r.data,"count",1,1,int(history_.size())));
        else if(cmd=="analyze"||cmd=="genmove") {
            if(r.data.contains("moves"))create(r.data,r);
            auto result=analyze(r);reply(r,result);search_reply=true;
        } else if(cmd=="raw_nn") {reply(r,{{"state",state()},{"network_planes",raw_(*game_)}});return;}
        else if(cmd!="state")throw std::runtime_error("Unknown command");
        if(!search_reply)reply(r,{{"state",state()}});
    }
    std::string rule_name() const {
        for(const auto& name:{"freestyle","standard","renju","hex","go","xiangqi"})if(parse_rule(name)==game_->rule())return name;
        throw std::runtime_error("Unknown rule");
    }
    void work() {
        for(;;) {
            Request r;
            {std::unique_lock<std::mutex> lock(mutex_);changed_.wait(lock,[&]{return eof_||!queue_.empty()||want_ponder_;});
                if(queue_.empty()) {
                    if(eof_)return;r.command="ponder";r.internal=true;want_ponder_=false;
                } else {r=std::move(queue_.front());queue_.pop_front();want_ponder_=false;}
                active_=r;}
            try{execute(r);}catch(const std::exception& error) {
                if(r.internal){std::cerr<<"Pondering failed: "<<error.what()<<'\n';std::lock_guard<std::mutex> lock(mutex_);active_.reset();continue;}
                if(protocol_=="gtp")output(r.streaming?"error "+std::string(error.what())+"\n\n":"?"+r.gtp_id+" "+error.what()+"\n\n");
                else {Json reply={{"ok",false},{"error",error.what()}};if(protocol_=="json") {
                    reply["id"]=r.id;reply["event"]="result";reply["final"]=true;
                    std::lock_guard<std::mutex> lock(mutex_);pending_ids_.erase(r.id.dump());
                }emit(reply);}
            }
            {std::lock_guard<std::mutex> lock(mutex_);active_.reset();}
        }
    }
public:
    Session(GameSearch& search,InferenceService& eval,const Config& c,uint64_t seed,std::string protocol,
            std::function<Json(const Game&)> raw,std::function<double(const Game&)> temperature,std::function<bool()> interrupted)
        :search_(search),evaluator_(eval),config_(c),seed_(seed),protocol_(std::move(protocol)),canvas_(c.integer("network.canvas")),
         raw_(std::move(raw)),temperature_(std::move(temperature)),interrupted_(std::move(interrupted)) {}
    int run() {
        if(protocol_=="gtp") {Request r;create(Json::object(),r);}
        else emit({{"ok",true},{"event","ready"},{"canvas_size",canvas_}});
        std::thread worker(&Session::work,this);
        for(std::string line;read_line(line);) {
            Request r;
            try {
                if(protocol_=="json") {
                    r.data=Json::parse(line);if(!r.data.is_object())throw std::runtime_error("Request must be an object");
                    r.id=r.data.at("id");if(!r.id.is_string()&&!r.id.is_number_integer())throw std::runtime_error("id must be a string or integer");
                    r.command=r.data.at("command").get<std::string>();
                    const std::set<std::string> allowed={"id","command","size","rule","moves","action","count","visits","report_interval_ms","fresh","opening","seed","timeout_ms","target_id","continuous","fen","komi","go_rules","avoid_moves","allow_moves","avoid_rescale_root","focus_moves","focus_weights","focus_probability","clock_type","clock_arguments","player","time","stones"};
                    for(auto it=r.data.begin();it!=r.data.end();++it)if(!allowed.count(it.key()))throw std::runtime_error("Unsupported request field: "+it.key());
                    r.continuous=r.data.value("continuous",false);
                    if(r.continuous&&r.command!="analyze")throw std::runtime_error("continuous requires analyze");
                } else {
                    auto parts=words(line);if(parts.empty())continue;size_t index=0;
                    if(protocol_=="gtp"&&std::all_of(parts[0].begin(),parts[0].end(),[](unsigned char c){return std::isdigit(c);}))r.gtp_id=parts[index++];
                    if(index==parts.size())throw std::runtime_error("Missing command");r.command=parts[index++];r.args.assign(parts.begin()+index,parts.end());
                    r.continuous=protocol_=="gtp"&&r.command=="kata-analyze";
                    if(protocol_=="text") {
                        auto& a=r.args;
                        if(r.command=="new"&&(a.size()==2||a.size()==5)) {r.data={{"size",number(a[0])},{"rule",a[1]}};if(a.size()==5)r.data.update({{"opening",a[2]},{"seed",a[3]},{"timeout_ms",number(a[4])}});}
                        else if(r.command=="play"&&a.size()==1)r.data["action"]=number(a[0]);
                        else if(r.command=="undo"&&a.size()==1)r.data["count"]=number(a[0]);
                        else if((r.command=="analyze"||r.command=="genmove")&&a.size()==1){size_t used=0;int64_t count=std::stoll(a[0],&used);if(used!=a[0].size())throw std::runtime_error("Invalid visit count");r.data["visits"]=count;}
                        else if(!a.empty()||!(r.command=="state"||r.command=="quit"||r.command=="stop"))throw std::runtime_error("Unknown command or arguments");
                    }
                }
                if(protocol_=="json" && r.command=="set_focus") {
                    {std::lock_guard<std::mutex> lock(mutex_);
                        if(!active_ || (active_->command!="analyze" && active_->command!="genmove") ||
                           !r.data.contains("target_id") || r.data.at("target_id")!=active_->id)
                            throw std::runtime_error("set_focus requires the active search target_id");
                        set_focus(r.data);
                        active_->data["focus_moves"]=r.data.value("focus_moves",Json::array());
                        active_->data["focus_weights"]=r.data.value("focus_weights",Json(std::vector<double>(active_->data["focus_moves"].size(),1)));
                        active_->data["focus_probability"]=r.data.value("focus_probability",.5);

                    }
                    emit({{"id",r.id},{"ok",true},{"event","result"},{"final",true}});continue;
                }
                bool quitting=r.command=="quit";
                {std::lock_guard<std::mutex> lock(mutex_);
                    if(protocol_=="json"&&!pending_ids_.insert(r.id.dump()).second)throw std::runtime_error("Duplicate pending request id");
                    if(active_ && active_->command=="ponder")active_->cancelled->store(true);
                    if(r.command=="stop"||r.command=="quit")want_ponder_=false;
                    bool mutation=r.command=="new"||r.command=="play"||r.command=="undo"||r.command=="boardsize"||r.command=="clear_board"||r.command=="etazero-rule"||r.command=="komi";
                    if(r.command=="stop"||mutation||(protocol_=="gtp"&&active_&&active_->continuous)) {
                        auto cancel=[&](Request& job){if(!r.data.contains("target_id")||job.id==r.data["target_id"])job.cancelled->store(true);};
                        if(active_)cancel(*active_);for(auto& job:queue_)if(r.command=="stop"||job.continuous)cancel(job);
                    }
                    if(protocol_=="gtp")for(auto& job:queue_)if(job.continuous)job.cancelled->store(true);
                    queue_.push_back(std::move(r));
                }
                changed_.notify_all();if(quitting)break;
            } catch(const std::exception& error) {
                if(protocol_=="gtp")output("?"+r.gtp_id+" "+error.what()+"\n\n");
                else {Json response={{"ok",false},{"error",error.what()},{"id",r.id},{"final",true},{"event","result"}};emit(response);}
            }
        }
        {std::lock_guard<std::mutex> lock(mutex_);eof_=true;if(active_&&(active_->continuous||active_->command=="ponder"))active_->cancelled->store(true);for(auto& r:queue_)if(r.continuous)r.cancelled->store(true);}
        changed_.notify_all();worker.join();return interrupted_()?130:0;
    }
};
}
Json analysis_json(const SearchResult& result,const Game& game,uint64_t requests,uint64_t batches,bool final) {
    auto local=[&](int a){return game.local_action(a);};
    Json out={{"action",local(result.action)},{"board_width",game.width()},{"board_height",game.height()},{"board_stride",game.geometry().stride},{"board_size",game.size()},{"canvas_size",game.canvas()},
        {"turn",game.turn()},{"player",game.player()},{"rule",int(game.rule())},{"board",game.board().cells},
        {"root_value",result.value},{"wdl",result.search_wdl},{"network_wdl",result.network_wdl},
        {"completed_visits",result.root_visits},{"seconds",result.seconds},{"requests",requests},{"batches",batches},
        {"root_search_algo",result.gumbel?"gumbel":"puct"},{"final",final},{"stopped_early",result.stopped_early},
        {"candidates",Json::array()}};
    int64_t total=std::accumulate(result.visits.begin(),result.visits.end(),int64_t(0));
    for(int a=0;a<game.actions();++a)if(game.legal(a)) {
        const AnalysisCandidate* data=nullptr;for(const auto& c:result.analysis)if(c.action==a){data=&c;break;}
        Json c={{"action",local(a)},{"visits",result.visits.at(a)},{"network_prior",result.network_policy.at(a)},
            {"visit_policy",total?double(result.visits.at(a))/total:0},{"selection_weight",result.move_policy.at(a)},
            {"improved_policy",result.policy.at(a)},{"sample_weight",data?data->weight:0},{"raw_selection_weight",nullptr},
            {"wdl",nullptr},{"winrate",nullptr},{"draw_rate",nullptr},{"win_loss",nullptr},{"utility",nullptr},
            {"lcb",nullptr},{"radius",nullptr},{"pv",Json::array()}};
        if(result.gumbel&&!final)c["selection_weight"]=nullptr;
        if(!result.gumbel)c["raw_selection_weight"]=data?data->selection.weight:0;
        if(data&&data->has_value) {
            c["wdl"]=data->wdl;c["winrate"]=(1+data->win_loss)*.5;c["draw_rate"]=data->wdl[1];c["win_loss"]=data->win_loss;c["utility"]=data->utility;
            if(data->selection.has_lcb){c["lcb"]=data->selection.lcb;c["radius"]=data->selection.radius;}
        }
        // Latent PVs may repeat occupied moves. Publish only the real-board legal prefix.
        if(data){Game pv=game;for(int move:data->pv){if(!pv.legal(move))break;c["pv"].push_back(local(move));pv.play(move);}}
        out["candidates"].push_back(std::move(c));
    }
    auto& candidates=out["candidates"];
    std::stable_sort(candidates.begin(),candidates.end(),[&](const Json& a,const Json& b){
        const char* field=result.gumbel?"improved_policy":"selection_weight";
        double x=a.at(field).get<double>(),y=b.at(field).get<double>();return x!=y?x>y:a.at("visits").get<int64_t>()>b.at("visits").get<int64_t>();
    });
    for(size_t i=0;i<candidates.size();++i)candidates[i]["rank"]=i+1;
    return out;
}
int engine_session(GameSearch& search,InferenceService& eval,const Config& config,uint64_t seed,const std::string& protocol,
                   const std::function<Json(const Game&)>& raw,const std::function<double(const Game&)>& temperature,const std::function<bool()>& interrupted) {
    if(protocol!="text"&&protocol!="json"&&protocol!="gtp")throw std::runtime_error("Unknown session protocol: "+protocol);
    return Session(search,eval,config,seed,protocol,raw,temperature,interrupted).run();
}
}
