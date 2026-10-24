#include "etazero/game_config.h"
#include "etazero/session.h"
#include "etazero/go_position.h"
#include "etazero/source_config.h"
#include "../../external/katago/program/playutils.h"
#include <fstream>
#include "benchmark_sgf.h"
#include <iomanip>
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
    if(a==g.area())return "pass";
    return std::string(1,columns.at(a%g.width()))+std::to_string(g.height()-a/g.width());
}
int action(const std::string& v,const Game& g) {
    if(g.rule()==Rule::GO && (v=="pass"||v=="PASS"))return g.area();
    if(g.rule()==Rule::XIANGQI){
        if(v.size()!=4)throw std::runtime_error("Expected Xiangqi move such as h2e2");
        int a=xiangqi_action(v[0]-'a',v[1]-'0',v[2]-'a',v[3]-'0');
        if(a<0)throw std::runtime_error("Unknown Xiangqi move");return a;
    }
    if(v.size()<2)throw std::runtime_error("Expected board coordinate");
    int width=g.width(),height=g.height();auto x=columns.find(std::toupper(static_cast<unsigned char>(v[0])));int y=number(v.substr(1));
    if(x>=size_t(width)||y<1||y>height)throw std::runtime_error("Move outside board");
    return (height-y)*width+int(x);
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
    "boardsize","rectangular_boardsize","set_position","clear_board","play","undo","genmove","quit","kata-analyze","kata-genmove_analyze",
    "stop","kata-raw-nn","kata-raw-human-nn","etazero-rule","komi","showboard","time_settings","time_left","kgs-time_settings","kata-time_settings","kata-list_time_settings",
    "get_komi","kata-get-rules","kata-set-rule","kata-set-rules","kgs-rules",
    "kata-list-params","kata-get-param","kata-set-param","kata-get-params","kata-set-params","kata-get-models",
    "fixed_handicap","place_free_handicap","set_free_handicap","loadsgf","printsgf","final_score","final_status_list",
    "clear_cache","kata-debug-print-tc","cputime","gomill-cpu_time","debug_moves","kata-benchmark",
    "genmove_debug","kata-search","kata-search_debug","kata-search_cancellable",
    "genmove_analyze","lz-genmove_analyze","kata-search_analyze","kata-search_analyze_cancellable","analyze","lz-analyze"};
class Session {
    std::unique_ptr<GameSearch> search_;InferenceService& evaluator_;Config config_;
    Json models_,overrides_=Json::object();std::function<std::unique_ptr<GameSearch>(const Json&)> factory_;
    std::optional<::BoardHistory> go_initial_;std::vector<Move> go_moves_;double genmove_seconds_=0;
    std::optional<SearchResult> last_result_;
    std::mt19937_64 gtp_random_{0};std::vector<double> recent_white_values_;
    uint64_t seed_;
    std::string protocol_;int canvas_;std::optional<Game> game_;std::vector<int> history_;Json opening_=nullptr;
    std::vector<std::vector<int>> boards_;std::vector<int> players_;
    std::function<Json(const Game&)> raw_;std::function<double(const Game&,const Config&)> temperature_;std::function<bool()> interrupted_;
    std::mutex mutex_,output_mutex_;std::condition_variable changed_;std::deque<Request> queue_;std::optional<Request> active_;bool eof_=false;
    std::array<TimeControls,2> clocks_;bool want_ponder_=false;int search_for_player_=0;
    std::set<std::string> pending_ids_;
    std::string input_buffer_;bool input_closed_=false;
    std::vector<std::string> auto_pattern_samples_;bool reload_auto_patterns_=false;
    std::shared_ptr<const PatternBonusTable> static_patterns_;
    ::BoardHistoryModes history_modes(const Config& config) const {
        return resolve_go_modes(config.contains("analysis.always_compute_pass_alive_under_suicide_rules")?config.text("analysis.always_compute_pass_alive_under_suicide_rules"):"auto",config.contains("analysis.exclude_territory_adjacent_to_atari")?config.text("analysis.exclude_territory_adjacent_to_atari"):"auto",evaluator_.description());
    }
    bool analysis_mode_=false;
    void select_search_mode(bool analysis) {
        if(protocol_!="gtp"||analysis_mode_==analysis)return;
        double wide=config_.contains("analysis.analysis_wide_root_noise")?config_.number("analysis.analysis_wide_root_noise"):.04;bool ignore=config_.contains("analysis.analysis_ignore_pre_root_history")&&config_.boolean("analysis.analysis_ignore_pre_root_history");
        if(wide==config_.number("analysis.wide_root_noise")&&ignore==config_.boolean("analysis.ignore_pre_root_history")){analysis_mode_=analysis;return;}
        auto overrides=overrides_;
        if(analysis){overrides["wide_root_noise"]=config_.contains("analysis.analysis_wide_root_noise")?config_.number("analysis.analysis_wide_root_noise"):.04;overrides["ignore_pre_root_history"]=config_.contains("analysis.analysis_ignore_pre_root_history")&&config_.boolean("analysis.analysis_ignore_pre_root_history");}
        search_=factory_(overrides);analysis_mode_=analysis;last_result_.reset();if(automatic_patterns())reload_patterns();
    }
    bool automatic_patterns() const {return config_.contains("analysis.auto_avoid_repeat_dir")&&!config_.text("analysis.auto_avoid_repeat_dir").empty();}
    void reload_patterns() {
        auto table=std::make_shared<PatternBonusTable>();if(static_patterns_)table->merge(*static_patterns_);
        auto automatic=PatternBonusTable::load_auto(config_,"analysis",canvas_,rule_options(config_,GameConfigMode::ANALYSIS));
        if(automatic)table->merge(*automatic);search_->set_external_patterns(std::move(table));
    }
    void save_patterns(bool force) {
        if(!automatic_patterns())return;
        if(force||auto_pattern_samples_.size()>=size_t(config_.integer("analysis.auto_avoid_repeat_save_chunk_size"))) {
            if(PatternBonusTable::save_auto(auto_pattern_samples_,config_,"analysis")){auto_pattern_samples_.clear();reload_auto_patterns_=true;}
        }
    }
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
        int size=integer(data,"size",config_.integer("analysis.board_size"),evaluator_.spec().kind==GameKind::GO?2:5,canvas_);
        auto options=rule_options(config_,GameConfigMode::ANALYSIS);options.fen=data.value("fen",std::string());
        options.go_width=data.value("width",config_.contains("analysis.board_width")&&config_.integer("analysis.board_width")?config_.integer("analysis.board_width"):size);options.go_height=data.value("height",config_.contains("analysis.board_height")&&config_.integer("analysis.board_height")?config_.integer("analysis.board_height"):size);
        size=std::max(options.go_width,options.go_height);
        auto modes=history_modes(config_);options.go_pass_alive=modes.alwaysComputePassAliveUnderSuicideRules;options.go_exclude_atari=modes.excludeTerritoryAdjacentToAtari;
        options.komi=data.value("komi",options.komi);options.go_rules=data.value("go_rules",options.go_rules);
        Game replacement(size,canvas_,parse_rule(data.value("rule",config_.text("analysis.rule"))),options);
        replacement.set_pda(config_.number("analysis.playout_doubling_advantage"),config_.text("analysis.playout_doubling_advantage_player")=="black"?1:-1);
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
        replacement.set_pda(config_.number("analysis.playout_doubling_advantage"),config_.text("analysis.playout_doubling_advantage_player")=="black"?1:-1);
        if(replacement.rule()==Rule::GO){go_initial_=replacement.go_history();go_initial_->clear(replacement.go_history().initialBoard,replacement.go_history().initialPla,replacement.go_history().rules,0);go_initial_->setInitialTurnNumber(replacement.go_history().initialTurnNumber);go_moves_=replacement.go_history().moveHistory;}else{go_initial_.reset();go_moves_.clear();}
        last_result_.reset();recent_white_values_.clear();game_=std::move(replacement);history_=std::move(moves);opening_=std::move(opening);search_->reset(seed_);
        boards_.clear();players_.clear();
        if(game_->rule()==Rule::GO||game_->rule()==Rule::XIANGQI){Game replay(size,canvas_,game_->rule(),options);
            boards_.push_back(replay.board().cells);players_.push_back(replay.player());
            for(int a:history_){replay.play(replay.native_action(a));boards_.push_back(replay.board().cells);players_.push_back(replay.player());}}
    }
    void play(int local) {if(local<0||local>=game_->local_actions())throw std::runtime_error("Move outside board");int a=native(local);Player previous=game_->player()==1?P_BLACK:P_WHITE;game_->play(a);if(go_initial_)go_moves_.emplace_back(a==canvas_*canvas_?::Board::PASS_LOC:Location::getLoc(a%canvas_,a/canvas_,game_->width()),previous);history_.push_back(local);search_->advance(a);
        if(!boards_.empty()){boards_.push_back(game_->board().cells);players_.push_back(game_->player());}}
    void undo(int count) {
        if(count<1||count>int(protocol_=="gtp"&&go_initial_?go_moves_.size():history_.size()))throw std::runtime_error("Invalid undo count");
        if(protocol_=="gtp"&&go_initial_){if(count>int(go_moves_.size()))throw std::runtime_error("cannot undo");auto moves=go_moves_;moves.resize(moves.size()-count);auto h=replay_go(go_initial_->rules,moves);go_moves_=std::move(moves);install_go(h,false);return;}
        Game replacement(game_->size(),canvas_,game_->rule(),game_->options());auto retained=history_;retained.resize(retained.size()-count);
        for(int a:retained)replacement.play(replacement.native_action(a));
        game_=std::move(replacement);history_=std::move(retained);search_->reset(seed_);
        if(!boards_.empty()){boards_.resize(history_.size()+1);players_.resize(history_.size()+1);}
        if(!opening_.is_null()&&history_.size()<opening_.at("moves").size())opening_=nullptr;
    }
    static void tolerant_go_move(::BoardHistory& h,::Board& b,Loc loc,Player pla) {
        if(!h.isLegalTolerant(b,loc,pla))throw std::runtime_error("illegal move");
        if(pla!=h.presumedNextMovePla){b.clearSimpleKoLoc();h.clear(b,pla,h.rules,h.encorePhase);h.setAssumeMultipleStartingBlackMovesAreHandicap(true);}
        h.makeBoardMoveAssumeLegal(b,loc,pla,nullptr);
    }
    ::BoardHistory replay_go(const ::Rules& rules,const std::vector<Move>& moves,std::optional<::BoardHistoryModes> modes={}) const {
        auto b=go_initial_->initialBoard;::BoardHistory h(b,go_initial_->initialPla,rules,0,modes.value_or(go_initial_->modes));
        h.setInitialTurnNumber(go_initial_->initialTurnNumber);h.setAssumeMultipleStartingBlackMovesAreHandicap(true);
        for(auto m:moves)tolerant_go_move(h,b,m.loc,m.pla);return h;
    }
    void install_go(const ::BoardHistory& h,bool reset_record=true) {
        Game g=game_from_go_history(h,canvas_);
        g.set_pda(config_.number("analysis.playout_doubling_advantage"),config_.text("analysis.playout_doubling_advantage_player")=="black"?1:-1);
        if(reset_record){go_initial_=h;go_initial_->clear(h.initialBoard,h.initialPla,h.rules,h.initialEncorePhase);go_initial_->setInitialTurnNumber(h.initialTurnNumber);go_moves_=h.moveHistory;}
        game_=std::move(g);history_.clear();for(auto m:go_moves_)history_.push_back(game_->local_action(go_action(m.loc,game_->width(),canvas_)));
        boards_={game_->board().cells};players_={game_->player()};opening_=nullptr;search_->reset(seed_);last_result_.reset();
    }
    void go_player(int side) {
        if(side==game_->player())return;
        if(game_->rule()!=Rule::GO)throw std::runtime_error("Color is not side to move");
        auto h=game_->go_history();auto b=h.getRecentBoard(0);b.clearSimpleKoLoc();
        h.clear(b,side==1?P_BLACK:P_WHITE,h.rules,h.encorePhase);h.setAssumeMultipleStartingBlackMovesAreHandicap(true);install_go(h,false);
    }
    void go_play(int side,int local) {
        auto h=game_->go_history();auto b=h.getRecentBoard(0);int a=game_->native_action(local);
        Loc loc=a==canvas_*canvas_?::Board::PASS_LOC:Location::getLoc(a%canvas_,a/canvas_,b.x_size);Player pla=side==1?P_BLACK:P_WHITE;
        tolerant_go_move(h,b,loc,pla);bool can_advance=side==game_->player()&&!game_->finished()&&game_->legal(a);
        go_moves_.emplace_back(loc,pla);
        if(can_advance){game_->play(a);history_.push_back(local);search_->advance(a);}
        else install_go(h,false);
    }
    std::string go_command(const Request& r) {
        const auto& cmd=r.command;const auto& args=r.args;auto h=game_->go_history();auto b=h.getRecentBoard(0);
        auto count=[&](size_t n){if(args.size()!=n)throw std::runtime_error("Unexpected command arguments");};
        auto joined=[&]{std::string out;for(auto& a:args){if(!out.empty())out+=' ';out+=a;}return out;};
        if(cmd=="showboard"){count(0);std::ostringstream out;h.printBasicInfo(out,b);return trim(out.str());}
        if(cmd=="get_komi"){count(0);return Global::doubleToString(h.rules.komi);}
        if(cmd=="kata-get-rules"){count(0);return h.rules.toJsonStringNoKomi();}
        if(cmd=="komi") {
            count(1);size_t used=0;double komi=std::stod(args[0],&used);
            if(used!=args[0].size()||!std::isfinite(komi)||komi< -400||komi>400||!::Rules::komiIsIntOrHalfInt(komi))throw std::runtime_error("komi must be an integer or half-integer in [-400,400]");
            h.setKomi(komi);go_initial_->setKomi(komi);install_go(h,false);return {};
        }
        if(cmd=="kata-set-rules"||cmd=="kata-set-rule"||cmd=="kgs-rules") {
            ::Rules rules=h.rules;
            if(cmd=="kata-set-rule"){count(2);rules=::Rules::updateRules(args[0],args[1],rules);}
            else {auto value=joined();if(cmd=="kgs-rules"){count(1);if(value=="chinese")value="chinese-kgs";else if(value!="aga"&&value!="new_zealand"&&value!="japanese")throw std::runtime_error("Unknown KGS rules");}rules=::Rules::parseRulesWithoutKomi(value,rules.komi);}
            rules.komi=h.rules.komi;auto next=replay_go(rules,go_moves_);install_go(next,false);go_initial_->rules=rules;return {};
        }
        if(cmd=="loadsgf") {
            if(args.empty()||args.size()>2)throw std::runtime_error("Expected filename [moveNumber]");
            auto sgf=CompactSgf::loadFile(args[0]);int64_t turns=sgf->moves.size();
            if(args.size()==2){int n=number(args[1]);if(n<1||n>10000001)throw std::runtime_error("Invalid moveNumber");turns=std::min<int64_t>(turns,n-1);}
            if(sgf->xSize>canvas_||sgf->ySize>canvas_)throw std::runtime_error("SGF exceeds model canvas");
            auto rules=sgf->getRulesOrWarn(h.rules,[](const std::string& s){std::cerr<<s<<'\n';});Player next;
            sgf->setupInitialBoardAndHist(rules,b,next,h,::BoardHistoryModes(true,true));h.setInitialTurnNumber(b.numStonesOnBoard());
            auto initial=h;sgf->playMovesTolerant(b,next,h,turns,false);install_go(h);go_initial_=initial;go_moves_.assign(sgf->moves.begin(),sgf->moves.begin()+turns);return {};
        }
        if(cmd=="printsgf") {
            if(args.size()>1)throw std::runtime_error("Expected [filename]");
            std::ostringstream out;WriteSgf::writeSgf(out,"","",h,nullptr,true,false);
            if(args.empty()||args[0]=="-")return out.str();std::ofstream file(args[0]);file<<out.str();file.close();if(!file)throw std::runtime_error("Cannot write SGF: "+args[0]);return {};
        }
        if(cmd=="final_score"||cmd=="final_status_list") {
            count(cmd=="final_score"?0:1);
            if(!h.isGameFinished)throw std::runtime_error("Unfinished position requires excluded score/ownership estimation; complete the game first");
            if(cmd=="final_score"){if(h.winner==C_EMPTY)return "0";return std::string(h.winner==P_BLACK?"B+":"W+")+Global::strprintf("%.1f",std::abs(h.finalWhiteMinusBlackScore));}
            if(args[0]!="alive"&&args[0]!="dead"&&args[0]!="seki")throw std::runtime_error("Expected alive, dead or seki");
            if(args[0]=="seki")return {}; // Source GTP also reports no separate seki list.
            Color area[::Board::MAX_ARR_SIZE];auto scored=h;scored.endAndScoreGameNow(b,area);std::string out;
            for(int y=0;y<b.y_size;++y)for(int x=0;x<b.x_size;++x){auto loc=Location::getLoc(x,y,b.x_size);bool alive=h.isNoResult||b.colors[loc]==area[loc];
                if(b.colors[loc]!=C_EMPTY&&alive==(args[0]=="alive")){if(!out.empty())out+=' ';out+=Location::toString(loc,b);}}
            return out;
        }
        if(cmd=="set_position"||cmd=="set_free_handicap"||cmd=="fixed_handicap"||cmd=="place_free_handicap") {
            if(cmd!="set_position"&&!b.isEmpty())throw std::runtime_error("Board is not empty");b=::Board(b.x_size,b.y_size);
            if(cmd=="fixed_handicap"){count(1);PlayUtils::placeFixedHandicap(b,number(args[0]));}
            else if(cmd=="place_free_handicap") {
                count(1);int n=number(args[0]);if(n<2)throw std::runtime_error("Handicap must be at least 2");n=std::min(n,std::min(30,b.x_size*b.y_size/4));
                std::mt19937_64 random(seed_++);::BoardHistory initial(b,P_BLACK,h.rules,0,h.modes);
                for(int i=0;i<n;++i){auto g=game_from_go_history(initial,canvas_);std::vector<int> legal;for(int a:g.static_actions())if(a!=g.canvas_area()&&g.legal(a))legal.push_back(a);if(legal.empty())break;
                    auto value=evaluator_.evaluate_position(g.observation(),1);auto policy=source_policy_probabilities(value,legal,1,0);std::vector<double> weights;for(int a:legal)weights.push_back(std::pow(policy[a],4));
                    int chosen=legal[std::discrete_distribution<size_t>(weights.begin(),weights.end())(random)];auto loc=Location::getLoc(chosen%canvas_,chosen/canvas_,b.x_size);
                    initial.makeBoardMoveAssumeLegal(b,loc,P_BLACK,nullptr);initial.clear(b,P_BLACK,h.rules,0);}
            } else {
                if(cmd=="set_position"&&args.size()%2)throw std::runtime_error("Expected color vertex pairs");std::vector<Move> placements;
                for(size_t i=0;i<args.size();){Player pla=P_BLACK;if(cmd=="set_position")pla=color(args[i++])==1?P_BLACK:P_WHITE;
                    Loc loc;if(!Location::tryOfString(args[i++],b,loc)||loc==::Board::PASS_LOC||loc==::Board::NULL_LOC)throw std::runtime_error("Invalid placement vertex");placements.emplace_back(loc,pla);}
                if(!b.setStonesFailIfNoLibs(placements))throw std::runtime_error("Illegal stone placements");for(auto m:placements)if(b.colors[m.loc]!=m.pla)throw std::runtime_error("Overlapping placements");
            }
            ::BoardHistory initial(b,cmd=="set_position"?P_BLACK:P_WHITE,h.rules,0,h.modes);initial.setInitialTurnNumber(b.numStonesOnBoard());initial.setAssumeMultipleStartingBlackMovesAreHandicap(true);install_go(initial);
            std::string out;if(cmd=="fixed_handicap"||cmd=="place_free_handicap")for(int y=0;y<b.y_size;++y)for(int x=0;x<b.x_size;++x){auto loc=Location::getLoc(x,y,b.x_size);if(b.colors[loc]!=C_EMPTY){if(!out.empty())out+=' ';out+=Location::toString(loc,b);}}return out;
        }
        throw std::runtime_error("Unknown Go command");
    }
    Json parameters() const {
        Json result=Json::object();for(const auto& [alias,key]:SOURCE_SEARCH_KEYS)if(alias.find('_')==std::string::npos&&config_.contains("analysis."+key)){
            const auto name="analysis."+key;auto type=SOURCE_SEARCH_TYPES.at(key);
            result[alias]=type=='b'?Json(config_.boolean(name)):type=='i'?Json(config_.integer64(name)):type=='f'?Json(config_.number(name)):Json(config_.text(name));}
        for(const auto& key:{"pondering_enabled","allow_resignation","resign_threshold","resign_consec_turns","delay_move_scale","delay_move_max","human_sl_profile"})if(config_.contains(std::string("analysis.")+key)) {
            std::string alias;bool upper=false;for(char c:std::string(key)){if(c=='_'){upper=true;continue;}alias+=upper?std::toupper(c):c;upper=false;}
            auto text=config_.text(std::string("analysis.")+key);if(std::string(key)=="human_sl_profile")result["humanSLProfile"]=text;else result[alias]=Json::parse(text);
        }
        result["alwaysComputePassAliveUnderSuicideRules"]=config_.contains("analysis.always_compute_pass_alive_under_suicide_rules")?config_.text("analysis.always_compute_pass_alive_under_suicide_rules"):"auto";result["excludeTerritoryAdjacentToAtari"]=config_.contains("analysis.exclude_territory_adjacent_to_atari")?config_.text("analysis.exclude_territory_adjacent_to_atari"):"auto";
        result["analysisWideRootNoise"]=config_.contains("analysis.analysis_wide_root_noise")?config_.number("analysis.analysis_wide_root_noise"):.04;result["analysisIgnorePreRootHistory"]=config_.contains("analysis.analysis_ignore_pre_root_history")&&config_.boolean("analysis.analysis_ignore_pre_root_history");return result;
    }

    std::string infos(const Json& analysis,const Request& r) {
        int maxmoves=integer(r.data,"maxmoves",10000000,0,999999999),minimum=integer(r.data,"minmoves",0,0,999999999),count=0;bool lz=r.data.value("lz",false);std::ostringstream out;out.precision(17);
        auto candidates=analysis.at("candidates");
        if(game_->rule()==Rule::GO&&config_.boolean("analysis.root_symmetry_pruning")) {
            auto h=game_->go_history();auto b=h.getRecentBoard(0);std::vector<int> avoid(::Board::MAX_ARR_SIZE,0);bool duplicate[::Board::MAX_ARR_SIZE];std::vector<int> symmetries;
            SearchRun limits;restrictions(r.data,limits);auto& depths=limits.avoid_until[game_->player()==1?0:1];
            if(!depths.empty())for(int a:game_->static_actions())avoid[a==canvas_*canvas_?::Board::PASS_LOC:Location::getLoc(a%canvas_,a/canvas_,b.x_size)]=depths[a];
            SymmetryHelpers::markDuplicateMoveLocs(b,h,nullptr,avoid,duplicate,symmetries);
            Json expanded=Json::array();std::set<int> seen;
            auto transformed=[&](int local,int sym){int a=game_->native_action(local);auto loc=a==canvas_*canvas_?::Board::PASS_LOC:Location::getLoc(a%canvas_,a/canvas_,b.x_size);return game_->local_action(go_action(SymmetryHelpers::getSymLoc(loc,b,sym),b.x_size,canvas_));};
            for(const auto& c:candidates)if(c.value("search_allowed",false))for(int sym:symmetries){int a=transformed(c["action"],sym);if(!seen.insert(a).second)continue;auto copy=c;copy["action"]=a;if(a!=c["action"].get<int>())copy["isSymmetryOf"]=c["action"];for(auto& move:copy["pv"])move=transformed(move,sym);expanded.push_back(std::move(copy));}candidates=std::move(expanded);
        }
        int perspective=game_->player();if(config_.contains("analysis.report_analysis_winrates_as")){auto value=config_.text("analysis.report_analysis_winrates_as");for(auto& c:value)c=std::toupper(c);if(value=="BLACK")perspective=1;else if(value=="WHITE")perspective=-1;}double sign=perspective==game_->player()?1:-1;
        for(const auto& c:candidates) {
            if(!c.value("search_allowed",false))continue;
            if((c.at("visits").get<int64_t>()<=0&&count>=minimum)||count>=maxmoves)continue;if(count++)out<<' ';
            double utility=c["utility"].is_null()?c.value("fpu_utility",0.0):c["utility"].get<double>();
            double winrate=c["winrate"].is_null()?.5+.5*c.value("fpu_win_loss",0.0):c["winrate"].get<double>();double prior=c["network_prior"];utility*=sign;winrate=.5+sign*(winrate-.5);
            out<<"info move "<<vertex(c.at("action"),*game_)<<" visits "<<c.at("node_visits");if(!lz)out<<" edgeVisits "<<c.at("visits");out<<" winrate ";
            if(lz)out<<std::lround(winrate*10000);else out<<winrate;
            out<<" prior ";if(lz)out<<std::lround(prior*10000);else out<<prior;
            out<<" order "<<count-1;
            if(!lz){out<<" utility "<<utility<<" drawRate "<<(c["draw_rate"].is_null()?0:c["draw_rate"].get<double>())<<" weight "<<c.at("node_weight");if(c.contains("isSymmetryOf"))out<<" isSymmetryOf "<<vertex(c["isSymmetryOf"],*game_);if(r.data.value("noResultValue",false))out<<" noResultValue "<<(c["wdl"].is_null()?0:c["wdl"][3].get<double>());}
            if(!c.at("lcb").is_null()){double lower=.5+.5*(2*winrate-1-sign*c["radius"].get<double>());out<<" lcb ";if(lz)out<<std::lround(lower*10000);else out<<lower<<" utilityLcb "<<sign*c.at("lcb").get<double>();}
            out<<" pv";for(const auto& a:c.at("pv"))out<<' '<<vertex(a,*game_);
            for(auto key:{"pvVisits","pvEdgeVisits"})if(!lz&&r.data.value(key,false)){out<<' '<<key;for(const auto& n:c.at(key))out<<' '<<n;}
        }
        if(r.data.value("rootInfo",false))out<<" rootInfo visits "<<analysis["completed_visits"]<<" utility "<<sign*analysis["pruned_utility"].get<double>()<<" winrate "<<.5+.5*sign*analysis["pruned_value"].get<double>()<<" weight "<<analysis["root_weight"];

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
        search_->set_root_focus(std::move(moves),std::move(weights),probability);
    }
    void restrictions(const Json& data,SearchRun& options) {
        bool seen_allow[2]={false,false},seen_avoid[2]={false,false};
        for(const auto* key:{"allow_moves","avoid_moves"})if(data.contains(key)) {
            bool allow=std::string(key)=="allow_moves";const auto& entries=data.at(key);if(!entries.is_array())throw std::runtime_error("Move restrictions must be an array");
            for(const auto& entry:entries){int player=entry.at("player").is_string()?color(entry.at("player").get<std::string>()):entry.at("player").get<int>();
                if(player!=1&&player!=-1)throw std::runtime_error("Restriction player must be black or white");int index=player==1?0:1,depth=integer(entry,"until_depth",-1,1,1000000000);
                if(depth<1||seen_allow[index]||(allow&&seen_avoid[index]))throw std::runtime_error("Conflicting restrictions for player");if(allow)seen_allow[index]=true;else seen_avoid[index]=true;
                auto moves=move_list(entry.at("moves"));auto& avoid=options.avoid_until[index];avoid.resize(game_->actions(),0);if(allow)std::fill(avoid.begin(),avoid.end(),depth);for(int a:moves)avoid[a]=allow?0:depth;
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
        if(!game_ || game_->finished())return;select_search_mode(false);
        SearchRun options;options.pondering=true;options.search_for_player=search_for_player_?search_for_player_:-game_->player();
        options.should_stop=[&]{return r.cancelled->load()||interrupted_();};
        auto result=search_->run(*game_,temperature_(*game_,config_),options);
        if(!r.internal)reply(r,{{"state",state()},{"analysis",analysis_json(result,*game_,0,0,true)}});
    }
    Json analyze(Request& r) {
        if(game_->finished()&&(protocol_!="gtp"||game_->rule()!=Rule::GO))throw std::runtime_error("Game has finished");
        select_search_mode(r.command=="kata-analyze");
        int64_t visits=integer(r.data,"visits",config_.integer64("analysis.visits"),1,SEARCH_CONFIG_LIMIT);
        double interval=r.data.value("report_interval_ms",protocol_=="text"?0.0:250.0);
        if(!std::isfinite(interval)||interval<0||interval>TimeControls::MAX_USER_INPUT_TIME*10)throw std::runtime_error("Invalid report interval");
        SearchRun options;restrictions(r.data,options);{std::lock_guard<std::mutex> lock(mutex_);set_focus(active_->data);}
        options.analysis_pv_length=config_.contains("analysis.analysis_pv_len")?config_.integer("analysis.analysis_pv_len"):13;
        options.max_visits=r.continuous?SEARCH_CONFIG_LIMIT:visits;
        const bool generate_move=r.command=="genmove"||r.command=="kata-genmove_analyze";
        if(generate_move){options.time_controls=&clocks_[game_->player()==1?0:1];options.search_for_player=game_->player();}
        options.clear_before_search=r.data.value("fresh",true);options.collect_analysis=true;
        options.should_stop=[&]{return r.cancelled->load()||interrupted_();};
        if(r.continuous){options.max_playouts=SEARCH_CONFIG_LIMIT;options.max_time=1e20;}
        auto requests=evaluator_.requests.load(),batches=evaluator_.batches.load();Game before=*game_;
        if(before.finished()){auto h=before.go_history();h.isGameFinished=false;h.isNoResult=false;h.winner=C_EMPTY;before=game_from_go_history(h,canvas_);before.set_pda(game_->pda_doublings(),game_->pda_player());}
        const bool gtp_stream=protocol_=="gtp"&&(r.command=="kata-analyze"||r.command=="kata-genmove_analyze");
        if(gtp_stream){output("="+r.gtp_id+"\n");r.streaming=true;}
        if(interval>0) {
            options.report_interval=interval*.001;options.on_report=[&](const SearchResult& result) {
                auto report=analysis_json(result,before,evaluator_.requests.load()-requests,evaluator_.batches.load()-batches,false);
                if(protocol_=="gtp")output(infos(report,r));else reply(r,{{"analysis",report},{"state",state()}},false);
            };
        }
        auto result=search_->run(before,temperature_(before,config_),options);
        if(generate_move)genmove_seconds_+=result.seconds;last_result_=result;
        if(r.data.value("debug",false))std::cerr<<"visits "<<result.root_visits<<" seconds "<<result.seconds<<" utility "<<result.value<<std::endl;
        auto report=analysis_json(result,before,evaluator_.requests.load()-requests,evaluator_.batches.load()-batches,true);
        if(result.action<0) {
            if(!r.cancelled->load())throw std::runtime_error("Search has no move: zero playout budget");
        } else if(protocol_!="gtp"&&!r.cancelled->load()&&!interrupted_())report["network_planes"]=raw_(before);
        if(gtp_stream)output(infos(report,r));
        bool generate=(r.command=="genmove"||r.command=="kata-genmove_analyze")&&!r.data.value("search_only",false);
        bool resigned=false;
        if(generate_move&&result.action>=0){
            recent_white_values_.push_back(-before.player()*result.value);
            if(config_.contains("analysis.allow_resignation")&&config_.boolean("analysis.allow_resignation")){
                int n=config_.integer("analysis.resign_consec_turns");resigned=recent_white_values_.size()>=size_t(n)&&before.turn()>=1+before.area()/6;
                for(int i=0;i<n&&resigned;++i)resigned=(-before.player()*recent_white_values_[recent_white_values_.size()-1-i])<config_.number("analysis.resign_threshold");
            }
            if(protocol_=="gtp"&&config_.contains("analysis.delay_move_scale")&&config_.number("analysis.delay_move_scale")>0&&config_.number("analysis.delay_move_max")>0){
                double prob=result.network_policy[result.action];if(evaluator_.secondary&&evaluator_.secondary->available()){auto e=evaluator_.secondary->evaluate_position(before.observation(),before.player());std::vector<int> legal;for(int a:before.static_actions())if(before.legal(a))legal.push_back(a);prob=source_policy_probabilities(e,legal,1,0)[result.action];}
                double mean=.5*config_.number("analysis.delay_move_scale")/(std::max(0.0,prob)+.1);double delay=std::max(.0001,std::min(config_.number("analysis.delay_move_max"),std::gamma_distribution<double>(2,mean/2)(gtp_random_)));
                auto end=std::chrono::steady_clock::now()+std::chrono::duration<double>(delay);while(!r.cancelled->load()&&!interrupted_()&&std::chrono::steady_clock::now()<end)std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        std::string chosen=resigned?"resign":result.action<0?"":vertex(before.local_action(result.action),before);
        bool moved=false;
        {
            // Cancellation and committing a generated move have one ordering point.
            std::lock_guard<std::mutex> lock(mutex_);
            moved=generate&&!resigned&&result.action>=0&&!r.cancelled->load()&&!interrupted_();
            if(moved){if(automatic_patterns())auto_pattern_samples_.push_back(before.position_json(result.action));
                search_for_player_=before.player();if(protocol_=="gtp"&&game_->rule()==Rule::GO)go_play(before.player(),before.local_action(result.action));else play(before.local_action(result.action));
                want_ponder_=config_.contains("analysis.pondering_enabled")&&config_.boolean("analysis.pondering_enabled")&&!game_->finished();}
            active_.reset();
        }
        save_patterns(false);
        if(protocol_=="gtp") {
            if(gtp_stream) {
                output((r.command=="kata-genmove_analyze"?(result.action>=0&&!r.cancelled->load()?"play "+chosen:"error search cancelled"):"")+(r.command=="kata-genmove_analyze"?"\n\n":"\n"));
                r.streaming=false;
            } else if(result.action>=0&&!r.cancelled->load())output("="+r.gtp_id+" "+chosen+"\n\n");
            else output("?"+r.gtp_id+" search cancelled\n\n");
        }
        return {{"state",state()},{"analysis",report},{"cancelled",r.cancelled->load()||interrupted_()},{"played",moved}};
    }
    std::string raw_gtp(const Request& request) {
        auto& args=request.args;size_t i=0;int perspective=game_->player();
        if(i<args.size()&&(args[i]=="b"||args[i]=="w"||args[i]=="B"||args[i]=="W"||args[i]=="black"||args[i]=="white"))perspective=color(args[i++]);
        if(i==args.size())throw std::runtime_error("Expected symmetry all or index");
        int selected=-1;if(i<args.size()){auto symmetry=args[i++];if(symmetry!="all"){selected=number(symmetry);if(selected<0)throw std::runtime_error("Invalid symmetry for game");}}
        if(selected< -1||selected>=game_->spec().symmetries)throw std::runtime_error("Invalid symmetry for game");
        double optimism=config_.number("analysis.root_policy_optimism");
        bool human=request.command=="kata-raw-human-nn";
        if(i<args.size()&&!human){size_t used=0;optimism=std::stod(args[i],&used);if(used!=args[i++].size())throw std::runtime_error("Invalid optimism");}
        if(i!=args.size()||!std::isfinite(optimism)||optimism<0||optimism>1)throw std::runtime_error("Expected [color] all|symmetry [optimism]");
        Evaluator* evaluator=human?evaluator_.secondary.get():&evaluator_;if(!evaluator)throw std::runtime_error("No human model loaded");
        Game position=*game_;
        if(position.rule()==Rule::GO&&perspective!=position.player()) {
            auto sample=Sgf::PositionSample::ofJsonLine(position.position_json());sample.board.clearSimpleKoLoc();sample.nextPla=perspective==1?P_BLACK:P_WHITE;
            sample.initialTurnNumber=0;position=game_from_go_sample(sample,canvas_,position.options());position.set_pda(game_->pda_doublings(),game_->pda_player());
        }
        std::vector<int> legal;for(int a:position.static_actions())if(position.raw_legal(a))legal.push_back(a);
        std::ostringstream out;out<<std::fixed<<std::setprecision(6);
        for(int symmetry=0;symmetry<position.spec().symmetries;++symmetry)if(selected<0||selected==symmetry) {
            auto value=evaluator->evaluate_position(position.observation(perspective),perspective,symmetry,true,1,false,optimism);
            auto policy=source_policy_probabilities(value,legal,1,optimism);
            out<<"symmetry "<<symmetry<<"\nwhiteWin "<<value.wdl[perspective==-1?0:2]<<"\nwhiteLoss "<<value.wdl[perspective==-1?2:0]
               <<"\nnoResult "<<value.wdl[3]<<"\ndraw "<<value.wdl[1]<<"\nshorttermWinlossError "<<value.shortterm_value_stdev<<"\npolicy\n";
            if(position.rule()==Rule::XIANGQI) {
                for(int a:legal)out<<vertex(position.local_action(a),position)<<' '<<policy[a]<<'\n';
            } else {
                for(int y=0;y<position.height();++y){for(int x=0;x<position.width();++x){int a=y*canvas_+x;if(position.raw_legal(a))out<<policy[a]<<' ';else out<<"NAN ";}out<<'\n';}
                if(position.rule()==Rule::GO)out<<"policyPass "<<policy[canvas_*canvas_]<<'\n';
            }
            out<<'\n';
        }
        return trim(out.str());
    }
    void execute(Request& r) {
        if(((r.cancelled->load()&&!r.continuous)||interrupted_())&&protocol_=="json") {reply(r,{{"cancelled",true},{"state",state()},{"played",false}});return;}
        const auto& cmd=r.command;
        if(cmd=="ponder"){ponder(r);return;}
        bool gtp=protocol_=="gtp";std::string value;bool search_reply=false;
        if(gtp) {
            if(std::find(commands.begin(),commands.end(),cmd)==commands.end())throw std::runtime_error("Unknown command");
            const std::map<std::string,size_t> arity={{"protocol_version",0},{"name",0},{"version",0},{"list_commands",0},{"known_command",1},{"clear_board",0},{"etazero-rule",1},{"play",2},{"undo",0},{"quit",0},{"stop",0},{"clear_cache",0},{"kata-get-models",0},{"kata-get-params",0},{"kata-list-params",0},{"kata-get-param",1},{"cputime",0},{"gomill-cpu_time",0},{"kata-debug-print-tc",0}};
            auto ar=arity.find(cmd);if(ar!=arity.end()&&r.args.size()!=ar->second)throw std::runtime_error("Unexpected command arguments");
            if(cmd=="protocol_version")value="2";else if(cmd=="name")value="EtaZero";else if(cmd=="version")value="3.2";
            else if(cmd=="known_command")value=std::find(commands.begin(),commands.end(),r.args[0])!=commands.end()?"true":"false";
            else if(cmd=="list_commands"){for(const auto& c:commands){if(!value.empty())value+='\n';value+=c;}}
            else if(cmd=="boardsize"||cmd=="rectangular_boardsize"||cmd=="clear_board"||cmd=="etazero-rule") {
                save_patterns(false);if(cmd=="clear_board"&&reload_auto_patterns_){reload_patterns();reload_auto_patterns_=false;}
                int width=game_->width(),height=game_->height();
                if(cmd=="boardsize"||cmd=="rectangular_boardsize"){
                    auto args=r.args;if(args.size()==1){auto colon=args[0].find(':');if(colon!=std::string::npos){auto v=args[0];args={v.substr(0,colon),v.substr(colon+1)};}}
                    if(args.empty()||args.size()>2)throw std::runtime_error("Expected size or width height");width=number(args[0]);height=args.size()==2?number(args[1]):width;
                    if(game_->rule()!=Rule::GO&&width!=height)throw std::runtime_error("Rectangular boards require Go");
                }
                Json data={{"size",std::max(width,height)},{"width",width},{"height",height},
                    {"rule",cmd=="etazero-rule"?r.args[0]:rule_name()},{"komi",game_->options().komi},{"go_rules",game_->options().go_rules}};create(data,r);
            } else if(cmd=="komi"||cmd=="get_komi"||cmd=="kata-get-rules"||cmd=="kata-set-rule"||cmd=="kata-set-rules"||cmd=="kgs-rules"||cmd=="set_position"||cmd=="fixed_handicap"||cmd=="place_free_handicap"||cmd=="set_free_handicap"||cmd=="loadsgf"||cmd=="printsgf"||cmd=="final_score"||cmd=="final_status_list")value=go_command(r);
            else if(cmd=="play") {int side=color(r.args[0]),local=action(r.args[1],*game_);if(game_->rule()==Rule::GO)go_play(side,local);else{if(side!=game_->player())throw std::runtime_error("Color is not side to move");play(local);}}
            else if(cmd=="clear_cache"){search_->reset(seed_);evaluator_.clear_cache();if(auto secondary=dynamic_cast<InferenceService*>(evaluator_.secondary.get()))secondary->clear_cache();last_result_.reset();}
            else if(cmd=="kata-get-models")value=models_.dump();
            else if(cmd=="kata-get-params")value=parameters().dump();
            else if(cmd=="kata-list-params"){auto params=parameters();for(auto it=params.begin();it!=params.end();++it){if(!value.empty())value+=' ';value+=it.key();}}
            else if(cmd=="kata-get-param"){auto params=parameters();if(!params.contains(r.args[0]))throw std::runtime_error("Unknown parameter: "+r.args[0]);auto v=params[r.args[0]];value=v.is_string()?v.get<std::string>():v.dump();}
            else if(cmd=="kata-set-param"||cmd=="kata-set-params") {
                Json requested;if(cmd=="kata-set-param"){if(r.args.empty()||r.args.size()>2)throw std::runtime_error("Expected parameter [value]");requested={{r.args[0],r.args.size()==2?r.args[1]:""}};}
                else {std::string text;for(auto& arg:r.args)text+=arg+" ";requested=Json::parse(text);if(!requested.is_object())throw std::runtime_error("Expected parameter object");}
                Config effective=config_;Json overrides=overrides_;std::optional<std::string> human_profile;
                for(auto it=requested.begin();it!=requested.end();++it){std::string key=it.key(),text=it.value().is_string()?it.value().get<std::string>():it.value().dump();
                    const std::map<std::string,std::string> special={{"alwaysComputePassAliveUnderSuicideRules","always_compute_pass_alive_under_suicide_rules"},{"excludeTerritoryAdjacentToAtari","exclude_territory_adjacent_to_atari"},{"analysisWideRootNoise","analysis_wide_root_noise"},{"analysisIgnorePreRootHistory","analysis_ignore_pre_root_history"},{"ponderingEnabled","pondering_enabled"},{"allowResignation","allow_resignation"},{"resignThreshold","resign_threshold"},{"resignConsecTurns","resign_consec_turns"},{"delayMoveScale","delay_move_scale"},{"delayMoveMax","delay_move_max"},{"humanSLProfile","human_sl_profile"}};
                    auto sp=special.find(key);
                    if(sp!=special.end()){
                        effective.override_field("analysis."+sp->second,text);
                        if(key=="humanSLProfile"){human_profile=text;if(!text.empty())SGFMetadata::getProfile(text);}
                        else if(key=="alwaysComputePassAliveUnderSuicideRules"||key=="excludeTerritoryAdjacentToAtari")history_modes(effective);
                        else if(key=="ponderingEnabled"||key=="allowResignation"||key=="analysisIgnorePreRootHistory")effective.boolean("analysis."+sp->second);
                        else {double v=effective.number("analysis."+sp->second);double low=key=="resignThreshold"?-1:0,high=key=="analysisWideRootNoise"?5:key=="resignThreshold"?0:key=="delayMoveScale"?10000:1000000;if(v<low||v>high)throw std::runtime_error("Parameter outside range");if(key=="resignConsecTurns"&&effective.integer("analysis."+sp->second)<1)throw std::runtime_error("resignConsecTurns must be positive");}
                    }else{auto found=SOURCE_SEARCH_KEYS.find(key);if(found==SOURCE_SEARCH_KEYS.end())throw std::runtime_error("Unsupported parameter: "+key);
                        for(auto fixed:{"node_table_shards_power_of_two","use_eval_cache","eval_cache_min_visits","avoid_repeated_pattern_utility"})if(found->second==fixed&&text!=config_.text("analysis."+found->second))throw std::runtime_error("Cannot change parameter after initialization: "+key);
                        if(found->second=="visit_cap_contempt_player"||found->second=="playout_doubling_advantage_player"){for(auto& c:text)c=std::tolower(c);if(found->second=="visit_cap_contempt_player"){if(text=="b"||text=="black")text="1";else if(text=="w"||text=="white")text="-1";else if(text=="unspecified")text="0";}else {if(text=="b")text="black";else if(text=="w")text="white";if(text!="black"&&text!="white")throw std::runtime_error("Invalid playout doubling advantage player");}}overrides[found->second]=text;effective.override_field("analysis."+found->second,text);}
                }
                bool replace_search=human_profile&&*human_profile!=config_.text("analysis.human_sl_profile");
                for(const auto& [key,type]:SOURCE_SEARCH_TYPES){auto name="analysis."+key;if(!effective.contains(name)||!config_.contains(name))continue;replace_search|=type=='b'?effective.boolean(name)!=config_.boolean(name):type=='i'||type=='f'?effective.number(name)!=config_.number(name):effective.text(name)!=config_.text(name);}
                if(analysis_mode_)for(auto key:{"analysis_wide_root_noise","analysis_ignore_pre_root_history"}){auto name=std::string("analysis.")+key;if(effective.contains(name)&&config_.contains(name))replace_search|=effective.text(name)!=config_.text(name);}
                auto active_overrides=overrides;if(analysis_mode_){active_overrides["wide_root_noise"]=effective.number("analysis.analysis_wide_root_noise");active_overrides["ignore_pre_root_history"]=effective.boolean("analysis.analysis_ignore_pre_root_history");}
                auto replacement=replace_search?factory_(active_overrides):nullptr;std::optional<::BoardHistory> replayed;if(go_initial_&&history_modes(effective)!=go_initial_->modes)replayed=replay_go(go_initial_->rules,go_moves_,history_modes(effective));
                if(human_profile){auto secondary=dynamic_cast<InferenceService*>(evaluator_.secondary.get());if(!secondary||!secondary->description().uses_human_profile)throw std::runtime_error("No metadata human model loaded");secondary->set_human_profile(*human_profile);}
                config_=std::move(effective);overrides_=std::move(overrides);if(replacement){search_=std::move(replacement);last_result_.reset();if(automatic_patterns())reload_patterns();}
                if(replayed){go_initial_->modes=replayed->modes;install_go(*replayed,false);}
                game_->set_pda(config_.number("analysis.playout_doubling_advantage"),config_.text("analysis.playout_doubling_advantage_player")=="black"?1:-1);
            }
            else if(cmd=="debug_moves") {
                Game position=*game_;std::vector<int> branch;bool raw=false;
                for(const auto& arg:r.args){if(arg=="rawstats"){raw=true;continue;}int a=position.native_action(action(arg,position));position.play(a);branch.push_back(a);}
                auto report=search_->inspect(position,branch);Json data=analysis_json(report,position,0,0,true);
                std::ostringstream out;if(position.rule()==Rule::GO)position.go_history().printBasicInfo(out,position.go_history().getRecentBoard(0));else out<<"player "<<position.player()<<" turn "<<position.turn()<<'\n';
                out<<data.dump();if(raw){auto e=evaluator_.evaluate_position(position.observation(),position.player());out<<"\nrawWinLoss "<<e.value()<<" draw "<<e.wdl[1]<<" noResult "<<e.wdl[3];}value=out.str();
            }
            else if(cmd=="kata-benchmark") {
                if(r.args.size()!=1)throw std::runtime_error("Expected visits");if(game_->width()!=game_->height())throw std::runtime_error("No built-in benchmark for rectangular boards");
                auto sgf=CompactSgf::parse(benchmark_sgf(game_->width()));int64_t visits=integer(Json{{"visits",[&]{size_t used=0;auto n=std::stoll(r.args[0],&used);if(used!=r.args[0].size())throw std::runtime_error("Invalid visits");return n;}()}},"visits",0,1,SEARCH_CONFIG_LIMIT);
                std::string seed="benchmarkPosSeed|";for(auto m:sgf->moves)seed+=std::to_string(int(m.loc))+"|";Rand rand(seed);std::vector<int> positions(sgf->moves.size());std::iota(positions.begin(),positions.end(),0);
                for(int i=int(positions.size())-1;i>1;--i)std::swap(positions[i],positions[rand.nextUInt(i)]);if(positions.size()>10)positions.resize(10);std::sort(positions.begin(),positions.end());
                auto params=overrides_;params["max_time"]=1e20;params["max_playouts"]=SEARCH_CONFIG_LIMIT;params["visits"]=visits;auto benchmark=factory_(params);
                auto rules=::Rules::getTrompTaylorish();rules.komi=sgf->getRulesOrFailAllowUnspecified(rules).komi;::Board board;::BoardHistory hist;Player pla;sgf->setupInitialBoardAndHist(rules,board,pla,hist,::BoardHistoryModes(true,true));
                int turn=0;double seconds=0;int64_t completed=0;auto requests=evaluator_.requests.load(),batches=evaluator_.batches.load();
                for(int end:positions){while(turn<end){auto m=sgf->moves[turn++];if(!hist.makeBoardMoveTolerant(board,m.loc,m.pla))throw std::runtime_error("Invalid benchmark SGF move");}benchmark->reset(seed_);evaluator_.clear_cache();auto game=game_from_go_history(hist,canvas_);SearchRun options;options.max_visits=visits;options.should_stop=[&]{return r.cancelled->load()||interrupted_();};auto result=benchmark->run(game,0,options);seconds+=result.seconds;completed+=result.root_visits;if(options.should_stop())break;}
                value="positions "+std::to_string(positions.size())+" visits "+std::to_string(completed)+" seconds "+Global::doubleToString(seconds)+" visitsPerSecond "+Global::doubleToString(completed/std::max(seconds,1e-30))+" nnRows "+std::to_string(evaluator_.requests.load()-requests)+" nnBatches "+std::to_string(evaluator_.batches.load()-batches);
            }
            else if(cmd=="cputime"||cmd=="gomill-cpu_time")value=Global::doubleToString(genmove_seconds_);
            else if(cmd=="kata-debug-print-tc") {int stones=std::count_if(game_->board().cells.begin(),game_->board().cells.end(),[](int x){return x!=0;});value="Black "+clocks_[0].toDebugString(game_->area(),stones,config_.number("analysis.lag_buffer"))+"\nWhite "+clocks_[1].toDebugString(game_->area(),stones,config_.number("analysis.lag_buffer"));}
            else if(cmd=="undo")undo(1);
            else if(cmd=="time_settings"||cmd=="kata-time_settings"||cmd=="kgs-time_settings"){if(cmd=="kgs-time_settings"&&!r.args.empty()&&r.args[0].rfind("fischer",0)==0)throw std::runtime_error("Use kata-time_settings for Fischer clocks");set_clock(r.args,cmd=="time_settings");}
            else if(cmd=="time_left")clock_left(r.args);
            else if(cmd=="kata-list_time_settings")value="none absolute byoyomi canadian fischer fischer-capped";
            else if(cmd=="genmove"||cmd=="kata-analyze"||cmd=="kata-genmove_analyze") {
                size_t i=0;bool needs_color=cmd=="genmove";
                if(i<r.args.size()&&(r.args[i]=="b"||r.args[i]=="w"||r.args[i]=="B"||r.args[i]=="W"||r.args[i]=="black"||r.args[i]=="white")) {
                    go_player(color(r.args[i++]));needs_color=false;
                }
                if(needs_color)throw std::runtime_error("Expected side to move");
                r.data["report_interval_ms"]=cmd=="genmove"?0:100;
                r.data["fresh"]=false;
                auto parse_interval=[&](const std::string& v){size_t used=0;double n=std::stod(v,&used);if(used!=v.size()||!std::isfinite(n)||n<0||n>=TimeControls::MAX_USER_INPUT_TIME)throw std::runtime_error("Invalid report interval");return n*10;};
                if(i<r.args.size()&&(std::isdigit(static_cast<unsigned char>(r.args[i][0]))||r.args[i][0]=='.'))r.data["report_interval_ms"]=parse_interval(r.args[i++]);
                while(i<r.args.size()) {
                    std::string key=r.args[i++];if(i==r.args.size())throw std::runtime_error("Missing analysis option value");
                    if(key=="interval")r.data["report_interval_ms"]=parse_interval(r.args[i++]);
                    else if(key=="maxmoves"||key=="visits"){auto value=r.args[i++];size_t used=0;int64_t count=std::stoll(value,&used);if(used!=value.size())throw std::runtime_error("Invalid visit count");r.data[key]=count;}
                    else if(key=="avoid"||key=="allow") {
                        if(i+2>=r.args.size())throw std::runtime_error("Expected player, comma-separated moves, depth");
                        int player=color(r.args[i++]);std::string list=r.args[i++];Json moves=Json::array();
                        std::istringstream stream(list);for(std::string v;std::getline(stream,v,',');)moves.push_back(action(v,*game_));
                        std::string field=key=="allow"?"allow_moves":"avoid_moves";
                        if(!r.data.contains(field))r.data[field]=Json::array();
                        r.data[field].push_back({{"player",player},{"moves",moves},{"until_depth",number(r.args[i++])}});
                    }
                    else if(key=="minmoves")r.data["minmoves"]=number(r.args[i++]);
                    else if(key=="pvVisits"||key=="pvEdgeVisits"||key=="rootInfo"||key=="noResultValue"){auto v=r.args[i++];if(v!="true"&&v!="false")throw std::runtime_error("Expected boolean");r.data[key]=v=="true";}
                    else if(key=="ownership"||key=="ownershipStdev"||key=="movesOwnership"||key=="movesOwnershipStdev"){auto v=r.args[i++];if(v!="false")throw std::runtime_error("Ownership heads are outside this model");}
                    else if(key=="focus"){
                        if(r.data.contains("focus_moves")||i+1>=r.args.size())throw std::runtime_error("Expected one focus move-list probability");
                        std::istringstream list(r.args[i++]);Json moves=Json::array(),weights=Json::array();
                        for(std::string v;std::getline(list,v,',');){if(v.empty())continue;auto colon=v.find(':');double weight=1;if(colon!=std::string::npos){auto w=v.substr(colon+1);size_t used=0;weight=std::stod(w,&used);if(used!=w.size()||!std::isfinite(weight)||weight<=0)throw std::runtime_error("Invalid focus weight");v.resize(colon);}moves.push_back(action(v,*game_));weights.push_back(weight);}
                        auto prob=r.args[i++];size_t used=0;double probability=std::stod(prob,&used);if(used!=prob.size()||!std::isfinite(probability)||probability<0||probability>1)throw std::runtime_error("Invalid focus probability");
                        r.data["focus_moves"]=moves;r.data["focus_weights"]=weights;r.data["focus_probability"]=probability;
                    }
                    else if(key=="focusProb"){size_t used=0;auto v=r.args[i++];double x=std::stod(v,&used);if(used!=v.size()||!std::isfinite(x)||x<0||x>1)throw std::runtime_error("Invalid focusProb");r.data["focus_probability"]=x;}
                    else throw std::runtime_error("Unsupported analysis option: "+key);
                }
                if(r.data.contains("visits"))r.continuous=false;
                integer(r.data,"maxmoves",10000000,0,999999999);integer(r.data,"minmoves",0,0,999999999);
                analyze(r);return;
            } else if(cmd=="kata-raw-nn"||cmd=="kata-raw-human-nn") {
                value=raw_gtp(r);
            } else if(cmd=="showboard")value=game_->rule()==Rule::GO?go_command(r):state().dump();
            while(value.find("\n\n")!=std::string::npos)value.replace(value.find("\n\n"),2,"\n");output("="+r.gtp_id+(value.empty()?"":" "+value)+"\n\n");return;
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
    Session(std::unique_ptr<GameSearch> search,InferenceService& eval,const Config& c,const Json& models,
            std::function<std::unique_ptr<GameSearch>(const Json&)> factory,uint64_t seed,std::string protocol,
            std::function<Json(const Game&)> raw,std::function<double(const Game&,const Config&)> temperature,std::function<bool()> interrupted)
        :search_(std::move(search)),evaluator_(eval),config_(c),models_(models),factory_(std::move(factory)),seed_(seed),protocol_(std::move(protocol)),canvas_(c.integer("network.canvas")),
         raw_(std::move(raw)),temperature_(std::move(temperature)),interrupted_(std::move(interrupted)) {}
    int run() {
        if(automatic_patterns()) {
            if(evaluator_.spec().kind!=GameKind::GO)throw std::runtime_error("Automatic pattern history requires Go");
            static_patterns_=PatternBonusTable::load(config_.text("analysis.pattern_corpora"),canvas_,rule_options(config_,GameConfigMode::ANALYSIS));reload_patterns();
        }
        if(protocol_=="gtp") {if(config_.contains("analysis.gtp_default_time_control")&&config_.boolean("analysis.gtp_default_time_control"))clocks_[0]=clocks_[1]=TimeControls::canadianOrByoYomiTime(1,5,5,1);Request r;create(Json::object(),r);}
        else emit({{"ok",true},{"event","ready"},{"canvas_size",canvas_}});
        std::thread worker(&Session::work,this);
        for(std::string line;read_line(line);) {
            Request r;
            try {
                if(protocol_=="json") {
                    r.data=Json::parse(line);if(!r.data.is_object())throw std::runtime_error("Request must be an object");
                    r.id=r.data.at("id");if(!r.id.is_string()&&!r.id.is_number_integer())throw std::runtime_error("id must be a string or integer");
                    r.command=r.data.at("command").get<std::string>();
                    const std::set<std::string> allowed={"id","command","size","width","height","rule","moves","action","count","visits","report_interval_ms","fresh","opening","seed","timeout_ms","target_id","continuous","fen","komi","go_rules","avoid_moves","allow_moves","avoid_rescale_root","focus_moves","focus_weights","focus_probability","clock_type","clock_arguments","player","time","stones"};
                    for(auto it=r.data.begin();it!=r.data.end();++it)if(!allowed.count(it.key()))throw std::runtime_error("Unsupported request field: "+it.key());
                    r.continuous=r.data.value("continuous",false);
                    if(r.continuous&&r.command!="analyze")throw std::runtime_error("continuous requires analyze");
                } else {
                    auto parts=words(line);if(parts.empty()){if(protocol_=="gtp"){std::lock_guard<std::mutex> lock(mutex_);if(active_&&(active_->continuous||active_->data.value("cancellable",false)))active_->cancelled->store(true);for(auto& job:queue_)if(job.continuous||job.data.value("cancellable",false))job.cancelled->store(true);}continue;}size_t index=0;
                    if(protocol_=="gtp"&&std::all_of(parts[0].begin(),parts[0].end(),[](unsigned char c){return std::isdigit(c);}))r.gtp_id=parts[index++];
                    if(index==parts.size())throw std::runtime_error("Missing command");r.command=parts[index++];r.args.assign(parts.begin()+index,parts.end());
                    if(protocol_=="gtp") {
                        const auto original=r.command;r.data["debug"]=original.find("debug")!=std::string::npos;r.data["lz"]=original=="analyze"||original.rfind("lz-",0)==0||original=="genmove_analyze";
                        r.data["cancellable"]=original.find("cancellable")!=std::string::npos;
                        r.data["search_only"]=original.rfind("kata-search",0)==0;
                        if(original=="analyze"||original=="lz-analyze")r.command="kata-analyze";
                        else if(original=="genmove_debug"||original=="kata-search"||original=="kata-search_debug"||original=="kata-search_cancellable")r.command="genmove";
                        else if(original=="genmove_analyze"||original=="lz-genmove_analyze"||original=="kata-search_analyze"||original=="kata-search_analyze_cancellable")r.command="kata-genmove_analyze";
                    }
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
                    bool running_search=active_&&(active_->command=="analyze"||active_->command=="genmove"||
                        (active_->command=="new"&&active_->data.value("opening",std::string("empty"))=="balanced"));
                    // State commands stay ordered. Mutations interrupt a private
                    // search/opening, never a preceding plain new/play/undo.
                    // Source GTP interrupts only continuous/cancellable searches.
                    if(r.command=="stop"||(protocol_!="gtp"&&mutation&&running_search)||
                       (protocol_=="gtp"&&active_&&(active_->continuous||active_->data.value("cancellable",false)))) {
                        auto cancel=[&](Request& job){if(!r.data.contains("target_id")||job.id==r.data["target_id"])job.cancelled->store(true);};
                        if(active_)cancel(*active_);for(auto& job:queue_)if(r.command=="stop"||job.continuous)cancel(job);
                    }
                    if(protocol_!="gtp"&&mutation)for(auto& job:queue_)if(job.continuous)job.cancelled->store(true);
                    if(protocol_=="gtp")for(auto& job:queue_)if(job.continuous||job.data.value("cancellable",false))job.cancelled->store(true);
                    queue_.push_back(std::move(r));
                }
                changed_.notify_all();if(quitting)break;
            } catch(const std::exception& error) {
                if(protocol_=="gtp")output("?"+r.gtp_id+" "+error.what()+"\n\n");
                else {Json response={{"ok",false},{"error",error.what()},{"id",r.id},{"final",true},{"event","result"}};emit(response);}
            }
        }
        {std::lock_guard<std::mutex> lock(mutex_);eof_=true;if(active_&&(active_->continuous||active_->command=="ponder"))active_->cancelled->store(true);for(auto& r:queue_)if(r.continuous)r.cancelled->store(true);}
        changed_.notify_all();worker.join();save_patterns(true);return interrupted_()?130:0;
    }
};
}
Json analysis_json(const SearchResult& result,const Game& game,uint64_t requests,uint64_t batches,bool final) {
    auto local=[&](int a){return game.local_action(a);};
    Json out={{"action",local(result.action)},{"board_width",game.width()},{"board_height",game.height()},{"board_stride",game.geometry().stride},{"board_size",game.size()},{"canvas_size",game.canvas()},
        {"turn",game.turn()},{"player",game.player()},{"rule",int(game.rule())},{"board",game.board().cells},
        {"pruned_value",result.pruned_value},{"pruned_utility",result.pruned_utility},{"root_value",result.value},{"root_utility",result.utility},{"root_weight",result.search_weight},{"wdl",result.search_wdl},{"network_wdl",result.network_wdl},
        {"completed_visits",result.root_visits},{"seconds",result.seconds},{"requests",requests},{"batches",batches},
        {"root_search_algo",result.gumbel?"gumbel":"puct"},{"final",final},{"stopped_early",result.stopped_early},
        {"candidates",Json::array()}};
    int64_t total=std::accumulate(result.visits.begin(),result.visits.end(),int64_t(0));
    for(int a=0;a<game.actions();++a)if(game.legal(a)) {
        const AnalysisCandidate* data=nullptr;for(const auto& c:result.analysis)if(c.action==a){data=&c;break;}
        Json c={{"action",local(a)},{"search_allowed",data!=nullptr},{"node_visits",data?data->node_visits:0},{"node_weight",data?data->node_weight:0},{"visits",result.visits.at(a)},{"network_prior",result.network_policy.at(a)},
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
        c["fpu_utility"]=data?data->utility:0;c["fpu_win_loss"]=data?data->win_loss:0;c["pvVisits"]=Json::array();c["pvEdgeVisits"]=Json::array();
        // Latent PVs may repeat occupied moves. Publish only the real-board legal prefix.
        if(data){Game pv=game;for(int move:data->pv){if(!pv.legal(move))break;c["pv"].push_back(local(move));pv.play(move);}}
        if(data){auto n=c["pv"].size();c["pvVisits"]=std::vector<int64_t>(data->pv_visits.begin(),data->pv_visits.begin()+std::min(n,data->pv_visits.size()));c["pvEdgeVisits"]=std::vector<int64_t>(data->pv_edge_visits.begin(),data->pv_edge_visits.begin()+std::min(n,data->pv_edge_visits.size()));}
        out["candidates"].push_back(std::move(c));
    }
    auto& candidates=out["candidates"];
    std::stable_sort(candidates.begin(),candidates.end(),[&](const Json& a,const Json& b){
        const char* field=result.gumbel?"improved_policy":"selection_weight";
        double x=a.at(field).get<double>(),y=b.at(field).get<double>();if(x!=y)return x>y;auto av=a.at("visits").get<int64_t>(),bv=b.at("visits").get<int64_t>();return av!=bv?av>bv:a["network_prior"].get<double>()>b["network_prior"].get<double>();
    });
    for(size_t i=0;i<candidates.size();++i)candidates[i]["rank"]=i+1;
    return out;
}
int engine_session(std::unique_ptr<GameSearch> search,InferenceService& eval,const Config& config,const Json& models,
                   const std::function<std::unique_ptr<GameSearch>(const Json&)>& factory,uint64_t seed,const std::string& protocol,
                   const std::function<Json(const Game&)>& raw,const std::function<double(const Game&,const Config&)>& temperature,const std::function<bool()>& interrupted) {
    if(protocol!="text"&&protocol!="json"&&protocol!="gtp")throw std::runtime_error("Unknown session protocol: "+protocol);
    return Session(std::move(search),eval,config,models,factory,seed,protocol,raw,temperature,interrupted).run();
}
}
