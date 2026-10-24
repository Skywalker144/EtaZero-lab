#include "search/pattern_bonus.h"
#include "external/nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstdio>
#include "game/go_position.h"
namespace etazero {
void PatternBonusTable::add(const Game& game,int action,double bonus,bool external,std::unordered_set<std::string>& seen) {
    for(bool flip:{false,true})for(int symmetry=0;symmetry<game.spec().symmetries;++symmetry){
        auto key=game.pattern_key(action,symmetry,flip);if(key.empty()||!seen.insert(key).second)continue;
        int player=game.player()*(flip?-1:1);
        entries_[key]+=external?(player==1?bonus:-bonus):bonus;
    }
}
void PatternBonusTable::add_game(const Game& game,double bonus,int only_player) {
    Game replay(game.size(),game.canvas(),game.rule(),game.options());std::unordered_set<std::string> seen;
    for(int action:game.moves()){if(!only_player||replay.player()==only_player)add(replay,action,bonus,false,seen);replay.play(action);}
}
std::shared_ptr<const PatternBonusTable> PatternBonusTable::load(const std::string& sources,int canvas,const RuleOptions& options) {
    if(sources.empty()||sources=="[]")return {};
    using Json=nlohmann::json;auto settings=Json::parse(sources);auto table=std::make_shared<PatternBonusTable>();
    if(!settings.is_array())throw std::runtime_error("pattern_corpora must be a JSON array");
    for(const auto& source:settings){
        double penalty=source.value("penalty",0.0),decay=source.value("decay",1.0);int64_t minimum=source.value("min_turn",0LL),maximum=source.value("max_files",1000000LL);
        if(!std::isfinite(penalty)||penalty< -3||penalty>3||!std::isfinite(decay)||decay<0||decay>1||minimum<0||minimum>1000000||maximum<1||maximum>1000000)throw std::runtime_error("Invalid pattern corpus settings");
        auto names=source.value("players",std::vector<std::string>{});std::vector<std::filesystem::path> files;
        for(const auto& value:source.at("paths")){std::filesystem::path path=value.get<std::string>();if(std::filesystem::is_directory(path)){for(const auto& e:std::filesystem::recursive_directory_iterator(path))if(e.is_regular_file()&&(e.path().extension()==".patterns"||e.path().extension()==".sgf"))files.push_back(e.path());}else files.push_back(path);}
        std::stable_sort(files.begin(),files.end(),[](const auto& a,const auto& b){auto ta=std::filesystem::last_write_time(a),tb=std::filesystem::last_write_time(b);return ta==tb?a<b:ta>tb;});
        double factor=1;
        for(size_t index=0;index<files.size()&&index<size_t(maximum);++index){
            if(files[index].extension()==".sgf") {
                try {
                    Game(std::min(5,canvas),canvas,Rule::GO,options);
                    std::unique_ptr<Sgf> sgf(Sgf::loadFile(files[index].string()));
                    std::set<Hash128> unique;std::unordered_set<std::string> seen;
                    sgf->iterAllUniquePositions(unique,true,true,false,false,nullptr,[&](Sgf::PositionSample&,const BoardHistory& history,const std::string& comments) {
                        if(history.moveHistory.empty()||int64_t(history.moveHistory.size())<minimum||comments.find("%SKIP%")!=std::string::npos)return;
                        const auto& move=history.moveHistory.back();
                        if(!names.empty()&&std::find(names.begin(),names.end(),sgf->getPlayerName(move.pla))==names.end())return;
                        Sgf::PositionSample before;before.board=history.getRecentBoard(1);before.nextPla=move.pla;
                        before.initialTurnNumber=history.getCurrentTurnNumber()-1;before.hintLoc=move.loc;before.weight=1;
                        int pattern_canvas=std::max({canvas,before.board.x_size,before.board.y_size});auto game=game_from_go_sample(before,pattern_canvas,options);
                        table->add(game,go_action(move.loc,before.board.x_size,pattern_canvas),penalty*factor,true,seen);
                    });
                    factor*=decay;
                }catch(const std::exception& e){std::cerr<<"Invalid pattern SGF "<<files[index]<<": "<<e.what()<<'\n';}
                continue;
            }
            std::ifstream in(files[index]);if(!in)throw std::runtime_error("Cannot open pattern corpus: "+files[index].string());
            std::unordered_set<std::string> seen;std::string line;bool used=false;
            while(std::getline(in,line)){if(line.empty())continue;
                try{auto record=Json::parse(line);auto rule=parse_rule(record.at("rule").get<std::string>());auto rules=options;if(record.contains("fen"))rules.fen=record["fen"].get<std::string>();
                    if(record.contains("go_rules"))rules.go_rules=record["go_rules"].get<std::string>();if(record.contains("komi"))rules.komi=record["komi"].get<double>();
                    Game game(record.at("size").get<int>(),canvas,rule,rules);auto moves=record.at("moves").get<std::vector<int>>();auto comments=record.value("comments",std::vector<std::string>(moves.size()));
                    if(comments.size()!=moves.size())throw std::runtime_error("Pattern comments and moves differ");
                    std::string players[2]={record.value("black",std::string()),record.value("white",std::string())};
                    for(size_t i=0;i<moves.size();++i){int action=game.native_action(moves[i]);if(!game.legal(action))throw std::runtime_error("Illegal pattern corpus move");
                        auto name=players[game.player()==1?0:1];if(int64_t(i+1)>=minimum&&comments[i].find("%SKIP%")==std::string::npos&&(names.empty()||std::find(names.begin(),names.end(),name)!=names.end()))table->add(game,action,penalty*factor,true,seen);
                        game.play(action);}used=true;
                }catch(const std::exception& e){std::cerr<<"Invalid pattern record "<<files[index]<<": "<<e.what()<<'\n';}
            }
            if(used)factor*=decay;
        }
    }
    return table;
}
}

namespace etazero {
namespace {
using Json=nlohmann::json;
double auto_parameter(const Config& c,const std::string& prefix,const std::string& size,const std::string& suffix,double low,double high) {
    const auto key="auto_avoid_repeat_"+suffix;
    auto sizes=Json::parse(c.text(prefix+".auto_avoid_repeat_sizes"));
    double value=sizes.contains(size)&&sizes[size].contains(key)?sizes[size][key].get<double>():c.number(prefix+"."+key);
    if(!std::isfinite(value)||value<low||value>high)throw std::runtime_error("Invalid "+key+size);
    return value;
}
}
std::shared_ptr<const PatternBonusTable> PatternBonusTable::load_auto(const Config& c,const std::string& prefix,int canvas,const RuleOptions& options) {
    if(!c.contains(prefix+".auto_avoid_repeat_dir")||c.text(prefix+".auto_avoid_repeat_dir").empty())return {};
    auto result=std::make_shared<PatternBonusTable>();std::filesystem::path base=c.text(prefix+".auto_avoid_repeat_dir");
    if(!std::filesystem::exists(base))return result;
    Game(std::min(5,canvas),canvas,Rule::GO,options);
    for(const auto& dir:std::filesystem::directory_iterator(base)) {
        if(!dir.is_directory())continue;auto name=dir.path().filename().string();
        int width=0,height=0;char extra=0;if(std::sscanf(name.c_str(),"%dx%d%c",&width,&height,&extra)!=2||width<2||height<2||width>25||height>25)continue;
        double penalty=auto_parameter(c,prefix,name,"utility",-3,3),decay=auto_parameter(c,prefix,name,"lambda",0,1);
        auto minimum=auto_parameter(c,prefix,name,"min_turn_number",0,1000000),maximum=auto_parameter(c,prefix,name,"max_turn_number",0,1000000);
        auto limit=auto_parameter(c,prefix,name,"max_poses",0,1e12);
        std::vector<std::filesystem::path> files;
        for(const auto& e:std::filesystem::recursive_directory_iterator(dir.path()))if(e.is_regular_file()&&Global::isSuffix(e.path().filename().string(),"_poses.txt"))files.push_back(e.path());
        std::stable_sort(files.begin(),files.end(),[](const auto& a,const auto& b){return std::filesystem::last_write_time(a)>std::filesystem::last_write_time(b);});
        size_t used=0,index=0;double factor=1;
        for(;index<files.size()&&used<limit;++index) {
            std::ifstream in(files[index]);std::unordered_set<std::string> seen;
            for(std::string line;std::getline(in,line);) {if(trim(line).empty())continue;
                try {
                    auto sample=Sgf::PositionSample::ofJsonLine(line);
                    if(sample.getCurrentTurnNumber()<minimum||sample.getCurrentTurnNumber()>maximum||!sample.moves.empty()||!sample.board.isLegal(sample.hintLoc,sample.nextPla,true))continue;
                    int pattern_canvas=std::max({canvas,sample.board.x_size,sample.board.y_size});
                    auto game=game_from_go_sample(sample,pattern_canvas,options);
                    result->add(game,go_action(sample.hintLoc,sample.board.x_size,pattern_canvas),penalty*factor,true,seen);++used;factor*=decay;
                }catch(const std::exception& e){std::cerr<<"Invalid automatic pattern record "<<files[index]<<": "<<e.what()<<'\n';}
            }
        }
        for(;index<files.size();++index){std::error_code error;std::filesystem::remove(files[index],error);if(error)std::cerr<<"Cannot prune pattern history: "<<error.message()<<'\n';}
    }
    return result;
}
bool PatternBonusTable::save_auto(const std::vector<std::string>& samples,const Config& c,const std::string& prefix) {
    if(samples.empty())return false;
    std::map<std::string,std::ofstream> outputs;
    std::random_device random;std::ostringstream file;file<<std::hex<<random()<<random()<<"_poses.txt";
    try {
        for(const auto& line:samples) {
            auto sample=Sgf::PositionSample::ofJsonLine(line);auto size=std::to_string(sample.board.x_size)+"x"+std::to_string(sample.board.y_size);
            if(sample.initialTurnNumber<auto_parameter(c,prefix,size,"min_turn_number",0,1000000)||sample.initialTurnNumber>auto_parameter(c,prefix,size,"max_turn_number",0,1000000))continue;
            auto& out=outputs[size];if(!out.is_open()) {
                auto dir=std::filesystem::path(c.text(prefix+".auto_avoid_repeat_dir"))/size;std::filesystem::create_directories(dir);
                out.open(dir/file.str());if(!out)throw std::runtime_error("Cannot open automatic pattern file");
            }
            out<<line<<'\n';if(!out)throw std::runtime_error("Cannot write automatic pattern file");
        }
        for(auto& entry:outputs){entry.second.close();if(!entry.second)throw std::runtime_error("Cannot close automatic pattern file");}
        return true;
    }catch(const std::exception& e){std::cerr<<"Saving automatic pattern history failed: "<<e.what()<<'\n';return false;}
}
}
