#include "etazero/pattern_bonus.h"
#include "../../external/katago/external/nlohmann_json/json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
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
        double penalty=source.value("penalty",0.0),decay=source.value("decay",1.0);int64_t minimum=source.value("min_turn",0LL),maximum=source.value("max_files",int64_t(1)<<50);
        if(!std::isfinite(penalty)||penalty<0||penalty>3||!std::isfinite(decay)||decay<0||decay>1||minimum<0||maximum<0)throw std::runtime_error("Invalid pattern corpus settings");
        auto names=source.value("players",std::vector<std::string>{});std::vector<std::filesystem::path> files;
        for(const auto& value:source.at("paths")){std::filesystem::path path=value.get<std::string>();if(std::filesystem::is_directory(path)){for(const auto& e:std::filesystem::recursive_directory_iterator(path))if(e.is_regular_file()&&e.path().extension()==".patterns")files.push_back(e.path());}else files.push_back(path);}
        std::stable_sort(files.begin(),files.end(),[](const auto& a,const auto& b){auto ta=std::filesystem::last_write_time(a),tb=std::filesystem::last_write_time(b);return ta==tb?a<b:ta>tb;});
        double factor=1;
        for(size_t index=0;index<files.size()&&index<size_t(maximum);++index){
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
