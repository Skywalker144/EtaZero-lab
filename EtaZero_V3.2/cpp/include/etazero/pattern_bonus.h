#pragma once
#include "game.h"
#include <unordered_map>
#include <unordered_set>
namespace etazero {
// Bonuses are expressed in White's utility perspective, as in KataGo.
class PatternBonusTable {
    std::unordered_map<std::string,double> entries_;
public:
    double get(const std::string& key) const {auto it=entries_.find(key);return it==entries_.end()?0:it->second;}
    size_t size() const {return entries_.size();}
    void add(const Game&,int action,double bonus,bool external,std::unordered_set<std::string>& seen);
    void add_game(const Game&,double bonus,int only_player=0);
    static std::shared_ptr<const PatternBonusTable> load(const std::string& json_sources,int canvas,const RuleOptions&);
};
}
