#ifndef PROGRAM_MATCHDISPATCH_H_
#define PROGRAM_MATCHDISPATCH_H_

#include "../core/global.h"
#include "../external/nlohmann_json/json.hpp"
#include <iostream>
#include <map>
#include <set>
#include <vector>

// A process boundary for outer scheduling. Native code supplies actual service
// identities and executes each returned batch only after the last has drained.
struct MatchDispatchWindow {
  std::set<int> models;
  std::vector<int64_t> games;
};

inline std::vector<MatchDispatchWindow> readMatchDispatch(
  const std::map<int64_t,std::pair<int,int>>& pending, bool external
) {
  using json = nlohmann::json;
  json requested = json::array();
  for(const auto& item : pending)
    requested.push_back({item.first,item.second.first,item.second.second});
  json batches;
  if(external) {
    std::cout << "ETAZERO_MATCH_DISPATCH " << json({{"games",requested}}).dump() << std::endl;
    std::string response;
    if(!std::getline(std::cin,response)) throw StringError("Match dispatch closed before receiving a schedule");
    batches = json::parse(response).at("windows");
  }
  else {
    // Direct pair-jobs without an outer scheduler retains all models.
    json ids = json::array();
    for(const auto& item : pending) ids.push_back(item.first);
    batches = json::array();
    if(!ids.empty()) batches.push_back(std::move(ids));
  }
  if(!batches.is_array()) throw StringError("Match dispatch windows must be an array");
  auto remaining = pending;
  std::vector<MatchDispatchWindow> windows;
  for(const auto& ids : batches) {
    if(!ids.is_array() || ids.empty()) throw StringError("Match dispatch windows must contain nonempty game arrays");
    MatchDispatchWindow window;
    for(const auto& id : ids) {
      if(!id.is_number_integer()) throw StringError("Match dispatch game IDs must be integers");
      const int64_t index = id.get<int64_t>();
      const auto it = remaining.find(index);
      if(it == remaining.end()) throw StringError("Match dispatch has an unknown, completed or repeated game ID");
      window.games.push_back(index);
      window.models.insert(it->second.first);
      window.models.insert(it->second.second);
      remaining.erase(it);
    }
    windows.push_back(std::move(window));
  }
  if(!remaining.empty()) throw StringError("Match dispatch omitted unfinished games");
  return windows;
}
#endif
