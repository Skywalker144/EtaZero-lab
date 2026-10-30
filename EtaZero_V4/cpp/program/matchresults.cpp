#include "matchresults.h"
#include "play.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

using json = nlohmann::json;
namespace {
void publish(const std::string& path, const json& data) {
  std::ofstream out;
  FileUtils::open(out,path + ".tmp");
  out << data.dump(2) << '\n';
  out.close();
  if(!out) throw StringError("Could not write match record: " + path);
  FileUtils::rename(path + ".tmp",path);
}
}

MatchResults::MatchResults(const std::string& dir, const std::string& config, const std::string& extension,
                           int64_t count, int64_t gamesPerPair)
  :directory(std::filesystem::absolute(dir).string()),lockHandle(-1),plan(),finished(),numGames(count),proposalGames(-1),
   extendPair(!extension.empty()),generate(false) {
  MakeDir::make(directory);
  const auto lockPath = directory + "/.native.lock";
#ifdef _WIN32
  const auto handle = CreateFileW(std::filesystem::path(lockPath).wstring().c_str(),GENERIC_READ|GENERIC_WRITE,
                                 0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
  if(handle == INVALID_HANDLE_VALUE) throw StringError("Another match is using the results directory, or it is inaccessible");
  lockHandle = reinterpret_cast<intptr_t>(handle);
#else
  lockHandle = open(lockPath.c_str(),O_RDWR|O_CREAT,0600);
  if(lockHandle < 0) throw StringError("Cannot open match results lock: " + lockPath);
  if(flock((int)lockHandle,LOCK_EX|LOCK_NB) != 0) {
    closeLock();
    throw StringError("Another match is using the results directory");
  }
#endif
  try {
    applyState(request({{"action","load"},{"resolved_config",config},{"extension_config",extension},
      {"working_directory",std::filesystem::current_path().string()},
      {"num_games",numGames},{"games_per_pair",gamesPerPair}}));
  }
  catch(...) { closeLock(); throw; }
}

void MatchResults::closeLock() {
  if(lockHandle == -1) return;
#ifdef _WIN32
  CloseHandle(reinterpret_cast<HANDLE>(lockHandle));
#else
  close((int)lockHandle);
#endif
  lockHandle = -1;
}
MatchResults::~MatchResults() { closeLock(); }

json MatchResults::request(json message) const {
  message["directory"] = directory;
  std::cout << "ETAZERO_MATCH_STATE " << message.dump() << std::endl;
  std::string response;
  if(!std::getline(std::cin,response))
    throw StringError("Match state control closed before receiving the outer schedule");
  return json::parse(response);
}

void MatchResults::applyState(const json& state) {
  plan = state.at("plan");
  generate = state.at("generate").get<bool>();
  proposalGames = state.at("proposal_games").is_null() ? -1 : state.at("proposal_games").get<int64_t>();
  finished.assign(hasPlan() ? plan.at("matchups").size() : 0,0);
  for(const auto& value : state.at("completed")) {
    if(!value.is_number_integer()) throw StringError("Completed match IDs must be integers");
    int64_t index = value.get<int64_t>();
    if(index < 0 || (size_t)index >= finished.size() || finished[(size_t)index])
      throw StringError("Outer match state has an invalid or repeated completed ID");
    finished[(size_t)index] = 1;
  }
}

int64_t MatchResults::completedCount() const {
  int64_t count = 0;
  for(char done: finished) count += done;
  return count;
}

void MatchResults::record(size_t index, const json& game) {
  const auto path = directory + "/game-" + std::to_string(index) + ".json";
  if(completed(index) || FileUtils::exists(path))
    throw StringError("Will not overwrite completed match game: " + path);
  publish(path,game);
  finished.at(index) = 1;
}

void MatchResults::prepareSchedule(MatchPairer& pairer, const json& bots, const std::string& game,
                                  const std::vector<std::pair<int,int>>& allowedPairs, Logger& logger) {
  if(generate) {
    json matchups = json::array();
    MatchPairer::BotSpec black, white;
    while((proposalGames < 0 || matchups.size() < (size_t)proposalGames) && pairer.getMatchup(black,white,logger))
      matchups.push_back({black.botIdx,white.botIdx});
    applyState(request({{"action","schedule"},{"proposal",{{"bots",bots},{"matchups",matchups},{"game",game}}}}));
  }
  if(!hasPlan() || generate || (!extendPair && plan.at("matchups").size() != (size_t)numGames) ||
     plan.at("matchups").size() < (size_t)numGames)
    throw StringError("Outer match schedule length differs from numGamesTotal");
  const std::set<std::pair<int,int>> allowed(allowedPairs.begin(),allowedPairs.end());
  for(const auto& pair : plan.at("matchups")) {
    if(!pair.is_array() || pair.size() != 2) throw StringError("Outer match schedule requires pairs of bot IDs");
    for(const auto& value : pair) {
      if(!value.is_number_integer()) throw StringError("Outer match bot IDs must be integers");
      int64_t index = value.get<int64_t>();
      if(index < 0 || (size_t)index >= bots.size()) throw StringError("Outer match bot ID is out of range");
    }
    if(!allowed.count({pair[0].get<int>(),pair[1].get<int>()}))
      throw StringError("Outer match schedule contains an unconfigured pairing");
  }
}
