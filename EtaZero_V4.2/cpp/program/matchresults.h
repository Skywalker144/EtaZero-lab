#ifndef PROGRAM_MATCHRESULTS_H_
#define PROGRAM_MATCHRESULTS_H_

#include "../core/global.h"
#include "../external/nlohmann_json/json.hpp"

class MatchPairer;
class Logger;
class ConfigParser;

// Native output sink and in-memory view of the outer controller's schedule.
// The writer keeps its directory lock even if that controller exits mid-game.
class MatchResults {
  std::string directory;
  intptr_t lockHandle;
  nlohmann::json plan;
  // Separate bytes, not vector<bool>: game workers publish distinct IDs.
  std::vector<char> finished;
  int64_t numGames, proposalGames;
  bool extendPair, generate;
  void closeLock();
  nlohmann::json request(nlohmann::json message) const;
  void applyState(const nlohmann::json& state);
 public:
  static bool scoresGoMoveLimit(ConfigParser& cfg);
  MatchResults(const std::string& directory, const std::string& resolvedConfig, const std::string& extensionConfig,
               int64_t numGames, int64_t gamesPerPair, bool scoreGoMoveLimit);
  ~MatchResults();
  MatchResults(const MatchResults&) = delete;
  MatchResults& operator=(const MatchResults&) = delete;
  bool hasPlan() const { return !plan.is_null(); }
  void prepareSchedule(MatchPairer& pairer, const nlohmann::json& bots, const std::string& game,
                       const std::vector<std::pair<int,int>>& allowedPairs, Logger& logger);
  const nlohmann::json& getPlan() const { return plan; }
  bool completed(size_t index) const { return finished.at(index); }
  int64_t completedCount() const;
  void record(size_t index, const nlohmann::json& game);
};
#endif
