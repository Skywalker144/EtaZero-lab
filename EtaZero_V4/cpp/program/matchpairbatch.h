#ifndef PROGRAM_MATCHPAIRBATCH_H_
#define PROGRAM_MATCHPAIRBATCH_H_

#include "matchresults.h"
#include "play.h"
#include "../core/fileutils.h"

// Autoelo keeps a separate native schedule and local bot indices for every
// pair. This view supplies global dispatch IDs without rewriting those files.
class MatchPairBatch {
 public:
  struct Job {
    std::unique_ptr<ConfigParser> config;
    std::unique_ptr<MatchResults> results;
    std::string extraConfig, sgfDirectory;
    std::vector<std::string> names, models;
    std::vector<bool> muzero;
    int64_t count;
    bool hasWork;
  };
  struct Game { size_t job, localIndex; int black, white; };
  std::vector<Job> jobs;
  std::vector<Game> games;

  MatchPairBatch(const ConfigParser& base, const std::string& manifest) {
    std::string text;
    FileUtils::loadFileIntoString(manifest,"",text);
    const auto input = nlohmann::json::parse(text);
    if(!input.is_array() || input.empty()) throw StringError("pair-jobs requires a nonempty task array");
    std::string sharedProfile;
    for(const auto& item : input) {
      Job job;
      job.extraConfig = item.at("extra_config").get<std::string>();
      job.sgfDirectory = item.at("sgf_output_dir").get<std::string>();
      job.config = std::make_unique<ConfigParser>(base);
      job.config->overrideKeys(job.extraConfig);
      auto& cfg = *job.config;
      if(cfg.getInt("numBots") != 2) throw StringError("pair-jobs requires two bots in each task");
      job.count = cfg.getInt64("numGamesTotal",1,((int64_t)1)<<62);
      if(cfg.getInt64("numGamesPerPair",1,((int64_t)1)<<62) != job.count)
        throw StringError("pair-jobs requires equal total/per-pair quotas");
      ConfigParser common(cfg);
      for(const std::string key : {"numGamesTotal","numGamesPerPair"}) common.overrideKey(key,"");
      for(int i = 0; i < 2; i++) {
        const auto index = std::to_string(i);
        job.names.push_back(cfg.getString("botName"+index));
        job.models.push_back(cfg.getString("nnModelFile"+index));
        job.muzero.push_back(cfg.contains("useMuZero"+index) ? cfg.getBool("useMuZero"+index) :
                              cfg.contains("useMuZero") && cfg.getBool("useMuZero"));
        for(const std::string key : {"botName","nnModelFile","useMuZero"}) common.overrideKey(key+index,"");
      }
      if(jobs.empty()) sharedProfile = common.getAllKeyVals();
      else if(common.getAllKeyVals() != sharedProfile)
        throw StringError("pair-jobs must share one profile; only participants and quotas may differ");
      jobs.push_back(std::move(job));
    }
    // Validate all profiles before opening output directories or taking locks.
    for(size_t i = 0; i < jobs.size(); i++) {
      auto& job = jobs[i];
      ConfigParser extension(*job.config);
      extension.overrideKey("numGamesTotal","");
      extension.overrideKey("numGamesPerPair","");
      job.results = std::make_unique<MatchResults>(input[i].at("results_dir").get<std::string>(),
                          job.config->getAllKeyVals(),extension.getAllKeyVals(),job.count,job.count);
      job.hasWork = true;
      if(job.results->hasPlan()) {
        job.hasWork = (size_t)job.count > job.results->getPlan().at("matchups").size();
        for(int64_t n = 0; !job.hasWork && n < job.count; n++)
          job.hasWork = !job.results->completed((size_t)n);
      }
    }
  }
  bool complete() const {
    for(const auto& job : jobs) if(job.hasWork) return false;
    return true;
  }
  void prepare(const std::vector<SearchParams>& localParams, const std::string& game, Logger& logger) {
    for(size_t j = 0; j < jobs.size(); j++) {
      auto& job = jobs[j];
      MatchPairer pairer(*job.config,2,job.names,{nullptr,nullptr},localParams,{{0,1},{1,0}},job.count+job.count%2);
      nlohmann::json bots = nlohmann::json::array();
      for(int i = 0; i < 2; i++)
        bots.push_back({{"name",job.names[i]},{"model",job.models[i]},{"muzero",(bool)job.muzero[i]}});
      job.results->prepareSchedule(pairer,bots,game,{{0,1},{1,0}},logger);
      for(size_t n = 0; n < (size_t)job.count; n++) {
        const auto& pair = job.results->getPlan().at("matchups").at(n);
        games.push_back({j,n,(int)(2*j)+pair[0].get<int>(),(int)(2*j)+pair[1].get<int>()});
      }
    }
  }
  bool completed(size_t id) const {
    const auto& g = games.at(id);
    return jobs[g.job].results->completed(g.localIndex);
  }
  int64_t completedCount() const {
    int64_t count = 0;
    for(size_t i = 0; i < games.size(); i++) count += completed(i);
    return count;
  }
  void record(size_t id, nlohmann::json game) {
    const auto& g = games.at(id);
    game["game_index"] = g.localIndex;
    game["black"] = g.black - 2*g.job;
    game["white"] = g.white - 2*g.job;
    jobs[g.job].results->record(g.localIndex,game);
  }
};
#endif
