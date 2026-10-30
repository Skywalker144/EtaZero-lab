#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../search/patternbonustable.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../program/selfplaymanager.h"
#include "../program/matchresults.h"
#include "../program/matchdispatch.h"
#include "../program/matchpairbatch.h"
#include "../command/commandline.h"
#include "../core/test.h"
#include "../main.h"

#include <csignal>

using namespace std;
using json = nlohmann::json;


static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

int MainCmds::match(const vector<string>& args) {
  Board::initHash();
  Rand seedRand;

  ConfigParser cfg;
  string logFile;
  string sgfOutputDir;
  string resultsDir;
  string pairJobsFile;
  bool extendPair = false;
  bool printConfig = false;
  bool externalDispatch = false;
  try {
    KataGoCommandLine cmd("Play different nets against each other with different search settings in a match or tournament.");
    cmd.addConfigFileArg("","match_example.cfg");

    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",false,string(),"FILE");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",false,string(),"DIR");
    TCLAP::ValueArg<string> resultsDirArg("","results-dir","Persist source pairings and completed games for resume and Elo",false,string(),"DIR");
    TCLAP::SwitchArg extendPairArg("","extend-pair","Allow increasing a two-bot pair quota while retaining its saved prefix",false);
    TCLAP::SwitchArg printConfigArg("","print-config","Print resolved match configuration as JSON without loading models",false);

    TCLAP::SwitchArg externalDispatchArg("","external-dispatch",
      "Report pending game/service IDs on stdout and read execution windows from stdin",false);
    TCLAP::ValueArg<string> pairJobsArg("","pair-jobs","Run extensible pair caches with shared model residency",false,string(),"JSON");
    cmd.add(pairJobsArg);
    cmd.add(externalDispatchArg);
    cmd.add(logFileArg);
    cmd.add(sgfOutputDirArg);
    cmd.add(resultsDirArg);
    cmd.add(extendPairArg);
    cmd.add(printConfigArg);

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    logFile = logFileArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();
    resultsDir = resultsDirArg.getValue();
    extendPair = extendPairArg.getValue();
    printConfig = printConfigArg.getValue();
    externalDispatch = externalDispatchArg.getValue();
    pairJobsFile = pairJobsArg.getValue();
    if(!pairJobsFile.empty() && (!resultsDir.empty() || extendPair || printConfig || !sgfOutputDir.empty()))
      throw StringError("pair-jobs supplies its own results, SGF directories and extensible quotas");
    if(externalDispatch && (printConfig || (resultsDir.empty() && pairJobsFile.empty())))
      throw StringError("external-dispatch requires results-dir or pair-jobs and a saved finite schedule");

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  ConfigParser quotaIndependent(cfg);
  quotaIndependent.overrideKey("numGamesTotal","");
  quotaIndependent.overrideKey("numGamesPerPair","");
  if(printConfig) {
    cout << json({{"resolved_config",cfg.getAllKeyVals()},
                  {"extension_config",quotaIndependent.getAllKeyVals()}}).dump() << endl;
    return 0;
  }
  if(extendPair && (resultsDir.empty() || cfg.getInt("numBots",1,Setup::MAX_BOT_PARAMS_FROM_CFG) != 2 ||
       cfg.getInt64("numGamesTotal",1,((int64_t)1)<<62) != cfg.getInt64("numGamesPerPair",1,((int64_t)1)<<62)))
    throw StringError("extend-pair requires two bots, results-dir and equal total/per-pair quotas");
  unique_ptr<MatchResults> results;
  if(!resultsDir.empty()) {
    results = make_unique<MatchResults>(resultsDir,cfg.getAllKeyVals(),extendPair ? quotaIndependent.getAllKeyVals() : "",
      cfg.getInt64("numGamesTotal",1,((int64_t)1)<<62),
      cfg.contains("numGamesPerPair") ? cfg.getInt64("numGamesPerPair",1,((int64_t)1)<<62) : 0);
    if(results->hasPlan()) {
      size_t requested = extendPair ? (size_t)cfg.getInt64("numGamesTotal",1,((int64_t)1)<<62) : results->getPlan().at("matchups").size();
      bool allComplete = requested <= results->getPlan().at("matchups").size();
      for(size_t i = 0; allComplete && i < requested; i++) allComplete = results->completed(i);
      if(allComplete) {
        cout << (extendPair ? "Requested saved match prefix is complete; no inference needed" :
                             "All saved match games are complete; no inference needed") << endl;
        return 0;
      }
    }
  }
  unique_ptr<MatchPairBatch> pairBatch;
  if(!pairJobsFile.empty()) {
    pairBatch = make_unique<MatchPairBatch>(cfg,pairJobsFile);
    if(pairBatch->complete()) {
      cout << "All requested pair prefixes are complete; no inference needed" << endl;
      return 0;
    }
    cfg.overrideKeys(pairBatch->jobs.front().extraConfig);
    // These fields were consumed by each job's parser before cloning the
    // shared profile; retain the source unused-key diagnostics for the rest.
    for(int i = 0; i < 2; i++) {
      const string index = std::to_string(i);
      cfg.markKeyUsed("botName"+index);
      cfg.markKeyUsed("nnModelFile"+index);
      cfg.markKeyUsed(cfg.contains("useMuZero"+index) ? "useMuZero"+index : "useMuZero");
    }
  }
  Logger logger(&cfg);
  logger.addFile(logFile);

  logger.write("Match Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load per-bot search config, first, which also tells us how many bots we're running
  vector<SearchParams> paramss = Setup::loadParams(cfg,Setup::SETUP_FOR_MATCH);
  assert(paramss.size() > 0);
  const vector<SearchParams> localPairParams = paramss;
  if(pairBatch) {
    paramss.clear();
    for(size_t i = 0; i < pairBatch->jobs.size(); i++)
      paramss.insert(paramss.end(),localPairParams.begin(),localPairParams.end());
  }
  int numBots = (int)paramss.size();

  //Figure out all pairs of bots that will be playing.
  std::vector<std::pair<int,int>> matchupsPerRound;
  {
    //Load a filter on what bots we actually want to run. By default, include everything.
    const int pairingBots = pairBatch ? 2 : numBots;
    vector<bool> includeBot(pairingBots);
    if(cfg.contains("includeBots")) {
      vector<int> includeBotIdxs = cfg.getInts("includeBots",0,Setup::MAX_BOT_PARAMS_FROM_CFG);
      for(int i = 0; i<pairingBots; i++) {
        if(contains(includeBotIdxs,i))
          includeBot[i] = true;
      }
    }
    else {
      for(int i = 0; i<pairingBots; i++) {
        includeBot[i] = true;
      }
    }

    std::vector<int> secondaryBotIdxs;
    if(cfg.contains("secondaryBots"))
      secondaryBotIdxs = cfg.getInts("secondaryBots",0,Setup::MAX_BOT_PARAMS_FROM_CFG);
    for(int i = 0; i<secondaryBotIdxs.size(); i++)
      if(secondaryBotIdxs[i] < 0 || secondaryBotIdxs[i] >= pairingBots)
        throw StringError("secondaryBots value " + Global::intToString(secondaryBotIdxs[i]) + " is out of range, numBots is " + Global::intToString(numBots));

    for(int i = 0; i<pairingBots; i++) {
      if(!includeBot[i])
        continue;
      for(int j = 0; j<pairingBots; j++) {
        if(!includeBot[j])
          continue;
        if(i < j && !(contains(secondaryBotIdxs,i) && contains(secondaryBotIdxs,j))) {
          matchupsPerRound.emplace_back(i,j);
          matchupsPerRound.emplace_back(j,i);
        }
      }
    }

    if(cfg.contains("extraPairs")) {
      std::vector<std::pair<int,int>> pairs = cfg.getNonNegativeIntDashedPairs("extraPairs",0,pairingBots-1);
      for(const std::pair<int,int>& pair: pairs) {
        int p0 = pair.first;
        int p1 = pair.second;
        if(cfg.contains("extraPairsAreOneSidedBW") && cfg.getBool("extraPairsAreOneSidedBW")) {
          matchupsPerRound.emplace_back(p0,p1);
        }
        else {
          matchupsPerRound.emplace_back(p0,p1);
          matchupsPerRound.emplace_back(p1,p0);
        }
      }
    }
  }

  if(pairBatch) {
    std::set<std::pair<int,int>> pairs(matchupsPerRound.begin(),matchupsPerRound.end());
    if(matchupsPerRound.size() != 2 || pairs != std::set<std::pair<int,int>>{{0,1},{1,0}})
      throw StringError("pair-jobs requires each local pair exactly once in each color");
    matchupsPerRound.clear();
    for(size_t i = 0; i < pairBatch->jobs.size(); i++) if(pairBatch->jobs[i].hasWork) {
      matchupsPerRound.emplace_back(2*i,2*i+1);
      matchupsPerRound.emplace_back(2*i+1,2*i);
    }
  }

  //Load the names of the bots and which model each bot is using
  vector<string> nnModelFilesByBot(numBots);
  vector<string> botNames(numBots);
  vector<bool> useMuZeroByBot(numBots);
  for(int i = 0; i<numBots; i++) {
    if(pairBatch) {
      const auto& job = pairBatch->jobs[(size_t)i/2];
      botNames[i] = job.names[i%2];
      nnModelFilesByBot[i] = job.models[i%2];
      useMuZeroByBot[i] = job.muzero[i%2];
      continue;
    }
    string idxStr = Global::intToString(i);
    useMuZeroByBot[i] = cfg.contains("useMuZero"+idxStr) ? cfg.getBool("useMuZero"+idxStr) :
      cfg.contains("useMuZero") && cfg.getBool("useMuZero");

    if(cfg.contains("botName"+idxStr))
      botNames[i] = cfg.getString("botName"+idxStr);
    else if(numBots == 1)
      botNames[i] = cfg.getString("botName");
    else
      throw StringError("If more than one bot, must specify botName0, botName1,... individually");

    if(cfg.contains("nnModelFile"+idxStr))
      nnModelFilesByBot[i] = cfg.getString("nnModelFile"+idxStr);
    else
      nnModelFilesByBot[i] = cfg.getString("nnModelFile");
  }

  vector<bool> botIsUsed(numBots);
  for(const std::pair<int,int>& pair : matchupsPerRound) {
    botIsUsed[pair.first] = true;
    botIsUsed[pair.second] = true;
  }

  //Dedup and load each necessary model exactly once
  vector<string> nnModelFiles;
  vector<bool> useMuZeroByModel;
  vector<int> whichNNModel(numBots);
  vector<int> modelConfigIndexes;
  vector<Setup::EvaluatorOptions> serviceOptions;
  for(int i = 0; !pairBatch && i<numBots; i++) {
    if(!botIsUsed[i])
      continue;

    const string& desiredFile = nnModelFilesByBot[i];
    int alreadyFoundIdx = -1;
    for(int j = 0; j<nnModelFiles.size(); j++) {
      if(nnModelFiles[j] == desiredFile) {
        if(useMuZeroByModel[j] != useMuZeroByBot[i])
          throw StringError("The same model path cannot name both an ordinary and a MuZero model");
        alreadyFoundIdx = j;
        break;
      }
    }
    if(alreadyFoundIdx != -1)
      whichNNModel[i] = alreadyFoundIdx;
    else {
      whichNNModel[i] = (int)nnModelFiles.size();
      nnModelFiles.push_back(desiredFile);
      useMuZeroByModel.push_back(useMuZeroByBot[i]);
    }
  }

  //Load match runner settings
  int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Work out an upper bound on how many concurrent nneval requests we could end up making.
  int expectedConcurrentEvals;
  {
    //Work out the max threads any one bot uses
    int maxBotThreads = 0;
    for(int i = 0; i<numBots; i++)
      if(paramss[i].numThreads > maxBotThreads)
        maxBotThreads = paramss[i].numThreads;
    //Mutiply by the number of concurrent games we could have
    expectedConcurrentEvals = maxBotThreads * numGameThreads;
  }

  //Initialize object for randomizing game settings and running games
  PlaySettings playSettings = PlaySettings::loadForMatch(cfg);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  if(pairBatch) {
    for(int i = 0; i < numBots; i++) {
      if(!botIsUsed[i]) continue;
      const auto& job = pairBatch->jobs[(size_t)i/2];
      // The source deduplicates within a pair before indexing NN options.
      const int localModel = i%2 == 0 || job.models[0] == job.models[1] ? 0 : 1;
      if(job.models[0] == job.models[1] && job.muzero[0] != job.muzero[1])
        throw StringError("The same model path cannot name both an ordinary and a MuZero model");
      int nx = maxBoardXSizeUsed, ny = maxBoardYSizeUsed;
      bool exact = minBoardXSizeUsed == nx && minBoardYSizeUsed == ny;
      if(useMuZeroByBot[i]) {
        // Planning needs only the canvas, not all three components' weights.
        // The native bundle loader validates the complete bundle on residency.
        string manifest;
        FileUtils::loadFileIntoString(nnModelFilesByBot[i]+"/model.json","",manifest);
        int canvas = json::parse(manifest).at("pos_len").get<int>();
        exact = exact && nx == canvas && ny == canvas;
        nx = ny = canvas;
      }
      auto options = Setup::readEvaluatorOptions(nnModelFilesByBot[i],localModel,cfg,logger,seedRand,
        expectedConcurrentEvals,nx,ny,Setup::MaxBatchSizeRequest::requireFromConfig(),exact,false,Setup::SETUP_FOR_MATCH,true);
      int index = -1;
      for(size_t j = 0; j < nnModelFiles.size(); j++) {
        if(nnModelFiles[j] == nnModelFilesByBot[i] && useMuZeroByModel[j] != useMuZeroByBot[i])
          throw StringError("The same model path cannot name both an ordinary and a MuZero model");
        if(nnModelFiles[j] == nnModelFilesByBot[i] && serviceOptions[j] == options) { index = (int)j; break; }
      }
      if(index < 0) {
        index = (int)nnModelFiles.size();
        nnModelFiles.push_back(nnModelFilesByBot[i]);
        useMuZeroByModel.push_back(useMuZeroByBot[i]);
        modelConfigIndexes.push_back(localModel);
        serviceOptions.push_back(std::move(options));
      }
      whichNNModel[i] = index;
    }
  }

  const bool windowed = externalDispatch || pairBatch != nullptr;

  //Initialize neural net inference engine globals, and load models
  Setup::initializeSession(cfg);
  const vector<string>& nnModelNames = nnModelFiles;
  const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
  const bool disableFP16 = false;
  vector<unique_ptr<SelfplayModel>> nnEvals(nnModelFiles.size());
  auto loadModel = [&](size_t i) {
    const int configIndex = pairBatch ? modelConfigIndexes[i] : (int)i;
    if(pairBatch) logger.write("Loading pair service " + std::to_string(i) + " with local model index " + std::to_string(configIndex));
    if(useMuZeroByModel[i]) {
      auto bundle = make_shared<MuZero::ModelBundle>(nnModelFiles[i]);
      if(Rules::parseGame(bundle->game()) != gameRunner->getGameInitializer()->getGame())
        throw StringError("MuZero model game differs from the configured match game");
      if(maxBoardXSizeUsed > bundle->posLen() || maxBoardYSizeUsed > bundle->posLen())
        throw StringError("Match board exceeds the MuZero model canvas");
      const bool exact = defaultRequireExactNNLen && maxBoardXSizeUsed == bundle->posLen() && maxBoardYSizeUsed == bundle->posLen();
      nnEvals[i] = Setup::initializeMuZeroSelfplayModel(nnModelNames[i],nnModelFiles[i],std::move(bundle),
        cfg,logger,seedRand,expectedConcurrentEvals,exact,configIndex,Setup::SETUP_FOR_MATCH);
    }
    else {
      nnEvals[i] = make_unique<SelfplayModel>(Setup::initializeNNEvaluator(
        nnModelNames[i],nnModelFiles[i],"",cfg,logger,seedRand,expectedConcurrentEvals,
        maxBoardXSizeUsed,maxBoardYSizeUsed,Setup::MaxBatchSizeRequest::requireFromConfig(),defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_MATCH,configIndex));
    }
  };
  if(!windowed) {
    for(size_t i = 0; i < nnModelFiles.size(); i++) loadModel(i);
    logger.write("Loaded neural net");
  }

  vector<NNEvaluator*> nnEvalsByBot(numBots);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;
    if(nnEvals[whichNNModel[i]]) nnEvalsByBot[i] = nnEvals[whichNNModel[i]]->boardEvaluator();
  }

  std::vector<std::unique_ptr<PatternBonusTable>> patternBonusTables = Setup::loadAvoidSgfPatternBonusTables(cfg,logger);
  testAssert(patternBonusTables.size() == (pairBatch ? 2 : numBots));

  //Initialize object for randomly pairing bots
  int64_t numGamesTotal = cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
  if(pairBatch) {
    pairBatch->prepare(localPairParams,Rules::gameName(gameRunner->getGameInitializer()->getGame()),logger);
    numGamesTotal = (int64_t)pairBatch->games.size();
  }
  int64_t gamesPerPair = cfg.contains("numGamesPerPair") ? cfg.getInt64("numGamesPerPair",1,((int64_t)1) << 62) : 0;
  int64_t pairingLimit = numGamesTotal;
  if(gamesPerPair > 0 && !pairBatch) {
    if(!results) throw StringError("numGamesPerPair requires -results-dir");
    std::set<std::pair<int,int>> orderedPairs(matchupsPerRound.begin(),matchupsPerRound.end());
    if(orderedPairs.empty() || orderedPairs.size() != matchupsPerRound.size())
      throw StringError("numGamesPerPair requires unique pairings in both colors");
    for(const auto& pair: orderedPairs)
      if(pair.first == pair.second || !orderedPairs.count({pair.second,pair.first}))
        throw StringError("numGamesPerPair requires distinct opponents paired in both colors");
    const int64_t numPairs = (int64_t)orderedPairs.size() / 2;
    if(numGamesTotal / numPairs != gamesPerPair || numGamesTotal % numPairs != 0)
      throw StringError("numGamesTotal must equal numGamesPerPair times the number of pairs");
    // For an odd quota, draw a full final source color round and keep the
    // first occurrence of each pair. This gives each pair exactly its quota,
    // with the unmatched color chosen by the source shuffled round.
    pairingLimit += (gamesPerPair % 2) * numPairs;
  }
  MatchPairer* matchPairer = new MatchPairer(cfg,numBots,botNames,nnEvalsByBot,paramss,matchupsPerRound,pairingLimit);

  if(results) {
    json bots = json::array();
    for(int i = 0; i < numBots; i++)
      bots.push_back({{"name",botNames[i]},{"model",nnModelFilesByBot[i]},{"muzero",(bool)useMuZeroByBot[i]}});
    results->prepareSchedule(*matchPairer,bots,Rules::gameName(gameRunner->getGameInitializer()->getGame()),
                             matchupsPerRound,logger);
  }

  std::vector<MatchDispatchWindow> dispatchWindows;
  if(windowed) {
    std::map<int64_t,std::pair<int,int>> pending;
    for(int64_t i = 0; i < numGamesTotal; i++) {
      if(pairBatch) {
        if(pairBatch->completed((size_t)i)) continue;
        const auto& game = pairBatch->games[(size_t)i];
        pending[i] = {whichNNModel[game.black],whichNNModel[game.white]};
      }
      else {
        if(results->completed((size_t)i)) continue;
        const auto& pair = results->getPlan().at("matchups").at(i);
        pending[i] = {whichNNModel[pair[0].get<int>()],whichNNModel[pair[1].get<int>()]};
      }
    }
    // Only dispatch IDs cross this boundary. Pairing, options and model service
    // identities are still resolved by the native components above.
    dispatchWindows = readMatchDispatch(pending,externalDispatch);
  }
  if(results || pairBatch) cout << "ETAZERO_MATCH_READY" << endl;

  //Check for unused config keys
  if(!windowed) cfg.warnUnusedKeys(cerr,&logger);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;
    if(nnEvalsByBot[i] != nullptr)
      Setup::maybeWarnHumanSLParams(paramss[i],nnEvalsByBot[i],NULL,cerr,&logger);
  }

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting matches");
  if(!logger.isLoggingToStdout())
    cout << "Loaded all config stuff, starting matches" << endl;

  if(sgfOutputDir != string())
    MakeDir::make(sgfOutputDir);
  if(pairBatch) for(const auto& job : pairBatch->jobs)
    if(!job.sgfDirectory.empty()) MakeDir::make(job.sgfDirectory);

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);


  std::mutex statsMutex;
  int64_t gameCount = pairBatch ? pairBatch->completedCount() : results ? results->completedCount() : 0;
  if(results && extendPair) {
    gameCount = 0;
    for(int64_t i = 0; i < numGamesTotal; i++) if(results->completed((size_t)i)) gameCount++;
  }
  atomic<int64_t> nextGame(0);
  vector<int64_t> windowGames; // Saved IDs; workers only read these until all have joined.
  std::map<string,double> timeUsedByBotMap;
  std::map<string,double> movesByBotMap;

  auto runMatchLoop = [
    &gameRunner,&matchPairer,&sgfOutputDir,&logger,&gameSeedBase,&patternBonusTables,
    &statsMutex, &gameCount, &timeUsedByBotMap, &movesByBotMap,
    &results,&pairBatch,&nextGame,&windowGames,windowed,numGamesTotal,&botNames,&paramss,&nnEvalsByBot,&nnEvals,&whichNNModel
  ](
    uint64_t threadHash
  ) {
    ofstream* sgfOut = NULL;
    std::map<size_t,unique_ptr<ofstream>> pairSgfOuts;
    if(sgfOutputDir.length() > 0) {
      sgfOut = new ofstream();
      FileUtils::open(*sgfOut, sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs");
    }
    auto shouldStopFunc = []() noexcept {
      return shouldStop.load();
    };
    WaitableFlag* shouldPause = nullptr;

    Rand thisLoopSeedRand;
    while(true) {
      if(shouldStop.load())
        break;

      FinishedGameData* gameData = NULL;

      MatchPairer::BotSpec botSpecB;
      MatchPairer::BotSpec botSpecW;
      int64_t gameIndex = -1;
      bool hasMatchup;
      if(results || pairBatch) {
        if(windowed) {
          int64_t index = nextGame.fetch_add(1);
          hasMatchup = index < (int64_t)windowGames.size();
          if(hasMatchup) gameIndex = windowGames[(size_t)index];
        }
        else {
          do { gameIndex = nextGame.fetch_add(1); }
          while(gameIndex < numGamesTotal && results->completed((size_t)gameIndex));
          hasMatchup = gameIndex < numGamesTotal;
        }
        if(hasMatchup) {
          json matchup = pairBatch ? json{pairBatch->games.at(gameIndex).black,pairBatch->games.at(gameIndex).white} :
                                    results->getPlan().at("matchups").at(gameIndex);
          auto populate = [&](MatchPairer::BotSpec& spec, int index) {
            spec.botIdx = index; spec.botName = botNames.at(index);
            spec.nnEval = nnEvalsByBot.at(index); spec.baseParams = paramss.at(index);
          };
          populate(botSpecB,matchup.at(0).get<int>());
          populate(botSpecW,matchup.at(1).get<int>());
        }
      }
      else hasMatchup = matchPairer->getMatchup(botSpecB, botSpecW, logger);
      if(hasMatchup) {
        botSpecB.muZeroEval = nnEvals[whichNNModel[botSpecB.botIdx]]->muZeroEvaluator();
        botSpecW.muZeroEval = nnEvals[whichNNModel[botSpecW.botIdx]]->muZeroEvaluator();
        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        std::function<void(const MatchPairer::BotSpec&, Search*)> afterInitialization = [&patternBonusTables,&pairBatch](const MatchPairer::BotSpec& spec, Search* search) {
          int index = pairBatch ? spec.botIdx%2 : spec.botIdx;
          assert(index < patternBonusTables.size());
          search->setCopyOfExternalPatternBonusTable(patternBonusTables[index]);
        };
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, NULL, NULL, logger,
          shouldStopFunc, shouldPause, nullptr, afterInitialization, nullptr
        );
      }

      bool shouldContinue = gameData != NULL;
      if(gameData != NULL) {
        if(results || pairBatch) {
          const auto& hist = gameData->endHist;
          if(!hist.isGameFinished)
            throw StringError("Match game reached its move limit without a rule result; not recording a rated game");
          const string outcome = hist.isNoResult ? "NR" : hist.winner == C_BLACK ? "B" : hist.winner == C_WHITE ? "W" : "D";
          ostringstream sgf, hash;
          WriteSgf::writeSgf(sgf,gameData->bName,gameData->wName,hist,gameData,false,true);
          hash << gameData->gameHash;
          json record = {{"game_index",gameIndex},{"black",botSpecB.botIdx},{"white",botSpecW.botIdx},
            {"outcome",outcome},{"moves",hist.getCurrentTurnNumber()},{"game_hash",hash.str()},
            {"black_time",gameData->bTimeUsed},{"white_time",gameData->wTimeUsed},{"sgf",sgf.str()}};
          if(pairBatch) pairBatch->record((size_t)gameIndex,std::move(record));
          else results->record((size_t)gameIndex,record);
        }
        if(pairBatch) {
          size_t job = pairBatch->games.at(gameIndex).job;
          const auto& directory = pairBatch->jobs[job].sgfDirectory;
          if(!directory.empty()) {
            auto& stream = pairSgfOuts[job];
            if(!stream) {
              stream = make_unique<ofstream>();
              FileUtils::open(*stream,directory+"/"+Global::uint64ToHexString(threadHash)+".sgfs");
            }
            WriteSgf::writeSgf(*stream,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true);
            *stream << endl;
          }
        }
        if(sgfOut != NULL) {
          WriteSgf::writeSgf(*sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true);
          (*sgfOut) << endl;
        }

        {
          std::lock_guard<std::mutex> lock(statsMutex);
          gameCount += 1;
          timeUsedByBotMap[gameData->bName] += gameData->bTimeUsed;
          timeUsedByBotMap[gameData->wName] += gameData->wTimeUsed;
          movesByBotMap[gameData->bName] += (double)gameData->bMoveCount;
          movesByBotMap[gameData->wName] += (double)gameData->wMoveCount;

          int64_t x = gameCount;
          while(x % 2 == 0 && x > 1) x /= 2;
          if(x == 1 || x == 3 || x == 5) {
            for(auto& pair : timeUsedByBotMap) {
              logger.write(
                "Avg move time used by " + pair.first + " " +
                Global::doubleToString(pair.second / movesByBotMap[pair.first]) + " " +
                Global::doubleToString(movesByBotMap[pair.first]) + " moves"
              );
            }
          }
        }

        delete gameData;
      }

      if(shouldStop.load())
        break;
      if(!shouldContinue)
        break;
    }
    if(sgfOut != NULL) {
      sgfOut->close();
      delete sgfOut;
    }
    logger.write("Match loop thread terminating");
  };
  auto runMatchLoopProtected = [&logger,&runMatchLoop](uint64_t threadHash) {
    Logger::logThreadUncaught("match loop", &logger, [&](){ runMatchLoop(threadHash); });
  };


  // Model switches only occur between drained windows. GameRunner and Search
  // have released their evaluators/latents before any service can be destroyed.
  auto releaseModel = [&](size_t i) {
    if(!nnEvals[i]) return;
    logger.write(nnEvals[i]->getModelFileName());
    logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
    logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
    logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
    nnEvals[i].reset();
  };
  Rand hashRand;
  auto runThreads = [&]() {
    vector<std::thread> threads;
    threads.reserve(numGameThreads);
    for(int i = 0; i<numGameThreads; i++)
      threads.emplace_back(runMatchLoopProtected, hashRand.nextUInt64());
    for(int i = 0; i<threads.size(); i++)
      threads[i].join();
  };
  if(!windowed) runThreads();
  else {
    std::set<int> resident;
    int windowNumber = 0;
    for(auto& window : dispatchWindows) {
      if(shouldStop.load()) break;
      ClockTimer prepareTimer;
      int retained = 0, loaded = 0;
      for(int i : resident) {
        if(window.models.count(i)) retained++;
        else releaseModel((size_t)i);
      }
      for(int i : window.models) {
        if(shouldStop.load()) break;
        if(!nnEvals[i]) { loadModel((size_t)i); loaded++; }
      }
      if(shouldStop.load()) break;
      for(int i = 0; i < numBots; i++) {
        if(!botIsUsed[i]) continue;
        auto* model = nnEvals[whichNNModel[i]].get();
        nnEvalsByBot[i] = model ? model->boardEvaluator() : nullptr;
        if(nnEvalsByBot[i] && !resident.count(whichNNModel[i]))
          Setup::maybeWarnHumanSLParams(paramss[i],nnEvalsByBot[i],NULL,cerr,&logger);
      }
      resident = window.models;
      windowGames = std::move(window.games);
      nextGame.store(0);
      logger.write("Model window " + std::to_string(++windowNumber) +
        ": resident=" + std::to_string(resident.size()) + " loaded=" + std::to_string(loaded) +
        " retained=" + std::to_string(retained) + " games=" + std::to_string(windowGames.size()) +
        " preparation seconds=" + Global::doubleToString(prepareTimer.getSeconds()));
      runThreads();
    }
    if(!shouldStop.load()) cfg.warnUnusedKeys(cerr,&logger);
  }

  delete matchPairer;
  delete gameRunner;

  nnEvalsByBot.clear();
  for(size_t i = 0; i < nnEvals.size(); i++) releaseModel(i);
  NeuralNet::globalCleanup();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}
