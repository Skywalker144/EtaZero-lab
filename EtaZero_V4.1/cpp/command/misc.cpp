#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/fancymath.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/parallel.h"
#include "../core/timer.h"
#include "../core/test.h"
#include "../dataio/sgf.h"
#include "../dataio/poswriter.h"
#include "../dataio/files.h"
#include "../game/xiangqi.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../program/selfplaymanager.h"
#include "../command/commandline.h"
#include "../tests/tests.h"
#include "../main.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <mutex>
#include <thread>

using namespace std;

static std::atomic<bool> sigReceived(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM)
    sigReceived.store(true);
}

int MainCmds::printclockinfo(const vector<string>& args) {
  (void)args;
#ifdef OS_IS_WINDOWS
  cout << "Does nothing on windows, disabled" << endl;
#endif
#ifdef OS_IS_UNIX_OR_APPLE
  cout << "Tick unit in seconds: " << std::chrono::steady_clock::period::num << " / " <<  std::chrono::steady_clock::period::den << endl;
  cout << "Ticks since epoch: " << std::chrono::steady_clock::now().time_since_epoch().count() << endl;
#endif
  return 0;
}

// These diagnostic commands use the source GTP evaluator defaults. The supplied
// game and board bounds constrain MuZero's fixed representation canvas only.
static unique_ptr<SelfplayModel> loadGtpDiagnosticModel(
  const string& modelFile, ConfigParser& cfg, Logger& logger, Rand& rand,
  int expectedConcurrentEvals, Rules::Game game, int maxBoardXSize, int maxBoardYSize
) {
  Setup::initializeSession(cfg);
  const bool defaultRequireExactNNLen = false;
  if(cfg.contains("useMuZero") && cfg.getBool("useMuZero")) {
    auto bundle = make_shared<MuZero::ModelBundle>(modelFile);
    if(Rules::parseGame(bundle->game()) != game)
      throw StringError("MuZero model game differs from the diagnostic positions");
    if(maxBoardXSize > bundle->posLen() || maxBoardYSize > bundle->posLen())
      throw StringError("MuZero model canvas must contain all configured boards");
    return Setup::initializeMuZeroSelfplayModel(modelFile,modelFile,std::move(bundle),cfg,logger,rand,
      expectedConcurrentEvals,defaultRequireExactNNLen,0,Setup::SETUP_FOR_GTP,Setup::MaxBatchSizeRequest::fromConcurrency());
  }
  const bool disableFP16 = false;
  const string expectedSha256 = "";
  return make_unique<SelfplayModel>(Setup::initializeNNEvaluator(
    modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
    Board::MAX_LEN,Board::MAX_LEN,Setup::MaxBatchSizeRequest::fromConcurrency(),defaultRequireExactNNLen,disableFP16,
    Setup::SETUP_FOR_GTP
  ));
}

int MainCmds::sampleinitializations(const vector<string>& args) {
  Board::initHash();

  ConfigParser cfg;
  string modelFile;
  int numToGen;
  bool evaluate;
  try {
    KataGoCommandLine cmd("View startposes");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numToGenArg("","num","Num to gen",false,1,"N");
    TCLAP::SwitchArg evaluateArg("","evaluate","Print out values on the inited poses");
    cmd.add(numToGenArg);
    cmd.add(evaluateArg);
    cmd.parseArgs(args);
    numToGen = numToGenArg.getValue();
    evaluate = evaluateArg.getValue();

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  //Play no moves in game, since we're sampling initializations
  cfg.overrideKey("maxMovesPerGame","0");

  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);

  unique_ptr<SelfplayModel> model;
  NNEvaluator* nnEval = NULL;
  MuZero::PositionEvaluator* muZeroEval = nullptr;
  if(cfg.getFileName() != "") {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    const GameInitializer* initializer = gameRunner->getGameInitializer();
    model = loadGtpDiagnosticModel(modelFile,cfg,logger,rand,params.numThreads,initializer->getGame(),
      initializer->getMaxBoardXSize(),initializer->getMaxBoardYSize());
    nnEval = model->boardEvaluator();
    muZeroEval = model->muZeroEvaluator();
    logger.write("Loaded neural net");
  }

  AsyncBot* evalBot;
  {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);
    params.maxVisits = 20;
    params.numThreads = 1;
    string seed = Global::uint64ToString(rand.nextUInt64());
    evalBot = muZeroEval ? new AsyncBot(params, *muZeroEval, &logger, seed) : new AsyncBot(params, nnEval, &logger, seed);
  }

  for(int i = 0; i<numToGen; i++) {
    string seed = Global::uint64ToString(rand.nextUInt64());
    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = "";
    botSpec.nnEval = nnEval;
    botSpec.muZeroEval = muZeroEval;
    botSpec.baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);

    FinishedGameData* data = gameRunner->runGame(
      seed,
      botSpec,
      botSpec,
      NULL,
      NULL,
      logger,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr
    );

    cout << data->startHist.rules.toString() << endl;
    if(data->startHist.rules.isXiangqi())
      data->startHist.printBasicInfo(cout, data->startBoard);
    else
      Board::printBoard(cout, data->startBoard, Board::NULL_LOC, &(data->startHist.moveHistory));
    cout << endl;
    if(evaluate) {
      ReportedSearchValues values;
      if(!data->startHist.rules.isGo() && data->startHist.isGameFinished) {
        // Policy initialization may reach a terminal position. It has no legal
        // policy to search; report its actual rule result, not a network value.
        const BoardHistory& hist = data->startHist;
        values.winValue = !hist.isNoResult && hist.winner == P_WHITE ? 1.0 : 0.0;
        values.lossValue = !hist.isNoResult && hist.winner == P_BLACK ? 1.0 : 0.0;
        values.drawValue = !hist.isNoResult && hist.winner == C_EMPTY ? 1.0 : 0.0;
        values.noResultValue = hist.isNoResult ? 1.0 : 0.0;
        values.winLossValue = values.winValue - values.lossValue;
        cout << "Terminal position (rule result)" << endl;
      }
      else {
        evalBot->setPosition(data->startPla, data->startBoard, data->startHist);
        evalBot->genMoveSynchronous(data->startPla,TimeControls());
        values = evalBot->getSearchStopAndWait()->getRootValuesRequireSuccess();
      }
      cout << "Winloss: " << values.winLossValue << endl;
      cout << "White W/D/L/NR: " << values.winValue << " " << values.drawValue << " " << values.lossValue << " " << values.noResultValue << endl;
    }

    delete data;
  }

  delete gameRunner;
  delete evalBot;
  model.reset();

  return 0;
}

int MainCmds::evalrandominits(const vector<string>& args) {
  Board::initHash();

  ConfigParser cfg;
  string modelFile;
  int numToGen;
  string initializationSeed;
  try {
    KataGoCommandLine cmd("Eval random inits");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numToGenArg("","num","Number of initializations to evaluate (0: run until interrupted)",false,0,"N");
    TCLAP::ValueArg<string> seedArg("","seed","Optional initialization RNG seed; does not seed search or neural inference",false,"","SEED");
    cmd.add(numToGenArg);
    cmd.add(seedArg);
    cmd.parseArgs(args);
    numToGen = numToGenArg.getValue();
    initializationSeed = seedArg.getValue();
    if(numToGen < 0) throw StringError("-num must be non-negative");

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  Rules::Game game = cfg.contains("rules") ? Rules::parseRules(cfg.getString("rules")).game : Rules::Game::GO;
  if(cfg.contains("game")) {
    Rules::Game declared = Rules::parseGame(cfg.getString("game"));
    if(cfg.contains("rules") && declared != game) throw StringError("Initialization config game and rules differ");
    game = declared;
  }
  if(!cfg.contains("game")) cfg.overrideKey("game",Rules::gameName(game));
  // Preserve the source diagnostic's Japanese Go rules; other games use their
  // native rule adapter. The shared board-size fields allow explicit sizes.
  const Rules rules = game == Rules::Game::GO ? Rules::parseRules("japanese") : Rules::forGame(game);
  int xSize = rules.isXiangqi() ? 9 : 19;
  int ySize = rules.isXiangqi() ? 10 : 19;
  Setup::loadDefaultBoardXYSize(cfg,logger,xSize,ySize);
  if(rules.isXiangqi() && (xSize != 9 || ySize != 10))
    throw StringError("Xiangqi initialization requires a 9x10 board");
  if(rules.isPlacement() && xSize != ySize)
    throw StringError("Placement initialization requires a square board");
  logger.write("Initialization rules: " + rules.toString() + ", board " + Global::intToString(xSize) + "x" + Global::intToString(ySize));

  unique_ptr<SelfplayModel> model;
  NNEvaluator* nnEval = NULL;
  MuZero::PositionEvaluator* muZeroEval = nullptr;
  if(cfg.getFileName() != "") {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    model = loadGtpDiagnosticModel(modelFile,cfg,logger,rand,params.numThreads,game,xSize,ySize);
    nnEval = model->boardEvaluator();
    muZeroEval = model->muZeroEvaluator();
    logger.write("Loaded neural net");
  }

  Search* evalBot;
  {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);
    params.maxVisits = 40;
    params.numThreads = 1;
    string seed = Global::uint64ToString(rand.nextUInt64());
    evalBot = muZeroEval ? new Search(params,*muZeroEval,&logger,seed) : new Search(params,nnEval,&logger,seed);
  }

  if(!std::atomic_is_lock_free(&sigReceived))
    throw StringError("sigReceived is not lock free, signal-quitting mechanism will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
  std::function<bool()> shouldStop = []() noexcept { return sigReceived.load(); };

  Rand gameRand;
  if(initializationSeed != "") gameRand.init(initializationSeed);
  cout << "requestedInitialMoves,whiteWinLoss,whiteWin,whiteDraw,whiteLoss,whiteNoResult,valueSource,initialMoves" << endl;
  for(int64_t generated = 0; (numToGen == 0 || generated < numToGen) && !shouldStop(); generated++) {
    Board board(xSize,ySize);
    Player pla = P_BLACK;
    //Keep pass-alive computations (e.g. endGameIfAllPassAlive below) consistent with how the bot's
    //own searches are performing them.
    BoardHistory hist = rules.isXiangqi() ? BoardHistory(Xiangqi::State(),evalBot->getRootHist().modes) :
      BoardHistory(board,pla,rules,0,evalBot->getRootHist().modes);
    if(rules.isXiangqi()) board = hist.initialBoard;
    int numInitialMovesToPlay = (int)gameRand.nextUInt(200);
    double temperature = 1.0;
    for(int i = 0; i<numInitialMovesToPlay; i++) {
      if(shouldStop()) break;
      NNResultBuf buf;
      Loc loc = PlayUtils::getGameInitializationMove(evalBot, evalBot, board, hist, pla, buf, gameRand, temperature);

      assert(hist.isLegal(board,loc,pla));
      hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
      pla = getOpp(pla);

      hist.endGameIfAllPassAlive(board);
      if(hist.isGameFinished)
        break;
    }
    if(shouldStop()) break;

    const bool ruleResult = !rules.isGo() && hist.isGameFinished;
    ReportedSearchValues values;
    if(ruleResult) {
      double winLoss = hist.winner == P_WHITE ? 1.0 : hist.winner == P_BLACK ? -1.0 : 0.0;
      values = ReportedSearchValues(winLoss,hist.winner == C_EMPTY ? 1.0 : 0.0,0.0,winLoss,0.0,0);
    }
    else {
      evalBot->setPosition(pla,board,hist);
      evalBot->runWholeSearch(pla,&shouldStop);
      if(shouldStop()) break;
      values = evalBot->getRootValuesRequireSuccess();
    }
    cout << numInitialMovesToPlay << "," << values.winLossValue
         << "," << values.winValue << "," << values.drawValue << "," << values.lossValue << "," << values.noResultValue
         << "," << (ruleResult ? "rule" : "search") << "," << hist.moveHistory.size() << endl;
  }
  delete evalBot;
  model.reset();

  return 0;
}

int MainCmds::searchentropyanalysis(const vector<string>& args) {
  Board::initHash();

  ConfigParser cfg;
  string modelFile;
  string boardSizeDataset;
  vector<string> sgfFiles;
  try {
    KataGoCommandLine cmd("Analyze search entropy across datasets");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> boardSizeDatasetArg("","boardsizedataset", "Built-in Go dataset to analyze (9,13,19,10x14,rectangle), instead of -sgf", false, "", "SIZE");
    TCLAP::MultiArg<string> sgfFilesArg("","sgf", "SGF to analyze instead of a built-in dataset; repeat for multiple games of the same type", false, "FILE");
    cmd.add(boardSizeDatasetArg);
    cmd.add(sgfFilesArg);
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    boardSizeDataset = boardSizeDatasetArg.getValue();
    sgfFiles = sgfFilesArg.getValue();
    if(boardSizeDataset.empty() == sgfFiles.empty())
      throw StringError("Specify either -boardsizedataset or one or more -sgf files");
    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  const bool builtinDataset = sgfFiles.empty();
  if(!builtinDataset) {
    boardSizeDataset = sgfFiles[0];
    for(size_t i = 1; i<sgfFiles.size(); i++) boardSizeDataset += ", " + sgfFiles[i];
  }
  logger.write("Search Entropy Analysis");
  logger.write("Model: " + modelFile);
  logger.write("Dataset: " + boardSizeDataset);

  vector<string> sgfData;
  if(!builtinDataset) {
    for(const string& file : sgfFiles) {
      string text;
      FileUtils::loadFileIntoString(file,"",text);
      sgfData.push_back(std::move(text));
    }
  }
  else if(boardSizeDataset == "9")
    sgfData = TestCommon::getMultiGameSize9Data();
  else if(boardSizeDataset == "13")
    sgfData = TestCommon::getMultiGameSize13Data();
  else if(boardSizeDataset == "19")
    sgfData = TestCommon::getMultiGameSize19Data();
  else if(boardSizeDataset == "10x14")
    sgfData = TestCommon::getMultiGameSize10x14Data();
  else if(boardSizeDataset == "rectangle")
    sgfData = TestCommon::getMultiGameRectangleData();
  else {
    throw StringError("Unknown search entropy dataset: " + boardSizeDataset);
  }

  // Build list of all (sgfIdx, turnIdx) pairs, then deterministically shuffle
  vector<std::pair<int,int>> positions;
  vector<Rules> initialRulesBySgf;
  int maxBoardXSize = 0;
  int maxBoardYSize = 0;
  {
    vector<std::unique_ptr<CompactSgf>> sgfObjs;
    for(const string& sgf : sgfData)
      sgfObjs.push_back(CompactSgf::parse(sgf));
    for(int sgfIdx = 0; sgfIdx < (int)sgfObjs.size(); sgfIdx++) {
      const auto& sgf = sgfObjs[sgfIdx];
      // Preserve the source built-in dataset's default-rules replay. Explicit
      // SGFs supply their own game/rules, including full Xiangqi starting history.
      Rules rules = builtinDataset ? Rules() : sgf->getRulesOrWarn(Rules(),[&logger](const string& msg) { logger.write(msg); });
      if(cfg.contains("game") && Rules::parseGame(cfg.getString("game")) != rules.game)
        throw StringError("searchentropyanalysis config and SGF game differ");
      if(!cfg.contains("game")) cfg.overrideKey("game",Rules::gameName(rules.game));
      initialRulesBySgf.push_back(rules);
      maxBoardXSize = std::max(maxBoardXSize,sgf->xSize);
      maxBoardYSize = std::max(maxBoardYSize,sgf->ySize);
      for(int turnIdx = 0; turnIdx < (int)sgfObjs[sgfIdx]->moves.size(); turnIdx++) {
        positions.push_back({sgfIdx, turnIdx});
      }
    }
  }
  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
  auto model = loadGtpDiagnosticModel(modelFile,cfg,logger,rand,params.numThreads,
    Rules::parseGame(cfg.getString("game")),maxBoardXSize,maxBoardYSize);
  logger.write("Loaded neural net");

  Search* bot;
  {
    SearchParams searchParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    string seed = Global::uint64ToString(rand.nextUInt64());
    bot = model->muZeroEvaluator() ? new Search(searchParams,*model->muZeroEvaluator(),&logger,seed) :
      new Search(searchParams,model->boardEvaluator(),&logger,seed);
  }
  {
    Rand shuffleRand("searchentropyanalysis_shuffle_seed");
    shuffleRand.shuffle(positions);
  }
  logger.write("Total positions to search: " + Global::intToString((int)positions.size()));

  vector<double> searchEntropies;
  vector<double> searchSurprises;
  int numPositions = 0;

  auto printRunningStats = [&]() {
    cout << "Searched numPositions: " << numPositions << endl;
    if(numPositions > 0) {
      double entropyMean = 0.0;
      for(double v : searchEntropies) entropyMean += v;
      entropyMean /= numPositions;
      double entropyVar = 0.0;
      for(double v : searchEntropies) { double d = v - entropyMean; entropyVar += d * d; }
      entropyVar /= numPositions;
      cout << "Mean search entropy: " << entropyMean << " standard deviation: " << sqrt(entropyVar) << endl;

      double surpriseMean = 0.0;
      for(double v : searchSurprises) surpriseMean += v;
      surpriseMean /= numPositions;
      double surpriseVar = 0.0;
      for(double v : searchSurprises) { double d = v - surpriseMean; surpriseVar += d * d; }
      surpriseVar /= numPositions;
      cout << "Mean search surprise: " << surpriseMean << " standard deviation: " << sqrt(surpriseVar) << endl;
    }
  };

  for(const auto& pos : positions) {
    int sgfIdx = pos.first;
    int turnIdx = pos.second;

    std::unique_ptr<CompactSgf> sgfObj = CompactSgf::parse(sgfData[sgfIdx]);

    Board board;
    Player pla;
    BoardHistory hist;
    sgfObj->setupInitialBoardAndHist(initialRulesBySgf[sgfIdx], board, pla, hist, bot->getRootHist().modes);

    if(!builtinDataset) {
      sgfObj->fileName = sgfFiles[sgfIdx];
      sgfObj->playMovesTolerant(board,pla,hist,turnIdx,false);
    }
    else {
      for(int i = 0; i < turnIdx; i++) {
        Loc moveLoc = sgfObj->moves[i].loc;
        if(moveLoc != Board::NULL_LOC && hist.isLegal(board, moveLoc, pla)) {
          hist.makeBoardMoveAssumeLegal(board, moveLoc, pla, NULL);
          pla = getOpp(pla);
        }
      }
    }

    if(!hist.isGameFinished) {
      bot->setPosition(pla, board, hist);
      bot->runWholeSearch(pla);

      double surprise, searchEntropy, policyEntropy;
      if(bot->getPolicySurpriseAndEntropy(surprise, searchEntropy, policyEntropy)) {
        searchEntropies.push_back(searchEntropy);
        searchSurprises.push_back(surprise);
        numPositions++;
        if(numPositions % 10 == 0)
          printRunningStats();
      }
    }
  }

  cout << modelFile << endl;
  cout << "Dataset: " << boardSizeDataset << endl;
  printRunningStats();

  delete bot;
  model.reset();
  return 0;
}

//One CSV row per recorded turn of the game, containing the stats that drive surprise-based selection of
//positions (policy surprise, value surprise) along with the per-game and per-turn covariates needed to
//analyze what they vary with.
static void writeSurpriseStatsRows(ostream& out, int64_t gameIdx, const FinishedGameData& data) {
  size_t numMoves = data.targetWeightByTurn.size();
  testAssert(data.hasFullData);
  testAssert(data.policySurpriseByTurn.size() == numMoves);
  testAssert(data.policyEntropyByTurn.size() == numMoves);
  testAssert(data.searchEntropyByTurn.size() == numMoves);
  testAssert(data.valueSurpriseByTurn.size() == numMoves);
  testAssert(data.wasCheapSearchByTurn.size() == numMoves);
  testAssert(data.nnRawStatsByTurn.size() == numMoves);
  testAssert(data.policyTargetsByTurn.size() == numMoves);
  testAssert(data.reanalysisByTurn.size() == numMoves);
  testAssert(data.whiteValueTargetsByTurn.size() == numMoves + 1);

  //Last entry is the terminal target, or the bootstrap target for a truncated non-Go game.
  const ValueTargets& outcome = data.whiteValueTargetsByTurn[numMoves];
  double pdaWhite =
    data.playoutDoublingAdvantagePla == P_WHITE ? data.playoutDoublingAdvantage :
    data.playoutDoublingAdvantagePla == P_BLACK ? -data.playoutDoublingAdvantage : 0.0;
  //Turn number of the first recorded turn, counting handicap/policy-init/fork setup moves and any initial
  //turn number offset of positions begun from sgfs, so that turn number is comparable across game modes.
  int64_t startTurnIdx = data.startHist.initialTurnNumber + (int64_t)data.startHist.moveHistory.size();

  for(size_t t = 0; t<numMoves; t++) {
    const ValueTargets& mcts = data.whiteValueTargetsByTurn[t];
    const NNRawStats& raw = data.nnRawStatsByTurn[t];
    const ReanalysisData& rean = data.reanalysisByTurn[t];
    out << gameIdx
        << "," << data.gameHash.toString()
        << "," << Rules::gameName(data.startHist.rules.game)
        << "," << data.startBoard.x_size
        << "," << data.startBoard.y_size
        << "," << data.startHist.rules.komi
        << "," << pdaWhite
        << "," << data.numExtraBlack
        << "," << data.mode
        << "," << (data.hitTurnLimit ? 1 : 0)
        << "," << numMoves
        << "," << outcome.win
        << "," << outcome.draw
        << "," << outcome.loss
        << "," << outcome.noResult
        << "," << (startTurnIdx + (int64_t)t)
        << "," << t
        << "," << (data.wasCheapSearchByTurn[t] ? 1 : 0)
        << "," << data.policyTargetsByTurn[t].unreducedNumVisits
        << "," << data.targetWeightByTurn[t]
        << "," << data.policySurpriseByTurn[t]
        << "," << data.valueSurpriseByTurn[t]
        << "," << data.policyEntropyByTurn[t]
        << "," << data.searchEntropyByTurn[t]
        << "," << raw.whiteWinLoss
        << "," << mcts.win
        << "," << mcts.draw
        << "," << mcts.loss
        << "," << mcts.noResult
        << "," << (rean.wasReanalyzed ? 1 : 0)
        << "," << rean.selectionPolicySurprise
        << "," << rean.selectionValueSurprise
        << "," << rean.originalNumVisits
        << "\n";
  }
}

int MainCmds::selfplaysurprisedump(const vector<string>& args) {
  Board::initHash();

  ConfigParser cfg;
  string modelFile;
  string outputFile;
  int64_t numGamesTotal;
  bool reanalyzeAll;
  try {
    KataGoCommandLine cmd("Run selfplay games with a fixed model and dump per-position policy and value surprise stats to csv");
    cmd.addConfigFileArg("","selfplay config file");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> outputArg("","output","Csv file to write one row per recorded turn to",true,string(),"FILE");
    TCLAP::ValueArg<int> numGamesArg("","num-games","Number of games to run",true,0,"N");
    TCLAP::SwitchArg reanalyzeAllArg("","reanalyze-all","Instead of disabling reanalysis, reanalyze every cheap-search position, so each such row records both the cheap search's (before) and the full search's (after) surprise stats");
    cmd.add(outputArg);
    cmd.add(numGamesArg);
    cmd.add(reanalyzeAllArg);
    cmd.parseArgs(args);
    outputFile = outputArg.getValue();
    numGamesTotal = numGamesArg.getValue();
    reanalyzeAll = reanalyzeAllArg.getValue();
    if(numGamesTotal <= 0)
      throw StringError("-num-games must be positive");

    cmd.getConfig(cfg);
    modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand seedRand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.write("Selfplay surprise stats dump");
  logger.write("Model: " + modelFile);

  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  const SearchParams baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER);
  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  if(reanalyzeAll) {
    //Reanalyze every cheap-search position, so that every cheap-search row records both the cheap search's
    //surprise stats (the ones that would drive reanalysis selection, preserved as origPolicySurprise and
    //origValueSurprise) and the full search's stats, without any selection bias in which positions get both.
    //Note that the post-reanalysis value surprise of a turn also absorbs the value changes of other reanalyzed
    //turns after it, via the smoothed forward-outcome track, so unlike policy surprise it is not a pure
    //cheap-vs-full comparison of this turn's search alone.
    logger.write("Reanalyzing all cheap-search positions to record before and after surprise stats");
    playSettings.useReanalyze = true;
    playSettings.reanalyzeProp = 1.0;
  }
  else {
    //Force no reanalysis, so that every turn's recorded stats are those of the search the game actually played
    //with - exactly the stats that would drive reanalysis selection - rather than partly overwritten by
    //reanalysis searches.
    if(playSettings.useReanalyze && playSettings.reanalyzeProp > 0.0)
      logger.write("Config has reanalyzeProp > 0, forcing it to 0 so that recorded stats are the pre-reanalysis ones");
    playSettings.reanalyzeProp = 0.0;
  }

  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);

  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  Setup::initializeSession(cfg);
  unique_ptr<SelfplayModel> model;
  {
    const int expectedConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads;
    const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    const bool useMuZero = cfg.contains("useMuZero") && cfg.getBool("useMuZero");
    if(useMuZero) {
      auto bundle = make_shared<MuZero::ModelBundle>(modelFile);
      if(Rules::parseGame(bundle->game()) != gameRunner->getGameInitializer()->getGame())
        throw StringError("MuZero model game differs from the configured selfplay game");
      if(maxBoardXSizeUsed > bundle->posLen() || maxBoardYSizeUsed > bundle->posLen())
        throw StringError("MuZero model canvas must contain all configured boards");
      const bool exact = defaultRequireExactNNLen && maxBoardXSizeUsed == bundle->posLen() && maxBoardYSizeUsed == bundle->posLen();
      model = Setup::initializeMuZeroSelfplayModel(modelFile,modelFile,std::move(bundle),cfg,logger,seedRand,
        expectedConcurrentEvals,exact,0,Setup::SETUP_FOR_SELFPLAY);
    }
    else {
      model = make_unique<SelfplayModel>(Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
        maxBoardXSizeUsed,maxBoardYSizeUsed,Setup::MaxBatchSizeRequest::requireFromConfig(),defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_SELFPLAY
      ));
    }
  }
  logger.write("Loaded neural net");

  cfg.warnUnusedKeys(cerr,&logger);

  ofstream out;
  FileUtils::open(out,outputFile);
  out.precision(9);
  out << "gameIdx,gameHash,game,xSize,ySize,komi,pdaWhite,numExtraBlack,mode,hitTurnLimit,gameNumMoves"
      << ",outcomeWhiteWin,outcomeWhiteDraw,outcomeWhiteLoss,outcomeWhiteNoResult"
      << ",turnIdx,turnAfterStart,wasCheapSearch,unreducedNumVisits,targetWeight"
      << ",policySurprise,valueSurprise,policyEntropy,searchEntropy"
      << ",rawNNWhiteWinLoss,mctsWhiteWin,mctsWhiteDraw,mctsWhiteLoss,mctsWhiteNoResult"
      << ",wasReanalyzed,origPolicySurprise,origValueSurprise,origNumVisits"
      << "\n";

  if(!std::atomic_is_lock_free(&sigReceived))
    throw StringError("sigReceived is not lock free, signal-quitting mechanism will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::mutex writeMutex;
  std::atomic<int64_t> numGamesStarted(0);
  int64_t numGamesWritten = 0;
  ForkData* forkData = new ForkData();

  auto gameLoop = [
    &gameRunner,&logger,&numGamesStarted,&forkData,numGamesTotal,&baseParams,&gameSeedBase,
    &model,&writeMutex,&out,&numGamesWritten
  ](int threadIdx) {
    (void)threadIdx;
    auto shouldStopFunc = []() noexcept {
      return sigReceived.load();
    };
    Rand thisLoopSeedRand;
    while(true) {
      if(sigReceived.load())
        break;
      int64_t gameIdx = numGamesStarted.fetch_add(1,std::memory_order_acq_rel);
      if(gameIdx >= numGamesTotal)
        break;

      MatchPairer::BotSpec botSpec;
      botSpec.botIdx = 0;
      botSpec.botName = model->getModelName();
      botSpec.nnEval = model->boardEvaluator();
      botSpec.muZeroEval = model->muZeroEvaluator();
      botSpec.baseParams = baseParams;

      string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
      FinishedGameData* gameData = gameRunner->runGame(
        seed, botSpec, botSpec, forkData, NULL, logger,
        shouldStopFunc, nullptr, nullptr, nullptr, nullptr
      );
      //Happens when interrupted by the signal handler mid-game
      if(gameData == NULL)
        break;

      {
        std::lock_guard<std::mutex> lock(writeMutex);
        writeSurpriseStatsRows(out, gameIdx, *gameData);
        out.flush();
        numGamesWritten += 1;
        logger.write(
          "Finished game " + Global::int64ToString(gameIdx) + " (" +
          Global::int64ToString(numGamesWritten) + "/" + Global::int64ToString(numGamesTotal) + " written)"
        );
      }
      delete gameData;
    }
  };

  vector<std::thread> threads;
  for(int i = 0; i<numGameThreads; i++)
    threads.push_back(std::thread(gameLoop,i));
  for(int i = 0; i<numGameThreads; i++)
    threads[i].join();

  out.close();
  logger.write("Wrote stats for " + Global::int64ToString(numGamesWritten) + " games to " + outputFile);

  delete forkData;
  delete gameRunner;
  model.reset();
  return 0;
}
