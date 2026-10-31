#include "tests.h"
#include "testsearchcommon.h"
#include "../core/test.h"
#include "../dataio/trainingwrite.h"
#include "../search/searchnode.h"
#include "../search/evalcache.h"
#include "../program/setup.h"
#include "../program/play.h"

#include <memory>

namespace {
void near(double actual, double expected) { testAssert(std::fabs(actual-expected) < 1e-6); }

ConfigParser fixedKomiConfig() {
  return ConfigParser(std::map<std::string,std::string>{
    {"koRules","SITUATIONAL"}, {"scoringRules","AREA"}, {"taxRules","NONE"},
    {"multiStoneSuicideLegals","false"}, {"hasButtons","false"},
    {"bSizes","7"}, {"bSizeRelProbs","1"}, {"komiMean","7.5"}
  });
}

void fixedKomiGames(NNEvaluator* nn, Logger& logger) {
  for(int mode = 0; mode < 3; mode++) {
    ConfigParser cfg;
    cfg.overrideKey("sekiForkHackProb","0");
    bool rejected = false;
    try {
      if(mode == 0) PlaySettings::loadForSelfplay(cfg,false);
      else if(mode == 1) PlaySettings::loadForGatekeeper(cfg);
      else PlaySettings::loadForMatch(cfg);
    }
    catch(const StringError& e) { rejected = std::string(e.what()).find("sekiForkHackProb is excluded") != std::string::npos; }
    testAssert(rejected);
  }
  for(const std::string key : {"komiStdev", "komiAllowIntegerProb", "handicapCompensateKomiProb",
                               "forkCompensateKomiProb", "sgfCompensateKomiProb", "komiBigStdevProb",
                               "komiBigStdev", "komiBiggerStdevProb", "komiBiggerStdev",
                               "handicapKomiInterpZeroProb", "sgfKomiInterpZeroProb", "komiAuto"}) {
    auto cfg = fixedKomiConfig();
    cfg.overrideKey(key,"0");
    bool rejected = false;
    try { GameInitializer init(cfg,logger); }
    catch(const StringError& e) { rejected = std::string(e.what()).find("removed setting: " + key) != std::string::npos; }
    testAssert(rejected);
  }
  for(const std::string key : {"compensateAfterPolicyInitProb", "compensateKomiVisits", "flipKomiProbWhenNoCompensate",
                               "fancyKomiVarying", "minAsymmetricCompensateKomiProb", "dynamicSelfKomiBonusMin",
                               "dynamicSelfKomiBonusMax", "dynamicSelfKomiWinLossMin", "dynamicSelfKomiWinLossMax",
                               "estimateLeadVisits", "estimateLeadProb"}) {
    for(int mode = 0; mode < 3; mode++) {
      ConfigParser cfg;
      cfg.overrideKey(key,"0");
      bool rejected = false;
      try {
        if(mode == 0) PlaySettings::loadForSelfplay(cfg,false);
        else if(mode == 1) PlaySettings::loadForGatekeeper(cfg);
        else PlaySettings::loadForMatch(cfg);
      }
      catch(const StringError& e) { rejected = std::string(e.what()).find("removed setting: " + key) != std::string::npos; }
      testAssert(rejected);
    }
  }
  auto invalidCfg = fixedKomiConfig();
  invalidCfg.overrideKey("komiMean","7.2");
  bool rejected = false;
  try { GameInitializer init(invalidCfg,logger); }
  catch(const StringError& e) { rejected = std::string(e.what()).find("half-integer") != std::string::npos; }
  testAssert(rejected);

  bool sawPolicyInit = false;
  for(float fixedKomi : {-5.0f,0.0f,7.5f}) for(int kind = 0; kind < 4; kind++) {
    auto cfg = fixedKomiConfig();
    cfg.overrideKey("komiMean",Global::floatToString(fixedKomi));
    cfg.overrideKey("handicapProb",kind == 1 ? "1" : "0");
    cfg.overrideKey("numExtraBlackFixed","2");
    GameInitializer init(cfg,logger,"fixed-komi-init");
    near(init.createRules().komi,fixedKomi);
    Board sourceBoard(7,7);
    Rules sourceRules = Rules::getTrompTaylorish();
    sourceRules.komi = -3.5;
    BoardHistory sourceHist(sourceBoard,P_BLACK,sourceRules,0,BoardHistoryModes(false,false));
    sourceHist.makeBoardMoveAssumeLegal(sourceBoard,Location::getLoc(3,3,7),P_BLACK,NULL);
    InitialPosition fork(sourceBoard,sourceHist,P_WHITE,true,false,0.75);
    Sgf::PositionSample sample;
    Sgf::PositionSample::writePosOfHist(sample,sourceHist,P_WHITE);
    PlaySettings settings;
    settings.initGamesWithPolicy = true;
    settings.policyInitAreaProp = 0.15;
    settings.startPosesPolicyInitAreaProp = 0.15;
    settings.normalAsymmetricPlayoutProb = 1;
    settings.handicapAsymmetricPlayoutProb = 1;
    settings.maxAsymmetricRatio = 4;
    for(int i = 0; i < 4; i++) {
      Board board;
      BoardHistory hist;
      Player pla;
      int handicap = -1;
      OtherGameProperties props;
      init.createGame(board,pla,hist,handicap,kind == 3 ? &fork : NULL,settings,props,
                      kind == 2 ? &sample : NULL,BoardHistoryModes(false,false));
      const float expectedKomi = kind == 3 ? sourceRules.komi : fixedKomi;
      near(hist.rules.komi,expectedKomi);
      testAssert(handicap == (kind == 1 ? 2 : 0));
      testAssert(props.isFork == (kind == 3) && props.isSgfPos == (kind == 2));
      testAssert(props.allowPolicyInit == (kind != 3));
      if(kind != 3) testAssert(props.playoutDoublingAdvantage > 0 && props.playoutDoublingAdvantage < 2);
      MatchPairer::BotSpec spec;
      spec.botIdx = 0;
      spec.botName = "fixed-komi";
      spec.nnEval = nn;
      spec.baseParams.maxVisits = 32;
      Rand rand("fixed-komi-play-" + Global::intToString(i));
      auto game = std::unique_ptr<FinishedGameData>(Play::runGame(
        board,pla,hist,handicap,spec,spec,"fixed-komi-search",false,true,logger,false,false,
        2,[]() noexcept { return false; },nullptr,settings,props,rand,nullptr,nullptr
      ));
      near(game->startHist.rules.komi,expectedKomi);
      near(game->endHist.rules.komi,expectedKomi);
      if(kind == 3) testAssert(game->startHist.moveHistory.size() == sourceHist.moveHistory.size());
      else if(game->startHist.moveHistory.size() > hist.moveHistory.size()) sawPolicyInit = true;
      testAssert(game->numExtraBlack == handicap);
    }
  }
  testAssert(sawPolicyInit);
}

void randomizedNoResultUtility(Logger& logger) {
  auto cfg = fixedKomiConfig();
  cfg.overrideKey("numSearchThreads","1");
  cfg.overrideKey("noResultUtilityForWhite","0.2");
  near(Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER).noResultUtilityForWhite,0.2);
  cfg.overrideKey("noResultStdev","0.2");
  GameInitializer init(cfg,logger,"nr-utility-init");
  GameInitializer duplicate(cfg,logger,"nr-utility-init");
  PlaySettings settings;
  Board board;
  BoardHistory hist;
  Player pla;
  int handicap;
  OtherGameProperties props;
  double sum = 0;
  bool positive = false, negative = false;
  for(int i = 0; i < 256; i++) {
    SearchParams params, repeated;
    init.createGame(board,pla,hist,handicap,params,NULL,settings,props,NULL,BoardHistoryModes(false,false));
    duplicate.createGame(board,pla,hist,handicap,repeated,NULL,settings,props,NULL,BoardHistoryModes(false,false));
    const double u = params.noResultUtilityForWhite;
    near(u,repeated.noResultUtilityForWhite);
    near(params.drawEquivalentWinsForWhite,0.5);
    testAssert(u >= -0.6 && u <= 0.6); // Three-sigma truncated Gaussian, without clipping atoms.
    positive |= u > 0.1;
    negative |= u < -0.1;
    sum += u;
  }
  testAssert(positive && negative && std::fabs(sum/256) < 0.05);
  // Reject samples outside [-1,1], including when the configured mean is near an endpoint.
  for(double mean : {-0.95,0.95}) for(int i = 0; i < 32; i++) {
    SearchParams params;
    params.noResultUtilityForWhite = mean;
    init.createGame(board,pla,hist,handicap,params,NULL,settings,props,NULL,BoardHistoryModes(false,false));
    testAssert(params.noResultUtilityForWhite >= -1 && params.noResultUtilityForWhite <= 1);
  }
  bool rejected = false;
  try { init.createGame(board,pla,hist,handicap,NULL,settings,props,NULL,BoardHistoryModes(false,false)); }
  catch(const StringError&) { rejected = true; }
  testAssert(rejected); // Non-selfplay entry cannot silently discard randomized search parameters.
  cfg.overrideKey("noResultStdev","0");
  GameInitializer fixed(cfg,logger,"nr-utility-fixed");
  SearchParams params;
  params.noResultUtilityForWhite = -0.3;
  fixed.createGame(board,pla,hist,handicap,params,NULL,settings,props,NULL,BoardHistoryModes(false,false));
  near(params.noResultUtilityForWhite,-0.3);
}

void excludedSearchOptions() {
  for(const std::string key : {"staticScoreUtilityFactor", "dynamicScoreUtilityFactor", "dynamicScoreCenterZeroWeight",
                               "dynamicScoreCenterScale", "rootEndingBonusPoints", "antiMirror", "genmoveAntiMirror",
                               "rootPruneUselessMoves", "conservativePass", "fillDameBeforePass",
                               "enablePassingHacks", "enableMorePassingHacks", "avoidMYTDaggerHack",
                               "avoidMYTDaggerHackPla", "friendlyPass", "cleanupBeforePass", "friendlyPassOk",
                               "searchFactorAfterOnePass", "searchFactorAfterTwoPass"}) {
    ConfigParser cfg;
    cfg.overrideKey("numSearchThreads","1");
    cfg.overrideKey(key,"0");
    bool rejected = false;
    try { Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP,false); }
    catch(const StringError& e) { rejected = std::string(e.what()).find("excluded from EtaZero") != std::string::npos; }
    testAssert(rejected);
  }
  for(const std::string spec : {"{\"friendlyPassOk\":true}", "koSIMPLEscoreAREAtaxNONEsui0fpok1"}) {
    Rules rules;
    testAssert(!Rules::tryParseRules(spec,rules));
  }
}

void passingRulesAndSearch(NNEvaluator* nn, Logger& logger) {
  for(const std::string ruleName : {"tromp-taylor", "chinese", "japanese"}) {
    for(int threads : {1,3}) for(bool graph : {false,true}) {
      Rules rules = Rules::parseRules(ruleName);
      rules.komi = 0;
      Board board(7,7);
      BoardHistory hist(board,P_BLACK,rules,0,BoardHistoryModes(false,false));
      const int phases = rules.scoringRule == Rules::SCORING_TERRITORY ? 3 : 1;
      SearchParams params;
      params.maxVisits = 32;
      params.numThreads = threads;
      params.useGraphSearch = graph;
      Search search(params,nn,&logger,"rule-passes");
      for(int phase = 0; phase < phases; phase++) {
        hist.makeBoardMoveAssumeLegal(board,Board::PASS_LOC,P_BLACK,NULL);
        testAssert(!hist.isGameFinished && hist.passWouldEndPhase(board,P_WHITE));
        testAssert(hist.passWouldEndGame(board,P_WHITE) == (phase == phases-1));
        // Before the real phase end, features retain the previous pass and the phase-end flag.
        float spatial[NNInputs::NUM_FEATURES_SPATIAL_V7 * 49];
        float global[NNInputs::NUM_FEATURES_GLOBAL_V7];
        MiscNNInputParams inputs;
        NNInputs::fillRowV7(board,hist,P_WHITE,inputs,7,7,false,spatial,global);
        near(global[0],1); near(global[14],1);

        search.setPosition(P_WHITE,board,hist);
        std::vector<int> avoid(Board::MAX_ARR_SIZE,1); avoid[Board::PASS_LOC] = 0;
        search.setAvoidMoveUntilByLoc({},avoid);
        search.runWholeSearch(P_WHITE);
        testAssert(search.getChosenMoveLoc() == Board::PASS_LOC);
        // A frequently visited pass no longer prevents recording root evaluations.
        EvalCacheTable rootCache(2);
        rootCache.update(Hash128(),search.rootNode,1);
        testAssert(rootCache.find(Hash128())->cacheWeight > 0);
        near(rootCache.find(Hash128())->avgDraw,search.rootNode->stats.drawValueAvg.load());
        auto children = search.rootNode->getChildren();
        testAssert(children.iterateAndCountChildren() == 1);
        auto* child = children[0].getIfAllocated();
        if(phase == phases-1) {
          testAssert(child->getNNOutput() == NULL && !child->forceNonTerminal);
          ReportedSearchValues values;
          testAssert(search.getNodeValues(child,values));
          near(values.drawValue,1); near(values.winValue,0); near(values.lossValue,0);
          near(values.noResultValue,0); near(values.utility,0);
        }
        else {
          testAssert(child->getNNOutput() != NULL);
        }
        hist.makeBoardMoveAssumeLegal(board,Board::PASS_LOC,P_WHITE,NULL);
        testAssert(hist.isGameFinished == (phase == phases-1));
        testAssert(hist.encorePhase == std::min(phase+1,phases-1));
      }
    }
  }
}

void trainingTargets() {
  TrainingWriteBuffers buffers(3,8,NNInputs::NUM_FEATURES_SPATIAL_V3,NNInputs::NUM_FEATURES_GLOBAL_V3,5,5,false);
  Board board(5,5);
  BoardHistory hist(board,P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes(false,false));
  std::vector<PolicyTargetMove> policy{PolicyTargetMove(Board::PASS_LOC,1)};
  std::vector<QValueTargets> q(2);
  q[0].targets.emplace_back(Location::getLoc(0,0,5),0.25f,9);
  q[0].targets.emplace_back(Board::PASS_LOC,-0.5f,40000);
  testAssert(buffers.qValueTargetsNCMove.shape[1] == 2);
  NNRawStats raw;
  raw.whiteWinLoss = 0; raw.policyEntropy = 0;
  Rand rand("four-result-targets");
  // Current distribution differs in every component; the last entry is a one-hot terminal result.
  // The five target groups use terminal-only, three source TD horizons, and immediate search.
  const int channels[5][4] = {{0,70,1,2},{4,71,5,6},{8,72,9,10},{12,73,13,14},{16,74,17,18}};
  const double now[5] = {0,1.0/5.4,1.0/2.4,1.0/1.4,1};
  const double current[4] = {0.2,0.4,0.1,0.3};
  for(int outcome = 0; outcome < 4; outcome++) for(Player pla : {P_WHITE,P_BLACK}) {
    std::vector<ValueTargets> values(2);
    values[0].win = 0.2f; values[0].draw = 0.4f; values[0].loss = 0.1f; values[0].noResult = 0.3f;
    values[1].win = outcome == 0; values[1].draw = outcome == 1;
    values[1].loss = outcome == 2; values[1].noResult = outcome == 3;
    buffers.addRow(board,hist,pla,hist,hist,0,1.0f,100,&policy,NULL,0.5,1.0,1.0,
                   values,q,0,1.0f,1.0f,raw,NULL,NULL,false,0,0.5,
                   C_EMPTY,0.0,Hash128(),{},false,0,FinishedGameData::MODE_NORMAL,NULL,rand,ReanalysisData());
    const float* row = buffers.globalTargetsNC.data + (buffers.curRows-1)*80;
    for(int channel : {3,7,11,15,19,20,21,29,58}) near(row[channel],0);
    const int16_t* qrow = buffers.qValueTargetsNCMove.data + (buffers.curRows-1)*2*26;
    const int sign = pla == P_WHITE ? 1 : -1;
    near(qrow[0],sign*8000); near(qrow[25],-sign*16000);
    near(qrow[26],9); near(qrow[51],32000);
    for(int pos = 1; pos < 25; pos++) {
      near(qrow[pos],0); near(qrow[26+pos],0);
    }
    for(int group = 0; group < 5; group++) {
      double sum = 0;
      for(int result = 0; result < 4; result++) {
        const int whiteResult = pla == P_BLACK && (result == 0 || result == 2) ? 2-result : result;
        near(row[channels[group][result]],now[group]*current[whiteResult] + (1-now[group])*(whiteResult == outcome));
        sum += row[channels[group][result]];
      }
      near(sum,1);
    }
  }
}

void futurePositionTargets() {
  TrainingWriteBuffers buffers(7,10,NNInputs::NUM_FEATURES_SPATIAL_V7,NNInputs::NUM_FEATURES_GLOBAL_V7,5,5,false);
  testAssert(buffers.valueTargetsNCHW.shape[1] == 2);
  Board board(5,5);
  BoardHistory hist(board,P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes(false,false));
  std::vector<Board> futureBoards(33,board);
  futureBoards[8].setStone(Location::getLoc(1,1,5),P_BLACK);
  futureBoards[32].setStone(Location::getLoc(3,2,5),P_WHITE);
  std::vector<ValueTargets> values(33);
  std::vector<QValueTargets> q(33);
  std::vector<PolicyTargetMove> policy{PolicyTargetMove(Board::PASS_LOC,1)};
  NNRawStats raw{};
  Rand rand("future-position-targets");
  for(Player pla : {P_BLACK,P_WHITE}) for(int kind = 0; kind < 5; kind++) {
    BoardHistory outcomeHist = hist;
    if(kind == 4) {
      outcomeHist.isGameFinished = true;
      outcomeHist.isNoResult = true;
    }
    buffers.addRow(board,hist,pla,hist,outcomeHist,0,1.0f,100,&policy,NULL,0.5,1.0,1.0,
                   values,q,kind == 1 ? 30 : 0,0.4f,1.0f,raw,kind >= 3 ? &board : NULL,
                   kind == 2 ? NULL : &futureBoards,false,0,0.5,C_EMPTY,0.0,Hash128(),{},
                   false,0,FinishedGameData::MODE_NORMAL,NULL,rand,ReanalysisData());
    const int row = buffers.curRows-1;
    near(buffers.globalTargetsNC.data[row*80+33],kind == 2 ? 0 : 1);
    near(buffers.globalTargetsNC.data[row*80+34],0);
    // Retain the source outcome mask, independently of future targets.
    near(buffers.globalTargetsNC.data[row*80+27],kind == 3 ? 0.4 : 0.0);
    near(buffers.globalTargetsNC.data[row*80+20],0.0);
    const int8_t* targets = buffers.valueTargetsNCHW.data + row*2*25;
    for(int pos = 0; pos < 25; pos++) {
      const int blackSign = pla == P_BLACK ? 1 : -1;
      const int nearExpected = kind == 2 ? 0 : kind == 1 ? (pos == 13 ? -blackSign : 0) : (pos == 6 ? blackSign : 0);
      const int farExpected = kind == 2 ? 0 : (pos == 13 ? -blackSign : 0);
      near(targets[pos],nearExpected);
      near(targets[25+pos],farExpected);
    }
  }
}

void searchResults() {
  NeuralNet::globalInitialize();
  Logger logger(nullptr,false,false,false);
  auto nn = std::unique_ptr<NNEvaluator>(TestSearchCommon::startNNEval("/dev/null",logger,"four-results",7,7,0,true,false,false,true,false));
  passingRulesAndSearch(nn.get(),logger);
  fixedKomiGames(nn.get(),logger);
  randomizedNoResultUtility(logger);
  SearchParams params;
  params.maxVisits = 64;
  for(int threads : {1,3}) for(bool graph : {false,true}) for(double nrUtility : {-0.6,0.0,0.6}) {
    params.numThreads = threads;
    params.useGraphSearch = graph;
    params.noResultUtilityForWhite = nrUtility;
    Search search(params,nn.get(),&logger,"results-search");
    // A placement draw is a terminal root with D=1, not half a win and half a loss.
    Board board(3,3);
    BoardHistory hist(board,P_BLACK,Rules::forGame(Rules::Game::STANDARD),0,BoardHistoryModes(false,false));
    for(int i = 0; i < 9; i++)
      hist.makeBoardMoveAssumeLegal(board,Location::getLoc(i%3,i/3,3),hist.presumedNextMovePla,NULL);
    search.setPosition(hist.presumedNextMovePla,board,hist);
    search.runWholeSearch(hist.presumedNextMovePla);
    ReportedSearchValues values;
    testAssert(search.getRootValues(values));
    near(values.winValue,0); near(values.drawValue,1); near(values.lossValue,0);
    near(values.noResultValue,0); near(values.utility,0);
    NodeStats drawnStats(search.rootNode->stats);
    near(drawnStats.drawValueAvg,1);
    EvalCacheTable drawCache(2);
    drawCache.update(Hash128(),search.rootNode,1);
    near(drawCache.find(Hash128())->avgDraw,1);

    // Source triple-ko fixture, one move before a genuine Go no-result.
    board = Board::parseBoard(7,6,"ooooooo\noxo.o.o\nx.xoxox\nxxxxxxx\nooooooo\n.......\n");
    Rules rules = Rules::getTrompTaylorish();
    rules.koRule = Rules::KO_SIMPLE; rules.multiStoneSuicideLegal = false;
    hist = BoardHistory(board,P_BLACK,rules,0,BoardHistoryModes(false,false));
    const int cycle[6][2] = {{3,1},{1,2},{5,1},{3,2},{1,1},{5,2}};
    for(int i = 0; i < 11; i++) {
      Loc loc = Location::getLoc(cycle[i%6][0],cycle[i%6][1],7);
      testAssert(hist.isLegal(board,loc,hist.presumedNextMovePla));
      hist.makeBoardMoveAssumeLegal(board,loc,hist.presumedNextMovePla,NULL);
    }
    testAssert(!hist.isGameFinished);
    Loc finalMove = Location::getLoc(5,2,7);
    search.setPosition(P_WHITE,board,hist);
    std::vector<int> avoid(Board::MAX_ARR_SIZE,1); avoid[finalMove] = 0;
    search.setAvoidMoveUntilByLoc({},avoid);
    search.runWholeSearch(P_WHITE);
    testAssert(search.getChosenMoveLoc() == finalMove);
    auto children = search.rootNode->getChildren();
    testAssert(children.iterateAndCountChildren() == 1);
    auto* child = children[0].getIfAllocated();
    testAssert(search.getNodeValues(child,values));
    near(values.winValue,0); near(values.drawValue,0); near(values.lossValue,0);
    near(values.noResultValue,1); near(values.utility,nrUtility);
    near(child->stats.utilitySqAvg.load(),nrUtility*nrUtility);
    // A single root evaluation must retain the NN's mixed outcome distribution in its utility.
    SearchParams singleParams = params;
    singleParams.maxVisits = 1;
    singleParams.numThreads = 1;
    Search single(singleParams,nn.get(),&logger,"nr-mixed-value");
    Board empty(7,7);
    BoardHistory emptyHist(empty,P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes(false,false));
    single.setPosition(P_BLACK,empty,emptyHist);
    single.runWholeSearch(P_BLACK);
    ReportedSearchValues mixedValues;
    testAssert(single.getRootValues(mixedValues));
    const NNOutput* mixedNN = single.rootNode->getNNOutput();
    near(mixedValues.utility,mixedNN->whiteWinProb-mixedNN->whiteLossProb+nrUtility*mixedNN->whiteNoResultProb);

    // Cache and node copies must carry D independently from NR.
    NodeStats stats(child->stats);
    near(stats.drawValueAvg,0); near(stats.noResultValueAvg,1);
    EvalCacheTable cache(2);
    cache.update(Hash128(),child,1);
    auto cached = cache.find(Hash128());
    testAssert(cached != nullptr);
    near(cached->avgDraw,0); near(cached->avgNoResult,1);
    auto a = std::make_shared<NNOutput>(*search.rootNode->getNNOutput());
    a->whiteWinProb = .2f; a->whiteDrawProb = .4f; a->whiteLossProb = .1f; a->whiteNoResultProb = .3f;
    auto b = std::make_shared<NNOutput>(*a);
    b->whiteWinProb = .1f; b->whiteDrawProb = .7f; b->whiteLossProb = .2f; b->whiteNoResultProb = 0;
    NNOutput averaged(std::vector<std::shared_ptr<NNOutput>>{a,b});
    NNOutput copied;
    copied = averaged;
    near(copied.whiteWinProb,.15); near(copied.whiteDrawProb,.55);
    near(copied.whiteLossProb,.15); near(copied.whiteNoResultProb,.15);
    ReportedSearchValues mixed(0.1,0.4,0.3,0.1,1,1);
    near(mixed.winValue,0.2); near(mixed.lossValue,0.1); near(mixed.drawValue,0.4); near(mixed.noResultValue,0.3);
  }
  nn.reset();
  NeuralNet::globalCleanup();
}
}

void Tests::runFourResultTests() {
  std::cout << "Running W/D/L/NR targets, terminal search, and cache tests" << std::endl;
  trainingTargets();
  futurePositionTargets();
  excludedSearchOptions();
  searchResults();
}
