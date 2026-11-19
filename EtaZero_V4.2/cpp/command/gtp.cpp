#include "../core/global.h"
#include "../core/commandloop.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../core/timer.h"
#include "../core/datetime.h"
#include "../core/makedir.h"
#include "../core/test.h"
#include "../dataio/sgf.h"
#include "../game/xiangqi.h"
#include "../search/searchnode.h"
#include "../search/asyncbot.h"
#include "../search/patternbonustable.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../program/selfplaymanager.h"
#include "../tests/tests.h"
#include "../command/commandline.h"
#include "../main.h"
#include <thread>

using namespace std;

static const vector<string> knownCommands = {
  //Basic GTP commands
  "protocol_version",
  "name",
  "version",
  "known_command",
  "list_commands",
  "quit",

  //GTP extension - specify "boardsize X:Y" or "boardsize X Y" for non-square sizes
  //rectangular_boardsize is an alias for boardsize, intended to make it more evident that we have such support
  "boardsize",
  "rectangular_boardsize",

  "clear_board",
  "set_position",
  "komi",
  //GTP extension - get KataGo's current komi setting
  "get_komi",
  "play",
  "undo",

  //GTP extension - specify rules
  "kata-get-rules",
  "kata-set-rule",
  "kata-set-rules",

  //Get or change a few limited params dynamically
  "kata-get-models",
  "kata-get-param",
  "kata-set-param",
  "kata-list-params",
  "kgs-rules",

  "genmove",
  "kata-search", //Doesn't actually make the move
  "kata-search_cancellable", //Doesn't actually make the move, also any command or newline cancels the search.

  "genmove_debug", //Prints additional info to stderr
  "kata-search_debug", //Prints additional info to stderr, doesn't actually make the move

  //Clears neural net cached evaluations and bot search tree, allows fresh randomization
  "clear_cache",

  "showboard",
  "kata-get-state", // Read-only structured board, rules, moves and legal actions.
  "kata-get-history", // Read source BoardHistory positions without modifying the engine.
  "kata-suggest-opening", // Cancellable native balancing; returns moves without playing them.
  "kata-get-analysis", // Read the existing search using the source JSON serializer.
  "kata-start-pondering", // Restart pondering after a client reads the board.
  "fixed_handicap",
  "place_free_handicap",
  "set_free_handicap",

  "time_settings",
  "kgs-time_settings",
  "time_left",
  //KataGo extensions for time settings
  "kata-list_time_settings",
  "kata-time_settings",

  "final_score",
  "final_status_list",

  "loadsgf",
  "printsgf",

  //GTP extensions for board analysis
  // "genmove_analyze",
  "lz-genmove_analyze",
  "kata-genmove_analyze",
  "kata-search_analyze",  //Doesn't actually make the move
  "kata-search_analyze_cancellable",  //Doesn't actually make the move, also any command or newline cancels the search.
  // "analyze",
  "lz-analyze",
  "kata-analyze",

  //Display raw neural net evaluations
  "kata-raw-nn",
  "kata-raw-human-nn",

  //Misc other stuff
  "cputime",
  "gomill-cpu_time",
  "kata-benchmark",

  //Some debug commands
  "kata-debug-print-tc",
  "debug_moves",

  //Stop any ongoing ponder or analyze
  "stop",
};

static nlohmann::json positionJson(const Board& board, const BoardHistory& hist, Player pla, Player perspective, bool includeLegal) {
  nlohmann::json state = {
    {"game",Rules::gameName(hist.rules.game)}, {"rules",hist.rules.toJson()},
    {"boardXSize",board.x_size}, {"boardYSize",board.y_size},
    {"currentPlayer",PlayerIO::playerToStringShort(pla)},
    {"analysisPerspective",PlayerIO::playerToStringShort(perspective == C_EMPTY ? pla : perspective)},
    {"turnNumber",hist.getCurrentTurnNumber()}, {"encorePhase",hist.encorePhase},
    {"isGameFinished",hist.isGameFinished}, {"isNoResult",hist.isNoResult},
    {"isResignation",hist.isResignation},
    {"result",hist.isGameFinished ? nlohmann::json(WriteSgf::gameResultNoSgfTag(hist)) : nlohmann::json(nullptr)}
  };
  state["board"] = nlohmann::json::array();
  state["legalMoves"] = nlohmann::json::array();
  for(int y = 0; y < board.y_size; y++) for(int x = 0; x < board.x_size; x++) {
    Loc loc = Location::getLoc(x,y,board.x_size);
    // Physical coordinates, row-major: 0 empty, 1 first player, -1 second.
    state["board"].push_back(board.colors[loc] == P_BLACK ? 1 : board.colors[loc] == P_WHITE ? -1 : 0);
    if(includeLegal && !hist.rules.isXiangqi() && !hist.isGameFinished && hist.isLegal(board,loc,pla))
      state["legalMoves"].push_back(Location::toString(loc,board));
  }
  if(includeLegal && hist.rules.isGo() && !hist.isGameFinished && hist.isLegal(board,Board::PASS_LOC,pla))
    state["legalMoves"].push_back("pass");
  if(hist.rules.isXiangqi()) {
    const auto& position = hist.getXiangqiState();
    state["fen"] = position.fen();
    state["pieces"] = position.pieces();
    if(includeLegal && !hist.isGameFinished)
      for(int action : position.legalActions())
        state["legalMoves"].push_back(Xiangqi::actionToString(action));
  }
  if(!includeLegal) state.erase("legalMoves");
  return state;
}

static bool tryParseLoc(const string& s, const Board& b, const Rules& rules, Loc& loc) {
  if(rules.isXiangqi()) {
    int action = Xiangqi::parseAction(s);
    if(action < 0) return false;
    loc = Xiangqi::actionToLoc(action);
    return true;
  }
  return Location::tryOfString(s,b,loc);
}

static void writePV(ostream& out, const AnalysisData& data, const Search* search) {
  if(!search->getRootHist().rules.isXiangqi()) {
    data.writePV(out,search->getRootBoard());
    return;
  }
  for(size_t i = 0; i < data.pv.size(); i++) {
    if(i > 0) out << " ";
    out << search->moveToString(data.pv[i]);
  }
}

//Filter out all double newlines, since double newline terminates GTP command responses
static string filterDoubleNewlines(const string& s) {
  string filtered;
  for(int i = 0; i<s.length(); i++) {
    if(i > 0 && s[i-1] == '\n' && s[i] == '\n')
      continue;
    filtered += s[i];
  }
  return filtered;
}

static bool timeIsValid(const double& time) {
  if(isnan(time) || time < 0.0 || time > TimeControls::MAX_USER_INPUT_TIME)
    return false;
  return true;
}
static bool timeIsValidAllowNegative(const double& time) {
  if(isnan(time) || time < -TimeControls::MAX_USER_INPUT_TIME || time > TimeControls::MAX_USER_INPUT_TIME)
    return false;
  return true;
}

static double parseTime(const vector<string>& args, int argIdx, const string& description) {
  double time = 0.0;
  if(args.size() <= argIdx || !Global::tryStringToDouble(args[argIdx],time))
    throw StringError("Expected float for " + description + " as argument " + Global::intToString(argIdx));
  if(!timeIsValid(time))
    throw StringError(description + " is an invalid value: " + args[argIdx]);
  return time;
}
static double parseTimeAllowNegative(const vector<string>& args, int argIdx, const string& description) {
  double time = 0.0;
  if(args.size() <= argIdx || !Global::tryStringToDouble(args[argIdx],time))
    throw StringError("Expected float for " + description + " as argument " + Global::intToString(argIdx));
  if(!timeIsValidAllowNegative(time))
    throw StringError(description + " is an invalid value: " + args[argIdx]);
  return time;
}
static int parseByoYomiStones(const vector<string>& args, int argIdx) {
  int byoYomiStones = 0;
  if(args.size() <= argIdx || !Global::tryStringToInt(args[argIdx],byoYomiStones))
    throw StringError("Expected int for byo-yomi overtime stones as argument " + Global::intToString(argIdx));
  if(byoYomiStones < 0 || byoYomiStones > 1000000)
    throw StringError("byo-yomi overtime stones is an invalid value: " + args[argIdx]);
  return byoYomiStones;
}
static int parseByoYomiPeriods(const vector<string>& args, int argIdx) {
  int byoYomiPeriods = 0;
  if(args.size() <= argIdx || !Global::tryStringToInt(args[argIdx],byoYomiPeriods))
    throw StringError("Expected int for byo-yomi overtime periods as argument " + Global::intToString(argIdx));
  if(byoYomiPeriods < 0 || byoYomiPeriods > 1000000)
    throw StringError("byo-yomi overtime periods is an invalid value: " + args[argIdx]);
  return byoYomiPeriods;
}

//Assumes that stones are worth 15 points area and 14 points territory, and that 7 komi is fair
static double initialBlackAdvantage(const BoardHistory& hist) {
  BoardHistory histCopy = hist;
  histCopy.setAssumeMultipleStartingBlackMovesAreHandicap(true);
  int handicapStones = histCopy.computeNumHandicapStones();
  if(handicapStones <= 1)
    return 7.0 - hist.rules.komi;

  //Subtract one since white gets the first move afterward
  int extraBlackStones = handicapStones - 1;
  double stoneValue = hist.rules.scoringRule == Rules::SCORING_AREA ? 15.0 : 14.0;
  double whiteHandicapBonus = 0.0;
  if(hist.rules.whiteHandicapBonusRule == Rules::WHB_N)
    whiteHandicapBonus += handicapStones;
  else if(hist.rules.whiteHandicapBonusRule == Rules::WHB_N_MINUS_ONE)
    whiteHandicapBonus += handicapStones-1;

  return stoneValue * extraBlackStones + (7.0 - hist.rules.komi - whiteHandicapBonus);
}

static double getBoardSizeScaling(const Board& board) {
  return pow(19.0 * 19.0 / (double)(board.x_size * board.y_size), 0.75);
}
static double getPointsThresholdForHandicapGame(double boardSizeScaling) {
  return std::max(4.0 / boardSizeScaling, 2.0);
}

static bool noWhiteStonesOnBoard(const Board& board) {
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      if(board.colors[loc] == P_WHITE)
        return false;
    }
  }
  return true;
}

static void updateDynamicPDAHelper(
  const Board& board, const BoardHistory& hist,
  const double dynamicPlayoutDoublingAdvantageCapPerOppLead,
  const vector<double>& recentWinLossValues,
  double& desiredDynamicPDAForWhite
) {
  (void)board;
  if(!hist.rules.isGo() || dynamicPlayoutDoublingAdvantageCapPerOppLead <= 0.0) {
    desiredDynamicPDAForWhite = 0.0;
  }
  else {
    double boardSizeScaling = getBoardSizeScaling(board);
    double pdaScalingStartPoints = getPointsThresholdForHandicapGame(boardSizeScaling);
    double initialBlackAdvantageInPoints = initialBlackAdvantage(hist);
    Player disadvantagedPla = initialBlackAdvantageInPoints >= 0 ? P_WHITE : P_BLACK;
    double initialAdvantageInPoints = std::fabs(initialBlackAdvantageInPoints);
    if(initialAdvantageInPoints < pdaScalingStartPoints || board.x_size <= 7 || board.y_size <= 7) {
      desiredDynamicPDAForWhite = 0.0;
    }
    else {
      double desiredDynamicPDAForDisadvantagedPla =
        (disadvantagedPla == P_WHITE) ? desiredDynamicPDAForWhite : -desiredDynamicPDAForWhite;

      //What increment to adjust desiredPDA at.
      //Power of 2 to avoid any rounding issues.
      const double increment = 0.125;

      //Hard cap of 2.75 in this parameter, since more extreme values start to reach into values without good training.
      double pdaCap = std::min(
        2.75,
        dynamicPlayoutDoublingAdvantageCapPerOppLead *
        (initialAdvantageInPoints - pdaScalingStartPoints) * boardSizeScaling
      );
      pdaCap = round(pdaCap / increment) * increment;

      //No history, or literally no white stones on board? Then this is a new game or a newly set position
      if(recentWinLossValues.size() <= 0 || noWhiteStonesOnBoard(board)) {
        //Just use the cap
        desiredDynamicPDAForDisadvantagedPla = pdaCap;
      }
      else {
        double winLossValue = recentWinLossValues[recentWinLossValues.size()-1];
        //Convert to perspective of disadvantagedPla
        if(disadvantagedPla == P_BLACK)
          winLossValue = -winLossValue;

        //Keep winLossValue between 5% and 25%, subject to available caps.
        if(winLossValue < -0.9)
          desiredDynamicPDAForDisadvantagedPla = desiredDynamicPDAForDisadvantagedPla + 0.125;
        else if(winLossValue > -0.5)
          desiredDynamicPDAForDisadvantagedPla = desiredDynamicPDAForDisadvantagedPla - 0.125;

        desiredDynamicPDAForDisadvantagedPla = std::max(desiredDynamicPDAForDisadvantagedPla, 0.0);
        desiredDynamicPDAForDisadvantagedPla = std::min(desiredDynamicPDAForDisadvantagedPla, pdaCap);
      }

      desiredDynamicPDAForWhite = (disadvantagedPla == P_WHITE) ? desiredDynamicPDAForDisadvantagedPla : -desiredDynamicPDAForDisadvantagedPla;
    }
  }
}

static bool shouldResign(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  const vector<double>& recentWinLossValues,
  const double resignThreshold,
  const int resignConsecTurns,
  const double resignMinMovesPerBoardArea
) {
  double initialBlackAdvantageInPoints = hist.rules.isGo() ? initialBlackAdvantage(hist) : 0.0;

  int minTurnForResignation = 0;
  if(initialBlackAdvantageInPoints > 0.9 && pla == P_WHITE) {
    //Play at least some moves no matter what
    minTurnForResignation = 1 + board.x_size * board.y_size / 5;

  }
  if(minTurnForResignation < resignMinMovesPerBoardArea * board.x_size * board.y_size)
    minTurnForResignation = (int)(resignMinMovesPerBoardArea * board.x_size * board.y_size);

  if(hist.moveHistory.size() < minTurnForResignation)
    return false;
  if(resignConsecTurns > recentWinLossValues.size())
    return false;

  for(int i = 0; i<resignConsecTurns; i++) {
    double winLossValue = recentWinLossValues[recentWinLossValues.size()-1-i];
    Player resignPlayerThisTurn = C_EMPTY;
    if(winLossValue < resignThreshold)
      resignPlayerThisTurn = P_WHITE;
    else if(winLossValue > -resignThreshold)
      resignPlayerThisTurn = P_BLACK;

    if(resignPlayerThisTurn != pla)
      return false;
  }

  return true;
}

struct GTPEngine {
  GTPEngine(const GTPEngine&) = delete;
  GTPEngine& operator=(const GTPEngine&) = delete;

  const string nnModelFile;
  const string humanModelFile;
  const bool useMuZero;
  const bool humanIsMuZero;
  const bool assumeMultipleStartingBlackMovesAreHandicap;
  const int analysisPVLen;
  const bool preventEncore;
  const bool autoAvoidPatterns;

  const double dynamicPlayoutDoublingAdvantageCapPerOppLead;
  bool staticPDATakesPrecedence;
  double normalAvoidRepeatedPatternUtility;
  double handicapAvoidRepeatedPatternUtility;

  std::shared_ptr<const MuZero::ModelBundle> muZeroBundle;
  std::shared_ptr<const MuZero::ModelBundle> humanMuZeroBundle;
  std::unique_ptr<SelfplayModel> primaryModel;
  std::unique_ptr<SelfplayModel> humanModel;
  MuZero::PositionEvaluator* muZeroEval;
  MuZero::PositionEvaluator* humanMuZeroEval;
  NNEvaluator* nnEval;
  NNEvaluator* humanEval;
  AsyncBot* bot;
  Rules currentRules; //Should always be the same as the rules in bot, if bot is not NULL.

  //Stores the params we want to be using during genmoves or analysis
  SearchParams genmoveParams;
  SearchParams analysisParams;
  bool isGenmoveParams;

  TimeControls bTimeControls;
  TimeControls wTimeControls;

  //This move history doesn't get cleared upon consecutive moves by the same side, and is used
  //for undo, whereas the one in search does.
  Board initialBoard;
  Player initialPla;
  vector<Move> moveHistory;

  vector<double> recentWinLossValues;
  double lastSearchFactor;
  double desiredDynamicPDAForWhite;
  std::unique_ptr<PatternBonusTable> patternBonusTable;

  double delayMoveScale;
  double delayMoveMax;

  Player perspective;

  Rand gtpRand;

  ClockTimer genmoveTimer;
  double genmoveTimeSum;
  std::atomic<int> genmoveExpectedId;

  //Positions during this game when genmove was called
  std::vector<Sgf::PositionSample> genmoveSamples;
  std::thread openingThread;
  std::atomic<bool> openingStop{false};

  GTPEngine(
    const string& modelFile, const string& hModelFile, bool isMuZero, bool isHumanMuZero,
    const SearchParams& initialGenmoveParams, const SearchParams& initialAnalysisParams,
    const Rules& initialRules,
    bool assumeMultiBlackHandicap, bool prevtEncore, bool autoPattern,
    double dynamicPDACapPerOppLead, bool staticPDAPrecedence,
    double normAvoidRepeatedPatternUtility, double hcapAvoidRepeatedPatternUtility,
    double delayScale, double delayMax,
    Player persp, int pvLen,
    std::unique_ptr<PatternBonusTable>&& pbTable
  )
    :nnModelFile(modelFile),
     humanModelFile(hModelFile),
     useMuZero(isMuZero),
     humanIsMuZero(isHumanMuZero),
     assumeMultipleStartingBlackMovesAreHandicap(assumeMultiBlackHandicap),
     analysisPVLen(pvLen),
     preventEncore(prevtEncore),
     autoAvoidPatterns(autoPattern),
     dynamicPlayoutDoublingAdvantageCapPerOppLead(dynamicPDACapPerOppLead),
     staticPDATakesPrecedence(staticPDAPrecedence),
     normalAvoidRepeatedPatternUtility(normAvoidRepeatedPatternUtility),
     handicapAvoidRepeatedPatternUtility(hcapAvoidRepeatedPatternUtility),
     muZeroEval(nullptr),
     humanMuZeroEval(nullptr),
     nnEval(NULL),
     humanEval(NULL),
     bot(NULL),
     currentRules(initialRules),
     genmoveParams(initialGenmoveParams),
     analysisParams(initialAnalysisParams),
     isGenmoveParams(true),
     bTimeControls(),
     wTimeControls(),
     initialBoard(),
     initialPla(P_BLACK),
     moveHistory(),
     recentWinLossValues(),
     lastSearchFactor(1.0),
     desiredDynamicPDAForWhite(0.0),
     patternBonusTable(std::move(pbTable)),
     delayMoveScale(delayScale),
     delayMoveMax(delayMax),
     perspective(persp),
     gtpRand(),
     genmoveTimer(),
     genmoveTimeSum(0.0),
     genmoveExpectedId(0),
     genmoveSamples()
  {
  }

  ~GTPEngine() {
    stopAndWait();
    delete bot;
    primaryModel.reset();
    humanModel.reset();
  }

  void stopAndWait() {
    openingStop.store(true);
    if(openingThread.joinable()) openingThread.join();
    // Invalidate any ongoing genmove
    int expectedSearchId = (genmoveExpectedId.load() + 1) & 0x3FFFFFFF;
    genmoveExpectedId.store(expectedSearchId);
    bot->stopAndWait();
  }

  void suggestOpening(const BalancedOpening::Settings& settings,
                      const std::function<void(const string&,bool)>& respond) {
    stopAndWait();
    Search* search = bot->getSearchStopAndWait();
    const Board& board = bot->getRootBoard();
    const BoardHistory& hist = bot->getRootHist();
    if(!hist.rules.isPlacement()) throw StringError("Balanced openings require Gomoku or Hex");
    if(hist.isGameFinished || bot->getRootPla() != P_BLACK || !hist.moveHistory.empty() || !board.isEmpty())
      throw StringError("Balanced openings require an empty ongoing board with Black to move");
    if(!primaryModel->isNeuralNetEnabled()) throw StringError("Balanced openings require a neural network model");
    openingStop.store(false);
    openingThread = std::thread([this,search,settings,respond]() {
      try {
        Board candidate = search->getRootBoard();
        BoardHistory history = search->getRootHist();
        Player pla = search->getRootPla();
        const auto result = BalancedOpening::initialize(search,search,candidate,history,pla,gtpRand,settings,
          [this]() noexcept { return openingStop.load(); });
        const bool cancelled = result == BalancedOpening::Result::INTERRUPTED || openingStop.load();
        nlohmann::json report;
        report["status"] = cancelled ? "cancelled" :
                           result == BalancedOpening::Result::APPLIED ? "complete" : "skipped";
        report["game"] = Rules::gameName(history.rules.game);
        report["moves"] = nlohmann::json::array();
        report["parameters"] = {{"balanceOpeningDropPow",settings.dropPow}};
        if(history.rules.game == Rules::Game::HEX) report["parameters"]["minAcceptRate"] = settings.minAcceptRate;
        if(!cancelled && result == BalancedOpening::Result::APPLIED) {
          for(const Move& move : history.moveHistory)
            report["moves"].push_back({PlayerIO::playerToStringShort(move.pla),Location::toString(move.loc,candidate)});
        }
        // The live board and search tree remain unchanged, even if the client
        // cancels after a completed suggestion has already reached stdout.
        respond(report.dump(),false);
      }
      catch(const std::exception& error) { respond(error.what(),true); }
    });
  }

  Rules getCurrentRules() {
    return currentRules;
  }

  BoardHistoryModes getHistoryModes() const {
    return muZeroEval ? Search::resolveHistoryModes(genmoveParams,*muZeroEval)
                      : Search::resolveHistoryModes(genmoveParams,nnEval);
  }

  void checkBoardSize(int xSize, int ySize) const {
    if(currentRules.isXiangqi() && (xSize != Xiangqi::WIDTH || ySize != Xiangqi::HEIGHT))
      throw StringError("Xiangqi requires a 9 x 10 board");
    if(currentRules.isXiangqi() && primaryModel && primaryModel->getRequireExactNNLen() &&
       (xSize != primaryModel->getNNXLen() || ySize != primaryModel->getNNYLen()))
      throw StringError("Xiangqi's rectangular board requires requireMaxBoardSize=false with a square NN canvas");
    if(currentRules.isPlacement() && xSize != ySize)
      throw StringError("Placement games require a square board");
    const auto& fixedBundle = useMuZero ? muZeroBundle : humanMuZeroBundle;
    if(fixedBundle && (xSize > fixedBundle->posLen() || ySize > fixedBundle->posLen() ||
       (primaryModel && primaryModel->getRequireExactNNLen() &&
        (xSize != fixedBundle->posLen() || ySize != fixedBundle->posLen()))))
      throw StringError("Board does not fit the MuZero model canvas");
  }

  void clearStatsForNewGame() {
  }

  //Specify -1 for the sizes for a default
  void setOrResetBoardSize(ConfigParser& cfg, Logger& logger, Rand& seedRand, int boardXSize, int boardYSize, bool loggingToStderr) {
    if(useMuZero && !muZeroBundle) {
      muZeroBundle = make_shared<MuZero::ModelBundle>(nnModelFile);
      if(Rules::parseGame(muZeroBundle->game()) != currentRules.game)
        throw StringError("MuZero model.json game must match the configured game");
    }
    if(humanIsMuZero && !humanMuZeroBundle) {
      humanMuZeroBundle = make_shared<MuZero::ModelBundle>(humanModelFile);
      if(Rules::parseGame(humanMuZeroBundle->game()) != currentRules.game)
        throw StringError("Secondary MuZero model.json game must match the configured game");
      if(muZeroBundle && humanMuZeroBundle->posLen() != muZeroBundle->posLen())
        throw StringError("Secondary MuZero must use the same canvas as the primary model");
    }
    const auto& fixedBundle = useMuZero ? muZeroBundle : humanMuZeroBundle;
    bool wasDefault = false;
    if(boardXSize == -1 || boardYSize == -1) {
      boardXSize = currentRules.isXiangqi() ? Xiangqi::WIDTH : fixedBundle ? fixedBundle->posLen() : Board::DEFAULT_LEN;
      boardYSize = currentRules.isXiangqi() ? Xiangqi::HEIGHT : fixedBundle ? fixedBundle->posLen() : Board::DEFAULT_LEN;
      wasDefault = true;
    }

    checkBoardSize(boardXSize,boardYSize);
    bool defaultRequireExactNNLen = !fixedBundle && !currentRules.isXiangqi();
    int nnXLen = fixedBundle ? fixedBundle->posLen() : currentRules.isXiangqi() ? Xiangqi::HEIGHT : boardXSize;
    int nnYLen = fixedBundle ? fixedBundle->posLen() : boardYSize;

    if(cfg.contains("gtpForceMaxNNSize") && cfg.getBool("gtpForceMaxNNSize")) {
      defaultRequireExactNNLen = false;
      if(!fixedBundle) {
        nnXLen = Board::MAX_LEN;
        nnYLen = Board::MAX_LEN;
      }
    }

    // Resolve the configured buffer before replacing an existing bot. A fixed
    // maximum buffer can override the requested dimensions, and exact-size
    // evaluators cannot search a smaller board inside that buffer.
    const int expectedConcurrentEvals = std::max(genmoveParams.numThreads, analysisParams.numThreads);
    auto checkEvaluatorSize = [&](const string& path, const char* label) {
      const auto options = Setup::readEvaluatorOptions(path,0,cfg,logger,seedRand,expectedConcurrentEvals,
        nnXLen,nnYLen,Setup::MaxBatchSizeRequest::fromConcurrency(),defaultRequireExactNNLen,
        false,Setup::SETUP_FOR_GTP,true);
      if(!wasDefault && (boardXSize > options.nnXLen || boardYSize > options.nnYLen ||
         (options.requireExactNNLen && (boardXSize != options.nnXLen || boardYSize != options.nnYLen))))
        throw StringError(string(label) + " NN buffer " + Global::intToString(options.nnXLen) + " x " +
          Global::intToString(options.nnYLen) + (options.requireExactNNLen ? " requires an exact board size" : " is too small for the board"));
      return options;
    };
    const auto primaryOptions = checkEvaluatorSize(nnModelFile,"Primary");
    if(!humanModelFile.empty()) checkEvaluatorSize(humanModelFile,"Secondary");

    //If the neural net is wrongly sized, we need to create or recreate it
    if(primaryModel == nullptr || !(primaryOptions.nnXLen == primaryModel->getNNXLen() &&
                                   primaryOptions.nnYLen == primaryModel->getNNYLen())) {

      if(primaryModel != nullptr) {
        assert(bot != NULL);
        bot->stopAndWait();
        delete bot;
        primaryModel.reset();
        humanModel.reset();
        bot = NULL;
        nnEval = NULL;
        muZeroEval = nullptr;
        humanMuZeroEval = nullptr;
        humanEval = NULL;
        logger.write("Cleaned up old neural net and bot");
      }

      const bool disableFP16 = false;
      const string expectedSha256 = "";
      if(useMuZero) {
        primaryModel = Setup::initializeMuZeroSelfplayModel(nnModelFile,nnModelFile,muZeroBundle,
          cfg,logger,seedRand,expectedConcurrentEvals,defaultRequireExactNNLen,0,Setup::SETUP_FOR_GTP,
          Setup::MaxBatchSizeRequest::fromConcurrency());
        muZeroEval = primaryModel->muZeroEvaluator();
      }
      else {
        primaryModel = make_unique<SelfplayModel>(Setup::initializeNNEvaluator(
          nnModelFile,nnModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
          nnXLen,nnYLen,Setup::MaxBatchSizeRequest::fromConcurrency(),defaultRequireExactNNLen,disableFP16,
          Setup::SETUP_FOR_GTP
        ));
        nnEval = primaryModel->boardEvaluator();
      }
      checkBoardSize(boardXSize,boardYSize);
      logger.write("Loaded neural net with nnXLen " + Global::intToString(primaryModel->getNNXLen()) + " nnYLen " + Global::intToString(primaryModel->getNNYLen()));
      if(humanModelFile != "") {
        if(humanIsMuZero) {
          humanModel = Setup::initializeMuZeroSelfplayModel(humanModelFile,humanModelFile,humanMuZeroBundle,
            cfg,logger,seedRand,expectedConcurrentEvals,defaultRequireExactNNLen,0,Setup::SETUP_FOR_GTP,
            Setup::MaxBatchSizeRequest::fromConcurrency());
          humanMuZeroEval = humanModel->muZeroEvaluator();
        }
        else {
          humanModel = make_unique<SelfplayModel>(Setup::initializeNNEvaluator(
            humanModelFile,humanModelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
            nnXLen,nnYLen,Setup::MaxBatchSizeRequest::fromConcurrency(),defaultRequireExactNNLen,disableFP16,
            Setup::SETUP_FOR_GTP
          ));
          humanEval = humanModel->boardEvaluator();
        }
        logger.write("Loaded human SL net with nnXLen " + Global::intToString(humanModel->getNNXLen()) + " nnYLen " + Global::intToString(humanModel->getNNYLen()));
        if(!humanModel->requiresSGFMetadata()) {
          string warning;
          warning += "WARNING: Human model was not trained from SGF metadata to vary by rank! Did you pass the wrong model for -human-model?\n";
          logger.write(warning);
          if(!loggingToStderr)
            cerr << warning << endl;
        }
      }

      {
        bool rulesWereSupported;
        primaryModel->getSupportedRules(currentRules,rulesWereSupported);
        if(!rulesWereSupported) {
          throw StringError("Rules " + currentRules.toJsonStringNoKomi() + " from config file " + cfg.getFileName() + " are NOT supported by neural net");
        }
        if(humanModel) {
          humanModel->getSupportedRules(currentRules,rulesWereSupported);
          if(!rulesWereSupported)
            throw StringError("Rules from config are not supported by secondary neural net");
        }
      }
    }

    //Point games default to the NN dimensions. Xiangqi always uses its physical
    //9 x 10 board, independent of padding in the neural canvas.
    if(wasDefault && !currentRules.isXiangqi()) {
      boardXSize = primaryModel->getNNXLen();
      boardYSize = primaryModel->getNNYLen();
    }

    //If the bot is wrongly sized, we need to create or recreate the bot
    if(bot == NULL || bot->getRootBoard().x_size != boardXSize || bot->getRootBoard().y_size != boardYSize) {
      if(bot != NULL) {
        assert(bot != NULL);
        bot->stopAndWait();
        delete bot;
        bot = NULL;
        logger.write("Cleaned up old bot");
      }

      logger.write("Initializing board with boardXSize " + Global::intToString(boardXSize) + " boardYSize " + Global::intToString(boardYSize));
      if(!loggingToStderr)
        cerr << ("Initializing board with boardXSize " + Global::intToString(boardXSize) + " boardYSize " + Global::intToString(boardYSize)) << endl;

      string searchRandSeed;
      if(cfg.contains("searchRandSeed"))
        searchRandSeed = cfg.getString("searchRandSeed");
      else
        searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

      if(muZeroEval)
        bot = humanMuZeroEval ? new AsyncBot(genmoveParams,*muZeroEval,*humanMuZeroEval,&logger,searchRandSeed)
                             : new AsyncBot(genmoveParams,*muZeroEval,humanEval,&logger,searchRandSeed);
      else
        bot = humanMuZeroEval ? new AsyncBot(genmoveParams,nnEval,*humanMuZeroEval,&logger,searchRandSeed)
                             : new AsyncBot(genmoveParams,nnEval,humanEval,&logger,searchRandSeed);
      bot->setCopyOfExternalPatternBonusTable(patternBonusTable);
      isGenmoveParams = true;

      Board board(boardXSize,boardYSize);
      Player pla = P_BLACK;
      BoardHistory hist = currentRules.isXiangqi() ? BoardHistory(Xiangqi::State(),getHistoryModes()) :
        BoardHistory(board,pla,currentRules,0,getHistoryModes());
      if(currentRules.isXiangqi()) board = hist.getXiangqiState().colorBoard();
      vector<Move> newMoveHistory;
      setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
      clearStatsForNewGame();
    }
  }

  void setPatternBonusTable(std::unique_ptr<PatternBonusTable>&& pbTable) {
    patternBonusTable = std::move(pbTable);
    if(bot != nullptr)
      bot->setCopyOfExternalPatternBonusTable(patternBonusTable);
  }

  void setPositionAndRules(Player pla, const Board& board, const BoardHistory& h, const Board& newInitialBoard, Player newInitialPla, const vector<Move>& newMoveHistory) {
    BoardHistory hist(h);
    //Ensure we always have this value correct
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);

    currentRules = hist.rules;
    bot->setPosition(pla,board,hist);
    initialBoard = newInitialBoard;
    initialPla = newInitialPla;
    moveHistory = newMoveHistory;
    recentWinLossValues.clear();
    updateDynamicPDA();
  }

  void clearBoard() {
    testAssert(bot->getRootHist().rules == currentRules);
    int newXSize = bot->getRootBoard().x_size;
    int newYSize = bot->getRootBoard().y_size;
    Board board(newXSize,newYSize);
    Player pla = P_BLACK;
    BoardHistory hist = currentRules.isXiangqi() ? BoardHistory(Xiangqi::State(),getHistoryModes()) :
      BoardHistory(board,pla,currentRules,0,getHistoryModes());
    if(currentRules.isXiangqi()) board = hist.getXiangqiState().colorBoard();
    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
  }

  bool setPosition(const vector<Move>& initialStones) {
    testAssert(bot->getRootHist().rules == currentRules);
    int newXSize = bot->getRootBoard().x_size;
    int newYSize = bot->getRootBoard().y_size;
    Board board(newXSize,newYSize);
    if(currentRules.isGo()) {
      if(!board.setStonesFailIfNoLibs(initialStones)) return false;
    }
    else {
      for(const Move& move : initialStones) {
        if(board.colors[move.loc] != C_EMPTY) return false;
        board.colors[move.loc] = move.pla;
      }
      board.regenChainsFromColors();
    }

    //Sanity check
    for(int i = 0; i<initialStones.size(); i++) {
      if(board.colors[initialStones[i].loc] != initialStones[i].pla) {
        return false;
      }
    }
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0,getHistoryModes());
    hist.setInitialTurnNumber(board.numStonesOnBoard()); //Heuristic to guess at what turn this is
    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
    return true;
  }

  void updateKomiIfNew(float newKomi) {
    bot->setKomiIfNew(newKomi);
    currentRules.komi = newKomi;
  }

  void updateDynamicPDA() {
    updateDynamicPDAHelper(
      bot->getRootBoard(),bot->getRootHist(),
      dynamicPlayoutDoublingAdvantageCapPerOppLead,
      recentWinLossValues,
      desiredDynamicPDAForWhite
    );
  }

  bool play(Loc loc, Player pla) {
    testAssert(bot->getRootHist().rules == currentRules);
    bool suc = bot->makeMove(loc,pla,preventEncore);
    if(suc)
      moveHistory.emplace_back(loc,pla);
    return suc;
  }

  BoardHistory initialHistory(const Rules& rules) const {
    if(!rules.isXiangqi()) return BoardHistory(initialBoard,initialPla,rules,0,getHistoryModes());
    // copyToInitial retains the SGF origin's full repetition/check/chase history.
    BoardHistory hist = bot->getRootHist().copyToInitial();
    hist.setModes(getHistoryModes());
    return hist;
  }

  bool undo() {
    if(moveHistory.size() <= 0)
      return false;
    testAssert(bot->getRootHist().rules == currentRules);

    vector<Move> moveHistoryCopy = moveHistory;

    Board undoneBoard = initialBoard;
    BoardHistory undoneHist = initialHistory(currentRules);
    undoneHist.setInitialTurnNumber(bot->getRootHist().initialTurnNumber);
    vector<Move> emptyMoveHistory;
    setPositionAndRules(initialPla,undoneBoard,undoneHist,initialBoard,initialPla,emptyMoveHistory);

    for(int i = 0; i<moveHistoryCopy.size()-1; i++) {
      Loc moveLoc = moveHistoryCopy[i].loc;
      Player movePla = moveHistoryCopy[i].pla;
      bool suc = play(moveLoc,movePla);
      testAssert(suc);
    }
    return true;
  }

  bool setRulesNotIncludingKomi(Rules newRules, string& error) {
    testAssert(primaryModel != nullptr);
    testAssert(bot->getRootHist().rules == currentRules);
    newRules.komi = currentRules.komi;
    if(newRules.isXiangqi() != currentRules.isXiangqi()) {
      error = "Cannot switch between Xiangqi and other games in this GTP engine";
      return false;
    }

    bool rulesWereSupported;
    primaryModel->getSupportedRules(newRules,rulesWereSupported);
    if(!rulesWereSupported) {
      error = "Rules " + newRules.toJsonStringNoKomi() + " are not supported by this neural net version";
      return false;
    }
    if(humanModel) {
      humanModel->getSupportedRules(newRules,rulesWereSupported);
      if(!rulesWereSupported) {
        error = "Rules are not supported by the secondary neural net";
        return false;
      }
    }

    vector<Move> moveHistoryCopy = moveHistory;

    Board board = initialBoard;
    BoardHistory hist = initialHistory(newRules);
    hist.setInitialTurnNumber(bot->getRootHist().initialTurnNumber);
    vector<Move> emptyMoveHistory;
    setPositionAndRules(initialPla,board,hist,initialBoard,initialPla,emptyMoveHistory);

    for(int i = 0; i<moveHistoryCopy.size(); i++) {
      Loc moveLoc = moveHistoryCopy[i].loc;
      Player movePla = moveHistoryCopy[i].pla;
      bool suc = play(moveLoc,movePla);

      //Because internally we use a highly tolerant test, we don't expect this to actually trigger
      //even if a rules change did make some earlier moves illegal. But this check simply futureproofs
      //things in case we ever do
      if(!suc) {
        error = "Could not make the rules change, some earlier moves in the game would now become illegal.";
        return false;
      }
    }
    return true;
  }

  //Re-replay the current game from the beginning under the currently resolved
  //BoardHistoryModes. Used when a runtime params change (kata-set-param)
  //flips a mode - the bot's own rootHistory gets re-stamped by setParams, but re-stamping
  //deliberately does not re-adjudicate game state recorded under the old modes (e.g. an
  //automatically detected game end), whereas replaying recomputes everything as if the engine had
  //been using the new modes all along. This also keeps the state consistent with what a later
  //rebuild-and-replay (undo, kata-set-rules) would produce.
  void rereplayGameForHistoryModesChange() {
    testAssert(bot->getRootHist().rules == currentRules);
    vector<Move> moveHistoryCopy = moveHistory;

    Board board = initialBoard;
    BoardHistory hist = initialHistory(currentRules);
    hist.setInitialTurnNumber(bot->getRootHist().initialTurnNumber);
    vector<Move> emptyMoveHistory;
    setPositionAndRules(initialPla,board,hist,initialBoard,initialPla,emptyMoveHistory);

    for(int i = 0; i<moveHistoryCopy.size(); i++) {
      //Tolerant internal replay, same as undo() - the mode never affects move legality.
      bool suc = play(moveHistoryCopy[i].loc,moveHistoryCopy[i].pla);
      testAssert(suc);
    }
  }

  void ponder() {
    if(currentRules.isXiangqi() && bot->getRootHist().isGameFinished) return;
    bot->ponder(lastSearchFactor);
  }

  struct GenmoveArgs {
    double searchFactorWhenWinningThreshold;
    double searchFactorWhenWinning;
    bool ogsChatToStderr;
    bool allowResignation;
    double resignThreshold;
    int resignConsecTurns;
    double resignMinMovesPerBoardArea;
    bool logSearchInfo;
    bool logSearchInfoForChosenMove;
    PrintTreeOptions logSearchTreeOptions;
    bool debug;
  };

  struct AnalyzeArgs {
    bool analyzing = false;
    bool lz = false;
    bool kata = false;
    int minMoves = 0;
    int maxMoves = 10000000;
    bool showRootInfo = false;
    bool showPVVisits = false;
    bool showPVEdgeVisits = false;
    bool showNoResultValue = false;
    bool jsonOutput = false;
    double secondsPerReport = TimeControls::UNLIMITED_TIME_DEFAULT;
    vector<int> avoidMoveUntilByLocBlack;
    vector<int> avoidMoveUntilByLocWhite;
    vector<Loc> focusMoves;
    vector<double> focusWeights;
    double focusProb = 0.0;
  };

  void filterZeroVisitMoves(const AnalyzeArgs& args, vector<AnalysisData> buf) {
    //Avoid printing moves that have 0 visits, unless we need them
    //These should already be sorted so that 0-visit moves only appear at the end.
    int keptMoves = 0;
    for(int i = 0; i<buf.size(); i++) {
      if(buf[i].childVisits > 0 || keptMoves < args.minMoves)
        buf[keptMoves++] = buf[i];
    }
    buf.resize(keptMoves);
  }

  std::function<void(const Search* search)> getAnalyzeCallback(Player pla, const AnalyzeArgs& args) {
    std::function<void(const Search* search)> callback;
    if(args.jsonOutput) {
      // This is the same report as the native JSON analysis engine. The GTP
      // framing and final play/cancelled response keep their source semantics.
      return [args,this](const Search* search) {
        nlohmann::json report;
        if(search->getAnalysisJson(perspective,analysisPVLen,preventEncore,true,
                                  args.showPVVisits || args.showPVEdgeVisits,args.showNoResultValue,report)) {
          report["turnNumber"] = search->getRootHist().getCurrentTurnNumber();
          cout << report.dump() << endl;
        }
      };
    }
    //lz-analyze
    if(args.lz && !args.kata) {
      //Avoid capturing anything by reference except [this], since this will potentially be used
      //asynchronously and called after we return
      callback = [args,pla,this](const Search* search) {
        vector<AnalysisData> buf;
        bool duplicateForSymmetries = true;
        search->getAnalysisData(buf,args.minMoves,false,analysisPVLen,duplicateForSymmetries);
        filterZeroVisitMoves(args,buf);
        if(buf.size() > args.maxMoves)
          buf.resize(args.maxMoves);
        if(buf.size() <= 0)
          return;

        const Board board = search->getRootBoard();
        for(int i = 0; i<buf.size(); i++) {
          if(i > 0)
            cout << " ";
          const AnalysisData& data = buf[i];
          double winrate = 0.5 * (1.0 + data.winLossValue);
          double lcb = PlayUtils::getHackedLCBForWinrate(search,data,pla);
          if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK)) {
            winrate = 1.0-winrate;
            lcb = 1.0 - lcb;
          }
          cout << "info";
          cout << " move " << search->moveToString(data.move);
          cout << " visits " << data.childVisits;
          cout << " winrate " << round(winrate * 10000.0);
          cout << " prior " << round(data.policyPrior * 10000.0);
          cout << " lcb " << round(lcb * 10000.0);
          cout << " order " << data.order;
          cout << " pv ";
          if(!useMuZero && preventEncore && data.pvContainsPass())
            data.writePVUpToPhaseEnd(cout,board,search->getRootHist(),search->getRootPla());
          else
            writePV(cout,data,search);
          if(args.showPVVisits) {
            cout << " pvVisits ";
            if(!useMuZero && preventEncore && data.pvContainsPass())
              data.writePVVisitsUpToPhaseEnd(cout,board,search->getRootHist(),search->getRootPla());
            else
              data.writePVVisits(cout);
          }
          if(args.showPVEdgeVisits) {
            cout << " pvEdgeVisits ";
            if(!useMuZero && preventEncore && data.pvContainsPass())
              data.writePVEdgeVisitsUpToPhaseEnd(cout,board,search->getRootHist(),search->getRootPla());
            else
              data.writePVEdgeVisits(cout);
          }
        }
        cout << endl;
      };
    }
    //kata-analyze, analyze (sabaki)
    else {
      callback = [args,pla,this](const Search* search) {
        vector<AnalysisData> buf;
        bool duplicateForSymmetries = true;
        search->getAnalysisData(buf,args.minMoves,false,analysisPVLen,duplicateForSymmetries);
        ReportedSearchValues rootVals;
        bool suc = search->getPrunedRootValues(rootVals);
        if(!suc)
          return;
        filterZeroVisitMoves(args,buf);
        if(buf.size() > args.maxMoves)
          buf.resize(args.maxMoves);
        if(buf.size() <= 0)
          return;
        const SearchNode* rootNode = search->getRootNode();

        ostringstream out;
        if(!args.kata) {
          //Hack for sabaki - ensure always showing decimal point. Also causes output to be more verbose with trailing zeros,
          //unfortunately, despite doing not improving the precision of the values.
          out << std::showpoint;
        }

        const Board board = search->getRootBoard();
        for(int i = 0; i<buf.size(); i++) {
          if(i > 0)
            out << " ";
          const AnalysisData& data = buf[i];
          double winrate = 0.5 * (1.0 + data.winLossValue);
          double utility = data.utility;
          //We still hack the LCB for consistency with LZ-analyze
          double lcb = PlayUtils::getHackedLCBForWinrate(search,data,pla);
          ///But now we also offer the proper LCB that KataGo actually uses.
          double utilityLcb = data.lcb;
          if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK)) {
            winrate = 1.0-winrate;
            lcb = 1.0 - lcb;
            utility = -utility;
            utilityLcb = -utilityLcb;
          }
          out << "info";
          out << " move " << search->moveToString(data.move);
          out << " visits " << data.childVisits;
          out << " edgeVisits " << data.numVisits;
          out << " utility " << utility;
          out << " winrate " << winrate;
          if(args.showNoResultValue)
            out << " noResultValue " << data.noResultValue;
          out << " prior " << data.policyPrior;
          out << " lcb " << lcb;
          out << " utilityLcb " << utilityLcb;
          out << " weight " << data.childWeightSum;
          if(data.isSymmetryOf != Board::NULL_LOC)
            out << " isSymmetryOf " << search->moveToString(data.isSymmetryOf);
          out << " order " << data.order;
          out << " pv ";
          if(!useMuZero && preventEncore && data.pvContainsPass())
            data.writePVUpToPhaseEnd(out,board,search->getRootHist(),search->getRootPla());
          else
            writePV(out,data,search);
          if(args.showPVVisits) {
            out << " pvVisits ";
            if(!useMuZero && preventEncore && data.pvContainsPass())
              data.writePVVisitsUpToPhaseEnd(out,board,search->getRootHist(),search->getRootPla());
            else
              data.writePVVisits(out);
          }
          if(args.showPVEdgeVisits) {
            out << " pvEdgeVisits ";
            if(!useMuZero && preventEncore && data.pvContainsPass())
              data.writePVEdgeVisitsUpToPhaseEnd(out,board,search->getRootHist(),search->getRootPla());
            else
              data.writePVEdgeVisits(out);
          }
        }

        if(args.showRootInfo) {
          out << " rootInfo";
          double winrate = 0.5 * (1.0 + rootVals.winLossValue);
          double utility = rootVals.utility;
          if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK)) {
            winrate = 1.0 - winrate;
            utility = -utility;
          }
          out << " visits " << rootVals.visits;
          out << " utility " << utility;
          out << " winrate " << winrate;
          out << " weight " << rootVals.weight;
          if(rootNode != NULL) {
            const NNOutput* nnOutput = rootNode->getNNOutput();
            if(nnOutput != NULL) {
              out << " rawStWrError " << nnOutput->shorttermWinlossError;
              out << " rawVarTimeLeft " << nnOutput->varTimeLeft;
            }
          }
        }

        cout << out.str() << endl;
      };
    }
    return callback;
  }

  void genMove(
    Player pla,
    Logger& logger,
    const GenmoveArgs& gargs,
    const AnalyzeArgs& args,
    bool playChosenMove,
    const std::function<void(const string&, bool)>& printGTPResponse,
    bool& maybeStartPondering
  ) {
    bool onMoveWasCalled = false;
    Loc genmoveMoveLoc = Board::NULL_LOC;
    auto onMove = [&genmoveMoveLoc,&onMoveWasCalled](Loc moveLoc, int searchId, Search* search) noexcept {
      (void)searchId;
      (void)search;
      onMoveWasCalled = true;
      genmoveMoveLoc = moveLoc;
    };
    launchGenMove(pla,gargs,args,onMove);
    bot->waitForSearchToEnd();
    testAssert(onMoveWasCalled);
    string response;
    bool responseIsError = false;
    Loc moveLocToPlay = Board::NULL_LOC;
    handleGenMoveResult(pla,bot->getSearchStopAndWait(),logger,gargs,args,genmoveMoveLoc,response,responseIsError,moveLocToPlay);
    printGTPResponse(response,responseIsError);
    if(moveLocToPlay != Board::NULL_LOC && playChosenMove) {
      bool suc = bot->makeMove(moveLocToPlay,pla,preventEncore);
      if(suc)
        moveHistory.emplace_back(moveLocToPlay,pla);
      if(!suc) {
        ostringstream sout;
        sout << "Engine chose move but makeMove failed" << "\n";
        sout << bot->getRootBoard() << "\n";
        sout << "Pla: " << PlayerIO::playerToString(pla) << "\n";
        sout << "MoveLoc: " << bot->getSearch()->moveToString(moveLocToPlay) << "\n";
        logger.write(sout.str());
        Global::fatalError(sout.str());
      }

      maybeStartPondering = true;
    }
  }

  void genMoveCancellable(
    Player pla,
    Logger& logger,
    const GenmoveArgs& gargs,
    const AnalyzeArgs& args,
    const std::function<void(const string&, bool)>& printGTPResponse
  ) {
    // Make sure to capture things by value unless they're long-lived, since the callback needs to survive past the current scope.
    auto onMove = [pla,&logger,gargs,args,printGTPResponse,this](Loc moveLoc, int searchId, Search* search) {
      string response;
      bool responseIsError = false;
      Loc moveLocToPlay = Board::NULL_LOC;

      // Search invalidated before completion
      if(searchId != genmoveExpectedId.load()) {
        if(args.analyzing)
          response = "play cancelled";
        else
          response = "cancelled";
        printGTPResponse(response,responseIsError);
        return;
      }
      handleGenMoveResult(pla,search,logger,gargs,args,moveLoc,response,responseIsError,moveLocToPlay);
      printGTPResponse(response,responseIsError);
    };
    launchGenMove(pla,gargs,args,onMove);
  }

  void launchGenMove(
    Player pla,
    const GenmoveArgs& gargs,
    const AnalyzeArgs& args,
    const std::function<void(Loc, int, Search*)>& onMove
  ) {
    genmoveTimer.reset();

    primaryModel->clearStats();
    if(humanModel != nullptr)
      humanModel->clearStats();
    TimeControls tc = pla == P_BLACK ? bTimeControls : wTimeControls;

    if(!isGenmoveParams) {
      bot->setParams(genmoveParams);
      isGenmoveParams = true;
    }

    //Update dynamic PDA given whatever the most recent values are, if we're using dynamic
    updateDynamicPDA();

    SearchParams paramsToUse = genmoveParams;
    //Make sure we have the right parameters, in case someone updated params in the meantime.
    if(!staticPDATakesPrecedence) {
      double desiredDynamicPDA =
        (paramsToUse.playoutDoublingAdvantagePla == P_WHITE) ? desiredDynamicPDAForWhite :
        (paramsToUse.playoutDoublingAdvantagePla == P_BLACK) ? -desiredDynamicPDAForWhite :
        (paramsToUse.playoutDoublingAdvantagePla == C_EMPTY && pla == P_WHITE) ? desiredDynamicPDAForWhite :
        (paramsToUse.playoutDoublingAdvantagePla == C_EMPTY && pla == P_BLACK) ? -desiredDynamicPDAForWhite :
        NAN;
      testAssert(!std::isnan(desiredDynamicPDA));

      paramsToUse.playoutDoublingAdvantage = desiredDynamicPDA;
    }

    {
      double avoidRepeatedPatternUtility = normalAvoidRepeatedPatternUtility;
      if(!args.analyzing && bot->getRootHist().rules.isGo()) {
        double initialOppAdvantage = initialBlackAdvantage(bot->getRootHist()) * (pla == P_WHITE ? 1 : -1);
        if(initialOppAdvantage > getPointsThresholdForHandicapGame(getBoardSizeScaling(bot->getRootBoard())))
          avoidRepeatedPatternUtility = handicapAvoidRepeatedPatternUtility;
      }
      paramsToUse.avoidRepeatedPatternUtility = avoidRepeatedPatternUtility;
    }

    if(paramsToUse != bot->getParams())
      bot->setParams(paramsToUse);


    //Play faster when winning
    double searchFactor = PlayUtils::getSearchFactor(gargs.searchFactorWhenWinningThreshold,gargs.searchFactorWhenWinning,paramsToUse,recentWinLossValues,pla);
    lastSearchFactor = searchFactor;

    bot->setAvoidMoveUntilByLoc(args.avoidMoveUntilByLocBlack,args.avoidMoveUntilByLocWhite);
    bot->setRootFocus(args.focusMoves,args.focusWeights,args.focusProb);

    //So that we can tell by the end of the search whether we still care for the result.
    int expectedSearchId = (genmoveExpectedId.load() + 1) & 0x3FFFFFFF;
    genmoveExpectedId.store(expectedSearchId);

    if(args.analyzing) {
      std::function<void(const Search* search)> callback = getAnalyzeCallback(pla,args);


      //Make sure callback happens at least once
      auto onMoveWrapped = [onMove,callback](Loc moveLoc, int searchId, Search* search) {
        callback(search);
        onMove(moveLoc,searchId,search);
      };
      bot->genMoveAsyncAnalyze(pla, expectedSearchId, tc, searchFactor, onMoveWrapped, args.secondsPerReport, args.secondsPerReport, callback);
    }
    else {
      bot->genMoveAsync(pla,expectedSearchId,tc,searchFactor,onMove);
    }
  }

  void handleGenMoveResult(
    Player pla,
    Search* searchBot,
    Logger& logger,
    const GenmoveArgs& gargs,
    const AnalyzeArgs& args,
    Loc moveLoc,
    string& response, bool& responseIsError,
    Loc& moveLocToPlay
  ) {
    response = "";
    responseIsError = false;
    moveLocToPlay = Board::NULL_LOC;

    const Search* search = searchBot;

    bool isLegal = search->isLegalStrict(moveLoc,pla);
    if(moveLoc == Board::NULL_LOC || !isLegal) {
      responseIsError = true;
      response = "genmove returned null location or illegal move";
      ostringstream sout;
      sout << "genmove null location or illegal move!?!" << "\n";
      sout << search->getRootBoard() << "\n";
      sout << "Pla: " << PlayerIO::playerToString(pla) << "\n";
      sout << "MoveLoc: " << search->moveToString(moveLoc) << "\n";
      logger.write(sout.str());
      genmoveTimeSum += genmoveTimer.getSeconds();
      return;
    }

    SearchNode* rootNode = search->rootNode;
    if(rootNode != NULL && delayMoveScale > 0.0 && delayMoveMax > 0.0) {
      int pos = search->getPos(moveLoc);
      const NNOutput* nnOutput = rootNode->getHumanOutput();
      nnOutput = nnOutput != NULL ? nnOutput : rootNode->getNNOutput();
      const float* policyProbs = nnOutput != NULL ? nnOutput->getPolicyProbsMaybeNoised() : NULL;
      if(policyProbs != NULL) {
        double prob = std::max(0.0,(double)policyProbs[pos]);
        double meanWait = 0.5 * delayMoveScale / (prob + 0.10);
        double waitTime = gtpRand.nextGamma(2.0) * meanWait / 2.0;
        waitTime = std::min(waitTime,delayMoveMax);
        waitTime = std::max(waitTime,0.0001);
        std::this_thread::sleep_for(std::chrono::duration<double>(waitTime));
      }
    }

    ReportedSearchValues values;
    double winLossValue;
    {
      values = search->getRootValuesRequireSuccess();
      winLossValue = values.winLossValue;
    }

    //Record data for resignation or adjusting handicap behavior ------------------------
    recentWinLossValues.push_back(winLossValue);

    //Decide whether we should resign---------------------
    bool resigned = gargs.allowResignation && shouldResign(
      search->getRootBoard(),search->getRootHist(),pla,recentWinLossValues,
      gargs.resignThreshold,gargs.resignConsecTurns,gargs.resignMinMovesPerBoardArea
    );

    //Snapshot the time NOW - all meaningful play-related computation time is done, the rest is just
    //output of various things.
    double timeTaken = genmoveTimer.getSeconds();
    genmoveTimeSum += timeTaken;

    //Chatting and logging ----------------------------

    const SearchParams& params = search->searchParams;

    if(gargs.ogsChatToStderr) {
      int64_t visits = search->getRootVisits();
      double winrate = 0.5 * (1.0 + (values.winValue - values.lossValue));
      //Print winrate from desired perspective
      if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && pla == P_BLACK)) {
        winrate = 1.0 - winrate;
      }
      cerr << "MALKOVICH:"
           << "Visits " << visits
           << " Winrate " << Global::strprintf("%.2f%%", winrate * 100.0);
      if(params.playoutDoublingAdvantage != 0.0) {
        cerr << Global::strprintf(
          " (PDA %.2f)",
          search->getRootPla() == getOpp(params.playoutDoublingAdvantagePla) ?
          -params.playoutDoublingAdvantage : params.playoutDoublingAdvantage);
      }
      cerr << " PV ";
      search->printPVForMove(cerr,search->rootNode, moveLoc, analysisPVLen);
      cerr << endl;
    }

    if(gargs.logSearchInfo) {
      ostringstream sout;
      PlayUtils::printGenmoveLog(sout,search,nnEval,moveLoc,timeTaken,perspective,gargs.logSearchInfoForChosenMove,gargs.logSearchTreeOptions,primaryModel.get());
      logger.write(sout.str());
    }
    if(gargs.debug) {
      PlayUtils::printGenmoveLog(cerr,search,nnEval,moveLoc,timeTaken,perspective,gargs.logSearchInfoForChosenMove,gargs.logSearchTreeOptions,primaryModel.get());
    }

    //Actual reporting of chosen move---------------------
    if(resigned)
      response = "resign";
    else
      response = search->moveToString(moveLoc);

    if(autoAvoidPatterns) {
      // Auto pattern expects moveless records using hintloc to contain the move.
      Sgf::PositionSample posSample;
      const BoardHistory& hist = search->getRootHist();
      posSample.game = hist.rules.game;
      if(hist.rules.isXiangqi())
        posSample.xiangqiState = make_shared<const Xiangqi::State>(hist.getXiangqiState());
      posSample.board = search->getRootBoard();
      posSample.nextPla = pla;
      posSample.initialTurnNumber = hist.getCurrentTurnNumber();
      posSample.hintLoc = moveLoc;
      posSample.weight = 1.0;
      genmoveSamples.push_back(posSample);
    }

    if(!resigned) {
      moveLocToPlay = moveLoc;
    }

    if(args.analyzing) {
      response = "play " + response;
    }

    return;
  }

  void clearCache() {
    bot->clearSearch();
    bot->clearEvalCache();
    primaryModel->clearCache();
    if(humanModel != nullptr)
      humanModel->clearCache();
  }

  void placeFixedHandicap(int n, string& response, bool& responseIsError) {
    int xSize = bot->getRootBoard().x_size;
    int ySize = bot->getRootBoard().y_size;
    Board board(xSize,ySize);
    try {
      PlayUtils::placeFixedHandicap(board,n);
    }
    catch(const StringError& e) {
      responseIsError = true;
      response = string(e.what()) + ", try place_free_handicap";
      return;
    }
    testAssert(bot->getRootHist().rules == currentRules);

    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0,getHistoryModes());

    //Also switch the initial player, expecting white should be next.
    hist.clear(board,P_WHITE,currentRules,0);
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
    hist.setInitialTurnNumber(board.numStonesOnBoard()); //Should give more accurate temperaure and time control behavior
    pla = P_WHITE;

    response = "";
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY) {
          response += " " + Location::toString(loc,board);
        }
      }
    }
    response = Global::trim(response);
    (void)responseIsError;

    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
  }

  void placeFreeHandicap(int n, string& response, bool& responseIsError, Rand& rand) {
    stopAndWait();

    //If asked to place more, we just go ahead and only place up to 30, or a quarter of the board
    int xSize = bot->getRootBoard().x_size;
    int ySize = bot->getRootBoard().y_size;
    int maxHandicap = xSize*ySize / 4;
    if(maxHandicap > 30)
      maxHandicap = 30;
    if(n > maxHandicap)
      n = maxHandicap;

    testAssert(bot->getRootHist().rules == currentRules);

    Board board(xSize,ySize);
    Player pla = P_BLACK;
    BoardHistory hist(board,pla,currentRules,0,getHistoryModes());
    double extraBlackTemperature = 0.25;
    PlayUtils::playExtraBlack(bot->getSearchStopAndWait(), n, board, hist, extraBlackTemperature, rand);
    //Also switch the initial player, expecting white should be next.
    hist.clear(board,P_WHITE,currentRules,0);
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
    hist.setInitialTurnNumber(board.numStonesOnBoard()); //Should give more accurate temperaure and time control behavior
    pla = P_WHITE;

    response = "";
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY) {
          response += " " + Location::toString(loc,board);
        }
      }
    }
    response = Global::trim(response);
    (void)responseIsError;

    vector<Move> newMoveHistory;
    setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
    clearStatsForNewGame();
  }

  void analyze(Player pla, const AnalyzeArgs& args) {
    testAssert(args.analyzing);
    if(isGenmoveParams) {
      bot->setParams(analysisParams);
      isGenmoveParams = false;
    }

    std::function<void(const Search* search)> callback = getAnalyzeCallback(pla,args);
    bot->setAvoidMoveUntilByLoc(args.avoidMoveUntilByLocBlack,args.avoidMoveUntilByLocWhite);
    bot->setRootFocus(args.focusMoves,args.focusWeights,args.focusProb);


    double searchFactor = 1e40; //go basically forever
    bot->analyzeAsync(pla, searchFactor, args.secondsPerReport, args.secondsPerReport, callback);
  }



  vector<bool> computeAnticipatedStatuses() {
    stopAndWait();
    return PlayUtils::computeAnticipatedStatusesSimple(bot->getRootBoard(),bot->getRootHist());
  }

  string rawNNBrief(const std::vector<Loc>& branch, int whichSymmetry) {
    if(muZeroEval) return rawMuZeroNNBrief(branch,whichSymmetry);
    if(nnEval == NULL)
      return "";
    ostringstream out;

    Player pla = bot->getRootPla();
    Board board = bot->getRootBoard();
    BoardHistory hist = bot->getRootHist();

    Player prevPla = pla;
    Board prevBoard = board;
    BoardHistory prevHist = hist;
    Loc prevLoc = Board::NULL_LOC;

    for(Loc loc: branch) {
      prevPla = pla;
      prevBoard = board;
      prevHist = hist;
      prevLoc = loc;
      bool suc = hist.makeBoardMoveTolerant(board, loc, pla, false);
      if(!suc)
        return "illegal move sequence";
      pla = getOpp(pla);
    }

    if(hist.rules.isXiangqi() && hist.isGameFinished) return "No policy at a terminal position";
    string policyStr = "Policy: ";
    string wlStr = "White winloss: ";

    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      if(!bot->getRootHist().rules.allowsSymmetry(symmetry))
        continue;
      if(whichSymmetry == NNInputs::SYMMETRY_ALL || whichSymmetry == symmetry) {
        {
          MiscNNInputParams nnInputParams;
          nnInputParams.playoutDoublingAdvantage =
            (analysisParams.playoutDoublingAdvantagePla == C_EMPTY || analysisParams.playoutDoublingAdvantagePla == pla) ?
            analysisParams.playoutDoublingAdvantage : -analysisParams.playoutDoublingAdvantage;
          nnInputParams.symmetry = symmetry;

          NNResultBuf buf;
          bool skipCache = true;
          nnEval->evaluate(board,hist,pla,&analysisParams.humanSLProfile,nnInputParams,buf,skipCache);

          NNOutput* nnOutput = buf.result.get();
          wlStr += Global::strprintf("%.2fc ", 100.0 * (nnOutput->whiteWinProb - nnOutput->whiteLossProb));
        }
        if(prevLoc != Board::NULL_LOC) {
          MiscNNInputParams nnInputParams;
          nnInputParams.playoutDoublingAdvantage =
            (analysisParams.playoutDoublingAdvantagePla == C_EMPTY || analysisParams.playoutDoublingAdvantagePla == prevPla) ?
            analysisParams.playoutDoublingAdvantage : -analysisParams.playoutDoublingAdvantage;
          nnInputParams.symmetry = symmetry;

          NNResultBuf buf;
          bool skipCache = true;
          nnEval->evaluate(prevBoard,prevHist,prevPla,&analysisParams.humanSLProfile,nnInputParams,buf,skipCache);

          NNOutput* nnOutput = buf.result.get();
          int pos = hist.rules.isXiangqi() ? Xiangqi::locToAction(prevLoc) :
            NNPos::locToPos(prevLoc,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
          policyStr += Global::strprintf("%.2f%% ", 100.0 * (nnOutput->policyProbs[pos]));
        }
      }
    }
    return Global::trim(policyStr + "\n" + wlStr);
  }

  string rawMuZeroNNBrief(const std::vector<Loc>& branch, int whichSymmetry) {
    const Board& board = bot->getRootBoard();
    const BoardHistory& hist = bot->getRootHist();
    if(!hist.rules.isGo() && hist.isGameFinished) return "No policy at a terminal position";
    string policyStr = "Policy: ", wlStr = "White winloss: ";
    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      if(!hist.rules.allowsSymmetry(symmetry) || (whichSymmetry != NNInputs::SYMMETRY_ALL && whichSymmetry != symmetry)) continue;
      MiscNNInputParams params;
      params.symmetry = symmetry;
      const Player pla = bot->getRootPla();
      params.playoutDoublingAdvantage =
        (analysisParams.playoutDoublingAdvantagePla == C_EMPTY || analysisParams.playoutDoublingAdvantagePla == pla)
          ? analysisParams.playoutDoublingAdvantage : -analysisParams.playoutDoublingAdvantage;
      MuZero::PositionBuffer buffer;
      auto value = muZeroEval->initial(board,hist,pla,&analysisParams.humanSLProfile,params,buffer,gtpRand,true);
      float prior = 0;
      for(size_t i = 0; i < branch.size(); i++) {
        const int action = hist.rules.isXiangqi() ? Xiangqi::locToAction(branch[i]) :
          NNPos::locToPos(branch[i],board.x_size,value->nnXLen,value->nnYLen);
        const int domain = hist.rules.isXiangqi() ? Xiangqi::NUM_ACTIONS :
          value->nnXLen*value->nnYLen + (hist.rules.isGo() ? 1 : 0);
        if(action < 0 || action >= domain || (i == 0 && value->policyProbs[action] < 0)) return "illegal move sequence";
        prior = value->policyProbs[action];
        value = muZeroEval->recurrent(*value,action,params,buffer);
      }
      wlStr += Global::strprintf("%.2fc ",100.0 * (value->whiteWinProb-value->whiteLossProb));
      if(!branch.empty()) policyStr += Global::strprintf("%.2f%% ",100.0 * prior);
    }
    return Global::trim(policyStr + "\n" + wlStr);
  }

  string rawNN(int whichSymmetry, double policyOptimism, bool useHumanModel, Player nextPla) {
    NNEvaluator* nnEvalToUse = useHumanModel ? humanEval : nnEval;
    MuZero::PositionEvaluator* muEvalToUse = useHumanModel ? humanMuZeroEval : muZeroEval;
    if(nnEvalToUse == NULL && muEvalToUse == nullptr)
      return "";
    ostringstream out;

    for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      if(!bot->getRootHist().rules.allowsSymmetry(symmetry))
        continue;
      if(whichSymmetry == NNInputs::SYMMETRY_ALL || whichSymmetry == symmetry) {
        Board board = bot->getRootBoard();
        BoardHistory hist = bot->getRootHist();
        //If evaluating from a player other than the one naturally to move, rebuild the history so that player
        //is to move (mirrors how the analysis engine handles a player switch). The NN evaluator asserts that
        //nextPla matches the history's presumed next mover. This resets ko/pass history for the position.
        if(nextPla != hist.presumedNextMovePla) {
          board.clearSimpleKoLoc();
          hist.clear(board,nextPla,hist.rules,hist.encorePhase);
        }

        MiscNNInputParams nnInputParams;
        nnInputParams.playoutDoublingAdvantage =
          (analysisParams.playoutDoublingAdvantagePla == C_EMPTY || analysisParams.playoutDoublingAdvantagePla == nextPla) ?
          analysisParams.playoutDoublingAdvantage : -analysisParams.playoutDoublingAdvantage;
        nnInputParams.symmetry = symmetry;
        nnInputParams.policyOptimism = policyOptimism;
        //When evaluating the human model, featurize per its own resolution (which may differ from the
        //main search's modes carried by the history), matching how in-search human evals featurize.
        if(useHumanModel) {
          const auto modes = humanMuZeroEval ? Search::resolveHistoryModes(analysisParams,*humanMuZeroEval)
                                            : Search::resolveHistoryModes(analysisParams,humanEval);
          nnInputParams.passAliveSuicideRulesOverride =
            modes.alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
          nnInputParams.excludeTerritoryAdjAtariOverride =
            modes.excludeTerritoryAdjacentToAtari ? 1 : 0;
        }
        NNResultBuf buf;
        bool skipCache = true;
        if(muEvalToUse) {
          MuZero::PositionBuffer buffer;
          buf.result = muEvalToUse->initial(board,hist,nextPla,&analysisParams.humanSLProfile,nnInputParams,buffer,gtpRand,skipCache);
        }
        else nnEvalToUse->evaluate(board,hist,nextPla,&analysisParams.humanSLProfile,nnInputParams,buf,skipCache);

        NNOutput* nnOutput = buf.result.get();
        out << "symmetry " << symmetry << endl;
        out << "whiteWin " << Global::strprintf("%.6f",nnOutput->whiteWinProb) << endl;
        out << "draw " << Global::strprintf("%.6f",nnOutput->whiteDrawProb) << endl;
        out << "whiteLoss " << Global::strprintf("%.6f",nnOutput->whiteLossProb) << endl;
        out << "noResult " << Global::strprintf("%.6f",nnOutput->whiteNoResultProb) << endl;
        if(!useHumanModel) {
          out << "varTimeLeft " << Global::strprintf("%.3f",nnOutput->varTimeLeft) << endl;
        }
        out << "shorttermWinlossError " << Global::strprintf("%.3f",nnOutput->shorttermWinlossError) << endl;

        out << "policy" << endl;
        if(hist.rules.isXiangqi()) {
          for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) {
            out << Xiangqi::actionToString(action) << " ";
            float prob = nnOutput->policyProbs[action];
            if(prob < 0) out << "NAN";
            else out << Global::strprintf("%.6f",prob);
            out << endl;
          }
          continue;
        }
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
            float prob = nnOutput->policyProbs[pos];
            if(prob < 0)
              out << "    NAN ";
            else
              out << Global::strprintf("%8.6f ", prob);
          }
          out << endl;
        }
        out << "policyPass ";
        {
          int pos = NNPos::locToPos(Board::PASS_LOC,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen);
          float prob = nnOutput->policyProbs[pos];
          if(prob < 0)
            out << "    NAN "; // Probably shouldn't ever happen for pass unles the rules change, but we handle it anyways
          else
            out << Global::strprintf("%8.6f ", prob);
          out << endl;
        }

      }
    }

    return Global::trim(out.str());
  }

  const SearchParams& getGenmoveParams() {
    return genmoveParams;
  }

  void setGenmoveParamsIfChanged(const SearchParams& p) {
    if(genmoveParams != p) {
      genmoveParams = p;
      if(isGenmoveParams)
        bot->setParams(genmoveParams);
    }
  }

  const SearchParams& getAnalysisParams() {
    return analysisParams;
  }

  void setAnalysisParamsIfChanged(const SearchParams& p) {
    if(analysisParams != p) {
      analysisParams = p;
      if(!isGenmoveParams)
        bot->setParams(analysisParams);
    }
  }
};


//User should pre-fill pla with a default value, as it will not get filled in if the parsed command doesn't specify
static GTPEngine::AnalyzeArgs parseAnalyzeCommand(
  const string& command,
  const vector<string>& pieces,
  Player& pla,
  bool& parseFailed,
  GTPEngine* engine
) {
  int numArgsParsed = 0;

  bool isLZ = (command == "lz-analyze" || command == "lz-genmove_analyze");
  bool isKata = (command == "kata-analyze" || command == "kata-genmove_analyze" || command == "kata-search_analyze" || command == "kata-search_analyze_cancellable");
  double lzAnalyzeInterval = TimeControls::UNLIMITED_TIME_DEFAULT;
  int minMoves = 0;
  int maxMoves = 10000000;
  bool showRootInfo = false;
  bool showPVVisits = false;
  bool showPVEdgeVisits = false;
  bool showNoResultValue = false;
  bool jsonOutput = false;
  bool moveLimitSpecified = false;
  vector<int> avoidMoveUntilByLocBlack;
  vector<int> avoidMoveUntilByLocWhite;
  bool gotAvoidMovesBlack = false;
  bool gotAllowMovesBlack = false;
  bool gotAvoidMovesWhite = false;
  bool gotAllowMovesWhite = false;
  vector<Loc> focusMoves;
  vector<double> focusWeights;
  double focusProb = 0.0;
  bool gotFocus = false;

  parseFailed = false;

  //Format:
  //lz-analyze [optional player] [optional interval float] <keys and values>
  //Keys and values consists of zero or more of:

  //interval <float interval in centiseconds>
  //avoid <player> <comma-separated moves> <until movenum>
  //focus <comma-separated moves, each optionally suffixed with :weight> <probability>
  //minmoves <int min number of moves to show>
  //maxmoves <int max number of moves to show>
  //pvVisits <bool whether to show pvVisits or not>
  //pvEdgeVisits <bool whether to show pvEdgeVisits or not>

  //Parse optional player
  if(pieces.size() > numArgsParsed && PlayerIO::tryParsePlayer(pieces[numArgsParsed],pla))
    numArgsParsed += 1;

  //Parse optional interval float
  if(pieces.size() > numArgsParsed &&
     Global::tryStringToDouble(pieces[numArgsParsed],lzAnalyzeInterval) &&
     !isnan(lzAnalyzeInterval) && lzAnalyzeInterval >= 0 && lzAnalyzeInterval < TimeControls::MAX_USER_INPUT_TIME)
    numArgsParsed += 1;

  //Now loop and handle all key value pairs
  while(pieces.size() > numArgsParsed) {
    const string& key = pieces[numArgsParsed];
    numArgsParsed += 1;
    //Make sure we have a value. If not, then we fail.
    if(pieces.size() <= numArgsParsed) {
      parseFailed = true;
      break;
    }

    const string& value = pieces[numArgsParsed];
    numArgsParsed += 1;

    if(key == "interval" && Global::tryStringToDouble(value,lzAnalyzeInterval) &&
       !isnan(lzAnalyzeInterval) && lzAnalyzeInterval >= 0 && lzAnalyzeInterval < TimeControls::MAX_USER_INPUT_TIME) {
      continue;
    }
    else if(key == "avoid" || key == "allow") {
      //Parse two more arguments
      if(pieces.size() < numArgsParsed+2) {
        parseFailed = true;
        break;
      }
      const string& movesStr = pieces[numArgsParsed];
      numArgsParsed += 1;
      const string& untilDepthStr = pieces[numArgsParsed];
      numArgsParsed += 1;

      int untilDepth = -1;
      if(!Global::tryStringToInt(untilDepthStr,untilDepth) || untilDepth < 1) {
        parseFailed = true;
        break;
      }
      Player avoidPla = C_EMPTY;
      if(!PlayerIO::tryParsePlayer(value,avoidPla)) {
        parseFailed = true;
        break;
      }
      vector<Loc> parsedLocs;
      vector<string> locPieces = Global::split(movesStr,',');
      for(size_t i = 0; i<locPieces.size(); i++) {
        string s = Global::trim(locPieces[i]);
        if(s.size() <= 0)
          continue;
        Loc loc;
        if(!tryParseLoc(s,engine->bot->getRootBoard(),engine->getCurrentRules(),loc)) {
          parseFailed = true;
          break;
        }
        parsedLocs.push_back(loc);
      }
      if(parseFailed)
        break;

      //Make sure the same analyze command can't specify both avoid and allow, and allow at most one allow.
      vector<int>& avoidMoveUntilByLoc = avoidPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
      bool& gotAvoidMoves = avoidPla == P_BLACK ? gotAvoidMovesBlack : gotAvoidMovesWhite;
      bool& gotAllowMoves = avoidPla == P_BLACK ? gotAllowMovesBlack : gotAllowMovesWhite;
      if((key == "allow" && gotAvoidMoves) || (key == "allow" && gotAllowMoves) || (key == "avoid" && gotAllowMoves)) {
        parseFailed = true;
        break;
      }
      avoidMoveUntilByLoc.resize(engine->bot->getSearch()->getMoveLocBufferSize());
      if(key == "allow") {
        std::fill(avoidMoveUntilByLoc.begin(),avoidMoveUntilByLoc.end(),untilDepth);
        for(Loc loc: parsedLocs) {
          avoidMoveUntilByLoc[loc] = 0;
        }
      }
      else {
        for(Loc loc: parsedLocs) {
          avoidMoveUntilByLoc[loc] = untilDepth;
        }
      }
      gotAvoidMoves |= (key == "avoid");
      gotAllowMoves |= (key == "allow");

      continue;
    }
    else if(key == "focus") {
      //Can only be specified once. Parse one more argument.
      if(gotFocus || pieces.size() < numArgsParsed+1) {
        parseFailed = true;
        break;
      }
      gotFocus = true;
      const string& probStr = pieces[numArgsParsed];
      numArgsParsed += 1;

      if(!Global::tryStringToDouble(probStr,focusProb) || isnan(focusProb) || focusProb < 0.0 || focusProb > 1.0) {
        parseFailed = true;
        break;
      }
      vector<string> locPieces = Global::split(value,',');
      for(size_t i = 0; i<locPieces.size(); i++) {
        string s = Global::trim(locPieces[i]);
        if(s.size() <= 0)
          continue;
        //Each move may carry a weight after a colon, such as C3:2. Missing weights default to 1.
        double weight = 1.0;
        size_t colonPos = s.find(':');
        if(colonPos != string::npos) {
          string weightStr = s.substr(colonPos+1);
          s = s.substr(0,colonPos);
          if(!Global::tryStringToDouble(weightStr,weight) || !isfinite(weight) || weight <= 0.0) {
            parseFailed = true;
            break;
          }
        }
        Loc loc;
        if(!tryParseLoc(s,engine->bot->getRootBoard(),engine->getCurrentRules(),loc)) {
          parseFailed = true;
          break;
        }
        focusMoves.push_back(loc);
        focusWeights.push_back(weight);
      }
      if(parseFailed)
        break;
      continue;
    }
    else if(key == "minmoves" && Global::tryStringToInt(value,minMoves) &&
            minMoves >= 0 && minMoves < 1000000000) {
      moveLimitSpecified = true;
      continue;
    }
    else if(key == "maxmoves" && Global::tryStringToInt(value,maxMoves) &&
            maxMoves >= 0 && maxMoves < 1000000000) {
      moveLimitSpecified = true;
      continue;
    }
    else if(isKata && key == "json" && Global::tryStringToBool(value,jsonOutput)) {
      continue;
    }
    else if(isKata && key == "rootInfo" && Global::tryStringToBool(value,showRootInfo)) {
      continue;
    }




    else if(isKata && key == "pvVisits" && Global::tryStringToBool(value,showPVVisits)) {
      continue;
    }
    else if(isKata && key == "pvEdgeVisits" && Global::tryStringToBool(value,showPVEdgeVisits)) {
      continue;
    }
    else if(isKata && key == "noResultValue" && Global::tryStringToBool(value,showNoResultValue)) {
      continue;
    }

    parseFailed = true;
    break;
  }

  GTPEngine::AnalyzeArgs args = GTPEngine::AnalyzeArgs();
  // The source JSON report emits its own candidate list and root information.
  // Do not silently accept text-only move truncation controls in this mode.
  if(jsonOutput && moveLimitSpecified)
    parseFailed = true;
  if(engine->getCurrentRules().isXiangqi() && pla != engine->bot->getRootPla())
    parseFailed = true;
  args.analyzing = true;
  args.lz = isLZ;
  args.kata = isKata;
  //Convert from centiseconds to seconds
  args.secondsPerReport = lzAnalyzeInterval * 0.01;
  args.minMoves = minMoves;
  args.maxMoves = maxMoves;
  args.showRootInfo = showRootInfo;
  args.showPVVisits = showPVVisits;
  args.showPVEdgeVisits = showPVEdgeVisits;
  args.showNoResultValue = showNoResultValue;
  args.jsonOutput = jsonOutput;
  args.avoidMoveUntilByLocBlack = avoidMoveUntilByLocBlack;
  args.avoidMoveUntilByLocWhite = avoidMoveUntilByLocWhite;
  args.focusMoves = focusMoves;
  args.focusWeights = focusWeights;
  args.focusProb = focusProb;
  return args;
}

//Parse args for kata-raw-nn / kata-raw-human-nn: an optional leading player color ("b"/"w"/"black"/"white"),
//a required symmetry ("all" or an index 0-7), and (if allowOptimism) an optional trailing policy optimism value
//in [0,1]. This is positional and mirrors the optional-player convention used by the analyze commands (see
//parseAnalyzeCommand): a color, if present, must come first, and the optimism, if present, must come last.
//'pla' and 'policyOptimism' should be pre-filled with defaults; each is overwritten only if its argument is given.
//Returns false and fills 'error' on any missing, extra, or unparseable argument.
//Note: only "b"/"w"/"black"/"white" parse as a player (see PlayerIO::tryParsePlayer). An integer never parses as a
//player, so a bare symmetry index such as "kata-raw-nn 3" is unambiguously the symmetry, exactly as before.
static bool parseRawNNArgs(
  const vector<string>& pieces,
  bool allowOptimism,
  Player& pla,
  int& whichSymmetry,
  double& policyOptimism,
  string& error
) {
  size_t idx = 0;

  //Optional leading player color.
  Player parsedPla;
  if(idx < pieces.size() && PlayerIO::tryParsePlayer(Global::trim(Global::toLower(pieces[idx])),parsedPla)) {
    pla = parsedPla;
    idx += 1;
  }

  //Required symmetry.
  if(idx >= pieces.size()) {
    error = "Expected a symmetry argument 'all' or index [0-7]";
    return false;
  }
  {
    string s = Global::trim(Global::toLower(pieces[idx]));
    int parsedSym;
    if(s == "all")
      whichSymmetry = NNInputs::SYMMETRY_ALL;
    else if(Global::tryStringToInt(s,parsedSym) && parsedSym >= 0 && parsedSym <= SymmetryHelpers::NUM_SYMMETRIES-1)
      whichSymmetry = parsedSym;
    else {
      error = "Expected a symmetry argument 'all' or index [0-7] but got '" + pieces[idx] + "'";
      return false;
    }
    idx += 1;
  }

  //Optional trailing policy optimism in [0,1].
  if(allowOptimism && idx < pieces.size()) {
    double parsedOpt;
    if(Global::tryStringToDouble(pieces[idx],parsedOpt) && !isnan(parsedOpt) && parsedOpt >= 0.0 && parsedOpt <= 1.0) {
      policyOptimism = parsedOpt;
      idx += 1;
    }
    else {
      error = "Expected an optimism value in [0,1] but got '" + pieces[idx] + "'";
      return false;
    }
  }

  //No further arguments allowed.
  if(idx < pieces.size()) {
    error = "Unexpected extra argument '" + pieces[idx] + "'";
    return false;
  }
  return true;
}


int MainCmds::gtp(const vector<string>& args) {
  Board::initHash();
  Rand seedRand;

  ConfigParser cfg;
  string nnModelFile;
  string humanModelFile;
  string overrideVersion;
  KataGoCommandLine cmd("Run KataGo main GTP engine for playing games or casual analysis.");
  try {
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    cmd.addHumanModelFileArg();
    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> overrideVersionArg("","override-version","Force KataGo to say a certain value in response to gtp version command",false,string(),"VERSION");
    cmd.add(overrideVersionArg);
    cmd.parseArgs(args);
    nnModelFile = cmd.getModelFile();
    humanModelFile = cmd.getHumanModelFile();
    overrideVersion = overrideVersionArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger(&cfg);

  const bool logAllGTPCommunication = cfg.getBool("logAllGTPCommunication");
  const bool logSearchInfo = cfg.getBool("logSearchInfo");
  const bool logSearchInfoForChosenMove = cfg.contains("logSearchInfoForChosenMove") ? cfg.getBool("logSearchInfoForChosenMove") : false;
  const PrintTreeOptions logSearchTreeOptions = Setup::loadLogSearchTreeOptions(cfg);

  bool startupPrintMessageToStderr = true;
  if(cfg.contains("startupPrintMessageToStderr"))
    startupPrintMessageToStderr = cfg.getBool("startupPrintMessageToStderr");

  logger.write("GTP Engine starting...");
  logger.write(Version::getKataGoVersionForHelp());
  //Also check loggingToStderr so that we don't duplicate the message from the log file
  if(startupPrintMessageToStderr && !logger.isLoggingToStderr()) {
    cerr << Version::getKataGoVersionForHelp() << endl;
  }

  //Defaults to 7.5 komi, gtp will generally override this
  const bool loadKomiFromCfg = false;
  Rules initialRules = Setup::loadSingleRules(cfg,loadKomiFromCfg);
  if(cfg.contains("game") && Rules::parseGame(cfg.getString("game")) != initialRules.game)
    throw StringError("Configured game must match the initial GTP rules");
  logger.write("Using " + initialRules.toStringNoKomiMaybeNice() + " rules initially, unless GTP/GUI overrides this");
  if(startupPrintMessageToStderr && !logger.isLoggingToStderr()) {
    cerr << "Using " + initialRules.toStringNoKomiMaybeNice() + " rules initially, unless GTP/GUI overrides this" << endl;
  }
  bool isForcingKomi = false;
  float forcedKomi = 0;
  if(cfg.contains("ignoreGTPAndForceKomi")) {
    isForcingKomi = true;
    forcedKomi = cfg.getFloat("ignoreGTPAndForceKomi", Rules::MIN_USER_KOMI, Rules::MAX_USER_KOMI);
    initialRules.komi = forcedKomi;
  }

  const bool hasHumanModel = humanModelFile != "";
  const bool humanIsMuZero = hasHumanModel && FileUtils::isDirectory(humanModelFile);
  const bool useMuZero = cfg.contains("useMuZero") && cfg.getBool("useMuZero");
  if(!initialRules.isGo() && initialRules.komi != 0)
    throw StringError("Non-Go games require zero komi");

  auto validateSearchParams = [useMuZero,hasHumanModel,humanIsMuZero](const SearchParams& params) {
    if(useMuZero && (params.useGraphSearch || params.subtreeValueBiasFactor != 0 || params.avoidRepeatedPatternUtility != 0))
      throw StringError("MuZero latent search excludes real-board graph, SVB and repeated-pattern bonuses");
    if(useMuZero && hasHumanModel && !humanIsMuZero && params.hasHumanSLTreeExploration())
      throw StringError("MuZero tree HumanSL exploration requires a secondary MuZero model; ordinary HumanSL is root-only");
    if(params.useGumbel && params.useGraphSearch)
      throw StringError("Gumbel GTP requires graph search disabled");
    if(params.useGumbel && hasHumanModel && params.humanSLRootExploreProbWeightful > 0)
      throw StringError("Gumbel root weightful HumanSL conflicts with sequential-halving quotas; use root weightless exploration");
  };

  auto loadParams = [&hasHumanModel](ConfigParser& config, SearchParams& genmoveOut, SearchParams& analysisOut) {
    SearchParams params = Setup::loadSingleParams(config,Setup::SETUP_FOR_GTP,hasHumanModel);

    const double analysisWideRootNoise =
      config.contains("analysisWideRootNoise") ? config.getDouble("analysisWideRootNoise",0.0,5.0) : Setup::DEFAULT_ANALYSIS_WIDE_ROOT_NOISE;
    const bool analysisIgnorePreRootHistory =
      config.contains("analysisIgnorePreRootHistory") ? config.getBool("analysisIgnorePreRootHistory") : Setup::DEFAULT_ANALYSIS_IGNORE_PRE_ROOT_HISTORY;
    genmoveOut = params;
    analysisOut = params;

    analysisOut.wideRootNoise = analysisWideRootNoise;
    analysisOut.ignorePreRootHistory = analysisIgnorePreRootHistory;
  };

  SearchParams initialGenmoveParams;
  SearchParams initialAnalysisParams;
  loadParams(cfg,initialGenmoveParams,initialAnalysisParams);
  validateSearchParams(initialGenmoveParams);
  validateSearchParams(initialAnalysisParams);
  logger.write("Using " + Global::intToString(initialGenmoveParams.numThreads) + " CPU thread(s) for search");

  bool ponderingEnabled = cfg.contains("ponderingEnabled") ? cfg.getBool("ponderingEnabled") : false;


  bool allowResignation = cfg.contains("allowResignation") ? cfg.getBool("allowResignation") : false;
  const double resignThreshold = cfg.contains("allowResignation") ? cfg.getDouble("resignThreshold",-1.0,0.0) : -1.0; //Threshold on [-1,1], regardless of winLossUtilityFactor
  const int resignConsecTurns = cfg.contains("resignConsecTurns") ? cfg.getInt("resignConsecTurns",1,100) : 3;
  if(cfg.contains("resignMinScoreDifference"))
    throw StringError("resignMinScoreDifference requires excluded score predictions");
  const double resignMinMovesPerBoardArea = cfg.contains("resignMinMovesPerBoardArea") ? cfg.getDouble("resignMinMovesPerBoardArea",0.0,1.0) : 0.0;

  Setup::initializeSession(cfg);

  const double searchFactorWhenWinning = cfg.contains("searchFactorWhenWinning") ? cfg.getDouble("searchFactorWhenWinning",0.01,1.0) : 1.0;
  const double searchFactorWhenWinningThreshold = cfg.contains("searchFactorWhenWinningThreshold") ? cfg.getDouble("searchFactorWhenWinningThreshold",0.0,1.0) : 1.0;
  const bool ogsChatToStderr = cfg.contains("ogsChatToStderr") ? cfg.getBool("ogsChatToStderr") : false;
  const int analysisPVLen = cfg.contains("analysisPVLen") ? cfg.getInt("analysisPVLen",1,1000) : 13;
  const bool assumeMultipleStartingBlackMovesAreHandicap =
    cfg.contains("assumeMultipleStartingBlackMovesAreHandicap") ? cfg.getBool("assumeMultipleStartingBlackMovesAreHandicap") : true;
  const bool preventEncore = cfg.contains("preventCleanupPhase") ? cfg.getBool("preventCleanupPhase") : true;
  const double dynamicPlayoutDoublingAdvantageCapPerOppLead =
    cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead") ? cfg.getDouble("dynamicPlayoutDoublingAdvantageCapPerOppLead",0.0,0.5) : 0.045;
  bool staticPDATakesPrecedence = cfg.contains("playoutDoublingAdvantage") && !cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead");
  const double normalAvoidRepeatedPatternUtility = initialGenmoveParams.avoidRepeatedPatternUtility;
  const double handicapAvoidRepeatedPatternUtility = cfg.contains("avoidRepeatedPatternUtility") ?
    initialGenmoveParams.avoidRepeatedPatternUtility : (useMuZero ? 0.0 : 0.005);
  const double initialDelayMoveScale = cfg.contains("delayMoveScale") ? cfg.getDouble("delayMoveScale",0.0,10000.0) : 0.0;
  const double initialDelayMoveMax = cfg.contains("delayMoveMax") ? cfg.getDouble("delayMoveMax",0.0,1000000.0) : 1000000.0;

  int defaultBoardXSize = -1;
  int defaultBoardYSize = -1;
  Setup::loadDefaultBoardXYSize(cfg,logger,defaultBoardXSize,defaultBoardYSize);

  const bool forDeterministicTesting =
    cfg.contains("forDeterministicTesting") ? cfg.getBool("forDeterministicTesting") : false;

  if(forDeterministicTesting)
    seedRand.init("forDeterministicTesting");

  std::unique_ptr<PatternBonusTable> patternBonusTable = nullptr;
  {
    std::vector<std::unique_ptr<PatternBonusTable>> tables = Setup::loadAvoidSgfPatternBonusTables(cfg,logger);
    testAssert(tables.size() == 1);
    patternBonusTable = std::move(tables[0]);
  }

  bool autoAvoidPatterns = false;
  {
    std::unique_ptr<PatternBonusTable> autoTable = Setup::loadAndPruneAutoPatternBonusTables(cfg,logger);
    if(autoTable != nullptr && patternBonusTable != nullptr)
      throw StringError("Providing both sgf avoid patterns and auto avoid patterns is not implemented right now");
    if(autoTable != nullptr) {
      autoAvoidPatterns = true;
      patternBonusTable = std::move(autoTable);
    }
  }
  // Toggled to true every time we save, toggled back to false once we do load.
  bool shouldReloadAutoAvoidPatterns = false;
  if(useMuZero && patternBonusTable != nullptr)
    throw StringError("MuZero latent search excludes external real-board pattern bonuses");

  Player perspective = Setup::parseReportAnalysisWinrates(cfg,C_EMPTY);

  GTPEngine* engine = new GTPEngine(
    nnModelFile,humanModelFile,useMuZero,humanIsMuZero,
    initialGenmoveParams,initialAnalysisParams,
    initialRules,
    assumeMultipleStartingBlackMovesAreHandicap,preventEncore,autoAvoidPatterns,
    dynamicPlayoutDoublingAdvantageCapPerOppLead,
    staticPDATakesPrecedence,
    normalAvoidRepeatedPatternUtility, handicapAvoidRepeatedPatternUtility,
    initialDelayMoveScale,initialDelayMoveMax,
    perspective,analysisPVLen,
    std::move(patternBonusTable)
  );
  engine->setOrResetBoardSize(cfg,logger,seedRand,defaultBoardXSize,defaultBoardYSize,logger.isLoggingToStderr());
  if(ponderingEnabled && engine->primaryModel->requiresSGFMetadata() && !initialGenmoveParams.humanSLProfile.initialized)
    throw StringError("Pondering with this primary model requires an initialized humanSLProfile");

  auto maybeSaveAvoidPatterns = [&](bool forceSave) {
    if(engine != NULL && autoAvoidPatterns) {
      int samplesPerSave = 200;
      if(cfg.contains("autoAvoidRepeatSaveChunkSize"))
        samplesPerSave = cfg.getInt("autoAvoidRepeatSaveChunkSize",1,10000);

      if(forceSave || engine->genmoveSamples.size() >= samplesPerSave) {
        bool suc = Setup::saveAutoPatternBonusData(engine->genmoveSamples, cfg, logger, seedRand);
        if(suc) {
          engine->genmoveSamples.clear();
          shouldReloadAutoAvoidPatterns = true;
        }
      }
    }
  };

  //If nobody specified any time limit in any way, then assume a relatively fast time control
  if(!cfg.contains("maxPlayouts") && !cfg.contains("maxVisits") && !cfg.contains("maxTime")) {
    double mainTime = 1.0;
    double byoYomiTime = 5.0;
    int byoYomiPeriods = 5;
    TimeControls tc = TimeControls::canadianOrByoYomiTime(mainTime,byoYomiTime,byoYomiPeriods,1);
    engine->bTimeControls = tc;
    engine->wTimeControls = tc;
  }

  //Check for unused config keys
  // Interactive generation always requests one opening; its sampling settings
  // use the same native fields and parser as match/gatekeeper.
  map<string,string> openingOptions;
  for(const char* key : {"balanceOpeningDropPow", "minAcceptRate"})
    if(cfg.contains(key)) openingOptions[key] = cfg.getString(key);
  cfg.warnUnusedKeys(cerr,&logger);
  Setup::maybeWarnHumanSLParams(initialGenmoveParams,engine->nnEval,engine->humanEval,cerr,&logger,engine->muZeroEval,engine->humanMuZeroEval);

  logger.write("Loaded config " + cfg.getFileName());
  logger.write("Loaded model " + nnModelFile);
  if(humanModelFile != "")
    logger.write("Loaded human SL model " + humanModelFile);
  cmd.logOverrides(logger);
  logger.write("Model name: "+ engine->primaryModel->getInternalModelName());
  if(engine->humanModel != nullptr)
    logger.write("Human SL model name: "+ (engine->humanModel->getInternalModelName()));
  logger.write("GTP ready, beginning main protocol loop");
  //Also check loggingToStderr so that we don't duplicate the message from the log file
  if(startupPrintMessageToStderr && !logger.isLoggingToStderr()) {
    cerr << "Loaded config " << cfg.getFileName() << endl;
    cerr << "Loaded model " << nnModelFile << endl;
    if(humanModelFile != "")
      cerr << "Loaded human SL model " << humanModelFile << endl;
    cerr << "Model name: "+ engine->primaryModel->getInternalModelName() << endl;
    if(engine->humanModel != nullptr)
      cerr << "Human SL model name: "+ (engine->humanModel->getInternalModelName()) << endl;
    cerr << "GTP ready, beginning main protocol loop" << endl;
  }

  if(humanModelFile != "" && !cfg.contains("humanSLProfile") && engine->humanModel->requiresSGFMetadata()) {
    logger.write("WARNING: Provided -human-model but humanSLProfile was not set in the config or overrides. The human SL model will not be used until it is set in the config or at runtime via kata-set-param.");
    if(!logger.isLoggingToStderr())
      cerr << "WARNING: Provided -human-model but humanSLProfile was not set in the config or overrides. The human SL model will not be used until it is set in the config or at runtime via kata-set-param." << endl;
  }

  bool currentlyGenmoving = false;
  bool currentlyAnalyzing = false;
  string line;
  while(getline(cin,line)) {
    //Parse command, extracting out the command itself, the arguments, and any GTP id number for the command.
    string command;
    vector<string> pieces;
    bool hasId = false;
    int id = 0;
    {
      line = CommandLoop::processSingleCommandLine(line);

      //Upon any input line at all, stop any analysis and output a newline
      //Only difference between analysis and genmove is that genmove handles its own
      //double newline in its onmove callback.
      if(currentlyAnalyzing) {
        currentlyAnalyzing = false;
        engine->stopAndWait();
        cout << endl;
      }
      if(currentlyGenmoving) {
        currentlyGenmoving = false;
        engine->stopAndWait();
      }

      if(line.length() == 0)
        continue;

      if(logAllGTPCommunication)
        logger.write("Controller: " + line);

      //Parse id number of command, if present
      size_t digitPrefixLen = 0;
      while(digitPrefixLen < line.length() && Global::isDigit(line[digitPrefixLen]))
        digitPrefixLen++;
      if(digitPrefixLen > 0) {
        hasId = true;
        try {
          id = Global::parseDigits(line,0,digitPrefixLen);
        }
        catch(const IOError& e) {
          cout << "? GTP id '" << id << "' could not be parsed: " << e.what() << endl;
          continue;
        }
        line = line.substr(digitPrefixLen);
      }

      line = Global::trim(line);
      if(line.length() <= 0) {
        cout << "? empty command" << endl;
        continue;
      }

      pieces = Global::split(line,' ');
      for(size_t i = 0; i<pieces.size(); i++)
        pieces[i] = Global::trim(pieces[i]);
      testAssert(pieces.size() > 0);

      command = pieces[0];
      pieces.erase(pieces.begin());
    }

    auto printGTPResponse = [hasId,id,&logger,logAllGTPCommunication](const string& response, bool responseIsError) {
      string postProcessed = response;
      if(hasId)
        postProcessed = Global::intToString(id) + " " + postProcessed;
      else
        postProcessed = " " + postProcessed;

      if(responseIsError)
        postProcessed = "?" + postProcessed;
      else
        postProcessed = "=" + postProcessed;

      cout << postProcessed << endl;
      cout << endl;

      if(logAllGTPCommunication)
        logger.write(postProcessed);
    };
    auto printGTPResponseHeader = [hasId,id,&logger,logAllGTPCommunication]() {
      if(hasId) {
        string s = "=" + Global::intToString(id);
        cout << s << endl;
        if(logAllGTPCommunication)
          logger.write(s);
      }
      else {
        cout << "=" << endl;
        if(logAllGTPCommunication)
          logger.write("=");
      }
    };

    auto printGTPResponseNoHeader = [&logger,logAllGTPCommunication](const string& response, bool responseIsError) {
      //Postprocessing of response in the case where we already printed the "=" and a newline ahead of time via printGTPResponseHeader.
      if(!responseIsError) {
        cout << response << endl;
        cout << endl;
      }
      else {
        cout << endl;
        if(!logger.isLoggingToStderr())
          cerr << response << endl;
      }
      if(logAllGTPCommunication)
        logger.write(response);
    };

    bool responseIsError = false;
    bool suppressResponse = false;
    bool shouldQuitAfterResponse = false;
    bool maybeStartPondering = false;
    string response;

    const bool needsPrimaryNN = command == "genmove" || command == "genmove_debug" || command == "kata-search" ||
      command == "kata-search_debug" || command == "kata-search_cancellable" || command == "genmove_analyze" ||
      command == "lz-genmove_analyze" || command == "kata-genmove_analyze" || command == "kata-search_analyze" ||
      command == "kata-search_analyze_cancellable" || command == "analyze" || command == "lz-analyze" ||
      command == "kata-analyze" || command == "kata-raw-nn" || command == "kata-suggest-opening" || command == "place_free_handicap" || command == "kata-benchmark" ||
      (command == "debug_moves" && std::find(pieces.begin(),pieces.end(),"rawstats") != pieces.end());
    if(needsPrimaryNN && engine->primaryModel->requiresSGFMetadata() && !engine->getGenmoveParams().humanSLProfile.initialized) {
      responseIsError = true;
      response = "The primary model requires an initialized humanSLProfile";
    }
    else if(engine->getCurrentRules().isXiangqi() && engine->bot->getRootHist().isGameFinished &&
            (needsPrimaryNN || command == "kata-raw-human-nn")) {
      responseIsError = true;
      response = "No policy is defined at a terminal Xiangqi position";
    }
    else if(engine->getCurrentRules().isXiangqi() && command == "final_status_list") {
      responseIsError = true;
      response = "Go stone life/death status is not defined for Xiangqi";
    }
    else if(!engine->getCurrentRules().isGo() &&
            (command == "fixed_handicap" || command == "place_free_handicap" || command == "set_free_handicap")) {
      responseIsError = true;
      response = "Handicap stones are Go-only";
    }
    else if(command == "protocol_version") {
      response = "2";
    }

    else if(command == "name") {
      response = "KataGo";
    }

    else if(command == "version") {
      if(overrideVersion.size() > 0)
        response = overrideVersion;
      else {
        std::vector<string> parts;
        parts.push_back(Version::getKataGoVersion());
        if(engine->nnEval != NULL)
          parts.push_back(engine->nnEval->getAbbrevInternalModelName());
        else parts.push_back(engine->primaryModel->getInternalModelName());
        if(engine->humanModel != nullptr)
          parts.push_back(engine->humanEval ? engine->humanEval->getAbbrevInternalModelName() : engine->humanModel->getInternalModelName());
        response = Global::concat(parts,"+");
      }
    }

    else if(command == "known_command") {
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected single argument for known_command but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        if(std::find(knownCommands.begin(), knownCommands.end(), pieces[0]) != knownCommands.end())
          response = "true";
        else
          response = "false";
      }
    }

    else if(command == "list_commands") {
      for(size_t i = 0; i<knownCommands.size(); i++) {
        response += knownCommands[i];
        if(i < knownCommands.size()-1)
          response += "\n";
      }
    }

    else if(command == "quit") {
      maybeSaveAvoidPatterns(true);
      shouldQuitAfterResponse = true;
      logger.write("Quit requested by controller");
    }

    else if(command == "boardsize" || command == "rectangular_boardsize") {
      maybeSaveAvoidPatterns(false);
      int newXSize = 0;
      int newYSize = 0;
      bool suc = false;

      if(pieces.size() == 1) {
        if(contains(pieces[0],':')) {
          vector<string> subpieces = Global::split(pieces[0],':');
          if(subpieces.size() == 2 && Global::tryStringToInt(subpieces[0], newXSize) && Global::tryStringToInt(subpieces[1], newYSize))
            suc = true;
        }
        else {
          if(Global::tryStringToInt(pieces[0], newXSize)) {
            suc = true;
            newYSize = newXSize;
          }
        }
      }
      else if(pieces.size() == 2) {
        if(Global::tryStringToInt(pieces[0], newXSize) && Global::tryStringToInt(pieces[1], newYSize))
          suc = true;
      }

      if(!suc) {
        responseIsError = true;
        response = "Expected int argument for boardsize or pair of ints but got '" + Global::concat(pieces," ") + "'";
      }
      else if(newXSize < 2 || newYSize < 2) {
        responseIsError = true;
        response = "unacceptable size";
      }
      else if(newXSize > Board::MAX_LEN || newYSize > Board::MAX_LEN) {
        responseIsError = true;
        response = Global::strprintf("unacceptable size (Board::MAX_LEN is %d, consider increasing and recompiling)",(int)Board::MAX_LEN);
      }
      else {
        try {
          engine->checkBoardSize(newXSize,newYSize);
          engine->setOrResetBoardSize(cfg,logger,seedRand,newXSize,newYSize,logger.isLoggingToStderr());
        }
        catch(const StringError& e) { responseIsError = true; response = e.what(); }
      }
    }

    else if(command == "clear_board") {
      maybeSaveAvoidPatterns(false);
      if(autoAvoidPatterns && shouldReloadAutoAvoidPatterns) {
        std::unique_ptr<PatternBonusTable> autoTable = Setup::loadAndPruneAutoPatternBonusTables(cfg,logger);
        engine->setPatternBonusTable(std::move(autoTable));
        shouldReloadAutoAvoidPatterns = false;
      }
      engine->clearBoard();
    }

    else if(command == "komi") {
      float newKomi = 0;
      if(pieces.size() != 1 || !Global::tryStringToFloat(pieces[0],newKomi)) {
        responseIsError = true;
        response = "Expected single float argument for komi but got '" + Global::concat(pieces," ") + "'";
      }
      //GTP spec says that we should accept any komi, but we're going to ignore that.
      else if(isnan(newKomi) || newKomi < Rules::MIN_USER_KOMI || newKomi > Rules::MAX_USER_KOMI) {
        responseIsError = true;
        response = "unacceptable komi";
      }
      else if(!Rules::komiIsIntOrHalfInt(newKomi)) {
        responseIsError = true;
        response = "komi must be an integer or half-integer";
      }
      else if(!engine->getCurrentRules().isGo() && newKomi != 0) {
        responseIsError = true;
        response = "Non-Go games require zero komi";
      }
      else {
        if(isForcingKomi)
          newKomi = forcedKomi;
        engine->updateKomiIfNew(newKomi);
        //In case the controller tells us komi every move, restart pondering afterward.
        maybeStartPondering = engine->bot->getRootHist().moveHistory.size() > 0;
      }
    }

    else if(command == "get_komi") {
      response = Global::doubleToString(engine->getCurrentRules().komi);
    }

    else if(command == "kata-get-rules") {
      if(pieces.size() != 0) {
        response = "Expected no arguments for kata-get-rules but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        response = engine->getCurrentRules().toJsonStringNoKomi();
      }
    }

    else if(command == "kata-set-rules") {
      string rest = Global::concat(pieces," ");
      bool parseSuccess = false;
      Rules newRules;
      try {
        newRules = Rules::parseRulesWithoutKomi(rest,engine->getCurrentRules().komi);
        parseSuccess = true;
      }
      catch(const StringError& err) {
        responseIsError = true;
        response = "Unknown rules '" + rest + "', " + err.what();
      }
      if(parseSuccess) {
        string error;
        bool suc = engine->setRulesNotIncludingKomi(newRules,error);
        if(!suc) {
          responseIsError = true;
          response = error;
        }
        logger.write("Changed rules to " + newRules.toStringNoKomiMaybeNice());
        if(!logger.isLoggingToStderr())
          cerr << "Changed rules to " + newRules.toStringNoKomiMaybeNice() << endl;
      }
    }

    else if(command == "kata-set-rule") {
      if(pieces.size() != 2) {
        responseIsError = true;
        response = "Expected two arguments for kata-set-rule but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        bool parseSuccess = false;
        Rules currentRules = engine->getCurrentRules();
        Rules newRules;
        try {
          newRules = Rules::updateRules(pieces[0], pieces[1], currentRules);
          parseSuccess = true;
        }
        catch(const StringError& err) {
          responseIsError = true;
          response = err.what();
        }
        if(parseSuccess) {
          string error;
          bool suc = engine->setRulesNotIncludingKomi(newRules,error);
          if(!suc) {
            responseIsError = true;
            response = error;
          }
          logger.write("Changed rules to " + newRules.toStringNoKomiMaybeNice());
          if(!logger.isLoggingToStderr())
            cerr << "Changed rules to " + newRules.toStringNoKomiMaybeNice() << endl;
        }
      }
    }

    else if(command == "kgs-rules") {
      bool parseSuccess = false;
      Rules newRules;
      if(pieces.size() <= 0) {
        responseIsError = true;
        response = "Expected one argument kgs-rules";
      }
      else {
        string s = Global::toLower(Global::trim(pieces[0]));
        if(s == "chinese") {
          newRules = Rules::parseRulesWithoutKomi("chinese-kgs",engine->getCurrentRules().komi);
          parseSuccess = true;
        }
        else if(s == "aga") {
          newRules = Rules::parseRulesWithoutKomi("aga",engine->getCurrentRules().komi);
          parseSuccess = true;
        }
        else if(s == "new_zealand") {
          newRules = Rules::parseRulesWithoutKomi("new_zealand",engine->getCurrentRules().komi);
          parseSuccess = true;
        }
        else if(s == "japanese") {
          newRules = Rules::parseRulesWithoutKomi("japanese",engine->getCurrentRules().komi);
          parseSuccess = true;
        }
        else {
          responseIsError = true;
          response = "Unknown rules '" + s + "'";
        }
      }
      if(parseSuccess) {
        string error;
        bool suc = engine->setRulesNotIncludingKomi(newRules,error);
        if(!suc) {
          responseIsError = true;
          response = error;
        }
        logger.write("Changed rules to " + newRules.toStringNoKomiMaybeNice());
        if(!logger.isLoggingToStderr())
          cerr << "Changed rules to " + newRules.toStringNoKomiMaybeNice() << endl;
      }
    }

    else if(command == "kata-list-params") {
      std::vector<string> paramsList;
      paramsList.emplace_back("analysisWideRootNoise");
      paramsList.emplace_back("analysisIgnorePreRootHistory");
      paramsList.emplace_back("humanSLProfile");
      paramsList.emplace_back("allowResignation");
      paramsList.emplace_back("ponderingEnabled");
      paramsList.emplace_back("delayMoveScale");
      paramsList.emplace_back("delayMoveMax");
      nlohmann::json params = engine->getGenmoveParams().changeableParametersToJson();
      for(auto& elt : params.items()) {
        paramsList.push_back(elt.key());
      }
      response = Global::concat(paramsList, " ");
    }

    else if(command == "kata-get-param") {
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one arguments for kata-get-param but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        const SearchParams& analysisParams = engine->getAnalysisParams();
        if(pieces[0] == "analysisWideRootNoise")
          response = Global::doubleToString(analysisParams.wideRootNoise);
        else if(pieces[0] == "analysisIgnorePreRootHistory")
          response = Global::boolToString(analysisParams.ignorePreRootHistory);
        else if(pieces[0] == "humanSLProfile")
          response = cfg.contains("humanSLProfile") ? cfg.getString("humanSLProfile") : "";
        else if(pieces[0] == "allowResignation")
          response = Global::boolToString(allowResignation);
        else if(pieces[0] == "ponderingEnabled")
          response = Global::boolToString(ponderingEnabled);
        else if(pieces[0] == "delayMoveScale")
          response = Global::doubleToString(engine->delayMoveScale);
        else if(pieces[0] == "delayMoveMax")
          response = Global::doubleToString(engine->delayMoveMax);
        else {
          nlohmann::json params = engine->getGenmoveParams().changeableParametersToJson();
          if(params.find(pieces[0]) == params.end()) {
            responseIsError = true;
            response = "Invalid parameter: " + pieces[0];
          }
          else {
            response = params[pieces[0]].dump();
          }
        }
      }
    }
    else if(command == "kata-get-models") {
      nlohmann::json modelsList = nlohmann::json::array();
      if(engine->primaryModel != nullptr) {
        nlohmann::json modelInfo;
        modelInfo["name"] = engine->primaryModel->getModelName();
        modelInfo["internalName"] = engine->primaryModel->getInternalModelName();
        modelInfo["maxBatchSize"] = engine->primaryModel->getMaxBatchSize();
        modelInfo["usesHumanSLProfile"] = engine->primaryModel->requiresSGFMetadata();
        modelInfo["version"] = engine->primaryModel->getModelVersion();
        modelInfo["usingFP16"] = engine->primaryModel->getUsingFP16Mode().toString();
        modelsList.push_back(modelInfo);
      }
      if(engine->humanModel != nullptr) {
        nlohmann::json modelInfo;
        modelInfo["name"] = engine->humanModel->getModelName();
        modelInfo["internalName"] = engine->humanModel->getInternalModelName();
        modelInfo["maxBatchSize"] = engine->humanModel->getMaxBatchSize();
        modelInfo["usesHumanSLProfile"] = engine->humanModel->requiresSGFMetadata();
        modelInfo["version"] = engine->humanModel->getModelVersion();
        modelInfo["usingFP16"] = engine->humanModel->getUsingFP16Mode().toString();
        modelsList.push_back(modelInfo);
      }
      response = modelsList.dump();
    }
    else if(command == "kata-get-params") {
      const SearchParams& analysisParams = engine->getAnalysisParams();
      nlohmann::json params = engine->getGenmoveParams().changeableParametersToJson();
      params["analysisWideRootNoise"] = Global::doubleToString(analysisParams.wideRootNoise);
      params["analysisIgnorePreRootHistory"] = Global::boolToString(analysisParams.ignorePreRootHistory);
      params["humanSLProfile"] = cfg.contains("humanSLProfile") ? cfg.getString("humanSLProfile") : "";
      params["allowResignation"] = Global::boolToString(allowResignation);
      params["ponderingEnabled"] = Global::boolToString(ponderingEnabled);
      params["delayMoveScale"] = Global::doubleToString(engine->delayMoveScale);
      params["delayMoveMax"] = Global::doubleToString(engine->delayMoveMax);
      response = params.dump();
    }
    else if(command == "kata-set-param" || command == "kata-set-params") {
      std::map<string,string> overrideSettings;
      if(command == "kata-set-param") {
        if(pieces.size() != 1 && pieces.size() != 2) {
          responseIsError = true;
          response = "Expected one or two arguments for kata-set-param but got '" + Global::concat(pieces," ") + "'";
        }
        else if(pieces.size() == 1) {
          overrideSettings[pieces[0]] = "";
        }
        else {
          overrideSettings[pieces[0]] = pieces[1];
        }
      }
      else {
        if(pieces.size() < 1) {
          responseIsError = true;
          response = "Expected argument for kata-set-param but got '" + Global::concat(pieces," ") + "'";
        }
        else {
          try {
            string rejoined = Global::concat(pieces, " ", 0, pieces.size());
            nlohmann::json settings = nlohmann::json::parse(rejoined);
            if(!settings.is_object())
              throw StringError("Argument to kata-set-params must be a json object");

            for(auto it = settings.begin(); it != settings.end(); ++it) {
              overrideSettings[it.key()] = it.value().is_string() ? it.value().get<string>(): it.value().dump(); // always convert to string
            }
          }
          catch(const StringError& exception) {
            responseIsError = true;
            response = string("Could not set params: ") + exception.what();
          }
        }
      }

      if(!responseIsError) {
        try {
          // First validate that everything in here is used by checking it by itself
          {
            ConfigParser cleanCfg;
            cleanCfg.overrideKeys(overrideSettings);
            // Add required parameter so that it passes validation
            if(!cleanCfg.contains("numSearchThreads"))
              cleanCfg.overrideKey("numSearchThreads",Global::intToString(engine->getGenmoveParams().numThreads));

            SearchParams buf1;
            SearchParams buf2;
            loadParams(cleanCfg, buf1, buf2);

            // These parameters have a bit of special handling so we can't change them easily right now
            if(contains(overrideSettings,"dynamicPlayoutDoublingAdvantageCapPerOppLead")) throw StringError("Cannot be overridden in kata-set-param: dynamicPlayoutDoublingAdvantageCapPerOppLead");
            if(contains(overrideSettings,"avoidRepeatedPatternUtility")) throw StringError("Cannot be overridden in kata-set-param: avoidRepeatedPatternUtility");
            cleanCfg.markKeyUsed("allowResignation");
            cleanCfg.markKeyUsed("ponderingEnabled");
            cleanCfg.markKeyUsed("delayMoveScale");
            cleanCfg.markKeyUsed("delayMoveMax");

            vector<string> unusedKeys = cleanCfg.unusedKeys();
            for(const string& unused: unusedKeys) {
              throw StringError("Unrecognized or non-overridable parameter in kata-set-params: " + unused);
            }
            ostringstream out;
            if(Setup::maybeWarnHumanSLParams(buf1,engine->nnEval,engine->humanEval,out,NULL,engine->muZeroEval,engine->humanMuZeroEval)) {
              throw StringError(out.str());
            }
          }

          //Ignore any unused keys in the original config so far
          cfg.markAllKeysUsedWithPrefix("");
          // Validate the complete candidate before installing settings. A rejected
          // graph/SVB/Gumbel combination must not poison subsequent commands.
          ConfigParser candidateCfg(cfg);
          candidateCfg.overrideKeys(overrideSettings);
          SearchParams genmoveParams;
          SearchParams analysisParams;
          loadParams(candidateCfg,genmoveParams,analysisParams);
          validateSearchParams(genmoveParams);
          validateSearchParams(analysisParams);

          bool desiredAllowResignation = candidateCfg.contains("allowResignation") ? candidateCfg.getBool("allowResignation") : allowResignation;
          bool desiredPonderingEnabled = candidateCfg.contains("ponderingEnabled") ? candidateCfg.getBool("ponderingEnabled") : ponderingEnabled;
          double desiredDelayMoveScale = candidateCfg.contains("delayMoveScale") ? candidateCfg.getDouble("delayMoveScale",0.0,10000.0) : engine->delayMoveScale;
          double desiredDelayMoveMax = candidateCfg.contains("delayMoveMax") ? candidateCfg.getDouble("delayMoveMax",0.0,1000000.0) : engine->delayMoveMax;

          SearchParams::failIfParamsDifferOnUnchangeableParameter(initialGenmoveParams,genmoveParams);
          SearchParams::failIfParamsDifferOnUnchangeableParameter(initialAnalysisParams,analysisParams);
          if(desiredPonderingEnabled && engine->primaryModel->requiresSGFMetadata() && !genmoveParams.humanSLProfile.initialized)
            throw StringError("Pondering with this primary model requires an initialized humanSLProfile");
          cfg.overrideKeys(overrideSettings);
          BoardHistoryModes oldHistoryModes = engine->getHistoryModes();
          engine->setGenmoveParamsIfChanged(genmoveParams);
          engine->setAnalysisParamsIfChanged(analysisParams);
          //If the params change flipped any resolved BoardHistoryModes flag, re-replay the game
          //so all recorded game state is recomputed under the new modes (re-stamping alone does not
          //re-adjudicate). Done after both param sets are updated so the replay runs under the new modes.
          BoardHistoryModes newHistoryModes = engine->getHistoryModes();
          if(newHistoryModes != oldHistoryModes)
            engine->rereplayGameForHistoryModesChange();
          staticPDATakesPrecedence = cfg.contains("playoutDoublingAdvantage") && !cfg.contains("dynamicPlayoutDoublingAdvantageCapPerOppLead");
          engine->staticPDATakesPrecedence = staticPDATakesPrecedence;
          allowResignation = desiredAllowResignation;
          ponderingEnabled = desiredPonderingEnabled;
          engine->delayMoveScale = desiredDelayMoveScale;
          engine->delayMoveMax = desiredDelayMoveMax;
        }
        catch(const StringError& exception) {
          responseIsError = true;
          response = string("Could not set params: ") + exception.what();
        }
      }
    }

    else if(command == "time_settings") {
      double mainTime;
      double byoYomiTime;
      int byoYomiStones;
      bool success = false;
      try {
        mainTime = parseTime(pieces,0,"main time");
        byoYomiTime = parseTime(pieces,1,"byo-yomi per-period time");
        byoYomiStones = parseByoYomiStones(pieces,2);
        success = true;
      }
      catch(const StringError& e) {
        responseIsError = true;
        response = e.what();
      }
      if(success) {
        TimeControls tc;
        //This means no time limits, according to gtp spec
        if(byoYomiStones == 0 && byoYomiTime > 0.0)
          tc = TimeControls();
        else if(byoYomiStones == 0)
          tc = TimeControls::absoluteTime(mainTime);
        else
          tc = TimeControls::canadianOrByoYomiTime(mainTime,byoYomiTime,1,byoYomiStones);
        engine->bTimeControls = tc;
        engine->wTimeControls = tc;
      }
    }

    else if(command == "kata-list_time_settings") {
      response = "none";
      response += " ";
      response += "absolute";
      response += " ";
      response += "byoyomi";
      response += " ";
      response += "canadian";
      response += " ";
      response += "fischer";
      response += " ";
      response += "fischer-capped";
    }

    else if(command == "kgs-time_settings" || command == "kata-time_settings") {
      if(pieces.size() < 1) {
        responseIsError = true;
        if(command == "kata-time_settings")
          response = "Expected 'none', 'absolute', 'byoyomi', 'canadian', 'fischer', or 'fischer-capped' as first argument for kata-time_settings";
        else
          response = "Expected 'none', 'absolute', 'byoyomi', or 'canadian' as first argument for kgs-time_settings";
      }
      else {
        string what = Global::toLower(Global::trim(pieces[0]));
        if(what == "none") {
          TimeControls tc = TimeControls();
          engine->bTimeControls = tc;
          engine->wTimeControls = tc;
        }
        else if(what == "absolute") {
          double mainTime;
          TimeControls tc;
          bool success = false;
          try {
            mainTime = parseTime(pieces,1,"main time");
            tc = TimeControls::absoluteTime(mainTime);
            success = true;
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(success) {
            engine->bTimeControls = tc;
            engine->wTimeControls = tc;
          }
        }
        else if(what == "canadian") {
          double mainTime;
          double byoYomiTime;
          int byoYomiStones;
          TimeControls tc;
          bool success = false;
          try {
            mainTime = parseTime(pieces,1,"main time");
            byoYomiTime = parseTime(pieces,2,"byo-yomi period time");
            byoYomiStones = parseByoYomiStones(pieces,3);
            //Use the same hack in time-settings - if somehow someone specifies positive overtime but 0 stones for it, intepret as no time control
            if(byoYomiStones == 0 && byoYomiTime > 0.0)
              tc = TimeControls();
            else if(byoYomiStones == 0)
              tc = TimeControls::absoluteTime(mainTime);
            else
              tc = TimeControls::canadianOrByoYomiTime(mainTime,byoYomiTime,1,byoYomiStones);
            success = true;
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(success) {
            engine->bTimeControls = tc;
            engine->wTimeControls = tc;
          }
        }
        else if(what == "byoyomi") {
          double mainTime;
          double byoYomiTime;
          int byoYomiPeriods;
          TimeControls tc;
          bool success = false;
          try {
            mainTime = parseTime(pieces,1,"main time");
            byoYomiTime = parseTime(pieces,2,"byo-yomi per-period time");
            byoYomiPeriods = parseByoYomiPeriods(pieces,3);
            if(byoYomiPeriods == 0)
              tc = TimeControls::absoluteTime(mainTime);
            else
              tc = TimeControls::canadianOrByoYomiTime(mainTime,byoYomiTime,byoYomiPeriods,1);
            success = true;
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(success) {
            engine->bTimeControls = tc;
            engine->wTimeControls = tc;
          }
        }
        else if(what == "fischer" && command == "kata-time_settings") {
          double mainTime;
          double increment;
          TimeControls tc;
          bool success = false;
          try {
            mainTime = parseTime(pieces,1,"main time");
            increment = parseTime(pieces,2,"increment time");
            tc = TimeControls::fischerTime(mainTime,increment);
            success = true;
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(success) {
            engine->bTimeControls = tc;
            engine->wTimeControls = tc;
          }
        }
        else if(what == "fischer-capped" && command == "kata-time_settings") {
          double mainTime;
          double increment;
          double mainTimeLimit;
          double maxTimePerMove;
          TimeControls tc;
          bool success = false;
          try {
            mainTime = parseTime(pieces,1,"main time");
            increment = parseTime(pieces,2,"increment time");
            mainTimeLimit = parseTimeAllowNegative(pieces,3,"main time limit");
            maxTimePerMove = parseTimeAllowNegative(pieces,4,"max time per move");
            if(mainTimeLimit < 0)
              mainTimeLimit = TimeControls::MAX_USER_INPUT_TIME;
            if(maxTimePerMove < 0)
              maxTimePerMove = TimeControls::MAX_USER_INPUT_TIME;
            tc = TimeControls::fischerCappedTime(mainTime,increment,mainTimeLimit,maxTimePerMove);
            success = true;
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(success) {
            engine->bTimeControls = tc;
            engine->wTimeControls = tc;
          }
        }
        else {
          responseIsError = true;
          if(command == "kata-time_settings")
            response = "Expected 'none', 'absolute', 'byoyomi', 'canadian', 'fischer', or 'fischer-capped' as first argument for kata-time_settings";
          else
            response = "Expected 'none', 'absolute', 'byoyomi', or 'canadian' as first argument for kgs-time_settings";
        }
      }
    }

    else if(command == "time_left") {
      Player pla;
      double time;
      int stones;
      if(pieces.size() != 3
         || !PlayerIO::tryParsePlayer(pieces[0],pla)
         || !Global::tryStringToDouble(pieces[1],time)
         || !Global::tryStringToInt(pieces[2],stones)
         ) {
        responseIsError = true;
        response = "Expected player and float time and int stones for time_left but got '" + Global::concat(pieces," ") + "'";
      }
      //Be slightly tolerant of negative time left
      else if(isnan(time) || time < -10.0 || time > TimeControls::MAX_USER_INPUT_TIME) {
        responseIsError = true;
        response = "invalid time";
      }
      else if(stones < 0 || stones > 100000) {
        responseIsError = true;
        response = "invalid stones";
      }
      else {
        TimeControls tc = pla == P_BLACK ? engine->bTimeControls : engine->wTimeControls;
        if(stones > 0 && tc.originalNumPeriods <= 0) {
          responseIsError = true;
          response = "stones left in period is > 0 but the time control used does not have any overtime periods";
        }
        else {
          //Main time
          if(stones == 0) {
            tc.mainTimeLeft = time;
            tc.inOvertime = false;
            tc.numPeriodsLeftIncludingCurrent = tc.originalNumPeriods;
            tc.numStonesLeftInPeriod = 0;
            tc.timeLeftInPeriod = 0;
          }
          else {
            //Hack for KGS byo-yomi - interpret num stones as periods instead
            if(tc.originalNumPeriods > 1 && tc.numStonesPerPeriod == 1) {
              tc.mainTimeLeft = 0.0;
              tc.inOvertime = true;
              tc.numPeriodsLeftIncludingCurrent = std::min(stones,tc.originalNumPeriods);
              tc.numStonesLeftInPeriod = 1;
              tc.timeLeftInPeriod = time;
            }
            //Normal canadian time interpertation of GTP
            else {
              tc.mainTimeLeft = 0.0;
              tc.inOvertime = true;
              tc.numPeriodsLeftIncludingCurrent = 1;
              tc.numStonesLeftInPeriod = std::min(stones,tc.numStonesPerPeriod);
              tc.timeLeftInPeriod = time;
            }
          }
          if(pla == P_BLACK)
            engine->bTimeControls = tc;
          else
            engine->wTimeControls = tc;

          //In case the controller tells us komi every move, restart pondering afterward.
          maybeStartPondering = engine->bot->getRootHist().moveHistory.size() > 0;
        }
      }
    }

    else if(command == "kata-debug-print-tc") {
      response += "Black "+ engine->bTimeControls.toDebugString(engine->bot->getRootBoard(),engine->bot->getRootHist(),engine->getGenmoveParams().lagBuffer);
      response += "\n";
      response += "White "+ engine->wTimeControls.toDebugString(engine->bot->getRootBoard(),engine->bot->getRootHist(),engine->getGenmoveParams().lagBuffer);
    }

    else if(command == "play") {
      Player pla;
      Loc loc;
      if(pieces.size() != 2) {
        responseIsError = true;
        response = "Expected two arguments for play but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!PlayerIO::tryParsePlayer(pieces[0],pla)) {
        responseIsError = true;
        response = "Could not parse color: '" + pieces[0] + "'";
      }
      else if(!tryParseLoc(pieces[1],engine->bot->getRootBoard(),engine->getCurrentRules(),loc)) {
        responseIsError = true;
        response = "Could not parse vertex: '" + pieces[1] + "'";
      }
      else {
        bool suc = engine->play(loc,pla);
        if(!suc) {
          responseIsError = true;
          response = "illegal move";
        }
        maybeStartPondering = true;
      }
    }

    else if(command == "set_position") {
      if(engine->getCurrentRules().isXiangqi()) {
        if(pieces.size() < 2 || pieces[0] != "fen") {
          responseIsError = true;
          response = "Xiangqi requires set_position fen <FEN>; use loadsgf to restore rule history";
        }
        else {
          try {
            vector<string> fenPieces(pieces.begin()+1,pieces.end());
            Xiangqi::State state(Global::concat(fenPieces," "));
            Board board = state.colorBoard();
            BoardHistory hist(state,engine->getHistoryModes());
            maybeSaveAvoidPatterns(false);
            engine->setPositionAndRules(state.nextPlayer(),board,hist,board,state.nextPlayer(),{});
            engine->clearStatsForNewGame();
          }
          catch(const StringError& e) { responseIsError = true; response = e.what(); }
        }
      }
      else if(pieces.size() % 2 != 0) {
        responseIsError = true;
        response = "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        vector<Move> initialStones;
        for(int i = 0; i<pieces.size(); i += 2) {
          Player pla;
          Loc loc;
          if(!PlayerIO::tryParsePlayer(pieces[i],pla)) {
            responseIsError = true;
            response = "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '" + Global::concat(pieces," ") + "': ";
            response += "could not parse color: '" + pieces[i] + "'";
            break;
          }
          else if(!tryParseLoc(pieces[i+1],engine->bot->getRootBoard(),engine->getCurrentRules(),loc)) {
            responseIsError = true;
            response = "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '" + Global::concat(pieces," ") + "': ";
            response += "could not parse vertex: '" + pieces[i+1] + "'";
            break;
          }
          else if(loc == Board::PASS_LOC) {
            responseIsError = true;
            response = "Expected a space-separated sequence of <COLOR> <VERTEX> pairs but got '" + Global::concat(pieces," ") + "': ";
            response += "could not parse vertex: '" + pieces[i+1] + "'";
            break;
          }
          initialStones.emplace_back(loc,pla);
        }
        if(!responseIsError) {
          maybeSaveAvoidPatterns(false);
          try {
            bool suc = engine->setPosition(initialStones);
            if(!suc) {
              responseIsError = true;
              response = "Illegal stone placements - overlapping stones or stones with no liberties?";
            }
          }
          catch(const StringError& e) { responseIsError = true; response = e.what(); }
          maybeStartPondering = false;
        }
      }
    }

    else if(command == "undo") {
      bool suc = engine->undo();
      if(!suc) {
        responseIsError = true;
        response = "cannot undo";
      }
    }

    else if(
      command == "genmove" ||
      command == "genmove_debug" ||
      command == "kata-search" ||
      command == "kata-search_cancellable" ||
      command == "kata-search_debug") {
      Player pla;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for " + command + " but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!PlayerIO::tryParsePlayer(pieces[0],pla)) {
        responseIsError = true;
        response = "Could not parse color: '" + pieces[0] + "'";
      }
      else if(engine->getCurrentRules().isXiangqi() && pla != engine->bot->getRootPla()) {
        responseIsError = true;
        response = "Xiangqi player must match the complete state's next player";
      }
      else if(const char* error = engine->getGenmoveParams().getGumbelVisitCapError(pla,pla)) {
        responseIsError = true;
        response = error;
      }
      else {
        bool debug = command == "genmove_debug" || command == "kata-search_debug";
        bool playChosenMove = command == "genmove" || command == "genmove_debug";

        GTPEngine::GenmoveArgs gargs;
        gargs.searchFactorWhenWinningThreshold = searchFactorWhenWinningThreshold;
        gargs.searchFactorWhenWinning = searchFactorWhenWinning;
        gargs.ogsChatToStderr = ogsChatToStderr;
        gargs.allowResignation = allowResignation;
        gargs.resignThreshold = resignThreshold;
        gargs.resignConsecTurns = resignConsecTurns;
        gargs.resignMinMovesPerBoardArea = resignMinMovesPerBoardArea;
        gargs.logSearchInfo = logSearchInfo;
        gargs.logSearchInfoForChosenMove = logSearchInfoForChosenMove;
        gargs.logSearchTreeOptions = logSearchTreeOptions;
        gargs.debug = debug;

        if(command == "kata-search_cancellable") {
          engine->genMoveCancellable(
            pla,
            logger,
            gargs,
            GTPEngine::AnalyzeArgs(),
            printGTPResponse
          );
          suppressResponse = true; // genmove handles it manually by calling printGTPResponse
          currentlyGenmoving = true; // so that any newline will interrupt us
        }
        else {
          engine->genMove(
            pla,
            logger,
            gargs,
            GTPEngine::AnalyzeArgs(),
            playChosenMove,
            printGTPResponse,
            maybeStartPondering
          );
          suppressResponse = true; // genmove handles it manually by calling printGTPResponse
        }
      }
    }

    else if(
      command == "genmove_analyze" ||
      command == "lz-genmove_analyze" ||
      command == "kata-genmove_analyze" ||
      command == "kata-search_analyze" ||
      command == "kata-search_analyze_cancellable"
    ) {
      Player pla = engine->bot->getRootPla();
      bool parseFailed = false;
      GTPEngine::AnalyzeArgs analyzeArgs = parseAnalyzeCommand(command, pieces, pla, parseFailed, engine);
      if(parseFailed) {
        responseIsError = true;
        response = "Could not parse genmove_analyze arguments or arguments out of range: '" + Global::concat(pieces," ") + "'";
      }
      else if(const char* error = engine->getGenmoveParams().getGumbelVisitCapError(pla,pla)) {
        responseIsError = true;
        response = error;
      }
      else {
        bool debug = false;
        bool playChosenMove = command == "genmove_analyze" || command == "lz-genmove_analyze" || command == "kata-genmove_analyze";

        GTPEngine::GenmoveArgs gargs;
        gargs.searchFactorWhenWinningThreshold = searchFactorWhenWinningThreshold;
        gargs.searchFactorWhenWinning = searchFactorWhenWinning;
        gargs.ogsChatToStderr = ogsChatToStderr;
        gargs.allowResignation = allowResignation;
        gargs.resignThreshold = resignThreshold;
        gargs.resignConsecTurns = resignConsecTurns;
        gargs.resignMinMovesPerBoardArea = resignMinMovesPerBoardArea;
        gargs.logSearchInfo = logSearchInfo;
        gargs.logSearchInfoForChosenMove = logSearchInfoForChosenMove;
        gargs.logSearchTreeOptions = logSearchTreeOptions;
        gargs.debug = debug;

        //Make sure the "equals" for GTP is printed out prior to the first analyze line, regardless of thread racing
        printGTPResponseHeader();

        if(command == "kata-search_analyze_cancellable") {
          engine->genMoveCancellable(
            pla,
            logger,
            gargs,
            analyzeArgs,
            printGTPResponseNoHeader
          );
          suppressResponse = true; // genmove handles it manually by calling printGTPResponse
          currentlyGenmoving = true; // so that any newline will interrupt us
        }
        else {
          engine->genMove(
            pla,
            logger,
            gargs,
            analyzeArgs,
            playChosenMove,
            printGTPResponseNoHeader,
            maybeStartPondering
          );
          suppressResponse = true; // genmove handles it manually by calling printGTPResponseNoHeader
        }
      }
    }

    else if(command == "clear_cache") {
      engine->clearCache();
    }
    else if(command == "showboard") {
      ostringstream sout;
      engine->bot->getRootHist().printBasicInfo(sout, engine->bot->getRootBoard());
      response = Global::trim(filterDoubleNewlines(sout.str()));
    }
    else if(command == "kata-suggest-opening") {
      try {
        auto options = openingOptions;
        if(!pieces.empty()) {
          const auto overrides = nlohmann::json::parse(Global::concat(pieces," "));
          if(!overrides.is_object()) throw StringError("Opening options must be a JSON object");
          for(auto it = overrides.begin(); it != overrides.end(); ++it) {
            if(it.key() != "balanceOpeningDropPow" && it.key() != "minAcceptRate")
              throw StringError("Unknown opening setting: " + it.key());
            if(!it.value().is_number()) throw StringError("Opening settings must be numeric: " + it.key());
            options[it.key()] = it.value().dump();
          }
        }
        options["game"] = Rules::gameName(engine->getCurrentRules().game);
        options["balanceOpeningProb"] = "1";
        if(engine->getCurrentRules().game == Rules::Game::HEX) options["makeOpeningFairRate"] = "1";
        ConfigParser openingCfg(options);
        auto settings = BalancedOpening::Settings::load(openingCfg,false);
        engine->suggestOpening(settings,printGTPResponse);
        suppressResponse = true;
        // Like source cancellable search, any subsequent input (including a
        // bare newline or EOF) stops and joins the task before another command.
        currentlyGenmoving = true;
      }
      catch(const std::exception& error) { responseIsError = true; response = error.what(); }
    }
    else if(command == "kata-get-state") {
      if(!pieces.empty()) {
        responseIsError = true;
        response = "kata-get-state takes no arguments";
      }
      else {
        engine->stopAndWait();
        const Board& board = engine->bot->getRootBoard();
        const BoardHistory& hist = engine->bot->getRootHist();
        const Player pla = engine->bot->getRootPla();
        nlohmann::json state = positionJson(board,hist,pla,perspective,true);
        state["moves"] = nlohmann::json::array();
        for(const Move& move : engine->moveHistory)
          state["moves"].push_back({PlayerIO::playerToStringShort(move.pla),
            hist.rules.isXiangqi() ? Xiangqi::actionToString(Xiangqi::locToAction(move.loc)) : Location::toString(move.loc,board)});
        response = state.dump();
      }
    }
    else if(command == "kata-get-history") {
      if(!pieces.empty()) {
        responseIsError = true;
        response = "kata-get-history takes no arguments";
      }
      else {
        engine->stopAndWait();
        const BoardHistory& current = engine->bot->getRootHist();
        BoardHistory hist = current.copyToInitial();
        Board board = hist.initialBoard;
        Player pla = hist.initialPla;
        nlohmann::json positions = nlohmann::json::array();
        positions.push_back(positionJson(board,hist,pla,perspective,false));
        // Use the same tolerant rule replay as loadsgf. In particular, the
        // Xiangqi origin includes its pre-existing check/repetition history.
        for(const Move& move : current.moveHistory) {
          bool suc = hist.makeBoardMoveTolerant(board,move.loc,move.pla,preventEncore);
          testAssert(suc);
          pla = getOpp(move.pla);
          positions.push_back(positionJson(board,hist,pla,perspective,false));
        }
        response = positions.dump();
      }
    }
    else if(command == "kata-get-analysis") {
      if(!pieces.empty()) {
        responseIsError = true;
        response = "kata-get-analysis takes no arguments";
      }
      else {
        engine->stopAndWait();
        const Search* search = engine->bot->getSearch();
        nlohmann::json report;
        if(!search->getAnalysisJson(engine->perspective,engine->analysisPVLen,engine->preventEncore,true,true,true,report)) {
          responseIsError = true;
          response = "No search result for the current position";
        }
        else {
          report["turnNumber"] = search->getRootHist().getCurrentTurnNumber();
          response = report.dump();
        }
      }
    }

    else if(command == "fixed_handicap") {
      int n;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for fixed_handicap but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!Global::tryStringToInt(pieces[0],n)) {
        responseIsError = true;
        response = "Could not parse number of handicap stones: '" + pieces[0] + "'";
      }
      else if(n < 2) {
        responseIsError = true;
        response = "Number of handicap stones less than 2: '" + pieces[0] + "'";
      }
      else if(!engine->bot->getRootBoard().isEmpty()) {
        responseIsError = true;
        response = "Board is not empty";
      }
      else {
        maybeSaveAvoidPatterns(false);
        engine->placeFixedHandicap(n,response,responseIsError);
      }
    }

    else if(command == "place_free_handicap") {
      int n;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for place_free_handicap but got '" + Global::concat(pieces," ") + "'";
      }
      else if(!Global::tryStringToInt(pieces[0],n)) {
        responseIsError = true;
        response = "Could not parse number of handicap stones: '" + pieces[0] + "'";
      }
      else if(n < 2) {
        responseIsError = true;
        response = "Number of handicap stones less than 2: '" + pieces[0] + "'";
      }
      else if(!engine->bot->getRootBoard().isEmpty()) {
        responseIsError = true;
        response = "Board is not empty";
      }
      else {
        maybeSaveAvoidPatterns(false);
        engine->placeFreeHandicap(n,response,responseIsError,seedRand);
      }
    }

    else if(command == "set_free_handicap") {
      if(!engine->bot->getRootBoard().isEmpty()) {
        responseIsError = true;
        response = "Board is not empty";
      }
      else {
        vector<Move> locs;
        int xSize = engine->bot->getRootBoard().x_size;
        int ySize = engine->bot->getRootBoard().y_size;
        Board board(xSize,ySize);
        for(int i = 0; i<pieces.size(); i++) {
          Loc loc;
          bool suc = tryParseLoc(pieces[i],board,engine->getCurrentRules(),loc);
          if(!suc || loc == Board::PASS_LOC) {
            responseIsError = true;
            response = "Invalid handicap location: " + pieces[i];
          }
          locs.emplace_back(loc,P_BLACK);
        }
        bool suc = board.setStonesFailIfNoLibs(locs);
        if(!suc) {
          responseIsError = true;
          response = "Handicap placement is invalid";
        }
        else {
          maybeSaveAvoidPatterns(false);
          Player pla = P_WHITE;
          BoardHistory hist(board,pla,engine->getCurrentRules(),0,engine->getHistoryModes());
          hist.setInitialTurnNumber(board.numStonesOnBoard()); //Should give more accurate temperaure and time control behavior
          vector<Move> newMoveHistory;
          engine->setPositionAndRules(pla,board,hist,board,pla,newMoveHistory);
        }
      }
    }

    else if(command == "final_score") {
      engine->stopAndWait();

      Player winner = C_EMPTY;
      double finalWhiteMinusBlackScore = 0.0;
      const BoardHistory& hist = engine->bot->getRootHist();
      winner = hist.winner;
      finalWhiteMinusBlackScore = hist.finalWhiteMinusBlackScore;

      if(!hist.isGameFinished || hist.isResignation) {
        responseIsError = true;
        response = "final_score requires a scored rules terminal; unfinished score estimation is unavailable";
      }
      else if(!hist.rules.isGo())
        response = WriteSgf::gameResultNoSgfTag(hist);
      else if(winner == C_EMPTY)
        response = "0";
      else if(winner == C_BLACK)
        response = "B+" + Global::strprintf("%.1f",-finalWhiteMinusBlackScore);
      else if(winner == C_WHITE)
        response = "W+" + Global::strprintf("%.1f",finalWhiteMinusBlackScore);
      else
        ASSERT_UNREACHABLE;
    }

    else if(command == "final_status_list") {
      int statusMode = 0;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for final_status_list but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        if(pieces[0] == "alive")
          statusMode = 0;
        else if(pieces[0] == "seki")
          statusMode = 1;
        else if(pieces[0] == "dead")
          statusMode = 2;
        else {
          responseIsError = true;
          response = "Argument to final_status_list must be 'alive' or 'seki' or 'dead'";
          statusMode = 3;
        }

        if(statusMode < 3 && !engine->bot->getRootHist().isGameFinished) {
          responseIsError = true;
          response = "final_status_list requires a finished game; ownership prediction is not supported";
          statusMode = 3;
        }
        if(statusMode < 3) {
          vector<bool> isAlive = engine->computeAnticipatedStatuses();
          Board board = engine->bot->getRootBoard();
          vector<Loc> locsToReport;

          if(statusMode == 0) {
            for(int y = 0; y<board.y_size; y++) {
              for(int x = 0; x<board.x_size; x++) {
                Loc loc = Location::getLoc(x,y,board.x_size);
                if(board.colors[loc] != C_EMPTY && isAlive[loc])
                  locsToReport.push_back(loc);
              }
            }
          }
          if(statusMode == 2) {
            for(int y = 0; y<board.y_size; y++) {
              for(int x = 0; x<board.x_size; x++) {
                Loc loc = Location::getLoc(x,y,board.x_size);
                if(board.colors[loc] != C_EMPTY && !isAlive[loc])
                  locsToReport.push_back(loc);
              }
            }
          }

          response = "";
          for(int i = 0; i<locsToReport.size(); i++) {
            Loc loc = locsToReport[i];
            if(i > 0)
              response += " ";
            response += Location::toString(loc,board);
          }
        }
      }
    }

    else if(command == "loadsgf") {
      if(pieces.size() != 1 && pieces.size() != 2) {
        responseIsError = true;
        response = "Expected one or two arguments for loadsgf but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        string filename = pieces[0];
        bool parseFailed = false;
        bool moveNumberSpecified = false;
        int moveNumber = 0;
        if(pieces.size() == 2) {
          bool suc = Global::tryStringToInt(pieces[1],moveNumber);
          moveNumber--;
          if(!suc || moveNumber < 0 || moveNumber > 10000000)
            parseFailed = true;
          else {
            moveNumberSpecified = true;
          }
        }
        if(parseFailed) {
          responseIsError = true;
          response = "Invalid value for moveNumber for loadsgf";
        }
        else {
          Board sgfInitialBoard;
          Player sgfInitialNextPla;
          BoardHistory sgfInitialHist;
          Rules sgfRules;
          Board sgfBoard;
          Player sgfNextPla;
          BoardHistory sgfHist;

          bool sgfParseSuccess = false;
          std::unique_ptr<CompactSgf> sgf = nullptr;
          try {
            sgf = CompactSgf::loadFile(filename);

            if(sgf->moves.size() > 0x3FFFFFFF)
              throw StringError("Sgf has too many moves");
            if(!moveNumberSpecified || moveNumber > sgf->moves.size())
              moveNumber = (int)sgf->moves.size();

            sgfRules = sgf->getRulesOrWarn(
              engine->getCurrentRules(), //Use current rules as default
              [&logger](const string& msg) { logger.write(msg); cerr << msg << endl; }
            );
            if(sgfRules.isXiangqi() != engine->getCurrentRules().isXiangqi())
              throw StringError("Cannot switch between Xiangqi and other games in this GTP engine");
            if(engine->primaryModel != nullptr) {
              bool rulesWereSupported;
              Rules supportedRules = engine->primaryModel->getSupportedRules(sgfRules,rulesWereSupported);
              if(!rulesWereSupported) {
                if(supportedRules == sgfRules)
                  throw StringError("SGF game is not supported by the loaded model");
                ostringstream out;
                out << "WARNING: Rules " << sgfRules.toJsonStringNoKomi()
                    << " from sgf not supported by neural net, using " << supportedRules.toJsonStringNoKomi() << " instead";
                logger.write(out.str());
                if(!logger.isLoggingToStderr())
                  cerr << out.str() << endl;
                sgfRules = supportedRules;
              }
            }
            if(engine->humanModel) {
              bool supported;
              engine->humanModel->getSupportedRules(sgfRules,supported);
              if(!supported)
                throw StringError("SGF game is not supported by the secondary model");
            }

            if(isForcingKomi)
              sgfRules.komi = forcedKomi;

            {
              //See if the rules differ, IGNORING komi differences
              Rules currentRules = engine->getCurrentRules();
              currentRules.komi = sgfRules.komi;
              if(sgfRules != currentRules) {
                ostringstream out;
                out << "Changing rules to " << sgfRules.toJsonStringNoKomi();
                logger.write(out.str());
                if(!logger.isLoggingToStderr())
                  cerr << out.str() << endl;
              }
            }

            //Set up with the BoardHistoryModes the bot will be using BEFORE replaying the
            //moves, so that any game-end adjudication happening during the replay matches what the
            //same moves would give if entered via play commands on the bot's own history.
            sgf->setupInitialBoardAndHist(
              sgfRules, sgfInitialBoard, sgfInitialNextPla, sgfInitialHist,
              engine->getHistoryModes()
            );
            if(!sgfRules.isXiangqi())
              sgfInitialHist.setInitialTurnNumber(sgfInitialBoard.numStonesOnBoard()); //Point-game turn heuristic
            sgfBoard = sgfInitialBoard;
            sgfNextPla = sgfInitialNextPla;
            sgfHist = sgfInitialHist;
            sgf->playMovesTolerant(sgfBoard,sgfNextPla,sgfHist,moveNumber,preventEncore);
            engine->checkBoardSize(sgfBoard.x_size,sgfBoard.y_size);

            sgf = nullptr;
            sgfParseSuccess = true;
          }
          catch(const StringError& err) {
            sgf = nullptr;
            responseIsError = true;
            response = "Could not load sgf: " + string(err.what());
          }
          catch(...) {
            sgf = nullptr;
            responseIsError = true;
            response = "Cannot load file";
          }

          if(sgfParseSuccess) {
            if(sgfRules.komi != engine->getCurrentRules().komi) {
              ostringstream out;
              out << "Changing komi to " << sgfRules.komi;
              logger.write(out.str());
              if(!logger.isLoggingToStderr())
                cerr << out.str() << endl;
            }
            maybeSaveAvoidPatterns(false);
            try {
              engine->setOrResetBoardSize(cfg,logger,seedRand,sgfBoard.x_size,sgfBoard.y_size,logger.isLoggingToStderr());
              engine->setPositionAndRules(sgfNextPla, sgfBoard, sgfHist, sgfInitialBoard, sgfInitialNextPla, sgfHist.moveHistory);
            }
            catch(const StringError& err) {
              responseIsError = true;
              response = "Could not load sgf: " + string(err.what());
            }
          }
        }
      }
    }

    else if(command == "printsgf") {
      if(pieces.size() != 0 && pieces.size() != 1) {
        responseIsError = true;
        response = "Expected zero or one argument for print but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        auto writeSgfToStream = [&](ostream& out) {
          double overrideFinalScore = std::numeric_limits<double>::quiet_NaN();
          const BoardHistory& hist = engine->bot->getRootHist();
          if(hist.isGameFinished && !hist.isResignation)
            overrideFinalScore = hist.finalWhiteMinusBlackScore;
          WriteSgf::writeSgf(out,"","",engine->bot->getRootHist(),NULL,true,false,overrideFinalScore);
        };

        if(pieces.size() == 0 || pieces[0] == "-") {
          ostringstream out;
          writeSgfToStream(out);
          response = out.str();
        }
        else {
          ofstream out;
          if(FileUtils::tryOpen(out,pieces[0])) {
            writeSgfToStream(out);
            out.close();
            response = "";
          }
          else {
            responseIsError = true;
            response = "Could not open or write to file: " + pieces[0];
          }
        }
      }
    }

    else if(command == "analyze" || command == "lz-analyze" || command == "kata-analyze") {
      Player pla = engine->bot->getRootPla();
      bool parseFailed = false;
      GTPEngine::AnalyzeArgs analyzeArgs = parseAnalyzeCommand(command, pieces, pla, parseFailed, engine);

      if(parseFailed) {
        responseIsError = true;
        response = "Could not parse analyze arguments or arguments out of range: '" + Global::concat(pieces," ") + "'";
      }
      else if(const char* error = engine->getAnalysisParams().getGumbelVisitCapError(pla,pla)) {
        responseIsError = true;
        response = error;
      }
      else {
        //Make sure the "equals" for GTP is printed out prior to the first analyze line, regardless of thread racing
        printGTPResponseHeader();

        engine->analyze(pla, analyzeArgs);

        //No response - currentlyAnalyzing will make sure we get a newline at the appropriate time, when stopped.
        suppressResponse = true;
        currentlyAnalyzing = true;
      }
    }

    else if(command == "kata-raw-nn") {
      int whichSymmetry = NNInputs::SYMMETRY_ALL;
      Player pla = engine->bot->getRootPla();
      double policyOptimism = engine->getGenmoveParams().rootPolicyOptimism;
      string error;
      bool parsed = parseRawNNArgs(pieces, true, pla, whichSymmetry, policyOptimism, error);
      if(parsed && engine->getCurrentRules().isXiangqi() && pla != engine->bot->getRootPla()) {
        parsed = false;
        error = "Xiangqi player must match the complete state's next player";
      }
      if(!parsed) {
        responseIsError = true;
        response = error + " for kata-raw-nn (expected an optional color, a symmetry 'all' or index [0-7], and an optional optimism [0-1])";
      }
      else {
        if(!engine->bot->getRootHist().rules.isGo() && engine->bot->getRootHist().isGameFinished) {
          responseIsError = true;
          response = "No policy is defined at a terminal placement position";
        }
        else if(whichSymmetry != NNInputs::SYMMETRY_ALL && !engine->bot->getRootHist().rules.allowsSymmetry(whichSymmetry)) {
          responseIsError = true;
          response = "Requested symmetry is invalid for the game";
        }
        else {
          const bool useHumanModel = false;
          response = engine->rawNN(whichSymmetry, policyOptimism, useHumanModel, pla);
        }
      }
    }

    else if(command == "kata-raw-human-nn") {
      int whichSymmetry = NNInputs::SYMMETRY_ALL;
      Player pla = engine->bot->getRootPla();
      double policyOptimism = engine->getGenmoveParams().rootPolicyOptimism;
      string error;
      bool parsed = parseRawNNArgs(pieces, false, pla, whichSymmetry, policyOptimism, error);
      if(parsed && engine->getCurrentRules().isXiangqi() && pla != engine->bot->getRootPla()) {
        parsed = false;
        error = "Xiangqi player must match the complete state's next player";
      }
      if(!parsed) {
        responseIsError = true;
        response = error + " for kata-raw-human-nn (expected an optional color followed by a symmetry 'all' or index [0-7])";
      }
      else {
        if(engine->humanModel == nullptr) {
          responseIsError = true;
          response = "Cannot run kata-raw-human-nn, -human-model was not provided";
        }
        else if(!(engine->getGenmoveParams().humanSLProfile.initialized || !engine->humanModel->requiresSGFMetadata())) {
          responseIsError = true;
          response = "Cannot run kata-raw-human-nn, humanSLProfile parameter was not set";
        }
        else if(!engine->getCurrentRules().isGo() && engine->bot->getRootHist().isGameFinished) {
          responseIsError = true;
          response = "No policy is defined at a terminal position";
        }
        else if(whichSymmetry != NNInputs::SYMMETRY_ALL && !engine->getCurrentRules().allowsSymmetry(whichSymmetry)) {
          responseIsError = true;
          response = "Requested symmetry is invalid for the game";
        }
        else {
          const bool useHumanModel = true;
          response = engine->rawNN(whichSymmetry, policyOptimism, useHumanModel, pla);
        }
      }
    }

    else if(command == "debug_moves") {
      PrintTreeOptions options;
      options = options.maxDepth(1);
      string printBranch;
      bool printRawStats = false;
      for(size_t i = 0; i<pieces.size(); i++) {
        if(pieces[i] == "rawstats") {
          printRawStats = true;
          continue;
        }
        if(i > 0)
          printBranch += " ";
        printBranch += pieces[i];
      }
      try {
        if(printBranch.length() > 0) {
          if(engine->getCurrentRules().isXiangqi()) {
            for(const string& token : Global::split(printBranch)) {
              Loc loc;
              if(!tryParseLoc(token,engine->bot->getRootBoard(),engine->getCurrentRules(),loc))
                throw StringError("Invalid joint action");
              options.branch_.push_back(loc);
            }
          }
          else options = options.onlyBranch(engine->bot->getRootBoard(),printBranch);
        }
      }
      catch(const StringError& e) {
        (void)e;
        responseIsError = true;
        response = "Invalid move sequence";
      }
      if(!responseIsError) {
        Search* search = engine->bot->getSearchStopAndWait();
        ostringstream sout;

        Player pla = engine->bot->getRootPla();
        Board board = engine->bot->getRootBoard();
        BoardHistory hist = engine->bot->getRootHist();
        bool allLegal = true;
        for(Loc loc: options.branch_) {
          bool suc = hist.makeBoardMoveTolerant(board, loc, pla, false);
          if(!suc) {
            allLegal = false;
            break;
          }
          pla = getOpp(pla);
        }
        if(allLegal) {
          if(hist.rules.isXiangqi()) hist.printBasicInfo(sout,board);
          else Board::printBoard(sout, board, Board::NULL_LOC, &hist.moveHistory);
        }
        search->printTree(sout, search->rootNode, options, perspective);
        if(printRawStats) {
          sout << engine->rawNNBrief(options.branch_, NNInputs::SYMMETRY_ALL);
        }
        response = filterDoubleNewlines(sout.str());
      }
    }
    else if(command == "cputime" || command == "gomill-cpu_time") {
      response = Global::doubleToString(engine->genmoveTimeSum);
    }

    else if(command == "kata-benchmark") {
      bool parsed = false;
      int64_t numVisits = 0;
      if(pieces.size() != 1) {
        responseIsError = true;
        response = "Expected one argument for kata-benchmark but got '" + Global::concat(pieces," ") + "'";
      }
      else {
        bool suc = Global::tryStringToInt64(pieces[0],numVisits);
        if(!suc || numVisits < 1) {
          responseIsError = true;
          response = "Expected a positive integer for benchmark visits: " + pieces[0];
        }
        else parsed = true;
      }

      if(parsed) {
        engine->stopAndWait();

        int boardSizeX = engine->bot->getRootBoard().x_size;
        int boardSizeY = engine->bot->getRootBoard().y_size;
        if(engine->getCurrentRules().isGo() && boardSizeX != boardSizeY) {
          responseIsError = true;
          response =
            "Current board size is " + Global::intToString(boardSizeX) + "x" + Global::intToString(boardSizeY) +
            ", no built-in benchmarks for rectangular boards";
        }
        else {
          std::unique_ptr<CompactSgf> sgf = nullptr;
          try {
            string sgfData = TestCommon::getBenchmarkSGFData(engine->getCurrentRules().game,boardSizeX,boardSizeY);
            sgf = CompactSgf::parse(sgfData);
          }
          catch(const StringError& e) {
            responseIsError = true;
            response = e.what();
          }
          if(sgf != nullptr) {
            const PlayUtils::BenchmarkResults* baseline = NULL;
            const double secondsPerGameMove = 1.0;
            const bool printElo = false;
            SearchParams params = engine->getGenmoveParams();
            params.maxTime = 1.0e20;
            params.maxPlayouts = ((int64_t)1) << 50;
            params.maxVisits = numVisits;
            //Make sure the "equals" for GTP is printed out prior to the benchmark line
            printGTPResponseHeader();

            try {
              PlayUtils::BenchmarkResults results = PlayUtils::benchmarkSearchOnPositionsAndPrint(
                params,
                *sgf,
                10,
                engine->nnEval,
                baseline,
                secondsPerGameMove,
                printElo,
                engine->primaryModel.get()
              );
              (void)results;
            }
            catch(const StringError& e) {
              responseIsError = true;
              response = e.what();
              sgf = nullptr;
            }
            if(sgf != nullptr) {
              //Act of benchmarking will write to stdout with a newline at the end, so we just need one more newline ourselves
              //to complete GTP protocol.
              suppressResponse = true;
              cout << endl;
            }
          }
        }
      }
    }

    else if(command == "kata-start-pondering") {
      if(!pieces.empty()) {
        responseIsError = true;
        response = "kata-start-pondering takes no arguments";
      }
      else if(!ponderingEnabled) {
        responseIsError = true;
        response = "Pondering is disabled";
      }
      else if(engine->bot->getRootHist().isGameFinished) {
        responseIsError = true;
        response = "Cannot ponder a finished game";
      }
      else
        maybeStartPondering = true;
    }

    else if(command == "stop") {
      //Stop any ongoing ponder or analysis
      engine->stopAndWait();
    }

    else {
      responseIsError = true;
      response = "unknown command";
    }

    if(!suppressResponse)
      printGTPResponse(response,responseIsError);

    if(shouldQuitAfterResponse)
      break;

    if(maybeStartPondering && ponderingEnabled) {
      try { engine->ponder(); }
      catch(const StringError& error) { logger.write("Ponder search not started: " + string(error.what())); }
    }

  } //Close read loop

  // Interrupt stuff if we close stdout
  if(currentlyAnalyzing) {
    currentlyAnalyzing = false;
    engine->stopAndWait();
    cout << endl;
  }
  if(currentlyGenmoving) {
    currentlyGenmoving = false;
    engine->stopAndWait();
  }


  maybeSaveAvoidPatterns(true);
  delete engine;
  engine = NULL;
  NeuralNet::globalCleanup();

  logger.write("All cleaned up, quitting");
  return 0;
}
