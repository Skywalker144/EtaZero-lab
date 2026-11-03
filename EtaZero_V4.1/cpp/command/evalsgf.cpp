#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../core/fileutils.h"
#include "../dataio/sgf.h"
#include "../game/xiangqi.h"
#include "../neuralnet/modelversion.h"
#include "../search/asyncbot.h"
#include "../search/searchnode.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../program/selfplaymanager.h"
#include "../command/commandline.h"
#include "../main.h"

using namespace std;

int MainCmds::evalsgf(const vector<string>& args) {
  Board::initHash();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  string humanModelFile;
  string sgfFile;
  int moveNumStart;
  int moveNumEnd;
  string printBranch;
  string extraMoves;
  string avoidMoves;
  string hintLoc;
  int64_t maxVisits;
  int numThreads;
  float overrideKomi;
  string overrideRules;
  bool printRootNNValues;
  bool printPolicy;
  bool printLogPolicy;
  bool printDirichletShape;
  bool printScoreNow;
  bool printAvgShorttermError;
  bool printGraph;
  bool printJson;
  int printMaxDepth;
  bool rawNN;
  string dumpNpzInputTo;
  try {
    KataGoCommandLine cmd("Run a search on a position from an sgf file, for debugging.");
    cmd.addConfigFileArg("","gtp_example.cfg");
    cmd.addModelFileArg();
    cmd.addHumanModelFileArg();

    TCLAP::UnlabeledValueArg<string> sgfFileArg("","Sgf file to analyze",true,string(),"FILE");
    TCLAP::ValueArg<int> moveNumArg("m","move-num","Sgf move num to analyze",true,0,"MOVENUM");
    TCLAP::ValueArg<int> moveNumEndArg("","move-num-end","End sgf move num range to analyze, inclusive",false,-1,"MOVENUM");

    TCLAP::ValueArg<string> printBranchArg("","print-branch","Move branch in search tree to print",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> printArg("p","print","Alias for -print-branch",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraMovesArg("","extra-moves","Extra moves to force-play before doing search",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> extraArg("e","extra","Alias for -extra-moves",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> avoidMovesArg("","avoid-moves","Avoid moves in search",false,string(),"MOVE MOVE ...");
    TCLAP::ValueArg<string> hintLocArg("","hint-loc","Hint loc",false,string(),"MOVE");
    TCLAP::ValueArg<long> visitsArg("v","visits","Set the number of visits",false,-1,"VISITS");
    TCLAP::ValueArg<int> threadsArg("t","threads","Set the number of threads",false,-1,"THREADS");
    TCLAP::ValueArg<float> overrideKomiArg("","override-komi","Artificially set komi",false,std::numeric_limits<float>::quiet_NaN(),"KOMI");
    TCLAP::ValueArg<string> overrideRulesArg("","override-rules","Artifically set rules",false,string(),"RULES");
    TCLAP::SwitchArg printRootNNValuesArg("","print-root-nn-values","Print root nn values");
    TCLAP::SwitchArg printPolicyArg("","print-policy","Print policy");
    TCLAP::SwitchArg printLogPolicyArg("","print-log-policy","Print log policy");
    TCLAP::SwitchArg printDirichletShapeArg("","print-dirichlet-shape","Print dirichlet shape");
    TCLAP::SwitchArg printScoreNowArg("","print-score-now","Print score now");
    TCLAP::SwitchArg printAvgShorttermErrorArg("","print-avg-shortterm-error","Compute and print avgShorttermError");
    TCLAP::SwitchArg printGraphArg("","print-graph","Print graph structure of the search");
    TCLAP::SwitchArg printJsonArg("","print-json","Print analysis json of the search");
    TCLAP::ValueArg<int> printMaxDepthArg("","print-max-depth","How deep to print",false,1,"DEPTH");
    TCLAP::SwitchArg rawNNArg("","raw-nn","Perform single raw neural net eval");
    TCLAP::ValueArg<string> dumpNpzInputToArg("","dump-npz-input-to","Dump the nn input tensor to npz file",false,string(),"NPZFILE");
    cmd.add(sgfFileArg);
    cmd.add(moveNumArg);
    cmd.add(moveNumEndArg);

    cmd.setShortUsageArgLimit();

    cmd.addOverrideConfigArg();

    cmd.add(printBranchArg);
    cmd.add(printArg);
    cmd.add(extraMovesArg);
    cmd.add(extraArg);
    cmd.add(avoidMovesArg);
    cmd.add(hintLocArg);
    cmd.add(visitsArg);
    cmd.add(threadsArg);
    cmd.add(overrideKomiArg);
    cmd.add(overrideRulesArg);
    cmd.add(printRootNNValuesArg);
    cmd.add(printPolicyArg);
    cmd.add(printLogPolicyArg);
    cmd.add(printDirichletShapeArg);
    cmd.add(printScoreNowArg);
    cmd.add(printAvgShorttermErrorArg);
    cmd.add(printGraphArg);
    cmd.add(printJsonArg);
    cmd.add(printMaxDepthArg);
    cmd.add(rawNNArg);
    cmd.add(dumpNpzInputToArg);
    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    humanModelFile = cmd.getHumanModelFile();
    sgfFile = sgfFileArg.getValue();
    moveNumStart = moveNumArg.getValue();
    moveNumEnd = moveNumEndArg.getValue();
    printBranch = printBranchArg.getValue();
    string print = printArg.getValue();
    extraMoves = extraMovesArg.getValue();
    string extra = extraArg.getValue();
    avoidMoves = avoidMovesArg.getValue();
    hintLoc = hintLocArg.getValue();
    maxVisits = (int64_t)visitsArg.getValue();
    numThreads = threadsArg.getValue();
    overrideKomi = overrideKomiArg.getValue();
    overrideRules = overrideRulesArg.getValue();
    printRootNNValues = printRootNNValuesArg.getValue();
    printPolicy = printPolicyArg.getValue();
    printLogPolicy = printLogPolicyArg.getValue();
    printDirichletShape = printDirichletShapeArg.getValue();
    printScoreNow = printScoreNowArg.getValue();
    printAvgShorttermError = printAvgShorttermErrorArg.getValue();
    printGraph = printGraphArg.getValue();
    printJson = printJsonArg.getValue();
    printMaxDepth = printMaxDepthArg.getValue();
    rawNN = rawNNArg.getValue();
    dumpNpzInputTo = dumpNpzInputToArg.getValue();

    if(printBranch.length() > 0 && print.length() > 0) {
      cerr << "Error: -print-branch and -print both specified" << endl;
      return 1;
    }
    if(printBranch.length() <= 0)
      printBranch = print;

    if(extraMoves.length() > 0 && extra.length() > 0) {
      cerr << "Error: -extra-moves and -extra both specified" << endl;
      return 1;
    }
    if(extraMoves.length() <= 0)
      extraMoves = extra;

    if(moveNumEnd < moveNumStart)
      moveNumEnd = moveNumStart;

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  //Parse rules -------------------------------------------------------------------
  Rules defaultRules = Rules::getTrompTaylorish();
  Player perspective = Setup::parseReportAnalysisWinrates(cfg,P_BLACK);

  //Parse sgf file and board ------------------------------------------------------------------

  std::unique_ptr<CompactSgf> sgf = CompactSgf::loadFile(sgfFile);

  Board board;
  Player nextPla;
  BoardHistory hist;

  //Set for real after the neural net is loaded, from the params and the model's declaration.
  BoardHistoryModes historyModes;
  auto parseMove = [&board,&hist](const string& text) {
    return hist.rules.isXiangqi() ? Xiangqi::actionToLoc(Xiangqi::parseAction(text)) : Location::ofString(text,board);
  };
  auto parseMoves = [&board,&hist,&parseMove](const string& text) {
    if(!hist.rules.isXiangqi()) return Location::parseSequence(text,board);
    vector<Loc> result;
    for(const string& token : Global::split(text)) result.push_back(parseMove(token));
    return result;
  };
  auto moveToString = [&board,&hist](Loc move) {
    return hist.rules.isXiangqi() ? Xiangqi::actionToString(Xiangqi::locToAction(move)) : Location::toString(move,board);
  };
  auto printPosition = [](ostream& out, const Board& b, const BoardHistory& h) {
    if(h.rules.isXiangqi()) h.printBasicInfo(out,b);
    else Board::printBoard(out,b,Board::NULL_LOC,&h.moveHistory);
  };
  auto setUpBoardUsingRules = [&](const Rules& initialRules, int moveNum) {
    //Set up before replaying moves so any adjudication during replay is consistent with the searches.
    sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist, historyModes);
    vector<Move>& moves = sgf->moves;

    if(!isnan(overrideKomi)) {
      if(!hist.rules.isGo() && overrideKomi != 0)
        throw StringError("Non-Go games require zero komi");
      if(overrideKomi > board.x_size * board.y_size + NNPos::KOMI_CLIP_RADIUS || overrideKomi < -board.x_size * board.y_size - NNPos::KOMI_CLIP_RADIUS)
        throw StringError("Invalid komi, too much greater than the area of the board");
      hist.setKomi(overrideKomi);
    }

    if(moveNum < 0)
      throw StringError("Move num " + Global::intToString(moveNum) + " requested but must be non-negative");
    if(moveNum > moves.size())
      throw StringError("Move num " + Global::intToString(moveNum) + " requested but sgf has only " + Global::int64ToString(moves.size()));

    sgf->playMovesTolerant(board,nextPla,hist,moveNum,false);

    vector<Loc> extraMoveLocs = parseMoves(extraMoves);
    for(size_t i = 0; i<extraMoveLocs.size(); i++) {
      Loc loc = extraMoveLocs[i];
      if(!hist.isLegal(board,loc,nextPla)) {
        printPosition(cerr,board,hist);
        cerr << "Extra illegal move for " << PlayerIO::colorToChar(nextPla) << ": " << moveToString(loc) << endl;
        throw StringError("Illegal extra move");
      }
      hist.makeBoardMoveAssumeLegal(board,loc,nextPla,NULL);
      nextPla = getOpp(nextPla);
    }
  };

  Rules initialRules = sgf->getRulesOrWarn(
    defaultRules,
    [](const string& msg) { cout << msg << endl; }
  );
  if(overrideRules != "") {
    initialRules = Rules::parseRules(overrideRules);
  }
  if(cfg.contains("game") && Rules::parseGame(cfg.getString("game")) != initialRules.game)
    throw StringError("evalsgf config game and SGF rules differ");
  if(!cfg.contains("game")) cfg.overrideKey("game",Rules::gameName(initialRules.game));
  if(printScoreNow && !initialRules.isGo())
    throw StringError("-print-score-now is a Go rules scoring operation");

  // Set up once now for error catcihng
  setUpBoardUsingRules(initialRules,moveNumStart);

  //Parse move sequence arguments------------------------------------------

  PrintTreeOptions options;
  options = options.maxDepth(printMaxDepth);
  if(printBranch.length() > 0)
    options.branch_ = parseMoves(printBranch);
  options = options.printAvgShorttermError(printAvgShorttermError);

  //Load neural net and start bot------------------------------------------

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.write("Engine starting...");

  bool hasHumanModel = humanModelFile != "";
  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP,hasHumanModel);
  if(maxVisits < -1 || maxVisits == 0)
    throw StringError("maxVisits: invalid value");
  else if(maxVisits == -1)
    logger.write("No max visits specified on cmdline, using defaults in " + cfg.getFileName());
  else {
    params.maxVisits = maxVisits;
    params.maxPlayouts = maxVisits; //Also set this so it doesn't cap us either
  }
  if(numThreads < -1 || numThreads == 0)
    throw StringError("numThreads: invalid value");
  else if(numThreads == -1)
    logger.write("No num threads specified on cmdline, using defaults in " + cfg.getFileName());
  else {
    params.numThreads = numThreads;
  }

  string searchRandSeed;
  if(cfg.contains("searchRandSeed"))
    searchRandSeed = cfg.getString("searchRandSeed");
  else
    searchRandSeed = Global::uint64ToString(seedRand.nextUInt64());

  unique_ptr<SelfplayModel> primaryModel;
  unique_ptr<SelfplayModel> humanModel;
  const bool useMuZero = cfg.contains("useMuZero") && cfg.getBool("useMuZero");
  {
    Setup::initializeSession(cfg);
    int expectedConcurrentEvals = params.numThreads;
    bool disableFP16 = false;
    string expectedSha256 = "";
    auto loadModel = [&](const string& file, bool muzero) -> unique_ptr<SelfplayModel> {
      if(muzero) {
        auto bundle = make_shared<MuZero::ModelBundle>(file);
        if(Rules::parseGame(bundle->game()) != initialRules.game)
          throw StringError("MuZero evalsgf model and SGF game differ");
        return Setup::initializeMuZeroSelfplayModel(file,file,std::move(bundle),cfg,logger,seedRand,
          expectedConcurrentEvals,false,0,Setup::SETUP_FOR_GTP);
      }
      return make_unique<SelfplayModel>(Setup::initializeNNEvaluator(
        file,file,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
        useMuZero && primaryModel ? primaryModel->getNNXLen() : board.x_size,
        useMuZero && primaryModel ? primaryModel->getNNYLen() : board.y_size,
        Setup::MaxBatchSizeRequest::fromConcurrency(),!useMuZero,disableFP16,Setup::SETUP_FOR_GTP
      ));
    };
    primaryModel = loadModel(modelFile,useMuZero);
    if(hasHumanModel) humanModel = loadModel(humanModelFile,FileUtils::isDirectory(humanModelFile));
  }
  NNEvaluator* nnEval = primaryModel->boardEvaluator();
  NNEvaluator* humanEval = humanModel ? humanModel->boardEvaluator() : nullptr;
  MuZero::PositionEvaluator* muZeroEval = primaryModel->muZeroEvaluator();
  MuZero::PositionEvaluator* humanMuZeroEval = humanModel ? humanModel->muZeroEvaluator() : nullptr;
  logger.write("Loaded neural net");

  historyModes = muZeroEval ? Search::resolveHistoryModes(params,*muZeroEval) : Search::resolveHistoryModes(params,nnEval);

  // Raw observations and NPZ diagnostics use the same initial inference as GTP.
  // Recurrent inference remains inside the native latent search.
  auto evaluate = [&](SelfplayModel& model, const MiscNNInputParams& inputParams, NNResultBuf& buf) {
    if(!hist.rules.isGo() && hist.isGameFinished)
      throw StringError("No policy is defined at a terminal non-Go position");
    if(model.muZeroEvaluator()) {
      MuZero::PositionBuffer buffer;
      buf.result = model.muZeroEvaluator()->initial(board,hist,nextPla,&params.humanSLProfile,inputParams,buffer,seedRand,true);
    }
    else model.boardEvaluator()->evaluate(board,hist,nextPla,&params.humanSLProfile,inputParams,buf,true);
  };

  {
    bool rulesWereSupported;
    Rules supportedRules = primaryModel->getSupportedRules(initialRules,rulesWereSupported);
    if(!rulesWereSupported) {
      cout << "Warning: Rules " << initialRules << " from sgf not supported by neural net, using " << supportedRules << " instead" << endl;
      initialRules = supportedRules;
    }
  }

  for(int moveNum = moveNumStart; moveNum <= moveNumEnd; moveNum++) {
    setUpBoardUsingRules(initialRules,moveNum);

    // {
    //   sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
    //   vector<Move>& moves = sgf->moves;

    //   for(size_t i = 0; i<moves.size(); i++) {
    //     bool preventEncore = false;
    //     bool suc = hist.makeBoardMoveTolerant(board,moves[i].loc,moves[i].pla,preventEncore);
    //     assert(suc);
    //     nextPla = getOpp(moves[i].pla);

    //     MiscNNInputParams nnInputParams;
    //     nnInputParams.nnPolicyTemperature = 1.2f;
    //     NNResultBuf buf;
    //     bool skipCache = true;
    //     nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

    //     NNOutput* nnOutput = buf.result.get();
    //     vector<double> probs;
    //     for(int y = 0; y<board.y_size; y++) {
    //       for(int x = 0; x<board.x_size; x++) {
    //         int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
    //         float prob = nnOutput->policyProbs[pos];
    //         probs.push_back(prob);
    //       }
    //     }
    //     std::sort(probs.begin(),probs.end());
    //     cout << probs[probs.size()-1] << " " << probs[probs.size()-2] << " " << probs[probs.size()-3] << endl;
    //   }
    //   continue;
    // }

    // {
    //   sgf->setupInitialBoardAndHist(initialRules, board, nextPla, hist);
    //   vector<Move>& moves = sgf->moves;

    //   for(size_t i = 0; i<moves.size(); i++) {
    //     bool preventEncore = false;
    //     bool suc = hist.makeBoardMoveTolerant(board,moves[i].loc,moves[i].pla,preventEncore);
    //     assert(suc);
    //     nextPla = getOpp(moves[i].pla);

    //     MiscNNInputParams nnInputParams;
    //     nnInputParams.drawEquivalentWinsForWhite = params.drawEquivalentWinsForWhite;
    //     NNResultBuf buf;
    //     bool skipCache = true;
    //     nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

    //     NNOutput* nnOutput = buf.result.get();
    //     cout << nnOutput->whiteWinProb << " " << nnOutput->shorttermWinlossError << " "
    //   }
    //   continue;
    // }

    //Check for unused config keys
    cfg.warnUnusedKeys(cerr,&logger);
    Setup::maybeWarnHumanSLParams(params,nnEval,humanEval,cerr,&logger,muZeroEval,humanMuZeroEval);

    if(rawNN) {
      NNResultBuf buf;
      MiscNNInputParams nnInputParams;
      nnInputParams.drawEquivalentWinsForWhite = params.drawEquivalentWinsForWhite;
      evaluate(*primaryModel,nnInputParams,buf);

      cout << "Rules: " << hist.rules << endl;
      cout << "Encore phase " << hist.encorePhase << endl;
      printPosition(cout,board,hist);
      buf.result->debugPrint(cout,board);

      if(humanModel) {
        //Featurize per the human model's own resolution, which may differ from the main model's.
        MiscNNInputParams humanNNInputParams = nnInputParams;
        const auto modes = humanMuZeroEval ? Search::resolveHistoryModes(params,*humanMuZeroEval) : Search::resolveHistoryModes(params,humanEval);
        humanNNInputParams.passAliveSuicideRulesOverride = modes.alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
        humanNNInputParams.excludeTerritoryAdjAtariOverride = modes.excludeTerritoryAdjacentToAtari ? 1 : 0;
        evaluate(*humanModel,humanNNInputParams,buf);
        buf.result->debugPrint(cout,board);
      }
      continue;
    }

    unique_ptr<AsyncBot> bot;
    if(muZeroEval && humanMuZeroEval)
      bot = make_unique<AsyncBot>(params,*muZeroEval,*humanMuZeroEval,&logger,searchRandSeed);
    else if(muZeroEval)
      bot = make_unique<AsyncBot>(params,*muZeroEval,humanEval,&logger,searchRandSeed);
    else if(humanMuZeroEval)
      bot = make_unique<AsyncBot>(params,nnEval,*humanMuZeroEval,&logger,searchRandSeed);
    else bot = make_unique<AsyncBot>(params,nnEval,humanEval,&logger,searchRandSeed);

    bot->setPosition(nextPla,board,hist);
    if(hintLoc != "") {
      bot->setRootHintLoc(parseMove(hintLoc));
    }

    if(avoidMoves != "") {
      vector<Loc> avoidMoveLocs = parseMoves(avoidMoves);
      vector<int> avoidMoveUntilByLoc(bot->getSearch()->getMoveLocBufferSize(),0);
      for(Loc loc: avoidMoveLocs)
        avoidMoveUntilByLoc[loc] = 1;
      bot->setAvoidMoveUntilByLoc(avoidMoveUntilByLoc,avoidMoveUntilByLoc);
    }


    //Print initial state----------------------------------------------------------------
    const Search* search = bot->getSearchStopAndWait();
    ostringstream sout;
    sout << "Rules: " << hist.rules << endl;
    sout << "Encore phase " << hist.encorePhase << endl;
    printPosition(sout,board,hist);

    if(options.branch_.size() > 0) {
      Board copy = board;
      BoardHistory copyHist = hist;
      Player pla = nextPla;
      for(int i = 0; i<options.branch_.size(); i++) {
        Loc loc = options.branch_[i];
        if(!copyHist.isLegal(copy,loc,pla)) {
          if(muZeroEval) {
            sout << "Branch leaves the real game; subsequent nodes are latent\n";
            break;
          }
          printPosition(cerr,board,hist);
          cerr << "Branch Illegal move for " << PlayerIO::colorToChar(pla) << ": " << moveToString(loc) << endl;
          return 1;
        }
        copyHist.makeBoardMoveAssumeLegal(copy,loc,pla,NULL);
        pla = getOpp(pla);
      }
      printPosition(sout,copy,copyHist);
    }

    sout << "\n";
    logger.write(sout.str());
    sout.clear();

    //Search!----------------------------------------------------------------

    ClockTimer timer;
    primaryModel->clearStats();
    if(humanModel) humanModel->clearStats();
    Loc loc = bot->genMoveSynchronous(bot->getSearch()->rootPla,TimeControls());
    (void)loc;

    //Postprocess------------------------------------------------------------



    if(printRootNNValues) {
      const NNOutput* nnOutput = search->rootNode->getNNOutput();
      if(nnOutput != NULL) {
        cout << "White win: " << nnOutput->whiteWinProb << endl;
        cout << "White draw: " << nnOutput->whiteDrawProb << endl;
        cout << "White loss: " << nnOutput->whiteLossProb << endl;
        cout << "White noresult: " << nnOutput->whiteNoResultProb << endl;
        cout << "Var time left " << nnOutput->varTimeLeft << endl;
        cout << "Shortterm winloss error " << nnOutput->shorttermWinlossError << endl;
      }
    }

    // {
    //   ReportedSearchValues values;
    //   bool suc = search->getRootValues(values);
    //   if(!suc)
    //     cout << "Unsuccessful getting root values" << endl;
    //   else
    //     cout << values << endl;
    // }
    // {
    //   ReportedSearchValues values;
    //   bool suc = search->getPrunedRootValues(values);
    //   if(!suc)
    //     cout << "Unsuccessful getting pruned root values" << endl;
    //   else
    //     cout << values << endl;
    // }




    if(printPolicy) {
      auto doPrintPolicy = [&](const float* policyProbs, int nnXLen, int nnYLen) {
        if(hist.rules.isXiangqi()) {
          for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) if(policyProbs[action] >= 0)
            cout << Xiangqi::actionToString(action) << " " << Global::strprintf("%5.2f",policyProbs[action]*100) << endl;
          return;
        }
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            double prob = policyProbs[pos];
            if(prob < 0)
              cout << "  -  " << " ";
            else
              cout << Global::strprintf("%5.2f",prob*100) << " ";
          }
          cout << endl;
        }
        if(hist.rules.isGo()) {
          double prob = policyProbs[NNPos::locToPos(Board::PASS_LOC,board.x_size,nnXLen,nnYLen)];
          cout << "Pass " << Global::strprintf("%5.2f",prob*100) << endl;
        }
      };

      const NNOutput* nnOutput = search->rootNode->getNNOutput();
      if(nnOutput != NULL) {
        const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
        cout << "Root policy: " << endl;
        doPrintPolicy(policyProbs, nnOutput->nnXLen, nnOutput->nnYLen);
      }
      const NNOutput* humanOutput = search->rootNode->getHumanOutput();
      if(humanOutput != NULL) {
        const float* policyProbs = humanOutput->getPolicyProbsMaybeNoised();
        cout << "Root human policy: " << endl;
        doPrintPolicy(policyProbs, humanOutput->nnXLen, humanOutput->nnYLen);
      }
    }
    if(printLogPolicy) {
      auto doPrintLogPolicy = [&](const float* policyProbs, int nnXLen, int nnYLen) {
        if(hist.rules.isXiangqi()) {
          for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) if(policyProbs[action] >= 0)
            cout << Xiangqi::actionToString(action) << " " << Global::strprintf("%+5.2f",log(policyProbs[action])) << endl;
          return;
        }
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            double prob = policyProbs[pos];
            if(prob < 0)
              cout << "  _  " << " ";
            else
              cout << Global::strprintf("%+5.2f",log(prob)) << " ";
          }
          cout << endl;
        }
        if(hist.rules.isGo()) {
          double prob = policyProbs[NNPos::locToPos(Board::PASS_LOC,board.x_size,nnXLen,nnYLen)];
          cout << "Pass " << Global::strprintf("%+5.2f",log(prob)) << endl;
        }
      };

      const NNOutput* nnOutput = search->rootNode->getNNOutput();
      if(nnOutput != NULL) {
        const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
        cout << "Root policy: " << endl;
        doPrintLogPolicy(policyProbs, nnOutput->nnXLen, nnOutput->nnYLen);
      }
      const NNOutput* humanOutput = search->rootNode->getHumanOutput();
      if(humanOutput != NULL) {
        const float* policyProbs = humanOutput->getPolicyProbsMaybeNoised();
        cout << "Root human policy: " << endl;
        doPrintLogPolicy(policyProbs, humanOutput->nnXLen, humanOutput->nnYLen);
      }
    }

    if(printDirichletShape) {
      const NNOutput* nnOutput = search->rootNode->getNNOutput();
      if(nnOutput != NULL) {
        const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
        vector<double> alphaDistr(search->policySize,-1.0);
        Search::computeDirichletAlphaDistribution(search->policySize, policyProbs, alphaDistr.data());
        cout << "Dirichlet alphas with 10.83 total concentration: " << endl;
        if(hist.rules.isXiangqi()) {
          for(int action = 0; action < Xiangqi::NUM_ACTIONS; action++) if(alphaDistr[action] >= 0)
            cout << Xiangqi::actionToString(action) << " " << Global::strprintf("%5.4f",alphaDistr[action]*10.83) << endl;
        }
        else {
          for(int y = 0; y<board.y_size; y++) {
            for(int x = 0; x<board.x_size; x++) {
              int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
              double alpha = alphaDistr[pos];
              if(alpha < 0)
                cout << "  -  " << " ";
              else
                cout << Global::strprintf("%5.4f",alpha * 10.83) << " ";
            }
            cout << endl;
          }
          if(hist.rules.isGo()) {
            double alpha = alphaDistr[NNPos::locToPos(Board::PASS_LOC,board.x_size,nnOutput->nnXLen,nnOutput->nnYLen)];
            cout << "Pass " << Global::strprintf("%5.2f",alpha * 10.83) << endl;
          }
        }
      }
    }

    if(printScoreNow) {
      sout << "Score now (ROOT position):\n";
      BoardHistory copyHist(hist);
      Color area[Board::MAX_ARR_SIZE];
      copyHist.endAndScoreGameNow(board,area);

      for(int y = 0; y<board.y_size; y++) {
        for(int x = 0; x<board.x_size; x++) {
          Loc l = Location::getLoc(x,y,board.x_size);
          sout << PlayerIO::colorToChar(area[l]);
        }
        sout << endl;
      }
      sout << endl;

      sout << "Komi: " << copyHist.rules.komi << endl;
      sout << "WBonus: " << copyHist.whiteBonusScore << endl;
      sout << "Final: "; WriteSgf::printGameResult(sout, copyHist); sout << endl;
    }

    sout << "Time taken: " << timer.getSeconds() << "\n";
    sout << "Root visits: " << search->getRootVisits() << "\n";
    sout << "NN rows: " << primaryModel->numRowsProcessed() << endl;
    sout << "NN batches: " << primaryModel->numBatchesProcessed() << endl;
    sout << "NN avg batch size: " << primaryModel->averageProcessedBatchSize() << endl;
    std::vector<SearchNode*> nodes = bot->getSearchStopAndWait()->enumerateTreePostOrder();
    sout << "True number of tree nodes: " << nodes.size() << endl;
    sout << "PV: ";
    search->printPV(sout, search->rootNode, 25);
    sout << "\n";
    sout << "Tree:\n";
    search->printTree(sout, search->rootNode, options, perspective);
    logger.write(sout.str());

    if(printGraph) {
      std::reverse(nodes.begin(),nodes.end());
      std::map<SearchNode*,size_t> idxOfNode;
      for(size_t nodeIdx = 0; nodeIdx<nodes.size(); nodeIdx++)
        idxOfNode[nodes[nodeIdx]] = nodeIdx;

      for(size_t nodeIdx = 0; nodeIdx<nodes.size(); nodeIdx++) {
        SearchNode& node = *(nodes[nodeIdx]);
        SearchNodeChildrenReference children = node.getChildren();
        int childrenCapacity = children.getCapacity();
        for(int i = 0; i<childrenCapacity; i++) {
          SearchNode* child = children[i].getIfAllocated();
          if(child == NULL)
            break;
          cout << nodeIdx << " -> " << idxOfNode[child] << "\n";
        }
      }
      cout << endl;
    }

    if(printJson) {
      int analysisPVLen = 7;
      bool preventEncore = false;
      bool includePolicy = printPolicy;
      bool includePVVisits = true;
      nlohmann::json ret;
      bool includeNoResultValue = true;
      bool suc = search->getAnalysisJson(
        perspective,
        analysisPVLen,
        preventEncore,
        includePolicy,
        includePVVisits,
        includeNoResultValue,
        ret
      );
      if(suc) {
        cout << ret << endl;
      }
    }



    if(dumpNpzInputTo != "") {
      bool inputsUseNHWC = false;
      int nnXLen = primaryModel->getNNXLen();
      int nnYLen = primaryModel->getNNYLen();
      int modelVersion = muZeroEval ? muZeroEval->getModel().representationDesc().modelVersion : nnEval->getModelVersion();
      int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
      int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);

      NumpyBuffer<float> binaryInputNCHW(std::vector<int64_t>({1,numSpatialFeatures,nnYLen,nnXLen}));
      NumpyBuffer<float> globalInputNC(std::vector<int64_t>({1,numGlobalFeatures}));

      MiscNNInputParams nnInputParams;
      nnInputParams.symmetry = 0;
      nnInputParams.policyOptimism = params.rootPolicyOptimism;
      NNInputs::fillRow(NNModelVersion::getInputsVersion(modelVersion),board,hist,nextPla,nnInputParams,
        nnXLen,nnYLen,inputsUseNHWC,binaryInputNCHW.data,globalInputNC.data);

      // ZipFile retains references to these buffers until close().
      unique_ptr<NumpyBuffer<float>> metadataInputNC;
      ZipFile zipFile(dumpNpzInputTo);
      uint64_t numBytes;

      numBytes = binaryInputNCHW.prepareHeaderWithNumRows(1);
      zipFile.writeBuffer("binaryInputNCHW", binaryInputNCHW.dataIncludingHeader, numBytes);
      numBytes = globalInputNC.prepareHeaderWithNumRows(1);
      zipFile.writeBuffer("globalInputNC", globalInputNC.dataIncludingHeader, numBytes);
      if(primaryModel->requiresSGFMetadata()) {
        metadataInputNC = make_unique<NumpyBuffer<float>>(std::vector<int64_t>({1,SGFMetadata::METADATA_INPUT_NUM_CHANNELS}));
        SGFMetadata::fillMetadataRow(&params.humanSLProfile,metadataInputNC->data,nextPla,board.x_size*board.y_size);
        numBytes = metadataInputNC->prepareHeaderWithNumRows(1);
        zipFile.writeBuffer("metadataInputNC",metadataInputNC->dataIncludingHeader,numBytes);
      }
      zipFile.close();
      cout << "Wrote to " << dumpNpzInputTo << endl;

      NNResultBuf buf;
      evaluate(*primaryModel,nnInputParams,buf);
      buf.result->debugPrint(cout,board);
    }

  }

  humanModel.reset();
  primaryModel.reset();
  NeuralNet::globalCleanup();

  return 0;
}
