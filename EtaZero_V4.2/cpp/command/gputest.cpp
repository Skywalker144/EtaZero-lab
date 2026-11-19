#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/selfplaymanager.h"
#include "../program/playutils.h"
#include "../tests/tests.h"
#include "../command/commandline.h"
#include "../main.h"

#include <chrono>
#include <map>
#include <sstream>
#include <fstream>

using namespace std;


int MainCmds::testgpuerror(const vector<string>& args) {
  Board::initHash();
  Rand seedRand;
  ConfigParser cfg;
  string modelFile, boardSizeDataset, referenceFileName;
  vector<string> sgfFiles;
  bool quickTest, sizeSpecified, stepsSpecified;
  int recurrentSteps;
  try {
    KataGoCommandLine cmd("Test inference error between FP32 and current precision, with and without batching");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> sizesArg("","boardsize","Go dataset (9,13,19,10x14,rectangle); other games use a board size, e.g. 9 or 9x10",false,"","SIZE");
    TCLAP::MultiArg<string> sgfsArg("","sgf","Read positions from same-game SGF files instead of the built-in dataset",false,"FILE");
    TCLAP::SwitchArg quickArg("","quick","Source shorter test, selecting 30% of positions");
    TCLAP::ValueArg<string> referenceArg("","reference-file","Eigen writes reference outputs; other backends load them. MuZero also uses FILE.recurrent-N per depth. Otherwise compare with unbatched FP32",false,"","FILE");
    TCLAP::ValueArg<int> stepsArg("","muzero-recurrent-steps","Check initial and every recurrent depth up to this value (default 3 for MuZero)",false,3,"N");
    cmd.add(sizesArg); cmd.add(sgfsArg); cmd.add(quickArg); cmd.add(referenceArg); cmd.add(stepsArg);
    cmd.setShortUsageArgLimit(); cmd.addOverrideConfigArg(); cmd.parseArgs(args);
    modelFile = cmd.getModelFile(); boardSizeDataset = sizesArg.getValue(); sizeSpecified = sizesArg.isSet();
    sgfFiles = sgfsArg.getValue(); quickTest = quickArg.getValue(); referenceFileName = referenceArg.getValue();
    recurrentSteps = stepsArg.getValue(); stepsSpecified = stepsArg.isSet(); cmd.getConfig(cfg);
    if(!sgfFiles.empty() && sizeSpecified) throw StringError("testgpuerror: sgf and boardsize are mutually exclusive");
    if(recurrentSteps < 0) throw StringError("testgpuerror: recurrent steps must be nonnegative");
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger(NULL,true,false,false);
  logger.write("Version " + Version::getGitRevisionWithBackend());
  logger.write("Testing " + modelFile);
  const string muKey = cfg.contains("useMuZero0") ? "useMuZero0" : "useMuZero";
  const bool muzero = cfg.contains(muKey) && cfg.getBool(muKey);
  if(!muzero && stepsSpecified) throw StringError("testgpuerror: recurrent steps require useMuZero=true");
  if(!muzero) recurrentSteps = 0;
  shared_ptr<const MuZero::ModelBundle> bundle;
  if(muzero) bundle = make_shared<MuZero::ModelBundle>(modelFile);
  Rules::Game game = cfg.contains("rules") ? Rules::parseRules(cfg.getString("rules")).game :
    bundle ? Rules::parseGame(bundle->game()) : Rules::Game::GO;
  if(cfg.contains("game")) {
    Rules::Game declared = Rules::parseGame(cfg.getString("game"));
    if(cfg.contains("rules") && game != declared) throw StringError("testgpuerror: config game and rules differ");
    game = declared;
  }
  int nnXSize = 0, nnYSize = 0;
  vector<string> sgfs;
  for(const string& file : sgfFiles) {
    const string data = FileUtils::readFile(file);
    auto sgf = CompactSgf::parse(data);
    const Rules rules = sgf->getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
    if((cfg.contains("game") || cfg.contains("rules") || bundle || !sgfs.empty()) && game != rules.game)
      throw StringError("testgpuerror: SGF and configured game differ");
    game = rules.game;
    nnXSize = std::max(nnXSize,sgf->xSize); nnYSize = std::max(nnYSize,sgf->ySize);
    sgfs.push_back(data);
  }
  if(bundle && Rules::parseGame(bundle->game()) != game) throw StringError("testgpuerror: MuZero model and game differ");
  if(sgfs.empty()) {
    if(game == Rules::Game::GO) {
      if(boardSizeDataset.empty()) boardSizeDataset = bundle ? Global::intToString(bundle->posLen()) : "19";
      if(boardSizeDataset == "19" || boardSizeDataset == "rectangle") nnXSize = nnYSize = 19;
      else if(boardSizeDataset == "13") nnXSize = nnYSize = 13;
      else if(boardSizeDataset == "9") nnXSize = nnYSize = 9;
      else if(boardSizeDataset == "10x14") { nnXSize = 10; nnYSize = 14; }
      else throw StringError("Unknown Go GPU-error dataset: " + boardSizeDataset + "; use -sgf for other sizes");
    }
    else {
      nnXSize = game == Rules::Game::XIANGQI ? 9 : bundle ? bundle->posLen() : 19;
      nnYSize = game == Rules::Game::XIANGQI ? 10 : nnXSize;
      if(boardSizeDataset.empty()) Setup::loadDefaultBoardXYSize(cfg,logger,nnXSize,nnYSize);
      else {
        const auto dimensions = Global::split(boardSizeDataset,'x');
        if(dimensions.empty() || dimensions.size() > 2) throw StringError("Invalid testgpuerror board size");
        nnXSize = Global::stringToInt(dimensions[0]);
        nnYSize = dimensions.size() == 2 ? Global::stringToInt(dimensions[1]) : nnXSize;
      }
      sgfs.push_back(TestCommon::getBenchmarkSGFData(game,nnXSize,nnYSize));
    }
  }
  if(!cfg.contains("game")) cfg.overrideKey("game",Rules::gameName(game));
  if(bundle && (nnXSize > bundle->posLen() || nnYSize > bundle->posLen()))
    throw StringError("testgpuerror: MuZero canvas is smaller than the dataset");
  logger.write("Testing game " + Rules::gameName(game) + (sgfFiles.empty() ? " built-in dataset " + boardSizeDataset : " SGF positions"));
  int maxBatchSize;
  if(cfg.contains("nnMaxBatchSize")) {
    maxBatchSize = cfg.getInt("nnMaxBatchSize",1,65536);
    logger.write("For batch test, using batch size from nnMaxBatchSize in config: " + Global::intToString(maxBatchSize));
  }
  else if(cfg.contains("numSearchThreads")) {
    maxBatchSize = cfg.getInt("numSearchThreads",1,65536);
    logger.write("For batch test, using batch size from numSearchThreads in config: " + Global::intToString(maxBatchSize));
  }
  else {
    maxBatchSize = 16;
    logger.write("For batch test, using default batch size 16");
  }
  auto loadModel = [&](bool disableFP16) {
    if(muzero) return Setup::initializeMuZeroSelfplayModel(modelFile,modelFile,bundle,cfg,logger,seedRand,
      maxBatchSize,false,0,Setup::SETUP_FOR_BENCHMARK,Setup::MaxBatchSizeRequest::explicitSize(maxBatchSize),nullptr,disableFP16);
    return make_unique<SelfplayModel>(Setup::initializeNNEvaluator(modelFile,modelFile,"",cfg,logger,seedRand,
      maxBatchSize,nnXSize,nnYSize,Setup::MaxBatchSizeRequest::explicitSize(maxBatchSize),false,disableFP16,Setup::SETUP_FOR_BENCHMARK));
  };
  logger.write("Initializing nneval using current config...");
  auto model = loadModel(false);
  if(!model->isNeuralNetEnabled()) throw StringError("testgpuerror requires a real neural net model");
  unique_ptr<SelfplayModel> fp32;
  // Ordinary backends retain the source actual-FP16 check. MuZero auto is
  // compared to explicit FP32 too, since its service may select FP16 per GPU.
  if(model->boardEvaluator() ? model->boardEvaluator()->isAnyThreadUsingFP16() : model->getUsingFP16Mode() != enabled_t::False) {
    logger.write("Initializing nneval in fp32...");
    fp32 = loadModel(true);
  }
  const double optimism = cfg.contains("policyOptimism") ? cfg.getDouble("policyOptimism") : 0.0;
  const double pda = cfg.contains("playoutDoublingAdvantage") ? cfg.getDouble("playoutDoublingAdvantage") : 0.0;
  const double temperature = cfg.contains("rootPolicyTemperature") ? cfg.getDouble("rootPolicyTemperature") : 1.0;
  if(optimism != 0 || pda != 0 || temperature != 1) {
    logger.write("Using policyOptimismForTest " + Global::doubleToString(optimism));
    logger.write("Using pdaForTest " + Global::doubleToString(pda));
    logger.write("Using nnPolicyTemperatureForTest " + Global::doubleToString(temperature));
  }
  bool success;
  if(!muzero && sgfs.empty()) {
    bool fp32BatchSuccess = true;
    success = Tests::runBackendErrorTest(model->boardEvaluator(),
      fp32 ? fp32->boardEvaluator() : model->boardEvaluator(),logger,
      boardSizeDataset,-1,true,quickTest,optimism,pda,temperature,fp32BatchSuccess,referenceFileName);
  }
  else success = Tests::runModelBackendErrorTest(*model,fp32 ? *fp32 : *model,logger,
    boardSizeDataset,sgfs,quickTest,optimism,pda,temperature,referenceFileName,recurrentSteps);
  fp32.reset(); model.reset();
  NeuralNet::globalCleanup();
  return success ? 0 : 1;
}

int MainCmds::testbackendreference(const vector<string>& args) {
  Board::initHash();
  Rand seedRand;

  ConfigParser cfg;
  string modelFile;
  string referenceDataFileOverride;
  string dumpCandidateFileName;
  string limitsFileName;
  bool dumpOnly;
  double lenienceFactor;
  try {
    KataGoCommandLine cmd(
      "Test a backend's absolute neural net outputs against compiled-in reference data blended across nets from "
      "KataGo's main distributed run. Only nets from that run are expected to pass - human-imitation nets and other "
      "specially-trained nets can deviate far enough to fail regardless of the backend."
    );
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> referenceDataFileArg("", "reference-data-file", "Load reference data from this file (same JSON-lines schema as backendreferencedata.cpp) instead of the compiled-in data", false, "", "FILE");
    TCLAP::ValueArg<string> dumpCandidateArg("", "dump-candidate", "Dump this net's raw outputs on the reference positions to this file, for reference calibration", false, "", "FILE");
    TCLAP::ValueArg<double> lenienceFactorArg("", "lenience-factor", "Scale all check limits by this factor, 0.01 to 10000 (default 1.0)", false, 1.0, "FACTOR");
    TCLAP::ValueArg<string> limitsArg("", "reference-limits-file", "Explicit calibrated limits as JSON, without source model-size lenience/exemption; required for new games/MuZero/four-way results", false, "", "FILE");
    TCLAP::SwitchArg dumpOnlyArg("", "dump-only", "Export candidate outputs without running or claiming reference checks; requires dump-candidate");
    cmd.add(referenceDataFileArg);
    cmd.add(dumpCandidateArg);
    cmd.add(lenienceFactorArg);
    cmd.add(limitsArg); cmd.add(dumpOnlyArg);

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    referenceDataFileOverride = referenceDataFileArg.getValue();
    dumpCandidateFileName = dumpCandidateArg.getValue();
    lenienceFactor = lenienceFactorArg.getValue();
    limitsFileName = limitsArg.getValue(); dumpOnly = dumpOnlyArg.getValue();
    if(dumpOnly && dumpCandidateFileName.empty()) throw StringError("dump-only requires dump-candidate");
    if(!limitsFileName.empty() && referenceDataFileOverride.empty())
      throw StringError("Explicit limits require reference-data-file for the intended model family");
    if(!(lenienceFactor >= 0.01 && lenienceFactor <= 10000.0))
      throw StringError("Lenience factor must be in the range 0.01 to 10000");
    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const bool logToStdoutDefault = true;
  const bool logToStderrDefault = false;
  const bool logTimeDefault = false;
  Logger logger(NULL, logToStdoutDefault, logToStderrDefault, logTimeDefault);
  logger.write("Version " + Version::getGitRevisionWithBackend());
  logger.write("Testing " + modelFile);

  const string expectedSha256 = "";
  const int maxBatchSize = 16;
  const int expectedConcurrentEvals = maxBatchSize;
  const bool defaultRequireExactNNLen = false;

  std::unique_ptr<SelfplayModel> model;
  {
    logger.write("Initializing nneval using current config...");
    const bool disableFP16 = false;
    const string muKey = cfg.contains("useMuZero0") ? "useMuZero0" : "useMuZero";
    if(cfg.contains(muKey) && cfg.getBool(muKey)) {
      if(referenceDataFileOverride.empty()) throw StringError("MuZero backend reference needs an explicit reference-data-file");
      if(!dumpOnly && limitsFileName.empty()) throw StringError("MuZero backend reference needs an explicit reference-limits-file");
      auto bundle = std::make_shared<MuZero::ModelBundle>(modelFile);
      const auto game = Rules::parseGame(bundle->game());
      if((cfg.contains("game") && Rules::parseGame(cfg.getString("game")) != game) ||
         (cfg.contains("rules") && Rules::parseRules(cfg.getString("rules")).game != game))
        throw StringError("Backend reference config game differs from MuZero model");
      if(!cfg.contains("game")) cfg.overrideKey("game",Rules::gameName(game));
      model = Setup::initializeMuZeroSelfplayModel(modelFile,modelFile,std::move(bundle),cfg,logger,seedRand,
        expectedConcurrentEvals,defaultRequireExactNNLen,0,Setup::SETUP_FOR_BENCHMARK,
        Setup::MaxBatchSizeRequest::explicitSize(maxBatchSize));
    }
    else model = std::make_unique<SelfplayModel>(Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,Setup::MaxBatchSizeRequest::explicitSize(maxBatchSize),defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_BENCHMARK
    ));
  }
  if(!model->isNeuralNetEnabled()) throw StringError("Backend reference needs a real neural net model");

  //Match contribute.cpp's values so that reference data calibrated from this command's dumps
  //matches what contribute measures at runtime.
  const double policyOptimismForTest = 0.25;
  const double pdaForTest = 0.0;
  const double nnPolicyTemperatureForTest = 1.0;

  const bool verbose = true;
  bool success = Tests::runBackendReferenceTest(
    model->boardEvaluator(),logger,verbose,
    policyOptimismForTest,pdaForTest,nnPolicyTemperatureForTest,
    lenienceFactor,
    referenceDataFileOverride,dumpCandidateFileName,model.get(),limitsFileName,dumpOnly
  );

  model.reset();
  NeuralNet::globalCleanup();

  return success ? 0 : 1;
}
