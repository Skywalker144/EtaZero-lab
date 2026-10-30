#include "../tests/tests.h"

#include "../core/fileutils.h"
#include "../neuralnet/nneval.h"
#include "../dataio/sgf.h"
#include "../game/xiangqi.h"
#include "../program/selfplaymanager.h"

#include "../external/nlohmann_json/json.hpp"

//------------------------
#include "../core/using.h"
//------------------------
using json = nlohmann::json;

// Absolute-output check of a backend against compiled-in reference data blended across a
// sampling of neural nets from the training run (see backendreferencedata.cpp for the data
// schema). Unlike runBackendErrorTest, whose reference is by default the same backend's own
// unbatched fp32 outputs, this can catch a backend that is systematically wrong in fp16 and
// fp32 alike, so the two are complementary.
//
// The reference blend is what a net from the run looks like, so only nets from the run are
// expected to pass. Human-imitation nets and nets trained for other special purposes deviate
// from the run's own trajectory by more than a backend bug would, and can fail on their own
// merits. Nets too small for any threshold to be meaningful are exempted outright rather than
// checked, see MIN_EFF_PARAMS_TO_CHECK.
//
// The detection floor is the spread of legitimate nets across the run on each metric, so
// thresholds are necessarily far looser than runBackendErrorTest's.
// Metrics and thresholds must stay valid for every net the run still uses, as nets improve:
//  - Policy KL(candidate || smoothed mixture): improving nets sharpen within the mixture's
//    support, so this stays bounded, and it fires when the backend puts mass on moves no
//    reference net ever considered.
//  - Binary KL of the candidate's winrate vs the reference average winrate.
//  - Consensus top move: on positions where the mixture is confident, any reasonable net
//    assigns the consensus move substantial mass.
//
// Positions may individually override the policy optimism and pda used for their eval (the
// function arguments are just the defaults), and may cap the history planes via maxHistory.
// Backends implement the blend of the two policy output channels themselves, so varying
// optimism across positions exercises that blend.

struct BackendRefPosData {
  Sgf::PositionSample sample;
  Rules rules;
  double policyOptimism = 0.0;
  double pda = 0.0;
  int maxHistory = 1000;
  //Full averaged policy across reference nets: physical row-major xSize*ySize+1
  //with pass last (zero for non-Go), or the complete Xiangqi joint-action layout.
  std::vector<double> policyMixture;
  bool hasPolicyMixture = false;
  double winrateAvg = 0.0;
  bool hasWinrateAvg = false;
  double stWinlossErrorAvg = 0.0;
  bool hasStWinlossErrorAvg = false;
  std::vector<double> resultAvg; // White W/D/L/NR, when explicitly calibrated.
};

// Base limits, calibrated August 2026 against 31 nets spanning 2020-2026: kata1-b6c96 through
// b28c512nbt, b40c768nbt and experimental transformers, including the worst and earliest net
// of each size class. Each was evaluated on the compiled-in positions against the compiled-in
// reference blend. Limits are set so that with the size lenience below, every legitimate net
// measured at most ~2/3 of each limit.
// Before checking, continuous limits are scaled up by the overall lenience (the size lenience
// below times the caller's lenienceFactor) and consensusMinProb is scaled down by it.
struct BackendRefLimits {
  double mixtureSmoothing = 1e-5;    // epsilon of uniform-over-legal mixed into the mixture
  double avgPolicyKL = 0.24;         // mean over positions of KL(candidate || smoothed mixture), nats
  double p95PolicyKL = 0.60;         // 95th percentile over positions
  double consensusTopProb = 0.80;    // mixture top prob >= this marks a consensus position
  double consensusMinProb = 0.10;    // candidate must assign at least this to the consensus move
  int maxConsensusViolations = 3;
  double avgWinrateKL = 0.040;       // binary KL of candidate winrate vs reference average, nats
  double p95WinrateKL = 0.19;
  //calibration: they are canaries that don't drive play the way policy/winrate do, and the
  //second-order ones could shift with future training hyperparameter changes.
  double avgStWinlossErrorAE = 0.096; // abs error vs reference average shorttermWinlossError
  double p95StWinlossErrorAE = 0.33;
  double avgResultKL = 0.0; // No source calibration for the four-way result.
  double p95ResultKL = 0.0;
};

static BackendRefLimits loadBackendRefLimits(const string& file) {
  const json obj = json::parse(FileUtils::readFile(file));
  if(!obj.is_object()) throw StringError("Backend reference limits must be a JSON object");
  BackendRefLimits limits;
  const std::map<string,double*> fields = {
    {"mixtureSmoothing",&limits.mixtureSmoothing}, {"avgPolicyKL",&limits.avgPolicyKL},
    {"p95PolicyKL",&limits.p95PolicyKL}, {"consensusTopProb",&limits.consensusTopProb},
    {"consensusMinProb",&limits.consensusMinProb}, {"avgWinrateKL",&limits.avgWinrateKL},
    {"p95WinrateKL",&limits.p95WinrateKL}, {"avgStWinlossErrorAE",&limits.avgStWinlossErrorAE},
    {"p95StWinlossErrorAE",&limits.p95StWinlossErrorAE},
    {"avgResultKL",&limits.avgResultKL}, {"p95ResultKL",&limits.p95ResultKL}
  };
  for(auto it = obj.begin(); it != obj.end(); ++it)
    if(fields.count(it.key()) == 0 && it.key() != "maxConsensusViolations")
      throw StringError("Unknown backend reference limit: " + it.key());
  for(const auto& field : fields) {
    if(!obj.contains(field.first) || !obj.at(field.first).is_number())
      throw StringError("Missing numeric backend reference limit: " + field.first);
    *field.second = obj.at(field.first).get<double>();
    if(!std::isfinite(*field.second) || *field.second < 0)
      throw StringError("Invalid backend reference limit: " + field.first);
  }
  if(!obj.contains("maxConsensusViolations") || !obj.at("maxConsensusViolations").is_number_integer())
    throw StringError("Missing integer backend reference limit: maxConsensusViolations");
  const int64_t violations = obj.at("maxConsensusViolations").get<int64_t>();
  if(violations < 0 || violations > std::numeric_limits<int>::max())
    throw StringError("maxConsensusViolations is outside the integer count range");
  limits.maxConsensusViolations = (int)violations;
  if(limits.maxConsensusViolations < 0 || limits.mixtureSmoothing <= 0 || limits.mixtureSmoothing > 1 ||
     limits.consensusTopProb > 1 || limits.consensusMinProb > 1)
    throw StringError("Backend reference probability limits must be in [0,1], smoothing in (0,1], violations nonnegative");
  return limits;
}

// Smaller/weaker nets legitimately deviate more from the blended reference. Measured mean
// policy KL vs the blend rises smoothly as parameter count shrinks: ~1.4x at 47M params, ~5x
// at 10M, ~11x at 1M, relative to the largest nets. Nested bottleneck nets behave like larger
// plain nets per parameter, and transformers larger still, so their parameters are counted at
// a multiple. The exponent is deliberately a bit steeper than the measured scaling: small nets
// only play rating games rather than generating training data, so extra buffer against
// stochastic variation is cheap there, while nets at or above the current main-run size stay
// near lenience 1.
static double computeBackendRefEffParams(const NNEvaluator* nnEval) {
  double effParams = (double)nnEval->getNumModelParameters();
  if(nnEval->modelHasAnyNestedBottleneckBlocks())
    effParams *= 2.0;
  if(nnEval->modelHasAnyTransformerBlocks())
    effParams *= 2.0;
  return effParams;
}

static double computeBackendRefSizeLenience(const NNEvaluator* nnEval) {
  return std::max(1.0, pow(1.4e8 / std::max(1.0, computeBackendRefEffParams(nnEval)), 0.75));
}

//Size classes at or below b6c96 include the run's very first nets, barely better than random.
//No thresholds can both pass those and stay meaningful, so below this the checks are advisory.
static constexpr double MIN_EFF_PARAMS_TO_CHECK = 2e6;

static bool parseBackendRefScalar(const json& obj, const char* key, double& out) {
  if(obj.find(key) == obj.end() || obj[key].is_null())
    return false;
  out = obj[key].get<double>();
  return true;
}

static BackendRefPosData parseBackendRefPosData(const string& line, double defaultPolicyOptimism, double defaultPda) {
  BackendRefPosData data;
  json obj = json::parse(line);
  data.sample = Sgf::PositionSample::ofJsonLine(obj["sample"].dump());
  data.rules = Rules::parseRules(obj["rules"].get<string>());
  data.rules.komi = (float)obj["komi"].get<double>();
  if(data.sample.game != data.rules.game)
    throw StringError("Backend reference sample and rules game differ");
  data.policyOptimism = defaultPolicyOptimism;
  if(parseBackendRefScalar(obj,"policyOptimism",data.policyOptimism)) {
    if(!(data.policyOptimism >= 0.0 && data.policyOptimism <= 1.0))
      throw StringError("Backend reference data: policyOptimism out of range");
  }
  data.pda = defaultPda;
  if(parseBackendRefScalar(obj,"pda",data.pda)) {
    if(!(data.pda >= -4.0 && data.pda <= 4.0))
      throw StringError("Backend reference data: pda out of range");
  }
  if(obj.find("maxHistory") != obj.end() && !obj["maxHistory"].is_null()) {
    data.maxHistory = obj["maxHistory"].get<int>();
    if(data.maxHistory < 0 || data.maxHistory > 1000)
      throw StringError("Backend reference data: maxHistory out of range");
  }
  if(obj.find("policyMixture") != obj.end() && !obj["policyMixture"].is_null()) {
    data.policyMixture = obj["policyMixture"].get<std::vector<double>>();
    const size_t size = data.rules.isXiangqi() ? Xiangqi::NUM_ACTIONS :
      data.sample.board.x_size * data.sample.board.y_size + 1;
    if(data.policyMixture.size() != size)
      throw StringError("Backend reference data: policyMixture size does not match board size");
    for(const double& p: data.policyMixture) {
      if(!(p >= 0.0 && p <= 1.0 + 1e-6))
        throw StringError("Backend reference data: bad policyMixture prob");
    }
    data.hasPolicyMixture = true;
  }
  data.hasWinrateAvg = parseBackendRefScalar(obj,"winrateAvg",data.winrateAvg);
  data.hasStWinlossErrorAvg = parseBackendRefScalar(obj,"shorttermWinlossErrorAvg",data.stWinlossErrorAvg);
  if(obj.contains("resultAvg") && !obj.at("resultAvg").is_null()) {
    data.resultAvg = obj.at("resultAvg").get<std::vector<double>>();
    if(data.resultAvg.size() != 4) throw StringError("Backend reference resultAvg must contain white W/D/L/NR");
    double total = 0;
    for(double p : data.resultAvg) {
      if(!std::isfinite(p) || p < 0 || p > 1) throw StringError("Invalid backend reference resultAvg probability");
      total += p;
    }
    if(std::abs(total-1.0) > 1e-5) throw StringError("Backend reference resultAvg must sum to one");
  }

  return data;
}

static double backendRefAvg(const std::vector<double>& vec) {
  if(vec.size() <= 0)
    return 0.0;
  double sum = 0;
  for(const double& x: vec)
    sum += x;
  return sum / (double)vec.size();
}
static double backendRefPercentile(std::vector<double> vec, double frac) {
  if(vec.size() <= 0)
    return 0.0;
  std::sort(vec.begin(),vec.end());
  return vec[(size_t)((double)(vec.size()-1) * frac)];
}
static double backendRefMax(const std::vector<double>& vec) {
  if(vec.size() <= 0)
    return 0.0;
  return *std::max_element(vec.begin(),vec.end());
}
static void backendRefReportStats(const string& name, const std::vector<double>& vec, Logger& logger) {
  auto rpad = [](const string& s, size_t n) {
    if(s.size() < n)
      return s + std::string(n - s.size(),' ');
    return s;
  };
  logger.write(
    rpad("backend reference " + name + ":", 46) +
    Global::strprintf(
      " %8.5f  %8.5f  %8.5f  %8.5f",
      backendRefAvg(vec), backendRefPercentile(vec,0.95), backendRefPercentile(vec,0.99), backendRefMax(vec)
    )
  );
}

bool Tests::runBackendReferenceTest(
  NNEvaluator* nnEval,
  Logger& logger,
  bool verbose,
  double policyOptimismForTest,
  double pdaForTest,
  double nnPolicyTemperatureForTest,
  double lenienceFactor,
  const string& referenceDataFileOverride,
  const string& dumpCandidateFileName,
  SelfplayModel* model,
  const string& limitsFileName,
  bool dumpOnly
) {
  auto* muEval = model ? model->muZeroEvaluator() : nullptr;
  const bool sourceCalibration = limitsFileName.empty();
  const string modelName = model ? model->getInternalModelName() : nnEval->getInternalModelName();
  const int modelVersion = model ? model->getModelVersion() : nnEval->getModelVersion();
  const int maxBatchSize = model ? model->getMaxBatchSize() : nnEval->getMaxBatchSize();
  const bool supportsShorttermError = modelVersion >= 9;
  if(dumpOnly && dumpCandidateFileName.empty()) throw StringError("dump-only requires dump-candidate");
  if(muEval && referenceDataFileOverride.empty())
    throw StringError("MuZero backend reference needs an explicit reference-data-file");
  BackendRefLimits limits = sourceCalibration ? BackendRefLimits() : loadBackendRefLimits(limitsFileName);
  const double sizeLenience = !dumpOnly && sourceCalibration && nnEval ? computeBackendRefSizeLenience(nnEval) : 1.0;
  const double lenience = sizeLenience * lenienceFactor;
  const double consensusMinProbUsed = sourceCalibration ? std::max(0.005, limits.consensusMinProb / lenience) :
    limits.consensusMinProb / lenience;

  std::vector<string> jsonLines;
  if(referenceDataFileOverride != "") {
    for(const string& line: FileUtils::readFileLines(referenceDataFileOverride,'\n')) {
      if(Global::trim(line) != "")
        jsonLines.push_back(line);
    }
    if(verbose)
      logger.write("Loaded " + Global::uint64ToString((uint64_t)jsonLines.size()) + " backend reference positions from: " + referenceDataFileOverride);
  }
  else
    jsonLines = TestCommon::getBackendReferenceJsonData();

  std::vector<BackendRefPosData> refData;
  for(const string& line: jsonLines)
    refData.push_back(parseBackendRefPosData(line, policyOptimismForTest, pdaForTest));

  if(refData.size() <= 0) {
    throw StringError("Backend reference test: no reference positions");
  }
  bool hasReference = false;
  for(const auto& data : refData) {
    if(!dumpOnly && sourceCalibration && (muEval || !data.rules.isGo() || !data.resultAvg.empty()))
      throw StringError("This model/game/result distribution needs explicit reference-limits-file; source calibration is only for its Go training family");
    hasReference = hasReference || data.hasPolicyMixture || data.hasWinrateAvg || data.hasStWinlossErrorAvg || !data.resultAvg.empty();
    if(muEval && data.rules.game != Rules::parseGame(muEval->getModel().game()))
      throw StringError("Backend reference game differs from MuZero model");
  }
  if(!dumpOnly && !hasReference) throw StringError("No reference metrics supplied; use dump-only to collect outputs");

  std::ofstream dumpOut;
  if(dumpCandidateFileName != "") {
    FileUtils::open(dumpOut,dumpCandidateFileName);
    if(!dumpOut)
      throw StringError("Unable to open dump file: " + dumpCandidateFileName);
  }

  // Per-position aggregates
  std::vector<double> policyKL;
  std::vector<double> winrateKL;
  std::vector<double> stWinlossErrorAE;
  std::vector<double> resultKL;
  std::vector<double> consensusProbs;      // candidate prob on the consensus move, over consensus positions
  int numConsensusViolations = 0;

  //Featurize per the model's own declared BoardHistoryModes preferences. Nets with different modes
  //thus see slightly different inputs on the same position - part of the model-family spread the
  //calibrated thresholds must absorb.
  const BoardHistoryModes modelModes(
    muEval ? muEval->getModel().representationDesc().preferPassAliveUnderSuicideRules : nnEval->modelPreferPassAliveUnderSuicideRules(),
    muEval ? muEval->getModel().representationDesc().preferExcludeTerritoryAdjacentToAtari : nnEval->modelPreferExcludeTerritoryAdjacentToAtari());
  const SGFMetadata sgfMeta = SGFMetadata::getProfile("preaz_5k");

  auto makeHistAndParams = [&](size_t posIdx, BoardHistory& hist, MiscNNInputParams& nnInputParams) {
    const BackendRefPosData& data = refData[posIdx];
    Player nextPla = C_EMPTY;
    bool histOkay = data.sample.tryGetCurrentBoardHistory(data.rules, nextPla, hist, modelModes);
    if(!histOkay)
      throw StringError("Backend reference test: reference position " + Global::uint64ToString((uint64_t)posIdx) + " has illegal moves");
    if(!hist.rules.isGo() && hist.isGameFinished)
      throw StringError("Backend reference test requires an ongoing position with a policy");
    const Board& board = hist.getRecentBoard(0);
    std::vector<int> symmetries;
    for(int s = 0; s < SymmetryHelpers::NUM_SYMMETRIES; s++)
      if(hist.rules.allowsSymmetry(s)) symmetries.push_back(s);
    nnInputParams.symmetry = symmetries[BoardHistory::getSituationRulesAndKoHash(board,hist,hist.presumedNextMovePla,0.5).hash0 % symmetries.size()];
    nnInputParams.policyOptimism = data.policyOptimism;
    nnInputParams.playoutDoublingAdvantage = data.pda;
    nnInputParams.maxHistory = data.maxHistory;
    nnInputParams.nnPolicyTemperature = (float)nnPolicyTemperatureForTest;
    nnInputParams.passAliveSuicideRulesOverride = modelModes.alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
    nnInputParams.excludeTerritoryAdjAtariOverride = modelModes.excludeTerritoryAdjacentToAtari ? 1 : 0;
  };

  auto evalPosition = [&](size_t posIdx) {
    BoardHistory hist;
    MiscNNInputParams nnInputParams;
    makeHistAndParams(posIdx, hist, nnInputParams);
    const Board& board = hist.getRecentBoard(0);
    NNResultBuf buf;
    const bool skipCache = true;
    if(muEval) {
      MuZero::PositionBuffer buffer;
      Rand rand("backend reference initial");
      const auto output = muEval->initial(board,hist,hist.presumedNextMovePla,&sgfMeta,nnInputParams,buffer,rand,skipCache);
      return std::make_shared<NNOutput>(*output);
    }
    nnEval->evaluate(board,hist,hist.presumedNextMovePla,&sgfMeta,nnInputParams,buf,skipCache);
    return std::move(buf.result);
  };

  //Evaluate the first half of the positions sequentially, one query in flight at a time, and
  //the second half from concurrent threads, so that the outputs checked cover both unbatched
  //evaluation and batched evaluation, with multiple NN server threads busy at once if the
  //evaluator is configured with more than one.
  std::vector<std::shared_ptr<NNOutput>> outputs(refData.size());
  const size_t numSequential = (refData.size()+1) / 2;
  for(size_t posIdx = 0; posIdx < numSequential; posIdx++)
    outputs[posIdx] = evalPosition(posIdx);
  if(numSequential < refData.size()) {
    int numEvalThreads = std::max(2, std::min(7, 1 + maxBatchSize/2));
    if((size_t)numEvalThreads > refData.size() - numSequential)
      numEvalThreads = (int)(refData.size() - numSequential);
    //Exceptions must not escape the worker threads, so they are collected and rethrown here.
    std::vector<std::exception_ptr> threadErrors(numEvalThreads);
    std::vector<std::thread> threads;
    for(int threadIdx = 0; threadIdx < numEvalThreads; threadIdx++) {
      threads.emplace_back([&,threadIdx]() {
        try {
          for(size_t posIdx = numSequential + threadIdx; posIdx < refData.size(); posIdx += numEvalThreads)
            outputs[posIdx] = evalPosition(posIdx);
        }
        catch(...) {
          threadErrors[threadIdx] = std::current_exception();
        }
      });
    }
    for(std::thread& thread: threads)
      thread.join();
    for(const std::exception_ptr& err: threadErrors) {
      if(err != nullptr)
        std::rethrow_exception(err);
    }
  }

  for(size_t posIdx = 0; posIdx < refData.size(); posIdx++) {
    const BackendRefPosData& data = refData[posIdx];

    BoardHistory hist;
    MiscNNInputParams nnInputParams;
    makeHistAndParams(posIdx, hist, nnInputParams);
    const Board& board = hist.getRecentBoard(0);
    const std::shared_ptr<NNOutput>& out = outputs[posIdx];

    int numLegal = 0;
    for(int i = 0; i<out->getPolicyStorageSize(); i++) {
      if(out->policyProbs[i] >= 0)
        numLegal += 1;
    }
    testAssert(numLegal > 0);

    const int mixtureLen = hist.rules.isXiangqi() ? Xiangqi::NUM_ACTIONS : board.x_size * board.y_size + 1;
    auto mixtureIdxOfPos = [&](int pos) {
      if(hist.rules.isXiangqi()) return pos;
      Loc loc = NNPos::posToLoc(pos,board.x_size,board.y_size,out->nnXLen,out->nnYLen);
      if(loc == Board::PASS_LOC)
        return mixtureLen-1;
      return Location::getY(loc,board.x_size) * board.x_size + Location::getX(loc,board.x_size);
    };

    if(!dumpOnly && data.hasPolicyMixture) {
      const double eps = limits.mixtureSmoothing;
      double klSum = 0.0;
      for(int i = 0; i<out->getPolicyStorageSize(); i++) {
        double c = out->policyProbs[i];
        if(c <= 1e-30)
          continue;
        double m = data.policyMixture[mixtureIdxOfPos(i)];
        m = (1.0-eps)*m + eps/numLegal;
        klSum += c * (log(c) - log(m));
      }
      policyKL.push_back(klSum);

      int topIdx = 0;
      for(int i = 1; i<mixtureLen; i++) {
        if(data.policyMixture[i] > data.policyMixture[topIdx])
          topIdx = i;
      }
      if(data.policyMixture[topIdx] >= limits.consensusTopProb) {
        Loc topLoc = (topIdx == mixtureLen-1) ? Board::PASS_LOC : Location::getLoc(topIdx % board.x_size, topIdx / board.x_size, board.x_size);
        int candPos = hist.rules.isXiangqi() ? topIdx : out->getPos(topLoc,board);
        double candidateProb = out->policyProbs[candPos];
        if(candidateProb < 0)
          throw StringError("Backend reference test: consensus move is illegal on reference position " + Global::uint64ToString((uint64_t)posIdx));
        consensusProbs.push_back(candidateProb);
        if(candidateProb < consensusMinProbUsed)
          numConsensusViolations += 1;
      }
    }

    double winrate = 0.5*(1.0 + out->whiteWinProb - out->whiteLossProb);
    if(!dumpOnly && data.hasWinrateAvg) {
      auto clamp01 = [](double x) { return std::min(1.0-1e-6, std::max(1e-6, x)); };
      double r = clamp01(data.winrateAvg);
      double c = clamp01(winrate);
      winrateKL.push_back(r*log(r/c) + (1.0-r)*log((1.0-r)/(1.0-c)));
    }
    //Model versions below 9 lack these heads entirely (the outputs would be softplus-of-zero
    //garbage rather than real predictions), so skip the checks for such models.
    if(!dumpOnly && data.hasStWinlossErrorAvg && supportsShorttermError && out->shorttermWinlossError >= 0)
      stWinlossErrorAE.push_back(std::abs((double)out->shorttermWinlossError - data.stWinlossErrorAvg));



    if(!dumpOnly && !data.resultAvg.empty()) {
      const double probabilities[] = {out->whiteWinProb,out->whiteDrawProb,out->whiteLossProb,out->whiteNoResultProb};
      double kl = 0;
      for(int k = 0; k < 4; k++) {
        const double r = data.resultAvg[k];
        if(r > 0) kl += r * log(r / std::max(1e-6,probabilities[k]));
      }
      resultKL.push_back(kl);
    }

    if(dumpCandidateFileName != "") {
      json dump;
      dump["posIdx"] = (int64_t)posIdx;
      dump["sample"] = json::parse(Sgf::PositionSample::toJsonLine(data.sample));
      dump["rules"] = data.rules.toStringNoKomi();
      dump["komi"] = data.rules.komi;
      //Effective values used, including ones that came from the defaults.
      dump["policyOptimism"] = data.policyOptimism;
      dump["pda"] = data.pda;
      dump["maxHistory"] = data.maxHistory;
      dump["symmetryUsed"] = nnInputParams.symmetry;
      dump["modelName"] = modelName;
      dump["modelVersion"] = modelVersion;
      //Full policy in the same physical-point or joint-action layout as policyMixture.
      std::vector<double> policy(mixtureLen, 0.0);
      for(int i = 0; i<out->getPolicyStorageSize(); i++) {
        double c = out->policyProbs[i];
        if(c < 0)
          continue;
        policy[mixtureIdxOfPos(i)] = c;
      }
      dump["policy"] = policy;
      dump["whiteWinProb"] = out->whiteWinProb;
      dump["whiteLossProb"] = out->whiteLossProb;
      dump["whiteDrawProb"] = out->whiteDrawProb;
      dump["whiteNoResultProb"] = out->whiteNoResultProb;
      dump["winrate"] = winrate;
      //Null for models that lack these heads, so blending excludes them.
      if(supportsShorttermError) {
        dump["shorttermWinlossError"] = out->shorttermWinlossError;
      }
      else {
        dump["shorttermWinlossError"] = nullptr;
      }
      dumpOut << dump.dump() << "\n";
    }
  }

  if(dumpCandidateFileName != "") {
    dumpOut.close();
    logger.write("Dumped candidate outputs for " + Global::uint64ToString((uint64_t)refData.size()) + " positions to: " + dumpCandidateFileName);
  }

  if(dumpOnly) {
    logger.write("Backend reference outputs exported; reference checks were not run");
    return true;
  }

  if(verbose) {
    logger.write(
      "Backend reference test: " + Global::uint64ToString((uint64_t)refData.size()) + " positions ("
      + Global::uint64ToString((uint64_t)policyKL.size()) + " policy mixture, "
      + Global::uint64ToString((uint64_t)consensusProbs.size()) + " consensus, "
      + Global::uint64ToString((uint64_t)winrateKL.size()) + " winrate, "
      + Global::uint64ToString((uint64_t)stWinlossErrorAE.size()) + " shortterm, "
      + Global::uint64ToString((uint64_t)resultKL.size()) + " four-result) on model "
      + modelName);
    logger.write(Global::strprintf(
      "Lenience: %.3f (size lenience %.3f x factor %.3f)",
      lenience, sizeLenience, lenienceFactor));
    logger.write("Reporting avg, 95%, 99%, max over positions:");
    backendRefReportStats("policyKL (nats)", policyKL, logger);
    backendRefReportStats("winrateKL (nats)", winrateKL, logger);
    backendRefReportStats("stWinlossErrorAE", stWinlossErrorAE, logger);
    if(!resultKL.empty()) backendRefReportStats("resultKL (nats)",resultKL,logger);
    if(consensusProbs.size() > 0)
      logger.write(Global::strprintf(
        "backend reference min consensus move prob: %8.5f  (%d of %d below %.5f, %d allowed)",
        *std::min_element(consensusProbs.begin(),consensusProbs.end()),
        numConsensusViolations, (int)consensusProbs.size(), consensusMinProbUsed, limits.maxConsensusViolations));
  }

  bool success = true;
  auto failCheck = [&](const string& msg) {
    //Always log failures even when not verbose, so that contribute logs record which check tripped.
    logger.write("Backend reference test failed check for " + modelName + ": " + msg);
    success = false;
  };
  auto checkLimit = [&](double value, double limit, const char* name) {
    if(!(value <= limit))
      failCheck(Global::strprintf("%s %.5f > limit %.5f", name, value, limit));
  };
  auto checkViolations = [&](int count, int maxAllowed, const char* name) {
    if(count > maxAllowed)
      failCheck(Global::strprintf("%s %d > allowed %d", name, count, maxAllowed));
  };

  auto p95 = [&](const std::vector<double>& v) { return backendRefPercentile(v, 0.95); };
  checkLimit(backendRefAvg(policyKL), limits.avgPolicyKL * lenience, "avg policyKL");
  checkLimit(p95(policyKL), limits.p95PolicyKL * lenience, "p95 policyKL");
  checkViolations(numConsensusViolations, limits.maxConsensusViolations, "consensus move violations");
  checkLimit(backendRefAvg(winrateKL), limits.avgWinrateKL * lenience, "avg winrateKL");
  checkLimit(p95(winrateKL), limits.p95WinrateKL * lenience, "p95 winrateKL");
  checkLimit(backendRefAvg(stWinlossErrorAE), limits.avgStWinlossErrorAE * lenience, "stWinlossError MAE");
  checkLimit(p95(stWinlossErrorAE), limits.p95StWinlossErrorAE * lenience, "p95 stWinlossErrorAE");

  if(!resultKL.empty()) {
    checkLimit(backendRefAvg(resultKL),limits.avgResultKL * lenience,"avg resultKL");
    checkLimit(p95(resultKL),limits.p95ResultKL * lenience,"p95 resultKL");
  }

  bool failuresExempted = false;
  if(sourceCalibration && !success && computeBackendRefEffParams(nnEval) < MIN_EFF_PARAMS_TO_CHECK) {
    logger.write("Backend reference test: failed checks ignored for " + modelName + ", model is below the minimum size for checking");
    success = true;
    failuresExempted = true;
  }

  if(verbose) {
    if(failuresExempted)
      logger.write("Backend reference test PASSED (checks not applicable at this model size)");
    else
      logger.write(string("Backend reference test ") + (success ? "PASSED" : "FAILED"));
  }
  return success;
}
