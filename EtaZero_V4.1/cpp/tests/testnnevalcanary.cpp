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

void Tests::runCanaryTests(NNEvaluator* nnEval, int symmetry, bool print) {
  // Be lenient on smaller models or special runs
  double policyLenience = 0;
  double winrateLenience = 0;

  const string& internalModelName = nnEval->getInternalModelName();
  if(print) {
    cout << "nnEval->getTrunkSpatialConvDepth() " << nnEval->getTrunkSpatialConvDepth() << endl;
    cout << "internalModelName " << internalModelName << endl;
  }

  if(nnEval->getTrunkSpatialConvDepth() <= 21) {
    policyLenience = 0.05;
    winrateLenience = 0.12;
  }
  else if(nnEval->getTrunkSpatialConvDepth() <= 31 ||
          Global::isPrefix(internalModelName,"rect15") ||
          Global::isPrefix(internalModelName,"special")
  ) {
    policyLenience = 0.15;
    winrateLenience = 0.05;
  }

  {
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[pd];W[pp];B[dd];W[dp];B[qn];W[nq];B[cq];W[dq];B[cp];W[do];B[bn];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[fc];B[ed];W[gb];B[db];W[fa];B[cb];W[qo];B[pn];W[nc];B[qj];W[qc];B[qd];W[pc];B[od];W[nd];B[ne];W[me];B[mf];W[nf])";
    std::unique_ptr<CompactSgf> sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 18;
    //Featurize per the model's own declared BoardHistoryModes preferences.
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx, BoardHistoryModes(nnEval->modelPreferPassAliveUnderSuicideRules(), nnEval->modelPreferExcludeTerritoryAdjacentToAtari()));

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("E16",board),board)] >= 0.95);
    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("Pass",board),board)] <= 0.005);
    testAssert(buf.result->whiteWinProb > 0.30 - winrateLenience);
    testAssert(buf.result->whiteWinProb < 0.70 + winrateLenience);
    testAssert(buf.result->whiteNoResultProb < 0.03);
  }

  {
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[pd];W[pp];B[dd];W[dp];B[qn];W[nq];B[cq];W[dq];B[cp];W[do];B[bn];W[cc];B[cd];W[dc];B[ec];W[eb];B[fb];W[fc];B[ed];W[gb];B[db];W[fa];B[cb];W[qo];B[pn];W[nc];B[qj];W[qc];B[qd];W[pc];B[od];W[nd];B[ne];W[me];B[mf];W[nf])";
    std::unique_ptr<CompactSgf> sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 36;
    //Featurize per the model's own declared BoardHistoryModes preferences.
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx, BoardHistoryModes(nnEval->modelPreferPassAliveUnderSuicideRules(), nnEval->modelPreferExcludeTerritoryAdjacentToAtari()));

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("P15",board),board)] >= 0.85 - policyLenience);
    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("Pass",board),board)] <= 0.005);
    testAssert(buf.result->whiteWinProb > 0.30 - winrateLenience);
    testAssert(buf.result->whiteWinProb < 0.70 + winrateLenience);
    testAssert(buf.result->whiteNoResultProb < 0.03);
  }
  {
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[qd];W[dd];B[pp];W[dp];B[cf];W[fc];B[nd];W[nq];B[cq];W[dq];B[cp];W[cn];B[co];W[do];B[bn];W[cm];B[bm];W[cl];B[qn];W[pq];B[qq];W[qr];B[oq])";
    std::unique_ptr<CompactSgf> sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 23;
    //Featurize per the model's own declared BoardHistoryModes preferences.
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx, BoardHistoryModes(nnEval->modelPreferPassAliveUnderSuicideRules(), nnEval->modelPreferExcludeTerritoryAdjacentToAtari()));

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("Q2",board),board)] >= 0.95);
    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("Pass",board),board)] <= 0.005);
    testAssert(buf.result->whiteWinProb > 0.30 - winrateLenience);
    testAssert(buf.result->whiteWinProb < 0.70 + winrateLenience);
    testAssert(buf.result->whiteNoResultProb < 0.03);
  }

  {
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[qd];W[dd];B[pp];W[dp];B[cf];W[fc];B[nd];W[nq];B[cq];W[dq];B[cp];W[cn];B[co];W[do];B[bn];W[cm];B[bm];W[cl];B[qn];W[pq];B[qq];W[qr];B[oq])";
    std::unique_ptr<CompactSgf> sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 23;
    //Featurize per the model's own declared BoardHistoryModes preferences.
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx, BoardHistoryModes(nnEval->modelPreferPassAliveUnderSuicideRules(), nnEval->modelPreferExcludeTerritoryAdjacentToAtari()));
    hist.setKomi(-7);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->whiteWinProb < 0.1 + winrateLenience);
  }

  {
    string sgfStr = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[19]KM[7]PW[White]PB[Black];B[qd];W[dd];B[pp];W[dp];B[cf];W[fc];B[nd];W[nq];B[cq];W[dq];B[cp];W[cn];B[co];W[do];B[bn];W[cm];B[bm];W[cl];B[qn];W[pq];B[qq];W[qr];B[oq])";
    std::unique_ptr<CompactSgf> sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 23;
    //Featurize per the model's own declared BoardHistoryModes preferences.
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx, BoardHistoryModes(nnEval->modelPreferPassAliveUnderSuicideRules(), nnEval->modelPreferExcludeTerritoryAdjacentToAtari()));
    hist.setKomi(21);

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->whiteWinProb > 0.9 - winrateLenience);
  }

  // 16x11 rectangular board. Skip when the evaluator requires the exact (max) NN length, since it is
  // locked to the configured max board size and would throw on a smaller board. This lets the canary
  // be run under requireMaxBoardSize=true to exercise exact-NNLen-only behavior on the 19x19 positions.
  if(!nnEval->getRequireExactNNLen()) {
    string sgfStr = "(;FF[4]GM[1]CA[UTF-8]RU[Japanese]KM[6]SZ[16:11];B[md];W[nh];B[dh];W[cd];B[lh];W[li];B[ki])";
    std::unique_ptr<CompactSgf> sgf = CompactSgf::parse(sgfStr);

    Board board;
    Player nextPla;
    BoardHistory hist;
    Rules initialRules = sgf->getRulesOrFail();
    int turnIdx = 7;
    //Featurize per the model's own declared BoardHistoryModes preferences.
    sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx, BoardHistoryModes(nnEval->modelPreferPassAliveUnderSuicideRules(), nnEval->modelPreferExcludeTerritoryAdjacentToAtari()));

    MiscNNInputParams nnInputParams;
    NNResultBuf buf;
    bool skipCache = true;
    nnInputParams.symmetry = symmetry;
    nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

    if(print) {
      cout << board << endl;
      cout << endl;
      buf.result->debugPrint(cout,board);
    }

    testAssert(buf.result->policyProbs[buf.result->getPos(Location::ofString("N3",board),board)] >= 0.80 - policyLenience);
    testAssert(buf.result->whiteWinProb > 0.20 - winrateLenience);
    testAssert(buf.result->whiteWinProb < 0.80 + winrateLenience);
    testAssert(buf.result->whiteNoResultProb < 0.06);
  }
}

struct GpuErrorStats {
  std::vector<double> winrateError;
  std::vector<double> drawError;
  std::vector<double> topPolicyDiff;
  std::vector<double> policyKLDiv;
  std::vector<double> shorttermWinlossErrorError;
  void appendStats(const std::shared_ptr<NNOutput>& base, const std::shared_ptr<NNOutput>& other) {
    if(base->nnXLen != other->nnXLen || base->nnYLen != other->nnYLen ||
       base->getPolicyStorageSize() != other->getPolicyStorageSize())
      throw StringError("GPU error test output layouts differ");
    for(const NNOutput* output : {base.get(),other.get()}) {
      for(float value : {output->whiteWinProb,output->whiteDrawProb,output->whiteLossProb,
                         output->whiteNoResultProb,output->shorttermWinlossError})
        if(!std::isfinite(value)) throw StringError("Nonfinite value in GPU error test output");
      for(int i = 0; i < output->getPolicyStorageSize(); i++)
        if(!std::isfinite(output->policyProbs[i])) throw StringError("Nonfinite policy in GPU error test output");
    }
    drawError.push_back(std::abs(base->whiteDrawProb - other->whiteDrawProb));
    winrateError.push_back(
      std::abs(0.5*(base->whiteWinProb - base->whiteLossProb) - 0.5*(other->whiteWinProb - other->whiteLossProb))
      + std::abs(base->whiteNoResultProb - other->whiteNoResultProb)
    );
    int topPolicyIdx = 0;
    double topPolicyProb = -1;
    for(int i = 0; i<base->getPolicyStorageSize(); i++) {
      if(base->policyProbs[i] > topPolicyProb) {
        topPolicyIdx = i;
        topPolicyProb = base->policyProbs[i];
      }
    }
    topPolicyDiff.push_back(std::abs(topPolicyProb - other->policyProbs[topPolicyIdx]));

    double klDivSum = 0;
    for(int i = 0; i<base->getPolicyStorageSize(); i++) {
      if((base->policyProbs[i] < 0) != (other->policyProbs[i] < 0))
        throw StringError("GPU error test policy masks differ");
      if(base->policyProbs[i] > 1e-30) {
        klDivSum += base->policyProbs[i] * (log(base->policyProbs[i]) - log(other->policyProbs[i]));
      }
    }
    policyKLDiv.push_back(klDivSum);

    //A metric indicating the "typical" error in the winloss value or the score that the net expects, relative to the
    //short-term future MCTS value.

    shorttermWinlossErrorError.push_back(std::abs(base->shorttermWinlossError - other->shorttermWinlossError));


  }

  double getAverage(const std::vector<double>& vec) {
    double sum = 0;
    for(const double& x: vec)
      sum += x;
    return sum / vec.size();
  }

  double get90Percentile(const std::vector<double>& sortedVec) {
    return sortedVec[(sortedVec.size()-1) * 9 / 10];
  }

  double get99Percentile(const std::vector<double>& sortedVec) {
    return sortedVec[(sortedVec.size()-1) * 99 / 100];
  }
  double getMaxPercentile(const std::vector<double>& sortedVec) {
    return sortedVec[sortedVec.size()-1];
  }

  void sortErrors() {
    std::sort(winrateError.begin(),winrateError.end());
    std::sort(drawError.begin(),drawError.end());
    std::sort(topPolicyDiff.begin(),topPolicyDiff.end());
    std::sort(policyKLDiv.begin(),policyKLDiv.end());
    std::sort(shorttermWinlossErrorError.begin(),shorttermWinlossErrorError.end());
  }

  bool checkStats99(double wr, double tpd, double pkld) {
    sortErrors();
    return (
      100*get99Percentile(winrateError) <= wr &&
      100*get99Percentile(drawError) <= wr &&
      100*get99Percentile(topPolicyDiff) <= tpd &&
      get99Percentile(policyKLDiv) <= pkld &&
      100*get99Percentile(shorttermWinlossErrorError) <= wr*1.8
    );
  }

  bool checkStatsMax(double wr, double tpd, double pkld) {
    sortErrors();
    return (
      100*getMaxPercentile(winrateError) <= wr &&
      100*getMaxPercentile(drawError) <= wr &&
      100*getMaxPercentile(topPolicyDiff) <= tpd &&
      getMaxPercentile(policyKLDiv) <= pkld &&
      100*getMaxPercentile(shorttermWinlossErrorError) <= wr*1.8
    );
  }

  // The largest ratio of measured error to its allowed limit across all metrics, for the same
  // limits and scalings as checkStats99 (useMax=false) or checkStatsMax (useMax=true).
  // A ratio > 1 means that metric failed its check; the largest ratio is the "closest" margin.
  double worstRatioVsLimits(bool useMax, double wr, double tpd, double pkld, std::string& whichMetricOut) {
    sortErrors();
    auto pct = [&](const std::vector<double>& v) { return useMax ? getMaxPercentile(v) : get99Percentile(v); };
    const std::pair<const char*, double> ratios[] = {
      {"winrateError", 100*pct(winrateError) / wr},
      {"drawError", 100*pct(drawError) / wr},
      {"topPolicyDelta", 100*pct(topPolicyDiff) / tpd},
      {"policyKLDiv", pct(policyKLDiv) / pkld},
      {"stWLErrorError", 100*pct(shorttermWinlossErrorError) / (wr*1.8)},
    };
    double worst = -1;
    whichMetricOut = "";
    for(const auto& r: ratios) {
      if(r.second > worst) {
        worst = r.second;
        whichMetricOut = r.first;
      }
    }
    return worst;
  }

  void reportClosestMargin(
    const string& name, Logger& logger,
    double wr99, double tpd99, double pkld99,
    double wrMax, double tpdMax, double pkldMax
  ) {
    auto rpad = [](const string& s, int n) {
      if(s.size() < n)
        return s + std::string(n - s.size(),' ');
      return s;
    };
    std::string which99, whichMax;
    double r99 = worstRatioVsLimits(false, wr99, tpd99, pkld99, which99);
    double rMax = worstRatioVsLimits(true, wrMax, tpdMax, pkldMax, whichMax);
    logger.write(
      rpad(name + " closest margin: ", 60) +
      Global::strprintf(
        " %.3gx of limit (%s 99%%), %.3gx of limit (%s max)",
        r99, which99.c_str(), rMax, whichMax.c_str())
    );
  }


  void reportStats(const string& name, Logger& logger) {
    sortErrors();
    auto rpad = [](const string& s, int n) {
      if(s.size() < n)
        return s + std::string(n - s.size(),' ');
      return s;
    };

    logger.write(
      rpad(name + " winrateError:   ", 60) +
      Global::strprintf(
        " %7.5f%%  %7.5f%%  %7.5f%%  %7.5f%%",
        100*getAverage(winrateError), 100*get90Percentile(winrateError), 100*get99Percentile(winrateError), 100*getMaxPercentile(winrateError)
      )
    );



    logger.write(
      rpad(name + " topPolicyDelta: ", 60) +
      Global::strprintf(
        " %7.5f%%  %7.5f%%  %7.5f%%  %7.5f%%",
        100*getAverage(topPolicyDiff), 100*get90Percentile(topPolicyDiff), 100*get99Percentile(topPolicyDiff), 100*getMaxPercentile(topPolicyDiff))
    );
    logger.write(rpad(name + " drawError: ", 60) + Global::strprintf(
      " %7.5f%%  %7.5f%%  %7.5f%%  %7.5f%%", 100*getAverage(drawError),
      100*get90Percentile(drawError),100*get99Percentile(drawError),100*getMaxPercentile(drawError)));
    logger.write(
      rpad(name + " policyKLDiv:    ", 60) +
      Global::strprintf(
        " %8.6f  %8.6f  %8.6f  %8.6f",
        getAverage(policyKLDiv), get90Percentile(policyKLDiv), get99Percentile(policyKLDiv), getMaxPercentile(policyKLDiv))
    );
    logger.write(
      rpad(name + " stWLErrorError:", 60) +
      Global::strprintf(
        " %7.5fc  %7.5fc  %7.5fc  %7.5fc",
        100*getAverage(shorttermWinlossErrorError), 100*get90Percentile(shorttermWinlossErrorError), 100*get99Percentile(shorttermWinlossErrorError), 100*getMaxPercentile(shorttermWinlossErrorError))
    );


  }
};

static std::string nnOutputToJson(const std::shared_ptr<NNOutput>& nnOutput) {
  json ret;
  ret["nnHash"] = nnOutput->nnHash.toString();
  ret["whiteWinProb"] = nnOutput->whiteWinProb;
  ret["whiteLossProb"] = nnOutput->whiteLossProb;
  ret["whiteDrawProb"] = nnOutput->whiteDrawProb;
  ret["whiteNoResultProb"] = nnOutput->whiteNoResultProb;
  ret["varTimeLeft"] = nnOutput->varTimeLeft;
  ret["shorttermWinlossError"] = nnOutput->shorttermWinlossError;
  ret["policyProbs"] = std::vector<float>(nnOutput->policyProbs, nnOutput->policyProbs + nnOutput->getPolicyStorageSize());
  ret["policyOptimismUsed"] = nnOutput->policyOptimismUsed;
  ret["nnXLen"] = nnOutput->nnXLen;
  ret["nnYLen"] = nnOutput->nnYLen;
  return std::string(ret.dump());
}

static std::shared_ptr<NNOutput> nnOutputOfJson(const std::string& s) {
  std::shared_ptr<NNOutput> nnOutput = std::make_shared<NNOutput>();
  json input = json::parse(s);
  nnOutput->nnHash = Hash128::ofString(input["nnHash"].get<string>());
  nnOutput->whiteWinProb = input["whiteWinProb"].get<float>();
  nnOutput->whiteLossProb = input["whiteLossProb"].get<float>();
  nnOutput->whiteDrawProb = input.at("whiteDrawProb").get<float>();
  nnOutput->whiteNoResultProb = input["whiteNoResultProb"].get<float>();
  nnOutput->varTimeLeft = input["varTimeLeft"].get<float>();
  nnOutput->shorttermWinlossError = input["shorttermWinlossError"].get<float>();
  std::vector<float> policyProbs = input["policyProbs"].get<std::vector<float>>();
  if(policyProbs.size() != NNPos::MAX_NN_POLICY_SIZE && policyProbs.size() != Xiangqi::NUM_ACTIONS)
    throw StringError("Invalid policy size in GPU error reference");
  nnOutput->resizePolicy((int)policyProbs.size());
  std::copy(policyProbs.begin(),policyProbs.end(),nnOutput->policyProbs);
  nnOutput->policyOptimismUsed = input["policyOptimismUsed"].get<float>();
  nnOutput->nnXLen = input["nnXLen"].get<int>();
  nnOutput->nnYLen = input["nnYLen"].get<int>();
  testAssert(nnOutput->nnXLen >= 2 && nnOutput->nnXLen <= NNPos::MAX_BOARD_LEN);
  testAssert(nnOutput->nnYLen >= 2 && nnOutput->nnYLen <= NNPos::MAX_BOARD_LEN);
  nnOutput->noisedPolicyProbs = nullptr;
  return nnOutput;
}

static void saveReferenceValuesToFile(const std::vector<std::shared_ptr<NNOutput>>& referenceValues, const string& referenceFileName, Logger& logger, bool verbose) {
  testAssert(referenceFileName != "");
  std::ofstream outFile;
  FileUtils::open(outFile,referenceFileName);
  if(!outFile)
    throw StringError("Unable to save reference values to: " + referenceFileName);

  for(const std::shared_ptr<NNOutput>& nnOutput : referenceValues) {
    testAssert(nnOutput != nullptr);
    outFile << nnOutputToJson(nnOutput) << "\n";
  }
  if(verbose)
    logger.write("Saved reference values for " + Global::uint64ToString((uint64_t)referenceValues.size()) + " positions to: " + referenceFileName);

  outFile.close();
}

static void loadReferenceValuesFromFile(std::vector<std::shared_ptr<NNOutput>>& referenceValues, const string& referenceFileName, Logger& logger, bool verbose) {
  testAssert(referenceFileName != "");
  referenceValues.clear();
  std::vector<std::string> lines = FileUtils::readFileLines(referenceFileName,'\n');

  for(const string& line: lines) {
    if(Global::trim(line) != "") {
      referenceValues.push_back(nnOutputOfJson(line));
    }
  }
  if(verbose)
    logger.write("Loaded reference values for " + Global::uint64ToString((uint64_t)referenceValues.size()) + " positions from: " + referenceFileName);
}

void Tests::runBackendErrorStatsTests() {
  auto base = std::make_shared<NNOutput>();
  base->nnXLen = 9; base->nnYLen = 10;
  base->whiteWinProb = 0.3f; base->whiteLossProb = 0.3f;
  base->whiteDrawProb = 0.3f; base->whiteNoResultProb = 0.1f;
  base->varTimeLeft = 10.0f; base->shorttermWinlossError = 0.2f;
  base->policyOptimismUsed = 0.25f;
  base->resizePolicy(Xiangqi::NUM_ACTIONS);
  std::fill_n(base->policyProbs,base->getPolicyStorageSize(),-1.0f);
  base->policyProbs[0] = 0.4f;
  base->policyProbs[2000] = 0.2f;
  base->policyProbs[2085] = 0.4f;
  auto roundtrip = nnOutputOfJson(nnOutputToJson(base));
  testAssert(roundtrip->whiteDrawProb == base->whiteDrawProb);
  testAssert(roundtrip->getPolicyStorageSize() == base->getPolicyStorageSize());
  testAssert(roundtrip->policyProbs[2085] == 0.4f);
  GpuErrorStats identical;
  identical.appendStats(base,roundtrip);
  testAssert(identical.checkStats99(0.45,0.45,0.0006));
  auto shifted = std::make_shared<NNOutput>(*base);
  shifted->whiteWinProb -= 0.1f; shifted->whiteLossProb -= 0.1f; shifted->whiteDrawProb += 0.2f;
  GpuErrorStats draw;
  draw.appendStats(base,shifted);
  testAssert(draw.winrateError[0] == 0);
  testAssert(!draw.checkStats99(0.45,0.45,0.0006));
  *shifted = *base;
  shifted->policyProbs[2000] = 0.5f; shifted->policyProbs[2085] = 0.1f;
  GpuErrorStats tail;
  tail.appendStats(base,shifted);
  testAssert(tail.topPolicyDiff[0] == 0);
  testAssert(!tail.checkStatsMax(1.35,1.35,0.0012));
  testAssert(std::abs(tail.policyKLDiv[0] - (0.2*log(0.2/0.5)+0.4*log(0.4/0.1))) < 1e-6);
  shifted->policyProbs[2000] = -1;
  bool rejected = false;
  try { tail.appendStats(base,shifted); } catch(const StringError&) { rejected = true; }
  testAssert(rejected);
}

static vector<BoardHistory> loadErrorTestHists(
  const string& boardSizeDataset, const vector<string>& sgfs, bool quickTest
) {
  Rand filterRand("Tests::runFP16Test filter rand");
  auto loadHists = [&](const std::vector<string>& sgfStrs) {
    std::vector<BoardHistory> hists;
    for(const string& sgfStr: sgfStrs) {
      std::unique_ptr<Sgf> sgf = Sgf::parse(sgfStr);
      std::set<Hash128> uniqueHashes;
      const bool hashComments = false;
      const bool hashParent = false;
      const bool flipIfPassOrWFirst = false;
      const bool allowGameOver = false;
      sgf->iterAllUniquePositions(
        uniqueHashes,
        hashComments,
        hashParent,
        flipIfPassOrWFirst,
        allowGameOver,
        NULL,
        [&](const Sgf::PositionSample& sample, const BoardHistory& hist, const string& comments) {
          (void)sample;
          (void)comments;
          if(!quickTest || filterRand.nextBool(0.3))
            hists.push_back(hist);
        }
      );
    }
    return hists;
  };

  std::vector<BoardHistory> hists;
  if(!sgfs.empty())
    hists = loadHists(sgfs);
  else if(boardSizeDataset == "9")
    hists = loadHists(TestCommon::getMultiGameSize9Data());
  else if(boardSizeDataset == "13")
    hists = loadHists(TestCommon::getMultiGameSize13Data());
  else if(boardSizeDataset == "19")
    hists = loadHists(TestCommon::getMultiGameSize19Data());
  else if(boardSizeDataset == "10x14")
    hists = loadHists(TestCommon::getMultiGameSize10x14Data());
  else if(boardSizeDataset == "rectangle")
    hists = loadHists(TestCommon::getMultiGameRectangleData());
  else
    throw StringError("Unknown dataset to test gpu error on: " + boardSizeDataset);

  if(hists.empty()) throw StringError("GPU error dataset has no sampled ongoing positions");
  return hists;
}

static MiscNNInputParams errorTestParams(
  const BoardHistory& hist, double optimism, double pda, double temperature
) {
  const Board& board = hist.getRecentBoard(0);
  const uint64_t hash = BoardHistory::getSituationRulesAndKoHash(board,hist,hist.presumedNextMovePla,0.5).hash0;
  vector<int> symmetries;
  for(int s = 0; s < SymmetryHelpers::NUM_SYMMETRIES; s++)
    if(hist.rules.allowsSymmetry(s)) symmetries.push_back(s);
  MiscNNInputParams params;
  params.symmetry = symmetries[hash % symmetries.size()];
  params.policyOptimism = optimism;
  params.playoutDoublingAdvantage = pda;
  params.nnPolicyTemperature = (float)temperature;
  return params;
}

static std::shared_ptr<NNOutput> evalErrorBoard(NNEvaluator* nnE, const BoardHistory& hist, MiscNNInputParams params) {
  params.passAliveSuicideRulesOverride = nnE->modelPreferPassAliveUnderSuicideRules() ? 1 : 0;
  params.excludeTerritoryAdjAtariOverride = nnE->modelPreferExcludeTerritoryAdjacentToAtari() ? 1 : 0;
  NNResultBuf buf;
  const SGFMetadata metadata = SGFMetadata::getProfile("preaz_5k");
  nnE->evaluate(hist.getRecentBoard(0),hist,hist.presumedNextMovePla,&metadata,params,buf,true);
  return buf.result;
}

static bool runErrorComparisons(
  Logger& logger, size_t count, int maxBatchSize, bool separateCurrent, bool verbose,
  const std::function<std::shared_ptr<NNOutput>(size_t,bool)>& evaluate, bool& fp32BatchSuccessBuf,
  const string& referenceFileName
) {
  std::vector<std::shared_ptr<NNOutput>> referenceValues;
  bool loadedReferenceValuesFromFile = false;
#ifndef USE_EIGEN_BACKEND
  if(referenceFileName != "") {
    loadReferenceValuesFromFile(referenceValues, referenceFileName, logger, verbose);
    loadedReferenceValuesFromFile = true;
  }
#endif
  (void)loadReferenceValuesFromFile;

  std::vector<std::shared_ptr<NNOutput>> fp32;
  std::vector<std::shared_ptr<NNOutput>> fp32Batched(count);
  std::vector<std::shared_ptr<NNOutput>> current;
  std::vector<std::shared_ptr<NNOutput>> currentBatched(count);

  if(verbose)
    logger.write("Beginning evaluations! These may take a long time on pure CPU, or on a weak GPU, but on a decent GPU shouldn't take too long.");

  if(verbose)
    logger.write("Running evaluations in fp32");
  fp32.reserve(count);
  for(size_t i = 0; i < count; i++)
    fp32.push_back(evaluate(i,true));

  Rand rand;

  auto batched = [&](bool fp32Mode, vector<std::shared_ptr<NNOutput>>& results) {
    vector<uint32_t> permutation(maxBatchSize);
    rand.fillShuffledUIntRange(maxBatchSize,permutation.data());
    vector<std::exception_ptr> errors(maxBatchSize);
    vector<std::thread> threads;
    threads.reserve(maxBatchSize);
    try {
      for(int t = 0; t < maxBatchSize; t++) threads.emplace_back([&,t]() {
        try {
          for(size_t i = permutation[t]; i < count; i += maxBatchSize)
            results[i] = evaluate(i,fp32Mode);
        }
        catch(...) { errors[t] = std::current_exception(); }
      });
    }
    catch(...) {
      for(auto& thread : threads) thread.join();
      throw;
    }
    for(auto& thread : threads) thread.join();
    for(const auto& error : errors) if(error) std::rethrow_exception(error);
  };
  if(maxBatchSize <= 1) fp32Batched = fp32;
  else {
    if(verbose) logger.write("Running batched evaluations in fp32");
    batched(true,fp32Batched);
  }
  if(separateCurrent) {
    if(verbose) logger.write("Running evaluations using current config");
    for(size_t i = 0; i < count; i++) current.push_back(evaluate(i,false));
    if(maxBatchSize <= 1) currentBatched = current;
    else {
      if(verbose) logger.write("Running batched evaluations using current config");
      batched(false,currentBatched);
    }
  }

  if(loadedReferenceValuesFromFile) {
    if(referenceValues.size() != fp32.size())
      throw StringError(
        "Number of reference values loaded from file does not match number of positions "
        + Global::uint64ToString(referenceValues.size()) + " " + Global::uint64ToString(fp32.size()));
  }
  else {
    logger.write("Using unbatched fp32 as the reference values");
    referenceValues = fp32;
  }

  if(verbose) {
    logger.write("Computed stats on " + Global::uint64ToString((uint64_t)referenceValues.size()) + " positions");
    logger.write("Reporting the average, 90%, 99%, and max abs error between the following configurations: ");
  }

  auto computeStats = [&](const string& name, const std::vector<std::shared_ptr<NNOutput>>& candidateValues, GpuErrorStats& stats) {
    for(size_t i = 0; i<referenceValues.size(); i++) {
      if(loadedReferenceValuesFromFile && referenceValues[i]->nnHash != candidateValues[i]->nnHash)
        throw StringError("GPU error reference input hash differs at position " + Global::uint64ToString(i));
      stats.appendStats(referenceValues[i], candidateValues[i]);
    }
    if(verbose)
      stats.reportStats(name, logger);
  };

  fp32BatchSuccessBuf = true;
  bool success = true;

  {
    GpuErrorStats stats;
    computeStats("fp32 error vs reference", fp32, stats);
    // some backends (notably TensorRT, whose "fp32" uses TF32/fused tensor-core GEMMs) on rectangle boards.
    fp32BatchSuccessBuf = fp32BatchSuccessBuf && stats.checkStats99( 0.45, 0.45, 0.0006);
    fp32BatchSuccessBuf = fp32BatchSuccessBuf && stats.checkStatsMax(1.35, 1.35, 0.0012);
    if(verbose)
      stats.reportClosestMargin("fp32 error vs reference", logger, 0.45, 0.45, 0.0006, 1.35, 1.35, 0.0012);
  }

  {
    GpuErrorStats stats;
    computeStats("batched fp32 error vs reference", fp32Batched, stats);
    fp32BatchSuccessBuf = fp32BatchSuccessBuf && stats.checkStats99( 0.45, 0.45, 0.0006);
    fp32BatchSuccessBuf = fp32BatchSuccessBuf && stats.checkStatsMax(1.35, 1.35, 0.0012);
    if(verbose)
      stats.reportClosestMargin("batched fp32 error vs reference", logger, 0.45, 0.45, 0.0006, 1.35, 1.35, 0.0012);
  }

  if(separateCurrent) {
    {
      GpuErrorStats stats;
      computeStats("current cfg error vs reference", current, stats);
      success = success && stats.checkStats99( 2.0, 2.50, 0.0020);
      success = success && stats.checkStatsMax(5.0, 6.00, 0.0040);
      if(verbose)
        stats.reportClosestMargin("current cfg error vs reference", logger, 2.0, 2.50, 0.0020, 5.0, 6.00, 0.0040);
    }
    {
      GpuErrorStats stats;
      computeStats("batched current cfg error vs reference", currentBatched, stats);
      success = success && stats.checkStats99( 2.0, 2.50, 0.0020);
      success = success && stats.checkStatsMax(5.0, 6.00, 0.0040);
      if(verbose)
        stats.reportClosestMargin("batched current cfg error vs reference", logger, 2.0, 2.50, 0.0020, 5.0, 6.00, 0.0040);
    }
  }

#ifdef USE_EIGEN_BACKEND
  if(referenceFileName != "")
    saveReferenceValuesToFile(referenceValues, referenceFileName, logger, verbose);
#endif
  (void)saveReferenceValuesToFile;

  return success && fp32BatchSuccessBuf;

}

bool Tests::runBackendErrorTest(
  NNEvaluator* nnEval,
  NNEvaluator* nnEval32,
  Logger& logger,
  const string& boardSizeDataset,
  int maxBatchSizeCap,
  bool verbose,
  bool quickTest,
  double policyOptimismForTest,
  double pdaForTest,
  double nnPolicyTemperatureForTest,
  bool& fp32BatchSuccessBuf,
  const string& referenceFileName
) {

  int maxBatchSize = nnEval->getMaxBatchSize();
  if(maxBatchSize != nnEval32->getMaxBatchSize())
    throw StringError("Inconsistent max batch size for fp16 test");
  if(maxBatchSizeCap > 0)
    maxBatchSize = std::min(maxBatchSize,maxBatchSizeCap);
  if(maxBatchSize <= 0)
    throw StringError("Invalid max batch size for fp16 test");

  const auto hists = loadErrorTestHists(boardSizeDataset,{},quickTest);
  auto evaluate = [&](size_t i, bool fp32) {
    return evalErrorBoard(fp32 ? nnEval32 : nnEval,hists[i],
      errorTestParams(hists[i],policyOptimismForTest,pdaForTest,nnPolicyTemperatureForTest));
  };
  return runErrorComparisons(logger,hists.size(),maxBatchSize,nnEval != nnEval32,verbose,
                            evaluate,fp32BatchSuccessBuf,referenceFileName);
}

bool Tests::runModelBackendErrorTest(
  SelfplayModel& model, SelfplayModel& model32, Logger& logger,
  const string& boardSizeDataset, const vector<string>& sgfs,
  bool quickTest, double optimism, double pda, double temperature,
  const string& referenceFileName, int recurrentSteps
) {
  const auto hists = loadErrorTestHists(boardSizeDataset,sgfs,quickTest);
  const int maxBatchSize = model.getMaxBatchSize();
  if(maxBatchSize <= 0 || maxBatchSize != model32.getMaxBatchSize())
    throw StringError("Inconsistent max batch size for GPU error test");
  const bool muzero = model.muZeroEvaluator() != nullptr;
  if(!muzero && recurrentSteps != 0) throw StringError("Recurrence requires MuZero");
  bool success = true;
  for(int depth = 0; depth <= recurrentSteps; depth++) {
    if(muzero) logger.write(depth == 0 ? "Testing MuZero initial" :
      "Testing MuZero recurrent depth " + Global::intToString(depth));
    auto evaluate = [&](size_t i, bool fp32) -> std::shared_ptr<NNOutput> {
      SelfplayModel& service = fp32 ? model32 : model;
      const BoardHistory& hist = hists[i];
      auto params = errorTestParams(hist,optimism,pda,temperature);
      if(service.boardEvaluator()) return evalErrorBoard(service.boardEvaluator(),hist,params);
      auto* evaluator = service.muZeroEvaluator();
      const auto& desc = evaluator->getModel().representationDesc();
      params.passAliveSuicideRulesOverride = desc.preferPassAliveUnderSuicideRules ? 1 : 0;
      params.excludeTerritoryAdjAtariOverride = desc.preferExcludeTerritoryAdjacentToAtari ? 1 : 0;
      MuZero::PositionBuffer buffer;
      // Explicit symmetry and deterministic static actions: no model-dependent
      // sampling, real successor board, cache read, or cross-service latent.
      Rand rand("GPU error MuZero trajectory " + Global::uint64ToString(i));
      const SGFMetadata metadata = SGFMetadata::getProfile("preaz_5k");
      auto output = evaluator->initial(hist.getRecentBoard(0),hist,hist.presumedNextMovePla,
        &metadata,params,buffer,rand,true);
      Hash128 trajectoryHash = output->nnHash;
      const Board& board = hist.getRecentBoard(0);
      const int domain = hist.rules.isXiangqi() ? Xiangqi::NUM_ACTIONS :
        board.x_size*board.y_size + (hist.rules.isGo() ? 1 : 0);
      for(int step = 0; step < depth; step++) {
        int action = rand.nextInt(0,domain-1);
        if(!hist.rules.isXiangqi()) action = action == board.x_size*board.y_size ?
          service.getNNXLen()*service.getNNYLen() :
          NNPos::xyToPos(action % board.x_size,action / board.x_size,service.getNNXLen());
        output = evaluator->recurrent(*output,action,params,buffer);
        // Recurrent states have no real-board NN hash. Extend the root input
        // hash in this diagnostic only, so reference checks identify the path.
        trajectoryHash.hash0 = Hash::splitMix64(trajectoryHash.hash0 + (uint64_t)action);
        trajectoryHash.hash1 = Hash::splitMix64(trajectoryHash.hash1 + (uint64_t)action + 1);
      }
      // Comparisons retain values only, releasing backend latents per request.
      auto values = std::make_shared<NNOutput>(*output);
      values->nnHash = trajectoryHash;
      return values;
    };
    string depthReferenceFile = referenceFileName;
    if(depth > 0 && !depthReferenceFile.empty())
      depthReferenceFile += ".recurrent-" + Global::intToString(depth);
    bool fp32Okay = true;
    const bool okay = runErrorComparisons(logger,hists.size(),maxBatchSize,&model != &model32,true,
      evaluate,fp32Okay,depthReferenceFile);
    success = okay && success;
  }
  return success;
}
