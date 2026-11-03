#ifndef TESTS_H
#define TESTS_H

#include <sstream>

#include "../core/global.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../core/test.h"
#include "../game/board.h"
#include "../game/rules.h"
#include "../game/boardhistory.h"

class NNEvaluator;
class SelfplayModel;

namespace Tests {
  void runNNBenchmarkTests();
  void runMuZeroData(const std::string& directory);
  void runPlacementTests();
  void runBalancedOpeningTests();
  void runPlacementPatternTests(const std::string& output);
  void runXiangqiTests();
  void runXiangqiPatternTests(const std::string& output);
  void runXiangqiInference(const std::string& model, const std::string& request, const std::string& response);
  void runGumbelTests();
  void runMuZeroModelDesc(const std::string& path);
  void runMuZeroInference(const std::string& directory, const std::string& request, const std::string& response);
  void runMuZeroBatcher(const std::string& directory, const std::string& request, const std::string& response);
  void runMuZeroPosition(const std::string& directory, const std::string& request, const std::string& response);
  void runMuZeroSearch(const std::string& directory, const std::string& request, const std::string& response);
  void runMuZeroPlay(const std::string& directory, const std::string& config, const std::string& output);
  void runGumbelReference();
  void runFourResultTests();
  //testboardbasic.cpp
  void runBoardIOTests();
  void runBoardBasicTests();
  void runBoardUndoTest();
  void runBoardHandicapTest();
  void runBoardStressTest();
  void runBoardReplayTest();


  //testboardarea.cpp
  void runBoardAreaTests();

  //testrules.cpp
  void runRulesTests();

  //testpassalivesuicide.cpp
  void runPassAliveSuicideModeTests();
  void runExcludeTerritoryAtariModeTests();

  //testscore.cpp
  void runScoreTests();

  //testsgf.cpp
  void runSgfTests();
  void runSgfFileTests();

  //testnninputs.cpp
  void runNNInputsV3V4Tests();
  void runExcludeTerritoryAtariNNInputsTests();

  //testsymmetries.cpp
  void runBasicSymmetryTests();
  void runBoardSymmetryTests();
  void runSymmetryDifferenceTests();

  //testsearchnonn.cpp
  void runNNLessSearchTests();
  //testsearch.cpp
  void runSearchTests(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16);
  //testsearchv3.cpp
  void runSearchTestsV3(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16);
  //testsearchv8.cpp
  void runSearchTestsV8(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, bool useFP16);
  //testsearchv9.cpp
  void runSearchTestsV9(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, bool useFP16);

  //testsearchmisc.cpp
  void runNNOnTinyBoard(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16);
  void runNNSymmetries(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, bool useFP16);
  void runNNOnManyPoses(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, int symmetry, bool useFP16, const std::string& comparisonFile);
  void runNNBatchingTest(const std::string& modelFile, bool inputsNHWC, bool cudaNHWC, bool useFP16);

  //testtime.cpp
  void runTimeControlsTests();

  //testtrainingwrite.cpp
  void runTrainingWriteTests();
  void runXiangqiData(const std::string& directory);
  void runXiangqiPlay(const std::string& model, const std::string& config, const std::string& output);
  void runXiangqiSgf(const std::string& directory);
  void runPassAliveSuicideGameTests();
  void runSelfplayInitTestsWithNN(const std::string& modelFile);
  void runSekiTrainWriteTests(const std::string& modelFile);
  void runMoreSelfplayTestsWithNN(const std::string& modelFile);
  void runSelfplayStatTestsWithNN(const std::string& modelFile);

  //testnn.cpp
  void runNNLayerTests();
  void runNNSymmetryTests();

  //testonnxmodelfile.cpp
  void runOnnxModelFileTests(const std::string& scratchDir, const std::string& modelFile);


  //testnnevalcanary.cpp
  void runCanaryTests(NNEvaluator* nnEval, int symmetry, bool print);
  void runBackendErrorStatsTests();
  bool runBackendErrorTest(
    NNEvaluator* nnEval,
    NNEvaluator* nnEval32,
    Logger& logger,
    const std::string& boardSizeDataset,
    int maxBatchSizeCap,
    bool verbose,
    bool quickTest,
    double policyOptimismForTest,
    double pdaForTest,
    double nnPolicyTemperatureForTest,
    bool& fp32BatchSuccessBuf,
    //Values on disk to compare correctness. We consider the pure-cpu float32 Eigen implementation of the neural network
    //to be the source of truth, since it is more likely to be stable and doesn't depend special hardware or drivers like
    //GPUs or other accelerators.
    //When running with Eigen backend, will overwrite this file with Eigen's results.
    const std::string& referenceFileName
  );
  // Same source comparison and batch scheduling, with game-specific SGFs and
  // optional MuZero recurrence. Empty sgfs select the original Go dataset.
  bool runModelBackendErrorTest(
    SelfplayModel& model, SelfplayModel& model32, Logger& logger,
    const std::string& boardSizeDataset, const std::vector<std::string>& sgfs,
    bool quickTest, double policyOptimism, double pda, double policyTemperature,
    const std::string& referenceFileName, int recurrentSteps
  );

  //testbackendreference.cpp
  //Absolute-output check against compiled-in reference data blended across a sampling of nets
  //from the training run (see backendreferencedata.cpp). Only nets from that run are expected
  //to pass. Source calibration exempts models too small for meaningful thresholds;
  //explicit limits do not apply this exemption or source model-size lenience.
  bool runBackendReferenceTest(
    NNEvaluator* nnEval,
    Logger& logger,
    bool verbose,
    //Defaults for positions whose reference data does not specify its own policyOptimism/pda.
    //The temperature applies to all positions.
    double policyOptimismForTest,
    double pdaForTest,
    double nnPolicyTemperatureForTest,
    //Scales all limits in the lenient direction as it grows. 1.0 for normal checking. Composed
    //with automatic model-size-based lenience only for source calibration.
    double lenienceFactor,
    //If nonempty, load reference data from this file instead of the compiled-in data.
    const std::string& referenceDataFileOverride,
    //If nonempty, dump this net's outputs on the reference positions, for calibration.
    const std::string& dumpCandidateFileName,
    // Optional service for MuZero initial inference and explicit calibration
    // for model families outside the source Go run. Export alone makes no
    // pass/fail claim and requires dumpCandidateFileName.
    SelfplayModel* model = nullptr,
    const std::string& limitsFileName = "",
    bool dumpOnly = false
  );

  //testconfig.cpp
  void runInlineConfigTests();
  void runConfigTests(const std::vector<std::string>& args);
  void runParseAllConfigsTest();
  void runTaskParsingTests();

  //testmisc.cpp
  void runCollectFilesTests();
  void runLoadModelTests();

  //testbook.cpp
  void runBookTests();
  void runBookGameTests(const std::string& outputDir = "");
}

namespace TestCommon {
  bool boardsSeemEqual(const Board& b1, const Board& b2);

  constexpr int MIN_BENCHMARK_SGF_DATA_SIZE = 7;
  constexpr int MAX_BENCHMARK_SGF_DATA_SIZE = 19;
  constexpr int DEFAULT_BENCHMARK_SGF_DATA_SIZE = std::min(Board::DEFAULT_LEN,MAX_BENCHMARK_SGF_DATA_SIZE);
  std::string getBenchmarkSGFData(int boardSize);
  std::string getBenchmarkSGFData(Rules::Game game, int xSize, int ySize);

  std::vector<std::string> getMultiGameSize9Data();
  std::vector<std::string> getMultiGameSize13Data();
  std::vector<std::string> getMultiGameSize19Data();
  std::vector<std::string> getMultiGameSize10x14Data();
  std::vector<std::string> getMultiGameRectangleData();

  //backendreferencedata.cpp (machine-generated, see that file for the JSON schema)
  std::vector<std::string> getBackendReferenceJsonData();

  void overrideForBackends(bool& inputsNHWC, bool& useNHWC);
}

#endif
