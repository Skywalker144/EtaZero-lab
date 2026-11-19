#include "../tests/tinymodel.h"

#include <fstream>
#include <sstream>

#include "../core/fileutils.h"
#include "../core/rand.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nneval.h"
#include "../program/setup.h"

using namespace std;

static void decodeBase64(const string& input, string& output) {
  output = "";

  string valueToCharMap = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  int32_t charToValueMap[256];
  for(int i = 0; i<256; i++)
    charToValueMap[i] = -1;
  for(int i = 0; i<valueToCharMap.size(); i++)
    charToValueMap[(int)valueToCharMap[i]] = i;

  int32_t carry = 0;
  int numBitsInCarry = 0;
  for(unsigned char c : input) {
    if(c == '\n' || c == '\r' || c == ' ')
      continue;
    int value = charToValueMap[c];
    if(value < 0)
      break;

    carry = (carry << 6) + value;
    numBitsInCarry += 6;
    if(numBitsInCarry >= 8) {
      int extracted = carry >> (numBitsInCarry-8);
      carry -= (extracted << (numBitsInCarry-8));
      numBitsInCarry -= 8;
      testAssert(extracted >= 0 && extracted < 256);
      output.push_back((char)extracted);
    }
  }
  if(carry != 0)
    throw StringError("decodeBase64 got leftover bits");
}

static void requireApproxEqual(double x, double expected, double scale, const NNResultBuf& buf, const Board& board, const char *file, int line) {
  if(!std::isfinite(x) || !std::isfinite(expected) || std::fabs(x-expected) > scale) {
    buf.result->debugPrint(cout,board);
    throw StringError(
      "Tiny neural net test got invalid values - is the GPU working? " +
      string(file) + " line " + Global::intToString(line) + " " + Global::doubleToString(x) + " " + Global::doubleToString(expected)
    );
  }
}

NNEvaluator* TinyModelTest::runTinyModelTest(const string& baseDir, Logger& logger, ConfigParser& cfg, bool randFileName, double errorTolFactor) {
  logger.write("Running tiny net to sanity-check that GPU is working");

  NNEvaluator* nnEvalRet;
  {
    string base64Data;
    base64Data += TinyModelTest::tinyModelBase64Part0;
    base64Data += TinyModelTest::tinyModelBase64Part1;
    base64Data += TinyModelTest::tinyModelBase64Part2;
    base64Data += TinyModelTest::tinyModelBase64Part3;
    base64Data += TinyModelTest::tinyModelBase64Part4;
    base64Data += TinyModelTest::tinyModelBase64Part5;
    base64Data += TinyModelTest::tinyModelBase64Part6;
    string binaryData;
    decodeBase64(base64Data, binaryData);

    Rand rand;
    const string tmpModelFile =
      randFileName ?
      (baseDir + "/" + "tmpTinyModel_" + Global::uint64ToHexString(rand.nextUInt64()) + ".bin.gz") :
      (baseDir + "/" + "tmpTinyModel.bin.gz");
    ofstream outModel;
    FileUtils::open(outModel,tmpModelFile.c_str(),ios::binary);
    outModel << binaryData;
    outModel.close();

    const int expectedConcurrentEvals = 1;
    const bool requireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      "tinyModel",tmpModelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,Setup::MaxBatchSizeRequest::fromConcurrency(),requireExactNNLen,disableFP16,
      Setup::SETUP_FOR_DISTRIBUTED
    );
    nnEval->setDoRandomize(false);
    nnEval->setDefaultSymmetry(6);

    Board board = Board::parseBoard(19,19,R"%%(
...................
...................
..xx......o.x..oo..
..xo.o.........ox..
................x..
...x..........oox..
...............x...
...................
...................
..x................
...................
...................
...................
.....o.......o.x...
...o.x.............
. xo.........o.x...
..xxooo......ox....
....xx.............
...................
)%%");

    const Player nextPla = P_BLACK;
    const Rules rules = Rules::getTrompTaylorish();
    const BoardHistory hist(board,nextPla,rules,0,BoardHistoryModes(false,false));

    auto runOneTest = [&]() {
      MiscNNInputParams nnInputParams;
      NNResultBuf buf;
      bool skipCache = true;
      nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

      //ostringstream out;
      //buf.result->debugPrint(out,board);
      //cout << out.str() << endl;

#define EQ(x,expected,scale) requireApproxEqual((x), (expected), (scale*errorTolFactor), buf, board, __FILE__, __LINE__)
      NNOutput& nnOutput = *(buf.result);
      EQ(nnOutput.whiteWinProb, .43298, 0.01);
      EQ(nnOutput.whiteLossProb, .56702, 0.01);
      EQ(nnOutput.whiteNoResultProb, .0000, 0.002);
      EQ(nnOutput.varTimeLeft, 15.79, 1.0);
      EQ(nnOutput.shorttermWinlossError, .22408, 0.01);

      double expectedPolicy[361] = {
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,
        0,    0,    0,    0,    3,   46,    7,    9,   14,    9,   39,   67,   12,   57,   52,    1,    1,   11,    0,
        0,    0,   -1,   -1,   36,   25,   83,  121,   79,   18,   -1,   22,   -1,    7,    3,   -1,   -1,  304,    1,
        0,    0,   -1,   -1,   33,   -1,    7,   95,   42,   37,   83,   78,   22,   33,   19,   -1,   -1,   31,    1,
        0,    0,    1,    3,    7,    3,   16,   22,   11,   20,   33,   84,  108,   14,    5,  147,   -1,    0,    0,
        0,    0,    0,   -1,    5,   23,   12,    8,    4,    5,    6,    8,   27,    4,   -1,   -1,   -1,    0,    0,
        0,    0,    2,    0,    1,    4,    4,    4,    3,    2,    3,    4,   14,   24,  294,   -1,   46,    7,    0,
        0,    0,    6,    4,    1,    2,    2,    2,    2,    2,    2,    3,    7,   45,   36,   57,   80,    4,    0,
        0,    1,    2,    3,    2,    1,    2,    2,    2,    2,    2,    2,    3,    6,    9,   91,   49,    3,    0,
        0,    0,   -1,    1,    3,    2,    2,    2,    2,    2,    2,    2,    2,    3,    4,   33,   45,    2,    0,
        0,    1,    2,    6,    4,    3,    3,    3,    2,    2,    3,    3,    4,    3,    5,   45,   78,    3,    0,
        0,    3,   40,   15,    8,   15,    7,    5,    3,    3,    4,    5,    6,   18,   31,  143,   98,    3,    0,
        0,   13,  152,  120,   30,   65,   21,   14,    4,    4,    5,   10,   13,   14,   22,   57,   46,    2,    0,
        0,   47,  270,   66,  723,   -1, 1055,   21,    6,    7,    6,   19,    2,   -1,   18,   -1,    3,    2,    0,
        0,   19,  489,   -1,   19,   -1,  480,  185,   19,   10,   11,   17,    6,    7,   36,    6,    8,    2,    0,
        0,    0,   -1,   -1,   40,    0,   10,   89,   69,   50,   40,   78,    1,   -1,   12,   -1,    5,    2,    0,
        0,    0,   -1,   -1,   -1,   -1,   -1,   89,  103,   19,   30,   85,    2,   -1,   -1,    1,   23,    1,    0,
        0,    1,    0,    4,   -1,   -1,  465,  195,   13,    1,    2,   17,   11,  287,  435,   33,    4,    0,    0,
        0,    0,    0,    0,    0,    0,    1,    1,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
      };
      for(int pos = 0; pos<361; pos++) {
        double prob = nnOutput.policyProbs[pos];
        if(expectedPolicy[pos] >= 0) {
          EQ(prob*10000, expectedPolicy[pos], std::min(60.0, expectedPolicy[pos] * 0.1 + 2.0) + std::min(10.0, expectedPolicy[pos] * 0.1));
        }
      }


    };

    runOneTest();

    auto runAFewTests = [&]() {
      try {
        for(int i = 0; i<10; i++)
          runOneTest();
      }
      catch(std::exception& e) {
        cout << "Tiny model test exception: " << e.what() << endl;
        throw;
      }
    };

    vector<std::thread> testThreads;
    testThreads.reserve(4);
    for(int i = 0; i<4; i++)
      testThreads.emplace_back(runAFewTests);
    for(int i = 0; i<4; i++)
      testThreads[i].join();

    if(!FileUtils::tryRemoveFile(tmpModelFile)) {
      logger.write("Warning: could not delete " + tmpModelFile);
    }

    // We return this one
    nnEvalRet = nnEval;
  }

  {
    string base64Data;
    base64Data += TinyModelTest::tinyMishModelBase64;
    string binaryData;
    decodeBase64(base64Data, binaryData);

    Rand rand;
    const string tmpModelFile =
      randFileName ?
      (baseDir + "/" + "tmpTinyMishModel_" + Global::uint64ToHexString(rand.nextUInt64()) + ".bin.gz") :
      (baseDir + "/" + "tmpTinyMishModel.bin.gz");
    ofstream outModel;
    FileUtils::open(outModel,tmpModelFile.c_str(),ios::binary);
    outModel << binaryData;
    outModel.close();

    const int expectedConcurrentEvals = 1;
    const bool requireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      "tinyModel",tmpModelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,Setup::MaxBatchSizeRequest::fromConcurrency(),requireExactNNLen,disableFP16,
      Setup::SETUP_FOR_DISTRIBUTED
    );
    nnEval->setDoRandomize(false);
    nnEval->setDefaultSymmetry(7);

    Board board = Board::parseBoard(19,19,R"%%(
...................
...................
..xx......o.x..oo..
..xo.o.........ox..
................x..
...x..........oox..
...............x...
...................
...................
..x................
...................
...................
...................
.....o.......o.x...
...o.x.............
. xo.........o.x...
..xxooo......ox....
....xx.............
...................
)%%");

    const Player nextPla = P_BLACK;
    const Rules rules = Rules::getTrompTaylorish();
    const BoardHistory hist(board,nextPla,rules,0,BoardHistoryModes(false,false));

    auto runOneTest = [&]() {
      MiscNNInputParams nnInputParams;
      NNResultBuf buf;
      bool skipCache = true;
      nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

      // ostringstream out;
      // buf.result->debugPrint(out,board);
      // cout << out.str() << endl;

#define EQ(x,expected,scale) requireApproxEqual((x), (expected), (scale*errorTolFactor), buf, board, __FILE__, __LINE__)
      NNOutput& nnOutput = *(buf.result);
      EQ(nnOutput.whiteWinProb, .4792, 0.02);
      EQ(nnOutput.whiteLossProb, .5208, 0.02);
      EQ(nnOutput.whiteNoResultProb, .0000, 0.004);
      EQ(nnOutput.varTimeLeft, 15.2, 2.0);
      EQ(nnOutput.shorttermWinlossError, .531, 0.02);

      double expectedPolicy[361] = {
        1,    5,    4,    4,    4,    5,    4,    4,    4,    5,    6,    8,    8,    6,    6,    6,    6,    7,    1,
        5,    8,    4,    9,   37,   40,   34,   19,   26,   51,   64,   56,   23,   43,   53,   20,   18,   31,    8,
        4,    4,   -1,   -1,   42,   51,   66,   37,   36,   56,   -1,   55,   -1,   35,   34,   -1,   -1,  468,    9,
        3,    5,   -1,   -1,   48,   -1,   48,   25,   24,   46,   69,   63,   20,   58,   30,   -1,   -1,   61,   14,
        3,   11,   20,   15,   10,   22,   46,   22,   19,   30,   37,   37,   51,   56,   36,  109,   -1,    3,   10,
        4,   13,    8,   -1,   12,   34,   28,   21,   22,   23,   23,   23,   37,   39,   -1,   -1,   -1,    6,    9,
        4,   19,   21,   11,   21,   24,   27,   24,   24,   24,   23,   22,   31,   60,  132,   -1,   66,   38,    6,
        4,   14,   22,   19,   20,   24,   25,   24,   24,   25,   24,   23,   28,   53,   59,   53,   70,   27,    5,
        4,   16,   12,   22,   23,   29,   24,   24,   24,   24,   24,   24,   26,   28,   34,   52,   48,   26,    4,
        5,    8,   -1,   12,   26,   28,   24,   24,   24,   24,   24,   25,   26,   26,   28,   33,   38,   23,    4,
        5,   21,   15,   25,   26,   26,   22,   23,   24,   24,   24,   24,   24,   27,   30,   34,   39,   22,    4,
        5,   27,   46,   30,   33,   36,   33,   25,   24,   24,   23,   22,   32,   41,   46,   42,   34,   21,    4,
        5,   35,   70,   64,   58,   96,   39,   36,   25,   24,   20,   24,   45,   62,   65,   30,   33,   19,    4,
        5,   55,  146,   75,  268,   -1,  237,   29,   24,   23,   18,   19,   28,   -1,   49,   -1,   11,   17,    4,
        5,   38,  231,   -1,   58,   -1,  113,   55,   24,   21,   15,   17,   50,   29,   60,    9,   14,   13,    4,
        6,    7,   -1,   -1,   30,   12,   19,   51,   37,   21,   16,   22,   11,   -1,   31,   -1,   11,   17,    4,
        7,    5,   -1,   -1,   -1,   -1,   -1,   82,   57,   31,   27,   37,   17,   -1,   -1,    5,   35,   24,    4,
        8,   22,    7,   19,   -1,   -1,  183,   76,   31,   19,   18,   24,   45,  201,  141,   41,   25,   17,    6,
        1,    8,    8,   11,    3,    4,    9,    7,    4,    4,    4,    4,    5,    6,    9,    6,    5,    6,    1,
      };
      for(int pos = 0; pos<361; pos++) {
        double prob = nnOutput.policyProbs[pos];
        if(expectedPolicy[pos] >= 0) {
          EQ(prob*10000, expectedPolicy[pos], std::min(60.0, expectedPolicy[pos] * 0.15 + 2.0) + std::min(10.0, expectedPolicy[pos] * 0.15));
        }
      }


    };

    runOneTest();

    auto runAFewTests = [&]() {
      try {
        for(int i = 0; i<10; i++)
          runOneTest();
      }
      catch(std::exception& e) {
        cout << "Tiny model test exception: " << e.what() << endl;
        throw;
      }
    };

    vector<std::thread> testThreads;
    testThreads.reserve(4);
    for(int i = 0; i<4; i++)
      testThreads.emplace_back(runAFewTests);
    for(int i = 0; i<4; i++)
      testThreads[i].join();

    if(!FileUtils::tryRemoveFile(tmpModelFile)) {
      logger.write("Warning: could not delete " + tmpModelFile);
    }

    //We delete this one
    delete nnEval;
  }

  {
    string base64Data;
    base64Data += TinyModelTest::tinyMishModelBase64;
    string binaryData;
    decodeBase64(base64Data, binaryData);

    Rand rand;
    const string tmpModelFile =
      randFileName ?
      (baseDir + "/" + "tmpTinyMishModel_" + Global::uint64ToHexString(rand.nextUInt64()) + ".bin.gz") :
      (baseDir + "/" + "tmpTinyMishModel.bin.gz");
    ofstream outModel;
    FileUtils::open(outModel,tmpModelFile.c_str(),ios::binary);
    outModel << binaryData;
    outModel.close();

    const int expectedConcurrentEvals = 1;
    const bool requireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      "tinyModel",tmpModelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
      NNPos::MAX_BOARD_LEN,NNPos::MAX_BOARD_LEN,Setup::MaxBatchSizeRequest::fromConcurrency(),requireExactNNLen,disableFP16,
      Setup::SETUP_FOR_DISTRIBUTED
    );
    nnEval->setDoRandomize(false);
    nnEval->setDefaultSymmetry(1);

    Board board = Board::parseBoard(13,6,R"%%(
.............
....o....xxx.
..xx......ox.
..xo.o.xoo.o.
...xo..oxxxxx
........x....
)%%");

    const Player nextPla = P_BLACK;
    const Rules rules = Rules::getTrompTaylorish();
    const BoardHistory hist(board,nextPla,rules,0,BoardHistoryModes(false,false));

    auto runOneTest = [&]() {
      MiscNNInputParams nnInputParams;
      NNResultBuf buf;
      bool skipCache = true;
      nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache);

      // ostringstream out;
      // buf.result->debugPrint(out,board);
      // cout << out.str() << endl;

#define EQ(x,expected,scale) requireApproxEqual((x), (expected), (scale*errorTolFactor), buf, board, __FILE__, __LINE__)
      NNOutput& nnOutput = *(buf.result);
      EQ(nnOutput.whiteWinProb, .3913, 0.02);
      EQ(nnOutput.whiteLossProb, .6087, 0.02);
      EQ(nnOutput.whiteNoResultProb, .0000, 0.004);
      EQ(nnOutput.varTimeLeft, 10.0, 2.0);
      EQ(nnOutput.shorttermWinlossError, .516, 0.02);

      double expectedPolicy[78] = {
        3,   24,   27,   34,   31,   24,   15,   15,   24,   14,   17,   17,    5,
        21,   45,   17,  159,   -1,  180,  160,  231,  141,   -1,   -1,   -1,   26,
        14,   12,   -1,   -1,  369,  156,  655,  841,  201,  148,   -1,   -1,   69,
        13,   15,   -1,   -1, 2189,   -1,  401,   -1,   -1,   -1,  262,   -1,  188,
        18,   41,   45,   -1,   -1,   74, 2108,   -1,   -1,   -1,   -1,   -1,   -1,
        2,   20,   52,  163,  241,   27,   38,  210,   -1,    7,   15,   23,   10,
      };
      for(int idx = 0; idx<78; idx++) {
        int pos = (idx % 13) + idx / 13 * NNPos::MAX_BOARD_LEN;
        double prob = nnOutput.policyProbs[pos];
        if(expectedPolicy[idx] >= 0) {
          EQ(prob*10000, expectedPolicy[idx], std::min(120.0, expectedPolicy[idx] * 0.15 + 2.0) + std::min(10.0, expectedPolicy[idx] * 0.15));
        }
      }


    };

    runOneTest();

    auto runAFewTests = [&]() {
      try {
        for(int i = 0; i<10; i++)
          runOneTest();
      }
      catch(std::exception& e) {
        cout << "Tiny model test exception: " << e.what() << endl;
        throw;
      }
    };

    vector<std::thread> testThreads;
    testThreads.reserve(4);
    for(int i = 0; i<4; i++)
      testThreads.emplace_back(runAFewTests);
    for(int i = 0; i<4; i++)
      testThreads[i].join();

    if(!FileUtils::tryRemoveFile(tmpModelFile)) {
      logger.write("Warning: could not delete " + tmpModelFile);
    }

    //We delete this one
    delete nnEval;
  }

  logger.write("Tiny net sanity check complete");

  return nnEvalRet;
}

