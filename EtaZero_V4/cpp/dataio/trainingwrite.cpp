#include "../dataio/trainingwrite.h"
#include "muzerowrite.h"

#include "../core/fileutils.h"
#include "../core/test.h"
#include "../neuralnet/modelversion.h"
#include "../game/xiangqi.h"

using namespace std;

ValueTargets::ValueTargets()
  :win(0),
   loss(0),
   draw(0),
   noResult(0)
{}
ValueTargets::~ValueTargets()
{}

//-------------------------------------------------------------------------------------

SidePosition::SidePosition()
  :board(),
   hist(),
   pla(P_BLACK),
   unreducedNumVisits(),
   policyTarget(),
   policySurprise(),
   policyEntropy(),
   searchEntropy(),
   whiteValueTargets(),
   whiteQValueTargets(),
   targetWeight(),
   targetWeightUnrounded(),
   numNeuralNetChangesSoFar(),
   playoutDoublingAdvantagePla(C_EMPTY),
   playoutDoublingAdvantage(0.0)
{}

SidePosition::SidePosition(const Board& b, const BoardHistory& h, Player p, int numNNChangesSoFar)
  :board(b),
   hist(h),
   pla(p),
   unreducedNumVisits(),
   policyTarget(),
   policySurprise(),
   policyEntropy(),
   searchEntropy(),
   whiteValueTargets(),
   whiteQValueTargets(),
   targetWeight(1.0f),
   targetWeightUnrounded(1.0f),
   numNeuralNetChangesSoFar(numNNChangesSoFar),
   playoutDoublingAdvantagePla(C_EMPTY),
   playoutDoublingAdvantage(0.0)
{}

SidePosition::~SidePosition()
{}

//-------------------------------------------------------------------------------------

FinishedGameData::FinishedGameData()
  :bName(),
   wName(),
   bIdx(0),
   wIdx(0),

   startBoard(),
   startHist(),
   endHist(),
   startPla(P_BLACK),
   gameHash(),

   drawEquivalentWinsForWhite(0.0),
   playoutDoublingAdvantagePla(P_BLACK),
   playoutDoublingAdvantage(0.0),

   hitTurnLimit(false),

   numExtraBlack(0),
   mode(0),
   beganInEncorePhase(0),
   usedInitialPosition(0),

   hasFullData(false),
   targetWeightByTurn(),
   targetWeightByTurnUnrounded(),
   policyTargetsByTurn(),
   whiteValueTargetsByTurn(),
   whiteQValueTargetsByTurn(),
   nnRawStatsByTurn(),
   reanalysisByTurn(),

   trainingWeight(1.0),

   sidePositions(),
   changedNeuralNets(),
   bTimeUsed(0.0),
   wTimeUsed(0.0),
   bMoveCount(0),
   wMoveCount(0)
{
}

FinishedGameData::~FinishedGameData() {
  for(size_t i = 0; i<policyTargetsByTurn.size(); i++)
    delete policyTargetsByTurn[i].policyTargets;


  for(size_t i = 0; i<sidePositions.size(); i++)
    delete sidePositions[i];

  for(size_t i = 0; i<changedNeuralNets.size(); i++)
    delete changedNeuralNets[i];
}

void FinishedGameData::printDebug(ostream& out) const {
  auto moveString = [&](Loc loc) {
    return startHist.rules.isXiangqi() ? Xiangqi::actionToString(Xiangqi::locToAction(loc)) : Location::toString(loc,startBoard);
  };
  out << "bName " << bName << endl;
  out << "wName " << wName << endl;
  out << "bIdx " << bIdx << endl;
  out << "wIdx " << wIdx << endl;
  out << "startPla " << PlayerIO::colorToChar(startPla) << endl;
  out << "start" << endl;
  startHist.printDebugInfo(out,startBoard);
  out << "end" << endl;
  endHist.printDebugInfo(out,endHist.getRecentBoard(0));
  out << "gameHash " << gameHash << endl;
  out << "hitTurnLimit " << hitTurnLimit << endl;
  out << "numExtraBlack " << numExtraBlack << endl;
  out << "mode " << mode << endl;
  out << "beganInEncorePhase " << beganInEncorePhase << endl;
  out << "usedInitialPosition " << usedInitialPosition << endl;
  out << "hasFullData " << hasFullData << endl;
  for(int i = 0; i<targetWeightByTurn.size(); i++)
    out << "targetWeightByTurn " << i << " " << targetWeightByTurn[i] << " " << "unrounded" << " " << targetWeightByTurnUnrounded[i] << endl;
  for(int i = 0; i<policyTargetsByTurn.size(); i++) {
    out << "policyTargetsByTurn " << i << " ";
    out << "unreducedNumVisits " << policyTargetsByTurn[i].unreducedNumVisits << " ";
    if(policyTargetsByTurn[i].policyTargets != NULL) {
      const vector<PolicyTargetMove>& target = *(policyTargetsByTurn[i].policyTargets);
      for(int j = 0; j<target.size(); j++)
        out << moveString(target[j].loc) << " " << target[j].policyTarget << " ";
    }
    out << endl;
  }
  for (int i = 0; i < policySurpriseByTurn.size(); i++)
    out << "policySurpriseByTurn " << i << " " << policySurpriseByTurn[i] << endl;
  for (int i = 0; i < policyEntropyByTurn.size(); i++)
    out << "policyEntropyByTurn " << i << " " << policyEntropyByTurn[i] << endl;
  for (int i = 0; i < searchEntropyByTurn.size(); i++)
    out << "searchEntropyByTurn " << i << " " << searchEntropyByTurn[i] << endl;

  for(int i = 0; i<whiteValueTargetsByTurn.size(); i++) {
    out << "whiteValueTargetsByTurn " << i << " ";
    out << whiteValueTargetsByTurn[i].win << " ";
    out << whiteValueTargetsByTurn[i].loss << " ";
    out << whiteValueTargetsByTurn[i].draw << " ";
    out << whiteValueTargetsByTurn[i].noResult << " ";
    out << endl;
  }

  for(int i = 0; i<whiteQValueTargetsByTurn.size(); i++) {
    out << "whiteQValueTargetsByTurn " << i << " ";
    const vector<QValueTargetMove>& target = whiteQValueTargetsByTurn[i].targets;
    for(int j = 0; j<target.size(); j++)
      out << moveString(target[j].loc) << " " << target[j].winLoss << " " << target[j].visits << " ";
    out << endl;
  }

  for(int i = 0; i<nnRawStatsByTurn.size(); i++) {
    out << "Raw Stats " << nnRawStatsByTurn[i].whiteWinLoss << " " << nnRawStatsByTurn[i].policyEntropy << endl;
  }

  for(int i = 0; i<reanalysisByTurn.size(); i++) {
    if(reanalysisByTurn[i].wasReanalyzed) {
      out << "reanalysisByTurn " << i << " ";
      out << "usedOutcomeTargets " << reanalysisByTurn[i].usedOutcomeTargets << " ";
      out << "selectionPolicySurprise " << reanalysisByTurn[i].selectionPolicySurprise << " ";
      out << "selectionValueSurprise " << reanalysisByTurn[i].selectionValueSurprise << " ";
      out << "originalNumVisits " << reanalysisByTurn[i].originalNumVisits << " ";
      out << endl;
    }
  }

  out << "trainingWeight " << trainingWeight << endl;
  for(int i = 0; i<sidePositions.size(); i++) {
    SidePosition* sp = sidePositions[i];
    out << "Side position " << i << endl;
    out << "targetWeight " << sp->targetWeight << " " << "unrounded" << " " << sp->targetWeightUnrounded << endl;
    sp->hist.printDebugInfo(out,sp->board);
    {
      out << "Side position policyTarget ";
      out << "unreducedNumVisits " << sp->unreducedNumVisits << " ";
      const vector<PolicyTargetMove>& target = sp->policyTarget;
      for(int j = 0; j<target.size(); j++)
        out << moveString(target[j].loc) << " " << target[j].policyTarget << " ";
      out << endl;
    }
    out << "Side position whiteValueTargets ";
    out << sp->whiteValueTargets.win << " ";
    out << sp->whiteValueTargets.loss << " ";
    out << sp->whiteValueTargets.draw << " ";
    out << sp->whiteValueTargets.noResult << " ";
    out << endl;
    {
      out << "Side position whiteQValueTargets ";
      const vector<QValueTargetMove>& target = sp->whiteQValueTargets.targets;
      for(int j = 0; j<target.size(); j++)
        out << moveString(target[j].loc) << " " << target[j].winLoss << " " << target[j].visits << " ";
      out << endl;
    }
  }
}

//-------------------------------------------------------------------------------------


//Don't forget to update everything else in the header file and the code below too if changing any of these
//And update the python code
static const int POLICY_TARGET_NUM_CHANNELS = 2;
static const int GLOBAL_TARGET_NUM_CHANNELS = 80;
static const int VALUE_SPATIAL_TARGET_NUM_CHANNELS = 2;
static const int QVALUE_SPATIAL_TARGET_NUM_CHANNELS = 2;

static int trainingPolicySize(int inputsVersion, int xLen, int yLen) {
  if(inputsVersion == 9) return Xiangqi::NUM_ACTIONS;
  return inputsVersion == 8 ? xLen*yLen : NNPos::getPolicySize(xLen,yLen);
}

TrainingWriteBuffers::TrainingWriteBuffers(int iVersion, int maxRws, int numBChannels, int numFChannels, int xLen, int yLen, bool includeMetadata)
  :inputsVersion(iVersion),
   maxRows(maxRws),
   numBinaryChannels(numBChannels),
   numGlobalChannels(numFChannels),
   dataXLen(xLen),
   dataYLen(yLen),
   packedBoardArea((xLen*yLen + 7)/8),
   hasMetadataInput(includeMetadata),
   curRows(0),
   binaryInputNCHWUnpacked(NULL),
   binaryInputNCHWPacked({maxRws, numBChannels, packedBoardArea}),
   globalInputNC({maxRws, numFChannels}),
   policyTargetsNCMove({maxRws, POLICY_TARGET_NUM_CHANNELS, trainingPolicySize(iVersion,xLen,yLen)}),
   globalTargetsNC({maxRws, GLOBAL_TARGET_NUM_CHANNELS}),
   valueTargetsNCHW({maxRws, VALUE_SPATIAL_TARGET_NUM_CHANNELS, yLen, xLen}),
   qValueTargetsNCMove({maxRws, QVALUE_SPATIAL_TARGET_NUM_CHANNELS, trainingPolicySize(iVersion,xLen,yLen)}),
   metadataInputNC({(includeMetadata ? maxRws : 1), SGFMetadata::METADATA_INPUT_NUM_CHANNELS})
{
  if(iVersion == 9 && includeMetadata)
    throw StringError("Xiangqi training inputs do not support Go metadata");
  binaryInputNCHWUnpacked = new float[numBChannels * xLen * yLen];
}

TrainingWriteBuffers::~TrainingWriteBuffers()
{
  delete[] binaryInputNCHWUnpacked;
}

void TrainingWriteBuffers::clear() {
  curRows = 0;
}

//Copy floats that are all 0-1 into bits, packing 8 to a byte, big-endian-style within each byte.
static void packBits(const float* binaryFloats, int len, uint8_t* bits) {
  for(int i = 0; i < len; i += 8) {
    if(i + 8 <= len) {
      bits[i >> 3] =
        ((uint8_t)binaryFloats[i + 0] << 7) |
        ((uint8_t)binaryFloats[i + 1] << 6) |
        ((uint8_t)binaryFloats[i + 2] << 5) |
        ((uint8_t)binaryFloats[i + 3] << 4) |
        ((uint8_t)binaryFloats[i + 4] << 3) |
        ((uint8_t)binaryFloats[i + 5] << 2) |
        ((uint8_t)binaryFloats[i + 6] << 1) |
        ((uint8_t)binaryFloats[i + 7] << 0);
    }
    else {
      bits[i >> 3] = 0;
      for(int di = 0; i + di < len; di++) {
        bits[i >> 3] |= ((uint8_t)binaryFloats[i + di] << (7-di));
      }
    }
  }
}

static void zeroPolicyTarget(int policySize, int16_t* target) {
  for(int pos = 0; pos<policySize; pos++)
    target[pos] = 0;
}

static void uniformPolicyTarget(int policySize, int16_t* target) {
  for(int pos = 0; pos<policySize; pos++)
    target[pos] = 1;
}

static int trainingMovePos(Loc loc, bool xiangqi, Player perspective, int boardXSize, int dataXLen, int dataYLen) {
  return xiangqi ? Xiangqi::transformAction(Xiangqi::locToAction(loc),perspective,false) :
    NNPos::locToPos(loc,boardXSize,dataXLen,dataYLen);
}

//Copy playouts into target, expanding out the sparse representation into a full plane.
static void fillPolicyTarget(const vector<PolicyTargetMove>& policyTargetMoves, bool xiangqi, Player perspective,
                            int policySize, int dataXLen, int dataYLen, int boardXSize, int16_t* target) {
  zeroPolicyTarget(policySize,target);
  size_t size = policyTargetMoves.size();
  for(size_t i = 0; i<size; i++) {
    const PolicyTargetMove& move = policyTargetMoves[i];
    int pos = trainingMovePos(move.loc,xiangqi,perspective,boardXSize,dataXLen,dataYLen);
    testAssert(pos >= 0 && pos < policySize);
    target[pos] = move.policyTarget;
  }
}

static int16_t clampToRadius32000(float x, Rand& rand) {
  //We need to pack this down to 16 bits, so clamp into an integer [-32000,32000].
  //Randomize to ensure the expectation is exactly correct.
  int low = (int)floor(x);
  int high = low+1;
  if(low < -32000) return -32000;
  if(high > 32000) return 32000;

  float lambda = (float)(x-low);
  if(lambda == 0.0f) return (int16_t)low;
  else return (int16_t)(rand.nextBool(lambda) ? high : low);
}

static void fillQValueTarget(const vector<QValueTargetMove>& whiteQValueTargets, bool xiangqi, Player nextPlayer, int policySize, int dataXLen, int dataYLen, int boardXSize, int16_t* cPosTarget, Rand& rand) {
  for(int i = 0; i < QVALUE_SPATIAL_TARGET_NUM_CHANNELS * policySize; i++) {
    cPosTarget[i] = 0;
  }


  size_t size = whiteQValueTargets.size();
  for(size_t i = 0; i<size; i++) {
    const QValueTargetMove& entry = whiteQValueTargets[i];
    int pos = trainingMovePos(entry.loc,xiangqi,nextPlayer,boardXSize,dataXLen,dataYLen);
    testAssert(pos >= 0 && pos < policySize);

    float winLoss = nextPlayer == P_WHITE ? entry.winLoss : -entry.winLoss;
    cPosTarget[pos] = clampToRadius32000(winLoss*32000.0f,rand);
    cPosTarget[pos+policySize] = (int16_t)(std::max((int64_t)0,std::min(entry.visits,(int64_t)32000)));
  }
}

static void fillValueTDTargets(const vector<ValueTargets>& whiteValueTargetsByTurn, int idx, Player nextPlayer, double nowFactor, float* buf, float* drawBuf) {
  double winValue = 0.0;
  double lossValue = 0.0;
  double drawValue = 0.0;
  double noResultValue = 0.0;

  double weightLeft = 1.0;
  for(int i = idx; i<whiteValueTargetsByTurn.size(); i++) {
    double weightNow;
    if(i == whiteValueTargetsByTurn.size() - 1) {
      weightNow = weightLeft;
      weightLeft = 0.0;
    }
    else {
      weightNow = weightLeft * nowFactor;
      weightLeft *= (1.0 - nowFactor);
    }

    //Training rows need things from the perspective of the player to move, so we flip as appropriate.
    const ValueTargets& targets = whiteValueTargetsByTurn[i];
    winValue += weightNow * (nextPlayer == P_WHITE ? targets.win : targets.loss);
    lossValue += weightNow * (nextPlayer == P_WHITE ? targets.loss : targets.win);
    drawValue += weightNow * targets.draw;
    noResultValue += weightNow * targets.noResult;
  }
  *drawBuf = (float)drawValue;
  buf[0] = (float)winValue;
  buf[1] = (float)lossValue;
  buf[2] = (float)noResultValue;
  buf[3] = 0.0f; // Reserved; score supervision is excluded.
}

void TrainingWriteBuffers::addRow(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const BoardHistory& startHist,
  const BoardHistory& actualGameEndHist,
  int turnIdx,
  float targetWeight,
  int64_t unreducedNumVisits,
  const vector<PolicyTargetMove>* policyTarget0, //can be null
  const vector<PolicyTargetMove>* policyTarget1, //can be null
  double policySurprise,
  double policyEntropy,
  double searchEntropy,
  const vector<ValueTargets>& whiteValueTargets,
  const vector<QValueTargets>& whiteQValueTargets,
  int whiteValueTargetsIdx, //index in whiteValueTargets corresponding to this turn.
  float valueTargetWeight,
  float tdValueTargetWeight,
  const NNRawStats& nnRawStats,
  const Board* finalBoard,
  const vector<Board>* posHistForFutureBoards, //can be null
  bool isSidePosition,
  int numNeuralNetsBehindLatest,
  double drawEquivalentWinsForWhite,
  Player playoutDoublingAdvantagePla,
  double playoutDoublingAdvantage,
  Hash128 gameHash,
  const std::vector<ChangedNeuralNet*>& changedNeuralNets,
  bool hitTurnLimit,
  int numExtraBlack,
  int mode,
  SGFMetadata* sgfMeta,
  Rand& rand,
  const ReanalysisData& reanalysisData
) {
  static_assert(NNModelVersion::latestInputsVersionImplemented == 9, "");
  const bool xiangqi = hist.rules.isXiangqi();
  if(xiangqi != (inputsVersion == 9))
    throw StringError("Training input version does not match Xiangqi rule state");
  if(inputsVersion < 3 || inputsVersion > 9)
    throw StringError("Training write buffers: Does not support input version: " + Global::intToString(inputsVersion));

  int posArea = dataXLen*dataYLen;
  testAssert(curRows < maxRows);

  {
    MiscNNInputParams nnInputParams;
    nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
    //Note: this is coordinated with the fact that selfplay does not use this feature on side positions
    if(!isSidePosition)
      nnInputParams.playoutDoublingAdvantage = getOpp(nextPlayer) == playoutDoublingAdvantagePla ? -playoutDoublingAdvantage : playoutDoublingAdvantage;
    else {
      testAssert(playoutDoublingAdvantagePla == C_EMPTY);
      testAssert(playoutDoublingAdvantage == 0.0);
    }

    bool inputsUseNHWC = false;
    float* rowBin = binaryInputNCHWUnpacked;
    float* rowGlobal = globalInputNC.data + curRows * numGlobalChannels;
    static_assert(NNModelVersion::latestInputsVersionImplemented == 9, "");
    if(inputsVersion == 3) {
      testAssert(NNInputs::NUM_FEATURES_SPATIAL_V3 == numBinaryChannels);
      testAssert(NNInputs::NUM_FEATURES_GLOBAL_V3 == numGlobalChannels);
      NNInputs::fillRowV3(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else if(inputsVersion == 4) {
      testAssert(NNInputs::NUM_FEATURES_SPATIAL_V4 == numBinaryChannels);
      testAssert(NNInputs::NUM_FEATURES_GLOBAL_V4 == numGlobalChannels);
      NNInputs::fillRowV4(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else if(inputsVersion == 5) {
      testAssert(NNInputs::NUM_FEATURES_SPATIAL_V5 == numBinaryChannels);
      testAssert(NNInputs::NUM_FEATURES_GLOBAL_V5 == numGlobalChannels);
      NNInputs::fillRowV5(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else if(inputsVersion == 6) {
      testAssert(NNInputs::NUM_FEATURES_SPATIAL_V6 == numBinaryChannels);
      testAssert(NNInputs::NUM_FEATURES_GLOBAL_V6 == numGlobalChannels);
      NNInputs::fillRowV6(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else if(inputsVersion == 7) {
      testAssert(NNInputs::NUM_FEATURES_SPATIAL_V7 == numBinaryChannels);
      testAssert(NNInputs::NUM_FEATURES_GLOBAL_V7 == numGlobalChannels);
      NNInputs::fillRowV7(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else if(inputsVersion == 8) {
      testAssert(numBinaryChannels == NNInputs::NUM_FEATURES_SPATIAL_V8 && numGlobalChannels == NNInputs::NUM_FEATURES_GLOBAL_V8);
      NNInputs::fillRowV8(board, hist, nextPlayer, nnInputParams, dataXLen, dataYLen, inputsUseNHWC, rowBin, rowGlobal);
    }
    else if(inputsVersion == 9) {
      testAssert(numBinaryChannels == Xiangqi::SPATIAL_FEATURES && numGlobalChannels == Xiangqi::GLOBAL_FEATURES);
      NNInputs::fillRow(inputsVersion,board,hist,nextPlayer,nnInputParams,dataXLen,dataYLen,inputsUseNHWC,rowBin,rowGlobal);
    }
    else
      ASSERT_UNREACHABLE;

    //Pack bools bitwise into uint8_t
    uint8_t* rowBinPacked = binaryInputNCHWPacked.data + curRows * numBinaryChannels * packedBoardArea;
    for(int c = 0; c<numBinaryChannels; c++)
      packBits(rowBin + c * posArea, posArea, rowBinPacked + c * packedBoardArea);
  }

  //Vector for global targets and metadata
  float* rowGlobal = globalTargetsNC.data + curRows * GLOBAL_TARGET_NUM_CHANNELS;

  //Target weight for the whole row
  rowGlobal[25] = targetWeight;

  //Fill policy
  const int policySize = trainingPolicySize(inputsVersion,dataXLen,dataYLen);
  int16_t* rowPolicy = policyTargetsNCMove.data + curRows * POLICY_TARGET_NUM_CHANNELS * policySize;

  if(policyTarget0 != NULL) {
    fillPolicyTarget(*policyTarget0, xiangqi, nextPlayer, policySize, dataXLen, dataYLen, board.x_size, rowPolicy + 0 * policySize);
    rowGlobal[26] = 1.0f;
  }
  else {
    uniformPolicyTarget(policySize, rowPolicy + 0 * policySize);
    rowGlobal[26] = 0.0f;
  }

  if(policyTarget1 != NULL) {
    // Both policy channels are expressed in this row's input orientation.
    fillPolicyTarget(*policyTarget1, xiangqi, nextPlayer, policySize, dataXLen, dataYLen, board.x_size, rowPolicy + 1 * policySize);
    rowGlobal[28] = 1.0f;
  }
  else {
    uniformPolicyTarget(policySize, rowPolicy + 1 * policySize);
    rowGlobal[28] = 0.0f;
  }

  //Fill td-like value targets
  int boardArea = board.x_size * board.y_size;
  testAssert(whiteValueTargetsIdx >= 0 && whiteValueTargetsIdx < whiteValueTargets.size());
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 0.0, rowGlobal, rowGlobal+70);
  //These three constants used to be 'nicer' numbers 0.18, 0.06, 0.02, but we screwed up the functional form
  //by omitting the "1.0 +" at the front (breaks scaling to small board sizes), so when we fixed this we also
  //decreased the other numbers slightly to try to maximally limit the impact of the fix on the numerical values
  //on the actual board sizes 9-19, since it would be costly to retest.
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0/(1.0 + boardArea * 0.176), rowGlobal+4, rowGlobal+71);
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0/(1.0 + boardArea * 0.056), rowGlobal+8, rowGlobal+72);
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0/(1.0 + boardArea * 0.016), rowGlobal+12, rowGlobal+73);
  fillValueTDTargets(whiteValueTargets, whiteValueTargetsIdx, nextPlayer, 1.0, rowGlobal+16, rowGlobal+74);

  rowGlobal[21] = 0.0f;
  rowGlobal[29] = 0.0f;

  //Expected time of arrival of winloss variance, in turns
  {
    double sum = 0.0;
    for(int i = whiteValueTargetsIdx+1; i<whiteValueTargets.size(); i++) {
      int turnsFromNow = i-whiteValueTargetsIdx;
      const ValueTargets& prevTargets = whiteValueTargets[i-1];
      const ValueTargets& targets = whiteValueTargets[i];
      double prevWL = prevTargets.win - prevTargets.loss;
      double nextWL = targets.win - targets.loss;
      double variance = (nextWL - prevWL) * (nextWL - prevWL);
      sum += turnsFromNow * variance;
    }
    rowGlobal[22] = (float)sum;
  }

  //Unused
  rowGlobal[23] = 0.0f;
  rowGlobal[24] = (float)(1.0f - tdValueTargetWeight);
  rowGlobal[30] = (float)policySurprise;
  rowGlobal[31] = (float)policyEntropy;
  rowGlobal[32] = (float)searchEntropy;
  // Value weight
  rowGlobal[35] = (float)(1.0f - valueTargetWeight);

  //Fill in whether we should use history or not
  bool useHist0 = rand.nextDouble() < 0.98;
  bool useHist1 = useHist0 && rand.nextDouble() < 0.98;
  bool useHist2 = useHist1 && rand.nextDouble() < 0.98;
  bool useHist3 = useHist2 && rand.nextDouble() < 0.98;
  bool useHist4 = useHist3 && rand.nextDouble() < 0.98;
  rowGlobal[36] = useHist0 ? 1.0f : 0.0f;
  rowGlobal[37] = useHist1 ? 1.0f : 0.0f;
  rowGlobal[38] = useHist2 ? 1.0f : 0.0f;
  rowGlobal[39] = useHist3 ? 1.0f : 0.0f;
  rowGlobal[40] = useHist4 ? 1.0f : 0.0f;

  //Fill in hash of game
  rowGlobal[41] = (float)(gameHash.hash0 & 0x3FFFFF);
  rowGlobal[42] = (float)((gameHash.hash0 >> 22) & 0x3FFFFF);
  rowGlobal[43] = (float)((gameHash.hash0 >> 44) & 0xFFFFF);
  rowGlobal[44] = (float)(gameHash.hash1 & 0x3FFFFF);
  rowGlobal[45] = (float)((gameHash.hash1 >> 22) & 0x3FFFFF);
  rowGlobal[46] = (float)((gameHash.hash1 >> 44) & 0xFFFFF);

  //Various other data
  rowGlobal[47] = hist.currentSelfKomi(nextPlayer,drawEquivalentWinsForWhite);
  rowGlobal[48] = (hist.encorePhase == 2 || hist.rules.scoringRule == Rules::SCORING_AREA) ? 1.0f : 0.0f;

  //Earlier neural net metadata
  rowGlobal[49] = changedNeuralNets.size() > 0 ? 1.0f : 0.0f;
  rowGlobal[50] = (float)numNeuralNetsBehindLatest;

  //Some misc metadata
  rowGlobal[51] = (float)turnIdx;
  rowGlobal[52] = hitTurnLimit ? 1.0f : 0.0f;
  rowGlobal[53] = (float)startHist.moveHistory.size();
  rowGlobal[54] = (float)numExtraBlack;

  //Metadata about how the game was initialized
  rowGlobal[55] = (float)mode;
  rowGlobal[56] = (float)hist.initialTurnNumber;

  //Some stats
  rowGlobal[57] = (float)(nextPlayer == P_WHITE ? nnRawStats.whiteWinLoss : -nnRawStats.whiteWinLoss);
  rowGlobal[58] = 0.0f;
  rowGlobal[59] = (float)nnRawStats.policyEntropy;

  //Original number of visits
  rowGlobal[60] = (float)unreducedNumVisits;

  //Bonus points
  if(!isSidePosition) {
    //Possibly this should count whiteHandicapBonusScore too, but in selfplay this never changes
    //after the start of a game
    float whiteBonusPoints = actualGameEndHist.whiteBonusScore - hist.whiteBonusScore;
    float selfBonusPoints = (nextPlayer == P_WHITE ? whiteBonusPoints : -whiteBonusPoints);
    //Note: we have a lot of data where this isn't reliable for side positions
    rowGlobal[61] = selfBonusPoints != 0 ? selfBonusPoints : 0.0f; //Conditional avoids negative zero
  }
  else {
    rowGlobal[61] = 0.0f;
  }

  //Game finished
  rowGlobal[62] = (!isSidePosition && actualGameEndHist.isGameFinished && !hitTurnLimit) ? 1.0f : 0.0f;

  //Version
  rowGlobal[63] = 3.0f;

  //Reanalysis info - whether this position originally got only a cheap search and was redone with a full
  //search after the game, and if so, the original cheap search's surprise stats that drove its selection
  //and its visit count.
  rowGlobal[64] = reanalysisData.wasReanalyzed ? 1.0f : 0.0f;
  rowGlobal[65] = reanalysisData.wasReanalyzed ? reanalysisData.selectionPolicySurprise : 0.0f;
  rowGlobal[66] = reanalysisData.wasReanalyzed ? reanalysisData.selectionValueSurprise : 0.0f;
  rowGlobal[67] = reanalysisData.wasReanalyzed ? (float)reanalysisData.originalNumVisits : 0.0f;

  //Whether pass-alive areas for this game's adjudication and featurization were being computed as if
  //multi-stone suicide were always legal regardless of the actual suicide rule.
  rowGlobal[68] = hist.modes.alwaysComputePassAliveUnderSuicideRules ? 1.0f : 0.0f;

  //Whether territory scoring with TaxRule NONE for this game's adjudication and featurization was
  //excluding empty points adjacent to chains in atari (rules version 3).
  rowGlobal[69] = hist.modes.excludeTerritoryAdjacentToAtari ? 1.0f : 0.0f;

  // C75/76 mask unavailable non-Go future horizons (0 means available).
  for(int i = 75; i<80; i++)
    rowGlobal[i] = 0.0f;

  testAssert(80 == GLOBAL_TARGET_NUM_CHANNELS);

  int8_t* rowFuturePositions = valueTargetsNCHW.data + curRows * VALUE_SPATIAL_TARGET_NUM_CHANNELS * posArea;
  rowGlobal[20] = 0.0f;
  // Retain the source mask for outcome-derived auxiliary targets.
  rowGlobal[27] = (finalBoard == NULL || (actualGameEndHist.isGameFinished && actualGameEndHist.isNoResult))
    ? 0.0f : valueTargetWeight;

  if(posHistForFutureBoards == NULL) {
    rowGlobal[33] = 0.0f;
    for(int i = 0; i<posArea; i++) {
      rowFuturePositions[i] = 0;
      rowFuturePositions[i+posArea] = 0;
    }
  }
  else {
    const vector<Board>& boards = *posHistForFutureBoards;
    testAssert(boards.size() == whiteValueTargets.size());
    testAssert(boards.size() > 0);

    // Future position weight
    rowGlobal[33] = 1.0f;
    int endIdx = (int)boards.size()-1;
    if(!hist.rules.isGo() && hitTurnLimit) {
      rowGlobal[75] = whiteValueTargetsIdx+8 > endIdx ? 1.0f : 0.0f;
      rowGlobal[76] = whiteValueTargetsIdx+32 > endIdx ? 1.0f : 0.0f;
    }
    const Board& board2 = boards[std::min(whiteValueTargetsIdx+8,endIdx)];
    const Board& board3 = boards[std::min(whiteValueTargetsIdx+32,endIdx)];
    testAssert(board2.y_size == board.y_size && board2.x_size == board.x_size);
    testAssert(board3.y_size == board.y_size && board3.x_size == board.x_size);

    for(int i = 0; i<posArea; i++) {
      rowFuturePositions[i] = 0;
      rowFuturePositions[i+posArea] = 0;
    }
    Player pla = nextPlayer;
    Player opp = getOpp(nextPlayer);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        const bool rotate = xiangqi && nextPlayer == P_WHITE;
        int pos = NNPos::xyToPos(rotate ? Xiangqi::WIDTH-1-x : x,rotate ? Xiangqi::HEIGHT-1-y : y,dataXLen);
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board2.colors[loc] == pla) rowFuturePositions[pos] = 1;
        else if(board2.colors[loc] == opp) rowFuturePositions[pos] = -1;
        if(board3.colors[loc] == pla) rowFuturePositions[pos+posArea] = 1;
        else if(board3.colors[loc] == opp) rowFuturePositions[pos+posArea] = -1;
      }
    }
  }


  rowGlobal[34] = 0.0f; // Reserved; per-point scoring supervision is excluded.

  //Q values
  {
    testAssert(whiteValueTargetsIdx < whiteQValueTargets.size());
    int16_t* rowQValues = qValueTargetsNCMove.data + curRows * QVALUE_SPATIAL_TARGET_NUM_CHANNELS * policySize;
    fillQValueTarget(whiteQValueTargets[whiteValueTargetsIdx].targets, xiangqi, nextPlayer, policySize, dataXLen, dataYLen, board.x_size, rowQValues, rand);
  }

  if(hasMetadataInput) {
    testAssert(sgfMeta != NULL);
    float* rowMetadata = metadataInputNC.data + curRows * SGFMetadata::METADATA_INPUT_NUM_CHANNELS;
    SGFMetadata::fillMetadataRow(sgfMeta, rowMetadata, nextPlayer, board.x_size * board.y_size);
  }

  curRows++;
}

void TrainingWriteBuffers::writeToZipFile(const string& fileName, MuZeroWriteBuffers* muZero) {
  ZipFile zipFile(fileName);

  uint64_t numBytes;

  numBytes = binaryInputNCHWPacked.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("binaryInputNCHWPacked", binaryInputNCHWPacked.dataIncludingHeader, numBytes);

  numBytes = globalInputNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("globalInputNC", globalInputNC.dataIncludingHeader, numBytes);

  numBytes = policyTargetsNCMove.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("policyTargetsNCMove", policyTargetsNCMove.dataIncludingHeader, numBytes);

  numBytes = globalTargetsNC.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("globalTargetsNC", globalTargetsNC.dataIncludingHeader, numBytes);


  numBytes = valueTargetsNCHW.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("valueTargetsNCHW", valueTargetsNCHW.dataIncludingHeader, numBytes);

  numBytes = qValueTargetsNCMove.prepareHeaderWithNumRows(curRows);
  zipFile.writeBuffer("qValueTargetsNCMove", qValueTargetsNCMove.dataIncludingHeader, numBytes);

  if(hasMetadataInput) {
    numBytes = metadataInputNC.prepareHeaderWithNumRows(curRows);
    zipFile.writeBuffer("metadataInputNC", metadataInputNC.dataIncludingHeader, numBytes);
  }

  if(muZero != nullptr)
    muZero->writeToZipFile(zipFile,curRows,dataXLen,dataYLen,(int)policyTargetsNCMove.shape[2]);

  zipFile.close();
}

void TrainingWriteBuffers::writeToTextOstream(ostream& out) {
  int64_t len;

  auto printHeader = [&out](const char* dataIncludingHeader) {
    //In actuality our headers aren't that long, so we cut it off at half the total header bytes
    for(int i = 0; i<10; i++)
      out << (int)dataIncludingHeader[i] << " ";
    for(int i = 10; i<NumpyBuffer<int>::TOTAL_HEADER_BYTES/2; i++)
      out << dataIncludingHeader[i];
    out << endl;
  };

  out << "binaryInputNCHWPacked" << endl;
  binaryInputNCHWPacked.prepareHeaderWithNumRows(curRows);
  char buf[32];
  printHeader((const char*)binaryInputNCHWPacked.dataIncludingHeader);
  len = binaryInputNCHWPacked.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    sprintf(buf,"%02X",binaryInputNCHWPacked.data[i]);
    out << buf;
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "globalInputNC" << endl;
  globalInputNC.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)globalInputNC.dataIncludingHeader);
  len = globalInputNC.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << globalInputNC.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "policyTargetsNCMove" << endl;
  policyTargetsNCMove.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)policyTargetsNCMove.dataIncludingHeader);
  len = policyTargetsNCMove.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << policyTargetsNCMove.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "globalTargetsNC" << endl;
  globalTargetsNC.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)globalTargetsNC.dataIncludingHeader);
  len = globalTargetsNC.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << globalTargetsNC.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "valueTargetsNCHW" << endl;
  valueTargetsNCHW.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)valueTargetsNCHW.dataIncludingHeader);
  len = valueTargetsNCHW.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << (int)valueTargetsNCHW.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  out << "qValueTargetsNCMove" << endl;
  qValueTargetsNCMove.prepareHeaderWithNumRows(curRows);
  printHeader((const char*)qValueTargetsNCMove.dataIncludingHeader);
  len = qValueTargetsNCMove.getActualDataLen(curRows);
  for(int i = 0; i<len; i++) {
    out << (int)qValueTargetsNCMove.data[i] << " ";
    if((i+1) % (len/curRows) == 0) out << endl;
  }
  out << endl;

  if(hasMetadataInput) {
    out << "metadataInputNC" << endl;
    metadataInputNC.prepareHeaderWithNumRows(curRows);
    printHeader((const char*)metadataInputNC.dataIncludingHeader);
    len = metadataInputNC.getActualDataLen(curRows);
    for(int i = 0; i<len; i++) {
      out << (int)metadataInputNC.data[i] << " ";
      if((i+1) % (len/curRows) == 0) out << endl;
    }
    out << endl;
  }
}

//-------------------------------------------------------------------------------------

TrainingDataWriter::TrainingDataWriter(const string& outDir, int iVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, const string& randSeed, bool writeMuZeroTrajectories)
  : TrainingDataWriter(outDir,NULL,iVersion,maxRowsPerFile,firstFileMinRandProp,dataXLen,dataYLen,1,randSeed,writeMuZeroTrajectories)
{}
TrainingDataWriter::TrainingDataWriter(ostream* dbgOut, int iVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, int onlyEvery, const string& randSeed)
  : TrainingDataWriter(string(),dbgOut,iVersion,maxRowsPerFile,firstFileMinRandProp,dataXLen,dataYLen,onlyEvery,randSeed)
{}

TrainingDataWriter::TrainingDataWriter(const string& outDir, ostream* dbgOut, int iVersion, int maxRowsPerFile, double firstFileMinRandProp, int dataXLen, int dataYLen, int onlyEvery, const string& randSeed, bool writeMuZeroTrajectories)
  :outputDir(outDir),inputsVersion(iVersion),rand(randSeed),writeBuffers(NULL),debugOut(dbgOut),debugOnlyWriteEvery(onlyEvery),rowCount(0)
{
  if(writeMuZeroTrajectories) {
    if(debugOut != nullptr)
      throw StringError("MuZero trajectories require NPZ output, not the row-only text format");
    muZeroRand.reset(new Rand(randSeed + ":muzero-trajectories"));
    muZeroBuffers.reset(new MuZeroWriteBuffers());
  }
  int numBinaryChannels;
  int numGlobalChannels;
  //Note that this inputsVersion is for data writing, it might be different than the inputsVersion used
  //to feed into a model during selfplay
  static_assert(NNModelVersion::latestInputsVersionImplemented == 9, "");
  if(inputsVersion == 3) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V3;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V3;
  }
  else if(inputsVersion == 4) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V4;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V4;
  }
  else if(inputsVersion == 5) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V5;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V5;
  }
  else if(inputsVersion == 6) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V6;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V6;
  }
  else if(inputsVersion == 7) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V7;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V7;
  }
  else if(inputsVersion == 8) {
    numBinaryChannels = NNInputs::NUM_FEATURES_SPATIAL_V8;
    numGlobalChannels = NNInputs::NUM_FEATURES_GLOBAL_V8;
  }
  else if(inputsVersion == 9) {
    numBinaryChannels = Xiangqi::SPATIAL_FEATURES;
    numGlobalChannels = Xiangqi::GLOBAL_FEATURES;
  }
  else {
    throw StringError("TrainingDataWriter: Unsupported inputs version: " + Global::intToString(inputsVersion));
  }

  const bool hasMetadataInput = false;
  writeBuffers = new TrainingWriteBuffers(
    inputsVersion,
    maxRowsPerFile,
    numBinaryChannels,
    numGlobalChannels,
    dataXLen,
    dataYLen,
    hasMetadataInput
  );

  if(firstFileMinRandProp < 0 || firstFileMinRandProp > 1)
    throw StringError("TrainingDataWriter: firstFileMinRandProp not in [0,1]: " + Global::doubleToString(firstFileMinRandProp));
  isFirstFile = true;
  if(firstFileMinRandProp >= 1.0)
    firstFileMaxRows = maxRowsPerFile;
  else
    firstFileMaxRows = maxRowsPerFile - (int)(maxRowsPerFile * (1.0-firstFileMinRandProp) * rand.nextDouble());
}



TrainingDataWriter::~TrainingDataWriter()
{
  delete writeBuffers;
}

bool TrainingDataWriter::isEmpty() const {
  return writeBuffers->curRows <= 0;
}
int64_t TrainingDataWriter::numRowsInBuffer() const {
  return writeBuffers->curRows;
}
int64_t TrainingDataWriter::numRowsWritten() const {
  return rowCount.load(std::memory_order_relaxed);
}

void TrainingDataWriter::writeAndClearIfFull() {
  if(writeBuffers->curRows >= writeBuffers->maxRows || (isFirstFile && writeBuffers->curRows >= firstFileMaxRows)) {
    flushIfNonempty();
  }
}



void TrainingDataWriter::flushIfNonempty() {
  string resultingFilename;
  flushIfNonempty(resultingFilename);
}

bool TrainingDataWriter::flushIfNonempty(string& resultingFilename) {
  if(writeBuffers->curRows <= 0)
    return false;

  isFirstFile = false;

  if(debugOut != NULL) {
    writeBuffers->writeToTextOstream(*debugOut);
    writeBuffers->clear();
    resultingFilename = "";
  }
  else {
    resultingFilename = outputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".npz";
    string tmpFilename = resultingFilename + ".tmp";
    writeBuffers->writeToZipFile(tmpFilename,muZeroBuffers.get());
    writeBuffers->clear();
    if(muZeroBuffers) muZeroBuffers->clear();
    FileUtils::rename(tmpFilename,resultingFilename);
  }
  return true;
}

void TrainingDataWriter::writeGame(const FinishedGameData& data) {
  int numMoves = (int)(data.endHist.moveHistory.size() - data.startHist.moveHistory.size());
  testAssert(numMoves >= 0);
  testAssert(data.startHist.moveHistory.size() <= data.endHist.moveHistory.size());
  testAssert(data.endHist.moveHistory.size() <= 100000000);
  testAssert(data.targetWeightByTurn.size() == numMoves);
  testAssert(data.targetWeightByTurnUnrounded.size() == numMoves);
  testAssert(data.policyTargetsByTurn.size() == numMoves);
  testAssert(data.policySurpriseByTurn.size() == numMoves);
  testAssert(data.policyEntropyByTurn.size() == numMoves);
  testAssert(data.searchEntropyByTurn.size() == numMoves);
  testAssert(data.whiteValueTargetsByTurn.size() == numMoves+1);
  testAssert(data.whiteQValueTargetsByTurn.size() == numMoves);
  testAssert(data.nnRawStatsByTurn.size() == numMoves);
  testAssert(data.reanalysisByTurn.size() == numMoves || data.reanalysisByTurn.size() == 0);

  //Some sanity checks
  {
    const ValueTargets& lastTargets = data.whiteValueTargetsByTurn[data.whiteValueTargetsByTurn.size()-1];
    if(!data.endHist.isGameFinished)
      testAssert(data.hitTurnLimit);
    else if(data.endHist.isNoResult)
      testAssert(lastTargets.win == 0.0f && lastTargets.loss == 0.0f && lastTargets.noResult == 1.0f);
    else if(data.endHist.winner == P_BLACK)
      testAssert(lastTargets.win == 0.0f && lastTargets.loss == 1.0f && lastTargets.noResult == 0.0f);
    else if(data.endHist.winner == P_WHITE)
      testAssert(lastTargets.win == 1.0f && lastTargets.loss == 0.0f && lastTargets.noResult == 0.0f);
    else
      testAssert(lastTargets.win == 0.0f && lastTargets.loss == 0.0f && lastTargets.draw == 1.0f && lastTargets.noResult == 0.0f);

    testAssert(!data.endHist.isResignation);
  }

  //Play out all the moves in a single pass first to compute all the future board states
  vector<Board> posHistForFutureBoards;
  {
    Board board(data.startBoard);
    BoardHistory hist(data.startHist);
    Player nextPlayer = data.startPla;
    posHistForFutureBoards.push_back(board);

    int startTurnIdx = (int)data.startHist.moveHistory.size();
    for(int turnAfterStart = 0; turnAfterStart<numMoves; turnAfterStart++) {
      int turnIdx = turnAfterStart + startTurnIdx;

      Move move = data.endHist.moveHistory[turnIdx];
      testAssert(move.pla == nextPlayer);
      testAssert(hist.isLegal(board,move.loc,move.pla));
      hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
      nextPlayer = getOpp(nextPlayer);

      posHistForFutureBoards.push_back(board);
    }
  }

  testAssert(data.hasFullData);

  std::shared_ptr<MuZeroTrajectory> trajectory;
  std::unique_ptr<TrainingWriteBuffers> trajectoryRow;
  auto writeMainRows = [&](bool fullTrajectory) {
    TrainingWriteBuffers* buffers = fullTrajectory ? trajectoryRow.get() : writeBuffers;
    Rand& rowRand = fullTrajectory ? *muZeroRand : rand;
    Board board(data.startBoard);
    BoardHistory hist(data.startHist);
    Player nextPlayer = data.startPla;

    //Write main game rows
    int startTurnIdx = (int)data.startHist.moveHistory.size();
    for(int turnAfterStart = 0; turnAfterStart<numMoves; turnAfterStart++) {
      double targetWeight = fullTrajectory ? 1.0 : data.targetWeightByTurn[turnAfterStart];
      int turnIdx = turnAfterStart + startTurnIdx;

      int64_t unreducedNumVisits = data.policyTargetsByTurn[turnAfterStart].unreducedNumVisits;
      const vector<PolicyTargetMove>* policyTarget0 = data.policyTargetsByTurn[turnAfterStart].policyTargets;
      const vector<PolicyTargetMove>* policyTarget1 = (turnAfterStart + 1 < numMoves) ? data.policyTargetsByTurn[turnAfterStart+1].policyTargets : NULL;
      bool isSidePosition = false;
      float valueTargetWeight = !data.endHist.rules.isGo() && data.hitTurnLimit ? 0.0f : 1.0f;
      float tdValueTargetWeight = 1.0f;

      ReanalysisData reanalysisData;
      if(turnAfterStart < data.reanalysisByTurn.size())
        reanalysisData = data.reanalysisByTurn[turnAfterStart];

      int numNeuralNetsBehindLatest = 0;
      if(reanalysisData.wasReanalyzed) {
        //Reanalysis searches happen after the game ends, so they use the net as of their creation rather than the net at this turn.
        numNeuralNetsBehindLatest = (int)data.changedNeuralNets.size() - reanalysisData.numNeuralNetChangesSoFar;
      }
      else {
        for(int i = 0; i<data.changedNeuralNets.size(); i++) {
          if(data.changedNeuralNets[i]->turnIdx > turnIdx) {
            numNeuralNetsBehindLatest = (int)data.changedNeuralNets.size()-i;
            break;
          }
        }
      }

      //If this position was reanalyzed and we're not using outcome targets for reanalyzed positions, then skip
      //the targets derived from the final board and the actual game continuation (    //future board positions, and the next-move policy target, which comes from the next turn's search along
      //the continuation the reanalysis search may not prefer). The value targets are still computed from the
      //game's value target array as normal - the entry at this turn is the reanalysis search's values, and
      //later turns and past positions read that array too.
      bool skipOutcomeDerivedTargets = reanalysisData.wasReanalyzed && !reanalysisData.usedOutcomeTargets;

      while(targetWeight > 0.0) {
        if(targetWeight >= 1.0 || rand.nextBool(targetWeight)) {
          if(fullTrajectory || debugOut == NULL || rowCount.load(std::memory_order_relaxed) % debugOnlyWriteEvery == 0) {
            buffers->addRow(
              board,hist,nextPlayer,
              data.startHist,
              data.endHist,
              turnIdx,
              (float)data.trainingWeight,
              unreducedNumVisits,
              policyTarget0,
              skipOutcomeDerivedTargets ? NULL : policyTarget1,
              data.policySurpriseByTurn[turnAfterStart],
              data.policyEntropyByTurn[turnAfterStart],
              data.searchEntropyByTurn[turnAfterStart],
              data.whiteValueTargetsByTurn,
              data.whiteQValueTargetsByTurn,
              turnAfterStart,
              valueTargetWeight,
              tdValueTargetWeight,
              data.nnRawStatsByTurn[turnAfterStart],
              skipOutcomeDerivedTargets ? NULL : &(data.endHist.getRecentBoard(0)),
              skipOutcomeDerivedTargets ? NULL : &posHistForFutureBoards,
              isSidePosition,
              numNeuralNetsBehindLatest,
              data.drawEquivalentWinsForWhite,
              data.playoutDoublingAdvantagePla,
              data.playoutDoublingAdvantage,
              data.gameHash,
              data.changedNeuralNets,
              data.hitTurnLimit,
              data.numExtraBlack,
              data.mode,
              NULL,
              rowRand,
              reanalysisData
            );
            if(fullTrajectory) {
              const Move& move = data.endHist.moveHistory[turnIdx];
              trajectory->append(*buffers,trainingMovePos(move.loc,hist.rules.isXiangqi(),nextPlayer,
                                 board.x_size,buffers->dataXLen,buffers->dataYLen),
                                 data.targetWeightByTurnUnrounded[turnAfterStart]);
              buffers->clear();
            }
            else {
              if(muZeroBuffers) muZeroBuffers->addRoot(trajectory,turnAfterStart);
              writeAndClearIfFull();
            }
          }
          if(!fullTrajectory) rowCount.fetch_add(1, std::memory_order_relaxed);
        }
        targetWeight -= 1.0;
      }


      Move move = data.endHist.moveHistory[turnIdx];
      testAssert(move.pla == nextPlayer);
      testAssert(hist.isLegal(board,move.loc,move.pla));
      hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
      nextPlayer = getOpp(nextPlayer);
    }
  };

  if(muZeroBuffers) {
    trajectory = std::make_shared<MuZeroTrajectory>();
    trajectory->gameHash = data.gameHash;
    trajectory->finished = data.endHist.isGameFinished && !data.hitTurnLimit;
    trajectoryRow.reset(new TrainingWriteBuffers(
      inputsVersion,1,writeBuffers->numBinaryChannels,writeBuffers->numGlobalChannels,
      writeBuffers->dataXLen,writeBuffers->dataYLen,false
    ));
    // Complete the trajectory before any sampled root can flush a shard.
    writeMainRows(true);

    // Final observed state: an actual terminal state may be extended as an
    // absorbing target. A move-limit endpoint only supplies its TD bootstrap;
    // it does not invent a terminal result or a continuation beyond observation.
    const Board& finalBoard = posHistForFutureBoards.back();
    const Player finalPla = numMoves % 2 == 0 ? data.startPla : getOpp(data.startPla);
    const vector<ValueTargets> finalValues{data.whiteValueTargetsByTurn.back()};
    const vector<QValueTargets> finalQ(1);
    const vector<Board> finalBoards{finalBoard};
    NNRawStats finalRaw;
    finalRaw.whiteWinLoss = 0.0;
    finalRaw.policyEntropy = 0.0;
    trajectoryRow->addRow(
      finalBoard,data.endHist,finalPla,data.startHist,data.endHist,(int)data.endHist.moveHistory.size(),
      (float)data.trainingWeight,0,NULL,NULL,0.0,0.0,0.0,finalValues,finalQ,0,
      trajectory->finished ? 1.0f : 0.0f,1.0f,finalRaw,
      trajectory->finished ? &finalBoard : NULL,trajectory->finished ? &finalBoards : NULL,
      false,0,data.drawEquivalentWinsForWhite,data.playoutDoublingAdvantagePla,data.playoutDoublingAdvantage,
      data.gameHash,data.changedNeuralNets,data.hitTurnLimit,data.numExtraBlack,data.mode,NULL,*muZeroRand
    );
    trajectory->append(*trajectoryRow,-1,1.0f);
  }
  writeMainRows(false);

  //Write side rows
  vector<ValueTargets> whiteValueTargetsBuf(1);
  vector<QValueTargets> whiteQValueTargetsBuf(1);
  for(int i = 0; i<data.sidePositions.size(); i++) {
    SidePosition* sp = data.sidePositions[i];

    double targetWeight = sp->targetWeight;
    while(targetWeight > 0.0) {
      if(targetWeight >= 1.0 || rand.nextBool(targetWeight)) {
        if(debugOut == NULL || rowCount.load(std::memory_order_relaxed) % debugOnlyWriteEvery == 0) {

          int turnIdx = (int)sp->hist.moveHistory.size();
          testAssert(turnIdx >= data.startHist.moveHistory.size());
          whiteValueTargetsBuf[0] = sp->whiteValueTargets;
          whiteQValueTargetsBuf[0] = sp->whiteQValueTargets;
          bool isSidePosition = true;
          int numNeuralNetsBehindLatest = (int)data.changedNeuralNets.size() - sp->numNeuralNetChangesSoFar;
          float valueTargetWeight = !data.endHist.rules.isGo() && data.hitTurnLimit ? 0.0f : 1.0f;
          float tdValueTargetWeight = 1.0f;

          writeBuffers->addRow(
            sp->board,sp->hist,sp->pla,
            data.startHist,
            data.endHist, // actual game ending hist, even for side position
            turnIdx,
            (float)data.trainingWeight,
            sp->unreducedNumVisits,
            &(sp->policyTarget),
            NULL,
            sp->policySurprise,
            sp->policyEntropy,
            sp->searchEntropy,
            whiteValueTargetsBuf,
            whiteQValueTargetsBuf,
            0,
            valueTargetWeight,
            tdValueTargetWeight,
            sp->nnRawStats,
            NULL,
            NULL,
            isSidePosition,
            numNeuralNetsBehindLatest,
            data.drawEquivalentWinsForWhite,
            sp->playoutDoublingAdvantagePla,
            sp->playoutDoublingAdvantage,
            data.gameHash,
            data.changedNeuralNets,
            data.hitTurnLimit, // actual game hit turn limit
            data.numExtraBlack,
            data.mode,
            NULL,
            rand
          );
          if(muZeroBuffers) muZeroBuffers->addRoot(trajectory,-1);
          writeAndClearIfFull();
        }
        rowCount.fetch_add(1, std::memory_order_relaxed);
      }
      targetWeight -= 1.0;
    }

  }

}
