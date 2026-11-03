#include "../search/search.h"
#include "policybuffer.h"

#include "../core/fancymath.h"
#include "../core/test.h"
#include "../search/searchnode.h"
#include "../search/patternbonustable.h"
#include "../game/xiangqi.h"

//------------------------
#include "../core/using.h"
//------------------------

uint32_t Search::chooseIndexWithTemperature(
  Rand& rand,
  const double* relativeProbs,
  int numRelativeProbs,
  double temperature,
  double onlyBelowProb,
  double* processedRelProbsBuf
) {
  testAssert(numRelativeProbs > 0);
  SearchPolicyBuffer<double,Board::MAX_ARR_SIZE> processedRelProbsStorage(processedRelProbsBuf == NULL ? numRelativeProbs : 0);
  double* processedRelProbs = processedRelProbsStorage.data();
  if(processedRelProbsBuf == NULL)
    processedRelProbsBuf = &processedRelProbs[0];

  double maxRelProb = 0.0;
  double sumRelProb = 0.0;
  for(int i = 0; i<numRelativeProbs; i++) {
    sumRelProb += std::max(0.0, relativeProbs[i]);
    if(relativeProbs[i] > maxRelProb)
      maxRelProb = relativeProbs[i];
  }
  testAssert(maxRelProb > 0.0);
  testAssert(sumRelProb > 0.0);

  //Temperature so close to 0 that we just calculate the max directly
  if(temperature <= 1.0e-4 && onlyBelowProb >= 1.0) {
    double bestProb = relativeProbs[0];
    int bestIdx = 0;
    processedRelProbsBuf[0] = 0;
    for(int i = 1; i<numRelativeProbs; i++) {
      processedRelProbsBuf[i] = 0;
      if(relativeProbs[i] > bestProb) {
        bestProb = relativeProbs[i];
        bestIdx = i;
      }
    }
    processedRelProbsBuf[bestIdx] = 1.0;
    return bestIdx;
  }
  //Actual temperature
  else {
    double logMaxRelProb = log(maxRelProb);
    double logSumRelProb = log(sumRelProb);
    double logOnlyBelowProb = log(std::max(1e-50,onlyBelowProb));
    double sum = 0.0;
    for(int i = 0; i<numRelativeProbs; i++) {
      if(relativeProbs[i] <= 0.0)
        processedRelProbsBuf[i] = 0.0;
      else {
        double logRelProb = log(relativeProbs[i]) - logMaxRelProb;
        double logRelProbThreshold = std::min(0.0, logOnlyBelowProb + logSumRelProb - logMaxRelProb);
        double newLogRelProb;
        if(logRelProb > logRelProbThreshold)
          newLogRelProb = logRelProb;
        else
          newLogRelProb = (logRelProb - logRelProbThreshold) / temperature + logRelProbThreshold;
        processedRelProbsBuf[i] = exp(newLogRelProb);
      }
      sum += processedRelProbsBuf[i];
    }
    testAssert(sum > 0.0);
    uint32_t idxChosen = rand.nextUInt(processedRelProbsBuf,numRelativeProbs);
    return idxChosen;
  }
}

void Search::computeDirichletAlphaDistribution(int policySize, const float* policyProbs, double* alphaDistr) {
  int legalCount = 0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0)
      legalCount += 1;
  }

  if(legalCount <= 0)
    throw StringError("computeDirichletAlphaDistribution: No move with nonnegative policy value - can't even pass?");

  //We're going to generate a gamma draw on each move with alphas that sum up to searchParams.rootDirichletNoiseTotalConcentration.
  //Half of the alpha weight are uniform.
  //The other half are shaped based on the log of the existing policy.
  double logPolicySum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      alphaDistr[i] = log(std::min(0.01, (double)policyProbs[i]) + 1e-20);
      logPolicySum += alphaDistr[i];
    }
  }
  double logPolicyMean = logPolicySum / legalCount;
  double alphaPropSum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      alphaDistr[i] = std::max(0.0, alphaDistr[i] - logPolicyMean);
      alphaPropSum += alphaDistr[i];
    }
  }
  double uniformProb = 1.0 / legalCount;
  if(alphaPropSum <= 0.0) {
    for(int i = 0; i<policySize; i++) {
      if(policyProbs[i] >= 0)
        alphaDistr[i] = uniformProb;
    }
  }
  else {
    for(int i = 0; i<policySize; i++) {
      if(policyProbs[i] >= 0)
        alphaDistr[i] = 0.5 * (alphaDistr[i] / alphaPropSum + uniformProb);
    }
  }
}

void Search::addDirichletNoise(const SearchParams& searchParams, Rand& rand, int policySize, float* policyProbs) {
  SearchPolicyBuffer<double> rStorage(policySize);
  double* r = rStorage.data();
  Search::computeDirichletAlphaDistribution(policySize, policyProbs, r);

  //r now contains the proportions with which we would like to split the alpha
  //The total of the alphas is searchParams.rootDirichletNoiseTotalConcentration
  //Generate gamma draw on each move
  double rSum = 0.0;
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      r[i] = rand.nextGamma(r[i] * searchParams.rootDirichletNoiseTotalConcentration);
      rSum += r[i];
    }
    else
      r[i] = 0.0;
  }

  //Normalized gamma draws -> dirichlet noise
  for(int i = 0; i<policySize; i++)
    r[i] /= rSum;

  //At this point, r[i] contains a dirichlet distribution draw, so add it into the nnOutput.
  for(int i = 0; i<policySize; i++) {
    if(policyProbs[i] >= 0) {
      double weight = searchParams.rootDirichletNoiseWeight;
      policyProbs[i] = (float)(r[i] * weight + policyProbs[i] * (1.0-weight));
    }
  }
}


std::shared_ptr<NNOutput>* Search::maybeAddPolicyNoiseAndTemp(SearchThread& thread, bool isRoot, const NNOutput* oldNNOutput) const {
  if(!isRoot)
    return NULL;
  if((!searchParams.rootNoiseEnabled || searchParams.useGumbel) &&
     (searchParams.useGumbel || (searchParams.rootPolicyTemperature == 1.0 && searchParams.rootPolicyTemperatureEarly == 1.0)) &&
     rootHintLoc == Board::NULL_LOC &&
     !avoidMoveUntilRescaleRoot
  )
    return NULL;
  if(oldNNOutput == NULL)
    return NULL;
  if(oldNNOutput->noisedPolicyProbs != NULL)
    return NULL;

  //Copy nnOutput as we're about to modify its policy to add noise or temperature
  std::shared_ptr<NNOutput>* newNNOutputSharedPtr = new std::shared_ptr<NNOutput>(muZeroEvaluator
    ? std::static_pointer_cast<NNOutput>(std::make_shared<MuZero::Evaluation>(*static_cast<const MuZero::Evaluation*>(oldNNOutput)))
    : std::make_shared<NNOutput>(*oldNNOutput));
  NNOutput* newNNOutput = newNNOutputSharedPtr->get();

  float* noisedPolicyProbs = new float[oldNNOutput->getPolicyStorageSize()];
  newNNOutput->noisedPolicyProbs = noisedPolicyProbs;
  std::copy(newNNOutput->policyProbs, newNNOutput->policyProbs + oldNNOutput->getPolicyStorageSize(), noisedPolicyProbs);

  if(!searchParams.useGumbel && (searchParams.rootPolicyTemperature != 1.0 || searchParams.rootPolicyTemperatureEarly != 1.0)) {
    double rootPolicyTemperature = interpolateEarly(
      searchParams.chosenMoveTemperatureHalflife, searchParams.rootPolicyTemperatureEarly, searchParams.rootPolicyTemperature
    );

    double maxValue = 0.0;
    for(int i = 0; i<policySize; i++) {
      double prob = noisedPolicyProbs[i];
      if(prob > maxValue)
        maxValue = prob;
    }
    testAssert(maxValue > 0.0);

    double logMaxValue = log(maxValue);
    double invTemp = 1.0 / rootPolicyTemperature;
    double sum = 0.0;

    for(int i = 0; i<policySize; i++) {
      if(noisedPolicyProbs[i] > 0) {
        //Numerically stable way to raise to power and normalize
        float p = (float)exp((log((double)noisedPolicyProbs[i]) - logMaxValue) * invTemp);
        noisedPolicyProbs[i] = p;
        sum += p;
      }
    }
    testAssert(sum > 0.0);
    for(int i = 0; i<policySize; i++) {
      if(noisedPolicyProbs[i] >= 0) {
        noisedPolicyProbs[i] = (float)(noisedPolicyProbs[i] / sum);
      }
    }
  }

  if(searchParams.rootNoiseEnabled && !searchParams.useGumbel) {
    addDirichletNoise(searchParams, thread.rand, policySize, noisedPolicyProbs);
  }

  if(avoidMoveUntilRescaleRoot) {
    const std::vector<int>& avoidMoveUntilByLoc = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
    if(avoidMoveUntilByLoc.size() > 0) {
      testAssert(avoidMoveUntilByLoc.size() >= (size_t)getMoveLocBufferSize());
      double policySum = 0.0;
      for(Loc loc = 0; loc<getMoveLocBufferSize(); loc++) {
        bool inDomain = rootHistory.rules.isXiangqi() ? getPos(loc) >= 0 : rootBoard.isOnBoard(loc) || loc == Board::PASS_LOC;
        if(inDomain && avoidMoveUntilByLoc[loc] <= 0) {
          int pos = getPos(loc);
          if(noisedPolicyProbs[pos] > 0) {
            policySum += noisedPolicyProbs[pos];
          }
        }
      }
      if(policySum > 0.0) {
        for(int i = 0; i<policySize; i++) {
          if(noisedPolicyProbs[i] > 0) {
            noisedPolicyProbs[i] = (float)(noisedPolicyProbs[i] / policySum);
          }
        }
      }
    }
  }

  //Move a small amount of policy to the hint move, around the same level that noising it would achieve
  if(rootHintLoc != Board::NULL_LOC) {
    const float propToMove = 0.02f;
    int pos = getPos(rootHintLoc);
    if(noisedPolicyProbs[pos] >= 0) {
      double amountToMove = 0.0;
      for(int i = 0; i<policySize; i++) {
        if(noisedPolicyProbs[i] >= 0) {
          amountToMove += noisedPolicyProbs[i] * propToMove;
          noisedPolicyProbs[i] *= (1.0f-propToMove);
        }
      }
      noisedPolicyProbs[pos] += (float)amountToMove;
    }
  }

  return newNNOutputSharedPtr;
}




double Search::getResultUtility(double winLossValue, double noResultValue) const {
  return (
    winLossValue * searchParams.winLossUtilityFactor +
    noResultValue * searchParams.noResultUtilityForWhite
  );
}

double Search::getResultUtilityFromNN(const NNOutput& nnOutput) const {
  return (
    (nnOutput.whiteWinProb - nnOutput.whiteLossProb) * searchParams.winLossUtilityFactor +
    nnOutput.whiteNoResultProb * searchParams.noResultUtilityForWhite
  );
}

double Search::getUtilityFromNN(const NNOutput& nnOutput) const {
  return getResultUtilityFromNN(nnOutput);
}


bool Search::isAllowedRootMove(Loc moveLoc) const {
  if(rootHistory.rules.isXiangqi()) {
    assert(getPos(moveLoc) >= 0);
    return !searchParams.rootSymmetryPruning || !rootJointSymDupLoc[moveLoc];
  }
  assert(moveLoc == Board::PASS_LOC || rootBoard.isOnBoard(moveLoc));

  if(searchParams.rootSymmetryPruning && moveLoc != Board::PASS_LOC && rootSymDupLoc[moveLoc]) {
    return false;
  }

  return true;
}

//Must be called after rootSymDupLoc and rootSymmetries are computed for the current search.
void Search::computeRootSymRepresentativeLocs() {
  if(rootHistory.rules.isXiangqi()) {
    rootJointSymRepresentativeLoc.resize(getMoveLocBufferSize());
    for(Loc loc = 0; loc < getMoveLocBufferSize(); loc++)
      rootJointSymRepresentativeLoc[loc] = rootJointSymDupLoc[loc] ? getSymMove(loc,2) : loc;
    return;
  }
  for(Loc loc = 0; loc < Board::MAX_ARR_SIZE; loc++) {
    rootSymRepresentativeLoc[loc] = loc;
    if(!searchParams.rootSymmetryPruning || loc == Board::PASS_LOC || !rootBoard.isOnBoard(loc) || !rootSymDupLoc[loc])
      continue;
    rootSymRepresentativeLoc[loc] = Board::NULL_LOC;
    for(int symmetry: rootSymmetries) {
      Loc symLoc = SymmetryHelpers::getSymLoc(loc, rootBoard, symmetry);
      if(!rootSymDupLoc[symLoc]) {
        rootSymRepresentativeLoc[loc] = symLoc;
        break;
      }
    }
  }
}

void Search::markXiangqiRootSymmetries() {
  rootJointSymDupLoc.assign(getMoveLocBufferSize(),0);
  rootSymmetries = {0};
  const auto& state = rootHistory.getXiangqiState();
  if(!searchParams.rootSymmetryPruning || state.historyHash() != state.mirroredHistoryHash()) return;
  if(!rootPruneOnlySymmetries.empty() && !contains(rootPruneOnlySymmetries,2)) return;
  rootSymmetries.push_back(2);
  const auto& avoid = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
  for(int i = 0; i < policySize; i++) {
    Loc loc = getLoc(rootPla == P_BLACK ? policySize-1-i : i);
    if(!avoid.empty() && avoid[loc] > 0) continue;
    Loc reflected = getSymMove(loc,2);
    if(!rootJointSymDupLoc[loc] && loc != reflected) rootJointSymDupLoc[reflected] = 1;
  }
}

double Search::getPatternBonus(Hash128 patternBonusHash, Player prevMovePla) const {
  if(patternBonusTable == NULL || prevMovePla != plaThatSearchIsFor)
    return 0;
  return patternBonusTable->get(patternBonusHash).utilityBonus;
}


double Search::interpolateEarly(double halflife, double earlyValue, double value) const {
  double rawHalflives = (double)rootHistory.getCurrentTurnNumber() / halflife;
  double halflives = rawHalflives * 19.0 / sqrt(rootBoard.x_size*rootBoard.y_size);
  return value + (earlyValue - value) * pow(0.5, halflives);
}

void Search::getSelfUtilityLCBAndRadiusZeroVisits(double& lcbBuf, double& radiusBuf) const {
  // Max radius of the entire utility range
  double utilityRangeRadius = searchParams.winLossUtilityFactor;
  radiusBuf = 2.0 * utilityRangeRadius * searchParams.lcbStdevs;
  lcbBuf = -radiusBuf;
  return;
}

void Search::getSelfUtilityLCBAndRadius(const SearchNode& parent, const SearchNode* child, int64_t edgeVisits, double& lcbBuf, double& radiusBuf) const {
  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
  double utilitySqAvg = child->stats.utilitySqAvg.load(std::memory_order_acquire);
  double weightSum = child->stats.getChildWeight(edgeVisits,childVisits);
  double weightSqSum = child->stats.getChildWeightSq(edgeVisits,childVisits);

  // Max radius of the entire utility range
  double utilityRangeRadius = searchParams.winLossUtilityFactor;
  radiusBuf = 2.0 * utilityRangeRadius * searchParams.lcbStdevs;
  lcbBuf = -radiusBuf;
  if(childVisits <= 0 || weightSum <= 0.0 || weightSqSum <= 0.0)
    return;

  // Effective sample size for weighted data
  double ess = weightSum * weightSum / weightSqSum;

  // To behave well at low playouts, we'd like a variance approximation that makes sense even with very small sample sizes.
  // We'd like to avoid using a T distribution approximation because we actually know a bound on the scale of the utilities
  // involved, namely utilityRangeRadius. So instead add a prior with a small weight that the variance is the largest it can be.
  // This should give a relatively smooth scaling that works for small discrete samples but diminishes for larger playouts.
  double priorWeight = weightSum / (ess * ess * ess);
  utilitySqAvg = std::max(utilitySqAvg, utilityAvg * utilityAvg + 1e-8);
  utilitySqAvg = (utilitySqAvg * weightSum + (utilitySqAvg + utilityRangeRadius * utilityRangeRadius) * priorWeight) / (weightSum + priorWeight);
  weightSum += priorWeight;
  weightSqSum += priorWeight*priorWeight;

  // Recompute effective sample size now that we have the prior
  ess = weightSum * weightSum / weightSqSum;

  double selfUtility = parent.nextPla == P_WHITE ? utilityAvg : -utilityAvg;

  double utilityVariance = utilitySqAvg - utilityAvg * utilityAvg;
  double estimateStdev = sqrt(utilityVariance / ess);
  double radius = estimateStdev * searchParams.lcbStdevs;

  lcbBuf = selfUtility - radius;
  radiusBuf = radius;
}
