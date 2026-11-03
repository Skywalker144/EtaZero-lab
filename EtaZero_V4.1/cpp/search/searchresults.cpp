
//-------------------------------------------------------------------------------------
//This file contains various functions for extracting stats and results from the search, choosing a move, etc
//-------------------------------------------------------------------------------------

#include "../search/search.h"
#include "policybuffer.h"

#include <cinttypes>

#include "../core/test.h"
#include "../program/playutils.h"
#include "../game/xiangqi.h"
#include "../search/searchnode.h"

using namespace std;
using nlohmann::json;

bool Search::usesHumanChosenMovePolicy() const {
  return needsHumanOutputAtRoot() && searchParams.humanSLChosenMoveProp > 0.0;
}

bool Search::isOkayRawPolicyMoveAtRoot(Loc moveLoc, double policyProb, bool obeyAllowedRootMove) const {
  if(!rootHistory.isLegal(rootBoard,moveLoc,rootPla) || policyProb < 0 || (obeyAllowedRootMove && !isAllowedRootMove(moveLoc)))
    return false;
  const std::vector<int>& avoidMoveUntilByLoc = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
  if(avoidMoveUntilByLoc.size() > 0) {
    testAssert(avoidMoveUntilByLoc.size() >= (size_t)getMoveLocBufferSize());
    int untilDepth = avoidMoveUntilByLoc[moveLoc];
    if(untilDepth > 0)
      return false;
  }
  return true;
}

void Search::blendHumanPolicy(const SearchNode& node, vector<Loc>& locs, vector<double>& playSelectionValues,
                             bool allowDirectPolicyMoves) const {
  int numChildren = (int)playSelectionValues.size();
  ConstSearchNodeChildrenReference children = node.getChildren();
  const int childrenCapacity = children.getCapacity();
  // Average in human policy
  if(needsHumanOutputAtRoot() && searchParams.humanSLChosenMoveProp > 0.0) {
    const NNOutput* humanOutput = node.getHumanOutput();
    const float* humanProbs = humanOutput != NULL ? humanOutput->getPolicyProbsMaybeNoised() : NULL;
    if(humanProbs != NULL) {
      // First, take a pass to just fill out all the legal/allowed moves into the play selection values, if allowed, and if at root.
      if(&node == rootNode && allowDirectPolicyMoves) {
        std::set<Loc> locsSet(locs.begin(),locs.end());
        for(int movePos = 0; movePos<policySize; movePos++) {
          Loc moveLoc = getLoc(movePos);
          double humanProb = humanProbs[movePos];
          const bool obeyAllowedRootMove = true;
          if(!isOkayRawPolicyMoveAtRoot(moveLoc,humanProb,obeyAllowedRootMove))
            continue;
          if(contains(locsSet,moveLoc))
            continue;
          locs.push_back(moveLoc);
          locsSet.insert(moveLoc);
          playSelectionValues.push_back(0.0); // Pushing zeros since we're just filling in
          numChildren++;
        }
      }

      // Grab utility on the moves we have utilities for.
      std::map<Loc,double> shiftedPolicy;
      std::map<Loc,double> selfUtilities;
      double selfUtilityMax = -1e10;
      double selfUtilitySum = 0.0;
      for(int i = 0; i<childrenCapacity; i++) {
        const SearchChildPointer& childPointer = children[i];
        const SearchNode* child = childPointer.getIfAllocated();
        if(child == NULL)
          break;
        Loc moveLoc = childPointer.getMoveLocRelaxed();
        double humanProb = humanProbs[getPos(moveLoc)];
        if(humanProb < 0)
          humanProb = 0.0;

        shiftedPolicy[moveLoc] = humanProb;
        selfUtilities[moveLoc] = (rootPla == P_WHITE ? 1 : -1) * child->stats.utilityAvg.load(std::memory_order_acquire);
        selfUtilityMax = std::max(selfUtilityMax, selfUtilities[moveLoc]);
        selfUtilitySum += selfUtilities[moveLoc];
      }
      // Straight linear average. Use this to complete the remaining utilities, i.e. for fpu
      double selfUtilityAvg = selfUtilitySum / std::max((size_t)1, selfUtilities.size());
      selfUtilityMax = std::max(selfUtilityMax, selfUtilityAvg); // In case of 0 size
      for(Loc loc: locs) {
        if(!contains(shiftedPolicy,loc)) {
          double humanProb = humanProbs[getPos(loc)];
          if(humanProb < 0)
            humanProb = 0.0;
          shiftedPolicy[loc] = humanProb;
          selfUtilities[loc] = selfUtilityAvg;
        }
      }
      // Perform shift
      for(Loc loc: locs)
        shiftedPolicy[loc] *= exp((selfUtilities[loc] - selfUtilityMax)/searchParams.humanSLChosenMovePiklLambda);

      double shiftedPolicySum = 0.0;
      for(Loc loc: locs)
        shiftedPolicySum += shiftedPolicy[loc];

      // Renormalize and average in to current play selection values, scaling up to the current sum scale of playSelectionValues.
      if(shiftedPolicySum > 0.0) {
        for(Loc loc: locs)
          shiftedPolicy[loc] /= shiftedPolicySum;

        double playSelectionValueSum = 0.0;
        double playSelectionValueNonPassSum = 0.0;
        for(int i = 0; i<numChildren; i++) {
          playSelectionValueSum += playSelectionValues[i];
          if(locs[i] != Board::PASS_LOC)
            playSelectionValueNonPassSum += playSelectionValues[i];
        }

        if(searchParams.humanSLChosenMoveIgnorePass) {
          double shiftedPolicyNonPassSum = 0.0;
          for(Loc loc: locs) {
            if(loc != Board::PASS_LOC)
              shiftedPolicyNonPassSum += shiftedPolicy[loc];
          }
          if(shiftedPolicyNonPassSum > 0.0) {
            for(Loc loc: locs) {
              if(loc != Board::PASS_LOC)
                shiftedPolicy[loc] = shiftedPolicy[loc] / shiftedPolicyNonPassSum * playSelectionValueNonPassSum / playSelectionValueSum;
              else
                shiftedPolicy[loc] = (playSelectionValueSum - playSelectionValueNonPassSum) / playSelectionValueSum;
            }
          }
        }

        for(int i = 0; i<numChildren; i++) {
          playSelectionValues[i] += searchParams.humanSLChosenMoveProp * (playSelectionValueSum * shiftedPolicy[locs[i]] - playSelectionValues[i]);
        }
      }
    }
  }

}

int64_t Search::getRootVisits() const {
  if(rootNode == NULL)
    return 0;
  int64_t n = rootNode->stats.visits.load(std::memory_order_acquire);
  return n;
}

bool Search::getPlaySelectionValues(
  vector<Loc>& locs, vector<double>& playSelectionValues, double scaleMaxToAtLeast
) const {
  if(rootNode == NULL) {
    locs.clear();
    playSelectionValues.clear();
    return false;
  }
  const bool allowDirectPolicyMoves = true;
  return getPlaySelectionValues(*rootNode, locs, playSelectionValues, NULL, scaleMaxToAtLeast, allowDirectPolicyMoves);
}

bool Search::getPlaySelectionValues(
  vector<Loc>& locs, vector<double>& playSelectionValues, vector<double>* retVisitCounts, double scaleMaxToAtLeast
) const {
  if(rootNode == NULL) {
    locs.clear();
    playSelectionValues.clear();
    if(retVisitCounts != NULL)
      retVisitCounts->clear();
    return false;
  }
  const bool allowDirectPolicyMoves = true;
  return getPlaySelectionValues(*rootNode, locs, playSelectionValues, retVisitCounts, scaleMaxToAtLeast, allowDirectPolicyMoves);
}

bool Search::getPlaySelectionValues(
  const SearchNode& node,
  vector<Loc>& locs, vector<double>& playSelectionValues, vector<double>* retVisitCounts, double scaleMaxToAtLeast,
  bool allowDirectPolicyMoves
) const {
  SearchPolicyBuffer<double> lcbBufStorage(getPolicyBufferSize());
  double* lcbBuf = lcbBufStorage.data();
  SearchPolicyBuffer<double> radiusBufStorage(getPolicyBufferSize());
  double* radiusBuf = radiusBufStorage.data();
  bool result = getPlaySelectionValues(
    node,locs,playSelectionValues,retVisitCounts,scaleMaxToAtLeast,allowDirectPolicyMoves,
    false,false,lcbBuf,radiusBuf
  );
  return result;
}

bool Search::getPlaySelectionValues(
  const SearchNode& node,
  vector<Loc>& locs, vector<double>& playSelectionValues, vector<double>* retVisitCounts, double scaleMaxToAtLeast,
  bool allowDirectPolicyMoves, bool alwaysComputeLcb, bool neverUseLcb,
  //Note: lcbBuf is signed from the player to move's perspective
  double* lcbBuf, double* radiusBuf
) const {
  locs.clear();
  playSelectionValues.clear();
  if(retVisitCounts != NULL)
    retVisitCounts->clear();

  const NNOutput* nnOutput = node.getNNOutput();
  const float* policyProbs = nnOutput != NULL ? nnOutput->getPolicyProbsMaybeNoised() : NULL;

  double totalChildWeight = 0.0;

  //Store up basic weights
  ConstSearchNodeChildrenReference children = node.getChildren();
  const int childrenCapacity = children.getCapacity();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    Loc moveLoc = childPointer.getMoveLocRelaxed();

    int64_t edgeVisits = childPointer.getEdgeVisits();
    double childWeight = child->stats.getChildWeight(edgeVisits);

    locs.push_back(moveLoc);
    totalChildWeight += childWeight;

    // If the move appears to be outright illegal in policy probs, zero out the selection value.
    // We always push a value on to playSelectionValues even if that value is 0,
    // because some callers rely on this to line up with the raw indices in the children array of the node.
    if(policyProbs[getPos(moveLoc)] < 0) {
      playSelectionValues.push_back(0.0);
      if(retVisitCounts != NULL)
        (*retVisitCounts).push_back(0.0);
    }
    else {
      playSelectionValues.push_back((double)childWeight);
      if(retVisitCounts != NULL)
        (*retVisitCounts).push_back((double)edgeVisits);
    }
  }

  int numChildren = (int)playSelectionValues.size();

  if(searchParams.useGumbel && (&node == rootNode || searchParams.useGumbelNonroot)) {
    // Root reports completed visits; Gumbel interior reports improved policy on allocated children.
    // Keep allocated-child order: PV and analysis index back into the node's children.
    vector<double> policy;
    if(&node != rootNode && nnOutput != NULL) {
      auto candidates = getGumbelChildren(node);
      double raw = (node.nextPla == P_WHITE ? 1.0 : -1.0) * getUtilityFromNN(*nnOutput);
      policy = Gumbel::gumbel_policy(candidates,raw,searchParams.gumbel);
    }
    for(int i = 0; i < numChildren; i++) {
      int64_t visits = children[i].getEdgeVisits();
      int pos = getPos(locs[i]);
      playSelectionValues[i] = policyProbs[pos] < 0 ? 0.0 : (&node == rootNode ? (double)visits : policy[pos]);
      if(alwaysComputeLcb && !neverUseLcb)
        getSelfUtilityLCBAndRadius(node,children[i].getIfAllocated(),visits,lcbBuf[i],radiusBuf[i]);
    }
    if(numChildren == 0) {
      if(&node != rootNode || !allowDirectPolicyMoves || !getPolicyTargetValues(node,locs,playSelectionValues)) return false;
      if(retVisitCounts != NULL) retVisitCounts->assign(locs.size(),0.0);
    }
    if(usesHumanChosenMovePolicy()) {
      if(&node == rootNode && !searchParams.gumbel.sample_visits) {
        // The default actual move is the SH winner. HumanSL mixes into that
        // one-hot base; sampled-visits mode instead mixes into the visit base.
        Loc winner = getGumbelMove(true);
        if(winner != Board::NULL_LOC) {
          if(std::find(locs.begin(),locs.end(),winner) == locs.end()) {
            // Preserve its base mass during blending even for reports limited
            // to allocated children. Remove the temporary entry below.
            locs.push_back(winner);
            playSelectionValues.push_back(0.0);
          }
          for(size_t i = 0; i < locs.size(); i++) playSelectionValues[i] = locs[i] == winner ? 1.0 : 0.0;
        }
      }
      if(*std::max_element(playSelectionValues.begin(),playSelectionValues.end()) <= 1e-50) {
        // No completed SH visit: sampled-visits mode uses the full root prior,
        // including moves not allocated by weightless HumanSL exploration.
        if(&node == rootNode && allowDirectPolicyMoves) {
          std::set<Loc> present(locs.begin(),locs.end());
          for(int pos = 0; pos < policySize; pos++) {
            Loc loc = getLoc(pos);
            if(isOkayRawPolicyMoveAtRoot(loc,policyProbs[pos],true) && present.insert(loc).second) {
              locs.push_back(loc);
              playSelectionValues.push_back(0.0);
            }
          }
        }
        for(size_t i = 0; i < locs.size(); i++) playSelectionValues[i] = std::max(0.0,(double)policyProbs[getPos(locs[i])]);
      }
      blendHumanPolicy(node,locs,playSelectionValues,allowDirectPolicyMoves);
      if(!allowDirectPolicyMoves) {
        locs.resize(numChildren);
        playSelectionValues.resize(numChildren);
      }
      if(retVisitCounts != NULL) retVisitCounts->resize(locs.size(),0.0);
    }
    double maximum = *std::max_element(playSelectionValues.begin(),playSelectionValues.end());
    if(maximum > 0 && maximum < scaleMaxToAtLeast)
      for(double& value : playSelectionValues) value *= scaleMaxToAtLeast / maximum;
    return true;
  }

  //Find the best child before LCB for pruning. Intended to be the most stably explored child.
  //This is the most weighted child, except with a tiny adjustment so that
  //at very low playouts, variable child weights and discretization doesn't do crazy things.
  int nonLCBBestIdx = 0;
  double nonLCBBestChildWeight = -1e30;
  {
    double maxGoodness = -1e30;
    for(int i = 0; i<numChildren; i++) {
      double weight = playSelectionValues[i];
      const SearchChildPointer& childPointer = children[i];
      double edgeVisits = childPointer.getEdgeVisits();
      Loc moveLoc = childPointer.getMoveLocRelaxed();
      double policyProb = policyProbs[getPos(moveLoc)];

      //Small weight on raw policy, and discount one visit's worth of weight since the most recent
      //visit could be overweighted.
      double g = weight * std::max(0.0,edgeVisits-1.0) / std::max(1.0, edgeVisits) + 2.0 * policyProb;
      if(g > maxGoodness) {
        maxGoodness = g;
        nonLCBBestChildWeight = weight;
        nonLCBBestIdx = i;
      }
    }
  }

  //Possibly reduce weight on children that we spend too many visits on in retrospect
  if(&node == rootNode && numChildren > 0) {
    const SearchChildPointer& bestChildPointer = children[nonLCBBestIdx];
    const SearchNode* bestChild = bestChildPointer.getIfAllocated();
    int64_t bestChildEdgeVisits = bestChildPointer.getEdgeVisits();
    Loc bestMoveLoc = bestChildPointer.getMoveLocRelaxed();
    testAssert(bestChild != NULL);
    const bool isRoot = true;
    const double policyProbMassVisited = 1.0; //doesn't matter, since fpu value computed from it isn't used here
    double parentUtility;
    double parentWeightPerVisit;
    double parentUtilityStdevFactor;
    double fpuValue = getFpuValueForChildrenAssumeVisited(
      node, rootPla, isRoot, policyProbMassVisited,
      parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
    );

    bool isDuringSearch = false;

    double exploreScaling = getExploreScaling(totalChildWeight, parentUtilityStdevFactor);

    testAssert(nnOutput != NULL);
    const bool countEdgeVisit = true;
    double bestChildExploreSelectionValue = getExploreSelectionValueOfChild(
      node,policyProbs,bestChild,
      bestMoveLoc,
      exploreScaling,
      totalChildWeight,bestChildEdgeVisits,fpuValue,
      parentUtility,parentWeightPerVisit,
      isDuringSearch,nonLCBBestChildWeight,
      countEdgeVisit,
      NULL
    );

    for(int i = 0; i<numChildren; i++) {
      const SearchChildPointer& childPointer = children[i];
      const SearchNode* child = childPointer.getIfAllocated();
      Loc moveLoc = childPointer.getMoveLocRelaxed();
      if(i != nonLCBBestIdx) {
        int64_t edgeVisits = childPointer.getEdgeVisits();
        double reduced = getReducedPlaySelectionWeight(
          node, policyProbs, child,
          moveLoc,
          exploreScaling,
          edgeVisits,
          bestChildExploreSelectionValue
        );
        playSelectionValues[i] = ceil(reduced);
      }
    }
  }

  //Now compute play selection values taking into account LCB
  if(!neverUseLcb && (alwaysComputeLcb || (searchParams.useLcbForSelection && numChildren > 0))) {
    double bestLcb = -1e10;
    int bestLcbIndex = -1;
    for(int i = 0; i<numChildren; i++) {
      const SearchChildPointer& childPointer = children[i];
      const SearchNode* child = childPointer.getIfAllocated();
      int64_t edgeVisits = childPointer.getEdgeVisits();
      getSelfUtilityLCBAndRadius(node,child,edgeVisits,lcbBuf[i],radiusBuf[i]);
      //Check if this node is eligible to be considered for best LCB
      double weight = playSelectionValues[i];
      if(weight > 0 && weight >= searchParams.minVisitPropForLCB * nonLCBBestChildWeight) {
        if(lcbBuf[i] > bestLcb) {
          bestLcb = lcbBuf[i];
          bestLcbIndex = i;
        }
      }
    }

    if(searchParams.useLcbForSelection && numChildren > 0 && (searchParams.useNonBuggyLcb ? (bestLcbIndex >= 0) : (bestLcbIndex > 0))) {
      //Best LCB move gets a bonus that ensures it is large enough relative to every other child
      double adjustedWeight = playSelectionValues[bestLcbIndex];
      for(int i = 0; i<numChildren; i++) {
        if(i != bestLcbIndex) {
          double excessValue = bestLcb - lcbBuf[i];
          //This move is actually worse lcb than some other move, it's just that the other
          //move failed its checks for having enough minimum weight. So don't actually
          //try to compute how much better this one is than that one, because it's not better.
          if(excessValue < 0)
            continue;

          double radius = radiusBuf[i];
          //How many times wider would the radius have to be before the lcb would be worse?
          //Add adjust the denom so that we cannot possibly gain more than a factor of 5, just as a guard
          double radiusFactor = (radius + excessValue) / (radius + 0.20 * excessValue);

          //That factor, squared, is the number of "weight" more that we should pretend we have, for
          //the purpose of selection, since normally stdev is proportional to 1/weight^2.
          double lbound = radiusFactor * radiusFactor * playSelectionValues[i];
          if(lbound > adjustedWeight)
            adjustedWeight = lbound;
        }
      }
      playSelectionValues[bestLcbIndex] = adjustedWeight;
    }
  }


  //If we have no children, then use the policy net directly. Only for the root, though, if calling this on any subtree
  //then just require that we have children, for implementation simplicity (since it requires that we have a board and a boardhistory too)
  //(and we also use isAllowedRootMove and avoidMoveUntilByLoc)
  if(numChildren == 0) {
    if(nnOutput == NULL || &node != rootNode || !allowDirectPolicyMoves)
      return false;

    bool obeyAllowedRootMove = true;
    while(true) {
      for(int movePos = 0; movePos<policySize; movePos++) {
        Loc moveLoc = getLoc(movePos);
        double policyProb = policyProbs[movePos];
        if(!isOkayRawPolicyMoveAtRoot(moveLoc,policyProb,obeyAllowedRootMove))
          continue;
        locs.push_back(moveLoc);
        playSelectionValues.push_back(policyProb);
        numChildren++;
      }
      //Still no children? Then at this point just ignore isAllowedRootMove.
      if(numChildren == 0 && obeyAllowedRootMove) {
        obeyAllowedRootMove = false;
        continue;
      }
      break;
    }
  }

  //Might happen absurdly rarely if we both have no children and don't properly have an nnOutput
  //but have a hash collision or something so we "found" an nnOutput anyways.
  //Could also happen if we have avoidMoveUntilByLoc pruning all the allowed moves.
  if(numChildren == 0)
    return false;

  double maxValue = 0.0;
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] > maxValue)
      maxValue = playSelectionValues[i];
  }

  if(maxValue <= 1e-50) {
    //If we reached this point we have nonzero many children but the children are all weightless.
    //In that case, at least set each one to be weighted by its policy.
    for(int i = 0; i<numChildren; i++) {
      playSelectionValues[i] = std::max(0.0,(double)policyProbs[getPos(locs[i])]);
    }
    //Recompute max
    for(int i = 0; i<numChildren; i++) {
      if(playSelectionValues[i] > maxValue)
        maxValue = playSelectionValues[i];
    }
    if(maxValue <= 1e-50) {
      return false;
    }
  }

  //Sanity check - if somehow we had more than this, something must have overflowed or gone wrong
  testAssert(maxValue < 1e40);

  double amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxValue/64.0);
  double amountToPrune = std::min(searchParams.chosenMovePrune, maxValue/64.0);
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] < amountToPrune)
      playSelectionValues[i] = 0.0;
    else {
      playSelectionValues[i] -= amountToSubtract;
      if(playSelectionValues[i] <= 0.0)
        playSelectionValues[i] = 0.0;
    }
  }

  blendHumanPolicy(node,locs,playSelectionValues,allowDirectPolicyMoves);
  numChildren = (int)playSelectionValues.size();

  maxValue = 0.0;
  for(int i = 0; i<numChildren; i++) {
    if(playSelectionValues[i] > maxValue)
      maxValue = playSelectionValues[i];
  }
  testAssert(maxValue > 0.0);
  if(maxValue < scaleMaxToAtLeast) {
    for(int i = 0; i<numChildren; i++) {
      playSelectionValues[i] *= scaleMaxToAtLeast / maxValue;
    }
  }

  return true;
}


bool Search::getRootValues(ReportedSearchValues& values) const {
  return getNodeValues(rootNode,values);
}

ReportedSearchValues Search::getRootValuesRequireSuccess() const {
  ReportedSearchValues values;
  if(rootNode == NULL)
    throw StringError("Bug? Bot search root was null");
  bool success = getNodeValues(rootNode,values);
  if(!success)
    throw StringError("Bug? Bot search returned no root values");
  return values;
}

bool Search::getRootRawNNValues(ReportedSearchValues& values) const {
  if(rootNode == NULL)
    return false;
  return getNodeRawNNValues(*rootNode,values);
}

ReportedSearchValues Search::getRootRawNNValuesRequireSuccess() const {
  ReportedSearchValues values;
  if(rootNode == NULL)
    throw StringError("Bug? Bot search root was null");
  bool success = getNodeRawNNValues(*rootNode,values);
  if(!success)
    throw StringError("Bug? Bot search returned no root values");
  return values;
}

bool Search::getNodeRawNNValues(const SearchNode& node, ReportedSearchValues& values) const {
  const NNOutput* nnOutput = node.getNNOutput();
  if(nnOutput == NULL)
    return false;

  values.winValue = nnOutput->whiteWinProb;
  values.lossValue = nnOutput->whiteLossProb;
  values.drawValue = nnOutput->whiteDrawProb;
  values.noResultValue = nnOutput->whiteNoResultProb;


  //Sanity check
  testAssert(values.winValue >= 0.0);
  testAssert(values.lossValue >= 0.0);
  testAssert(values.noResultValue >= 0.0);
  testAssert(values.winValue + values.drawValue + values.lossValue + values.noResultValue < 1.001);

  double winLossValue = values.winValue - values.lossValue;
  if(winLossValue > 1.0) winLossValue = 1.0;
  if(winLossValue < -1.0) winLossValue = -1.0;
  values.winLossValue = winLossValue;

  values.weight = computeWeightFromNNOutput(nnOutput);
  values.visits = 1;

  return true;
}


bool Search::getNodeValues(const SearchNode* node, ReportedSearchValues& values) const {
  if(node == NULL)
    return false;
  int64_t visits = node->stats.visits.load(std::memory_order_acquire);
  double weightSum = node->stats.weightSum.load(std::memory_order_acquire);
  double winLossValueAvg = node->stats.winLossValueAvg.load(std::memory_order_acquire);
  double drawValueAvg = node->stats.drawValueAvg.load(std::memory_order_acquire);
  double noResultValueAvg = node->stats.noResultValueAvg.load(std::memory_order_acquire);
  double utilityAvg = node->stats.utilityAvg.load(std::memory_order_acquire);

  if(weightSum <= 0.0)
    return false;
  assert(visits >= 0);
  if(node == rootNode && rootHistory.rules.isGo()) {
    //For terminal nodes, we may have no nnoutput and yet we have legitimate visits and terminal evals.
    //But for the root, the root is never treated as a terminal node and always gets an nneval, so if
    //it has visits and weight, it has an nnoutput unless something has gone wrong.
    const NNOutput* nnOutput = node->getNNOutput();
    testAssert(nnOutput != NULL);
    (void)nnOutput;
  }

  values = ReportedSearchValues(
    winLossValueAvg,
    drawValueAvg,
    noResultValueAvg,
    utilityAvg,
    weightSum,
    visits
  );
  return true;
}

const SearchNode* Search::getRootNode() const {
  return rootNode;
}
const SearchNode* Search::getChildForMove(const SearchNode* node, Loc moveLoc) const {
  if(node == NULL)
    return NULL;
  ConstSearchNodeChildrenReference children = node->getChildren();
  int childrenCapacity = children.getCapacity();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    Loc childMoveLoc = childPointer.getMoveLocRelaxed();
    if(moveLoc == childMoveLoc)
      return child;
  }
  return NULL;
}

Loc Search::getChosenMoveLoc() {
  if(rootNode == NULL)
    return Board::NULL_LOC;

  if(searchParams.useGumbel && !searchParams.gumbel.sample_visits && !usesHumanChosenMovePolicy()) return getGumbelMove();
  vector<Loc> locs;
  vector<double> playSelectionValues;
  if(searchParams.useGumbel && !usesHumanChosenMovePolicy()) {
    auto children = getGumbelChildren(*rootNode);
    locs.clear(); playSelectionValues.clear();
    double total = 0; for(const auto& child : children) total += child.visits;
    for(int pos = 0; pos < policySize; pos++) if(children[pos].prior >= 0) {
      locs.push_back(getLoc(pos));
      playSelectionValues.push_back(total > 0 ? children[pos].visits : children[pos].prior);
    }
  }
  else if(!getPlaySelectionValues(locs,playSelectionValues,0.0)) return Board::NULL_LOC;
  if(locs.empty()) return Board::NULL_LOC;
  testAssert(locs.size() == playSelectionValues.size());

  double temperature = interpolateEarly(
    searchParams.chosenMoveTemperatureHalflife, searchParams.chosenMoveTemperatureEarly, searchParams.chosenMoveTemperature
  );

  uint32_t idxChosen = chooseIndexWithTemperature(
    nonSearchRand,
    playSelectionValues.data(),
    (int)playSelectionValues.size(),
    temperature,
    searchParams.chosenMoveTemperatureOnlyBelowProb,
    NULL
  );
  return locs[idxChosen];
}


bool Search::getPolicy(float* policyProbs) const {
  return getPolicy(rootNode, policyProbs);
}
bool Search::getPolicy(const SearchNode* node, float* policyProbs) const {
  if(node == NULL)
    return false;
  const NNOutput* nnOutput = node->getNNOutput();
  if(nnOutput == NULL)
    return false;

  std::copy(nnOutput->policyProbs, nnOutput->policyProbs+getPolicyBufferSize(), policyProbs);
  return true;
}


//Safe to call concurrently with search
double Search::getPolicySurprise() const {
  double surprise = 0.0;
  double searchEntropy = 0.0;
  double policyEntropy = 0.0;
  if(getPolicySurpriseAndEntropy(surprise,searchEntropy,policyEntropy))
    return surprise;
  return 0.0;
}

bool Search::getPolicySurpriseAndEntropy(double& surpriseRet, double& searchEntropyRet, double& policyEntropyRet) const {
  return getPolicySurpriseAndEntropy(surpriseRet, searchEntropyRet, policyEntropyRet, rootNode);
}

//Safe to call concurrently with search
bool Search::getPolicySurpriseAndEntropy(double& surpriseRet, double& searchEntropyRet, double& policyEntropyRet, const SearchNode* node, const bool* policyMask) const {
  if(node == NULL)
    return false;
  const NNOutput* nnOutput = node->getNNOutput();
  if(nnOutput == NULL)
    return false;

  vector<Loc> locs;
  vector<double> playSelectionValues;
  const bool allowDirectPolicyMoves = true;
  const bool alwaysComputeLcb = false;
  SearchPolicyBuffer<double> lcbBufStorage(getPolicyBufferSize());
  double* lcbBuf = lcbBufStorage.data();
  SearchPolicyBuffer<double> radiusBufStorage(getPolicyBufferSize());
  double* radiusBuf = radiusBufStorage.data();
  bool suc = searchParams.useGumbel ? getPolicyTargetValues(*node,locs,playSelectionValues) : getPlaySelectionValues(
    *node,locs,playSelectionValues,NULL,1.0,allowDirectPolicyMoves,alwaysComputeLcb,false,lcbBuf,radiusBuf
  );
  if(!suc)
    return false;

  SearchPolicyBuffer<float> policyProbsFromNNBufStorage(getPolicyBufferSize());

  float* policyProbsFromNNBuf = policyProbsFromNNBufStorage.data();
  {
    const float* policyProbsFromNN = nnOutput->getPolicyProbsMaybeNoised();
    std::copy(policyProbsFromNN, policyProbsFromNN+getPolicyBufferSize(), policyProbsFromNNBuf);
  }

  double sumPlaySelectionValues = 0.0;
  // A latent node collected as a real side position has a real-root action
  // mask. Apply it only to the extracted supervision, never to latent search.
  if(policyMask) {
    double policyMass = 0.0;
    for(int pos = 0; pos < getPolicyBufferSize(); pos++) {
      if(!policyMask[pos]) policyProbsFromNNBuf[pos] = -1.0f;
      else if(policyProbsFromNNBuf[pos] > 0.0f) policyMass += policyProbsFromNNBuf[pos];
    }
    if(policyMass <= 0.0) return false;
    for(int pos = 0; pos < getPolicyBufferSize(); pos++)
      if(policyProbsFromNNBuf[pos] > 0.0f) policyProbsFromNNBuf[pos] /= policyMass;
    for(size_t i = 0; i < playSelectionValues.size(); i++)
      if(!policyMask[getPos(locs[i])]) playSelectionValues[i] = 0.0;
  }
  for(size_t i = 0; i < playSelectionValues.size(); i++)
    sumPlaySelectionValues += playSelectionValues[i];
  if(policyMask && sumPlaySelectionValues <= 0.0) return false;

  double surprise = 0.0;
  double searchEntropy = 0.0;
  for(size_t i = 0; i < playSelectionValues.size(); i++) {
    int pos = getPos(locs[i]);
    double policy = std::max((double)policyProbsFromNNBuf[pos],1e-100);
    double target = playSelectionValues[i] / sumPlaySelectionValues;
    if(target > 1e-100) {
      double logTarget = log(target);
      double logPolicy = log(policy);
      surprise += target * (logTarget - logPolicy);
      searchEntropy += -target * logTarget;
    }
  }

  double policyEntropy = 0.0;
  for(int pos = 0; pos<getPolicyBufferSize(); pos++) {
    double policy = policyProbsFromNNBuf[pos];
    if(policy > 1e-100) {
      policyEntropy += -policy * log(policy);
    }
  }

  //Just in case, guard against float imprecision
  if(surprise < 0.0)
    surprise = 0.0;
  if(searchEntropy < 0.0)
    searchEntropy = 0.0;
  if(policyEntropy < 0.0)
    policyEntropy = 0.0;

  surpriseRet = surprise;
  searchEntropyRet = searchEntropy;
  policyEntropyRet = policyEntropy;

  return true;
}



void Search::printRootPolicyMap(ostream& out) const {
  if(rootNode == NULL)
    return;
  const NNOutput* nnOutput = rootNode->getNNOutput();
  if(nnOutput == NULL)
    return;

  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
  if(rootHistory.rules.isXiangqi()) {
    for(int pos = 0; pos < policySize; pos++) if(policyProbs[pos] >= 0)
      out << moveToString(getLoc(pos)) << " " << Global::strprintf("%6.1f",policyProbs[pos]*100) << endl;
    return;
  }
  for(int y = 0; y<rootBoard.y_size; y++) {
    for(int x = 0; x<rootBoard.x_size; x++) {
      int pos = NNPos::xyToPos(x,y,nnOutput->nnXLen);
      out << Global::strprintf("%6.1f ", policyProbs[pos]*100);
    }
    out << endl;
  }
  out << endl;
}

void Search::appendPV(
  vector<Loc>& buf,
  vector<int64_t>& visitsBuf,
  vector<int64_t>& edgeVisitsBuf,
  vector<Loc>& scratchLocs,
  vector<double>& scratchValues,
  const SearchNode* node,
  int maxDepth
) const {
  appendPVForMove(buf,visitsBuf,edgeVisitsBuf,scratchLocs,scratchValues,node,Board::NULL_LOC,maxDepth);
}

void Search::appendPVForMove(
  vector<Loc>& buf,
  vector<int64_t>& visitsBuf,
  vector<int64_t>& edgeVisitsBuf,
  vector<Loc>& scratchLocs,
  vector<double>& scratchValues,
  const SearchNode* node,
  Loc move,
  int maxDepth
) const {
  if(node == NULL)
    return;

  if(node == rootNode && move == Board::NULL_LOC && searchParams.useGumbel && !searchParams.gumbel.sample_visits && !usesHumanChosenMovePolicy())
    move = getGumbelMove(true);
  for(int depth = 0; depth < maxDepth; depth++) {
    const bool allowDirectPolicyMoves = true;
    bool success = getPlaySelectionValues(*node, scratchLocs, scratchValues, NULL, 1.0, allowDirectPolicyMoves);
    if(!success)
      return;

    double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
    int bestChildIdx = -1;
    Loc bestChildMoveLoc = Board::NULL_LOC;

    for(int i = 0; i<scratchValues.size(); i++) {
      Loc moveLoc = scratchLocs[i];
      double selectionValue = scratchValues[i];

      if(depth == 0 && moveLoc == move) {
        maxSelectionValue = selectionValue;
        bestChildIdx = i;
        bestChildMoveLoc = moveLoc;
        break;
      }

      if(selectionValue > maxSelectionValue) {
        maxSelectionValue = selectionValue;
        bestChildIdx = i;
        bestChildMoveLoc = moveLoc;
      }
    }

    if(bestChildIdx < 0 || bestChildMoveLoc == Board::NULL_LOC)
      return;
    if(depth == 0 && move != Board::NULL_LOC && bestChildMoveLoc != move)
      return;

    ConstSearchNodeChildrenReference children = node->getChildren();
    int childrenCapacity = children.getCapacity();
    //Direct policy move
    if(bestChildIdx >= childrenCapacity) {
      buf.push_back(bestChildMoveLoc);
      visitsBuf.push_back(0);
      edgeVisitsBuf.push_back(0);
      return;
    }
    const SearchNode* child = children[bestChildIdx].getIfAllocated();
    //Direct policy move
    if(child == NULL) {
      buf.push_back(bestChildMoveLoc);
      visitsBuf.push_back(0);
      edgeVisitsBuf.push_back(0);
      return;
    }

    node = child;

    int64_t visits = node->stats.visits.load(std::memory_order_acquire);
    int64_t edgeVisits = children[bestChildIdx].getEdgeVisits();

    buf.push_back(bestChildMoveLoc);
    visitsBuf.push_back(visits);
    edgeVisitsBuf.push_back(edgeVisits);
  }
}


void Search::printPV(ostream& out, const SearchNode* n, int maxDepth) const {
  vector<Loc> buf;
  vector<int64_t> visitsBuf;
  vector<int64_t> edgeVisitsBuf;
  vector<Loc> scratchLocs;
  vector<double> scratchValues;
  appendPV(buf,visitsBuf,edgeVisitsBuf,scratchLocs,scratchValues,n,maxDepth);
  printPV(out,buf);
}

void Search::printPV(ostream& out, const vector<Loc>& buf) const {
  bool printedAnything = false;
  for(int i = 0; i<buf.size(); i++) {
    if(printedAnything)
      out << " ";
    if(buf[i] == Board::NULL_LOC)
      continue;
    out << moveToString(buf[i]);
    printedAnything = true;
  }
}

//Child should NOT be locked.
AnalysisData Search::getAnalysisDataOfSingleChild(
  const SearchNode* child, int64_t edgeVisits, vector<Loc>& scratchLocs, vector<double>& scratchValues,
  Loc move, double policyProb, double fpuValue, double parentUtility, double parentWinLossValue, int maxPVDepth
) const {
  int64_t childVisits = 0;
  double winLossValueAvg = 0.0;
  double drawValueAvg = 0.0;
  double noResultValueAvg = 0.0;
  double utilityAvg = 0.0;
  double utilitySqAvg = 0.0;
  double weightSum = 0.0;
  double weightSqSum = 0.0;
  double childWeightSum = 0.0;

  if(child != NULL) {
    childVisits = child->stats.visits.load(std::memory_order_acquire);
    winLossValueAvg = child->stats.winLossValueAvg.load(std::memory_order_acquire);
    drawValueAvg = child->stats.drawValueAvg.load(std::memory_order_acquire);
    noResultValueAvg = child->stats.noResultValueAvg.load(std::memory_order_acquire);
    utilityAvg = child->stats.utilityAvg.load(std::memory_order_acquire);
    utilitySqAvg = child->stats.utilitySqAvg.load(std::memory_order_acquire);
    weightSum = child->stats.getChildWeight(edgeVisits,childVisits);
    weightSqSum = child->stats.getChildWeightSq(edgeVisits,childVisits);
    childWeightSum = child->stats.weightSum.load(std::memory_order_acquire);
  }

  AnalysisData data;
  data.move = move;
  data.numVisits = edgeVisits;
  if(childVisits <= 0 || childWeightSum <= 1e-30) {
    data.utility = fpuValue;
    data.resultUtility = fpuValue;
    data.winLossValue = searchParams.winLossUtilityFactor == 1.0 ? parentWinLossValue + (fpuValue - parentUtility) : 0.0;
    data.drawValue = 0.0;
    data.noResultValue = 0.0;
    // Make sure winloss values due to FPU don't go out of bounds for purposes of reporting to UI
    if(data.winLossValue < -1.0)
      data.winLossValue = -1.0;
    if(data.winLossValue > 1.0)
      data.winLossValue = 1.0;
    data.ess = 0.0;
    data.weightSum = 0.0;
    data.weightSqSum = 0.0;
    data.utilitySqAvg = data.utility * data.utility;
    data.childVisits = childVisits;
    data.childWeightSum = childWeightSum;
  }
  else {
    data.utility = utilityAvg;
    data.resultUtility = getResultUtility(winLossValueAvg, noResultValueAvg);
    data.winLossValue = winLossValueAvg;
    data.drawValue = drawValueAvg;
    data.noResultValue = noResultValueAvg;
    data.ess = weightSum * weightSum / std::max(1e-8,weightSqSum);
    data.weightSum = weightSum;
    data.weightSqSum = weightSqSum;
    data.utilitySqAvg = utilitySqAvg;
    data.childVisits = childVisits;
    data.childWeightSum = childWeightSum;
  }

  data.policyPrior = policyProb;
  data.order = 0;

  data.pv.clear();
  data.pv.push_back(move);
  data.pvVisits.clear();
  data.pvVisits.push_back(childVisits);
  data.pvEdgeVisits.clear();
  data.pvEdgeVisits.push_back(edgeVisits);
  appendPV(data.pv, data.pvVisits, data.pvEdgeVisits, scratchLocs, scratchValues, child, maxPVDepth);

  data.node = child;

  return data;
}

void Search::getAnalysisData(
  vector<AnalysisData>& buf,int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth, bool duplicateForSymmetries
) const {
  buf.clear();
  if(rootNode == NULL)
    return;
  getAnalysisData(*rootNode, buf, minMovesToTryToGet, includeWeightFactors, maxPVDepth, duplicateForSymmetries);
}

void Search::getAnalysisData(
  const SearchNode& node, vector<AnalysisData>& buf, int minMovesToTryToGet, bool includeWeightFactors, int maxPVDepth, bool duplicateForSymmetries
) const {
  buf.clear();
  vector<const SearchNode*> children;
  vector<int64_t> childrenEdgeVisits;
  vector<Loc> childrenMoveLocs;
  children.reserve(rootBoard.x_size * rootBoard.y_size + 1);
  childrenEdgeVisits.reserve(rootBoard.x_size * rootBoard.y_size + 1);
  childrenMoveLocs.reserve(rootBoard.x_size * rootBoard.y_size + 1);

  int numChildren;
  vector<Loc> scratchLocs;
  vector<double> scratchValues;
  SearchPolicyBuffer<double> lcbBufStorage(getPolicyBufferSize());
  double* lcbBuf = lcbBufStorage.data();
  SearchPolicyBuffer<double> radiusBufStorage(getPolicyBufferSize());
  double* radiusBuf = radiusBufStorage.data();
  SearchPolicyBuffer<float> policyProbsStorage(getPolicyBufferSize());
  float* policyProbs = policyProbsStorage.data();
  {
    ConstSearchNodeChildrenReference childrenArr = node.getChildren();
    int childrenCapacity = childrenArr.getCapacity();
    for(int i = 0; i<childrenCapacity; i++) {
      const SearchChildPointer& childPointer = childrenArr[i];
      const SearchNode* child = childPointer.getIfAllocated();
      if(child == NULL)
        break;
      children.push_back(child);
      childrenEdgeVisits.push_back(childPointer.getEdgeVisits());
      childrenMoveLocs.push_back(childPointer.getMoveLocRelaxed());
    }
    numChildren = (int)children.size();

    if(numChildren <= 0)
      return;
    assert(numChildren <= getPolicyBufferSize());

    const bool alwaysComputeLcb = true;
    bool gotPlaySelectionValues = getPlaySelectionValues(node, scratchLocs, scratchValues, NULL, 1.0, false, alwaysComputeLcb, false, lcbBuf, radiusBuf);

    // No play selection values - then fill with values consistent with all 0 visits.
    // We want it to be possible to get analysis data even when all visits are weightless.
    if(!gotPlaySelectionValues) {
      for(int i = 0; i<numChildren; i++) {
        scratchLocs.push_back(childrenMoveLocs[i]);
        scratchValues.push_back(0.0);
      }
      double lcbBufValue;
      double radiusBufValue;
      getSelfUtilityLCBAndRadiusZeroVisits(lcbBufValue,radiusBufValue);
      std::fill(lcbBuf,lcbBuf+numChildren,lcbBufValue);
      std::fill(radiusBuf,radiusBuf+numChildren,radiusBufValue);
    }

    const NNOutput* nnOutput = node.getNNOutput();
    const float* policyProbsFromNN = nnOutput->getPolicyProbsMaybeNoised();
    for(int i = 0; i<getPolicyBufferSize(); i++)
      policyProbs[i] = policyProbsFromNN[i];
  }

  //Copy to make sure we keep these values so we can reuse scratch later for PV
  vector<double> playSelectionValues = scratchValues;

  double policyProbMassVisited = 0.0;
  {
    for(int i = 0; i<numChildren; i++) {
      policyProbMassVisited += std::max(0.0, (double)policyProbs[getPos(childrenMoveLocs[i])]);
    }
    //Probability mass should not sum to more than 1, giving a generous allowance
    //for floating point error.
    assert(policyProbMassVisited <= 1.0001);
  }

  double parentWinLossValue;
  {
    double winLossValueAvg = node.stats.winLossValueAvg.load(std::memory_order_acquire);
    assert(node.stats.weightSum.load(std::memory_order_acquire) > 0.0);

    parentWinLossValue = winLossValueAvg;
  }

  double parentUtility;
  double parentWeightPerVisit;
  double parentUtilityStdevFactor;
  double fpuValue = getFpuValueForChildrenAssumeVisited(
    node, node.nextPla, true, policyProbMassVisited,
    parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );

  vector<MoreNodeStats> statsBuf(numChildren);
  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = children[i];
    int64_t edgeVisits = childrenEdgeVisits[i];
    Loc moveLoc = childrenMoveLocs[i];
    double policyProb = policyProbs[getPos(moveLoc)];
    AnalysisData data = getAnalysisDataOfSingleChild(
      child, edgeVisits, scratchLocs, scratchValues, moveLoc, policyProb, fpuValue, parentUtility, parentWinLossValue, maxPVDepth
    );
    data.playSelectionValue = playSelectionValues[i];
    //Make sure data.lcb is from white's perspective, for consistency with everything else
    //In lcbBuf, it's from self perspective, unlike values at nodes.
    data.lcb = node.nextPla == P_BLACK ? -lcbBuf[i] : lcbBuf[i];
    data.radius = radiusBuf[i];
    buf.push_back(data);

    if(includeWeightFactors) {
      MoreNodeStats& stats = statsBuf[i];
      stats.stats = NodeStats(child->stats);
      stats.selfUtility = node.nextPla == P_WHITE ? data.utility : -data.utility;
      stats.weightAdjusted = stats.stats.getChildWeight(edgeVisits);
      stats.prevMoveLoc = moveLoc;
    }
  }

  //Find all children and compute weighting of the children based on their values
  if(includeWeightFactors) {
    double totalChildWeight = 0.0;
    for(int i = 0; i<numChildren; i++) {
      totalChildWeight += statsBuf[i].weightAdjusted;
    }
    if(searchParams.useNoisePruning) {
      SearchPolicyBuffer<double> policyProbsBufStorage(getPolicyBufferSize());
      double* policyProbsBuf = policyProbsBufStorage.data();
      for(int i = 0; i<numChildren; i++)
        policyProbsBuf[i] = std::max(1e-30, (double)policyProbs[getPos(statsBuf[i].prevMoveLoc)]);
      totalChildWeight = pruneNoiseWeight(statsBuf, numChildren, totalChildWeight, policyProbsBuf);
    }
    double amountToSubtract = 0.0;
    double amountToPrune = 0.0;
    downweightBadChildrenAndNormalizeWeight(
      numChildren, totalChildWeight, totalChildWeight,
      amountToSubtract, amountToPrune, statsBuf
    );
    for(int i = 0; i<numChildren; i++)
      buf[i].weightFactor = statsBuf[i].weightAdjusted;
  }

  //Fill the rest of the moves directly from policy
  if(numChildren < minMovesToTryToGet) {
    //A bit inefficient, but no big deal
    for(int i = 0; i<minMovesToTryToGet - numChildren; i++) {
      int bestPos = -1;
      double bestPolicy = -1.0;
      for(int pos = 0; pos<getPolicyBufferSize(); pos++) {
        if(policyProbs[pos] < bestPolicy)
          continue;

        bool alreadyUsed = false;
        for(int j = 0; j<buf.size(); j++) {
          if(getPos(buf[j].move) == pos) {
            alreadyUsed = true;
            break;
          }
        }
        if(alreadyUsed)
          continue;

        bestPos = pos;
        bestPolicy = policyProbs[pos];
      }
      if(bestPos < 0 || bestPolicy < 0.0)
        break;

      Loc bestMove = getLoc(bestPos);
      AnalysisData data = getAnalysisDataOfSingleChild(
        NULL, 0, scratchLocs, scratchValues, bestMove, bestPolicy, fpuValue, parentUtility, parentWinLossValue, maxPVDepth
      );
      buf.push_back(data);
    }
  }
  std::stable_sort(buf.begin(),buf.end());
  if(&node == rootNode && searchParams.useGumbel && !searchParams.gumbel.sample_visits && !usesHumanChosenMovePolicy()) {
    Loc selected = getGumbelMove(true);
    auto winner = std::find_if(buf.begin(),buf.end(),[&](const AnalysisData& data) { return data.move == selected; });
    if(winner != buf.end()) std::rotate(buf.begin(),winner,winner+1);
  }

  if(duplicateForSymmetries && searchParams.rootSymmetryPruning && rootSymmetries.size() > 1) {
    vector<AnalysisData> newBuf;
    std::set<Loc> isDone;
    for(int i = 0; i<buf.size(); i++) {
      const AnalysisData& data = buf[i];
      for(int symmetry : rootSymmetries) {
        Loc symMove = getSymMove(data.move,symmetry);
        if(contains(isDone,symMove))
          continue;
        const std::vector<int>& avoidMoveUntilByLoc = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
        if(avoidMoveUntilByLoc.size() > 0 && avoidMoveUntilByLoc[symMove] > 0)
          continue;

        isDone.insert(symMove);
        newBuf.push_back(data);
        //Replace the fields that need to be adjusted for symmetry
        AnalysisData& newData = newBuf.back();
        newData.move = symMove;
        if(symmetry != 0)
          newData.isSymmetryOf = data.move;
        newData.symmetry = symmetry;
        for(int j = 0; j<newData.pv.size(); j++)
          newData.pv[j] = getSymMove(newData.pv[j],symmetry);
      }
    }
    buf = std::move(newBuf);
  }

  for(int i = 0; i<buf.size(); i++)
    buf[i].order = i;
}

void Search::printPVForMove(ostream& out, const SearchNode* n, Loc move, int maxDepth) const {
  vector<Loc> buf;
  vector<int64_t> visitsBuf;
  vector<int64_t> edgeVisitsBuf;
  vector<Loc> scratchLocs;
  vector<double> scratchValues;
  appendPVForMove(buf,visitsBuf,edgeVisitsBuf,scratchLocs,scratchValues,n,move,maxDepth);
  for(int i = 0; i<buf.size(); i++) {
    if(i > 0)
      out << " ";
    out << moveToString(buf[i]);
  }
}

void Search::printTree(ostream& out, const SearchNode* node, const PrintTreeOptions& options, Player perspective) const {
  if(node == NULL)
    return;
  string prefix;
  AnalysisData data;
  {
    vector<Loc> scratchLocs;
    vector<double> scratchValues;
    //Use dummy values for parent
    double policyProb = NAN;
    double fpuValue = 0;
    double parentUtility = 0;
    double parentWinLossValue = 0;
    //Since we don't have an edge from another parent we are following, we just use the visits on the node itself as the edge visits.
    int64_t edgeVisits = node->stats.visits.load(std::memory_order_acquire);
    data = getAnalysisDataOfSingleChild(
      node, edgeVisits, scratchLocs, scratchValues,
      Board::NULL_LOC, policyProb, fpuValue, parentUtility, parentWinLossValue, options.maxPVDepth_
    );
    data.weightFactor = NAN;
  }
  perspective = (perspective != P_BLACK && perspective != P_WHITE) ? node->nextPla : perspective;
  std::unordered_set<const SearchNode*> graphPath;
  printTreeHelper(out, node, options, prefix, 0, 0, data, perspective, graphPath);
}

void Search::printTreeHelper(
  ostream& out, const SearchNode* n, const PrintTreeOptions& options,
  string& prefix, int64_t origVisits, int depth, const AnalysisData& data, Player perspective,
  std::unordered_set<const SearchNode*>& graphPath
) const {
  if(n == NULL)
    return;

  const SearchNode& node = *n;

  Player perspectiveToUse = (perspective != P_BLACK && perspective != P_WHITE) ? n->nextPla : perspective;
  double perspectiveFactor = perspectiveToUse == P_BLACK ? -1.0 : 1.0;

  if(depth == 0)
    origVisits = data.numVisits;

  //Output for this node
  {
    out << prefix;
    char buf[128];

    out << ": ";

    if(data.childVisits > 0) {
      sprintf(buf,"T %6.2fc ",(perspectiveFactor * data.utility * 100.0));
      out << buf;
      sprintf(buf,"W %6.2fc ",(perspectiveFactor * data.resultUtility * 100.0));
      out << buf;
    }

    // bool hasNNValue = false;
    // double nnResultValue;
    // double nnTotalValue;
    // lock.lock();
    // if(node.nnOutput != nullptr) {
    //   nnResultValue = getResultUtilityFromNN(*node.nnOutput);
    //   nnTotalValue = getUtilityFromNN(*node.nnOutput);
    //   hasNNValue = true;
    // }
    // lock.unlock();

    // if(hasNNValue) {
    //   sprintf(buf,"VW %6.2fc VS %6.2fc ", nnResultValue * 100.0, (nnTotalValue - nnResultValue) * 100.0);
    //   out << buf;
    // }
    // else {
    //   sprintf(buf,"VW ---.--c VS ---.--c ");
    //   out << buf;
    // }

    if(depth > 0 && !isnan(data.lcb)) {
      sprintf(buf,"LCB %7.2fc ", perspectiveFactor * data.lcb * 100.0);
      out << buf;
    }

    if(!isnan(data.policyPrior)) {
      sprintf(buf,"P %5.2f%% ", data.policyPrior * 100.0);
      out << buf;
    }
    if(!isnan(data.weightFactor)) {
      sprintf(buf,"WF %5.1f ", data.weightFactor);
      out << buf;
    }
    if(data.playSelectionValue >= 0 && depth > 0) {
      sprintf(buf,"PSV %7.0f ", data.playSelectionValue);
      out << buf;
    }

    if(options.printSqs_) {
      sprintf(buf,"USQ %7.5f W %6.2f WSQ %8.2f ", data.utilitySqAvg, data.weightSum, data.weightSqSum);
      out << buf;
    }

    if(options.printAvgShorttermError_) {
      double wlError = getShallowAverageShorttermWLError(&node);
      sprintf(buf,"STWL %6.2fc ", wlError * 100.0);
      out << buf;
    }

    // Using child visits here instead of edge visits because edge visits is at least
    // semi-reflected in WF and PSV.
    sprintf(buf,"N %7" PRIu64 "  --  ", data.childVisits);
    out << buf;

    printPV(out, data.pv);
    out << endl;
  }

  if(depth >= options.branch_.size()) {
    if(depth >= options.maxDepth_ + options.branch_.size())
      return;
    if(data.numVisits < options.minVisitsToExpand_)
      return;
    if((double)data.numVisits < origVisits * options.minVisitsPropToExpand_)
      return;
  }
  //Graph search can transpose back to an ancestor on this path. Recursing into it again would repeat the cycle
  //until maxDepth_, so stop after printing its own line.
  if(!graphPath.insert(&node).second)
    return;

  //The top of the printed tree, which is below the root when only printing a branch.
  const bool isDisplayRoot = options.alsoBranch_ ? depth == 0 : depth == options.branch_.size();
  if(isDisplayRoot) {
    out << "---" << PlayerIO::playerToString(node.nextPla) << "(" << (node.nextPla == perspectiveToUse ? "^" : "v") << ")---" << endl;
  }

  vector<AnalysisData> analysisData;
  bool duplicateForSymmetries = false;
  getAnalysisData(node,analysisData,0,true,options.maxPVDepth_,duplicateForSymmetries);

  int numChildren = (int)analysisData.size();

  //Apply filtering conditions, but include children that don't match the filtering condition
  //but where there are children afterward that do, in case we ever use something more complex
  //than plain visits as a filter criterion. Do this by finding the last child that we want as the threshold.
  int lastIdxWithEnoughVisits = numChildren-1;
  while(true) {
    if(lastIdxWithEnoughVisits <= 0)
      break;

    int64_t childVisits = analysisData[lastIdxWithEnoughVisits].numVisits;
    bool hasEnoughVisits = childVisits >= options.minVisitsToShow_
      && (double)childVisits >= origVisits * options.minVisitsPropToShow_;
    if(hasEnoughVisits)
      break;
    lastIdxWithEnoughVisits--;
  }

  int numChildrenToRecurseOn = numChildren;
  //A child shown only due to showAllRootChildren_ is still not expanded if below minVisitsToExpand_.
  if(!(options.showAllRootChildren_ && isDisplayRoot)) {
    if(options.maxChildrenToShow_ < numChildrenToRecurseOn)
      numChildrenToRecurseOn = options.maxChildrenToShow_;
    if(lastIdxWithEnoughVisits+1 < numChildrenToRecurseOn)
      numChildrenToRecurseOn = lastIdxWithEnoughVisits+1;
  }


  for(int i = 0; i<numChildren; i++) {
    const SearchNode* child = analysisData[i].node;
    Loc moveLoc = analysisData[i].move;

    if((depth >= options.branch_.size() && i < numChildrenToRecurseOn) ||
       (depth < options.branch_.size() && moveLoc == options.branch_[depth]) ||
       (depth < options.branch_.size() && options.alsoBranch_ && i < numChildrenToRecurseOn)
    ) {
      size_t oldLen = prefix.length();
      string locStr = moveToString(moveLoc);
      if(locStr == "pass")
        prefix += "pss";
      else
        prefix += locStr;
      prefix += " ";
      while(prefix.length() < oldLen+4)
        prefix += " ";
      int nextDepth = depth+1;
      if(depth < options.branch_.size() && moveLoc != options.branch_[depth])
        nextDepth = (int)options.branch_.size() + 1;
      printTreeHelper(out,child,options,prefix,origVisits,nextDepth,analysisData[i], perspective, graphPath);
      prefix.erase(oldLen);
    }
  }

  graphPath.erase(&node);
}


double Search::getShallowAverageShorttermWLError(const SearchNode* node) const {
  if(node == NULL)
    node = rootNode;
  if(node == NULL)
    return 0.0;
  if(!supportsShorttermError())
    return -1.0;
  std::unordered_set<const SearchNode*> graphPath;
  SearchPolicyBuffer<double> policyProbsBufStorage(getPolicyBufferSize());
  double* policyProbsBuf = policyProbsBufStorage.data();
  double wlError = 0.0;
  // Stop deepening when we hit a node whose proportion in the final average would be less than this.
  // Sublinear in visits so that the cost of this grows more slowly than overall search depth.
  int64_t visits = node->stats.visits.load(std::memory_order_acquire);
  double minProp = 0.25 / pow(std::max(1.0,(double)visits),0.625);
  double desiredProp = 1.0;
  getShallowAverageShorttermWLErrorHelper(
    node,
    graphPath,
    policyProbsBuf,
    minProp,
    desiredProp,
    wlError);
  return wlError;
}

void Search::getShallowAverageShorttermWLErrorHelper(
  const SearchNode* node,
  std::unordered_set<const SearchNode*>& graphPath,
  double* policyProbsBuf,
  double minProp,
  double desiredProp,
  double& wlError) const {
  const NNOutput* nnOutput = node->getNNOutput();
  if(nnOutput == NULL) {
    // Accumulate nothing. This will be correct for terminal nodes, which have no uncertainty.
    // Not quite correct for multithreading, but no big deal, this value isn't used for anything critical
    // and currently isn't called while multithreaded. Yay code debt.
    return;
  }

  if(desiredProp < minProp) {
    // We don't track the average errors on nodes, so just use the error of this node's raw nn output.
    wlError += desiredProp * nnOutput->shorttermWinlossError;
    return;
  }

  std::pair<std::unordered_set<const SearchNode*>::iterator,bool> result = graphPath.insert(node);
  // No insertion, node was already there, this means we hit a cycle in the graph
  if(!result.second) {
    //Just treat it as base case and immediately terminate.
    wlError += desiredProp * nnOutput->shorttermWinlossError;
    return;
  }

  ConstSearchNodeChildrenReference children = node->getChildren();
  int childrenCapacity = children.getCapacity();

  vector<MoreNodeStats> statsBuf;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    int64_t edgeVisits = childPointer.getEdgeVisits();
    Loc moveLoc = childPointer.getMoveLocRelaxed();
    MoreNodeStats stats;
    stats.stats = NodeStats(child->stats);
    stats.selfUtility = node->nextPla == P_WHITE ? stats.stats.utilityAvg : -stats.stats.utilityAvg;
    stats.weightAdjusted = stats.stats.getChildWeight(edgeVisits);
    stats.prevMoveLoc = moveLoc;
    statsBuf.push_back(stats);
  }
  int numChildren = (int)statsBuf.size();

  // Find all children and compute weighting of the children based on their values
  {
    double totalChildWeight = 0.0;
    for(int i = 0; i<numChildren; i++) {
      totalChildWeight += statsBuf[i].weightAdjusted;
    }
    const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
    if(searchParams.useNoisePruning) {
      for(int i = 0; i<numChildren; i++)
        policyProbsBuf[i] = std::max(1e-30, (double)policyProbs[getPos(statsBuf[i].prevMoveLoc)]);
      totalChildWeight = pruneNoiseWeight(statsBuf, numChildren, totalChildWeight, policyProbsBuf);
    }
    double amountToSubtract = 0.0;
    double amountToPrune = 0.0;
    downweightBadChildrenAndNormalizeWeight(
      numChildren, totalChildWeight, totalChildWeight,
      amountToSubtract, amountToPrune, statsBuf
    );
  }

  //What we actually weight the children by for averaging
  double relativeChildrenWeightSum = 0.0;
  //What the weights of the children sum to from the search.
  double childrenWeightSum = 0;
  for(int i = 0; i<numChildren; i++) {
    double childWeight = statsBuf[i].weightAdjusted;
    relativeChildrenWeightSum += childWeight;
    childrenWeightSum += childWeight;
  }
  double parentNNWeight = computeWeightFromNNOutput(nnOutput);
  parentNNWeight = std::max(parentNNWeight,1e-10);
  double desiredPropFromChildren = desiredProp * childrenWeightSum / (childrenWeightSum + parentNNWeight);
  double selfProp = desiredProp * parentNNWeight / (childrenWeightSum + parentNNWeight);

  // In multithreading we may sometimes have children but with no weight at all yet, in that case just use parent alone.
  if(desiredPropFromChildren <= 0.0 || relativeChildrenWeightSum <= 0.0) {
    selfProp += desiredPropFromChildren;
  }
  else {

    for(int i = 0; i<numChildren; i++) {
      const SearchNode* child = children[i].getIfAllocated();
      assert(child != NULL);
      double childWeight = statsBuf[i].weightAdjusted;
      double desiredPropFromChild = childWeight / relativeChildrenWeightSum * desiredPropFromChildren;
      getShallowAverageShorttermWLErrorHelper(child,graphPath,policyProbsBuf,minProp,desiredPropFromChild,wlError);
    }
  }

  graphPath.erase(node);

  // Also add in the direct evaluation of this node.
  {
    wlError += selfProp * nnOutput->shorttermWinlossError;
  }
}






bool Search::getAnalysisJson(
  const Player perspective,
  int analysisPVLen,
  bool preventEncore,
  bool includePolicy,
  bool includePVVisits,
  bool includeNoResultValue,
  json& ret
) const {
  vector<AnalysisData> buf;
  static constexpr int minMoves = 0;
  static constexpr int OUTPUT_PRECISION = 8;

  const Board& board = rootBoard;
  const BoardHistory& hist = rootHistory;
  bool duplicateForSymmetries = true;
  getAnalysisData(buf, minMoves, false, analysisPVLen, duplicateForSymmetries);

  const NNOutput* nnOutput = NULL;
  const NNOutput* humanOutput = NULL;
  if(rootNode != NULL) {
    nnOutput = rootNode->getNNOutput();
    humanOutput = rootNode->getHumanOutput();
  }

  // Stats for all the individual moves
  json moveInfos = json::array();
  for(int i = 0; i < buf.size(); i++) {
    const AnalysisData& data = buf[i];
    double winrate = 0.5 * (1.0 + data.winLossValue);
    double utility = data.utility;
    double lcb = PlayUtils::getHackedLCBForWinrate(this, data, rootPla);
    double utilityLcb = data.lcb;
    if(perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && rootPla == P_BLACK)) {
      winrate = 1.0 - winrate;
      lcb = 1.0 - lcb;
      utility = -utility;
      utilityLcb = -utilityLcb;
    }

    json moveInfo;
    moveInfo["move"] = moveToString(data.move);
    moveInfo["visits"] = data.childVisits;
    moveInfo["weight"] = Global::roundDynamic(data.childWeightSum,OUTPUT_PRECISION);
    moveInfo["utility"] = Global::roundDynamic(utility,OUTPUT_PRECISION);
    moveInfo["winrate"] = Global::roundDynamic(winrate,OUTPUT_PRECISION);
    moveInfo["drawValue"] = Global::roundDynamic(data.drawValue,OUTPUT_PRECISION);
    if(includeNoResultValue)
      moveInfo["noResultValue"] = Global::roundDynamic(data.noResultValue,OUTPUT_PRECISION);
    moveInfo["prior"] = Global::roundDynamic(data.policyPrior,OUTPUT_PRECISION);
    if(humanOutput != NULL)
      moveInfo["humanPrior"] = Global::roundDynamic(std::max(0.0,(double)humanOutput->policyProbs[getPos(data.move)]),OUTPUT_PRECISION);
    moveInfo["lcb"] = Global::roundDynamic(lcb,OUTPUT_PRECISION);
    moveInfo["utilityLcb"] = Global::roundDynamic(utilityLcb,OUTPUT_PRECISION);
    moveInfo["order"] = data.order;
    if(data.isSymmetryOf != Board::NULL_LOC)
      moveInfo["isSymmetryOf"] = moveToString(data.isSymmetryOf);
    moveInfo["edgeVisits"] = data.numVisits;
    moveInfo["edgeWeight"] = Global::roundDynamic(data.weightSum,OUTPUT_PRECISION);
    moveInfo["playSelectionValue"] = Global::roundDynamic(data.playSelectionValue,OUTPUT_PRECISION);

    json pv = json::array();
    int pvLen =
      (!muZeroEvaluator && preventEncore && data.pvContainsPass()) ? data.getPVLenUpToPhaseEnd(board, hist, rootPla) : (int)data.pv.size();
    for(int j = 0; j < pvLen; j++)
      pv.push_back(moveToString(data.pv[j]));
    moveInfo["pv"] = pv;

    if(includePVVisits) {
      testAssert(data.pvVisits.size() >= pvLen);
      json pvVisits = json::array();
      for(int j = 0; j < pvLen; j++)
        pvVisits.push_back(data.pvVisits[j]);
      moveInfo["pvVisits"] = pvVisits;

      testAssert(data.pvEdgeVisits.size() >= pvLen);
      json pvEdgeVisits = json::array();
      for(int j = 0; j < pvLen; j++)
        pvEdgeVisits.push_back(data.pvEdgeVisits[j]);
      moveInfo["pvEdgeVisits"] = pvEdgeVisits;
    }

    moveInfos.push_back(moveInfo);
  }
  ret["moveInfos"] = moveInfos;

  // Stats for root position
  {
    ReportedSearchValues rootVals;
    bool suc = getPrunedRootValues(rootVals);
    if(!suc)
      return false;

    double winloss = rootVals.winLossValue;
    double utility = rootVals.utility;
    double flipFactor = (perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && rootPla == P_BLACK)) ? -1.0 : 1.0;

    json rootInfo;
    rootInfo["visits"] = rootVals.visits;
    rootInfo["weight"] = rootVals.weight;
    rootInfo["winrate"] = Global::roundDynamic(0.5 + 0.5*winloss*flipFactor,OUTPUT_PRECISION);
    rootInfo["winValue"] = Global::roundDynamic(flipFactor > 0 ? rootVals.winValue : rootVals.lossValue,OUTPUT_PRECISION);
    rootInfo["lossValue"] = Global::roundDynamic(flipFactor > 0 ? rootVals.lossValue : rootVals.winValue,OUTPUT_PRECISION);
    rootInfo["drawValue"] = Global::roundDynamic(rootVals.drawValue,OUTPUT_PRECISION);
    rootInfo["noResultValue"] = Global::roundDynamic(rootVals.noResultValue,OUTPUT_PRECISION);
    rootInfo["utility"] = Global::roundDynamic(utility*flipFactor,OUTPUT_PRECISION);

    if(nnOutput != NULL) {
      rootInfo["rawWinrate"] = Global::roundDynamic(0.5 + 0.5*(nnOutput->whiteWinProb - nnOutput->whiteLossProb)*flipFactor,OUTPUT_PRECISION);
      rootInfo["rawDrawProb"] = Global::roundDynamic(nnOutput->whiteDrawProb,OUTPUT_PRECISION);
      rootInfo["rawNoResultProb"] = Global::roundDynamic(nnOutput->whiteNoResultProb,OUTPUT_PRECISION);
      rootInfo["rawStWrError"] = Global::roundDynamic(nnOutput->shorttermWinlossError * 0.5,OUTPUT_PRECISION);
      rootInfo["rawVarTimeLeft"] = Global::roundDynamic(nnOutput->varTimeLeft,OUTPUT_PRECISION);
    }
    if(humanOutput != NULL) {
      rootInfo["humanWinrate"] = Global::roundDynamic(0.5 + 0.5*(humanOutput->whiteWinProb - humanOutput->whiteLossProb)*flipFactor,OUTPUT_PRECISION);
      rootInfo["humanStWrError"] = Global::roundDynamic(humanOutput->shorttermWinlossError * 0.5,OUTPUT_PRECISION);
    }

    Hash128 thisHash;
    Hash128 symHash;
    if(hist.rules.isXiangqi()) {
      thisHash = hist.getXiangqiState().historyHash();
      symHash = std::min(thisHash,hist.getXiangqiState().mirroredHistoryHash());
    }
    else if(hist.rules.isPlacement()) {
      for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
        if(!hist.rules.allowsSymmetry(symmetry)) continue;
        const bool transpose = SymmetryHelpers::isTranspose(symmetry);
        Board symBoard(transpose ? board.y_size : board.x_size,transpose ? board.x_size : board.y_size);
        // Placement setups may contain zero-liberty stones; the Go symmetry
        // helper deliberately rejects them. Preserve colors without captures.
        for(int y = 0; y < board.y_size; y++) for(int x = 0; x < board.x_size; x++)
          symBoard.colors[SymmetryHelpers::getSymLoc(x,y,board,symmetry)] = board.colors[Location::getLoc(x,y,board.x_size)];
        symBoard.regenChainsFromColors();
        Hash128 hash = BoardHistory::getSituationRulesAndKoHash(symBoard,hist,rootPla,0.5);
        if(symmetry == 0) thisHash = symHash = hash;
        else if(hash < symHash) symHash = hash;
      }
    }
    else for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++) {
      Board symBoard = SymmetryHelpers::getSymBoard(board,symmetry);
      Hash128 hash = symBoard.getSitHashWithSimpleKo(rootPla);
      if(symmetry == 0) {
        thisHash = hash;
        symHash = hash;
      }
      else {
        if(hash < symHash)
          symHash = hash;
      }
    }
    rootInfo["thisHash"] = Global::uint64ToHexString(thisHash.hash1) + Global::uint64ToHexString(thisHash.hash0);
    rootInfo["symHash"] = Global::uint64ToHexString(symHash.hash1) + Global::uint64ToHexString(symHash.hash0);
    rootInfo["currentPlayer"] = PlayerIO::playerToStringShort(rootPla);

    ret["rootInfo"] = rootInfo;
  }

  // Raw policy prior
  if(includePolicy) {
    {
      SearchPolicyBuffer<float> policyProbsStorage(getPolicyBufferSize());
      float* policyProbs = policyProbsStorage.data();
      if(!hist.rules.isGo() && hist.isGameFinished)
        std::fill(policyProbs,policyProbs+policySize,-1.0f);
      else if(!getPolicy(policyProbs)) return false;
      json policy = json::array();
      if(hist.rules.isXiangqi()) {
        for(int pos = 0; pos < policySize; pos++)
          policy.push_back(Global::roundDynamic(policyProbs[pos],OUTPUT_PRECISION));
      }
      else {
        for(int y = 0; y < board.y_size; y++) {
          for(int x = 0; x < board.x_size; x++) {
            int pos = NNPos::xyToPos(x, y, nnXLen);
            policy.push_back(Global::roundDynamic(policyProbs[pos],OUTPUT_PRECISION));
          }
        }

        int passPos = NNPos::locToPos(Board::PASS_LOC, board.x_size, nnXLen, nnYLen);
        policy.push_back(Global::roundDynamic(policyProbs[passPos],OUTPUT_PRECISION));
      }
      ret["policy"] = policy;
    }

    if(humanOutput != NULL) {
      const float* policyProbs = humanOutput->getPolicyProbsMaybeNoised();
      json policy = json::array();
      if(hist.rules.isXiangqi()) {
        for(int pos = 0; pos < policySize; pos++)
          policy.push_back(Global::roundDynamic(policyProbs[pos],OUTPUT_PRECISION));
      }
      else {
        for(int y = 0; y < board.y_size; y++) {
          for(int x = 0; x < board.x_size; x++) {
            int pos = NNPos::xyToPos(x, y, nnXLen);
            policy.push_back(Global::roundDynamic(policyProbs[pos],OUTPUT_PRECISION));
          }
        }

        int passPos = NNPos::locToPos(Board::PASS_LOC, board.x_size, nnXLen, nnYLen);
        policy.push_back(Global::roundDynamic(policyProbs[passPos],OUTPUT_PRECISION));
      }
      ret["humanPolicy"] = policy;
    }
  }

  return true;
}

//Compute all the stats of the node based on its children, pruning weights such that they are as expected
//based on policy and utility. This is used to give accurate rootInfo even with a lot of wide root noise
bool Search::getPrunedRootValues(ReportedSearchValues& values) const {
  return getPrunedNodeValues(rootNode,values);
}

bool Search::getPrunedNodeValues(const SearchNode* nodePtr, ReportedSearchValues& values) const {
  // Gumbel's allocation is not forced PUCT exploration to be pruned retrospectively.
  if(searchParams.useGumbel) return getNodeValues(nodePtr,values);
  if(nodePtr == NULL)
    return false;
  const SearchNode& node = *nodePtr;

  ConstSearchNodeChildrenReference children = node.getChildren();
  int childrenCapacity = children.getCapacity();

  vector<double> playSelectionValues;
  vector<Loc> locs; // not used
  const bool allowDirectPolicyMoves = false;
  const bool alwaysComputeLcb = false;
  const bool neverUseLcb = true;
  bool suc = getPlaySelectionValues(node,locs,playSelectionValues,NULL,1.0,allowDirectPolicyMoves,alwaysComputeLcb,neverUseLcb,NULL,NULL);
  //If there are no children, or otherwise values could not be computed,
  //then fall back to the normal case and just listen to the values on the node rather than trying
  //to recompute things.
  if(!suc) {
    return getNodeValues(nodePtr,values);
  }

  double winLossValueSum = 0.0;
  double drawValueSum = 0.0;
  double noResultValueSum = 0.0;
  double utilitySum = 0.0;
  double weightSum = 0.0;
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    int64_t edgeVisits = childPointer.getEdgeVisits();
    NodeStats stats = NodeStats(child->stats);

    if(stats.visits <= 0 || stats.weightSum <= 0.0 || edgeVisits <= 0)
      continue;
    double weight = playSelectionValues[i];
    winLossValueSum += weight * stats.winLossValueAvg;
    drawValueSum += weight * stats.drawValueAvg;
    noResultValueSum += weight * stats.noResultValueAvg;
    utilitySum += weight * stats.utilityAvg;
    weightSum += weight;
  }

  //Also add in the direct evaluation of this node.
  {
    const NNOutput* nnOutput = node.getNNOutput();
    //If somehow the nnOutput is still null here, skip
    if(nnOutput == NULL)
      return false;
    double winProb = (double)nnOutput->whiteWinProb;
    double lossProb = (double)nnOutput->whiteLossProb;
    double drawProb = (double)nnOutput->whiteDrawProb;
    double noResultProb = (double)nnOutput->whiteNoResultProb;
    double utility =
      getResultUtility(winProb-lossProb, noResultProb);

    double weight = computeWeightFromNNOutput(nnOutput);
    winLossValueSum += (winProb - lossProb) * weight;
    drawValueSum += drawProb * weight;
    noResultValueSum += noResultProb * weight;
    utilitySum += utility * weight;
    weightSum += weight;
  }
  values = ReportedSearchValues(
    winLossValueSum / weightSum,
    drawValueSum / weightSum,
    noResultValueSum / weightSum,
    utilitySum / weightSum,
    node.stats.weightSum.load(std::memory_order_acquire),
    node.stats.visits.load(std::memory_order_acquire)
  );
  return true;
}

void Search::debugPrintChildrenSummary(std::ostream& out, const SearchNode& node, NNOutput* nnOutput) {
  SearchNodeState nodeState = node.state.load(std::memory_order_acquire);
  int numChildren = 0;
  ConstSearchNodeChildrenReference children = node.getChildren(nodeState);
  int childrenCapacity = children.getCapacity();
  out << "childrenCapacity " << childrenCapacity << endl;
  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
  for(int i = 0; i<childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    numChildren += 1;
    Loc moveLoc = childPointer.getMoveLocRelaxed();
    int movePos = getPos(moveLoc);
    float nnPolicyProb = policyProbs[movePos];
    int64_t edgeVisits = childPointer.getEdgeVisits();
    double childWeight = child->stats.getChildWeight(edgeVisits);
    out << i << " " << moveToString(moveLoc) << " " << nnPolicyProb << " " << edgeVisits << " " << childWeight << endl;
  }
  out << "numChildren " << numChildren << endl;
}
