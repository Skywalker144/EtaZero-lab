#include "../search/search.h"

#include "../search/searchnode.h"

//------------------------
#include "../core/using.h"
//------------------------

void Search::computeRootNNEvaluation(NNResultBuf& nnResultBuf) {
  Board board = rootBoard;
  const BoardHistory& hist = rootHistory;
  Player pla = rootPla;
  bool skipCache = false;
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
  nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
  nnInputParams.policyOptimism = searchParams.rootPolicyOptimism;
  if(searchParams.playoutDoublingAdvantage != 0) {
    Player playoutDoublingAdvantagePla = getPlayoutDoublingAdvantagePla();
    nnInputParams.playoutDoublingAdvantage = (
      getOpp(pla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
    );
  }
  if(searchParams.ignorePreRootHistory || searchParams.ignoreAllHistory)
    nnInputParams.maxHistory = 0;
  if(muZeroEvaluator) {
    MuZero::PositionBuffer buffer;
    nnResultBuf.result = muZeroEvaluator->initial(board,hist,pla,&searchParams.humanSLProfile,
      nnInputParams,buffer,nonSearchRand,skipCache);
    nnResultBuf.hasResult = true;
  }
  else {
  nnEvaluator->evaluate(
    board, hist, pla, &searchParams.humanSLProfile,
    nnInputParams,
    nnResultBuf, skipCache);
  }
}

MiscNNInputParams Search::paramsForHumanEvaluator(const MiscNNInputParams& nnInputParams) const {
  MiscNNInputParams humanNNInputParams = nnInputParams;
  if(humanMuZeroEvaluator) {
    const BoardHistoryModes modes = resolveHistoryModes(searchParams,*humanMuZeroEvaluator);
    humanNNInputParams.passAliveSuicideRulesOverride = modes.alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
    humanNNInputParams.excludeTerritoryAdjAtariOverride = modes.excludeTerritoryAdjacentToAtari ? 1 : 0;
    return humanNNInputParams;
  }
  //Featurize for the human net per its own resolution (its own declaration when the params are auto),
  //which may differ from the modes the search itself is using.
  humanNNInputParams.passAliveSuicideRulesOverride =
    resolveAlwaysComputePassAliveUnderSuicideRules(searchParams, humanEvaluator) ? 1 : 0;
  humanNNInputParams.excludeTerritoryAdjAtariOverride =
    resolveExcludeTerritoryAdjacentToAtari(searchParams, humanEvaluator) ? 1 : 0;
  return humanNNInputParams;
}

bool Search::needsHumanOutputAtRoot() const {
  if(humanMuZeroEvaluator)
    return searchParams.humanSLProfile.initialized || humanMuZeroEvaluator->getModel().representationDesc().numInputMetaChannels == 0;
  return humanEvaluator != NULL && (searchParams.humanSLProfile.initialized || !humanEvaluator->requiresSGFMetadata());
}
bool Search::needsHumanOutputInTree() const {
  return needsHumanOutputAtRoot() && searchParams.hasHumanSLTreeExploration();
}

//If isReInit is false, among any threads trying to store, the first one wins
//If isReInit is true, we always replace, even for threads that come later.
//Returns true if a nnOutput was set where there was none before.
bool Search::initNodeNNOutput(
  SearchThread& thread, SearchNode& node,
  bool isRoot, bool skipCache, bool isReInit
) {
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
  nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
  nnInputParams.policyOptimism = isRoot ? searchParams.rootPolicyOptimism : searchParams.policyOptimism;
  if(searchParams.playoutDoublingAdvantage != 0) {
    Player playoutDoublingAdvantagePla = getPlayoutDoublingAdvantagePla();
    nnInputParams.playoutDoublingAdvantage = (
      getOpp(thread.pla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
    );
  }
  if(searchParams.ignoreAllHistory)
    nnInputParams.maxHistory = 0;
  else if(searchParams.ignorePreRootHistory) {
    nnInputParams.maxHistory = isRoot ? 0 : std::max(0, (int)thread.history.moveHistory.size() - (int)rootHistory.moveHistory.size());
  }

  std::shared_ptr<NNOutput>* result = NULL;
  std::shared_ptr<NNOutput>* humanResult = NULL;
  if(muZeroEvaluator) {
    // A source cache collision can request a root reevaluation. Changing its
    // latent while other threads use descendants would mix different trees.
    // Fresh searches never need an in-place refresh; require a clean restart.
    if(isReInit)
      throw StringError("MuZero cannot replace an active root latent; clear the root cache and restart search");
    std::shared_ptr<MuZero::Evaluation> output;
    if(isRoot) {
      if(searchParams.rootNumSymmetriesToSample > 1)
        output = muZeroEvaluator->averageMultipleSymmetries(thread.board,thread.history,thread.pla,
          &searchParams.humanSLProfile,nnInputParams,thread.muZeroBuffer,thread.rand,searchParams.rootNumSymmetriesToSample);
      else
        output = muZeroEvaluator->initial(thread.board,thread.history,thread.pla,&searchParams.humanSLProfile,
          nnInputParams,thread.muZeroBuffer,thread.rand,skipCache);
    }
    else {
      testAssert(thread.muZeroParent != nullptr);
      output = muZeroEvaluator->recurrent(*thread.muZeroParent,thread.muZeroAction,nnInputParams,thread.muZeroBuffer);
    }
    result = new std::shared_ptr<NNOutput>(std::move(output));
  }
  else if(isRoot && searchParams.rootNumSymmetriesToSample > 1) {
    result = nnEvaluator->averageMultipleSymmetries(
      thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
      nnInputParams,
      thread.nnResultBuf,
      thread.rand, searchParams.rootNumSymmetriesToSample
    );
  }
  else {
    nnEvaluator->evaluate(
      thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
      nnInputParams,
      thread.nnResultBuf, skipCache);
    result = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
  }

  try {
    if(needsHumanOutputInTree() || (isRoot && needsHumanOutputAtRoot())) {
      if(humanMuZeroEvaluator) {
        std::shared_ptr<MuZero::Evaluation> output;
        const MiscNNInputParams humanParams = paramsForHumanEvaluator(nnInputParams);
        if(isRoot || !muZeroEvaluator) {
          //Ordinary search has a real position at every node. Only latent
          //search advances a secondary from its own retained parent state.
          if(isRoot && searchParams.rootNumSymmetriesToSample > 1)
            output = humanMuZeroEvaluator->averageMultipleSymmetries(thread.board,thread.history,thread.pla,
              &searchParams.humanSLProfile,humanParams,thread.humanMuZeroBuffer,thread.rand,searchParams.rootNumSymmetriesToSample);
          else
            output = humanMuZeroEvaluator->initial(thread.board,thread.history,thread.pla,&searchParams.humanSLProfile,
              humanParams,thread.humanMuZeroBuffer,thread.rand,skipCache);
        }
        else {
          testAssert(thread.humanMuZeroParent != nullptr);
          output = humanMuZeroEvaluator->recurrent(*thread.humanMuZeroParent,thread.muZeroAction,humanParams,thread.humanMuZeroBuffer);
        }
        humanResult = new std::shared_ptr<NNOutput>(std::move(output));
      }
      else if(isRoot && searchParams.rootNumSymmetriesToSample > 1) {
        humanResult = humanEvaluator->averageMultipleSymmetries(
          thread.board,thread.history,thread.pla,&searchParams.humanSLProfile,paramsForHumanEvaluator(nnInputParams),
          thread.nnResultBuf,thread.rand,searchParams.rootNumSymmetriesToSample);
      }
      else {
        //An ordinary secondary is queried on real positions only. The MuZero
        //root-only restriction is validated before beginning the search.
        humanEvaluator->evaluate(thread.board,thread.history,thread.pla,&searchParams.humanSLProfile,
          paramsForHumanEvaluator(nnInputParams),thread.nnResultBuf,skipCache);
        humanResult = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
      }
    }
  }
  catch(...) {
    //In particular, do not retain the primary device latent if the secondary
    //evaluation fails before either result is published to the node.
    delete result;
    throw;
  }

  assert((*result)->noisedPolicyProbs == NULL);
  std::shared_ptr<NNOutput>* noisedResult = maybeAddPolicyNoiseAndTemp(thread,isRoot,result->get());
  if(noisedResult != NULL) {
    std::shared_ptr<NNOutput>* tmp = result;
    result = noisedResult;
    delete tmp;
  }

  node.nodeAge.store(searchNodeAge,std::memory_order_release);
  //If this is a re-initialization of the nnOutput, we don't want to add any visits or anything.
  //Also don't bother updating any of the stats. Technically we should do so because winLossValueSum
  //and such will have changed potentially due to a new orientation of the neural net eval
  //slightly affecting the evals, but this is annoying to recompute from scratch, and on the next
  //visit updateStatsAfterPlayout should fix it all up anyways.
  if(isReInit) {
    if(humanResult != NULL)
      node.storeHumanOutput(humanResult,thread); // ignore the wasNullBefore from this one
    bool wasNullBefore = node.storeNNOutput(result,thread);
    return wasNullBefore;
  }
  else {
    // Store human result first, so that the presence of the main result guarantees
    // that the human result exists in the case we have a human evaluator.
    if(humanResult != NULL) {
      bool humanSuc = node.storeHumanOutputIfNull(humanResult);
      if(!humanSuc)
        delete humanResult;
    }
    bool suc = node.storeNNOutputIfNull(result);
    if(!suc)
      delete result;
    else {
      addCurrentNNOutputAsLeafValue(node,true);
    }
    return suc;
  }
}


//Assumes node already has an nnOutput
bool Search::maybeRecomputeExistingNNOutput(
  SearchThread& thread, SearchNode& node, bool isRoot
) {
  bool recomputeHappened = false;
  //Right now only the root node currently ever needs to recompute, and only if it's old
  if(isRoot && node.nodeAge.load(std::memory_order_acquire) != searchNodeAge) {
    //See if we're the lucky thread that gets to do the update!
    //Threads that pass by here later will NOT wait for us to be done before proceeding with search.
    //We accept this and tolerate that for a few iterations potentially we will be using the OLD policy - without noise,
    //or without root temperature, etc.
    //Or if we have none of those things, then we'll not end up updating anything except the age, which is okay too.
    uint32_t oldAge = node.nodeAge.exchange(searchNodeAge,std::memory_order_acq_rel);
    if(oldAge < searchNodeAge) {
      NNOutput* nnOutput = node.getNNOutput();
      NNOutput* humanOutput = node.getHumanOutput();
      testAssert(nnOutput != NULL);

      //If averaging a bunch of symmetries, then we need to recompute it too
      //If root needs different optimism, we need to recompute it.
      //If humanSL is missing, but we want it, we need to recompute.
      //Also do so when ignoring history pre root
      if(searchParams.rootNumSymmetriesToSample > 1 ||
         searchParams.rootPolicyOptimism != searchParams.policyOptimism ||
         (searchParams.ignorePreRootHistory && !searchParams.ignoreAllHistory) ||
         (humanOutput == NULL && needsHumanOutputAtRoot())
      ) {
        //We *can* use cached evaluations even though parameters are changing, because:
        //Symmetry averaging skips the cache on its own when it does symmetry sampling without replacement
        //The optimism is part of the nn hash
        //When pre-root history is ignored at the root, maxHistory is 0 and the nn cache distinguishes 0 from nonzero.
        const bool skipCache = false;
        initNodeNNOutput(thread,node,isRoot,skipCache,true);
        recomputeHappened = true;
      }
      //We also need to recompute the root nn if we have root noise or temperature and that's missing.
      else {
        //We don't need to go all the way to the nnEvaluator, we just need to maybe add those transforms
        //to the existing policy.
        std::shared_ptr<NNOutput>* result = maybeAddPolicyNoiseAndTemp(thread,isRoot,nnOutput);
        if(result != NULL) {
          node.storeNNOutput(result,thread);
          recomputeHappened = true;
        }
      }
    }
  }
  return recomputeHappened;
}
