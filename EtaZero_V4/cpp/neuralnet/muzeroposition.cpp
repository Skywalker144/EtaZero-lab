#include "muzeroposition.h"
#include "modelversion.h"
#include "nnpostprocess.h"
#include "../core/test.h"
#include "../game/xiangqi.h"

namespace MuZero {
PositionEvaluator::PositionEvaluator(Batcher& service, bool exact, bool randomize_, int symmetry,
                                     int cachePower, int mutexPower, Logger* logger_, const std::string& symmetrySeed_)
  : batcher(service), model(service.model()), requireExactNNLen(exact), symmetrySeed(symmetrySeed_), randomize(randomize_),
    defaultSymmetry(symmetry), logger(logger_), postprocess(model.predictionDesc().postProcessParams) {
  // The raw MuZero backend already restores scale8 before returning CPU logits.
  postprocess.outputScaleMultiplier = 1.0f;
  if(cachePower >= 0) cache = std::make_unique<NNCacheTable>(cachePower,mutexPower);
}

std::shared_ptr<Evaluation> PositionEvaluator::finish(Output output, Player nextPlayer,
    std::shared_ptr<const RootOrientation> orientation, const bool* legal,
    const MiscNNInputParams& inputParams, PositionBuffer& buffer) const {
  const int canvas = model.posLen(), area = canvas*canvas;
  const bool xiangqi = model.game() == "xiangqi";
  const int actions = xiangqi ? Xiangqi::NUM_ACTIONS : area+(model.game() == "go");
  MiscNNInputParams params = inputParams;
  if(model.representationDesc().numInputMetaChannels > 0) params.policyOptimism = 0.0;
  auto result = std::make_shared<Evaluation>();
  result->state = std::move(output.state);
  result->orientation = std::move(orientation);
  result->nextPlayer = nextPlayer;
  result->nnXLen = result->nnYLen = canvas;
  const float optimism = (float)params.policyOptimism;
  if(xiangqi) {
    result->resizePolicy(actions);
    for(int action = 0; action < actions; action++) {
      const int oriented = Xiangqi::transformAction(action,result->orientation->rootPlayer,result->orientation->symmetry == 2);
      const float p = output.policy[oriented];
      result->policyProbs[action] = model.predictionDesc().numPolicyChannels > 1 ?
        p+(output.policy[actions+oriented]-p)*optimism : p;
    }
  }
  else {
    float mixed[NNPos::MAX_NN_POLICY_SIZE];
    for(int i = 0; i < actions; i++) {
      float p = output.policy[i];
      mixed[i] = model.predictionDesc().numPolicyChannels > 1 ? p+(output.policy[actions+i]-p)*optimism : p;
    }
    SymmetryHelpers::copyOutputsWithSymmetry(mixed,result->policyProbs,1,canvas,canvas,result->orientation->symmetry);
    result->policyProbs[area] = model.game() == "go" ? mixed[area] : -1e30f;
  }
  result->whiteWinProb = output.value[0];
  result->whiteDrawProb = output.value[1];
  result->whiteLossProb = output.value[2];
  result->whiteNoResultProb = output.value[3];
  result->varTimeLeft = output.auxiliary[0];
  result->shorttermWinlossError = output.auxiliary[1];
  NNPostprocess::policy(*result,legal,xiangqi ? actions : area+1,postprocess.outputScaleMultiplier,
    params.nnPolicyTemperature,params.policyOptimism,buffer.errorLogLockout,logger,model.predictionDesc().name);
  NNPostprocess::value(*result,nextPlayer,result->orientation->suppressNoResult,
    model.predictionDesc().modelVersion,postprocess);
  return result;
}

std::shared_ptr<Evaluation> PositionEvaluator::initial(const Board& board, const BoardHistory& history,
    Player nextPlayer, const SGFMetadata* metadata, const MiscNNInputParams& inputParams,
    PositionBuffer& buffer, Rand& rand, bool skipCache) {
  const int canvas = model.posLen(), area = canvas*canvas;
  const auto& desc = model.representationDesc();
  if(board.x_size > canvas || board.y_size > canvas ||
     (requireExactNNLen && (board.x_size != canvas || board.y_size != canvas)))
    throw StringError("MuZero board dimensions differ from the configured NN canvas");
  if(history.rules.game != Rules::parseGame(model.game()))
    throw StringError("MuZero model game differs from the requested position");
  const bool xiangqi = history.rules.isXiangqi();
  if(xiangqi && history.getXiangqiState().result().finished)
    throw StringError("Cannot evaluate a terminal Xiangqi position with no legal policy");
  testAssert(nextPlayer == history.presumedNextMovePla);
  const bool doRandomize = randomize.load(std::memory_order_acquire);
  const int fallback = defaultSymmetry.load(std::memory_order_acquire);
  if(inputParams.symmetry != NNInputs::SYMMETRY_NOTSPECIFIED &&
     (inputParams.symmetry < 0 || inputParams.symmetry >= SymmetryHelpers::NUM_SYMMETRIES || !history.rules.allowsSymmetry(inputParams.symmetry)))
    throw StringError("Requested neural-net symmetry is invalid for the game");
  if(inputParams.symmetry == NNInputs::SYMMETRY_NOTSPECIFIED && !doRandomize &&
     (fallback < 0 || fallback >= SymmetryHelpers::NUM_SYMMETRIES || !history.rules.allowsSymmetry(fallback)))
    throw StringError("Configured neural-net symmetry is invalid for the game");
  MiscNNInputParams params = inputParams;
  if(desc.numInputMetaChannels > 0) params.policyOptimism = 0.0;
  Hash128 hash = NNInputs::getHash(board,history,nextPlayer,params);
  if(desc.numInputMetaChannels > 0) {
    if(metadata == nullptr || !metadata->initialized)
      throw StringError("Initialized SGFMetadata is required for this MuZero representation");
    hash ^= metadata->getHash(nextPlayer);
  }
  if(cache && !skipCache) {
    std::shared_ptr<NNOutput> cached;
    if(cache->get(hash,cached)) {
      cacheHits.fetch_add(1,std::memory_order_relaxed);
      // This private table only stores make_shared<Evaluation> entries. The
      // shared control block retains the derived destructor and device state.
      return std::static_pointer_cast<Evaluation>(cached);
    }
  }
  bool pointLegal[NNPos::MAX_NN_POLICY_SIZE];
  if(xiangqi && !buffer.jointLegal) buffer.jointLegal = std::make_unique<bool[]>(Xiangqi::NUM_ACTIONS);
  bool* legal = xiangqi ? buffer.jointLegal.get() : pointLegal;
  int legalCount = 0;
  if(xiangqi) {
    std::fill_n(legal,Xiangqi::NUM_ACTIONS,false);
    for(int action: history.getXiangqiState().legalActions()) { legal[action] = true; legalCount++; }
  }
  else for(int i = 0; i <= area; i++) {
    legal[i] = history.isLegal(board,NNPos::posToLoc(i,board.x_size,board.y_size,canvas,canvas),nextPlayer);
    legalCount += legal[i];
  }
  if(legalCount == 0) throw StringError("MuZero root has no legal actions");
  buffer.spatial.resize((size_t)desc.numInputChannels*area);
  buffer.global.resize(desc.numInputGlobalChannels);
  buffer.metadata.resize(desc.numInputMetaChannels);
  buffer.transformed.resize(buffer.spatial.size());
  NNInputs::fillRow(NNModelVersion::getInputsVersion(desc.modelVersion),board,history,nextPlayer,
                    params,canvas,canvas,false,buffer.spatial.data(),buffer.global.data());
  if(desc.numInputMetaChannels > 0)
    SGFMetadata::fillMetadataRow(metadata,buffer.metadata.data(),nextPlayer,board.x_size*board.y_size);
  int symmetry = params.symmetry;
  if(symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
    Rand* symmetrySource = &rand;
    if(doRandomize && !symmetrySeed.empty()) {
      if(!buffer.symmetryRand)
        buffer.symmetryRand = std::make_unique<Rand>(symmetrySeed + "$" + Global::uint64ToString(rand.nextUInt64()));
      symmetrySource = buffer.symmetryRand.get();
    }
    symmetry = doRandomize ? (xiangqi ? (int)symmetrySource->nextUInt(2)*2 :
      history.rules.game == Rules::Game::HEX ? (int)symmetrySource->nextUInt(2)*3 : (int)symmetrySource->nextUInt(8)) : fallback;
  }
  if(xiangqi)
    Xiangqi::copyInputsWithMirror(buffer.spatial.data(),buffer.transformed.data(),canvas,canvas,desc.numInputChannels,false,symmetry == 2);
  else
    SymmetryHelpers::copyInputsWithSymmetry(buffer.spatial.data(),buffer.transformed.data(),
      1,canvas,canvas,desc.numInputChannels,false,symmetry);
  const bool suppressNoResult = !history.rules.isGo() ||
    (history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY);
  auto orientation = std::make_shared<RootOrientation>(RootOrientation{symmetry,board.x_size,board.y_size,suppressNoResult,nextPlayer});
  auto result = finish(batcher.initial(buffer.transformed,buffer.global,buffer.metadata),
                       nextPlayer,std::move(orientation),legal,params,buffer);
  result->nnHash = hash;
  if(cache) cache->set(result); // Like source, skipCache bypasses reads, not writes.
  return result;
}

std::shared_ptr<Evaluation> PositionEvaluator::recurrent(const Evaluation& parent, int action,
    const MiscNNInputParams& params, PositionBuffer& buffer) {
  if(!parent.orientation) throw StringError("MuZero recurrent state has no root orientation");
  const auto& root = *parent.orientation;
  const int canvas = model.posLen(), area = canvas*canvas;
  const bool xiangqi = model.game() == "xiangqi";
  bool pointLegal[NNPos::MAX_NN_POLICY_SIZE];
  if(xiangqi && !buffer.jointLegal) buffer.jointLegal = std::make_unique<bool[]>(Xiangqi::NUM_ACTIONS);
  bool* legal = xiangqi ? buffer.jointLegal.get() : pointLegal;
  int transformedAction = action;
  if(xiangqi) {
    if(action < 0 || action >= Xiangqi::NUM_ACTIONS)
      throw StringError("MuZero recurrent action outside the static root domain");
    std::fill_n(legal,Xiangqi::NUM_ACTIONS,true);
    transformedAction = Xiangqi::transformAction(action,root.rootPlayer,root.symmetry == 2);
  }
  else {
    for(int i = 0; i < area; i++) legal[i] = i%canvas < root.xSize && i/canvas < root.ySize;
    legal[area] = model.game() == "go";
    if(action < 0 || action > area || !legal[action])
      throw StringError("MuZero recurrent action outside the static root domain");
    if(action < area) {
      const Loc transformed = SymmetryHelpers::getSymLoc(action%canvas,action/canvas,canvas,canvas,root.symmetry);
      transformedAction = NNPos::locToPos(transformed,canvas,canvas,canvas);
    }
  }
  return finish(batcher.recurrent(parent.state,transformedAction),getOpp(parent.nextPlayer),
                parent.orientation,legal,params,buffer);
}

std::shared_ptr<Evaluation> PositionEvaluator::averageMultipleSymmetries(const Board& board, const BoardHistory& history,
    Player nextPlayer, const SGFMetadata* metadata, const MiscNNInputParams& baseParams,
    PositionBuffer& buffer, Rand& rand, int count) {
  if(count < 1) throw StringError("MuZero symmetry sample count must be positive");
  std::array<int,SymmetryHelpers::NUM_SYMMETRIES> indexes;
  int allowed = 0;
  for(int symmetry = 0; symmetry < SymmetryHelpers::NUM_SYMMETRIES; symmetry++)
    if(history.rules.allowsSymmetry(symmetry)) indexes[allowed++] = symmetry;
  count = std::min(count,allowed);
  std::vector<std::shared_ptr<NNOutput>> outputs;
  std::shared_ptr<Evaluation> result;
  MiscNNInputParams params = baseParams;
  for(int i = 0; i < count; i++) {
    std::swap(indexes[i],indexes[rand.nextInt(i,allowed-1)]);
    params.symmetry = indexes[i];
    auto output = initial(board,history,nextPlayer,metadata,params,buffer,rand,true);
    // Keep one uniformly sampled orientation for subsequent latent transitions;
    // nonlinear hidden states are never averaged or transformed between steps.
    if(i == 0) result = std::make_shared<Evaluation>(*output);
    outputs.push_back(std::move(output));
  }
  static_cast<NNOutput&>(*result) = NNOutput(outputs);
  return result;
}

void PositionEvaluator::clearCache() { if(cache) cache->clear(); }
uint64_t PositionEvaluator::numCacheHits() const { return cacheHits.load(std::memory_order_relaxed); }
void PositionEvaluator::setDoRandomize(bool enabled) { randomize.store(enabled,std::memory_order_release); }
void PositionEvaluator::setDefaultSymmetry(int symmetry) { defaultSymmetry.store(symmetry,std::memory_order_release); }
}
