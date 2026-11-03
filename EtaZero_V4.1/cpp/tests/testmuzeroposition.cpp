#include "tests.h"
#include "../neuralnet/muzeroposition.h"
#include "../game/xiangqi.h"
#include "../external/nlohmann_json/json.hpp"
#include <fstream>

namespace {
using json = nlohmann::json;
json outputJSON(const MuZero::Evaluation& output) {
  const int size = output.getPolicyStorageSize() == Xiangqi::NUM_ACTIONS ? Xiangqi::NUM_ACTIONS : output.nnXLen*output.nnYLen+1;
  for(int i = size; i < NNPos::MAX_NN_POLICY_SIZE; i++) testAssert(output.policyProbs[i] == -1);
  return {{"policy",std::vector<float>(output.policyProbs,output.policyProbs+size)},
    {"value",{output.whiteWinProb,output.whiteDrawProb,output.whiteLossProb,output.whiteNoResultProb}},
    {"auxiliary",{output.varTimeLeft,output.shorttermWinlossError}},
    {"optimism",output.policyOptimismUsed},{"symmetry",output.orientation->symmetry},{"player",output.nextPlayer},
    {"root_player",output.orientation->rootPlayer}};
}
void reject(const std::function<void()>& call) {
  bool rejected = false;
  try { call(); } catch(const StringError&) { rejected = true; }
  testAssert(rejected);
}
}

void Tests::runMuZeroPosition(const std::string& directory, const std::string& request, const std::string& response) {
  Board::initHash();
  std::ifstream in(request);
  const json input = json::parse(in);
  if(input.contains("random_game")) {
    auto model = std::make_shared<MuZero::ModelBundle>(input.at("random_game").get<std::string>(),input.at("canvas").get<int>());
    MuZero::PositionEvaluator evaluator(model,false,false,0,4,2,nullptr,"random-position");
    const bool xiangqi = model->game() == "xiangqi", go = model->game() == "go";
    const int canvas = model->posLen(), area = canvas*canvas;
    Rules rules = Rules::forGame(Rules::parseGame(model->game()));
    if(go) rules.koRule = Rules::KO_SIMPLE;
    Board board(xiangqi ? 9 : 4,xiangqi ? 10 : 4);
    const BoardHistoryModes modes(false,false);
    BoardHistory history = xiangqi ? BoardHistory(Xiangqi::State(),modes) : BoardHistory(board,P_BLACK,rules,0,modes);
    if(xiangqi) board = history.getRecentBoard(0);
    const int action = xiangqi ? Xiangqi::parseAction("b0c2") : 0;
    history.makeBoardMoveAssumeLegal(board,xiangqi ? Xiangqi::actionToLoc(action) : Location::getLoc(0,0,4),P_BLACK,nullptr);
    MuZero::PositionBuffer buffer;
    Rand rand("random-position");
    MiscNNInputParams params;
    params.policyOptimism = 1.0;
    auto root = evaluator.initial(board,history,P_WHITE,nullptr,params,buffer,rand,false);
    testAssert(root->policyProbs[action] == -1 && !root->state);
    testAssert(evaluator.initial(board,history,P_WHITE,nullptr,params,buffer,rand,false) == root);
    testAssert(evaluator.numCacheHits() == 1);
    // Recurrence may start on another search thread with a fresh buffer. Even
    // repeated occupied actions stay in the latent tree's static vocabulary.
    MuZero::PositionBuffer otherThread;
    auto current = root;
    json outputs = json::array({outputJSON(*root)});
    for(int depth = 0; depth < 4; depth++) {
      current = evaluator.recurrent(*current,action,params,otherThread);
      testAssert(current->orientation == root->orientation && !current->state);
      testAssert(current->policyProbs[action] >= 0);
      const int count = xiangqi ? Xiangqi::NUM_ACTIONS : area+1;
      for(int i = 0; i < count; i++) {
        const bool allowed = xiangqi || (i == area ? go : i%canvas < 4 && i/canvas < 4);
        testAssert((current->policyProbs[i] >= 0) == allowed);
      }
      outputs.push_back(outputJSON(*current));
    }
    reject([&]() { evaluator.recurrent(*root,xiangqi ? Xiangqi::NUM_ACTIONS : 4,params,buffer); });
    std::ofstream out(response);
    out << outputs.dump();
    return;
  }
  ConfigParser config(input.value("cuda_config",std::map<std::string,std::string>{}));
  auto model = std::make_shared<MuZero::ModelBundle>(directory);
  MuZero::Batcher batcher(model,8,{0,0},input.at("fp16").get<bool>() ? enabled_t::True : enabled_t::False,config,nullptr,input.value("inputs_nhwc",true));
  MuZero::PositionEvaluator evaluator(batcher,false,false,0,4,2);
  MuZero::PositionBuffer buffer;
  SGFMetadata metadata = SGFMetadata::makeDummyWarmupProfile();
  const bool go = model->game() == "go";
  const bool xiangqi = model->game() == "xiangqi";
  const int alternateSymmetry = xiangqi ? 2 : 3;
  const bool hasMetadata = model->representationDesc().numInputMetaChannels > 0;
  const auto* meta = hasMetadata ? &metadata : nullptr;
  const int canvas = model->posLen(), area = canvas*canvas;
  Rand rand("muzero-position");
  json result = {{"cases",json::array()},{"averages",json::array()}};
  for(int nr = 0; nr < (go ? 2 : 1); nr++) {
    Rules rules = Rules::forGame(Rules::parseGame(model->game()));
    if(go) { rules = Rules::getTrompTaylorish(); rules.koRule = nr ? Rules::KO_SIMPLE : Rules::KO_POSITIONAL; }
    for(int player = 0; player < 2; player++) {
      Board board(xiangqi ? 9 : 4,xiangqi ? 10 : go ? 3 : 4);
      const auto& desc = model->representationDesc();
      const BoardHistoryModes modes(desc.preferPassAliveUnderSuicideRules,desc.preferExcludeTerritoryAdjacentToAtari);
      BoardHistory history = xiangqi ? BoardHistory(Xiangqi::State(),modes) : BoardHistory(board,P_BLACK,rules,0,modes);
      if(xiangqi) board = history.getRecentBoard(0);
      const char* opening[] = {"b0c2","b9c7","a3a4"};
      for(int i = 0; i < 2+player; i++) {
        const Loc move = xiangqi ? Xiangqi::actionToLoc(Xiangqi::parseAction(opening[i])) : Location::getLoc(i,0,board.x_size);
        testAssert(history.isLegal(board,move,history.presumedNextMovePla));
        history.makeBoardMoveAssumeLegal(board,move,history.presumedNextMovePla,nullptr);
      }
      const Player pla = history.presumedNextMovePla;
      MiscNNInputParams params;
      params.nnPolicyTemperature = 0.7f;
      params.policyOptimism = 0.35;
      params.playoutDoublingAdvantage = 0.5;
      std::vector<int> symmetries;
      for(int s = 0; s < 8; s++) if(rules.allowsSymmetry(s)) symmetries.push_back(s);
      std::vector<json> roots;
      for(int symmetry: symmetries) {
        params.symmetry = symmetry;
        auto root = evaluator.initial(board,history,pla,meta,params,buffer,rand,true);
        testAssert(root->policyProbs[xiangqi ? Xiangqi::parseAction("e0e9") : 0] == -1);
        json entry = {{"spatial",buffer.spatial},{"global",buffer.global},{"metadata",buffer.metadata},
          {"xsize",board.x_size},{"ysize",board.y_size},{"player",pla},{"suppress_nr",!nr},
          {"symmetry",symmetry},{"temperature",params.nnPolicyTemperature},
          {"optimism",hasMetadata ? 0.0 : params.policyOptimism},
          {"outputs",json::array({outputJSON(*root)})}};
        roots.push_back(outputJSON(*root));
        if(xiangqi) entry["legal"] = history.getXiangqiState().legalActions();
        std::vector<int> actions = xiangqi ?
          std::vector<int>{Xiangqi::parseAction("i9i0"),Xiangqi::parseAction("i9i0"),2085,Xiangqi::parseAction("b0c2")} :
          std::vector<int>{0,go ? area : 0,go ? area : 0,canvas*2};
        entry["actions"] = actions;
        auto current = root;
        for(int action: actions) {
          current = evaluator.recurrent(*current,action,params,buffer);
          testAssert(current->orientation == root->orientation);
          testAssert(current->orientation->rootPlayer == pla);
          testAssert(current->policyProbs[0] >= 0); // Occupancy never masks latent nodes.
          if(xiangqi) {
            testAssert(current->getPolicyStorageSize() == Xiangqi::NUM_ACTIONS);
            for(int a = 0; a < Xiangqi::NUM_ACTIONS; a++) testAssert(current->policyProbs[a] >= 0);
          }
          else {
            testAssert(current->policyProbs[4] == -1); // Canvas padding remains masked.
            testAssert((current->policyProbs[area] >= 0) == go);
          }
          entry["outputs"].push_back(outputJSON(*current));
        }
        reject([&]() { evaluator.recurrent(*root,-1,params,buffer); });
        reject([&]() { evaluator.recurrent(*root,xiangqi ? Xiangqi::NUM_ACTIONS : 4,params,buffer); });
        result["cases"].push_back(entry);
      }
      // All-symmetry averaging uses source output averaging; the retained latent
      // must agree with one of the individually evaluated root orientations.
      auto average = evaluator.averageMultipleSymmetries(board,history,pla,meta,params,buffer,rand,8);
      auto next = evaluator.recurrent(*average,0,params,buffer);
      params.symmetry = average->orientation->symmetry;
      auto selected = evaluator.initial(board,history,pla,meta,params,buffer,rand,true);
      selected = evaluator.recurrent(*selected,0,params,buffer);
      result["averages"].push_back({{"individual",roots},{"average",outputJSON(*average)},
        {"recurrent",outputJSON(*next)},{"selected_recurrent",outputJSON(*selected)}});

      evaluator.clearCache();
      params.symmetry = 0;
      auto first = evaluator.initial(board,history,pla,meta,params,buffer,rand,false);
      const auto rows = batcher.stats().rows;
      const auto hits = evaluator.numCacheHits();
      params.symmetry = alternateSymmetry;
      auto hit = evaluator.initial(board,history,pla,meta,params,buffer,rand,false);
      testAssert(hit == first && hit->orientation->symmetry == 0);
      testAssert(batcher.stats().rows == rows && evaluator.numCacheHits() == hits+1);
      auto bypass = evaluator.initial(board,history,pla,meta,params,buffer,rand,true);
      params.symmetry = 0;
      hit = evaluator.initial(board,history,pla,meta,params,buffer,rand,false);
      testAssert(hit == bypass && hit->orientation->symmetry == alternateSymmetry);
      testAssert(batcher.stats().rows == rows+1);
      // Metadata, temperature, PDA and history visibility use source hash keys.
      params.nnPolicyTemperature = 1.0f;
      auto different = evaluator.initial(board,history,pla,meta,params,buffer,rand,false);
      testAssert(different->nnHash != hit->nnHash);
      params.playoutDoublingAdvantage = 1.0;
      auto pda = evaluator.initial(board,history,pla,meta,params,buffer,rand,false);
      testAssert(pda->nnHash != different->nnHash);
      params.maxHistory = 0;
      auto noHistory = evaluator.initial(board,history,pla,meta,params,buffer,rand,false);
      testAssert(noHistory->nnHash != pda->nnHash);
      if(hasMetadata) {
        SGFMetadata other = metadata;
        other.inverseBRank++;
        auto changed = evaluator.initial(board,history,pla,&other,params,buffer,rand,false);
        testAssert(changed->nnHash != noHistory->nnHash);
        reject([&]() { evaluator.initial(board,history,pla,nullptr,params,buffer,rand,false); });
      }
      params.symmetry = NNInputs::SYMMETRY_NOTSPECIFIED;
      evaluator.setDefaultSymmetry(alternateSymmetry);
      auto defaulted = evaluator.initial(board,history,pla,meta,params,buffer,rand,true);
      testAssert(defaulted->orientation->symmetry == alternateSymmetry);
      evaluator.setDoRandomize(true);
      std::set<int> sampled;
      for(int i = 0; i < 32; i++) {
        auto sample = evaluator.initial(board,history,pla,meta,params,buffer,rand,true);
        testAssert(rules.allowsSymmetry(sample->orientation->symmetry));
        sampled.insert(sample->orientation->symmetry);
      }
      testAssert(sampled.size() > 1);
      evaluator.setDoRandomize(false);
      evaluator.setDefaultSymmetry(0);
      if(xiangqi || model->game() == "hex") {
        params.symmetry = 4;
        reject([&]() { evaluator.initial(board,history,pla,meta,params,buffer,rand,false); });
      }
      MuZero::PositionEvaluator exact(batcher,true,false,0,-1,0);
      reject([&]() { exact.initial(board,history,pla,meta,params,buffer,rand,true); });
      for(int invalid: {-2,8}) {
        params.symmetry = invalid;
        reject([&]() { evaluator.initial(board,history,pla,meta,params,buffer,rand,false); });
      }
      evaluator.clearCache();
    }
  }
  // Cache eviction/clear must release GPU states, including derived objects.
  Board board(xiangqi ? 9 : 4,xiangqi ? 10 : 4);
  BoardHistory history = xiangqi ? BoardHistory(Xiangqi::State(),BoardHistoryModes(false,false)) :
    BoardHistory(board,P_BLACK,Rules::forGame(Rules::parseGame(model->game())),0,BoardHistoryModes(false,false));
  if(xiangqi) board = history.getRecentBoard(0);
  MiscNNInputParams params;
  std::weak_ptr<const MuZero::StateData> weak;
  {
    auto root = evaluator.initial(board,history,P_BLACK,meta,params,buffer,rand,true);
    weak = root->state;
  }
  testAssert(!weak.expired());
  evaluator.clearCache();
  testAssert(weak.expired());
  if(xiangqi) {
    // This full prior history adjudicates the next real move as a long-check
    // loss. A FEN-only copy has the same pieces but not that rule history.
    Xiangqi::State state("5k3/3RR4/9/9/4P4/9/9/9/9/4K4 w - - 0 1");
    for(const char* move : {"e8f8","f9e9","f8e8","e9f9","e8f8","f9e9","f8e8"})
      state.play(Xiangqi::parseAction(move));
    BoardHistory prior(state,BoardHistoryModes(false,false));
    BoardHistory fresh(Xiangqi::State(state.fen()),BoardHistoryModes(false,false));
    const Board sameBoard = prior.getRecentBoard(0);
    testAssert(sameBoard.pos_hash == fresh.getRecentBoard(0).pos_hash);
    params.symmetry = 2;
    auto withHistory = evaluator.initial(sameBoard,prior,state.nextPlayer(),nullptr,params,buffer,rand,false);
    auto withoutHistory = evaluator.initial(sameBoard,fresh,state.nextPlayer(),nullptr,params,buffer,rand,false);
    testAssert(withHistory->nnHash != withoutHistory->nnHash && withHistory->state != withoutHistory->state);
    const int move = Xiangqi::parseAction("e9f9");
    auto latent = evaluator.recurrent(*withHistory,move,params,buffer);
    for(int a = 0; a < Xiangqi::NUM_ACTIONS; a++) testAssert(latent->policyProbs[a] >= 0);
    Board actual = sameBoard;
    prior.makeBoardMoveAssumeLegal(actual,Xiangqi::actionToLoc(move),prior.presumedNextMovePla,nullptr);
    testAssert(prior.isGameFinished && prior.winner == P_WHITE);
    const auto rows = batcher.stats().rows;
    reject([&]() { evaluator.initial(actual,prior,prior.presumedNextMovePla,nullptr,params,buffer,rand,false); });
    testAssert(batcher.stats().rows == rows);
    actual = sameBoard;
    fresh.makeBoardMoveAssumeLegal(actual,Xiangqi::actionToLoc(move),fresh.presumedNextMovePla,nullptr);
    testAssert(!fresh.isGameFinished);
    evaluator.clearCache();
  }
  batcher.finish();
  std::ofstream out(response);
  out << result.dump() << '\n';
}
