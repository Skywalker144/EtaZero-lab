#include "../neuralnet/desc.h"
#include "../neuralnet/muzero.h"
#include "../game/xiangqi.h"
#include "../external/nlohmann_json/json.hpp"
#include "tests.h"
#include <iostream>
#include <fstream>

void Tests::runMuZeroModelDesc(const std::string& path) {
  ModelDesc desc;
  ModelDesc::loadFromFileMaybeGZipped(path,desc,"",true);
  if(desc.muZeroComponent != 0) {
    bool rejected = false;
    try {
      ModelDesc ordinary;
      ModelDesc::loadFromFileMaybeGZipped(path,ordinary,"");
    }
    catch(const StringError&) { rejected = true; }
    testAssert(rejected);
  }
  // Exercise source optimizations and move/release lifecycle for headless bodies.
  const auto parameters = desc.getNumParameters();
  const auto scale8 = desc.applyScale8ToReduceActivations();
  ModelDesc moved(std::move(desc));
  desc = std::move(moved);
  testAssert(desc.getNumParameters() == parameters);
  int convolutions = 0;
  desc.iterConvLayers([&](const ConvLayerDesc&) noexcept { convolutions++; });
  nlohmann::json output = {
    {"role",desc.muZeroComponent}, {"latent_channels",desc.numLatentChannels},
    {"spatial_channels",desc.numInputChannels}, {"global_channels",desc.numInputGlobalChannels},
    {"policy_channels",desc.numPolicyChannels}, {"value_channels",desc.numValueChannels},
    {"aux_channels",desc.numAuxValueChannels}, {"parameters",parameters},
    {"convolutions",convolutions}, {"scale8",scale8},
    {"projection_weights",desc.latentProjection.weights}
  };
  if(desc.modelVersion == 19 && !desc.isMuZeroBody()) {
    output["spatial_policy_channels"] = desc.policyHead.p2Conv.outChannels;
    output["policy_weights"] = desc.policyHead.p2Conv.weights;
  }
  desc.releaseWeights();
  desc.iterConvLayers([](const ConvLayerDesc& conv) { testAssert(conv.weights.empty()); });
  testAssert(desc.trunk.initialMatMul.weights.empty());
  std::cout << output.dump() << std::endl;
}

void Tests::runMuZeroInference(const std::string& directory, const std::string& request, const std::string& response) {
  using json = nlohmann::json;
  std::ifstream in(request);
  const json input = json::parse(in);
  ConfigParser config(input.value("cuda_config",std::map<std::string,std::string>{}));
  const auto precision = input.at("fp16").get<bool>() ? enabled_t::True : enabled_t::False;
  const int maxBatch = input.at("batch").get<int>();
  auto evaluator = std::make_unique<MuZero::Evaluator>(directory,maxBatch,0,precision,config,nullptr,input.value("inputs_nhwc",true));
  if(input.value("warmup",false)) evaluator->warmup();
  const auto spatial = input.at("spatial").get<std::vector<float>>();
  const auto global = input.at("global").get<std::vector<float>>();
  const auto metadata = input.value("metadata",std::vector<float>{});
  std::vector<std::vector<MuZero::Output>> generations;
  generations.push_back(evaluator->initial(maxBatch,spatial,global,metadata));
  for(const auto& step: input.at("steps")) {
    std::vector<MuZero::State> states;
    for(const auto& index: step.at("states"))
      states.push_back(generations.at(index.at(0).get<int>()).at(index.at(1).get<int>()).state);
    generations.push_back(evaluator->recurrent(states,step.at("actions").get<std::vector<int>>()));
  }
  // No state downloads occur until all device-to-device recurrences finish.
  json output = json::array();
  for(const auto& generation: generations) {
    json rows = json::array();
    for(const auto& row: generation)
      rows.push_back({{"latent",evaluator->readStateForTesting(row.state)}, {"policy",row.policy},
                     {"value",row.value}, {"auxiliary",row.auxiliary}});
    output.push_back(rows);
  }
  const auto expectRejected = [](const std::function<void()>& run) {
    bool rejected = false;
    try { run(); } catch(const StringError&) { rejected = true; }
    testAssert(rejected);
  };
  expectRejected([&]() { evaluator->initial(0,{},{}); });
  expectRejected([&]() { evaluator->initial(maxBatch+1,spatial,global,metadata); });
  expectRejected([&]() { evaluator->initial(maxBatch,{},global,metadata); });
  expectRejected([&]() { evaluator->recurrent({},{}); });
  expectRejected([&]() { evaluator->recurrent({generations[0][0].state},{}); });
  expectRejected([&]() { evaluator->recurrent({nullptr},{0}); });
  expectRejected([&]() { evaluator->recurrent({generations[0][0].state},{-1}); });
  const int area = evaluator->posLen()*evaluator->posLen();
  const bool xiangqi = evaluator->game() == "xiangqi";
  const int actionCount = xiangqi ? Xiangqi::NUM_ACTIONS : area+(evaluator->game() == "go");
  expectRejected([&]() { evaluator->recurrent({generations[0][0].state},{actionCount}); });
  for(int row = 0; !xiangqi && row < maxBatch; row++) {
    for(int point = 0; point < area; point++) {
      if(spatial[row*area*evaluator->representationDesc().numInputChannels+point] == 0) {
        expectRejected([&]() { evaluator->recurrent({generations[0][row].state},{point}); });
        break;
      }
    }
  }
  auto badMask = spatial;
  std::fill(badMask.begin(),badMask.begin()+area,0);
  expectRejected([&]() { evaluator->initial(maxBatch,badMask,global,metadata); });
  badMask[0] = 0.5f;
  expectRejected([&]() { evaluator->initial(maxBatch,badMask,global,metadata); });
  if(xiangqi) {
    badMask = spatial;
    badMask[Xiangqi::WIDTH] = 1; // Padding must remain outside the physical board.
    expectRejected([&]() { evaluator->initial(maxBatch,badMask,global,metadata); });
    badMask = spatial;
    badMask[0] = 0;
    expectRejected([&]() { evaluator->initial(maxBatch,badMask,global,metadata); });
  }
  if(input.value("lifecycle",false)) {
    // Releasing the evaluator must not free outstanding immutable states, and
    // those states must never be reinterpreted as belonging to a new model.
    evaluator.reset();
    evaluator = std::make_unique<MuZero::Evaluator>(directory,maxBatch,0,precision,config,nullptr,input.value("inputs_nhwc",true));
    expectRejected([&]() { evaluator->recurrent({generations[0][0].state},{0}); });
  }
  evaluator.reset();
  generations.clear();
  std::ofstream out(response);
  out << output.dump() << '\n';
}
