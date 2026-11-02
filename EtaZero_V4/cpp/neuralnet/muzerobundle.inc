// Shared bundle loading after a backend defines LoadedModel.
#include "../core/fileutils.h"
#include "../game/xiangqi.h"
#include "modelversion.h"
#include "../external/nlohmann_json/json.hpp"

namespace MuZero {
struct ModelBundle::Impl {
  int canvas, channels;
  std::string gameName;
  bool transformer;
  std::unique_ptr<LoadedModel> representation, dynamics, prediction;
  ModelDesc randomDesc;
  Impl(const std::string& game, int posLen) : canvas(posLen), channels(0), gameName(game), transformer(false) {
    randomDesc.name = "random";
    randomDesc.modelVersion = game == "xiangqi" ? 19 : game == "go" ? 17 : 18;
    randomDesc.numInputChannels = NNModelVersion::getNumSpatialFeatures(randomDesc.modelVersion);
    randomDesc.numInputGlobalChannels = NNModelVersion::getNumGlobalFeatures(randomDesc.modelVersion);
    randomDesc.numPolicyChannels = 1;
    randomDesc.numValueChannels = 4;
  }
  explicit Impl(const std::string& directory) {
    std::string manifest;
    FileUtils::loadFileIntoString(directory+"/model.json","",manifest);
    const auto info = nlohmann::json::parse(manifest);
    canvas = info.at("pos_len").get<int>();
    channels = info.at("latent_channels").get<int>();
    gameName = info.at("game").get<std::string>();
    if(canvas < 1 || canvas > NNPos::MAX_BOARD_LEN || channels < 1 ||
       (gameName == "xiangqi" && canvas < Xiangqi::HEIGHT))
      throw StringError("MuZero canvas/latent dimensions are invalid");
    // Packing/normalization use the source backend's signed 32-bit indexing.
    // Divide first so hostile or erroneous dimensions cannot overflow the check.
    if((int64_t)channels+2+(gameName == "xiangqi") > (((int64_t)1<<31)-1)/(canvas*canvas))
      throw StringError("MuZero GPU tensor exceeds the supported 2^31 entry bound");
    if(gameName != "go" && gameName != "freestyle" && gameName != "standard" && gameName != "renju" && gameName != "hex" && gameName != "xiangqi")
      throw StringError("MuZero action layout is not connected for " + gameName);
    representation = std::make_unique<LoadedModel>(directory+"/representation.bin","",true);
    dynamics = std::make_unique<LoadedModel>(directory+"/dynamics.bin","",true);
    prediction = std::make_unique<LoadedModel>(directory+"/prediction.bin","",true);
    transformer = false;
    int role = 0;
    for(const auto* model: {representation.get(),dynamics.get(),prediction.get()}) {
      const auto& desc = model->modelDesc;
      if(desc.muZeroComponent != ++role || desc.numLatentChannels != channels ||
         ((desc.modelVersion == 18) != (gameName != "go" && gameName != "xiangqi")) ||
         ((desc.modelVersion == 19) != (gameName == "xiangqi")))
        throw StringError("MuZero bundle component roles, latent widths or game layouts differ");
      transformer |= desc.hasAnyTransformerBlocks();
    }
  }
};
ModelBundle::ModelBundle(const std::string& directory) : impl(std::make_unique<Impl>(directory)) {}
ModelBundle::ModelBundle(const std::string& game, int canvas) : impl(std::make_unique<Impl>(game,canvas)), randomEvaluation(true) {}
ModelBundle::~ModelBundle() = default;
const ModelDesc& ModelBundle::representationDesc() const { return isRandom() ? impl->randomDesc : impl->representation->modelDesc; }
const ModelDesc& ModelBundle::predictionDesc() const { return isRandom() ? impl->randomDesc : impl->prediction->modelDesc; }
int ModelBundle::posLen() const { return impl->canvas; }
int ModelBundle::latentChannels() const { return impl->channels; }
const std::string& ModelBundle::game() const { return impl->gameName; }
bool ModelBundle::hasTransformer() const { return impl->transformer; }
}
