#include "muzero.h"
#include "modelversion.h"
#include "../game/xiangqi.h"

ModelDesc MuZero::ModelBundle::makeRandomDesc(const std::string& game, int canvas) {
  const auto rules = Rules::forGame(Rules::parseGame(game));
  if(canvas < 1 || canvas > NNPos::MAX_BOARD_LEN || (rules.isXiangqi() && canvas < Xiangqi::HEIGHT))
    throw StringError("Invalid random MuZero canvas size");
  ModelDesc desc;
  desc.name = "random";
  desc.modelVersion = rules.isXiangqi() ? 19 : rules.isGo() ? 17 : 18;
  desc.numInputChannels = NNModelVersion::getNumSpatialFeatures(desc.modelVersion);
  desc.numInputGlobalChannels = NNModelVersion::getNumGlobalFeatures(desc.modelVersion);
  desc.numPolicyChannels = 1;
  desc.numValueChannels = 4;
  return desc;
}

#if !defined(USE_CUDA_BACKEND) && !defined(USE_EIGEN_BACKEND) && !defined(USE_OPENCL_BACKEND)
namespace MuZero {
struct Evaluator::Impl {};
struct ModelBundle::Impl {
  ModelDesc desc;
  int canvas;
  std::string game;
  Impl(const std::string& game_, int canvas_) : desc(makeRandomDesc(game_,canvas_)), canvas(canvas_), game(game_) {}
};
[[noreturn]] static void unsupported() { throw StringError("MuZero native inference currently requires the CUDA, Eigen or OpenCL backend"); }
ModelBundle::ModelBundle(const std::string&) { unsupported(); }
ModelBundle::ModelBundle(const std::string& game, int canvas) : impl(std::make_unique<Impl>(game,canvas)), randomEvaluation(true) {}
ModelBundle::~ModelBundle() = default;
const ModelDesc& ModelBundle::representationDesc() const { return impl->desc; }
const ModelDesc& ModelBundle::predictionDesc() const { return impl->desc; }
int ModelBundle::posLen() const { return impl->canvas; }
int ModelBundle::latentChannels() const { return 0; }
const std::string& ModelBundle::game() const { return impl->game; }
bool ModelBundle::hasTransformer() const { return false; }
const void* stateOwner(const State&) { unsupported(); }
Evaluator::Evaluator(std::shared_ptr<const ModelBundle>,int,int,enabled_t,ConfigParser&,Logger*,bool) { unsupported(); }
const void* Evaluator::identity() const { unsupported(); }
bool Evaluator::isUsingFP16() const { unsupported(); }
void Evaluator::warmup() { unsupported(); }
Evaluator::Evaluator(const std::string&,int,int,enabled_t,ConfigParser&,Logger*,bool) { unsupported(); }
Evaluator::~Evaluator() = default;
const ModelDesc& Evaluator::representationDesc() const { unsupported(); }
const ModelDesc& Evaluator::predictionDesc() const { unsupported(); }
int Evaluator::posLen() const { unsupported(); }
int Evaluator::latentChannels() const { unsupported(); }
const std::string& Evaluator::game() const { unsupported(); }
std::vector<Output> Evaluator::initial(int,const std::vector<float>&,const std::vector<float>&,const std::vector<float>&) { unsupported(); }
std::vector<Output> Evaluator::recurrent(const std::vector<State>&,const std::vector<int>&) { unsupported(); }
std::vector<float> Evaluator::readStateForTesting(const State&) const { unsupported(); }
}
#endif
