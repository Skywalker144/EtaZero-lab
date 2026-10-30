#include "muzero.h"
#if !defined(USE_CUDA_BACKEND) && !defined(USE_EIGEN_BACKEND) && !defined(USE_OPENCL_BACKEND)
namespace MuZero {
struct Evaluator::Impl {};
struct ModelBundle::Impl {};
[[noreturn]] static void unsupported() { throw StringError("MuZero native inference currently requires the CUDA, Eigen or OpenCL backend"); }
ModelBundle::ModelBundle(const std::string&) { unsupported(); }
ModelBundle::~ModelBundle() = default;
const ModelDesc& ModelBundle::representationDesc() const { unsupported(); }
const ModelDesc& ModelBundle::predictionDesc() const { unsupported(); }
int ModelBundle::posLen() const { unsupported(); }
int ModelBundle::latentChannels() const { unsupported(); }
const std::string& ModelBundle::game() const { unsupported(); }
bool ModelBundle::hasTransformer() const { unsupported(); }
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
