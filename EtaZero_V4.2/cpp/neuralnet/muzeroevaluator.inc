// Included inside MuZero after the backend defines StateData and Evaluator::Impl.
const void* stateOwner(const State& state) { return state ? state->owner.get() : nullptr; }

Evaluator::Evaluator(const std::string& directory, int maxBatch, int device, enabled_t fp16, ConfigParser& cfg, Logger* logger, bool inputsUseNHWC)
  : Evaluator(std::make_shared<ModelBundle>(directory),maxBatch,device,fp16,cfg,logger,inputsUseNHWC) {}
Evaluator::Evaluator(std::shared_ptr<const ModelBundle> bundle, int maxBatch, int device, enabled_t fp16, ConfigParser& cfg, Logger* logger, bool inputsUseNHWC) {
  if(!bundle || bundle->isRandom()) throw StringError("MuZero evaluator requires a loaded model bundle");
  impl = std::make_unique<Impl>(std::move(bundle),maxBatch,device,fp16,cfg,logger,inputsUseNHWC);
}
Evaluator::~Evaluator() = default;
const ModelDesc& Evaluator::representationDesc() const { return impl->representation->modelDesc; }
const ModelDesc& Evaluator::predictionDesc() const { return impl->prediction->modelDesc; }
const void* Evaluator::identity() const { return impl->owner.get(); }
bool Evaluator::isUsingFP16() const { return NeuralNet::isUsingFP16(impl->handle.get()); }
int Evaluator::posLen() const { return impl->canvas; }
int Evaluator::latentChannels() const { return impl->channels; }
const std::string& Evaluator::game() const { return impl->gameName; }
