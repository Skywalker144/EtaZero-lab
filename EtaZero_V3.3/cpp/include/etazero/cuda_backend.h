#pragma once
#include "etazero/inference.h"
namespace etazero {
std::vector<Evaluation> decode_cuda_outputs(std::vector<float>&,std::vector<float>&,std::vector<float>&,std::vector<float>&,
    int n,int canvas,GameSpec,int policyChannels,int valueChannels,int scoreChannels,float scale,const std::vector<uint8_t>& allow_no_result);
// Dedicated KataGo CUDA/cuDNN kernels and dynamic batches; no JIT execution.
class CudaBackend final : public Backend {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    CudaBackend(const std::string& path,int canvas,int max_batch,int device,
                const std::string& precision,GameSpec spec,const std::string& profile="");
    ~CudaBackend() override;
    bool supports_auxiliary() const override {return true;}
    int metadata_channels() const override;
    bool available() const override;
    ModelDescription description() const override;
    void set_human_profile(const std::string&) override;
    void validate_human_profile(const std::string&) const override;
    void append_profile(std::vector<float>&,int,const SGFMetadata&) const override;
    void append_metadata(std::vector<float>&,int player) const override;
    void initialize() override;
    std::vector<Evaluation> evaluate(const InferenceInputs&) override;
};
}
