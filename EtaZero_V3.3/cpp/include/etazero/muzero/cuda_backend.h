#pragma once
#include "etazero/muzero/inference.h"
namespace etazero::muzero {
class CudaBackend final : public Backend {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    CudaBackend(const std::string& path,int canvas,int max_batch,int device,const std::string& precision,GameSpec);
    ~CudaBackend() override;
    void initialize() override;
    std::vector<InferenceOutput> initial(const InferenceInputs&) override;
    std::vector<InferenceOutput> recurrent(const std::vector<InferenceAction>&) override;
};
}
