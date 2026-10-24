#pragma once
#include "neuralnet/torch_backend.h"
#include "neuralnet/muzero/torch_backend.h"
namespace etazero {
bool is_cuda_model(const std::string& path);
std::string reference_model_path(const std::string& path);
GameSpec cuda_model_spec(const std::string& path,int canvas);
std::string cuda_model_algorithm(const std::string& path);
std::unique_ptr<muzero::Backend> make_muzero_backend(const std::string& path,const std::string& device,
    int canvas,int batch,const std::string& precision,std::shared_ptr<PreparedModel> prepared=nullptr);
std::unique_ptr<Backend> make_backend(const std::string& path,const std::string& device,
    int canvas,int batch,const std::string& precision,std::shared_ptr<PreparedModel> prepared=nullptr,
    const std::string& human_profile="");
}
