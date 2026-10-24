#include "neuralnet/backend_factory.h"
#ifdef ETAZERO_WITH_CUDA
#include "neuralnet/cuda_backend.h"
#endif
#include "external/nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
namespace etazero {
bool is_cuda_model(const std::string& path){return std::filesystem::path(path).extension()==".bin" || std::filesystem::path(path).extension()==".gz";}
std::string reference_model_path(const std::string& path){
    return is_cuda_model(path)?(std::filesystem::path(path).parent_path()/"reference.pt").string():path;
}
std::string cuda_model_algorithm(const std::string& path){
    if(!std::filesystem::exists(std::filesystem::path(path).parent_path()/"manifest.json"))return "alphazero";
    std::ifstream input(std::filesystem::path(path).parent_path()/"manifest.json");nlohmann::json manifest;input>>manifest;
    return manifest.at("algorithm").get<std::string>();
}
GameSpec cuda_model_spec(const std::string& path,int canvas){
    if(!std::filesystem::exists(std::filesystem::path(path).parent_path()/"manifest.json"))return game_spec(GameKind::GO);
    std::ifstream input(std::filesystem::path(path).parent_path()/"manifest.json");nlohmann::json manifest;input>>manifest;
    if(manifest.at("canvas").get<int>()!=canvas)
        throw std::runtime_error("CUDA model canvas/algorithm mismatch");
    return game_spec(manifest.at("game").get<std::string>());
}
std::unique_ptr<Backend> make_backend(const std::string& path,const std::string& device,
    int canvas,int batch,const std::string& precision,std::shared_ptr<PreparedModel> prepared,const std::string& human_profile){
    std::string profile=human_profile;
    auto manifest_path=std::filesystem::path(path).parent_path()/"manifest.json";
    if(profile.empty() && std::filesystem::exists(manifest_path)) {
        std::ifstream input(manifest_path);nlohmann::json manifest;input>>manifest;
        profile=manifest.value("metadata_profile",std::string());
    }
    if(!is_cuda_model(path))return std::make_unique<TorchBackend>(path,device,canvas,batch,precision,prepared,profile);
#ifdef ETAZERO_WITH_CUDA
    if(device.rfind("cuda",0)!=0)throw std::runtime_error("KataGo CUDA model requires a CUDA device");
    int index=device=="cuda"?0:std::stoi(device.substr(5));
    return std::make_unique<CudaBackend>(path,canvas,batch,index,precision=="auto"?"float16":precision,
        cuda_model_spec(path,canvas),profile);
#else
    throw std::runtime_error("This executable was built without the CUDA backend");
#endif
}
}
