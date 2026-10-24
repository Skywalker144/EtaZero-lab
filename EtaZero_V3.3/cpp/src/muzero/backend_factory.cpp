#include "etazero/backend_factory.h"
#ifdef ETAZERO_WITH_CUDA
#include "etazero/muzero/cuda_backend.h"
#endif
namespace etazero {
std::unique_ptr<muzero::Backend> make_muzero_backend(const std::string& path,const std::string& device,
    int canvas,int batch,const std::string& precision,std::shared_ptr<PreparedModel> prepared){
    if(!is_cuda_model(path))return std::make_unique<muzero::TorchBackend>(path,device,canvas,batch,precision,prepared);
#ifdef ETAZERO_WITH_CUDA
    if(device.rfind("cuda",0)!=0)throw std::runtime_error("MuZero CUDA model requires a CUDA device");
    if(cuda_model_algorithm(path)!="muzero")throw std::runtime_error("Expected a MuZero CUDA model");
    int index=device=="cuda"?0:std::stoi(device.substr(5));
    return std::make_unique<muzero::CudaBackend>(path,canvas,batch,index,precision=="auto"?"float16":precision,cuda_model_spec(path,canvas));
#else
    throw std::runtime_error("This executable was built without the CUDA backend");
#endif
}
}
