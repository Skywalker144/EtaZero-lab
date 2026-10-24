option(ETAZERO_WITH_CUDA "Build the pinned KataGo CUDA/cuDNN inference engine" ON)
if(ETAZERO_WITH_CUDA)
    set(ETAZERO_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures for the inference engine")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pathlib,nvidia.cudnn; print(pathlib.Path(nvidia.cudnn.__path__[0]))"
        OUTPUT_VARIABLE PIP_CUDNN_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${PIP_CUDNN_ROOT} PATH_SUFFIXES include REQUIRED)
    find_library(CUDNN_LIBRARY NAMES cudnn libcudnn.so.9 HINTS ${CUDNN_ROOT_DIR} ${PIP_CUDNN_ROOT} PATH_SUFFIXES lib lib64 REQUIRED)
    get_filename_component(PIP_NVIDIA_ROOT ${PIP_CUDNN_ROOT} DIRECTORY)
    find_library(CUBLAS_LIBRARY NAMES cublas libcublas.so.13 libcublas.so.12 HINTS ${PIP_NVIDIA_ROOT}/cublas/lib ${PIP_NVIDIA_ROOT}/cu13/lib ${CUDAToolkit_LIBRARY_DIR} REQUIRED)
    add_library(etazero_cuda_engine
        neuralnet/cudabackend.cpp neuralnet/cudautils.cpp neuralnet/cudahelpers.cu
        neuralnet/cudafusedffn.cu neuralnet/muzerohelpers.cu neuralnet/desc.cpp neuralnet/debugprint.cpp neuralnet/modelversion.cpp)
    set_target_properties(etazero_cuda_engine PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES ${ETAZERO_CUDA_ARCHITECTURES})
    set_source_files_properties(neuralnet/cudafusedffn.cu PROPERTIES COMPILE_OPTIONS "--expt-relaxed-constexpr")
    target_compile_definitions(etazero_cuda_engine PRIVATE USE_CUDA_BACKEND USE_CUTLASS_FUSED_FFN CUDA_TARGET_VERSION=${CMAKE_CUDA_COMPILER_VERSION})
    target_include_directories(etazero_cuda_engine SYSTEM PRIVATE ${CUDNN_INCLUDE_DIR} ${PIP_NVIDIA_ROOT}/cublas/include ${PIP_NVIDIA_ROOT}/cu13/include
        external/cudnn-frontend/include external/cutlass/include
        external/cutlass/examples/45_dual_gemm external/filesystem-1.5.8/include)
    target_link_libraries(etazero_cuda_engine PUBLIC etazero_go_features ZLIB::ZLIB CUDA::cudart ${CUBLAS_LIBRARY} CUDA::nvrtc ${CUDNN_LIBRARY})
    add_library(etazero_cuda neuralnet/cuda_backend.cpp neuralnet/muzero/cuda_backend.cpp)
    target_include_directories(etazero_cuda PRIVATE external)
    target_link_libraries(etazero_cuda PUBLIC etazero_neuralnet etazero_cuda_engine)
    add_executable(cuda_inference_probe tests/cuda_inference_probe.cpp)
    target_link_libraries(cuda_inference_probe PRIVATE etazero_cuda)
    add_executable(muzero_cuda_probe tests/muzero_cuda_probe.cpp)
    target_link_libraries(muzero_cuda_probe PRIVATE etazero_cuda)
endif()
