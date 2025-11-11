#define CHECK_CUDA(val) {\
cudaError_t err = (val); \
if(err != cudaSuccess)    \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; }

#define CHECK_CUBLAS(val) {\
cublasStatus_t s = (val); \
if(s != CUBLAS_STATUS_SUCCESS)    \
    std::cerr << "cuBLAS Error: " << s << std::endl;}
