#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main(int argc, char** argc) {
    const int N = 10;
    float A[N] = {0,1,2,3,4,5,6,7,8,9};


    std::cout << "Result: ";
    for(int i=0; i<N; ++i)
        std::cout << A[i] << " ";
    std::cout << std::endl;

    return 0;
}
