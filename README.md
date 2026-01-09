# Nvidia RTX-5070
## CUDA matmul kernels up to cuBLASS


MxK @ KxN = MxN

### Naive kernel

| Matrix size | GFLOPS |
|-------------|--------|
| 512         | 234.1 ± 1.4 |
| 1024        | 255.7 ± 5.6 |
| 2048        | 264.9 ± 0.0 |
| 4096        | 266.6 ± 0.0 |


### Memory-Coalescing kernel

| Matrix size | GFLOPS |
|-------------|--------|
| 512         | 1729.2 ± 14.4 |
| 1024        | 1980.9 ± 4.4 |
| 2048        | 2030.0 ± 1.1 |
| 4096        | 1429.9 ± 1.2 |


### Shared Memory Tiling kernel

 | Matrix size | GFLOPS |
|-------------|--------|
| 512         | 2256.6 ± 43.4 |
| 1024        | 2643.0 ± 8.9 |
| 2048        | 2681.1 ± 9.7 |
| 4096        | 2340.9 ± 0.2 |


### CUDA cuBLASS

| Matrix size | GFLOPS |
|-------------|--------|
| 512         | 786.2 ± 146.6 |
| 1024        | 5533.1 ± 1303.3 |
| 2048        | 19163.4 ± 545.7 |
| 4096        | 30582.2 ± 362.4 |
