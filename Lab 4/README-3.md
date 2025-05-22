## Compile
```
nvcc popcorn_kernel_kmeans.cu -o popcorn_kmeans -arch=sm_60 -std=c++17 -lcublas -lcusparse -lineinfo -O3
```
Use proper parameters for -arch=sm_60, If the file cannot be compiled, use 70 or 75 instead.
## Run
```
./popcorn_kmeans
```