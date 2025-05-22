# Here is the instruction of the code

## I have upload 2 files, TiledMatrixMul.cu and CPU.cu.


### To compile the TiledMatrixMul.cu, use the command: nvcc -o TiledMatrixMul TiledMatrixMul.cu -lcublas 

### To run the code , use the command: ./TiledMatrixMul -i \<rowDimA> \<colDimA> \<colDimB>

### Because the code contain cuBLAS for comparing.

#### The result may shown as:

Naive: \<time> ms, \<number> TFLOPS
Coalesced: \<time> ms, \<number> TFLOPS
Tiled (Shared Memory): \<time> ms, \<number> TFLOPS
Warp-Optimized: \<time> ms, \<number> TFLOPS
1D Thread Tiling: \<time> ms, \<number> TFLOPS
2D Thread Tiling: \<time> ms, \<number> TFLOPS
CUBLAS: \<time> ms, \<number> TFLOPS

#### which contain 7 results.


### To compile the CPU.cu, use the command: nvcc -o CPU CPU.cu

### To run the code , use the command: ./CPU -i \<rowDimA> \<colDimA> \<colDimB>

#### The result will only show:
CPU                \<time> ms
