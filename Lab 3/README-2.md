# Here is the instruction of the code

### To compile the histo.cu, use the command: nvcc -o convolution2D convolution2D.cu

### To run the code , use the command: ./convolution2D -i \<dimX> -j \<dimY> -k \<dimK>

### The code many have problem when dimK is over 15

### The sample output is like:

CPU Time: \<time> ms, GFLOPS: \<number>
GPU Basic Time: \<time> ms, GFLOPS: \<number>
GPU Basic Error: 0
GPU Tiled Time: \<time> ms, GFLOPS: \<number>
GPU Tiled Error: 0