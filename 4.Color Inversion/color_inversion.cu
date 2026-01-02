#include <cuda_runtime.h>
#include <stdio.h>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < width * height) {
        // printf("%d => %d\n", idx, (int)image[idx]);
        int curIdx = idx*4;
        image[curIdx] = 255 - (int)image[curIdx];
        image[curIdx + 1] = 255 - (int)image[curIdx + 1];
        image[curIdx + 2] = 255 - (int)image[curIdx + 2];
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}