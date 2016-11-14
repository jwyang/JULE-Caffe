/*====================================== */
/* compute merging loss for two clusters */
/* the computation is based on GPU, for  */
/* acceleration upon original cpu compute*/
/*====================================== */

//// input: 
// 1-affnity matrix among clustering
//// output:
// 1-optimal pair for merging

#include <opencv2/opencv.hpp>
#include <cuda.h>
#include "caffe/util/computeMergeLoss.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/bmat.h"
#define block_thread_size 512
#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) std::cout << "CUDA Error: " << \
        cudaGetErrorString(XXX) << ", at line " << __LINE__ \
        << std::endl; cudaDeviceSynchronize(); } while (0)

/*===============*/
/* sort matrix */
/*===============*/
__global__ void kernel_sort_affinity(float* source, float* target, int* ind_sort, 
	const int K, const int height, const int width, bool order) {
	// order=0: descend
	// order=1: ascend
	unsigned int idx = blockIdx.x * block_thread_size + threadIdx.x;
	if (idx < height) { // idx must be less than height
		ind_sort[idx] = 0;
		// find top K_c affinity values from affinity matrix
		for (int i = 0; i < K; ++i) {
			int ind_sort_idx = i;
			float max_val = source[idx * width + i];
			for (int j = i + 1; j < width; ++j) {
				if (max_val < source[idx * width + j]) { // if 
					max_val = source[idx * width + j];
					ind_sort_idx = j;
				}
			}
			target[idx * K + i] = max_val;
			float temp = source[idx * width + i];
			source[idx * width + i] = source[idx * width + ind_sort_idx];
			source[idx * width + ind_sort_idx] = temp;
			if (i == 0)
				ind_sort[idx] = ind_sort_idx;
		}
	}
}

/*==============*/
/* compute loss */
/*==============*/
__global__ void kernel_merge_loss(float* affinity_Kc, float* loss, const int rows, const int cols, const float scale) {
	unsigned int idx = blockIdx.x * block_thread_size + threadIdx.x;
	if (idx < rows) {
		loss[idx] = 0;
		for (int i = 1; i < cols; ++i) {
			loss[idx] += (affinity_Kc[idx * cols + 0] - affinity_Kc[idx * cols + i]) * scale;
		}
		loss[idx] += affinity_Kc[idx * cols + 0];
	}
}

std::pair<int, int>& seekOptimalPairs(float* affinity_cpu, const int rows, const int cols, const int K_c, const float lambda) {
	// affnity_inter_sym: symmetric affinity matrix
	// K_c: the number of nearest neighbours
	size_t num_elements = size_t(rows) * size_t(cols);
	// define pointer on device and initialize it
	float* affinity_gpu;
	CUDA_WARN(cudaMalloc((void**)&affinity_gpu, num_elements * sizeof(float)));

	// bmat h_writebmat;
	// h_writebmat.write_bmat(string("affinity.bmat"), affinity_cpu, int64(rows), int64(cols), string("float"), 0, 1);
	// memcopy from host to device
	CUDA_WARN(cudaMemcpy(affinity_gpu, affinity_cpu, num_elements * sizeof(float), cudaMemcpyHostToDevice));

	float* affinity_Kc_gpu;
	CUDA_WARN(cudaMalloc((void**)&affinity_Kc_gpu, K_c * rows * sizeof(float)));

	int* ind_sort_gpu;
	CUDA_WARN(cudaMalloc((void**)&ind_sort_gpu, rows * sizeof(int)));

	// sort (Kc + 1) affinity
	int n_blocks = rows / block_thread_size + 1;
	kernel_sort_affinity << <n_blocks, block_thread_size >> >(
		affinity_gpu, affinity_Kc_gpu, ind_sort_gpu, K_c, rows, cols, false);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	// compute delta = sum_k A(c_i, c_j) - A(c_i, c_k)
	float* loss_gpu;
	const float scale = lambda / (K_c - 1);
	CUDA_WARN(cudaMalloc((void**)&loss_gpu, rows * sizeof(float)));
	kernel_merge_loss << <n_blocks, block_thread_size >> >(
		affinity_Kc_gpu, loss_gpu, rows, K_c, scale);

	int* ind_sort_cpu = new int[rows];
	CUDA_WARN(cudaMemcpy(ind_sort_cpu, ind_sort_gpu, rows * sizeof(int), cudaMemcpyDeviceToHost));
	//for (int i = 0; i < rows; ++i) {
	//	std::cout << i << " " << ind_sort_cpu[i] << std::endl;
	//}

	float* loss_cpu = new float[rows];
	CUDA_WARN(cudaMemcpy(loss_cpu, loss_gpu, rows * sizeof(float), cudaMemcpyDeviceToHost));

	float max_loss = 0;
	int max_row = 0;
	int max_col = 0;
	for (int i = 0; i < rows; ++i) {
		// std::cout << i << " " << loss_cpu[i] << std::endl;
		if (max_loss < loss_cpu[i]) {
			max_loss = loss_cpu[i];
			max_row = i;
		}
	}
	max_col = ind_sort_cpu[max_row];

	// free host and device memory
	free(ind_sort_cpu);
	CUDA_WARN(cudaFree(affinity_gpu));
	CUDA_WARN(cudaFree(affinity_Kc_gpu));
	CUDA_WARN(cudaFree(ind_sort_gpu));
	CUDA_WARN(cudaFree(loss_gpu));

	return std::make_pair(max_row, max_col);
}

