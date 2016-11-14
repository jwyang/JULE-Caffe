/*=============================*/
/*update affinity A_{c->merged}*/
/*it costs much time to update */
/*this by cpu, we would like to*/
/*compute using gpu. */
/*=============================*/
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include "caffe/util/updateAc2merged.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/bmat.h"
#include "caffe/common.hpp"

#define block_thread_size 512
#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) std::cout << "CUDA Error: " << \
        cudaGetErrorString(XXX) << ", at line " << __LINE__ \
        << std::endl; cudaDeviceSynchronize(); } while (0)

/**
* Computes the distance between two matrix A (reference points) and
* B (query points) containing respectively wA and wB points.
*
* @param A     pointer on the matrix A
* @param wA    width of the matrix A = number of points in A
* @param pA    pitch of matrix A given in number of columns
* @param B     pointer on the matrix B
* @param wB    width of the matrix B = number of points in B
* @param pB    pitch of matrix B given in number of columns
* @param dim   dimension of points = height of matrices A and B
* @param AB    pointer on the matrix containing the wA*wB distances computed
*/
//__global__ void cuComputeDistanceGlobal(float* A, int wA, int pA, float* B, int wB, int pB, int dim, float* AB){
//
//	// Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
//	__shared__ float shared_A[block_thread_size][block_thread_size];
//	__shared__ float shared_B[block_thread_size][block_thread_size];
//
//	// Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
//	__shared__ int begin_A;
//	__shared__ int begin_B;
//	__shared__ int step_A;
//	__shared__ int step_B;
//	__shared__ int end_A;
//
//	// Thread index
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
//
//	// Other variables
//	float tmp;
//	float ssd = 0;
//
//	// Loop parameters
//	begin_A = block_thread_size * blockIdx.y;
//	begin_B = block_thread_size * blockIdx.x;
//	step_A = block_thread_size * pA;
//	step_B = block_thread_size * pB;
//	end_A = begin_A + (dim - 1) * pA;
//
//	// Conditions
//	int cond0 = (begin_A + tx < wA); // used to write in shared memory
//	int cond1 = (begin_B + tx < wB); // used to write in shared memory & to computations and to write in output matrix
//	int cond2 = (begin_A + ty < wA); // used to computations and to write in output matrix
//
//	// Loop over all the sub-matrices of A and B required to compute the block sub-matrix
//	for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {
//
//		// Load the matrices from device memory to shared memory; each thread loads one element of each matrix
//		if (a / pA + ty < dim){
//			shared_A[ty][tx] = (cond0) ? A[a + pA * ty + tx] : 0;
//			shared_B[ty][tx] = (cond1) ? B[b + pB * ty + tx] : 0;
//		}
//		else{
//			shared_A[ty][tx] = 0;
//			shared_B[ty][tx] = 0;
//		}
//
//		// Synchronize to make sure the matrices are loaded
//		__syncthreads();
//
//		// Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
//		if (cond2 && cond1){
//			for (int k = 0; k < block_thread_size; ++k){
//				tmp = shared_A[k][ty] - shared_B[k][tx];
//				ssd += tmp*tmp;
//			}
//		}
//
//		// Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
//		__syncthreads();
//	}
//
//	// Write the block sub-matrix to device memory; each thread writes one element
//	if (cond2 && cond1)
//		AB[(begin_A + ty) * pB + begin_B + tx] = ssd;
//}

/*===============*/
/* sort matrix */
/*===============*/
__global__ void kernel_sort_distance(float* source, float* target, int* ind_sort,
	const int K, const int height, const int width, bool order) {
	// order=0: descend
	// order=1: ascend
	unsigned int idx = blockIdx.x * block_thread_size + threadIdx.x;
	if (idx < height) { // idx must be less than height
		ind_sort[idx] = 0;
		// find top K_c affinity values from affinity matrix
		for (int i = 0; i < K; ++i) {
			int ind_sort_idx = i;
			float opt_val = source[idx * width + i];
			for (int j = i + 1; j < width; ++j) {
				if (order == false) {
					if (opt_val < source[idx * width + j]) { // if 
						opt_val = source[idx * width + j];
						ind_sort_idx = j;
					}
				}
				else if (order == true) {
					if (opt_val > source[idx * width + j]) { // if 
						opt_val = source[idx * width + j];
						ind_sort_idx = j;
					}
				}
			}
			target[idx * K + i] = opt_val;
			float temp = source[idx * width + i];
			source[idx * width + i] = source[idx * width + ind_sort_idx];
			source[idx * width + ind_sort_idx] = temp;
			ind_sort[idx * K + i] = ind_sort_idx;
		}
	}
}

void batchDistance(float* feat, float* d_samples, int* nIdx, const int Knn, const int n, int d) { 
	// find knn for samples, note that feat is num_samples X dim
	size_t num_samples = n;
	size_t dim = d;
	size_t num_elements = num_samples * dim;
	size_t num_elements_output = size_t(num_samples) * size_t(Knn);
	size_t pitch;
	float* feat_gpu;

	// CUDA_WARN(cudaMallocPitch((void**)&feat_gpu, &pitch, num_samples, dim));
	// CUDA_WARN(cudaMemcpy2D(feat_gpu, pitch, feat, dim * sizeof(float), dim * sizeof(float), 
		// num_samples, cudaMemcpyHostToDevice));
	CUDA_WARN(cudaMalloc((void**)&feat_gpu, num_elements * sizeof(float)));
	CUDA_WARN(cudaMemcpy(feat_gpu, feat, num_elements * sizeof(float), cudaMemcpyHostToDevice));

	// compute A*A';
	size_t batch_size = 50;
	size_t num_elements_batch = batch_size * dim;
	float *ABt;
	// float* Fnorm_gpu;
	CUDA_WARN(cudaMalloc((void**)&ABt, batch_size * num_samples * sizeof(float)));
	// CUDA_WARN(cudaMalloc((void**)&Fnorm_gpu, num_samples * sizeof(float)));

	//float *ones_A, *ones_B, *Anorm_mat, *Bnorm_mat;
	//CUDA_WARN(cudaMalloc((void**)&ones_A, batch_size * sizeof(float)));
	//CUDA_WARN(cudaMalloc((void**)&ones_B, num_samples * sizeof(float)));
	//CUDA_WARN(cudaMalloc((void**)&Anorm_mat, batch_size * num_samples * sizeof(float)));
	//CUDA_WARN(cudaMalloc((void**)&Bnorm_mat, batch_size * num_samples * sizeof(float)));

	// float *batchD;
	// CUDA_WARN(cudaMalloc((void**)&batchD, batch_size * num_samples * sizeof(float)));

	//float* Fnorm_cpu = new float[num_samples];
	//// before compute distance, we first compute norm for all samples
	//for (int i = 0; i < num_samples; ++i) {	
	//	caffe::caffe_gpu_dot(dim, feat_gpu + i * dim, feat_gpu + i * dim, Fnorm_cpu + i);		
	//}
	//CUDA_WARN(cudaMemcpy(Fnorm_gpu, Fnorm_cpu, num_samples * sizeof(float), cudaMemcpyHostToDevice));

	//float* batch_feat_gpu;
	//CUDA_WARN(cudaMalloc((void**)&batch_feat_gpu, num_elements_batch * sizeof(float)));
	// float* ABt_cpu = new float[batch_size * num_samples];
	float* sortedD;
	int* sortedInd;
	CUDA_WARN(cudaMalloc((void**)&sortedD, batch_size * Knn * sizeof(float)));
	CUDA_WARN(cudaMalloc((void**)&sortedInd, batch_size * Knn * sizeof(int)));

	for (size_t i = 0; i < (num_samples + batch_size - 1) / batch_size; ++i)  {
		// compyte A x B'
		size_t batch_size_i = batch_size;
		if ((i + 1) * batch_size > num_samples)
			batch_size_i = num_samples - i * batch_size;

		caffe::caffe_gpu_gemm(CblasNoTrans, CblasTrans, batch_size_i, num_samples, dim, -2.0f, feat_gpu + i * num_elements_batch, feat_gpu, 0.0f, ABt);
		// caffe::caffe_cpu_gemm(CblasNoTrans, CblasTrans, batch_size, num_samples, dim, -2.0f, feat + i * num_elements_batch, feat, 0.0f, ABt_cpu);

		// convert ||A||^2 to matrix
		// caffe::caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, batch_size, num_samples, 1, 1.0f, Fnorm_gpu + i * batch_size, ones_B, 0.0f, Anorm_mat);

		// convert ||B||^2 to matrix
		// caffe::caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, batch_size, num_samples, 1, 1.0f, ones_A, Fnorm_gpu, 0.0f, Bnorm_mat);

		// ||A||^2 + ||B||^2
		// caffe::caffe_gpu_add(batch_size * num_samples, Anorm_mat, Bnorm_mat, batchD);
		// caffe::caffe_gpu_axpy(batch_size * num_samples, -2.0f, ABt, batchD);
		caffe::caffe_gpu_add_scalar(batch_size_i * num_samples, 2.0f, ABt);

		// for batch distance, find top Knn minimal values
		int n_blocks = (batch_size_i + block_thread_size - 1) / block_thread_size;
		kernel_sort_distance << <n_blocks, block_thread_size >> >(ABt, sortedD, sortedInd, Knn, batch_size_i, num_samples, 1);

		CUDA_WARN(cudaMemcpy(d_samples + i * batch_size * Knn, sortedD, batch_size_i * Knn * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_WARN(cudaMemcpy(nIdx + i * batch_size * Knn, sortedInd, batch_size_i * Knn * sizeof(int), cudaMemcpyDeviceToHost));
	}

	CUDA_WARN(cudaFree(feat_gpu));
	CUDA_WARN(cudaFree(ABt));
	CUDA_WARN(cudaFree(sortedD));
	CUDA_WARN(cudaFree(sortedInd));

}

void entityDistance(float* feat, float* d_samples, const int n, int d) {
	// find knn for samples, note that feat is num_samples X dim
	size_t num_samples = n;
	size_t dim = d;
	size_t num_elements = num_samples * dim;
	size_t pitch;
	float* feat_gpu;

	// CUDA_WARN(cudaMallocPitch((void**)&feat_gpu, &pitch, num_samples, dim));
	// CUDA_WARN(cudaMemcpy2D(feat_gpu, pitch, feat, dim * sizeof(float), dim * sizeof(float), 
	// num_samples, cudaMemcpyHostToDevice));
	CUDA_WARN(cudaMalloc((void**)&feat_gpu, num_elements * sizeof(float)));
	CUDA_WARN(cudaMemcpy(feat_gpu, feat, num_elements * sizeof(float), cudaMemcpyHostToDevice));

	// compute A*A';
	size_t batch_size = 100;
	size_t num_elements_batch = batch_size * dim;
	float *ABt;
	// float* Fnorm_gpu;
	CUDA_WARN(cudaMalloc((void**)&ABt, batch_size * num_samples * sizeof(float)));
	// CUDA_WARN(cudaMalloc((void**)&Fnorm_gpu, num_samples * sizeof(float)));

	for (size_t i = 0; i < (num_samples + batch_size - 1) / batch_size; ++i)  {
		// compyte A x B'
		size_t batch_size_i = batch_size;
		if ((i + 1) * batch_size > num_samples)
			batch_size_i = num_samples - i * batch_size;

		caffe::caffe_gpu_gemm(CblasNoTrans, CblasTrans, batch_size_i, num_samples, dim, -2.0f, feat_gpu + i * num_elements_batch, feat_gpu, 0.0f, ABt);
		// caffe::caffe_cpu_gemm(CblasNoTrans, CblasTrans, batch_size, num_samples, dim, -2.0f, feat + i * num_elements_batch, feat, 0.0f, ABt_cpu);

		// convert ||A||^2 to matrix
		// caffe::caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, batch_size, num_samples, 1, 1.0f, Fnorm_gpu + i * batch_size, ones_B, 0.0f, Anorm_mat);

		// convert ||B||^2 to matrix
		// caffe::caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, batch_size, num_samples, 1, 1.0f, ones_A, Fnorm_gpu, 0.0f, Bnorm_mat);

		// ||A||^2 + ||B||^2
		// caffe::caffe_gpu_add(batch_size * num_samples, Anorm_mat, Bnorm_mat, batchD);
		// caffe::caffe_gpu_axpy(batch_size * num_samples, -2.0f, ABt, batchD);
		caffe::caffe_gpu_add_scalar(batch_size_i * num_samples, 2.0f, ABt);

		// for batch distance, find top Knn minimal values
		// int n_blocks = (batch_size_i + block_thread_size - 1) / block_thread_size;
		// kernel_sort_distance << <n_blocks, block_thread_size >> >(ABt, sortedD, sortedInd, Knn, batch_size_i, num_samples, 1);

		CUDA_WARN(cudaMemcpy(d_samples + i * batch_size * num_samples, ABt, batch_size_i * num_samples * sizeof(float), cudaMemcpyDeviceToHost));
		// CUDA_WARN(cudaMemcpy(nIdx + i * batch_size * Knn, sortedInd, batch_size_i * Knn * sizeof(int), cudaMemcpyDeviceToHost));
	}

	CUDA_WARN(cudaFree(feat_gpu));
	CUDA_WARN(cudaFree(ABt));

}

float computeAc2merged(float* W_sub_mc, float* W_sub_cm, const int rows, const int cols) {
	float* W_sub_mc_gpu;
	float* W_sub_cm_gpu;

	size_t num_elements = size_t(rows) * size_t(cols);
	CUDA_WARN(cudaMalloc((void**)&W_sub_mc_gpu, num_elements * sizeof(float)));
	CUDA_WARN(cudaMalloc((void**)&W_sub_cm_gpu, num_elements * sizeof(float)));

	CUDA_WARN(cudaMemcpy(W_sub_mc_gpu, W_sub_mc, num_elements * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_WARN(cudaMemcpy(W_sub_cm_gpu, W_sub_cm, num_elements * sizeof(float), cudaMemcpyHostToDevice));

	float* W_sub_product;
	CUDA_WARN(cudaMalloc((void**)&W_sub_product, rows * rows * sizeof(float)));

	caffe::caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, rows, rows, cols, 1.0f, W_sub_mc_gpu, W_sub_cm_gpu, 0.0f, W_sub_product);

	float sum;
	caffe::caffe_gpu_asum(rows * rows, W_sub_product, &sum);
	cudaFree(W_sub_mc_gpu);
	cudaFree(W_sub_cm_gpu);
	cudaFree(W_sub_product);
	return sum;
}