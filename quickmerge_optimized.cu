// QuickMerge++ Optimized CUDA Kernels
// Author: Dong Liu and Yanxuan Yu
// ICML 2025

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

// Optimized 1. Attention Entropy Kernel
extern "C" __global__ void attention_entropy_kernel(
    const float* __restrict__ X, // [B*N*D]
    float* __restrict__ entropy_scores, // [B*N]
    int B, int N, int D, int L, float sqrt_D_inv
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / N;
    int token_idx = tid % N;
    if (batch_idx >= B || token_idx >= N) return;
    
    float total_entropy = 0.0f;
    
    // Shared memory for attention computation
    extern __shared__ float shared[];
    float* attn_scores = shared;
    float* attn_probs = shared + N;
    
    for (int layer = 0; layer < L; ++layer) {
        const float* X_layer = X + layer * B * N * D;
        const float* x_i = X_layer + (batch_idx * N + token_idx) * D;
        
        // Compute attention scores with loop unrolling
        for (int j = 0; j < N; ++j) {
            float score = 0.0f;
            const float* x_j = X_layer + (batch_idx * N + j) * D;
            
            // Unroll by 4 for better memory bandwidth
            int d;
            for (d = 0; d < D - 3; d += 4) {
                score += x_i[d] * x_j[d] + x_i[d+1] * x_j[d+1] + 
                        x_i[d+2] * x_j[d+2] + x_i[d+3] * x_j[d+3];
            }
            // Handle remaining elements
            for (; d < D; ++d) {
                score += x_i[d] * x_j[d];
            }
            attn_scores[j] = score * sqrt_D_inv;
        }
        
        // Softmax with numerical stability
        float max_score = attn_scores[0];
        for (int j = 1; j < N; ++j) {
            max_score = fmaxf(max_score, attn_scores[j]);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            attn_probs[j] = expf(attn_scores[j] - max_score);
            sum_exp += attn_probs[j];
        }
        
        // Normalize and compute entropy
        float entropy = 0.0f;
        for (int j = 0; j < N; ++j) {
            attn_probs[j] /= sum_exp;
            if (attn_probs[j] > 1e-8f) {
                entropy -= attn_probs[j] * logf(attn_probs[j]);
            }
        }
        total_entropy += entropy;
    }
    entropy_scores[tid] = total_entropy / L;
}

// Optimized 2. Saliency-Weighted Token Merging with atomic operations
extern "C" __global__ void saliency_merging_kernel(
    const float* __restrict__ X, // [B*N*D]
    const float* __restrict__ saliency_mass, // [B*N]
    const int* __restrict__ cluster_labels, // [B*N]
    float* __restrict__ merged_tokens, // [B*K*D]
    int B, int N, int K, int D
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (K * D);
    int cluster_idx = (tid % (K * D)) / D;
    int dim_idx = tid % D;
    
    if (batch_idx >= B || cluster_idx >= K || dim_idx >= D) return;
    
    // Use registers for better performance
    float weighted_sum = 0.0f;
    float weight_sum = 0.0f;
    
    // Process tokens in chunks for better cache utilization
    const int chunk_size = 32;
    for (int chunk = 0; chunk < N; chunk += chunk_size) {
        int end = min(chunk + chunk_size, N);
        for (int token_idx = chunk; token_idx < end; ++token_idx) {
            if (cluster_labels[batch_idx * N + token_idx] == cluster_idx) {
                float w = saliency_mass[batch_idx * N + token_idx];
                weighted_sum += w * X[(batch_idx * N + token_idx) * D + dim_idx];
                weight_sum += w;
            }
        }
    }
    
    merged_tokens[tid] = (weight_sum > 1e-8f) ? (weighted_sum / weight_sum) : 0.0f;
}

// Optimized 3. Cosine Similarity Matrix with symmetry exploitation
extern "C" __global__ void cosine_similarity_kernel(
    const float* __restrict__ X, // [B*N*D]
    float* __restrict__ sim_matrix, // [B*N*N]
    int B, int N, int D
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (N * N);
    int i = (tid % (N * N)) / N;
    int j = tid % N;
    
    if (batch_idx >= B || i >= N || j >= N) return;
    
    // Exploit symmetry: sim[i][j] = sim[j][i]
    if (i > j) {
        sim_matrix[tid] = sim_matrix[batch_idx * N * N + j * N + i];
        return;
    }
    
    const float* x_i = X + (batch_idx * N + i) * D;
    const float* x_j = X + (batch_idx * N + j) * D;
    
    float dot = 0.0f, norm_i = 0.0f, norm_j = 0.0f;
    
    // Loop unrolling for better performance
    int d;
    for (d = 0; d < D - 3; d += 4) {
        float xi0 = x_i[d], xi1 = x_i[d+1], xi2 = x_i[d+2], xi3 = x_i[d+3];
        float xj0 = x_j[d], xj1 = x_j[d+1], xj2 = x_j[d+2], xj3 = x_j[d+3];
        
        dot += xi0 * xj0 + xi1 * xj1 + xi2 * xj2 + xi3 * xj3;
        norm_i += xi0 * xi0 + xi1 * xi1 + xi2 * xi2 + xi3 * xi3;
        norm_j += xj0 * xj0 + xj1 * xj1 + xj2 * xj2 + xj3 * xj3;
    }
    
    // Handle remaining elements
    for (; d < D; ++d) {
        float xi = x_i[d], xj = x_j[d];
        dot += xi * xj;
        norm_i += xi * xi;
        norm_j += xj * xj;
    }
    
    float similarity = dot / (sqrtf(norm_i) * sqrtf(norm_j) + 1e-8f);
    sim_matrix[tid] = similarity;
    
    // Fill symmetric part
    if (i != j) {
        sim_matrix[batch_idx * N * N + j * N + i] = similarity;
    }
}

// Optimized 4. Gumbel-Softmax Sampling with better random number generation
extern "C" __global__ void gumbel_softmax_kernel(
    const float* __restrict__ saliency, // [B*N]
    float* __restrict__ mask, // [B*N]
    float temperature,
    int B, int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / N;
    int token_idx = tid % N;
    
    if (batch_idx >= B || token_idx >= N) return;
    
    // Use better random number generation
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    
    // Generate uniform random number
    float u = curand_uniform(&state);
    
    // Compute Gumbel noise with better numerical stability
    float gumbel = -logf(-logf(u + 1e-8f) + 1e-8f);
    
    // Apply temperature scaling
    mask[tid] = (saliency[tid] + gumbel) / temperature;
}

// Half-precision version for memory efficiency
extern "C" __global__ void attention_entropy_kernel_half(
    const __half* __restrict__ X, // [B*N*D]
    __half* __restrict__ entropy_scores, // [B*N]
    int B, int N, int D, int L, float sqrt_D_inv
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / N;
    int token_idx = tid % N;
    if (batch_idx >= B || token_idx >= N) return;
    
    float total_entropy = 0.0f;
    
    extern __shared__ float shared[];
    float* attn_scores = shared;
    float* attn_probs = shared + N;
    
    for (int layer = 0; layer < L; ++layer) {
        const __half* X_layer = X + layer * B * N * D;
        const __half* x_i = X_layer + (batch_idx * N + token_idx) * D;
        
        for (int j = 0; j < N; ++j) {
            float score = 0.0f;
            const __half* x_j = X_layer + (batch_idx * N + j) * D;
            
            for (int d = 0; d < D; ++d) {
                score += __half2float(x_i[d]) * __half2float(x_j[d]);
            }
            attn_scores[j] = score * sqrt_D_inv;
        }
        
        float max_score = attn_scores[0];
        for (int j = 1; j < N; ++j) {
            max_score = fmaxf(max_score, attn_scores[j]);
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            attn_probs[j] = expf(attn_scores[j] - max_score);
            sum_exp += attn_probs[j];
        }
        
        float entropy = 0.0f;
        for (int j = 0; j < N; ++j) {
            attn_probs[j] /= sum_exp;
            if (attn_probs[j] > 1e-8f) {
                entropy -= attn_probs[j] * logf(attn_probs[j]);
            }
        }
        total_entropy += entropy;
    }
    entropy_scores[tid] = __float2half(total_entropy / L);
} 