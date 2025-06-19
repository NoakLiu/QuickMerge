// QuickMerge++ Core CUDA Kernels
// Author: Dong Liu and Yanxuan Yu
// ICML 2025

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// 1. Attention Entropy Kernel
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
    for (int layer = 0; layer < L; ++layer) {
        const float* X_layer = X + layer * B * N * D;
        // Compute attention scores
        extern __shared__ float shared[];
        float* attn_scores = shared;
        float* attn_probs = shared + N;
        for (int j = 0; j < N; ++j) {
            float score = 0.0f;
            const float* x_i = X_layer + (batch_idx * N + token_idx) * D;
            const float* x_j = X_layer + (batch_idx * N + j) * D;
            for (int d = 0; d < D; ++d) score += x_i[d] * x_j[d];
            attn_scores[j] = score * sqrt_D_inv;
        }
        float max_score = attn_scores[0];
        for (int j = 1; j < N; ++j) max_score = fmaxf(max_score, attn_scores[j]);
        float sum_exp = 0.0f;
        for (int j = 0; j < N; ++j) {
            attn_probs[j] = expf(attn_scores[j] - max_score);
            sum_exp += attn_probs[j];
        }
        for (int j = 0; j < N; ++j) attn_probs[j] /= sum_exp;
        float entropy = 0.0f;
        for (int j = 0; j < N; ++j) if (attn_probs[j] > 1e-8f) entropy -= attn_probs[j] * logf(attn_probs[j]);
        total_entropy += entropy;
    }
    entropy_scores[tid] = total_entropy / L;
}

// 2. Saliency-Weighted Token Merging
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
    float weighted_sum = 0.0f, weight_sum = 0.0f;
    for (int token_idx = 0; token_idx < N; ++token_idx) {
        if (cluster_labels[batch_idx * N + token_idx] == cluster_idx) {
            float w = saliency_mass[batch_idx * N + token_idx];
            weighted_sum += w * X[(batch_idx * N + token_idx) * D + dim_idx];
            weight_sum += w;
        }
    }
    merged_tokens[tid] = (weight_sum > 1e-8f) ? (weighted_sum / weight_sum) : 0.0f;
}

// 3. Cosine Similarity Matrix
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
    const float* x_i = X + (batch_idx * N + i) * D;
    const float* x_j = X + (batch_idx * N + j) * D;
    float dot = 0.0f, norm_i = 0.0f, norm_j = 0.0f;
    for (int d = 0; d < D; ++d) {
        dot += x_i[d] * x_j[d];
        norm_i += x_i[d] * x_i[d];
        norm_j += x_j[d] * x_j[d];
    }
    sim_matrix[tid] = dot / (sqrtf(norm_i) * sqrtf(norm_j) + 1e-8f);
}

// 4. Gumbel-Softmax Sampling
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
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    float u = curand_uniform(&state);
    float gumbel = -logf(-logf(u + 1e-8f) + 1e-8f);
    mask[tid] = (saliency[tid] + gumbel) / temperature;
} 