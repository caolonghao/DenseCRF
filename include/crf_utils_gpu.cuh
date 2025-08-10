#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math_constants.h>

#ifdef CUDA_CHECK
#undef CUDA_CHECK
#endif
#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                   __FILE__, __LINE__, cudaGetErrorString(err__));            \
      cudaDeviceReset();                                                      \
      std::abort(); /* 或者使用: exit(EXIT_FAILURE); */                       \
    }                                                                         \
  } while (0)

// --------------------- 基础工具 ---------------------
__device__ __forceinline__
int lin3d(int x, int y, int z, int W, int H) { return z * W * H + y * W + x; }

__device__ __forceinline__
float sat_at(const float* __restrict__ sat, int x, int y, int z, int W, int H, int D){
    // 积分体是包含式前缀和（0..idx），对于 <0 视作 0
    if (x < 0 || y < 0 || z < 0) return 0.0f;
    if (x >= W) x = W - 1;
    if (y >= H) y = H - 1;
    if (z >= D) z = D - 1;
    return sat[lin3d(x, y, z, W, H)];
}

__device__ __forceinline__
float sat_sum_box(const float* __restrict__ sat,
                  int x0, int y0, int z0,
                  int x1, int y1, int z1,
                  int W, int H, int D){
    // 3D 包含-排除：S(1,1,1)-S(0,1,1)-S(1,0,1)-S(1,1,0)+S(0,0,1)+S(0,1,0)+S(1,0,0)-S(0,0,0)
    float S111 = sat_at(sat, x1, y1, z1, W, H, D);
    float S011 = sat_at(sat, x0 - 1, y1, z1, W, H, D);
    float S101 = sat_at(sat, x1, y0 - 1, z1, W, H, D);
    float S110 = sat_at(sat, x1, y1, z0 - 1, W, H, D);
    float S001 = sat_at(sat, x0 - 1, y0 - 1, z1, W, H, D);
    float S010 = sat_at(sat, x0 - 1, y1, z0 - 1, W, H, D);
    float S100 = sat_at(sat, x1, y0 - 1, z0 - 1, W, H, D);
    float S000 = sat_at(sat, x0 - 1, y0 - 1, z0 - 1, W, H, D);
    return S111 - S011 - S101 - S110 + S001 + S010 + S100 - S000;
}

// --------------------- 距离近似（BFS 波前） ---------------------
__global__ void init_distance_kernel(const short* __restrict__ anno,
                                     uint8_t* __restrict__ dist,
                                     int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x) {
        // 你的约定：0=前景，非0 或 -1 视作背景/未知
        dist[idx] = (anno[idx] == 0) ? 255 : 0;
    }
}

__global__ void bfs_expand_kernel(const uint8_t* __restrict__ dist_prev,
                                  uint8_t* __restrict__ dist_next,
                                  const short* __restrict__ anno,
                                  int W, int H, int D,
                                  uint8_t t)
{
    int N = W * H * D;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x) {
        if (anno[idx] != 0) continue;          // 仅前景中扩散
        if (dist_prev[idx] != 255) continue;   // 尚未赋值的才可能被触达

        int z = idx / (W * H);
        int y = (idx / W) % H;
        int x = idx % W;

        bool touched = false;
        if (x > 0)           touched |= (dist_prev[idx - 1] == (t - 1));
        if (x + 1 < W)       touched |= (dist_prev[idx + 1] == (t - 1));
        if (y > 0)           touched |= (dist_prev[idx - W] == (t - 1));
        if (y + 1 < H)       touched |= (dist_prev[idx + W] == (t - 1));
        if (z > 0)           touched |= (dist_prev[idx - W * H] == (t - 1));
        if (z + 1 < D)       touched |= (dist_prev[idx + W * H] == (t - 1));

        dist_next[idx] = touched ? t : dist_prev[idx];
    }
}

// --------------------- 3D 积分图：构建 ---------------------
// init: s1 = img, s2 = img*img
__global__ void sat_init_kernel(const float* __restrict__ img,
                                float* __restrict__ s1,
                                float* __restrict__ s2,
                                int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x) {
        float v = img[idx];
        s1[idx] = v;
        s2[idx] = v * v;
    }
}

// 对 X 方向做前缀和：每条 (y,z) 线由一个线程顺序扫描
__global__ void prefix_sum_x_kernel(float* __restrict__ arr, int W, int H, int D)
{
    int lines = H * D;
    int lid = blockDim.x * blockIdx.x + threadIdx.x;
    if (lid >= lines) return;
    int z = lid / H;
    int y = lid % H;
    int base = z * W * H + y * W;
    float acc = 0.f;
    #pragma unroll 4
    for (int x = 0; x < W; ++x){
        int idx = base + x;
        acc += arr[idx];
        arr[idx] = acc;
    }
}

// 对 Y 方向做前缀和：每条 (x,z) 线一个线程
__global__ void prefix_sum_y_kernel(float* __restrict__ arr, int W, int H, int D)
{
    int lines = W * D;
    int lid = blockDim.x * blockIdx.x + threadIdx.x;
    if (lid >= lines) return;
    int z = lid / W;
    int x = lid % W;
    int base = z * W * H + x;
    float acc = 0.f;
    #pragma unroll 4
    for (int y = 0; y < H; ++y){
        int idx = base + y * W;
        acc += arr[idx];
        arr[idx] = acc;
    }
}

// 对 Z 方向做前缀和：每条 (x,y) 线一个线程
__global__ void prefix_sum_z_kernel(float* __restrict__ arr, int W, int H, int D)
{
    int lines = W * H;
    int lid = blockDim.x * blockIdx.x + threadIdx.x;
    if (lid >= lines) return;
    int y = lid / W;
    int x = lid % W;
    int base = y * W + x;
    float acc = 0.f;
    #pragma unroll 4
    for (int z = 0; z < D; ++z){
        int idx = z * W * H + base;
        acc += arr[idx];
        arr[idx] = acc;
    }
}

// Host 端封装：构建两张积分体（s1, s2）
inline void build_integral3d(const float* imgGPU, int W, int H, int D,
                             float*& s1, float*& s2)
{
    const int N = W * H * D;
    CUDA_CHECK(cudaMalloc(&s1, sizeof(float)*N));
    CUDA_CHECK(cudaMalloc(&s2, sizeof(float)*N));

    // init
    {
        dim3 blk(256);
        dim3 grd((N + blk.x - 1) / blk.x);
        sat_init_kernel<<<grd, blk>>>(imgGPU, s1, s2, N);
        CUDA_CHECK(cudaGetLastError());
    }
    // X
    {
        int lines = H * D;
        dim3 blk(256);
        dim3 grd((lines + blk.x - 1) / blk.x);
        prefix_sum_x_kernel<<<grd, blk>>>(s1, W, H, D);
        prefix_sum_x_kernel<<<grd, blk>>>(s2, W, H, D);
        CUDA_CHECK(cudaGetLastError());
    }
    // Y
    {
        int lines = W * D;
        dim3 blk(256);
        dim3 grd((lines + blk.x - 1) / blk.x);
        prefix_sum_y_kernel<<<grd, blk>>>(s1, W, H, D);
        prefix_sum_y_kernel<<<grd, blk>>>(s2, W, H, D);
        CUDA_CHECK(cudaGetLastError());
    }
    // Z
    {
        int lines = W * H;
        dim3 blk(256);
        dim3 grd((lines + blk.x - 1) / blk.x);
        prefix_sum_z_kernel<<<grd, blk>>>(s1, W, H, D);
        prefix_sum_z_kernel<<<grd, blk>>>(s2, W, H, D);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// --------------------- 核心：计算 unary ---------------------
__global__ void compute_unary_kernel(
    const float* __restrict__ img,
    const short* __restrict__ anno,
    const uint8_t* __restrict__ dist6, // 可空
    const float* __restrict__ sat1,    // 可空：sum(img)
    const float* __restrict__ sat2,    // 可空：sum(img^2)
    float* __restrict__ unary,         // 输出: N*M
    int W, int H, int D, int M,
    int strategy,
    int max_dist_for_scale,
    int local_radius
){
    int N = W * H * D;
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    for (int idx = i; idx < N; idx += blockDim.x * gridDim.x) {
        const bool is_fg = (anno[idx] == 0);

        // 坐标
        int z = idx / (W * H);
        int y = (idx / W) % H;
        int x = idx % W;

        auto get = [&](int xx, int yy, int zz)->float{
            if (xx < 0 || xx >= W || yy < 0 || yy >= H || zz < 0 || zz >= D) return 0.0f;
            return img[lin3d(xx, yy, zz, W, H)];
        };

        // 1) 梯度（中心差分）
        float gx = 0.f, gy = 0.f, gz = 0.f;
        if (x > 0 && x + 1 < W) gx = 0.5f * (get(x + 1, y, z) - get(x - 1, y, z));
        if (y > 0 && y + 1 < H) gy = 0.5f * (get(x, y + 1, z) - get(x, y - 1, z));
        if (z > 0 && z + 1 < D) gz = 0.5f * (get(x, y, z + 1) - get(x, y, z - 1));
        float grad_mag = sqrtf(gx*gx + gy*gy + gz*gz);

        // 2) 边界判定（6邻域）
        bool is_boundary = false;
        if (is_fg){
            if ((x > 0     && anno[idx - 1]       != 0) ||
                (x + 1 < W && anno[idx + 1]       != 0) ||
                (y > 0     && anno[idx - W]       != 0) ||
                (y + 1 < H && anno[idx + W]       != 0) ||
                (z > 0     && anno[idx - W*H]     != 0) ||
                (z + 1 < D && anno[idx + W*H]     != 0))
                is_boundary = true;
        }

        // 3) 局部统计（积分体可 O(1) 求和）
        float mean = 0.f, stdv = 0.f;
        if (sat1 && sat2){
            int r = local_radius;
            int x0 = x - r, x1 = x + r;
            int y0 = y - r, y1 = y + r;
            int z0 = z - r, z1 = z + r;
            if (x0 < 0) x0 = 0; if (x1 >= W) x1 = W - 1;
            if (y0 < 0) y0 = 0; if (y1 >= H) y1 = H - 1;
            if (z0 < 0) z0 = 0; if (z1 >= D) z1 = D - 1;

            int nx = x1 - x0 + 1;
            int ny = y1 - y0 + 1;
            int nz = z1 - z0 + 1;
            float cnt = float(nx * ny * nz);

            float sum1 = sat_sum_box(sat1, x0, y0, z0, x1, y1, z1, W, H, D);
            float sum2 = sat_sum_box(sat2, x0, y0, z0, x1, y1, z1, W, H, D);

            mean = sum1 / cnt;
            float var = fmaxf(0.f, sum2 / cnt - mean * mean);
            stdv = sqrtf(var);
        } else {
            // 不需要局部统计（如策略 1/2/4）时，避免开销
            mean = 0.f; stdv = 0.f;
        }

        // 4) 置信度
        float confidence = 0.95f;
        if (!is_fg){
            confidence = 0.02f;
        } else {
            switch (strategy){
            case 1: {
                float d = 0.f;
                if (dist6) {
                    float di = float(dist6[idx]);
                    if (di > max_dist_for_scale) di = float(max_dist_for_scale);
                    d = di;
                }
                float df = fminf(1.f, d / 5.0f);
                confidence = 0.85f + 0.13f * df;
                break;
            }
            case 2: {
                float ng = fminf(1.f, grad_mag / 50.0f);
                confidence = 0.99f - 0.25f * ng;
                break;
            }
            case 3: {
                float d = 0.f;
                if (dist6) {
                    float di = float(dist6[idx]);
                    if (di > max_dist_for_scale) di = float(max_dist_for_scale);
                    d = di;
                }
                float dist_factor = fminf(1.f, d / 3.0f);
                float grad_factor = 1.0f - fminf(1.f, grad_mag / 30.0f);
                float consistency  = 1.0f - fminf(1.f, stdv / 20.0f);

                confidence = 0.75f
                           + 0.15f * dist_factor
                           + 0.08f * grad_factor
                           + 0.02f * consistency;
                confidence = fminf(0.99f, fmaxf(0.75f, confidence));
                break;
            }
            case 4: {
                if (is_boundary) confidence = (grad_mag > 15.0f) ? 0.85f : 0.70f;
                else             confidence = 0.95f;
                break;
            }
            default:
                confidence = 0.95f;
            }
        }

        // 5) 写 unary（M==2）
        float p_fg = confidence;
        float p_bg = 1.0f - p_fg;
        unary[idx * M + 1] = -__logf(fmaxf(1e-6f, p_fg));
        unary[idx * M + 0] = -__logf(fmaxf(1e-6f, p_bg));
    }
}
