#include <cmath>
#include <algorithm>
#include <vector>

// ==================== 动态置信度计算函数 ====================

// 计算3D梯度幅值
float calculateGradientMagnitude3D(const float* img, int idx, int W, int H, int D) {
    int x = idx % W;
    int y = (idx / W) % H;
    int z = idx / (W * H);
    
    float gx = 0, gy = 0, gz = 0;
    
    // X方向梯度
    if (x > 0 && x < W-1) {
        int idx_left = z * W * H + y * W + (x-1);
        int idx_right = z * W * H + y * W + (x+1);
        gx = (img[idx_right] - img[idx_left]) / 2.0f;
    }
    
    // Y方向梯度
    if (y > 0 && y < H-1) {
        int idx_up = z * W * H + (y-1) * W + x;
        int idx_down = z * W * H + (y+1) * W + x;
        gy = (img[idx_down] - img[idx_up]) / 2.0f;
    }
    
    // Z方向梯度
    if (z > 0 && z < D-1) {
        int idx_front = (z-1) * W * H + y * W + x;
        int idx_back = (z+1) * W * H + y * W + x;
        gz = (img[idx_back] - img[idx_front]) / 2.0f;
    }
    
    return sqrt(gx*gx + gy*gy + gz*gz);
}

// 计算到前景边界的距离
float calculateDistanceToForegroundBoundary(const short* anno, int idx, int W, int H, int D, int max_dist = 10) {
    int x = idx % W;
    int y = (idx / W) % H;
    int z = idx / (W * H);
    
    bool is_fg = (anno[idx] == 0);  // 假设0是前景
    
    // 如果是背景点，返回最大距离
    if (!is_fg) return max_dist;
    
    float min_dist = max_dist;
    
    // 搜索周围的背景点
    for (int dz = -max_dist; dz <= max_dist; dz++) {
        for (int dy = -max_dist; dy <= max_dist; dy++) {
            for (int dx = -max_dist; dx <= max_dist; dx++) {
                int nx = x + dx, ny = y + dy, nz = z + dz;
                
                // 边界检查
                if (nx < 0 || nx >= W || ny < 0 || ny >= H || nz < 0 || nz >= D) continue;
                
                int nidx = nz * W * H + ny * W + nx;
                
                // 如果找到背景点
                if (anno[nidx] != 0) {
                    float dist = sqrt(dx*dx + dy*dy + dz*dz);
                    min_dist = std::min(min_dist, dist);
                }
            }
        }
    }
    
    return min_dist;
}

// 计算局部强度统计
struct LocalStats {
    float mean;
    float std;
    float min_val;
    float max_val;
};

LocalStats calculateLocalStats(const float* img, int idx, int W, int H, int D, int radius = 3) {
    int x = idx % W;
    int y = (idx / W) % H;
    int z = idx / (W * H);
    
    std::vector<float> values;
    
    for (int dz = -radius; dz <= radius; dz++) {
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = x + dx, ny = y + dy, nz = z + dz;
                
                if (nx >= 0 && nx < W && ny >= 0 && ny < H && nz >= 0 && nz < D) {
                    int nidx = nz * W * H + ny * W + nx;
                    values.push_back(img[nidx]);
                }
            }
        }
    }
    
    LocalStats stats;
    stats.mean = 0;
    stats.min_val = values[0];
    stats.max_val = values[0];
    
    for (float val : values) {
        stats.mean += val;
        stats.min_val = std::min(stats.min_val, val);
        stats.max_val = std::max(stats.max_val, val);
    }
    stats.mean /= values.size();
    
    // 计算标准差
    float variance = 0;
    for (float val : values) {
        variance += (val - stats.mean) * (val - stats.mean);
    }
    stats.std = sqrt(variance / values.size());
    
    return stats;
}

// ==================== 动态置信度策略 ====================

// 策略1：基于边界距离的置信度
float calculateDistanceBasedConfidence(const short* anno, int idx, int W, int H, int D) {
    float dist = calculateDistanceToForegroundBoundary(anno, idx, W, H, D, 8);
    
    // 距离边界越远，置信度越高
    float confidence = 0.85f + 0.13f * std::min(1.0f, dist / 5.0f);
    
    return confidence;
}

// 策略2：基于梯度的置信度
float calculateGradientBasedConfidence(const float* img, const short* anno, int idx, int W, int H, int D) {
    if (anno[idx] != 0) return 0.02f;  // 背景固定低置信度
    
    float grad = calculateGradientMagnitude3D(img, idx, W, H, D);
    
    // 梯度强的地方（边界）置信度低，梯度弱的地方（内部）置信度高
    float normalized_grad = std::min(1.0f, grad / 50.0f);  // 假设最大梯度50
    float confidence = 0.99f - 0.25f * normalized_grad;    // 0.74 - 0.99
    
    return confidence;
}

// 策略3：混合策略（推荐）
float calculateHybridConfidence(const float* img, const short* anno, int idx, int W, int H, int D) {
    if (anno[idx] != 0) return 0.02f;  // 背景固定
    
    // 1. 距离因子
    float dist = calculateDistanceToForegroundBoundary(anno, idx, W, H, D, 6);
    float dist_factor = std::min(1.0f, dist / 3.0f);  // 0-1
    
    // 2. 梯度因子
    float grad = calculateGradientMagnitude3D(img, idx, W, H, D);
    float grad_factor = 1.0f - std::min(1.0f, grad / 30.0f);  // 1-0
    
    // 3. 局部一致性因子
    LocalStats stats = calculateLocalStats(img, idx, W, H, D, 2);
    float consistency_factor = 1.0f - std::min(1.0f, stats.std / 20.0f);  // 标准差小=一致性高
    
    // 加权组合
    float confidence = 0.75f +  // 基础置信度
                      0.15f * dist_factor +      // 距离贡献
                      0.08f * grad_factor +      // 梯度贡献  
                      0.02f * consistency_factor; // 一致性贡献
    
    return std::min(0.99f, std::max(0.75f, confidence));
}

// 策略4：自适应边界置信度
float calculateAdaptiveBoundaryConfidence(const float* img, const short* anno, int idx, int W, int H, int D) {
    if (anno[idx] != 0) return 0.02f;  // 背景
    
    // 计算是否在边界附近
    bool is_boundary = false;
    int x = idx % W;
    int y = (idx / W) % H;
    int z = idx / (W * H);
    
    // 检查6连通邻域
    int neighbors[][3] = {{-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}};
    for (int i = 0; i < 6; i++) {
        int nx = x + neighbors[i][0];
        int ny = y + neighbors[i][1]; 
        int nz = z + neighbors[i][2];
        
        if (nx >= 0 && nx < W && ny >= 0 && ny < H && nz >= 0 && nz < D) {
            int nidx = nz * W * H + ny * W + nx;
            if (anno[nidx] != 0) {  // 发现背景邻居
                is_boundary = true;
                break;
            }
        }
    }
    
    if (is_boundary) {
        // 边界点：基于图像梯度决定置信度
        float grad = calculateGradientMagnitude3D(img, idx, W, H, D);
        
        // 如果梯度强（真实边界），给中等置信度
        // 如果梯度弱（可能的误分），给低置信度
        if (grad > 15.0f) {
            return 0.85f;  // 真实边界，中等置信度
        } else {
            return 0.70f;  // 可能误分，低置信度
        }
    } else {
        // 内部点：高置信度
        return 0.95f;
    }
}

// ==================== 主要接口函数 ====================

// 计算整个体积的动态置信度
void calculateDynamicConfidence(const float* hostImg, const short* hostAnno, 
                               std::vector<float>& unary_host, 
                               int W, int H, int D, int M,
                               int strategy = 3) {
    size_t N = W * H * D;
    
    for (size_t i = 0; i < N; ++i) {
        float confidence;
        
        switch(strategy) {
            case 1:
                confidence = calculateDistanceBasedConfidence(hostAnno, i, W, H, D);
                break;
            case 2:
                confidence = calculateGradientBasedConfidence(hostImg, hostAnno, i, W, H, D);
                break;
            case 3:
                confidence = calculateHybridConfidence(hostImg, hostAnno, i, W, H, D);
                break;
            case 4:
                confidence = calculateAdaptiveBoundaryConfidence(hostImg, hostAnno, i, W, H, D);
                break;
            default:
                confidence = 0.95f;  // 默认固定置信度
        }
        
        bool is_fg = (hostAnno[i] == 0);
        
        if (is_fg) {
            // 前景点使用动态置信度
            unary_host[i * M + 1] = -logf(confidence);
            unary_host[i * M + 0] = -logf(1.0f - confidence);
        } else {
            // 背景点使用固定低置信度
            unary_host[i * M + 1] = -logf(0.02f);
            unary_host[i * M + 0] = -logf(0.98f);
        }
    }
}

// ==================== 调试和可视化函数 ====================

void analyzeConfidenceDistribution(const std::vector<float>& unary_host, int N, int M) {
    std::vector<float> confidences;

    for (int i = 0; i < N; i++) {
        // 从unary energy反推置信度
        float fg_energy = unary_host[i * M + 1];
        float confidence = exp(-fg_energy);
        confidences.push_back(confidence);
    }
    
    // 统计分析
    std::sort(confidences.begin(), confidences.end());
    
    std::cout << "Confidence distribution:\n";
    std::cout << "Min: " << confidences.front() << "\n";
    std::cout << "25%: " << confidences[N/4] << "\n"; 
    std::cout << "50%: " << confidences[N/2] << "\n";
    std::cout << "75%: " << confidences[3*N/4] << "\n";
    std::cout << "Max: " << confidences.back() << "\n";
}