#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <string>
#include <map>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "densecrf_gpu.cuh"
#include "pairwise_gpu.cuh"
#include "crf_utils.h"
#include "nifti1_io.h"
#include "crf_utils_gpu.cuh"

using namespace dcrf_cuda;
using namespace std::chrono;

// Number of labels (must match your annotation labels count)
constexpr int M = 2;

// Helper to parse command line arguments
void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name
              << " --image <path> --annotation <path> --output <path>"
              << " [--confidence_strategy <1-4>]"
              << " [--smooth3d_w <float>] [--smooth3d_posdev <float>]"
              << " [--appear3d_w <float>] [--appear3d_posdev <float>] [--appear3d_featuredev <float>]"
              << " [--inference_iter <int>]"
              << " [--local_radius <int>]  (0=disable local stats in strategy 3)\n";
}

int main(int argc, char** argv) {
    // --- 0) Parse Arguments ---
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0 && i + 1 < argc) {
            args[arg] = argv[++i];
        }
    }

    const char* image_path = args.count("--image") ? args["--image"].c_str() : nullptr;
    const char* anno_path = args.count("--annotation") ? args["--annotation"].c_str() : nullptr;
    const char* output_path = args.count("--output") ? args["--output"].c_str() : nullptr;

    if (!image_path || !anno_path || !output_path) {
        print_usage(argv[0]);
        return 1;
    }

    // Default CRF parameters
    int local_radius = 2;   // 默认 2（5x5x5 体素窗口）；设为 0 可关闭（策略3下不再构建积分体）
    int confidence_strategy = 3;
    float smooth3d_w = 0.3f;
    float smooth3d_posdev = 2.5f;
    float appear3d_w = 2.0f;
    float appear3d_posdev = 35.0f;
    float appear3d_featuredev = 12.0f;
    int inference_iter = 8;

    // Override defaults from args
    if (args.count("--confidence_strategy")) confidence_strategy = std::stoi(args["--confidence_strategy"]);
    if (args.count("--smooth3d_w")) smooth3d_w = std::stof(args["--smooth3d_w"]);
    if (args.count("--smooth3d_posdev")) smooth3d_posdev = std::stof(args["--smooth3d_posdev"]);
    if (args.count("--appear3d_w")) appear3d_w = std::stof(args["--appear3d_w"]);
    if (args.count("--appear3d_posdev")) appear3d_posdev = std::stof(args["--appear3d_posdev"]);
    if (args.count("--appear3d_featuredev")) appear3d_featuredev = std::stof(args["--appear3d_featuredev"]);
    if (args.count("--inference_iter")) inference_iter = std::stoi(args["--inference_iter"]);
    if (args.count("--local_radius")) local_radius = std::stoi(args["--local_radius"]);
    // 简单约束，避免极端值；如需更大窗口可自行调高上限
    if (local_radius < 0) local_radius = 0;
    if (local_radius > 8) local_radius = 8;

    std::cout << "--- CRF Parameters ---\n"
              << "confidence_strategy: " << confidence_strategy << "\n"
              << "smooth3d_w: " << smooth3d_w << "\n"
              << "smooth3d_posdev: " << smooth3d_posdev << "\n"
              << "appear3d_w: " << appear3d_w << "\n"
              << "appear3d_posdev: " << appear3d_posdev << "\n"
              << "appear3d_featuredev: " << appear3d_featuredev << "\n"
              << "inference_iter: " << inference_iter << "\n"
              << "local_radius: " << local_radius << "\n"
              << "----------------------\n";

    // --- 1) Read input image ---
    nifti_image* img = nifti_image_read(image_path, 1);
    if (!img) {
        std::cerr << "Failed to read image: " << image_path << "\n";
        return 1;
    }
    int W = img->nx, H = img->ny, D = img->nz;
    size_t N = size_t(W) * H * D;

    std::cout << "Image size: " << W << "x" << H << "x" << D << " (" << N << " voxels)\n";

    // Ensure image data in float array
    float* hostImg = nullptr;

    // std::cout << "img->datatype = " << img->datatype << std::endl;
    // std::cout << "NIFTI_TYPE_INT16 = " << NIFTI_TYPE_INT16 << std::endl;
    if (img->datatype == NIFTI_TYPE_FLOAT32) {
        hostImg = static_cast<float*>(img->data);
    } else {
        hostImg = new float[N];
        if (img->datatype == NIFTI_TYPE_UINT8) {
            auto ptr = static_cast<uint8_t*>(img->data);
            for (size_t i = 0; i < N; ++i) hostImg[i] = ptr[i];
        } else if (img->datatype == NIFTI_TYPE_INT16) {
            auto ptr = static_cast<int16_t*>(img->data);
            for (size_t i = 0; i < N; ++i) hostImg[i] = ptr[i];
        } else {
            std::cerr << "Unsupported image datatype.\n";
            nifti_image_free(img);
            delete[] hostImg;
            return 1;
        }
    }

    // --- 2) Read annotation volume ---
    nifti_image* anno = nifti_image_read(anno_path, 1);
    if (!anno) {
        std::cerr << "Failed to read annotation: " << anno_path << "\n";
        if (img->datatype != NIFTI_TYPE_FLOAT32) delete[] hostImg;
        nifti_image_free(img);
        return 1;
    }
    if (anno->nx != W || anno->ny != H || anno->nz != D) {
        std::cerr << "Dimension mismatch between image and annotation.\n";
        if (img->datatype != NIFTI_TYPE_FLOAT32) delete[] hostImg;
        nifti_image_free(img);
        nifti_image_free(anno);
        return 1;
    }

    // Convert annotation into labels [-1 = unknown, 0..M-1]
    short* hostAnno = new short[N];
    if (anno->datatype == NIFTI_TYPE_INT16) {
        auto ptr = static_cast<int16_t*>(anno->data);
        for (size_t i = 0; i < N; ++i) {
            if (ptr[i] == 0) hostAnno[i] = -1;
            else           hostAnno[i] = ptr[i] - 1;
        }
    } else if (anno->datatype == NIFTI_TYPE_UINT8) {
        auto ptr = static_cast<uint8_t*>(anno->data);
        for (size_t i = 0; i < N; ++i) {
            if (ptr[i] == 0) hostAnno[i] = -1;
            else             hostAnno[i] = ptr[i] - 1;
        }
    } else if (anno->datatype == NIFTI_TYPE_UINT16) { // ✅ ADD THIS NEW BRANCH
    // Datatype 512 is UINT16 (unsigned 16-bit integer)
    auto ptr = static_cast<uint16_t*>(anno->data);
    for (size_t i = 0; i < N; ++i) {
        if (ptr[i] == 0) hostAnno[i] = -1;
        else           hostAnno[i] = ptr[i] - 1;
        }
     } else if (anno->datatype == NIFTI_TYPE_UINT8) { // Your other existing branch
        auto ptr = static_cast<uint8_t*>(anno->data);
        for (size_t i = 0; i < N; ++i) {
            if (ptr[i] == 0) hostAnno[i] = -1;
            else             hostAnno[i] = ptr[i] - 1;
        }
    } else {
        std::cerr << "Unsupported annotation datatype.\n";
        if (img->datatype != NIFTI_TYPE_FLOAT32) delete[] hostImg;
        delete[] hostAnno;
        nifti_image_free(img);
        nifti_image_free(anno);
        return 1;
    }

    // --- 3) Upload data to GPU ---
    float*  imgGPU   = nullptr;
    short*  labelGPU = nullptr;
    cudaMalloc(&imgGPU,   sizeof(float) * N);
    cudaMemcpy(imgGPU, hostImg, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMalloc(&labelGPU, sizeof(short) * N);
    cudaMemcpy(labelGPU, hostAnno, sizeof(short) * N, cudaMemcpyHostToDevice);

    

    // --- 4) Set up DenseCRF ---
    DenseCRFGPU<M> crf(N);
    // Unary from ground truth
    // crf.setUnaryEnergyFromLabel(labelGPU, 0.5f);


    // >>> GPU unary begin（积分体版）
    const int max_dist = 8;        // 与原逻辑一致
    uint8_t *dist_prev = nullptr, *dist_next = nullptr;
    bool need_distance     = (confidence_strategy == 1 || confidence_strategy == 3);
    bool need_local_stats  = (confidence_strategy == 3);

    // 1) （可选）距离（BFS）
    if (need_distance){
        cudaMalloc(&dist_prev, sizeof(uint8_t) * N);
        cudaMalloc(&dist_next, sizeof(uint8_t) * N);
        {
            dim3 blk(256);
            dim3 grd( (N + blk.x - 1) / blk.x );
            init_distance_kernel<<<grd, blk>>>(labelGPU, dist_prev, (int)N);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int t = 1; t <= max_dist; ++t){
            CUDA_CHECK(cudaMemcpy(dist_next, dist_prev, sizeof(uint8_t)*N, cudaMemcpyDeviceToDevice));
            dim3 blk(256);
            dim3 grd( (N + blk.x - 1) / blk.x );
            bfs_expand_kernel<<<grd, blk>>>(dist_prev, dist_next, labelGPU, W, H, D, (uint8_t)t);
            CUDA_CHECK(cudaGetLastError());
            uint8_t* tmp = dist_prev; dist_prev = dist_next; dist_next = tmp;
        }
    }

    // 2) （可选）构建 3D 积分体（仅策略 3 需要）
    float *sat1 = nullptr, *sat2 = nullptr;
    if (need_local_stats){
        build_integral3d(imgGPU, W, H, D, sat1, sat2);
    }

    // 3) 直接在 GPU 上生成 unary
    float* unary_dev = nullptr;
    cudaMalloc(&unary_dev, sizeof(float)*N*M);
    {
        dim3 blk(256);
        dim3 grd( (N + blk.x - 1) / blk.x );
        compute_unary_kernel<<<grd, blk>>>(
            imgGPU, labelGPU,
            need_distance ? dist_prev : nullptr,
            need_local_stats ? sat1 : nullptr,
            need_local_stats ? sat2 : nullptr,
            unary_dev, W, H, D, M,
            confidence_strategy,
            max_dist, local_radius
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    crf.setUnaryEnergy(unary_dev);
    cudaFree(unary_dev);

    // 清理
    if (need_distance){
        cudaFree(dist_prev);
        cudaFree(dist_next);
    }
    if (need_local_stats){
        cudaFree(sat1);
        cudaFree(sat2);
    }

    // Pairwise: 3D smoothness (xyz)
    auto* smooth3d = PottsPotentialGPU<M,3>::FromImage3D<float>(
        W, H, D,
        smooth3d_w,       // weight
        smooth3d_posdev,  // posdev
        nullptr, // features
        0.0f     // featuredev
    );
    crf.addPairwiseEnergy(smooth3d);

    // Pairwise: 3D appearance (xyz + intensity)
    auto* appear3d = PottsPotentialGPU<M,4>::FromImage3D(
        W, H, D,
        /*weight=*/appear3d_w,
        /*posdev=*/appear3d_posdev,
        /*features=*/imgGPU,
        /*featuredev=*/appear3d_featuredev
    );
    crf.addPairwiseEnergy(appear3d);

    // --- 5) Run inference ---
    // auto t0 = steady_clock::now();
    // crf.inference(/*iterations=*/10, /*with_map=*/true);
    // auto t1 = steady_clock::now();
    // double elapsed = duration<double, std::milli>(t1 - t0).count();
    // std::cout << "Inference time: " << elapsed << " ms\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    crf.inference(inference_iter, true);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Total inference host time: "
            << std::chrono::duration<double,std::milli>(t_end - t_start).count()
            << " ms\n";

    // Retrieve result map
    short* mapGPU  = crf.getMap();
    short* hostMap = new short[N];
    cudaMemcpy(hostMap, mapGPU, sizeof(short) * N, cudaMemcpyDeviceToHost);

    // --- 6) Write output NIfTI ---
    nifti_image* out = nifti_copy_nim_info(img);
    out->datatype = NIFTI_TYPE_INT16;
    out->nbyper   = sizeof(short);
    out->scl_slope = 1.0f;     // 关键：重置缩放
    out->scl_inter = 0.0f;
    out->cal_min   = 0;        // （可选）标注统计范围
    out->cal_max   = M - 1;
    // Replace data pointer
    if (out->data) free(out->data);
    out->data = hostMap;
    nifti_set_filenames(out, output_path, /*use_ext=*/0, /*check=*/1);
    nifti_image_write(out);
    
    

    std::cout << "out->datatype=" << out->datatype
          << " nvox=" << out->nvox
          << " scl_slope=" << out->scl_slope
          << " scl_inter=" << out->scl_inter << std::endl;


    // --- Cleanup ---
    cudaFree(imgGPU);
    cudaFree(labelGPU);
    delete[] hostAnno;
    nifti_image_free(img);
    nifti_image_free(anno);
    nifti_image_free(out);

    return 0;
}

// ========== 额外的调试和分析功能 ==========

// 可选：保存置信度图用于可视化
void saveConfidenceMap(const std::vector<float>& unary_host, 
                      const nifti_image* template_img,
                      const char* filename, int N, int M) {
    
    // 创建置信度数据
    float* conf_data = new float[N];
    for (int i = 0; i < N; i++) {
        float fg_energy = unary_host[i * M + 1];
        conf_data[i] = exp(-fg_energy);  // 转换回置信度
    }
    
    // 创建NIfTI图像
    nifti_image* conf_img = nifti_copy_nim_info(template_img);
    conf_img->datatype = NIFTI_TYPE_FLOAT32;
    conf_img->nbyper = sizeof(float);
    conf_img->scl_slope = 1.0f;
    conf_img->scl_inter = 0.0f;
    
    if (conf_img->data) free(conf_img->data);
    conf_img->data = conf_data;
    
    nifti_set_filenames(conf_img, filename, 0, 1);
    nifti_image_write(conf_img);
    
    // 注意：不要free conf_data，因为它被nifti_image拥有
    nifti_image_free(conf_img);
    
    std::cout << "Confidence map saved to: " << filename << std::endl;
}

// ========== 不同策略的性能对比测试 ==========
void compareConfidenceStrategies(const float* hostImg, const short* hostAnno, 
                                int W, int H, int D, int M) {
    size_t N = W * H * D;
    
    for (int strategy = 1; strategy <= 4; strategy++) {
        std::vector<float> unary_test(N * M);
        
        auto start = std::chrono::high_resolution_clock::now();
        calculateDynamicConfidence(hostImg, hostAnno, unary_test, W, H, D, M, strategy);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double,std::milli>(end - start).count();
        
        std::cout << "Strategy " << strategy << " time: " << time_ms << " ms" << std::endl;
        analyzeConfidenceDistribution(unary_test, N, M);
        std::cout << "---" << std::endl;
    }
}