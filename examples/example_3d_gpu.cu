#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "densecrf_gpu.cuh"
#include "pairwise_gpu.cuh"
#include "nifti1_io.h"

using namespace dcrf_cuda;
using namespace std::chrono;

// Number of labels (must match your annotation labels count)
constexpr int M = 2;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image.nii> <annotation.nii> <output.nii>\n";
        return 1;
    }

    // --- 1) Read input image ---
    nifti_image* img = nifti_image_read(argv[1], 1);
    if (!img) {
        std::cerr << "Failed to read image: " << argv[1] << "\n";
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
    nifti_image* anno = nifti_image_read(argv[2], 1);
    if (!anno) {
        std::cerr << "Failed to read annotation: " << argv[2] << "\n";
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

    // Free host image buffer if we allocated one
    if (img->datatype != NIFTI_TYPE_FLOAT32) delete[] hostImg;

    // --- 4) Set up DenseCRF ---
    DenseCRFGPU<M> crf(N);
    // Unary from ground truth
    // crf.setUnaryEnergyFromLabel(labelGPU, 0.5f);
    std::vector<float> unary_host(N * M);
    for (size_t i=0; i<N; ++i){
        bool fg = (hostAnno[i] == 0); // 原逻辑: 原 mask=1 => hostAnno=0
        float p = fg ? 0.9f : 0.05f;  // 可调: 0.85/0.1 等
        unary_host[i*M + 1] = -logf(p);
        unary_host[i*M + 0] = -logf(1.0f - p);
    }
    float* unary_dev;
    cudaMalloc(&unary_dev, sizeof(float)*N*M);
    cudaMemcpy(unary_dev, unary_host.data(), sizeof(float)*N*M, cudaMemcpyHostToDevice);
    crf.setUnaryEnergy(unary_dev);
    cudaFree(unary_dev);

    // Pairwise: 3D smoothness (xyz)
    auto* smooth3d = PottsPotentialGPU<M,3>::FromImage3D<float>(
        W, H, D,
        3.0f,    // weight
        3.0f,    // posdev
        nullptr, // features
        0.0f     // featuredev
    );
    crf.addPairwiseEnergy(smooth3d);

    // Pairwise: 3D appearance (xyz + intensity)
    auto* appear3d = PottsPotentialGPU<M,4>::FromImage3D(
        W, H, D,
        /*weight=*/10.0f,
        /*posdev=*/60.0f,
        /*features=*/imgGPU,
        /*featuredev=*/20.0f
    );
    crf.addPairwiseEnergy(appear3d);

    // --- 5) Run inference ---
    // auto t0 = steady_clock::now();
    // crf.inference(/*iterations=*/10, /*with_map=*/true);
    // auto t1 = steady_clock::now();
    // double elapsed = duration<double, std::milli>(t1 - t0).count();
    // std::cout << "Inference time: " << elapsed << " ms\n";

    auto t_start = std::chrono::high_resolution_clock::now();
    crf.inference(10, true);
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
    nifti_set_filenames(out, argv[3], /*use_ext=*/0, /*check=*/1);
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