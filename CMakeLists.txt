cmake_minimum_required(VERSION 3.8)
project(densecrf_cuda LANGUAGES CXX)

# provide FindCUDAToolkit for cmake versions below 3.17
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
enable_language(CUDA)

# 手动查找 nifti1_io.h
find_path(NIFTI_INCLUDE_DIR
  NAMES nifti1_io.h
  HINTS
    ${NIFTI_ROOT}/include
    /usr/local/include
    /usr/include
  PATH_SUFFIXES nifti
)
# 手动查找 libniftiio
find_library(NIFTIIO_LIB
  NAMES niftiio
  HINTS
    ${NIFTI_ROOT}/lib
    /usr/local/lib
    /usr/lib
)
# 手动查找 libznz (压缩支持)
find_library(ZNZ_LIB
  NAMES znz
  HINTS
    ${NIFTI_ROOT}/lib
    /usr/local/lib
    /usr/lib
    /usr/lib/x86_64-linux-gnu
)
find_package(ZLIB REQUIRED)  # znz 库通常依赖 zlib

if (NOT NIFTI_INCLUDE_DIR OR NOT NIFTIIO_LIB OR NOT ZNZ_LIB)
  message(FATAL_ERROR
    "Could not find NIfTI:\n"
    "  - nifti1_io.h  => ${NIFTI_INCLUDE_DIR}\n"
    "  - libniftiio   => ${NIFTIIO_LIB}\n"
    "  - libznz       => ${ZNZ_LIB}\n"
    "Install nifti_clib (nifti1-dev/nifti-devel) or pass -DNIFTI_ROOT=/your/prefix"
  )
endif()
message(STATUS "NIfTI include: ${NIFTI_INCLUDE_DIR}")
message(STATUS "NIfTI libraries: ${NIFTIIO_LIB}; ${ZNZ_LIB}")

find_package(ZLIB REQUIRED)
# find_package(NIFTI REQUIRED)

add_library(${PROJECT_NAME}_cpu INTERFACE)
target_include_directories(${PROJECT_NAME}_cpu INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>/")

add_library(${PROJECT_NAME}_cuda INTERFACE)
target_include_directories(${PROJECT_NAME}_cuda INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>/")

# CPU Example Target
add_executable(example_cpu
  examples/example_cpu.cpp
  examples/util.cpp
  examples/util.h
)
target_link_libraries(example_cpu ${PROJECT_NAME}_cpu)
target_compile_options(example_cpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fPIC -O3 -march=native >)
target_compile_features(example_cpu PUBLIC cxx_std_11)

# GPU Example Target
add_executable(example_gpu
  examples/example_gpu.cu
  examples/util.cpp
  examples/util.h
)
target_link_libraries(example_gpu ${PROJECT_NAME}_cuda)
target_compile_options(example_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math >
  $<$<COMPILE_LANGUAGE:CXX>: -fPIC -O3 -march=native >
)
target_compile_features(example_gpu PUBLIC cxx_std_11)
set_target_properties( example_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# GPU 3D Example Target
add_executable(example_3d_gpu
  examples/example_3d_gpu.cu
  examples/util.cpp
  examples/util.h
)
# 链接项目自身的 cuda 接口库以及 NIFTI 库
target_compile_options(example_3d_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>: --use_fast_math >
  $<$<COMPILE_LANGUAGE:CXX>: -fPIC -O3 -march=native >
)
target_compile_features(example_3d_gpu PUBLIC cxx_std_11)
set_target_properties(example_3d_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(example_3d_gpu PRIVATE
  ${NIFTI_INCLUDE_DIR}
)

target_link_libraries(example_3d_gpu PRIVATE
  ${PROJECT_NAME}_cuda   # 你的 CUDA INTERFACE target
  ${NIFTIIO_LIB}
  ${ZNZ_LIB}
  ZLIB::ZLIB
)

install(TARGETS ${PROJECT_NAME}_cpu EXPORT ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}_cuda EXPORT ${PROJECT_NAME})

include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

install(EXPORT ${PROJECT_NAME}
  DESTINATION share/${PROJECT_NAME}/cmake
  FILE ${PROJECT_NAME}Config.cmake
)
