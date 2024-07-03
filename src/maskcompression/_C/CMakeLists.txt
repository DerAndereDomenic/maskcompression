cmake_minimum_required(VERSION 3.27)

project(maskcompression LANGUAGES C CXX CUDA VERSION 1.0)

find_package(charonload)

if(charonload_FOUND)
    charonload_add_torch_library(${TORCH_EXTENSION_NAME} MODULE)

    target_sources(${TORCH_EXTENSION_NAME} PRIVATE python/maskcompression.cpp
                                                   src/decompress.cu
                                                   src/compress.cpp)

    target_include_directories(${TORCH_EXTENSION_NAME} PUBLIC include)
endif()