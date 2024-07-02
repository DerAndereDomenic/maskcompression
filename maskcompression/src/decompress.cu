#include "decompress.h"

#include <ATen/cuda/ApplyGridUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

namespace maskcompression
{

namespace detail
{

template<typename T>
inline __device__ uint32_t
binary_search(const torch::PackedTensorAccessor32<T, 1, torch::RestrictPtrTraits> sorted_array, T value)
{
    // Find first element in sorted_array that is larger than value.
    uint32_t left  = 0;
    uint32_t right = sorted_array.size(0) - 1;
    while(left < right)
    {
        uint32_t mid = (left + right) / 2;
        if(sorted_array[mid] < value)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

__global__ void decompressImage(const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> cumsum,
                                const uint32_t width,
                                const uint32_t height,
                                torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> output)
{
    auto id = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    auto num_threads = static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
    for(int32_t tid = id; tid < width * height; tid += num_threads)
    {
        int pixel_x = tid % width;
        int pixel_y = tid / width;

        uint32_t bin_index = binary_search(cumsum, tid + 1);

        output[pixel_y][pixel_x] = bin_index % 2 == 0 ? 0.0f : 1.0f;
    }
}
}    // namespace detail

torch::Tensor decompress(const torch::Tensor& compressed, at::IntArrayRef& resolution)
{
    torch::Tensor output =
        torch::zeros(resolution, torch::TensorOptions {}.dtype(torch::kFloat32).device(torch::kCUDA));
    auto cumsum = torch::cumsum(compressed, 0).to(torch::kInt32);    // TODO: copy

    auto device = output.device();

    at::cuda::CUDAGuard device_guard {device};
    const auto stream = at::cuda::getCurrentCUDAStream();

    const int threads_per_block = 128;
    dim3 grid;
    at::cuda::getApplyGrid(resolution[0] * resolution[1], grid, device.index(), threads_per_block);
    dim3 threads = at::cuda::getApplyBlock(threads_per_block);

    detail::decompressImage<<<grid, threads, 0, stream>>>(
        cumsum.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        resolution[1],
        resolution[0],
        output.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaStreamSynchronize(stream));

    return output;
}
}    // namespace maskcompression