#include "decompress.h"

namespace maskcompression
{
torch::Tensor decompress(const torch::Tensor& compressed, at::IntArrayRef& resolution)
{
    torch::Tensor output =
        torch::zeros(resolution, torch::TensorOptions {}.dtype(torch::kFloat32).device(torch::kCUDA));
    auto cumsum = torch::cumsum(compressed, 0);


    return output;
}
}    // namespace maskcompression