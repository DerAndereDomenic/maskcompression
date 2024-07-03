#include "compress.h"

namespace maskcompression
{
std::vector<torch::Tensor> compress(const torch::Tensor& masks)
{
    uint32_t batch_size = masks.size(0);
    std::vector<torch::Tensor> result(batch_size);

    for(int i = 0; i < batch_size; ++i)
    {
        auto compressed =
            std::get<2>(torch::unique_consecutive(
                            masks.index({i, torch::indexing::Slice(), torch::indexing::Slice()}).flatten(),
                            /*return_inverse = */ false,
                            /*return_counts = */ true))
                .to(torch::kInt32);
        result[i] = compressed;
    }

    return result;
}
}    // namespace maskcompression