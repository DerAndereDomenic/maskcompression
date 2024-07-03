#include <maskcompression/compress.h>

namespace maskcompression
{
std::vector<torch::Tensor> compress(const torch::Tensor& masks)
{
    uint32_t batch_size = masks.size(0);
    std::vector<torch::Tensor> result(batch_size);

    for(int i = 0; i < batch_size; ++i)
    {
        int32_t leading_one = (int32_t)(masks.index({0, 0, 0}).item<float>() != 0.0f);
        auto compressed     = std::get<2>(
            torch::unique_consecutive(masks.index({i, torch::indexing::Slice(), torch::indexing::Slice()}).flatten(),
                                      /*return_inverse = */ false,
                                      /*return_counts = */ true));

        result[i] = torch::cat(
            {torch::tensor({leading_one}, torch::TensorOptions {}.dtype(torch::kInt32).device(compressed.device())),
             torch::cumsum(compressed, 0).to(torch::kInt32)});
    }

    return result;
}
}    // namespace maskcompression