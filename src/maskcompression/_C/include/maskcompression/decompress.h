#pragma once

#include <torch/types.h>

namespace maskcompression
{
/**
 * @brief Decompress a list of compressed masks.
 * Only works on cuda devices. The compressed tensors are expected to be cuda tensors.
 *
 * @param compressed List of linear tensors representing the masks
 * @param resolution The resolution of the mask. Has to be constant for all input masks
 *
 * @return (B, H, W) tensor with decoded masks
 */
torch::Tensor decompress(const std::vector<torch::Tensor>& compressed, at::IntArrayRef& resolution);
}    // namespace maskcompression