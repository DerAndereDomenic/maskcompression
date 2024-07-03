#pragma once

#include <torch/types.h>

namespace maskcompression
{
torch::Tensor decompress(const std::vector<torch::Tensor>& compressed, at::IntArrayRef& resolution);
}