#pragma once

#include <torch/types.h>

namespace maskcompression
{
torch::Tensor decompress(const torch::Tensor& compressed, at::IntArrayRef& resolution);
}