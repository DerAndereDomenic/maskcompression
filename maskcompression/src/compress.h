#pragma once

#include <torch/types.h>

namespace maskcompression
{
std::vector<torch::Tensor> compress(const torch::Tensor& masks);
}