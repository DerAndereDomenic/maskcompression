#include <torch/python.h>

#include "decompress.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("decompress", &maskcompression::decompress);
}