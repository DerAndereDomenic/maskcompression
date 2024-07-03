#include <torch/python.h>

#include <maskcompression/decompress.h>
#include <maskcompression/compress.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("decompress", &maskcompression::decompress);
    m.def("compress", &maskcompression::compress);
}