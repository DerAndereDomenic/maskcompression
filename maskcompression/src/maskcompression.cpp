#include <torch/python.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("helloworld", [] { printf("Hello World!"); });
}