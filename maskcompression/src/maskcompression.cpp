#include <torch/python.h>

torch::Tensor generate_tensor();  // Implemented somewhere in <my_cpp_cuda_ext>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("helloworld", []{printf("Hello World!");});
}