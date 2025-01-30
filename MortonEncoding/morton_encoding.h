#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <torch/extension.h>

at::Tensor morton_encode(
        const at::Tensor& positions,
        const at::Tensor& minimum_coordinates,
        const at::Tensor& cube_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("morton_encode_cuda", &morton_encode);
}
