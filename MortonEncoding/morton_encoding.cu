/*
Implementation based on
1. https://github.com/m-schuetz/compute_rasterizer/blob/f2cbb658e6bf58407c385c75d21f3f615f11d5c9/tools/sort_points/Sort_Frugal/src/main.cpp#L79
2. https://gitlab.inria.fr/sibr/sibr_core/-/blob/gaussian_code_release_linux/src/projects/gaussianviewer/renderer/GaussianView.cpp?ref_type=heads#L90
*/

#include "morton_encoding.h"

__device__ __forceinline__ uint64_t splitBy3(uint32_t a) {
	uint64_t x = a & 0x1fffff;
	x = (x | x << 32) & 0x1f00000000ffff;
	x = (x | x << 16) & 0x1f0000ff0000ff;
	x = (x | x << 8) & 0x100f00f00f00f00f;
	x = (x | x << 4) & 0x10c30c30c30c30c3;
	x = (x | x << 2) & 0x1249249249249249;
	return x;
}

__global__ void morton_encode_cu(
        float3 const *const __restrict__ positions,
        float3 const *const __restrict__ minimum_coordinates,
        float const *const __restrict__ cube_size,
        int64_t *const __restrict__ morton_encoding,
        const int n_positions) {
    const int position_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (position_idx >= n_positions) return;
    const float3 position = positions[position_idx];
    const float3 minimum_coordinate = minimum_coordinates[0];
    // could use float instead of double if performance is critical
    const double size = double(cube_size[0]);
    const double normalized_x = double(position.x - minimum_coordinate.x) / size;
    const double normalized_y = double(position.y - minimum_coordinate.y) / size;
    const double normalized_z = double(position.z - minimum_coordinate.z) / size;
    constexpr double factor = 2097151.0; // 2^21 - 1
    const uint32_t x = static_cast<uint32_t>(normalized_x * factor);
    const uint32_t y = static_cast<uint32_t>(normalized_y * factor);
    const uint32_t z = static_cast<uint32_t>(normalized_z * factor);
    const uint64_t morton_code = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;
    constexpr int64_t int64_min = -9223372036854775808;
    const int64_t morton_code_torch = static_cast<int64_t>(morton_code) + int64_min;
    morton_encoding[position_idx] = morton_code_torch;
}


at::Tensor morton_encode(
        const at::Tensor& positions,
        const at::Tensor& minimum_coordinates,
        const at::Tensor& cube_size) {
    const int n_positions = positions.size(0);
    at::Tensor morton_encoding = torch::empty({n_positions}, positions.options().dtype(torch::kLong));
    constexpr int block_size = 256;
    const int grid_size = (n_positions + block_size - 1) / block_size;
    morton_encode_cu<<<grid_size, block_size>>>(
        reinterpret_cast<const float3*>(positions.contiguous().data_ptr<float>()),
        reinterpret_cast<const float3*>(minimum_coordinates.contiguous().data_ptr<float>()),
        reinterpret_cast<const float*>(cube_size.contiguous().data_ptr<float>()),
        morton_encoding.data_ptr<int64_t>(),
        n_positions);
    return morton_encoding;
}
