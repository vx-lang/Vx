// The triad kernel the bandwidth spike launches: x = a*y + b*z, one float4 per
// work-item, grid-strided so any grid size is correct.
//
// Written by hand and compiled to SPIR-V with the same `clang -target
// spirv64-unknown-unknown` the Level Zero spike uses, so the measurement goes
// through the same load-and-launch path a Vx kernel does. It works on float4
// because a scalar triad on this card measures the compiler's scalar code
// instead of the memory: the same kernel with one float per work-item gets
// 337 GB/s, well under the copy's 449 GB/s, while this one gets 423 GB/s --
// close to the copy, which is what says it is bandwidth-bound.
//
// `n4` counts float4 vectors, so the buffer must hold a multiple of four
// floats. The caller passes (elements / 4).

__kernel void triad(__global float4 *x, __global const float4 *y,
                    __global const float4 *z, const float a, const float b,
                    const unsigned long n4) {
  unsigned long gid = get_global_id(0);
  unsigned long stride = get_global_size(0);
  for (unsigned long i = gid; i < n4; i += stride)
    x[i] = a * y[i] + b * z[i];
}
