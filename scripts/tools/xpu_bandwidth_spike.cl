// The triad kernel the bandwidth spike launches: x = a*y + b*z, one float4 per
// work-item, grid-strided so any grid size is correct.
//
// Compiled to SPIR-V and launched through Level Zero with the call sequence a
// device image uses, so the measurement takes the path a Vx-compiled kernel
// takes. It works on float4 because a float-per-work-item triad on this card
// measures 337 GB/s against the copy's 448-449 GB/s, while this one measures
// 427-428 GB/s.
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
