// Runs a kernel that vxc compiled for the Apple GPU. Its payload carries Metal
// Shading Language (`format=msl`), and npu_dispatch.mm sends such a payload
// here.
//
// src/dialect/MetalImage.cpp gives the binding convention. Each tensor becomes
// one buffer over the pages that the tensor touches, with no copy. Each of
// those pages holds part of the tensor, so each one is mapped. The kernel finds
// the tensor's first element from the words. Metal compiles the MSL with safe
// math, so a kernel gives the same bits as its host copy.
//
// Each launch waits until the GPU finishes, so a tensor that the kernel wrote
// is ready when the call returns.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../include/vx_hardware_runtime.h"
#include "vx_dispatch_plan.h"
#include "vx_kernel_launch.h"

#include <mach/vm_page_size.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

bool verbose() {
  static const bool on = [] {
    const char *v = getenv("VX_DISPATCH_VERBOSE");
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
  }();
  return on;
}

[[noreturn]] void die(const char *kernel, const char *format, ...) {
  fprintf(stderr, "[Vx Metal] FATAL: kernel %s: ", kernel);
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
  abort();
}

// Whether this thread sends kernels to the GPU. VX_METAL=0 turns the GPU off
// for every thread. vx_metal_set_enabled changes it for the calling thread.
thread_local int enabledHere = -1;

bool enabled() {
  static const bool byEnvironment = [] {
    const char *v = getenv("VX_METAL");
    return !(v && strcmp(v, "0") == 0);
  }();
  return enabledHere < 0 ? byEnvironment : enabledHere == 1;
}

struct Gpu {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
};

Gpu &gpu() {
  static Gpu g = [] {
    Gpu made;
    made.device = MTLCreateSystemDefaultDevice();
    made.queue = [made.device newCommandQueue];
    return made;
  }();
  return g;
}

// Compiled kernels, by image and name. The image is a constant of the program,
// so its address names it.
std::mutex cacheLock;
std::unordered_map<const char *, id<MTLLibrary>> libraries;
std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

id<MTLComputePipelineState> pipelineFor(const char *image, const char *kernel) {
  std::lock_guard<std::mutex> hold(cacheLock);
  std::string key =
      std::to_string(reinterpret_cast<uintptr_t>(image)) + "/" + kernel;
  auto found = pipelines.find(key);
  if (found != pipelines.end())
    return found->second;
  id<MTLDevice> device = gpu().device;
  if (!device)
    die(kernel, "this machine has no Metal GPU");
  NSError *error = nil;
  id<MTLLibrary> library = libraries[image];
  if (!library) {
    MTLCompileOptions *options = [MTLCompileOptions new];
    // A kernel must give the same bits on the CPU and the GPU.
    options.mathMode = MTLMathModeSafe;
    library = [device newLibraryWithSource:@(image)
                                   options:options
                                     error:&error];
    if (!library)
      die(kernel, "Metal could not compile the kernel:\n%s",
          error.localizedDescription.UTF8String);
    libraries[image] = library;
  }
  id<MTLFunction> function = [library newFunctionWithName:@(kernel)];
  if (!function)
    die(kernel, "the image has no kernel of this name");
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (!pipeline)
    die(kernel, "Metal could not make a pipeline for the kernel: %s",
        error.localizedDescription.UTF8String);
  pipelines[key] = pipeline;
  return pipeline;
}

// The parameters that the `// vx.kernel` line in `image` gives for `kernel`.
std::vector<std::string> parametersOf(const char *image, const char *kernel) {
  std::string key = std::string("// vx.kernel ") + kernel;
  for (const char *at = strstr(image, key.c_str()); at;
       at = strstr(at + 1, key.c_str())) {
    const char *end = at + key.size();
    if (*end != ' ' && *end != '\n')
      continue;
    const char *eol = strchr(end, '\n');
    std::string rest(end, eol ? eol : end + strlen(end));
    std::vector<std::string> out;
    size_t from = 0;
    while (from < rest.size()) {
      size_t space = rest.find(' ', from);
      if (space == std::string::npos)
        space = rest.size();
      if (space > from)
        out.push_back(rest.substr(from, space - from));
      from = space + 1;
    }
    return out;
  }
  die(kernel, "the image has no `// vx.kernel` line for this kernel");
}

// A tensor's buffer. `copied` is set when Metal could not use the memory as it
// is, and the kernel ran on a copy.
struct Bound {
  id<MTLBuffer> buffer;
  char *copied = nullptr;
  uintptr_t first = 0, last = 0, start = 0;
  bool written = false;
};

void pushWord(std::vector<uint32_t> &words, int64_t value, const char *kernel,
              const char *what) {
  if (value < INT32_MIN || value > INT32_MAX)
    die(kernel, "%s is %lld, and a Metal kernel takes 32-bit indices", what,
        (long long)value);
  words.push_back(static_cast<uint32_t>(static_cast<int32_t>(value)));
}

} // namespace

extern "C" void vx_metal_set_enabled(int on) { enabledHere = on ? 1 : 0; }

// Returns 1 when the kernel ran on the GPU, and 0 when the GPU is turned off.
// The caller then runs the host copy of the kernel.
extern "C" int vx_metal_dispatch(const void *payload, size_t payload_size,
                                 void **device_args, const int32_t *arg_tags,
                                 int64_t num_args) {
  if (!enabled())
    return 0;
  const char *kernel = static_cast<const char *>(payload);
  const char *image = vx_payload_field(payload, payload_size, "image=");
  if (!image)
    die(kernel, "the payload says `format=msl` and has no image");

  @autoreleasepool {
    std::vector<std::string> params = parametersOf(image, kernel);
    if ((int64_t)params.size() != num_args)
      die(kernel, "the kernel has %zu parameters, and the launch passes %lld",
          params.size(), (long long)num_args);

    id<MTLDevice> device = gpu().device;
    id<MTLComputePipelineState> pipeline = pipelineFor(image, kernel);
    const uintptr_t page = vm_page_size;
    std::vector<Bound> bound;
    std::vector<uint32_t> words;

    for (int64_t i = 0; i < num_args; ++i) {
      const std::string &p = params[i];
      const int32_t tag = arg_tags[i];
      const int32_t kind = VX_ABI_KIND(tag);
      if (p[0] == 'm') {
        int rank = atoi(p.c_str() + 1);
        if (kind != VX_ABI_KIND_MEMREF || VX_ABI_RANK(tag) != rank)
          die(kernel,
              "parameter %lld is a tensor of rank %d, and the launch "
              "passes something else",
              (long long)i, rank);
        const void *desc = vx_operand_desc(device_args, arg_tags, (int)i);
        const size_t bytes = vx_dtype_bytes(VX_ABI_ELEM(tag));
        if (!desc || bytes == 0)
          die(kernel, "parameter %lld has no tensor", (long long)i);
        char *aligned = static_cast<char *>(vx_memref_aligned(desc));
        const int64_t offset = vx_memref_offset(desc);
        const int64_t *sizes = vx_memref_sizes(desc);
        const int64_t *strides = vx_memref_strides(desc, rank);

        // The elements the tensor can reach, from `lo` to before `hi`. The
        // kernel reads a stride or an offset that its type fixes from the
        // type, so the buffer covers the contiguous layout too.
        int64_t lo = offset, hi = offset + 1, contiguous = 1;
        for (int d = 0; d < rank; ++d) {
          contiguous *= sizes[d];
          if (sizes[d] == 0) {
            lo = hi = offset;
            break;
          }
          int64_t reach = (sizes[d] - 1) * strides[d];
          (reach < 0 ? lo : hi) += reach;
        }
        lo = std::min<int64_t>(lo, 0);
        hi = std::max<int64_t>(hi, contiguous);
        Bound b;
        b.first = reinterpret_cast<uintptr_t>(aligned + lo * (int64_t)bytes);
        b.last = reinterpret_cast<uintptr_t>(aligned + hi * (int64_t)bytes);
        b.start = b.first & ~(page - 1);
        uintptr_t end = (b.last + page - 1) & ~(page - 1);
        if (end == b.start)
          end = b.start + page;
        b.written = p.back() != 'r';
        const int64_t base =
            (int64_t)(reinterpret_cast<uintptr_t>(aligned) - b.start) /
            (int64_t)bytes;
        if ((reinterpret_cast<uintptr_t>(aligned) - b.start) % bytes != 0)
          die(kernel, "tensor %lld is not aligned to its element size",
              (long long)i);
        if ((end - b.start) / bytes > (uintptr_t)INT32_MAX)
          die(kernel,
              "tensor %lld spans %llu elements, and a Metal kernel "
              "takes 32-bit indices",
              (long long)i, (unsigned long long)((end - b.start) / bytes));
        b.buffer =
            [device newBufferWithBytesNoCopy:reinterpret_cast<void *>(b.start)
                                      length:end - b.start
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
        if (!b.buffer) {
          // Metal refused this memory. The kernel runs on a copy, and a tensor
          // that it writes is copied back.
          b.buffer =
              [device newBufferWithBytes:reinterpret_cast<void *>(b.start)
                                  length:end - b.start
                                 options:MTLResourceStorageModeShared];
          b.copied = static_cast<char *>(b.buffer.contents);
          if (verbose())
            fprintf(stderr, "[Vx Metal] %s: tensor %lld is copied\n", kernel,
                    (long long)i);
        }
        if (!b.buffer)
          die(kernel, "Metal could not make a buffer for tensor %lld",
              (long long)i);
        pushWord(words, base, kernel, "a tensor's start in its buffer");
        pushWord(words, offset, kernel, "a tensor's offset");
        for (int d = 0; d < rank; ++d)
          pushWord(words, sizes[d], kernel, "a tensor's size");
        for (int d = 0; d < rank; ++d)
          pushWord(words, strides[d], kernel, "a tensor's stride");
        bound.push_back(b);
        continue;
      }

      const void *value = device_args[i];
      uint32_t word = 0;
      if (p == "i1" && kind == VX_ABI_KIND_I1) {
        words.push_back(*static_cast<const uint8_t *>(value) & 1);
      } else if (p == "i8" && kind == VX_ABI_KIND_I8) {
        words.push_back(*static_cast<const uint8_t *>(value));
      } else if (p == "i16" && kind == VX_ABI_KIND_I16) {
        words.push_back(*static_cast<const uint16_t *>(value));
      } else if ((p == "i32" && kind == VX_ABI_KIND_I32) ||
                 (p == "f32" && kind == VX_ABI_KIND_F32)) {
        memcpy(&word, value, 4);
        words.push_back(word);
      } else if (p == "i64" && kind == VX_ABI_KIND_I64) {
        uint64_t v;
        memcpy(&v, value, 8);
        words.push_back(static_cast<uint32_t>(v));
        words.push_back(static_cast<uint32_t>(v >> 32));
      } else if (p == "index" && kind == VX_ABI_KIND_I64) {
        int64_t v;
        memcpy(&v, value, 8);
        pushWord(words, v, kernel, "an index parameter");
      } else {
        die(kernel,
            "parameter %lld is %s, and the launch passes a value of "
            "kind %d",
            (long long)i, p.c_str(), kind);
      }
    }

    // With the copies, two written tensors that share memory would undo each
    // other's writes when they are copied back.
    for (size_t a = 0; a < bound.size(); ++a)
      for (size_t b = 0; b < bound.size(); ++b)
        if (a != b && bound[a].copied && bound[a].written && bound[b].written &&
            bound[a].first < bound[b].last && bound[b].first < bound[a].last)
          die(kernel, "two tensors share memory that Metal could not use as "
                      "it is");

    unsigned grid[3] = {1, 1, 1};
    unsigned block = 1;
    char why[256] = {0};
    if (vx_launch_shape(payload, payload_size, device_args, arg_tags, num_args,
                        grid, &block, why, sizeof why) < 0)
      die(kernel, "%s", why);
    if (block > pipeline.maxTotalThreadsPerThreadgroup)
      die(kernel,
          "a block of %u threads, and this kernel can run at most %lu "
          "threads in a block on this GPU",
          block, (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);

    if (words.empty())
      words.push_back(0);
    id<MTLCommandBuffer> commands = [gpu().queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    for (size_t k = 0; k < bound.size(); ++k)
      [encoder setBuffer:bound[k].buffer offset:0 atIndex:k];
    const size_t wordBytes = words.size() * sizeof(uint32_t);
    // `setBytes` takes at most 4 KB.
    if (wordBytes <= 4096)
      [encoder setBytes:words.data() length:wordBytes atIndex:bound.size()];
    else
      [encoder
          setBuffer:[device newBufferWithBytes:words.data()
                                        length:wordBytes
                                       options:MTLResourceStorageModeShared]
             offset:0
            atIndex:bound.size()];
    [encoder dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2])
            threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    if (commands.status != MTLCommandBufferStatusCompleted)
      die(kernel, "the GPU did not finish the kernel: %s",
          commands.error.localizedDescription.UTF8String);

    for (const Bound &b : bound)
      if (b.copied && b.written)
        memcpy(reinterpret_cast<void *>(b.first),
               b.copied + (b.first - b.start), b.last - b.first);

    if (verbose())
      fprintf(
          stderr,
          "[Vx Metal] %s: grid %u x %u x %u, block %u, %.3f ms on the GPU\n",
          kernel, grid[0], grid[1], grid[2], block,
          (commands.GPUEndTime - commands.GPUStartTime) * 1e3);
  }
  return 1;
}
