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
// Metal maps every page of a buffer when it makes the buffer. So GPU memory
// from vx_metal_alloc and vx_metal_map_copy gets one buffer over all of its
// pages, once, and a residency set keeps it resident. Other memory gets a
// buffer at each batch, because the program can free it without telling this
// file.
//
// The launches of each host thread make one batch until a wait. The batch
// commits its command buffer after every few launches, so the GPU runs them
// while the CPU makes the next. A wait commits the rest and waits for the GPU.
// The GPU runs the launches of a command buffer at the same time, with a
// barrier before each launch that uses memory that an earlier one writes, or
// writes memory that an earlier one reads.
// The Mac dispatcher waits before anything that can touch the memory of a
// kernel, but only for the launches that use that memory. A `pub fn` waits
// before it returns to C. A failed assert in a kernel stops the program at the
// wait that finishes its launch.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../include/vx_hardware_runtime.h"
#include "vx_dispatch_plan.h"
#include "vx_kernel_launch.h"

#include <fcntl.h>
#include <libproc.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_page_size.h>
#include <mach/vm_prot.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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

// VX_METAL_SYNC=1 waits after each launch, so a GPU fault names its kernel.
bool syncEachLaunch() {
  static const bool on = [] {
    const char *v = getenv("VX_METAL_SYNC");
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
  }();
  return on;
}

// VX_METAL_CONCURRENT=0 runs each launch after the launch before it ends, as
// if every launch had a barrier.
bool concurrent() {
  static const bool on = [] {
    const char *v = getenv("VX_METAL_CONCURRENT");
    return !(v && strcmp(v, "0") == 0);
  }();
  return on;
}

// VX_METAL_SENTINEL=1 ends a wait of the GPU in a spin on the sentinel that a
// commit's kernel writes. The default waits in `waitUntilCompleted`, as before
// the sentinel: on the M4 of fleet/m4-uma.vx the probe of a launch and a read
// takes longer with the spin than without it.
bool sentinelOn() {
  static const bool on = [] {
    const char *v = getenv("VX_METAL_SENTINEL");
    return v && v[0] != '\0' && strcmp(v, "0") != 0;
  }();
  return on;
}

id<MTLComputeCommandEncoder> newEncoder(id<MTLCommandBuffer> commands) {
  return concurrent() ? [commands computeCommandEncoderWithDispatchType:
                                      MTLDispatchTypeConcurrent]
                      : [commands computeCommandEncoder];
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

bool enabledByEnvironment() {
  static const bool on = [] {
    const char *v = getenv("VX_METAL");
    return !(v && strcmp(v, "0") == 0);
  }();
  return on;
}

bool enabled() {
  return enabledHere < 0 ? enabledByEnvironment() : enabledHere == 1;
}

struct Gpu {
  id<MTLDevice> device;
  id<MTLCommandQueue> queue;
  id resident;
};

Gpu &gpu() {
  static Gpu g = [] {
    Gpu made;
    made.device = MTLCreateSystemDefaultDevice();
    made.queue = [made.device newCommandQueue];
    if (@available(macOS 15.0, *)) {
      MTLResidencySetDescriptor *d = [MTLResidencySetDescriptor new];
      id<MTLResidencySet> set = [made.device newResidencySetWithDescriptor:d
                                                                     error:nil];
      if (set) {
        [made.queue addResidencySet:set];
        made.resident = set;
      }
    }
    return made;
  }();
  return g;
}

void setResident(id<MTLBuffer> buffer, bool resident) {
  if (@available(macOS 15.0, *)) {
    id<MTLResidencySet> set = gpu().resident;
    if (!set)
      return;
    if (resident)
      [set addAllocation:buffer];
    else
      [set removeAllocation:buffer];
    [set commit];
    if (resident)
      [set requestResidency];
  }
}

struct Access {
  uintptr_t first = 0, last = 0;
  bool written = false;

  bool mustFollow(const Access &other) const {
    return first < other.last && other.first < last &&
           (written || other.written);
  }
};

struct Region {
  uintptr_t start = 0, end = 0;
  id<MTLBuffer> buffer;
};

const Region *holding(const std::vector<Region> &regions, uintptr_t start,
                      uintptr_t end) {
  for (const Region &r : regions)
    if (r.start <= start && end <= r.end)
      return &r;
  return nullptr;
}

std::mutex keptLock;

std::map<uintptr_t, Region> &kept() {
  static auto *all = new std::map<uintptr_t, Region>;
  return *all;
}

Region keptHolding(uintptr_t start, uintptr_t end) {
  std::lock_guard<std::mutex> hold(keptLock);
  auto at = kept().upper_bound(start);
  if (at == kept().begin())
    return {};
  --at;
  return end <= at->second.end ? at->second : Region{};
}

struct Check {
  std::string kernel;
  const char *image;
  id<MTLBuffer> words;
  size_t at;
};

constexpr size_t assertWords = 1024;

struct Batch {
  std::thread::id owner;
  id<MTLCommandBuffer> commands;
  std::vector<id<MTLCommandBuffer>> sent;
  id<MTLComputeCommandEncoder> encoder;
  size_t unsent = 0;
  std::vector<std::string> kernels;
  std::vector<Region> regions;
  std::vector<Access> accesses;
  std::vector<Access> sinceBarrier;
  std::vector<Check> checks;
  id<MTLBuffer> assertBuffer;
  size_t assertsGiven = 0;
  id<MTLBuffer> done;
  uint32_t sentinel = 0;
  bool hasSentinel = false;
  bool proved = false;
  bool reported = false;
  size_t cmdBufferKernelStart = 0;
  size_t cmdBufferCheckStart = 0;

  bool open() const { return encoder != nil; }

  const Region *holding(uintptr_t start, uintptr_t end) const {
    return ::holding(regions, start, end);
  }

  bool overlaps(uintptr_t start, uintptr_t end) const {
    for (const Region &r : regions)
      if (r.start < end && start < r.end)
        return true;
    return false;
  }

  bool mustPrecede(const Access &a) const {
    for (const Access &b : accesses)
      if (a.mustFollow(b))
        return true;
    return false;
  }

  void add(const Access &a) {
    for (Access &b : accesses)
      if (b.first == a.first && b.last == a.last) {
        b.written |= a.written;
        return;
      }
    accesses.push_back(a);
  }
};

std::mutex batchLock;

std::vector<std::shared_ptr<Batch>> &batches() {
  static auto *all = new std::vector<std::shared_ptr<Batch>>;
  return *all;
}

std::shared_ptr<Batch> openBatchHere() {
  for (const std::shared_ptr<Batch> &b : batches())
    if (b->open() && b->owner == std::this_thread::get_id())
      return b;
  return nullptr;
}

const char *assertLine(const char *image, const char *kernel, uint32_t number) {
  std::string key = std::string("// vx.assert ") + kernel + " " +
                    std::to_string(number) + " ";
  for (const char *at = strstr(image, key.c_str()); at;
       at = strstr(at + 1, key.c_str()))
    if (at == image || at[-1] == '\n')
      return at + key.size();
  return nullptr;
}

std::string assertMessage(const char *image, const char *kernel,
                          uint32_t number) {
  const char *at = assertLine(image, kernel, number);
  if (!at)
    return "assert " + std::to_string(number) +
           " failed, and the image has no message for it";
  std::string out;
  for (; *at && *at != '\n'; ++at) {
    if (at[0] == '%' && isxdigit(at[1]) && isxdigit(at[2])) {
      out += static_cast<char>(std::stoi(std::string(at + 1, 2), nullptr, 16));
      at += 2;
    } else {
      out += *at;
    }
  }
  return out;
}

void report(const Batch &b) {
  for (const Check &c : b.checks) {
    const uint32_t number =
        static_cast<const volatile uint32_t *>(c.words.contents)[c.at];
    if (number != 0)
      die(c.kernel.c_str(), "%s",
          assertMessage(c.image, c.kernel.c_str(), number).c_str());
  }
  const size_t count = b.kernels.size();
  std::vector<id<MTLCommandBuffer>> all = b.sent;
  all.push_back(b.commands);
  double seconds = 0;
  bool timeKnown = true;
  for (id<MTLCommandBuffer> commands : all) {
    const bool fault = commands.status == MTLCommandBufferStatusError;
    if (fault ||
        (!b.proved && commands.status != MTLCommandBufferStatusCompleted)) {
      std::string names;
      for (const std::string &k : b.kernels)
        names += (names.empty() ? "" : ", ") + k;
      fprintf(stderr, "[Vx Metal] FATAL: the GPU did not finish %s: %s\n",
              names.c_str(), commands.error.localizedDescription.UTF8String);
      if (count > 1)
        fprintf(stderr, "           Set VX_METAL_SYNC=1 to name the kernel "
                        "that failed.\n");
      abort();
    }
    if (commands.status != MTLCommandBufferStatusCompleted) {
      timeKnown = false;
      continue;
    }
    seconds += commands.GPUEndTime - commands.GPUStartTime;
  }
  if (verbose()) {
    if (timeKnown)
      fprintf(stderr, "[Vx Metal] waited for %zu kernel%s, %.3f ms on the GPU\n",
              count, count == 1 ? "" : "s", seconds * 1e3);
    else
      fprintf(stderr,
              "[Vx Metal] waited for %zu kernel%s, the time on the GPU is not "
              "known yet\n",
              count, count == 1 ? "" : "s");
  }
}

id<MTLComputePipelineState> sentinelPipeline();

static void dispatchSentinel(id<MTLComputeCommandEncoder> encoder, Batch &b) {
  b.hasSentinel = false;

  // If the batch has multiple command buffers, we cannot use the sentinel.
  if (!sentinelOn() || !b.sent.empty() || b.done == nil || b.regions.size() > 29)
    return;

  id<MTLComputePipelineState> pipeline = sentinelPipeline();
  if (!pipeline)
    return;

  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:b.done offset:0 atIndex:0];

  static thread_local uint32_t nextSentinel = 1;
  const uint32_t value = nextSentinel++;
  b.sentinel = value;
  [encoder setBytes:&value length:sizeof value atIndex:1];

  size_t given = 0;
  for (const Region &r : b.regions) {
    // A tensor of size 0 has no valid buffer to read. The sentinel reads r[0],
    // so we must not bind a 0-byte buffer.
    if (r.end > r.start) {
      [encoder setBuffer:r.buffer offset:0 atIndex:2 + given++];
    }
  }

  // Bind b.done to all remaining slots to force the sentinel to wait for
  // all preceding launches without reading 0-size buffers.
  for (size_t i = given; i < 29; ++i)
    [encoder setBuffer:b.done offset:0 atIndex:2 + i];

  [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  b.hasSentinel = true;
}

static bool spin(Batch &b) {
  if (!b.sent.empty() || b.done == nil || !b.hasSentinel)
    return false;

  const uint32_t value = b.sentinel;
  if (value == 0)
    return false;

  const uint32_t *word = static_cast<const uint32_t *>(b.done.contents);
  // Apple CPU cores have no SMT, so yield does not give the core to another
  // thread. This keeps the core busy for up to 2 ms before it blocks.
  const auto giveUp =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
  while (__atomic_load_n(word, __ATOMIC_ACQUIRE) < value) {
    if (std::chrono::steady_clock::now() > giveUp)
      return false;
#if defined(__aarch64__)
    __builtin_arm_yield();
#elif defined(__x86_64__)
    __builtin_ia32_pause();
#endif
  }
  return true;
}

static void reportLateFault(id<MTLCommandBuffer> commands,
                            std::shared_ptr<Batch> b) {
  std::string names;
  for (size_t i = b->cmdBufferKernelStart; i < b->kernels.size(); ++i)
    names += (names.empty() ? "" : ", ") + b->kernels[i];

  std::vector<Check> checks(b->checks.begin() + b->cmdBufferCheckStart,
                            b->checks.end());

  [commands addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
    if (buffer.status == MTLCommandBufferStatusError) {
      if (!b->reported) {
        report(*b); // Aborts if any buffer status is Error
        b->reported = true;
      }

      if (b->proved) {
        for (const Check &c : checks) {
          const uint32_t number =
              static_cast<const volatile uint32_t *>(c.words.contents)[c.at];
          if (number != 0)
            die(c.kernel.c_str(), "%s",
                assertMessage(c.image, c.kernel.c_str(), number).c_str());
        }
        fprintf(stderr, "[Vx Metal] FATAL: the GPU did not finish %s: %s\n",
                names.c_str(), buffer.error.localizedDescription.UTF8String);
        if (checks.size() > 1 || names.find(',') != std::string::npos) {
          fprintf(stderr, "           Set VX_METAL_SYNC=1 to name the kernel "
                          "that failed.\n");
        }
        abort();
      }
    }

    {
      std::lock_guard<std::mutex> hold(batchLock);
      if (b->proved && b->commands == buffer) {
        auto it = std::find(batches().begin(), batches().end(), b);
        if (it != batches().end()) {
          batches().erase(it);
        }
      }
    }
  }];
}

void finish(const std::vector<std::shared_ptr<Batch>> &chosen,
            std::unique_lock<std::mutex> &hold, bool block = false) {
  if (chosen.empty())
    return;
  for (const std::shared_ptr<Batch> &b : chosen) {
    if (b->open()) {
      if (b->unsent > 0)
        dispatchSentinel(b->encoder, *b);
      [b->encoder endEncoding];
      b->encoder = nil;
      reportLateFault(b->commands, b);
      [b->commands commit];
    }
  }
  hold.unlock();
  std::vector<bool> proved(chosen.size(), false);
  for (size_t i = 0; i < chosen.size(); ++i) {
    if (block) {
      for (id<MTLCommandBuffer> commands : chosen[i]->sent)
        [commands waitUntilCompleted];
      [chosen[i]->commands waitUntilCompleted];
      continue;
    }
    proved[i] = spin(*chosen[i]);
    if (proved[i]) {
      for (id<MTLCommandBuffer> commands : chosen[i]->sent)
        [commands waitUntilCompleted];
      continue;
    }
    for (id<MTLCommandBuffer> commands : chosen[i]->sent)
      [commands waitUntilCompleted];
    [chosen[i]->commands waitUntilCompleted];
  }
  hold.lock();
  std::vector<std::shared_ptr<Batch>> &all = batches();
  for (size_t i = 0; i < chosen.size(); ++i) {
    const std::shared_ptr<Batch> &b = chosen[i];
    auto at = std::find(all.begin(), all.end(), b);
    if (at == all.end())
      continue;
    b->proved = proved[i];
    if (proved[i]) {
      if (verbose())
        fprintf(stderr,
                "[Vx Metal] the sentinel ended the wait for %zu kernel%s\n",
                b->kernels.size(), b->kernels.size() == 1 ? "" : "s");
      report(*b);
      b->reported = true;
      continue;
    }
    all.erase(at);
    report(*b);
  }
}

template <typename Pick> void finishEach(Pick pick, bool block = false) {
  std::unique_lock<std::mutex> hold(batchLock);
  std::vector<std::shared_ptr<Batch>> chosen;
  for (const std::shared_ptr<Batch> &b : batches())
    if (pick(*b))
      chosen.push_back(b);
  finish(chosen, hold, block);
}

std::mutex cacheLock;
std::unordered_map<const char *, id<MTLLibrary>> libraries;
std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;

id<MTLComputePipelineState> sentinelPipeline() {
  std::lock_guard<std::mutex> hold(cacheLock);
  static id<MTLComputePipelineState> made;
  if (made)
    return made;

  // Metal does not expose a release fence in MSL. Only a measurement supports
  // this order: the CPU saw the order on one M4 and one driver.
  static const char *source =
      "kernel void vx_sentinel(device uint *done [[buffer(0)]],\n"
      "                        constant uint &value [[buffer(1)]],\n"
      "                        device uchar *r0 [[buffer(2)]],\n"
      "                        device uchar *r1 [[buffer(3)]],\n"
      "                        device uchar *r2 [[buffer(4)]],\n"
      "                        device uchar *r3 [[buffer(5)]],\n"
      "                        device uchar *r4 [[buffer(6)]],\n"
      "                        device uchar *r5 [[buffer(7)]],\n"
      "                        device uchar *r6 [[buffer(8)]],\n"
      "                        device uchar *r7 [[buffer(9)]],\n"
      "                        device uchar *r8 [[buffer(10)]],\n"
      "                        device uchar *r9 [[buffer(11)]],\n"
      "                        device uchar *r10 [[buffer(12)]],\n"
      "                        device uchar *r11 [[buffer(13)]],\n"
      "                        device uchar *r12 [[buffer(14)]],\n"
      "                        device uchar *r13 [[buffer(15)]],\n"
      "                        device uchar *r14 [[buffer(16)]],\n"
      "                        device uchar *r15 [[buffer(17)]],\n"
      "                        device uchar *r16 [[buffer(18)]],\n"
      "                        device uchar *r17 [[buffer(19)]],\n"
      "                        device uchar *r18 [[buffer(20)]],\n"
      "                        device uchar *r19 [[buffer(21)]],\n"
      "                        device uchar *r20 [[buffer(22)]],\n"
      "                        device uchar *r21 [[buffer(23)]],\n"
      "                        device uchar *r22 [[buffer(24)]],\n"
      "                        device uchar *r23 [[buffer(25)]],\n"
      "                        device uchar *r24 [[buffer(26)]],\n"
      "                        device uchar *r25 [[buffer(27)]],\n"
      "                        device uchar *r26 [[buffer(28)]],\n"
      "                        device uchar *r27 [[buffer(29)]],\n"
      "                        device uchar *r28 [[buffer(30)]]) {\n"
      "  uchar sum = 0;\n"
      "  sum += r0[0] + r1[0] + r2[0] + r3[0] + r4[0] + r5[0] + r6[0] + r7[0];\n"
      "  sum += r8[0] + r9[0] + r10[0] + r11[0] + r12[0] + r13[0] + r14[0] + r15[0];\n"
      "  sum += r16[0] + r17[0] + r18[0] + r19[0] + r20[0] + r21[0] + r22[0] + r23[0];\n"
      "  sum += r24[0] + r25[0] + r26[0] + r27[0] + r28[0];\n"
      "  done[1] = sum;\n"
      "  done[0] = value;\n"
      "}\n";
  NSError *error = nil;
  id<MTLLibrary> library = [gpu().device newLibraryWithSource:@(source)
                                                      options:nil
                                                        error:&error];
  if (!library) {
    die("vx_sentinel", "Metal could not compile the sentinel:\n%s",
        error.localizedDescription.UTF8String);
  }
  id<MTLFunction> function = [library newFunctionWithName:@"vx_sentinel"];
  if (!function)
    return nil;
  made = [gpu().device newComputePipelineStateWithFunction:function error:nil];
  return made;
}

std::string keyOf(const char *image, const char *kernel) {
  return std::to_string(reinterpret_cast<uintptr_t>(image)) + "/" + kernel;
}

id<MTLComputePipelineState> pipelineFor(const char *image, const char *kernel) {
  std::lock_guard<std::mutex> hold(cacheLock);
  std::string key = keyOf(image, kernel);
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

struct Kernel {
  std::vector<std::string> params;
  bool asserts = false;
};
std::unordered_map<std::string, Kernel> kernels;

const Kernel &kernelOf(const char *image, const char *kernel) {
  std::lock_guard<std::mutex> hold(cacheLock);
  std::string key = keyOf(image, kernel);
  auto found = kernels.find(key);
  if (found != kernels.end())
    return found->second;
  Kernel k{parametersOf(image, kernel),
           assertLine(image, kernel, 1) != nullptr};
  return kernels.emplace(key, std::move(k)).first->second;
}

Access reachOf(const void *desc, int rank, size_t bytes) {
  char *aligned = static_cast<char *>(vx_memref_aligned(desc));
  const int64_t offset = vx_memref_offset(desc);
  const int64_t *sizes = vx_memref_sizes(desc);
  const int64_t *strides = vx_memref_strides(desc, rank);
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
  Access a;
  a.first = reinterpret_cast<uintptr_t>(aligned + lo * (int64_t)bytes);
  a.last = reinterpret_cast<uintptr_t>(aligned + hi * (int64_t)bytes);
  return a;
}

struct Bound {
  id<MTLBuffer> buffer;
  char *copied = nullptr;
  uintptr_t first = 0, last = 0, start = 0, end = 0, offset = 0;
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

namespace {

bool tensorsOf(void **device_args, const int32_t *arg_tags, int64_t num_args,
               std::vector<Access> &out) {
  for (int64_t i = 0; i < num_args; ++i) {
    const int32_t tag = arg_tags[i];
    if (VX_ABI_KIND(tag) != VX_ABI_KIND_MEMREF)
      continue;
    const void *desc = vx_operand_desc(device_args, arg_tags, (int)i);
    const size_t bytes = vx_dtype_bytes(VX_ABI_ELEM(tag));
    if (!desc || bytes == 0)
      return false;
    out.push_back(reachOf(desc, VX_ABI_RANK(tag), bytes));
    out.back().written = true;
  }
  return true;
}

bool anyMustPrecede(const Batch &b, const std::vector<Access> &accesses) {
  return std::any_of(accesses.begin(), accesses.end(),
                     [&](const Access &a) { return b.mustPrecede(a); });
}

} // namespace

extern "C" void vx_metal_wait(void) {
  finishEach([](const Batch &) { return true; }, false);
}

extern "C" void vx_metal_wait_for_thread(void) {
  const std::thread::id me = std::this_thread::get_id();
  finishEach([&](const Batch &b) { return b.owner == me; }, false);
}

extern "C" void vx_metal_wait_for_memory(const void *start, size_t bytes,
                                         int written) {
  if (!start || bytes == 0)
    return;
  Access a;
  a.first = reinterpret_cast<uintptr_t>(start);
  a.last = a.first + bytes;
  a.written = written != 0;
  finishEach([&](const Batch &b) { return b.mustPrecede(a); }, true);
}

static bool keepPages(void *memory, size_t length) {
  Region r;
  r.start = reinterpret_cast<uintptr_t>(memory);
  r.end = r.start + length;
  r.buffer =
      [gpu().device newBufferWithBytesNoCopy:memory
                                      length:length
                                     options:MTLResourceStorageModeShared
                                 deallocator:^(void *pointer, NSUInteger n) {
                                   munmap(pointer, n);
                                 }];
  if (!r.buffer) {
    munmap(memory, length);
    return false;
  }
  std::lock_guard<std::mutex> hold(keptLock);
  kept()[r.start] = r;
  setResident(r.buffer, true);
  return true;
}

static size_t gpuPagesFor(size_t bytes) {
  const size_t page = vm_page_size;
  if (bytes == 0 || !enabledByEnvironment())
    return 0;
  id<MTLDevice> device = gpu().device;
  const size_t length = (bytes + page - 1) & ~(page - 1);
  return device && length <= device.maxBufferLength ? length : 0;
}

extern "C" void *vx_metal_alloc(size_t bytes) {
  @autoreleasepool {
    const size_t length = gpuPagesFor(bytes);
    if (length == 0)
      return nullptr;
    void *memory = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                        MAP_ANON | MAP_PRIVATE, -1, 0);
    if (memory == MAP_FAILED || !keepPages(memory, length))
      return nullptr;
    if (verbose())
      fprintf(stderr, "[Vx Metal] made a buffer of %zu bytes of GPU memory\n",
              length);
    return memory;
  }
}

struct FileCopy {
  uintptr_t start = 0, end = 0;
  uint64_t offset = 0;
  std::string path;
  uintptr_t copy = 0;
  std::vector<std::pair<uintptr_t, uintptr_t>> given;
};

static std::mutex fileCopiesLock;

static std::vector<FileCopy> &fileCopies() {
  static auto *all = new std::vector<FileCopy>;
  return *all;
}

static FileCopy *fileCopyOf(uintptr_t start, uintptr_t end, uint64_t offset,
                            const char *path) {
  for (FileCopy &f : fileCopies())
    if (f.start == start && f.end == end && f.offset == offset &&
        f.path == path)
      return &f;
  struct proc_regionwithpathinfo info;
  if (proc_pidinfo(getpid(), PROC_PIDREGIONPATHINFO, start, &info,
                   sizeof info) != sizeof info)
    return nullptr;
  const int fd = open(path, O_RDONLY);
  if (fd < 0)
    return nullptr;
  struct stat now;
  const struct vinfo_stat &mapped = info.prp_vip.vip_vi.vi_stat;
  void *copy = MAP_FAILED;
  if (fstat(fd, &now) == 0 && (uint64_t)now.st_dev == mapped.vst_dev &&
      (uint64_t)now.st_ino == mapped.vst_ino)
    copy = mmap(nullptr, end - start, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd,
                (off_t)offset);
  close(fd);
  if (copy == MAP_FAILED)
    return nullptr;
  FileCopy made;
  made.start = start;
  made.end = end;
  made.offset = offset;
  made.path = path;
  made.copy = reinterpret_cast<uintptr_t>(copy);
  fileCopies().push_back(made);
  return &fileCopies().back();
}

extern "C" void *vx_metal_map_copy(const void *source, size_t bytes) {
  const uintptr_t at = reinterpret_cast<uintptr_t>(source);
  if (!source || at % vm_page_size != 0)
    return nullptr;
  @autoreleasepool {
    const size_t length = gpuPagesFor(bytes);
    if (length == 0)
      return nullptr;
    mach_vm_address_t start = at;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t region;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &start, &size, VM_REGION_BASIC_INFO_64,
                       reinterpret_cast<vm_region_info_t>(&region), &count,
                       &object) != KERN_SUCCESS)
      return nullptr;
    if (object != MACH_PORT_NULL)
      mach_port_deallocate(mach_task_self(), object);
    char path[MAXPATHLEN];
    if (at < start || at + length > start + size ||
        (region.max_protection & VM_PROT_WRITE) != 0 ||
        proc_regionfilename(getpid(), at, path, sizeof path) <= 0)
      return nullptr;
    void *memory = nullptr;
    {
      std::lock_guard<std::mutex> hold(fileCopiesLock);
      FileCopy *file = fileCopyOf(start, start + size, region.offset, path);
      if (!file)
        return nullptr;
      for (const auto &g : file->given)
        if (at < g.second && g.first < at + length)
          return nullptr;
      file->given.push_back({at, at + length});
      memory = reinterpret_cast<void *>(file->copy + (at - start));
    }
    if (!keepPages(memory, length))
      return nullptr;
    if (verbose())
      fprintf(stderr,
              "[Vx Metal] made a buffer over %zu bytes of the file %s, with "
              "no copy\n",
              length, path);
    return memory;
  }
}

extern "C" int vx_metal_free(void *memory) {
  const uintptr_t at = reinterpret_cast<uintptr_t>(memory);
  size_t length = 0;
  {
    std::lock_guard<std::mutex> hold(keptLock);
    auto found = kept().find(at);
    if (found == kept().end())
      return 0;
    length = found->second.end - at;
  }
  vx_metal_wait_for_memory(memory, length, 1);
  {
    std::lock_guard<std::mutex> hold(keptLock);
    auto found = kept().find(at);
    setResident(found->second.buffer, false);
    kept().erase(found);
  }
  if (verbose())
    fprintf(stderr, "[Vx Metal] freed %zu bytes of GPU memory\n", length);
  return 1;
}

extern "C" void vx_metal_wait_for_tensors(void **args, const int32_t *tags,
                                          int64_t count) {
  std::vector<Access> accesses;
  if (!tensorsOf(args, tags, count, accesses)) {
    vx_metal_wait();
    return;
  }
  for (Access &a : accesses)
    a.written = false;
  finishEach([&](const Batch &b) { return anyMustPrecede(b, accesses); }, true);
}

extern "C" void vx_metal_wait_for_launch(const void *payload,
                                         size_t payload_size,
                                         void **device_args,
                                         const int32_t *arg_tags,
                                         int64_t num_args) {
  std::vector<Access> accesses;
  if (!tensorsOf(device_args, arg_tags, num_args, accesses)) {
    vx_metal_wait();
    return;
  }
  const char *format = vx_payload_field(payload, payload_size, "format=");
  if (format && strcmp(format, "msl") == 0) {
    const char *image = vx_payload_field(payload, payload_size, "image=");
    if (!image)
      die(static_cast<const char *>(payload),
          "the payload says `format=msl` and has no image");
    const std::vector<std::string> &params =
        kernelOf(image, static_cast<const char *>(payload)).params;
    size_t k = 0;
    for (int64_t i = 0; i < num_args && i < (int64_t)params.size(); ++i)
      if (VX_ABI_KIND(arg_tags[i]) == VX_ABI_KIND_MEMREF)
        accesses[k++].written = params[i][0] != 'm' || params[i].back() != 'r';
  }
  finishEach([&](const Batch &b) { return anyMustPrecede(b, accesses); }, true);
}

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
    const Kernel &info = kernelOf(image, kernel);
    const std::vector<std::string> &params = info.params;
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
        const Access reach = reachOf(desc, rank, bytes);
        Bound b;
        b.first = reach.first;
        b.last = reach.last;
        b.start = b.first & ~(page - 1);
        b.end = (b.last + page - 1) & ~(page - 1);
        if (b.end == b.start)
          b.end = b.start + page;
        b.written = p.back() != 'r';
        const int64_t base =
            (int64_t)(reinterpret_cast<uintptr_t>(aligned) - b.start) /
            (int64_t)bytes;
        if ((reinterpret_cast<uintptr_t>(aligned) - b.start) % bytes != 0)
          die(kernel, "tensor %lld is not aligned to its element size",
              (long long)i);
        if ((b.end - b.start) / bytes > (uintptr_t)INT32_MAX)
          die(kernel,
              "tensor %lld spans %llu elements, and a Metal kernel "
              "takes 32-bit indices",
              (long long)i, (unsigned long long)((b.end - b.start) / bytes));
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

    std::vector<std::pair<uintptr_t, uintptr_t>> spans;
    std::vector<Access> accesses;
    for (const Bound &b : bound) {
      spans.push_back({b.start, b.end});
      accesses.push_back({b.first, b.last, b.written});
    }
    std::sort(spans.begin(), spans.end());
    std::vector<std::pair<uintptr_t, uintptr_t>> joined;
    for (const auto &span : spans)
      if (!joined.empty() && span.first < joined.back().second)
        joined.back().second = std::max(joined.back().second, span.second);
      else
        joined.push_back(span);

    std::unique_lock<std::mutex> hold(batchLock);
    for (;;) {
      std::shared_ptr<Batch> here = openBatchHere();
      std::vector<std::shared_ptr<Batch>> chosen;
      for (const std::shared_ptr<Batch> &b : batches()) {
        if (b != here) {
          if (anyMustPrecede(*b, accesses))
            chosen.push_back(b);
          continue;
        }
        for (const auto &span : joined)
          if (!here->holding(span.first, span.second) &&
              here->overlaps(span.first, span.second)) {
            chosen.push_back(b);
            break;
          }
      }
      if (chosen.empty())
        break;
      finish(chosen, hold, true);
    }
    std::shared_ptr<Batch> here = openBatchHere();
    std::vector<Region> made;
    for (const auto &span : joined) {
      if ((here && here->holding(span.first, span.second)) ||
          holding(made, span.first, span.second))
        continue;
      Region r = keptHolding(span.first, span.second);
      if (!r.buffer) {
        r.start = span.first;
        r.end = span.second;
        r.buffer =
            [device newBufferWithBytesNoCopy:reinterpret_cast<void *>(r.start)
                                      length:r.end - r.start
                                     options:MTLResourceStorageModeShared
                                 deallocator:nil];
        if (r.buffer && verbose())
          fprintf(stderr, "[Vx Metal] made a buffer of %zu bytes for a batch\n",
                  (size_t)(r.end - r.start));
      }
      if (r.buffer)
        made.push_back(r);
    }
    for (Bound &b : bound) {
      const Region *r = here ? here->holding(b.start, b.end) : nullptr;
      if (!r)
        r = holding(made, b.start, b.end);
      if (r) {
        b.buffer = r->buffer;
        b.offset = b.start - r->start;
      }
    }
    const bool copies = std::any_of(bound.begin(), bound.end(),
                                    [](const Bound &b) { return !b.buffer; });
    if (copies) {
      const std::vector<std::shared_ptr<Batch>> all = batches();
      finish(all, hold, true);
      here = nullptr;
    }
    for (size_t i = 0; i < bound.size(); ++i) {
      Bound &b = bound[i];
      if (b.buffer)
        continue;
      b.buffer = [device newBufferWithBytes:reinterpret_cast<void *>(b.start)
                                     length:b.end - b.start
                                    options:MTLResourceStorageModeShared];
      if (!b.buffer)
        die(kernel, "Metal could not make a buffer for tensor %zu", i);
      b.copied = static_cast<char *>(b.buffer.contents);
      if (verbose())
        fprintf(stderr, "[Vx Metal] %s: tensor %zu is copied\n", kernel, i);
    }
    for (size_t a = 0; a < bound.size(); ++a)
      for (size_t b = 0; b < bound.size(); ++b)
        if (a != b && bound[a].copied && bound[a].written && bound[b].written &&
            bound[a].first < bound[b].last && bound[b].first < bound[a].last)
          die(kernel, "two tensors share memory that Metal could not use as "
                      "it is");

    if (!here) {
      here = std::make_shared<Batch>();
      here->owner = std::this_thread::get_id();
      here->commands = [gpu().queue commandBuffer];
      here->encoder = newEncoder(here->commands);

      static std::map<std::thread::id, id<MTLBuffer>> threadDones;
      id<MTLBuffer> threadDone = threadDones[here->owner];
      if (!threadDone) {
        threadDone = [device newBufferWithLength:2 * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
        if (threadDone) memset(threadDone.contents, 0, 2 * sizeof(uint32_t));
        threadDones[here->owner] = threadDone;
      }
      here->done = threadDone;

      here->cmdBufferKernelStart = here->kernels.size();
      here->cmdBufferCheckStart = here->checks.size();

      batches().push_back(here);
      static const bool waitAtExit = atexit(vx_metal_wait) == 0;
      (void)waitAtExit;
    }
    here->regions.insert(here->regions.end(), made.begin(), made.end());
    for (const Access &a : accesses)
      here->add(a);
    id<MTLComputeCommandEncoder> encoder = here->encoder;
    if (concurrent()) {
      const bool after =
          std::any_of(accesses.begin(), accesses.end(), [&](const Access &a) {
            return std::any_of(
                here->sinceBarrier.begin(), here->sinceBarrier.end(),
                [&](const Access &b) { return a.mustFollow(b); });
          });
      if (after) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        here->sinceBarrier.clear();
        if (verbose())
          fprintf(stderr, "[Vx Metal] %s: a barrier before it\n", kernel);
      }
      here->sinceBarrier.insert(here->sinceBarrier.end(), accesses.begin(),
                                accesses.end());
    }
    [encoder setComputePipelineState:pipeline];
    for (size_t k = 0; k < bound.size(); ++k)
      [encoder setBuffer:bound[k].buffer offset:bound[k].offset atIndex:k];
    const size_t wordBytes = words.size() * sizeof(uint32_t);
    if (wordBytes <= 4096)
      [encoder setBytes:words.data() length:wordBytes atIndex:bound.size()];
    else
      [encoder
          setBuffer:[device newBufferWithBytes:words.data()
                                        length:wordBytes
                                       options:MTLResourceStorageModeShared]
             offset:0
            atIndex:bound.size()];
    if (info.asserts) {
      if (!here->assertBuffer || here->assertsGiven == assertWords) {
        here->assertBuffer =
            [device newBufferWithLength:assertWords * sizeof(uint32_t)
                                options:MTLResourceStorageModeShared];
        if (!here->assertBuffer)
          die(kernel, "Metal could not make the buffer for its asserts");
        memset(here->assertBuffer.contents, 0, assertWords * sizeof(uint32_t));
        here->assertsGiven = 0;
      }
      Check c{kernel, image, here->assertBuffer, here->assertsGiven++};
      [encoder setBuffer:c.words
                  offset:c.at * sizeof(uint32_t)
                 atIndex:bound.size() + 1];
      here->checks.push_back(c);
    }
    [encoder dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2])
            threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
    here->kernels.push_back(kernel);
    if (verbose())
      fprintf(stderr, "[Vx Metal] %s: grid %u x %u x %u, block %u\n", kernel,
              grid[0], grid[1], grid[2], block);

    if (copies || syncEachLaunch()) {
      finish({here}, hold, true);
      for (const Bound &b : bound)
        if (b.copied && b.written)
          memcpy(reinterpret_cast<void *>(b.first),
                 b.copied + (b.first - b.start), b.last - b.first);
    }
    if (here->open() && ++here->unsent >= 16) {
      if (verbose())
        fprintf(stderr, "[Vx Metal] committed %zu kernels before the wait\n",
                here->unsent);
      dispatchSentinel(here->encoder, *here);
      [here->encoder endEncoding];
      reportLateFault(here->commands, here);
      [here->commands commit];
      here->sent.push_back(here->commands);
      here->commands = [gpu().queue commandBuffer];
      here->encoder = newEncoder(here->commands);
      here->sinceBarrier.clear();
      here->unsent = 0;
      here->cmdBufferKernelStart = here->kernels.size();
      here->cmdBufferCheckStart = here->checks.size();
    }
  }
  return 1;
}
