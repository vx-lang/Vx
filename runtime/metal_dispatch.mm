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
// from vx_metal_alloc gets one buffer over all of its pages, once. Other memory
// gets a buffer at each batch, because the program can free it without telling
// this file.
//
// The launches of each host thread make one batch until a wait. The batch
// commits its command buffer after every few launches, so the GPU runs them
// while the CPU makes the next. A wait commits the rest and waits for the GPU.
// The Mac dispatcher waits before anything that can touch the memory of a
// kernel, but only for the launches that use that memory. A `pub fn` waits
// before it returns to C. A failed assert in a kernel stops the program at the
// wait that finishes its launch.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../include/vx_hardware_runtime.h"
#include "vx_dispatch_plan.h"
#include "vx_kernel_launch.h"

#include <mach/vm_page_size.h>
#include <sys/mman.h>

#include <algorithm>
#include <cctype>
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

// The bytes from `first` to before `last` that a launch reads, or writes when
// `written` is true.
struct Access {
  uintptr_t first = 0, last = 0;
  bool written = false;

  // Whether this access must wait for `other`: they share bytes, and one of
  // them writes.
  bool mustFollow(const Access &other) const {
    return first < other.last && other.first < last &&
           (written || other.written);
  }
};

// A buffer over pages from `start` to before `end`, with no copy.
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

// The GPU memory from vx_metal_alloc, by its first byte. Its pages hold
// nothing else, so each launch uses the same buffer for a tensor in it.
std::mutex keptLock;

std::map<uintptr_t, Region> &kept() {
  static auto *all = new std::map<uintptr_t, Region>;
  return *all;
}

// The region of GPU memory that holds the bytes from `start` to before `end`,
// or a region with no buffer.
Region keptHolding(uintptr_t start, uintptr_t end) {
  std::lock_guard<std::mutex> hold(keptLock);
  auto at = kept().upper_bound(start);
  if (at == kept().begin())
    return {};
  --at;
  return end <= at->second.end ? at->second : Region{};
}

// A launch of a kernel with asserts, and the word where a failed assert
// stores its number.
struct Check {
  std::string kernel;
  const char *image;
  id<MTLBuffer> word;
};

// The launches of one host thread that the GPU did not finish, in one command
// buffer. A wait for some memory finishes only the batches that use it, so a
// wait on one thread does not end the batch of another thread.
//
// Metal orders two launches only when they use the same buffer object. On an
// M4, a kernel did not see what the kernel before it wrote through another
// buffer over the same memory, even with a barrier between them. So no two
// regions of a batch overlap, and a launch uses the region that holds each of
// its tensors. Before a launch that must follow a launch of another batch,
// that batch finishes. So two batches use the same memory only to read it.
struct Batch {
  std::thread::id owner;
  // The command buffer that takes the next launch, and the ones that the batch
  // committed before it.
  id<MTLCommandBuffer> commands;
  std::vector<id<MTLCommandBuffer>> sent;
  // Nil after the commit.
  id<MTLComputeCommandEncoder> encoder;
  // The launches in `commands`.
  size_t unsent = 0;
  std::vector<std::string> kernels;
  std::vector<Region> regions;
  std::vector<Access> accesses;
  std::vector<Check> checks;

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

// Guards `batches()`. Only the owner of a batch adds launches to it, while it
// holds this lock.
std::mutex batchLock;

// The batches that the GPU did not finish, open and committed.
std::vector<std::shared_ptr<Batch>> &batches() {
  static auto *all = new std::vector<std::shared_ptr<Batch>>;
  return *all;
}

// The open batch of the calling thread, or null. The caller holds `batchLock`.
std::shared_ptr<Batch> openBatchHere() {
  for (const std::shared_ptr<Batch> &b : batches())
    if (b->open() && b->owner == std::this_thread::get_id())
      return b;
  return nullptr;
}

// The text after the number in the line `// vx.assert <kernel> <number> ...`
// of `image`, or null when the image has no such line.
const char *assertLine(const char *image, const char *kernel, uint32_t number) {
  std::string key = std::string("// vx.assert ") + kernel + " " +
                    std::to_string(number) + " ";
  for (const char *at = strstr(image, key.c_str()); at;
       at = strstr(at + 1, key.c_str()))
    if (at == image || at[-1] == '\n')
      return at + key.size();
  return nullptr;
}

// The message of assert `number` of `kernel`, without its percent-encoding.
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

// Stops the program when an assert failed in a batch, or when the GPU did not
// run the batch to its end.
void report(const Batch &b) {
  // A failed assert can also stop the GPU, so it is reported first.
  for (const Check &c : b.checks) {
    const uint32_t number =
        *static_cast<const volatile uint32_t *>(c.word.contents);
    if (number != 0)
      die(c.kernel.c_str(), "%s",
          assertMessage(c.image, c.kernel.c_str(), number).c_str());
  }
  const size_t count = b.kernels.size();
  std::vector<id<MTLCommandBuffer>> all = b.sent;
  all.push_back(b.commands);
  double seconds = 0;
  for (id<MTLCommandBuffer> commands : all) {
    if (commands.status != MTLCommandBufferStatusCompleted) {
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
    seconds += commands.GPUEndTime - commands.GPUStartTime;
  }
  if (verbose())
    fprintf(stderr, "[Vx Metal] waited for %zu kernel%s, %.3f ms on the GPU\n",
            count, count == 1 ? "" : "s", seconds * 1e3);
}

// Commits each batch in `chosen` that is open, and waits until the GPU
// finishes them. `hold` holds `batchLock`. The wait itself runs without the
// lock, so other threads can add launches to their own batches meanwhile.
void finish(const std::vector<std::shared_ptr<Batch>> &chosen,
            std::unique_lock<std::mutex> &hold) {
  if (chosen.empty())
    return;
  for (const std::shared_ptr<Batch> &b : chosen)
    if (b->open()) {
      [b->encoder endEncoding];
      b->encoder = nil;
      [b->commands commit];
    }
  hold.unlock();
  for (const std::shared_ptr<Batch> &b : chosen) {
    for (id<MTLCommandBuffer> commands : b->sent)
      [commands waitUntilCompleted];
    [b->commands waitUntilCompleted];
  }
  hold.lock();
  std::vector<std::shared_ptr<Batch>> &all = batches();
  for (const std::shared_ptr<Batch> &b : chosen) {
    auto at = std::find(all.begin(), all.end(), b);
    // Another thread waited for this batch too, and reported it.
    if (at == all.end())
      continue;
    all.erase(at);
    report(*b);
  }
}

// Finishes each batch for which `pick` is true.
template <typename Pick> void finishEach(Pick pick) {
  std::unique_lock<std::mutex> hold(batchLock);
  std::vector<std::shared_ptr<Batch>> chosen;
  for (const std::shared_ptr<Batch> &b : batches())
    if (pick(*b))
      chosen.push_back(b);
  finish(chosen, hold);
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

// The bytes that a tensor can reach. A kernel reads a stride or an offset that
// its type fixes from the type, so they cover the contiguous layout too.
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

// A tensor's buffer. `copied` is set when Metal could not use the memory as it
// is, and the kernel ran on a copy.
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

// The bytes of each tensor among the arguments of a launch, each one written.
// Returns false when the bytes of a tensor are not known.
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

// Waits for every launch.
extern "C" void vx_metal_wait(void) {
  finishEach([](const Batch &) { return true; });
}

// Waits for every launch of the calling thread.
extern "C" void vx_metal_wait_for_thread(void) {
  const std::thread::id me = std::this_thread::get_id();
  finishEach([&](const Batch &b) { return b.owner == me; });
}

// Waits for each launch that writes the `bytes` at `start`. With `written`, it
// also waits for each launch that reads them.
extern "C" void vx_metal_wait_for_memory(const void *start, size_t bytes,
                                         int written) {
  if (!start || bytes == 0)
    return;
  Access a;
  a.first = reinterpret_cast<uintptr_t>(start);
  a.last = a.first + bytes;
  a.written = written != 0;
  finishEach([&](const Batch &b) { return b.mustPrecede(a); });
}

// Maps new pages for `bytes` of GPU memory, and makes their buffer. Returns
// null when the GPU is off, when `bytes` is less than a page, or when Metal
// refuses. The caller then uses `malloc`.
extern "C" void *vx_metal_alloc(size_t bytes) {
  const size_t page = vm_page_size;
  if (bytes < page || !enabledByEnvironment())
    return nullptr;
  @autoreleasepool {
    id<MTLDevice> device = gpu().device;
    const size_t length = (bytes + page - 1) & ~(page - 1);
    if (!device || length > device.maxBufferLength)
      return nullptr;
    void *memory = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                        MAP_ANON | MAP_PRIVATE, -1, 0);
    if (memory == MAP_FAILED)
      return nullptr;
    // A batch can hold the buffer after vx_metal_free, so the pages go when
    // Metal releases the buffer.
    Region r;
    r.start = reinterpret_cast<uintptr_t>(memory);
    r.end = r.start + length;
    r.buffer = [device newBufferWithBytesNoCopy:memory
                                         length:length
                                        options:MTLResourceStorageModeShared
                                    deallocator:^(void *pointer, NSUInteger n) {
                                      munmap(pointer, n);
                                    }];
    if (!r.buffer) {
      munmap(memory, length);
      return nullptr;
    }
    {
      std::lock_guard<std::mutex> hold(keptLock);
      kept()[r.start] = r;
    }
    if (verbose())
      fprintf(stderr, "[Vx Metal] made a buffer of %zu bytes of GPU memory\n",
              length);
    return memory;
  }
}

// Frees memory from vx_metal_alloc, after the launches that use it. Returns 0
// when `memory` did not come from vx_metal_alloc.
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
    kept().erase(at);
  }
  if (verbose())
    fprintf(stderr, "[Vx Metal] freed %zu bytes of GPU memory\n", length);
  return 1;
}

// Waits for each launch that writes the memory of one of these tensors, before
// the CPU reads them.
extern "C" void vx_metal_wait_for_tensors(void **args, const int32_t *tags,
                                          int64_t count) {
  std::vector<Access> accesses;
  if (!tensorsOf(args, tags, count, accesses)) {
    vx_metal_wait();
    return;
  }
  for (Access &a : accesses)
    a.written = false;
  finishEach([&](const Batch &b) { return anyMustPrecede(b, accesses); });
}

// Waits for each GPU launch that a launch with these arguments must follow,
// when that launch does not run on this Mac's GPU. The `// vx.kernel` line of
// an Apple GPU kernel says which tensors it only reads. Any other kernel can
// write each of its tensors.
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
    std::vector<std::string> params =
        parametersOf(image, static_cast<const char *>(payload));
    size_t k = 0;
    for (int64_t i = 0; i < num_args && i < (int64_t)params.size(); ++i)
      if (VX_ABI_KIND(arg_tags[i]) == VX_ABI_KIND_MEMREF)
        accesses[k++].written = params[i][0] != 'm' || params[i].back() != 'r';
  }
  finishEach([&](const Batch &b) { return anyMustPrecede(b, accesses); });
}

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

    // The pages of the tensors, where two tensors that share a page share one
    // region.
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
    // The batches of other threads that this launch must follow finish first.
    // A region that only overlaps a region of this thread's batch cannot get a
    // buffer of its own, so then this thread's batch finishes too.
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
      finish(chosen, hold);
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
      // When Metal refuses this memory, its tensors run on a copy below.
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
    // A copy must hold what every launch writes, so all batches finish first.
    const bool copies = std::any_of(bound.begin(), bound.end(),
                                    [](const Bound &b) { return !b.buffer; });
    if (copies) {
      const std::vector<std::shared_ptr<Batch>> all = batches();
      finish(all, hold);
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
    // With the copies, two written tensors that share memory would undo each
    // other's writes when they are copied back.
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
      here->encoder = [here->commands computeCommandEncoder];
      batches().push_back(here);
      // Report a fault of a kernel that nothing waited for.
      static const bool waitAtExit = atexit(vx_metal_wait) == 0;
      (void)waitAtExit;
    }
    here->regions.insert(here->regions.end(), made.begin(), made.end());
    for (const Access &a : accesses)
      here->add(a);
    id<MTLComputeCommandEncoder> encoder = here->encoder;
    [encoder setComputePipelineState:pipeline];
    for (size_t k = 0; k < bound.size(); ++k)
      [encoder setBuffer:bound[k].buffer offset:bound[k].offset atIndex:k];
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
    if (assertLine(image, kernel, 1)) {
      Check c{kernel, image,
              [device newBufferWithLength:sizeof(uint32_t)
                                  options:MTLResourceStorageModeShared]};
      if (!c.word)
        die(kernel, "Metal could not make the buffer for its asserts");
      memset(c.word.contents, 0, sizeof(uint32_t));
      [encoder setBuffer:c.word offset:0 atIndex:bound.size() + 1];
      here->checks.push_back(c);
    }
    [encoder dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2])
            threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
    here->kernels.push_back(kernel);
    if (verbose())
      fprintf(stderr, "[Vx Metal] %s: grid %u x %u x %u, block %u\n", kernel,
              grid[0], grid[1], grid[2], block);

    // A copy goes back to its tensor after the GPU ran the kernel.
    if (copies || syncEachLaunch()) {
      finish({here}, hold);
      for (const Bound &b : bound)
        if (b.copied && b.written)
          memcpy(reinterpret_cast<void *>(b.first),
                 b.copied + (b.first - b.start), b.last - b.first);
    }
    // The GPU runs a command buffer only after its commit. So the batch commits
    // after a few launches, and the GPU runs them while the CPU makes the next.
    if (here->open() && ++here->unsent >= 16) {
      if (verbose())
        fprintf(stderr, "[Vx Metal] committed %zu kernels before the wait\n",
                here->unsent);
      [here->encoder endEncoding];
      [here->commands commit];
      here->sent.push_back(here->commands);
      here->commands = [gpu().queue commandBuffer];
      here->encoder = [here->commands computeCommandEncoder];
      here->unsent = 0;
    }
  }
  return 1;
}
