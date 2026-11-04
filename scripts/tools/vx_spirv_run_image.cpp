// Run a Vx-compiled SPIR-V kernel on an Intel GPU, through the dispatch
// library's own entry points.
//
// `vxc prog.vx --emit-llvm` embeds the dispatch payload in the program: the
// kernel's name, some `key=value` entries, and the SPIR-V image in a
// length-prefixed section after them. The shell driver beside this file takes
// that payload out and hands it here. This program then does what a compiled Vx
// program does -- allocates the tensors with `vx_plugin_alloc_and_transfer`,
// dispatches with `vx_plugin_dispatch_async`, reads the result back with
// `vx_plugin_transfer_device_to_host`, and compares it with the same
// computation on the CPU. The library path never opens a Level Zero device
// itself, so a correct answer is the library's ABI working, not a launch
// written beside it.
//
// The kernel is `tests/backend/pass/spirv_device_image_loop_kernel.vx`:
//
//   o[i][j] = a[i][j] * b[i][j] + c[i][j]      for 8 rows of 4
//
// so it has four rank-2 operands. The ABI gives each operand seven separate
// kernel arguments -- the allocation's base pointer, the aligned base pointer,
// the offset, two sizes and two strides -- which is 28 arguments, and
// `spirv-dis` on the image shows exactly 28 `OpFunctionParameter`s. The library
// re-derives that count from the image and refuses to launch when the arguments
// do not produce it, so a wrong parameter list is caught before the kernel
// runs.
//
// `direct` mode keeps a launch through Level Zero alone as a control: the same
// image with the same descriptors, without the library. It tells a wrong result
// caused by the library apart from one caused by the image.
//
// Memory: four tensors of 128 bytes, allocated through the library. This prints
// its footprint and its ceiling and refuses to allocate past `VX_VRAM_CEILING`
// (2 GiB by default), the same rule the other helper in this directory follows.
//
// Run with `VX_DISPATCH_VERBOSE=1` to see the target the library picked and the
// format it accepted; the shell driver sets it.
//
// Build and run: scripts/tools/vx_spirv_run_image.sh

#include "vram_ceiling.h"

#include <level_zero/ze_api.h>
#include <vx_hardware_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#define L0_CHECK(call)                                                         \
  do {                                                                         \
    ze_result_t r_ = (call);                                                   \
    if (r_ != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "%s:%d: %s failed with 0x%x\n", __FILE__, __LINE__, \
                   #call, static_cast<unsigned>(r_));                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static std::vector<char> readFile(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  return std::vector<char>((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
}

constexpr uint64_t kRows = 8, kCols = 4;
constexpr uint64_t kElems = kRows * kCols;
constexpr uint64_t kBytes = kElems * sizeof(float);
constexpr uint64_t kOperands = 4;  // a, b, c, o
constexpr uint64_t kWorkgroup = 8; // the payload's `launch=8`: one group of 8
constexpr uint64_t kGroups = 1;    // the kernel grid-strides by the whole grid
constexpr uint64_t kFootprint = kOperands * kBytes;
constexpr int kWarmup = 3, kReps = 20;

// The GPU band's topology id. The compiler emits this for a
// `transfer(t, Memory::GPU_HBM)` (src/arch.rs), and it is the id the library
// decodes to a device ordinal -- so it is the id a compiled program passes.
constexpr uint32_t kGpuTopology = 500;

// The sabotage that corrupts the image flips the low byte of the 64-bit
// constant that holds the row stride (4 elements). The module still validates;
// only the addresses change, so the result is the thing that catches it.
static constexpr uint32_t kOpConstant = 43;

/// Flip one byte of the row-stride constant, in place. Returns false if this
/// module has no 64-bit constant equal to 4, which means the sabotage no
/// longer applies and the caller should say so rather than pretend to test.
static bool corruptRowStride(std::vector<char> &image) {
  if (image.size() % 4 != 0)
    return false;
  const uint32_t *w = reinterpret_cast<const uint32_t *>(image.data());
  size_t words = image.size() / 4;
  for (size_t i = 5 /* past the header */; i + 4 < words;) {
    uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
    if (wc == 0)
      break;
    if (op == kOpConstant && wc == 5) {
      uint64_t value = (uint64_t(w[i + 4]) << 32) | w[i + 3];
      if (value == kCols) {
        image[(i + 3) * 4] ^= 0x01; // 4 -> 5
        return true;
      }
    }
    i += wc;
  }
  return false;
}

/// Overwrite the payload's `format=` value in place, keeping its length so the
/// section after the entries stays where it is. Returns false when this payload
/// does not say `format=spirv`, so a payload that changed shape fails the
/// sabotage rather than silently testing nothing.
static bool rewriteFormat(std::vector<char> &payload, const char *value) {
  const char *entry = "format=spirv";
  const size_t entry_len = std::strlen(entry);
  if (std::strlen(value) != std::strlen(entry) - 7 /* "format=" */) {
    std::fprintf(stderr, "rewriteFormat: %s does not fit where spirv was\n",
                 value);
    std::exit(2);
  }
  const auto found =
      std::search(payload.begin(), payload.end(), entry, entry + entry_len);
  if (found == payload.end())
    return false;
  std::memcpy(&*found + 7, value, std::strlen(value));
  return true;
}

/// The tensors this program launches: values that vary along both axes, so no
/// operand can be dropped or swapped without changing the result, and the same
/// formula on the CPU. Every value is a small integer, which fp32 holds
/// exactly, so the check is equality and no tolerance is needed: any difference
/// is the kernel, not rounding.
static void fillReference(std::vector<float> &a, std::vector<float> &b,
                          std::vector<float> &c, std::vector<float> &expected) {
  a.assign(kElems, 0.0f);
  b.assign(kElems, 0.0f);
  c.assign(kElems, 0.0f);
  expected.assign(kElems, 0.0f);
  for (uint64_t i = 0; i < kRows; ++i)
    for (uint64_t j = 0; j < kCols; ++j) {
      a[i * kCols + j] = float(i + j);
      b[i * kCols + j] = 2.0f;
      c[i * kCols + j] = float(j + 2);
    }
  for (uint64_t i = 0; i < kElems; ++i)
    expected[i] = a[i] * b[i] + c[i];
}

/// Count the elements that came back different, printing the first few.
static uint64_t countWrong(const std::vector<float> &got,
                           const std::vector<float> &expected) {
  uint64_t wrong = 0;
  for (uint64_t i = 0; i < kElems; ++i)
    if (got[i] != expected[i]) {
      if (wrong < 4)
        std::printf("  element %llu: got %f, wanted %f\n",
                    (unsigned long long)i, static_cast<double>(got[i]),
                    static_cast<double>(expected[i]));
      ++wrong;
    }
  return wrong;
}

static void reportResult(uint64_t wrong) {
  std::printf("result: %s (%llu of %llu elements differ)\n",
              wrong ? "WRONG" : "correct", (unsigned long long)wrong,
              (unsigned long long)kElems);
}

template <typename Run> static double bestOf(const Run &run) {
  for (int i = 0; i < kWarmup; ++i)
    run();
  double best = 1e30;
  for (int i = 0; i < kReps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    run();
    auto t1 = std::chrono::steady_clock::now();
    best = std::min(best,
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  return best;
}

/// Launch the image through Level Zero alone. This is the control: it uses the
/// same payload's image and the same flattened argument list the library
/// builds, with nothing between them and the driver.
static uint64_t runDirect(const char *image, size_t imageSize,
                          const char *kernelName) {
  std::vector<float> a, b, c, expected;
  fillReference(a, b, c, expected);

  L0_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t driverCount = 0;
  L0_CHECK(zeDriverGet(&driverCount, nullptr));
  std::vector<ze_driver_handle_t> drivers(driverCount);
  L0_CHECK(zeDriverGet(&driverCount, drivers.data()));

  ze_driver_handle_t driver = nullptr;
  ze_device_handle_t device = nullptr;
  for (ze_driver_handle_t d : drivers) {
    uint32_t count = 0;
    L0_CHECK(zeDeviceGet(d, &count, nullptr));
    std::vector<ze_device_handle_t> devices(count);
    L0_CHECK(zeDeviceGet(d, &count, devices.data()));
    for (ze_device_handle_t dev : devices) {
      ze_device_properties_t props = {};
      props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      L0_CHECK(zeDeviceGetProperties(dev, &props));
      if (props.type == ZE_DEVICE_TYPE_GPU) {
        driver = d;
        device = dev;
        std::printf("device: %s  (%u EUs)\n", props.name,
                    props.numEUsPerSubslice * props.numSubslicesPerSlice *
                        props.numSlices);
        break;
      }
    }
    if (device)
      break;
  }
  if (!device) {
    std::fprintf(stderr, "no Level Zero GPU found\n");
    std::exit(1);
  }

  ze_context_desc_t contextDesc = {};
  contextDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ze_context_handle_t context = nullptr;
  L0_CHECK(zeContextCreate(driver, &contextDesc, &context));

  ze_command_queue_desc_t queueDesc = {};
  queueDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  ze_command_queue_handle_t queue = nullptr;
  L0_CHECK(zeCommandQueueCreate(context, device, &queueDesc, &queue));

  ze_command_list_desc_t listDesc = {};
  listDesc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  ze_command_list_handle_t list = nullptr;
  L0_CHECK(zeCommandListCreate(context, device, &listDesc, &list));

  ze_device_mem_alloc_desc_t allocDesc = {};
  allocDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  void *dev[kOperands] = {};
  for (uint64_t i = 0; i < kOperands; ++i)
    L0_CHECK(zeMemAllocDevice(context, &allocDesc, kBytes, 8, device, &dev[i]));

  ze_module_desc_t moduleDesc = {};
  moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  moduleDesc.inputSize = imageSize;
  moduleDesc.pInputModule = reinterpret_cast<const uint8_t *>(image);
  ze_module_handle_t module = nullptr;
  ze_module_build_log_handle_t buildLog = nullptr;
  ze_result_t mr =
      zeModuleCreate(context, device, &moduleDesc, &module, &buildLog);
  if (mr != ZE_RESULT_SUCCESS) {
    size_t logSize = 0;
    if (buildLog) {
      zeModuleBuildLogGetString(buildLog, &logSize, nullptr);
      std::vector<char> log(logSize + 1, 0);
      zeModuleBuildLogGetString(buildLog, &logSize, log.data());
      std::fprintf(stderr, "the driver refused the module: %s\n", log.data());
    } else {
      std::fprintf(stderr, "zeModuleCreate failed with 0x%x\n",
                   static_cast<unsigned>(mr));
    }
    return kElems;
  }

  ze_kernel_desc_t kernelDesc = {};
  kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernelDesc.pKernelName = kernelName;
  ze_kernel_handle_t kernel = nullptr;
  L0_CHECK(zeKernelCreate(module, &kernelDesc, &kernel));
  L0_CHECK(zeKernelSetGroupSize(kernel, kWorkgroup, 1, 1));

  std::vector<float> o(kElems, 0.0f);
  struct {
    void *buf;
    const float *src;
  } uploads[3] = {{dev[0], a.data()}, {dev[1], b.data()}, {dev[2], c.data()}};
  for (auto &u : uploads)
    L0_CHECK(zeCommandListAppendMemoryCopy(list, u.buf, u.src, kBytes, nullptr,
                                           0, nullptr));
  L0_CHECK(zeCommandListAppendMemoryCopy(list, dev[3], o.data(), kBytes,
                                         nullptr, 0, nullptr));

  // The flattened ABI, in order: per tensor, two pointers, an offset, two
  // sizes and two strides, each its own argument.
  for (uint64_t t = 0; t < kOperands; ++t) {
    void *allocated = dev[t], *aligned = dev[t];
    uint64_t offset = 0;
    uint64_t n0 = kRows, n1 = kCols, s0 = kCols, s1 = 1;
    uint64_t base = t * 7;
    L0_CHECK(
        zeKernelSetArgumentValue(kernel, base + 0, sizeof(void *), &allocated));
    L0_CHECK(
        zeKernelSetArgumentValue(kernel, base + 1, sizeof(void *), &aligned));
    L0_CHECK(
        zeKernelSetArgumentValue(kernel, base + 2, sizeof(uint64_t), &offset));
    L0_CHECK(zeKernelSetArgumentValue(kernel, base + 3, sizeof(uint64_t), &n0));
    L0_CHECK(zeKernelSetArgumentValue(kernel, base + 4, sizeof(uint64_t), &n1));
    L0_CHECK(zeKernelSetArgumentValue(kernel, base + 5, sizeof(uint64_t), &s0));
    L0_CHECK(zeKernelSetArgumentValue(kernel, base + 6, sizeof(uint64_t), &s1));
  }

  ze_group_count_t groups = {kGroups, 1, 1};
  auto run = [&]() {
    L0_CHECK(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0,
                                             nullptr));
    L0_CHECK(zeCommandListClose(list));
    L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    L0_CHECK(zeCommandListReset(list));
  };

  const double best = bestOf(run);

  L0_CHECK(zeCommandListAppendMemoryCopy(list, o.data(), dev[3], kBytes,
                                         nullptr, 0, nullptr));
  L0_CHECK(zeCommandListClose(list));
  L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  L0_CHECK(zeCommandListReset(list));

  const uint64_t wrong = countWrong(o, expected);
  reportResult(wrong);
  std::printf("kernel time (Level Zero directly): best of %d after %d warm-up "
              "runs = %.4f ms\n",
              kReps, kWarmup, best);
  return wrong;
}

/// Run the payload through the dispatch library's entry points.
///
/// The allocation, the dispatch and the read-back are the ABI a compiled
/// program calls. The sabotage modes change what the library is handed, so the
/// checks below can fail: a corrupted image, a wrong parameter count and a
/// changed argument.
static uint64_t runLibrary(std::vector<char> &payload, const char *kernelName,
                           const char *mode) {
  const bool wrongCount = std::strcmp(mode, "wrong-param-count") == 0;
  const bool shiftBase = std::strcmp(mode, "shift-base") == 0;

  std::vector<float> a, b, c, expected;
  fillReference(a, b, c, expected);

  // Every tensor goes through the library's allocator, in the GPU band's
  // topology -- the id the compiler emits for a transfer into
  // `Memory::GPU_HBM`. The library hands back memory the device can address and
  // the host can read, so the read-back below is a copy and not a decode.
  void *dev[kOperands];
  dev[0] = vx_plugin_alloc_and_transfer(kBytes, a.data(), kGpuTopology,
                                        VX_SPACE_ACCESS_HOST_READABLE);
  dev[1] = vx_plugin_alloc_and_transfer(kBytes, b.data(), kGpuTopology,
                                        VX_SPACE_ACCESS_HOST_READABLE);
  dev[2] = vx_plugin_alloc_and_transfer(kBytes, c.data(), kGpuTopology,
                                        VX_SPACE_ACCESS_HOST_READABLE);
  // The result buffer starts with nothing in it; the kernel writes all of it.
  dev[3] = vx_plugin_alloc_and_transfer(kBytes, nullptr, kGpuTopology,
                                        VX_SPACE_ACCESS_HOST_READABLE);
  for (uint64_t t = 0; t < kOperands; ++t)
    if (!dev[t]) {
      std::fprintf(stderr, "the library did not allocate tensor %llu\n",
                   (unsigned long long)t);
      std::exit(1);
    }

  // One descriptor per tensor, in the layout the library reads, and one word
  // per argument holding that descriptor's address: the word is what
  // `device_args[i]` points at.
  struct Descriptor {
    void *allocated;
    void *aligned;
    int64_t offset;
    int64_t sizes[2];
    int64_t strides[2];
  };
  Descriptor desc[kOperands] = {};
  void *descPtr[kOperands];
  void *deviceArgs[kOperands];
  int32_t tags[kOperands];
  for (uint64_t t = 0; t < kOperands; ++t) {
    desc[t].allocated = dev[t];
    desc[t].aligned = dev[t];
    desc[t].offset = 0;
    desc[t].sizes[0] = kRows;
    desc[t].sizes[1] = kCols;
    desc[t].strides[0] = kCols;
    desc[t].strides[1] = 1;
    descPtr[t] = &desc[t];
    deviceArgs[t] = &descPtr[t];
    tags[t] = VX_ABI_MEMREF_TAG(VX_DTYPE_F32, 2);
  }
  /* The sabotage moves the first tensor's base pointer by one element, which
     is the field the kernel reads its addresses from. The offset field would
     be the other candidate, and it is not: the image this compiler emits
     indexes from the aligned pointer and never loads the offset, so changing
     it changes nothing and the check would pass while proving nothing. */
  if (shiftBase) {
    desc[0].aligned = static_cast<char *>(dev[0]) + sizeof(float);
    std::printf(
        "sabotage: moved the first tensor 4 bytes past its allocation\n");
  }

  // The library caches the loaded module and its kernel, so the first dispatch
  // pays the driver's compile and the timed ones pay the launch.
  auto dispatch = [&]() {
    vx_plugin_dispatch_async(payload.data(), payload.size(), deviceArgs, tags,
                             wrongCount ? (int64_t)(kOperands - 1)
                                        : (int64_t)kOperands);
  };
  if (wrongCount)
    std::printf("sabotage: dispatching with %llu arguments instead of %llu\n",
                (unsigned long long)(kOperands - 1),
                (unsigned long long)kOperands);

  const double best = bestOf(dispatch);

  std::vector<float> got(kElems, 0.0f);
  vx_plugin_transfer_device_to_host(dev[3], got.data(), kBytes, kGpuTopology);

  const uint64_t wrong = countWrong(got, expected);
  reportResult(wrong);
  std::printf("kernel time (through the library): best of %d after %d warm-up "
              "launches = %.4f ms\n",
              kReps, kWarmup, best);

  for (uint64_t t = 0; t < kOperands; ++t)
    vx_plugin_free(dev[t], kGpuTopology);
  return wrong;
}

int main(int argc, char **argv) {
  const char *payloadPath = argc > 1 ? argv[1] : "loop.payload";
  const char *kernelName = argc > 2 ? argv[2] : "vx_npu_kernel_0";
  const char *mode = argc > 3 ? argv[3] : "";

  // The footprint is the four tensors the kernel's arguments describe, and it
  // is a compile-time constant. Checked against the ceiling before anything is
  // allocated, and printed on every run.
  requireWithinCeiling("vx_spirv_run_image", kFootprint, vramCeiling());
  std::printf("footprint: %llu bytes of device memory (%llu tensors of %llu)\n",
              (unsigned long long)kFootprint, (unsigned long long)kOperands,
              (unsigned long long)kBytes);

  std::vector<char> payload = readFile(payloadPath);
  const void *image = nullptr;
  const int64_t imageSize =
      vx_payload_section(payload.data(), payload.size(), &image);
  if (imageSize <= 0 || !image) {
    std::fprintf(stderr, "%s carries no usable image section (code %lld)\n",
                 payloadPath, (long long)imageSize);
    return 1;
  }
  const size_t imageOffset = (const char *)image - payload.data();

  if (std::strcmp(mode, "direct") == 0)
    return runDirect(static_cast<const char *>(image), (size_t)imageSize,
                     kernelName) != 0;

  if (std::strcmp(mode, "corrupt-image") == 0) {
    // Corrupt a copy of the image, then put it back so the library reads the
    // same payload a real dispatch would get, with one byte changed.
    std::vector<char> corrupted(static_cast<const char *>(image),
                                static_cast<const char *>(image) + imageSize);
    if (!corruptRowStride(corrupted)) {
      std::fprintf(stderr, "sabotage: no row-stride constant to corrupt\n");
      return 2;
    }
    std::memcpy(payload.data() + imageOffset, corrupted.data(),
                corrupted.size());
    std::printf(
        "sabotage: flipped one byte of the image's row-stride constant\n");
  } else if (std::strcmp(mode, "bad-format") == 0) {
    if (!rewriteFormat(payload, "metal")) {
      std::fprintf(stderr, "sabotage: no format=spirv entry to change\n");
      return 2;
    }
    std::printf("sabotage: changed format=spirv to format=metal\n");
  }

  return runLibrary(payload, kernelName, mode) != 0;
}
