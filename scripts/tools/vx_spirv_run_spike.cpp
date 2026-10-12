// Run a Vx-compiled SPIR-V kernel on an Intel GPU through Level Zero.
//
// `vxc prog.vx --emit-llvm` puts the device image in the dispatch payload; the
// shell driver beside this file extracts it and hands it here. This program
// loads the image, allocates the tensors the kernel needs on the device,
// copies them in, launches the kernel with the flattened argument list in ABI
// order, copies the result back, and compares it with the same computation on
// the CPU.
//
// The kernel is `tests/backend/pass/spirv_device_image_loop_kernel.vx`:
//
//   o[i][j] = a[i][j] * b[i][j] + c[i][j]      for 8 rows of 4
//
// so it has four rank-2 operands. The ABI gives each operand seven separate
// kernel arguments -- the allocation's base pointer, the aligned base pointer,
// the offset, two sizes and two strides -- which is 28 arguments, and
// `spirv-dis` on the image shows exactly 28 `OpFunctionParameter`s. Nothing is
// passed as an aggregate.
//
// Memory. The card is shared and this machine is unstable above 12 GiB of VRAM
// in use, so this prints its footprint (four tensors, 512 bytes) and there is
// no way for it to grow: the sizes are compile-time constants of this fixture.
//
// Build and run: scripts/tools/vx_spirv_run_spike.sh

#include <level_zero/ze_api.h>

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

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "loop.spv";
  const char *kernelName = argc > 2 ? argv[2] : "vx_npu_kernel_0";
  const char *mode = argc > 3 ? argv[3] : "";
  const bool shiftOffset = std::strcmp(mode, "shift-offset") == 0;
  const bool corruptImage = std::strcmp(mode, "corrupt-image") == 0;

  std::vector<char> image = readFile(spvPath);
  if (image.size() < 4 ||
      std::memcmp(image.data(), "\x03\x02\x23\x07", 4) != 0) {
    std::fprintf(stderr, "%s is not a SPIR-V module\n", spvPath);
    return 1;
  }
  if (corruptImage) {
    if (!corruptRowStride(image)) {
      std::fprintf(stderr, "sabotage: no row-stride constant to corrupt\n");
      return 2;
    }
    std::printf(
        "sabotage: flipped one byte of the image's row-stride constant\n");
  }

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
        std::printf("device: %s  (%u compute units)\n", props.name,
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
    return 1;
  }

  std::printf("footprint: %llu bytes of device memory (%llu tensors of %llu)\n",
              (unsigned long long)kFootprint, (unsigned long long)kOperands,
              (unsigned long long)kBytes);

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
  moduleDesc.inputSize = image.size();
  moduleDesc.pInputModule = reinterpret_cast<const uint8_t *>(image.data());
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
    return 1;
  }

  ze_kernel_desc_t kernelDesc = {};
  kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  kernelDesc.pKernelName = kernelName;
  ze_kernel_handle_t kernel = nullptr;
  L0_CHECK(zeKernelCreate(module, &kernelDesc, &kernel));
  L0_CHECK(zeKernelSetGroupSize(kernel, kWorkgroup, 1, 1));

  // The host's view of the tensors: values that vary along both axes, so no
  // operand can be dropped or swapped without changing the result.
  std::vector<float> a(kElems), b(kElems), c(kElems), o(kElems, 0.0f);
  for (uint64_t i = 0; i < kRows; ++i)
    for (uint64_t j = 0; j < kCols; ++j) {
      a[i * kCols + j] = float(i + j);
      b[i * kCols + j] = 2.0f;
      c[i * kCols + j] = float(j + 2);
    }
  // The same formula on the CPU. Every value here is a small integer, which
  // fp32 holds exactly, so the check is equality and no tolerance is needed:
  // any difference is the kernel, not rounding.
  std::vector<float> expected(kElems);
  for (uint64_t i = 0; i < kElems; ++i)
    expected[i] = a[i] * b[i] + c[i];

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
    uint64_t offset = (shiftOffset && t == 0) ? 1 : 0;
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
  if (shiftOffset)
    std::printf("sabotage: gave the first tensor offset 1 instead of 0\n");

  ze_group_count_t groups = {kGroups, 1, 1};
  auto run = [&]() {
    L0_CHECK(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0,
                                             nullptr));
    L0_CHECK(zeCommandListClose(list));
    L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    L0_CHECK(zeCommandListReset(list));
  };

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

  L0_CHECK(zeCommandListAppendMemoryCopy(list, o.data(), dev[3], kBytes,
                                         nullptr, 0, nullptr));
  L0_CHECK(zeCommandListClose(list));
  L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
  L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
  L0_CHECK(zeCommandListReset(list));

  uint64_t wrong = 0;
  for (uint64_t i = 0; i < kElems; ++i)
    if (o[i] != expected[i]) {
      if (wrong < 4)
        std::printf("  element %llu: got %f, wanted %f\n",
                    (unsigned long long)i, static_cast<double>(o[i]),
                    static_cast<double>(expected[i]));
      ++wrong;
    }
  std::printf("result: %s (%llu of %llu elements differ)\n",
              wrong ? "WRONG" : "correct", (unsigned long long)wrong,
              (unsigned long long)kElems);
  std::printf("kernel time: best of %d after %d warm-up runs = %.4f ms\n",
              kReps, kWarmup, best);
  return wrong ? 1 : 0;
}
