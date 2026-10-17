// Measure the Intel Arc A770's achieved memory bandwidth through Level Zero:
// a device-to-device copy (twice the buffer, one read and one write), a triad
// kernel x = a*y + b*z (three times the buffer, two reads and one write), and
// the two host-transfer directions through pinned host memory. Each runs at a
// size the caches cannot answer for, is timed best-of-several after a warm-up,
// and is checked against the CPU or against the bytes that were written.
//
// Memory: the program prints its footprint and refuses to allocate past the
// ceiling in `vram_ceiling.h` (`VX_VRAM_CEILING`, 2 GiB by default). At the
// default 512 MiB per buffer the peak is the triad phase's 1.5 GiB, so a larger
// run has to be asked for on purpose.
//
// Build and run: scripts/tools/xpu_bandwidth_spike.sh [size in MiB]

#include <level_zero/ze_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "vram_ceiling.h"

#define L0_CHECK(call)                                                         \
  do {                                                                         \
    ze_result_t r_ = (call);                                                   \
    if (r_ != ZE_RESULT_SUCCESS) {                                             \
      std::fprintf(stderr, "%s:%d: %s failed with 0x%x\n", __FILE__, __LINE__, \
                   #call, static_cast<unsigned>(r_));                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static constexpr uint64_t kMiB = 1024 * 1024;
static constexpr uint64_t kWorkgroup = 256;
static constexpr uint64_t kGroups = 4096; // 1M work-items
static constexpr int kWarmup = 2, kReps = 10;
static constexpr float kA = 1.5f, kB = 2.0f;
// The transfer phase's size per buffer, one on the device and one pinned on the
// host. It is part of the peak, because it is allocated while the phase runs.
static constexpr uint64_t kXfer = 256 * kMiB;

static std::vector<char> readFile(const char *path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    std::exit(1);
  }
  return std::vector<char>((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
}

static double secondsSince(const std::chrono::steady_clock::time_point &t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count();
}

int main(int argc, char **argv) {
  const char *spvPath = argc > 1 ? argv[1] : "triad.spv";
  const char *kernelName = argc > 2 ? argv[2] : "triad";
  // The number is MiB per buffer. It is capped at a third of the ceiling, so
  // the three-buffer peak below can neither wrap a 64-bit count nor exceed what
  // the operator allowed, and a size that is not a number is refused rather
  // than read as 0.
  const uint64_t sizeMiB = argc > 3 ? parseCount("size in MiB", argv[3], "MiB",
                                                 vramCeilingMaxSizeMiB())
                                    : 512;
  if (sizeMiB == 0) {
    std::fprintf(stderr, "size in MiB: 0 measures nothing\n");
    return 1;
  }
  const uint64_t bytesPerBuffer = sizeMiB * kMiB;
  const uint64_t elems = bytesPerBuffer / sizeof(float);
  if (elems % 4 != 0) {
    std::fprintf(stderr, "the size must be a multiple of 16 bytes\n");
    return 1;
  }
  // A deliberate fault the run script turns on to prove each check can fail.
  const char *sabotage = argc > 4 ? argv[4] : "";
  const bool shortN = std::strcmp(sabotage, "short-n") == 0;
  const bool skipCopy = std::strcmp(sabotage, "skip-copy") == 0;
  const bool skipReadback = std::strcmp(sabotage, "skip-readback") == 0;
  const bool skipUpload = std::strcmp(sabotage, "skip-upload") == 0;
  const bool skipDownload = std::strcmp(sabotage, "skip-download") == 0;

  // The most device memory this run holds at one time: the triad phase's three
  // buffers, or the transfer phase's one device buffer, whichever is larger.
  // The phases free their buffers before the next phase starts, so this is a
  // peak and not a sum. `sizeMiB` is capped at a third of the ceiling above, so
  // the three-buffer product is at most the ceiling and cannot wrap a 64-bit
  // count; the transfer phase's buffer is a constant, which the ceiling still
  // limits. Checked before `zeInit` and before any allocation.
  const uint64_t triadPeak = 3 * bytesPerBuffer;
  requireWithinCeiling("xpu_bandwidth_spike", std::max(triadPeak, kXfer),
                       vramCeiling());

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
    return 1;
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

  // --- device-to-device copy ------------------------------------------------
  {
    std::printf("copy phase: 2 buffers of %llu MiB = %llu MiB of VRAM\n",
                (unsigned long long)(bytesPerBuffer / kMiB),
                (unsigned long long)(2 * bytesPerBuffer / kMiB));
    void *a = nullptr, *b = nullptr;
    L0_CHECK(
        zeMemAllocDevice(context, &allocDesc, bytesPerBuffer, 8, device, &a));
    L0_CHECK(
        zeMemAllocDevice(context, &allocDesc, bytesPerBuffer, 8, device, &b));

    std::vector<char> host(bytesPerBuffer, 1);
    L0_CHECK(zeCommandListAppendMemoryCopy(list, a, host.data(), bytesPerBuffer,
                                           nullptr, 0, nullptr));
    L0_CHECK(zeCommandListClose(list));
    L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    L0_CHECK(zeCommandListReset(list));

    const char zero = 0;
    auto copy = [&]() {
      // The sabotage fills b with zeroes rather than leaving it alone, so a
      // fresh buffer cannot pass by happening to hold the source's pattern.
      if (skipCopy)
        L0_CHECK(zeCommandListAppendMemoryFill(
            list, b, &zero, 1, bytesPerBuffer, nullptr, 0, nullptr));
      else
        L0_CHECK(zeCommandListAppendMemoryCopy(list, b, a, bytesPerBuffer,
                                               nullptr, 0, nullptr));
      L0_CHECK(zeCommandListClose(list));
      L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
      L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
      L0_CHECK(zeCommandListReset(list));
    };
    for (int i = 0; i < kWarmup; ++i)
      copy();

    double best = 1e30;
    for (int i = 0; i < kReps; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      copy();
      best = std::min(best, secondsSince(t0));
    }

    // Read the destination back and check it matches the source. Clear the
    // staging buffer first: it still holds the source's bytes from the upload
    // above, so a readback that did nothing would leave them and still pass.
    std::fill(host.begin(), host.end(), 0);
    if (!skipReadback)
      L0_CHECK(zeCommandListAppendMemoryCopy(
          list, host.data(), b, bytesPerBuffer, nullptr, 0, nullptr));
    L0_CHECK(zeCommandListClose(list));
    L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    L0_CHECK(zeCommandListReset(list));
    // Every byte came back 1, the pattern the source holds.
    bool ok =
        std::all_of(host.begin(), host.end(), [](char c) { return c == 1; });
    std::printf("  copy: %.3f ms, %.1f GB/s (read+write), %s\n", best * 1e3,
                2.0 * bytesPerBuffer / best / 1e9, ok ? "checked" : "WRONG");
    if (!ok)
      return 1;
    L0_CHECK(zeMemFree(context, a));
    L0_CHECK(zeMemFree(context, b));
  }

  // --- triad ----------------------------------------------------------------
  {
    std::printf("triad phase: 3 buffers of %llu MiB = %llu MiB of VRAM\n",
                (unsigned long long)(bytesPerBuffer / kMiB),
                (unsigned long long)(3 * bytesPerBuffer / kMiB));
    std::vector<char> image = readFile(spvPath);
    if (image.size() < 4 ||
        std::memcmp(image.data(), "\x03\x02\x23\x07", 4) != 0) {
      std::fprintf(stderr, "%s is not a SPIR-V module\n", spvPath);
      return 1;
    }

    void *x = nullptr, *y = nullptr, *z = nullptr;
    L0_CHECK(
        zeMemAllocDevice(context, &allocDesc, bytesPerBuffer, 8, device, &x));
    L0_CHECK(
        zeMemAllocDevice(context, &allocDesc, bytesPerBuffer, 8, device, &y));
    L0_CHECK(
        zeMemAllocDevice(context, &allocDesc, bytesPerBuffer, 8, device, &z));

    std::vector<float> hy(elems), hz(elems), hx(elems, 0.0f);
    for (uint64_t i = 0; i < elems; ++i) {
      hy[i] = float(i % 7) * 0.5f;
      hz[i] = float(i % 5) * 0.25f + 1.0f;
    }
    L0_CHECK(zeCommandListAppendMemoryCopy(list, y, hy.data(), bytesPerBuffer,
                                           nullptr, 0, nullptr));
    L0_CHECK(zeCommandListAppendMemoryCopy(list, z, hz.data(), bytesPerBuffer,
                                           nullptr, 0, nullptr));
    L0_CHECK(zeCommandListAppendMemoryCopy(list, x, hx.data(), bytesPerBuffer,
                                           nullptr, 0, nullptr));
    L0_CHECK(zeCommandListClose(list));
    L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    L0_CHECK(zeCommandListReset(list));

    ze_module_desc_t moduleDesc = {};
    moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.inputSize = image.size();
    moduleDesc.pInputModule = reinterpret_cast<const uint8_t *>(image.data());
    ze_module_handle_t module = nullptr;
    L0_CHECK(zeModuleCreate(context, device, &moduleDesc, &module, nullptr));
    ze_kernel_desc_t kernelDesc = {};
    kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernelDesc.pKernelName = kernelName;
    ze_kernel_handle_t kernel = nullptr;
    L0_CHECK(zeKernelCreate(module, &kernelDesc, &kernel));
    L0_CHECK(zeKernelSetGroupSize(kernel, kWorkgroup, 1, 1));

    // The kernel works one float4 per work-item, so it is told the number of
    // vectors, at a quarter of the element count.
    uint64_t nvec = elems / 4;
    // The sabotage halves the length the kernel is told, so half of x stays
    // zero and the check against the CPU must fail.
    uint64_t n = shortN ? nvec / 2 : nvec;
    L0_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void *), &x));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void *), &y));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(void *), &z));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 3, sizeof(float), &kA));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 4, sizeof(float), &kB));
    L0_CHECK(zeKernelSetArgumentValue(kernel, 5, sizeof(uint64_t), &n));

    ze_group_count_t groups = {kGroups, 1, 1};
    auto launch = [&]() {
      L0_CHECK(zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr,
                                               0, nullptr));
      L0_CHECK(zeCommandListClose(list));
      L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
      L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
      L0_CHECK(zeCommandListReset(list));
    };
    for (int i = 0; i < kWarmup; ++i)
      launch();

    double best = 1e30;
    for (int i = 0; i < kReps; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      launch();
      best = std::min(best, secondsSince(t0));
    }

    L0_CHECK(zeCommandListAppendMemoryCopy(list, hx.data(), x, bytesPerBuffer,
                                           nullptr, 0, nullptr));
    L0_CHECK(zeCommandListClose(list));
    L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
    L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
    L0_CHECK(zeCommandListReset(list));

    // The device may fuse the multiply-add, so the check allows a relative
    // 1e-5 rather than exact equality. The CPU computes a*y + b*z here.
    uint64_t wrong = 0;
    for (uint64_t i = 0; i < elems; ++i) {
      float want = kA * hy[i] + kB * hz[i];
      if (std::fabs(hx[i] - want) > 1e-5f * std::fabs(want)) {
        if (wrong < 4)
          std::printf("  element %llu: got %f, wanted %f\n",
                      (unsigned long long)i, (double)hx[i], (double)want);
        ++wrong;
      }
    }
    std::printf("  triad: %.3f ms, %.1f GB/s (2 reads + 1 write), %s\n",
                best * 1e3, 3.0 * bytesPerBuffer / best / 1e9,
                wrong ? "WRONG" : "checked");
    if (wrong)
      return 1;

    L0_CHECK(zeMemFree(context, x));
    L0_CHECK(zeMemFree(context, y));
    L0_CHECK(zeMemFree(context, z));
  }

  // --- host to device and back ---------------------------------------------
  {
    const uint64_t xfer = kXfer;
    std::printf("transfer phase: 256 MiB pinned host + 256 MiB device\n");
    void *devbuf = nullptr, *hostbuf = nullptr;
    ze_host_mem_alloc_desc_t hostDesc = {};
    hostDesc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
    L0_CHECK(zeMemAllocDevice(context, &allocDesc, xfer, 8, device, &devbuf));
    L0_CHECK(zeMemAllocHost(context, &hostDesc, xfer, 8, &hostbuf));
    std::memset(hostbuf, 3, xfer);

    auto upload = [&]() {
      L0_CHECK(zeCommandListAppendMemoryCopy(list, devbuf, hostbuf, xfer,
                                             nullptr, 0, nullptr));
      L0_CHECK(zeCommandListClose(list));
      L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
      L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
      L0_CHECK(zeCommandListReset(list));
    };
    auto download = [&]() {
      L0_CHECK(zeCommandListAppendMemoryCopy(list, hostbuf, devbuf, xfer,
                                             nullptr, 0, nullptr));
      L0_CHECK(zeCommandListClose(list));
      L0_CHECK(zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
      L0_CHECK(zeCommandQueueSynchronize(queue, UINT64_MAX));
      L0_CHECK(zeCommandListReset(list));
    };
    for (int i = 0; i < kWarmup; ++i) {
      upload();
      download();
    }
    double up = 1e30, down = 1e30;
    for (int i = 0; i < kReps; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      upload();
      up = std::min(up, secondsSince(t0));
      t0 = std::chrono::steady_clock::now();
      download();
      down = std::min(down, secondsSince(t0));
    }

    // The timed loop above is not a check. Both directions are checked here,
    // and the host buffer is cleared before each readback so a download that
    // did nothing cannot pass on the pattern just uploaded. A skipped upload
    // leaves the device holding what the timed loop put there, not nothing.
    auto allBytesAre = [&](char want) {
      const char *begin = static_cast<const char *>(hostbuf);
      return std::all_of(begin, begin + xfer,
                         [want](char c) { return c == want; });
    };
    auto check = [&](char want, bool doUpload, bool doDownload) {
      std::memset(hostbuf, want, xfer);
      if (doUpload)
        upload();
      std::memset(hostbuf, 0, xfer);
      if (doDownload)
        download();
      return allBytesAre(want);
    };
    const bool upOk = check(9, !skipUpload, true);
    const bool downOk = check(4, true, !skipDownload);
    std::printf("  host to device: %.1f GB/s, %s\n", xfer / up / 1e9,
                upOk ? "checked" : "WRONG");
    std::printf("  device to host: %.1f GB/s, %s\n", xfer / down / 1e9,
                downOk ? "checked" : "WRONG");
    if (!upOk || !downOk)
      return 1;
    L0_CHECK(zeMemFree(context, devbuf));
    L0_CHECK(zeMemFree(context, hostbuf));
  }

  return 0;
}
