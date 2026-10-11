// How much device memory a measurement tool may hold, and how a size typed on
// the command line or set in the environment becomes a number.
//
// Every tool in this directory that touches the card includes this rather than
// deciding for itself, because the failure it prevents is the machine's: the
// owner's machine becomes unstable above 12 GiB of VRAM in use, and a run that
// asks for more can take the driver down with it.
//
// Two mistakes are worth refusing by name. A size that is not a number used to
// become 0 through `strtoull`, and the run then failed a length check that
// never mentioned the argument that caused it. A size that is a number can
// still be larger than the card can safely give.
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// The ceiling a tool starts with. A run that wants more has to ask for more by
// setting `VX_VRAM_CEILING`, so the larger allocation is always deliberate.
static constexpr uint64_t kVramCeilingDefault = 2ull * 1024 * 1024 * 1024;

// The most `VX_VRAM_CEILING` may be set to. This machine is unstable above
// 12 GiB of VRAM in use, so the variable cannot be used to walk past that
// point: a value above this is refused instead of honoured.
static constexpr uint64_t kVramCeilingMax = 12ull * 1024 * 1024 * 1024;

// Parse `text` as a decimal byte count and return it, or print why `name` is
// not usable and exit. An empty string, a `-`, a trailing character, a value
// too large for 64 bits and a value above `max` are all refused, so no caller
// has to guess what a bad argument meant.
static uint64_t parseByteCount(const char *name, const char *text,
                               uint64_t max) {
  if (text[0] == '\0' || text[0] == '-') {
    std::fprintf(stderr, "%s: \"%s\" is not a decimal number of bytes\n", name,
                 text);
    std::exit(1);
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || errno == ERANGE) {
    std::fprintf(stderr, "%s: \"%s\" is not a decimal number of bytes\n", name,
                 text);
    std::exit(1);
  }
  if (value > max) {
    std::fprintf(stderr,
                 "%s: \"%s\" is %llu bytes, more than the %llu bytes "
                 "allowed\n",
                 name, text, value, (unsigned long long)max);
    std::exit(1);
  }
  return value;
}

// The VRAM ceiling in bytes: `VX_VRAM_CEILING` when it is set, else
// `kVramCeilingDefault`. The variable cannot raise the ceiling past
// `kVramCeilingMax`.
static uint64_t vramCeiling() {
  const char *env = std::getenv("VX_VRAM_CEILING");
  if (!env || env[0] == '\0')
    return kVramCeilingDefault;
  return parseByteCount("VX_VRAM_CEILING", env, kVramCeilingMax);
}

// Print the ceiling and the peak this run needs, then refuse if the peak is
// over. `peak` is the most device memory the run holds at one time, which is
// what the ceiling is about -- not the sum over the run.
//
// The refusal names the tool, the bytes asked for and the bytes permitted, and
// says how to permit more. A requested size is never clamped down to fit: a
// measurement that silently ran at a size other than the one asked for would
// report a number for the wrong thing.
static void requireWithinCeiling(const char *tool, uint64_t peak,
                                 uint64_t ceiling) {
  const uint64_t mib = 1024 * 1024;
  // A helper whose footprint is a few hundred bytes would print "0 MiB" and
  // read like a bug, so a size below a mebibyte is printed in bytes.
  if (ceiling >= mib)
    std::printf("ceiling: %llu MiB%s\n", (unsigned long long)(ceiling / mib),
                std::getenv("VX_VRAM_CEILING") ? " (VX_VRAM_CEILING)"
                                               : " (default)");
  else
    std::printf("ceiling: %llu bytes%s\n", (unsigned long long)ceiling,
                std::getenv("VX_VRAM_CEILING") ? " (VX_VRAM_CEILING)"
                                               : " (default)");
  if (peak >= mib)
    std::printf("peak: %llu MiB of device memory at once\n",
                (unsigned long long)(peak / mib));
  else
    std::printf("peak: %llu bytes of device memory at once\n",
                (unsigned long long)peak);
  if (peak > ceiling) {
    std::fflush(stdout);
    std::fprintf(stderr,
                 "%s: refusing to allocate %llu bytes (requested); the VRAM "
                 "ceiling is %llu bytes (permitted). Raise VX_VRAM_CEILING to "
                 "at most %llu bytes deliberately.\n",
                 tool, (unsigned long long)peak, (unsigned long long)ceiling,
                 (unsigned long long)kVramCeilingMax);
    std::exit(1);
  }
}
