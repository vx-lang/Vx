// How much device memory a measurement tool may hold, and how a size typed on
// the command line or set in the environment becomes a number.
//
// Every tool in this directory that touches the card includes this rather than
// deciding for itself, so no two helpers disagree about how much of the card
// they may hold. A card asked for more device memory than it can give can take
// the driver down with it, so a helper states a ceiling and refuses before it
// allocates anything.
//
// The ceiling is the operator's, not the card's. `VX_VRAM_CEILING` names it in
// bytes, and the default below is deliberately small so a larger run has to be
// asked for on purpose. How much of a card a deployment may use is a fact about
// the driver and the card -- the driver reports what is available to allocate
// -- and the operator decides how much of that a measurement may hold.
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// The ceiling a tool starts with. A run that wants more has to ask for more by
// setting `VX_VRAM_CEILING`, so the larger allocation is always deliberate.
static constexpr uint64_t kVramCeilingDefault = 2ull * 1024 * 1024 * 1024;

// The most `VX_VRAM_CEILING` may be set to. This is arithmetic and not a fact
// about any card: it is the largest ceiling under which three buffers sized
// from the ceiling still fit a 64-bit count, so no ceiling the variable accepts
// can make a helper's peak arithmetic wrap.
static constexpr uint64_t kVramCeilingMax = UINT64_MAX / 3;

// Parse `text` as a decimal count of `unit` and return it, or print why `name`
// is not usable and exit. An empty string, a `-`, a trailing character, a value
// too large for 64 bits and a value above `max` are all refused, so no caller
// has to guess what a bad argument meant. `unit` names what is counted, so a
// value that is in MiB is not reported as a number of bytes.
static uint64_t parseCount(const char *name, const char *text, const char *unit,
                           uint64_t max) {
  if (text[0] == '\0' || text[0] == '-') {
    std::fprintf(stderr, "%s: \"%s\" is not a decimal number of %s\n", name,
                 text, unit);
    std::exit(1);
  }
  char *end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0' || errno == ERANGE) {
    std::fprintf(stderr, "%s: \"%s\" is not a decimal number of %s\n", name,
                 text, unit);
    std::exit(1);
  }
  if (value > max) {
    std::fprintf(stderr,
                 "%s: \"%s\" is %llu %s, more than the %llu %s allowed\n", name,
                 text, value, unit, (unsigned long long)max, unit);
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
  return parseCount("VX_VRAM_CEILING", env, "bytes", kVramCeilingMax);
}

// The largest per-buffer size in MiB a helper may be asked for when its peak is
// three buffers of that size. It is computed from the ceiling, so `3 * size *
// MiB` is at most the ceiling: a size this cap lets through can neither wrap a
// 64-bit count nor exceed what the operator allowed.
static inline uint64_t vramCeilingMaxSizeMiB() {
  return vramCeiling() / 3 / (1024 * 1024);
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
                 "permit more.\n",
                 tool, (unsigned long long)peak, (unsigned long long)ceiling);
    std::exit(1);
  }
}
