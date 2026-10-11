//===- payload_test.cpp - The dispatch payload's own format ----*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The payload a dispatch carries: the kernel name, then NUL-separated
// `key=value` entries. This checks the two that are about the format itself.
//
// `abi=` is the version, and it exists because the entry list grows. Every
// consumer walks past a key it was not asked for, which is what let `kind=`,
// `roles=`, `topo=` and the rest land one at a time without breaking anything
// -- but that protects a reader from a key it does not know, not from a value
// it misreads. The version is what a dispatch library checks before it trusts
// anything else.
//
// A device image rides in one of two places. PTX is text, so `image=` holds it
// as a NUL-terminated entry like any other. SPIR-V is not: its first word holds
// NUL bytes, so it rides in a section after the entries, introduced by an empty
// entry and counted by a little-endian length. There is no base64 anywhere --
// that would make a text format carry a binary image, which is the wrong shape
// for the problem (docs/gpu_backends.md, "Shared work before a second backend",
// item 2).
//
// Driven by tests/integration_test/payload_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../include/vx_hardware_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

/// A payload laid out the way the compiler lays one out: the kernel name first,
/// then `key=value` entries, each NUL-terminated, ending with the last entry's
/// NUL. An empty `abi` leaves the key out, which is what a producer that
/// predates the version would have written.
std::string
make_payload(const std::string &name, const std::string &abi,
             const std::vector<std::pair<const char *, std::string>> &entries) {
  std::string payload = name;
  payload.push_back('\0');
  if (!abi.empty()) {
    payload += "abi=";
    payload += abi;
    payload.push_back('\0');
  }
  for (const auto &entry : entries) {
    payload += entry.first;
    payload += entry.second;
    payload.push_back('\0');
  }
  return payload;
}

void the_version_key_is_read() {
  // Pinned literally, because this is the number the compiler stamps and every
  // dispatch library checks. The Rust half of this test asserts the compiler
  // writes exactly `abi=1` into a real payload, so the two ends hold each
  // other.
  check(VX_PAYLOAD_ABI == 1, "the payload version this header defines is 1");

  const std::string with_version = make_payload(
      "vx_npu_kernel_0", std::to_string(VX_PAYLOAD_ABI), {{"kind=", "matmul"}});
  check(vx_payload_abi(with_version.data(), with_version.size()) ==
            VX_PAYLOAD_ABI,
        "a payload carrying this header's version reads back as that version");
  check(vx_payload_field(with_version.data(), with_version.size(), "abi=") !=
            nullptr,
        "and the entry is readable as text too");

  const std::string without_version =
      make_payload("vx_npu_kernel_0", "", {{"kind=", "matmul"}});
  check(vx_payload_abi(without_version.data(), without_version.size()) == -1,
        "a producer that predates the key reports absence, not a version");
  check(vx_payload_field(without_version.data(), without_version.size(),
                         "kind=") != nullptr,
        "and its entries are still readable");

  const std::string empty_version =
      make_payload("vx_npu_kernel_0", "", {{"abi=", ""}});
  check(vx_payload_abi(empty_version.data(), empty_version.size()) == -1,
        "an empty version is not version 0");
  check(vx_payload_field(empty_version.data(), empty_version.size(), "abi=") !=
            nullptr,
        "an empty `abi=` is present to a field lookup, not absent");

  const std::string not_a_number =
      make_payload("vx_npu_kernel_0", "1x", {{"kind=", "matmul"}});
  check(vx_payload_abi(not_a_number.data(), not_a_number.size()) == -1,
        "a version that is not a decimal number is refused");
  check(vx_payload_field(not_a_number.data(), not_a_number.size(), "abi=") !=
            nullptr,
        "and a malformed one is still distinct from an absent one");

  const std::string huge =
      make_payload("vx_npu_kernel_0", "99999999999999999999", {});
  check(vx_payload_abi(huge.data(), huge.size()) == -1,
        "a version that overflows is refused rather than wrapped");

  check(vx_payload_abi(nullptr, 0) == -1, "no payload, no version");
}

void entries_still_walk_with_one_more() {
  const std::string payload =
      make_payload("vx_npu_kernel_7", "1",
                   {{"kind=", "matmul"},
                    {"topo=", "500"},
                    {"image=", ".version 7.6\n.target sm_80\n"}});

  const char *kind = vx_payload_field(payload.data(), payload.size(), "kind=");
  check(kind != nullptr && strcmp(kind, "matmul") == 0,
        "a key after the version is still found");
  const char *topo = vx_payload_field(payload.data(), payload.size(), "topo=");
  check(topo != nullptr && strcmp(topo, "500") == 0,
        "and so is one after two others");

  // The name is still the first thing a consumer that ignores all of this
  // reads, which is what makes the whole scheme backward-compatible.
  check(strcmp(payload.c_str(), "vx_npu_kernel_7") == 0,
        "the payload still reads as the kernel name as a C string");

  check(vx_payload_field(payload.data(), payload.size(), "roles=") == nullptr,
        "a key that is not there reports absence");
  check(vx_payload_field(payload.data(), payload.size(), nullptr) == nullptr,
        "no key, no answer");
  check(vx_payload_field(nullptr, 0, "kind=") == nullptr, "no payload");

  // The walk refuses rather than reading past the blob. Truncating the payload
  // so its last entry has no terminator means a lookup that has to walk
  // *through* that entry finds nothing -- it cannot tell absence from
  // truncation, which is the documented answer. A key that appears before it is
  // still found, because the walk stops at the first match and never reaches
  // the damaged tail.
  std::string truncated = payload;
  truncated.resize(truncated.size() - 1);
  check(vx_payload_field(truncated.data(), truncated.size(), "image=") ==
            nullptr,
        "an unterminated entry stops the walk instead of being read past");
  check(vx_payload_field(truncated.data(), truncated.size(), "kind=") !=
            nullptr,
        "a key before the damage is still found");
}

/// A section appended to a text payload: the empty entry that ends the text
/// part, then a little-endian 64-bit length, then the bytes. This is the layout
/// the reader must agree with, written out rather than built by a helper the
/// reader also uses.
std::string with_section(std::string payload, const std::string &image) {
  payload.push_back('\0');
  uint64_t length = image.size();
  for (int i = 0; i < 8; ++i) {
    payload.push_back((char)((length >> (8 * i)) & 0xFF));
  }
  payload += image;
  return payload;
}

void the_section_is_read_back() {
  // A SPIR-V header and then a string that looks exactly like an entry. Both
  // parts of that are deliberate: the header is why the section exists at all
  // (NUL bytes cannot ride in a NUL-terminated entry), and the entry-shaped
  // tail is what a reader that walked into the section would pick up.
  const std::string image =
      std::string("\x03\x02\x23\x07", 4) + "\x00" + std::string("evil=1\0", 7);
  std::string payload = with_section(
      make_payload("vx_npu_kernel_7", "1",
                   {{"format=", "spirv"}, {"topo=", "500"}}),
      image);

  const char *format =
      vx_payload_field(payload.data(), payload.size(), "format=");
  check(format != nullptr && std::string(format) == "spirv",
        "a text entry before the section is still found");
  check(vx_payload_field(payload.data(), payload.size(), "topo=") != nullptr,
        "the last text entry is still found");
  check(vx_payload_field(payload.data(), payload.size(), "evil=") == nullptr,
        "the walk stops at the section instead of reading its bytes");

  const void *out = nullptr;
  int64_t length = vx_payload_section(payload.data(), payload.size(), &out);
  check(length == (int64_t)image.size(), "the section's length");
  check(out != nullptr && memcmp(out, image.data(), image.size()) == 0,
        "the section's bytes, NUL bytes and all");
}

void a_text_only_payload_carries_no_section() {
  // What a PTX payload looks like today: entries and no empty entry, so there
  // is nothing after the text and the section reader says so rather than
  // reading the end of the blob.
  std::string payload = make_payload(
      "vx_npu_kernel_1", "1", {{"format=", "ptx"}, {"image=", ".version 7.6\n"}});
  const void *out = nullptr;
  check(vx_payload_section(payload.data(), payload.size(), &out) == -1,
        "no section in a text-only payload");
  check(vx_payload_text_end(payload.data(), payload.size()) == payload.size(),
        "the text part ends the payload");
}

void refusals() {
  const std::string image = "0123456789";
  std::string payload = with_section(make_payload("k", "1", {}), image);
  const void *out = nullptr;
  const size_t length_at = payload.size() - image.size() - 8;

  // A length that does not account for the rest of the payload describes a
  // truncated or overlapping blob. Both directions are refused, because a
  // reader that trusted either would read past the end or leave bytes unread.
  std::string too_long = payload;
  too_long[length_at] += 1;
  check(vx_payload_section(too_long.data(), too_long.size(), &out) == -2,
        "a length longer than the payload");

  std::string too_short = payload;
  too_short[length_at] -= 1;
  check(vx_payload_section(too_short.data(), too_short.size(), &out) == -2,
        "a length shorter than the payload");

  std::string cut = payload.substr(0, payload.size() - 4);
  check(vx_payload_section(cut.data(), cut.size(), &out) == -2,
        "a payload cut short inside the section");

  check(vx_payload_section(nullptr, 0, &out) == -1, "no payload");
  // A NULL `out` is a caller bug, not a payload shape, so it must not answer
  // the same -1 a text-only payload does: -3 is its own code.
  check(vx_payload_section(payload.data(), payload.size(), nullptr) == -3,
        "nowhere to put the answer is not the same as no section");
}

} // namespace

int main() {
  the_version_key_is_read();
  entries_still_walk_with_one_more();
  the_section_is_read_back();
  a_text_only_payload_carries_no_section();
  refusals();

  if (failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all dispatch payload checks passed\n");
  return 0;
}
