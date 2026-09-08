//===- manifest_test.cpp - Which machine a worker's name means --*- C++ -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Exercises runtime/vx_manifest.h, which is the last step of turning
// `toponame=DecodeWorker` into a machine (#348).
//
// The rule under test is the one the location-transparency claim rests on: a
// name that is absent means *local*. A program with no manifest runs entirely
// on this machine, and adding an entry is what distributes it -- which is why
// `llama2.vx` is the same text either way.
//
// That rule also decides what a malformed line has to do. Skipping one would
// leave a worker silently local: the program runs, the output is right, and the
// distribution it was supposed to demonstrate did not happen. There is no error
// to notice afterwards, which is why the parse refuses instead.
//
// Driven by tests/integration_test/manifest_test.rs.
//
//===----------------------------------------------------------------------===//

#include "../../runtime/vx_manifest.h"

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

void test_absent_means_local() {
  vx_manifest m;
  vx_manifest_init(&m);

  check(vx_manifest_find(&m, "DecodeWorker") == NULL,
        "an empty manifest makes every worker local");

  check(vx_manifest_add(&m, "DecodeWorker", "10.0.0.5", 9001) == 1,
        "a worker can be named");
  check(vx_manifest_find(&m, "DecodeWorker") != NULL, "and is then found");
  check(vx_manifest_find(&m, "PrefillWorker") == NULL,
        "while a worker that was not named stays local");

  // Names are exact. A near-miss must not resolve, because resolving it would
  // send a dispatch to a machine the program did not name.
  check(vx_manifest_find(&m, "decodeworker") == NULL, "lookup is case exact");
  check(vx_manifest_find(&m, "DecodeWorker ") == NULL, "and space exact");
  check(vx_manifest_find(&m, "Decode") == NULL, "and not a prefix match");
  check(vx_manifest_find(&m, NULL) == NULL, "a null name is local");
}

// The three outcomes of resolving a placement, and why the middle one has a
// name. `vx_manifest_find` returning NULL cannot distinguish "no manifest, so
// everything is local" from "a manifest exists and forgot this one" -- and the
// second is how a distributed run becomes a local one in silence. The fleet
// demo named its workers `PrefillWorker`/`DecodeWorker` while the program
// dispatched to `GPU[0]`/`GPU[1]`; every dispatch fell through to local and the
// run still reported two GPUs and identical output.
void test_classify_separates_unlisted_from_local() {
  vx_manifest empty;
  vx_manifest_init(&empty);
  const vx_manifest_entry *w = (const vx_manifest_entry *)1;
  check(vx_manifest_classify(&empty, "GPU[0]", 500, &w) == VX_PLACED_LOCAL,
        "an empty manifest is a single-machine run, not a mismatch");
  check(w == NULL, "and yields no worker");

  vx_manifest m;
  vx_manifest_init(&m);
  vx_manifest_add(&m, "DecodeWorker", "10.0.0.5", 9002);

  check(vx_manifest_classify(&m, "GPU[0]", 500, &w) == VX_PLACED_UNLISTED,
        "a populated manifest that does not name the placement is unlisted");
  check(w == NULL, "and still yields no worker");

  check(vx_manifest_classify(&m, "DecodeWorker", 0, &w) == VX_PLACED_REMOTE,
        "a named placement resolves");
  check(w != NULL && strcmp(w->host, "10.0.0.5") == 0,
        "to the endpoint the manifest gave it");

  // The name is the identity and the id is a hash of it, so the name wins and
  // the id is only the fallback for a producer that sent no name.
  const vx_manifest_entry *d = vx_manifest_find(&m, "DecodeWorker");
  check(vx_manifest_classify(&m, NULL, d->dispatch_id, &w) == VX_PLACED_REMOTE,
        "the id alone still resolves");
  check(w == d, "to the same entry");

  // A null manifest is local rather than a crash: a backend may ask before one
  // has been loaded.
  check(vx_manifest_classify(NULL, "GPU[0]", 500, &w) == VX_PLACED_LOCAL,
        "no manifest at all is local");
}

void test_lookup_returns_the_endpoint() {
  vx_manifest m;
  vx_manifest_init(&m);
  vx_manifest_add(&m, "PrefillWorker", "10.0.0.4", 9001);
  vx_manifest_add(&m, "DecodeWorker", "10.0.0.5", 9002);

  const vx_manifest_entry *p = vx_manifest_find(&m, "PrefillWorker");
  const vx_manifest_entry *d = vx_manifest_find(&m, "DecodeWorker");
  check(p && strcmp(p->host, "10.0.0.4") == 0 && p->port == 9001,
        "prefill resolves to its own endpoint");
  check(d && strcmp(d->host, "10.0.0.5") == 0 && d->port == 9002,
        "decode resolves to its own endpoint");
  check(p != d, "and they are different entries");
}

void test_parsing() {
  vx_manifest m;
  vx_manifest_init(&m);

  check(vx_manifest_parse_line(&m, "\n") == 0, "a blank line yields nothing");
  check(vx_manifest_parse_line(&m, "   \t \n") == 0, "whitespace likewise");
  check(vx_manifest_parse_line(&m, "# a comment\n") == 0, "a comment likewise");
  check(m.count == 0, "and none of them added an entry");

  check(vx_manifest_parse_line(&m, "DecodeWorker 10.0.0.5 9001\n") == 1,
        "a well-formed line parses");
  check(vx_manifest_parse_line(&m, "  PrefillWorker\t10.0.0.4\t9002  \n") == 1,
        "tabs and surrounding space are fine");
  check(vx_manifest_parse_line(&m,
                               "Third 10.0.0.6 9003 # trailing comment\n") == 1,
        "a trailing comment is stripped, not treated as a field");
  check(m.count == 3, "three workers");

  const vx_manifest_entry *e = vx_manifest_find(&m, "Third");
  check(e && e->port == 9003, "and the one with a comment kept its port");
}

// Every one of these would otherwise leave a worker quietly local.
void test_malformed_lines_are_refused() {
  vx_manifest m;
  vx_manifest_init(&m);

  check(vx_manifest_parse_line(&m, "DecodeWorker\n") < 0,
        "a name with no host is refused");
  check(vx_manifest_parse_line(&m, "DecodeWorker 10.0.0.5\n") < 0,
        "a host with no port is refused");
  check(vx_manifest_parse_line(&m, "DecodeWorker 10.0.0.5 9001 extra\n") < 0,
        "a fourth field is refused rather than ignored");
  check(vx_manifest_parse_line(&m, "DecodeWorker 10.0.0.5 nine\n") < 0,
        "a port that is not a number is refused");
  check(vx_manifest_parse_line(&m, "DecodeWorker 10.0.0.5 9001x\n") < 0,
        "and one with trailing rubbish too");
  check(vx_manifest_parse_line(&m, "DecodeWorker 10.0.0.5 0\n") < 0,
        "port 0 is refused");
  check(vx_manifest_parse_line(&m, "DecodeWorker 10.0.0.5 65536\n") < 0,
        "and a port past the range");
  check(vx_manifest_parse_line(&m, "DecodeWorker 10.0.0.5 -1\n") < 0,
        "and a negative one");
  check(m.count == 0, "no malformed line left an entry behind");
}

// Two lines naming one worker disagree about where it is, and whichever won
// would be an accident of ordering.
void test_duplicates_are_refused() {
  vx_manifest m;
  vx_manifest_init(&m);
  check(vx_manifest_parse_line(&m, "DecodeWorker 10.0.0.5 9001\n") == 1,
        "the first entry is taken");
  check(vx_manifest_parse_line(&m, "DecodeWorker 10.0.0.9 9001\n") < 0,
        "a second entry for the same worker is refused");
  const vx_manifest_entry *e = vx_manifest_find(&m, "DecodeWorker");
  check(e && strcmp(e->host, "10.0.0.5") == 0,
        "and the first is not overwritten");
}

void test_bounds() {
  vx_manifest m;
  char name[VX_MANIFEST_MAX_NAME + 8];
  char host[VX_MANIFEST_MAX_HOST + 8];
  vx_manifest_init(&m);

  memset(name, 'a', sizeof(name) - 1);
  name[sizeof(name) - 1] = '\0';
  memset(host, 'b', sizeof(host) - 1);
  host[sizeof(host) - 1] = '\0';

  check(vx_manifest_add(&m, name, "10.0.0.5", 9001) == 0,
        "an over-long name is refused rather than truncated");
  check(vx_manifest_add(&m, "DecodeWorker", host, 9001) == 0,
        "an over-long host likewise");
  check(vx_manifest_add(&m, "", "10.0.0.5", 9001) == 0, "an empty name too");
  check(vx_manifest_add(&m, "DecodeWorker", "", 9001) == 0,
        "an empty host too");

  /* Truncation is what makes this matter: a name silently cut to fit would
     match a *different* worker's dispatch. */
  for (int i = 0; i < VX_MANIFEST_MAX_WORKERS; ++i) {
    char n[32];
    snprintf(n, sizeof(n), "w%d", i);
    check(vx_manifest_add(&m, n, "10.0.0.1", 9000 + i) == 1, "the table fills");
  }
  check(vx_manifest_add(&m, "one_too_many", "10.0.0.1", 9999) == 0,
        "and then refuses rather than overwriting");
}

// The C mirror of the compiler's fnv_dispatch_id must agree with it exactly.
//
// These constants are not arbitrary: they are what src/arch.rs computes for
// these names, cross-checked against `vxc --action emit-mlir` output, and
// PrefillWorker/DecodeWorker are the same pair
// tests/optimizations/pass/topology_name_in_payload.vx pins on the compiler
// side. A divergence between the two implementations does not fail to build --
// it routes a dispatch to the wrong machine, or to none.
void test_dispatch_ids_match_the_compiler() {
  check(vx_manifest_dispatch_id("PrefillWorker") == 1180846465,
        "PrefillWorker -> 1180846465");
  check(vx_manifest_dispatch_id("DecodeWorker") == 702233669,
        "DecodeWorker -> 702233669");
  check(vx_manifest_dispatch_id("Node") == 879234789, "Node -> 879234789");
  check(vx_manifest_dispatch_id("Device") == 1036766083,
        "Device -> 1036766083");
  check(vx_manifest_dispatch_id("MyTPU") == 192274502, "MyTPU -> 192274502");
  check(vx_manifest_dispatch_id("AcmeCore") == 1926585836,
        "AcmeCore -> 1926585836");

  // Built-in spellings, which are what a real program mostly places on and are
  // *not* hashes of anything. llama2.vx uses GPU[0] and GPU[1]; a manifest that
  // understood only declared names would resolve its dispatches by `toponame=`
  // and then fail to route the allocations, which are given an id and never a
  // name -- so the weights would stay here while the dispatches went elsewhere.
  //
  // Every one of these was cross-checked against `vxc --action emit-mlir`.
  check(vx_manifest_dispatch_id("GPU[0]") == 500, "GPU[0] -> 500");
  check(vx_manifest_dispatch_id("GPU[1]") == 501, "GPU[1] -> 501");
  check(vx_manifest_dispatch_id("GPU[7]") == 507, "GPU[7] -> 507");
  check(vx_manifest_dispatch_id("NPU[3]") == 103, "NPU[3] -> 103");
  check(vx_manifest_dispatch_id("AccCore[2]") == 202, "AccCore[2] -> 202");
  check(vx_manifest_dispatch_id("ANE") == 400, "ANE -> 400");
  check(vx_manifest_dispatch_id("AMX") == 300, "AMX -> 300");
  check(vx_manifest_dispatch_id("CPU") == 0, "CPU -> 0");
  check(vx_manifest_dispatch_id("CpuNeon") == 700, "CpuNeon -> 700");
  check(vx_manifest_dispatch_id("CpuAvx512") == 600, "CpuAvx512 -> 600");

  // A declared name must not collide with a built-in band.
  check(vx_manifest_dispatch_id("DecodeWorker") >= VX_MANIFEST_CUSTOM_ID_BASE,
        "a declared name stays in the hashed range");

  // And the range: declared names start at 3000, which is what keeps them clear
  // of the built-in kinds below and of slices (2000..2999).
  check(vx_manifest_dispatch_id("") >= VX_MANIFEST_CUSTOM_ID_BASE,
        "even an empty name lands in the declared range");

  vx_manifest m;
  vx_manifest_init(&m);
  vx_manifest_add(&m, "DecodeWorker", "10.0.0.5", 9001);
  check(vx_manifest_find_by_id(&m, 702233669) != NULL, "lookup by id finds it");
  check(vx_manifest_find_by_id(&m, 1180846465) == NULL,
        "and an id nobody claimed is local");
}

void test_load_missing_file_is_not_an_error() {
  vx_manifest m;
  check(vx_manifest_load(&m, "/nonexistent/vx-manifest") == 0,
        "a missing manifest is not an error");
  check(m.count == 0, "and leaves everything local");
  check(vx_manifest_load(&m, NULL) == 0, "nor is no path at all");
  check(vx_manifest_load(&m, "") == 0, "nor an empty one");
}

void test_load_from_file() {
  const char *path = "/tmp/vx_manifest_test.txt";
  FILE *f = fopen(path, "w");
  if (!f) {
    check(false, "the temporary manifest can be written");
    return;
  }
  fprintf(f, "# the two workers a disaggregated llama uses\n");
  fprintf(f, "PrefillWorker  10.0.0.4  9001\n");
  fprintf(f, "\n");
  fprintf(f, "DecodeWorker   10.0.0.5  9001\n");
  fclose(f);

  vx_manifest m;
  check(vx_manifest_load(&m, path) == 1, "a well-formed manifest loads");
  check(m.count == 2, "with both workers");
  check(vx_manifest_find(&m, "PrefillWorker") != NULL, "prefill is there");
  check(vx_manifest_find(&m, "DecodeWorker") != NULL, "decode is there");
  check(vx_manifest_find(&m, "SomeOtherWorker") == NULL,
        "and anything else is local");

  /* One bad line fails the whole file, rather than loading the good lines and
     leaving one worker mysteriously local. */
  f = fopen(path, "w");
  fprintf(f, "PrefillWorker 10.0.0.4 9001\n");
  fprintf(f, "DecodeWorker 10.0.0.5 not-a-port\n");
  fclose(f);
  check(vx_manifest_load(&m, path) < 0, "one bad line fails the load");

  remove(path);
}

} // namespace

int main() {
  test_absent_means_local();
  test_classify_separates_unlisted_from_local();
  test_lookup_returns_the_endpoint();
  test_parsing();
  test_malformed_lines_are_refused();
  test_duplicates_are_refused();
  test_bounds();
  test_dispatch_ids_match_the_compiler();
  test_load_missing_file_is_not_an_error();
  test_load_from_file();

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
