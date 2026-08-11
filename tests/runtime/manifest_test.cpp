//===- manifest_test.cpp - Which machine a worker's name means --*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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
  test_lookup_returns_the_endpoint();
  test_parsing();
  test_malformed_lines_are_refused();
  test_duplicates_are_refused();
  test_bounds();
  test_load_missing_file_is_not_an_error();
  test_load_from_file();

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
