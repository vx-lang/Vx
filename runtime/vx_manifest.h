//===- vx_manifest.h - Which machine a worker's name means ------*- C++ -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The last step of resolving a placement to a machine: `toponame=DecodeWorker`
// in a dispatch payload, and a file that says DecodeWorker is a host and a
// port.
//
// This is where the compiler stops. A program names a *role*, the machine file
// says what that role is made of, and this says where it lives -- so an address
// never appears in program text, in a machine model, or in the compiler, and
// moving a worker is editing one line of deployment configuration (#348).
//
// **A name that is not in the manifest is local.** That is the rule the whole
// location-transparency claim rests on, and it is a default rather than an
// error on purpose: a program with no manifest at all runs entirely on this
// machine, which is what makes `llama2.vx` the same text whether it is
// disaggregated or not. Adding a manifest entry is what distributes it.
//
// Vendor-free and header-only, so parsing is tested on any machine
// (tests/runtime/manifest_test.cpp).
//
// Format, one worker per line, `#` to end of line is a comment:
//
//   # name          host            port
//   PrefillWorker   10.0.0.4        9001
//   DecodeWorker    10.0.0.5        9001
//
//===----------------------------------------------------------------------===//

#ifndef VX_MANIFEST_H
#define VX_MANIFEST_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VX_MANIFEST_MAX_WORKERS 32
#define VX_MANIFEST_MAX_NAME 64
#define VX_MANIFEST_MAX_HOST 128

typedef struct {
  char name[VX_MANIFEST_MAX_NAME];
  char host[VX_MANIFEST_MAX_HOST];
  int port;
} vx_manifest_entry;

typedef struct {
  vx_manifest_entry workers[VX_MANIFEST_MAX_WORKERS];
  size_t count;
} vx_manifest;

static inline void vx_manifest_init(vx_manifest *m) { m->count = 0; }

/// Add one worker. Returns 0 if the table is full, a field is too long, the
/// port is out of range, or the name is already present.
///
/// A duplicate is refused rather than overwriting or being appended, because
/// the two entries would disagree about where a worker is and whichever won
/// would be an accident of ordering. Deployment configuration that contradicts
/// itself should be fixed, not resolved.
static inline int vx_manifest_add(vx_manifest *m, const char *name,
                                  const char *host, int port) {
  size_t i;
  if (!name || !host || m->count == VX_MANIFEST_MAX_WORKERS) {
    return 0;
  }
  if (strlen(name) == 0 || strlen(name) >= VX_MANIFEST_MAX_NAME ||
      strlen(host) == 0 || strlen(host) >= VX_MANIFEST_MAX_HOST) {
    return 0;
  }
  if (port <= 0 || port > 65535) {
    return 0;
  }
  for (i = 0; i < m->count; ++i) {
    if (strcmp(m->workers[i].name, name) == 0) {
      return 0;
    }
  }
  snprintf(m->workers[m->count].name, VX_MANIFEST_MAX_NAME, "%s", name);
  snprintf(m->workers[m->count].host, VX_MANIFEST_MAX_HOST, "%s", host);
  m->workers[m->count].port = port;
  ++m->count;
  return 1;
}

/// Look a worker up by the name a dispatch carried.
///
/// Returns NULL when the name is absent, which means *local* rather than
/// *unknown*. A caller distinguishing the two would be inventing a distinction
/// the format deliberately does not make.
static inline const vx_manifest_entry *vx_manifest_find(const vx_manifest *m,
                                                        const char *name) {
  size_t i;
  if (!name) {
    return NULL;
  }
  for (i = 0; i < m->count; ++i) {
    if (strcmp(m->workers[i].name, name) == 0) {
      return &m->workers[i];
    }
  }
  return NULL;
}

/// Parse one line. Returns 1 if it produced an entry, 0 if it was blank or a
/// comment, and -1 if it was meant to be an entry and could not be read.
///
/// A malformed line is an error rather than something to skip. A manifest is
/// deployment configuration, and the failure mode of ignoring a line one cannot
/// read is a worker that silently stays local -- the program runs, produces
/// correct output, and the distribution it was supposed to demonstrate did not
/// happen.
static inline int vx_manifest_parse_line(vx_manifest *m, const char *line) {
  char buf[512];
  char *p, *name, *host, *port_s;
  char *hash;
  long port;
  char *end;

  if (strlen(line) >= sizeof(buf)) {
    return -1;
  }
  snprintf(buf, sizeof(buf), "%s", line);

  hash = strchr(buf, '#');
  if (hash) {
    *hash = '\0';
  }

  p = buf;
  while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
    ++p;
  }
  if (*p == '\0') {
    return 0;
  }

  name = strtok(p, " \t\r\n");
  host = strtok(NULL, " \t\r\n");
  port_s = strtok(NULL, " \t\r\n");
  if (!name || !host || !port_s) {
    return -1;
  }
  if (strtok(NULL, " \t\r\n") != NULL) {
    return -1; /* a fourth field is something this format does not mean */
  }

  port = strtol(port_s, &end, 10);
  if (*end != '\0' || port <= 0 || port > 65535) {
    return -1;
  }

  return vx_manifest_add(m, name, host, (int)port) ? 1 : -1;
}

/// Read a manifest file. Returns 1 on success, 0 if the file cannot be opened,
/// -1 if any line is malformed.
///
/// A file that cannot be opened is not an error here: the caller asked whether
/// there is a manifest, and there is not, so everything is local. A file that
/// exists and is wrong *is* an error, because someone meant something by it.
static inline int vx_manifest_load(vx_manifest *m, const char *path) {
  char line[512];
  FILE *f;

  vx_manifest_init(m);
  if (!path || path[0] == '\0') {
    return 0;
  }
  f = fopen(path, "r");
  if (!f) {
    return 0;
  }
  while (fgets(line, sizeof(line), f)) {
    if (vx_manifest_parse_line(m, line) < 0) {
      fclose(f);
      return -1;
    }
  }
  fclose(f);
  return 1;
}

#endif /* VX_MANIFEST_H */
