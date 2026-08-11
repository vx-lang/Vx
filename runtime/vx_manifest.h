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
  /* The dispatch id the compiler derives from this name. Computed here so a
     plugin can answer "is topology 1113 remote?" -- the entry points that
     allocate and free are given an id and never a name, because only a
     dispatch payload carries `toponame=`. */
  int32_t dispatch_id;
} vx_manifest_entry;

typedef struct {
  vx_manifest_entry workers[VX_MANIFEST_MAX_WORKERS];
  size_t count;
} vx_manifest;

static inline void vx_manifest_init(vx_manifest *m) { m->count = 0; }

/// The compiler's `fnv_dispatch_id`, mirrored byte for byte.
///
/// src/arch.rs is the original and this must agree with it exactly: it is what
/// turns a name in this file into the number a dispatch, an allocation and a
/// free all carry. A divergence would not fail to build, it would route to the
/// wrong machine or to none.
static inline uint32_t vx_manifest_fnv32(const char *s) {
  uint32_t hash = 2166136261u;
  for (; *s; ++s) {
    hash ^= (uint32_t)(unsigned char)*s;
    hash *= 16777619u;
  }
  return hash;
}

/// The dispatch id the compiler gives a topology written this way.
///
/// Mirrors `topology_dispatch_id` over `display_name`'s spellings, not only the
/// hashed band. A manifest that understood just declared names could not name
/// the devices a program actually uses most: llama2.vx places on
/// `Topology::GPU[0]` and `GPU[1]`, whose ids are 500 and 501 and are not
/// hashes of anything. Naming them would have looked right, resolved a dispatch
/// by its `toponame=`, and then failed to route the *allocation* -- which is
/// given an id and never a name -- so the weights would have stayed on the host
/// while the dispatches went elsewhere.
static inline int32_t vx_manifest_dispatch_id(const char *name) {
  int index = 0;

  if (strcmp(name, "CPU") == 0 || strcmp(name, "Current") == 0) {
    return 0;
  }
  if (strcmp(name, "AMX") == 0) {
    return 300;
  }
  if (strcmp(name, "ANE") == 0) {
    return 400;
  }
  if (strcmp(name, "CpuAvx512") == 0) {
    return 600;
  }
  if (strcmp(name, "CpuNeon") == 0) {
    return 700;
  }
  if (sscanf(name, "NPU[%d]", &index) == 1) {
    return 100 + index;
  }
  if (sscanf(name, "AccCore[%d]", &index) == 1) {
    return 200 + index;
  }
  if (sscanf(name, "GPU[%d]", &index) == 1) {
    return 500 + index;
  }
  /* A name declared in a machine file, which is the hashed band. Slices
     (2000..2999) are deliberately absent: a slice is an extent of a device
     rather than a machine, and naming one in a manifest would be asking to send
     a dispatch to half a GPU. */
  return 1000 + (int32_t)(vx_manifest_fnv32(name) % 1000u);
}

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
  /* Two names hashing to one id would make a dispatch for either resolve to
     whichever was added first, silently. The band is only a thousand wide, so
     this is a real possibility rather than a theoretical one, and it is
     detectable exactly here -- refusing costs a rename and catching it later
     costs a run on the wrong machine. */
  {
    int32_t id = vx_manifest_dispatch_id(name);
    for (i = 0; i < m->count; ++i) {
      if (m->workers[i].dispatch_id == id) {
        fprintf(stderr,
                "[Vx manifest] '%s' and '%s' both hash to dispatch id %d; "
                "rename one\n",
                m->workers[i].name, name, id);
        return 0;
      }
    }
    m->workers[m->count].dispatch_id = id;
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

/// Look a worker up by the dispatch id a topology resolves to.
///
/// The counterpart of `vx_manifest_find` for the entry points that are handed
/// an id rather than a name -- allocation, free, and a transfer between
/// devices. Same rule: absent means local.
static inline const vx_manifest_entry *
vx_manifest_find_by_id(const vx_manifest *m, int32_t dispatch_id) {
  size_t i;
  for (i = 0; i < m->count; ++i) {
    if (m->workers[i].dispatch_id == dispatch_id) {
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
