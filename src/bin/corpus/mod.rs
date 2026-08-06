//===- corpus/mod.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// The corpus generator for the parallel-frontend measurements (#296). Shared by `parallel_demo`
// and `intern_bench` via `#[path]`, so both measure the same programs -- a demo and a benchmark
// that disagree about the workload produce two numbers that cannot be compared.
//
// It lives in a subdirectory rather than as `src/bin/corpus.rs` because cargo auto-discovers
// `src/bin/*.rs` as binaries; a directory with no `main.rs` is not a target.
//
//===----------------------------------------------------------------------===//
//
// WHAT ACTUALLY CREATES INTERNING PRESSURE
//
// A generic GID is minted in exactly one place: `emit_type_gid`, reached from
// `emit_function_type_gids`, which walks a function's *parameter and return types*. A
// `Type::GenericInstance` found there mints one entry keyed on its resolved argument GIDs.
//
// Three consequences shape everything below.
//
//  1. **Instantiations in function bodies mint nothing.** `let x = wrap<i32>(1);` is checked and
//     monomorphized, but monomorphized functions are appended to modules in
//     `codegen_and_metadata_phase` -- after the type stream has been extracted, and after
//     `compile_pipeline_type_stream` has already returned. A corpus that puts its generics in
//     bodies exercises the type-checker and leaves the interner idle.
//
//  2. **The key is the argument list, not the instantiation.** `intern_generic` and
//     `deduplication_phase` both key on `Vec<TypeId>` of the arguments; the base type lives in
//     words 0-1 of the GID. `Pair<i32>` and `Box<i32>` therefore share one arena entry, and
//     distinct keys come from distinct *argument lists*. That is why `arity` is a knob: over a
//     vocabulary of size V, arity 1 admits V keys and arity 2 admits V^2.
//
//  3. **Only resolvable nominals count as arguments.** `nominal_gid` answers `Some` for structs,
//     enums, scalars and tensors, and `None` for a bare type parameter or a nested
//     `GenericInstance` -- which `filter_map` then drops. So the generator emits only non-nested
//     concrete arguments; anything else would silently intern the empty key and measure nothing.

// Two binaries include this module and each uses a different subset of it -- `intern_bench` sweeps
// with `arg_list`, `parallel_demo` reports `dir`. Neither is dead; `dead_code` just cannot see
// across the two inclusions.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// Everything that determines the generated text. Two corpora with equal parameters are
/// byte-identical, which is what lets a run be reproduced from its log.
#[derive(Debug, Clone, PartialEq)]
pub struct CorpusParams {
    /// N: modules (the measurement plan sweeps 8 / 64 / 512).
    pub modules: usize,
    /// M: functions per module (the plan sweeps 16 / 128).
    pub fns_per_module: usize,
    /// The generics-density knob: the fraction of each function's parameter slots holding a generic
    /// instantiation rather than a plain nominal. 0.0 is the "plain" variant, 1.0 the
    /// "instantiation-dense" one.
    ///
    /// Density changes the *type* in a slot, never the number of slots, so the two variants have
    /// matched function counts, matched signature widths, and matched type-stream lengths. What
    /// differs is whether a slot's GID is settled or interned -- which is the variable under study.
    pub density: f64,
    /// Type parameters on the generated carriers. The key space is `vocabulary^arity`, so this
    /// decides whether the interner's map stays cache-resident.
    pub arity: usize,
    /// Fraction of generic arguments drawn from the shared scalar vocabulary rather than from
    /// module-local structs. Scalars have module-independent GIDs, so a shared argument is a key
    /// two threads can collide on, while a local struct is a key only one thread ever mints. This
    /// is the difference between measuring lock *contention* and measuring lock *overhead*.
    pub shared_frac: f64,
    /// Parameter slots per function; constant across variants by construction.
    pub params_per_fn: usize,
    /// Module-local structs per module: the local half of the argument vocabulary, and the type
    /// used in a plain slot.
    pub locals_per_module: usize,
    pub seed: u64,
}

impl Default for CorpusParams {
    fn default() -> Self {
        Self {
            modules: 64,
            fns_per_module: 16,
            density: 1.0,
            arity: 1,
            shared_frac: 0.5,
            params_per_fn: 4,
            locals_per_module: 8,
            seed: 0x5EED,
        }
    }
}

/// The shared vocabulary: scalars, whose GIDs are module-independent (`scalar_gid`), so every
/// module naming `i32` mints the same key. Deliberately narrow -- a wide shared vocabulary would
/// spread out the collisions the `locked` baseline exists to suffer.
const SHARED_SCALARS: [&str; 6] = ["i32", "i64", "u32", "u64", "f32", "f64"];

/// Generic carriers per module. More than one so a slot picks a base *and* an argument list
/// independently: distinct bases sharing an argument list is precisely the case the arena's key
/// design (arguments only, base in words 0-1) exists to handle.
const CARRIERS_PER_MODULE: usize = 2;

/// Bump whenever the emitted text changes for the same parameters.
///
/// The corpus is cached and reused when its manifest matches, and the manifest is keyed on the
/// parameters -- so without this, changing the generator leaves every cached corpus on disk looking
/// current, and the next run measures the old programs while reporting the new flags. That is not a
/// hypothetical: the first version of this generator named its parameters `p0..p3`, which tripped
/// W1009 and printed a warning line per parameter from inside the timed region; renaming them to
/// `_p0..` changed no parameter, so the fix appeared to do nothing until this constant existed.
const GENERATOR_VERSION: u32 = 3;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform draw in [0, 1) from the top 53 bits, so the density and shared-fraction comparisons
/// are against a real uniform rather than a modulo of low bits.
fn unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

impl CorpusParams {
    /// A short stable digest of the parameters. It names the corpus directory and appears in the
    /// run log, so a log line and a directory on disk can be matched up later without having to
    /// trust that nobody re-ran the generator with different flags in between.
    pub fn id(&self) -> String {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in self.canonical_string().bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        format!("{h:016x}")
    }

    fn canonical_string(&self) -> String {
        format!(
            "v={GENERATOR_VERSION} n={} m={} d={:.4} a={} s={:.4} p={} l={} seed={}",
            self.modules,
            self.fns_per_module,
            self.density,
            self.arity,
            self.shared_frac,
            self.params_per_fn,
            self.locals_per_module,
            self.seed
        )
    }
}

/// A generated corpus, with the counts a run log should quote.
pub struct Corpus {
    pub params: CorpusParams,
    pub dir: PathBuf,
    pub paths: Vec<String>,
    /// Signature slots that will mint a generic GID. This is the honest measure of interning work,
    /// as opposed to "instantiation sites", which counts source-level instantiations -- most of
    /// which never reach the interner at all (see note 1 in the header).
    pub generic_slots: usize,
    /// Distinct argument lists across the whole corpus: the number of entries the interner's map
    /// will end up holding, and so the thing that decides whether it stays in cache.
    pub distinct_keys: usize,
    /// True when this run reused an existing directory rather than writing one.
    pub reused: bool,
}

impl Corpus {
    /// One line for the run log, so a raw log is self-describing.
    pub fn log_line(&self) -> String {
        let p = &self.params;
        format!(
            "corpus {} | N={} M={} density={:.2} arity={} shared={:.2} params/fn={} \
             locals/mod={} seed={} | {} fns, {} generic signature slots, {} distinct keys{}",
            p.id(),
            p.modules,
            p.fns_per_module,
            p.density,
            p.arity,
            p.shared_frac,
            p.params_per_fn,
            p.locals_per_module,
            p.seed,
            p.modules * p.fns_per_module,
            self.generic_slots,
            self.distinct_keys,
            if self.reused { " (reused)" } else { "" },
        )
    }

    fn manifest_json(&self) -> String {
        let p = &self.params;
        format!(
            "{{\n  \"schema\": \"vx-corpus-v1\",\n  \"corpus_id\": \"{}\",\n  \"modules\": {},\n  \
             \"fns_per_module\": {},\n  \"density\": {},\n  \"arity\": {},\n  \
             \"shared_frac\": {},\n  \"params_per_fn\": {},\n  \"locals_per_module\": {},\n  \
             \"seed\": {},\n  \"functions\": {},\n  \"generic_signature_slots\": {},\n  \
             \"distinct_argument_lists\": {}\n}}\n",
            p.id(),
            p.modules,
            p.fns_per_module,
            p.density,
            p.arity,
            p.shared_frac,
            p.params_per_fn,
            p.locals_per_module,
            p.seed,
            p.modules * p.fns_per_module,
            self.generic_slots,
            self.distinct_keys,
        )
    }
}

/// Generate module `m`, returning its source and the argument lists its signatures will intern.
///
/// The PRNG is seeded from `(seed, m)` alone, so module 3 is the same text whether the corpus has
/// 8 modules or 512. Growing N extends the corpus rather than reshuffling it, which is what makes
/// an N-sweep vary only N.
fn module_source(p: &CorpusParams, m: usize) -> (String, Vec<Vec<String>>) {
    let mut rng = p.seed ^ (m as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut s = String::new();
    let mut keys = Vec::new();

    s.push_str(&format!(
        "// corpus {} module {m} -- generated by src/bin/corpus, do not edit\n\n",
        p.id()
    ));

    // The module-local nominal vocabulary. These serve as both the plain-slot type and the local
    // half of the generic-argument vocabulary, so plain and dense signatures emit the same number
    // of GIDs into the type stream and differ only in whether those GIDs need interning.
    for j in 0..p.locals_per_module {
        s.push_str(&format!("struct L{m}_{j} {{\n  a: i32,\n}}\n\n"));
    }

    // Carriers hold each type parameter behind a pointer (`*mut T`), not by value.
    //
    // This is a codegen constraint, not a modelling choice. `lowered_ty` resolves a generic instance
    // to its *base* layout, which is only instance-independent when every type parameter appears
    // behind a pointer -- a pointer field is 8 bytes for any `T`, whereas a by-value `f0: T0` leaves
    // the base a 0/0 stub and the whole function declines out of the flat subset. With a by-value
    // carrier the density-1 arm compiles its frontend and then generates nothing (#311), so the arm
    // that exists to *show* interning pressure would be the one arm that never reaches codegen.
    //
    // Nothing the density knob controls changes: a parameter slot still holds either a settled
    // nominal GID or an interned instantiation GID, and the key space is still `vocabulary^arity`.
    // What a carrier's field looks like never reaches the interner.
    for k in 0..CARRIERS_PER_MODULE {
        let tps: Vec<String> = (0..p.arity).map(|i| format!("T{i}")).collect();
        let fields: Vec<String> = (0..p.arity)
            .map(|i| format!("  f{i}: *mut T{i},"))
            .collect();
        s.push_str(&format!(
            "struct G{m}_{k}<{}> {{\n{}\n}}\n\n",
            tps.join(", "),
            fields.join("\n")
        ));
    }

    for f in 0..p.fns_per_module {
        let mut sig: Vec<String> = vec!["a: i32".to_string(), "b: i32".to_string()];
        for n in 0..p.params_per_fn {
            let generic = unit(&mut rng) < p.density;
            let args: Vec<String> = (0..p.arity)
                .map(|_| {
                    let shared = unit(&mut rng) < p.shared_frac;
                    let pick = splitmix64(&mut rng) as usize;
                    if shared {
                        SHARED_SCALARS[pick % SHARED_SCALARS.len()].to_string()
                    } else {
                        format!("L{m}_{}", pick % p.locals_per_module)
                    }
                })
                .collect();
            let which = splitmix64(&mut rng) as usize;
            // `_`-prefixed: W1009 warns on an unused parameter unless the name starts with `_`,
            // and stdout is written from inside the timed region. At the plan's largest cell
            // (512 modules x 128 functions x 4 slots) that would be a quarter of a million warning
            // lines landing in the middle of the measurement.
            if generic {
                sig.push(format!(
                    "_p{n}: G{m}_{}<{}>",
                    which % CARRIERS_PER_MODULE,
                    args.join(", ")
                ));
                keys.push(args);
            } else {
                sig.push(format!("_p{n}: L{m}_{}", which % p.locals_per_module));
            }
        }
        // A body with real work for the checker -- a mutable local, a loop, a branch -- so the
        // measurement is not dominated by signature walking alone. The generic parameters are
        // deliberately unused: nothing calls these functions, so a parameter type costs a signature
        // slot and nothing else, which is the isolation the density knob needs.
        s.push_str(&format!(
            "fn m{m}_f{f}({}) -> i32 {{\n\
             \x20 let mut s: i32 = a;\n\
             \x20 for k in 0..b {{\n\
             \x20   if a < k {{ s = s + a * k; }} else {{ s = s - k; }}\n\
             \x20 }}\n\
             \x20 return s + a * {f};\n\
             }}\n\n",
            sig.join(", ")
        ));
    }
    (s, keys)
}

/// Materialise the corpus, reusing an existing directory whose manifest carries the same
/// `corpus_id`. Because the id is a digest of the parameters, a matching id means matching
/// parameters; a directory holding some other corpus is regenerated rather than silently measured.
pub fn generate(params: &CorpusParams, out: Option<&Path>) -> Corpus {
    let dir = out
        .map(|d| d.to_path_buf())
        .unwrap_or_else(|| std::env::temp_dir().join(format!("vx_corpus_{}", params.id())));

    let mut sources = Vec::with_capacity(params.modules);
    let mut distinct = std::collections::HashSet::new();
    let mut generic_slots = 0usize;
    for m in 0..params.modules {
        let (src, keys) = module_source(params, m);
        generic_slots += keys.len();
        for k in keys {
            distinct.insert(k);
        }
        sources.push(src);
    }

    let manifest_path = dir.join("manifest.json");
    let reused = std::fs::read_to_string(&manifest_path)
        .map(|got| got.contains(&format!("\"corpus_id\": \"{}\"", params.id())))
        .unwrap_or(false);

    let corpus = Corpus {
        params: params.clone(),
        dir: dir.clone(),
        paths: (0..params.modules)
            .map(|m| dir.join(format!("m{m}.vx")).to_string_lossy().into_owned())
            .collect(),
        generic_slots,
        distinct_keys: distinct.len(),
        reused,
    };

    if !reused {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("corpus dir");
        for (m, src) in sources.iter().enumerate() {
            std::fs::write(dir.join(format!("m{m}.vx")), src).expect("corpus file");
        }
        std::fs::write(&manifest_path, corpus.manifest_json()).expect("manifest");
    }

    corpus
}

/// Parse `--flag value` out of an argv slice.
pub fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.windows(2)
        .find(|w| w[0] == name)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

/// Parse a comma-separated list, e.g. `--modules 8,64,512`.
pub fn arg_list<T: std::str::FromStr>(args: &[String], name: &str, default: Vec<T>) -> Vec<T> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].split(',').filter_map(|t| t.parse().ok()).collect())
        .unwrap_or(default)
}

/// Build params from argv, taking `modules`/`fns` from a caller that may be sweeping them.
pub fn params_from_args(args: &[String], modules: usize, fns_per_module: usize) -> CorpusParams {
    let d = CorpusParams::default();
    CorpusParams {
        modules,
        fns_per_module,
        density: arg(args, "--density", d.density),
        arity: arg(args, "--arity", d.arity),
        shared_frac: arg(args, "--shared-frac", d.shared_frac),
        params_per_fn: arg(args, "--params-per-fn", d.params_per_fn),
        locals_per_module: arg(args, "--locals-per-module", d.locals_per_module),
        seed: arg(args, "--seed", d.seed),
    }
}
