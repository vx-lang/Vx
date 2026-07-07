# Vx Language Grammar

This document provides a complete, formal description of the syntax and grammatical rules for the Vx programming language. It is represented in Extended Backus-Naur Form (EBNF).

## 1. Program Structure and Modules

A Vx program consists of a sequence of module-level declarations.

```ebnf
program ::= ( import_decl | macro_def | extern_block | trait_decl | impl_block | struct_decl | enum_decl | topology_decl | memory_decl | function_decl )*

import_decl ::= "import" identifier ( "::" identifier )* ";"

macro_def ::= "macro_rules!" identifier "{" macro_rule* "}"
macro_rule ::= "(" token_tree* ")" "=>" "{" token_tree* "}" ";"?
```

## 2. Declarations

Top-level declarations outline the types and functionality of the language.

```ebnf
extern_block ::= "extern" string_literal? "{" extern_fn* "}"
extern_fn ::= "safe"? "fn" identifier "(" param_list? ")" "->" type ";"

trait_decl ::= "trait" identifier generic_params? "{" trait_method* "}"
trait_method ::= "fn" identifier "(" param_list? ")" "->" type ";"

impl_block ::= "impl" generic_params? ( type "for" )? type "{" function_decl* "}"

struct_decl ::= "struct" identifier generic_params? "{" ( identifier ":" type ","? )* "}"

enum_decl ::= "enum" identifier generic_params? "{" ( identifier ( "(" type ( "," type )* ")" )? ","? )* "}"

function_decl ::= "fn" identifier generic_params? "(" param_list? ")" ( "on" topology )? "->" type where_clause? ( "requires" expr )* ( "ensures" expr )* "{" statement* "}"

where_clause ::= "where" transfer_constraint ( "," transfer_constraint )*
transfer_constraint ::= "Transfer" "<" identifier "," identifier ">"

generic_params ::= "<" ( generic_param ","? )* ">"
generic_param ::= "const" identifier ":" type | identifier ( ":" ( "Topology" | identifier ) )?

param_list ::= ( identifier ":" type ","? )*

// A user-defined topology, registered at parse time. `memory:` is required; an omitted
// `visible:` defaults to just the topology's own memory. `transfer` clauses contribute
// morphisms (with a cost and a `relaxed`/`sync` consistency marker) to the cost graph.
topology_decl ::= "Topology" identifier "{" ( topology_field ","? )* "}"
topology_field ::=
    | "memory" ":" memory_space
    | "visible" ":" "[" ( memory_space ","? )* "]"
    | "transfer" memory_space "->" memory_space ":" number ( "relaxed" | "sync" )?

// A first-class memory space (all fields optional except the name). Unlike a topology, the
// descriptor is stored on the AST, not a global registry. `within:` forms the hierarchy tree.
memory_decl  ::= "Memory" identifier "{" ( memory_field ","? )* "}"
memory_field ::=
    | "within"    ":" memory_space
    | "capacity"  ":" size_literal
    | "bandwidth" ":" rate_literal
    | "granule"   ":" size_literal
    | "managed"   ":" ( "explicit" | "cached" )
    | "scope"     ":" ( "device" | "sm" | "cta" | "thread" )   // execution level the space is private to
    | "overcommit"                                             // bare flag: allow the working set to exceed capacity (warn, not error)

size_literal ::= number ( "B" | "KB" | "MB" | "GB" | "TB" )   // binary multipliers (KB = 1024)
rate_literal ::= size_literal "/" ( "s" | "cyc" )
```

## 3. Statements

Statements make up the execution body of functions and blocks.

```ebnf
statement ::= 
    | "let" "mut"? identifier ( ":" type )? "=" expr ";"
    | "comptime" "{" statement* "}"
    | "assert" "(" expr ( "," string_literal )? ")" ";"
    | "return" expr ";"
    | "loop" ( "invariant" "(" expr ")" )* "{" statement* "}"
    | "break" ";"
    | "continue" ";"
    | "for" identifier "in" expr ( "invariant" "(" expr ")" )* "{" statement* "}"
    | expr "=" expr ";"
    | expr "+=" expr ";"
    | identifier "!" token_tree block_tree? ";"?
    | expr ";"?
```

## 4. Expressions

Expressions evaluate to values. Vx supports precedence-climbed binary operators, method chains, and specialized hardware primitives.

```ebnf
expr ::= binary_expr

binary_expr ::= 
    | binary_expr "||" binary_expr
    | binary_expr "&&" binary_expr
    | binary_expr "==" binary_expr
    | binary_expr "!=" binary_expr
    | binary_expr "<=" binary_expr
    | binary_expr ">=" binary_expr
    | binary_expr "<" binary_expr
    | binary_expr ">" binary_expr
    | binary_expr "+" binary_expr
    | binary_expr "-" binary_expr
    | binary_expr "*" binary_expr
    | binary_expr "/" binary_expr
    | binary_expr "@" binary_expr
    | binary_expr ".." binary_expr
    | primary_expr

primary_expr ::= 
    | "!" primary_expr
    | "-" primary_expr
    | "&" "mut"? primary_expr
    | "*" primary_expr
    | "unsafe" "{" statement* "}"
    | "if" "comptime"? expr "{" statement* "}" ( "else" ( "if" primary_expr | "{" statement* "}" ) )?
    | "match" expr "{" ( pattern "=>" ( "{" statement* "}" | statement ) ","? )* "}"
    | "spawn" "on" "(" topology ")" "{" statement* "}"   // routes its body to a topology; yields Pinned<T, topology>
    | "transfer" "(" expr "," memory_space ")"
    | "Transfer" "<" topology "," topology ">"            // comptime transferability predicate (a bool)
    | "[" ( expr ","? )* "]"
    | "Memory" "::" identifier
    | "Topology" "::" identifier
    | "Verified" "(" expr ")"
    | "grad" "(" identifier ( "," expr )* ")"
    | "vjp" "(" identifier "," ( expr "," )* expr ")"
    | "jvp" "(" identifier "," ( expr "," )* expr ")"
    | "(" expr ")"
    | identifier "!" token_tree block_tree?
    | "sizeof" "<" type ">" "(" ")"
    | identifier ( "<" generic_args ">" )? ( "::" identifier ( "<" generic_args ">" )? )* "(" ( expr ","? )* ")"
    | identifier ( "<" generic_args ">" )? "{" ( identifier ":" expr ","? )* "}"
    | identifier "::" identifier ( "(" ( expr ","? )* ")" )?
    | number_literal
    | string_literal
    | identifier

pattern ::= 
    | "_"
    | number_literal
    | string_literal
    | identifier ( "<" identifier ">" )? "::" identifier ( "(" ( pattern ","? )* ")" )?
    | identifier
```

## 5. Types & Memory Semantics

The type system includes primitives, complex types like vectors and tensors, and topological constraints.

```ebnf
type ::= 
    | "&" "mut"? type
    | "*" ( "mut" | "const" ) type
    | "Ref" "<" type "," memory_space ">"
    | "Verified" "<" type ">"
    | "Pinned" "<" type "," topology ">"
    | "<" number ">" "x" element_type
    | "fn" "(" ( type ","? )* ")" "->" type
    | ( "|" ( type ","? )* "|" )? "->" type
    | named_type

named_type ::= 
    | "Tensor" ( "<" element_type ( "," "[" ( expr ","? )* "]" )? ( "," topology )? ">" )?
    | "Matrix"
    | element_type
    | identifier ( "<" type ( "," type )* ">" )?

element_type ::=
    | "i4" | "i8" | "i16" | "i32" | "i64" | "i128"
    | "u4" | "u8" | "u16" | "u32" | "u64" | "u128"
    | "f16" | "bf16" | "f32" | "f64"
    | "bool"

// Any identifier that is not a built-in name is a user-defined (custom) memory space,
// so a `Topology` declaration can introduce a novel memory.
memory_space ::= "Memory" "::" ( "CPU_DRAM" | "NPU_HBM" | "GPU_HBM" | "Local_SRAM" | "NIC_RAM" | "Remote_HBM" | identifier )

topology ::= "Topology" "::" topology_kind
topology_kind ::= 
    | "CPU"
    | "Current"
    | "GPU"
    | "AMX"
    | "ANE"
    | "CpuAvx512"                        // also accepts "CPU_AVX512"
    | "CpuNeon"                          // also accepts "CPU_Neon"
    | "NPU" "[" expr "]"
    | "NPU" "[" expr ".." expr "]"       // Slice: spawns across a range of NPUs
    | "AccCore" "[" expr "]"
    | identifier                         // a user-declared `Topology`
```

> **Reserved but not yet parsed.** The lexer reserves `unroll`, `across`, and `HardwareState`
> as keywords, but the parser does not yet accept them (see the "unimplemented" notes in
> `syntax.md` / `types.md`). `safe` (on `extern` functions) and the `..` range operator are
> parsed and implemented. Only `+=` compound assignment is supported (`*=` is not).

## 6. ABI Mangling

Internally, the compiler uses a deterministic mangling scheme based on the `$` character to represent types, generics, and traits in the C-ABI. This guarantees no naming collisions with user-defined identifiers (which cannot contain `$`).

### Mangling Rules

- Scalar Types: Used verbatim (e.g. `f32`, `i64`).
- Tensors: `Tensor$<element_type>$<rank>` (e.g., `Tensor$f32$2`).
- Vectors: `Simd$<element_type>$<width>` (e.g., `Simd$f32$4`).
- Methods: `<struct_name>$<method_name>` (e.g., `Point$distance`).
- Shadowed Variables: Appended with `_mangled<index>` starting from `_mangled1` when a local variable shadows an earlier variable in the same scope or parent scope (e.g., `x_mangled1`).

*Note: The user-facing syntax uses `< >` for generics and `::` for paths. The `$` symbol is strictly reserved for backend lowering and internal compiler representation.*
