# Vx Language Grammar

This document provides a complete, formal description of the syntax and grammatical rules for the Vx programming language. It is represented in Extended Backus-Naur Form (EBNF).

## 1. Program Structure and Modules

A Vx program consists of a sequence of module-level declarations.

```ebnf
program ::= ( import_decl | macro_def | extern_block | trait_decl | impl_block | struct_decl | enum_decl | function_decl )*

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

function_decl ::= "fn" identifier generic_params? "(" param_list? ")" ( "on" topology )? "->" type ( "requires" expr )* ( "ensures" expr )* "{" statement* "}"

generic_params ::= "<" ( generic_param ","? )* ">"
generic_param ::= "const" identifier ":" type | identifier ( ":" identifier )?

param_list ::= ( identifier ":" type ","? )*
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
    | "transfer" "(" expr "," memory_space ")"
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

element_type ::= "f32" | "f64" | "i32" | "i64" | "i128" | "bool"

memory_space ::= "CPU_DRAM" | "NPU_HBM" | "Local_SRAM" | "NIC_RAM" | "Remote_HBM"

topology ::= 
    | "Topology" "::" "CPU"
    | "Topology" "::" "Current"
    | "Topology" "::" "NPU" "[" expr "]"
    | "Topology" "::" "NPU" "[" expr ".." expr "]"   // Slice: spawns across a range of NPUs
    | "Topology" "::" "AccCore" "[" expr "]"
    | "Topology" "::" "AMX"
    | "Topology" "::" "ANE"
    | "Topology" "::" "GPU"
    | "Topology" "::" "CpuAvx512"
    | "Topology" "::" "CpuNeon"
```

## 6. ABI Mangling

Internally, the compiler uses a deterministic mangling scheme based on the `$` character to represent types, generics, and traits in the C-ABI. This guarantees no naming collisions with user-defined identifiers (which cannot contain `$`).

### Mangling Rules

- Scalar Types: Used verbatim (e.g. `f32`, `i64`).
- Tensors: `Tensor$<element_type>$<rank>` (e.g., `Tensor$f32$2`).
- Vectors: `Simd$<element_type>$<width>` (e.g., `Simd$f32$4`).
- Methods: `<struct_name>$<method_name>` (e.g., `Point$distance`).
- Shadowed Variables: Appended with `_mangled<index>` starting from `_mangled1` when a local variable shadows an earlier variable in the same scope or parent scope (e.g., `x_mangled1`).

*Note: The user-facing syntax uses `< >` for generics and `::` for paths. The `$` symbol is strictly reserved for backend lowering and internal compiler representation.*
