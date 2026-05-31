# Vx Language Formal Grammar

This document provides a formal Extended Backus-Naur Form (EBNF) specification for the Vx programming language.

## 1. Notation

The grammar is specified using the following notation:
* `...` : Literal text
* `Capitalized` : Lexical token or terminal rule
* `lowercase_with_underscores` : Syntactic non-terminal rule
* `[ x ]` : Optional item `x`
* `{ x }` : Zero or more repetitions of `x`
* `x | y` : Alternation (either `x` or `y`)
* `( x )` : Grouping

---

## 2. Lexical Elements

### 2.1 Identifiers

```ebnf
Identifier ::= XID_Start { XID_Continue }
```

### 2.2 Keywords

The following identifiers are reserved keywords and cannot be used as variable or function names:
`fn`, `let`, `mut`, `for`, `in`, `if`, `else`, `loop`, `break`, `continue`, `return`, `spawn`, `on`, `transfer`, `unroll`, `across`, `match`, `struct`, `enum`, `trait`, `impl`, `extern`, `unsafe`, `safe`, `comptime`, `import`, `assert`, `grad`, `vjp`, `jvp`.

### 2.3 Literals

```ebnf
Literal ::= IntegerLiteral | FloatLiteral | StringLiteral | BooleanLiteral

IntegerLiteral ::= Digit { Digit }
FloatLiteral   ::= Digit { Digit } `.` Digit { Digit } [ Exponent ]
Exponent       ::= ( `e` | `E` ) [ `+` | `-` ] Digit { Digit }

BooleanLiteral ::= `true` | `false`
StringLiteral  ::= `"` { StringCharacter } `"`
```

---

## 3. Types

```ebnf
type ::=
    | PrimitiveType
    | tensor_type
    | matrix_type
    | reference_type
    | ptr_type
    | array_type
    | simd_type
    | topology_state_type
    | Identifier [ generic_args ]

PrimitiveType ::= `i4` | `u4` | `i8` | `u8` | `i16` | `u16` | `i32` | `u32` 
                | `i64` | `u64` | `i128` | `u128` 
                | `f16` | `bf16` | `f32` | `f64` 
                | `bool`

tensor_type   ::= `Tensor` `<` type `,` `[` { expression `,` } `]` [ `,` Topology ] `>`
matrix_type   ::= `Matrix`
reference_type ::= `&` [ `mut` ] type [ `in` MemorySpace ]
ptr_type      ::= `*` ( `const` | `mut` ) type [ `in` MemorySpace ]
array_type    ::= `[` type `;` expression `]`
simd_type     ::= `<` IntegerLiteral `x` PrimitiveType `>`

topology_state_type ::= 
    | `Verified` `<` type `>`
    | `Pinned` `<` type `,` Topology `>`
    | `HardwareState` `<` type `,` Topology `>`
    | `Ref` `<` type `,` MemorySpace `>`

generic_args ::= `<` type { `,` type } `>`

Topology ::= `Topology` `::` Identifier [ `[` expression `]` | `[` expression `..` expression `]` ]
MemorySpace ::= `Memory` `::` Identifier
```

---

## 4. Expressions

```ebnf
expression ::= 
    | literal
    | Identifier
    | tuple_expression
    | array_expression
    | block_expression
    | if_expression
    | match_expression
    | spawn_expression
    | autodiff_expression
    | expression binary_op expression
    | unary_op expression
    | expression `(` [ argument_list ] `)`
    | expression `[` expression `]`
    | expression `.` Identifier
    | expression `with` Topology

argument_list ::= expression { `,` expression }

binary_op ::= `+` | `-` | `*` | `/` | `%` | `==` | `!=` | `<` | `>` | `<=` | `>=` | `&&` | `||`
unary_op  ::= `-` | `!` | `&` [ `mut` ] | `*`
```

### 4.1 Control Flow Expressions

```ebnf
if_expression ::= `if` expression block_expression [ `else` ( block_expression | if_expression ) ]

match_expression ::= `match` expression `{` { match_arm } `}`
match_arm ::= pattern `=>` expression `,`

spawn_expression ::= `spawn` `on` `(` Topology `)` block_expression

autodiff_expression ::= ( `grad` | `vjp` | `jvp` ) `(` expression `)`
```

---

## 5. Statements

```ebnf
statement ::= 
    | let_statement
    | expression_statement
    | loop_statement
    | for_statement
    | return_statement
    | break_statement
    | continue_statement
    | assert_statement
    | transfer_statement

let_statement ::= `let` [ `mut` ] Identifier [ `:` type ] `=` expression `;`

expression_statement ::= expression `;`

loop_statement ::= `loop` block_expression
for_statement ::= `for` Identifier `in` expression block_expression
                | `unroll` `across` `(` Topology `)` `{` `|` Identifier `|` block_expression `}`

return_statement ::= `return` [ expression ] `;`
break_statement ::= `break` `;`
continue_statement ::= `continue` `;`

assert_statement ::= `assert` `(` expression [ `,` StringLiteral ] `)` `;`

transfer_statement ::= `let` Identifier `=` `transfer` `(` expression `,` MemorySpace `)` `;`

block_expression ::= `{` { statement } [ expression ] `}`
```

---

## 6. Declarations

```ebnf
declaration ::= 
    | function_declaration
    | struct_declaration
    | enum_declaration
    | trait_declaration
    | impl_declaration
    | extern_block
    | import_declaration

function_declaration ::= [ `safe` ] `fn` Identifier [ generic_params ] `(` [ param_list ] `)` 
                         [ `on` Topology ] [ `->` type ] [ `effects` `(` effect_list `)` ]
                         block_expression

param_list ::= param { `,` param }
param ::= [ `mut` ] Identifier `:` type

effect_list ::= effect { `,` effect }
effect ::= `DataMovement` `(` MemorySpace `->` MemorySpace `)`

struct_declaration ::= `struct` Identifier [ generic_params ] `{` { struct_field `,` } `}`
struct_field ::= Identifier `:` type

enum_declaration ::= `enum` Identifier [ generic_params ] `{` { enum_variant `,` } `}`
enum_variant ::= Identifier [ `(` type { `,` type } `)` ]

trait_declaration ::= `trait` Identifier `{` { function_signature `;` } `}`
impl_declaration ::= `impl` Identifier `for` type `{` { function_declaration } `}`

extern_block ::= `extern` `{` { [ `safe` ] function_signature `;` } `}`
function_signature ::= `fn` Identifier `(` [ param_list ] `)` [ `->` type ]

import_declaration ::= `import` Identifier [ `::` Identifier | `::` `{` Identifier { `,` Identifier } `}` ] `;`
```
