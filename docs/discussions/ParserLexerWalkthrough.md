# Parser & Lexer Refactoring Walkthrough

I have implemented the improvements detailed in the `docs/discussions/Vx_src_lexer.md` and `docs/discussions/Vx_src_parser_mod.md` proposals.

## Changes Made

### Zero-Copy Token Error Handling

The previous implementation used `OwnedToken` for errors, forcing eager allocations. We replaced `OwnedToken` by introducing a generic `TokenBase<S, C>`, which allowed unifying `Token` and `OwnedToken`. `ParserError` now natively uses zero-copy references (`Token<'a>`), which eliminates allocations when errors occur!

### Fast Keyword Lookup

Replaced the monolithic `match` statement for keywords with an O(1) hash map using `once_cell::sync::Lazy` and `rustc-hash::FxHashMap`. This drastically improves keyword identification during the hot parsing loop.

### Avoid Eager String Allocations

The lexer's `string_literal` parsing now defers allocations entirely. It parses standard literal chunks into a `Cow<'a, str>` and only allocates to `String` when an escape sequence like `\n` is actually encountered.

### Streamlined Parser Loop

- Eliminated redundant `iter.clone()` peeks using explicit string-slice lookaheads across numbers, strings, and whitespace skipping.
- Avoided token cloning during `Parser::consume` success paths by letting `ParserResult` natively propagate the borrowed lifetimes from the parser core.

## Validation Results

All code passes the extensive test suites. The entire recursive descent stack compiles flawlessly and `cargo bench` metrics confirm our allocation pressure is massively reduced.
