//! The flat emitter, one module per opcode family.
//!
//! Each module adds an `impl` block to `FnEmit` covering the opcodes it names. A new opcode
//! gets a method in the family it belongs to and an arm in `step`.

mod aggregate;
mod arith;
mod call;
mod control;
mod io;
mod memory;
mod parallel;
mod tensor;
