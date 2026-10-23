/-!
# Executions

An execution is one run of a concurrent program, seen as a graph. Its nodes are the memory
accesses and fences of the run. Its edges say which write each read takes its value from
(`rf`), in which order the writes to one place happen (`co`), and the order of the events in
each thread (`po`). A memory model is a set of rules that says which executions can happen.

This file follows the CPU part of `../vx.cat` and `../memalloy/vx_cpu.cat`: no scopes and no
barriers. A read-modify-write is two events, a read and a write, joined by `rmw`.
-/

namespace Vx

inductive Kind where
  | read
  | write
  | fence
  deriving DecidableEq

/-- The memory order of an access or a fence. `plain` is an access that is not atomic. -/
inductive Order where
  | plain
  | rlx
  | acq
  | rel
  | acqRel
  | sc
  deriving DecidableEq

structure Execution where
  /-- The events of the run. -/
  E : Type
  kind : E → Kind
  order : E → Order
  /-- The place that an access reads or writes. A fence has no place, so it is ignored. -/
  loc : E → Nat
  thread : E → Nat
  /-- Program order: `po a b` when `a` comes before `b` in one thread. -/
  po : E → E → Prop
  /-- Reads from: `rf w r` when the read `r` takes its value from the write `w`. -/
  rf : E → E → Prop
  /-- Coherence order: the order of the writes to one place. -/
  co : E → E → Prop
  /-- `rmw r w` when the read `r` and the write `w` are one read-modify-write. -/
  rmw : E → E → Prop

namespace Execution

variable (X : Execution)

def IsRead (e : X.E) : Prop := X.kind e = .read
def IsWrite (e : X.E) : Prop := X.kind e = .write
def IsFence (e : X.E) : Prop := X.kind e = .fence
def IsAtomic (e : X.E) : Prop := X.order e ≠ .plain
def IsAcquire (e : X.E) : Prop := X.order e = .acq ∨ X.order e = .acqRel ∨ X.order e = .sc
def IsRelease (e : X.E) : Prop := X.order e = .rel ∨ X.order e = .acqRel ∨ X.order e = .sc
def IsSeqCst (e : X.E) : Prop := X.order e = .sc

/-- From-reads: `fr r w` when the write `w` comes after the write that `r` reads from. -/
def fr (r w : X.E) : Prop := ∃ w₀, X.rf w₀ r ∧ X.co w₀ w

/-- The rules that every execution follows, whatever the memory model. -/
structure WellFormed : Prop where
  po_trans : ∀ a b c, X.po a b → X.po b c → X.po a c
  /-- Two events of one thread are ordered. -/
  po_total : ∀ a b, X.thread a = X.thread b → a ≠ b → X.po a b ∨ X.po b a
  rf_write_read : ∀ w r, X.rf w r → X.IsWrite w ∧ X.IsRead r ∧ X.loc w = X.loc r
  /-- A read reads from one write. -/
  rf_unique : ∀ w w' r, X.rf w r → X.rf w' r → w = w'
  co_writes : ∀ a b, X.co a b → X.IsWrite a ∧ X.IsWrite b ∧ X.loc a = X.loc b
  co_irrefl : ∀ a, ¬ X.co a a
  rmw_po : ∀ r w, X.rmw r w → X.po r w
  rmw_atomic : ∀ r w, X.rmw r w → X.IsRead r ∧ X.IsWrite w ∧ X.IsAtomic r ∧ X.IsAtomic w

end Execution

end Vx
