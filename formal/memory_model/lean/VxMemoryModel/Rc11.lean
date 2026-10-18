import VxMemoryModel.Execution
import VxMemoryModel.Relation

/-!
# The Vx memory model on CPUs

RC11 (Lahav et al., "Repairing Sequential Consistency in C/C++11", PLDI 2017), written as in
`../memalloy/vx_cpu.cat`. memalloy checks that file against RC11 up to 6 events.
-/

namespace Vx.Execution

open Relation

variable (X : Execution)

/-- Release sequence: a write, then maybe an atomic write of the same thread to the same place,
then a chain of read-modify-writes that read from each other. -/
def rs (a b : X.E) : Prop :=
  ∃ w, X.IsWrite a ∧ (a = w ∨ (X.po a w ∧ X.loc a = X.loc w)) ∧ X.IsWrite w ∧ X.IsAtomic w ∧
    (w = b ∨ TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) w b)

/-- Synchronizes-with: a release (a write or a fence before a write) is read by an acquire (a
read, or a read before a fence). -/
def sw (a b : X.E) : Prop :=
  X.IsRelease a ∧ X.IsAcquire b ∧
    ∃ x y z, (a = x ∨ (X.IsFence a ∧ X.po a x)) ∧ X.rs x y ∧ X.rf y z ∧ X.IsAtomic z ∧
      (z = b ∨ (X.po z b ∧ X.IsFence b))

/-- Happens-before. -/
def hb : X.E → X.E → Prop := TransGen (fun a b => X.po a b ∨ X.sw a b)

/-- Extended coherence order. -/
def eco : X.E → X.E → Prop := TransGen (fun a b => X.rf a b ∨ X.co a b ∨ X.fr a b)

def scb (a b : X.E) : Prop :=
  X.po a b ∨
    (∃ c d, X.po a c ∧ X.loc a ≠ X.loc c ∧ X.hb c d ∧ X.po d b ∧ X.loc d ≠ X.loc b) ∨
    (X.hb a b ∧ X.loc a = X.loc b) ∨ X.co a b ∨ X.fr a b

def pscb (a b : X.E) : Prop :=
  ∃ c d,
    ((X.IsSeqCst a ∧ a = c) ∨ (X.IsFence a ∧ X.IsSeqCst a ∧ X.hb a c)) ∧ X.scb c d ∧
    ((X.IsSeqCst b ∧ d = b) ∨ (X.IsFence b ∧ X.IsSeqCst b ∧ X.hb d b))

def pscf (a b : X.E) : Prop :=
  X.IsFence a ∧ X.IsSeqCst a ∧ X.IsFence b ∧ X.IsSeqCst b ∧
    (X.hb a b ∨ ∃ c d, X.hb a c ∧ X.eco c d ∧ X.hb d b)

/-- The order that all sequentially consistent events agree on. -/
def psc (a b : X.E) : Prop := X.pscb a b ∨ X.pscf a b

structure Consistent : Prop where
  coherence : ∀ a b, X.hb a b → ¬ (a = b ∨ X.eco b a)
  coherence_rmw : ∀ a b, X.rmw a b → ¬ X.eco b a
  atomicity : ∀ a b c, X.rmw a b → X.fr a c → ¬ X.co c b
  seq_cst : ∀ a, ¬ TransGen X.psc a a
  /-- No value comes out of thin air: no cycle of program order and reads. -/
  no_thin_air : ∀ a, ¬ TransGen (fun a b => X.po a b ∨ X.rf a b) a a

def IsAccess (e : X.E) : Prop := X.IsRead e ∨ X.IsWrite e

/-- Two accesses conflict when they touch one place and at least one of them writes. -/
def Conflict (a b : X.E) : Prop :=
  a ≠ b ∧ X.IsAccess a ∧ X.IsAccess b ∧ X.loc a = X.loc b ∧ (X.IsWrite a ∨ X.IsWrite b)

/-- A data race: two threads make conflicting accesses that happens-before does not order,
and at least one of them is not atomic. -/
def DataRace (a b : X.E) : Prop :=
  X.Conflict a b ∧ X.thread a ≠ X.thread b ∧ ¬ X.hb a b ∧ ¬ X.hb b a ∧
    ¬ (X.IsAtomic a ∧ X.IsAtomic b)

def RaceFree : Prop := ∀ a b, ¬ X.DataRace a b

/-- The edges of an interleaving. -/
def ord (a b : X.E) : Prop := X.po a b ∨ X.rf a b ∨ X.co a b ∨ X.fr a b

variable {X} in
theorem ord_po {a b : X.E} (h : X.po a b) : X.ord a b := Or.inl h
variable {X} in
theorem ord_rf {a b : X.E} (h : X.rf a b) : X.ord a b := Or.inr (Or.inl h)
variable {X} in
theorem ord_co {a b : X.E} (h : X.co a b) : X.ord a b := Or.inr (Or.inr (Or.inl h))
variable {X} in
theorem ord_fr {a b : X.E} (h : X.fr a b) : X.ord a b := Or.inr (Or.inr (Or.inr h))

/-- Sequential consistency: every run is one interleaving of the threads. -/
structure SeqConsistent : Prop where
  acyclic : ∀ a, ¬ TransGen X.ord a a
  atomicity : ∀ a b c, X.rmw a b → X.fr a c → ¬ X.co c b

end Vx.Execution
