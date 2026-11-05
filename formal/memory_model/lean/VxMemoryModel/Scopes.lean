import VxMemoryModel.DrfSc
import VxMemoryModel.Program

/-!
# Scopes and barriers on the GPU

A GPU puts its threads in subgroups, the subgroups in blocks and the blocks in devices. An atomic
or a fence has a scope, and it reaches only the threads inside its scope. A barrier makes the
threads of one subgroup, block or device wait for each other. This file adds both to an
execution, as `../vx.cat` does. The CPU files do not change: with every scope `system`, the
definitions here are those of `Rc11.lean` (`sw_eq`, `hb_eq`, `consistent_iff`, `dataRace_iff`).
-/

namespace Vx

open Relation

inductive Scope where
  | subgroup
  | block
  | device
  | system
  deriving DecidableEq

/-- An execution where each event has a scope, and each thread has a subgroup, a block and a
device. -/
structure ScopedExecution extends Execution where
  /-- The scope of an event. An event without a scope tag has `system` scope. -/
  scope : E → Scope
  /-- `group t s` names the subgroup, block or device of the thread `t`, for `s` of that scope. -/
  group : Nat → Scope → Nat
  /-- A fence that is a barrier of its scope. -/
  barrier : E → Bool

namespace Execution

variable (X : Execution) (hb : X.E → X.E → Prop)

/-! The rules of `Rc11.lean`, for any happens-before order `hb`. -/

def scbWith (a b : X.E) : Prop :=
  X.po a b ∨
    (∃ c d, X.po a c ∧ X.loc a ≠ X.loc c ∧ hb c d ∧ X.po d b ∧ X.loc d ≠ X.loc b) ∨
    (hb a b ∧ X.loc a = X.loc b) ∨ X.co a b ∨ X.fr a b

def pscWith (a b : X.E) : Prop :=
  (∃ c d,
    ((X.IsSeqCst a ∧ a = c) ∨ (X.IsFence a ∧ X.IsSeqCst a ∧ hb a c)) ∧ X.scbWith hb c d ∧
    ((X.IsSeqCst b ∧ d = b) ∨ (X.IsFence b ∧ X.IsSeqCst b ∧ hb d b))) ∨
  (X.IsFence a ∧ X.IsSeqCst a ∧ X.IsFence b ∧ X.IsSeqCst b ∧
    (hb a b ∨ ∃ c d, hb a c ∧ X.eco c d ∧ hb d b))

structure ConsistentWith : Prop where
  coherence : ∀ a b, hb a b → ¬ (a = b ∨ X.eco b a)
  coherence_rmw : ∀ a b, X.rmw a b → ¬ X.eco b a
  atomicity : ∀ a b c, X.rmw a b → X.fr a c → ¬ X.co c b
  seq_cst : ∀ a, ¬ TransGen (X.pscWith hb) a a
  no_thin_air : ∀ a, ¬ TransGen (fun a b => X.po a b ∨ X.rf a b) a a

theorem consistentWith_hb : X.ConsistentWith X.hb ↔ X.Consistent :=
  ⟨fun h => ⟨h.1, h.2, h.3, h.4, h.5⟩, fun h => ⟨h.1, h.2, h.3, h.4, h.5⟩⟩

end Execution

namespace ScopedExecution

variable (S : ScopedExecution)

/-- The threads of `a` and `b` are in one group of scope `s`. -/
def SameGroup (s : Scope) (a b : S.E) : Prop := S.group (S.thread a) s = S.group (S.thread b) s

/-- `reaches` of `vx.cat`: `a` is atomic, and its scope includes the thread of `b`. -/
def Reaches (a b : S.E) : Prop :=
  S.IsAtomic a ∧ (S.scope a = .system ∨ S.SameGroup (S.scope a) a b)

/-- `strong` of `vx.cat`: each one of `a` and `b` reaches the thread of the other. -/
def Strong (a b : S.E) : Prop := S.Reaches a b ∧ S.Reaches b a

/-- Synchronizes-with, as in `Rc11.lean`. The release and the acquire must reach each other, and
so must the write and the read that link them. -/
def sw (a b : S.E) : Prop :=
  S.IsRelease a ∧ S.IsAcquire b ∧ S.Strong a b ∧
    ∃ x y z, (a = x ∨ (S.IsFence a ∧ S.po a x)) ∧ S.rs x y ∧ S.rf y z ∧ S.Strong y z ∧
      S.IsAtomic z ∧ (z = b ∨ (S.po z b ∧ S.IsFence b))

def IsBarrier (e : S.E) : Prop := S.IsFence e ∧ S.barrier e = true

/-- `bar` of `vx.cat`: an event before a barrier, and an event after the matching barrier of
another thread in the same group of the barriers' scope. -/
def bar (a b : S.E) : Prop :=
  ∃ x y, S.po a x ∧ S.IsBarrier x ∧ S.IsBarrier y ∧ S.scope x = S.scope y ∧
    S.scope x ≠ .system ∧ S.SameGroup (S.scope x) x y ∧ S.thread x ≠ S.thread y ∧ S.po y b

def hb : S.E → S.E → Prop := TransGen (fun a b => S.po a b ∨ S.sw a b ∨ S.bar a b)

def Consistent : Prop := S.ConsistentWith S.hb

/-- A data race: two threads make conflicting accesses that `hb` does not order, and the two
accesses do not reach each other. -/
def DataRace (a b : S.E) : Prop :=
  S.Conflict a b ∧ S.thread a ≠ S.thread b ∧ ¬ S.hb a b ∧ ¬ S.hb b a ∧ ¬ S.Strong a b

def RaceFree : Prop := ∀ a b, ¬ S.DataRace a b

/-- Every event has `system` scope, as on a CPU. -/
def AllSystem : Prop := ∀ e, S.scope e = .system

/-! ## With every scope `system`, the definitions are those of `Rc11.lean` -/

variable {S}

theorem strong_iff (hs : S.AllSystem) {a b : S.E} : S.Strong a b ↔ S.IsAtomic a ∧ S.IsAtomic b :=
  ⟨fun h => ⟨h.1.1, h.2.1⟩, fun h => ⟨⟨h.1, Or.inl (hs a)⟩, ⟨h.2, Or.inl (hs b)⟩⟩⟩

theorem rs_atomic (wf : S.WellFormed) {x y : S.E} (h : S.rs x y) : S.IsAtomic y := by
  obtain ⟨w, -, -, -, hwa, rfl | hc⟩ := h
  · exact hwa
  · cases hc with
    | single h => obtain ⟨r, -, hr⟩ := h; exact (wf.rmw_atomic _ _ hr).2.2.2
    | tail _ h => obtain ⟨r, -, hr⟩ := h; exact (wf.rmw_atomic _ _ hr).2.2.2

theorem sw_eq (hs : S.AllSystem) (wf : S.WellFormed) : S.sw = S.toExecution.sw := by
  funext a b
  apply propext
  constructor
  · rintro ⟨hr, ha, -, x, y, z, h1, h2, h3, -, h5, h6⟩
    exact ⟨hr, ha, x, y, z, h1, h2, h3, h5, h6⟩
  · rintro ⟨hr, ha, x, y, z, h1, h2, h3, h5, h6⟩
    exact ⟨hr, ha, (strong_iff hs).2 ⟨Execution.atomic_of_release hr,
      Execution.atomic_of_acquire ha⟩, x, y, z, h1, h2, h3,
      (strong_iff hs).2 ⟨rs_atomic wf h2, h5⟩, h5, h6⟩

theorem not_bar (hs : S.AllSystem) (a b : S.E) : ¬ S.bar a b := by
  rintro ⟨x, -, -, -, -, -, hx, -⟩
  exact hx (hs x)

theorem hb_eq (hs : S.AllSystem) (wf : S.WellFormed) : S.hb = S.toExecution.hb := by
  unfold hb Execution.hb
  rw [sw_eq hs wf]
  congr
  funext a b
  apply propext
  exact ⟨fun h => h.elim Or.inl (fun h => h.elim Or.inr (fun h => (not_bar hs a b h).elim)),
    fun h => h.elim Or.inl (fun h => Or.inr (Or.inl h))⟩

theorem consistent_iff (hs : S.AllSystem) (wf : S.WellFormed) :
    S.Consistent ↔ S.toExecution.Consistent := by
  unfold Consistent
  rw [hb_eq hs wf]
  exact Execution.consistentWith_hb _

theorem dataRace_iff (hs : S.AllSystem) (wf : S.WellFormed) {a b : S.E} :
    S.DataRace a b ↔ S.toExecution.DataRace a b := by
  unfold DataRace Execution.DataRace
  rw [hb_eq hs wf, strong_iff hs]

/-! ## The rules of RC11 that do not depend on the orders -/

/-- The same execution with every event relaxed. Its `hb` is `po`, so it keeps only the rules of
RC11 that hold whatever the orders and the scopes. The lemmas of `Counter.lean` need no more. -/
def relaxed (S : ScopedExecution) : Execution := { S.toExecution with order := fun _ => .rlx }

theorem relaxed_wellFormed (wf : S.WellFormed) : S.relaxed.WellFormed where
  po_trans := wf.po_trans
  po_total := wf.po_total
  rf_write_read := wf.rf_write_read
  rf_unique := wf.rf_unique
  co_writes := wf.co_writes
  co_irrefl := wf.co_irrefl
  rmw_po := wf.rmw_po
  rmw_atomic r w h := by
    obtain ⟨h1, h2, -, -⟩ := wf.rmw_atomic r w h
    exact ⟨h1, h2, nofun, nofun⟩

theorem relaxed_hb {a b : S.E} (h : S.relaxed.hb a b) : S.hb a b := by
  induction h with
  | single h =>
    rcases h with h | ⟨h, -⟩
    · exact TransGen.single (Or.inl h)
    · rcases h with h | h | h <;> cases h
  | tail _ h ih =>
    rcases h with h | ⟨h, -⟩
    · exact ih.tail (Or.inl h)
    · rcases h with h | h | h <;> cases h

theorem relaxed_consistent (hc : S.Consistent) : S.relaxed.Consistent where
  coherence a b h := hc.coherence a b (relaxed_hb h)
  coherence_rmw := hc.coherence_rmw
  atomicity := hc.atomicity
  seq_cst a h := by
    obtain ⟨c, h⟩ := h.exists_first
    rcases h with ⟨c, d, h, -⟩ | ⟨-, h, -⟩
    · rcases h with ⟨h, -⟩ | ⟨-, h, -⟩ <;> cases h
    · cases h
  no_thin_air := hc.no_thin_air

end ScopedExecution

end Vx
