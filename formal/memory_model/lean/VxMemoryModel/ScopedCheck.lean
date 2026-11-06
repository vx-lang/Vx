import VxMemoryModel.Check
import VxMemoryModel.Scopes

/-!
# Deciding the model with scopes for a small execution

As `Check.lean`, but for the definitions of `Scopes.lean`. `run_lean.py` uses it for the litmus
tests with scopes, fences and barriers.
-/

namespace Vx

open Relation

/-- An execution with `n` events, with the scope of each event, the groups of each thread, and
the barriers. -/
structure SData (n : Nat) extends Data n where
  scope : Fin n → Scope
  group : Nat → Scope → Nat
  barrier : Fin n → Bool

/-- `groups` gives the subgroup, the block and the device of each thread. A thread that is not in
the list gets groups of its own. All the threads are in one system. -/
def SData.ofLists (n : Nat) (kind : List Kind) (order : List Order) (loc thread : List Nat)
    (po rf co rmw : List (Nat × Nat)) (scope : List Scope) (groups : List (Nat × Nat × Nat))
    (barrier : List Bool) : SData n where
  toData := Data.ofLists n kind order loc thread po rf co rmw
  scope i := scope.getD i.val .system
  group t s :=
    match groups[t]?, s with
    | some (g, _, _), .subgroup => g
    | some (_, g, _), .block => g
    | some (_, _, g), .device => g
    | _, .system => 0
    | _, _ => 1000000 + t
  barrier i := barrier.getD i.val false

abbrev SData.toScoped (d : SData n) : ScopedExecution :=
  { d.toData.toExecution with scope := d.scope, group := d.group, barrier := d.barrier }

namespace SData

variable (d : SData n)

instance : DecidableEq d.toScoped.E := inferInstanceAs (DecidableEq (Fin n))
instance {P : d.toScoped.E → Prop} [DecidablePred P] : Decidable (∃ e, P e) :=
  Nat.decidableExistsFin P
instance {P : d.toScoped.E → Prop} [DecidablePred P] : Decidable (∀ e, P e) :=
  Nat.decidableForallFin P
instance : DecidableRel d.toScoped.po := fun a b => inferInstanceAs (Decidable (d.po a b = true))
instance : DecidableRel d.toScoped.rf := fun a b => inferInstanceAs (Decidable (d.rf a b = true))
instance : DecidableRel d.toScoped.co := fun a b => inferInstanceAs (Decidable (d.co a b = true))
instance : DecidableRel d.toScoped.rmw := fun a b =>
  inferInstanceAs (Decidable (d.rmw a b = true))
instance : DecidableRel d.toScoped.fr := fun a b =>
  inferInstanceAs (Decidable (∃ w : Fin n, d.rf w a = true ∧ d.co w b = true))
instance (e : d.toScoped.E) : Decidable (d.toScoped.IsRead e) := inferInstanceAs (Decidable (_ = _))
instance (e : d.toScoped.E) : Decidable (d.toScoped.IsWrite e) :=
  inferInstanceAs (Decidable (_ = _))
instance (e : d.toScoped.E) : Decidable (d.toScoped.IsFence e) :=
  inferInstanceAs (Decidable (_ = _))
instance (e : d.toScoped.E) : Decidable (d.toScoped.IsAtomic e) :=
  inferInstanceAs (Decidable (¬ _ = _))
instance (e : d.toScoped.E) : Decidable (d.toScoped.IsAcquire e) := by
  unfold Execution.IsAcquire; infer_instance
instance (e : d.toScoped.E) : Decidable (d.toScoped.IsRelease e) := by
  unfold Execution.IsRelease; infer_instance
instance (e : d.toScoped.E) : Decidable (d.toScoped.IsSeqCst e) :=
  inferInstanceAs (Decidable (_ = _))
instance (e : d.toScoped.E) : Decidable (d.toScoped.IsAccess e) := by
  unfold Execution.IsAccess; infer_instance
instance : DecidableRel d.toScoped.Strong := fun a b => by
  unfold ScopedExecution.Strong ScopedExecution.Reaches ScopedExecution.SameGroup; infer_instance
instance : DecidableRel d.toScoped.bar := fun a b => by
  unfold ScopedExecution.bar ScopedExecution.IsBarrier ScopedExecution.SameGroup; infer_instance

def chainStep (x y : Fin n) : Prop := d.toScoped.rsStep x y
def hbStep (a b : Fin n) : Prop := d.toScoped.po a b ∨ d.toScoped.sw a b ∨ d.toScoped.bar a b
def ecoStep (a b : Fin n) : Prop :=
  d.toScoped.rf a b ∨ d.toScoped.co a b ∨ d.toScoped.fr a b
def thinAirStep (a b : Fin n) : Prop := d.toScoped.po a b ∨ d.toScoped.rf a b

instance : DecidableRel d.chainStep := fun x y => by
  unfold chainStep ScopedExecution.rsStep; infer_instance
instance : DecidableRel d.ecoStep := fun a b => by unfold ecoStep; infer_instance
instance : DecidableRel d.thinAirStep := fun a b => by unfold thinAirStep; infer_instance

theorem consistent_iff : d.toScoped.Consistent ↔
    (∀ a b, d.toScoped.hb a b → ¬ (a = b ∨ d.toScoped.eco b a)) ∧
    (∀ a b, d.toScoped.rmw a b → ¬ d.toScoped.eco b a) ∧
    (∀ a b c, d.toScoped.rmw a b → d.toScoped.fr a c → ¬ d.toScoped.co c b) ∧
    (∀ a, ¬ TransGen (d.toScoped.pscWith d.toScoped.hb) a a) ∧
    (∀ a, ¬ TransGen d.thinAirStep a a) :=
  ⟨fun h => ⟨h.1, h.2, h.3, h.4, h.5⟩, fun ⟨h1, h2, h3, h4, h5⟩ => ⟨h1, h2, h3, h4, h5⟩⟩

section
variable [hchain : DecidableRel (TransGen d.chainStep)]

instance : DecidableRel d.toScoped.rs := fun a b => by
  unfold ScopedExecution.rs
  have : ∀ x y, Decidable (TransGen d.toScoped.rsStep x y) := fun x y => hchain x y
  infer_instance

instance : DecidableRel d.toScoped.sw := fun a b => by
  unfold ScopedExecution.sw; infer_instance

instance : DecidableRel d.hbStep := fun a b => by unfold hbStep; infer_instance

variable [hhb : DecidableRel (TransGen d.hbStep)]
variable [heco : DecidableRel (TransGen d.ecoStep)]

instance : DecidableRel d.toScoped.hb := fun a b => hhb a b
instance : DecidableRel d.toScoped.eco := fun a b => heco a b

instance : DecidableRel (d.toScoped.pscWith d.toScoped.hb) := fun a b => by
  unfold Execution.pscWith Execution.scbWith; infer_instance

instance : DecidableRel d.toScoped.DataRace := fun a b => by
  unfold ScopedExecution.DataRace Execution.Conflict; infer_instance

instance : Decidable d.toScoped.RaceFree := by
  unfold ScopedExecution.RaceFree; infer_instance

variable [hpsc : DecidableRel (TransGen (d.toScoped.pscWith d.toScoped.hb))]
variable [hair : DecidableRel (TransGen d.thinAirStep)]

instance : Decidable d.toScoped.Consistent := by
  have : ∀ a, Decidable (TransGen d.thinAirStep a a) := fun a => hair a a
  exact decidable_of_iff _ (consistent_iff d).symm

end

/-- Whether the model with scopes allows the execution, and whether the execution has a data
race. `none` when a closure did not converge. -/
def check : Option (Bool × Bool) := do
  let ⟨_, hchain⟩ ← certify d.chainStep
  have : DecidableRel (TransGen d.chainStep) := fun a b => decidable_of_iff _ (hchain a b)
  let ⟨_, hhb⟩ ← certify d.hbStep
  have : DecidableRel (TransGen d.hbStep) := fun a b => decidable_of_iff _ (hhb a b)
  let ⟨_, heco⟩ ← certify d.ecoStep
  have : DecidableRel (TransGen d.ecoStep) := fun a b => decidable_of_iff _ (heco a b)
  let ⟨_, hpsc⟩ ← certify (d.toScoped.pscWith d.toScoped.hb)
  have : DecidableRel (TransGen (d.toScoped.pscWith d.toScoped.hb)) := fun a b =>
    decidable_of_iff _ (hpsc a b)
  let ⟨_, hair⟩ ← certify d.thinAirStep
  have : DecidableRel (TransGen d.thinAirStep) := fun a b => decidable_of_iff _ (hair a b)
  return (decide d.toScoped.Consistent, !decide d.toScoped.RaceFree)

/-- Whether the accesses `a` and `b` race. `none` when a closure did not converge. -/
def raceOn (a b : Fin n) : Option Bool := do
  let ⟨_, hchain⟩ ← certify d.chainStep
  have : DecidableRel (TransGen d.chainStep) := fun a b => decidable_of_iff _ (hchain a b)
  let ⟨_, hhb⟩ ← certify d.hbStep
  have : DecidableRel (TransGen d.hbStep) := fun a b => decidable_of_iff _ (hhb a b)
  return decide (d.toScoped.DataRace a b)

end SData

end Vx
