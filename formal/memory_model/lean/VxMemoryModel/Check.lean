import VxMemoryModel.Rc11

/-!
# Deciding the model for a small execution

`check` says whether the model allows an execution with `n` events, and whether it has a data
race. It decides the definitions of `Rc11.lean` themselves, so its answer is correct by
construction. `run_lean.py` uses it to compare the Lean model with herd7 on the litmus tests.

A transitive closure is computed as a table, by squaring. `certify` then checks that the table
contains the relation and is transitive. With `closure_sound`, that proves the table equals the
transitive closure. If the check fails, `check` gives no answer.
-/

namespace Vx

open Relation

/-- An execution with `n` events, given as tables. -/
structure Data (n : Nat) where
  kind : Fin n → Kind
  order : Fin n → Order
  loc : Fin n → Nat
  thread : Fin n → Nat
  po : Fin n → Fin n → Bool
  rf : Fin n → Fin n → Bool
  co : Fin n → Fin n → Bool
  rmw : Fin n → Fin n → Bool

/-- An execution from lists: one entry for each event, and the edges as pairs of indices. -/
def Data.ofLists (n : Nat) (kind : List Kind) (order : List Order) (loc thread : List Nat)
    (po rf co rmw : List (Nat × Nat)) : Data n where
  kind i := kind.getD i.val .fence
  order i := order.getD i.val .plain
  loc i := loc.getD i.val 0
  thread i := thread.getD i.val 0
  po a b := po.contains (a.val, b.val)
  rf a b := rf.contains (a.val, b.val)
  co a b := co.contains (a.val, b.val)
  rmw a b := rmw.contains (a.val, b.val)

abbrev Data.toExecution (d : Data n) : Execution where
  E := Fin n
  kind := d.kind
  order := d.order
  loc := d.loc
  thread := d.thread
  po a b := d.po a b = true
  rf a b := d.rf a b = true
  co a b := d.co a b = true
  rmw a b := d.rmw a b = true

/-- A relation on `n` events, stored as an array so that each entry is computed once. -/
structure Table (n : Nat) where
  rows : Array (Array Bool)

def Table.get (c : Table n) (a b : Fin n) : Bool := ((c.rows[a.val]?).bind (·[b.val]?)).getD false

def Table.ofFn (f : Fin n → Fin n → Bool) : Table n :=
  ⟨Array.ofFn fun a => Array.ofFn fun b => f a b⟩

theorem Table.get_ofFn (f : Fin n → Fin n → Bool) (a b : Fin n) :
    (Table.ofFn f).get a b = f a b := by
  simp [Table.get, Table.ofFn]

def Table.square (c : Table n) : Table n :=
  Table.ofFn fun a b => c.get a b || decide (∃ m, c.get a m = true ∧ c.get m b = true)

def Table.iterate (c : Table n) : Nat → Table n
  | 0 => c
  | k + 1 => (c.iterate k).square

theorem Table.iterate_sound {R : Fin n → Fin n → Prop} {c : Table n}
    (hc : ∀ a b, c.get a b = true → TransGen R a b) :
    ∀ k a b, (c.iterate k).get a b = true → TransGen R a b := by
  intro k
  induction k with
  | zero => exact hc
  | succ k ih =>
    intro a b h
    simp only [Table.iterate, Table.square, Table.get_ofFn, Bool.or_eq_true,
      decide_eq_true_eq] at h
    rcases h with h | ⟨m, ham, hmb⟩
    · exact ih a b h
    · exact (ih a m ham).trans (ih m b hmb)

theorem closure_complete {R : Fin n → Fin n → Prop} {c : Table n}
    (hsub : ∀ a b, R a b → c.get a b = true)
    (htrans : ∀ a m b, c.get a m = true → c.get m b = true → c.get a b = true) {a b : Fin n}
    (h : TransGen R a b) : c.get a b = true := by
  induction h with
  | single h => exact hsub _ _ h
  | tail _ h ih => exact htrans _ _ _ ih (hsub _ _ h)

/-- The transitive closure of `R` as a table, if `n` rounds of squaring reach it. -/
def certify (R : Fin n → Fin n → Prop) [DecidableRel R] :
    Option {c : Table n // ∀ a b, c.get a b = true ↔ TransGen R a b} :=
  let base : Table n := Table.ofFn fun a b => decide (R a b)
  let c := base.iterate n
  if h : (∀ a b, R a b → c.get a b = true) ∧
      (∀ a m b, c.get a m = true → c.get m b = true → c.get a b = true)
  then
    some ⟨c, fun a b => ⟨Table.iterate_sound (fun a b hab => by
      simp only [base, Table.get_ofFn, decide_eq_true_eq] at hab
      exact TransGen.single hab) n a b, closure_complete h.1 h.2⟩⟩
  else none

namespace Execution

variable (X : Execution)

instance (e : X.E) : Decidable (X.IsRead e) := inferInstanceAs (Decidable (_ = _))
instance (e : X.E) : Decidable (X.IsWrite e) := inferInstanceAs (Decidable (_ = _))
instance (e : X.E) : Decidable (X.IsFence e) := inferInstanceAs (Decidable (_ = _))
instance (e : X.E) : Decidable (X.IsAtomic e) := inferInstanceAs (Decidable (¬ _ = _))
instance (e : X.E) : Decidable (X.IsAcquire e) := by unfold IsAcquire; infer_instance
instance (e : X.E) : Decidable (X.IsRelease e) := by unfold IsRelease; infer_instance
instance (e : X.E) : Decidable (X.IsSeqCst e) := inferInstanceAs (Decidable (_ = _))
instance (e : X.E) : Decidable (X.IsAccess e) := by unfold IsAccess; infer_instance

theorem consistent_iff : X.Consistent ↔
    (∀ a b, X.hb a b → ¬ (a = b ∨ X.eco b a)) ∧ (∀ a b, X.rmw a b → ¬ X.eco b a) ∧
    (∀ a b c, X.rmw a b → X.fr a c → ¬ X.co c b) ∧ (∀ a, ¬ TransGen X.psc a a) ∧
    (∀ a, ¬ TransGen (fun a b => X.po a b ∨ X.rf a b) a a) :=
  ⟨fun h => ⟨h.1, h.2, h.3, h.4, h.5⟩, fun ⟨h1, h2, h3, h4, h5⟩ => ⟨h1, h2, h3, h4, h5⟩⟩

end Execution

namespace Data

variable (d : Data n)

instance : DecidableEq d.toExecution.E := inferInstanceAs (DecidableEq (Fin n))
instance {P : d.toExecution.E → Prop} [DecidablePred P] : Decidable (∃ e, P e) :=
  Nat.decidableExistsFin P
instance {P : d.toExecution.E → Prop} [DecidablePred P] : Decidable (∀ e, P e) :=
  Nat.decidableForallFin P
instance : DecidableRel d.toExecution.po := fun a b => inferInstanceAs (Decidable (d.po a b = true))
instance : DecidableRel d.toExecution.rf := fun a b => inferInstanceAs (Decidable (d.rf a b = true))
instance : DecidableRel d.toExecution.co := fun a b => inferInstanceAs (Decidable (d.co a b = true))
instance : DecidableRel d.toExecution.rmw := fun a b => inferInstanceAs (Decidable (d.rmw a b = true))
instance : DecidableRel d.toExecution.fr := fun a b =>
  inferInstanceAs (Decidable (∃ w : Fin n, d.rf w a = true ∧ d.co w b = true))

def chainStep (x y : Fin n) : Prop := ∃ r, d.toExecution.rf x r ∧ d.toExecution.rmw r y
def hbStep (a b : Fin n) : Prop := d.toExecution.po a b ∨ d.toExecution.sw a b
def ecoStep (a b : Fin n) : Prop := d.toExecution.rf a b ∨ d.toExecution.co a b ∨ d.toExecution.fr a b
def thinAirStep (a b : Fin n) : Prop := d.toExecution.po a b ∨ d.toExecution.rf a b

instance : DecidableRel d.chainStep := fun x y => by unfold chainStep; infer_instance

section
variable [hchain : DecidableRel (TransGen d.chainStep)]

instance : DecidableRel d.toExecution.rs := fun a b => by
  unfold Execution.rs
  have : ∀ x y, Decidable
      (TransGen (fun x y => ∃ r, d.toExecution.rf x r ∧ d.toExecution.rmw r y) x y) :=
    fun x y => hchain x y
  infer_instance

instance : DecidableRel d.toExecution.sw := fun a b => by unfold Execution.sw; infer_instance

instance : DecidableRel d.hbStep := fun a b => by unfold hbStep; infer_instance

variable [hhb : DecidableRel (TransGen d.hbStep)]
variable [heco : DecidableRel (TransGen d.ecoStep)]

instance : DecidableRel d.toExecution.hb := fun a b => hhb a b
instance : DecidableRel d.toExecution.eco := fun a b => heco a b

instance : DecidableRel d.toExecution.scb := fun a b => by unfold Execution.scb; infer_instance
instance : DecidableRel d.toExecution.psc := fun a b => by
  unfold Execution.psc Execution.pscb Execution.pscf; infer_instance

instance : Decidable d.toExecution.RaceFree := by
  unfold Execution.RaceFree Execution.DataRace Execution.Conflict; infer_instance

variable [hpsc : DecidableRel (TransGen d.toExecution.psc)]
variable [hair : DecidableRel (TransGen d.thinAirStep)]

instance : Decidable d.toExecution.Consistent := by
  have : ∀ a, Decidable (TransGen (fun a b => d.toExecution.po a b ∨ d.toExecution.rf a b) a a) := fun a => hair a a
  exact decidable_of_iff _ (Execution.consistent_iff d.toExecution).symm

end

instance : DecidableRel d.ecoStep := fun a b => by unfold ecoStep; infer_instance
instance : DecidableRel d.thinAirStep := fun a b => by unfold thinAirStep; infer_instance

/-- Whether the model allows the execution, and whether the execution has a data race. `none`
when a closure did not converge. -/
def check : Option (Bool × Bool) := do
  let ⟨_, hchain⟩ ← certify d.chainStep
  have : DecidableRel (TransGen d.chainStep) := fun a b => decidable_of_iff _ (hchain a b)
  let ⟨_, hhb⟩ ← certify d.hbStep
  have : DecidableRel (TransGen d.hbStep) := fun a b => decidable_of_iff _ (hhb a b)
  let ⟨_, heco⟩ ← certify d.ecoStep
  have : DecidableRel (TransGen d.ecoStep) := fun a b => decidable_of_iff _ (heco a b)
  let ⟨_, hpsc⟩ ← certify d.toExecution.psc
  have : DecidableRel (TransGen d.toExecution.psc) := fun a b => decidable_of_iff _ (hpsc a b)
  let ⟨_, hair⟩ ← certify d.thinAirStep
  have : DecidableRel (TransGen d.thinAirStep) := fun a b => decidable_of_iff _ (hair a b)
  return (decide d.toExecution.Consistent, !decide d.toExecution.RaceFree)

end Data

end Vx
