import VxMemoryModel.Thread
import VxMemoryModel.Program

/-!
# Programs of the core language

A program gives the code of each thread. Thread 0 runs first, and the other threads start when
it ends, so thread 0 can write the initial values.

An execution of a program gives each event a value and a position in its thread. The events of
each thread, in order, must be a run of its code, and each read must read a write of the same
value. This file shows that these executions have the three properties of `Program`, so DRF-SC
holds for every program of the language.
-/

namespace Vx

open Classical

structure Prog where
  /-- Which places are atomic. -/
  atomic : Nat → Bool
  code : Nat → Stmt

namespace Prog

/-- The facts that make `X` an execution of `P`. -/
structure Run (P : Prog) (X : Execution) where
  val : X.E → Int
  /-- The position of an event in its thread. -/
  idx : X.E → Nat
  /-- The accesses of each thread, in order. -/
  lab : Nat → Nat → Label
  po_iff : ∀ a b, X.po a b ↔
    (X.thread a = X.thread b ∧ idx a < idx b) ∨ (X.thread a = 0 ∧ X.thread b ≠ 0)
  idx_inj : ∀ a b, X.thread a = X.thread b → idx a = idx b → a = b
  /-- The positions of a thread have no gaps. -/
  idx_down : ∀ e i, i < idx e → ∃ e', X.thread e' = X.thread e ∧ idx e' = i
  lab_eq : ∀ e, lab (X.thread e) (idx e) = ⟨X.kind e, X.order e, X.loc e, val e⟩
  /-- The accesses of a thread up to each event are a run of its code. -/
  trace : ∀ e, Trace P.atomic (P.code (X.thread e)) (idx e + 1) (lab (X.thread e))
  rf_val : ∀ w r, X.rf w r → val w = val r
  rf_write_read : ∀ w r, X.rf w r → X.IsWrite w ∧ X.IsRead r ∧ X.loc w = X.loc r
  rf_source : ∀ r, X.IsRead r → ∃ w, X.rf w r
  no_rmw : ∀ a b, ¬ X.rmw a b

def Execs (P : Prog) (X : Execution) : Prop := Nonempty (P.Run X)

variable {P : Prog}

theorem Run.prefix_closed {X : Execution} (r : P.Run X) {S : X.E → Prop} (hS : X.Downclosed S) :
    P.Execs (X.restrict S) := ⟨{
  val := fun e => r.val e.val
  idx := fun e => r.idx e.val
  lab := r.lab
  po_iff := fun a b => r.po_iff a.val b.val
  idx_inj := fun a b ht hi => Subtype.ext (r.idx_inj _ _ ht hi)
  idx_down := fun e i hi => by
    obtain ⟨e', ht, hi'⟩ := r.idx_down e.val i hi
    have hp : X.po e' e.val := (r.po_iff _ _).2 (Or.inl ⟨ht, hi' ▸ hi⟩)
    exact ⟨⟨e', hS _ _ e.property (Or.inl hp)⟩, ht, hi'⟩
  lab_eq := fun e => r.lab_eq e.val
  trace := fun e => r.trace e.val
  rf_val := fun w r' h => r.rf_val w.val r'.val h
  rf_write_read := fun w r' h => r.rf_write_read w.val r'.val h
  rf_source := fun r' hr => by
    obtain ⟨w, hw⟩ := r.rf_source r'.val hr
    exact ⟨⟨w, hS _ _ r'.property (Or.inr hw)⟩, hw⟩
  no_rmw := fun a b => r.no_rmw a.val b.val }⟩

theorem Run.any_co {X : Execution} (r : P.Run X) (co : X.E → X.E → Prop) :
    P.Execs (X.withEdges X.rf co) := ⟨{
  val := r.val
  idx := r.idx
  lab := r.lab
  po_iff := r.po_iff
  idx_inj := r.idx_inj
  idx_down := r.idx_down
  lab_eq := r.lab_eq
  trace := r.trace
  rf_val := r.rf_val
  rf_write_read := r.rf_write_read
  rf_source := r.rf_source
  no_rmw := r.no_rmw }⟩

theorem Run.receptive {X : Execution} (r : P.Run X) {rd w : X.E} (hr : X.IsRead rd)
    (hlast : ∀ x, ¬ X.po rd x) (hw : X.IsWrite w) (hl : X.loc w = X.loc rd) :
    P.Execs (X.withEdges (X.readsFrom rd w) X.co) := by
  have hne : ∀ e, X.IsWrite e → e ≠ rd := by
    rintro e he rfl
    unfold Execution.IsWrite at he
    unfold Execution.IsRead at hr
    rw [hr] at he
    cases he
  -- Each event of the thread of `rd` comes before it or is `rd`.
  have hle : ∀ e, X.thread e = X.thread rd → r.idx e ≤ r.idx rd := fun e ht =>
    Nat.le_of_not_lt fun h => hlast e ((r.po_iff _ _).2 (Or.inl ⟨ht.symm, h⟩))
  have hnew : (⟨.read, X.order rd, X.loc rd, r.val w⟩ : Label) =
      { r.lab (X.thread rd) (r.idx rd) with val := r.val w } := by
    rw [r.lab_eq rd]
    unfold Execution.IsRead at hr
    rw [hr]
  exact ⟨{
    val := fun e => if e = rd then r.val w else r.val e
    idx := r.idx
    lab := fun t i => if t = X.thread rd ∧ i = r.idx rd then ⟨.read, X.order rd, X.loc rd, r.val w⟩
      else r.lab t i
    po_iff := r.po_iff
    idx_inj := r.idx_inj
    idx_down := r.idx_down
    lab_eq := fun e => by
      by_cases he : e = rd
      · rw [he]
        unfold Execution.IsRead at hr
        simp [hr]
      · have : ¬ (X.thread e = X.thread rd ∧ r.idx e = r.idx rd) := fun ⟨ht, hi⟩ =>
          he (r.idx_inj _ _ ht hi)
        simp only [this, he, ite_false]
        exact r.lab_eq e
    trace := fun e => by
      by_cases ht : X.thread e = X.thread rd
      · rcases Nat.lt_or_eq_of_le (hle e ht) with hi | hi
        · refine (r.trace e).congr fun i hi' => ?_
          have : ¬ (X.thread e = X.thread rd ∧ i = r.idx rd) := fun ⟨_, h⟩ => by omega
          simp only [this, ite_false]
        · have he : e = rd := r.idx_inj _ _ ht hi
          rw [he]
          have hk : (r.lab (X.thread rd) (r.idx rd)).kind = .read := by rw [r.lab_eq]; exact hr
          refine ((r.trace rd).receptive hk (r.val w)).congr fun i _ => ?_
          by_cases hi : i = r.idx rd
          · rw [hi]
            simp [hnew]
          · simp [hi]
      · refine (r.trace e).congr fun i _ => ?_
        have : ¬ (X.thread e = X.thread rd ∧ i = r.idx rd) := fun ⟨h, _⟩ => ht h
        simp only [this, ite_false]
    rf_val := fun a b h => by
      rcases h with ⟨hb, ha⟩ | ⟨hb, h⟩
      · rw [ha, hb]
        simp [hne _ hw]
      · have ha := hne _ (r.rf_write_read _ _ h).1
        simp only [ha, hb, ite_false]
        exact r.rf_val _ _ h
    rf_write_read := fun a b h => by
      rcases h with ⟨hb, ha⟩ | ⟨_, h⟩
      · rw [ha, hb]
        exact ⟨hw, hr, hl⟩
      · exact r.rf_write_read _ _ h
    rf_source := fun b hb => by
      by_cases h : b = rd
      · exact ⟨w, Or.inl ⟨h, rfl⟩⟩
      · obtain ⟨a, ha⟩ := r.rf_source b hb
        exact ⟨a, Or.inr ⟨h, ha⟩⟩
    no_rmw := r.no_rmw }⟩

def toProgram (P : Prog) : Program where
  Execs := P.Execs
  prefix_closed _ _ hX hS := let ⟨r⟩ := hX; r.prefix_closed hS
  any_co _ co hX := let ⟨r⟩ := hX; r.any_co co
  receptive _ _ _ hX hr hlast hw hl := let ⟨r⟩ := hX; r.receptive hr hlast hw hl

theorem Run.typed {X : Execution} (r : P.Run X) (e : X.E) :
    X.kind e ≠ .fence ∧ X.order e = orderOf P.atomic (X.loc e) := by
  have h := (r.trace e).typed
  rw [r.lab_eq e] at h
  exact h

/-- DRF-SC for the core language: if no interleaving of a program has a data race, then each
finite execution that the model allows is an interleaving. -/
theorem drf_sc (P : Prog)
    (hP : ∀ Y, P.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.RaceFree)
    {X : Execution} (hX : P.Execs X) (wf : X.WellFormed) (hfin : X.Finite)
    (hc : X.Consistent) : X.SeqConsistent := by
  obtain ⟨r⟩ := hX
  refine P.toProgram.drf_sc hP ⟨r⟩ wf hfin ?_ ?_ ?_ hc
  · intro a b _ _ hl
    unfold Execution.IsAtomic
    rw [(r.typed a).2, (r.typed b).2, hl]
  · intro e he
    unfold Execution.IsAtomic at he
    unfold Execution.IsSeqCst
    rw [(r.typed e).2] at he ⊢
    unfold orderOf at he ⊢
    cases hA : P.atomic (X.loc e) <;> simp [hA] at he ⊢
  · exact fun e => (r.typed e).1

end Prog

end Vx
