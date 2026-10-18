import VxMemoryModel.Thread
import VxMemoryModel.Program

/-!
# Programs of the core language

A program gives the code of each thread. Thread 0 runs first, and the other threads start when
it ends, so thread 0 can write the initial values.

An execution of a program gives each event a value and a position in its thread. The events of
each thread, in order, must be a run of its code, and each read must read a write of the same
value. The write of a read-modify-write comes right after its read, and `rmw` joins them. This
file shows that these executions have the three properties of `Program`. So DRF-SC holds for
each program whose atomics are sequentially consistent and which has no fences.
-/

namespace Vx

open Classical

structure Prog where
  code : Nat → Stmt

namespace Prog

/-- The facts that make `X` an execution of `P`. -/
structure Run (P : Prog) (X : Execution) where
  val : X.E → Int
  /-- True for the write of a read-modify-write. -/
  isRmw : X.E → Bool
  /-- The position of an event in its thread. -/
  idx : X.E → Nat
  /-- The accesses of each thread, in order. -/
  lab : Nat → Nat → Label
  po_iff : ∀ a b, X.po a b ↔
    (X.thread a = X.thread b ∧ idx a < idx b) ∨ (X.thread a = 0 ∧ X.thread b ≠ 0)
  idx_inj : ∀ a b, X.thread a = X.thread b → idx a = idx b → a = b
  /-- The positions of a thread have no gaps. -/
  idx_down : ∀ e i, i < idx e → ∃ e', X.thread e' = X.thread e ∧ idx e' = i
  lab_eq : ∀ e, lab (X.thread e) (idx e) = ⟨X.kind e, X.order e, X.loc e, val e, isRmw e⟩
  /-- The accesses of a thread up to each event are a run of its code. -/
  trace : ∀ e, Trace (P.code (X.thread e)) (idx e + 1) (lab (X.thread e))
  rf_val : ∀ w r, X.rf w r → val w = val r
  rf_write_read : ∀ w r, X.rf w r → X.IsWrite w ∧ X.IsRead r ∧ X.loc w = X.loc r
  rf_source : ∀ r, X.IsRead r → ∃ w, X.rf w r
  rmw_iff : ∀ a b, X.rmw a b ↔ X.thread a = X.thread b ∧ idx b = idx a + 1 ∧ isRmw b = true

def Execs (P : Prog) (X : Execution) : Prop := Nonempty (P.Run X)

variable {P : Prog}

theorem Run.prefix_closed {X : Execution} (r : P.Run X) {S : X.E → Prop} (hS : X.Downclosed S) :
    P.Execs (X.restrict S) := ⟨{
  val := fun e => r.val e.val
  isRmw := fun e => r.isRmw e.val
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
  rmw_iff := fun a b => r.rmw_iff a.val b.val }⟩

theorem Run.any_co {X : Execution} (r : P.Run X) (co : X.E → X.E → Prop) :
    P.Execs (X.withEdges X.rf co) := ⟨{
  val := r.val
  isRmw := r.isRmw
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
  rmw_iff := r.rmw_iff }⟩

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
  have hnew : (⟨.read, X.order rd, X.loc rd, r.val w, r.isRmw rd⟩ : Label) =
      { r.lab (X.thread rd) (r.idx rd) with val := r.val w } := by
    rw [r.lab_eq rd]
    unfold Execution.IsRead at hr
    rw [hr]
  exact ⟨{
    val := fun e => if e = rd then r.val w else r.val e
    isRmw := r.isRmw
    idx := r.idx
    lab := fun t i => if t = X.thread rd ∧ i = r.idx rd then
      ⟨.read, X.order rd, X.loc rd, r.val w, r.isRmw rd⟩ else r.lab t i
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
    rmw_iff := r.rmw_iff }⟩

def toProgram (P : Prog) : Program where
  Execs := P.Execs
  prefix_closed _ _ hX hS := let ⟨r⟩ := hX; r.prefix_closed hS
  any_co _ co hX := let ⟨r⟩ := hX; r.any_co co
  receptive _ _ _ hX hr hlast hw hl := let ⟨r⟩ := hX; r.receptive hr hlast hw hl

/-- Each thread follows `Stmt.SeqCstOnly`: only sequentially consistent atomics, no fences. -/
def SeqCstOnly (P : Prog) : Prop := ∀ t, (P.code t).SeqCstOnly

theorem Run.seqCst {X : Execution} (r : P.Run X) (hP : P.SeqCstOnly) (e : X.E) :
    X.kind e ≠ .fence ∧ (X.order e = .plain ∨ X.order e = .sc) := by
  have h := seqCstRule.trace (r.trace e) (hP _)
  rw [r.lab_eq e] at h
  exact h

/-- DRF-SC for the core language: take a program whose atomics are sequentially consistent and
which has no fences. If no interleaving of it has a data race, then each finite
execution that the model allows is an interleaving. -/
theorem drf_sc (P : Prog) (hsc : P.SeqCstOnly)
    (hP : ∀ Y, P.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.RaceFree)
    {X : Execution} (hX : P.Execs X) (wf : X.WellFormed) (hfin : X.Finite)
    (hc : X.Consistent) : X.SeqConsistent := by
  obtain ⟨r⟩ := hX
  refine P.toProgram.drf_sc hP ⟨r⟩ wf hfin ?_ ?_ hc
  · intro e he
    rcases (r.seqCst hsc e).2 with h | h
    · exact absurd h he
    · exact h
  · exact fun e => (r.seqCst hsc e).1

end Prog

end Vx
