import VxMemoryModel.Rc11
import VxMemoryModel.Finite
import VxMemoryModel.Program

/-!
# Counters

A counter is a place that one write sets to 0, and that each other write increments by a
`fetchAdd` of 1. A channel gives each `send` a number from such a counter. `tickets` shows that
two `fetchAdd`s never read the same value, so each `send` gets its own number.

This needs the rule of RC11 that `co` orders the writes to each place (`CoOrder`). Without it,
two read-modify-writes can read from the same write.
-/

namespace Vx.Execution

open Relation Classical

variable {X : Execution}

/-- `co` is transitive, and it orders each two writes to one place. -/
structure CoOrder (X : Execution) : Prop where
  trans : ∀ a b c, X.co a b → X.co b c → X.co a c
  total : ∀ a b, X.IsWrite a → X.IsWrite b → X.loc a = X.loc b → a ≠ b → X.co a b ∨ X.co b a

/-- A read-modify-write comes after the write that it reads from, in `co`. -/
theorem rmw_source_co (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) {s r w : X.E}
    (hs : X.rf s r) (hrw : X.rmw r w) (hloc : X.loc r = X.loc w) : X.co s w := by
  obtain ⟨hsw, -, hsl⟩ := wf.rf_write_read s r hs
  obtain ⟨-, hww, -, -⟩ := wf.rmw_atomic r w hrw
  have hpo := wf.rmw_po r w hrw
  have hne : s ≠ w := by
    rintro rfl
    exact hc.no_thin_air r (TransGen.tail (TransGen.single (Or.inl hpo)) (Or.inr hs))
  rcases ho.total s w hsw hww (hsl.trans hloc) hne with h | h
  · exact h
  · exact (hc.coherence r w (TransGen.single (Or.inl hpo))
      (Or.inr (TransGen.tail (TransGen.single (Or.inr (Or.inl h))) (Or.inl hs)))).elim

/-- No write comes between a read-modify-write and the write that it reads from, in `co`. -/
theorem rmw_between (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) {s r w c : X.E}
    (hs : X.rf s r) (hrw : X.rmw r w) (hloc : X.loc r = X.loc w) (hcw : X.co c w) :
    c = s ∨ X.co c s := by
  by_cases hcs : c = s
  · exact Or.inl hcs
  obtain ⟨hcw', -, hcl⟩ := wf.co_writes c w hcw
  obtain ⟨hsw, -, hsl⟩ := wf.rf_write_read s r hs
  rcases ho.total c s hcw' hsw (by rw [hcl, hsl, hloc]) hcs with h | h
  · exact Or.inr h
  · exact absurd hcw (hc.atomicity r w c hrw ⟨s, hs, h⟩)

/-- Two read-modify-writes never read from the same write. -/
theorem rmw_source_unique (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hrmw : ∀ r1 r2 w, X.rmw r1 w → X.rmw r2 w → r1 = r2) {s r1 r2 w1 w2 : X.E}
    (h1 : X.rf s r1) (hrw1 : X.rmw r1 w1) (hl1 : X.loc r1 = X.loc w1)
    (h2 : X.rf s r2) (hrw2 : X.rmw r2 w2) (hl2 : X.loc r2 = X.loc w2) : r1 = r2 := by
  have hc1 := rmw_source_co wf hc ho h1 hrw1 hl1
  have hc2 := rmw_source_co wf hc ho h2 hrw2 hl2
  by_cases hw : w1 = w2
  · subst hw
    exact hrmw r1 r2 w1 hrw1 hrw2
  obtain ⟨-, hww1, hcl1⟩ := wf.co_writes s w1 hc1
  obtain ⟨-, hww2, hcl2⟩ := wf.co_writes s w2 hc2
  rcases ho.total w1 w2 hww1 hww2 (hcl1.symm.trans hcl2) hw with h | h
  · exact absurd h (hc.atomicity r2 w2 w1 hrw2 ⟨s, h2, hc1⟩)
  · exact absurd h (hc.atomicity r1 w1 w2 hrw1 ⟨s, h1, hc2⟩)

/-- The facts that make the place `x` a counter, with the values `val` of the events. The write
`i` sets it to 0, is not a read-modify-write, and happens before each other write to it. Each other write to it is the write
of a read-modify-write of `x` that adds 1. -/
structure Counter (X : Execution) (val : X.E → Int) (x : Nat) (i : X.E) : Prop where
  init_write : X.IsWrite i
  init_loc : X.loc i = x
  init_val : val i = 0
  init_not_rmw : ∀ r, ¬ X.rmw r i
  init_hb : ∀ w, X.IsWrite w → X.loc w = x → w ≠ i → X.hb i w
  add : ∀ w, X.IsWrite w → X.loc w = x → w ≠ i → ∃ r, X.rmw r w ∧ X.loc r = x ∧ val w = val r + 1
  rf_val : ∀ w r, X.rf w r → val w = val r
  rf_source : ∀ r, X.IsRead r → ∃ w, X.rf w r
  rmw_unique : ∀ r1 r2 w, X.rmw r1 w → X.rmw r2 w → r1 = r2

/-- The value of a counter grows along `co`. -/
theorem Counter.mono (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite)
    {val : X.E → Int} {x : Nat} {i : X.E} (hC : X.Counter val x i) :
    ∀ a b, X.IsWrite a → X.IsWrite b → X.loc a = x → X.loc b = x → X.co a b → val a < val b := by
  obtain ⟨l, hl⟩ := hfin
  have key : ∀ n b, count l (X.co · b) ≤ n → ∀ a, X.IsWrite a → X.IsWrite b → X.loc a = x →
      X.loc b = x → X.co a b → val a < val b := by
    intro n
    induction n with
    | zero =>
      intro b hn a _ _ _ _ hab
      have := count_pos (p := (X.co · b)) (hl a) hab
      omega
    | succ n ih =>
      intro b hn a ha hb hax hbx hab
      by_cases hbi : b = i
      · subst hbi
        have hai : a ≠ b := fun h => wf.co_irrefl a (h ▸ hab)
        exact (hc.coherence b a (hC.init_hb a ha hax hai)
          (Or.inr (TransGen.single (Or.inr (Or.inl hab))))).elim
      obtain ⟨r, hrb, hrx, hvb⟩ := hC.add b hb hbx hbi
      obtain ⟨s, hs⟩ := hC.rf_source r (wf.rmw_atomic r b hrb).1
      have hsb := rmw_source_co wf hc ho hs hrb (hrx.trans hbx.symm)
      have hvs := hC.rf_val s r hs
      obtain ⟨hsw, -, hsx⟩ := wf.co_writes s b hsb
      rcases rmw_between wf hc ho hs hrb (hrx.trans hbx.symm) hab with rfl | has
      · omega
      · have hlt : count l (X.co · s) < count l (X.co · b) :=
          count_lt (fun c h => ho.trans c s b h hsb) (hl s) hsb (wf.co_irrefl s)
        have := ih s (by omega) a ha hsw hax (hsx.trans hbx) has
        omega
  exact fun a b => key _ b (Nat.le_refl _) a

/-- Two read-modify-writes of a counter never read the same value. -/
theorem Counter.tickets (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {val : X.E → Int} {x : Nat} {i : X.E} (hC : X.Counter val x i)
    {r1 r2 w1 w2 : X.E} (hrw1 : X.rmw r1 w1) (hw1 : X.loc w1 = x) (hrw2 : X.rmw r2 w2)
    (hw2 : X.loc w2 = x) (hv : val r1 = val r2) : r1 = r2 := by
  have hloc {r w} (hrw : X.rmw r w) (hw : X.loc w = x) : X.loc r = X.loc w := by
    have hww := (wf.rmw_atomic r w hrw).2.1
    have hwi : w ≠ i := by
      rintro rfl
      exact hC.init_not_rmw r hrw
    obtain ⟨r', hr', hr'x, -⟩ := hC.add w hww hw hwi
    rw [← hC.rmw_unique r' r w hr' hrw, hr'x, hw]
  obtain ⟨s1, hs1⟩ := hC.rf_source r1 (wf.rmw_atomic r1 w1 hrw1).1
  obtain ⟨s2, hs2⟩ := hC.rf_source r2 (wf.rmw_atomic r2 w2 hrw2).1
  have hl1 := hloc hrw1 hw1
  have hl2 := hloc hrw2 hw2
  have hsv : val s1 = val s2 := by rw [hC.rf_val s1 r1 hs1, hC.rf_val s2 r2 hs2, hv]
  by_cases hs : s1 = s2
  · subst hs
    exact rmw_source_unique wf hc ho hC.rmw_unique hs1 hrw1 hl1 hs2 hrw2 hl2
  obtain ⟨hsw1, -, hsl1⟩ := wf.rf_write_read s1 r1 hs1
  obtain ⟨hsw2, -, hsl2⟩ := wf.rf_write_read s2 r2 hs2
  have hx1 : X.loc s1 = x := by rw [hsl1, hl1, hw1]
  have hx2 : X.loc s2 = x := by rw [hsl2, hl2, hw2]
  rcases ho.total s1 s2 hsw1 hsw2 (hx1.trans hx2.symm) hs with h | h
  · have := hC.mono wf hc ho hfin s1 s2 hsw1 hsw2 hx1 hx2 h
    omega
  · have := hC.mono wf hc ho hfin s2 s1 hsw2 hsw1 hx2 hx1 h
    omega

end Vx.Execution
