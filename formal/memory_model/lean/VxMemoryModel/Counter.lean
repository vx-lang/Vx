import VxMemoryModel.Rc11
import VxMemoryModel.Finite
import VxMemoryModel.Program
import VxMemoryModel.Language

/-!
# Counters

A counter is a place that one write sets at the start, and that each other write increments by a
`fetchAdd` of 1. A channel gives each `send` a number from such a counter. `tickets` shows that
two `fetchAdd`s never read the same value, so each `send` gets its own number. The first write
need not happen before the others: `init_first` shows that it comes first in `co` anyway. When
the accesses are `sc`, `hb_of_lt` shows that a `fetchAdd` happens before each read of a larger
value.

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

/-- The facts that make the place `x` a counter, with the values `val` of the events. Each write
to `x` is the write of a read-modify-write of `x` that adds 1, except one write that sets it at the
start. -/
structure Counter (X : Execution) (val : X.E → Int) (x : Nat) : Prop where
  init_unique : ∀ a b, X.IsWrite a → X.IsWrite b → X.loc a = x → X.loc b = x →
    (∀ r, ¬ X.rmw r a) → (∀ r, ¬ X.rmw r b) → a = b
  add : ∀ r w, X.rmw r w → X.loc w = x → X.loc r = x ∧ val w = val r + 1
  rf_val : ∀ w r, X.rf w r → val w = val r
  rf_source : ∀ r, X.IsRead r → ∃ w, X.rf w r
  rmw_unique : ∀ r1 r2 w, X.rmw r1 w → X.rmw r2 w → r1 = r2

/-- No write comes before the write that sets a counter, in `co`. Each other write reads from a
write before it, so the first write in `co` cannot be one of them. -/
theorem Counter.init_first (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {val : X.E → Int} {x : Nat} (hC : X.Counter val x) {b : X.E}
    (hbw : X.IsWrite b) (hbx : X.loc b = x) (hbi : ∀ r, ¬ X.rmw r b) : ∀ a, ¬ X.co a b := by
  obtain ⟨l, hl⟩ := hfin
  have step : ∀ a, X.co a b → ∃ s, X.co s b ∧ count l (X.co · s) < count l (X.co · a) := by
    intro a hab
    obtain ⟨haw, -, hal⟩ := wf.co_writes a b hab
    by_cases hr : ∃ r, X.rmw r a
    · obtain ⟨r, hra⟩ := hr
      obtain ⟨hrx, -⟩ := hC.add r a hra (hal.trans hbx)
      obtain ⟨s, hs⟩ := hC.rf_source r (wf.rmw_atomic r a hra).1
      have hsa := rmw_source_co wf hc ho hs hra (hrx.trans (hal.trans hbx).symm)
      exact ⟨s, ho.trans s a b hsa hab,
        count_lt (fun c h => ho.trans c s a h hsa) (hl s) hsa (wf.co_irrefl s)⟩
    · have := hC.init_unique a b haw hbw (hal.trans hbx) hbx (fun r h => hr ⟨r, h⟩) hbi
      subst this
      exact (wf.co_irrefl a hab).elim
  have key : ∀ n a, count l (X.co · a) ≤ n → ¬ X.co a b := by
    intro n
    induction n with
    | zero =>
      intro a hn hab
      obtain ⟨s, -, hlt⟩ := step a hab
      omega
    | succ n ih =>
      intro a hn hab
      obtain ⟨s, hsb, hlt⟩ := step a hab
      exact ih s (by omega) hsb
  exact fun a => key _ a (Nat.le_refl _)

/-- The value of a counter grows along `co`. -/
theorem Counter.mono (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite)
    {val : X.E → Int} {x : Nat} (hC : X.Counter val x) :
    ∀ a b, X.IsWrite a → X.IsWrite b → X.loc a = x → X.loc b = x → X.co a b → val a < val b := by
  have hfin' := hfin
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
      by_cases hbi : ∃ r, X.rmw r b
      · obtain ⟨r, hrb⟩ := hbi
        obtain ⟨hrx, hvb⟩ := hC.add r b hrb hbx
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
      · exact (hC.init_first wf hc ho hfin' hb hbx (fun r h => hbi ⟨r, h⟩) a hab).elim
  exact fun a b => key _ b (Nat.le_refl _) a

/-- Two read-modify-writes of a counter never read the same value. -/
theorem Counter.tickets (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {val : X.E → Int} {x : Nat} (hC : X.Counter val x)
    {r1 r2 w1 w2 : X.E} (hrw1 : X.rmw r1 w1) (hw1 : X.loc w1 = x) (hrw2 : X.rmw r2 w2)
    (hw2 : X.loc w2 = x) (hv : val r1 = val r2) : r1 = r2 := by
  obtain ⟨hr1, -⟩ := hC.add r1 w1 hrw1 hw1
  obtain ⟨hr2, -⟩ := hC.add r2 w2 hrw2 hw2
  have hl1 : X.loc r1 = X.loc w1 := hr1.trans hw1.symm
  have hl2 : X.loc r2 = X.loc w2 := hr2.trans hw2.symm
  obtain ⟨s1, hs1⟩ := hC.rf_source r1 (wf.rmw_atomic r1 w1 hrw1).1
  obtain ⟨s2, hs2⟩ := hC.rf_source r2 (wf.rmw_atomic r2 w2 hrw2).1
  have hsv : val s1 = val s2 := by rw [hC.rf_val s1 r1 hs1, hC.rf_val s2 r2 hs2, hv]
  by_cases hs : s1 = s2
  · subst hs
    exact rmw_source_unique wf hc ho hC.rmw_unique hs1 hrw1 hl1 hs2 hrw2 hl2
  obtain ⟨hsw1, -, hsl1⟩ := wf.rf_write_read s1 r1 hs1
  obtain ⟨hsw2, -, hsl2⟩ := wf.rf_write_read s2 r2 hs2
  have hx1 : X.loc s1 = x := hsl1.trans hr1
  have hx2 : X.loc s2 = x := hsl2.trans hr2
  rcases ho.total s1 s2 hsw1 hsw2 (hx1.trans hx2.symm) hs with h | h
  · have := hC.mono wf hc ho hfin s1 s2 hsw1 hsw2 hx1 hx2 h
    omega
  · have := hC.mono wf hc ho hfin s2 s1 hsw2 hsw1 hx2 hx1 h
    omega

/-- On a counter with `sc` accesses, a `fetchAdd` happens before each read of a larger value. The
read gets its value from a `fetchAdd` that read one less, and so on down to the first one. -/
theorem Counter.hb_of_lt (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {val : X.E → Int} {x : Nat} (hC : X.Counter val x)
    (hsc : ∀ e, X.loc e = x → X.kind e ≠ .fence → X.IsSeqCst e) {r w : X.E} (hrw : X.rmw r w)
    (hwx : X.loc w = x) : ∀ e, X.IsRead e → X.loc e = x → val r < val e → X.hb r e := by
  obtain ⟨-, hvw⟩ := hC.add r w hrw hwx
  have hww := (wf.rmw_atomic r w hrw).2.1
  have key : ∀ n e, (val e - val r).toNat ≤ n → X.IsRead e → X.loc e = x → val r < val e →
      X.hb r e := by
    intro n
    induction n with
    | zero => intro e hn _ _ hlt; omega
    | succ n ih =>
      intro e hn he hex hlt
      obtain ⟨s, hs⟩ := hC.rf_source e he
      obtain ⟨hsw, -, hsl⟩ := wf.rf_write_read s e hs
      have hsx : X.loc s = x := hsl.trans hex
      have hvs := hC.rf_val s e hs
      have sig : ∀ a, a = s ∨ X.po a s → X.hb a e := fun a ha =>
        X.hb_of_signal ha hsw (hsc s hsx (by rw [hsw]; nofun)) hs (hsc e hex (by rw [he]; nofun))
          (Or.inl rfl)
      by_cases hws : w = s
      · subst hws
        exact sig r (Or.inr (wf.rmw_po r w hrw))
      rcases ho.total w s hww hsw (hwx.trans hsx.symm) hws with h | h
      · by_cases hbi : ∃ r', X.rmw r' s
        · obtain ⟨r', hr's⟩ := hbi
          obtain ⟨hr'x, hvs'⟩ := hC.add r' s hr's hsx
          have hm := hC.mono wf hc ho hfin w s hww hsw hwx hsx h
          have := ih r' (by omega) (wf.rmw_atomic r' s hr's).1 hr'x (by omega)
          exact this.trans (sig r' (Or.inr (wf.rmw_po r' s hr's)))
        · exact (hC.init_first wf hc ho hfin hsw hsx (fun r h => hbi ⟨r, h⟩) w h).elim
      · have := hC.mono wf hc ho hfin s w hsw hww hsx hwx h
        omega
  exact fun e => key _ e (Nat.le_refl _)

end Vx.Execution
