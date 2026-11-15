import VxMemoryModel.ReleaseSequence

/-!
# The orders of the counts of a broadcast channel

`stdlib/std/sync/broadcast.vx` gives each slot a count of the receivers that did not read its
value. A `send` sets the count with a store, and then sets the number of the slot with a release
store. A receiver reads the number with an acquire load, and then reads the count:

- When the count is 1, each other receiver read the value. The receiver reads the count with an
  acquire load, stores 0, and moves the value out.
- Else it clones the value and takes 1 from the count with an acquire-release read-modify-write.
  The receiver whose read-modify-write reads 1 drops the value.

The drop of a receiver also takes 1 from the count of each value that it did not read, with an
acquire-release read-modify-write, and drops the value when it reads 1.

The theorems here look at the count of one value: the writes from the store of its `send` up
to the last read. `last_rmw_hb` covers a last receiver that takes 1 with a read-modify-write.
`last_load_hb` covers a last receiver that reads 1 with a load and moves the value out. Each one
shows that the clone of another receiver happens before the drop or the move. The other
receiver must take 1 with a release write. The last receiver must read with an acquire read.
The drop of a receiver does not use the value before its read-modify-write, so the release part
of that write is not needed. `BroadcastExample.lean` shows a race for each weaker order.

The number of the slot holds a different value at each write, as in a `channel`. So
`ChannelOrders.lean` gives the edges from a `send` to each receiver and from the last receiver
to the next `send`. `tests/integration_test/library_model_check.rs` checks the algorithm with
sequential consistency: the count is the number of receivers that still read the value. The
hypotheses `hlive` and `honly` below state the part of that fact that the theorems use.
-/

namespace Vx.Execution

open Relation Classical

variable {X : Execution}

/-- A chain of read-modify-writes from `a` to `b`, when each write after `a` and up to `c` in
`co` is the write of a read-modify-write. `rmw_chain_of_co` asks this of each write. -/
theorem rmw_chain_up_to (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {x : Nat} {a c : X.E}
    (hrmw : ∀ y, X.IsWrite y → X.loc y = x → X.co a y → (y = c ∨ X.co y c) → ∃ r, X.rmw r y)
    (hsrc : ∀ r y, X.rmw r y → X.loc y = x → X.loc r = x ∧ ∃ s, X.rf s r) :
    ∀ b, X.loc b = x → X.co a b → (b = c ∨ X.co b c) →
      TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) a b := by
  obtain ⟨l, hl⟩ := hfin
  have key : ∀ n b, count l (X.co · b) ≤ n → X.loc b = x → X.co a b → (b = c ∨ X.co b c) →
      TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) a b := by
    intro n
    induction n with
    | zero =>
      intro b hn _ hab _
      have := count_pos (p := (X.co · b)) (hl a) hab
      omega
    | succ n ih =>
      intro b hn hbx hab hbc
      obtain ⟨-, hbw, -⟩ := wf.co_writes a b hab
      obtain ⟨r, hrb⟩ := hrmw b hbw hbx hab hbc
      obtain ⟨hrx, s, hs⟩ := hsrc r b hrb hbx
      have hloc : X.loc r = X.loc b := hrx.trans hbx.symm
      have hsb := rmw_source_co wf hc ho hs hrb hloc
      rcases rmw_between wf hc ho hs hrb hloc hab with rfl | has
      · exact TransGen.single ⟨r, hs, hrb⟩
      · have hsx : X.loc s = x := (wf.co_writes s b hsb).2.2.trans hbx
        have hsc : s = c ∨ X.co s c := by
          rcases hbc with rfl | hbc
          · exact Or.inr hsb
          · exact Or.inr (ho.trans s b c hsb hbc)
        have hlt := count_lt (fun c h => ho.trans c s b h hsb) (hl s) hsb (wf.co_irrefl s)
        exact (ih s (by omega) hsx has hsc).tail ⟨r, hs, hrb⟩
  exact fun b hbx hab hbc => key _ b (Nat.le_refl _) hbx hab hbc

/-- The count `x` of one value, where each write but the first is a read-modify-write and no
read-modify-write reads 0. Another receiver takes 1 with the release write `wd`. The last
receiver takes 1 with `rl`, `wl`: an acquire read of 1. Then each event before `wd` happens
before each event after `rl`, such as the drop of the value. -/
theorem last_rmw_hb (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {x : Nat} {val : X.E → Int}
    (hrmw : ∀ y, X.IsWrite y → X.loc y = x → (∃ a, X.co a y) → ∃ r, X.rmw r y)
    (hsrc : ∀ r y, X.rmw r y → X.loc y = x → X.loc r = x ∧ ∃ s, X.rf s r)
    (hval : ∀ w r, X.rf w r → val w = val r)
    (hlive : ∀ r y, X.rmw r y → X.loc r = x → val r ≠ 0)
    {rd wd : X.E} (hd : X.rmw rd wd) (hdx : X.loc wd = x) (hdrel : X.IsRelease wd)
    {rl wl : X.E} (hl : X.rmw rl wl) (hlx : X.loc wl = x) (hlacq : X.IsAcquire rl)
    (hl1 : val rl = 1) (hl0 : val wl = val rl - 1) (hne : wd ≠ wl)
    {a b : X.E} (ha : a = wd ∨ X.po a wd) (hb : rl = b ∨ X.po rl b) : X.hb a b := by
  have reach := rmw_chain_of_co wf hc ho hfin hrmw hsrc
  obtain ⟨-, hdw, -, hda⟩ := wf.rmw_atomic rd wd hd
  obtain ⟨-, hlw, -, -⟩ := wf.rmw_atomic rl wl hl
  obtain ⟨hrx, y, hy⟩ := hsrc rl wl hl hlx
  have hloc : X.loc rl = X.loc wl := hrx.trans hlx.symm
  -- `wd` comes before `wl` in `co`. Else a read-modify-write after `wl` reads its 0.
  have hco : X.co wd wl := by
    rcases ho.total wd wl hdw hlw (hdx.trans hlx.symm) hne with h | h
    · exact h
    · obtain ⟨c, r, hr, hrc⟩ := (reach wl wd hdx h).exists_first
      have hrx' : X.loc r = x := ((wf.rf_write_read wl r hr).2.2).symm.trans hlx
      exact absurd (by rw [← hval wl r hr, hl0, hl1]; rfl) (hlive r c hrc hrx')
  have hyx : X.loc y = x := (wf.rf_write_read y rl hy).2.2.trans hrx
  have hchain : wd = y ∨ TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) wd y := by
    rcases rmw_between wf hc ho hy hl hloc hco with rfl | hdy
    · exact Or.inl rfl
    · exact Or.inr (reach wd y hyx hdy)
  exact hb_of_release_read ha hdw hdrel hda hchain hy hlacq hb

/-- The count `x` of one value, read by the last receiver with the acquire load `e`. The load
reads 1 from `y`, and the receiver then stores 0 with `z`. Each write of the count but the
first and `z` is a read-modify-write. No read-modify-write reads 0, and none comes between `y`
and `z`. Another receiver takes 1 with the release write `wd`. Then each event before `wd`
happens before each event after `e`, such as the move of the value out of its slot. -/
theorem last_load_hb (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {x : Nat} {val : X.E → Int} {y e z : X.E}
    (hrmw : ∀ w, X.IsWrite w → X.loc w = x → (∃ a, X.co a w) → w ≠ z → ∃ r, X.rmw r w)
    (hsrc : ∀ r y, X.rmw r y → X.loc y = x → X.loc r = x ∧ ∃ s, X.rf s r)
    (hval : ∀ w r, X.rf w r → val w = val r)
    (hlive : ∀ r y, X.rmw r y → X.loc r = x → val r ≠ 0)
    (honly : ∀ r w, X.rmw r w → X.loc w = x → X.co y w → X.co w z → False)
    (hye : X.rf y e) (hex : X.loc e = x) (hacq : X.IsAcquire e)
    (hez : X.po e z) (hzw : X.IsWrite z) (hzx : X.loc z = x) (hz0 : val z = 0)
    (hzr : ∀ r, ¬ X.rmw r z)
    {rd wd : X.E} (hd : X.rmw rd wd) (hdx : X.loc wd = x) (hdrel : X.IsRelease wd)
    {a b : X.E} (ha : a = wd ∨ X.po a wd) (hb : e = b ∨ X.po e b) : X.hb a b := by
  obtain ⟨-, hdw, -, hda⟩ := wf.rmw_atomic rd wd hd
  obtain ⟨hyw, -, hyl⟩ := wf.rf_write_read y e hye
  have hyx : X.loc y = x := hyl.trans hex
  -- `z` comes after `y` in `co`, since `e` reads `y` and comes before `z`.
  have hyz : X.co y z := by
    have hne : y ≠ z := by
      rintro rfl
      exact hc.no_thin_air y (TransGen.tail (TransGen.single (Or.inr hye)) (Or.inl hez))
    rcases ho.total y z hyw hzw (hyx.trans hzx.symm) hne with h | h
    · exact h
    · have hez' : X.hb e z := TransGen.single (Or.inl hez)
      have heco : X.eco z e := TransGen.tail (TransGen.single (Or.inr (Or.inl h))) (Or.inl hye)
      exact absurd (Or.inr heco) (hc.coherence e z hez')
  have hdz : wd ≠ z := fun h => hzr rd (h ▸ hd)
  have hchain : wd = y ∨ TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) wd y := by
    by_cases hdy : wd = y
    · exact Or.inl hdy
    rcases ho.total wd y hdw hyw (hdx.trans hyx.symm) hdy with h | h
    · -- Each write from `wd` to `y` comes before `z`, so it is a read-modify-write.
      refine Or.inr (rmw_chain_up_to wf hc ho hfin (c := y) ?_ hsrc y hyx h (Or.inl rfl))
      intro w hw hwx haw hwy
      refine hrmw w hw hwx ⟨wd, haw⟩ ?_
      rintro rfl
      rcases hwy with rfl | hwy
      · exact wf.co_irrefl _ hyz
      · exact wf.co_irrefl _ (ho.trans _ _ _ hwy hyz)
    · -- `wd` after `y`: it is not between `y` and `z`, so it comes after `z`. The first
      -- read-modify-write after `z` reads its 0.
      exfalso
      rcases ho.total wd z hdw hzw (hdx.trans hzx.symm) hdz with h' | h'
      · exact honly rd wd hd hdx h h'
      · have hz : ∀ w, X.IsWrite w → X.loc w = x → X.co z w → (w = wd ∨ X.co w wd) →
            ∃ r, X.rmw r w := fun w hw hwx hzw' _ =>
          hrmw w hw hwx ⟨z, hzw'⟩ (fun h => wf.co_irrefl _ (h ▸ hzw'))
        obtain ⟨c, r, hr, hrc⟩ :=
          (rmw_chain_up_to wf hc ho hfin hz hsrc wd hdx h' (Or.inl rfl)).exists_first
        have hrx' : X.loc r = x := ((wf.rf_write_read z r hr).2.2).symm.trans hzx
        exact hlive r c hrc hrx' (by rw [← hval z r hr, hz0])
  exact hb_of_release_read ha hdw hdrel hda hchain hye hacq hb

end Vx.Execution
