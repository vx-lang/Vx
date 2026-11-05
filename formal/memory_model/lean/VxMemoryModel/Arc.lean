import VxMemoryModel.ReleaseSequence

/-!
# The last drop of an `Arc`

`stdlib/std/sync/arc.vx` keeps a count of the owners of a value. A clone adds 1 with a relaxed
read-modify-write, and a drop takes 1 away with a release one. The drop that reads 1 runs an
acquire fence and then frees the value. `arc_last_drop_hb_free` shows that each event before
another drop happens before each event after that fence. The clones and drops between the two
drops form a release sequence.
-/

namespace Vx.Execution

open Relation Classical

variable {X : Execution}

/-- Each use of the value of an `Arc` before another drop happens before each event after the
acquire fence of the last drop, so the free of the value does not race with the use. -/
theorem arc_last_drop_hb_free (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {x : Nat} {val : X.E → Int}
    (hrmw : ∀ y, X.IsWrite y → X.loc y = x → (∃ a, X.co a y) → ∃ r, X.rmw r y)
    (hsrc : ∀ r y, X.rmw r y → X.loc y = x → X.loc r = x ∧ ∃ s, X.rf s r)
    (hval : ∀ w r, X.rf w r → val w = val r)
    (hlive : ∀ r y, X.rmw r y → X.loc r = x → val r ≠ 0)
    {rd wd : X.E} (hd : X.rmw rd wd) (hdx : X.loc wd = x) (hdrel : X.IsRelease wd)
    {rl wl f : X.E} (hl : X.rmw rl wl) (hlx : X.loc wl = x) (hl1 : val rl = 1)
    (hl0 : val wl = val rl - 1) (hne : wd ≠ wl)
    (hlf : X.po rl f) (hf : X.IsFence f) (hfa : X.IsAcquire f)
    {a b : X.E} (ha : a = wd ∨ X.po a wd) (hb : f = b ∨ X.po f b) : X.hb a b := by
  have reach := rmw_chain_of_co wf hc ho hfin hrmw hsrc
  obtain ⟨-, hdw, -, hda⟩ := wf.rmw_atomic rd wd hd
  obtain ⟨-, hlw, hra, -⟩ := wf.rmw_atomic rl wl hl
  obtain ⟨hrx, y, hy⟩ := hsrc rl wl hl hlx
  have hloc : X.loc rl = X.loc wl := hrx.trans hlx.symm
  -- `wd` comes before `wl` in `co`. Otherwise a read-modify-write reads the 0 that `wl` wrote.
  have hco : X.co wd wl := by
    rcases ho.total wd wl hdw hlw (hdx.trans hlx.symm) hne with h | h
    · exact h
    · obtain ⟨c, r, hr, hrc⟩ := (reach wl wd hdx h).exists_first
      have hrx' : X.loc r = x := ((wf.rf_write_read wl r hr).2.2).symm.trans hlx
      exact absurd (by rw [← hval wl r hr, hl0, hl1]; rfl) (hlive r c hrc hrx')
  rcases rmw_between wf hc ho hy hl hloc hco with rfl | hdy
  · exact hb_of_release_sequence ha hdw hdrel hda (Or.inl rfl) hy hra hlf hf hfa hb
  · have hyx : X.loc y = x := (wf.rf_write_read y rl hy).2.2.trans hrx
    exact hb_of_release_sequence ha hdw hdrel hda (Or.inr (reach wd y hyx hdy)) hy hra hlf hf
      hfa hb

end Vx.Execution
