import VxMemoryModel.Counter

/-!
# Release sequences

The edges that a release write gives to an acquire read of it, or of a read-modify-write after it.
`Arc.lean`, `MutexOrders.lean` and `ChannelOrders.lean` use them.
-/

namespace Vx.Execution

open Relation Classical

variable {X : Execution}

/-- A release write `w`, a chain of read-modify-writes from `w` to `y`, and an acquire read of
`y`: then `w` synchronizes with the read. -/
theorem hb_of_release_read {a w y e b : X.E}
    (haw : a = w ∨ X.po a w) (hw : X.IsWrite w) (hwr : X.IsRelease w) (hwa : X.IsAtomic w)
    (hwy : w = y ∨ TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) w y)
    (hye : X.rf y e) (hea : X.IsAcquire e) (heb : e = b ∨ X.po e b) : X.hb a b := by
  have hat : X.IsAtomic e := fun h => by rcases hea with h' | h' | h' <;> rw [h'] at h <;> cases h
  have hsw : X.hb w e := TransGen.single (Or.inr ⟨hwr, hea, w, y, e, Or.inl rfl,
    ⟨w, hw, Or.inl rfl, hw, hwa, hwy⟩, hye, hat, Or.inl rfl⟩)
  have h1 : X.hb a e := haw.elim (fun h => h ▸ hsw) (fun h => TransGen.head (Or.inl h) hsw)
  exact heb.elim (fun h => h ▸ h1) (fun h => h1.tail (Or.inl h))

/-- A release write `w`, a chain of read-modify-writes from `w` to `y`, and a read of `y` before
an acquire fence: then `w` synchronizes with the fence. -/
theorem hb_of_release_sequence {a w y z f b : X.E}
    (haw : a = w ∨ X.po a w) (hw : X.IsWrite w) (hwr : X.IsRelease w) (hwa : X.IsAtomic w)
    (hwy : w = y ∨ TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) w y)
    (hyz : X.rf y z) (hza : X.IsAtomic z) (hzf : X.po z f) (hf : X.IsFence f)
    (hfa : X.IsAcquire f) (hfb : f = b ∨ X.po f b) : X.hb a b := by
  have hsw : X.hb w f := TransGen.single (Or.inr ⟨hwr, hfa, w, y, z, Or.inl rfl,
    ⟨w, hw, Or.inl rfl, hw, hwa, hwy⟩, hyz, hza, Or.inr ⟨hzf, hf⟩⟩)
  have h1 : X.hb a f := haw.elim (fun h => h ▸ hsw) (fun h => TransGen.head (Or.inl h) hsw)
  exact hfb.elim (fun h => h ▸ h1) (fun h => h1.tail (Or.inl h))

/-- On a place where each write but the first in `co` is the write of a read-modify-write, a write
reaches each later write through a chain of read-modify-writes that read from each other. -/
theorem rmw_chain_of_co (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {x : Nat}
    (hrmw : ∀ y, X.IsWrite y → X.loc y = x → (∃ a, X.co a y) → ∃ r, X.rmw r y)
    (hsrc : ∀ r y, X.rmw r y → X.loc y = x → X.loc r = x ∧ ∃ s, X.rf s r) :
    ∀ a b, X.loc b = x → X.co a b →
      TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) a b := by
  obtain ⟨l, hl⟩ := hfin
  have key : ∀ n b, count l (X.co · b) ≤ n → ∀ a, X.loc b = x → X.co a b →
      TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) a b := by
    intro n
    induction n with
    | zero =>
      intro b hn a _ hab
      have := count_pos (p := (X.co · b)) (hl a) hab
      omega
    | succ n ih =>
      intro b hn a hbx hab
      obtain ⟨-, hbw, -⟩ := wf.co_writes a b hab
      obtain ⟨r, hrb⟩ := hrmw b hbw hbx ⟨a, hab⟩
      obtain ⟨hrx, s, hs⟩ := hsrc r b hrb hbx
      have hloc : X.loc r = X.loc b := hrx.trans hbx.symm
      have hsb := rmw_source_co wf hc ho hs hrb hloc
      rcases rmw_between wf hc ho hs hrb hloc hab with rfl | has
      · exact TransGen.single ⟨r, hs, hrb⟩
      · have hsx : X.loc s = x := (wf.co_writes s b hsb).2.2.trans hbx
        have hlt := count_lt (fun c h => ho.trans c s b h hsb) (hl s) hsb (wf.co_irrefl s)
        exact (ih s (by omega) a hsx has).tail ⟨r, hs, hrb⟩
  exact fun a b hbx hab => key _ b (Nat.le_refl _) a hbx hab

end Vx.Execution
