import VxMemoryModel.Hom

/-!
# Prefixes

A set of events is downward closed when it holds everything that comes before its events in
program order or in reads-from. Such a set is a prefix of the run: the program can stop there.
A happens-before path that ends in a prefix stays in it, so the prefix keeps the same
happens-before order.
-/

namespace Vx.Execution

open Relation

/-- A path that ends in `S` stays in `S`, when each step into `S` starts in `S`. -/
theorem transGen_reflect {α : Type _} {S : α → Prop} {r : α → α → Prop}
    {r' : {e // S e} → {e // S e} → Prop}
    (hlift : ∀ a b (hb : S b), r a b → ∃ ha : S a, r' ⟨a, ha⟩ ⟨b, hb⟩) {a b : α}
    (h : TransGen r a b) : ∀ hb : S b, ∃ ha : S a, TransGen r' ⟨a, ha⟩ ⟨b, hb⟩ := by
  induction h with
  | single h =>
    intro hb
    have ⟨ha, h'⟩ := hlift _ _ hb h
    exact ⟨ha, TransGen.single h'⟩
  | tail _ h ih =>
    intro hb
    have ⟨hc, h'⟩ := hlift _ _ hb h
    have ⟨ha, p⟩ := ih hc
    exact ⟨ha, p.tail h'⟩

variable {X : Execution} {S : X.E → Prop}

def Downclosed (X : Execution) (S : X.E → Prop) : Prop :=
  ∀ a b, S b → (X.po a b ∨ X.rf a b) → S a

theorem rs_reflect (hrmw : ∀ r w, X.rmw r w → X.po r w) (hS : X.Downclosed S) {a b : X.E} (h : X.rs a b)
    (hb : S b) : ∃ ha : S a, (X.restrict S).rs ⟨a, ha⟩ ⟨b, hb⟩ := by
  obtain ⟨w, hwa, haw, hw, hwat, hwb⟩ := h
  have ⟨hwS, hwb'⟩ : ∃ hwS : S w, (⟨w, hwS⟩ : {e // S e}) = ⟨b, hb⟩ ∨
      TransGen (fun x y : {e // S e} => ∃ r : {e // S e}, X.rf x.val r.val ∧ X.rmw r.val y.val)
        ⟨w, hwS⟩ ⟨b, hb⟩ := by
    rcases hwb with rfl | hwb
    · exact ⟨hb, Or.inl rfl⟩
    · have ⟨hwS, p⟩ := transGen_reflect
        (r' := fun x y : {e // S e} => ∃ r : {e // S e}, X.rf x.val r.val ∧ X.rmw r.val y.val)
        (fun x y hy ⟨r, hxr, hry⟩ =>
        have hr : S r := hS _ _ hy (Or.inl (hrmw _ _ hry))
        ⟨hS _ _ hr (Or.inr hxr), ⟨r, hr⟩, hxr, hry⟩) hwb hb
      exact ⟨hwS, Or.inr p⟩
  rcases haw with rfl | ⟨hp, hl⟩
  · exact ⟨hwS, _, hwa, Or.inl rfl, hw, hwat, hwb'⟩
  · have ha : S a := hS _ _ hwS (Or.inl hp)
    exact ⟨ha, ⟨w, hwS⟩, hwa, Or.inr ⟨hp, hl⟩, hw, hwat, hwb'⟩

theorem sw_reflect (hrmw : ∀ r w, X.rmw r w → X.po r w) (hS : X.Downclosed S) {a b : X.E} (h : X.sw a b)
    (hb : S b) : ∃ ha : S a, (X.restrict S).sw ⟨a, ha⟩ ⟨b, hb⟩ := by
  obtain ⟨hrel, hacq, x, y, z, hax, hxy, hyz, hz, hzb⟩ := h
  have hzS : S z := by
    rcases hzb with rfl | ⟨hp, _⟩
    · exact hb
    · exact hS _ _ hb (Or.inl hp)
  have hyS : S y := hS _ _ hzS (Or.inr hyz)
  have ⟨hxS, hxy'⟩ := rs_reflect hrmw hS hxy hyS
  have hzb' : (⟨z, hzS⟩ : {e // S e}) = ⟨b, hb⟩ ∨ (X.po z b ∧ X.IsFence b) := by
    rcases hzb with rfl | hzb
    · exact Or.inl rfl
    · exact Or.inr hzb
  rcases hax with rfl | ⟨hf, hp⟩
  · exact ⟨hxS, hrel, hacq, ⟨_, hxS⟩, ⟨y, hyS⟩, ⟨z, hzS⟩, Or.inl rfl, hxy', hyz, hz, hzb'⟩
  · have ha : S a := hS _ _ hxS (Or.inl hp)
    exact ⟨ha, hrel, hacq, ⟨x, hxS⟩, ⟨y, hyS⟩, ⟨z, hzS⟩, Or.inr ⟨hf, hp⟩, hxy', hyz, hz, hzb'⟩

/-- A happens-before path that ends in a prefix is a path of the prefix. -/
theorem hb_reflect (hrmw : ∀ r w, X.rmw r w → X.po r w) (hS : X.Downclosed S) {a b : X.E} (h : X.hb a b)
    (hb : S b) : ∃ ha : S a, (X.restrict S).hb ⟨a, ha⟩ ⟨b, hb⟩ :=
  transGen_reflect (fun x y hy s => by
    rcases s with hp | hsw
    · exact ⟨hS _ _ hy (Or.inl hp), Or.inl hp⟩
    · have ⟨hx, h'⟩ := sw_reflect hrmw hS hsw hy
      exact ⟨hx, Or.inr h'⟩) h hb

/-- No edge of `po`, `rf`, `co` or `fr` leaves the events outside `S` for an event in `S`. -/
def OrdClosed (X : Execution) (S : X.E → Prop) : Prop := ∀ a b, S b → X.ord a b → S a

theorem ord_reflect (hS : X.OrdClosed S) (hrf : X.Downclosed S) {a b : X.E}
    (h : TransGen X.ord a b) (hb : S b) :
    ∃ ha : S a, TransGen (X.restrict S).ord ⟨a, ha⟩ ⟨b, hb⟩ :=
  transGen_reflect (fun x y hy s => by
    have hx := hS _ _ hy s
    refine ⟨hx, ?_⟩
    rcases s with s | s | s | ⟨w, hwx, hwy⟩
    · exact Or.inl s
    · exact Or.inr (Or.inl s)
    · exact Or.inr (Or.inr (Or.inl s))
    · exact Or.inr (Or.inr (Or.inr ⟨⟨w, hrf _ _ hx (Or.inr hwx)⟩, hwx, hwy⟩))) h hb

end Vx.Execution
