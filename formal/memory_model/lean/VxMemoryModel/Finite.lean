import VxMemoryModel.Relation

/-!
# Finite sets of events

An execution of a program has finitely many events: a list holds them all. This file counts
the events in a set, and finds an event with no step after it for a relation without cycles.
-/

namespace Vx

open Relation Classical

theorem _root_.Relation.TransGen.exists_first {α : Sort _} {r : α → α → Prop} {a b : α}
    (h : TransGen r a b) : ∃ c, r a c := by
  induction h with
  | single h => exact ⟨_, h⟩
  | tail _ _ ih => exact ih

/-- The number of entries of `l` in `p`. -/
noncomputable def count {α : Type _} (l : List α) (p : α → Prop) : Nat :=
  l.countP fun x => decide (p x)

theorem count_le {α : Type _} {p q : α → Prop} (h : ∀ x, p x → q x) (l : List α) :
    count l p ≤ count l q := by
  induction l with
  | nil => simp [count]
  | cons a l ih =>
    simp only [count, List.countP_cons] at ih ⊢
    by_cases hp : p a
    · simp [hp, h a hp]; omega
    · by_cases hq : q a <;> simp [hp, hq] <;> omega

theorem count_lt {α : Type _} {p q : α → Prop} (h : ∀ x, p x → q x) {l : List α} {x : α}
    (hx : x ∈ l) (hq : q x) (hp : ¬ p x) : count l p < count l q := by
  induction l with
  | nil => cases hx
  | cons a l ih =>
    have hle := count_le h l
    simp only [count, List.countP_cons] at ih hle ⊢
    rcases List.mem_cons.1 hx with rfl | hx
    · simp [hp, hq]; omega
    · have := ih hx
      by_cases hpa : p a
      · simp [hpa, h a hpa]; omega
      · by_cases hqa : q a <;> simp [hpa, hqa] <;> omega

theorem count_pos {α : Type _} {p : α → Prop} {l : List α} {x : α} (hx : x ∈ l) (h : p x) :
    0 < count l p :=
  Nat.lt_of_le_of_lt (Nat.zero_le _) (count_lt (p := fun _ => False) (fun _ h => h.elim) hx h id)

/-- In a finite type, a relation without cycles has, after each element, an element with no
step after it. -/
theorem exists_maximal {α : Type _} {l : List α} (hl : ∀ a, a ∈ l) {R : α → α → Prop}
    (hacyc : ∀ a, ¬ TransGen R a a) (a : α) :
    ∃ m, (a = m ∨ TransGen R a m) ∧ ∀ x, ¬ R m x := by
  have key : ∀ n a, count l (TransGen R a) ≤ n →
      ∃ m, (a = m ∨ TransGen R a m) ∧ ∀ x, ¬ R m x := by
    intro n
    induction n with
    | zero =>
      intro a hn
      refine ⟨a, Or.inl rfl, fun x hx => ?_⟩
      have := count_pos (hl x) (TransGen.single hx)
      omega
    | succ n ih =>
      intro a hn
      by_cases h : ∃ x, R a x
      · obtain ⟨x, hx⟩ := h
        have hlt : count l (TransGen R x) < count l (TransGen R a) :=
          count_lt (fun _ h => TransGen.head hx h) (hl x) (TransGen.single hx) (hacyc x)
        obtain ⟨m, hxm, hm⟩ := ih x (by omega)
        exact ⟨m, Or.inr (hxm.elim (fun h => h ▸ TransGen.single hx) (TransGen.head hx)), hm⟩
      · exact ⟨a, Or.inl rfl, fun x hx => h ⟨x, hx⟩⟩
  exact key _ a (Nat.le_refl _)

end Vx
