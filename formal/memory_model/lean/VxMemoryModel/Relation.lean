/-! Small facts about the transitive closure that Lean's core library does not have. -/

namespace Relation.TransGen

variable {α : Sort u} {r s : α → α → Prop}

theorem head {a b c : α} (hab : r a b) (hbc : TransGen r b c) : TransGen r a c :=
  (single hab).trans hbc

theorem mono (h : ∀ a b, r a b → s a b) {a b : α} (hab : TransGen r a b) : TransGen s a b := by
  induction hab with
  | single hab => exact single (h _ _ hab)
  | tail _ hbc ih => exact ih.tail (h _ _ hbc)

/-- If each step of `r` is a path of `s`, then each path of `r` is a path of `s`. -/
theorem flatten (h : ∀ a b, r a b → TransGen s a b) {a b : α} (hab : TransGen r a b) :
    TransGen s a b := by
  induction hab with
  | single hab => exact h _ _ hab
  | tail _ hbc ih => exact ih.trans (h _ _ hbc)

/-- `a = b`, or a path from `a` to `b`, followed by a path from `b` to `c`. -/
theorem of_refl {a b c : α} (hab : a = b ∨ TransGen r a b) (hbc : TransGen r b c) :
    TransGen r a c := by
  rcases hab with rfl | hab
  · exact hbc
  · exact hab.trans hbc

end Relation.TransGen
