import VxMemoryModel.Machine

/-!
# The WF1 rule

Lamport's rule for progress in TLA. Take a condition `P` on configurations and an action `A`.
Assume that `A` can happen while `P` holds, and that `P` holds until `A` happens. Then in a fair
run, `A` happens at some point after each point where `P` holds (`wf1_taken`).

`wf1` is the rule in its usual form, "`P` leads to `Q`". `delivers` uses `wf1_taken`: in a fair
run, the fleet delivers each message in flight at some point.
-/

namespace Vx.Fleet

variable {S M : Type} {F : Fleet S M} {A : Act}

/-- In the run `r`, after each point where `P` holds, `Q` holds at some point. -/
def LeadsTo (r : Nat → Config S M) (P Q : Config S M → Prop) : Prop :=
  ∀ i, P (r i) → ∃ j, i ≤ j ∧ Q (r j)

theorem LeadsTo.trans {r : Nat → Config S M} {P Q T : Config S M → Prop} (h₁ : LeadsTo r P Q)
    (h₂ : LeadsTo r Q T) : LeadsTo r P T := fun i hi =>
  let ⟨j, hij, hj⟩ := h₁ i hi
  let ⟨k, hjk, hk⟩ := h₂ j hj
  ⟨k, Nat.le_trans hij hjk, hk⟩

/-- WF1 for an action: if `A` can happen while `P` holds, and each other step keeps `P`, then in
a fair run `A` happens after each point where `P` holds. -/
theorem wf1_taken {c₀ : Config S M} {r : Nat → Config S M} (hr : F.IsRun c₀ r)
    (hf : F.WeakFair r A) {P : Config S M → Prop}
    (hkeep : ∀ act c c', P c → F.Step act c c' → act ≠ A → P c')
    (hen : ∀ c, P c → F.Enabled A c) {i : Nat} (hi : P (r i)) :
    ∃ j, i ≤ j ∧ F.Step A (r j) (r (j + 1)) := by
  refine Classical.byContradiction fun hno => ?_
  -- If `A` never happens from `i` on, then `P` holds from `i` on.
  have hP : ∀ k, P (r (i + k)) := by
    intro k
    induction k with
    | zero => exact hi
    | succ k ih =>
      rcases hr.next (i + k) with ⟨act, hs⟩ | he
      · by_cases ha : act = A
        · subst ha
          exact absurd ⟨i + k, Nat.le_add_right _ _, hs⟩ hno
        · exact hkeep act _ _ ih hs ha
      · rw [← Nat.add_assoc, he]
        exact ih
  obtain ⟨j, hij, hj⟩ := hf i fun j hij => by
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hij
    exact hen _ (hP k)
  exact hno ⟨j, hij, hj⟩

/-- WF1: `P` holds until `Q` holds, `A` can happen while `P` holds, and `A` makes `Q` true. Then
in a fair run, `P` leads to `Q`. -/
theorem wf1 {c₀ : Config S M} {r : Nat → Config S M} (hr : F.IsRun c₀ r) (hf : F.WeakFair r A)
    {P Q : Config S M → Prop} (hkeep : ∀ c c', P c → F.Next c c' → P c' ∨ Q c')
    (hen : ∀ c, P c → F.Enabled A c) (hA : ∀ c c', P c → F.Step A c c' → Q c') :
    LeadsTo r P Q := by
  intro i hi
  refine Classical.byContradiction fun hno => ?_
  have hnot : ∀ j, i ≤ j → ¬ Q (r j) := fun j hij hq => hno ⟨j, hij, hq⟩
  have hP : ∀ k, P (r (i + k)) := by
    intro k
    induction k with
    | zero => exact hi
    | succ k ih =>
      rcases hkeep _ _ ih (hr.next (i + k)) with h | h
      · exact h
      · exact absurd h (hnot _ (by omega))
  obtain ⟨j, hij, hj⟩ := hf i fun j hij => by
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hij
    exact hen _ (hP k)
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hij
  exact hnot _ (by omega) (hA _ _ (hP k) hj)

/-- The message with the number `k` is in flight. -/
def InFlight (k : Nat) (c : Config S M) : Prop := ∃ m ∈ c.flight, m.id = k

/-- Only its own delivery takes a message out of flight. -/
theorem inFlight_keep {k : Nat} {act : Act} {c c' : Config S M} (h : InFlight k c)
    (hs : F.Step act c c') (ha : act ≠ .deliver k) : InFlight k c' := by
  obtain ⟨m, hm, hk⟩ := h
  cases hs with
  | move => exact ⟨m, List.mem_append_left _ hm, hk⟩
  | deliver i hi =>
    obtain ⟨p, hp, rfl⟩ := List.mem_iff_getElem.1 hm
    refine ⟨c.flight[p], List.mem_eraseIdx_iff_getElem.2 ⟨p, hp, fun e => ?_, rfl⟩, hk⟩
    subst e
    exact ha (by rw [hk])

/-- A message in flight can be delivered. -/
theorem inFlight_enabled {k : Nat} {c : Config S M} (h : InFlight k c) :
    F.Enabled (.deliver k) c := by
  obtain ⟨m, hm, rfl⟩ := h
  obtain ⟨p, hp, rfl⟩ := List.mem_iff_getElem.1 hm
  exact ⟨_, .deliver p hp⟩

/-- A delivery takes one message out of flight and gives it to its node. -/
theorem Step.deliver_inv {k : Nat} {c c' : Config S M} (h : F.Step (.deliver k) c c') :
    ∃ i, ∃ hi : i < c.flight.length, c.flight[i].id = k ∧
      c'.state c.flight[i].dst = (F.node c.flight[i].dst).recv (c.state c.flight[i].dst)
        c.flight[i].data := by
  generalize ha : Act.deliver k = act at h
  cases h with
  | move => cases ha
  | deliver i hi =>
    cases ha
    exact ⟨i, hi, rfl, by simp⟩

/-- In a fair run, the fleet delivers each message in flight at some point. -/
theorem delivers {c₀ : Config S M} {r : Nat → Config S M} (hr : F.IsRun c₀ r) (hf : F.Fair r)
    {k i : Nat} (h : InFlight k (r i)) :
    ∃ j, i ≤ j ∧ F.Step (.deliver k) (r j) (r (j + 1)) :=
  wf1_taken hr (hf _) (fun _ _ _ h hs ha => inFlight_keep h hs ha) (fun _ h => inFlight_enabled h) h

end Vx.Fleet
