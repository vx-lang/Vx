import VxMemoryModel.Machine

/-!
# A deadlock across nodes

Take a predicate `Q n s` for "the node `n` in the state `s` is quiet": each of its tasks is done
or waits. A quiet node sends nothing, and its own steps keep it quiet (`KeepsQuiet`). Only a
message can wake it.

The fleet is deadlocked when each node is quiet and no message is in flight. Then nothing can
wake a node, so the fleet stays deadlocked (`Deadlocked.forever`, the soundness part of F9).

No one node can see a deadlock across nodes. A snapshot in the style of Chandy and Lamport (1985)
lets each node record its state at its own point. `snapshot` shows that the recorded result is
sound: if each node records that it is quiet, and no message crosses the cut, then the fleet is
deadlocked when the last node records. The marker protocol that gives such a cut is not modeled.
-/

namespace Vx.Fleet

variable {S M : Type} {F : Fleet S M} {Q : Nat → S → Prop}

/-- A quiet node sends nothing, and its steps keep it quiet. -/
def KeepsQuiet (F : Fleet S M) (Q : Nat → S → Prop) : Prop :=
  ∀ n s a s' out, Q n s → (F.node n).step s a s' out → Q n s' ∧ out = []

/-- Each node is quiet, and no message is in flight. -/
def Deadlocked (Q : Nat → S → Prop) (c : Config S M) : Prop :=
  c.flight = [] ∧ ∀ n, Q n (c.state n)

/-- The two kinds of step, with the configuration after the step as a variable. -/
theorem Step.cases {act : Act} {c c' : Config S M} (h : F.Step act c c') :
    (∃ n a s' out, (F.node n).step (c.state n) a s' out ∧ c'.state = setAt c.state n s' ∧
      c'.flight = c.flight ++ number n c.sent out ∧ c'.sent = c.sent + out.length) ∨
    (∃ i, ∃ hi : i < c.flight.length, c'.state = setAt c.state c.flight[i].dst
      ((F.node c.flight[i].dst).recv (c.state c.flight[i].dst) c.flight[i].data) ∧
      c'.flight = c.flight.eraseIdx i ∧ c'.sent = c.sent) := by
  cases h with
  | move h => exact .inl ⟨_, _, _, _, h, rfl, rfl, rfl⟩
  | deliver i hi => exact .inr ⟨i, hi, rfl, rfl, rfl⟩

theorem Deadlocked.step {act : Act} {c c' : Config S M} (hq : KeepsQuiet F Q)
    (h : Deadlocked Q c) (hs : F.Step act c c') : Deadlocked Q c' := by
  obtain ⟨hf, hn⟩ := h
  rcases hs.cases with ⟨n, a, s', out, hs, hst, hfl, _⟩ | ⟨i, hi, _⟩
  · obtain ⟨hq', rfl⟩ := hq _ _ _ _ _ (hn n) hs
    refine ⟨by simp [hfl, hf, number], fun k => ?_⟩
    rw [hst]
    by_cases e : k = n
    · subst e; simpa using hq'
    · rw [setAt_other _ _ e]; exact hn k
  · simp [hf] at hi

/-- F9: once the fleet is deadlocked, it stays deadlocked. -/
theorem Deadlocked.forever {c₀ : Config S M} {r : Nat → Config S M} (hr : F.IsRun c₀ r)
    (hq : KeepsQuiet F Q) {i : Nat} (h : Deadlocked Q (r i)) : ∀ j, i ≤ j → Deadlocked Q (r j) := by
  intro j hij
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hij
  induction k with
  | zero => exact h
  | succ k ih =>
    rcases hr.next (i + k) with ⟨_, hs⟩ | he
    · exact (ih (Nat.le_add_right _ _)).step hq hs
    · rw [← Nat.add_assoc, he]; exact ih (Nat.le_add_right _ _)

/-! ## The snapshot -/

theorem mem_number {src n : Nat} {out : List (Nat × M)} {m : Msg M} (h : m ∈ number src n out) :
    m.src = src ∧ m.id < n + out.length := by
  induction out generalizing n with
  | nil => simp [number] at h
  | cons p out ih =>
    simp only [number, List.mem_cons] at h
    rcases h with rfl | h
    · simp
    · obtain ⟨h1, h2⟩ := ih h
      simp only [List.length_cons]
      exact ⟨h1, by omega⟩

/-- The number of messages sent never goes down. -/
theorem IsRun.sent_mono {c₀ : Config S M} {r : Nat → Config S M} (hr : F.IsRun c₀ r) {i j : Nat}
    (hij : i ≤ j) : (r i).sent ≤ (r j).sent := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hij
  induction k with
  | zero => exact Nat.le_refl _
  | succ k ih =>
    refine Nat.le_trans (ih (Nat.le_add_right _ _)) ?_
    rw [← Nat.add_assoc]
    rcases hr.next (i + k) with ⟨_, hs⟩ | he
    · rcases hs.cases with ⟨_, _, _, _, _, _, _, hse⟩ | ⟨_, _, _, _, hse⟩ <;> omega
    · rw [he]; exact Nat.le_refl _

/-- A snapshot is sound. The node `n` records its state at the point `t n`, and each node records
by the point `T`. Each node records that it is quiet. No message crosses the cut: a message that
its sender sent before it recorded is not in flight at or after the point where its receiver
recorded. Then the fleet is deadlocked at `T`. -/
theorem snapshot {c₀ : Config S M} {r : Nat → Config S M} (hr : F.IsRun c₀ r)
    (h0 : ∀ m ∈ c₀.flight, m.id < c₀.sent) (hq : KeepsQuiet F Q) (t : Nat → Nat) {T : Nat}
    (hT : ∀ n, t n ≤ T) (hrec : ∀ n, Q n ((r (t n)).state n))
    (hcut : ∀ j, ∀ m ∈ (r j).flight, m.id < (r (t m.src)).sent → j < t m.dst) :
    Deadlocked Q (r T) := by
  -- Each node that recorded is still quiet, and each message in flight was sent before the cut.
  have inv : ∀ j, (∀ n, t n ≤ j → Q n ((r j).state n)) ∧
      ∀ m ∈ (r j).flight, m.id < (r (t m.src)).sent := by
    intro j
    induction j with
    | zero =>
      refine ⟨fun n hn => ?_, fun m hm => ?_⟩
      · have e : t n = 0 := by omega
        simpa [e] using hrec n
      · have h1 := hr.sent_mono (Nat.zero_le (t m.src))
        rw [hr.start] at hm h1
        exact Nat.lt_of_lt_of_le (h0 m hm) h1
    | succ j ih =>
      obtain ⟨ih1, ih2⟩ := ih
      -- A node that records at `j + 1` is quiet by `hrec`. The others need the step.
      suffices h : (∀ n, t n ≤ j → Q n ((r (j + 1)).state n)) ∧
          ∀ m ∈ (r (j + 1)).flight, m.id < (r (t m.src)).sent by
        refine ⟨fun n hn => ?_, h.2⟩
        by_cases e : t n = j + 1
        · rw [← e]; exact hrec n
        · exact h.1 n (by omega)
      rcases hr.next j with ⟨_, hs⟩ | he
      · rcases hs.cases with ⟨n, a, s', out, hs, hst, hfl, hse⟩ | ⟨i, hi, hst, hfl, _⟩
        · by_cases hc : t n ≤ j
          · -- A node that recorded is quiet: it stays quiet and sends nothing.
            obtain ⟨hq', rfl⟩ := hq _ _ _ _ _ (ih1 n hc) hs
            refine ⟨fun k hk => ?_, fun m hm => ?_⟩
            · rw [hst]
              by_cases e : k = n
              · subst e; simpa using hq'
              · rw [setAt_other _ _ e]; exact ih1 k hk
            · simp only [hfl, number, List.append_nil] at hm
              exact ih2 m hm
          · -- A node that did not record yet sends before its cut.
            refine ⟨fun k hk => ?_, fun m hm => ?_⟩
            · rw [hst, setAt_other _ _ (fun e => hc (by subst e; exact hk))]; exact ih1 k hk
            · rw [hfl] at hm
              rcases List.mem_append.1 hm with hm | hm
              · exact ih2 m hm
              · obtain ⟨hsrc, hid⟩ := mem_number hm
                have := hr.sent_mono (show j + 1 ≤ t n by omega)
                rw [hsrc]
                omega
        · -- The message goes to a node that did not record yet.
          have hd := hcut j _ (List.getElem_mem hi) (ih2 _ (List.getElem_mem hi))
          refine ⟨fun k hk => ?_, fun m hm => ?_⟩
          · rw [hst, setAt_other _ _ (fun e => by subst e; omega)]; exact ih1 k hk
          · rw [hfl] at hm; exact ih2 m (List.mem_of_mem_eraseIdx hm)
      · rw [he]; exact ⟨ih1, ih2⟩
  refine ⟨List.eq_nil_iff_forall_not_mem.2 fun m hm => ?_, fun n => (inv T).1 n (hT n)⟩
  have := hcut T m hm ((inv T).2 m hm)
  have := hT m.dst
  omega

/-- After a sound snapshot, the fleet stays deadlocked. -/
theorem snapshot_forever {c₀ : Config S M} {r : Nat → Config S M} (hr : F.IsRun c₀ r)
    (h0 : ∀ m ∈ c₀.flight, m.id < c₀.sent) (hq : KeepsQuiet F Q) (t : Nat → Nat) {T : Nat}
    (hT : ∀ n, t n ≤ T) (hrec : ∀ n, Q n ((r (t n)).state n))
    (hcut : ∀ j, ∀ m ∈ (r j).flight, m.id < (r (t m.src)).sent → j < t m.dst) :
    ∀ j, T ≤ j → Deadlocked Q (r j) :=
  Deadlocked.forever hr hq (snapshot hr h0 hq t hT hrec hcut)

end Vx.Fleet
