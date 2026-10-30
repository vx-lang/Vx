import VxMemoryModel.CpuDeadlock
import VxMemoryModel.Progress

/-!
# A thread finishes (W7)

`Cpu.finishes` is W7 for a thread of a CPU node: in a fair run, a thread with no loop in its code
reaches its end, if each of its waits can continue. The hypotheses are:

- The thread is never stuck, except at a wait: when it is not done and does not wait, it can move.
- Each wait can continue at some point: the thread stops waiting.
- A wait that can continue stays so until the thread moves. The flags of `tasks` and of channels
  have this property: only one write sets each flag, and no write clears it.

The proof uses a weight of the code. A step of the thread keeps its code, or makes the weight
smaller. A step that keeps the code is a wait that read 0 (`Cpu.Step.code_eq`). So a thread that
never stops waiting cannot keep its code forever, and WF1 gives it a step that is not a spin.
-/

namespace Vx

/-- The code has no `loop`. -/
def Stmt.NoLoop : Stmt → Prop
  | .loop .. => False
  | .seq s t => s.NoLoop ∧ t.NoLoop
  | .ite _ s t => s.NoLoop ∧ t.NoLoop
  | _ => True

/-- A weight that each step of code with no loop makes smaller, except a spin of a wait and the
read of a read-modify-write. -/
def Stmt.weight : Stmt → Nat
  | .skip => 0
  | .seq s t => s.weight + t.weight + 1
  | .ite _ s t => s.weight + t.weight + 1
  | _ => 1

theorem Step.weight_le {s regs a s' regs'} (h : Step s regs a s' regs') (hn : s.NoLoop) :
    s'.NoLoop ∧
      (s' = s ∨ s'.weight < s.weight ∨ (a ≠ none ∧ s'.Pending ∧ s'.weight = s.weight)) := by
  induction h with
  | seq _ ih =>
    obtain ⟨hl, h | h | ⟨ha, hp, h⟩⟩ := ih hn.1
    · exact ⟨⟨hl, hn.2⟩, .inl (by rw [h])⟩
    · exact ⟨⟨hl, hn.2⟩, .inr (.inl (by simp only [Stmt.weight]; omega))⟩
    · exact ⟨⟨hl, hn.2⟩, .inr (.inr ⟨ha, hp, by simp only [Stmt.weight]; omega⟩)⟩
  | seq_skip => exact ⟨hn.2, .inr (.inl (by simp only [Stmt.weight]; omega))⟩
  | ite_true => exact ⟨hn.1, .inr (.inl (by simp only [Stmt.weight]; omega))⟩
  | ite_false => exact ⟨hn.2, .inr (.inl (by simp only [Stmt.weight]; omega))⟩
  | loop => exact hn.elim
  | wait v | waitAt _ v => by_cases hv : v = 0 <;> simp [hv, Stmt.weight, Stmt.NoLoop]
  | fetchAdd | cas_eq => exact ⟨trivial, .inr (.inr ⟨nofun, trivial, rfl⟩)⟩
  | load | store | cas_ne | rmwStore | fence | assign | loadAt | storeAt | assignAt =>
    exact ⟨trivial, .inr (.inl (by simp [Stmt.weight]))⟩

/-- The write of a read-modify-write makes the weight smaller. -/
theorem Step.weight_pending {s regs a s' regs'} (h : Step s regs a s' regs') (hp : s.Pending)
    (hn : s.NoLoop) : s'.NoLoop ∧ s'.weight < s.weight := by
  induction h with
  | rmwStore => exact ⟨trivial, by simp [Stmt.weight]⟩
  | seq _ ih =>
    obtain ⟨hl, hw⟩ := ih hp hn.1
    exact ⟨⟨hl, hn.2⟩, by simp only [Stmt.weight]; omega⟩
  | wait v | waitAt _ v => simp [Stmt.Pending] at hp
  | _ => simp [Stmt.Pending] at hp

/-- A step of a thread keeps its code, or makes its weight smaller. -/
theorem Cpu.Step.weight {c c' : Cpu} {t ls} (h : Cpu.Step c t ls c') (hn : (c.code t).NoLoop) :
    (c'.code t).NoLoop ∧ (c'.code t = c.code t ∨ (c'.code t).weight < (c.code t).weight) := by
  cases h with
  | silent hs =>
    simp only [Cpu.move, setAt_same]
    obtain ⟨hl, h | h | ⟨ha, _⟩⟩ := hs.weight_le hn
    · exact ⟨hl, .inl h⟩
    · exact ⟨hl, .inr h⟩
    · exact absurd rfl ha
  | one hs _ hp' _ =>
    simp only [Cpu.move, setAt_same]
    obtain ⟨hl, h | h | ⟨_, hp, _⟩⟩ := hs.weight_le hn
    · exact ⟨hl, .inl h⟩
    · exact ⟨hl, .inr h⟩
    · exact absurd hp hp'
  | two hs hn' hp _ hs₂ _ =>
    simp only [Cpu.move, setAt_same]
    obtain ⟨hl, h | h | ⟨_, _, h⟩⟩ := hs.weight_le hn
    · rw [h] at hp; exact absurd hp hn'
    · obtain ⟨hl₂, h₂⟩ := hs₂.weight_pending hp hl
      exact ⟨hl₂, .inr (by omega)⟩
    · obtain ⟨hl₂, h₂⟩ := hs₂.weight_pending hp hl
      exact ⟨hl₂, .inr (by omega)⟩

namespace Fleet

variable {S M : Type} {F : Fleet S M} {A : Act}

/-- WF1 with a condition on the run. If `A` can happen while `P` holds, and `P` holds until `A`
happens, then in a fair run `A` happens at a point where `P` holds. -/
theorem wf1_run {r : Nat → Config S M} (hf : F.WeakFair r A) {P : Config S M → Prop} {i : Nat}
    (hkeep : ∀ j, i ≤ j → P (r j) → ¬ F.Step A (r j) (r (j + 1)) → P (r (j + 1)))
    (hen : ∀ j, i ≤ j → P (r j) → F.Enabled A (r j)) (hi : P (r i)) :
    ∃ j, i ≤ j ∧ P (r j) ∧ F.Step A (r j) (r (j + 1)) := by
  refine Classical.byContradiction fun hno => ?_
  have hP : ∀ k, P (r (i + k)) := by
    intro k
    induction k with
    | zero => exact hi
    | succ k ih =>
      exact hkeep _ (Nat.le_add_right _ _) ih fun hs => hno ⟨_, Nat.le_add_right _ _, ih, hs⟩
  obtain ⟨j, hij, hj⟩ := hf i fun j hij => by
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hij
    exact hen _ hij (hP k)
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hij
  exact hno ⟨_, hij, hP k, hj⟩

end Fleet

namespace Cpu

variable {M : Type} {F : Fleet Cpu M} {n t : Nat}

/-- Only a step of thread `t` of node `n` changes its code. -/
theorem code_keep (hF : F.node n = node M) {act : Fleet.Act} {c c' : Fleet.Config Cpu M}
    (hs : F.Step act c c') (ha : act ≠ .move n t) : (c'.state n).code t = (c.state n).code t := by
  cases hs with
  | @move _ n' a s' _ h =>
    by_cases e : n = n'
    · subst e
      rw [hF] at h
      obtain ⟨_, _, h⟩ := h
      have hat : t ≠ a := fun e => ha (by rw [e])
      simp only [setAt_same]
      cases h <;> simp [move, setAt_other _ _ hat]
    · simp [setAt_other _ _ e]
  | deliver i hi =>
    by_cases e : n = c.flight[i].dst
    · subst e
      simp only [setAt_same]
      rw [hF]
      rfl
    · simp [setAt_other _ _ e]

/-- A step of thread `t` of node `n` is a step of the CPU. -/
theorem move_step (hF : F.node n = node M) {c c' : Fleet.Config Cpu M}
    (hs : F.Step (.move n t) c c') : ∃ ls, Step (c.state n) t ls (c'.state n) := by
  obtain ⟨s', _, h, e⟩ := hs.move_state rfl
  rw [hF] at h
  obtain ⟨_, ls, h⟩ := h
  exact ⟨ls, e ▸ h⟩

/-- W7: in a fair run, a thread with no loop finishes if it is never stuck except at a wait, each
of its waits can continue at some point, and a wait that can continue stays so until the thread
moves. -/
theorem finishes (hF : F.node n = node M) {c₀ : Fleet.Config Cpu M}
    {r : Nat → Fleet.Config Cpu M} (hr : F.IsRun c₀ r) (hf : F.WeakFair r (.move n t)) {i : Nat}
    (hloop : (((r i).state n).code t).NoLoop)
    (hok : ∀ j, i ≤ j → ((r j).state n).code t ≠ .skip → ¬ ((r j).state n).Waits t →
      F.Enabled (.move n t) (r j))
    (hcont : ∀ j, i ≤ j → ((r j).state n).Waits t → ∃ k, j ≤ k ∧ ¬ ((r k).state n).Waits t)
    (hstay : ∀ j, i ≤ j → ¬ ((r j).state n).Waits t → ¬ F.Step (.move n t) (r j) (r (j + 1)) →
      ¬ ((r (j + 1)).state n).Waits t) :
    ∃ k, i ≤ k ∧ ((r k).state n).code t = .skip := by
  let code := fun j => ((r j).state n).code t
  -- Each point keeps the code of the thread, or makes its weight smaller.
  have one : ∀ j, (code j).NoLoop →
      (code (j + 1)).NoLoop ∧
        (code (j + 1) = code j ∨ (code (j + 1)).weight < (code j).weight) := by
    intro j hn
    rcases hr.next j with ⟨act, hs⟩ | he
    · by_cases ha : act = .move n t
      · subst ha
        obtain ⟨_, h⟩ := move_step hF hs
        exact h.weight hn
      · have e : code (j + 1) = code j := code_keep hF hs ha
        exact ⟨e ▸ hn, .inl e⟩
    · have e : code (j + 1) = code j := by simp only [code, he]
      exact ⟨e ▸ hn, .inl e⟩
  have noloop : ∀ k, (code (i + k)).NoLoop := by
    intro k
    induction k with
    | zero => exact hloop
    | succ k ih => exact (one _ ih).1
  have later : ∀ j, i ≤ j → ∀ k,
      code (j + k) = code j ∨ (code (j + k)).weight < (code j).weight := by
    intro j hij k
    obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hij
    induction k with
    | zero => exact .inl rfl
    | succ k ih =>
      have h := (one (i + d + k) (by rw [Nat.add_assoc]; exact noloop _)).2
      rw [← Nat.add_assoc]
      rcases ih with e | e <;> rcases h with h | h
      · exact .inl (h.trans e)
      · rw [e] at h; exact .inr h
      · rw [h]; exact .inr e
      · exact .inr (by omega)
  -- A thread that is not done does not keep its code forever.
  have moves : ∀ j, i ≤ j → code j ≠ .skip → ¬ ∀ k, j ≤ k → code k = code j := by
    intro j hij hd hsame
    obtain ⟨j₁, hj₁, hw⟩ : ∃ j₁, j ≤ j₁ ∧ ¬ ((r j₁).state n).Waits t := by
      by_cases hw : ((r j).state n).Waits t
      · exact hcont j hij hw
      · exact ⟨j, Nat.le_refl _, hw⟩
    obtain ⟨k, hk, hwk, hs⟩ := Fleet.wf1_run hf (P := fun c => ¬ (c.state n).Waits t)
      (fun j' hj' => hstay j' (by omega))
      (fun j' hj' hw' =>
        hok j' (by omega) (show code j' ≠ _ by rw [hsame j' (by omega)]; exact hd) hw') hw
    obtain ⟨_, h⟩ := move_step hF hs
    have e : code (k + 1) = code k := (hsame _ (by omega)).trans (hsame _ (by omega)).symm
    exact hwk (h.code_eq e).1
  -- Induction on the weight.
  have main : ∀ w j, i ≤ j → (code j).weight ≤ w → ∃ k, j ≤ k ∧ code k = .skip := by
    intro w
    induction w with
    | zero =>
      intro j hij hw
      refine Classical.byContradiction fun hno => moves j hij (fun e => hno ⟨j, Nat.le_refl _, e⟩)
        fun k hjk => ?_
      obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hjk
      rcases later j hij d with e | e
      · exact e
      · omega
    | succ w ih =>
      intro j hij hw
      by_cases hd : code j = .skip
      · exact ⟨j, Nat.le_refl _, hd⟩
      by_cases hsame : ∀ k, j ≤ k → code k = code j
      · exact absurd hsame (moves j hij hd)
      · obtain ⟨k, hk⟩ := Classical.not_forall.1 hsame
        have hjk : j ≤ k := Classical.byContradiction fun h => hk fun h' => absurd h' h
        replace hk : code k ≠ code j := fun e => hk fun _ => e
        obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hjk
        rcases later j hij d with e | e
        · exact absurd e hk
        · obtain ⟨m, hm, hs⟩ := ih (j + d) (by omega) (by omega)
          exact ⟨m, by omega, hs⟩
  exact main _ i (Nat.le_refl _) (Nat.le_refl _)

end Cpu

end Vx
