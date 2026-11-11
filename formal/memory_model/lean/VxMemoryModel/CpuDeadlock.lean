import VxMemoryModel.Deadlock
import VxMemoryModel.Interleave

/-!
# Waits on a CPU node

In `Thread.lean` a wait is a spin: `wait x` reads `x` and stays at the `wait` while `x` is 0. So
"waits" is a defined state (`Cpu.Waits`): the next step of the thread reads a flag that holds 0,
or that has no value yet.

- `Cpu.stops` is W6: if no step of a CPU node changes it, each of its threads waits or has no
  step. A thread whose code is `skip` is done and has no step.
- `Cpu.quiet_step`: when each thread is done or waits, each step of the node leaves it as it was.
  So `Cpu.keepsQuiet` gives the hypothesis of `Deadlocked.forever` for a fleet of CPU nodes.
- `Cpu.frozen`: in a deadlocked fleet of CPU nodes, no node ever changes again.
-/

namespace Vx

/-- The code waits on the place `x`: its next step is the read of `x` by a `wait` or a
`waitAt`. -/
def Stmt.WaitsOn (regs : Nat → Int) (x : Nat) : Stmt → Prop
  | .wait y => y = x
  | .waitAt base len i => ∃ j : Nat, i.eval regs = j ∧ j < len ∧ base + j = x
  | .seq s _ => s.WaitsOn regs x
  | _ => False

theorem Stmt.size_lt_ite_left {c : Expr} {s t : Stmt} : sizeOf s < sizeOf (Stmt.ite c s t) := by
  simp; omega

theorem Stmt.size_lt_ite_right {c : Expr} {s t : Stmt} : sizeOf t < sizeOf (Stmt.ite c s t) := by
  simp; omega

theorem Stmt.size_lt_seq_right {s t : Stmt} : sizeOf t < sizeOf (Stmt.seq s t) := by
  simp; omega

/-- A step that leaves the code as it was is a wait that read 0. -/
theorem Step.eq_self {s regs a s' regs'} (h : Step s regs a s' regs') (he : s' = s) :
    ∃ x, s.WaitsOn regs x ∧ a = some ⟨.read, .sc, x, 0, false⟩ ∧ regs' = regs := by
  induction h with
  | wait v =>
    by_cases hv : v = 0
    · subst hv; exact ⟨_, rfl, rfl, rfl⟩
    · simp [hv] at he
  | waitAt j v hi hj =>
    by_cases hv : v = 0
    · subst hv; exact ⟨_, ⟨j, hi, hj, rfl⟩, rfl, rfl⟩
    · simp [hv] at he
  | seq _ ih =>
    obtain ⟨x, hx, ha, hr⟩ := ih (Stmt.seq.inj he).1
    exact ⟨x, hx, ha, hr⟩
  | ite_true => exact absurd (congrArg sizeOf he) (Nat.ne_of_lt Stmt.size_lt_ite_left)
  | ite_false => exact absurd (congrArg sizeOf he) (Nat.ne_of_lt Stmt.size_lt_ite_right)
  | seq_skip => exact absurd (congrArg sizeOf he) (Nat.ne_of_lt Stmt.size_lt_seq_right)
  | load | store | fetchAdd | cas_eq | cas_ne | rmwStore | fence | assign | loop | loadAt
  | storeAt | assignAt => cases he

/-- Each step of a code that waits on `x` reads `x`. When it reads 0, the code stays as it was. -/
theorem Stmt.WaitsOn.step {s regs x a s' regs'} (hw : s.WaitsOn regs x)
    (h : Step s regs a s' regs') :
    ∃ v, a = some ⟨.read, .sc, x, v, false⟩ ∧ regs' = regs ∧ (v = 0 → s' = s) := by
  induction h with
  | wait v => exact ⟨v, by rw [show _ = x from hw], rfl, fun hv => by simp [hv]⟩
  | waitAt j v hi _ =>
    obtain ⟨j', hi', _, rfl⟩ := hw
    obtain rfl : j = j' := by rw [hi] at hi'; exact Int.ofNat.inj hi'
    exact ⟨v, rfl, rfl, fun hv => by simp [hv]⟩
  | seq _ ih =>
    obtain ⟨v, ha, hr, hs⟩ := ih hw
    exact ⟨v, ha, hr, fun hv => by rw [hs hv]⟩
  | seq_skip => exact hw.elim
  | load | store | fetchAdd | cas_eq | cas_ne | rmwStore | fence | assign | ite_true | ite_false
  | loop | loadAt | storeAt | assignAt => exact hw.elim

/-- The two accesses of a read-modify-write leave the code at a new point. -/
theorem Step.rmw_ne {s regs a s₁ regs₁ b s₂ regs₂} (h₁ : Step s regs a s₁ regs₁) (ha : a ≠ none)
    (hp : s₁.Pending) (h₂ : Step s₁ regs₁ b s₂ regs₂) : s₂ ≠ s := by
  induction h₁ generalizing s₂ regs₂ b with
  | fetchAdd | cas_eq => cases h₂; exact nofun
  | seq _ ih =>
    cases h₂ with
    | seq_skip => exact hp.elim
    | seq h₂ => exact fun he => ih ha hp h₂ (Stmt.seq.inj he).1
  | wait v | waitAt _ v =>
    by_cases hv : v = 0 <;> simp [hv, Stmt.Pending] at hp
  | load | store | cas_ne | rmwStore | fence | loadAt | storeAt => simp [Stmt.Pending] at hp
  | assign | seq_skip | ite_true | ite_false | loop | assignAt => exact absurd rfl ha

namespace Cpu

theorem move_self (c : Cpu) (t : Nat) : c.move t (c.code t) (c.regs t) c.mem = c := by
  cases c
  simp only [move, mk.injEq, true_and]
  exact ⟨funext fun i => by by_cases h : i = t <;> simp [setAt, h],
    funext fun i => by by_cases h : i = t <;> simp [setAt, h]⟩

/-- Thread `t` waits: its code waits on a place that holds 0 or has no value. -/
def Waits (c : Cpu) (t : Nat) : Prop :=
  ∃ x, (c.code t).WaitsOn (c.regs t) x ∧ ∀ v, c.mem x = some v → v = 0

/-- Each thread is done or waits. -/
def Quiet (c : Cpu) : Prop := ∀ t, c.code t = .skip ∨ c.Waits t

/-- A step of a thread that leaves its code as it was is a wait that read 0. Then the step leaves
the whole node as it was. -/
theorem Step.code_eq {c c' : Cpu} {t ls} (h : Step c t ls c') (he : c'.code t = c.code t) :
    c.Waits t ∧ c' = c := by
  cases h with
  | silent hs =>
    obtain ⟨_, _, ha, _⟩ := hs.eq_self (by simpa [move] using he)
    cases ha
  | one hs _ _ hf =>
    obtain ⟨x, hx, ha, hr⟩ := hs.eq_self (by simpa [move] using he)
    cases ha
    have hm := hf rfl
    refine ⟨⟨x, hx, fun v hv => by rw [hm] at hv; exact (Option.some.inj hv).symm⟩, ?_⟩
    simp only [move, setAt_same] at he
    rw [he, hr]
    exact move_self c t
  | two hs _ hp _ hs₂ => exact absurd (by simpa [move] using he) (hs.rmw_ne nofun hp hs₂)

/-- W6: if no step of a CPU node changes it, then each of its threads waits or has no step. -/
theorem stops {c : Cpu} (h : ∀ t ls c', Step c t ls c' → c' = c) (t : Nat) :
    c.Waits t ∨ ∀ ls c', ¬ Step c t ls c' :=
  Classical.byCases (p := ∃ ls c', Step c t ls c')
    (fun ⟨_, _, hs⟩ => .inl (hs.code_eq (by rw [h _ _ _ hs])).1)
    (fun hn => .inr fun ls c' hs => hn ⟨ls, c', hs⟩)

/-- When each thread is done or waits, each step of the node leaves it as it was. -/
theorem quiet_step {c c' : Cpu} {t ls} (hq : c.Quiet) (h : Step c t ls c') : c' = c := by
  rcases hq t with hd | ⟨x, hx, h0⟩
  · cases h with
    | silent hs | one hs | two hs => rw [hd] at hs; cases hs
  · cases h with
    | silent hs => obtain ⟨_, ha, _⟩ := hx.step hs; cases ha
    | one hs _ _ hf =>
      obtain ⟨v, ha, _, _⟩ := hx.step hs
      cases ha
      have hv := h0 v (hf rfl)
      subst hv
      exact (Step.code_eq (.one hs ‹_› ‹_› hf) (by simp_all [move])).2
    | two hs hn hp hf =>
      obtain ⟨v, ha, _, hs'⟩ := hx.step hs
      cases ha
      rw [hs' (h0 v (hf rfl))] at hp
      exact absurd hp hn

/-- In a fleet of CPU nodes, a quiet node stays quiet and sends nothing. -/
theorem keepsQuiet {M : Type} {F : Fleet Cpu M} (hF : ∀ n, F.node n = node M) :
    F.KeepsQuiet fun _ c => c.Quiet := by
  intro n s a s' out hq hs
  rw [hF] at hs
  obtain ⟨rfl, _, hs⟩ := hs
  exact ⟨quiet_step hq hs ▸ hq, rfl⟩

/-- F9 for CPU nodes: in a deadlocked fleet, each step leaves the configuration as it was. So no
waiting thread ever continues. -/
theorem frozen {M : Type} {F : Fleet Cpu M} (hF : ∀ n, F.node n = node M)
    {c₀ : Fleet.Config Cpu M} {r : Nat → Fleet.Config Cpu M} (hr : F.IsRun c₀ r) {i : Nat}
    (h : Fleet.Deadlocked (fun _ c => c.Quiet) (r i)) : ∀ j, i ≤ j → r j = r i := by
  intro j hij
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hij
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [← ih (Nat.le_add_right _ _), ← Nat.add_assoc]
    rcases hr.next (i + k) with ⟨_, hs⟩ | he
    · have hd := Fleet.Deadlocked.forever hr (keepsQuiet hF) h (i + k) (Nat.le_add_right _ _)
      generalize r (i + k + 1) = c' at hs ⊢
      rcases hs.cases with ⟨n, a, s', out, hs, hst, hfl, hse⟩ | ⟨_, hi, _⟩
      · rw [hF] at hs
        obtain ⟨rfl, _, hs⟩ := hs
        have e := quiet_step (hd.2 n) hs
        subst e
        cases c'
        simp only at hst hfl hse
        subst hst hfl hse
        simp only [Fleet.number, List.append_nil, List.length_nil, Nat.add_zero]
        congr 1
        funext m
        by_cases e : m = n <;> simp [setAt, e]
      · simp [hd.1] at hi
    · exact he

end Cpu

end Vx
