import VxMemoryModel.Finish

/-!
# A FIFO scheduler is fair

The shared runtime runs many tasks on a few worker threads (`stdlib/rust_core/src/ffi/sched.rs`).
A task there is a fiber. The ready fibers wait in one queue. A worker takes the fiber at the
front. A wake or a spawn puts a fiber at the back. A fiber runs on its worker until it waits or
ends. A fiber that waits is parked: it is in no queue until a wake.

- `queue_bound`: assume that, while the queue is not empty, a worker takes a fiber within `B`
  points. Then the workers take a fiber at the position `p` of the queue within `(p + 1) * B`
  points. A push does not delay it, because a push goes to the back.
- `fifo_fair`: the scheduler gives the weak fairness that `Cpu.finishes` takes as a hypothesis,
  for a thread that does not wait. A thread that waits is parked and does not spin. Its spin
  would change nothing (`Cpu.Step.code_eq`).
- `Cpu.finishes_fifo`: W7 for a CPU node whose threads run on this scheduler, with no fairness
  hypothesis on the threads.
- `lifo_starves`: with a push to the front, a fiber can wait forever.
-/

namespace Vx.Sched

/-- A step of the ready queue. A worker takes the fiber at the front, a wake or a spawn puts a
fiber at the back, or the queue stays as it was. -/
inductive QStep : List Nat → List Nat → Prop
  | take (t : Nat) (q : List Nat) : QStep (t :: q) q
  | push (q : List Nat) (t : Nat) : QStep q (q ++ [t])
  | stay (q : List Nat) : QStep q q

/-- The queue at each point of a run of the scheduler. -/
def IsQueueRun (q : Nat → List Nat) : Prop := ∀ j, QStep (q j) (q (j + 1))

/-- At the point `m`, a worker takes the fiber `t` from the front of the queue. -/
def Takes (q : Nat → List Nat) (t m : Nat) : Prop := q m = t :: q (m + 1)

/-- At the point `m`, a worker takes a fiber. -/
def TakesAny (q : Nat → List Nat) (m : Nat) : Prop := ∃ t, Takes q t m

/-- A step that takes no fiber keeps the queue as a prefix. -/
theorem QStep.keep {a b : List Nat} (h : QStep a b) (hn : ∀ t, a ≠ t :: b) : ∃ l, b = a ++ l := by
  cases h with
  | take t => exact absurd rfl (hn t)
  | push => exact ⟨_, rfl⟩
  | stay => exact ⟨[], by simp⟩

variable {q : Nat → List Nat}

/-- While no worker takes a fiber, the queue only grows at the back. -/
theorem keep (hq : IsQueueRun q) {j : Nat} :
    ∀ d, (∀ m, j ≤ m → m < j + d → ¬ TakesAny q m) → ∃ l, q (j + d) = q j ++ l := by
  intro d
  induction d with
  | zero => exact fun _ => ⟨[], by simp⟩
  | succ d ih =>
    intro hn
    obtain ⟨l, hl⟩ := ih fun m h1 h2 => hn m h1 (by omega)
    obtain ⟨l', hl'⟩ := (hq (j + d)).keep fun t e => hn (j + d) (by omega) (by omega) ⟨t, e⟩
    exact ⟨l ++ l', by rw [← Nat.add_assoc, hl', hl, List.append_assoc]⟩

/-- The first point from `j` on where `P` holds. -/
theorem exists_first {P : Nat → Prop} {j : Nat} :
    ∀ m, (∃ k, j ≤ k ∧ k ≤ m ∧ P k) →
      ∃ k, j ≤ k ∧ k ≤ m ∧ P k ∧ ∀ k', j ≤ k' → k' < k → ¬ P k' := by
  intro m
  induction m with
  | zero =>
    rintro ⟨k, hj, hk, hp⟩
    exact ⟨k, hj, hk, hp, fun k' _ h => absurd h (by omega)⟩
  | succ m ih =>
    rintro ⟨k, hj, hk, hp⟩
    by_cases h : ∃ k, j ≤ k ∧ k ≤ m ∧ P k
    · obtain ⟨k₁, h1, h2, h3, h4⟩ := ih h
      exact ⟨k₁, h1, by omega, h3, h4⟩
    · exact ⟨k, hj, hk, hp, fun k' h1 h2 hp' => h ⟨k', h1, by omega, hp'⟩⟩

/-- The workers take a fiber at the position `p` of the queue within `(p + 1) * B` points, if
they take some fiber within `B` points of each point where the queue is not empty. -/
theorem queue_bound (hq : IsQueueRun q) {B : Nat}
    (hB : ∀ j, q j ≠ [] → ∃ m, j ≤ m ∧ m < j + B ∧ TakesAny q m) {t : Nat} :
    ∀ p j (l₁ l₂ : List Nat), q j = l₁ ++ t :: l₂ → l₁.length = p →
      ∃ m, j ≤ m ∧ m < j + (p + 1) * B ∧ Takes q t m := by
  intro p
  induction p with
  | zero =>
    intro j l₁ l₂ hj hp
    obtain ⟨m₀, h1, h2, h3⟩ := hB j (by simp [hj])
    obtain ⟨m, hjm, hm, ⟨t', ht'⟩, hfirst⟩ := exists_first m₀ ⟨m₀, h1, Nat.le_refl _, h3⟩
    obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hjm
    obtain ⟨l, hl⟩ := keep hq d hfirst
    rw [List.eq_nil_of_length_eq_zero hp] at hj
    have ht := ht'
    unfold Takes at ht'
    rw [hl, hj] at ht'
    simp only [List.nil_append, List.cons_append, List.cons.injEq] at ht'
    obtain ⟨rfl, _⟩ := ht'
    exact ⟨j + d, hjm, by omega, ht⟩
  | succ p ih =>
    intro j l₁ l₂ hj hp
    obtain ⟨m₀, h1, h2, h3⟩ := hB j (by simp [hj])
    obtain ⟨m, hjm, hm, ⟨t', ht'⟩, hfirst⟩ := exists_first m₀ ⟨m₀, h1, Nat.le_refl _, h3⟩
    obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hjm
    obtain ⟨l, hl⟩ := keep hq d hfirst
    obtain ⟨a, l₁, rfl⟩ : ∃ a l, l₁ = a :: l :=
      match l₁, hp with
      | a :: l, _ => ⟨a, l, rfl⟩
    unfold Takes at ht'
    rw [hl, hj] at ht'
    simp only [List.cons_append, List.append_assoc, List.cons.injEq] at ht'
    obtain ⟨m', h1', h2', h3'⟩ := ih (j + d + 1) l₁ (l₂ ++ l) (by rw [← ht'.2])
      (by simpa using hp)
    refine ⟨m', by omega, ?_, h3'⟩
    rw [Nat.succ_mul]
    omega

/-- The workers take each fiber in the queue at some point. -/
theorem taken (hq : IsQueueRun q) {B : Nat}
    (hB : ∀ j, q j ≠ [] → ∃ m, j ≤ m ∧ m < j + B ∧ TakesAny q m) {t j : Nat} (h : t ∈ q j) :
    ∃ m, j ≤ m ∧ Takes q t m := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem h
  obtain ⟨m, hm, _, ht⟩ := queue_bound hq hB _ j l₁ l₂ hl rfl
  exact ⟨m, hm, ht⟩

/-! ## The fairness that the scheduler gives -/

section Fair

variable {S M : Type} {F : Fleet S M} {r : Nat → Fleet.Config S M} {A : Fleet.Act}
  {P : Fleet.Config S M → Prop}

/-- A FIFO scheduler is weakly fair to the action `A` of the fiber `t` while `P` holds. The
scheduler runs the fiber `t` on a worker, and each step of the fiber is `A`. The hypotheses are:

- `hB`: while the queue is not empty, a worker takes a fiber within `B` points. The workers and
  the monitor thread of the runtime give it.
- `hwake`: when `P` holds and `A` can happen, the fiber is in the queue at some later point, or
  it moves. A fiber that can continue is ready or runs: a wake puts it in the queue.
- `hrun`: a fiber that a worker takes makes a step. -/
theorem fifo_fair (hq : IsQueueRun q) {B : Nat}
    (hB : ∀ j, q j ≠ [] → ∃ m, j ≤ m ∧ m < j + B ∧ TakesAny q m) {t : Nat}
    (hwake : ∀ j, P (r j) → F.Enabled A (r j) →
      ∃ k, j ≤ k ∧ (t ∈ q k ∨ F.Step A (r k) (r (k + 1))))
    (hrun : ∀ m, Takes q t m → ∃ k, m ≤ k ∧ F.Step A (r k) (r (k + 1))) :
    F.WeakFairWhen r A P := by
  intro i hi
  obtain ⟨k, hik, hk | hk⟩ := hwake i (hi i (Nat.le_refl _)).1 (hi i (Nat.le_refl _)).2
  · obtain ⟨m, hkm, hm⟩ := taken hq hB hk
    obtain ⟨k', hmk, hs⟩ := hrun m hm
    exact ⟨k', by omega, hs⟩
  · exact ⟨k, hik, hk⟩

end Fair

end Sched

namespace Cpu

open Sched

variable {M : Type} {F : Fleet Cpu M} {n t : Nat}

/-- W7 on the scheduler: a thread with no loop finishes, if it runs on a FIFO scheduler and meets
the conditions of `finishes`. The run need not be fair: the scheduler gives the fairness. -/
theorem finishes_fifo (hF : F.node n = node M) {c₀ : Fleet.Config Cpu M}
    {r : Nat → Fleet.Config Cpu M} (hr : F.IsRun c₀ r) {q : Nat → List Nat} (hq : IsQueueRun q)
    {B : Nat} (hB : ∀ j, q j ≠ [] → ∃ m, j ≤ m ∧ m < j + B ∧ TakesAny q m)
    (hwake : ∀ j, ¬ ((r j).state n).Waits t → F.Enabled (.move n t) (r j) →
      ∃ k, j ≤ k ∧ (t ∈ q k ∨ F.Step (.move n t) (r k) (r (k + 1))))
    (hrun : ∀ m, Takes q t m → ∃ k, m ≤ k ∧ F.Step (.move n t) (r k) (r (k + 1))) {i : Nat}
    (hloop : (((r i).state n).code t).NoLoop)
    (hok : ∀ j, i ≤ j → ((r j).state n).code t ≠ .skip → ¬ ((r j).state n).Waits t →
      F.Enabled (.move n t) (r j))
    (hcont : ∀ j, i ≤ j → ((r j).state n).Waits t → ∃ k, j ≤ k ∧ ¬ ((r k).state n).Waits t)
    (hstay : ∀ j, i ≤ j → ¬ ((r j).state n).Waits t → ¬ F.Step (.move n t) (r j) (r (j + 1)) →
      ¬ ((r (j + 1)).state n).Waits t) :
    ∃ k, i ≤ k ∧ ((r k).state n).code t = .skip :=
  finishes hF hr (fifo_fair hq hB hwake hrun) hloop hok hcont hstay

end Cpu

namespace Sched

/-! ## A stack is not fair -/

/-- The fiber 1 runs, and a wake puts it back at the front, before the fiber 0. -/
def lifo (j : Nat) : List Nat := if j % 2 = 0 then [1, 0] else [0]

/-- In `lifo`, a worker takes a fiber at each second point, but never the fiber 0. The queue does
not follow `QStep`: the wake puts the fiber at the front. -/
theorem lifo_starves :
    (∀ j, lifo j ≠ [] → ∃ m, j ≤ m ∧ m < j + 2 ∧ TakesAny lifo m) ∧ (∀ m, ¬ Takes lifo 0 m) ∧
      ¬ IsQueueRun lifo := by
  refine ⟨fun j _ => ?_, fun m h => ?_, fun h => ?_⟩
  · rcases Nat.mod_two_eq_zero_or_one j with e | e
    · refine ⟨j, Nat.le_refl _, by omega, 1, ?_⟩
      simp [Takes, lifo, e, show (j + 1) % 2 = 1 by omega]
    · refine ⟨j + 1, by omega, by omega, 1, ?_⟩
      simp [Takes, lifo, show (j + 1) % 2 = 0 by omega, show (j + 1 + 1) % 2 = 1 by omega]
  · rcases Nat.mod_two_eq_zero_or_one m with e | e
    · simp [Takes, lifo, e] at h
    · simp [Takes, lifo, e, show (m + 1) % 2 = 0 by omega] at h
  · have h1 := h 1
    simp only [lifo, show (1 : Nat) % 2 = 1 by rfl, show (1 + 1) % 2 = 0 by rfl] at h1
    cases h1

end Sched

end Vx
