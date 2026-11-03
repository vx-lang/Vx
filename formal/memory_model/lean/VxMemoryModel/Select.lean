/-!
# The round-robin bound of `select` (M3)

`Selector::select` in `stdlib/std/sync/channel.vx` holds a list of `n` receivers. A call tries
them in a round, from a start index: `start`, `start + 1`, ... modulo `n`. It returns the first one that
gives a value. `pick` models this search. Each try sees whether its receiver has a value then
(`ready`).

M3 asks for a bound: if a receiver is ready at each of `n` calls in a row, one of them chooses
it. `within_n` shows the bound for each rule of the start that moves the start past the
receiver that the last call chose, but not past the ready one. Two rules do this:

- `after_last_within_n`: the next call starts after the receiver that this call chose. This is
  the rule of the code: `Selector` keeps the start in its field `next`.
- `count_within_n`: the rule that `select` used before. The start is the count of channel events
  modulo `n`. Each value that `select` takes is one event. So when no other event comes between
  two calls, the start moves by one.
- `count_starves`: with the rule of the count, other events can keep each call from choosing a
  ready receiver. With two receivers and two events between calls, each call starts at 0. This
  is why the code changed to the rule of `after_last_within_n`.
-/

namespace Vx.Select

/-- The search of a call: try the receivers `(s + j) % n`, `(s + j + 1) % n`, ... for `fuel`
tries, and give the first one that has a value. -/
def scan (n s : Nat) (ready : Nat → Bool) (j : Nat) : Nat → Option Nat
  | 0 => none
  | fuel + 1 => if ready ((s + j) % n) then some ((s + j) % n) else scan n s ready (j + 1) fuel

/-- The receiver that a call with the start `s` chooses, if one has a value. -/
def pick (n s : Nat) (ready : Nat → Bool) : Option Nat := scan n s ready 0 n

/-- If the receiver at the offset `d` from the start has a value, the search chooses the receiver
at an offset `e ≤ d`. -/
theorem scan_le {n s : Nat} {ready : Nat → Bool} :
    ∀ fuel j d, j ≤ d → d < j + fuel → ready ((s + d) % n) = true →
      ∃ e, j ≤ e ∧ e ≤ d ∧ scan n s ready j fuel = some ((s + e) % n) := by
  intro fuel
  induction fuel with
  | zero => intro j d _ h; omega
  | succ fuel ih =>
    intro j d hj hd hr
    by_cases h : ready ((s + j) % n) = true
    · exact ⟨j, Nat.le_refl _, hj, by simp [scan, h]⟩
    · have hjd : j ≠ d := fun e => h (e ▸ hr)
      obtain ⟨e, h1, h2, h3⟩ := ih (j + 1) d (by omega) (by omega) hr
      exact ⟨e, by omega, h2, by simp [scan, h, h3]⟩

theorem pick_le {n s d : Nat} {ready : Nat → Bool} (hd : d < n) (hr : ready ((s + d) % n) = true) :
    ∃ e, e ≤ d ∧ pick n s ready = some ((s + e) % n) := by
  obtain ⟨e, _, h2, h3⟩ := scan_le n 0 d (Nat.zero_le _) (by omega) hr
  exact ⟨e, h2, h3⟩

/-- Each receiver `i < n` is at some offset `d < n` from each start `s`. -/
theorem exists_offset {n s i : Nat} (hi : i < n) : ∃ d, d < n ∧ (s + d) % n = i := by
  have hs := Nat.mod_add_div s n
  have hlt : s % n < n := Nat.mod_lt _ (by omega)
  by_cases h : s % n ≤ i
  · refine ⟨i - s % n, by omega, ?_⟩
    rw [show s + (i - s % n) = i + n * (s / n) by omega, Nat.add_mul_mod_self_left,
      Nat.mod_eq_of_lt hi]
  · refine ⟨i + n - s % n, by omega, ?_⟩
    rw [show s + (i + n - s % n) = i + n * (s / n + 1) by rw [Nat.mul_add]; omega,
      Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hi]

/-- The bound for each rule of the start. The call `c` starts at `start c`, and its tries see
`ready c`. Assume that when a call does not choose the receiver `i`, the offset of `i` from the
start of the next call is smaller. Let `d` be the offset of `i` from the start of the call `c₀`.
If `i` is ready at the calls `c₀` to `c₀ + d`, one of them chooses it. -/
theorem within {n i : Nat} {start : Nat → Nat} {ready : Nat → Nat → Bool}
    (hnext : ∀ c d e, d < n → (start c + d) % n = i → e < d →
      pick n (start c) (ready c) = some ((start c + e) % n) →
      ∃ d', d' < d ∧ (start (c + 1) + d') % n = i) :
    ∀ d c₀, d < n → (start c₀ + d) % n = i → (∀ c, c₀ ≤ c → c ≤ c₀ + d → ready c i = true) →
      ∃ c, c₀ ≤ c ∧ c ≤ c₀ + d ∧ pick n (start c) (ready c) = some i := by
  intro d
  induction d using Nat.strongRecOn with
  | _ d ih =>
    intro c₀ hd hoff hready
    obtain ⟨e, he, hp⟩ := pick_le (ready := ready c₀) hd
      (by rw [hoff]; exact hready c₀ (Nat.le_refl _) (by omega))
    by_cases h : e = d
    · subst h
      exact ⟨c₀, Nat.le_refl _, by omega, by rw [hp, hoff]⟩
    · obtain ⟨d', hd', hoff'⟩ := hnext c₀ d e hd hoff (by omega) hp
      obtain ⟨c, h1, h2, h3⟩ := ih d' hd' (c₀ + 1) (by omega) hoff'
        fun c h1 h2 => hready c (by omega) (by omega)
      exact ⟨c, by omega, by omega, h3⟩

/-- M3 for each rule of the start that meets `hnext`: if a receiver is ready at each of the `n`
calls from `c₀` on, one of them chooses it. -/
theorem within_n {n i : Nat} {start : Nat → Nat} {ready : Nat → Nat → Bool} (hi : i < n)
    (hnext : ∀ c d e, d < n → (start c + d) % n = i → e < d →
      pick n (start c) (ready c) = some ((start c + e) % n) →
      ∃ d', d' < d ∧ (start (c + 1) + d') % n = i)
    {c₀ : Nat} (hready : ∀ c, c₀ ≤ c → c < c₀ + n → ready c i = true) :
    ∃ c, c₀ ≤ c ∧ c < c₀ + n ∧ pick n (start c) (ready c) = some i := by
  obtain ⟨d, hd, hoff⟩ := exists_offset (s := start c₀) hi
  obtain ⟨c, h1, h2, h3⟩ := within hnext d c₀ hd hoff fun c h1 h2 => hready c h1 (by omega)
  exact ⟨c, h1, by omega, h3⟩

/-- The rule of the code: the next call starts after the receiver that this call chose. -/
theorem after_last_within_n {n i : Nat} {start : Nat → Nat} {ready : Nat → Nat → Bool}
    (hi : i < n)
    (hstart : ∀ c k, pick n (start c) (ready c) = some k → start (c + 1) = (k + 1) % n)
    {c₀ : Nat} (hready : ∀ c, c₀ ≤ c → c < c₀ + n → ready c i = true) :
    ∃ c, c₀ ≤ c ∧ c < c₀ + n ∧ pick n (start c) (ready c) = some i := by
  refine within_n hi (fun c d e _ hoff he hp => ⟨d - e - 1, by omega, ?_⟩) hready
  rw [hstart c _ hp, Nat.mod_add_mod, Nat.add_assoc, Nat.mod_add_mod, ← hoff]
  congr 1
  omega

/-- The rule that `select` used before: the start is the count `seen c` of channel events modulo
`n`. When the only event between two calls is the value that the first call took, the count goes
up by one. -/
theorem count_within_n {n i : Nat} {seen : Nat → Nat} {ready : Nat → Nat → Bool} (hi : i < n)
    {c₀ : Nat} (hseen : ∀ c, c₀ ≤ c → seen (c + 1) = seen c + 1)
    (hready : ∀ c, c₀ ≤ c → c < c₀ + n → ready c i = true) :
    ∃ c, c₀ ≤ c ∧ c < c₀ + n ∧ pick n (seen c % n) (ready c) = some i := by
  -- Count the calls from `c₀`, so that the hypothesis on `seen` holds at each of them.
  let start := fun c => seen (c₀ + c) % n
  have h := within_n (start := start) (ready := fun c => ready (c₀ + c)) hi
    (fun c d e _ hoff he _ => ⟨d - 1, by omega, ?_⟩) (c₀ := 0)
    fun c _ h2 => hready (c₀ + c) (by omega) (by omega)
  · obtain ⟨c, _, h2, h3⟩ := h
    exact ⟨c₀ + c, by omega, by omega, h3⟩
  · simp only [start] at hoff ⊢
    rw [show c₀ + (c + 1) = c₀ + c + 1 by omega, hseen _ (by omega), Nat.mod_add_mod, ← hoff,
      Nat.mod_add_mod]
    congr 1
    omega

/-- The rule of the count does not give M3 when other events come between the calls. Two
receivers have a value at each call. Each call takes one value, and then a `send` to the
receiver 0 adds a second event. So the count at the call `c` is `2 * c`, each call starts at 0,
and no call chooses the receiver 1. -/
theorem count_starves : ∀ c, pick 2 ((2 * c) % 2) (fun _ => true) ≠ some 1 := by
  intro c
  rw [Nat.mul_mod_right]
  decide

end Vx.Select
