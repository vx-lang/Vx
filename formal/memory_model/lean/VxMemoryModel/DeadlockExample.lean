import VxMemoryModel.CpuDeadlock
import VxMemoryModel.ProgressExample

/-!
# Examples of a deadlock

- `cross_frozen`: in `cross`, each thread waits for the flag of the other, and each flag holds 0.
  The fleet is deadlocked at the start, so by `Cpu.frozen` no run ever moves it.
- `crossing_cut`: a cut of the fleet `ping` in which each node records that it is quiet, but the
  message of node 0 crosses the cut. The fleet is not deadlocked at the end of that snapshot. So
  `snapshot` needs its hypothesis on the cut.
-/

namespace Vx.DeadlockExample

open Fleet

/-- Each thread waits for the flag of the other, and then sets its own. -/
def cross : Prog where
  code
    | 0 => .seq (.wait 1) (.signal 0)
    | 1 => .seq (.wait 0) (.signal 1)
    | _ => .skip

/-- A CPU that runs `cross`. Each place holds 0. -/
def waiting : Cpu := ⟨fun _ => some 0, cross.code, fun _ _ => 0⟩

theorem waiting_quiet : waiting.Quiet
  | 0 => .inr ⟨1, rfl, fun _ h => (Option.some.inj h).symm⟩
  | 1 => .inr ⟨0, rfl, fun _ h => (Option.some.inj h).symm⟩
  | _ + 2 => .inl rfl

def fleet : Fleet Cpu Unit := ⟨fun _ => Cpu.node Unit⟩

/-- Each node runs `cross`, and no message is in flight. -/
def cfg : Config Cpu Unit := ⟨fun _ => waiting, [], 0⟩

/-- No run from `cfg` ever moves. So thread 0 of node 0 stays at its wait. -/
theorem cross_frozen {r : Nat → Config Cpu Unit} (hr : fleet.IsRun cfg r) (j : Nat) :
    r j = cfg ∧ ((r j).state 0).code 0 = .seq (.wait 1) (.signal 0) := by
  have h := Cpu.frozen (fun _ => rfl) hr (i := 0)
    (by rw [hr.start]; exact ⟨rfl, fun _ => waiting_quiet⟩) j (Nat.zero_le _)
  rw [hr.start] at h
  exact ⟨h, by rw [h]; rfl⟩

/-! ## A cut that a message crosses -/

open ProgressExample

/-- Node 0 is quiet after it sent its message. Node 1 is always quiet. -/
def Q (n : Nat) (s : Nat) : Prop := n = 0 → s = 1

theorem ping_keepsQuiet : ping.KeepsQuiet Q := by
  intro n s a s' out hq ⟨hn, _, hs, _⟩
  exact absurd (hs ▸ hq hn) (by decide)

/-- After node 0 sent. -/
def c1 : Config Nat Unit :=
  ⟨setAt start.state 0 1, start.flight ++ number 0 start.sent [(1, ())], start.sent + 1⟩

/-- After the fleet delivered the message. -/
def c2 : Config Nat Unit := ⟨setAt c1.state 1 1, [], 1⟩

def run : Nat → Config Nat Unit
  | 0 => start
  | 1 => c1
  | _ => c2

theorem run_isRun : ping.IsRun start run where
  start := rfl
  next
    | 0 => .inl ⟨_, .move ⟨rfl, rfl, rfl, rfl, rfl⟩⟩
    | 1 => by
      have e : run 2 = ⟨setAt c1.state 1 (c1.state 1 + 1), c1.flight.eraseIdx 0, c1.sent⟩ := by
        simp only [run, c2, Config.mk.injEq]
        exact ⟨funext fun i => by simp [setAt, c1, start], rfl, rfl⟩
      rw [e]
      exact .inl ⟨_, .deliver (c := c1) 0 (by simp [c1, number])⟩
    | _ + 2 => .inr rfl

/-- Node 0 records at point 1, after it sent. Node 1 records at point 0, before the delivery. -/
def t : Nat → Nat
  | 0 => 1
  | _ => 0

/-- Each node records that it is quiet, but the fleet is not deadlocked at point 1: the message
is still in flight. It crosses the cut, because node 0 sent it before it recorded and node 1
gets it after it recorded. -/
theorem crossing_cut :
    (∀ n, Q n ((run (t n)).state n)) ∧ ¬ Deadlocked Q (run 1) ∧
    ¬ ∀ j, ∀ m ∈ (run j).flight, m.id < (run (t m.src)).sent → j < t m.dst := by
  refine ⟨fun n hn => by subst hn; rfl,
    fun h => by simp [Deadlocked, run, c1, start, number] at h, fun h => ?_⟩
  have := h 1 ⟨0, 0, 1, ()⟩ (by simp [run, c1, start, number]) (by simp [run, t, c1, start])
  simp [t] at this

end Vx.DeadlockExample
