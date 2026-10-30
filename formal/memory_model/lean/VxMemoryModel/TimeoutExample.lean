import VxMemoryModel.Progress

/-!
# A wait across nodes ends: a reply comes, or the timeout fails the task

A node cannot tell a dead node from a slow one. So a wait for a message from another node has a
timeout, and the timeout is an error in the task. Then W7 needs no hypothesis on the other nodes.

- `wait_ends`: in each fair run of each fleet whose node 0 is `waiter`, the wait of node 0 ends.
  The task is done, or it failed with the timeout error.
- `no_timeout_hangs`: without the timeout, a fair run can keep node 0 waiting forever, when no
  node ever replies.
-/

namespace Vx.TimeoutExample

open Fleet

/-- The state 0 is "waits for a reply", 1 is "done" and 2 is "failed with the timeout error".
The action 0 is the timeout. A reply ends the wait. -/
def waiter : Node Nat Unit :=
  ⟨fun s a s' out => a = 0 ∧ s = 0 ∧ s' = 2 ∧ out = [], fun s _ => if s = 0 then 1 else s⟩

/-- Whatever the other nodes do, in a fair run the wait of node 0 ends. -/
theorem wait_ends {F : Fleet Nat Unit} (hF : F.node 0 = waiter) {c₀ : Config Nat Unit}
    {r : Nat → Config Nat Unit} (hr : F.IsRun c₀ r) (hf : F.Fair r) {i : Nat}
    (hi : (r i).state 0 = 0) : ∃ k, i ≤ k ∧ (r k).state 0 ≠ 0 :=
  wf1 hr (hf (.move 0 0)) (P := fun c => c.state 0 = 0) (Q := fun c => c.state 0 ≠ 0)
    (fun _ c' _ _ => Classical.em (c'.state 0 = 0))
    (fun c h => ⟨_, .move (s' := 2) (out := []) (by rw [hF]; exact ⟨rfl, h, rfl, rfl⟩)⟩)
    (fun c c' _ hs => by
      obtain ⟨_, _, h, e⟩ := hs.move_state rfl
      rw [hF] at h
      obtain ⟨_, _, rfl, _⟩ := h
      rw [e]
      decide)
    i hi

/-- A fleet with no timeout, in which no node ever moves. A node that gets a message is done. -/
def silent : Fleet Nat Unit := ⟨fun _ => ⟨fun _ _ _ _ => False, fun s _ => if s = 0 then 1 else s⟩⟩

def start : Config Nat Unit := ⟨fun _ => 0, [], 0⟩

/-- The run that stays at `start` is fair, because no action can happen. Node 0 waits forever. -/
theorem no_timeout_hangs :
    silent.IsRun start (fun _ => start) ∧ silent.Fair (fun _ => start) ∧ start.state 0 = 0 := by
  refine ⟨⟨rfl, fun _ => .inr rfl⟩, fair_of_halt (k := 0) fun act j _ ⟨c', h⟩ => ?_, rfl⟩
  cases h with
  | move h => exact h
  | deliver i hi => exact absurd hi (Nat.not_lt_zero _)

end Vx.TimeoutExample
