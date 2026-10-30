import VxMemoryModel.Progress

/-!
# An example of WF1: a message arrives

In the fleet `ping`, node 0 sends one message to node 1 and then stops. Each node counts the
messages that it takes. `ping_arrives` uses WF1 twice: node 0 sends at some point, and then the
fleet delivers the message. `idle_never_arrives` shows a run that misses it, and so is not fair.
-/

namespace Vx.ProgressExample

open Fleet

/-- Node 0 sends one message to node 1, and then stops. A node counts the messages it takes. -/
def ping : Fleet Nat Unit :=
  ⟨fun n => ⟨fun s a s' out => n = 0 ∧ a = 0 ∧ s = 0 ∧ s' = 1 ∧ out = [(1, ())], fun s _ => s + 1⟩⟩

def start : Config Nat Unit := ⟨fun _ => 0, [], 0⟩

/-- Each message in flight goes to node 1. -/
theorem to_one {r : Nat → Config Nat Unit} (hr : ping.IsRun start r) :
    ∀ i, ∀ m ∈ (r i).flight, m.dst = 1 := by
  intro i
  induction i with
  | zero => rw [hr.start]; simp [start]
  | succ i ih =>
    rcases hr.next i with ⟨act, hs⟩ | he
    · generalize r (i + 1) = c' at hs ⊢
      cases hs with
      | move h =>
        obtain ⟨_, _, _, _, rfl⟩ := h
        intro m hm
        rcases List.mem_append.1 hm with hm | hm
        · exact ih m hm
        · simp [number] at hm
          rw [hm]
      | deliver j hj => exact fun m hm => ih m (List.mem_of_mem_eraseIdx hm)
    · rw [he]; exact ih

/-- In a fair run, node 1 takes a message at some point. -/
theorem ping_arrives {r : Nat → Config Nat Unit} (hr : ping.IsRun start r) (hf : ping.Fair r) :
    ∃ j, 0 < (r j).state 1 := by
  -- Node 0 sends: until it does, nothing else can happen.
  obtain ⟨j, _, hj⟩ := wf1_taken hr (hf (.move 0 0))
    (P := fun c => c.state 0 = 0 ∧ c.flight = [])
    (fun act c c' hP hs ha => by
      cases hs with
      | move h => obtain ⟨rfl, rfl, _⟩ := h; exact absurd rfl ha
      | deliver i hi => simp [hP.2] at hi)
    (fun c hP => ⟨_, .move ⟨rfl, rfl, hP.1, rfl, rfl⟩⟩)
    (i := 0) (by rw [hr.start]; exact ⟨rfl, rfl⟩)
  -- The message is in flight, so the fleet delivers it.
  have hin : InFlight (r j).sent (r (j + 1)) := by
    generalize r (j + 1) = c' at hj ⊢
    cases hj with
    | move h =>
      obtain ⟨_, _, _, _, rfl⟩ := h
      exact ⟨⟨(r j).sent, 1, ()⟩, by simp [number], rfl⟩
  obtain ⟨k, _, hk⟩ := delivers hr hf hin
  refine ⟨k + 1, ?_⟩
  have hdst := to_one hr k
  obtain ⟨i, hi, _, hs⟩ := hk.deliver_inv
  rw [hdst _ (List.getElem_mem hi)] at hs
  rw [hs]
  simp [ping]

/-- A run where nothing moves. -/
def idle (_ : Nat) : Config Nat Unit := start

/-- In `idle`, node 1 never takes a message. So by `ping_arrives`, the run is not fair. -/
theorem idle_never_arrives :
    ping.IsRun start idle ∧ (∀ j, (idle j).state 1 = 0) ∧ ¬ ping.Fair idle := by
  have hr : ping.IsRun start idle := ⟨rfl, fun _ => Or.inr rfl⟩
  refine ⟨hr, fun _ => rfl, fun hf => ?_⟩
  obtain ⟨j, hj⟩ := ping_arrives hr hf
  exact absurd hj (by simp [idle, start])

end Vx.ProgressExample
