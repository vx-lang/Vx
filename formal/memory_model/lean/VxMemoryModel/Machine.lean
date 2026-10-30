/-!
# A fleet of nodes that pass messages

The progress model of #2360. A fleet is a set of nodes. A node is a group of memories that use
each other directly, such as a CPU with its memory, or a GPU. The nodes share no memory. They pass
messages, as in Lamport's model of a distributed system.

A node is opaque: the fleet sees only its steps, the messages that it sends, and the messages that
it takes. A node can always take a message, so a message in flight stays deliverable.
`Interleave.lean` makes a CPU node from the code of `Thread.lean`.

A run is an infinite sequence of configurations. Each step of a run is a step of one node, a
delivery of one message, or a step that changes nothing. A run is fair when each action that can
happen at each point from some point on happens again later (weak fairness).
-/

namespace Vx

/-- The function `f` with the value `v` at `k`. -/
def setAt {α : Type _} (f : Nat → α) (k : Nat) (v : α) : Nat → α :=
  fun i => if i = k then v else f i

@[simp] theorem setAt_same {α : Type _} (f : Nat → α) (k : Nat) (v : α) : setAt f k v k = v := by
  simp [setAt]

theorem setAt_other {α : Type _} (f : Nat → α) {k i : Nat} (v : α) (h : i ≠ k) :
    setAt f k v i = f i := by
  simp [setAt, h]

/-- A node with states in `S` and messages in `M`. -/
structure Node (S M : Type) where
  /-- `step s a s' out`: the action `a` of the node takes it from `s` to `s'` and sends the
  messages `out`. Each message names the node that gets it. -/
  step : S → Nat → S → List (Nat × M) → Prop
  /-- The node takes the message `m`. A node can always take a message. -/
  recv : S → M → S

/-- A message in flight. The fleet gives each message its own number. -/
structure Msg (M : Type) where
  id : Nat
  dst : Nat
  data : M

structure Fleet (S M : Type) where
  node : Nat → Node S M

namespace Fleet

variable {S M : Type} (F : Fleet S M)

structure Config (S M : Type) where
  state : Nat → S
  /-- The messages that the fleet sent and did not deliver yet. -/
  flight : List (Msg M)
  /-- The number of messages sent so far. The next message gets this number. -/
  sent : Nat

/-- Give the numbers `n`, `n + 1`, ... to the messages `out`. -/
def number (n : Nat) : List (Nat × M) → List (Msg M)
  | [] => []
  | p :: out => ⟨n, p.1, p.2⟩ :: number (n + 1) out

/-- An action of the fleet, for fairness. -/
inductive Act where
  /-- The action `a` of the node `n`. -/
  | move (n a : Nat)
  /-- The delivery of the message with the number `id`. -/
  | deliver (id : Nat)
  deriving DecidableEq

inductive Step : Act → Config S M → Config S M → Prop
  | move {c n a s' out} : (F.node n).step (c.state n) a s' out →
    Step (.move n a) c
      ⟨setAt c.state n s', c.flight ++ number c.sent out, c.sent + out.length⟩
  | deliver {c} (i : Nat) (h : i < c.flight.length) :
    Step (.deliver c.flight[i].id) c
      ⟨setAt c.state c.flight[i].dst ((F.node c.flight[i].dst).recv (c.state c.flight[i].dst)
        c.flight[i].data), c.flight.eraseIdx i, c.sent⟩

/-- A step of a run: a step of the fleet, or a step that changes nothing. -/
def Next (c c' : Config S M) : Prop := (∃ act, F.Step act c c') ∨ c' = c

def Enabled (act : Act) (c : Config S M) : Prop := ∃ c', F.Step act c c'

structure IsRun (c₀ : Config S M) (r : Nat → Config S M) : Prop where
  start : r 0 = c₀
  next : ∀ i, F.Next (r i) (r (i + 1))

/-- Weak fairness of `act`: if `act` can happen at each point from `i` on, it happens at some
point from `i` on. -/
def WeakFair (r : Nat → Config S M) (act : Act) : Prop :=
  ∀ i, (∀ j, i ≤ j → F.Enabled act (r j)) → ∃ j, i ≤ j ∧ F.Step act (r j) (r (j + 1))

def Fair (r : Nat → Config S M) : Prop := ∀ act, F.WeakFair r act

variable {F}

/-- After a step of the node `n`, the node is in a state that its own step gave. -/
theorem Step.move_state {act : Act} {n a : Nat} {c c' : Config S M} (h : F.Step act c c')
    (hact : act = .move n a) : ∃ s' out, (F.node n).step (c.state n) a s' out ∧ c'.state n = s' := by
  cases h with
  | move h =>
    cases hact
    exact ⟨_, _, h, by simp⟩
  | deliver => cases hact

/-- An action that stops being possible at some point is weakly fair. -/
theorem weakFair_of_disabled {r : Nat → Config S M} {act : Act} {k : Nat}
    (h : ∀ j, k ≤ j → ¬ F.Enabled act (r j)) : F.WeakFair r act := fun i hi =>
  absurd (hi (max i k) (Nat.le_max_left _ _)) (h _ (Nat.le_max_right _ _))

/-- A run that reaches a configuration where no action is possible is fair. -/
theorem fair_of_halt {r : Nat → Config S M} {k : Nat} (h : ∀ act j, k ≤ j → ¬ F.Enabled act (r j)) :
    F.Fair r := fun act => weakFair_of_disabled (h act)

end Fleet

end Vx
