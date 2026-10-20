import VxMemoryModel.Counter
import VxMemoryModel.Tasks

/-!
# Channels

A program with channels, as `channel<T>(n)` of Vx gives them. `send` and `recv` lower to `sc`
atomics on places that the program code cannot use, as `spawn` and `join` do in `Tasks.lean`. So
`Prog.drf_sc` covers a program with channels, with no new proof.

Each channel has two counters. A `send` takes its number `k` from the first counter with a
`fetchAdd`. It waits for the flag `taken k`, writes its value to element `k` of the values, and
sets the flag `full k`. A `recv` takes its number `k` from the second counter. It waits for
`full k`, reads element `k` of the values, and sets `taken (k + n)`, where `n` is the room of the
channel. At the start, thread 0 sets both counters and the flags `taken 0` to `taken (n - 1)`.
So each `send` waits for a `recv` `n` places before it.

`Counter.tickets` shows that no two `send`s get the same number, and no two `recv`s. So each flag
has one writer. Then `send_hb_recv` and `recv_hb_send_capacity` give the edges of M4 in
`docs/discussions/waiting_tasks_channels_and_task_errors.md`.

Each channel can carry `msgs` values in a run. A `send` or a `recv` with the number `msgs` or more
stops its thread, as a failed bounds check does. This keeps the number of places finite. The
theorems need `CoOrder` (see `Counter.lean`).
-/

namespace Vx

/-- The code of a program with channels: the code of `Tasks.lean`, with `send` and `recv`. -/
inductive CStmt where
  | base (s : Stmt)
  /-- Send the value of `e` on the channel `c`. -/
  | send (c : Nat) (e : Expr)
  /-- Take the next value of the channel `c` into the register `r`. -/
  | recv (r c : Nat)
  | seq (s t : CStmt)
  | ite (c : Expr) (s t : CStmt)
  | loop (c : Expr) (s : CStmt)

/-- `s` holds the code `b` of `Tasks.lean`. -/
def CStmt.Bases : CStmt → Stmt → Prop
  | .base s, b => s = b
  | .send .., _ | .recv .., _ => False
  | .seq s t, b | .ite _ s t, b => s.Bases b ∨ t.Bases b
  | .loop _ s, b => s.Bases b

/-- The two operations of a channel. -/
inductive Op where
  | send
  | recv
  deriving DecidableEq

/-- The operation that a `send` waits for is a `recv`, and the other way around. -/
def Op.other : Op → Op
  | .send => .recv
  | .recv => .send

structure ChanProg where
  main : CStmt
  /-- The code of task `t`. -/
  task : Nat → CStmt
  /-- The task ids are below `count`. -/
  count : Nat
  /-- The code uses the places below `places`. The channels come next, then the flags of the
  tasks. -/
  places : Nat
  /-- The channels are `0` to `chans - 1`. -/
  chans : Nat
  /-- The room of each channel. -/
  cap : Nat → Nat
  /-- Each channel carries at most `msgs` values in a run. -/
  msgs : Nat
  /-- The register that holds the number of a `send` or a `recv`. -/
  tmp : Nat

namespace ChanProg

variable (P : ChanProg)

/-- The number of places of one channel. -/
def size : Nat := 2 + 4 * P.msgs

/-- The first place of the channel `c`. -/
def chan (c : Nat) : Nat := P.places + c * P.size

/-- The counter that numbers the operations `o` of the channel `c`. -/
def ctr : Op → Nat → Nat
  | .send, c => P.chan c
  | .recv, c => P.chan c + 1

/-- The values: element `k` holds the value of the `send` with the number `k`. -/
def data (c : Nat) : Nat := P.chan c + 2 + P.msgs

/-- The flags that the operation `o` sets: `full` for a `send`, `taken` for a `recv`. -/
def sig : Op → Nat → Nat
  | .send, c => P.chan c + 2
  | .recv, c => P.chan c + 2 + 2 * P.msgs

/-- The number of flags of each kind. -/
def sigLen : Op → Nat
  | .send => P.msgs
  | .recv => 2 * P.msgs

/-- The operation with the number `k` sets the flag `k + off`. -/
def off : Op → Nat → Nat
  | .send, _ => 0
  | .recv, c => P.cap c

/-- The last step of an operation: set its flag. -/
abbrev fin (o : Op) (c : Nat) : Stmt :=
  .storeAt (P.sig o c) (P.sigLen o) (.add (.reg P.tmp) (.const (P.off o c))) (.const 1) .sc

/-- An operation after it took its number: wait for the flag of the other side, then move the
value with `d`, then set the flag. -/
abbrev rest (o : Op) (c : Nat) (d : Stmt) : Stmt :=
  .seq (.waitAt (P.sig o.other c) (P.sigLen o.other) (.reg P.tmp)) (.seq d (P.fin o c))

abbrev opCode (o : Op) (c : Nat) (d : Stmt) : Stmt :=
  .seq (.fetchAdd P.tmp (P.ctr o c) (.const 1) .sc) (P.rest o c d)

/-- Replace `send` and `recv` with their code. -/
def lowerC : CStmt → Stmt
  | .base s => s
  | .send c e => P.opCode .send c (.storeAt (P.data c) P.msgs (.reg P.tmp) e .plain)
  | .recv r c => P.opCode .recv c (.loadAt r (P.data c) P.msgs (.reg P.tmp) .plain)
  | .seq s t => .seq (lowerC s) (lowerC t)
  | .ite e s t => .ite e (lowerC s) (lowerC t)
  | .loop e s => .loop e (lowerC s)

/-- `sc` writes of constants, in order. -/
def stores : List (Nat × Int) → Stmt
  | [] => .skip
  | (x, v) :: ps => .seq (.store x (.const v) .sc) (stores ps)

/-- The counters: the `send` counter of each channel, and then its `recv` counter. -/
def ctrPlace (i : Nat) : Nat := P.chan (i / 2) + i % 2

/-- The writes at the start of thread 0: each counter, and the first `cap c` flags `taken` of each
channel `c`. -/
def initList : List (Nat × Int) :=
  (List.range (2 * P.chans)).map (fun i => (P.ctrPlace i, 0)) ++
    (List.range P.chans).flatMap fun c => (List.range (P.cap c)).map fun j => (P.sig .recv c + j, 1)

def toTaskProg : TaskProg where
  main := .seq (stores P.initList) (P.lowerC P.main)
  task t := P.lowerC (P.task t)
  count := P.count
  flags := P.places + P.chans * P.size

/-- The rules for the code of the program: it uses the places below `places`, its channels exist,
and a `recv` does not write the register `tmp`. -/
def StmtOk : CStmt → Prop
  | .base s => s.Below P.places P.count
  | .send c _ => c < P.chans
  | .recv r c => c < P.chans ∧ r ≠ P.tmp
  | .seq s t | .ite _ s t => StmtOk s ∧ StmtOk t
  | .loop _ s => StmtOk s

structure Ok : Prop where
  main : P.StmtOk P.main
  task : ∀ t, P.StmtOk (P.task t)
  cap : ∀ c, c < P.chans → P.cap c ≤ P.msgs

/-! ## The places of the channels -/

variable {P}

theorem size_eq : P.size = 2 + 4 * P.msgs := rfl

theorem chan_le {c c' : Nat} (h : c < c') : P.chan c + P.size ≤ P.chan c' := by
  unfold chan
  have := Nat.mul_le_mul_right P.size (Nat.succ_le_of_lt h)
  rw [Nat.succ_mul] at this
  omega

theorem chan_end {c : Nat} (h : c < P.chans) : P.chan c + P.size ≤ P.places + P.chans * P.size := by
  unfold chan
  have := Nat.mul_le_mul_right P.size (Nat.succ_le_of_lt h)
  rw [Nat.succ_mul] at this
  omega

theorem places_le_chan {c : Nat} : P.places ≤ P.chan c := Nat.le_add_right _ _

/-- A place of the channels belongs to one channel. -/
theorem chan_eq {c c' a a' : Nat} (h : P.chan c + a = P.chan c' + a') (ha : a < P.size)
    (ha' : a' < P.size) : c = c' ∧ a = a' := by
  rcases Nat.lt_trichotomy c c' with hc | rfl | hc
  · have := chan_le (P := P) hc
    omega
  · omega
  · have := chan_le (P := P) hc
    omega

variable (P) in
/-- Where the flags of the operation `o` start, in the places of its channel. -/
def sigOff : Op → Nat
  | .send => 2
  | .recv => 2 + 2 * P.msgs

theorem sig_eq (o : Op) (c j : Nat) : P.sig o c + j = P.chan c + (P.sigOff o + j) := by
  cases o <;> simp only [sig, sigOff] <;> omega

theorem sig_in {o : Op} {j : Nat} (hj : j < P.sigLen o) : P.sigOff o + j < P.size := by
  cases o <;> simp only [sigOff, sigLen] at * <;> rw [size_eq] <;> omega

theorem sig_inj {o o' : Op} {c c' j j' : Nat} (hj : j < P.sigLen o) (hj' : j' < P.sigLen o')
    (h : P.sig o c + j = P.sig o' c' + j') : o = o' ∧ c = c' ∧ j = j' := by
  rw [sig_eq, sig_eq] at h
  obtain ⟨rfl, h⟩ := chan_eq h (sig_in hj) (sig_in hj')
  cases o <;> cases o' <;> simp only [sigOff, sigLen] at h hj hj' <;>
    first | exact ⟨rfl, rfl, by omega⟩ | omega

theorem ctr_eq (o : Op) (c : Nat) :
    P.ctr o c = P.chan c + (match o with | .send => 0 | .recv => 1) := by
  cases o <;> rfl

theorem ctr_inj {o o' : Op} {c c' : Nat} (h : P.ctr o c = P.ctr o' c') : o = o' ∧ c = c' := by
  have hs : 1 < P.size := by rw [size_eq]; omega
  cases o <;> cases o' <;> simp only [ctr] at h
  · exact ⟨rfl, (chan_eq (c := c) (c' := c') (a := 0) (a' := 0) (by omega) (by omega)
      (by omega)).1⟩
  · have := chan_eq (c := c) (c' := c') (a := 0) (a' := 1) (by omega) (by omega) (by omega)
    omega
  · have := chan_eq (c := c) (c' := c') (a := 1) (a' := 0) (by omega) (by omega) (by omega)
    omega
  · exact ⟨rfl, (chan_eq h (by omega) (by omega)).1⟩

theorem ctr_ne_sig {o o' : Op} {c c' j : Nat} (hj : j < P.sigLen o') :
    P.ctr o c ≠ P.sig o' c' + j := by
  intro h
  rw [sig_eq] at h
  have hs : 1 < P.size := by rw [size_eq]; omega
  have hin := sig_in hj
  cases o <;> simp only [ctr] at h
  · have := chan_eq (c := c) (c' := c') (a := 0) (by omega) (by omega) hin
    cases o' <;> simp only [sigOff] at this <;> omega
  · have := chan_eq (a := 1) h (by omega) hin
    cases o' <;> simp only [sigOff] at this <;> omega

theorem data_ne_sig {o : Op} {c c' j j' : Nat} (hj : j < P.msgs) (hj' : j' < P.sigLen o) :
    P.data c + j ≠ P.sig o c' + j' := by
  intro h
  rw [sig_eq] at h
  have hin := sig_in hj'
  have := chan_eq (c := c) (c' := c') (a := 2 + P.msgs + j) (by simp only [data] at h; omega)
    (by rw [size_eq]; omega) hin
  cases o <;> simp only [sigOff, sigLen] at this hj' <;> omega

theorem data_ne_ctr {o : Op} {c c' j : Nat} (hj : j < P.msgs) : P.data c + j ≠ P.ctr o c' := by
  intro h
  have hs : 1 < P.size := by rw [size_eq]; omega
  cases o <;> simp only [ctr] at h
  · have := chan_eq (P := P) (c := c) (c' := c') (a := 2 + P.msgs + j) (a' := 0)
      (by simp only [data] at h; omega)
      (by rw [size_eq]; omega) (by omega)
    omega
  · have := chan_eq (P := P) (c := c) (c' := c') (a := 2 + P.msgs + j) (a' := 1)
      (by simp only [data] at h; omega)
      (by rw [size_eq]; omega) (by omega)
    omega

theorem sig_inside {o : Op} {c j : Nat} (hc : c < P.chans) (hj : j < P.sigLen o) :
    P.places ≤ P.sig o c + j ∧ P.sig o c + j < P.places + P.chans * P.size := by
  have := chan_end hc
  have := places_le_chan (P := P) (c := c)
  have := sig_in hj
  rw [sig_eq]
  omega

theorem ctr_inside {o : Op} {c : Nat} (hc : c < P.chans) :
    P.places ≤ P.ctr o c ∧ P.ctr o c < P.places + P.chans * P.size := by
  have := chan_end hc
  have := places_le_chan (P := P) (c := c)
  have hs : 1 < P.size := by rw [size_eq]; omega
  cases o <;> simp only [ctr] <;> omega

theorem data_inside {c j : Nat} (hc : c < P.chans) (hj : j < P.msgs) :
    P.places ≤ P.data c + j ∧ P.data c + j < P.places + P.chans * P.size := by
  have := chan_end hc
  have := places_le_chan (P := P) (c := c)
  simp only [data]
  rw [size_eq] at *
  omega

/-! ## The states of lowered code -/

variable (P)

/-- An access that the code of the program can make: a fence, or an access outside the
channels. -/
def Avoid (k : Kind) (_ : Order) (x : Nat) : Prop :=
  k = .fence ∨ x < P.places ∨ P.places + P.chans * P.size ≤ x

/-- Where a thread is in an operation of a channel. -/
inductive Phase where
  | idle
  /-- It read the counter of the operation `o` of the channel `c`, and got the number `k`. -/
  | took (o : Op) (c : Nat) (k : Int)
  /-- It wrote the counter, and did not read the flag that it waits for yet. -/
  | counted (o : Op) (c : Nat) (k : Int)
  /-- It read the flag that it waits for at least once. -/
  | waited (o : Op) (c : Nat) (k : Int)

/-- The accesses that lowered code can make in each phase. -/
inductive Trans : Phase → Option Label → Phase → Prop
  | idle : Trans .idle none .idle
  | counted_silent {o c k} : Trans (.counted o c k) none (.counted o c k)
  | waited_silent {o c k} : Trans (.waited o c k) none (.waited o c k)
  | other {l : Label} : P.Avoid l.kind l.order l.loc → Trans .idle (some l) .idle
  | take {o c k} : c < P.chans → Trans .idle (some ⟨.read, .sc, P.ctr o c, k, false⟩) (.took o c k)
  | count {o c k} :
    Trans (.took o c k) (some ⟨.write, .sc, P.ctr o c, k + 1, true⟩) (.counted o c k)
  | wait {o c k v ph} {j : Nat} : ph = .counted o c k ∨ ph = .waited o c k → (j : Int) = k →
    j < P.sigLen o.other →
    Trans ph (some ⟨.read, .sc, P.sig o.other c + j, v, false⟩) (.waited o c k)
  | move {o c k kd v} {j : Nat} : (j : Int) = k → j < P.msgs →
    Trans (.waited o c k) (some ⟨kd, .plain, P.data c + j, v, false⟩) (.waited o c k)
  | set {o c k} {j : Nat} : (j : Int) = k + P.off o c → j < P.sigLen o →
    Trans (.waited o c k) (some ⟨.write, .sc, P.sig o c + j, 1, false⟩) .idle

/-- The step that moves the value: a `send` writes it, a `recv` reads it. -/
inductive Mid : Op → Nat → Stmt → Prop
  | send {c e} : Mid .send c (.storeAt (P.data c) P.msgs (.reg P.tmp) e .plain)
  | recv {c r} : r ≠ P.tmp → Mid .recv c (.loadAt r (P.data c) P.msgs (.reg P.tmp) .plain)

/-- Lowered code that is not in the middle of an operation. -/
inductive Clean : Stmt → Prop
  | other {s} : s.Accesses P.Avoid → Clean s
  | op {o c d} : c < P.chans → P.Mid o c d → Clean (P.opCode o c d)
  | seq {s t} : Clean s → Clean t → Clean (.seq s t)
  | ite {e s t} : Clean s → Clean t → Clean (.ite e s t)
  | loop {e s} : Clean s → Clean (.loop e s)

/-- The states of lowered code, with their phase. In an operation, the register `tmp` holds its
number. -/
inductive Good (regs : Nat → Int) : Phase → Stmt → Prop
  | clean {s} : P.Clean s → Good regs .idle s
  | seq {ph s t} : Good regs ph s → P.Clean t → Good regs ph (.seq s t)
  | took {o c d k} : P.Mid o c d → regs P.tmp = k →
    Good regs (.took o c k) (.seq (.rmwStore (P.ctr o c) (k + 1) .sc) (P.rest o c d))
  | counted {o c d k} : P.Mid o c d → regs P.tmp = k →
    Good regs (.counted o c k) (.seq .skip (P.rest o c d))
  | waiting {o c d k ph} : ph = .counted o c k ∨ ph = .waited o c k → P.Mid o c d →
    regs P.tmp = k → Good regs ph (P.rest o c d)
  | waited {o c d k} : P.Mid o c d → regs P.tmp = k →
    Good regs (.waited o c k) (.seq .skip (.seq d (P.fin o c)))
  | moving {o c d k} : P.Mid o c d → regs P.tmp = k →
    Good regs (.waited o c k) (.seq d (P.fin o c))
  | moved {o c k} : regs P.tmp = k → Good regs (.waited o c k) (.seq .skip (P.fin o c))
  | setting {o c k} : regs P.tmp = k → Good regs (.waited o c k) (P.fin o c)

variable {P}

theorem other_step {s regs a s' regs'} (h : Step s regs a s' regs') (hs : s.Accesses P.Avoid) :
    P.Good regs' .idle s' ∧ P.Trans .idle a .idle := by
  obtain ⟨hs', ha⟩ := (accessRule P.Avoid).step h hs
  refine ⟨.clean (.other hs'), ?_⟩
  cases a with
  | none => exact .idle
  | some l => exact .other (ha l rfl)

theorem Clean.step {s} (hcl : P.Clean s) :
    ∀ {regs a s' regs'}, Step s regs a s' regs' → ∃ ph, P.Good regs' ph s' ∧ P.Trans .idle a ph := by
  induction hcl with
  | other hs => exact fun h => ⟨_, other_step h hs⟩
  | op hc hm =>
    intro regs a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | fetchAdd v => exact ⟨_, .took hm (by simp [update]), .take hc⟩
  | seq _ ht ih =>
    intro regs a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .clean ht, .idle⟩
    | seq h =>
      obtain ⟨ph, hg, htr⟩ := ih h
      exact ⟨ph, .seq hg ht, htr⟩
  | ite hs ht =>
    intro regs a s' regs' h
    cases h with
    | ite_true => exact ⟨_, .clean hs, .idle⟩
    | ite_false => exact ⟨_, .clean ht, .idle⟩
  | loop hs =>
    intro regs a s' regs' h
    cases h with
    | loop => exact ⟨_, .clean (.ite (.seq hs (.loop hs)) (.other trivial)), .idle⟩

/-- Each step of lowered code keeps `Good`, and its access follows `Trans`. -/
theorem Good.step {ph s regs} (hg : P.Good regs ph s) :
    ∀ {a s' regs'}, Step s regs a s' regs' → ∃ ph', P.Good regs' ph' s' ∧ P.Trans ph a ph' := by
  induction hg with
  | clean hcl => exact hcl.step
  | seq hg ht ih =>
    intro a s' regs' h
    cases h with
    | seq_skip =>
      cases hg with
      | clean => exact ⟨_, .clean ht, .idle⟩
    | seq h =>
      obtain ⟨ph', hg', htr⟩ := ih h
      exact ⟨ph', .seq hg' ht, htr⟩
  | took hm hk =>
    intro a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | rmwStore => exact ⟨_, .counted hm hk, .count⟩
  | counted hm hk =>
    intro a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .waiting (Or.inl rfl) hm hk, .counted_silent⟩
    | seq h => cases h
  | waiting hph hm hk =>
    intro a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | waitAt j v hi hj =>
        refine ⟨_, ?_, .wait hph (hi.symm.trans hk) hj⟩
        by_cases hv : v = 0
        · simp only [hv, ↓reduceIte]
          exact .waiting (Or.inr rfl) hm hk
        · simp only [hv, ↓reduceIte]
          exact .waited hm hk
  | waited hm hk =>
    intro a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .moving hm hk, .waited_silent⟩
    | seq h => cases h
  | moving hm hk =>
    intro a s' regs' h
    cases h with
    | seq_skip => cases hm
    | seq h =>
      cases hm with
      | send =>
        cases h with
        | storeAt j hi hj => exact ⟨_, .moved hk, .move (hi.symm.trans hk) hj⟩
      | recv hr =>
        cases h with
        | loadAt j v hi hj =>
          refine ⟨_, .moved ?_, .move (hi.symm.trans hk) hj⟩
          simp [update, Ne.symm hr, hk]
  | moved hk =>
    intro a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .setting hk, .waited_silent⟩
    | seq h => cases h
  | setting hk =>
    intro a s' regs' h
    cases h with
    | storeAt j hi hj =>
      refine ⟨_, .clean (.other trivial), .set ?_ hj⟩
      simp only [Expr.eval] at hi
      omega

variable (P) in
/-- The lists of accesses that `Trans` allows from a phase. -/
inductive Accepts : Phase → List Label → Prop
  | nil {ph} : Accepts ph []
  | cons {ph ph' a L} : P.Trans ph (some a) ph' → Accepts ph' L → Accepts ph (a :: L)

theorem Trans.none_eq {ph ph'} (h : P.Trans ph none ph') : ph' = ph := by
  cases h <;> rfl

/-- A run of lowered code makes a list of accesses that `Trans` allows. -/
theorem Runs.accepts {s regs L} (h : Runs s regs L) : ∀ {ph}, P.Good regs ph s → P.Accepts ph L := by
  induction h with
  | stop => exact fun _ => .nil
  | silent hs _ ih =>
    intro ph hg
    obtain ⟨ph', hg', htr⟩ := hg.step hs
    cases htr.none_eq
    exact ih hg'
  | visible hs _ ih =>
    intro ph hg
    obtain ⟨ph', hg', htr⟩ := hg.step hs
    exact .cons htr (ih hg')

/-- An access of the code of the program is outside the channels. -/
theorem Avoid.not_inside {k o x} (h : P.Avoid k o x) (hk : k ≠ .fence) (h1 : P.places ≤ x)
    (h2 : x < P.places + P.chans * P.size) : False := by
  rcases h with h | h | h
  · exact hk h
  · omega
  · omega

/-- Each write of a counter is the write of a `fetchAdd`. -/
theorem Accepts.ctr_rmw {ph L} (h : P.Accepts ph L) {o c} (hc : c < P.chans) :
    ∀ {m : Nat} {l : Label}, L[m]? = some l → l.kind = .write → l.loc = P.ctr o c → l.rmw = true := by
  induction h with
  | nil => intro m l h; simp at h
  | cons htr _ ih =>
    intro m l hl hk hx
    cases m with
    | succ m => exact ih (by simpa using hl) hk hx
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hav => exact (hav.not_inside (by rw [hk]; nofun) (hx ▸ (ctr_inside hc).1)
          (hx ▸ (ctr_inside hc).2)).elim
      | take | wait => cases hk
      | count => rfl
      | move _ hj => exact (data_ne_ctr hj hx).elim
      | set _ hj => exact (ctr_ne_sig hj hx.symm).elim

/-- The read of a counter comes right before the write of its `fetchAdd`. -/
theorem Accepts.take_count {ph L} (h : P.Accepts ph L) {o c} (hc : c < P.chans) :
    ∀ {m : Nat} {l l' : Label}, L[m]? = some l → L[m + 1]? = some l' → l.kind = .read → l.loc = P.ctr o c →
      l' = ⟨.write, .sc, P.ctr o c, l.val + 1, true⟩ := by
  induction h with
  | nil => intro m l l' h; simp at h
  | cons htr hacc ih =>
    intro m l l' hl hl' hk hx
    cases m with
    | succ m => exact ih (by simpa using hl) (by simpa using hl') hk hx
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      simp only [List.getElem?_cons_succ] at hl'
      subst hl
      cases htr with
      | other hav => exact (hav.not_inside (by rw [hk]; nofun) (hx ▸ (ctr_inside hc).1)
          (hx ▸ (ctr_inside hc).2)).elim
      | take =>
        obtain ⟨rfl, rfl⟩ := ctr_inj hx
        cases hacc with
        | nil => simp at hl'
        | cons htr' _ =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at hl'
          subst hl'
          cases htr' with
          | count => rfl
          | wait hph => rcases hph with h | h <;> cases h
      | count | set => cases hk
      | wait _ _ hj => exact (ctr_ne_sig hj hx.symm).elim
      | move _ hj => exact (data_ne_ctr hj hx).elim

/-- The write of a `fetchAdd` of a counter comes right after the read of the counter. -/
theorem Accepts.count_take {ph L} (h : P.Accepts ph L) {o c} (hc : c < P.chans) :
    ∀ {m : Nat} {l : Label}, L[m]? = some l → l.kind = .write → l.rmw = true →
      l.loc = P.ctr o c → (m = 0 ∧ ph = .took o c (l.val - 1)) ∨
        ∃ m', m = m' + 1 ∧ L[m']? = some ⟨.read, .sc, P.ctr o c, l.val - 1, false⟩ := by
  induction h with
  | nil => intro m l h; simp at h
  | @cons ph ph' a L htr _ ih =>
    intro m l hl hw hr hx
    cases m with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hav => exact (hav.not_inside (by rw [hw]; nofun)
          (hx ▸ (ctr_inside hc).1) (hx ▸ (ctr_inside hc).2)).elim
      | take | wait | move | set => cases hr
      | count =>
        obtain ⟨rfl, rfl⟩ := ctr_inj hx
        exact Or.inl ⟨rfl, by simp⟩
    | succ m =>
      rcases ih (by simpa using hl) hw hr hx with ⟨rfl, rfl⟩ | ⟨m', rfl, hm'⟩
      · cases htr
        exact Or.inr ⟨0, rfl, by simp⟩
      · exact Or.inr ⟨m' + 1, rfl, by simpa using hm'⟩

/-- Each read and write of a flag is `sc`. -/
theorem Accepts.sig_sc {ph L} (h : P.Accepts ph L) {o c j} (hc : c < P.chans)
    (hj : j < P.sigLen o) :
    ∀ {m : Nat} {l : Label}, L[m]? = some l → l.kind ≠ .fence → l.loc = P.sig o c + j → l.order = .sc := by
  induction h with
  | nil => intro m l h; simp at h
  | cons htr _ ih =>
    intro m l hl hk hx
    cases m with
    | succ m => exact ih (by simpa using hl) hk hx
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hav => exact (hav.not_inside hk (hx ▸ (sig_inside hc hj).1)
          (hx ▸ (sig_inside hc hj).2)).elim
      | take | count | wait | set => rfl
      | move _ hj' => exact (data_ne_sig hj' hj hx).elim

variable (P) in
/-- No access from `i0` to before `i1` sets a flag of the operation `o` of the channel `c`. -/
def NoSet (o : Op) (c : Nat) (L : List Label) (i0 i1 : Nat) : Prop :=
  ∀ m l, i0 ≤ m → m < i1 → L[m]? = some l →
    ¬ (l.kind = .write ∧ ∃ j < P.sigLen o, l.loc = P.sig o c + j)

variable (P) in
/-- An access from `i0` to before `i1` reads the flag that the operation number `t` waits for. -/
def Waits (o : Op) (c : Nat) (t : Int) (L : List Label) (i0 i1 : Nat) : Prop :=
  ∃ (m j : Nat) (v : Int), i0 ≤ m ∧ m < i1 ∧ (j : Int) = t ∧ L[m]? = some ⟨.read, .sc, P.sig o.other c + j, v, false⟩

variable (P) in
/-- What a phase says about an operation that sets its flag at the position `i`. -/
def Pre (ph : Phase) (o : Op) (c : Nat) (t : Int) (L : List Label) (i : Nat) : Prop :=
  match ph with
  | .idle => False
  | .took o' c' k | .counted o' c' k =>
    o' = o ∧ c' = c ∧ k = t ∧ P.NoSet o c L 0 i ∧ P.Waits o c t L 0 i
  | .waited o' c' k => o' = o ∧ c' = c ∧ k = t ∧ P.NoSet o c L 0 i

theorem NoSet.cons {o c L i0 i1} {a : Label} (h : P.NoSet o c L i0 i1) :
    P.NoSet o c (a :: L) (i0 + 1) (i1 + 1) := by
  intro m l h0 h1 hl
  obtain ⟨m, rfl⟩ : ∃ m', m = m' + 1 := ⟨m - 1, by omega⟩
  exact h m l (by omega) (by omega) (by simpa using hl)

theorem NoSet.cons_zero {o c L i1} {a : Label} (h : P.NoSet o c L 0 i1)
    (ha : ¬ (a.kind = .write ∧ ∃ j < P.sigLen o, a.loc = P.sig o c + j)) :
    P.NoSet o c (a :: L) 0 (i1 + 1) := by
  intro m l _ h1 hl
  cases m with
  | zero =>
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
    exact hl ▸ ha
  | succ m => exact h m l (by omega) (by omega) (by simpa using hl)

theorem Waits.cons {o c t L i0 i1} {a : Label} (h : P.Waits o c t L i0 i1) :
    P.Waits o c t (a :: L) (i0 + 1) (i1 + 1) := by
  obtain ⟨m, j, v, h0, h1, hj, hm⟩ := h
  exact ⟨m + 1, j, v, by omega, by omega, hj, by simpa using hm⟩

theorem Waits.weaken {o c t L i0 i0' i1} (h : P.Waits o c t L i0 i1) (hle : i0' ≤ i0) :
    P.Waits o c t L i0' i1 := by
  obtain ⟨m, j, v, h0, h1, hj, hm⟩ := h
  exact ⟨m, j, v, by omega, h1, hj, hm⟩

/-- Before an operation sets its flag, it read its counter and got its number, then read the
flag that it waits for, and set no other flag of its kind. -/
theorem Accepts.set_hist {ph L} (h : P.Accepts ph L) {o c j} (hc : c < P.chans)
    (hj : j < P.sigLen o) :
    ∀ {i : Nat} {l : Label}, L[i]? = some l → l.kind = .write → l.loc = P.sig o c + j →
      P.Pre ph o c ((j : Int) - P.off o c) L i ∨
        ∃ i' < i, L[i']? = some ⟨.read, .sc, P.ctr o c, (j : Int) - P.off o c, false⟩ ∧
          P.NoSet o c L (i' + 1) i ∧ P.Waits o c ((j : Int) - P.off o c) L (i' + 1) i := by
  induction h with
  | nil => intro i l h; simp at h
  | @cons ph ph' a L htr _ ih =>
    intro i l hl hk hx
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hav => exact (hav.not_inside (by rw [hk]; nofun) (hx ▸ (sig_inside hc hj).1)
          (hx ▸ (sig_inside hc hj).2)).elim
      | take | wait => cases hk
      | count => exact (ctr_ne_sig hj hx).elim
      | move _ hj' => exact (data_ne_sig hj' hj hx).elim
      | set hk' hj' =>
        obtain ⟨rfl, rfl, rfl⟩ := sig_inj hj' hj hx
        exact Or.inl ⟨rfl, rfl, by omega, fun m _ _ h1 _ => absurd h1 (Nat.not_lt_zero m)⟩
    | succ i =>
      have notSet {b : Label} (hb : b.kind ≠ .write ∨ b.loc = P.ctr o c ∨
          ∃ j' < P.msgs, b.loc = P.data c + j') :
          ¬ (b.kind = .write ∧ ∃ j < P.sigLen o, b.loc = P.sig o c + j) := by
        rintro ⟨hw, j'', hj'', hl''⟩
        rcases hb with hb | hb | ⟨j', hj', hb⟩
        · exact hb hw
        · exact ctr_ne_sig hj'' (hb.symm.trans hl'')
        · exact data_ne_sig hj' hj'' (hb.symm.trans hl'')
      rcases ih (by simpa using hl) hk hx with hpre | ⟨i', hi', hread, hno, hw⟩
      · cases ph' with
        | idle => exact hpre.elim
        | took o' c' k =>
          obtain ⟨rfl, rfl, rfl, hno, hw⟩ := hpre
          cases htr
          exact Or.inr ⟨0, by omega, by simp, hno.cons, hw.cons⟩
        | counted o' c' k =>
          obtain ⟨rfl, rfl, rfl, hno, hw⟩ := hpre
          cases htr
          exact Or.inl ⟨rfl, rfl, rfl, hno.cons_zero (notSet (Or.inr (Or.inl rfl))),
            hw.cons.weaken (Nat.zero_le _)⟩
        | waited o' c' k =>
          obtain ⟨rfl, rfl, rfl, hno⟩ := hpre
          cases htr with
          | wait hph hj' hjl =>
            rcases hph with rfl | rfl
            · exact Or.inl ⟨rfl, rfl, rfl, hno.cons_zero (notSet (Or.inl nofun)),
                0, _, _, Nat.le_refl 0, by omega, hj', rfl⟩
            · exact Or.inl ⟨rfl, rfl, rfl, hno.cons_zero (notSet (Or.inl nofun))⟩
          | move hj' hjl =>
            exact Or.inl ⟨rfl, rfl, rfl, hno.cons_zero (notSet (Or.inr (Or.inr ⟨_, hjl, rfl⟩)))⟩
      · exact Or.inr ⟨i' + 1, by omega, by simpa using hread, hno.cons, hw.cons⟩

/-! ## The code of each thread -/

/-- The access of one write at the start of thread 0. -/
def initLabel (p : Nat × Int) : Label := ⟨.write, .sc, p.1, p.2, false⟩

theorem lower_stores (Q : TaskProg) (ps : List (Nat × Int)) : Q.lower (stores ps) = stores ps := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    obtain ⟨x, v⟩ := p
    simp only [stores, TaskProg.lower, ih]

/-- A run of `stores ps` and then `t` makes a prefix of the writes of `ps`, or all of them and then
a run of `t`. -/
theorem runs_stores {ps t regs M} (h : Runs (.seq (stores ps) t) regs M) :
    (∃ k ≤ ps.length, M = (ps.map initLabel).take k) ∨
      ∃ M', M = ps.map initLabel ++ M' ∧ Runs t regs M' := by
  induction ps generalizing M with
  | nil =>
    cases h with
    | stop => exact Or.inl ⟨0, Nat.le_refl _, rfl⟩
    | silent hs hr =>
      cases hs with
      | seq_skip => exact Or.inr ⟨M, rfl, hr⟩
      | seq hs => cases hs
    | visible hs => cases hs with | seq hs => cases hs
  | cons p ps ih =>
    obtain ⟨x, v⟩ := p
    cases h with
    | stop => exact Or.inl ⟨0, Nat.zero_le _, rfl⟩
    | silent hs => cases hs with | seq hs => cases hs with | seq hs => cases hs
    | visible hs hr =>
      cases hs with
      | seq hs =>
        cases hs with
        | seq hs =>
          cases hs
          cases hr with
          | stop => exact Or.inl ⟨1, by simp, rfl⟩
          | silent hs hr =>
            cases hs with
            | seq hs =>
              cases hs with
              | seq_skip =>
                rcases ih hr with ⟨k, hk, rfl⟩ | ⟨M', rfl, hM'⟩
                · exact Or.inl ⟨k + 1, by simp; omega, rfl⟩
                · exact Or.inr ⟨M', rfl, hM'⟩
              | seq hs => cases hs
          | visible hs => cases hs with | seq hs => cases hs with | seq hs => cases hs

theorem stores_seqCstOnly (ps : List (Nat × Int)) : (stores ps).SeqCstOnly := by
  induction ps with
  | nil => trivial
  | cons p ps ih => exact ⟨Or.inr rfl, ih⟩

/-- The code of the program stays outside the channels, also after the lowering of `spawn` and
`join`. -/
theorem lower_avoid {s : Stmt} (h : s.Below P.places P.count) :
    (P.toTaskProg.lower s).Accesses P.Avoid := by
  induction s with
  | spawnAt base =>
    refine fun j hj => Or.inr (Or.inr ?_)
    simp only [TaskProg.start, toTaskProg]
    omega
  | joinAt base =>
    refine fun j hj => Or.inr (Or.inr ?_)
    simp only [TaskProg.done, toTaskProg]
    omega
  | load | store | rmwStore | wait => exact Or.inr (Or.inl h)
  | fetchAdd | cas => exact ⟨Or.inr (Or.inl h), Or.inr (Or.inl h)⟩
  | loadAt | storeAt | waitAt =>
    exact fun j hj => Or.inr (Or.inl (by simp only [Stmt.Below] at h; omega))
  | fence => exact Or.inl rfl
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | loop _ _ ih => exact ih h
  | skip | assign => trivial

theorem clean_lower {s : CStmt} (h : P.StmtOk s) : P.Clean (P.toTaskProg.lower (P.lowerC s)) := by
  induction s with
  | base s => exact .other (lower_avoid h)
  | send c e => exact .op h .send
  | recv r c => exact .op h.1 (.recv h.2)
  | seq _ _ ih1 ih2 => exact .seq (ih1 h.1) (ih2 h.2)
  | ite _ _ _ ih1 ih2 => exact .ite (ih1 h.1) (ih2 h.2)
  | loop _ _ ih => exact .loop (ih h)

theorem code_zero :
    P.toTaskProg.toProg.code 0 = .seq (stores P.initList) (P.toTaskProg.lower (P.lowerC P.main)) := by
  show P.toTaskProg.lower (.seq _ _) = _
  simp only [TaskProg.lower, lower_stores]

theorem clean_task (hP : P.Ok) (t : Nat) : P.Clean (P.toTaskProg.toProg.code (t + 1)) := by
  rw [TaskProg.code_succ]
  unfold TaskProg.taskCode
  split
  · refine .seq (.other (Or.inr (Or.inr ?_))) (.seq (clean_lower (hP.task t)) (.other (Or.inr (Or.inr ?_))))
    · simp only [TaskProg.start, toTaskProg]
      omega
    · simp only [TaskProg.done, toTaskProg]
      omega
  · exact .other trivial

variable (P) in
/-- The number of writes at the start of the thread `t`. -/
def pre (t : Nat) : Nat := if t = 0 then P.initList.length else 0

theorem getElem?_range_map {n i : Nat} {f : Nat → Label} (hi : i < n) :
    ((List.range n).map f)[i]? = some (f i) := by
  simp [List.getElem?_map, List.getElem?_range hi]

/-- The accesses of a thread: first the writes at the start, then a list that `Trans` allows. -/
theorem trace_split (hP : P.Ok) {t n : Nat} {f : Nat → Label}
    (h : Trace (P.toTaskProg.toProg.code t) n f) :
    (∀ i < n, i < P.pre t → P.initList[i]?.map initLabel = some (f i)) ∧
      P.Accepts .idle ((List.range (n - P.pre t)).map fun m => f (P.pre t + m)) := by
  unfold Trace at h
  cases t with
  | succ t =>
    simp only [pre, Nat.add_one_ne_zero, ↓reduceIte, Nat.sub_zero, Nat.zero_add]
    exact ⟨fun _ _ h => absurd h (Nat.not_lt_zero _), Runs.accepts h (.clean (clean_task hP t))⟩
  | zero =>
    simp only [pre, ↓reduceIte]
    rw [code_zero] at h
    rcases runs_stores h with ⟨k, hk, hM⟩ | ⟨M', hM, hr⟩
    · have hn : n = k := by
        have := congrArg List.length hM
        simp only [List.length_map, List.length_range, List.length_take] at this
        omega
      subst hn
      refine ⟨fun i hi _ => ?_, by rw [Nat.sub_eq_zero_of_le hk]; exact .nil⟩
      have := congrArg (·[i]?) hM
      simp only [getElem?_range_map hi, List.getElem?_take, hi, ↓reduceIte, List.getElem?_map]
        at this
      exact this.symm
    · have hlen := congrArg List.length hM
      simp only [List.length_map, List.length_range, List.length_append] at hlen
      refine ⟨fun i hi hi' => ?_, ?_⟩
      · have := congrArg (·[i]?) hM
        simp only [getElem?_range_map hi] at this
        rw [List.getElem?_append_left (by simpa using hi'), List.getElem?_map] at this
        exact this.symm
      · have : M' = (List.range (n - P.initList.length)).map fun m => f (P.initList.length + m) := by
          apply List.ext_getElem?
          intro m
          have := congrArg (·[P.initList.length + m]?) hM
          rw [List.getElem?_append_right (by simp)] at this
          simp only [List.length_map, Nat.add_sub_cancel_left] at this
          by_cases hm : m < n - P.initList.length
          · rw [getElem?_range_map hm, ← this, getElem?_range_map (by omega)]
          · rw [List.getElem?_eq_none (show M'.length ≤ m by omega),
              List.getElem?_eq_none (by simp; omega)]
        rw [← this]
        exact Runs.accepts hr (.clean (clean_lower hP.main))

/-! ## The writes at the start -/

/-- The position of each counter in `initList`. -/
def initIdx : Op → Nat → Nat
  | .send, c => 2 * c
  | .recv, c => 2 * c + 1

theorem initList_cases {i : Nat} {p : Nat × Int} (h : P.initList[i]? = some p) :
    (i < 2 * P.chans ∧ p = (P.ctrPlace i, 0)) ∨
      ∃ c < P.chans, ∃ j < P.cap c, p = (P.sig .recv c + j, 1) := by
  unfold initList at h
  by_cases hi : i < 2 * P.chans
  · rw [List.getElem?_append_left (by simpa using hi), List.getElem?_map,
      List.getElem?_range hi] at h
    exact Or.inl ⟨hi, (Option.some.inj h).symm⟩
  · right
    rw [List.getElem?_append_right (by simp; omega)] at h
    obtain ⟨c, hc, hp⟩ := List.mem_flatMap.1 (List.mem_of_getElem? h)
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hp
    exact ⟨c, List.mem_range.1 hc, j, List.mem_range.1 hj, rfl⟩

/-- The writes at the start write each counter once. -/
theorem initList_ctr (hP : P.Ok) {i : Nat} {p : Nat × Int} (h : P.initList[i]? = some p) {o c}
    (hp : p.1 = P.ctr o c) : i = initIdx o c := by
  have hs : 1 < P.size := by rw [size_eq]; omega
  rcases initList_cases h with ⟨hi, rfl⟩ | ⟨c', hc', j, hj, rfl⟩
  · simp only [ctrPlace] at hp
    have h2 : i % 2 < 2 := Nat.mod_lt _ (by omega)
    cases o <;> simp only [ctr, initIdx] at hp ⊢
    · obtain ⟨rfl, h⟩ := chan_eq (a' := 0) hp (by omega) (by omega)
      omega
    · obtain ⟨rfl, h⟩ := chan_eq hp (by omega) (by omega)
      omega
  · have := hP.cap c' hc'
    exact (ctr_ne_sig (o' := .recv) (by simp only [sigLen]; omega) hp.symm).elim

/-- The only flags that the writes at the start set are the first `cap c` flags `taken`. -/
theorem initList_sig (hP : P.Ok) {i : Nat} {p : Nat × Int} (h : P.initList[i]? = some p) {o c j}
    (hj : j < P.sigLen o) (hp : p.1 = P.sig o c + j) : o = .recv ∧ j < P.cap c := by
  have hs : 1 < P.size := by rw [size_eq]; omega
  rcases initList_cases h with ⟨hi, rfl⟩ | ⟨c', hc', j', hj', rfl⟩
  · simp only [ctrPlace] at hp
    rw [sig_eq] at hp
    have := chan_eq hp (Nat.lt_of_lt_of_le (Nat.mod_lt _ (by omega)) (by omega)) (sig_in hj)
    cases o <;> simp only [sigOff] at this <;> omega
  · have := hP.cap c' hc'
    obtain ⟨rfl, rfl, rfl⟩ := sig_inj (o := .recv) (by simp only [sigLen]; omega) hj hp
    exact ⟨rfl, hj'⟩

theorem seqCstOnly_lowerC {s : CStmt}
    (h : ∀ b, s.Bases b → b.SeqCstOnly) : (P.lowerC s).SeqCstOnly := by
  induction s with
  | base s => exact h s rfl
  | send | recv => exact ⟨Or.inr rfl, trivial, Or.inl rfl, Or.inr rfl⟩
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 =>
    exact ⟨ih1 fun b hb => h b (Or.inl hb), ih2 fun b hb => h b (Or.inr hb)⟩
  | loop _ _ ih => exact ih h

/-- DRF-SC for programs with tasks and channels. -/
theorem drf_sc (P : ChanProg) (hm : ∀ b, P.main.Bases b → b.SeqCstOnly)
    (ht : ∀ t b, (P.task t).Bases b → b.SeqCstOnly)
    (hP : ∀ Y, P.toTaskProg.toProg.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.RaceFree)
    {X : Execution} (hX : P.toTaskProg.toProg.Execs X) (wf : X.WellFormed) (hfin : X.Finite)
    (hc : X.Consistent) : X.SeqConsistent :=
  P.toTaskProg.drf_sc ⟨stores_seqCstOnly _, seqCstOnly_lowerC hm⟩
    (fun t => seqCstOnly_lowerC (ht t)) hP hX wf hfin hc

/-! ## The executions of a program with channels -/

variable {X : Execution} (run : P.toTaskProg.toProg.Run X)
include run

theorem lab_of {d : X.E} {t i : Nat} (ht : X.thread d = t) (hi : run.idx d = i) {l : Label}
    (h : run.lab t i = l) :
    X.kind d = l.kind ∧ X.order d = l.order ∧ X.loc d = l.loc ∧ run.val d = l.val ∧
      run.isRmw d = l.rmw := by
  have := run.lab_eq d
  rw [ht, hi, h] at this
  subst this
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- The accesses of the thread of `e`, up to `e`. -/
theorem view (hP : P.Ok) (e : X.E) :
    (∀ i ≤ run.idx e, i < P.pre (X.thread e) →
      P.initList[i]?.map initLabel = some (run.lab (X.thread e) i)) ∧
      P.Accepts .idle ((List.range (run.idx e + 1 - P.pre (X.thread e))).map
        fun m => run.lab (X.thread e) (P.pre (X.thread e) + m)) := by
  obtain ⟨h1, h2⟩ := trace_split hP (run.trace e)
  exact ⟨fun i hi => h1 i (by omega), h2⟩

omit run in
theorem pre_pos {t : Nat} (h : 0 < P.pre t) : t = 0 := by
  unfold pre at h
  split at h
  · assumption
  · omega

/-- The access of an event in the list of `view`. -/
theorem view_get {e d : X.E} (ht : X.thread d = X.thread e) (hd : run.idx d ≤ run.idx e)
    (hp : P.pre (X.thread e) ≤ run.idx d) :
    ((List.range (run.idx e + 1 - P.pre (X.thread e))).map
        fun m => run.lab (X.thread e) (P.pre (X.thread e) + m))[run.idx d - P.pre (X.thread e)]? =
      some ⟨X.kind d, X.order d, X.loc d, run.val d, run.isRmw d⟩ := by
  rw [getElem?_range_map (by omega), Nat.add_sub_cancel' hp, ← ht, run.lab_eq d]

/-- An event of the prefix is one of the writes at the start. -/
theorem init_of (hP : P.Ok) {d : X.E} (hp : run.idx d < P.pre (X.thread d)) :
    X.thread d = 0 ∧ ∃ p, P.initList[run.idx d]? = some p ∧ X.kind d = .write ∧
      X.order d = .sc ∧ X.loc d = p.1 ∧ run.isRmw d = false := by
  refine ⟨pre_pos (P := P) (t := X.thread d) (by omega), ?_⟩
  have := (view run hP d).1 _ (Nat.le_refl _) hp
  rw [run.lab_eq d] at this
  cases hi : P.initList[run.idx d]? with
  | none => rw [hi] at this; cases this
  | some p =>
    rw [hi] at this
    simp only [Option.map_some, Option.some.injEq, initLabel, Label.mk.injEq] at this
    obtain ⟨h1, h2, h3, -, h5⟩ := this
    exact ⟨p, rfl, h1.symm, h2.symm, h3.symm, h5.symm⟩

/-- Each counter of a channel is a counter in the sense of `Counter.lean`. -/
theorem counter (hP : P.Ok) (wf : X.WellFormed) {o c} (hc : c < P.chans) :
    X.Counter run.val (P.ctr o c) := by
  have hadd : ∀ r w, X.rmw r w → X.loc w = P.ctr o c →
      X.loc r = P.ctr o c ∧ run.val w = run.val r + 1 := by
    intro r w hrw hwl
    obtain ⟨ht, hidx, hrm⟩ := (run.rmw_iff r w).1 hrw
    have hww := (wf.rmw_atomic r w hrw).2.1
    by_cases hp : run.idx w < P.pre (X.thread w)
    · obtain ⟨-, -, -, -, -, -, h⟩ := init_of run hP hp
      rw [hrm] at h
      cases h
    · have hacc := (view run hP w).2
      rcases hacc.count_take hc (view_get run rfl (Nat.le_refl _) (by omega)) hww hrm hwl with
        ⟨h0, h⟩ | ⟨m', hm, hm'⟩
      · cases h
      · have := view_get run (e := w) ht (by omega) (by omega)
        rw [show run.idx r - P.pre (X.thread w) = m' by omega, hm'] at this
        simp only [Option.some.injEq, Label.mk.injEq] at this
        obtain ⟨-, -, h3, h4, -⟩ := this
        exact ⟨h3.symm, by omega⟩
  have hfirst : ∀ a, X.IsWrite a → X.loc a = P.ctr o c → (∀ r, ¬ X.rmw r a) →
      X.thread a = 0 ∧ run.idx a = initIdx o c := by
    intro a haw hal hna
    by_cases hp : run.idx a < P.pre (X.thread a)
    · obtain ⟨ht, p, hi, -, -, hl, -⟩ := init_of run hP hp
      exact ⟨ht, initList_ctr hP hi (hl.symm.trans hal)⟩
    · have hacc := (view run hP a).2
      have hget := view_get run (e := a) rfl (Nat.le_refl _) (by omega)
      have hrm := hacc.ctr_rmw hc hget haw hal
      rcases hacc.count_take hc hget haw hrm hal with ⟨h0, h⟩ | ⟨m', hm, -⟩
      · cases h
      · obtain ⟨r, hr, hri⟩ := run.idx_down a (run.idx a - 1) (by omega)
        exact (hna r ((run.rmw_iff r a).2 ⟨hr, by omega, hrm⟩)).elim
  exact {
    init_unique := fun a b haw hbw hal hbl hna hnb => by
      obtain ⟨hta, hia⟩ := hfirst a haw hal hna
      obtain ⟨htb, hib⟩ := hfirst b hbw hbl hnb
      exact run.idx_inj a b (hta.trans htb.symm) (hia.trans hib.symm)
    add := hadd
    rf_val := run.rf_val
    rf_source := run.rf_source
    rmw_unique := fun r1 r2 w h1 h2 => by
      obtain ⟨ht1, hi1, -⟩ := (run.rmw_iff r1 w).1 h1
      obtain ⟨ht2, hi2, -⟩ := (run.rmw_iff r2 w).1 h2
      exact run.idx_inj r1 r2 (ht1.trans ht2.symm) (by omega) }

/-- Each read and write of a flag is `sc`. -/
theorem sig_sc (hP : P.Ok) {o c j} (hc : c < P.chans) (hj : j < P.sigLen o) {e : X.E}
    (hk : X.kind e ≠ .fence) (hl : X.loc e = P.sig o c + j) : X.IsSeqCst e := by
  by_cases hp : run.idx e < P.pre (X.thread e)
  · obtain ⟨-, -, -, -, h, -⟩ := init_of run hP hp
    exact h
  · exact (view run hP e).2.sig_sc hc hj (view_get run rfl (Nat.le_refl _) (by omega)) hk hl

/-- The write of a flag comes from the operation whose number the flag names. In the thread of
the write, a `fetchAdd` of the counter gave that number before it. Between them, the thread read
the flag that the operation waits for, and set no other flag of its kind. -/
theorem set_writer (hP : P.Ok) {o c j} (hc : c < P.chans) (hj : j < P.sigLen o)
    (hoff : P.off o c ≤ j) {w : X.E} (hw : X.IsWrite w) (hwl : X.loc w = P.sig o c + j) :
    ∃ r r', X.rmw r r' ∧ X.loc r' = P.ctr o c ∧ run.val r = (j : Int) - P.off o c ∧
      X.thread r = X.thread w ∧ run.idx r < run.idx w ∧
      (∀ d, X.thread d = X.thread w → run.idx r < run.idx d → run.idx d < run.idx w →
        ¬ (X.IsWrite d ∧ ∃ j' < P.sigLen o, X.loc d = P.sig o c + j')) ∧
      ∃ e, X.thread e = X.thread w ∧ run.idx r < run.idx e ∧ run.idx e < run.idx w ∧
        X.IsRead e ∧ X.IsSeqCst e ∧ X.loc e = P.sig o.other c + (j - P.off o c) := by
  have hp : P.pre (X.thread w) ≤ run.idx w := by
    refine Nat.le_of_not_lt fun hp => ?_
    obtain ⟨-, p, hi, -, -, hl, -⟩ := init_of run hP hp
    obtain ⟨rfl, hj'⟩ := initList_sig hP hi hj (hl.symm.trans hwl)
    simp only [off] at hoff
    omega
  have hacc := (view run hP w).2
  have hget := view_get run (e := w) rfl (Nat.le_refl _) hp
  rcases hacc.set_hist hc hj hget hw hwl with h | ⟨i', hi', hread, hno, hwt⟩
  · exact h.elim
  obtain ⟨mw, jw, v, h0, h1, hjw, hmw⟩ := hwt
  obtain ⟨r, hrt, hri⟩ := run.idx_down w (P.pre (X.thread w) + i') (by omega)
  obtain ⟨r', hrt', hri'⟩ := run.idx_down w (P.pre (X.thread w) + i' + 1) (by omega)
  obtain ⟨e, het, hei⟩ := run.idx_down w (P.pre (X.thread w) + mw) (by omega)
  have hr := view_get run (e := w) hrt (by omega) (by omega)
  rw [show run.idx r - P.pre (X.thread w) = i' by omega, hread] at hr
  have hr' := view_get run (e := w) hrt' (by omega) (by omega)
  rw [show run.idx r' - P.pre (X.thread w) = i' + 1 by omega] at hr'
  have hcount := hacc.take_count hc hread hr' rfl rfl
  have he := view_get run (e := w) het (by omega) (by omega)
  rw [show run.idx e - P.pre (X.thread w) = mw by omega, hmw] at he
  simp only [Option.some.injEq, Label.mk.injEq] at hr hcount he
  obtain ⟨-, -, hrl, hrv, -⟩ := hr
  obtain ⟨-, -, hrl', -, hrm'⟩ := hcount
  obtain ⟨hek, heo, hel, -, -⟩ := he
  refine ⟨r, r', (run.rmw_iff r r').2 ⟨hrt.trans hrt'.symm, by omega, hrm'⟩, hrl',
    hrv.symm, hrt, by omega, ?_, e, het, by omega, by omega, hek.symm, heo.symm, ?_⟩
  · intro d hdt hd1 hd2 ⟨hdw, j', hj', hdl⟩
    exact hno (run.idx d - P.pre (X.thread w)) _ (by omega) (by omega)
      (view_get run hdt (by omega) (by omega)) ⟨hdw, j', hj', hdl⟩
  · rw [← hel]
    congr 1
    simp only [off] at *
    cases o <;> simp only at hoff hjw ⊢ <;> omega

/-- Each flag has one writer. -/
theorem set_unique (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {o c j} (hc : c < P.chans) (hj : j < P.sigLen o) (hoff : P.off o c ≤ j)
    {w1 w2 : X.E} (hw1 : X.IsWrite w1) (hl1 : X.loc w1 = P.sig o c + j) (hw2 : X.IsWrite w2)
    (hl2 : X.loc w2 = P.sig o c + j) : w1 = w2 := by
  obtain ⟨r1, r1', hrw1, hrl1, hv1, ht1, hi1, hno1, -⟩ := set_writer run hP hc hj hoff hw1 hl1
  obtain ⟨r2, r2', hrw2, hrl2, hv2, ht2, hi2, hno2, -⟩ := set_writer run hP hc hj hoff hw2 hl2
  have := (counter run hP wf hc).tickets wf hcons ho hfin hrw1 hrl1 hrw2 hrl2 (hv1.trans hv2.symm)
  subst this
  rcases Nat.lt_trichotomy (run.idx w1) (run.idx w2) with h | h | h
  · exact (hno2 w1 (ht1.symm.trans ht2) hi1 h ⟨hw1, j, hj, hl1⟩).elim
  · exact run.idx_inj w1 w2 (ht1.symm.trans ht2) h
  · exact (hno1 w2 (ht2.symm.trans ht1) hi2 h ⟨hw2, j, hj, hl2⟩).elim

/-- A `send` happens before the `recv` that gets its value. The `send` with the number `k` ends
with its write of the flag `full k`, and the `recv` with the number `k` waits for that flag. So
each access before the write of `full k` happens before each access after a read of it. -/
theorem send_hb_recv (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {c k} (hc : c < P.chans) (hk : k < P.msgs) {a w e b : X.E}
    (hw : X.IsWrite w) (hwl : X.loc w = P.sig .send c + k) (haw : a = w ∨ X.po a w)
    (her : X.IsRead e) (hel : X.loc e = P.sig .send c + k) (heb : e = b ∨ X.po e b) :
    X.hb a b := by
  obtain ⟨w', hrf⟩ := run.rf_source e her
  obtain ⟨hw', -, hwl'⟩ := run.rf_write_read w' e hrf
  have := set_unique run hP wf hcons ho hfin hc hk (Nat.zero_le _) hw' (hwl'.trans hel) hw hwl
  subst this
  have hwk : X.kind w' ≠ .fence := by rw [hw]; nofun
  have hek : X.kind e ≠ .fence := by rw [her]; nofun
  exact X.hb_of_signal haw hw (sig_sc run hP hc hk hwk hwl) hrf (sig_sc run hP hc hk hek hel) heb

/-- For a channel with room for `n` values, the `recv` with the number `k` happens before the
`send` with the number `k + n` ends. The `recv` ends with its write of the flag `taken (k + n)`.
The `send` waits for that flag, and ends with its write of `full (k + n)`. -/
theorem recv_hb_send_capacity (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent)
    (ho : X.CoOrder) (hfin : X.Finite) {c k} (hc : c < P.chans) (hk : k + P.cap c < P.msgs)
    {a t w : X.E} (ht : X.IsWrite t) (htl : X.loc t = P.sig .recv c + (k + P.cap c))
    (hat : a = t ∨ X.po a t) (hw : X.IsWrite w) (hwl : X.loc w = P.sig .send c + (k + P.cap c)) :
    X.hb a w := by
  obtain ⟨r, -, -, -, -, hrt, -, -, e, het, -, hew, her, hes, hel⟩ :=
    set_writer run hP hc hk (Nat.zero_le _) hw hwl
  simp only [off, Op.other, Nat.sub_zero] at hel
  obtain ⟨t', hrf⟩ := run.rf_source e her
  obtain ⟨ht', -, htl'⟩ := run.rf_write_read t' e hrf
  have hj : k + P.cap c < P.sigLen .recv := by simp only [sigLen]; omega
  have := set_unique run hP wf hcons ho hfin hc hj (by simp only [off]; omega) ht'
    (htl'.trans hel) ht htl
  subst this
  have htk : X.kind t' ≠ .fence := by rw [ht]; nofun
  exact X.hb_of_signal hat ht (sig_sc run hP hc hj htk htl) hrf hes
    (Or.inr ((run.po_iff e w).2 ⟨het, hew⟩))


end ChanProg

end Vx
