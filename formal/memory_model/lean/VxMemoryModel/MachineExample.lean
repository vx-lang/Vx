import VxMemoryModel.Interleave

/-!
# Examples of runs of a CPU node and of a fleet

The program `mp` passes a message. Thread 0 writes 42 to the place 0 and then sets the flag at
the place 1. Thread 1 waits for the flag and then reads the place 0 into its register 0.

- `mp_reach`: a finite run in which thread 1 reads 42. `Cpu.Reach.execs` turns it into an
  execution of `mp` in which the accesses happen one at a time.
- `mp_fair`: a run of a fleet whose node 0 runs `mp` to its end. Then no action can happen, so the
  run is fair.
- `idle_not_fair`: a run in which no node ever moves is not fair, because thread 0 can move at each
  point.
-/

namespace Vx.MachineExample

def mp : Prog where
  code
    | 0 => .seq (.store 0 (.const 42) .plain) (.signal 1)
    | 1 => .seq (.wait 1) (.load 0 0 .plain)
    | _ => .skip

theorem not_pending_seq {s t : Stmt} (h : ¬ s.Pending) : ¬ (Stmt.seq s t).Pending := h

/-- A finite run of `mp`, in which thread 1 reads 42. -/
theorem mp_reach : ∃ evs c, Cpu.Reach (Cpu.init mp) evs c ∧ c.regs 1 0 = 42 := by
  refine ⟨_, _, .tail (.tail (.tail (.tail (.tail (.tail .refl
    (.one (t := 0) (.seq .store) nofun nofun nofun))
    (.silent (t := 0) .seq_skip))
    (.one (t := 0) .store nofun nofun nofun))
    (.one (t := 1) (.seq (.wait 1)) nofun nofun fun _ => rfl))
    (.silent (t := 1) .seq_skip))
    (.one (t := 1) (.load 42) nofun nofun fun _ => rfl), rfl⟩

/-- The accesses of the run of `mp_reach` are an execution of `mp`. -/
theorem mp_execs : ∃ evs, mp.Execs (ofEvents evs) ∧ (ofEvents evs).SeqConsistent := by
  obtain ⟨evs, _, h, _⟩ := mp_reach
  exact ⟨evs, h.execs.1, h.execs.2.1⟩

/-! ## A fleet -/

/-- A fleet of CPU nodes. -/
def fleet : Fleet Cpu Unit := ⟨fun _ => Cpu.node Unit⟩

/-- A program that does nothing. -/
def idle : Prog := ⟨fun _ => .skip⟩

/-- Node 0 is in the state `s`. The other nodes do nothing, and no message is in flight. -/
def cfg (s : Cpu) : Fleet.Config Cpu Unit := ⟨fun n => if n = 0 then s else Cpu.init idle, [], 0⟩

theorem cfg_step {s s' : Cpu} {t ls} (h : Cpu.Step s t ls s') :
    fleet.Step (.move 0 t) (cfg s) (cfg s') := by
  have e : cfg s' = ⟨setAt (cfg s).state 0 s', (cfg s).flight ++ Fleet.number (cfg s).sent [],
      (cfg s).sent + ([] : List (Nat × Unit)).length⟩ := by
    simp only [cfg, Fleet.Config.mk.injEq]
    refine ⟨funext fun n => ?_, rfl, rfl⟩
    by_cases hn : n = 0 <;> simp [setAt, hn]
  rw [e]
  exact .move ⟨rfl, _, h⟩

theorem no_step_of_skip {c : Cpu} {t ls c'} (hc : c.code t = .skip) : ¬ Cpu.Step c t ls c' := by
  intro h
  cases h with
  | silent hs | one hs | two hs => rw [hc] at hs; cases hs

/-- The states of node 0 in the run of `mp_reach`. -/
def s0 : Cpu := Cpu.init mp
def s1 : Cpu := s0.move 0 (.seq .skip (.signal 1)) (s0.regs 0)
  (Label.apply ⟨.write, .plain, 0, 42, false⟩ s0.mem)
def s2 : Cpu := s1.move 0 (.signal 1) (s1.regs 0) s1.mem
def s3 : Cpu := s2.move 0 .skip (s2.regs 0) (Label.apply ⟨.write, .sc, 1, 1, false⟩ s2.mem)
def s4 : Cpu := s3.move 1 (.seq .skip (.load 0 0 .plain)) (s3.regs 1) s3.mem
def s5 : Cpu := s4.move 1 (.load 0 0 .plain) (s4.regs 1) s4.mem
def s6 : Cpu := s5.move 1 .skip (update (s5.regs 1) 0 42) s5.mem

theorem step0 : Cpu.Step s0 0 [⟨.write, .plain, 0, 42, false⟩] s1 :=
  .one (.seq .store) nofun nofun nofun
theorem step1 : Cpu.Step s1 0 [] s2 := .silent .seq_skip
theorem step2 : Cpu.Step s2 0 [⟨.write, .sc, 1, 1, false⟩] s3 := .one .store nofun nofun nofun
theorem step3 : Cpu.Step s3 1 [⟨.read, .sc, 1, 1, false⟩] s4 :=
  .one (.seq (.wait 1)) nofun nofun fun _ => rfl
theorem step4 : Cpu.Step s4 1 [] s5 := .silent .seq_skip
theorem step5 : Cpu.Step s5 1 [⟨.read, .plain, 0, 42, false⟩] s6 :=
  .one (.load 42) nofun nofun fun _ => rfl

def states : Nat → Cpu
  | 0 => s0
  | 1 => s1
  | 2 => s2
  | 3 => s3
  | 4 => s4
  | 5 => s5
  | _ => s6

/-- Node 0 runs `mp` to its end, and then nothing changes. -/
def run (i : Nat) : Fleet.Config Cpu Unit := cfg (states i)

theorem run_isRun : fleet.IsRun (cfg s0) run where
  start := rfl
  next
    | 0 => Or.inl ⟨_, cfg_step step0⟩
    | 1 => Or.inl ⟨_, cfg_step step1⟩
    | 2 => Or.inl ⟨_, cfg_step step2⟩
    | 3 => Or.inl ⟨_, cfg_step step3⟩
    | 4 => Or.inl ⟨_, cfg_step step4⟩
    | 5 => Or.inl ⟨_, cfg_step step5⟩
    | _ + 6 => Or.inr rfl

theorem s6_code (t : Nat) : s6.code t = .skip := by
  match t with
  | 0 | 1 => rfl
  | t + 2 => simp [s6, s5, s4, s3, s2, s1, s0, Cpu.move, Cpu.init, setAt, mp]

/-- The run ends: from point 6 on, no action can happen. So the run is fair. -/
theorem mp_fair : fleet.Fair run := by
  refine Fleet.fair_of_halt (k := 6) fun act j hj ⟨c', h⟩ => ?_
  obtain ⟨j, rfl⟩ : ∃ j', j = j' + 6 := ⟨j - 6, by omega⟩
  cases h with
  | @move _ n _ _ _ h =>
    obtain ⟨_, _, h⟩ := h
    refine no_step_of_skip ?_ h
    by_cases hn : n = 0
    · simp [run, cfg, states, hn, s6_code]
    · simp [run, cfg, hn, Cpu.init, idle]
  | deliver i hi => exact absurd hi (Nat.not_lt_zero _)

/-- Each step of thread 0 from `s0` changes its code. -/
theorem step0_changes {ls c} (h : Cpu.Step s0 0 ls c) : c.code 0 ≠ s0.code 0 := by
  have e : s0.code 0 = .seq (.store 0 (.const 42) .plain) (.signal 1) := rfl
  cases h with
  | silent hs => rw [e] at hs; cases hs with | seq hs => cases hs
  | one hs =>
    rw [e] at hs
    cases hs with
    | seq hs => cases hs; simp [Cpu.move, setAt, e]
  | two hs _ hp => rw [e] at hs; cases hs with | seq hs => cases hs; exact (nomatch hp)

/-- In a run where nothing moves, thread 0 of node 0 can move at each point and never does. So
the run is not fair. -/
theorem idle_not_fair : fleet.IsRun (cfg s0) (fun _ => cfg s0) ∧ ¬ fleet.Fair fun _ => cfg s0 := by
  refine ⟨⟨rfl, fun _ => Or.inr rfl⟩, fun hf => ?_⟩
  obtain ⟨j, _, h⟩ := hf (.move 0 0) 0 fun _ _ => ⟨_, cfg_step step0⟩
  obtain ⟨s', _, ⟨_, ls, h⟩, hs⟩ := h.move_state rfl
  simp only [cfg, ↓reduceIte] at hs h
  subst hs
  exact step0_changes h rfl

end Vx.MachineExample
