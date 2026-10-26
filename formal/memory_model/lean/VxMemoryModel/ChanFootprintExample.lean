import VxMemoryModel.LockFootprint

/-!
# Programs with channels, locks and failures that follow the footprint rule

In `chanFoot`, the main code starts two tasks in a block. Task 0 sends 7 on channel 0. Task 1
receives it into the register 4 and writes it to the place 0. After the block, the main code
reads the place 0. The two tasks share no place of the code.

In `lockFoot`, both tasks add 1 to the place 0, each in a `locked` block of lock 0. The rule
gives the place 0 to lock 0, so the two tasks can share it. After the block, the main code reads the
place 0 without the lock, because no task runs then.

In `failFoot`, task 0 writes the place 0 and then fails. Task 1 checks the cancel flag and then
writes the place 1. After the block, the main code reads the failure slot and both places.

In `nestChanFoot`, the main code starts task 0 in a block, and task 0 starts tasks 1 and 2 in its
own block. Task 1 sends 7 on channel 0, and task 2 receives it and writes it to the place 0.
After their blocks, task 0 and then the main code read the place 0.

The rule holds for the four programs, so no execution of them has a data race.
-/

namespace Vx

open ChanProg

def chanPhases : List CPhase :=
  [.block (.seq (.base (.spawn 0)) (.base (.spawn 1))) none, .code (.base (.load 4 0 .plain))]

def chanFoot : ChanProg where
  main := cseqAll chanPhases
  task
    | 0 => .send 0 (.const 7)
    | 1 => .seq (.recv 4 5 0) (.base (.store 0 (.reg 4) .plain))
    | _ => .base .skip
  count := 2
  places := 1
  chans := 1
  locks := 0
  cap _ := 1
  senders _ := 1
  msgs := 1
  tmp := 6

theorem chanFoot_footprints : chanFoot.CFootprints (fun _ => none) (fun _ => []) chanPhases where
  ok :=
    { main := by simp [chanFoot, chanPhases, cseqAll, CPhase.stmt, StmtOk, Stmt.Below]
      task
        | 0 => by simp [chanFoot, StmtOk]
        | 1 => by simp [chanFoot, StmtOk, Stmt.Below]
        | _ + 2 => by simp [chanFoot, StmtOk, Stmt.Below]
      cap _ _ := by simp [chanFoot] }
  main := rfl
  phases := by
    simp [CPhasesOk, chanPhases, CPhase.Ok, CPhase.lower, TaskProg.Phase.ids, CStmt.NoTasks,
      CStmt.NoBlocks, CStmt.RegsBelow, Stmt.NoTasks, Stmt.NoBlocks, Stmt.RegsBelow,
      Expr.RegsBelow, lowerC, Stmt.spawnIds, Stmt.allIds, CStmt.Apart, CStmt.Accesses, Stmt.Accesses, Compat,
      chanFoot]
  tasks
    | 0 | 1 | _ + 2 => Or.inr ⟨by simp [chanFoot, CStmt.NoTasks, Stmt.NoTasks], rfl⟩
  task_phases _ := trivial
  one_parent := one_parent_nil _
  order _ _ := nofun
  guard := nofun
  guarded_blocks B rep hB := by
    simp only [chanPhases, List.mem_cons, CPhase.block.injEq, reduceCtorEq, List.not_mem_nil,
      or_false] at hB
    obtain ⟨rfl, rfl⟩ := hB
    simp [CStmt.Guarded, Stmt.Accesses]
  guarded_task
    | 0 | 1 | _ + 2 => by simp [chanFoot, CStmt.Guarded, Stmt.Accesses]

theorem chanFoot_raceFree {X : Execution} (hX : chanFoot.toTaskProg.toProg.Execs X)
    (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite) : X.RaceFree :=
  raceFree chanFoot_footprints hX wf hc ho hfin

/-- Add 1 to the place 0, under lock 0. -/
def addOne : CStmt :=
  .locked 0 (.seq (.base (.load 4 0 .plain)) (.base (.store 0 (.add (.reg 4) (.const 1)) .plain)))

def lockPhases : List CPhase :=
  [.block (.seq (.base (.spawn 0)) (.base (.spawn 1))) none, .code (.base (.load 4 0 .plain))]

def lockFoot : ChanProg where
  main := cseqAll lockPhases
  task
    | 0 | 1 => addOne
    | _ => .base .skip
  count := 2
  places := 1
  chans := 0
  locks := 1
  cap _ := 1
  senders _ := 1
  msgs := 1
  tmp := 6

/-- The lock 0 guards the place 0. -/
def lockGuard (x : Nat) : Option Nat := if x = 0 then some 0 else none

theorem lockFoot_footprints : lockFoot.CFootprints lockGuard (fun _ => []) lockPhases where
  ok :=
    { main := by simp [lockFoot, lockPhases, cseqAll, CPhase.stmt, StmtOk, Stmt.Below]
      task
        | 0 | 1 => by simp [lockFoot, addOne, StmtOk, Stmt.Below, CStmt.Locks]
        | _ + 2 => by simp [lockFoot, StmtOk, Stmt.Below]
      cap _ h := by simp [lockFoot] at h }
  main := rfl
  phases := by
    simp [CPhasesOk, lockPhases, CPhase.Ok, CPhase.lower, TaskProg.Phase.ids, CStmt.NoTasks,
      CStmt.NoBlocks, CStmt.RegsBelow, Stmt.NoTasks, Stmt.NoBlocks, Stmt.RegsBelow,
      Expr.RegsBelow, lowerC, Stmt.spawnIds, Stmt.allIds, CStmt.Apart, CStmt.Accesses, Stmt.Accesses, Compat,
      lockFoot, addOne, lockGuard]
  tasks
    | 0 | 1 | _ + 2 => Or.inr ⟨by simp [lockFoot, addOne, CStmt.NoTasks, Stmt.NoTasks], rfl⟩
  task_phases _ := trivial
  one_parent := one_parent_nil _
  order _ _ := nofun
  guard x m h := by
    simp only [lockGuard] at h
    split at h <;> simp_all [lockFoot]
  guarded_blocks B rep hB := by
    simp only [lockPhases, List.mem_cons, CPhase.block.injEq, reduceCtorEq, List.not_mem_nil,
      or_false] at hB
    obtain ⟨rfl, rfl⟩ := hB
    simp [CStmt.Guarded, Stmt.Accesses]
  guarded_task
    | 0 | 1 => by simp [lockFoot, addOne, CStmt.Guarded, Stmt.Accesses, lockGuard]
    | _ + 2 => by simp [lockFoot, CStmt.Guarded, Stmt.Accesses]

theorem lockFoot_raceFree {X : Execution} (hX : lockFoot.toTaskProg.toProg.Execs X)
    (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite) : X.RaceFree :=
  raceFree lockFoot_footprints hX wf hc ho hfin

def failPhases : List CPhase :=
  [.block (.seq (.base (.spawn 0)) (.base (.spawn 1))) none,
    .code (.seq (.result 4) (.seq (.base (.load 5 0 .plain)) (.base (.load 5 1 .plain))))]

def failFoot : ChanProg where
  main := cseqAll failPhases
  task
    | 0 => .seq (.base (.store 0 (.const 1) .plain)) (.fail 5)
    | 1 => .seq .checkCancel (.base (.store 1 (.const 2) .plain))
    | _ => .base .skip
  count := 2
  places := 2
  chans := 0
  locks := 0
  cap _ := 1
  senders _ := 1
  msgs := 1
  tmp := 6
  errors := true

theorem failFoot_footprints : failFoot.CFootprints (fun _ => none) (fun _ => []) failPhases where
  ok :=
    { main := by simp [failFoot, failPhases, cseqAll, CPhase.stmt, StmtOk, Stmt.Below]
      task
        | 0 | 1 | _ + 2 => by simp [failFoot, StmtOk, Stmt.Below]
      cap _ h := by simp [failFoot] at h }
  main := rfl
  phases := by
    simp [CPhasesOk, failPhases, CPhase.Ok, CPhase.lower, TaskProg.Phase.ids, CStmt.NoTasks,
      CStmt.NoBlocks, CStmt.RegsBelow, Stmt.NoTasks, Stmt.NoBlocks, Stmt.RegsBelow,
      Expr.RegsBelow, lowerC, Stmt.spawnIds, Stmt.allIds, CStmt.Apart, CStmt.Accesses, Stmt.Accesses, Compat,
      failFoot]
  tasks
    | 0 | 1 | _ + 2 => Or.inr ⟨by simp [failFoot, CStmt.NoTasks, Stmt.NoTasks], rfl⟩
  task_phases _ := trivial
  one_parent := one_parent_nil _
  order _ _ := nofun
  guard := nofun
  guarded_blocks B rep hB := by
    simp only [failPhases, List.mem_cons, CPhase.block.injEq, reduceCtorEq, List.not_mem_nil,
      or_false] at hB
    obtain ⟨rfl, rfl⟩ := hB
    simp [CStmt.Guarded, Stmt.Accesses]
  guarded_task
    | 0 | 1 | _ + 2 => by simp [failFoot, CStmt.Guarded, Stmt.Accesses]

theorem failFoot_raceFree {X : Execution} (hX : failFoot.toTaskProg.toProg.Execs X)
    (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite) : X.RaceFree :=
  raceFree failFoot_footprints hX wf hc ho hfin

def nestChanMain : List CPhase :=
  [.block (.seq (.base (.spawn 0)) (.base (.load 4 1 .plain))) none, .code (.base (.load 4 0 .plain))]

def nestChanTask0 : List CPhase :=
  [.block (.seq (.base (.spawn 1)) (.base (.spawn 2))) none, .code (.base (.load 4 0 .plain))]

def nestChanPhases : Nat → List CPhase
  | 0 => nestChanTask0
  | _ => []

def nestChanFoot : ChanProg where
  main := cseqAll nestChanMain
  task
    | 0 => cseqAll nestChanTask0
    | 1 => .send 0 (.const 7)
    | 2 => .seq (.recv 4 5 0) (.base (.store 0 (.reg 4) .plain))
    | _ => .base .skip
  count := 3
  places := 2
  chans := 1
  locks := 0
  cap _ := 1
  senders _ := 1
  msgs := 1
  tmp := 6

/-- Task 0 starts tasks 1 and 2. They start no task. -/
theorem nestChan_desc {u w : Nat} (h : TaskProg.Desc (nestChanFoot.lowerPhases nestChanPhases) u w) :
    w = u ∨ u = 0 ∧ (w = 1 ∨ w = 2) := by
  cases h with
  | refl => exact Or.inl rfl
  | @step _ v _ hs hd =>
    obtain ⟨p, hp, hv⟩ := hs
    cases u with
    | succ => simp [lowerPhases, nestChanPhases] at hp
    | zero =>
      simp only [lowerPhases, nestChanPhases, nestChanTask0, List.map, List.mem_cons,
        List.not_mem_nil, or_false] at hp
      rcases hp with rfl | rfl <;>
        simp [CPhase.lower, lowerC, TaskProg.Phase.ids, Stmt.allIds] at hv
      rcases hv with rfl | rfl
      · exact Or.inr ⟨rfl, Or.inl ((TaskProg.desc_of_nil rfl).1 hd)⟩
      · exact Or.inr ⟨rfl, Or.inr ((TaskProg.desc_of_nil rfl).1 hd)⟩

/-- The main code starts task 0, and task 0 starts tasks 1 and 2. -/
theorem nestChan_starts {q v : Nat}
    (h : ∃ p ∈ cthr nestChanMain nestChanPhases q, v ∈ (p.lower nestChanFoot).ids) :
    q = 0 ∧ v = 0 ∨ q = 1 ∧ (v = 1 ∨ v = 2) := by
  obtain ⟨p, hp, hv⟩ := h
  rcases q with _ | _ | q
  · simp only [cthr, nestChanMain, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl <;>
      simp [CPhase.lower, lowerC, TaskProg.Phase.ids, Stmt.allIds] at hv
    exact Or.inl ⟨rfl, hv⟩
  · simp only [cthr, nestChanPhases, nestChanTask0, List.mem_cons, List.not_mem_nil,
      or_false] at hp
    rcases hp with rfl | rfl <;>
      simp [CPhase.lower, lowerC, TaskProg.Phase.ids, Stmt.allIds] at hv
    exact Or.inr ⟨rfl, hv⟩
  · simp [cthr, nestChanPhases] at hp

theorem nestChanFoot_footprints :
    nestChanFoot.CFootprints (fun _ => none) nestChanPhases nestChanMain where
  ok :=
    { main := by simp [nestChanFoot, nestChanMain, cseqAll, CPhase.stmt, StmtOk, Stmt.Below]
      task
        | 0 => by simp [nestChanFoot, nestChanTask0, cseqAll, CPhase.stmt, StmtOk, Stmt.Below]
        | 1 => by simp [nestChanFoot, StmtOk]
        | 2 | _ + 3 => by simp [nestChanFoot, StmtOk, Stmt.Below]
      cap _ _ := by simp [nestChanFoot] }
  main := rfl
  phases := by
    simp only [CPhasesOk, nestChanMain, CPhase.Ok, CPhase.lower, TaskProg.Phase.ids, lowerC,
      Stmt.spawnIds, Stmt.allIds]
    refine ⟨⟨by simp [CStmt.NoBlocks, Stmt.NoBlocks],
      by simp [CStmt.RegsBelow, Stmt.RegsBelow, Expr.RegsBelow, nestChanFoot],
      fun u hu w hw => ?_, by simp⟩, by simp, by simp [CStmt.NoTasks, Stmt.NoTasks], by simp,
      trivial⟩
    simp at hu
    subst hu
    rcases nestChan_desc hw with rfl | ⟨-, rfl | rfl⟩ <;>
      simp [CStmt.Apart, CStmt.Accesses, Stmt.Accesses, Compat, nestChanFoot, nestChanTask0,
        cseqAll, CPhase.stmt]
  tasks
    | 0 => Or.inl rfl
    | 1 | 2 | _ + 3 => Or.inr ⟨by simp [nestChanFoot, CStmt.NoTasks, Stmt.NoTasks], rfl⟩
  task_phases
    | 0 => by
      simp only [CPhasesOk, nestChanPhases, nestChanTask0, CPhase.Ok, CPhase.lower,
        TaskProg.Phase.ids, lowerC, Stmt.spawnIds, Stmt.allIds]
      refine ⟨⟨by simp [CStmt.NoBlocks, Stmt.NoBlocks],
        by simp [CStmt.RegsBelow, Stmt.RegsBelow, Expr.RegsBelow],
        fun u _ w _ => by simp [CStmt.Apart, CStmt.Accesses, Stmt.Accesses],
        fun u hu v hv huv w w' hw hw' => ?_⟩, by simp, by simp [CStmt.NoTasks, Stmt.NoTasks],
        by simp, trivial⟩
      simp at hu hv
      rcases hu with rfl | rfl <;> rcases hv with rfl | rfl <;> simp at huv <;>
        rw [(TaskProg.desc_of_nil rfl).1 hw, (TaskProg.desc_of_nil rfl).1 hw'] <;>
        simp [CStmt.Apart, CStmt.Accesses, Stmt.Accesses, Compat, nestChanFoot]
    | _ + 1 => trivial
  one_parent q q' v h h' := by
    rcases nestChan_starts h with ⟨rfl, rfl⟩ | ⟨rfl, rfl | rfl⟩ <;>
      rcases nestChan_starts h' with ⟨rfl, h⟩ | ⟨rfl, h | h⟩ <;> first | rfl | omega
  order t v h := by
    rcases nestChan_starts (q := t + 1) h with ⟨h, -⟩ | ⟨h, h' | h'⟩ <;> omega
  guard := nofun
  guarded_blocks B rep hB := by
    simp only [nestChanMain, List.mem_cons, CPhase.block.injEq, reduceCtorEq, List.not_mem_nil,
      or_false] at hB
    obtain ⟨rfl, rfl⟩ := hB
    simp [CStmt.Guarded, Stmt.Accesses]
  guarded_task
    | 0 => by simp [nestChanFoot, nestChanTask0, cseqAll, CPhase.stmt, CStmt.Guarded,
        Stmt.Accesses]
    | 1 | 2 | _ + 3 => by simp [nestChanFoot, CStmt.Guarded, Stmt.Accesses]

theorem nestChanFoot_raceFree {X : Execution} (hX : nestChanFoot.toTaskProg.toProg.Execs X)
    (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite) : X.RaceFree :=
  raceFree nestChanFoot_footprints hX wf hc ho hfin

end Vx
