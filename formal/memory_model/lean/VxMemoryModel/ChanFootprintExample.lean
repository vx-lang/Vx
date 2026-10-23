import VxMemoryModel.LockFootprint

/-!
# Programs with a channel and a lock that follow the footprint rule

In `chanFoot`, the main code starts two tasks in a block. Task 0 sends 7 on channel 0. Task 1
receives it into the register 4 and writes it to the place 0. After the block, the main code
reads the place 0. The two tasks share no place of the code.

In `lockFoot`, both tasks add 1 to the place 0, each in a `locked` block of lock 0. The rule
gives the place 0 to lock 0, so the two tasks can share it. The main code reads the place 0 after
the block, also under the lock.

The rule holds for both programs, so no execution of them has a data race.
-/

namespace Vx

open ChanProg

def chanPhases : List CPhase :=
  [.block (.seq (.base (.spawn 0)) (.base (.spawn 1))), .code (.base (.load 4 0 .plain))]

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

theorem chanFoot_footprints : chanFoot.CFootprints (fun _ => none) chanPhases where
  ok :=
    { main := by simp [chanFoot, chanPhases, cseqAll, CPhase.stmt, StmtOk, Stmt.Below]
      task
        | 0 => by simp [chanFoot, StmtOk]
        | 1 => by simp [chanFoot, StmtOk, Stmt.Below]
        | _ + 2 => by simp [chanFoot, StmtOk, Stmt.Below]
      cap _ _ := by simp [chanFoot] }
  errors := rfl
  main := rfl
  phases := by
    simp [CPhasesOk, chanPhases, CPhase.Ok, CPhase.lower, TaskProg.Phase.ids, CStmt.NoTasks,
      CStmt.NoBlocks, CStmt.RegsBelow, Stmt.NoTasks, Stmt.NoBlocks, Stmt.RegsBelow,
      Expr.RegsBelow, lowerC, Stmt.spawnIds, CStmt.Apart, CStmt.Accesses, Stmt.Accesses, Compat,
      chanFoot]
  tasks
    | 0 | 1 | _ + 2 => by simp [chanFoot, CStmt.NoTasks, Stmt.NoTasks]
  guard := nofun
  guarded_main := by
    simp [chanFoot, chanPhases, cseqAll, CPhase.stmt, CStmt.Guarded, Stmt.Accesses]
  guarded_task
    | 0 | 1 | _ + 2 => by simp [chanFoot, CStmt.Guarded, Stmt.Accesses]

theorem chanFoot_raceFree {X : Execution} (hX : chanFoot.toTaskProg.toProg.Execs X)
    (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite) : X.RaceFree :=
  raceFree chanFoot_footprints hX wf hc ho hfin

/-- Add 1 to the place 0, under lock 0. -/
def addOne : CStmt :=
  .locked 0 (.seq (.base (.load 4 0 .plain)) (.base (.store 0 (.add (.reg 4) (.const 1)) .plain)))

def lockPhases : List CPhase :=
  [.block (.seq (.base (.spawn 0)) (.base (.spawn 1))), .code (.locked 0 (.base (.load 4 0 .plain)))]

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

theorem lockFoot_footprints : lockFoot.CFootprints lockGuard lockPhases where
  ok :=
    { main := by simp [lockFoot, lockPhases, cseqAll, CPhase.stmt, StmtOk, Stmt.Below, CStmt.Locks]
      task
        | 0 | 1 => by simp [lockFoot, addOne, StmtOk, Stmt.Below, CStmt.Locks]
        | _ + 2 => by simp [lockFoot, StmtOk, Stmt.Below]
      cap _ h := by simp [lockFoot] at h }
  errors := rfl
  main := rfl
  phases := by
    simp [CPhasesOk, lockPhases, CPhase.Ok, CPhase.lower, TaskProg.Phase.ids, CStmt.NoTasks,
      CStmt.NoBlocks, CStmt.RegsBelow, Stmt.NoTasks, Stmt.NoBlocks, Stmt.RegsBelow,
      Expr.RegsBelow, lowerC, Stmt.spawnIds, CStmt.Apart, CStmt.Accesses, Stmt.Accesses, Compat,
      lockFoot, addOne, lockGuard]
  tasks
    | 0 | 1 | _ + 2 => by simp [lockFoot, addOne, CStmt.NoTasks, Stmt.NoTasks]
  guard x m h := by
    simp only [lockGuard] at h
    split at h <;> simp_all [lockFoot]
  guarded_main := by
    simp [lockFoot, lockPhases, cseqAll, CPhase.stmt, CStmt.Guarded, Stmt.Accesses, lockGuard]
  guarded_task
    | 0 | 1 => by simp [lockFoot, addOne, CStmt.Guarded, Stmt.Accesses, lockGuard]
    | _ + 2 => by simp [lockFoot, CStmt.Guarded, Stmt.Accesses]

theorem lockFoot_raceFree {X : Execution} (hX : lockFoot.toTaskProg.toProg.Execs X)
    (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite) : X.RaceFree :=
  raceFree lockFoot_footprints hX wf hc ho hfin

end Vx
