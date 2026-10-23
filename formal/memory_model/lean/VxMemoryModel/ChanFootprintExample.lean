import VxMemoryModel.ChanFootprint

/-!
# A program with a channel that follows the footprint rule

The main code starts two tasks in a block. Task 0 sends 7 on channel 0. Task 1 receives it into
the register 4 and writes it to the place 0. After the block, the main code reads the place 0.
The two tasks share no place of the code, so the rule holds, and no execution of the program
has a data race.
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

theorem chanFoot_footprints : chanFoot.CFootprints chanPhases where
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

theorem chanFoot_raceFree {X : Execution} (hX : chanFoot.toTaskProg.toProg.Execs X)
    (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite) : X.RaceFree :=
  raceFree chanFoot_footprints hX wf hc ho hfin

end Vx
