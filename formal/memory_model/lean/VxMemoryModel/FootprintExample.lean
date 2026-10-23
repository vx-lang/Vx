import VxMemoryModel.Footprint
import VxMemoryModel.LanguageExample

/-!
# A program that follows the footprint rule

The tasks of `forkJoin`, in a `tasks` block. The main code writes `input`, starts both tasks in a
block, and reads `out1` and `out2` after the block. Each task reads `input` and writes its own
output. The rule holds, so no execution of the program has a data race.
-/

namespace Vx

open TaskProg

def blockPhases : List Phase :=
  [.code (.store input (.const 5) .plain), .block (.seq (.spawn 0) (.spawn 1)),
    .code (.seq (.load 1 out1 .plain) (.load 2 out2 .plain))]

def blockFoot : TaskProg := { forkJoin with main := seqAll blockPhases }

theorem blockFoot_footprints : blockFoot.Footprints blockFoot.flags blockPhases where
  ok :=
    { main := by
        simp [blockFoot, blockPhases, seqAll, Phase.stmt, Stmt.Below, forkJoin, input, out1, out2]
      task := forkJoin_ok.task }
  le_flags := Nat.le_refl _
  main := rfl
  phases := by
    simp [PhasesOk, blockPhases, Phase.Ok, Phase.ids, Stmt.NoTasks, Stmt.NoBlocks, Stmt.RegsBelow,
      Expr.RegsBelow, Stmt.spawnIds, Stmt.Apart, Stmt.Accesses, Compat, blockFoot, forkJoin, input,
      out1, out2]
  tasks
    | 0 | 1 | _ + 2 => by simp [blockFoot, forkJoin, Stmt.NoTasks]

theorem blockFoot_raceFree {X : Execution} (hX : blockFoot.toProg.Execs X) : X.RaceFree :=
  raceFree blockFoot_footprints hX

end Vx
