import VxMemoryModel.Footprint
import VxMemoryModel.LanguageExample

/-!
# Programs that follow the footprint rule

`blockFoot` has the tasks of `forkJoin`, in a `tasks` block. The main code writes `input`, starts
both tasks in a block, and reads `out1` and `out2` after the block. Each task reads `input` and
writes its own output.

`loopFoot` has the same tasks, with a block in a loop. Each pass of the loop starts one task:
task 0 in the first pass, and task 1 in the second.

In `nestFoot`, a task starts tasks. The main code writes the place 3 and starts task 0 in a block.
The code of that block reads the place 3. Task 0 writes the place 2, and then starts tasks 1 and 2
in its own block. Task 1 reads the places 2 and 3 and writes the place 0. Task 2 reads the place 3
and writes the place 1. After its block, task 0 reads the places 0 and 1. After the main block,
the main code writes them.

The rule holds for the three programs, so no execution of them has a data race.
-/

namespace Vx

open TaskProg

def blockPhases : List Phase :=
  [.code (.store input (.const 5) .plain), .block (.seq (.spawn 0) (.spawn 1)) none,
    .code (.seq (.load 1 out1 .plain) (.load 2 out2 .plain))]

def blockFoot : TaskProg := { forkJoin with main := seqAll blockPhases }

theorem blockFoot_footprints :
    blockFoot.Footprints (· < blockFoot.flags) (fun _ => []) blockPhases where
  ok :=
    { main := by
        simp [blockFoot, blockPhases, seqAll, Phase.stmt, Stmt.Below, forkJoin, input, out1, out2]
      task := forkJoin_ok.task }
  le_flags _ h := h
  main := rfl
  phases := by
    simp [PhasesOk, blockPhases, Phase.Ok, BlockOk, Phase.ids, Stmt.NoTasks, Stmt.NoBlocks, Stmt.RegsBelow,
      Expr.RegsBelow, Stmt.spawnIds, Stmt.Apart, Stmt.Accesses, Compat, blockFoot, forkJoin, input,
      out1, out2]
  tasks
    | 0 | 1 | _ + 2 => by simp [blockFoot, forkJoin, Stmt.NoTasks]
  task_phases _ := trivial
  one_parent := one_parent_nil _
  order _ _ := nofun

theorem blockFoot_raceFree {X : Execution} (hX : blockFoot.toProg.Execs X) : X.RaceFree :=
  raceFree blockFoot_footprints hX

/-- Register 0 counts the passes, and the loop runs while the register is not 2. A pass starts
task `regs 0` in its block. After the block, it reads `out1` and adds 1 to the register. Task 0
writes `out1`, but the block joined it before the read. -/
def loopPhases : List Phase :=
  [.code (.seq (.store input (.const 5) .plain) (.assign 0 (.const 0))),
    .block (.spawnAt 0 2 (.reg 0))
      (some ⟨.sub (.const 2) (.reg 0), .skip,
        .seq (.load 3 out1 .plain) (.assign 0 (.add (.reg 0) (.const 1))), trivial,
        ⟨trivial, trivial⟩⟩),
    .code (.seq (.load 1 out1 .plain) (.load 2 out2 .plain))]

def loopFoot : TaskProg := { forkJoin with main := seqAll loopPhases }

theorem loopFoot_footprints :
    loopFoot.Footprints (· < loopFoot.flags) (fun _ => []) loopPhases where
  ok :=
    { main := by
        simp [loopFoot, loopPhases, seqAll, Phase.stmt, Stmt.Below, forkJoin, input, out1, out2]
      task := forkJoin_ok.task }
  le_flags _ h := h
  main := rfl
  phases := by
    simp [PhasesOk, loopPhases, Phase.Ok, BlockOk, Phase.ids, Stmt.NoTasks, Stmt.NoBlocks,
      Stmt.RegsBelow, Expr.RegsBelow, Stmt.spawnIds, Stmt.Apart, Stmt.Accesses, loopFoot,
      forkJoin, input, out1, out2]
    -- The two tasks of the block, one for each pass.
    intro u hu v hv huv
    obtain rfl | rfl : u = 0 ∨ u = 1 := by omega
    all_goals obtain rfl | rfl : v = 0 ∨ v = 1 := by omega
    all_goals simp_all [Stmt.Accesses, Compat]
  tasks
    | 0 | 1 | _ + 2 => by simp [loopFoot, forkJoin, Stmt.NoTasks]
  task_phases _ := trivial
  one_parent := one_parent_nil _
  order _ _ := nofun

theorem loopFoot_raceFree {X : Execution} (hX : loopFoot.toProg.Execs X) : X.RaceFree :=
  raceFree loopFoot_footprints hX

def nestMain : List Phase :=
  [.code (.store 3 (.const 1) .plain), .block (.seq (.spawn 0) (.load 0 3 .plain)) none,
    .code (.seq (.store 0 (.const 3) .plain) (.store 1 (.const 4) .plain))]

def nestTask0 : List Phase :=
  [.code (.store 2 (.const 5) .plain), .block (.seq (.spawn 1) (.spawn 2)) none,
    .code (.seq (.load 0 0 .plain) (.load 1 1 .plain))]

def nestPhases : Nat → List Phase
  | 0 => nestTask0
  | _ => []

def nestFoot : TaskProg where
  main := seqAll nestMain
  task
    | 0 => seqAll nestTask0
    | 1 => .seq (.load 0 2 .plain) (.seq (.load 1 3 .plain)
        (.store 0 (.add (.reg 0) (.reg 1)) .plain))
    | 2 => .seq (.load 0 3 .plain) (.store 1 (.reg 0) .plain)
    | _ => .skip
  count := 3
  flags := 4
  marks := 2

/-- Task 0 starts tasks 1 and 2. They start no task. -/
theorem nest_desc {u w : Nat} (h : Desc nestPhases u w) : w = u ∨ u = 0 ∧ (w = 1 ∨ w = 2) := by
  cases h with
  | refl => exact Or.inl rfl
  | @step _ v _ hs hd =>
    obtain ⟨p, hp, hv⟩ := hs
    cases u with
    | succ => simp [nestPhases] at hp
    | zero =>
      simp only [nestPhases, nestTask0, List.mem_cons, List.not_mem_nil, or_false] at hp
      rcases hp with rfl | rfl | rfl <;> simp [Phase.ids, Stmt.spawnIds] at hv
      rcases hv with rfl | rfl
      · exact Or.inr ⟨rfl, Or.inl ((desc_of_nil rfl).1 hd)⟩
      · exact Or.inr ⟨rfl, Or.inr ((desc_of_nil rfl).1 hd)⟩

/-- The main code starts task 0, and task 0 starts tasks 1 and 2. -/
theorem nest_starts {q v : Nat} (h : ∃ p ∈ thr nestMain nestPhases q, v ∈ p.ids) :
    q = 0 ∧ v = 0 ∨ q = 1 ∧ (v = 1 ∨ v = 2) := by
  obtain ⟨p, hp, hv⟩ := h
  rcases q with _ | _ | q
  · simp only [thr, nestMain, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl <;> simp [Phase.ids, Stmt.spawnIds] at hv
    exact Or.inl ⟨rfl, hv⟩
  · simp only [thr, nestPhases, nestTask0, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl <;> simp [Phase.ids, Stmt.spawnIds] at hv
    exact Or.inr ⟨rfl, hv⟩
  · simp [thr, nestPhases] at hp

theorem nestFoot_footprints : nestFoot.Footprints (· < nestFoot.flags) nestPhases nestMain where
  ok :=
    { main := by simp [nestFoot, nestMain, seqAll, Phase.stmt, Stmt.Below]
      task
        | 0 => by simp [nestFoot, nestTask0, seqAll, Phase.stmt, Stmt.Below]
        | 1 | 2 | _ + 3 => by simp [nestFoot, Stmt.Below] }
  le_flags _ h := h
  main := rfl
  phases := by
    simp only [PhasesOk, nestMain, Phase.Ok, BlockOk, Phase.ids, Stmt.spawnIds]
    refine ⟨by simp [Stmt.NoTasks], by simp, ⟨by simp [Stmt.NoBlocks],
      by simp [Stmt.RegsBelow, Expr.RegsBelow, nestFoot], fun u hu w hw => ?_, by simp⟩,
      by simp, by simp [Stmt.NoTasks], by simp, trivial⟩
    simp at hu
    subst hu
    rcases nest_desc hw with rfl | ⟨-, rfl | rfl⟩ <;>
      simp [Stmt.Apart, Stmt.Accesses, Compat, nestFoot, nestTask0, seqAll, Phase.stmt]
  tasks
    | 0 => Or.inl rfl
    | 1 | 2 | _ + 3 => Or.inr ⟨by simp [nestFoot, Stmt.NoTasks], rfl⟩
  task_phases
    | 0 => by
      simp only [PhasesOk, nestPhases, nestTask0, Phase.Ok, BlockOk, Phase.ids, Stmt.spawnIds]
      refine ⟨by simp [Stmt.NoTasks], by simp, ⟨by simp [Stmt.NoBlocks],
        by simp [Stmt.RegsBelow, Expr.RegsBelow], fun u _ w _ => by simp [Stmt.Apart, Stmt.Accesses],
        fun u hu v hv huv w w' hw hw' => ?_⟩, by simp, by simp [Stmt.NoTasks], by simp, trivial⟩
      simp at hu hv
      rcases hu with rfl | rfl <;> rcases hv with rfl | rfl <;> simp at huv <;>
        rw [(desc_of_nil rfl).1 hw, (desc_of_nil rfl).1 hw'] <;>
        simp [Stmt.Apart, Stmt.Accesses, Compat, nestFoot]
    | _ + 1 => trivial
  one_parent q q' v h h' := by
    rcases nest_starts h with ⟨rfl, rfl⟩ | ⟨rfl, rfl | rfl⟩ <;>
      rcases nest_starts h' with ⟨rfl, h⟩ | ⟨rfl, h | h⟩ <;> first | rfl | omega
  order t v h := by
    rcases nest_starts (q := t + 1) h with ⟨h, -⟩ | ⟨h, h' | h'⟩ <;> omega

theorem nestFoot_raceFree {X : Execution} (hX : nestFoot.toProg.Execs X) : X.RaceFree :=
  raceFree nestFoot_footprints hX

end Vx
