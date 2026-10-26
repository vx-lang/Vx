import VxMemoryModel.Typed

/-!
# Examples of the typed layer

`counterProg` has five locals: an atomic counter (local 0), a struct of two numbers (local 1), a
number (local 2), a reference `&counter` (local 3) and a reference `&mut stats` (local 4). Task 0
moves `&counter` into itself and borrows the number with `ref`. Task 1 borrows the counter with
`ref` and moves `&mut stats` into itself. Both tasks add to the counter, and only task 1 writes
the struct. The code of the block reads the number and adds to the counter.

`cellProg` borrows a cell with `ref` in a task, and `cellRefProg` moves a reference to a cell
into a task. The checker rejects both, because a cell is not `Sync`.
-/

namespace Vx

namespace Typed

def counterProg : TProg where
  ty
    | 0 => .val (.leaf .atomic)
    | 1 => .val (.pair (.leaf .int) (.leaf .int))
    | 3 => .ref 0
    | 4 => .mutRef 1
    | _ => .val (.leaf .int)
  locals := 5
  width := 2
  regs := 3
  count := 2
  caps
    | 0 => [(3, .move), (2, .ref)]
    | 1 => [(0, .ref), (4, .move)]
    | _ => []
  body
    | 0 => .seq (.read 0 2 0) (.fetchAdd 1 3 0 (.reg 0))
    | 1 => .seq (.fetchAdd 0 0 0 (.const 1)) (.seq (.write 4 0 (.const 1)) (.write 4 1 (.const 2)))
    | _ => .skip
  main := [.code (.write 2 0 (.const 10)),
    .block [.spawn 0, .spawn 1, .code (.seq (.read 2 2 0) (.fetchAdd 2 0 0 (.const 1)))],
    .code (.read 0 1 0)]

theorem counterProg_wellTyped : counterProg.WellTyped where
  width x := by
    rcases x with _ | _ | _ | _ | _ | x <;> simp [counterProg, TProg.vty, Ty.leaves]
  targets x y h := by
    rcases x with _ | _ | _ | _ | _ | x <;> simp [counterProg] at h <;> subst h <;> simp [counterProg]
  caps u x m h := by
    rcases u with _ | _ | u <;> simp [counterProg] at h <;>
      rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;>
      simp [counterProg, TProg.SyncL, TProg.SendL, TProg.vty, Ty.Sync, Ty.Send, Leaf.Sync]
  bodies u := by
    rcases u with _ | _ | u <;>
      simp [counterProg, TProg.Fits, TProg.At, TProg.pathPerm, TProg.capPerm, TProg.target,
        TProg.vty, via, Allowed, Expr.RegsBelow, Ty.leaves]
  phases p hp := by
    simp only [counterProg, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl <;>
      simp [TProg.PhaseOk, TProg.Fits, TProg.At, TProg.blockFits, TProg.blockPerm, TProg.taskPerm,
        TProg.pathPerm, TProg.capPerm, TProg.target, TProg.vty, spawnsOf, via, Allowed,
        Expr.RegsBelow, Ty.leaves, counterProg]
    exact ⟨fun x => by rcases x with _ | _ | _ | x <;> simp,
      fun x => by rcases x with _ | _ | _ | x <;> simp⟩
  nodup := by decide
  spawn_lt := by decide

/-- No execution of `counterProg` has a data race. -/
theorem counterProg_raceFree {X : Execution} (hX : counterProg.toTaskProg.toProg.Execs X) :
    X.RaceFree :=
  counterProg_wellTyped.raceFree hX

/-- A task borrows a cell with `ref`. Two such tasks could write the cell at the same time. -/
def cellProg : TProg where
  ty _ := .val (.leaf .cell)
  locals := 1
  width := 1
  regs := 1
  count := 2
  caps _ := [(0, .ref)]
  body _ := .write 0 0 (.const 1)
  main := [.block [.spawn 0, .spawn 1]]

/-- The checker rejects `cellProg`, because a cell is not `Sync`. -/
theorem cellProg_rejected : ¬ cellProg.WellTyped := fun h => by
  have := (h.caps 0 0 .ref (by simp [cellProg])).1 rfl
  simp [cellProg, TProg.SyncL, Ty.Sync, Leaf.Sync] at this

/-- Each task moves a copy of `&cell` into itself, and writes the cell through it. -/
def cellRefProg : TProg where
  ty
    | 0 => .val (.leaf .cell)
    | _ => .ref 0
  locals := 2
  width := 1
  regs := 1
  count := 2
  caps _ := [(1, .move)]
  body _ := .write 1 0 (.const 1)
  main := [.block [.spawn 0, .spawn 1]]

/-- The checker rejects `cellRefProg`, because `&cell` is not `Send`. -/
theorem cellRefProg_rejected : ¬ cellRefProg.WellTyped := fun h => by
  have := (h.caps 0 1 .move (by simp [cellRefProg])).2 nofun
  simp [cellRefProg, TProg.SendL, TProg.vty, Ty.Sync, Leaf.Sync] at this

end Typed

end Vx
