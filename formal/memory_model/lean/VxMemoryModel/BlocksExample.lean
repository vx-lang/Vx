import VxMemoryModel.Blocks
import VxMemoryModel.LanguageExample

/-!
# Examples of `tasks` blocks
-/

namespace Vx

/-! ## The end of a block joins its tasks

The program of `forkJoin`, with no `join` in the code. A `tasks` block starts both tasks, and its
end joins them. The run gives the same accesses as `forkJoin`. -/

def blockJoin : TaskProg :=
  { forkJoin with
    main := .seq (.store input (.const 5) .plain) <|
      .seq (.tasksBlock (.seq (.spawn 0) (.spawn 1))) <|
        .seq (.load 1 out1 .plain) (.load 2 out2 .plain) }

theorem blockJoin_ok : blockJoin.Ok where
  main := by simp [blockJoin, forkJoin, Stmt.Below, input, out1, out2]
  task := forkJoin_ok.task

/-- The block clears the marks 3 and 4, starts both tasks, and joins each task that has its mark
set. The block makes the accesses 1 to 4 of `forkJoinLabels 0`. -/
theorem blockJoin_block :
    Steps (blockJoin.lower (.tasksBlock (.seq (.spawn 0) (.spawn 1)))) (fun _ => 0)
      [forkJoinLabels 0 1, forkJoinLabels 0 2, forkJoinLabels 0 3, forkJoinLabels 0 4] .skip
      (update (update (update (update (fun _ => 0) 3 0) 4 0) 3 1) 4 1) :=
  .silent (.seq (.seq .assign)) <| .silent (.seq .seq_skip) <|
    .silent (.seq (.seq .assign)) <| .silent (.seq .seq_skip) <| .silent .seq_skip <|
    .silent (.seq (.seq (.seq (.assignAt 0 rfl Nat.one_pos)))) <|
    .silent (.seq (.seq .seq_skip)) <|
    .visible (.seq (.seq (.storeAt 0 rfl Nat.one_pos))) <| .silent (.seq .seq_skip) <|
    .silent (.seq (.seq (.assignAt 0 rfl Nat.one_pos))) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.storeAt 0 rfl Nat.one_pos)) <| .silent .seq_skip <|
    .silent (.seq (.ite_true (by decide))) <| .visible (.seq (.waitAt 0 1 rfl Nat.one_pos)) <|
    .silent .seq_skip <|
    .silent (.seq (.ite_true (by decide))) <| .visible (.seq (.waitAt 0 1 rfl Nat.one_pos)) <|
    .silent .seq_skip <| .refl

theorem blockJoin_main : Trace (blockJoin.toProg.code 0) 7 (forkJoinLabels 0) :=
  .visible (.seq .store) <| .silent .seq_skip <|
    .silent (.seq (.seq (.seq .assign))) <| .silent (.seq (.seq .seq_skip)) <|
    .silent (.seq (.seq (.seq .assign))) <| .silent (.seq (.seq .seq_skip)) <|
    .silent (.seq .seq_skip) <|
    .silent (.seq (.seq (.seq (.seq (.assignAt 0 rfl Nat.one_pos))))) <|
    .silent (.seq (.seq (.seq .seq_skip))) <|
    .visible (.seq (.seq (.seq (.storeAt 0 rfl Nat.one_pos)))) <| .silent (.seq (.seq .seq_skip)) <|
    .silent (.seq (.seq (.seq (.assignAt 0 rfl Nat.one_pos)))) <| .silent (.seq (.seq .seq_skip)) <|
    .visible (.seq (.seq (.storeAt 0 rfl Nat.one_pos))) <| .silent (.seq .seq_skip) <|
    .silent (.seq (.seq (.ite_true (by decide)))) <|
    .visible (.seq (.seq (.waitAt 0 1 rfl Nat.one_pos))) <| .silent (.seq .seq_skip) <|
    .silent (.seq (.seq (.ite_true (by decide)))) <|
    .visible (.seq (.seq (.waitAt 0 1 rfl Nat.one_pos))) <| .silent (.seq .seq_skip) <|
    .silent .seq_skip <|
    .visible (.seq (.load 6)) <| .silent .seq_skip <| .visible (.load 7) .stop

def blockJoinRun : blockJoin.toProg.Run forkJoinData.toExecution :=
  { forkJoinRun with
    trace
      | ⟨0, _⟩ | ⟨1, _⟩ | ⟨2, _⟩ | ⟨3, _⟩ | ⟨4, _⟩ | ⟨5, _⟩ =>
        blockJoin_main.prefix (by decide +revert)
      | ⟨6, _⟩ => blockJoin_main
      | ⟨7, _⟩ | ⟨8, _⟩ | ⟨9, _⟩ => forkJoin_task0.prefix (by decide +revert)
      | ⟨10, _⟩ => forkJoin_task0
      | ⟨11, _⟩ | ⟨12, _⟩ | ⟨13, _⟩ => forkJoin_task1.prefix (by decide +revert)
      | ⟨14, _⟩ => forkJoin_task1 }

/-- `block_hb` applies to this run: the write of `out1` by task 0 (event 9) happens before the
read of `out1` after the block (event 5). The `spawn` of task 0 is event 1. -/
theorem blockJoin_hb : forkJoinData.toExecution.hb ⟨9, by decide⟩ ⟨5, by decide⟩ :=
  TaskProg.block_hb blockJoin_ok blockJoinRun (s := .seq (.spawn 0) (.spawn 1))
    (by simp [blockJoin, forkJoin, Stmt.Below])
    (by simp [Stmt.RegsBelow, Expr.RegsBelow]) ⟨trivial, trivial⟩
    (p := 0) (n := 1) blockJoin_block
    (fun | 0, _ => rfl | 1, _ => rfl | 2, _ => rfl | 3, _ => rfl | _ + 4, h => absurd h (by simp))
    (t := 0) (w := ⟨1, by decide⟩) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide)

-- The block lets the plain reads after it see the writes of the tasks, with no data race.
#guard forkJoinData.check == some (true, false)

/-! ## A block that starts no task does not wait

The block starts task 0 only when register 0 is not 0. Register 0 is 0, so the block starts no
task, and its end does not wait. Then the main code writes 1 to the place 0. -/

def skipBlock : TaskProg where
  main := .seq (.tasksBlock (.ite (.reg 0) (.spawn 0) .skip)) (.store 0 (.const 1) .plain)
  task
    | 0 => .store 1 (.const 1) .plain
    | _ => .skip
  count := 1
  flags := 2
  marks := 1

def skipBlockData : Data 1 := Data.ofLists 1 [.write] [.plain] [0] [0] [] [] [] []

def skipBlockLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .plain, 0, 1, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

/-- The end of the block finds the mark of task 0 not set, and goes on. -/
theorem skipBlock_main : Trace (skipBlock.toProg.code 0) 1 (skipBlockLabels 0) :=
  .silent (.seq (.seq (.seq .assign))) <| .silent (.seq (.seq .seq_skip)) <|
    .silent (.seq .seq_skip) <| .silent (.seq (.seq (.ite_false (by decide)))) <|
    .silent (.seq .seq_skip) <| .silent (.seq (.seq (.ite_false (by decide)))) <|
    .silent (.seq .seq_skip) <| .silent .seq_skip <| .visible .store .stop

def skipBlockRun : skipBlock.toProg.Run skipBlockData.toExecution where
  val _ := 1
  isRmw _ := false
  idx _ := 0
  lab := skipBlockLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace | ⟨0, _⟩ => skipBlock_main
  rf_val := by decide
  rf_write_read := by
    simp only [Execution.IsWrite, Execution.IsRead]
    decide
  rf_source := by
    simp only [Execution.IsRead]
    decide
  rmw_iff := by decide

#guard skipBlockData.check == some (true, false)

end Vx
