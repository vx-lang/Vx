import VxMemoryModel.Tasks
import VxMemoryModel.Check

/-!
# Executions of two programs

This file shows that `Prog.Run` is not empty for real programs, so `Prog.drf_sc` is not true
only because no execution qualifies.
-/

namespace Vx

/-! ## Message passing

Place 0 holds the data and place 1 holds a flag. Thread 1 writes the data and then sets the flag
with a release write. Thread 2 reads the flag with an acquire read, and if it is set, reads the
data. In the execution below, thread 2 sees the flag and reads 42. -/

def messagePassing : Prog where
  code
    | 1 => .seq (.store 0 (.const 42) .plain) (.store 1 (.const 1) .rel)
    | 2 => .seq (.load 0 1 .acq) (.ite (.reg 0) (.load 1 0 .plain) .skip)
    | _ => .skip

/-- Events: 0 and 1 are the writes of thread 1, and 2 and 3 are the reads of thread 2. -/
def messagePassingData : Data 4 := Data.ofLists 4
  [.write, .write, .read, .read]
  [.plain, .rel, .acq, .plain]
  [0, 1, 1, 0]
  [1, 1, 2, 2]
  [(0, 1), (2, 3)]
  [(1, 2), (0, 3)]
  []
  []

def messagePassingLabels : Nat → Nat → Label
  | 1, 0 => ⟨.write, .plain, 0, 42, false⟩
  | 1, 1 => ⟨.write, .rel, 1, 1, false⟩
  | 2, 0 => ⟨.read, .acq, 1, 1, false⟩
  | 2, 1 => ⟨.read, .plain, 0, 42, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

theorem messagePassing_thread1 :
    Trace (messagePassing.code 1) 2 (messagePassingLabels 1) :=
  .visible (.seq .store) (.silent .seq_skip (.visible .store .stop))

theorem messagePassing_thread2 :
    Trace (messagePassing.code 2) 2 (messagePassingLabels 2) :=
  .visible (.seq (.load 1)) (.silent .seq_skip (.silent (.ite_true (by decide))
    (.visible (.load 42) .stop)))

def messagePassingRun : messagePassing.Run messagePassingData.toExecution where
  val i := [42, 1, 1, 42].getD i.val 0
  isRmw _ := false
  idx i := [0, 1, 0, 1].getD i.val 0
  lab := messagePassingLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ => messagePassing_thread1.prefix (Nat.le_succ 1)
    | ⟨1, _⟩ => messagePassing_thread1
    | ⟨2, _⟩ => messagePassing_thread2.prefix (Nat.le_succ 1)
    | ⟨3, _⟩ => messagePassing_thread2
  rf_val := by decide
  rf_write_read := by
    simp only [Execution.IsWrite, Execution.IsRead]
    decide
  rf_source := by
    simp only [Execution.IsRead]
    decide
  rmw_iff := by decide

-- The model allows this execution, and it has no data race: the flag orders the data.
#guard messagePassingData.check == some (true, false)

/-! ## Message passing through an array

The places 0 and 1 are an array, and place 2 holds a flag. Thread 1 computes the index 1 in a
register, writes 42 to that element, and then sets the flag with a release write. Thread 2 reads
the flag with an acquire read, and uses the value that it read as the index into the array. -/

def arrayPassing : Prog where
  code
    | 1 => .seq (.assign 0 (.const 1)) <|
        .seq (.storeAt 0 2 (.reg 0) (.const 42) .plain) (.store 2 (.const 1) .rel)
    | 2 => .seq (.load 0 2 .acq) (.ite (.reg 0) (.loadAt 1 0 2 (.reg 0) .plain) .skip)
    | _ => .skip

/-- Events: 0 and 1 are the writes of thread 1, and 2 and 3 are the reads of thread 2. -/
def arrayPassingData : Data 4 := Data.ofLists 4
  [.write, .write, .read, .read]
  [.plain, .rel, .acq, .plain]
  [1, 2, 2, 1]
  [1, 1, 2, 2]
  [(0, 1), (2, 3)]
  [(1, 2), (0, 3)]
  []
  []

def arrayPassingLabels : Nat → Nat → Label
  | 1, 0 => ⟨.write, .plain, 1, 42, false⟩
  | 1, 1 => ⟨.write, .rel, 2, 1, false⟩
  | 2, 0 => ⟨.read, .acq, 2, 1, false⟩
  | 2, 1 => ⟨.read, .plain, 1, 42, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

theorem arrayPassing_thread1 :
    Trace (arrayPassing.code 1) 2 (arrayPassingLabels 1) :=
  .silent (.seq .assign) <| .silent .seq_skip <|
    .visible (.seq (.storeAt 1 (by decide) (by decide))) <| .silent .seq_skip <|
    .visible .store .stop

theorem arrayPassing_thread2 :
    Trace (arrayPassing.code 2) 2 (arrayPassingLabels 2) :=
  .visible (.seq (.load 1)) <| .silent .seq_skip <| .silent (.ite_true (by decide)) <|
    .visible (.loadAt 1 42 (by decide) (by decide)) .stop

def arrayPassingRun : arrayPassing.Run arrayPassingData.toExecution where
  val i := [42, 1, 1, 42].getD i.val 0
  isRmw _ := false
  idx i := [0, 1, 0, 1].getD i.val 0
  lab := arrayPassingLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ => arrayPassing_thread1.prefix (Nat.le_succ 1)
    | ⟨1, _⟩ => arrayPassing_thread1
    | ⟨2, _⟩ => arrayPassing_thread2.prefix (Nat.le_succ 1)
    | ⟨3, _⟩ => arrayPassing_thread2
  rf_val := by decide
  rf_write_read := by
    simp only [Execution.IsWrite, Execution.IsRead]
    decide
  rf_source := by
    simp only [Execution.IsRead]
    decide
  rmw_iff := by decide

-- The model allows this execution, and it has no data race: the flag orders the element.
#guard arrayPassingData.check == some (true, false)

/-! ## Fork and join

The main code writes 5 to the place `input` and starts tasks 0 and 1. Each task reads `input`
and writes its result. The main code joins both tasks and reads the results. All accesses to
the data are plain: only the fork and the join order them. -/

def input := 0
def out1 := 1
def out2 := 2

def forkJoin : TaskProg where
  main := .seq (.store input (.const 5) .plain) <| .seq (.spawn 0) <| .seq (.spawn 1) <|
    .seq (.join 0) <| .seq (.join 1) <| .seq (.load 1 out1 .plain) (.load 2 out2 .plain)
  task
    | 0 => .seq (.load 1 input .plain) (.store out1 (.add (.reg 1) (.const 1)) .plain)
    | 1 => .seq (.load 1 input .plain) (.store out2 (.add (.reg 1) (.const 2)) .plain)
    | _ => .skip
  count := 2
  flags := 3
  marks := 3

theorem forkJoin_ok : forkJoin.Ok where
  main := by simp [forkJoin, Stmt.Below, input, out1, out2]
  task
    | 0 | 1 | _ + 2 => by simp [forkJoin, Stmt.Below, input, out1, out2]

theorem forkJoin_seqCstOnly : forkJoin.toProg.SeqCstOnly :=
  TaskProg.seqCstOnly (by simp [forkJoin, Stmt.SeqCstOnly])
    fun | 0 | 1 | _ + 2 => by simp [forkJoin, Stmt.SeqCstOnly]

/-- Events: 0 to 6 are the main code, 7 to 10 are task 0, and 11 to 14 are task 1. The flags
`start 0`, `start 1`, `done 0` and `done 1` are the places 3, 4, 5 and 6. -/
def forkJoinData : Data 15 := Data.ofLists 15
  [.write, .write, .write, .read, .read, .read, .read,
    .read, .read, .write, .write,
    .read, .read, .write, .write]
  [.plain, .sc, .sc, .sc, .sc, .plain, .plain,
    .sc, .plain, .plain, .sc,
    .sc, .plain, .plain, .sc]
  [0, 3, 4, 5, 6, 1, 2,
    3, 0, 1, 5,
    4, 0, 2, 6]
  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
  ((List.range 15).flatMap fun a => (List.range 15).filterMap fun b =>
    if a < b ∧ (b < 7 ∨ 7 ≤ a ∧ b < 11 ∨ 11 ≤ a) then some (a, b) else none)
  [(1, 7), (2, 11), (0, 8), (0, 12), (10, 3), (14, 4), (9, 5), (13, 6)]
  []
  []

def forkJoinLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .plain, input, 5, false⟩
  | 0, 1 => ⟨.write, .sc, 3, 1, false⟩
  | 0, 2 => ⟨.write, .sc, 4, 1, false⟩
  | 0, 3 => ⟨.read, .sc, 5, 1, false⟩
  | 0, 4 => ⟨.read, .sc, 6, 1, false⟩
  | 0, 5 => ⟨.read, .plain, out1, 6, false⟩
  | 0, 6 => ⟨.read, .plain, out2, 7, false⟩
  | 1, 0 => ⟨.read, .sc, 3, 1, false⟩
  | 1, 1 => ⟨.read, .plain, input, 5, false⟩
  | 1, 2 => ⟨.write, .plain, out1, 6, false⟩
  | 1, 3 => ⟨.write, .sc, 5, 1, false⟩
  | 2, 0 => ⟨.read, .sc, 4, 1, false⟩
  | 2, 1 => ⟨.read, .plain, input, 5, false⟩
  | 2, 2 => ⟨.write, .plain, out2, 7, false⟩
  | 2, 3 => ⟨.write, .sc, 6, 1, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

theorem forkJoin_main : Trace (forkJoin.toProg.code 0) 7 (forkJoinLabels 0) :=
  .visible (.seq .store) <| .silent .seq_skip <|
    .silent (.seq (.seq (.assignAt 0 rfl Nat.one_pos))) <| .silent (.seq (.seq_skip)) <|
    .visible (.seq (.storeAt 0 rfl Nat.one_pos)) <| .silent .seq_skip <|
    .silent (.seq (.seq (.assignAt 0 rfl Nat.one_pos))) <| .silent (.seq (.seq_skip)) <|
    .visible (.seq (.storeAt 0 rfl Nat.one_pos)) <| .silent .seq_skip <|
    .waitAt_seq 0 rfl Nat.one_pos <| .waitAt_seq 0 rfl Nat.one_pos <|
    .visible (.seq (.load 6)) <| .silent .seq_skip <| .visible (.load 7) .stop

theorem forkJoin_task0 : Trace (forkJoin.toProg.code 1) 4 (forkJoinLabels 1) := by
  rw [TaskProg.code_succ, TaskProg.taskCode_of_lt (by decide)]
  exact .wait_seq <| .visible (.seq (.seq (.load 5))) <| .silent (.seq .seq_skip) <|
    .visible (.seq .store) <| .silent .seq_skip <| .visible .store .stop

theorem forkJoin_task1 : Trace (forkJoin.toProg.code 2) 4 (forkJoinLabels 2) := by
  rw [TaskProg.code_succ, TaskProg.taskCode_of_lt (by decide)]
  exact .wait_seq <| .visible (.seq (.seq (.load 5))) <| .silent (.seq .seq_skip) <|
    .visible (.seq .store) <| .silent .seq_skip <| .visible .store .stop

def forkJoinRun : forkJoin.toProg.Run forkJoinData.toExecution where
  val i := [5, 1, 1, 1, 1, 6, 7, 1, 5, 6, 1, 1, 5, 7, 1].getD i.val 0
  isRmw _ := false
  idx i := [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 0, 1, 2, 3].getD i.val 0
  lab := forkJoinLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ | ⟨1, _⟩ | ⟨2, _⟩ | ⟨3, _⟩ | ⟨4, _⟩ | ⟨5, _⟩ =>
      forkJoin_main.prefix (by decide +revert)
    | ⟨6, _⟩ => forkJoin_main
    | ⟨7, _⟩ | ⟨8, _⟩ | ⟨9, _⟩ => forkJoin_task0.prefix (by decide +revert)
    | ⟨10, _⟩ => forkJoin_task0
    | ⟨11, _⟩ | ⟨12, _⟩ | ⟨13, _⟩ => forkJoin_task1.prefix (by decide +revert)
    | ⟨14, _⟩ => forkJoin_task1
  rf_val := by decide
  rf_write_read := by
    simp only [Execution.IsWrite, Execution.IsRead]
    decide
  rf_source := by
    simp only [Execution.IsRead]
    decide
  rmw_iff := by decide

-- The model allows this execution, and it has no data race: the fork orders the write of
-- `input` before the reads, and the join orders the writes of the results before the reads.
#guard forkJoinData.check == some (true, false)

/-! ## A task that starts a task

The main code writes 5 to the place 0 and starts task 0. Task 0 starts task 1 and joins it. Task 1
reads the place 0 and writes 6 to the place 1. The main code joins task 0 and reads the place 1.
Only the forks and the joins order the plain accesses, through two levels of the tree. -/

def nested : TaskProg where
  main := .seq (.store 0 (.const 5) .plain) <| .seq (.spawn 0) <| .seq (.join 0) (.load 1 1 .plain)
  task
    | 0 => .seq (.spawn 1) (.join 1)
    | 1 => .seq (.load 1 0 .plain) (.store 1 (.add (.reg 1) (.const 1)) .plain)
    | _ => .skip
  count := 2
  flags := 2
  marks := 3

theorem nested_ok : nested.Ok where
  main := by simp [nested, Stmt.Below]
  task
    | 0 | 1 | _ + 2 => by simp [nested, Stmt.Below]

/-- Events: 0 to 3 are the main code, 4 to 7 are task 0, and 8 to 11 are task 1. The flags
`start 0`, `start 1`, `done 0` and `done 1` are the places 2, 3, 4 and 5. -/
def nestedData : Data 12 := Data.ofLists 12
  [.write, .write, .read, .read,
    .read, .write, .read, .write,
    .read, .read, .write, .write]
  [.plain, .sc, .sc, .plain,
    .sc, .sc, .sc, .sc,
    .sc, .plain, .plain, .sc]
  [0, 2, 4, 1,
    2, 3, 5, 4,
    3, 0, 1, 5]
  [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
  ((List.range 12).flatMap fun a => (List.range 12).filterMap fun b =>
    if a < b ∧ a / 4 = b / 4 then some (a, b) else none)
  [(1, 4), (5, 8), (11, 6), (7, 2), (0, 9), (10, 3)]
  []
  []

def nestedLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .plain, 0, 5, false⟩
  | 0, 1 => ⟨.write, .sc, 2, 1, false⟩
  | 0, 2 => ⟨.read, .sc, 4, 1, false⟩
  | 0, 3 => ⟨.read, .plain, 1, 6, false⟩
  | 1, 0 => ⟨.read, .sc, 2, 1, false⟩
  | 1, 1 => ⟨.write, .sc, 3, 1, false⟩
  | 1, 2 => ⟨.read, .sc, 5, 1, false⟩
  | 1, 3 => ⟨.write, .sc, 4, 1, false⟩
  | 2, 0 => ⟨.read, .sc, 3, 1, false⟩
  | 2, 1 => ⟨.read, .plain, 0, 5, false⟩
  | 2, 2 => ⟨.write, .plain, 1, 6, false⟩
  | 2, 3 => ⟨.write, .sc, 5, 1, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

theorem nested_main : Trace (nested.toProg.code 0) 4 (nestedLabels 0) :=
  .visible (.seq .store) <| .silent .seq_skip <|
    .silent (.seq (.seq (.assignAt 0 rfl Nat.one_pos))) <| .silent (.seq (.seq_skip)) <|
    .visible (.seq (.storeAt 0 rfl Nat.one_pos)) <| .silent .seq_skip <|
    .waitAt_seq 0 rfl Nat.one_pos <| .visible (.load 6) .stop

theorem nested_task0 : Trace (nested.toProg.code 1) 4 (nestedLabels 1) := by
  rw [TaskProg.code_succ, TaskProg.taskCode_of_lt (by decide)]
  exact .wait_seq <| .silent (.seq (.seq (.seq (.assignAt 0 rfl Nat.one_pos)))) <| .silent (.seq (.seq (.seq_skip))) <|
    .visible (.seq (.seq (.storeAt 0 rfl Nat.one_pos))) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.waitAt 0 1 rfl Nat.one_pos)) <|
    .silent .seq_skip <| .visible .store .stop

theorem nested_task1 : Trace (nested.toProg.code 2) 4 (nestedLabels 2) := by
  rw [TaskProg.code_succ, TaskProg.taskCode_of_lt (by decide)]
  exact .wait_seq <| .visible (.seq (.seq (.load 5))) <| .silent (.seq .seq_skip) <|
    .visible (.seq .store) <| .silent .seq_skip <| .visible .store .stop

def nestedRun : nested.toProg.Run nestedData.toExecution where
  val i := [5, 1, 1, 6, 1, 1, 1, 1, 1, 5, 6, 1].getD i.val 0
  isRmw _ := false
  idx i := [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3].getD i.val 0
  lab := nestedLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ | ⟨1, _⟩ | ⟨2, _⟩ => nested_main.prefix (by decide +revert)
    | ⟨3, _⟩ => nested_main
    | ⟨4, _⟩ | ⟨5, _⟩ | ⟨6, _⟩ => nested_task0.prefix (by decide +revert)
    | ⟨7, _⟩ => nested_task0
    | ⟨8, _⟩ | ⟨9, _⟩ | ⟨10, _⟩ => nested_task1.prefix (by decide +revert)
    | ⟨11, _⟩ => nested_task1
  rf_val := by decide
  rf_write_read := by
    simp only [Execution.IsWrite, Execution.IsRead]
    decide
  rf_source := by
    simp only [Execution.IsRead]
    decide
  rmw_iff := by decide

-- The model allows this execution, and it has no data race: the write of the place 0 happens
-- before the read in task 1 through both forks, and the write of the place 1 happens before the
-- read in the main code through both joins.
#guard nestedData.check == some (true, false)

/-! ## A spawn in a loop

The main code runs a loop of two passes. Pass `i` starts task `i` and joins it, so each pass gets
a new task id from the counter in register 0. Task `i` writes `10 + i` to the place `i`. After the
loop, the main code reads both places. -/

def spawnLoop : TaskProg where
  main := .seq
    (.loop (.sub (.const 1) (.eq (.reg 0) (.const 2))) <|
      .seq (.spawnAt 0 2 (.reg 0)) <| .seq (.joinAt 0 2 (.reg 0)) <|
        .assign 0 (.add (.reg 0) (.const 1)))
    (.seq (.load 1 0 .plain) (.load 2 1 .plain))
  task
    | 0 => .store 0 (.const 10) .plain
    | 1 => .store 1 (.const 11) .plain
    | _ => .skip
  count := 2
  flags := 2
  marks := 3

theorem spawnLoop_ok : spawnLoop.Ok where
  main := by simp [spawnLoop, Stmt.Below]
  task
    | 0 | 1 | _ + 2 => by simp [spawnLoop, Stmt.Below]

/-- Events: 0 to 5 are the main code, 6 to 8 are task 0, and 9 to 11 are task 1. The flags
`start 0`, `start 1`, `done 0` and `done 1` are the places 2, 3, 4 and 5. -/
def spawnLoopData : Data 12 := Data.ofLists 12
  [.write, .read, .write, .read, .read, .read,
    .read, .write, .write,
    .read, .write, .write]
  [.sc, .sc, .sc, .sc, .plain, .plain,
    .sc, .plain, .sc,
    .sc, .plain, .sc]
  [2, 4, 3, 5, 0, 1,
    2, 0, 4,
    3, 1, 5]
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
  ((List.range 12).flatMap fun a => (List.range 12).filterMap fun b =>
    if a < b ∧ (b < 6 ∨ 6 ≤ a ∧ b < 9 ∨ 9 ≤ a) then some (a, b) else none)
  [(0, 6), (8, 1), (2, 9), (11, 3), (7, 4), (10, 5)]
  []
  []

def spawnLoopLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .sc, 2, 1, false⟩
  | 0, 1 => ⟨.read, .sc, 4, 1, false⟩
  | 0, 2 => ⟨.write, .sc, 3, 1, false⟩
  | 0, 3 => ⟨.read, .sc, 5, 1, false⟩
  | 0, 4 => ⟨.read, .plain, 0, 10, false⟩
  | 0, 5 => ⟨.read, .plain, 1, 11, false⟩
  | 1, 0 => ⟨.read, .sc, 2, 1, false⟩
  | 1, 1 => ⟨.write, .plain, 0, 10, false⟩
  | 1, 2 => ⟨.write, .sc, 4, 1, false⟩
  | 2, 0 => ⟨.read, .sc, 3, 1, false⟩
  | 2, 1 => ⟨.write, .plain, 1, 11, false⟩
  | 2, 2 => ⟨.write, .sc, 5, 1, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

/-- The two passes of the loop, and then the two reads. -/
theorem spawnLoop_main : Trace (spawnLoop.toProg.code 0) 6 (spawnLoopLabels 0) :=
  .silent (.seq .loop) <| .silent (.seq (.ite_true (by decide))) <|
    .silent (.seq (.seq (.seq (.seq (.assignAt 0 rfl (by decide)))))) <| .silent (.seq (.seq (.seq (.seq_skip)))) <|
    .visible (.seq (.seq (.seq (.storeAt 0 rfl (by decide))))) <|
    .silent (.seq (.seq .seq_skip)) <|
    .visible (.seq (.seq (.seq (.waitAt 0 1 rfl (by decide))))) <|
    .silent (.seq (.seq .seq_skip)) <| .silent (.seq (.seq .assign)) <| .silent (.seq .seq_skip) <|
    .silent (.seq .loop) <| .silent (.seq (.ite_true (by decide))) <|
    .silent (.seq (.seq (.seq (.seq (.assignAt 1 rfl (by decide)))))) <| .silent (.seq (.seq (.seq (.seq_skip)))) <|
    .visible (.seq (.seq (.seq (.storeAt 1 rfl (by decide))))) <|
    .silent (.seq (.seq .seq_skip)) <|
    .visible (.seq (.seq (.seq (.waitAt 1 1 rfl (by decide))))) <|
    .silent (.seq (.seq .seq_skip)) <| .silent (.seq (.seq .assign)) <| .silent (.seq .seq_skip) <|
    .silent (.seq .loop) <| .silent (.seq (.ite_false (by decide))) <| .silent .seq_skip <|
    .visible (.seq (.load 10)) <| .silent .seq_skip <| .visible (.load 11) .stop

theorem spawnLoop_task0 : Trace (spawnLoop.toProg.code 1) 3 (spawnLoopLabels 1) := by
  rw [TaskProg.code_succ, TaskProg.taskCode_of_lt (by decide)]
  exact .wait_seq <| .visible (.seq .store) <| .silent .seq_skip <| .visible .store .stop

theorem spawnLoop_task1 : Trace (spawnLoop.toProg.code 2) 3 (spawnLoopLabels 2) := by
  rw [TaskProg.code_succ, TaskProg.taskCode_of_lt (by decide)]
  exact .wait_seq <| .visible (.seq .store) <| .silent .seq_skip <| .visible .store .stop

def spawnLoopRun : spawnLoop.toProg.Run spawnLoopData.toExecution where
  val i := [1, 1, 1, 1, 10, 11, 1, 10, 1, 1, 11, 1].getD i.val 0
  isRmw _ := false
  idx i := [0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2].getD i.val 0
  lab := spawnLoopLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ | ⟨1, _⟩ | ⟨2, _⟩ | ⟨3, _⟩ | ⟨4, _⟩ => spawnLoop_main.prefix (by decide +revert)
    | ⟨5, _⟩ => spawnLoop_main
    | ⟨6, _⟩ | ⟨7, _⟩ => spawnLoop_task0.prefix (by decide +revert)
    | ⟨8, _⟩ => spawnLoop_task0
    | ⟨9, _⟩ | ⟨10, _⟩ => spawnLoop_task1.prefix (by decide +revert)
    | ⟨11, _⟩ => spawnLoop_task1
  rf_val := by decide
  rf_write_read := by
    simp only [Execution.IsWrite, Execution.IsRead]
    decide
  rf_source := by
    simp only [Execution.IsRead]
    decide
  rmw_iff := by decide

-- The model allows this execution, and it has no data race: the join of each pass orders the
-- write of the task before the reads after the loop.
#guard spawnLoopData.check == some (true, false)

end Vx
