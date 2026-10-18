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
  flags := 3

theorem forkJoin_ok : forkJoin.Ok where
  main := by simp [forkJoin, Stmt.Below, input, out1, out2]
  task
    | 0 | 1 | _ + 2 => by simp [forkJoin, Stmt.Below, input, out1, out2]

theorem forkJoin_seqCstOnly : forkJoin.toProg.SeqCstOnly :=
  TaskProg.seqCstOnly (by simp [forkJoin, Stmt.SeqCstOnly])
    fun | 0 | 1 | _ + 2 => by simp [forkJoin, Stmt.SeqCstOnly]

/-- Events: 0 to 6 are the main code, 7 to 10 are task 0, and 11 to 14 are task 1. The flags
`start 0`, `done 0`, `start 1` and `done 1` are the places 3, 4, 5 and 6. -/
def forkJoinData : Data 15 := Data.ofLists 15
  [.write, .write, .write, .read, .read, .read, .read,
    .read, .read, .write, .write,
    .read, .read, .write, .write]
  [.plain, .sc, .sc, .sc, .sc, .plain, .plain,
    .sc, .plain, .plain, .sc,
    .sc, .plain, .plain, .sc]
  [0, 3, 5, 4, 6, 1, 2,
    3, 0, 1, 4,
    5, 0, 2, 6]
  [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
  ((List.range 15).flatMap fun a => (List.range 15).filterMap fun b =>
    if a < b ∧ (b < 7 ∨ 7 ≤ a ∧ b < 11 ∨ 11 ≤ a) then some (a, b) else none)
  [(1, 7), (2, 11), (0, 8), (0, 12), (10, 3), (14, 4), (9, 5), (13, 6)]
  []
  []

def forkJoinLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .plain, input, 5, false⟩
  | 0, 1 => ⟨.write, .sc, 3, 1, false⟩
  | 0, 2 => ⟨.write, .sc, 5, 1, false⟩
  | 0, 3 => ⟨.read, .sc, 4, 1, false⟩
  | 0, 4 => ⟨.read, .sc, 6, 1, false⟩
  | 0, 5 => ⟨.read, .plain, out1, 6, false⟩
  | 0, 6 => ⟨.read, .plain, out2, 7, false⟩
  | 1, 0 => ⟨.read, .sc, 3, 1, false⟩
  | 1, 1 => ⟨.read, .plain, input, 5, false⟩
  | 1, 2 => ⟨.write, .plain, out1, 6, false⟩
  | 1, 3 => ⟨.write, .sc, 4, 1, false⟩
  | 2, 0 => ⟨.read, .sc, 5, 1, false⟩
  | 2, 1 => ⟨.read, .plain, input, 5, false⟩
  | 2, 2 => ⟨.write, .plain, out2, 7, false⟩
  | 2, 3 => ⟨.write, .sc, 6, 1, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

theorem forkJoin_main : Trace (forkJoin.toProg.code 0) 7 (forkJoinLabels 0) :=
  .visible (.seq .store) <| .silent .seq_skip <| .visible (.seq .store) <| .silent .seq_skip <|
    .visible (.seq .store) <| .silent .seq_skip <| .wait_seq <| .wait_seq <|
    .visible (.seq (.load 6)) <| .silent .seq_skip <| .visible (.load 7) .stop

theorem forkJoin_task0 : Trace (forkJoin.toProg.code 1) 4 (forkJoinLabels 1) :=
  .wait_seq <| .visible (.seq (.seq (.load 5))) <| .silent (.seq .seq_skip) <|
    .visible (.seq .store) <| .silent .seq_skip <| .visible .store .stop

theorem forkJoin_task1 : Trace (forkJoin.toProg.code 2) 4 (forkJoinLabels 2) :=
  .wait_seq <| .visible (.seq (.seq (.load 5))) <| .silent (.seq .seq_skip) <|
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

end Vx
