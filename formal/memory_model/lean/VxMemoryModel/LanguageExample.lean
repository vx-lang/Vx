import VxMemoryModel.Language
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

Thread 0 writes 5 to the place `input`, and starts threads 1 and 2 with the flag `start`. Each
thread reads `input`, writes its result, and sets its flag `done`. Thread 0 waits for both
flags and reads the results. All accesses to the data are plain: only the fork and the join
order them. -/

def input := 0
def start := 1
def done1 := 2
def done2 := 3
def out1 := 4
def out2 := 5

def forkJoin : Prog where
  code
    | 0 => .seq (.store input (.const 5) .plain) <| .seq (.signal start) <|
        .seq (.await 0 done1) <| .seq (.await 0 done2) <|
        .seq (.load 1 out1 .plain) (.load 2 out2 .plain)
    | 1 => .seq (.await 0 start) <| .seq (.load 1 input .plain) <|
        .seq (.store out1 (.add (.reg 1) (.const 1)) .plain) (.signal done1)
    | 2 => .seq (.await 0 start) <| .seq (.load 1 input .plain) <|
        .seq (.store out2 (.add (.reg 1) (.const 2)) .plain) (.signal done2)
    | _ => .skip

theorem forkJoin_seqCstOnly : forkJoin.SeqCstOnly
  | 0 | 1 | 2 | _ + 3 => by simp [forkJoin, Stmt.SeqCstOnly]

/-- Events: 0 to 5 are thread 0, 6 to 9 are thread 1, and 10 to 13 are thread 2. -/
def forkJoinData : Data 14 := Data.ofLists 14
  [.write, .write, .read, .read, .read, .read,
    .read, .read, .write, .write,
    .read, .read, .write, .write]
  [.plain, .sc, .sc, .sc, .plain, .plain,
    .sc, .plain, .plain, .sc,
    .sc, .plain, .plain, .sc]
  [0, 1, 2, 3, 4, 5,
    1, 0, 4, 2,
    1, 0, 5, 3]
  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
  ((List.range 14).flatMap fun a => (List.range 14).filterMap fun b =>
    if a < b ∧ (b < 6 ∨ 6 ≤ a ∧ b < 10 ∨ 10 ≤ a) then some (a, b) else none)
  [(1, 6), (1, 10), (0, 7), (0, 11), (9, 2), (13, 3), (8, 4), (12, 5)]
  []
  []

def forkJoinLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .plain, input, 5, false⟩
  | 0, 1 => ⟨.write, .sc, start, 1, false⟩
  | 0, 2 => ⟨.read, .sc, done1, 1, false⟩
  | 0, 3 => ⟨.read, .sc, done2, 1, false⟩
  | 0, 4 => ⟨.read, .plain, out1, 6, false⟩
  | 0, 5 => ⟨.read, .plain, out2, 7, false⟩
  | 1, 0 => ⟨.read, .sc, start, 1, false⟩
  | 1, 1 => ⟨.read, .plain, input, 5, false⟩
  | 1, 2 => ⟨.write, .plain, out1, 6, false⟩
  | 1, 3 => ⟨.write, .sc, done1, 1, false⟩
  | 2, 0 => ⟨.read, .sc, start, 1, false⟩
  | 2, 1 => ⟨.read, .plain, input, 5, false⟩
  | 2, 2 => ⟨.write, .plain, out2, 7, false⟩
  | 2, 3 => ⟨.write, .sc, done2, 1, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

theorem forkJoin_thread0 : Trace (forkJoin.code 0) 6 (forkJoinLabels 0) :=
  .visible (.seq .store) <| .silent .seq_skip <| .visible (.seq .store) <| .silent .seq_skip <|
    .await_seq <| .await_seq <| .visible (.seq (.load 6)) <| .silent .seq_skip <|
    .visible (.load 7) .stop

theorem forkJoin_thread1 : Trace (forkJoin.code 1) 4 (forkJoinLabels 1) :=
  .await_seq <| .visible (.seq (.load 5)) <| .silent .seq_skip <| .visible (.seq .store) <|
    .silent .seq_skip <| .visible .store .stop

theorem forkJoin_thread2 : Trace (forkJoin.code 2) 4 (forkJoinLabels 2) :=
  .await_seq <| .visible (.seq (.load 5)) <| .silent .seq_skip <| .visible (.seq .store) <|
    .silent .seq_skip <| .visible .store .stop

def forkJoinRun : forkJoin.Run forkJoinData.toExecution where
  val i := [5, 1, 1, 1, 6, 7, 1, 5, 6, 1, 1, 5, 7, 1].getD i.val 0
  isRmw _ := false
  idx i := [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 3].getD i.val 0
  lab := forkJoinLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ => forkJoin_thread0.prefix (by decide +revert)
    | ⟨1, _⟩ => forkJoin_thread0.prefix (by decide +revert)
    | ⟨2, _⟩ => forkJoin_thread0.prefix (by decide +revert)
    | ⟨3, _⟩ => forkJoin_thread0.prefix (by decide +revert)
    | ⟨4, _⟩ => forkJoin_thread0.prefix (by decide +revert)
    | ⟨5, _⟩ => forkJoin_thread0
    | ⟨6, _⟩ => forkJoin_thread1.prefix (by decide +revert)
    | ⟨7, _⟩ => forkJoin_thread1.prefix (by decide +revert)
    | ⟨8, _⟩ => forkJoin_thread1.prefix (by decide +revert)
    | ⟨9, _⟩ => forkJoin_thread1
    | ⟨10, _⟩ => forkJoin_thread2.prefix (by decide +revert)
    | ⟨11, _⟩ => forkJoin_thread2.prefix (by decide +revert)
    | ⟨12, _⟩ => forkJoin_thread2.prefix (by decide +revert)
    | ⟨13, _⟩ => forkJoin_thread2
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
