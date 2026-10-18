import VxMemoryModel.Language
import VxMemoryModel.Check

/-!
# An execution of message passing

This file shows that `Prog.Run` is not empty for real programs, so `Prog.drf_sc` is not true
only because no execution qualifies. Place 0 holds the data and place 1, which is atomic, holds
a flag. Thread 1 writes the data and then sets the flag with a release write. Thread 2 reads
the flag with an acquire read, and if it is set, reads the data. Thread 0 writes the initial
values.

In the execution below, thread 2 sees the flag and reads 42.
-/

namespace Vx

def messagePassing : Prog where
  code
    | 0 => .seq (.store 0 (.const 0) .plain) (.store 1 (.const 0) .rel)
    | 1 => .seq (.store 0 (.const 42) .plain) (.store 1 (.const 1) .rel)
    | 2 => .seq (.load 0 1 .acq) (.ite (.reg 0) (.load 1 0 .plain) .skip)
    | _ => .skip


/-- Events: 0 and 1 are the initial writes, 2 and 3 are the writes of thread 1, and 4 and 5
are the reads of thread 2. -/
def messagePassingData : Data 6 := Data.ofLists 6
  [.write, .write, .write, .write, .read, .read]
  [.plain, .rel, .plain, .rel, .acq, .plain]
  [0, 1, 0, 1, 1, 0]
  [0, 0, 1, 1, 2, 2]
  [(0, 1), (2, 3), (4, 5), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5)]
  [(3, 4), (2, 5)]
  [(0, 2), (1, 3)]
  []

def messagePassingLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .plain, 0, 0, false⟩
  | 0, 1 => ⟨.write, .rel, 1, 0, false⟩
  | 1, 0 => ⟨.write, .plain, 0, 42, false⟩
  | 1, 1 => ⟨.write, .rel, 1, 1, false⟩
  | 2, 0 => ⟨.read, .acq, 1, 1, false⟩
  | 2, 1 => ⟨.read, .plain, 0, 42, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

theorem messagePassing_thread0 :
    Trace (messagePassing.code 0) 2 (messagePassingLabels 0) :=
  .visible (.seq .store) (.silent .seq_skip (.visible .store .stop))

theorem messagePassing_thread1 :
    Trace (messagePassing.code 1) 2 (messagePassingLabels 1) :=
  .visible (.seq .store) (.silent .seq_skip (.visible .store .stop))

theorem messagePassing_thread2 :
    Trace (messagePassing.code 2) 2 (messagePassingLabels 2) :=
  .visible (.seq (.load 1)) (.silent .seq_skip (.silent (.ite_true (by decide))
    (.visible (.load 42) .stop)))

def messagePassingRun : messagePassing.Run messagePassingData.toExecution where
  val i := [0, 0, 42, 1, 1, 42].getD i.val 0
  isRmw _ := false
  idx i := [0, 1, 0, 1, 0, 1].getD i.val 0
  lab := messagePassingLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ => messagePassing_thread0.prefix (Nat.le_succ 1)
    | ⟨1, _⟩ => messagePassing_thread0
    | ⟨2, _⟩ => messagePassing_thread1.prefix (Nat.le_succ 1)
    | ⟨3, _⟩ => messagePassing_thread1
    | ⟨4, _⟩ => messagePassing_thread2.prefix (Nat.le_succ 1)
    | ⟨5, _⟩ => messagePassing_thread2
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

/-! ## Two threads add 1 to one counter

Place 0 is an atomic counter. Threads 1 and 2 each add 1 with a sequentially consistent
`fetchAdd`. In the execution below, thread 1 goes first and thread 2 reads its write. The
program follows `Prog.SeqCstOnly`, so `Prog.drf_sc` applies to it. -/

def counter : Prog where
  code
    | 0 => .store 0 (.const 0) .sc
    | 1 | 2 => .fetchAdd 0 0 (.const 1) .sc
    | _ => .skip

theorem counter_seqCstOnly : counter.SeqCstOnly
  | 0 | 1 | 2 | _ + 3 => by simp [counter, Stmt.SeqCstOnly]

/-- Events: 0 is the initial write. 1 and 2 are the read and the write of thread 1, and 3 and 4
those of thread 2. -/
def counterData : Data 5 := Data.ofLists 5
  [.write, .read, .write, .read, .write]
  [.sc, .sc, .sc, .sc, .sc]
  [0, 0, 0, 0, 0]
  [0, 1, 1, 2, 2]
  [(1, 2), (3, 4), (0, 1), (0, 2), (0, 3), (0, 4)]
  [(0, 1), (2, 3)]
  [(0, 2), (0, 4), (2, 4)]
  [(1, 2), (3, 4)]

def counterLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .sc, 0, 0, false⟩
  | 1, 0 => ⟨.read, .sc, 0, 0, false⟩
  | 1, 1 => ⟨.write, .sc, 0, 1, true⟩
  | 2, 0 => ⟨.read, .sc, 0, 1, false⟩
  | 2, 1 => ⟨.write, .sc, 0, 2, true⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

theorem counter_thread0 : Trace (counter.code 0) 1 (counterLabels 0) := .visible .store .stop

theorem counter_thread1 : Trace (counter.code 1) 2 (counterLabels 1) :=
  .visible (.fetchAdd 0) (.visible .rmwStore .stop)

theorem counter_thread2 : Trace (counter.code 2) 2 (counterLabels 2) :=
  .visible (.fetchAdd 1) (.visible .rmwStore .stop)

def counterRun : counter.Run counterData.toExecution where
  val i := [0, 0, 1, 1, 2].getD i.val 0
  isRmw i := [false, false, true, false, true].getD i.val false
  idx i := [0, 0, 1, 0, 1].getD i.val 0
  lab := counterLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ => counter_thread0
    | ⟨1, _⟩ => counter_thread1.prefix (Nat.le_succ 1)
    | ⟨2, _⟩ => counter_thread1
    | ⟨3, _⟩ => counter_thread2.prefix (Nat.le_succ 1)
    | ⟨4, _⟩ => counter_thread2
  rf_val := by decide
  rf_write_read := by
    simp only [Execution.IsWrite, Execution.IsRead]
    decide
  rf_source := by
    simp only [Execution.IsRead]
    decide
  rmw_iff := by decide

-- The model allows this execution, and it has no data race: all accesses are atomic.
#guard counterData.check == some (true, false)

end Vx
