import VxMemoryModel.Language
import VxMemoryModel.Check

/-!
# An execution of message passing

This file shows that `Prog.Run` is not empty for real programs, so `Prog.drf_sc` is not true
only because no execution qualifies. Place 0 holds the data and place 1, which is atomic, holds
a flag. Thread 1 writes the data and then sets the flag. Thread 2 reads the flag, and if it is
set, reads the data. Thread 0 writes the initial values.

In the execution below, thread 2 sees the flag and reads 42.
-/

namespace Vx

def messagePassing : Prog where
  atomic x := x == 1
  code
    | 0 => .seq (.store 0 (.const 0)) (.store 1 (.const 0))
    | 1 => .seq (.store 0 (.const 42)) (.store 1 (.const 1))
    | 2 => .seq (.load 0 1) (.ite (.reg 0) (.load 1 0) .skip)
    | _ => .skip

/-- Events: 0 and 1 are the initial writes, 2 and 3 are the writes of thread 1, and 4 and 5
are the reads of thread 2. -/
def messagePassingData : Data 6 := Data.ofLists 6
  [.write, .write, .write, .write, .read, .read]
  [.plain, .sc, .plain, .sc, .sc, .plain]
  [0, 1, 0, 1, 1, 0]
  [0, 0, 1, 1, 2, 2]
  [(0, 1), (2, 3), (4, 5), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), (1, 5)]
  [(3, 4), (2, 5)]
  [(0, 2), (1, 3)]
  []

def messagePassingLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .plain, 0, 0⟩
  | 0, 1 => ⟨.write, .sc, 1, 0⟩
  | 1, 0 => ⟨.write, .plain, 0, 42⟩
  | 1, 1 => ⟨.write, .sc, 1, 1⟩
  | 2, 0 => ⟨.read, .sc, 1, 1⟩
  | 2, 1 => ⟨.read, .plain, 0, 42⟩
  | _, _ => ⟨.fence, .plain, 0, 0⟩

theorem messagePassing_thread0 :
    Trace messagePassing.atomic (messagePassing.code 0) 2 (messagePassingLabels 0) :=
  .visible (.seq .store) (.silent .seq_skip (.visible .store .stop))

theorem messagePassing_thread1 :
    Trace messagePassing.atomic (messagePassing.code 1) 2 (messagePassingLabels 1) :=
  .visible (.seq .store) (.silent .seq_skip (.visible .store .stop))

theorem messagePassing_thread2 :
    Trace messagePassing.atomic (messagePassing.code 2) 2 (messagePassingLabels 2) :=
  .visible (.seq (.load 1)) (.silent .seq_skip (.silent (.ite_true (by decide))
    (.visible (.load 42) .stop)))

def messagePassingRun : messagePassing.Run messagePassingData.toExecution where
  val i := [0, 0, 42, 1, 1, 42].getD i.val 0
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
  no_rmw := by decide

-- The model allows this execution, and it has no data race: the flag orders the data.
#guard messagePassingData.check == some (true, false)

end Vx
