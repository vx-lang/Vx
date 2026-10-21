import VxMemoryModel.Channel
import VxMemoryModel.Check

/-!
# An example of a lock

The main code starts task 0. Then it takes lock 0, writes 7 to the place 0 with a plain write,
and lets the lock go. Task 0 takes the lock, reads the place 0 with a plain read, and lets the
lock go. In the run below, the main code gets ticket 0 and the task gets ticket 1. The task reads
`serving` as 0 and waits, then reads it as 1 and reads 7. The `unlock` of the main code happens
before the read, so the two plain accesses do not race.
-/

namespace Vx

def oneLock : ChanProg where
  main := .seq (.base (.spawn 0))
    (.seq (.locked 0 (.base (.store 0 (.const 7) .plain))) (.base (.join 0)))
  task
    | 0 => .locked 0 (.base (.load 5 0 .plain))
    | _ => .base .skip
  count := 1
  places := 1
  chans := 0
  locks := 1
  cap _ := 0
  senders _ := 0
  msgs := 0
  tmp := 0

theorem oneLock_ok : oneLock.Ok where
  main := by simp [oneLock, ChanProg.StmtOk, Stmt.Below, CStmt.Locks]
  task
    | 0 => by simp [oneLock, ChanProg.StmtOk, Stmt.Below, CStmt.Locks]
    | _ + 1 => by simp [oneLock, ChanProg.StmtOk, Stmt.Below]
  cap _ h := absurd h (by simp [oneLock])

/-- `next` and `serving` of lock 0 are the places 1 and 2. The flags `start 0` and `done 0` of the
task are 3 and 4. -/
theorem oneLock_initList : oneLock.initList = [(1, 0), (2, 0)] := by decide

/-- Events: 0 to 9 are the main code, and 10 to 18 are task 0. -/
def oneLockData : Data 19 := Data.ofLists 19
  [.write, .write, .write, .read, .write, .read, .write, .read, .write, .read,
    .read, .read, .write, .read, .read, .read, .read, .write, .write]
  [.sc, .sc, .sc, .sc, .sc, .sc, .plain, .sc, .sc, .sc,
    .sc, .sc, .sc, .sc, .sc, .plain, .sc, .sc, .sc]
  [1, 2, 3, 1, 1, 2, 0, 2, 2, 4,
    3, 1, 1, 2, 2, 0, 2, 2, 4]
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  ((List.range 19).flatMap fun a => (List.range 19).filterMap fun b =>
    if a < b ∧ (b < 10 ∨ 10 ≤ a) then some (a, b) else none)
  [(0, 3), (1, 5), (1, 7), (2, 10), (4, 11), (1, 13), (8, 14), (6, 15), (8, 16), (18, 9)]
  [(0, 4), (0, 12), (4, 12), (1, 8), (1, 17), (8, 17)]
  [(3, 4), (7, 8), (11, 12), (16, 17)]

def oneLockLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .sc, 1, 0, false⟩
  | 0, 1 => ⟨.write, .sc, 2, 0, false⟩
  | 0, 2 => ⟨.write, .sc, 3, 1, false⟩
  | 0, 3 => ⟨.read, .sc, 1, 0, false⟩
  | 0, 4 => ⟨.write, .sc, 1, 1, true⟩
  | 0, 5 => ⟨.read, .sc, 2, 0, false⟩
  | 0, 6 => ⟨.write, .plain, 0, 7, false⟩
  | 0, 7 => ⟨.read, .sc, 2, 0, false⟩
  | 0, 8 => ⟨.write, .sc, 2, 1, true⟩
  | 0, 9 => ⟨.read, .sc, 4, 1, false⟩
  | 1, 0 => ⟨.read, .sc, 3, 1, false⟩
  | 1, 1 => ⟨.read, .sc, 1, 1, false⟩
  | 1, 2 => ⟨.write, .sc, 1, 2, true⟩
  | 1, 3 => ⟨.read, .sc, 2, 0, false⟩
  | 1, 4 => ⟨.read, .sc, 2, 1, false⟩
  | 1, 5 => ⟨.read, .plain, 0, 7, false⟩
  | 1, 6 => ⟨.read, .sc, 2, 1, false⟩
  | 1, 7 => ⟨.write, .sc, 2, 2, true⟩
  | 1, 8 => ⟨.write, .sc, 4, 1, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

/-- The writes at the start, the `spawn`, the `lock` with ticket 0, the write, the `unlock` and the
`join`. `serving` is 0 at the first read, so the `lock` does not wait. -/
theorem oneLock_main : Trace (oneLock.toTaskProg.toProg.code 0) 10 (oneLockLabels 0) := by
  rw [ChanProg.code_zero, oneLock_initList]
  exact .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <| .silent .seq_skip <|
    .silent (.seq (.seq (.assignAt 0 rfl Nat.one_pos))) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.storeAt 0 rfl Nat.one_pos)) <| .silent .seq_skip <|
    .visible (.seq (.seq (.seq (.fetchAdd 0)))) <| .visible (.seq (.seq (.seq .rmwStore))) <|
    .silent (.seq (.seq .seq_skip)) <| .visible (.seq (.seq (.seq (.load 0)))) <|
    .silent (.seq (.seq .seq_skip)) <| .silent (.seq (.seq .loop)) <|
    .silent (.seq (.seq (.ite_false (by decide)))) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.fetchAdd 0)) <| .visible (.seq .rmwStore) <| .silent .seq_skip <|
    .visible (.waitAt 0 1 rfl Nat.one_pos) .stop

/-- The task waits for `start 0`, and takes ticket 1. It reads `serving` as 0 and waits again, then
reads it as 1. It reads 7, lets the lock go, and sets `done 0`. -/
theorem oneLock_task : Trace (oneLock.toTaskProg.toProg.code 1) 9 (oneLockLabels 1) := by
  rw [TaskProg.code_succ, TaskProg.taskCode_of_lt (by decide)]
  exact .wait_seq <|
    .visible (.seq (.seq (.seq (.fetchAdd 1)))) <| .visible (.seq (.seq (.seq .rmwStore))) <|
    .silent (.seq (.seq .seq_skip)) <|
    .visible (.seq (.seq (.seq (.load 0)))) <| .silent (.seq (.seq .seq_skip)) <|
    .silent (.seq (.seq .loop)) <| .silent (.seq (.seq (.ite_true (by decide)))) <|
    .visible (.seq (.seq (.seq (.load 1)))) <| .silent (.seq (.seq .seq_skip)) <|
    .silent (.seq (.seq .loop)) <| .silent (.seq (.seq (.ite_false (by decide)))) <|
    .silent (.seq .seq_skip) <|
    .visible (.seq (.seq (.load 7))) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.fetchAdd 1)) <| .visible (.seq .rmwStore) <| .silent .seq_skip <|
    .visible .store .stop

def oneLockRun : oneLock.toTaskProg.toProg.Run oneLockData.toExecution where
  val i := [0, 0, 1, 0, 1, 0, 7, 0, 1, 1, 1, 1, 2, 0, 1, 7, 1, 2, 1].getD i.val 0
  isRmw i := [4, 8, 12, 17].contains i.val
  idx i := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8].getD i.val 0
  lab := oneLockLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ | ⟨1, _⟩ | ⟨2, _⟩ | ⟨3, _⟩ | ⟨4, _⟩ | ⟨5, _⟩ | ⟨6, _⟩ | ⟨7, _⟩ | ⟨8, _⟩ =>
      oneLock_main.prefix (by decide +revert)
    | ⟨9, _⟩ => oneLock_main
    | ⟨10, _⟩ | ⟨11, _⟩ | ⟨12, _⟩ | ⟨13, _⟩ | ⟨14, _⟩ | ⟨15, _⟩ | ⟨16, _⟩ | ⟨17, _⟩ =>
      oneLock_task.prefix (by decide +revert)
    | ⟨18, _⟩ => oneLock_task
    | ⟨_ + 19, h⟩ => absurd h (by omega)
  rf_val := by decide
  rf_write_read := by
    simp only [Execution.IsWrite, Execution.IsRead]
    decide
  rf_source := by
    simp only [Execution.IsRead]
    decide
  rmw_iff := by decide

-- The model allows this execution, and it has no data race: the plain write happens before the
-- plain read, through the `unlock` and the read of `serving` that sees it.
#guard oneLockData.check == some (true, false)

end Vx
