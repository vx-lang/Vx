import VxMemoryModel.Channel
import VxMemoryModel.Check

/-!
# An example of a channel

The main code sends 7 on channel 0, which has room for one value. Then it starts task 0 and joins
it. Task 0 takes the value into register 1. The run below shows that the lowered `send` and `recv`
can both reach their end, in an execution that the model allows.
-/

namespace Vx

def oneSend : ChanProg where
  main := .seq (.send 0 (.const 7)) (.seq (.base (.spawn 0)) (.base (.join 0)))
  task
    | 0 => .recv 1 0
    | _ => .base .skip
  count := 1
  places := 0
  chans := 1
  cap _ := 1
  msgs := 1
  tmp := 0

theorem oneSend_ok : oneSend.Ok where
  main := by simp [oneSend, ChanProg.StmtOk, Stmt.Below]
  task
    | 0 => by simp [oneSend, ChanProg.StmtOk]
    | _ + 1 => by simp [oneSend, ChanProg.StmtOk, Stmt.Below]
  cap _ _ := by simp [oneSend]

/-- The counters are the places 0 and 1, `full 0` is 2, the value is 3, and `taken 0` and
`taken 1` are 4 and 5. The flags `start 0` and `done 0` of the task are 6 and 7. -/
theorem oneSend_initList : oneSend.initList = [(0, 0), (1, 0), (4, 1)] := by decide

/-- Events: 0 to 9 are the main code, and 10 to 16 are task 0. -/
def oneSendData : Data 17 := Data.ofLists 17
  [.write, .write, .write, .read, .write, .read, .write, .write, .write, .read,
    .read, .read, .write, .read, .read, .write, .write]
  [.sc, .sc, .sc, .sc, .sc, .sc, .plain, .sc, .sc, .sc,
    .sc, .sc, .sc, .sc, .plain, .sc, .sc]
  [0, 1, 4, 0, 0, 4, 3, 2, 6, 7,
    6, 1, 1, 2, 3, 5, 7]
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  ((List.range 17).flatMap fun a => (List.range 17).filterMap fun b =>
    if a < b ∧ (b < 10 ∨ 10 ≤ a) then some (a, b) else none)
  [(0, 3), (2, 5), (8, 10), (1, 11), (7, 13), (6, 14), (16, 9)]
  [(0, 4), (1, 12)]
  [(3, 4), (11, 12)]

def oneSendLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .sc, 0, 0, false⟩
  | 0, 1 => ⟨.write, .sc, 1, 0, false⟩
  | 0, 2 => ⟨.write, .sc, 4, 1, false⟩
  | 0, 3 => ⟨.read, .sc, 0, 0, false⟩
  | 0, 4 => ⟨.write, .sc, 0, 1, true⟩
  | 0, 5 => ⟨.read, .sc, 4, 1, false⟩
  | 0, 6 => ⟨.write, .plain, 3, 7, false⟩
  | 0, 7 => ⟨.write, .sc, 2, 1, false⟩
  | 0, 8 => ⟨.write, .sc, 6, 1, false⟩
  | 0, 9 => ⟨.read, .sc, 7, 1, false⟩
  | 1, 0 => ⟨.read, .sc, 6, 1, false⟩
  | 1, 1 => ⟨.read, .sc, 1, 0, false⟩
  | 1, 2 => ⟨.write, .sc, 1, 1, true⟩
  | 1, 3 => ⟨.read, .sc, 2, 1, false⟩
  | 1, 4 => ⟨.read, .plain, 3, 7, false⟩
  | 1, 5 => ⟨.write, .sc, 5, 1, false⟩
  | 1, 6 => ⟨.write, .sc, 7, 1, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

/-- The writes at the start, the `send`, the `spawn` and the `join`. The `send` gets the number 0,
and the writes at the start set `taken 0`, so its wait ends at once. -/
theorem oneSend_main : Trace (oneSend.toTaskProg.toProg.code 0) 10 (oneSendLabels 0) := by
  rw [ChanProg.code_zero, oneSend_initList]
  exact .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <| .silent .seq_skip <|
    .visible (.seq (.seq (.fetchAdd 0))) <| .visible (.seq (.seq .rmwStore)) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.seq (.waitAt 0 1 rfl (by decide)))) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.seq (.storeAt 0 rfl Nat.one_pos))) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.storeAt 0 rfl Nat.one_pos)) <|
    .silent .seq_skip <| .visible (.seq (.storeAt 0 rfl Nat.one_pos)) <| .silent .seq_skip <|
    .visible (.waitAt 0 1 rfl Nat.one_pos) .stop

/-- The task waits for `start 0`, takes the number 0, waits for `full 0`, reads the value 7, sets
`taken 1`, and sets `done 0`. -/
theorem oneSend_task : Trace (oneSend.toTaskProg.toProg.code 1) 7 (oneSendLabels 1) := by
  rw [TaskProg.code_succ, TaskProg.taskCode_of_lt (by decide)]
  exact .wait_seq <| .visible (.seq (.seq (.fetchAdd 0))) <| .visible (.seq (.seq .rmwStore)) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.seq (.waitAt 0 1 rfl Nat.one_pos))) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.seq (.loadAt 0 7 rfl Nat.one_pos))) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.storeAt 1 rfl (by decide))) <|
    .silent .seq_skip <| .visible .store .stop

def oneSendRun : oneSend.toTaskProg.toProg.Run oneSendData.toExecution where
  val i := [0, 0, 1, 0, 1, 1, 7, 1, 1, 1, 1, 0, 1, 1, 7, 1, 1].getD i.val 0
  isRmw i := [4, 12].contains i.val
  idx i := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6].getD i.val 0
  lab := oneSendLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ | ⟨1, _⟩ | ⟨2, _⟩ | ⟨3, _⟩ | ⟨4, _⟩ | ⟨5, _⟩ | ⟨6, _⟩ | ⟨7, _⟩ | ⟨8, _⟩ =>
      oneSend_main.prefix (by decide +revert)
    | ⟨9, _⟩ => oneSend_main
    | ⟨10, _⟩ | ⟨11, _⟩ | ⟨12, _⟩ | ⟨13, _⟩ | ⟨14, _⟩ | ⟨15, _⟩ =>
      oneSend_task.prefix (by decide +revert)
    | ⟨16, _⟩ => oneSend_task
    | ⟨_ + 17, h⟩ => absurd h (by omega)
  rf_val := by decide
  rf_write_read := by
    simp only [Execution.IsWrite, Execution.IsRead]
    decide
  rf_source := by
    simp only [Execution.IsRead]
    decide
  rmw_iff := by decide

-- The model allows this execution, and it has no data race: the plain write of the value
-- happens before its plain read, through `full 0`.
#guard oneSendData.check == some (true, false)

end Vx
