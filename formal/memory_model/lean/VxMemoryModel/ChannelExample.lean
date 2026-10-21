import VxMemoryModel.Channel
import VxMemoryModel.Check

/-!
# An example of a channel

The main code sends 7 on channel 0, which has room for one value and one sender. Then it drops
the sender, starts task 0, and joins it. Task 0 makes two `recv`s into the register 4, with the
status in the register 5. The first gets 7. The second reads that the sender dropped, then reads
`full 1` as 0, so the channel is closed. The run below shows that both ends of a `recv` happen in
an execution that the model allows.
-/

namespace Vx

def oneSend : ChanProg where
  main := .seq (.send 0 (.const 7)) (.seq (.drop 0) (.seq (.base (.spawn 0)) (.base (.join 0))))
  task
    | 0 => .seq (.recv 4 5 0) (.recv 4 5 0)
    | _ => .base .skip
  count := 1
  places := 0
  chans := 1
  cap _ := 1
  senders _ := 1
  msgs := 2
  tmp := 0

theorem oneSend_ok : oneSend.Ok where
  main := by simp [oneSend, ChanProg.StmtOk, Stmt.Below]
  task
    | 0 => by simp [oneSend, ChanProg.StmtOk]
    | _ + 1 => by simp [oneSend, ChanProg.StmtOk, Stmt.Below]
  cap _ _ := by simp [oneSend]

/-- The counters are the places 0 to 2: `send`s, `recv`s and drops. `full 0` and `full 1` are 3
and 4, the values are 5 and 6, and `taken 0` to `taken 3` are 7 to 10. The flags `start 0` and
`done 0` of the task are 11 and 12. -/
theorem oneSend_initList :
    oneSend.initList = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (7, 1)] := by decide

/-- Events: 0 to 14 are the main code, and 15 to 27 are task 0. -/
def oneSendData : Data 28 := Data.ofLists 28
  [.write, .write, .write, .write, .write, .write, .read, .write, .read, .write, .write, .read,
    .write, .write, .read,
    .read, .read, .write, .read, .read, .read, .write, .read, .write, .read, .read, .write, .write]
  [.sc, .sc, .sc, .sc, .sc, .sc, .sc, .sc, .sc, .plain, .sc, .sc, .sc, .sc, .sc,
    .sc, .sc, .sc, .sc, .sc, .plain, .sc, .sc, .sc, .sc, .sc, .sc, .sc]
  [0, 1, 2, 3, 4, 7, 0, 0, 7, 5, 3, 2, 2, 11, 12,
    11, 1, 1, 2, 3, 5, 8, 1, 1, 2, 4, 9, 12]
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  ((List.range 28).flatMap fun a => (List.range 28).filterMap fun b =>
    if a < b ∧ (b < 15 ∨ 15 ≤ a) then some (a, b) else none)
  [(0, 6), (5, 8), (2, 11), (13, 15), (1, 16), (12, 18), (10, 19), (9, 20), (17, 22), (12, 24),
    (4, 25), (27, 14)]
  [(0, 7), (1, 17), (1, 23), (17, 23), (2, 12), (3, 10)]
  [(6, 7), (11, 12), (16, 17), (22, 23)]

def oneSendLabels : Nat → Nat → Label
  | 0, 0 => ⟨.write, .sc, 0, 0, false⟩
  | 0, 1 => ⟨.write, .sc, 1, 0, false⟩
  | 0, 2 => ⟨.write, .sc, 2, 0, false⟩
  | 0, 3 => ⟨.write, .sc, 3, 0, false⟩
  | 0, 4 => ⟨.write, .sc, 4, 0, false⟩
  | 0, 5 => ⟨.write, .sc, 7, 1, false⟩
  | 0, 6 => ⟨.read, .sc, 0, 0, false⟩
  | 0, 7 => ⟨.write, .sc, 0, 1, true⟩
  | 0, 8 => ⟨.read, .sc, 7, 1, false⟩
  | 0, 9 => ⟨.write, .plain, 5, 7, false⟩
  | 0, 10 => ⟨.write, .sc, 3, 1, false⟩
  | 0, 11 => ⟨.read, .sc, 2, 0, false⟩
  | 0, 12 => ⟨.write, .sc, 2, 1, true⟩
  | 0, 13 => ⟨.write, .sc, 11, 1, false⟩
  | 0, 14 => ⟨.read, .sc, 12, 1, false⟩
  | 1, 0 => ⟨.read, .sc, 11, 1, false⟩
  | 1, 1 => ⟨.read, .sc, 1, 0, false⟩
  | 1, 2 => ⟨.write, .sc, 1, 1, true⟩
  | 1, 3 => ⟨.read, .sc, 2, 1, false⟩
  | 1, 4 => ⟨.read, .sc, 3, 1, false⟩
  | 1, 5 => ⟨.read, .plain, 5, 7, false⟩
  | 1, 6 => ⟨.write, .sc, 8, 1, false⟩
  | 1, 7 => ⟨.read, .sc, 1, 1, false⟩
  | 1, 8 => ⟨.write, .sc, 1, 2, true⟩
  | 1, 9 => ⟨.read, .sc, 2, 1, false⟩
  | 1, 10 => ⟨.read, .sc, 4, 0, false⟩
  | 1, 11 => ⟨.write, .sc, 9, 1, false⟩
  | 1, 12 => ⟨.write, .sc, 12, 1, false⟩
  | _, _ => ⟨.fence, .plain, 0, 0, false⟩

/-- The writes at the start, the `send`, the drop, the `spawn` and the `join`. The `send` gets the
number 0, and the writes at the start set `taken 0`, so its wait ends at once. -/
theorem oneSend_main : Trace (oneSend.toTaskProg.toProg.code 0) 15 (oneSendLabels 0) := by
  rw [ChanProg.code_zero, oneSend_initList]
  exact .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <|
    .visible (.seq (.seq .store)) <| .silent (.seq .seq_skip) <| .silent .seq_skip <|
    .visible (.seq (.seq (.fetchAdd 0))) <| .visible (.seq (.seq .rmwStore)) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.seq (.waitAt 0 1 rfl (by decide)))) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.seq (.storeAt 0 rfl (by decide)))) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.storeAt 0 rfl (by decide))) <|
    .silent .seq_skip <|
    .visible (.seq (.fetchAdd 0)) <| .visible (.seq .rmwStore) <| .silent .seq_skip <|
    .silent (.seq (.seq (.assignAt 0 rfl Nat.one_pos))) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.storeAt 0 rfl Nat.one_pos)) <| .silent .seq_skip <|
    .visible (.waitAt 0 1 rfl Nat.one_pos) .stop

/-- The task waits for `start 0`. The first `recv` takes the number 0, reads 1 drop and `full 0`
set, reads the value 7, and sets `taken 1`. The second takes the number 1, reads 1 drop and
`full 1` not set, sets the status to 0, and sets `taken 2`. Then the task sets `done 0`. -/
theorem oneSend_task : Trace (oneSend.toTaskProg.toProg.code 1) 13 (oneSendLabels 1) := by
  rw [TaskProg.code_succ, TaskProg.taskCode_of_lt (by decide)]
  exact .wait_seq <|
    .visible (.seq (.seq (.seq (.fetchAdd 0)))) <| .visible (.seq (.seq (.seq .rmwStore))) <|
    .silent (.seq (.seq .seq_skip)) <|
    .visible (.seq (.seq (.seq (.seq (.seq (.load 1)))))) <|
    .silent (.seq (.seq (.seq (.seq .seq_skip)))) <|
    .visible (.seq (.seq (.seq (.seq (.loadAt 0 1 rfl (by decide)))))) <|
    .silent (.seq (.seq (.seq .seq_skip))) <| .silent (.seq (.seq (.seq .loop))) <|
    .silent (.seq (.seq (.seq (.ite_false (by decide))))) <| .silent (.seq (.seq .seq_skip)) <|
    .silent (.seq (.seq (.seq (.ite_true (by decide))))) <|
    .visible (.seq (.seq (.seq (.seq (.loadAt 0 7 rfl (by decide)))))) <|
    .silent (.seq (.seq (.seq .seq_skip))) <| .silent (.seq (.seq (.seq .assign))) <|
    .silent (.seq (.seq .seq_skip)) <| .visible (.seq (.seq (.storeAt 1 rfl (by decide)))) <|
    .silent (.seq .seq_skip) <|
    .visible (.seq (.seq (.fetchAdd 1))) <| .visible (.seq (.seq .rmwStore)) <|
    .silent (.seq .seq_skip) <|
    .visible (.seq (.seq (.seq (.seq (.load 1))))) <|
    .silent (.seq (.seq (.seq .seq_skip))) <|
    .visible (.seq (.seq (.seq (.loadAt 1 0 rfl (by decide))))) <|
    .silent (.seq (.seq .seq_skip)) <| .silent (.seq (.seq .loop)) <|
    .silent (.seq (.seq (.ite_false (by decide)))) <| .silent (.seq .seq_skip) <|
    .silent (.seq (.seq (.ite_false (by decide)))) <| .silent (.seq (.seq .assign)) <|
    .silent (.seq .seq_skip) <| .visible (.seq (.storeAt 2 rfl (by decide))) <|
    .silent .seq_skip <| .visible .store .stop

def oneSendRun : oneSend.toTaskProg.toProg.Run oneSendData.toExecution where
  val i := [0, 0, 0, 0, 0, 1, 0, 1, 1, 7, 1, 0, 1, 1, 1,
    1, 0, 1, 1, 1, 7, 1, 1, 2, 1, 0, 1, 1].getD i.val 0
  isRmw i := [7, 12, 17, 23].contains i.val
  idx i := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].getD i.val 0
  lab := oneSendLabels
  po_iff := by decide
  idx_inj := by decide
  idx_down := by decide
  lab_eq := by decide
  trace
    | ⟨0, _⟩ | ⟨1, _⟩ | ⟨2, _⟩ | ⟨3, _⟩ | ⟨4, _⟩ | ⟨5, _⟩ | ⟨6, _⟩ | ⟨7, _⟩ | ⟨8, _⟩ | ⟨9, _⟩
    | ⟨10, _⟩ | ⟨11, _⟩ | ⟨12, _⟩ | ⟨13, _⟩ =>
      oneSend_main.prefix (by decide +revert)
    | ⟨14, _⟩ => oneSend_main
    | ⟨15, _⟩ | ⟨16, _⟩ | ⟨17, _⟩ | ⟨18, _⟩ | ⟨19, _⟩ | ⟨20, _⟩ | ⟨21, _⟩ | ⟨22, _⟩ | ⟨23, _⟩
    | ⟨24, _⟩ | ⟨25, _⟩ | ⟨26, _⟩ =>
      oneSend_task.prefix (by decide +revert)
    | ⟨27, _⟩ => oneSend_task
    | ⟨_ + 28, h⟩ => absurd h (by omega)
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
