import VxMemoryModel.TypedChan

/-!
# Examples of the typed layer with channels and a lock

In `mutexProg`, local 0 is a cell in a mutex with lock 0, and local 1 is a number. Both tasks
borrow the mutex with `ref`, and each one adds 1 to the cell under the lock. A cell is not
`Sync`, but a mutex of a cell is. Task 0 also writes local 1 and sends the value that it read on
channel 0. Task 1 receives it. After the block, the main code reads local 1 with no lock and the
cell under the lock.

`unlockedProg` writes the cell of the mutex without the lock, so the checker rejects it. The
checker also rejects `noMutexProg`, where the two tasks share the cell with no mutex.
-/

namespace Vx

namespace Typed

/-- The types and the captures of `mutexProg`. -/
def mutexEnv : TProg where
  ty
    | 0 => .val (.leaf .cell)
    | _ => .val (.leaf .int)
  locals := 2
  width := 1
  regs := 3
  count := 2
  caps
    | 0 => [(0, .ref), (1, .mut)]
    | 1 => [(0, .ref)]
    | _ => []
  body _ := .skip
  main := []

/-- Add 1 to the cell of the mutex, under its lock. The value read stays in register 0. -/
def addCell : CSrc :=
  .locked 0 (.code (.seq (.read 0 0 0) (.write 0 0 (.add (.reg 0) (.const 1)))))

def mutexProg : CTProg where
  env := mutexEnv
  guard
    | 0 => some 0
    | _ => none
  locks := 1
  chans := 1
  cap _ := 1
  senders _ := 1
  msgs := 2
  body
    | 0 => .seq addCell (.seq (.code (.write 1 0 (.reg 0))) (.send 0 (.reg 0)))
    | 1 => .seq addCell (.recv 1 2 0)
    | _ => .code .skip
  main := [.block [.spawn 0, .spawn 1],
    .code (.seq (.code (.read 0 1 0)) (.locked 0 (.code (.read 1 0 0))))]

theorem mutexProg_wellTyped : mutexProg.WellTyped where
  width x := by rcases x with _ | x <;> simp [mutexProg, mutexEnv, TProg.vty, Ty.leaves]
  guard_lt x m h := by rcases x with _ | x <;> simp_all [mutexProg]
  guard_val x m h := by rcases x with _ | x <;> simp [mutexProg, mutexEnv] at h ⊢
  noPtrs x k := by rcases x with _ | x <;> simp [mutexProg, mutexEnv]
  caps u x m h := by
    rcases u with _ | _ | u <;> simp [mutexProg, mutexEnv] at h <;>
      rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;>
      simp [mutexProg, mutexEnv, TProg.SyncL, TProg.SendL, TProg.vty, Ty.Sync, Ty.Send, Leaf.Sync]
  bodies u := by
    rcases u with _ | _ | u <;>
      simp [mutexProg, mutexEnv, addCell, CTProg.CFits, CTProg.SFits, CTProg.AccessOk, TProg.At,
        TProg.target, TProg.vty, TProg.pathPerm, TProg.capPerm, via, Allowed, Expr.RegsBelow,
        Ty.leaves]
  phases p hp := by
    simp only [mutexProg, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · simp [CTProg.PhaseOk, TProg.taskPerm, TProg.pathPerm, TProg.capPerm, TProg.target,
        cspawnsOf, via, mutexProg, mutexEnv]
      exact ⟨fun x => by rcases x with _ | _ | x <;> simp,
        fun x => by rcases x with _ | _ | x <;> simp⟩
    · simp [CTProg.PhaseOk, CTProg.CFits, CTProg.SFits, CTProg.AccessOk, TProg.At, TProg.target,
        TProg.vty, via, Allowed, Ty.leaves, mutexProg, mutexEnv]
  nodup := by decide
  spawn_lt := by decide
  cap _ _ := by simp [mutexProg]

/-- No execution of `mutexProg` has a data race. -/
theorem mutexProg_raceFree {X : Execution} (hX : mutexProg.toChanProg.toTaskProg.toProg.Execs X)
    (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite) : X.RaceFree :=
  mutexProg_wellTyped.raceFree hX wf hc ho hfin

/-- Task 0 writes the cell of the mutex without its lock. -/
def unlockedProg : CTProg :=
  { mutexProg with body
      | 0 => .code (.write 0 0 (.const 1))
      | 1 => addCell
      | _ => .code .skip }

/-- The checker rejects `unlockedProg`: the write needs the lock 0. -/
theorem unlockedProg_rejected : ¬ unlockedProg.WellTyped := fun h => by
  have := h.bodies 0
  simp [unlockedProg, mutexProg, mutexEnv, CTProg.CFits, CTProg.SFits, CTProg.AccessOk, TProg.At,
    TProg.target, TProg.vty, Ty.leaves] at this

/-- The same tasks with no mutex: both tasks borrow the cell with `ref`. -/
def noMutexProg : CTProg := { mutexProg with guard _ := none }

/-- The checker rejects `noMutexProg`, because a cell with no lock is not `Sync`. -/
theorem noMutexProg_rejected : ¬ noMutexProg.WellTyped := fun h => by
  have := ((h.caps 0 0 .ref (by simp [noMutexProg, mutexProg, mutexEnv])).1 rfl).1 rfl
  simp [noMutexProg, mutexProg, mutexEnv, TProg.SyncL, Ty.Sync, Leaf.Sync] at this

end Typed

end Vx
