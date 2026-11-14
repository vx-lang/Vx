import VxMemoryModel.Typed

/-!
# Examples of raw pointers and `unsafe impl`

`bufProg` has a raw pointer `buf` (local 0) of pointer type 0, which points to one number, and a
number `n` (local 1). Pointer type 0 has `unsafe impl Send`, as `Vec` does. Task 0 moves `buf`
into itself and adds to the number that it points to. The code of the block writes `n`. The
checker accepts it, and its claims hold, so it has no data race.

The checker rejects three changes of it: a pointer type with no `unsafe impl` (`rawProg`), an
access through `buf` outside `unsafe` (`safeDerefProg`), and code of a block that accesses
through a pointer with no `unsafe impl` while a task runs (`aliasProg`).

`syncCellProg` claims `unsafe impl Sync` for a pointer to a cell. The checker accepts it, because
it cannot check the claim. The claim is false, so the theorem says nothing about it.
-/

namespace Vx

namespace Typed

def bufProg : TProg where
  ty
    | 0 => .ptr 0
    | _ => .val (.leaf .int)
  locals := 2
  width := 1
  regs := 1
  count := 1
  caps
    | 0 => [(0, .move)]
    | _ => []
  body
    | 0 => .unsafeCode (.fetchAdd 0 0 0 (.const 1))
    | _ => .skip
  main := [.code (.unsafeCode (.write 0 0 (.const 5))),
    .block [.spawn 0, .code (.write 1 0 (.const 1))],
    .code (.unsafeCode (.read 0 0 0))]
  implSend _ := true
  /- The number that `buf` points to is in slot 0. No other local uses that slot. -/
  points _ := 0

theorem bufProg_wellTyped : bufProg.WellTyped where
  width x := by rcases x with _ | x <;> simp [bufProg, TProg.vty, Ty.leaves]
  targets x y h := by rcases x with _ | x <;> simp [bufProg] at h
  caps u x m h := by
    rcases u with _ | u <;> simp [bufProg] at h
    obtain ⟨rfl, rfl⟩ := h
    simp [bufProg, TProg.SendL]
  bodies u := by
    rcases u with _ | u <;>
      simp [bufProg, TProg.Fits, TProg.PtrOk, TProg.IsPtr, TProg.At, TProg.pathPerm,
        TProg.capPerm, TProg.target, TProg.vty, via, Allowed, Expr.RegsBelow, Ty.leaves]
  phases k p hp := by
    rcases k with _ | _ | _ | k <;> simp [bufProg] at hp <;> subst hp <;>
      simp [TProg.PhaseOk, TProg.Fits, TProg.PtrOk, TProg.IsPtr, TProg.At, TProg.blockFits,
        TProg.blockPerm, TProg.taskPerm, TProg.pathPerm, TProg.capPerm, TProg.target, TProg.vty,
        TProg.Live, spawnsOf, via, Allowed, Expr.RegsBelow, Ty.leaves, bufProg]
  nodup := by decide
  spawn_lt := by decide
  slot_lt _ h := h
  slots _ _ _ _ _ h := h
  refs _ _ _ := Or.inl rfl

theorem bufProg_claims : bufProg.Claims where
  valid _ _ _ := by simp [bufProg]
  owns x y _ hx _ _ _ _ h := by
    obtain ⟨_, hk⟩ := hx
    rcases x with _ | x <;> simp [bufProg] at hk
    rcases y with _ | y
    · rfl
    · simp [TProg.store, bufProg] at h
  sync _ _ _ h := by simp [bufProg] at h

/-- No execution of `bufProg` has a data race. -/
theorem bufProg_raceFree {X : Execution} (hX : bufProg.toTaskProg.toProg.Execs X) : X.RaceFree :=
  bufProg_wellTyped.raceFree bufProg_claims hX

/-- `bufProg` with no `unsafe impl Send`. -/
def rawProg : TProg := { bufProg with implSend := fun _ => false }

/-- The checker rejects `rawProg`, because a raw pointer is not `Send`. -/
theorem rawProg_rejected : ¬ rawProg.WellTyped := fun h => by
  have := (h.caps 0 0 .move (by simp [rawProg, bufProg])).2 nofun
  simp [TProg.SendL, rawProg, bufProg] at this

/-- `bufProg`, but its first phase writes through `buf` outside `unsafe`. -/
def safeDerefProg : TProg := { bufProg with main := [.code (.write 0 0 (.const 5))] }

/-- The checker rejects `safeDerefProg`, because an access through a raw pointer needs
`unsafe`. -/
theorem safeDerefProg_rejected : ¬ safeDerefProg.WellTyped := fun h => by
  have := h.phases 0 _ rfl
  simp [TProg.PhaseOk, TProg.Fits, TProg.PtrOk, TProg.IsPtr, safeDerefProg, bufProg] at this

/-- Local 0 is a raw pointer with no `unsafe impl`, and it points to the slot of local 1. Task 0
borrows local 1 with `mut` and writes it. At the same time, the code of the block writes through
the pointer. The two writes race. -/
def aliasProg : TProg where
  ty
    | 0 => .ptr 0
    | _ => .val (.leaf .int)
  locals := 2
  width := 1
  regs := 1
  count := 1
  caps _ := [(1, .mut)]
  body _ := .write 1 0 (.const 1)
  main := [.block [.spawn 0, .code (.unsafeCode (.write 0 0 (.const 2)))]]
  points _ := 1

/-- The checker rejects `aliasProg`. The code of a block uses only raw pointers whose type has an
`unsafe impl`. -/
theorem aliasProg_rejected : ¬ aliasProg.WellTyped := fun h => by
  have := (h.phases 0 _ rfl).1 (.unsafeCode (.write 0 0 (.const 2))) (by simp)
  simp [TProg.Fits, TProg.PtrOk, TProg.IsPtr, TProg.Shared, aliasProg] at this

/-- Pointer type 0 points to a cell and claims `unsafe impl Sync`. Two tasks borrow the pointer
with `ref` and write the cell through it. -/
def syncCellProg : TProg where
  ty _ := .ptr 0
  locals := 1
  width := 1
  regs := 1
  count := 2
  caps _ := [(0, .ref)]
  body _ := .unsafeCode (.write 0 0 (.const 1))
  main := [.block [.spawn 0, .spawn 1]]
  pointee _ := .leaf .cell
  implSync _ := true

/-- The checker accepts `syncCellProg`: it cannot check the claim of `unsafe impl Sync`. -/
theorem syncCellProg_wellTyped : syncCellProg.WellTyped where
  width x := by simp [syncCellProg, TProg.vty, Ty.leaves]
  targets x y h := by simp [syncCellProg] at h
  caps u x m h := by
    simp [syncCellProg] at h
    obtain ⟨rfl, rfl⟩ := h
    simp [syncCellProg, TProg.SyncL]
  bodies u := by
    simp [syncCellProg, TProg.Fits, TProg.PtrOk, TProg.IsPtr, TProg.At, TProg.pathPerm,
      TProg.capPerm, TProg.target, TProg.vty, via, Allowed, Expr.RegsBelow, Ty.leaves]
  phases k p hp := by
    rcases k with _ | k <;> simp [syncCellProg] at hp
    subst hp
    simp [TProg.PhaseOk, TProg.taskPerm, TProg.pathPerm, TProg.capPerm, TProg.target, TProg.Live,
      spawnsOf, via, syncCellProg]
    intro x
    split <;> simp
  nodup := by decide
  spawn_lt := by decide
  slot_lt _ h := h
  slots _ _ _ _ _ h := h
  refs _ _ _ := Or.inl rfl

/-- The claim of `syncCellProg` is false: a cell is not `Sync`. -/
theorem syncCellProg_claim_false : ¬ syncCellProg.Claims := fun h => by
  have := h.sync 0 0 rfl rfl
  simp [syncCellProg, Ty.Sync, Leaf.Sync] at this

end Typed

end Vx
