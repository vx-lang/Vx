import VxMemoryModel.ReleaseSequence

/-!
# The orders of the ticket lock in `Mutex`

`stdlib/std/sync/mutex.vx` takes a ticket from `next` with a relaxed read-modify-write. It waits
with acquire loads until `serving` holds the ticket. The drop of the guard adds 1 to `serving` with
an `sc` read-modify-write. `lock_unlock_hb_lock` gives the edge from an unlock to a later `lock`.
`mutex_excl` shows that two `lock`s still never hold the lock at the same time.
-/

namespace Vx.Execution

open Relation Classical

variable {X : Execution}

/-- Each read-modify-write of `x` adds 1, so the value grows along a chain of them. -/
private theorem chain_val_lt (wf : X.WellFormed) {x : Nat} {val : X.E → Int}
    (hsrc : ∀ r y, X.rmw r y → X.loc y = x → X.loc r = x ∧ ∃ s, X.rf s r)
    (hadd : ∀ r y, X.rmw r y → X.loc y = x → val y = val r + 1)
    (hval : ∀ w r, X.rf w r → val w = val r) {a b : X.E}
    (hab : TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) a b) (hbx : X.loc b = x) :
    X.loc a = x ∧ val a < val b := by
  induction hab with
  | single h =>
    obtain ⟨r, hr, hrb⟩ := h
    have hrx := (hsrc r _ hrb hbx).1
    have := hadd r _ hrb hbx
    have := hval _ r hr
    exact ⟨(wf.rf_write_read _ r hr).2.2.trans hrx, by omega⟩
  | tail _ h ih =>
    obtain ⟨r, hr, hrb⟩ := h
    have hrx := (hsrc r _ hrb hbx).1
    obtain ⟨hax, hlt⟩ := ih ((wf.rf_write_read _ r hr).2.2.trans hrx)
    have := hadd r _ hrb hbx
    have := hval _ r hr
    exact ⟨hax, by omega⟩

/-- The unlock `ru wu` adds 1 to `serving` (the place `x`) with a release read-modify-write. A
`lock` reads `serving` with the acquire read `e`, and sees the value of `wu` or more. Then each
event before the unlock happens before each event after `e`. -/
theorem lock_unlock_hb_lock (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {x : Nat} {val : X.E → Int}
    (hrmw : ∀ y, X.IsWrite y → X.loc y = x → (∃ a, X.co a y) → ∃ r, X.rmw r y)
    (hsrc : ∀ r y, X.rmw r y → X.loc y = x → X.loc r = x ∧ ∃ s, X.rf s r)
    (hadd : ∀ r y, X.rmw r y → X.loc y = x → val y = val r + 1)
    (hval : ∀ w r, X.rf w r → val w = val r)
    {ru wu : X.E} (hu : X.rmw ru wu) (hux : X.loc wu = x) (hrel : X.IsRelease wu)
    {y e : X.E} (hye : X.rf y e) (hex : X.loc e = x) (hacq : X.IsAcquire e)
    (hle : val wu ≤ val e) {a b : X.E} (ha : a = wu ∨ X.po a wu) (hb : e = b ∨ X.po e b) :
    X.hb a b := by
  obtain ⟨-, huw, -, hua⟩ := wf.rmw_atomic ru wu hu
  obtain ⟨hyw, -, hyl⟩ := wf.rf_write_read y e hye
  have hyx : X.loc y = x := hyl.trans hex
  have reach := rmw_chain_of_co wf hc ho hfin hrmw hsrc
  -- `wu` is `y` or comes before it in `co`. If `y` came first, `e` would see less than `wu`.
  have hchain : wu = y ∨ TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) wu y := by
    by_cases hne : wu = y
    · exact Or.inl hne
    rcases ho.total wu y huw hyw (hux.trans hyx.symm) hne with h | h
    · exact Or.inr (reach wu y hyx h)
    · have := (chain_val_lt wf hsrc hadd hval (reach y wu hux h) hux).2
      have := hval y e hye
      omega
  exact hb_of_release_read ha huw hrel hua hchain hye hacq hb

/-- One `lock` of a `Mutex` and the drop of its guard. The `lock` takes the ticket `t` from
`next`, and its acquire read `e` sees `serving` equal to the ticket. The drop reads `serving` as
the ticket with `u`, and writes 1 more with the release write `w`. -/
structure LockCall (X : Execution) (val : X.E → Int) (next serving : Nat) where
  t : X.E
  t' : X.E
  e : X.E
  u : X.E
  w : X.E
  ticket : X.rmw t t' ∧ X.loc t' = next
  sees : X.IsRead e ∧ X.loc e = serving ∧ X.IsAcquire e ∧ val e = val t
  unlock : X.rmw u w ∧ X.loc w = serving ∧ X.IsRelease w ∧ val u = val t

/-- The edge from the drop of `L` to the `lock` of `M`, when `L` has the smaller ticket. -/
private theorem LockCall.hb_of_lt (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {val : X.E → Int} {next serving : Nat} (hS : X.Counter val serving)
    (L M : LockCall X val next serving) (hlt : val L.t < val M.t) :
    ∀ a b, (a = L.w ∨ X.po a L.w) → (M.e = b ∨ X.po M.e b) → X.hb a b := by
  intro a b ha hb
  obtain ⟨hLu, hLw, hLrel, hLv⟩ := L.unlock
  obtain ⟨hMr, hMl, hMacq, hMv⟩ := M.sees
  obtain ⟨y, hy⟩ := hS.rf_source M.e hMr
  have hrmw : ∀ y, X.IsWrite y → X.loc y = serving → (∃ a, X.co a y) → ∃ r, X.rmw r y := by
    intro y hyw hyx ⟨a, hay⟩
    refine byContradiction fun hno => ?_
    exact hS.init_first wf hc ho hfin hyw hyx (fun r h => hno ⟨r, h⟩) a hay
  have hsrc : ∀ r y, X.rmw r y → X.loc y = serving → X.loc r = serving ∧ ∃ s, X.rf s r :=
    fun r y h hy => ⟨(hS.add r y h hy).1, hS.rf_source r (wf.rmw_atomic r y h).1⟩
  have hadd : ∀ r y, X.rmw r y → X.loc y = serving → val y = val r + 1 :=
    fun r y h hy => (hS.add r y h hy).2
  have := (hS.add L.u L.w hLu hLw).2
  exact lock_unlock_hb_lock wf hc ho hfin hrmw hsrc hadd hS.rf_val hLu hLw hLrel hy hMl hMacq
    (by omega) ha hb

/-- Two `lock`s of a `Mutex` never hold it at the same time. They take different tickets, with
any order on `next`. So the drop of one happens before the acquire read of the other. -/
theorem mutex_excl (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite)
    {val : X.E → Int} {next serving : Nat} (hN : X.Counter val next)
    (hS : X.Counter val serving) (L M : LockCall X val next serving) (hne : L.t ≠ M.t) :
    (∀ a b, (a = L.w ∨ X.po a L.w) → (M.e = b ∨ X.po M.e b) → X.hb a b) ∨
      (∀ a b, (a = M.w ∨ X.po a M.w) → (L.e = b ∨ X.po L.e b) → X.hb a b) := by
  have hv : val L.t ≠ val M.t := fun h =>
    hne (hN.tickets wf hc ho hfin L.ticket.1 L.ticket.2 M.ticket.1 M.ticket.2 h)
  rcases Int.lt_or_gt_of_ne hv with h | h
  · exact Or.inl (L.hb_of_lt wf hc ho hfin hS M h)
  · exact Or.inr (M.hb_of_lt wf hc ho hfin hS L h)

end Vx.Execution
