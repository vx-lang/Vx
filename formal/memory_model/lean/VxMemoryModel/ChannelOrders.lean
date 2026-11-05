import VxMemoryModel.ReleaseSequence

/-!
# The orders of the slots of a channel

`stdlib/std/sync/channel.vx` gives each `send` a number `k` with a relaxed read-modify-write of a
counter. The send waits with an acquire load until its slot holds `2k`, writes its value, and sets
the slot to `2k + 1` with a release store. The `recv` with the number `k` waits with an acquire
load for `2k + 1`, reads the value, and sets the slot to `2(k + n)` with a release store, for the
send `n` places later. A `pass_a_skip` sets the slot to `2(k + n)` with an `sc` store, which is
also a release store. `slot_send_hb_recv` and `slot_recv_hb_send` give the two edges.
-/

namespace Vx.Execution

open Relation Classical

variable {X : Execution}

/-- The facts about the slot word `x` with the values `val`. Each read of it reads some write,
and no two writes to it write the same value. -/
structure SlotWord (X : Execution) (val : X.E → Int) (x : Nat) : Prop where
  rf_source : ∀ r, X.IsRead r → X.loc r = x → ∃ w, X.rf w r
  rf_val : ∀ w r, X.rf w r → X.loc r = x → val w = val r
  unique : ∀ a b, X.IsWrite a → X.IsWrite b → X.loc a = x → X.loc b = x → val a = val b → a = b

/-- An acquire read of a slot word that sees the value of a release store reads that store. So
each event before the store happens before each event after the read. -/
theorem SlotWord.hb_of_read (wf : X.WellFormed) {val : X.E → Int} {x : Nat}
    (hS : X.SlotWord val x) {s e : X.E} (hs : X.IsWrite s) (hsx : X.loc s = x)
    (hrel : X.IsRelease s) (he : X.IsRead e) (hex : X.loc e = x) (hacq : X.IsAcquire e)
    (hv : val e = val s) {a b : X.E} (ha : a = s ∨ X.po a s) (hb : e = b ∨ X.po e b) :
    X.hb a b := by
  obtain ⟨y, hy⟩ := hS.rf_source e he hex
  obtain ⟨hyw, -, hyl⟩ := wf.rf_write_read y e hy
  have hys : y = s :=
    hS.unique y s hyw hs (hyl.trans hex) hsx ((hS.rf_val y e hy hex).trans hv)
  subst hys
  have hsa : X.IsAtomic y := fun h => by
    rcases hrel with h' | h' | h' <;> rw [h'] at h <;> cases h
  exact hb_of_release_read ha hs hrel hsa (Or.inl rfl) hy hacq hb

/-- The send with the number `k` sets its slot to `2k + 1` with the release store `s`. When the
acquire load `e` of a `recv` reads `2k + 1`, the write of the value happens before its read. -/
theorem slot_send_hb_recv (wf : X.WellFormed) {val : X.E → Int} {x : Nat}
    (hS : X.SlotWord val x) {k : Int} {s e : X.E} (hs : X.IsWrite s) (hsx : X.loc s = x)
    (hrel : X.IsRelease s) (hsk : val s = 2 * k + 1) (he : X.IsRead e) (hex : X.loc e = x)
    (hacq : X.IsAcquire e) (hek : val e = 2 * k + 1) {a b : X.E} (ha : a = s ∨ X.po a s)
    (hb : e = b ∨ X.po e b) : X.hb a b :=
  hS.hb_of_read wf hs hsx hrel he hex hacq (hek.trans hsk.symm) ha hb

/-- The `recv` with the number `k` reads the value, then sets the slot to `2(k + n)` with the
release store `s`. When the send with the number `k + n` reads `2(k + n)` with the acquire load
`e`, the read of the old value happens before the write of the new one. -/
theorem slot_recv_hb_send (wf : X.WellFormed) {val : X.E → Int} {x : Nat}
    (hS : X.SlotWord val x) {k n : Int} {s e : X.E} (hs : X.IsWrite s) (hsx : X.loc s = x)
    (hrel : X.IsRelease s) (hsk : val s = 2 * (k + n)) (he : X.IsRead e) (hex : X.loc e = x)
    (hacq : X.IsAcquire e) (hek : val e = 2 * (k + n)) {a b : X.E} (ha : a = s ∨ X.po a s)
    (hb : e = b ∨ X.po e b) : X.hb a b :=
  hS.hb_of_read wf hs hsx hrel he hex hacq (hek.trans hsk.symm) ha hb

/-- Two sends never get the same number, though they take it from the counter `SENDS` with a
relaxed read-modify-write. `Counter.tickets` holds for any orders. -/
theorem send_numbers_differ (wf : X.WellFormed) (hc : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {val : X.E → Int} {sends : Nat} (hN : X.Counter val sends)
    {r1 r2 w1 w2 : X.E} (h1 : X.rmw r1 w1) (hw1 : X.loc w1 = sends) (h2 : X.rmw r2 w2)
    (hw2 : X.loc w2 = sends) (hne : r1 ≠ r2) : val r1 ≠ val r2 :=
  fun hv => hne (hN.tickets wf hc ho hfin h1 hw1 h2 hw2 hv)

end Vx.Execution
