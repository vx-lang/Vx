import VxMemoryModel.Rc11

/-!
# DRF-SC for one execution

Take a consistent execution with no data race, whose atomics are all sequentially consistent
and which has no fences. Then the execution is sequentially consistent: the program cannot see
that the hardware reorders memory accesses.

The proof: each edge of a cycle of `po`, `rf`, `co` and `fr` is either in happens-before or in
`psc`, the order of the sequentially consistent events. A path of happens-before between two
sequentially consistent events is a path of `psc`. So the cycle gives a cycle in happens-before
or in `psc`, and the model forbids both.
-/

namespace Vx.Execution

open Relation

variable {X : Execution}

theorem atomic_of_seqCst {e : X.E} (h : X.IsSeqCst e) : X.IsAtomic e := by
  intro hp
  unfold IsSeqCst at h
  rw [h] at hp
  cases hp

theorem atomic_of_release {e : X.E} (h : X.IsRelease e) : X.IsAtomic e := by
  intro hp
  rcases h with h | h | h <;> rw [h] at hp <;> cases hp

theorem atomic_of_acquire {e : X.E} (h : X.IsAcquire e) : X.IsAtomic e := by
  intro hp
  rcases h with h | h | h <;> rw [h] at hp <;> cases hp

theorem release_of_seqCst {e : X.E} (h : X.IsSeqCst e) : X.IsRelease e := Or.inr (Or.inr h)

theorem acquire_of_seqCst {e : X.E} (h : X.IsSeqCst e) : X.IsAcquire e := Or.inr (Or.inr h)

theorem po_hb {a b : X.E} (h : X.po a b) : X.hb a b := TransGen.single (Or.inl h)

theorem sw_hb {a b : X.E} (h : X.sw a b) : X.hb a b := TransGen.single (Or.inr h)

theorem ne_of_write_read {a b : X.E} (ha : X.IsWrite a) (hb : X.IsRead b) : a ≠ b := by
  rintro rfl
  exact Kind.noConfusion (ha.symm.trans hb)

/-- A read of a sequentially consistent write by a sequentially consistent read synchronizes. -/
theorem sw_of_rf (wf : X.WellFormed) {w r : X.E} (h : X.rf w r) (hw : X.IsSeqCst w)
    (hr : X.IsSeqCst r) : X.sw w r := by
  have hwrite := (wf.rf_write_read _ _ h).1
  exact ⟨release_of_seqCst hw, acquire_of_seqCst hr, w, w, r, Or.inl rfl,
    ⟨w, hwrite, Or.inl rfl, hwrite, atomic_of_seqCst hw, Or.inl rfl⟩, h,
    atomic_of_seqCst hr, Or.inl rfl⟩

theorem psc_of_scb {a b : X.E} (ha : X.IsSeqCst a) (hb : X.IsSeqCst b) (h : X.scb a b) :
    X.psc a b :=
  Or.inl ⟨a, b, Or.inl ⟨ha, rfl⟩, h, Or.inl ⟨hb, rfl⟩⟩

theorem psc_of_po {a b : X.E} (ha : X.IsSeqCst a) (hb : X.IsSeqCst b) (h : X.po a b) :
    X.psc a b := psc_of_scb ha hb (Or.inl h)

section NoFences

variable (wf : X.WellFormed) (hsc : ∀ e, X.IsAtomic e → X.IsSeqCst e)
  (hnf : ∀ e, ¬ X.IsFence e)

include wf hsc in
theorem psc_of_rf {a b : X.E} (ha : X.IsAtomic a) (hb : X.IsAtomic b) (h : X.rf a b) :
    X.psc a b :=
  psc_of_scb (hsc _ ha) (hsc _ hb)
    (Or.inr (Or.inr (Or.inl ⟨sw_hb (sw_of_rf wf h (hsc _ ha) (hsc _ hb)),
      (wf.rf_write_read _ _ h).2.2⟩)))

include wf hsc in
/-- A chain of read-modify-writes that read from each other is a path of `psc`. -/
theorem rmw_chain_psc {w y : X.E} (hw : X.IsAtomic w)
    (h : TransGen (fun x y => ∃ r, X.rf x r ∧ X.rmw r y) w y) :
    TransGen X.psc w y ∧ X.IsAtomic y := by
  have step : ∀ x y, X.IsAtomic x → (∃ r, X.rf x r ∧ X.rmw r y) →
      TransGen X.psc x y ∧ X.IsAtomic y := by
    intro x y hx ⟨r, hxr, hry⟩
    have ⟨_, _, hr, hy⟩ := wf.rmw_atomic _ _ hry
    exact ⟨(TransGen.single (psc_of_rf wf hsc hx hr hxr)).tail
      (psc_of_po (hsc _ hr) (hsc _ hy) (wf.rmw_po _ _ hry)), hy⟩
  induction h with
  | single h => exact step _ _ hw h
  | tail _ h ih =>
    have ⟨p, hx⟩ := ih
    have ⟨q, hy⟩ := step _ _ hx h
    exact ⟨p.trans q, hy⟩

include wf hsc hnf in
/-- Synchronization between two events is a path of `psc`. -/
theorem sw_psc {a b : X.E} (h : X.sw a b) : TransGen X.psc a b := by
  obtain ⟨ha, _, x, y, z, hax, ⟨w, _, hxw, _, hwa, hwy⟩, hyz, hz, hzb⟩ := h
  have hax : a = x := hax.resolve_right (fun h => hnf _ h.1)
  have hzb : z = b := hzb.resolve_right (fun h => hnf _ h.2)
  subst hax hzb
  have ha := atomic_of_release ha
  have hwy : (w = y ∨ TransGen X.psc w y) ∧ X.IsAtomic y := by
    rcases hwy with rfl | hwy
    · exact ⟨Or.inl rfl, hwa⟩
    · have ⟨p, hy⟩ := rmw_chain_psc wf hsc hwa hwy
      exact ⟨Or.inr p, hy⟩
  have hyb : TransGen X.psc w z :=
    TransGen.of_refl hwy.1 (TransGen.single (psc_of_rf wf hsc hwy.2 hz hyz))
  rcases hxw with rfl | ⟨hxw, _⟩
  · exact hyb
  · exact TransGen.head (psc_of_po (hsc _ ha) (hsc _ hwa) hxw) hyb

include wf hsc hnf in
/-- A path of happens-before between two sequentially consistent events is a path of `psc`. -/
theorem hb_psc {t b : X.E} (ht : X.IsSeqCst t) (hb : X.IsSeqCst b) (h : X.hb t b) :
    TransGen X.psc t b := by
  have inv : ∀ x, X.hb t x →
      X.po t x ∨ ∃ u, X.IsSeqCst u ∧ TransGen X.psc t u ∧ (u = x ∨ X.po u x) := by
    intro x h
    induction h with
    | single h =>
      rcases h with h | h
      · exact Or.inl h
      · exact Or.inr ⟨_, hsc _ (atomic_of_acquire h.2.1), sw_psc wf hsc hnf h, Or.inl rfl⟩
    | tail _ hxy ih =>
      rcases hxy with hxy | hxy
      · rcases ih with htx | ⟨u, hu, p, rfl | hux⟩
        · exact Or.inl (wf.po_trans _ _ _ htx hxy)
        · exact Or.inr ⟨u, hu, p, Or.inr hxy⟩
        · exact Or.inr ⟨u, hu, p, Or.inr (wf.po_trans _ _ _ hux hxy)⟩
      · have hx := hsc _ (atomic_of_release hxy.1)
        have q := sw_psc wf hsc hnf hxy
        have hy := hsc _ (atomic_of_acquire hxy.2.1)
        rcases ih with htx | ⟨u, hu, p, rfl | hux⟩
        · exact Or.inr ⟨_, hy, TransGen.head (psc_of_po ht hx htx) q, Or.inl rfl⟩
        · exact Or.inr ⟨_, hy, p.trans q, Or.inl rfl⟩
        · exact Or.inr ⟨_, hy, (p.tail (psc_of_po hu hx hux)).trans q, Or.inl rfl⟩
  rcases inv b h with htb | ⟨u, hu, p, rfl | hub⟩
  · exact TransGen.single (psc_of_po ht hb htb)
  · exact p
  · exact p.tail (psc_of_po hu hb hub)

theorem seqCst_of_psc {a b : X.E} (h : X.psc a b) : X.IsSeqCst a ∧ X.IsSeqCst b := by
  rcases h with ⟨_, _, ha, _, hb⟩ | ⟨_, ha, _, hb, _⟩
  · exact ⟨ha.elim (·.1) (·.2.1), hb.elim (·.1) (·.2.1)⟩
  · exact ⟨ha, hb⟩

theorem seqCst_of_psc_path {a b : X.E} (h : TransGen X.psc a b) :
    X.IsSeqCst a ∧ X.IsSeqCst b := by
  induction h with
  | single h => exact seqCst_of_psc h
  | tail _ h ih => exact ⟨ih.1, (seqCst_of_psc h).2⟩

include wf in
/-- Without a data race, two conflicting accesses in `eco` order are in happens-before order,
or are both atomic. -/
theorem hb_or_atomic (hc : X.Consistent) (hrace : X.RaceFree) {a b : X.E}
    (hcnf : X.Conflict a b) (heco : X.eco a b) : X.hb a b ∨ (X.IsAtomic a ∧ X.IsAtomic b) := by
  have back : ¬ X.hb b a := fun h => hc.coherence b a h (Or.inr heco)
  by_cases ht : X.thread a = X.thread b
  · rcases wf.po_total a b ht hcnf.1 with h | h
    · exact Or.inl (po_hb h)
    · exact absurd (po_hb h) back
  · by_cases hab : X.hb a b
    · exact Or.inl hab
    · by_cases hat : X.IsAtomic a ∧ X.IsAtomic b
      · exact Or.inr hat
      · exact absurd ⟨hcnf, ht, hab, back, hat⟩ (hrace a b)

include wf hsc in
/-- Each edge of an interleaving is in happens-before or in `psc`. -/
theorem hb_or_psc (hc : X.Consistent) (hrace : X.RaceFree) {a b : X.E} (h : X.ord a b) :
    X.hb a b ∨ X.psc a b := by
  rcases h with h | h | h | h
  · exact Or.inl (po_hb h)
  · have ⟨hw, hr, hloc⟩ := wf.rf_write_read _ _ h
    rcases hb_or_atomic wf hc hrace ⟨ne_of_write_read hw hr, Or.inr hw, Or.inl hr, hloc,
      Or.inl hw⟩ (TransGen.single (Or.inl h)) with h' | ⟨ha, hb⟩
    · exact Or.inl h'
    · exact Or.inl (sw_hb (sw_of_rf wf h (hsc _ ha) (hsc _ hb)))
  · have ⟨hwa, hwb, hloc⟩ := wf.co_writes _ _ h
    have hne : a ≠ b := by
      rintro rfl
      exact wf.co_irrefl a h
    rcases hb_or_atomic wf hc hrace ⟨hne, Or.inr hwa, Or.inr hwb, hloc, Or.inl hwa⟩
      (TransGen.single (Or.inr (Or.inl h))) with h' | ⟨ha, hb⟩
    · exact Or.inl h'
    · exact Or.inr (psc_of_scb (hsc _ ha) (hsc _ hb) (Or.inr (Or.inr (Or.inr (Or.inl h)))))
  · have ⟨w, hwa, hwb⟩ := h
    have ⟨_, hra, hla⟩ := wf.rf_write_read _ _ hwa
    have ⟨_, hwb', hlb⟩ := wf.co_writes _ _ hwb
    rcases hb_or_atomic wf hc hrace ⟨fun e => ne_of_write_read hwb' hra e.symm, Or.inl hra,
      Or.inr hwb', hla.symm.trans hlb, Or.inr hwb'⟩ (TransGen.single (Or.inr (Or.inr h)))
      with h' | ⟨ha, hb⟩
    · exact Or.inl h'
    · exact Or.inr (psc_of_scb (hsc _ ha) (hsc _ hb) (Or.inr (Or.inr (Or.inr (Or.inr h)))))

/-- A path of happens-before, or a path of `psc` with happens-before at each end. -/
def HbPsc (X : Execution) (x y : X.E) : Prop :=
  X.hb x y ∨ ∃ s t, (x = s ∨ X.hb x s) ∧ TransGen X.psc s t ∧ (t = y ∨ X.hb t y)

include wf hsc hnf in
theorem hbPsc_of_path (hc : X.Consistent) (hrace : X.RaceFree) {x y : X.E}
    (h : TransGen X.ord x y) : X.HbPsc x y := by
  induction h with
  | single h =>
    rcases hb_or_psc wf hsc hc hrace h with h | h
    · exact Or.inl h
    · exact Or.inr ⟨_, _, Or.inl rfl, TransGen.single h, Or.inl rfl⟩
  | tail _ hyz ih =>
    rcases hb_or_psc wf hsc hc hrace hyz with hyz | hyz
    · rcases ih with hxy | ⟨s, t, hxs, p, hty⟩
      · exact Or.inl (hxy.trans hyz)
      · exact Or.inr ⟨s, t, hxs, p, Or.inr (TransGen.of_refl hty hyz)⟩
    · rcases ih with hxy | ⟨s, t, hxs, p, rfl | hty⟩
      · exact Or.inr ⟨_, _, Or.inr hxy, TransGen.single hyz, Or.inl rfl⟩
      · exact Or.inr ⟨s, _, hxs, p.tail hyz, Or.inl rfl⟩
      · have q := hb_psc wf hsc hnf (seqCst_of_psc_path p).2 (seqCst_of_psc hyz).1 hty
        exact Or.inr ⟨s, _, hxs, (p.trans q).tail hyz, Or.inl rfl⟩

include wf hsc hnf in
/-- DRF-SC for one execution: a consistent execution with no data race, whose atomics are all
sequentially consistent and which has no fences, is sequentially consistent. -/
theorem seqConsistent_of_raceFree (hc : X.Consistent) (hrace : X.RaceFree) :
    X.SeqConsistent where
  atomicity := hc.atomicity
  acyclic a hcyc := by
    rcases hbPsc_of_path wf hsc hnf hc hrace hcyc with h | ⟨s, t, has, p, hta⟩
    · exact hc.coherence a a h (Or.inl rfl)
    · have hts : t = s ∨ X.hb t s := by
        rcases hta with rfl | hta
        · exact has
        · rcases has with rfl | has
          · exact Or.inr hta
          · exact Or.inr (hta.trans has)
      rcases hts with rfl | hts
      · exact hc.seq_cst t p
      · have ⟨hs, ht⟩ := seqCst_of_psc_path p
        exact hc.seq_cst s (p.trans (hb_psc wf hsc hnf ht hs hts))

end NoFences

end Vx.Execution
