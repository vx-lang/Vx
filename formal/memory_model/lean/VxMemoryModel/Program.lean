import VxMemoryModel.DrfSc
import VxMemoryModel.Prefix
import VxMemoryModel.Finite

/-!
# DRF-SC for programs

A program is the set of its executions. The proof needs three facts about that set, which hold
for the programs of any language with threads:

- A prefix of an execution is an execution: the threads can stop at any point.
- The coherence order is any order: the program does not choose it.
- A read at the end of its thread can read any write to its place. Nothing after it depends on
  the value yet.

The theorem: if no interleaving of a program has a data race, then every execution that the
model allows is an interleaving. The atomics must be sequentially consistent, and there are no
fences. A place can have both atomic and plain accesses.

The proof takes a racy execution and its smallest racy prefix. Without its last event `m` the
prefix has no race, so it is an interleaving. Then `m` reads the latest write, or writes after
all other writes, and the result is an interleaving with the same race. That contradicts the
hypothesis.

If `m` is atomic, the race is with a plain access. If another event has nothing after it, it is
in the race too, and it is plain, so the proof uses it as `m`. Otherwise all events come before
`m`, and each read before `m` is ordered after its write by happens-before. Then an atomic
write `m` cannot race, and an atomic read `m` reads a plain write. The latest write is then
plain too, so `m` does not synchronize with it, and the race stays.
-/

namespace Vx

open Relation

namespace Execution

/-- The same events, with other reads-from and coherence edges. -/
abbrev withEdges (X : Execution) (rf co : X.E → X.E → Prop) : Execution :=
  { X with rf := rf, co := co }

/-- The reads-from edges after the read `r` takes its value from `w`. -/
def readsFrom (X : Execution) (r w : X.E) (a b : X.E) : Prop :=
  (b = r ∧ a = w) ∨ (b ≠ r ∧ X.rf a b)

def Finite (X : Execution) : Prop := ∃ l : List X.E, ∀ e, e ∈ l

/-- `co` is transitive, and it orders each two writes to one place. -/
structure CoOrder (X : Execution) : Prop where
  trans : ∀ a b c, X.co a b → X.co b c → X.co a c
  total : ∀ a b, X.IsWrite a → X.IsWrite b → X.loc a = X.loc b → a ≠ b → X.co a b ∨ X.co b a

theorem Finite.restrict {X : Execution} (h : X.Finite) (S : X.E → Prop) : (X.restrict S).Finite := by
  classical
  obtain ⟨l, hl⟩ := h
  refine ⟨l.filterMap fun x => if hx : S x then some ⟨x, hx⟩ else none, fun e => ?_⟩
  exact List.mem_filterMap.2 ⟨e.val, hl _, by simp [e.property]; rfl⟩

theorem CoOrder.restrict {X : Execution} (h : X.CoOrder) (S : X.E → Prop) :
    (X.restrict S).CoOrder where
  trans a b c := h.trans a.val b.val c.val
  total a b ha hb hl hne := h.total a.val b.val ha hb hl fun e => hne (Subtype.ext e)

end Execution

structure Program where
  Execs : Execution → Prop
  prefix_closed : ∀ X S, Execs X → X.Downclosed S → Execs (X.restrict S)
  any_co : ∀ X co, Execs X → Execs (X.withEdges X.rf co)
  receptive : ∀ X r w, Execs X → X.IsRead r → (∀ x, ¬ X.po r x) → X.IsWrite w →
    X.loc w = X.loc r → Execs (X.withEdges (X.readsFrom r w) X.co)

namespace Execution

variable {Y : Execution}

/-- The last write of a release sequence is atomic. -/
theorem rs_atomic {X : Execution} (hrmwa : ∀ r w, X.rmw r w → X.IsAtomic w) {a b : X.E}
    (h : X.rs a b) : X.IsAtomic b := by
  obtain ⟨w, _, _, _, hw, hwb⟩ := h
  rcases hwb with rfl | hwb
  · exact hw
  · cases hwb with
    | single h => obtain ⟨_, _, h⟩ := h; exact hrmwa _ _ h
    | tail _ h => obtain ⟨_, _, h⟩ := h; exact hrmwa _ _ h

/-- Change the edges into one event `m` that has nothing after it. If the other events form an
interleaving, the result is an interleaving, and a race of `m` stays a race. -/
theorem replace_last (m : Y.E) (rf' co' : Y.E → Y.E → Prop)
    (hrf : ∀ a b, b ≠ m → (rf' a b ↔ Y.rf a b))
    (hco : ∀ a b, a ≠ m → b ≠ m → (co' a b ↔ Y.co a b))
    (hout : ∀ x, ¬ (Y.withEdges rf' co').ord m x) (hpo : ∀ x, ¬ Y.po m x)
    (hin : ∀ y, rf' y m → ¬ Y.IsAcquire m ∨ ¬ Y.IsAtomic y) (hm : ∀ r, ¬ Y.rmw r m)
    (hrmw : ∀ r w, Y.rmw r w → Y.po r w) (hrmwa : ∀ r w, Y.rmw r w → Y.IsAtomic w)
    (hnf : ∀ e, ¬ Y.IsFence e) (hT : (Y.restrict (· ≠ m)).SeqConsistent)
    {p : Y.E} (hrace : Y.DataRace p m) :
    (Y.withEdges rf' co').SeqConsistent ∧ ¬ (Y.withEdges rf' co').RaceFree := by
  let G := Y.withEdges rf' co'
  let Z := G.restrict (· ≠ m)
  have hrf_m : ∀ x, ¬ rf' m x := fun x h => hout x (Or.inr (Or.inl h))
  let toT : Hom Z (Y.restrict (· ≠ m)) := {
    f := fun e => ⟨e.val, e.property⟩
    kind := fun _ => rfl
    order := fun _ => rfl
    loc := fun _ => rfl
    thread := fun _ => rfl
    po := fun _ _ h => h
    rf := fun a b h => (hrf a.val b.val b.property).1 h
    co := fun a b h => (hco a.val b.val a.property b.property).1 h
    rmw := fun _ _ h => h }
  have hZ : Z.SeqConsistent := toT.seqConsistent hT
  have hclosed : G.OrdClosed (· ≠ m) := by
    rintro a b _ hab rfl
    exact hout b hab
  have hdown : G.Downclosed (· ≠ m) := fun a b hb h =>
    hclosed a b hb (h.elim (fun h => Or.inl h) (fun h => Or.inr (Or.inl h)))
  refine ⟨⟨fun a hcyc => ?_, fun a b c hab hac hcb => ?_⟩, fun hfree => ?_⟩
  · have ha : a ≠ m := by
      rintro rfl
      obtain ⟨c, hc⟩ := hcyc.exists_first
      exact hout c hc
    obtain ⟨_, p⟩ := ord_reflect hclosed hdown hcyc ha
    exact hZ.acyclic _ p
  · have ha : a ≠ m := fun h => hpo b (h ▸ hrmw _ _ hab)
    have hb : b ≠ m := fun h => hm a (h ▸ hab)
    have hc : c ≠ m := by
      rintro rfl
      exact hout b (Or.inr (Or.inr (Or.inl hcb)))
    obtain ⟨w, hwa, hwc⟩ := hac
    have hw : w ≠ m := by
      rintro rfl
      exact hrf_m a hwa
    exact hZ.atomicity ⟨a, ha⟩ ⟨b, hb⟩ ⟨c, hc⟩ hab ⟨⟨w, hw⟩, hwa, hwc⟩ hcb
  · apply hfree p m
    refine ⟨hrace.1, hrace.2.1, fun hpm => ?_, fun hmp => ?_, hrace.2.2.2.2⟩
    · -- The last step into `m` is in program order: nothing synchronizes with `m`.
      have last : ∀ x, (G.po x m ∨ G.sw x m) → G.po x m := by
        rintro x (h | ⟨_, hacq, _, y, z, _, hxy, hyz, _, hzm⟩)
        · exact h
        · rcases hzm with rfl | ⟨_, hf⟩
          · exact (hin y hyz).elim (absurd hacq) (absurd (rs_atomic (X := G) hrmwa hxy))
          · exact absurd hf (hnf m)
      have back : ∀ x, x ≠ m → G.hb p x → Y.hb p x := by
        intro x hx h
        obtain ⟨_, h'⟩ := hb_reflect (X := G) hrmw hdown h hx
        exact (restrictHom Y _).hb (toT.hb h')
      cases hpm with
      | single h => exact hrace.2.2.1 (TransGen.single (Or.inl (last p h)))
      | @tail x _ h hx =>
        have hx := last _ hx
        have hxm : x ≠ m := fun e => hpo m (e ▸ hx)
        exact hrace.2.2.1 ((back _ hxm h).tail (Or.inl hx))
    · -- Nothing comes after `m`.
      obtain ⟨c, hc⟩ := hmp.exists_first
      rcases hc with h | ⟨_, _, x, y, z, hax, ⟨w, _, hxw, _, _, hwy⟩, hyz, _, _⟩
      · exact hpo c h
      · have hx : m = x := hax.resolve_right (fun h => hpo x h.2)
        subst hx
        have hw : m = w := hxw.resolve_right (fun h => hpo w h.1)
        subst hw
        rcases hwy with rfl | hwy
        · exact hrf_m z hyz
        · obtain ⟨_, r, hr, _⟩ := hwy.exists_first
          exact hrf_m r hr

theorem dataRace_symm {X : Execution} {a b : X.E} (h : X.DataRace a b) : X.DataRace b a := by
  obtain ⟨⟨hne, ha, hb, hl, hw⟩, ht, hab, hba, hat⟩ := h
  exact ⟨⟨fun e => hne e.symm, hb, ha, hl.symm, hw.symm⟩, fun e => ht e.symm, hba, hab,
    fun ⟨x, y⟩ => hat ⟨y, x⟩⟩

theorem mem_of_path {α : Type _} {S : α → Prop} {r : α → α → Prop} {a b : α}
    (hstep : ∀ x y, r x y → S y) (h : TransGen r a b) : S b := by
  cases h with
  | single h => exact hstep _ _ h
  | tail _ h => exact hstep _ _ h

end Execution

open Execution

/-- The same execution, with fewer events. -/
def Execution.restrictLe (X : Execution) {S T : X.E → Prop} (h : ∀ x, S x → T x) :
    Hom (X.restrict S) (X.restrict T) where
  f x := ⟨x.val, h _ x.property⟩
  kind _ := rfl
  order _ := rfl
  loc _ := rfl
  thread _ := rfl
  po _ _ h := h
  rf _ _ h := h
  co _ _ h := h
  rmw _ _ h := h

/-- A race between events of `S` in a larger set of events is a race in `S`. -/
theorem dataRace_restrictLe {X : Execution} {S T : X.E → Prop} (h : ∀ x, S x → T x)
    {a b : (X.restrict S).E} (hr : (X.restrict T).DataRace (X.restrictLe h |>.f a)
      (X.restrictLe h |>.f b)) : (X.restrict S).DataRace a b := by
  obtain ⟨⟨hne, h1, h2, h3, h4⟩, ht, hab, hba, hat⟩ := hr
  exact ⟨⟨fun e => hne (by rw [e]), h1, h2, h3, h4⟩, ht, fun hh => hab ((X.restrictLe h).hb hh),
    fun hh => hba ((X.restrictLe h).hb hh), hat⟩

/-- The step of the proof. `S` is a prefix, `m` is an event of `S` with nothing after it, and
`S` without `m` has no race. Then `S` has no race with `m`, when `m` is not atomic or when all
other events of `S` come before `m`. -/
theorem Program.no_race_with_last (P : Program) {X : Execution}
    (hP : ∀ Y, P.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.Finite → (X.CoOrder → Y.CoOrder) →
      Y.RaceFree)
    (hX : P.Execs X) (wf : X.WellFormed)
    (hsc : ∀ e, X.IsAtomic e → X.IsSeqCst e) (hnf : ∀ e, ¬ X.IsFence e) (hc : X.Consistent)
    {l : List X.E} (hl : ∀ e, e ∈ l) {S : X.E → Prop} (hS : X.Downclosed S) {m : X.E}
    (hmS : S m) (hmax' : ∀ x, S x → ¬ (X.po m x ∨ X.rf m x))
    (hfree' : (X.restrict (fun x => S x ∧ x ≠ m)).RaceFree)
    (hcase : ¬ X.IsAtomic m ∨
      ∀ x, S x → x = m ∨ TransGen (fun a b => S b ∧ (X.po a b ∨ X.rf a b)) x m)
    (p : (X.restrict S).E) (hrace : (X.restrict S).DataRace p ⟨m, hmS⟩) : False := by
  let S' : X.E → Prop := fun x => S x ∧ x ≠ m
  let Y := X.restrict S
  let mY : Y.E := ⟨m, hmS⟩
  let toY : Hom (X.restrict S') Y := X.restrictLe (fun _ h => h.1)
  -- The race involves `m`: the prefix without `m` has none.
  have hlift : ∀ a b : Y.E, a.val ≠ m → b.val ≠ m → ¬ Y.DataRace a b := fun a b ha hb h =>
    hfree' ⟨a.val, a.property, ha⟩ ⟨b.val, b.property, hb⟩ (dataRace_restrictLe (fun _ h => h.1) h)
  have wfY : Y.WellFormed := wf.restrict S
  have hYc : Y.Consistent := (restrictHom X S).consistent hc
  have hpoY : ∀ x : Y.E, ¬ Y.po mY x := fun x h => hmax' x.val x.property (Or.inl h)
  have hrfY : ∀ x : Y.E, ¬ Y.rf mY x := fun x h => hmax' x.val x.property (Or.inr h)
  have hnfY : ∀ x : Y.E, ¬ Y.IsFence x := fun x => hnf x.val
  have hS'sc : (X.restrict S').SeqConsistent :=
    seqConsistent_of_raceFree (wf.restrict S') (fun e h => hsc e.val h)
      (fun e => hnf e.val) ((restrictHom X S').consistent hc) hfree'
  have hT : (Y.restrict (· ≠ mY)).SeqConsistent := by
    let toS' : Hom (Y.restrict (· ≠ mY)) (X.restrict S') := {
      f := fun y => ⟨y.val.val, y.val.property, fun h => y.property (Subtype.ext h)⟩
      kind := fun _ => rfl
      order := fun _ => rfl
      loc := fun _ => rfl
      thread := fun _ => rfl
      po := fun _ _ h => h
      rf := fun _ _ h => h
      co := fun _ _ h => h
      rmw := fun _ _ h => h }
    exact toS'.seqConsistent hS'sc
  have hPY : P.Execs Y := P.prefix_closed X S hX hS
  have hfinY : Y.Finite := Finite.restrict ⟨l, hl⟩ S
  have hpm : p ≠ mY := hrace.1.1
  have hpm' : p.val ≠ m := fun h => hpm (Subtype.ext h)
  have hrmwa : ∀ r w, Y.rmw r w → Y.IsAtomic w := fun r w h => (wfY.rmw_atomic r w h).2.2.2
  -- Two sequentially consistent atomics that read from each other synchronize.
  have sw_rf : ∀ a b : Y.E, Y.rf a b → Y.IsAtomic a → Y.IsAtomic b → Y.sw a b :=
    fun a b hab ha hb =>
      ⟨Or.inr (Or.inr (hsc _ ha)), Or.inr (Or.inr (hsc _ hb)), a, a, b, Or.inl rfl,
        ⟨a, (wfY.rf_write_read _ _ hab).1, Or.inl rfl, (wfY.rf_write_read _ _ hab).1, ha,
          Or.inl rfl⟩, hab, hb, Or.inl rfl⟩
  -- A read that is not `m` is ordered after its write by happens-before: if not, they race.
  have covered : ∀ a b : Y.E, a ≠ mY → b ≠ mY → Y.rf a b → Y.hb a b := by
    intro a b ham hbm hab
    have ⟨haW, hbR, hl⟩ := wfY.rf_write_read _ _ hab
    have hne : a ≠ b := fun h => by
      rw [h] at haW
      exact Kind.noConfusion (haW.symm.trans hbR)
    by_cases ht : Y.thread a = Y.thread b
    · rcases wfY.po_total a b ht hne with h | h
      · exact TransGen.single (Or.inl h)
      · exact absurd (TransGen.head (r := fun x y => Y.po x y ∨ Y.rf x y) (Or.inr hab)
          (TransGen.single (Or.inl h)))
          (hYc.no_thin_air a)
    by_cases hh : Y.hb a b
    · exact hh
    exfalso
    by_cases hba : Y.hb b a
    · exact hYc.coherence b a hba (Or.inr (TransGen.single (Or.inl hab)))
    by_cases hat : Y.IsAtomic a ∧ Y.IsAtomic b
    · exact hh (TransGen.single (Or.inr (sw_rf a b hab hat.1 hat.2)))
    · exact hlift a b (fun h => ham (Subtype.ext h)) (fun h => hbm (Subtype.ext h))
        ⟨⟨hne, Or.inr haW, Or.inl hbR, hl, Or.inl haW⟩, ht, hh, hba, hat⟩
  -- A path of program order and reads to `b` is in happens-before, except for a last read
  -- into `m`.
  let R : X.E → X.E → Prop := fun a b => S b ∧ (X.po a b ∨ X.rf a b)
  have path : ∀ a b, TransGen R a b → ∀ (ha : S a) (hb : S b),
      Y.hb ⟨a, ha⟩ ⟨b, hb⟩ ∨ (b = m ∧ ∃ w : Y.E, (⟨a, ha⟩ = w ∨ Y.hb ⟨a, ha⟩ w) ∧ Y.rf w mY) := by
    intro a b h
    induction h with
    | @single b h =>
      intro ha hb
      have ham : a ≠ m := fun e => hmax' b hb (e ▸ h.2)
      rcases h.2 with hp | hr
      · exact Or.inl (TransGen.single (Or.inl hp))
      · by_cases hbm : b = m
        · exact Or.inr ⟨hbm, ⟨a, ha⟩, Or.inl rfl, by show X.rf a m; rw [← hbm]; exact hr⟩
        · exact Or.inl (covered ⟨a, ha⟩ ⟨b, hb⟩ (fun e => ham (congrArg Subtype.val e))
            (fun e => hbm (congrArg Subtype.val e)) hr)
    | @tail c b _ hstep ih =>
      intro ha hb
      have hcS : S c := hS c b hb hstep.2
      have hcm : c ≠ m := fun e => hmax' b hb (e ▸ hstep.2)
      have hac : Y.hb ⟨a, ha⟩ ⟨c, hcS⟩ := (ih ha hcS).resolve_right (fun h => hcm h.1)
      rcases hstep.2 with hp | hr
      · exact Or.inl (hac.tail (Or.inl hp))
      · by_cases hbm : b = m
        · exact Or.inr ⟨hbm, ⟨c, hcS⟩, Or.inr hac, by show X.rf c m; rw [← hbm]; exact hr⟩
        · exact Or.inl (hac.trans (covered ⟨c, hcS⟩ ⟨b, hb⟩
            (fun e => hcm (congrArg Subtype.val e)) (fun e => hbm (congrArg Subtype.val e)) hr))
  rcases hrace.1.2.2.1 with hR | hW
  · -- `m` reads: it reads the last write to its place.
    have hpW : Y.IsWrite p := hrace.1.2.2.2.2.resolve_right (fun h =>
      Kind.noConfusion (h.symm.trans hR))
    let R2 : X.E → X.E → Prop := fun a b => S' a ∧ S' b ∧ X.co a b
    have hacyc2 : ∀ a, ¬ TransGen R2 a a := by
      intro a h
      obtain ⟨_, h'⟩ := transGen_reflect (r' := (X.restrict S').ord)
        (fun x y hy hxy => ⟨hxy.1, Or.inr (Or.inr (Or.inl hxy.2.2))⟩) h
        (mem_of_path (fun _ _ h => h.2.1) h)
      exact hS'sc.acyclic _ h'
    have hpath : ∀ w0 b, X.IsWrite w0 → TransGen R2 w0 b → X.IsWrite b ∧ X.loc b = X.loc w0 := by
      intro w0 b _ hpb
      induction hpb with
      | single h =>
        have ⟨_, hb, hl⟩ := wf.co_writes _ _ h.2.2
        exact ⟨hb, hl.symm⟩
      | tail _ h ih =>
        have ⟨_, hb, hl⟩ := wf.co_writes _ _ h.2.2
        exact ⟨hb, hl.symm.trans ih.2⟩
    have hpS' : S' p.val := ⟨p.property, hpm'⟩
    -- The search for the last write starts at `p`. When `m` is atomic, it starts at the write
    -- that `m` reads, which is not atomic. The last write is then not atomic either.
    obtain ⟨w0, hw0S', hw0W, hw0loc, hgood⟩ : ∃ w0, S' w0 ∧ X.IsWrite w0 ∧ X.loc w0 = X.loc m ∧
        ∀ w, (w0 = w ∨ TransGen R2 w0 w) → S' w → ¬ X.IsAtomic m ∨ ¬ X.IsAtomic w := by
      by_cases hmA : X.IsAtomic m
      · have huniq := hcase.resolve_left (fun h => h hmA)
        rcases path _ _ ((huniq p.val p.property).resolve_left hpm') p.property hmS with
          hh | ⟨_, w, hpw, hwm⟩
        · exact absurd hh hrace.2.2.1
        have hwA : ¬ Y.IsAtomic w := fun hwA => hrace.2.2.1 (by
          have hsw : Y.hb w mY := TransGen.single (Or.inr (sw_rf w mY hwm hwA hmA))
          rcases hpw with h | h
          · rw [← h] at hsw
            exact hsw
          · exact h.trans hsw)
        have ⟨hwW, _, hwl⟩ := wfY.rf_write_read _ _ hwm
        have hwS' : S' w.val := ⟨w.property, fun h => by
          have : Y.kind w = Kind.write := hwW
          rw [show w = mY from Subtype.ext h] at this
          exact Kind.noConfusion (this.symm.trans hR)⟩
        refine ⟨w.val, hwS', hwW, hwl, fun wY hwY hwYS' => Or.inr fun hwYA => ?_⟩
        have hne : w.val ≠ wY := fun h => hwA (by show X.IsAtomic w.val; rw [h]; exact hwYA)
        have hchain : TransGen R2 w.val wY := hwY.resolve_left hne
        -- `m` reads `w`, and `wY` comes after `w` in coherence order.
        have heco : ∀ b, TransGen R2 w.val b → X.eco m b := by
          intro b h
          induction h with
          | single h => exact TransGen.single (Or.inr (Or.inr ⟨w.val, hwm, h.2.2⟩))
          | tail _ h ih => exact ih.tail (Or.inr (Or.inl h.2.2))
        rcases path _ _ ((huniq wY hwYS'.1).resolve_left hwYS'.2) hwYS'.1 hmS with
          hh | ⟨_, w', hw', hw'm⟩
        · exact hc.coherence _ _ ((restrictHom X S).hb hh) (Or.inr (heco _ hchain))
        · have hw'w : w' = w := wfY.rf_unique _ _ _ hw'm hwm
          rw [hw'w] at hw'
          rcases hw' with h | h
          · exact hne (congrArg Subtype.val h).symm
          · exact hc.coherence _ _ ((restrictHom X S).hb h)
              (Or.inr (TransGen.mono (fun _ _ h => Or.inr (Or.inl h.2.2)) hchain))
      · exact ⟨p.val, hpS', hpW, hrace.1.2.2.2.1, fun _ _ _ => Or.inl hmA⟩
    obtain ⟨w, hpw, hwmax⟩ := exists_maximal hl hacyc2 w0
    have hwS' : S' w := hpw.elim (fun h => h ▸ hw0S') (mem_of_path (fun _ _ h => h.2.1))
    have hwl : X.IsWrite w ∧ X.loc w = X.loc m := by
      rcases hpw with rfl | hpw
      · exact ⟨hw0W, hw0loc⟩
      · have := hpath _ _ hw0W hpw
        exact ⟨this.1, this.2.trans hw0loc⟩
    have hgw := hgood w hpw hwS'
    let wY : Y.E := ⟨w, hwS'.1⟩
    have hwm : wY ≠ mY := fun h => hwS'.2 (congrArg Subtype.val h)
    have hwlast : ∀ c : Y.E, ¬ Y.co wY c := by
      intro c h
      have hcW := (wf.co_writes _ _ h).2.1
      refine hwmax c.val ⟨hwS', ⟨c.property, fun e => ?_⟩, h⟩
      have hmW : X.kind m = Kind.write := e ▸ hcW
      exact Kind.noConfusion (hmW.symm.trans hR)
    have hloc : Y.loc wY = Y.loc mY := hwl.2
    let G := Y.withEdges (Y.readsFrom mY wY) Y.co
    have wfG : G.WellFormed := {
      po_trans := wfY.po_trans
      po_total := wfY.po_total
      rf_write_read := fun a b h => by
        rcases h with ⟨rfl, rfl⟩ | ⟨_, h⟩
        · exact ⟨hwl.1, hR, hloc⟩
        · exact wfY.rf_write_read _ _ h
      rf_unique := fun a a' b h h' => by
        rcases h with ⟨hb, rfl⟩ | ⟨hb, h⟩ <;> rcases h' with ⟨hb', rfl⟩ | ⟨hb', h'⟩
        · rfl
        · exact absurd hb hb'
        · exact absurd hb' hb
        · exact wfY.rf_unique _ _ _ h h'
      co_writes := wfY.co_writes
      co_irrefl := wfY.co_irrefl
      rmw_po := wfY.rmw_po
      rmw_atomic := wfY.rmw_atomic }
    have ⟨hGsc, hGracy⟩ := replace_last (Y := Y) mY (Y.readsFrom mY wY) Y.co
      (fun a b hb => ⟨fun h => h.elim (fun h => absurd h.1 hb) (·.2),
        fun h => Or.inr ⟨hb, h⟩⟩)
      (fun _ _ _ _ => Iff.rfl)
      (by
        rintro x (h | h | h | ⟨w0, hw0, hw0x⟩)
        · exact hpoY x h
        · rcases h with ⟨_, h⟩ | ⟨_, h⟩
          · exact hwm h.symm
          · exact hrfY x h
        · exact Kind.noConfusion ((wf.co_writes _ _ h).1.symm.trans hR)
        · rcases hw0 with ⟨_, rfl⟩ | ⟨h, _⟩
          · exact hwlast x hw0x
          · exact h rfl)
      hpoY
      (fun y h => by
        have hy : y = wY := h.elim (·.2) (fun h => absurd rfl h.1)
        rw [hy]
        exact hgw.imp (fun hna hacq => hna (atomic_of_acquire hacq)) id)
      (fun r h => Kind.noConfusion ((wfY.rmw_atomic _ _ h).2.1.symm.trans hR))
      wfY.rmw_po hrmwa hnfY hT hrace
    exact hGracy (hP G (P.receptive Y mY wY hPY hR hpoY hwl.1 hloc) wfG hGsc hfinY
      fun ho => let h := ho.restrict S; ⟨h.trans, h.total⟩)
  · by_cases hmA : X.IsAtomic m
    · -- An atomic write that comes after all other events comes after `p`.
      have huniq := hcase.resolve_left (fun h => h hmA)
      rcases path _ _ ((huniq p.val p.property).resolve_left hpm') p.property hmS with
        hh | ⟨_, w, _, hwm⟩
      · exact hrace.2.2.1 hh
      · exact Kind.noConfusion ((wfY.rf_write_read _ _ hwm).2.1.symm.trans hW)
    -- `m` writes: it comes after all other writes to its place.
    let co'' : Y.E → Y.E → Prop := fun a b =>
      (a ≠ mY ∧ b ≠ mY ∧ Y.co a b) ∨ (b = mY ∧ a ≠ mY ∧ Y.IsWrite a ∧ Y.loc a = Y.loc mY)
    let G := Y.withEdges Y.rf co''
    have wfG : G.WellFormed := {
      po_trans := wfY.po_trans
      po_total := wfY.po_total
      rf_write_read := wfY.rf_write_read
      rf_unique := wfY.rf_unique
      co_writes := fun a b h => by
        rcases h with ⟨_, _, h⟩ | ⟨rfl, _, ha, hl⟩
        · exact wfY.co_writes _ _ h
        · exact ⟨ha, hW, hl⟩
      co_irrefl := fun a h => by
        rcases h with ⟨_, _, h⟩ | ⟨rfl, h, _⟩
        · exact wfY.co_irrefl a h
        · exact h rfl
      rmw_po := wfY.rmw_po
      rmw_atomic := wfY.rmw_atomic }
    have ⟨hGsc, hGracy⟩ := replace_last (Y := Y) mY Y.rf co''
      (fun _ _ _ => Iff.rfl)
      (fun a b ha hb => ⟨fun h => h.elim (·.2.2) (fun h => absurd h.1 hb),
        fun h => Or.inl ⟨ha, hb, h⟩⟩)
      (by
        rintro x (h | h | h | ⟨w0, hw0, _⟩)
        · exact hpoY x h
        · exact hrfY x h
        · rcases h with ⟨h, _⟩ | ⟨_, h, _⟩
          · exact h rfl
          · exact h rfl
        · exact Kind.noConfusion ((wfY.rf_write_read _ _ hw0).2.1.symm.trans hW))
      hpoY
      (fun y h => absurd hW (fun hw =>
        Kind.noConfusion ((wfY.rf_write_read _ _ h).2.1.symm.trans hw)))
      (fun r h => hmA (hrmwa _ _ h)) wfY.rmw_po hrmwa hnfY hT hrace
    -- `co''` keeps the order of `co`, and puts `m` after each other write to its place.
    have hcoG : X.CoOrder → G.CoOrder := fun ho =>
      let h := ho.restrict S
      { trans := fun a b c hab hbc => by
          rcases hab with ⟨ha, hb, hab⟩ | ⟨rfl, -, -, -⟩
          · rcases hbc with ⟨-, hc, hbc⟩ | ⟨rfl, -, -, hbl⟩
            · exact Or.inl ⟨ha, hc, h.trans a b c hab hbc⟩
            · obtain ⟨haW, -, hl⟩ := wfY.co_writes a b hab
              exact Or.inr ⟨rfl, ha, haW, hl.trans hbl⟩
          · rcases hbc with ⟨h', -, -⟩ | ⟨-, h', -, -⟩ <;> exact absurd rfl h'
        total := fun a b ha hb hl hne => by
          by_cases ham : a = mY
          · subst ham
            exact Or.inr (Or.inr ⟨rfl, fun e => hne e.symm, hb, hl.symm⟩)
          by_cases hbm : b = mY
          · subst hbm
            exact Or.inl (Or.inr ⟨rfl, ham, ha, hl⟩)
          rcases h.total a b ha hb hl hne with h | h
          · exact Or.inl (Or.inl ⟨ham, hbm, h⟩)
          · exact Or.inr (Or.inl ⟨hbm, ham, h⟩) }
    exact hGracy (hP G (P.any_co Y co'' hPY) wfG hGsc hfinY hcoG)

/-- The proof of DRF-SC. The interleavings that it asks to be race free are finite, and their
`co` orders the writes to each place when the `co` of `X` does. -/
theorem Program.drf_sc_core (P : Program) {X : Execution}
    (hP : ∀ Y, P.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.Finite → (X.CoOrder → Y.CoOrder) →
      Y.RaceFree)
    (hX : P.Execs X) (wf : X.WellFormed) (hfin : X.Finite)
    (hsc : ∀ e, X.IsAtomic e → X.IsSeqCst e)
    (hnf : ∀ e, ¬ X.IsFence e) (hc : X.Consistent) : X.SeqConsistent := by
  obtain ⟨l, hl⟩ := hfin
  have key : ∀ n S, X.Downclosed S → count l S ≤ n → (X.restrict S).RaceFree := by
    intro n
    induction n with
    | zero =>
      intro S _ hn a _ _
      have := count_pos (hl a.val) a.property
      omega
    | succ n ih =>
      intro S hS hn d e hde
      let R : X.E → X.E → Prop := fun a b => S b ∧ (X.po a b ∨ X.rf a b)
      have hacyc : ∀ a, ¬ TransGen R a a := fun a h =>
        hc.no_thin_air a (TransGen.mono (fun _ _ h => h.2) h)
      let Last : X.E → Prop := fun x => S x ∧ ∀ y, S y → ¬ (X.po x y ∨ X.rf x y)
      -- Without one event with nothing after it, the prefix is smaller, so it has no race.
      have hfree : ∀ x, Last x → (X.restrict (fun y => S y ∧ y ≠ x)).RaceFree := by
        intro x ⟨hxS, hxmax⟩
        apply ih
        · intro a b hb h
          refine ⟨hS a b hb.1 h, ?_⟩
          rintro rfl
          exact hxmax b hb.1 h
        · have := count_lt (p := fun y => S y ∧ y ≠ x) (fun y h => h.1) (hl x) hxS
            (fun h => h.2 rfl)
          omega
      -- So each race of the prefix involves each event with nothing after it.
      have hinv : ∀ x, Last x → ∀ a b : (X.restrict S).E, (X.restrict S).DataRace a b →
          a.val = x ∨ b.val = x := by
        intro x hx a b hab
        refine Classical.byContradiction fun hne => ?_
        obtain ⟨ha, hb⟩ := not_or.1 hne
        exact hfree x hx ⟨a.val, a.property, ha⟩ ⟨b.val, b.property, hb⟩
          (dataRace_restrictLe (fun _ h => h.1) hab)
      obtain ⟨m, hem, hmax⟩ := exists_maximal hl hacyc e.val
      have hmS : S m := hem.elim (fun h => h ▸ e.property) (mem_of_path (fun _ _ h => h.1))
      have hmL : Last m := ⟨hmS, fun x hx h => hmax x ⟨hx, h⟩⟩
      obtain ⟨p, hrace⟩ : ∃ p, (X.restrict S).DataRace p ⟨m, hmS⟩ := by
        rcases hinv m hmL d e hde with hd | he
        · exact ⟨e, (show d = ⟨m, hmS⟩ from Subtype.ext hd) ▸ dataRace_symm hde⟩
        · exact ⟨d, (show e = ⟨m, hmS⟩ from Subtype.ext he) ▸ hde⟩
      by_cases hmA : X.IsAtomic m
      · by_cases h2 : ∃ m2, Last m2 ∧ m2 ≠ m
        · -- Another last event `m2` is in the race too. It is not atomic, so it is the `m`
          -- of the step.
          obtain ⟨m2, hm2, hm2m⟩ := h2
          have hp : p.val = m2 := (hinv m2 hm2 p _ hrace).resolve_right (fun h => hm2m h.symm)
          have hp' : p = ⟨m2, hm2.1⟩ := Subtype.ext hp
          rw [hp'] at hrace
          exact P.no_race_with_last hP hX wf hsc hnf hc hl hS hm2.1 hm2.2 (hfree m2 hm2)
            (Or.inl fun h => hrace.2.2.2.2 ⟨h, hmA⟩) ⟨m, hmS⟩ (dataRace_symm hrace)
        · -- `m` is the only last event, so all other events come before it.
          refine P.no_race_with_last hP hX wf hsc hnf hc hl hS hmS hmL.2 (hfree m hmL)
            (Or.inr fun x hx => ?_) p hrace
          obtain ⟨m', hxm', hm'⟩ := exists_maximal hl hacyc x
          have hm'S : S m' := hxm'.elim (fun h => h ▸ hx) (mem_of_path (fun _ _ h => h.1))
          have : m' = m := Classical.byContradiction fun hne =>
            h2 ⟨m', ⟨hm'S, fun y hy h => hm' y ⟨hy, h⟩⟩, hne⟩
          exact this ▸ hxm'
      · exact P.no_race_with_last hP hX wf hsc hnf hc hl hS hmS hmL.2 (hfree m hmL)
          (Or.inl hmA) p hrace
  have hrace : X.RaceFree := by
    intro a b ⟨hcnf, ht, hab, hba, hat⟩
    have hdown : X.Downclosed (fun _ => True) := fun _ _ _ _ => trivial
    refine key l.length (fun _ => True) hdown (List.countP_le_length) ⟨a, trivial⟩ ⟨b, trivial⟩
      ⟨⟨fun h => hcnf.1 (congrArg Subtype.val h), hcnf.2⟩, ht,
        fun h => hab ((restrictHom X _).hb h), fun h => hba ((restrictHom X _).hb h), hat⟩
  exact seqConsistent_of_raceFree wf hsc hnf hc hrace

/-- DRF-SC for programs. If no interleaving of `P` has a data race, then every execution of `P`
that the model allows is an interleaving. -/
theorem Program.drf_sc (P : Program)
    (hP : ∀ Y, P.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.RaceFree)
    {X : Execution} (hX : P.Execs X) (wf : X.WellFormed) (hfin : X.Finite)
    (hsc : ∀ e, X.IsAtomic e → X.IsSeqCst e)
    (hnf : ∀ e, ¬ X.IsFence e) (hc : X.Consistent) : X.SeqConsistent :=
  P.drf_sc_core (fun Y hY wf hsc _ _ => hP Y hY wf hsc) hX wf hfin hsc hnf hc

/-- DRF-SC for programs, when `co` orders the writes to each place, as RC11 asks. Then only the
finite interleavings in which `co` orders the writes to each place need to be race free. -/
theorem Program.drf_sc_co (P : Program)
    (hP : ∀ Y, P.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.Finite → Y.CoOrder → Y.RaceFree)
    {X : Execution} (hX : P.Execs X) (wf : X.WellFormed) (hfin : X.Finite) (ho : X.CoOrder)
    (hsc : ∀ e, X.IsAtomic e → X.IsSeqCst e)
    (hnf : ∀ e, ¬ X.IsFence e) (hc : X.Consistent) : X.SeqConsistent :=
  P.drf_sc_core (fun Y hY wf hsc hf hco => hP Y hY wf hsc hf (hco ho)) hX wf hfin hsc hnf hc

end Vx
