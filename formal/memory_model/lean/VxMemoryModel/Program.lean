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
model allows is an interleaving. The atomics must be sequentially consistent, there are no
fences, and each place is either atomic or not atomic.

The proof takes a racy execution and its smallest racy prefix. Without its last event `m` the
prefix has no race, so it is an interleaving. Then `m` reads the latest write, or writes after
all other writes, and the result is an interleaving with the same race. That contradicts the
hypothesis.
-/

namespace Vx

open Relation

namespace Execution

/-- The same events, with other reads-from and coherence edges. -/
def withEdges (X : Execution) (rf co : X.E → X.E → Prop) : Execution :=
  { X with rf := rf, co := co }

/-- The reads-from edges after the read `r` takes its value from `w`. -/
def readsFrom (X : Execution) (r w : X.E) (a b : X.E) : Prop :=
  (b = r ∧ a = w) ∨ (b ≠ r ∧ X.rf a b)

/-- Each place is either atomic or not atomic: all its accesses agree. -/
def TypedLocations (X : Execution) : Prop :=
  ∀ a b, X.IsAccess a → X.IsAccess b → X.loc a = X.loc b → (X.IsAtomic a ↔ X.IsAtomic b)

def Finite (X : Execution) : Prop := ∃ l : List X.E, ∀ e, e ∈ l

end Execution

structure Program where
  Execs : Execution → Prop
  prefix_closed : ∀ X S, Execs X → X.Downclosed S → Execs (X.restrict S)
  any_co : ∀ X co, Execs X → Execs (X.withEdges X.rf co)
  receptive : ∀ X r w, Execs X → X.IsRead r → (∀ x, ¬ X.po r x) → X.IsWrite w →
    X.loc w = X.loc r → Execs (X.withEdges (X.readsFrom r w) X.co)

namespace Execution

variable {Y : Execution}

/-- Change the edges into one event `m` that has nothing after it. If the other events form an
interleaving, the result is an interleaving, and a race of `m` stays a race. -/
theorem replace_last (m : Y.E) (rf' co' : Y.E → Y.E → Prop)
    (hrf : ∀ a b, b ≠ m → (rf' a b ↔ Y.rf a b))
    (hco : ∀ a b, a ≠ m → b ≠ m → (co' a b ↔ Y.co a b))
    (hout : ∀ x, ¬ (Y.withEdges rf' co').ord m x) (hpo : ∀ x, ¬ Y.po m x)
    (hin : ∀ y, rf' y m → ¬ Y.IsAcquire m) (hm : ¬ Y.IsAtomic m)
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
    have hb : b ≠ m := fun h => hm (h ▸ hrmwa _ _ hab)
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
        rintro x (h | ⟨_, hacq, _, y, z, _, _, hyz, _, hzm⟩)
        · exact h
        · rcases hzm with rfl | ⟨_, hf⟩
          · exact absurd hacq (hin y hyz)
          · exact absurd hf (hnf m)
      have back : ∀ x, x ≠ m → G.hb p x → Y.hb p x := by
        intro x hx h
        obtain ⟨_, h'⟩ := hb_reflect (X := G) hrmw hdown h hx
        exact (restrictHom Y _).hb (toT.hb h')
      cases hpm with
      | single h => exact hrace.2.2.1 (TransGen.single (Or.inl (last p h)))
      | tail h hx =>
        have hx := last _ hx
        have hxm : _ ≠ m := fun e => hpo m (e ▸ hx)
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

/-- DRF-SC for programs. If no interleaving of `P` has a data race, then every execution of `P`
that the model allows is an interleaving. -/
theorem Program.drf_sc (P : Program)
    (hP : ∀ Y, P.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.RaceFree)
    {X : Execution} (hX : P.Execs X) (wf : X.WellFormed) (hfin : X.Finite)
    (htyped : X.TypedLocations) (hsc : ∀ e, X.IsAtomic e → X.IsSeqCst e)
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
      -- `m` is an event of the prefix with nothing after it.
      let R : X.E → X.E → Prop := fun a b => S b ∧ (X.po a b ∨ X.rf a b)
      have hacyc : ∀ a, ¬ TransGen R a a := fun a h =>
        hc.no_thin_air a (TransGen.mono (fun _ _ h => h.2) h)
      obtain ⟨m, hem, hmax⟩ := exists_maximal hl hacyc e.val
      have hmS : S m := hem.elim (fun h => h ▸ e.property) (mem_of_path (fun _ _ h => h.1))
      have hmax' : ∀ x, S x → ¬ (X.po m x ∨ X.rf m x) := fun x hx h => hmax x ⟨hx, h⟩
      let S' : X.E → Prop := fun x => S x ∧ x ≠ m
      have hS' : X.Downclosed S' := by
        intro a b hb h
        refine ⟨hS a b hb.1 h, ?_⟩
        rintro rfl
        exact hmax' b hb.1 h
      have hfree' : (X.restrict S').RaceFree :=
        ih S' hS' (by
          have := count_lt (p := S') (fun x h => h.1) (hl m) hmS (fun h => h.2 rfl)
          omega)
      let Y := X.restrict S
      let mY : Y.E := ⟨m, hmS⟩
      let toY : Hom (X.restrict S') Y := {
        f := fun x => ⟨x.val, x.property.1⟩
        kind := fun _ => rfl
        order := fun _ => rfl
        loc := fun _ => rfl
        thread := fun _ => rfl
        po := fun _ _ h => h
        rf := fun _ _ h => h
        co := fun _ _ h => h
        rmw := fun _ _ h => h }
      -- The race involves `m`: the prefix without `m` has none.
      have hlift : ∀ a b : Y.E, a.val ≠ m → b.val ≠ m → ¬ Y.DataRace a b := by
        intro a b ha hb ⟨⟨hne, h1, h2, h3, h4⟩, ht, hab, hba, hat⟩
        exact hfree' ⟨a.val, a.property, ha⟩ ⟨b.val, b.property, hb⟩
          ⟨⟨fun h => hne (Subtype.ext (Subtype.mk.inj h)), h1, h2, h3, h4⟩, ht,
            fun h => hab (toY.hb h), fun h => hba (toY.hb h), hat⟩
      obtain ⟨p, hrace⟩ : ∃ p, Y.DataRace p mY := by
        by_cases hd : d.val = m
        · have : d = mY := Subtype.ext hd
          exact ⟨e, this ▸ dataRace_symm hde⟩
        · by_cases he : e.val = m
          · have : e = mY := Subtype.ext he
            exact ⟨d, this ▸ hde⟩
          · exact absurd hde (hlift d e hd he)
      have wfY : Y.WellFormed := wf.restrict S
      have hpoY : ∀ x : Y.E, ¬ Y.po mY x := fun x h => hmax' x.val x.property (Or.inl h)
      have hrfY : ∀ x : Y.E, ¬ Y.rf mY x := fun x h => hmax' x.val x.property (Or.inr h)
      have hnfY : ∀ x : Y.E, ¬ Y.IsFence x := fun x => hnf x.val
      have hmNA : ¬ Y.IsAtomic mY := by
        obtain ⟨⟨_, hpa, hma, hloc, _⟩, _, _, _, hat⟩ := hrace
        exact fun h => hat ⟨(htyped _ _ hpa hma hloc).2 h, h⟩
      have hT : (Y.restrict (· ≠ mY)).SeqConsistent := by
        have hS'sc : (X.restrict S').SeqConsistent :=
          seqConsistent_of_raceFree (wf.restrict S') (fun e h => hsc e.val h)
            (fun e => hnf e.val) ((restrictHom X S').consistent hc) hfree'
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
      have hpm : p ≠ mY := hrace.1.1
      have hrmwa : ∀ r w, Y.rmw r w → Y.IsAtomic w := fun r w h => (wfY.rmw_atomic r w h).2.2.2
      rcases hrace.1.2.2.1 with hR | hW
      · -- `m` reads: it reads the last write to its place.
        have hpW : Y.IsWrite p := hrace.1.2.2.2.2.resolve_right (fun h =>
          Kind.noConfusion (h.symm.trans hR))
        let R2 : X.E → X.E → Prop := fun a b => S' a ∧ S' b ∧ X.co a b
        have hS'sc : (X.restrict S').SeqConsistent :=
          seqConsistent_of_raceFree (wf.restrict S') (fun e h => hsc e.val h)
            (fun e => hnf e.val) ((restrictHom X S').consistent hc) hfree'
        have hacyc2 : ∀ a, ¬ TransGen R2 a a := by
          intro a h
          obtain ⟨_, h'⟩ := transGen_reflect (r' := (X.restrict S').ord)
            (fun x y hy hxy => ⟨hxy.1, Or.inr (Or.inr (Or.inl hxy.2.2))⟩) h
            (mem_of_path (fun _ _ h => h.2.1) h)
          exact hS'sc.acyclic _ h'
        have hpath : ∀ b, TransGen R2 p.val b → X.IsWrite b ∧ X.loc b = X.loc p.val := by
          intro b hpb
          induction hpb with
          | single h =>
            have ⟨_, hb, hl⟩ := wf.co_writes _ _ h.2.2
            exact ⟨hb, hl.symm⟩
          | tail _ h ih =>
            have ⟨_, hb, hl⟩ := wf.co_writes _ _ h.2.2
            exact ⟨hb, hl.symm.trans ih.2⟩
        obtain ⟨w, hpw, hwmax⟩ := exists_maximal hl hacyc2 p.val
        have hpS' : S' p.val := ⟨p.property, fun h => hpm (Subtype.ext h)⟩
        have hwS' : S' w := hpw.elim (fun h => h ▸ hpS') (mem_of_path (fun _ _ h => h.2.1))
        have hwl : X.IsWrite w ∧ X.loc w = X.loc p.val := by
          rcases hpw with rfl | hpw
          · exact ⟨hpW, rfl⟩
          · exact hpath w hpw
        let wY : Y.E := ⟨w, hwS'.1⟩
        have hwm : wY ≠ mY := fun h => hwS'.2 (congrArg Subtype.val h)
        have hwlast : ∀ c : Y.E, ¬ Y.co wY c := by
          intro c h
          have hcW := (wf.co_writes _ _ h).2.1
          refine hwmax c.val ⟨hwS', ⟨c.property, fun e => ?_⟩, h⟩
          have hmW : X.kind m = Kind.write := e ▸ hcW
          exact Kind.noConfusion (hmW.symm.trans hR)
        have hloc : Y.loc wY = Y.loc mY := hwl.2.trans hrace.1.2.2.2.1
        let G := Y.withEdges (Y.readsFrom mY wY) Y.co
        have wfG : G.WellFormed := {
          po_trans := wfY.po_trans
          po_total := wfY.po_total
          rf_write_read := fun a b h => by
            rcases h with ⟨rfl, rfl⟩ | ⟨_, h⟩
            · exact ⟨hwl.1, hR, hloc⟩
            · exact wfY.rf_write_read _ _ h
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
          hpoY (fun _ _ h => hmNA (atomic_of_acquire h)) hmNA wfY.rmw_po hrmwa hnfY hT hrace
        exact hGracy (hP G (P.receptive Y mY wY hPY hR hpoY hwl.1 hloc) wfG hGsc)
      · -- `m` writes: it comes after all other writes to its place.
        let co'' : Y.E → Y.E → Prop := fun a b =>
          (a ≠ mY ∧ b ≠ mY ∧ Y.co a b) ∨ (b = mY ∧ a ≠ mY ∧ Y.IsWrite a ∧ Y.loc a = Y.loc mY)
        let G := Y.withEdges Y.rf co''
        have wfG : G.WellFormed := {
          po_trans := wfY.po_trans
          po_total := wfY.po_total
          rf_write_read := wfY.rf_write_read
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
          hpoY (fun y h => absurd hW (fun hw =>
            Kind.noConfusion ((wfY.rf_write_read _ _ h).2.1.symm.trans hw)))
          hmNA wfY.rmw_po hrmwa hnfY hT hrace
        exact hGracy (hP G (P.any_co Y co'' hPY) wfG hGsc)
  have hrace : X.RaceFree := by
    intro a b ⟨hcnf, ht, hab, hba, hat⟩
    have hdown : X.Downclosed (fun _ => True) := fun _ _ _ _ => trivial
    refine key l.length (fun _ => True) hdown (List.countP_le_length) ⟨a, trivial⟩ ⟨b, trivial⟩
      ⟨⟨fun h => hcnf.1 (congrArg Subtype.val h), hcnf.2⟩, ht,
        fun h => hab ((restrictHom X _).hb h), fun h => hba ((restrictHom X _).hb h), hat⟩
  exact seqConsistent_of_raceFree wf hsc hnf hc hrace

end Vx
