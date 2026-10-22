import VxMemoryModel.Channel

/-!
# Each `recv` gets the value of one `send`, in order

Element `k` of the values of a channel belongs to the operations with the number `k`. Only a
`send` writes an element, and only a `recv` reads one. `data_op` shows that each access of element
`k` comes after a `fetchAdd` that gave its thread the number `k`. So `ring_fifo` follows: the
`recv` with the number `k` gets a value that the `send` with the number `k` wrote, before it set
its flag `full`. `ring_once` shows that two reads of the same write belong to the same `recv`, so
no value arrives twice.

`ring_fifo` needs no rule of the memory model. A read gets its value from a write to its place,
and only one `send` writes that place.
-/

namespace Vx

namespace ChanProg

variable {P : ChanProg}

variable (P) in
/-- What a phase says about an operation `o` with the number `t` that accesses its element of the
values at the position `i`. -/
def DPre (ph : Phase) (o : Op) (c : Nat) (t : Int) (L : List Label) (i : Nat) : Prop :=
  match ph with
  | .idle => False
  | .took q c' k => q = .op o ∧ c' = c ∧ k = t ∧ P.NoSet o c L 0 i
  | .counted o' c' k => o' = o ∧ c' = c ∧ k = t ∧ P.NoSet o c L 0 i
  | .waited o' c' k => o' = o ∧ c' = c ∧ k = t ∧ P.NoSet o c L 0 i

theorem data_inj {c c' j j' : Nat} (hj : j < P.msgs) (hj' : j' < P.msgs)
    (h : P.data c + j = P.data c' + j') : c = c' ∧ j = j' := by
  have := chan_eq (P := P) (c := c) (c' := c') (a := 3 + P.msgs + j) (a' := 3 + P.msgs + j')
    (by simp only [data] at h; omega) (by rw [size_eq]; omega) (by rw [size_eq]; omega)
  omega

/-- Before an operation accesses its element of the values, it read its counter and got its
number, and set no flag of its kind. A `send` writes the element, and a `recv` reads it. -/
theorem Accepts.data_hist {ph L} (h : P.Accepts ph L) {c j} (hc : c < P.chans)
    (hj : j < P.msgs) :
    ∀ {i : Nat} {l : Label}, L[i]? = some l → l.kind ≠ .fence → l.loc = P.data c + j →
      ∃ o, l.kind = o.moveKind ∧ (P.DPre ph o c j L i ∨
        ∃ i' < i, L[i']? = some ⟨.read, .sc, P.ctr o c, j, false⟩ ∧ P.NoSet o c L (i' + 1) i) := by
  induction h with
  | nil => intro i l h; simp at h
  | @cons ph ph' a L htr _ ih =>
    intro i l hl hk hx
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hav => exact (hav.not_inside hk (hx ▸ (data_inside hc hj).1)
          (hx ▸ (data_inside hc hj).2)).elim
      | take hq => exact (data_ne_cnt hq.lockOk hj hx.symm).elim
      | count => exact (data_ne_cnt (q := .op _) trivial hj hx.symm).elim
      | bumped _ hq => exact (data_ne_cnt hq.lockOk hj hx.symm).elim
      | peek hm => exact (data_ne_cnt (q := .serving) hm hj hx.symm).elim
      | poll => exact (data_ne_cnt (q := .drop) trivial hj hx.symm).elim
      | wait _ _ hj' => exact (data_ne_sig hj hj' hx.symm).elim
      | set _ hj' => exact (data_ne_sig hj hj' hx.symm).elim
      | @move o c' k v j' hk' hj' =>
        obtain ⟨rfl, rfl⟩ := data_inj hj' hj hx
        exact ⟨o, rfl, Or.inl ⟨rfl, rfl, hk'.symm, fun m _ _ h1 _ => absurd h1 (Nat.not_lt_zero m)⟩⟩
    | succ i =>
      have notSet {o : Op} {b : Label} (hb : b.kind ≠ .write ∨ b.loc = P.ctr o c ∨
          ∃ j' < P.msgs, b.loc = P.data c + j') :
          ¬ (b.kind = .write ∧ ∃ j < P.sigLen o, b.loc = P.sig o c + j) := by
        rintro ⟨hw, j'', hj'', hl''⟩
        rcases hb with hb | hb | ⟨j', hj', hb⟩
        · exact hb hw
        · exact cnt_ne_sig (q := .op o) trivial hj'' (hb.symm.trans hl'')
        · exact data_ne_sig hj' hj'' (hb.symm.trans hl'')
      obtain ⟨o, hko, hpre | ⟨i', hi', hread, hno⟩⟩ := ih (by simpa using hl) hk hx
      · refine ⟨o, hko, ?_⟩
        cases ph' with
        | idle => exact hpre.elim
        | took q c' k =>
          obtain ⟨rfl, rfl, rfl, hno⟩ := hpre
          cases htr with
          | take => exact Or.inr ⟨0, by omega, by simp, hno.cons⟩
          | poll hph => rcases hph with h | h <;> cases h
        | counted o' c' k =>
          obtain ⟨rfl, rfl, rfl, hno⟩ := hpre
          cases htr with
          | count => exact Or.inl ⟨rfl, rfl, rfl, hno.cons_zero (notSet (Or.inr (Or.inl rfl)))⟩
          | poll => exact Or.inl ⟨rfl, rfl, rfl, hno.cons_zero (notSet (Or.inl nofun))⟩
        | waited o' c' k =>
          obtain ⟨rfl, rfl, rfl, hno⟩ := hpre
          cases htr with
          | wait hph =>
            rcases hph with rfl | rfl <;>
              exact Or.inl ⟨rfl, rfl, rfl, hno.cons_zero (notSet (Or.inl nofun))⟩
          | move _ hjl =>
            exact Or.inl ⟨rfl, rfl, rfl, hno.cons_zero (notSet (Or.inr (Or.inr ⟨_, hjl, rfl⟩)))⟩
          | poll => exact Or.inl ⟨rfl, rfl, rfl, hno.cons_zero (notSet (Or.inl nofun))⟩
      · exact ⟨o, hko, Or.inr ⟨i' + 1, by omega, by simpa using hread, hno.cons⟩⟩

variable {X : Execution} (run : P.toTaskProg.toProg.Run X)
include run

/-- The writes at the start do not write the values. -/
theorem data_not_init (hP : P.Ok) {c j} (hc : c < P.chans) (hj : j < P.msgs) {d : X.E}
    (hl : X.loc d = P.data c + j) : P.pre (X.thread d) ≤ run.idx d := by
  refine Nat.le_of_not_lt fun hp => ?_
  obtain ⟨-, p, hi, -, -, hl', -⟩ := init_of run hP hp
  have hs : 2 < P.size := by rw [size_eq]; omega
  have hg := chan_ge (P := P) (c := c)
  have hx : P.data c + j = p.1 := hl.symm.trans hl'
  rcases initList_cases hi with ⟨hi, rfl⟩ | ⟨hi, hi', rfl⟩ | ⟨c', hc', j', hj', rfl⟩ |
      ⟨c', hc', j', hj', rfl⟩ | ⟨-, rfl | rfl⟩
  · simp only [data] at hx
    omega
  · simp only [ctrPlace, data] at hx
    have := chan_eq (P := P) (c := c) (c' := (run.idx d - 2 * P.locks) / 3)
      (a := 3 + P.msgs + j) (a' := (run.idx d - 2 * P.locks) % 3) (by omega)
      (by rw [size_eq]; omega) (Nat.lt_of_lt_of_le (Nat.mod_lt _ (by omega)) (by omega))
    omega
  · exact data_ne_sig (o := .send) hj hj' hx
  · have := hP.cap c' hc'
    exact data_ne_sig (o := .recv) hj (by simp only [sigLen]; omega) hx
  all_goals
    have := (data_inside hc hj).2
    have := top_le_slot (P := P)
    simp only [flag] at hx
    omega

/-- Each access of element `j` of the values comes from the operation with the number `j`. In the
thread of the access, a `fetchAdd` of the counter gave that number before it, and the thread set
no flag of its kind in between. The access is a write for a `send`, and a read for a `recv`. -/
theorem data_op (hP : P.Ok) {c j} (hc : c < P.chans) (hj : j < P.msgs) {d : X.E}
    (hk : X.kind d ≠ .fence) (hl : X.loc d = P.data c + j) :
    ∃ o r r', X.kind d = o.moveKind ∧ X.rmw r r' ∧ X.loc r' = P.ctr o c ∧ run.val r = j ∧
      X.thread r = X.thread d ∧ run.idx r < run.idx d ∧
      ∀ d', X.thread d' = X.thread d → run.idx r < run.idx d' → run.idx d' < run.idx d →
        ¬ (X.IsWrite d' ∧ ∃ j' < P.sigLen o, X.loc d' = P.sig o c + j') := by
  have hp := data_not_init run hP hc hj hl
  have hacc := (view run hP d).2
  have hget := view_get run (e := d) rfl (Nat.le_refl _) hp
  obtain ⟨o, hko, hpre | ⟨i', hi', hread, hno⟩⟩ := hacc.data_hist hc hj hget hk hl
  · exact hpre.elim
  have hne : i' + 1 ≠ run.idx d - P.pre (X.thread d) := by
    intro heq
    have := hacc.take_count hc hread (by rw [heq]; exact hget) rfl rfl
    simp only [Label.mk.injEq] at this
    exact data_ne_cnt (q := .op o) trivial hj (hl.symm.trans this.2.2.1)
  obtain ⟨r, hrt, hri⟩ := run.idx_down d (P.pre (X.thread d) + i') (by omega)
  obtain ⟨r', hrt', hri'⟩ := run.idx_down d (P.pre (X.thread d) + i' + 1) (by omega)
  have hr := view_get run (e := d) hrt (by omega) (by omega)
  rw [show run.idx r - P.pre (X.thread d) = i' by omega, hread] at hr
  have hr' := view_get run (e := d) hrt' (by omega) (by omega)
  rw [show run.idx r' - P.pre (X.thread d) = i' + 1 by omega] at hr'
  have hcount := hacc.take_count hc hread hr' rfl rfl
  simp only [Option.some.injEq, Label.mk.injEq] at hr hcount
  obtain ⟨-, -, -, hrv, -⟩ := hr
  obtain ⟨-, -, hrl', -, hrm'⟩ := hcount
  refine ⟨o, r, r', hko, (run.rmw_iff r r').2 ⟨hrt.trans hrt'.symm, by omega, hrm'⟩, hrl',
    hrv.symm, hrt, by omega, ?_⟩
  intro d' hdt hd1 hd2 ⟨hdw, j', hj', hdl⟩
  exact hno (run.idx d' - P.pre (X.thread d)) _ (by omega) (by omega)
    (view_get run hdt (by omega) (by omega)) ⟨hdw, j', hj', hdl⟩

/-- The `recv` with the number `j` gets a value that the `send` with the number `j` wrote. The
read of element `j` comes after the `fetchAdd` that gave the `recv` its number. The write comes
after the `fetchAdd` that gave the `send` its number, and before the `send` set its flag. -/
theorem ring_fifo (hP : P.Ok) {c j} (hc : c < P.chans) (hj : j < P.msgs) {w e : X.E}
    (hrf : X.rf w e) (hel : X.loc e = P.data c + j) :
    run.val e = run.val w ∧
      (∃ r r', X.rmw r r' ∧ X.loc r' = P.ctr .recv c ∧ run.val r = j ∧
        X.thread r = X.thread e ∧ run.idx r < run.idx e) ∧
      ∃ r r', X.rmw r r' ∧ X.loc r' = P.ctr .send c ∧ run.val r = j ∧
        X.thread r = X.thread w ∧ run.idx r < run.idx w ∧
        ∀ d, X.thread d = X.thread w → run.idx r < run.idx d → run.idx d < run.idx w →
          ¬ (X.IsWrite d ∧ ∃ j' < P.msgs, X.loc d = P.sig .send c + j') := by
  obtain ⟨hw, he, hwl⟩ := run.rf_write_read w e hrf
  obtain ⟨o, r, r', hko, hrw, hrl, hrv, hrt, hri, -⟩ :=
    data_op run hP hc hj (d := e) (by rw [he]; nofun) hel
  obtain ⟨o', s, s', hko', hsw, hsl, hsv, hst, hsi, hno⟩ :=
    data_op run hP hc hj (d := w) (by rw [hw]; nofun) (hwl.trans hel)
  cases o
  · rw [he] at hko
    cases hko
  cases o'
  · exact ⟨(run.rf_val w e hrf).symm, ⟨r, r', hrw, hrl, hrv, hrt, hri⟩, s, s', hsw, hsl, hsv, hst,
      hsi, hno⟩
  · rw [hw] at hko'
    cases hko'

/-- Two reads that get their value from the same write belong to the same `recv`: the one whose
number names the element. So no value arrives twice. -/
theorem ring_once (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {c j} (hc : c < P.chans) (hj : j < P.msgs) {w e1 e2 : X.E}
    (h1 : X.rf w e1) (h2 : X.rf w e2) (hel : X.loc e1 = P.data c + j) :
    ∃ r r', X.rmw r r' ∧ X.loc r' = P.ctr .recv c ∧ run.val r = j ∧
      X.thread r = X.thread e1 ∧ X.thread r = X.thread e2 ∧
      run.idx r < run.idx e1 ∧ run.idx r < run.idx e2 := by
  have hel2 : X.loc e2 = P.data c + j := by
    rw [← (run.rf_write_read w e2 h2).2.2, (run.rf_write_read w e1 h1).2.2, hel]
  obtain ⟨-, ⟨r1, r1', hrw1, hrl1, hrv1, hrt1, hri1⟩, -⟩ := ring_fifo run hP hc hj h1 hel
  obtain ⟨-, ⟨r2, r2', hrw2, hrl2, hrv2, hrt2, hri2⟩, -⟩ := ring_fifo run hP hc hj h2 hel2
  have := (counter run hP wf (q := .op .recv) hc).tickets wf hcons ho hfin hrw1 hrl1 hrw2 hrl2
    (hrv1.trans hrv2.symm)
  subst this
  exact ⟨r1, r1', hrw1, hrl1, hrv1, hrt1, hrt2, hri1, hri2⟩

end ChanProg

end Vx
