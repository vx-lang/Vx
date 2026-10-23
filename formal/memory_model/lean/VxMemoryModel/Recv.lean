import VxMemoryModel.Fifo

/-!
# The places of a channel never race

A `recv` with the number `k` reads element `k` of the values only after a read of the flag
`full k` that saw it set. `recv_full` shows this. `RTrans` follows one element `j` of one channel
`c` through a thread. Its state is `true` when the last read of `full j` saw a value that is not
0. A read of element `j` needs `true`.

Then `data_hb` follows. Only the `send` with the number `j` writes element `j`. It sets `full j`
after it, so it happens before each `recv` that reads element `j`. The other places of a channel
or a lock have only `sc` accesses. So `chan_no_race` shows that no two accesses to these places
make a data race.
-/

namespace Vx

namespace ChanProg

variable (P : ChanProg) (c j : Nat)

/-- An access that is not a read of `full j` or of element `j` of the channel `c`. -/
def NotRecv (k : Kind) (_ : Order) (x : Nat) : Prop :=
  k ≠ .read ∨ (x ≠ P.sig .send c + j ∧ x ≠ P.data c + j)

/-- The accesses that a thread can make. The state is `true` when the last read of `full j` saw
a value that is not 0. A read of element `j` needs `true`. -/
inductive RTrans : Bool → Option Label → Bool → Prop
  | silent {h} : RTrans h none h
  | other {h l} : P.NotRecv c j l.kind l.order l.loc → RTrans h (some l) h
  | full {h v} : RTrans h (some ⟨.read, .sc, P.sig .send c + j, v, false⟩) (decide (v ≠ 0))
  | value {v} : RTrans true (some ⟨.read, .plain, P.data c + j, v, false⟩) true

/-- Lowered code, with each `recv` whole. -/
inductive RClean : Stmt → Prop
  | other {s} : s.Accesses (P.NotRecv c j) → RClean s
  | recv {c' r s} : c' < P.chans → RClean (P.recvCode c' r s)
  | seq {s t} : RClean s → RClean t → RClean (.seq s t)
  | ite {e s t} : RClean s → RClean t → RClean (.ite e s t)
  | loop {e s} : RClean s → RClean (.loop e s)

/-- In a `recv` of the channel `c'`: if it is the `recv` of element `j` of `c`, the state tells
whether the register `fullReg` holds a value that is not 0. -/
def Synced (c' : Nat) (regs : Nat → Int) (h : Bool) : Prop :=
  c' = c → regs P.tmp = j → h = decide (regs P.fullReg ≠ 0)

/-- The states of lowered code, with their state. Before its first read of `full`, a `recv` can
be in any state. After it, `Synced` holds. -/
inductive RGood (regs : Nat → Int) : Bool → Stmt → Prop
  | clean {h s} : P.RClean c j s → RGood regs h s
  | seq {h s t} : RGood regs h s → P.RClean c j t → RGood regs h (.seq s t)
  | took {h c' r s v} : c' < P.chans →
    RGood regs h (.seq (.rmwStore (P.ctr .recv c') v .sc) (P.recvRest c' r s))
  | counted {h c' r s} : c' < P.chans → RGood regs h (.seq .skip (P.recvRest c' r s))
  | polling {h c' r s x} : c' < P.chans →
    (x = P.poll c' ∨ x = .seq .skip (P.loadFull c') ∨ x = P.loadFull c') →
    RGood regs h (.seq (.seq x (P.polls c')) (P.recvEnd c' r s))
  | polled {h c' r s y} : c' < P.chans → P.Synced c j c' regs h →
    (y = .seq .skip (P.polls c') ∨ y = P.polls c' ∨
      y = .ite (P.again c') (.seq (P.poll c') (P.polls c')) .skip ∨ y = .skip) →
    RGood regs h (.seq y (P.recvEnd c' r s))
  | ending {h c' r s} : c' < P.chans → P.Synced c j c' regs h →
    RGood regs h (P.recvEnd c' r s)
  | reading {h c' r s} : c' < P.chans → P.Synced c j c' regs h → regs P.fullReg ≠ 0 →
    RGood regs h (.seq (.seq (.loadAt r (P.data c') P.msgs (.reg P.tmp) .plain)
      (.assign s (.const 1))) (P.fin .recv c'))

variable {P c j}

section Step

variable (hc : c < P.chans) (hj : j < P.msgs)
include hj

theorem notRecv_cnt {q c'} (hq : P.LockOk q c') {k o} : P.NotRecv c j k o (P.cnt q c') :=
  Or.inr ⟨cnt_ne_sig (o' := .send) hq hj, fun h => data_ne_cnt hq hj h.symm⟩

theorem notRecv_sig {c' j'} (hj' : j' < P.sigLen .recv) {k o} :
    P.NotRecv c j k o (P.sig .recv c' + j') :=
  Or.inr ⟨fun h => absurd (sig_inj (o' := .send) hj' hj h).1 nofun,
    fun h => data_ne_sig hj hj' h.symm⟩

omit hj in
theorem notRecv_write {k o x} (hk : k = .write) : P.NotRecv c j k o x :=
  Or.inl (by rw [hk]; nofun)

theorem RClean.step {s} (hcl : P.RClean c j s) :
    ∀ {h regs a s' regs'}, Step s regs a s' regs' →
      ∃ h', P.RGood c j regs' h' s' ∧ P.RTrans c j h a h' := by
  induction hcl with
  | other hs =>
    intro h regs a s' regs' hst
    obtain ⟨hs', ha⟩ := (accessRule (P.NotRecv c j)).step hst hs
    refine ⟨h, .clean (.other hs'), ?_⟩
    cases a with
    | none => exact .silent
    | some l => exact .other (ha l rfl)
  | recv hc' =>
    intro h regs a s' regs' hst
    cases hst with
    | seq hst =>
      cases hst with
      | fetchAdd v => exact ⟨h, .took hc', .other (notRecv_cnt hj (q := .op .recv) trivial)⟩
  | seq _ ht ih =>
    intro h regs a s' regs' hst
    cases hst with
    | seq_skip => exact ⟨h, .clean ht, .silent⟩
    | seq hst =>
      obtain ⟨h', hg, htr⟩ := ih hst
      exact ⟨h', .seq hg ht, htr⟩
  | ite hs ht =>
    intro h regs a s' regs' hst
    cases hst with
    | ite_true => exact ⟨h, .clean hs, .silent⟩
    | ite_false => exact ⟨h, .clean ht, .silent⟩
  | loop hs =>
    intro h regs a s' regs' hst
    cases hst with
    | loop => exact ⟨h, .clean (.ite (.seq hs (.loop hs)) (.other trivial)), .silent⟩

/-- Each step of lowered code keeps `RGood`, and its access follows `RTrans`. -/
theorem RGood.step {h s regs} (hg : P.RGood c j regs h s) :
    ∀ {a s' regs'}, Step s regs a s' regs' → ∃ h', P.RGood c j regs' h' s' ∧ P.RTrans c j h a h' := by
  induction hg with
  | clean hcl => exact (hcl.step hj)
  | seq hg ht ih =>
    intro a s' regs' hst
    cases hst with
    | seq_skip =>
      cases hg with
      | clean => exact ⟨_, .clean ht, .silent⟩
    | seq hst =>
      obtain ⟨h', hg', htr⟩ := ih hst
      exact ⟨h', .seq hg' ht, htr⟩
  | took hc' =>
    intro a s' regs' hst
    cases hst with
    | seq hst =>
      cases hst with
      | rmwStore => exact ⟨_, .counted hc', .other (notRecv_write rfl)⟩
  | counted hc' =>
    intro a s' regs' hst
    cases hst with
    | seq_skip => exact ⟨_, .polling hc' (Or.inl rfl), .silent⟩
    | seq hst => cases hst
  | @polling c' r s x hc' hx =>
    intro a s' regs' hst
    rcases hx with rfl | rfl | rfl
    · cases hst with
      | seq hst =>
        cases hst with
        | seq hst =>
          cases hst with
          | seq hst =>
            cases hst with
            | load v =>
              exact ⟨_, .polling hc' (Or.inr (Or.inl rfl)),
                .other (notRecv_cnt hj (q := .drop) trivial)⟩
    · cases hst with
      | seq hst =>
        cases hst with
        | seq hst =>
          cases hst with
          | seq_skip => exact ⟨_, .polling hc' (Or.inr (Or.inr rfl)), .silent⟩
          | seq hst => cases hst
    · cases hst with
      | seq hst =>
        cases hst with
        | seq hst =>
          cases hst with
          | @loadAt _ _ _ _ _ _ j' v hi hj' =>
            simp only [Expr.eval] at hi
            have htmp : update regs P.fullReg v P.tmp = regs P.tmp := by simp [update]
            by_cases hcj : c' = c ∧ j' = j
            · obtain ⟨rfl, rfl⟩ := hcj
              refine ⟨_, .polled hc' (fun _ _ => by simp [update]) (Or.inl rfl), .full⟩
            · refine ⟨_, .polled hc' (fun h1 h2 => ?_) (Or.inl rfl), .other (Or.inr ⟨fun he => ?_,
                fun he => data_ne_sig hj hj' he.symm⟩)⟩
              · rw [htmp, hi] at h2
                exact (hcj ⟨h1, by omega⟩).elim
              · obtain ⟨-, h1, h2⟩ := sig_inj hj' hj he
                exact hcj ⟨h1, h2⟩
  | polled hc' hsy hy =>
    intro a s' regs' hst
    rcases hy with rfl | rfl | rfl | rfl
    · cases hst with
      | seq hst =>
        cases hst with
        | seq_skip => exact ⟨_, .polled hc' hsy (Or.inr (Or.inl rfl)), .silent⟩
        | seq hst => cases hst
    · cases hst with
      | seq hst =>
        cases hst with
        | loop => exact ⟨_, .polled hc' hsy (Or.inr (Or.inr (Or.inl rfl))), .silent⟩
    · cases hst with
      | seq hst =>
        cases hst with
        | ite_true => exact ⟨_, .polling hc' (Or.inl rfl), .silent⟩
        | ite_false => exact ⟨_, .polled hc' hsy (Or.inr (Or.inr (Or.inr rfl))), .silent⟩
    · cases hst with
      | seq_skip => exact ⟨_, .ending hc' hsy, .silent⟩
      | seq hst => cases hst
  | ending hc' hsy =>
    intro a s' regs' hst
    cases hst with
    | seq hst =>
      cases hst with
      | ite_true hv =>
        simp only [Expr.eval] at hv
        exact ⟨_, .reading hc' hsy hv, .silent⟩
      | ite_false =>
        exact ⟨_, .clean (.other ⟨trivial, fun _ _ => notRecv_write rfl⟩), .silent⟩
  | @reading c' r s hc' hsy hv =>
    intro a s' regs' hst
    cases hst with
    | seq hst =>
      cases hst with
      | seq hst =>
        cases hst with
        | @loadAt _ _ _ _ _ _ j' v hi hj' =>
          simp only [Expr.eval] at hi
          by_cases hcj : c' = c ∧ j' = j
          · obtain ⟨rfl, rfl⟩ := hcj
            have hh := hsy rfl hi
            simp only [hv, ne_eq, not_false_eq_true, decide_true] at hh
            subst hh
            exact ⟨_, .clean (.other ⟨⟨trivial, trivial⟩, fun _ _ => notRecv_write rfl⟩), .value⟩
          · refine ⟨_, .clean (.other ⟨⟨trivial, trivial⟩, fun _ _ => notRecv_write rfl⟩),
              .other (Or.inr ⟨fun he => data_ne_sig hj' hj he, fun he => ?_⟩)⟩
            obtain ⟨h1, h2⟩ := data_inj hj' hj he
            exact hcj ⟨h1, h2⟩

end Step

variable (P c j) in
/-- The lists of accesses that `RTrans` allows from a state. -/
inductive RAccepts : Bool → List Label → Prop
  | nil {h} : RAccepts h []
  | cons {h h' a L} : P.RTrans c j h (some a) h' → RAccepts h' L → RAccepts h (a :: L)

theorem RTrans.none_eq {h h'} (ht : P.RTrans c j h none h') : h' = h := by
  cases ht <;> rfl

/-- A run of lowered code makes a list of accesses that `RTrans` allows. -/
theorem Runs.raccepts (hj : j < P.msgs) {s regs L} (hr : Runs s regs L) :
    ∀ {h}, P.RGood c j regs h s → P.RAccepts c j h L := by
  induction hr with
  | stop => exact fun _ => .nil
  | silent hs _ ih =>
    intro h hg
    obtain ⟨h', hg', htr⟩ := hg.step hj hs
    cases htr.none_eq
    exact ih hg'
  | visible hs _ ih =>
    intro h hg
    obtain ⟨h', hg', htr⟩ := hg.step hj hs
    exact .cons htr (ih hg')

/-- Each read of element `j` comes after a read of `full j` that saw a value that is not 0, or
the state was `true` at the start. -/
theorem RAccepts.full_hist (hj : j < P.msgs) {h L} (hacc : P.RAccepts c j h L) :
    ∀ {i : Nat} {l : Label}, L[i]? = some l → l.kind = .read → l.loc = P.data c + j →
      h = true ∨ ∃ i' < i, ∃ v, v ≠ 0 ∧ L[i']? = some ⟨.read, .sc, P.sig .send c + j, v, false⟩ := by
  induction hacc with
  | nil => intro i l hl; simp at hl
  | @cons h h' a L htr _ ih =>
    intro i l hl hk hx
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hn =>
        rcases hn with hn | hn
        · exact (hn hk).elim
        · exact (hn.2 hx).elim
      | full => exact (data_ne_sig hj (o := .send) hj hx.symm).elim
      | value => exact Or.inl rfl
    | succ i =>
      rcases ih (by simpa using hl) hk hx with hh | ⟨i', hi', v, hv, hl'⟩
      · cases htr with
        | other => exact Or.inl hh
        | @full _ v => exact Or.inr ⟨0, by omega, v, by simpa using hh, rfl⟩
        | value => exact Or.inl rfl
      · exact Or.inr ⟨i' + 1, by omega, v, hv, by simpa using hl'⟩

/-! ## The code of the program follows `RTrans` -/

section Code

variable (hc : c < P.chans) (hj : j < P.msgs)
include hc hj

theorem Avoid.notRecv {k o x} (h : P.Avoid k o x) : P.NotRecv c j k o x := by
  have h1 := sig_inside (o := .send) hc hj
  have h2 := data_inside hc hj
  rcases h with h | h | h
  · exact Or.inl (by rw [h]; nofun)
  · exact Or.inr ⟨by omega, by omega⟩
  · exact Or.inr ⟨by omega, by omega⟩

theorem rclean_lower {s : CStmt} (hs : P.StmtOk s) :
    P.RClean c j (P.toTaskProg.lower (P.lowerC s)) := by
  have av : ∀ {t : Stmt}, t.Accesses P.Avoid → P.RClean c j t := fun h =>
    .other (h.mono fun _ _ _ h => Avoid.notRecv hc hj h)
  induction s with
  | base => exact av (lower_avoid hs)
  | send c' e =>
    exact .other ⟨⟨notRecv_cnt hj (q := .op .send) trivial, notRecv_cnt hj (q := .op .send) trivial⟩,
      fun _ hj' => notRecv_sig hj hj', fun _ _ => notRecv_write rfl,
      fun _ _ => notRecv_write rfl⟩
  | recv r s c' => exact .recv hs.1
  | drop c' =>
    exact .other ⟨notRecv_cnt hj (q := .drop) trivial, notRecv_cnt hj (q := .drop) trivial⟩
  | locked m s ih =>
    exact .seq (.other ⟨⟨notRecv_cnt hj (q := .next) hs.1, notRecv_cnt hj (q := .next) hs.1⟩,
        notRecv_cnt hj (q := .serving) (k := .read) (o := .sc) hs.1,
        notRecv_cnt hj (q := .serving) (k := .read) (o := .sc) hs.1⟩)
      (.seq (ih hs.2.2) (.other ⟨notRecv_cnt hj (q := .serving) hs.1,
        notRecv_cnt hj (q := .serving) hs.1⟩))
  | seq _ _ ih1 ih2 => exact .seq (ih1 hs.1) (ih2 hs.2)
  | ite _ _ _ ih1 ih2 => exact .ite (ih1 hs.1) (ih2 hs.2)
  | loop _ _ ih => exact .loop (ih hs)
  | tasksBlock _ ih => exact .seq (av (clearMarks_avoid _)) (.seq (ih hs) (av (joinMarked_avoid _)))
  | fail => exact av (failCode_avoid _)
  | checkCancel => exact av checkCode_avoid
  | result => exact .other (Avoid.notRecv hc hj slot_avoid)

theorem rclean_body (hP : P.Ok) (t : Nat) : P.RClean c j (P.body t) := by
  cases t with
  | zero => exact rclean_lower hc hj hP.main
  | succ t =>
    show P.RClean c j (P.toTaskProg.taskCode t)
    unfold TaskProg.taskCode
    split
    · refine .seq (.other (Avoid.notRecv hc hj (Or.inr (Or.inr ?_))))
        (.seq (rclean_lower hc hj (hP.task t)) (.other (Avoid.notRecv hc hj (Or.inr (Or.inr ?_)))))
      · simp only [TaskProg.start, toTaskProg]
        omega
      · simp only [TaskProg.done, toTaskProg]
        omega
    · exact .other trivial

end Code

/-- An access to the places of the channels and the locks is `sc`, or it moves a value. -/
theorem Accepts.inside {ph L} (h : P.Accepts ph L) :
    ∀ {m : Nat} {l : Label}, L[m]? = some l → l.kind ≠ .fence → P.places ≤ l.loc →
      l.loc < P.top → l.order = .sc ∨ ∃ c j, j < P.msgs ∧ l.loc = P.data c + j := by
  induction h with
  | nil => intro m l h; simp at h
  | cons htr _ ih =>
    intro m l hl hk h1 h2
    cases m with
    | succ m => exact ih (by simpa using hl) hk h1 h2
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hav => exact (hav.not_inside hk h1 h2).elim
      | take | count | bumped | poll | peek | wait | set => exact Or.inl rfl
      | move _ hj => exact Or.inr ⟨_, _, hj, rfl⟩

theorem chans_of_data {c j : Nat} (h : P.data c + j < P.top) : c < P.chans := by
  refine Nat.lt_of_not_le fun hc => ?_
  have := Nat.mul_le_mul_right P.size hc
  simp only [data, chan, top] at h
  omega

/-! ## The executions of a program with channels -/

variable {X : Execution} (run : P.toTaskProg.toProg.Run X)
include run

/-- A read of element `j` of the channel `c` comes after a read of `full j`, in its thread, that
saw a value that is not 0. -/
theorem recv_full (hP : P.Ok) (hc : c < P.chans) (hj : j < P.msgs) {b : X.E} (hb : X.IsRead b)
    (hbl : X.loc b = P.data c + j) :
    ∃ e, X.thread e = X.thread b ∧ run.idx e < run.idx b ∧ X.IsRead e ∧ X.IsSeqCst e ∧
      X.loc e = P.sig .send c + j ∧ run.val e ≠ 0 := by
  have hp := data_not_init run hP hc hj hbl
  have hacc := Runs.raccepts (h := false) hj (trace_rest (run.trace b)).2
    (.clean (rclean_body hc hj hP _))
  have hget := view_get run (e := b) rfl (Nat.le_refl _) hp
  rcases hacc.full_hist hj hget hb hbl with h | ⟨i', hi', v, hv, hl'⟩
  · cases h
  obtain ⟨e, het, hei⟩ := run.idx_down b (P.pre (X.thread b) + i') (by omega)
  have he := view_get run (e := b) het (by omega) (by omega)
  rw [show run.idx e - P.pre (X.thread b) = i' by omega, hl'] at he
  simp only [Option.some.injEq, Label.mk.injEq] at he
  obtain ⟨hk, ho, hl, hvl, -⟩ := he
  exact ⟨e, het, by omega, hk.symm, ho.symm, hl.symm, by rw [← hvl]; exact hv⟩

/-- The write of element `j` by a `send` happens before each access of element `j` in another
thread. Only that `send` writes element `j`, and a `recv` reads it after `full j`. -/
theorem data_hb (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) (hc : c < P.chans) (hj : j < P.msgs) {a b : X.E} (haw : X.IsWrite a)
    (hal : X.loc a = P.data c + j) (hb : X.IsAccess b) (hbl : X.loc b = P.data c + j)
    (ht : X.thread a ≠ X.thread b) : X.hb a b := by
  have hS := counter run hP wf (q := .op .send) hc
  obtain ⟨oa, ra, ra', hka, hra, hral, hrav, hrat, hrai, hnoa⟩ :=
    data_op run hP hc hj (d := a) (by rw [haw]; nofun) hal
  have hoa : oa = .send := by
    cases oa
    · rfl
    · rw [haw] at hka
      cases hka
  subst hoa
  rcases hb with hb | hb
  · -- `b` reads: the `send` sets `full j` after its write, and `b` comes after a read of it.
    obtain ⟨e, het, hei, her, -, hel, hev⟩ := recv_full run hP hc hj hb hbl
    obtain ⟨w, hrf⟩ := run.rf_source e her
    obtain ⟨hww, -, hwl⟩ := run.rf_write_read w e hrf
    have hwv : run.val w ≠ 0 := by rw [run.rf_val w e hrf]; exact hev
    obtain ⟨r, r', hrr, hrl, hrv, hrt, hri, -⟩ :=
      set_writer run hP hc hj (Nat.zero_le _) hww (hwl.trans hel) (Or.inr hwv)
    have := hS.tickets wf hcons ho hfin hra hral hrr hrl (by rw [hrav, hrv]; simp [off])
    subst this
    have haw' : X.po a w := by
      refine (run.po_iff a w).2 ⟨hrat.symm.trans hrt, ?_⟩
      rcases Nat.lt_trichotomy (run.idx a) (run.idx w) with h | h | h
      · exact h
      · have := run.idx_inj a w (hrat.symm.trans hrt) h
        subst this
        exact (data_ne_sig hj (o := .send) hj (hal.symm.trans (hwl.trans hel))).elim
      · exact (hnoa w (hrt.symm.trans hrat) hri h ⟨hww, j, hj, hwl.trans hel⟩).elim
    exact send_hb_recv run hP wf hcons ho hfin hc hj hww (hwl.trans hel) hwv (Or.inr haw') her hel hev
      (Or.inr ((run.po_iff e b).2 ⟨het, hei⟩))
  · -- `b` writes: it is the `send` with the number `j` too, so it is in the thread of `a`.
    obtain ⟨ob, rb, rb', hkb, hrb, hrbl, hrbv, hrbt, -, -⟩ :=
      data_op run hP hc hj (d := b) (by rw [hb]; nofun) hbl
    have hob : ob = .send := by
      cases ob
      · rfl
      · rw [hb] at hkb
        cases hkb
    subst hob
    have := hS.tickets wf hcons ho hfin hra hral hrb hrbl (hrav.trans hrbv.symm)
    subst this
    exact (ht (hrat.symm.trans hrbt)).elim

/-- An access to the places of the channels and the locks is `sc`, or it moves a value. -/
theorem inside_sc (hP : P.Ok) {e : X.E} (hk : X.kind e ≠ .fence) (h1 : P.places ≤ X.loc e)
    (h2 : X.loc e < P.top) :
    X.IsSeqCst e ∨ ∃ c j, c < P.chans ∧ j < P.msgs ∧ X.loc e = P.data c + j := by
  by_cases hp : run.idx e < P.pre (X.thread e)
  · obtain ⟨-, -, -, -, h, -⟩ := init_of run hP hp
    exact Or.inl h
  · rcases (view run hP e).2.inside (view_get run rfl (Nat.le_refl _) (by omega)) hk h1 h2 with
      h | ⟨c, j, hj, hl⟩
    · exact Or.inl h
    · have hl : X.loc e = P.data c + j := hl
      exact Or.inr ⟨c, j, chans_of_data (hl ▸ h2), hj, hl⟩

/-- No two accesses to the places of the channels and the locks make a data race. -/
theorem chan_no_race (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {a b : X.E} (hr : X.DataRace a b) (h1 : P.places ≤ X.loc a)
    (h2 : X.loc a < P.top) : False := by
  obtain ⟨⟨-, ha, hb, hl, hw⟩, ht, hab, hba, hat⟩ := hr
  have nf : ∀ {e : X.E}, X.IsAccess e → X.kind e ≠ .fence := by
    rintro e (h | h) hf
    · simp [Execution.IsRead, hf] at h
    · simp [Execution.IsWrite, hf] at h
  have atomic : ∀ {e : X.E}, X.IsSeqCst e → X.IsAtomic e := fun h => by
    unfold Execution.IsAtomic
    rw [h]
    nofun
  have value : ∀ {c j}, c < P.chans → j < P.msgs → X.loc a = P.data c + j → False := by
    intro c j hc hj hx
    rcases hw with hw | hw
    · exact hab (data_hb run hP wf hcons ho hfin hc hj hw hx hb (hl ▸ hx) ht)
    · exact hba (data_hb run hP wf hcons ho hfin hc hj hw (hl ▸ hx) ha hx fun h => ht h.symm)
  rcases inside_sc run hP (nf ha) h1 h2 with hsa | ⟨c, j, hc, hj, hx⟩
  · rcases inside_sc run hP (nf hb) (hl ▸ h1) (hl ▸ h2) with hsb | ⟨c, j, hc, hj, hx⟩
    · exact hat ⟨atomic hsa, atomic hsb⟩
    · exact value hc hj (hl.trans hx)
  · exact value hc hj hx

end ChanProg

end Vx
