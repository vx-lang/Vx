import VxMemoryModel.Channel

/-!
# Two tasks never hold a lock at the same time

A `locked m s` block takes a ticket `k` from the counter `next m` with a `fetchAdd`. It reads
`serving m` until it is `k`, runs `s`, and adds 1 to `serving m`. `Hold` follows one thread
through its blocks of the lock `m`. A block of `m` does not hold another block of `m`, so between
its ticket and its `unlock`, the thread does not write `serving m`.

`unlock_ticket` shows that the `unlock` of the ticket `k` reads `serving m` as `k`. The proof
goes by `k`. The block read `serving m` as `k` before its `unlock`, so the `unlock` reads `k` or
more. If it read more, an `unlock` read `k` before it, and that one is the `unlock` of the same
block. Then `lock_excl` follows from `unlock_hb_lock`.
-/

namespace Vx

namespace ChanProg

variable (P : ChanProg) (m : Nat)

/-- An access that is not to a counter of the lock `m`. -/
def NotLock (k : Kind) (_ : Order) (x : Nat) : Prop :=
  k = .fence ∨ (x ≠ P.cnt .next m ∧ x ≠ P.cnt .serving m)

/-- Where a thread is in a `locked` block of the lock `m`. -/
inductive Hold where
  | free
  /-- It read the ticket `t` from `next`, and did not write `next` yet. -/
  | took (t : Int)
  /-- It waits until `serving` is `t`. -/
  | wait (t : Int)
  /-- It read `serving` as `t`, and holds the lock. -/
  | held (t : Int)
  /-- It read `serving` to let the lock go, and did not write it yet. -/
  | leaving (t : Int)

/-- A thread that does not run the code of a `lock` or an `unlock`. -/
def Hold.Calm : Hold → Prop
  | .free | .held _ => True
  | _ => False

/-- The accesses that a thread can make in each state of `Hold`. -/
inductive HTrans : Hold → Option Label → Hold → Prop
  | silent {h} : HTrans h none h
  | other {h l} : h.Calm → P.NotLock m l.kind l.order l.loc → HTrans h (some l) h
  | take {t} : HTrans .free (some ⟨.read, .sc, P.cnt .next m, t, false⟩) (.took t)
  | count {t} : HTrans (.took t) (some ⟨.write, .sc, P.cnt .next m, t + 1, true⟩) (.wait t)
  | peek {t v} : v ≠ t → HTrans (.wait t) (some ⟨.read, .sc, P.cnt .serving m, v, false⟩) (.wait t)
  | enter {t} : HTrans (.wait t) (some ⟨.read, .sc, P.cnt .serving m, t, false⟩) (.held t)
  | leave {t v} :
    HTrans (.held t) (some ⟨.read, .sc, P.cnt .serving m, v, false⟩) (.leaving t)
  | unlock {t v} :
    HTrans (.leaving t) (some ⟨.write, .sc, P.cnt .serving m, v, true⟩) .free

/-- The end of a `locked` block: add 1 to `serving`. -/
abbrev unl : Stmt := P.bump .serving m

abbrev peeks : Stmt := .loop P.waitCond (P.peek m)

/-- Lowered code, with each `locked` block of the lock `m` whole. -/
inductive HClean : Stmt → Prop
  | other {s} : s.Accesses (P.NotLock m) → HClean s
  | locked {b} : b.Accesses (P.NotLock m) → HClean (.seq (P.lockCode m) (.seq b (P.unl m)))
  | seq {s t} : HClean s → HClean t → HClean (.seq s t)
  | ite {e s t} : HClean s → HClean t → HClean (.ite e s t)
  | loop {e s} : HClean s → HClean (.loop e s)

/-- The states of lowered code, with their state of `Hold`. While the thread waits for the lock,
the register `tmp` holds its ticket. -/
inductive HGood (regs : Nat → Int) : Hold → Stmt → Prop
  | clean {s} : P.HClean m s → HGood regs .free s
  | seq {h s t} : HGood regs h s → P.HClean m t → HGood regs h (.seq s t)
  | took {t b} : b.Accesses (P.NotLock m) → regs P.tmp = t →
    HGood regs (.took t) (.seq (.seq (.rmwStore (P.cnt .next m) (t + 1) .sc)
      (.seq (P.peek m) (P.peeks m))) (.seq b (P.unl m)))
  | counted {t b} : b.Accesses (P.NotLock m) → regs P.tmp = t →
    HGood regs (.wait t) (.seq (.seq .skip (.seq (P.peek m) (P.peeks m))) (.seq b (P.unl m)))
  | peeking {t b} : b.Accesses (P.NotLock m) → regs P.tmp = t →
    HGood regs (.wait t) (.seq (.seq (P.peek m) (P.peeks m)) (.seq b (P.unl m)))
  | peeked {h t b x} : b.Accesses (P.NotLock m) → regs P.tmp = t →
    (x = .seq .skip (P.peeks m) ∨ x = P.peeks m ∨
      x = .ite P.waitCond (.seq (P.peek m) (P.peeks m)) .skip) →
    (regs P.ctrReg = t ∧ h = .held t ∨ regs P.ctrReg ≠ t ∧ h = .wait t) →
    HGood regs h (.seq x (.seq b (P.unl m)))
  | entered {t b} : b.Accesses (P.NotLock m) → HGood regs (.held t) (.seq .skip (.seq b (P.unl m)))
  | inside {t b} : b.Accesses (P.NotLock m) → HGood regs (.held t) (.seq b (P.unl m))
  | unlocking {t} : HGood regs (.held t) (P.unl m)
  | leaving {t v} : HGood regs (.leaving t) (.rmwStore (P.cnt .serving m) v .sc)

variable {P m}

theorem HClean.step {s} (hcl : P.HClean m s) :
    ∀ {regs a s' regs'}, Step s regs a s' regs' → ∃ h, P.HGood m regs' h s' ∧ P.HTrans m .free a h := by
  induction hcl with
  | other hs =>
    intro regs a s' regs' h
    obtain ⟨hs', ha⟩ := (accessRule (P.NotLock m)).step h hs
    refine ⟨_, .clean (.other hs'), ?_⟩
    cases a with
    | none => exact .silent
    | some l => exact .other trivial (ha l rfl)
  | locked hb =>
    intro regs a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | seq h =>
        cases h with
        | fetchAdd v => exact ⟨_, .took hb (by simp [update]), .take⟩
  | seq _ ht ih =>
    intro regs a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .clean ht, .silent⟩
    | seq h =>
      obtain ⟨h', hg, htr⟩ := ih h
      exact ⟨h', .seq hg ht, htr⟩
  | ite hs ht =>
    intro regs a s' regs' h
    cases h with
    | ite_true => exact ⟨_, .clean hs, .silent⟩
    | ite_false => exact ⟨_, .clean ht, .silent⟩
  | loop hs =>
    intro regs a s' regs' h
    cases h with
    | loop => exact ⟨_, .clean (.ite (.seq hs (.loop hs)) (.other trivial)), .silent⟩

/-- Each step of lowered code keeps `HGood`, and its access follows `HTrans`. -/
theorem HGood.step {h s regs} (hg : P.HGood m regs h s) :
    ∀ {a s' regs'}, Step s regs a s' regs' → ∃ h', P.HGood m regs' h' s' ∧ P.HTrans m h a h' := by
  induction hg with
  | clean hcl => exact hcl.step
  | seq hg ht ih =>
    intro a s' regs' h
    cases h with
    | seq_skip =>
      cases hg with
      | clean => exact ⟨_, .clean ht, .silent⟩
    | seq h =>
      obtain ⟨h', hg', htr⟩ := ih h
      exact ⟨h', .seq hg' ht, htr⟩
  | took hb hk =>
    intro a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | seq h =>
        cases h with
        | rmwStore => exact ⟨_, .counted hb hk, .count⟩
  | counted hb hk =>
    intro a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | seq_skip => exact ⟨_, .peeking hb hk, .silent⟩
      | seq h => cases h
  | @peeking t b hb hk =>
    intro a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | seq h =>
        cases h with
        | load v =>
          have hk' : update regs P.ctrReg v P.tmp = _ := update_tmp (r := P.tmp + 1) (by omega) hk
          by_cases hv : v = t
          · subst hv
            exact ⟨_, .peeked hb hk' (Or.inl rfl) (Or.inl ⟨by simp [update], rfl⟩), .enter⟩
          · exact ⟨_, .peeked hb hk' (Or.inl rfl) (Or.inr ⟨by simpa [update] using hv, rfl⟩),
              .peek hv⟩
  | peeked hb hk hx hh =>
    intro a s' regs' h
    rcases hx with rfl | rfl | rfl
    · cases h with
      | seq h =>
        cases h with
        | seq_skip => exact ⟨_, .peeked hb hk (Or.inr (Or.inl rfl)) hh, .silent⟩
        | seq h => cases h
    · cases h with
      | seq h =>
        cases h with
        | loop => exact ⟨_, .peeked hb hk (Or.inr (Or.inr rfl)) hh, .silent⟩
    · cases h with
      | seq h =>
        cases h with
        | ite_true hc =>
          simp only [Expr.eval] at hc
          rcases hh with ⟨hh, rfl⟩ | ⟨-, rfl⟩
          · simp [hh, hk] at hc
          · exact ⟨_, .peeking hb hk, .silent⟩
        | ite_false hc =>
          simp only [Expr.eval] at hc
          rcases hh with ⟨-, rfl⟩ | ⟨hh, rfl⟩
          · exact ⟨_, .entered hb, .silent⟩
          · exact absurd hc (by simp [hk, hh])
  | entered hb =>
    intro a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .inside hb, .silent⟩
    | seq h => cases h
  | @inside t b hb =>
    intro a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .unlocking, .silent⟩
    | seq h =>
      obtain ⟨hb', ha⟩ := (accessRule (P.NotLock m)).step h hb
      refine ⟨.held t, .inside hb', ?_⟩
      cases a with
      | none => exact .silent
      | some l => exact .other trivial (ha l rfl)
  | unlocking =>
    intro a s' regs' h
    cases h with
    | fetchAdd v => exact ⟨_, .leaving, .leave⟩
  | leaving =>
    intro a s' regs' h
    cases h with
    | rmwStore => exact ⟨_, .clean (.other trivial), .unlock⟩

variable (P m) in
/-- The lists of accesses that `HTrans` allows from a state. -/
inductive HAccepts : Hold → List Label → Prop
  | nil {h} : HAccepts h []
  | cons {h h' a L} : P.HTrans m h (some a) h' → HAccepts h' L → HAccepts h (a :: L)

theorem HTrans.none_eq {h h'} (ht : P.HTrans m h none h') : h' = h := by
  cases ht <;> rfl

/-- A run of lowered code makes a list of accesses that `HTrans` allows. -/
theorem Runs.haccepts {s regs L} (hr : Runs s regs L) :
    ∀ {h}, P.HGood m regs h s → P.HAccepts m h L := by
  induction hr with
  | stop => exact fun _ => .nil
  | silent hs _ ih =>
    intro h hg
    obtain ⟨h', hg', htr⟩ := hg.step hs
    cases htr.none_eq
    exact ih hg'
  | visible hs _ ih =>
    intro h hg
    obtain ⟨h', hg', htr⟩ := hg.step hs
    exact .cons htr (ih hg')

/-! ## The code of the program follows `HTrans` -/

section Code

variable (hm : m < P.locks)
include hm

theorem notLock_cnt {q c} (hq : P.LockOk q c) (hc : q.IsLock → c ≠ m) {k o} :
    P.NotLock m k o (P.cnt q c) := by
  refine Or.inr ⟨fun h => ?_, fun h => ?_⟩
  · obtain ⟨rfl, rfl⟩ := cnt_inj h hq (q' := .next) hm
    exact hc trivial rfl
  · obtain ⟨rfl, rfl⟩ := cnt_inj h hq (q' := .serving) hm
    exact hc trivial rfl

theorem notLock_sig {o c j} (hj : j < P.sigLen o) {k ord} : P.NotLock m k ord (P.sig o c + j) :=
  Or.inr ⟨fun h => cnt_ne_sig (q := .next) hm hj h.symm,
    fun h => cnt_ne_sig (q := .serving) hm hj h.symm⟩

theorem notLock_data {c j} (hj : j < P.msgs) {k o} : P.NotLock m k o (P.data c + j) :=
  Or.inr ⟨data_ne_cnt (q := .next) hm hj, data_ne_cnt (q := .serving) hm hj⟩

theorem Avoid.notLock {k o x} (h : P.Avoid k o x) : P.NotLock m k o x := by
  have h1 := cnt_inside (P := P) (q := .next) hm
  have h2 := cnt_inside (P := P) (q := .serving) hm
  rcases h with h | h | h
  · exact Or.inl h
  · exact Or.inr ⟨by omega, by omega⟩
  · exact Or.inr ⟨by omega, by omega⟩

theorem lockCode_notLock {m'} (hm' : m' < P.locks) (hne : m' ≠ m) :
    (P.lockCode m').Accesses (P.NotLock m) :=
  ⟨⟨notLock_cnt (q := .next) hm hm' fun _ => hne, notLock_cnt (q := .next) hm hm' fun _ => hne⟩,
    notLock_cnt (q := .serving) (k := .read) (o := .sc) hm hm' fun _ => hne,
    notLock_cnt (q := .serving) (k := .read) (o := .sc) hm hm' fun _ => hne⟩

/-- The code of the program outside the blocks of the lock `m` does not touch its counters. -/
theorem lower_notLock {s : CStmt} (hs : P.StmtOk s) (hn : ¬ s.Locks m) :
    (P.toTaskProg.lower (P.lowerC s)).Accesses (P.NotLock m) := by
  induction s with
  | base s => exact (lower_avoid hs).mono fun _ _ _ h => Avoid.notLock hm h
  | send c e =>
    exact ⟨⟨notLock_cnt hm (q := .op .send) trivial nofun,
        notLock_cnt hm (q := .op .send) trivial nofun⟩,
      fun j hj => notLock_sig hm hj, fun j hj => notLock_data hm hj,
      fun j hj => notLock_sig hm hj⟩
  | recv r s c =>
    exact ⟨⟨notLock_cnt hm (q := .op .recv) trivial nofun,
        notLock_cnt hm (q := .op .recv) trivial nofun⟩,
      ⟨⟨notLock_cnt hm (q := .drop) trivial nofun, fun j hj => notLock_sig hm (o := .send) hj⟩,
        notLock_cnt hm (q := .drop) trivial nofun, fun j hj => notLock_sig hm (o := .send) hj⟩,
      ⟨⟨fun j hj => notLock_data hm hj, trivial⟩, trivial⟩, fun j hj => notLock_sig hm hj⟩
  | drop c =>
    exact ⟨notLock_cnt hm (q := .drop) trivial nofun, notLock_cnt hm (q := .drop) trivial nofun⟩
  | locked m' s ih =>
    have hne : m' ≠ m := fun h => hn (Or.inl h)
    exact ⟨lockCode_notLock hm hs.1 hne, ih hs.2.2 fun h => hn (Or.inr h),
      notLock_cnt (q := .serving) hm hs.1 fun _ => hne,
      notLock_cnt (q := .serving) hm hs.1 fun _ => hne⟩
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 =>
    exact ⟨ih1 hs.1 fun h => hn (Or.inl h), ih2 hs.2 fun h => hn (Or.inr h)⟩
  | loop _ _ ih => exact ih hs hn
  | tasksBlock _ ih =>
    exact ⟨(clearMarks_avoid _).mono fun _ _ _ h => Avoid.notLock hm h, ih hs hn,
      (joinMarked_avoid _).mono fun _ _ _ h => Avoid.notLock hm h⟩

theorem hclean_lower {s : CStmt} (hs : P.StmtOk s) :
    P.HClean m (P.toTaskProg.lower (P.lowerC s)) := by
  induction s with
  | base | send | recv | drop => exact .other (lower_notLock hm hs nofun)
  | locked m' s ih =>
    by_cases hne : m' = m
    · subst hne
      exact .locked (lower_notLock hm hs.2.2 hs.2.1)
    · exact .seq (.other (lockCode_notLock hm hs.1 hne))
        (.seq (ih hs.2.2) (.other ⟨notLock_cnt (q := .serving) hm hs.1 fun _ => hne,
          notLock_cnt (q := .serving) hm hs.1 fun _ => hne⟩))
  | seq _ _ ih1 ih2 => exact .seq (ih1 hs.1) (ih2 hs.2)
  | ite _ _ _ ih1 ih2 => exact .ite (ih1 hs.1) (ih2 hs.2)
  | loop _ _ ih => exact .loop (ih hs)
  | tasksBlock _ ih =>
    exact .seq (.other ((clearMarks_avoid _).mono fun _ _ _ h => Avoid.notLock hm h))
      (.seq (ih hs) (.other ((joinMarked_avoid _).mono fun _ _ _ h => Avoid.notLock hm h)))

theorem hclean_body (hP : P.Ok) (t : Nat) : P.HClean m (P.body t) := by
  cases t with
  | zero => exact hclean_lower hm hP.main
  | succ t =>
    show P.HClean m (P.toTaskProg.taskCode t)
    unfold TaskProg.taskCode
    split
    · refine .seq (.other (Avoid.notLock hm (Or.inr (Or.inr ?_))))
        (.seq (hclean_lower hm (hP.task t)) (.other (Avoid.notLock hm (Or.inr (Or.inr ?_)))))
      · simp only [TaskProg.start, toTaskProg]
        omega
      · simp only [TaskProg.done, toTaskProg]
        omega
    · exact .other trivial

end Code

/-! ## The history of an `unlock` -/

variable (P m) in
/-- No access from `i0` to before `i1` writes `serving`. -/
def NoUnlock (L : List Label) (i0 i1 : Nat) : Prop :=
  ∀ j l, i0 ≤ j → j < i1 → L[j]? = some l → ¬ (l.kind = .write ∧ l.loc = P.cnt .serving m)

theorem NoUnlock.cons {L i0 i1} {a : Label} (h : P.NoUnlock m L i0 i1) :
    P.NoUnlock m (a :: L) (i0 + 1) (i1 + 1) := by
  intro j l h0 h1 hl
  obtain ⟨j, rfl⟩ : ∃ j', j = j' + 1 := ⟨j - 1, by omega⟩
  exact h j l (by omega) (by omega) (by simpa using hl)

theorem NoUnlock.cons_zero {L i1} {a : Label} (h : P.NoUnlock m L 0 i1)
    (ha : ¬ (a.kind = .write ∧ a.loc = P.cnt .serving m)) : P.NoUnlock m (a :: L) 0 (i1 + 1) := by
  intro j l _ h1 hl
  cases j with
  | zero =>
    simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
    exact hl ▸ ha
  | succ j => exact h j l (by omega) (by omega) (by simpa using hl)

variable (P m) in
/-- The `unlock` that writes `serving` at the position `i`: its block took the ticket `t` at `iT`,
read `serving` as `t` at `iE`, and wrote `serving` nowhere between. -/
def UnlockHist (L : List Label) (i : Nat) : Prop :=
  ∃ (t : Int) (iT iE : Nat), iT + 1 < iE ∧ iE + 1 < i ∧
    L[iT]? = some ⟨.read, .sc, P.cnt .next m, t, false⟩ ∧
    L[iT + 1]? = some ⟨.write, .sc, P.cnt .next m, t + 1, true⟩ ∧
    L[iE]? = some ⟨.read, .sc, P.cnt .serving m, t, false⟩ ∧ P.NoUnlock m L (iT + 1) i

theorem UnlockHist.cons {L i} {a : Label} (h : P.UnlockHist m L i) :
    P.UnlockHist m (a :: L) (i + 1) := by
  obtain ⟨t, iT, iE, h1, h2, hT, hC, hE, hno⟩ := h
  exact ⟨t, iT + 1, iE + 1, by omega, by omega, by simpa using hT, by simpa using hC,
    by simpa using hE, hno.cons⟩

variable (P m) in
/-- What a state of `Hold` says about an `unlock` that writes `serving` at the position `i`. -/
def HPre (h : Hold) (L : List Label) (i : Nat) : Prop :=
  match h with
  | .free => False
  | .took t => L[0]? = some ⟨.write, .sc, P.cnt .next m, t + 1, true⟩ ∧
    ∃ iE, 0 < iE ∧ iE + 1 < i ∧ L[iE]? = some ⟨.read, .sc, P.cnt .serving m, t, false⟩ ∧
      P.NoUnlock m L 0 i
  | .wait t => ∃ iE, iE + 1 < i ∧ L[iE]? = some ⟨.read, .sc, P.cnt .serving m, t, false⟩ ∧
      P.NoUnlock m L 0 i
  | .held _ => 0 < i ∧ P.NoUnlock m L 0 i
  | .leaving _ => i = 0

theorem notLock_unlock {k o x} (h : P.NotLock m k o x) (hk : k = .write) :
    x ≠ P.cnt .serving m := by
  rcases h with h | h
  · rw [hk] at h
    cases h
  · exact h.2

/-- Before an `unlock`, its block took its ticket, read `serving` as the ticket, and wrote
`serving` nowhere. -/
theorem HAccepts.unlock_hist (hm : m < P.locks) {h L} (hacc : P.HAccepts m h L) :
    ∀ {i : Nat} {l : Label}, L[i]? = some l → l.kind = .write → l.loc = P.cnt .serving m →
      P.HPre m h L i ∨ P.UnlockHist m L i := by
  induction hacc with
  | nil => intro i l hl; simp at hl
  | @cons h h' a L htr _ ih =>
    intro i l hl hk hx
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other _ hav => exact (notLock_unlock hav hk hx).elim
      | take | peek | enter | leave => cases hk
      | count => exact absurd (cnt_inj (q := .next) (q' := .serving) hx hm hm).1 nofun
      | unlock => exact Or.inl rfl
    | succ i =>
      have notW {b : Label} (hb : b.kind ≠ .write ∨ b.loc ≠ P.cnt .serving m) :
          ¬ (b.kind = .write ∧ b.loc = P.cnt .serving m) := fun ⟨h1, h2⟩ => hb.elim (· h1) (· h2)
      rcases ih (by simpa using hl) hk hx with hpre | hist
      · cases htr with
        | other hc hav =>
          have ha : ¬ (a.kind = .write ∧ a.loc = P.cnt .serving m) := by
            rcases hav with h | h
            · exact notW (Or.inl (by rw [h]; nofun))
            · exact notW (Or.inr h.2)
          cases h with
          | free => exact hpre.elim
          | held t => exact Or.inl ⟨by omega, hpre.2.cons_zero ha⟩
          | took | wait | leaving => exact hc.elim
        | @take t =>
          obtain ⟨hC, iE, h0, h1, hE, hno⟩ := hpre
          exact Or.inr ⟨t, 0, iE + 1, by omega, by omega, by simp, by simpa using hC,
            by simpa using hE, hno.cons⟩
        | count =>
          obtain ⟨iE, h1, hE, hno⟩ := hpre
          exact Or.inl ⟨by simp, iE + 1, by omega, by omega, by simpa using hE,
            hno.cons_zero (notW (Or.inr fun h => absurd (cnt_inj (q := .next) (q' := .serving) h hm hm).1 nofun))⟩
        | peek =>
          obtain ⟨iE, h1, hE, hno⟩ := hpre
          exact Or.inl ⟨iE + 1, by omega, by simpa using hE, hno.cons_zero (notW (Or.inl nofun))⟩
        | enter =>
          obtain ⟨h0, hno⟩ := hpre
          exact Or.inl ⟨0, by omega, by simp, hno.cons_zero (notW (Or.inl nofun))⟩
        | leave =>
          have h0 : i = 0 := hpre
          subst h0
          exact Or.inl ⟨by omega, NoUnlock.cons_zero (fun _ _ _ h => absurd h (Nat.not_lt_zero _))
            (notW (Or.inl nofun))⟩
        | unlock => exact hpre.elim
      · exact Or.inr hist.cons

/-! ## The executions of a program with locks -/

variable {X : Execution} (run : P.toTaskProg.toProg.Run X)
include run

/-- The accesses of the thread of `e` after the writes at the start, up to `e`, follow `HTrans`. -/
theorem hview (hP : P.Ok) (hm : m < P.locks) (e : X.E) :
    P.HAccepts m .free ((List.range (run.idx e + 1 - P.pre (X.thread e))).map
      fun i => run.lab (X.thread e) (P.pre (X.thread e) + i)) :=
  Runs.haccepts (trace_rest (run.trace e)).2 (.clean (hclean_body hm hP _))

/-- A counter starts at 0. -/
theorem init_zero (hP : P.Ok) {q c} (hc : P.Has q c) {a : X.E} (haw : X.IsWrite a)
    (hal : X.loc a = P.cnt q c) (hna : ∀ r, ¬ X.rmw r a) : run.val a = 0 := by
  by_cases hp : run.idx a < P.pre (X.thread a)
  · obtain ⟨-, p, hi, -, -, hl, -⟩ := init_of run hP hp
    rw [init_val run hP hp hi]
    rcases initList_cases hi with ⟨-, rfl⟩ | ⟨-, -, rfl⟩ | ⟨c', -, j, -, rfl⟩ | ⟨c', hc', j, hj, rfl⟩
    · rfl
    · rfl
    · rfl
    · have := hP.cap c' hc'
      exact (cnt_ne_sig (o' := .recv) hc.lockOk (by simp only [sigLen]; omega)
        (hal.symm.trans hl)).elim
  · have hacc := (view run hP a).2
    have hget := view_get run (e := a) rfl (Nat.le_refl _) (by omega)
    have hrm := hacc.cnt_rmw hc hget haw hal
    rcases hacc.count_take hc hget haw hrm hal with ⟨-, h⟩ | ⟨m', hm', -⟩
    · cases h
    · obtain ⟨r, hr, hri⟩ := run.idx_down a (run.idx a - 1) (by omega)
      exact (hna r ((run.rmw_iff r a).2 ⟨hr, by omega, hrm⟩)).elim

/-- Each read of a counter sees 0 or more. -/
theorem read_nonneg (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) {q c} (hc : P.Has q c) {e : X.E} (he : X.IsRead e)
    (hel : X.loc e = P.cnt q c) : 0 ≤ run.val e := by
  obtain ⟨s, hs⟩ := run.rf_source e he
  obtain ⟨hsw, -, hsl⟩ := run.rf_write_read s e hs
  rw [← run.rf_val s e hs]
  exact (counter run hP wf hc).ge_init wf hcons ho hfin
    (fun a ha hal hna => by rw [init_zero run hP hc ha hal hna]; exact Int.le_refl 0) s hsw
    (hsl.trans hel)

/-- Before each `unlock` of the lock `m`, its thread took a ticket, then read `serving` as the
ticket, and wrote `serving` nowhere from the ticket to the `unlock`. -/
theorem unlock_hist (hP : P.Ok) (hm : m < P.locks) {w : X.E} (hw : X.IsWrite w)
    (hwl : X.loc w = P.cnt .serving m) (hrm : run.isRmw w = true) :
    ∃ t t' e, X.rmw t t' ∧ X.loc t' = P.cnt .next m ∧ X.thread t = X.thread w ∧
      X.IsRead e ∧ X.loc e = P.cnt .serving m ∧ run.val e = run.val t ∧
      X.thread e = X.thread w ∧ run.idx t < run.idx e ∧ run.idx e + 1 < run.idx w ∧
      ∀ d, X.thread d = X.thread w → run.idx t < run.idx d → run.idx d < run.idx w →
        ¬ (X.IsWrite d ∧ X.loc d = P.cnt .serving m) := by
  have hp : P.pre (X.thread w) ≤ run.idx w := by
    refine Nat.le_of_not_lt fun hp => ?_
    obtain ⟨-, -, -, -, -, -, h⟩ := init_of run hP hp
    rw [hrm] at h
    cases h
  have hacc := hview run hP hm w
  have hget := view_get run (e := w) rfl (Nat.le_refl _) hp
  rcases hacc.unlock_hist hm hget hw hwl with h | ⟨t, iT, iE, h1, h2, hT, hC, hE, hno⟩
  · exact h.elim
  obtain ⟨d1, hd1t, hd1i⟩ := run.idx_down w (P.pre (X.thread w) + iT) (by omega)
  obtain ⟨d2, hd2t, hd2i⟩ := run.idx_down w (P.pre (X.thread w) + iT + 1) (by omega)
  obtain ⟨e, het, hei⟩ := run.idx_down w (P.pre (X.thread w) + iE) (by omega)
  have g1 := view_get run (e := w) hd1t (by omega) (by omega)
  rw [show run.idx d1 - P.pre (X.thread w) = iT by omega, hT] at g1
  have g2 := view_get run (e := w) hd2t (by omega) (by omega)
  rw [show run.idx d2 - P.pre (X.thread w) = iT + 1 by omega, hC] at g2
  have g3 := view_get run (e := w) het (by omega) (by omega)
  rw [show run.idx e - P.pre (X.thread w) = iE by omega, hE] at g3
  simp only [Option.some.injEq, Label.mk.injEq] at g1 g2 g3
  obtain ⟨-, -, -, hv1, -⟩ := g1
  obtain ⟨-, -, hl2, -, hr2⟩ := g2
  obtain ⟨hk3, -, hl3, hv3, -⟩ := g3
  refine ⟨d1, d2, e, (run.rmw_iff d1 d2).2 ⟨hd1t.trans hd2t.symm, by omega, hr2.symm⟩, hl2.symm,
    hd1t, hk3.symm, hl3.symm, by rw [← hv3, ← hv1], het, by omega, by omega, ?_⟩
  intro d hdt hd1 hd2 hdw
  exact hno (run.idx d - P.pre (X.thread w)) _ (by omega) (by omega)
    (view_get run hdt (by omega) (by omega)) hdw

/-- The `unlock` of the ticket `k` reads `serving` as `k`. Its block read `serving` as `k`
before it, so it reads `k` or more. If it read more, an `unlock` read `k` before it. That one is
the `unlock` of the same block, since each ticket has one block. -/
theorem unlock_ticket (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) (hm : m < P.locks) {r w : X.E} (hrw : X.rmw r w)
    (hwl : X.loc w = P.cnt .serving m) :
    ∃ t t', X.rmw t t' ∧ X.loc t' = P.cnt .next m ∧ run.val t = run.val r ∧
      X.thread t = X.thread w ∧ run.idx t < run.idx r ∧
      ∀ d, X.thread d = X.thread w → run.idx t < run.idx d → run.idx d < run.idx w →
        ¬ (X.IsWrite d ∧ X.loc d = P.cnt .serving m) := by
  have hS := counter run hP wf (q := .serving) hm
  have hN := counter run hP wf (q := .next) hm
  have key : ∀ n (r w : X.E), (run.val r).toNat = n → X.rmw r w → X.loc w = P.cnt .serving m →
      ∃ t t', X.rmw t t' ∧ X.loc t' = P.cnt .next m ∧ run.val t = run.val r ∧
        X.thread t = X.thread w ∧ run.idx t < run.idx r ∧
        ∀ d, X.thread d = X.thread w → run.idx t < run.idx d → run.idx d < run.idx w →
          ¬ (X.IsWrite d ∧ X.loc d = P.cnt .serving m) := by
    intro n
    induction n using Nat.strongRecOn with
    | ind n ih =>
      intro r w hn hrw hwl
      obtain ⟨hrl, -⟩ := hS.add r w hrw hwl
      obtain ⟨hrr, hww, -, -⟩ := wf.rmw_atomic r w hrw
      obtain ⟨hthr, hidx, hrm⟩ := (run.rmw_iff r w).1 hrw
      obtain ⟨T, T', E, hTT, hT'l, hTt, hEr, hEl, hEv, hEt, hTE, hEw, hno⟩ :=
        unlock_hist run hP hm hww hwl hrm
      refine ⟨T, T', hTT, hT'l, ?_, hTt, by omega, hno⟩
      have hTl := (hN.add T T' hTT hT'l).1
      have hT0 := read_nonneg run hP wf hcons ho hfin (q := .next) hm (wf.rmw_atomic T T' hTT).1 hTl
      have hr0 := read_nonneg run hP wf hcons ho hfin (q := .serving) hm hrr hrl
      rcases Int.lt_trichotomy (run.val T) (run.val r) with hlt | heq | hgt
      · obtain ⟨r0, w0, hrw0, hw0l, hv0⟩ := hS.exists_of_lt wf (v := run.val T)
          (fun a ha hal hna => by rw [init_zero run hP (q := .serving) hm ha hal hna]; exact hT0)
          r hrr hrl hlt
        obtain ⟨T0, T0', hTT0, hT0l, hv0', hT0t, hT0r, hno0⟩ :=
          ih _ (by omega) r0 w0 rfl hrw0 hw0l
        have := hN.tickets wf hcons ho hfin hTT0 hT0l hTT hT'l (hv0'.trans hv0)
        subst this
        obtain ⟨hthr0, hidx0, -⟩ := (run.rmw_iff r0 w0).1 hrw0
        have hww0 := (wf.rmw_atomic r0 w0 hrw0).2.1
        rcases Nat.lt_trichotomy (run.idx r0) (run.idx r) with h | h | h
        · exact (hno w0 (hT0t.symm.trans hTt) (by omega) (by omega) ⟨hww0, hw0l⟩).elim
        · have := run.idx_inj r0 r (by rw [hthr0, hthr, ← hT0t, hTt]) h
          subst this
          omega
        · exact (hno0 w (hTt.symm.trans hT0t) (by omega) (by omega) ⟨hww, hwl⟩).elim
      · exact heq
      · obtain ⟨sE, hsE⟩ := run.rf_source _ hEr
        obtain ⟨sr, hsr⟩ := run.rf_source r hrr
        obtain ⟨hsEw, -, hsEl⟩ := run.rf_write_read sE _ hsE
        obtain ⟨hsrw, -, hsrl⟩ := run.rf_write_read sr r hsr
        have hvE := run.rf_val sE _ hsE
        have hvr := run.rf_val sr r hsr
        have hne : sr ≠ sE := by
          rintro rfl
          omega
        rcases ho.total sr sE hsrw hsEw (by rw [hsrl, hsEl, hrl, hEl]) hne with h | h
        · have hpo : X.po _ r := (run.po_iff _ r).2 ⟨hEt.trans hthr.symm, by omega⟩
          exact (hcons.coherence _ r (Relation.TransGen.single (Or.inl hpo))
            (Or.inr (Relation.TransGen.tail (Relation.TransGen.single (Or.inr (Or.inr ⟨sr, hsr, h⟩)))
              (Or.inl hsE)))).elim
        · have := hS.mono wf hcons ho hfin sE sr hsEw hsrw (hsEl.trans hEl) (hsrl.trans hrl) h
          omega
  exact key _ r w rfl hrw hwl

/-- Two tasks never hold the lock `m` at the same time. The thread that took the ticket `k` with
`t` holds the lock up to its `unlock`. Each event `d` of that thread before the `unlock` happens
before each read of `serving` that sees more than `k`. The `lock` of a later ticket ends with such
a read, so `d` happens before each event of the block of that ticket. -/
theorem lock_excl (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder)
    (hfin : X.Finite) (hm : m < P.locks) {t t' d e b : X.E} (htt : X.rmw t t')
    (htl : X.loc t' = P.cnt .next m) (hdt : X.thread d = X.thread t)
    (hheld : ∀ w, X.thread w = X.thread t → run.idx t < run.idx w → run.idx w ≤ run.idx d →
      ¬ (X.IsWrite w ∧ X.loc w = P.cnt .serving m))
    (he : X.IsRead e) (hel : X.loc e = P.cnt .serving m) (hlt : run.val t < run.val e)
    (heb : e = b ∨ X.po e b) : X.hb d b := by
  have hS := counter run hP wf (q := .serving) hm
  have hN := counter run hP wf (q := .next) hm
  have htl' := (hN.add t t' htt htl).1
  have ht0 := read_nonneg run hP wf hcons ho hfin (q := .next) hm (wf.rmw_atomic t t' htt).1 htl'
  obtain ⟨U, W, hUW, hWl, hUv⟩ := hS.exists_of_lt wf (v := run.val t)
    (fun a ha hal hna => by rw [init_zero run hP (q := .serving) hm ha hal hna]; exact ht0)
    e he hel hlt
  obtain ⟨T, T', hTT, hT'l, hTv, hTt, hTU, -⟩ := unlock_ticket run hP wf hcons ho hfin hm hUW hWl
  have := hN.tickets wf hcons ho hfin hTT hT'l htt htl (hTv.trans hUv)
  subst this
  obtain ⟨hUt, hWi, -⟩ := (run.rmw_iff U W).1 hUW
  have hWw := (wf.rmw_atomic U W hUW).2.1
  have hdU : d = U ∨ X.po d U := by
    rcases Nat.lt_trichotomy (run.idx d) (run.idx U) with h | h | h
    · exact Or.inr ((run.po_iff d U).2 ⟨by rw [hdt, hTt, hUt], h⟩)
    · exact Or.inl (run.idx_inj d U (by rw [hdt, hTt, hUt]) h)
    · exact (hheld W hTt.symm (by omega) (by omega) ⟨hWw, hWl⟩).elim
  exact unlock_hb_lock run hP wf hcons ho hfin hm hUW hWl hdU he hel (by omega) heb

end ChanProg

end Vx
