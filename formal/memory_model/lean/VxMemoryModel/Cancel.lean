import VxMemoryModel.Channel

/-!
# The first failure of a `tasks` block, and cancellation

A program with errors has a failure slot and a cancel flag. Both start at 0. `fail v` writes the
error `v` to the slot with a CAS from 0. If the CAS wrote the slot, it then sets the cancel flag
with a release write. `checkCancel` is a wait that can be cancelled: it reads the flag with
acquire, and if the flag is set, it fails with `cancelled`. `result r` reads the slot: the value of
the block.

`FState` follows one thread through these steps. From it, `one_failure_wins` shows that at most
one CAS writes the slot. `block_value_is_failure` shows that a read of the slot that the winning
CAS happens before gets its value. `block_value_not_cancelled` shows that `cancelled` never
reaches the slot. A task that sees the flag set also sees the slot set, so its CAS fails.
`each_error_once` shows that the error of at most one failure moves into the slot, and that each
other failure drops its error. A task with no `result` never reads the slot, so `join` gives no
error to its caller.

This is safety only: the model does not say that a cancelled task stops.
-/

namespace Vx

/-- A release write orders the events before it before the events after an acquire read of
it. -/
theorem Execution.hb_of_release {X : Execution} {a w r b : X.E}
    (haw : a = w ∨ X.po a w) (hw : X.IsWrite w) (hwr : X.IsRelease w) (hwa : X.IsAtomic w)
    (hrf : X.rf w r) (hra : X.IsAcquire r) (hrat : X.IsAtomic r) (hrb : r = b ∨ X.po r b) :
    X.hb a b := by
  have hsw : X.hb w r := Relation.TransGen.single (Or.inr ⟨hwr, hra, w, w, r, Or.inl rfl,
    ⟨w, hw, Or.inl rfl, hw, hwa, Or.inl rfl⟩, hrf, hrat, Or.inl rfl⟩)
  have h1 : X.hb a r := haw.elim (fun h => h ▸ hsw) (fun h => Relation.TransGen.head (Or.inl h) hsw)
  exact hrb.elim (fun h => h ▸ h1) (fun h => h1.tail (Or.inl h))

namespace ChanProg

variable (P : ChanProg)

/-- An access that is not to the failure slot or the cancel flag. -/
def NotFail (k : Kind) (_ : Order) (x : Nat) : Prop :=
  k = .fence ∨ (x ≠ P.slot ∧ x ≠ P.flag)

/-- An access that is not an acquire read of the slot: not a `result`. -/
def NoResult (k : Kind) (o : Order) (x : Nat) : Prop :=
  k = .read → x = P.slot → o = .sc

/-- Where a thread is in a failure. -/
inductive FState where
  | free
  /-- It read the cancel flag set, and did not read the slot yet. -/
  | saw
  /-- Its CAS read 0 from the slot. `c` is true for the CAS of a task that was cancelled. -/
  | zero (c : Bool)
  /-- Its CAS wrote the slot, and it did not set the cancel flag yet. -/
  | won

/-- The accesses that a thread can make in each state of `FState`. -/
inductive FTrans : FState → Option Label → FState → Prop
  | silent {f} : FTrans f none f
  | other {l : Label} : P.NotFail l.kind l.order l.loc → FTrans .free (some l) .free
  /-- `checkCancel` reads the cancel flag as 0. -/
  | clear : FTrans .free (some ⟨.read, .acq, P.flag, 0, false⟩) .free
  /-- `checkCancel` reads the cancel flag set. -/
  | see {v} : v ≠ 0 → FTrans .free (some ⟨.read, .acq, P.flag, v, false⟩) .saw
  /-- The CAS of a failure reads the slot set, and does not write it. -/
  | lose {f v} : (f = .free ∨ f = .saw) → v ≠ 0 →
    FTrans f (some ⟨.read, .sc, P.slot, v, false⟩) .free
  | attempt : FTrans .free (some ⟨.read, .sc, P.slot, 0, false⟩) (.zero false)
  | attemptCancel : FTrans .saw (some ⟨.read, .sc, P.slot, 0, false⟩) (.zero true)
  /-- The CAS writes the slot. Only the CAS of a cancelled task writes `cancelled`. -/
  | win {c v} : v ≠ 0 → (v = cancelled → c = true) →
    FTrans (.zero c) (some ⟨.write, .sc, P.slot, v, true⟩) .won
  | cancel : FTrans .won (some ⟨.write, .rel, P.flag, 1, false⟩) .free
  /-- `result` reads the slot. -/
  | result {v} : FTrans .free (some ⟨.read, .acq, P.slot, v, false⟩) .free

/-- Lowered code, with each failure, each `checkCancel` and each `result` whole. A `result` needs
`R`. -/
inductive FClean (R : Prop) : Stmt → Prop
  | other {s} : s.Accesses P.NotFail → FClean R s
  | fail {v} : v ≠ 0 → v ≠ cancelled → FClean R (P.failCode v)
  | check : FClean R P.checkCode
  | result {r} : R → FClean R (.load r P.slot .acq)
  | seq {s t} : FClean R s → FClean R t → FClean R (.seq s t)
  | ite {e s t} : FClean R s → FClean R t → FClean R (.ite e s t)
  | loop {e s} : FClean R s → FClean R (.loop e s)

/-- The states of lowered code, with their state of `FState`. After the CAS of a failure or the
read of the cancel flag, the register `ctrReg` holds the value that it read. -/
inductive FGood (regs : Nat → Int) : FState → Stmt → Prop
  | clean {s} : P.FClean True s → FGood regs .free s
  | seq {f s t} : FGood regs f s → P.FClean True t → FGood regs f (.seq s t)
  | zero {c v} : v ≠ 0 → (v = cancelled → c = true) → regs P.ctrReg = 0 →
    FGood regs (.zero c) (.seq (.rmwStore P.slot v .sc) P.raise)
  | wrote : regs P.ctrReg = 0 → FGood regs .won (.seq .skip P.raise)
  | raising : regs P.ctrReg = 0 → FGood regs .won P.raise
  | setting : FGood regs .won (.store P.flag (.const 1) .rel)
  | lost : regs P.ctrReg ≠ 0 → FGood regs .free (.seq .skip P.raise)
  | lostRaise : regs P.ctrReg ≠ 0 → FGood regs .free P.raise
  | looked {f} : (regs P.ctrReg = 0 ∧ f = .free ∨ regs P.ctrReg ≠ 0 ∧ f = .saw) →
    FGood regs f (.seq .skip P.checkTail)
  | tail {f} : (regs P.ctrReg = 0 ∧ f = .free ∨ regs P.ctrReg ≠ 0 ∧ f = .saw) →
    FGood regs f P.checkTail
  | cancelling : FGood regs .saw (P.failCode cancelled)

variable {P}

theorem FClean.step {s} (hcl : P.FClean True s) :
    ∀ {regs a s' regs'}, Step s regs a s' regs' → ∃ f, P.FGood regs' f s' ∧ P.FTrans .free a f := by
  induction hcl with
  | other hs =>
    intro regs a s' regs' h
    obtain ⟨hs', ha⟩ := (accessRule P.NotFail).step h hs
    refine ⟨_, .clean (.other hs'), ?_⟩
    cases a with
    | none => exact .silent
    | some l => exact .other (ha l rfl)
  | fail hv hc =>
    intro regs a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | cas_eq v' he =>
        simp only [Expr.eval] at he
        subst he
        exact ⟨_, .zero hv (fun h => (hc h).elim) (by simp [update]), .attempt⟩
      | cas_ne v' he =>
        simp only [Expr.eval] at he
        exact ⟨_, .lost (by simpa [update] using he), .lose (Or.inl rfl) he⟩
  | check =>
    intro regs a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | load v =>
        by_cases hv : v = 0
        · subst hv
          exact ⟨_, .looked (Or.inl ⟨by simp [update], rfl⟩), .clear⟩
        · exact ⟨_, .looked (Or.inr ⟨by simpa [update] using hv, rfl⟩), .see hv⟩
  | result _ =>
    intro regs a s' regs' h
    cases h with
    | load v => exact ⟨_, .clean (.other trivial), .result⟩
  | seq _ ht ih =>
    intro regs a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .clean ht, .silent⟩
    | seq h =>
      obtain ⟨f, hg, htr⟩ := ih h
      exact ⟨f, .seq hg ht, htr⟩
  | ite hs ht =>
    intro regs a s' regs' h
    cases h with
    | ite_true => exact ⟨_, .clean hs, .silent⟩
    | ite_false => exact ⟨_, .clean ht, .silent⟩
  | loop hs =>
    intro regs a s' regs' h
    cases h with
    | loop => exact ⟨_, .clean (.ite (.seq hs (.loop hs)) (.other trivial)), .silent⟩

/-- Each step of lowered code keeps `FGood`, and its access follows `FTrans`. -/
theorem FGood.step {f s regs} (hg : P.FGood regs f s) :
    ∀ {a s' regs'}, Step s regs a s' regs' → ∃ f', P.FGood regs' f' s' ∧ P.FTrans f a f' := by
  induction hg with
  | clean hcl => exact hcl.step
  | seq hg ht ih =>
    intro a s' regs' h
    cases h with
    | seq_skip =>
      cases hg with
      | clean => exact ⟨_, .clean ht, .silent⟩
    | seq h =>
      obtain ⟨f', hg', htr⟩ := ih h
      exact ⟨f', .seq hg' ht, htr⟩
  | zero hv hc hk =>
    intro a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | rmwStore => exact ⟨_, .wrote hk, .win hv hc⟩
  | wrote hk =>
    intro a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .raising hk, .silent⟩
    | seq h => cases h
  | raising hk =>
    intro a s' regs' h
    cases h with
    | ite_true => exact ⟨_, .setting, .silent⟩
    | ite_false hc => simp [Expr.eval, hk] at hc
  | setting =>
    intro a s' regs' h
    cases h with
    | store => exact ⟨_, .clean (.other trivial), .cancel⟩
  | lost hk =>
    intro a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .lostRaise hk, .silent⟩
    | seq h => cases h
  | lostRaise hk =>
    intro a s' regs' h
    cases h with
    | ite_true hc => simp [Expr.eval, hk] at hc
    | ite_false => exact ⟨_, .clean (.other trivial), .silent⟩
  | looked hf =>
    intro a s' regs' h
    cases h with
    | seq_skip => exact ⟨_, .tail hf, .silent⟩
    | seq h => cases h
  | tail hf =>
    intro a s' regs' h
    cases h with
    | ite_true hc =>
      simp only [Expr.eval] at hc
      rcases hf with ⟨hk, rfl⟩ | ⟨-, rfl⟩
      · exact absurd hk hc
      · exact ⟨_, .cancelling, .silent⟩
    | ite_false hc =>
      simp only [Expr.eval] at hc
      rcases hf with ⟨-, rfl⟩ | ⟨hk, rfl⟩
      · exact ⟨_, .clean (.other trivial), .silent⟩
      · exact absurd hc hk
  | cancelling =>
    intro a s' regs' h
    cases h with
    | seq h =>
      cases h with
      | cas_eq v' he =>
        simp only [Expr.eval] at he
        subst he
        exact ⟨_, .zero (by simp [cancelled, Expr.eval]) (fun _ => rfl) (by simp [update]), .attemptCancel⟩
      | cas_ne v' he =>
        simp only [Expr.eval] at he
        exact ⟨_, .lost (by simpa [update] using he), .lose (Or.inr rfl) he⟩

variable (P) in
/-- The lists of accesses that `FTrans` allows from a state. -/
inductive FAccepts : FState → List Label → Prop
  | nil {f} : FAccepts f []
  | cons {f f' a L} : P.FTrans f (some a) f' → FAccepts f' L → FAccepts f (a :: L)

theorem FTrans.none_eq {f f'} (h : P.FTrans f none f') : f' = f := by
  cases h; rfl

/-- A run of lowered code makes a list of accesses that `FTrans` allows. -/
theorem Runs.faccepts {s regs L} (hr : Runs s regs L) :
    ∀ {f}, P.FGood regs f s → P.FAccepts f L := by
  induction hr with
  | stop => exact fun _ => .nil
  | silent hs _ ih =>
    intro f hg
    obtain ⟨f', hg', htr⟩ := hg.step hs
    cases htr.none_eq
    exact ih hg'
  | visible hs _ ih =>
    intro f hg
    obtain ⟨f', hg', htr⟩ := hg.step hs
    exact .cons htr (ih hg')

/-! ## The code of the program follows `FTrans` -/

theorem slot_ne_flag : P.slot ≠ P.flag := by
  unfold flag
  omega

theorem notFail_of_lt {k o x} (h : x < P.slot) : P.NotFail k o x :=
  Or.inr ⟨by omega, by unfold flag; omega⟩

theorem notFail_of_top {k o x} (h : x < P.top) : P.NotFail k o x :=
  notFail_of_lt (Nat.lt_of_lt_of_le h top_le_slot)

theorem notFail_of_flags {k o x} (h : P.flag < x) : P.NotFail k o x :=
  Or.inr ⟨by unfold flag at h; omega, by omega⟩

/-- The code of the program and the code of `spawn` and `join` stay below the failure slot. -/
theorem lower_notFail {s : Stmt} (h : s.Below P.places P.count) :
    (P.toTaskProg.lower s).Accesses P.NotFail := by
  have hps : P.places ≤ P.slot := by unfold slot top; omega
  induction s with
  | spawnAt base =>
    refine ⟨trivial, fun j hj => notFail_of_flags ?_⟩
    simp only [Stmt.Below] at h
    simp only [TaskProg.start, toTaskProg, flag, slot]
    omega
  | tasksBlock s ih =>
    refine ⟨P.toTaskProg.clearMarks_accesses _ _, ih h,
      P.toTaskProg.joinMarked_accesses fun u hu => notFail_of_flags ?_⟩
    have := TaskProg.spawnIds_lt (s := s) h u hu
    simp only [TaskProg.done, toTaskProg, flag, slot]
    omega
  | joinAt base =>
    refine fun j hj => notFail_of_flags ?_
    simp only [Stmt.Below] at h
    simp only [TaskProg.done, toTaskProg, flag, slot]
    omega
  | load | store | rmwStore | wait => exact notFail_of_lt (by simp only [Stmt.Below] at h; omega)
  | fetchAdd | cas =>
    exact ⟨notFail_of_lt (by simp only [Stmt.Below] at h; omega),
      notFail_of_lt (by simp only [Stmt.Below] at h; omega)⟩
  | loadAt | storeAt | waitAt =>
    exact fun j hj => notFail_of_lt (by simp only [Stmt.Below] at h; omega)
  | fence => exact Or.inl rfl
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | loop _ _ ih => exact ih h
  | skip | assign | assignAt => trivial

/-- The tasks that lowered code starts have ids below `count`. -/
theorem spawnIds_lowerC {s : CStmt} (hs : P.StmtOk s) : ∀ u ∈ (P.lowerC s).spawnIds, u < P.count := by
  induction s with
  | base s => exact TaskProg.spawnIds_lt hs
  | send | recv | drop | fail | checkCancel | result | tasksBlock =>
    intro u hu
    simp [lowerC, Stmt.spawnIds] at hu
  | locked _ _ ih =>
    intro u hu
    simp only [lowerC, Stmt.spawnIds, List.append_nil, List.nil_append] at hu
    exact ih hs.2.2 u hu
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 =>
    intro u hu
    rcases List.mem_append.1 hu with h | h
    · exact ih1 hs.1 u h
    · exact ih2 hs.2 u h
  | loop _ _ ih => exact ih hs

theorem fclean_lower {R : Prop} {s : CStmt} (hs : P.StmtOk s) (hr : s.Results → R) :
    P.FClean R (P.toTaskProg.lower (P.lowerC s)) := by
  induction s with
  | base s => exact .other (lower_notFail hs)
  | send c e =>
    exact .other ⟨⟨notFail_of_top (cnt_inside (q := .op .send) hs).2,
        notFail_of_top (cnt_inside (q := .op .send) hs).2⟩,
      fun j hj => notFail_of_top (sig_inside hs hj).2, fun j hj => notFail_of_top (data_inside hs hj).2,
      fun j hj => notFail_of_top (sig_inside hs hj).2⟩
  | recv r s c =>
    exact .other ⟨⟨notFail_of_top (cnt_inside (q := .op .recv) hs.1).2,
        notFail_of_top (cnt_inside (q := .op .recv) hs.1).2⟩,
      ⟨⟨notFail_of_top (cnt_inside (q := .drop) hs.1).2,
          fun j hj => notFail_of_top (sig_inside (o := .send) hs.1 hj).2⟩,
        notFail_of_top (cnt_inside (q := .drop) hs.1).2,
        fun j hj => notFail_of_top (sig_inside (o := .send) hs.1 hj).2⟩,
      ⟨⟨fun j hj => notFail_of_top (data_inside hs.1 hj).2, trivial⟩, trivial⟩,
      fun j hj => notFail_of_top (sig_inside hs.1 hj).2⟩
  | drop c =>
    exact .other ⟨notFail_of_top (cnt_inside (q := .drop) hs).2,
      notFail_of_top (cnt_inside (q := .drop) hs).2⟩
  | locked m s ih =>
    exact .seq (.other ⟨⟨notFail_of_top (cnt_inside (q := .next) hs.1).2,
        notFail_of_top (cnt_inside (q := .next) hs.1).2⟩,
      notFail_of_top (k := .read) (o := .sc) (cnt_inside (q := .serving) hs.1).2,
      notFail_of_top (k := .read) (o := .sc) (cnt_inside (q := .serving) hs.1).2⟩)
      (.seq (ih hs.2.2 hr) (.other ⟨notFail_of_top (cnt_inside (q := .serving) hs.1).2,
        notFail_of_top (cnt_inside (q := .serving) hs.1).2⟩))
  | seq _ _ ih1 ih2 => exact .seq (ih1 hs.1 fun h => hr (.inl h)) (ih2 hs.2 fun h => hr (.inr h))
  | ite _ _ _ ih1 ih2 => exact .ite (ih1 hs.1 fun h => hr (.inl h)) (ih2 hs.2 fun h => hr (.inr h))
  | loop _ _ ih => exact .loop (ih hs hr)
  | tasksBlock s ih =>
    refine .seq (.other (P.toTaskProg.clearMarks_accesses _ _))
      (.seq (ih hs hr) (.other (P.toTaskProg.joinMarked_accesses fun u hu => notFail_of_flags ?_)))
    have := spawnIds_lowerC (s := s) hs u hu
    simp only [TaskProg.done, toTaskProg, flag, slot]
    omega
  | fail v => exact .fail (by have := hs.2; omega) (by have := hs.2; simp only [cancelled]; omega)
  | checkCancel => exact .check
  | result => exact .result (hr trivial)

theorem fclean_task {R : Prop} (hP : P.Ok) (t : Nat) (hr : (P.task t).Results → R) :
    P.FClean R (P.body (t + 1)) := by
  show P.FClean R (P.toTaskProg.taskCode t)
  unfold TaskProg.taskCode
  split
  · refine .seq (.other (notFail_of_flags ?_)) (.seq (fclean_lower (hP.task t) hr)
      (.other (notFail_of_flags ?_)))
    · simp only [TaskProg.start, toTaskProg, flag, slot] at *
      omega
    · simp only [TaskProg.done, toTaskProg, flag, slot] at *
      omega
  · exact .other trivial

theorem fclean_body (hP : P.Ok) (t : Nat) : P.FClean True (P.body t) := by
  cases t with
  | zero => exact fclean_lower hP.main fun _ => trivial
  | succ t => exact fclean_task hP t fun _ => trivial

/-- Code with no `result` never reads the slot with acquire. -/
theorem FClean.noResult {s} (h : P.FClean False s) : s.Accesses P.NoResult := by
  have hsf := slot_ne_flag (P := P)
  induction h with
  | other hs =>
    refine hs.mono fun k o x h hk hx => ?_
    rcases h with h | h
    · rw [h] at hk
      cases hk
    · exact (h.1 hx).elim
  | fail => exact ⟨⟨fun _ _ => rfl, nofun⟩, nofun, trivial⟩
  | check => exact ⟨fun _ hx => (hsf hx.symm).elim, ⟨⟨fun _ _ => rfl, nofun⟩, nofun, trivial⟩, trivial⟩
  | result h => exact h.elim
  | seq _ _ ih1 ih2 | ite _ _ ih1 ih2 => exact ⟨ih1, ih2⟩
  | loop _ ih => exact ih

/-! ## The history of a failure -/

/-- What `FTrans` says about each access to the slot and the flag. -/
theorem FAccepts.labels {f L} (h : P.FAccepts f L) :
    ∀ {i : Nat} {l : Label}, L[i]? = some l →
      (l.kind = .write → l.loc = P.slot → l.rmw = true ∧ l.order = .sc ∧ l.val ≠ 0) ∧
      (l.kind = .write → l.loc = P.flag → l.order = .rel) ∧
      (l.kind = .read → l.loc = P.flag → l.order = .acq) := by
  induction h with
  | nil => intro i l h; simp at h
  | cons htr _ ih =>
    intro i l hl
    cases i with
    | succ i => exact ih (by simpa using hl)
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      have hsf := slot_ne_flag (P := P)
      cases htr with
      | other hnf =>
        rcases hnf with h | ⟨h1, h2⟩
        · refine ⟨fun hk => ?_, fun hk => ?_, fun hk => ?_⟩ <;> rw [h] at hk <;> cases hk
        · exact ⟨fun _ hx => (h1 hx).elim, fun _ hx => (h2 hx).elim, fun _ hx => (h2 hx).elim⟩
      | clear | see => exact ⟨nofun, nofun, fun _ _ => rfl⟩
      | lose | attempt | attemptCancel | result => exact ⟨nofun, nofun, fun _ hx => (hsf hx).elim⟩
      | win hv => exact ⟨fun _ _ => ⟨rfl, rfl, hv⟩, fun _ hx => (hsf hx).elim, nofun⟩
      | cancel => exact ⟨fun _ hx => (hsf hx.symm).elim, fun _ _ => rfl, nofun⟩

variable (P) in
/-- What a state says about a write of the value `v` to the slot at the position `i`. -/
def WPre (f : FState) (L : List Label) (i : Nat) (v : Int) : Prop :=
  match f with
  | .free | .won => False
  | .saw => i = 1 ∧ L[0]? = some ⟨.read, .sc, P.slot, 0, false⟩
  | .zero c => i = 0 ∧ (v = cancelled → c = true)

variable (P) in
/-- The write of `v` to the slot at the position `i` comes right after a read of the slot as 0.
If `v` is `cancelled`, a read of the cancel flag set comes right before that read. -/
def WHist (L : List Label) (i : Nat) (v : Int) : Prop :=
  ∃ i', i = i' + 1 ∧ L[i']? = some ⟨.read, .sc, P.slot, 0, false⟩ ∧
    (v = cancelled → ∃ i'', i' = i'' + 1 ∧ ∃ u, u ≠ 0 ∧
      L[i'']? = some ⟨.read, .acq, P.flag, u, false⟩)

theorem WHist.cons {L i v} {a : Label} (h : P.WHist L i v) : P.WHist (a :: L) (i + 1) v := by
  obtain ⟨i', rfl, h1, h2⟩ := h
  refine ⟨i' + 1, rfl, by simpa using h1, fun hv => ?_⟩
  obtain ⟨i'', rfl, u, hu, h3⟩ := h2 hv
  exact ⟨i'' + 1, rfl, u, hu, by simpa using h3⟩

/-- Before a CAS writes the slot, it read the slot as 0. Before the CAS of a cancelled task, its
thread read the cancel flag set. -/
theorem FAccepts.win_hist {f L} (h : P.FAccepts f L) :
    ∀ {i : Nat} {l : Label}, L[i]? = some l → l.kind = .write → l.loc = P.slot →
      P.WPre f L i l.val ∨ P.WHist L i l.val := by
  induction h with
  | nil => intro i l h; simp at h
  | @cons f f' a L htr _ ih =>
    intro i l hl hk hx
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hnf =>
        rcases hnf with h | ⟨h1, -⟩
        · rw [h] at hk
          cases hk
        · exact (h1 hx).elim
      | clear | see | lose | attempt | attemptCancel | result => cases hk
      | win _ hc => exact Or.inl ⟨rfl, hc⟩
      | cancel => exact (slot_ne_flag hx.symm).elim
    | succ i =>
      rcases ih (by simpa using hl) hk hx with hpre | hist
      · cases f' with
        | free | won => exact hpre.elim
        | saw =>
          obtain ⟨rfl, h0⟩ := hpre
          cases htr with
          | see hv => exact Or.inr ⟨1, rfl, by simpa using h0, fun _ => ⟨0, rfl, _, hv, by simp⟩⟩
        | zero c =>
          obtain ⟨rfl, hc⟩ := hpre
          cases htr with
          | attempt => exact Or.inr ⟨0, rfl, by simp, fun h => absurd (hc h) nofun⟩
          | attemptCancel => exact Or.inl ⟨rfl, by simp⟩
      · exact Or.inr hist.cons

/-- Before a write of the cancel flag, its thread wrote the slot. -/
theorem FAccepts.flag_hist {f L} (h : P.FAccepts f L) :
    ∀ {i : Nat} {l : Label}, L[i]? = some l → l.kind = .write → l.loc = P.flag →
      (f = .won ∧ i = 0) ∨ ∃ i' v, i = i' + 1 ∧ L[i']? = some ⟨.write, .sc, P.slot, v, true⟩ := by
  induction h with
  | nil => intro i l h; simp at h
  | @cons f f' a L htr _ ih =>
    intro i l hl hk hx
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hnf =>
        rcases hnf with h | ⟨-, h2⟩
        · rw [h] at hk
          cases hk
        · exact (h2 hx).elim
      | clear | see | lose | attempt | attemptCancel | result => cases hk
      | win => exact (slot_ne_flag hx).elim
      | cancel => exact Or.inl ⟨rfl, rfl⟩
    | succ i =>
      rcases ih (by simpa using hl) hk hx with ⟨rfl, rfl⟩ | ⟨i', v, rfl, h'⟩
      · cases htr with
        | @win c v _ _ => exact Or.inr ⟨0, v, rfl, by simp⟩
      · exact Or.inr ⟨i' + 1, v, rfl, by simpa using h'⟩

/-- Each access to the slot or the flag is atomic. -/
theorem FAccepts.atomic {f L} (h : P.FAccepts f L) :
    ∀ {i : Nat} {l : Label}, L[i]? = some l → l.kind ≠ .fence → (l.loc = P.slot ∨ l.loc = P.flag) →
      l.order ≠ .plain := by
  induction h with
  | nil => intro i l h; simp at h
  | cons htr _ ih =>
    intro i l hl hk hx
    cases i with
    | succ i => exact ih (by simpa using hl) hk hx
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hnf =>
        rcases hnf with h | ⟨h1, h2⟩
        · exact (hk h).elim
        · exact (hx.elim h1 h2).elim
      | _ => nofun

/-! ## The writes at the start -/

theorem initFirst_lt (hP : P.Ok) : ∀ q ∈ P.initFirst, q.1 < P.top := by
  intro q hq
  simp only [initFirst, List.mem_append, List.mem_map, List.mem_range, List.mem_flatMap] at hq
  rcases hq with ((⟨a, ha, rfl⟩ | ⟨a, ha, rfl⟩) | ⟨c, hc, a, ha, rfl⟩) | ⟨c, hc, a, ha, rfl⟩
  · simp only [top]
    omega
  · simp only [ctrPlace]
    have := chan_end (P := P) (c := a / 3) (by omega)
    have : a % 3 < 3 := Nat.mod_lt _ (by omega)
    rw [size_eq] at *
    omega
  · exact (sig_inside (o := .send) hc (by simp only [sigLen]; omega)).2
  · have := hP.cap c hc
    exact (sig_inside (o := .recv) hc (by simp only [sigLen]; omega)).2

/-- A write at the start is to a place below `top`, or it is the write to the slot or to the
flag. -/
theorem initList_split (hP : P.Ok) {i : Nat} {p : Nat × Int} (h : P.initList[i]? = some p) :
    p.1 < P.top ∨ (i = P.initFirst.length ∧ p = (P.slot, 0) ∨
      i = P.initFirst.length + 1 ∧ p = (P.flag, 0)) := by
  unfold initList at h
  by_cases hi : i < P.initFirst.length
  · rw [List.getElem?_append_left hi] at h
    exact Or.inl (initFirst_lt hP _ (List.mem_of_getElem? h))
  · rw [List.getElem?_append_right (by omega)] at h
    right
    cases he : P.errors
    · simp [he] at h
    · simp only [he, ↓reduceIte] at h
      rcases Nat.lt_trichotomy (i - P.initFirst.length) 1 with h1 | h1 | h1
      · rw [show i - P.initFirst.length = 0 by omega] at h
        simp only [List.getElem?_cons_zero, Option.some.injEq] at h
        exact Or.inl ⟨by omega, h.symm⟩
      · rw [h1] at h
        simp only [List.getElem?_cons_succ, List.getElem?_cons_zero, Option.some.injEq] at h
        exact Or.inr ⟨by omega, h.symm⟩
      · rw [List.getElem?_eq_none (by simp; omega)] at h
        cases h

/-! ## The executions of a program with failures -/

variable {X : Execution} (run : P.toTaskProg.toProg.Run X)
include run

/-- The accesses of the thread of `e` after the writes at the start, up to `e`, follow `FTrans`. -/
theorem fview (hP : P.Ok) (e : X.E) :
    P.FAccepts .free ((List.range (run.idx e + 1 - P.pre (X.thread e))).map
      fun i => run.lab (X.thread e) (P.pre (X.thread e) + i)) :=
  Runs.faccepts (trace_rest (run.trace e)).2 (.clean (fclean_body hP _))

theorem lab_facts (hP : P.Ok) {d : X.E} (hp : P.pre (X.thread d) ≤ run.idx d) :
    (X.kind d = .write → X.loc d = P.slot →
      run.isRmw d = true ∧ X.order d = .sc ∧ run.val d ≠ 0) ∧
    (X.kind d = .write → X.loc d = P.flag → X.order d = .rel) ∧
    (X.kind d = .read → X.loc d = P.flag → X.order d = .acq) :=
  (fview run hP d).labels (view_get run rfl (Nat.le_refl _) hp)

/-- The write at the start to the slot writes 0. -/
theorem prefix_slot (hP : P.Ok) {d : X.E} (hp : run.idx d < P.pre (X.thread d))
    (hl : X.loc d = P.slot) : X.thread d = 0 ∧ run.idx d = P.initFirst.length ∧ run.val d = 0 := by
  obtain ⟨ht, p, hi, -, -, hl', -⟩ := init_of run hP hp
  have hv := init_val run hP hp hi
  have := top_le_slot (P := P)
  rcases initList_split hP hi with h | ⟨h1, rfl⟩ | ⟨-, rfl⟩
  · omega
  · exact ⟨ht, h1, hv⟩
  · simp only [flag] at hl'
    omega

/-- The write at the start to the flag writes 0. -/
theorem prefix_flag (hP : P.Ok) {d : X.E} (hp : run.idx d < P.pre (X.thread d))
    (hl : X.loc d = P.flag) : run.val d = 0 := by
  obtain ⟨-, p, hi, -, -, hl', -⟩ := init_of run hP hp
  have hv := init_val run hP hp hi
  have := top_le_slot (P := P)
  simp only [flag] at hl
  rcases initList_split hP hi with h | ⟨-, rfl⟩ | ⟨-, rfl⟩
  · omega
  · simp only at hl'
    omega
  · exact hv

/-- The only write of 0 to the slot is the write at the start. -/
theorem zero_unique (hP : P.Ok) {s s' : X.E} (hs : X.IsWrite s) (hsl : X.loc s = P.slot)
    (hv : run.val s = 0) (hs' : X.IsWrite s') (hsl' : X.loc s' = P.slot) (hv' : run.val s' = 0) :
    s = s' := by
  have key : ∀ d : X.E, X.IsWrite d → X.loc d = P.slot → run.val d = 0 →
      X.thread d = 0 ∧ run.idx d = P.initFirst.length := by
    intro d hd hdl hdv
    by_cases hp : run.idx d < P.pre (X.thread d)
    · exact ⟨(prefix_slot run hP hp hdl).1, (prefix_slot run hP hp hdl).2.1⟩
    · exact absurd hdv ((lab_facts run hP (by omega)).1 hd hdl).2.2
  obtain ⟨h1, h2⟩ := key s hs hsl hv
  obtain ⟨h1', h2'⟩ := key s' hs' hsl' hv'
  exact run.idx_inj s s' (h1.trans h1'.symm) (h2.trans h2'.symm)

/-- A write of a value other than 0 to the slot is the write of a CAS that read 0. If it writes
`cancelled`, its thread read the cancel flag set two steps before it. -/
theorem slot_win (hP : P.Ok) {w : X.E} (hw : X.IsWrite w) (hwl : X.loc w = P.slot)
    (hv : run.val w ≠ 0) :
    ∃ r, X.rmw r w ∧ X.IsRead r ∧ X.loc r = P.slot ∧ run.val r = 0 ∧
      (run.val w = cancelled → ∃ f, X.thread f = X.thread w ∧ run.idx f + 2 = run.idx w ∧
        X.IsRead f ∧ X.loc f = P.flag ∧ X.order f = .acq ∧ run.val f ≠ 0) := by
  have hp : P.pre (X.thread w) ≤ run.idx w :=
    Nat.le_of_not_lt fun hp => hv (prefix_slot run hP hp hwl).2.2
  have hacc := fview run hP w
  have hget := view_get run (e := w) rfl (Nat.le_refl _) hp
  have hrm := ((hacc.labels hget).1 hw hwl).1
  rcases hacc.win_hist hget hw hwl with h | hist
  · exact h.elim
  obtain ⟨i', hi', hread, hcan⟩ := hist
  have h1 : P.pre (X.thread w) + i' < run.idx w := by omega
  obtain ⟨r, hrt, hri⟩ := run.idx_down w _ h1
  have hr := view_get run (e := w) hrt (by omega) (by omega)
  rw [show run.idx r - P.pre (X.thread w) = i' by omega, hread] at hr
  simp only [Option.some.injEq, Label.mk.injEq] at hr
  obtain ⟨hrk, -, hrl, hrv, -⟩ := hr
  have h2 : run.idx w = run.idx r + 1 := by omega
  refine ⟨r, (run.rmw_iff r w).2 ⟨hrt, h2, hrm⟩, hrk.symm, hrl.symm, hrv.symm, fun hc => ?_⟩
  obtain ⟨i'', hi'', u, hu, hf⟩ := hcan hc
  have h3 : P.pre (X.thread w) + i'' < run.idx w := by omega
  obtain ⟨f, hft, hfi⟩ := run.idx_down w _ h3
  have hf' := view_get run (e := w) hft (by omega) (by omega)
  rw [show run.idx f - P.pre (X.thread w) = i'' by omega, hf] at hf'
  simp only [Option.some.injEq, Label.mk.injEq] at hf'
  obtain ⟨hfk, hfo, hfl, hfv, -⟩ := hf'
  have h4 : run.idx f + 2 = run.idx w := by omega
  exact ⟨f, hft, h4, hfk.symm, hfl.symm, hfo.symm, by rw [← hfv]; exact hu⟩

/-- A write of the cancel flag that sets it is a release write. Right before it, its thread wrote
the slot. -/
theorem flag_win (hP : P.Ok) {w : X.E} (hw : X.IsWrite w) (hwl : X.loc w = P.flag)
    (hv : run.val w ≠ 0) :
    X.IsRelease w ∧ X.IsAtomic w ∧ ∃ s, X.thread s = X.thread w ∧ run.idx s + 1 = run.idx w ∧
      X.IsWrite s ∧ X.loc s = P.slot ∧ run.val s ≠ 0 := by
  have hp : P.pre (X.thread w) ≤ run.idx w :=
    Nat.le_of_not_lt fun hp => hv (prefix_flag run hP hp hwl)
  have hacc := fview run hP w
  have hget := view_get run (e := w) rfl (Nat.le_refl _) hp
  have ho : X.order w = .rel := (hacc.labels hget).2.1 hw hwl
  rcases hacc.flag_hist hget hw hwl with ⟨h, -⟩ | ⟨i', v, hi', hs⟩
  · cases h
  obtain ⟨s, hst, hsi⟩ := run.idx_down w (P.pre (X.thread w) + i') (by omega)
  have g := view_get run (e := w) hst (by omega) (by omega)
  rw [show run.idx s - P.pre (X.thread w) = i' by omega, hs] at g
  simp only [Option.some.injEq, Label.mk.injEq] at g
  obtain ⟨hk, -, hsl, -, -⟩ := g
  refine ⟨Or.inl ho, by unfold Execution.IsAtomic; rw [ho]; nofun, s, hst, by omega, hk.symm,
    hsl.symm, ?_⟩
  exact ((lab_facts run hP (d := s) (by rw [hst]; omega)).1 hk.symm hsl.symm).2.2

/-- A read of the cancel flag is an acquire read. -/
theorem flag_acq (hP : P.Ok) {f : X.E} (hf : X.IsRead f) (hfl : X.loc f = P.flag) :
    X.IsAcquire f ∧ X.IsAtomic f := by
  have hp : P.pre (X.thread f) ≤ run.idx f := by
    refine Nat.le_of_not_lt fun hp => ?_
    obtain ⟨-, -, -, hk, -⟩ := init_of run hP hp
    unfold Execution.IsRead at hf
    rw [hf] at hk
    cases hk
  have ho := (lab_facts run hP hp).2.2 hf hfl
  exact ⟨Or.inl ho, by unfold Execution.IsAtomic; rw [ho]; nofun⟩

/-- Each access to the slot or the flag is atomic, so these places never race. -/
theorem fail_atomic (hP : P.Ok) {e : X.E} (hk : X.kind e ≠ .fence)
    (hl : X.loc e = P.slot ∨ X.loc e = P.flag) : X.IsAtomic e := by
  by_cases hp : run.idx e < P.pre (X.thread e)
  · obtain ⟨-, -, -, -, ho, -⟩ := init_of run hP hp
    unfold Execution.IsAtomic
    rw [ho]
    nofun
  · exact (fview run hP e).atomic (view_get run rfl (Nat.le_refl _) (by omega)) hk hl

/-- The write at the start to the slot comes before each CAS that wrote it, in `co`. -/
theorem init_co (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder) {w : X.E}
    (hw : X.IsWrite w) (hwl : X.loc w = P.slot) (hv : run.val w ≠ 0) :
    ∃ s, X.IsWrite s ∧ X.loc s = P.slot ∧ run.val s = 0 ∧ X.co s w := by
  obtain ⟨r, hrw, hr, hrl, hrv, -⟩ := slot_win run hP hw hwl hv
  obtain ⟨s, hs⟩ := run.rf_source r hr
  obtain ⟨hsw, -, hsl⟩ := run.rf_write_read s r hs
  exact ⟨s, hsw, hsl.trans hrl, (run.rf_val s r hs).trans hrv,
    Execution.rmw_source_co wf hcons ho hs hrw (hrl.trans hwl.symm)⟩

/-- At most one CAS writes the failure slot. Each CAS that writes it read 0, which only the write
at the start wrote. Two CAS that read the same write cannot both write, by the atomicity of a
read-modify-write. -/
theorem one_failure_wins (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder)
    {w1 w2 : X.E} (hw1 : X.IsWrite w1) (hl1 : X.loc w1 = P.slot) (hv1 : run.val w1 ≠ 0)
    (hw2 : X.IsWrite w2) (hl2 : X.loc w2 = P.slot) (hv2 : run.val w2 ≠ 0) : w1 = w2 := by
  refine Classical.byContradiction fun hne => ?_
  obtain ⟨r1, hrw1, hr1, hrl1, hrv1, -⟩ := slot_win run hP hw1 hl1 hv1
  obtain ⟨r2, hrw2, hr2, hrl2, hrv2, -⟩ := slot_win run hP hw2 hl2 hv2
  obtain ⟨s1, hs1⟩ := run.rf_source r1 hr1
  obtain ⟨s2, hs2⟩ := run.rf_source r2 hr2
  obtain ⟨hs1w, -, hs1l⟩ := run.rf_write_read s1 r1 hs1
  obtain ⟨hs2w, -, hs2l⟩ := run.rf_write_read s2 r2 hs2
  have h12 := zero_unique run hP hs1w (hs1l.trans hrl1) ((run.rf_val _ _ hs1).trans hrv1) hs2w
    (hs2l.trans hrl2) ((run.rf_val _ _ hs2).trans hrv2)
  subst h12
  have hc1 := Execution.rmw_source_co wf hcons ho hs1 hrw1 (hrl1.trans hl1.symm)
  have hc2 := Execution.rmw_source_co wf hcons ho hs2 hrw2 (hrl2.trans hl2.symm)
  rcases ho.total w1 w2 hw1 hw2 (hl1.trans hl2.symm) hne with h | h
  · exact hcons.atomicity r2 w2 w1 hrw2 ⟨s1, hs2, hc1⟩ h
  · exact hcons.atomicity r1 w1 w2 hrw1 ⟨s1, hs1, hc2⟩ h

/-- A read of the slot that the winning CAS happens before gets the value of that CAS. The joins
of the block give this edge for the `result` after them. -/
theorem block_value_is_failure (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent)
    (ho : X.CoOrder) {w e : X.E} (hw : X.IsWrite w) (hwl : X.loc w = P.slot) (hv : run.val w ≠ 0)
    (he : X.IsRead e) (hel : X.loc e = P.slot) (hwe : X.hb w e) : run.val e = run.val w := by
  obtain ⟨s, hs⟩ := run.rf_source e he
  obtain ⟨hsw, -, hsl⟩ := run.rf_write_read s e hs
  by_cases h0 : run.val s = 0
  · obtain ⟨i0, hi0w, hi0l, hi0v, hco⟩ := init_co run hP wf hcons ho hw hwl hv
    have := zero_unique run hP hsw (hsl.trans hel) h0 hi0w hi0l hi0v
    subst this
    exact (hcons.coherence w e hwe
      (Or.inr (Relation.TransGen.single (Or.inr (Or.inr ⟨s, hs, hco⟩))))).elim
  · have := one_failure_wins run hP wf hcons ho hsw (hsl.trans hel) h0 hw hwl hv
    subst this
    exact (run.rf_val s e hs).symm

/-- A read of the slot that does not see 0 gets its value from the CAS of a failure: of a task, or
of the code of the block. -/
theorem block_value_from_fail (hP : P.Ok) {e : X.E} (he : X.IsRead e) (hel : X.loc e = P.slot)
    (hv : run.val e ≠ 0) :
    ∃ w r, X.rf w e ∧ X.rmw r w ∧ X.loc r = P.slot ∧ run.val r = 0 ∧ run.val w = run.val e := by
  obtain ⟨w, hrf⟩ := run.rf_source e he
  obtain ⟨hw, -, hwl⟩ := run.rf_write_read w e hrf
  have hwv := run.rf_val w e hrf
  obtain ⟨r, hrw, -, hrl, hrv, -⟩ := slot_win run hP hw (hwl.trans hel) (by rw [hwv]; exact hv)
  exact ⟨w, r, hrf, hrw, hrl, hrv, hwv⟩

/-- `cancelled` never reaches the slot. A task that was cancelled read the cancel flag set. The
CAS that set the slot happens before that read, by the release write and the acquire read of the
flag. So the CAS of the cancelled task reads the slot set, and it does not write. -/
theorem block_value_not_cancelled (hP : P.Ok) (wf : X.WellFormed) (hcons : X.Consistent)
    (ho : X.CoOrder) {w : X.E} (hw : X.IsWrite w) (hwl : X.loc w = P.slot) :
    run.val w ≠ cancelled := by
  intro hc
  have hv : run.val w ≠ 0 := by rw [hc]; simp [cancelled]
  obtain ⟨r, hrw, hr, hrl, hrv, hcan⟩ := slot_win run hP hw hwl hv
  obtain ⟨f, hft, hfi, hf, hfl, hfo, hfv⟩ := hcan hc
  obtain ⟨g, hrf⟩ := run.rf_source f hf
  obtain ⟨hg, -, hgl⟩ := run.rf_write_read g f hrf
  have hgv : run.val g ≠ 0 := by rw [run.rf_val _ _ hrf]; exact hfv
  obtain ⟨hrel, hat, s, hst, hsi, hsw, hsl, hsv⟩ := flag_win run hP hg (hgl.trans hfl) hgv
  obtain ⟨hfacq, hfat⟩ := flag_acq run hP hf hfl
  obtain ⟨hthr, hri, -⟩ := (run.rmw_iff r w).1 hrw
  have hhb : X.hb s r := Execution.hb_of_release (Or.inr ((run.po_iff s g).2 ⟨hst, by omega⟩))
    hg hrel hat hrf hfacq hfat (Or.inr ((run.po_iff f r).2 ⟨hft.trans hthr.symm, by omega⟩))
  obtain ⟨s0, hs0⟩ := run.rf_source r hr
  obtain ⟨hs0w, -, hs0l⟩ := run.rf_write_read s0 r hs0
  obtain ⟨i0, hi0w, hi0l, hi0v, hco⟩ := init_co run hP wf hcons ho hsw hsl hsv
  have := zero_unique run hP hs0w (hs0l.trans hrl) ((run.rf_val s0 r hs0).trans hrv) hi0w hi0l hi0v
  subst this
  exact hcons.coherence s r hhb (Or.inr (Relation.TransGen.single (Or.inr (Or.inr ⟨s0, hs0, hco⟩))))

/-! ## Each error once -/

/-- A task with no `result` never reads the slot with acquire. -/
theorem task_noResult (hP : P.Ok) (htr : ∀ t, ¬ (P.task t).Results) {e : X.E}
    (he : X.thread e ≠ 0) : P.NoResult (X.kind e) (X.order e) (X.loc e) := by
  obtain ⟨t, ht⟩ := Nat.exists_eq_succ_of_ne_zero he
  have hc : P.FClean False (P.toTaskProg.toProg.code (X.thread e)) := by
    have := fclean_task hP t (htr t)
    rw [ht]
    simpa [body] using this
  have h := (accessRule P.NoResult).trace (run.trace e) hc.noResult
  simp only [accessRule, run.lab_eq e] at h
  exact h

theorem rmw_eq {r r' w : X.E} (h : X.rmw r w) (h' : X.rmw r' w) : r = r' := by
  obtain ⟨ht, hi, -⟩ := (run.rmw_iff r w).1 h
  obtain ⟨ht', hi', -⟩ := (run.rmw_iff r' w).1 h'
  exact run.idx_inj r r' (ht.trans ht'.symm) (by omega)

/-- The block owns each error, and `join` gives none to its caller. The error of at most one
failure moves into the slot, by the CAS of that failure. A CAS that read the slot set does not
write it: its failure drops its error. Other than a CAS, only the code of the block reads the
slot. A value other than 0 that it reads is the error that moved. -/
theorem each_error_once (hP : P.Ok) (htr : ∀ t, ¬ (P.task t).Results) (wf : X.WellFormed)
    (hcons : X.Consistent) (ho : X.CoOrder) :
    (∀ {r1 w1 r2 w2 : X.E}, X.rmw r1 w1 → X.loc w1 = P.slot → run.val w1 ≠ 0 →
      X.rmw r2 w2 → X.loc w2 = P.slot → run.val w2 ≠ 0 → r1 = r2) ∧
    (∀ {r w : X.E}, X.rmw r w → X.loc w = P.slot → run.val w ≠ 0 → run.val r = 0) ∧
    (∀ {e : X.E}, X.IsRead e → X.loc e = P.slot → X.order e ≠ .sc →
      X.thread e = 0 ∧ (run.val e ≠ 0 → ∃ r w, X.rmw r w ∧ X.rf w e)) := by
  refine ⟨fun h1 hl1 hv1 h2 hl2 hv2 => ?_, fun h hl hv => ?_, fun he hel ho' => ⟨?_, fun hv => ?_⟩⟩
  · have := one_failure_wins run hP wf hcons ho (wf.rmw_atomic _ _ h1).2.1 hl1 hv1
      (wf.rmw_atomic _ _ h2).2.1 hl2 hv2
    subst this
    exact rmw_eq run h1 h2
  · obtain ⟨r', hr', -, -, hv', -⟩ := slot_win run hP (wf.rmw_atomic _ _ h).2.1 hl hv
    rw [rmw_eq run h hr']
    exact hv'
  · refine Classical.byContradiction fun ht => ho' ?_
    exact task_noResult run hP htr ht he hel
  · obtain ⟨w, r, hrf, hrw, -⟩ := block_value_from_fail run hP he hel hv
    exact ⟨r, w, hrw, hrf⟩

end ChanProg

end Vx
