import VxMemoryModel.Interleave

/-!
# Each execution with no reordering comes from a run of a CPU node

The converse of `Cpu.Reach.execs`. Take a finite execution of a program in which the accesses
happen one at a time. Then a run of the CPU node makes the same accesses in each thread
(`Cpu.complete`).

A run adds one event of the execution at each step. It adds an event when each event before it
in the execution is in the run. The read and the write of a read-modify-write go in together,
in one step. The value of a read is the value of the last write to its place in the run.

An execution can end between the read and the write of a read-modify-write, and a run cannot. So
the run can lack such a read at the end of its thread.

The code must not hold the write of a read-modify-write (`Stmt.NoRmw`). Only a `fetchAdd` or a
`cas` makes it.
-/

namespace Vx

/-! ## Facts about the code -/

/-- `s` holds no write of a read-modify-write. -/
def Stmt.NoRmw : Stmt → Prop
  | rmwStore .. => False
  | seq s t | ite _ s t => s.NoRmw ∧ t.NoRmw
  | loop _ s | tasksBlock s => s.NoRmw
  | _ => True

/-- The write of a read-modify-write can only come first. -/
def Stmt.Ok : Stmt → Prop
  | rmwStore .. => True
  | seq s t => s.Ok ∧ t.NoRmw
  | s => s.NoRmw

/-- The place of the write that comes next, when `s.Pending`. -/
def Stmt.pendLoc : Stmt → Nat
  | rmwStore x _ _ => x
  | seq s _ => s.pendLoc
  | _ => 0

theorem Stmt.NoRmw.ok {s : Stmt} (h : s.NoRmw) : s.Ok := by
  cases s with
  | rmwStore => exact h.elim
  | seq s t => exact ⟨Stmt.NoRmw.ok h.1, h.2⟩
  | _ => exact h

theorem Stmt.NoRmw.not_pending {s : Stmt} (h : s.NoRmw) : ¬ s.Pending := by
  induction s with
  | rmwStore => exact h.elim
  | seq _ _ ih _ => exact ih h.1
  | _ => exact id

theorem Step.ok {s regs a s' regs'} (h : Step s regs a s' regs') (hs : s.Ok) : s'.Ok := by
  induction h with
  | seq _ ih => exact ⟨ih hs.1, hs.2⟩
  | seq_skip => exact hs.2.ok
  | ite_true => exact hs.1.ok
  | ite_false => exact hs.2.ok
  | loop => exact Stmt.NoRmw.ok (s := .ite _ _ _) ⟨⟨hs, hs⟩, trivial⟩
  | fetchAdd | cas_eq => trivial
  | wait v | waitAt _ v => by_cases hv : v = 0 <;> simp [hv, Stmt.Ok, Stmt.NoRmw]
  | _ => trivial

/-- A step that makes no access does not stop before the write of a read-modify-write. -/
theorem Step.silent_not_pending {s regs s' regs'} (h : Step s regs none s' regs') (hs : s.Ok) :
    ¬ s'.Pending := by
  generalize ha : (none : Option Label) = a at h
  induction h with
  | seq _ ih => exact ih hs.1 ha
  | seq_skip => exact hs.2.not_pending
  | ite_true => exact hs.1.not_pending
  | ite_false => exact hs.2.not_pending
  | assign | assignAt | loop => exact nofun
  | _ => cases ha

/-- From a state before the write of a read-modify-write, the only step is that write. -/
theorem Step.of_pending {s regs a s' regs'} (h : Step s regs a s' regs') (hs : s.Pending) :
    ∃ l, a = some l ∧ l.kind = .write ∧ l.rmw = true ∧ l.loc = s.pendLoc ∧ ¬ s'.Pending := by
  induction h with
  | rmwStore => exact ⟨_, rfl, rfl, rfl, rfl, id⟩
  | seq _ ih =>
    obtain ⟨l, hl, hk, hr, hx, hp⟩ := ih hs
    exact ⟨l, hl, hk, hr, hx, hp⟩
  | seq_skip => exact hs.elim
  | _ => exact hs.elim

/-- A step to a state before the write of a read-modify-write is its read, of the same place. -/
theorem Step.to_pending {s regs a s' regs'} (h : Step s regs a s' regs') (hok : s.Ok)
    (hs : ¬ s.Pending) (hs' : s'.Pending) : ∃ l, a = some l ∧ l.kind = .read ∧ s'.pendLoc = l.loc := by
  induction h with
  | fetchAdd | cas_eq => exact ⟨_, rfl, rfl, rfl⟩
  | seq _ ih => exact ih hok.1 hs hs'
  | seq_skip => exact absurd hs' hok.2.not_pending
  | ite_true => exact absurd hs' hok.1.not_pending
  | ite_false => exact absurd hs' hok.2.not_pending
  | wait v | waitAt _ v => by_cases hv : v = 0 <;> simp [hv, Stmt.Pending] at hs'
  | _ => exact (nomatch hs')

/-! ## Runs of code -/

/-- `Silent s regs s' regs'`: steps that make no access take `s` to `s'`. -/
inductive Silent : Stmt → (Nat → Int) → Stmt → (Nat → Int) → Prop
  | refl {s regs} : Silent s regs s regs
  | step {s regs s₁ regs₁ s₂ regs₂} : Step s regs none s₁ regs₁ → Silent s₁ regs₁ s₂ regs₂ →
    Silent s regs s₂ regs₂

/-- A run that makes the access `l` first: steps with no access, then the step of `l`. -/
theorem Runs.cons_inv {s regs l L} (h : Runs s regs (l :: L)) :
    ∃ s₁ regs₁ s₂ regs₂, Silent s regs s₁ regs₁ ∧ Step s₁ regs₁ (some l) s₂ regs₂ ∧
      Runs s₂ regs₂ L := by
  generalize hL : l :: L = L' at h
  induction h with
  | stop => cases hL
  | silent hs _ ih =>
    obtain ⟨_, _, _, _, h1, h2, h3⟩ := ih hL
    exact ⟨_, _, _, _, .step hs h1, h2, h3⟩
  | visible hs hr =>
    cases hL
    exact ⟨_, _, _, _, .refl, hs, hr⟩

theorem Silent.ok {s regs s' regs'} (h : Silent s regs s' regs') (hs : s.Ok) : s'.Ok := by
  induction h with
  | refl => exact hs
  | step h₁ _ ih => exact ih (h₁.ok hs)

theorem Silent.not_pending {s regs s' regs'} (h : Silent s regs s' regs') (hs : s.Ok)
    (hp : ¬ s.Pending) : ¬ s'.Pending := by
  induction h with
  | refl => exact hp
  | step h₁ _ ih => exact ih (h₁.ok hs) (h₁.silent_not_pending hs)

/-- In a run of code that is `Ok`, the write of a read-modify-write comes right after a read of
the same place. -/
theorem Runs.rmw_facts {s regs L} (h : Runs s regs L) (hs : s.Ok) :
    (∀ l, L.head? = some l → l.rmw = true → s.Pending ∧ l.loc = s.pendLoc) ∧
    ∀ i (hi : i + 1 < L.length), L[i + 1].rmw = true →
      L[i].kind = .read ∧ L[i + 1].loc = L[i].loc := by
  induction h with
  | stop => exact ⟨nofun, fun _ hi => by simp at hi⟩
  | silent hstep _ ih =>
    obtain ⟨ih1, ih2⟩ := ih (hstep.ok hs)
    refine ⟨fun l hl hr => absurd (ih1 l hl hr).1 (hstep.silent_not_pending hs), ih2⟩
  | @visible s regs s' regs' a L hstep _ ih =>
    obtain ⟨ih1, ih2⟩ := ih (hstep.ok hs)
    refine ⟨fun l hl hr => ?_, fun i hi hr => ?_⟩
    · cases hl
      have hp := hstep.pending_of_rmw a rfl hr
      obtain ⟨l, hl, _, _, hx, _⟩ := hstep.of_pending hp
      cases hl
      exact ⟨hp, hx⟩
    · cases i with
      | zero =>
        have h0 : 0 < L.length := by simp at hi; omega
        obtain ⟨hp', hx⟩ := ih1 L[0] (by simp [List.head?_eq_getElem?, h0]) hr
        have hp : ¬ s.Pending := fun hp => (hstep.of_pending hp).choose_spec.2.2.2.2 hp'
        obtain ⟨l, hl, hk, hx'⟩ := hstep.to_pending hs hp hp'
        cases hl
        exact ⟨hk, hx.trans hx'⟩
      | succ i => exact ih2 i (by simp at hi; omega) hr


/-- A run of the CPU node can take the silent steps of one thread. -/
theorem Cpu.Reach.silent {c₀ evs c t s regs} (h : Cpu.Reach c₀ evs c)
    (hs : Silent (c.code t) (c.regs t) s regs) :
    ∃ c', Cpu.Reach c₀ evs c' ∧ c'.code t = s ∧ c'.regs t = regs ∧ c'.mem = c.mem ∧
      ∀ t', t' ≠ t → c'.code t' = c.code t' ∧ c'.regs t' = c.regs t' := by
  generalize hs₀ : c.code t = s₀ at hs
  generalize hr₀ : c.regs t = r₀ at hs
  induction hs generalizing c with
  | refl => exact ⟨c, h, hs₀, hr₀, rfl, fun _ _ => ⟨rfl, rfl⟩⟩
  | step h₁ _ ih =>
    subst hs₀ hr₀
    have h' := h.tail (Cpu.Step.silent h₁)
    simp only [List.map_nil, List.append_nil] at h'
    obtain ⟨c', hr, hc, hg, hm, ho⟩ := ih h' (by simp [Cpu.move]) (by simp [Cpu.move])
    refine ⟨c', hr, hc, hg, hm, fun t' ht => ?_⟩
    rw [(ho t' ht).1, (ho t' ht).2]
    exact ⟨setAt_other _ _ ht, setAt_other _ _ ht⟩

/-! ## Facts about the execution -/

section Exec

variable {P : Prog} {X : Execution} (R : P.Run X)

/-- The number of events of thread `t`. -/
theorem Prog.Run.exists_len (hfin : X.Finite) (t : Nat) :
    ∃ n, ∀ i, (∃ e, X.thread e = t ∧ R.idx e = i) ↔ i < n := by
  obtain ⟨l, hl⟩ := hfin
  have key : ∀ l : List X.E, ∃ n, (∀ e ∈ l, X.thread e = t → R.idx e < n) ∧
      (n = 0 ∨ ∃ e, X.thread e = t ∧ R.idx e + 1 = n) := by
    intro l
    induction l with
    | nil => exact ⟨0, nofun, Or.inl rfl⟩
    | cons e l ih =>
      obtain ⟨n, h1, h2⟩ := ih
      by_cases he : X.thread e = t ∧ n ≤ R.idx e
      · refine ⟨R.idx e + 1, fun x hx ht => ?_, Or.inr ⟨e, he.1, rfl⟩⟩
        rcases List.mem_cons.1 hx with rfl | hx
        · omega
        · have := h1 x hx ht; omega
      · refine ⟨n, fun x hx ht => ?_, h2⟩
        rcases List.mem_cons.1 hx with rfl | hx
        · omega
        · exact h1 x hx ht
  obtain ⟨n, h1, h2⟩ := key l
  refine ⟨n, fun i => ⟨fun ⟨e, ht, hi⟩ => hi ▸ h1 e (hl e) ht, fun hi => ?_⟩⟩
  rcases h2 with rfl | ⟨e, ht, he⟩
  · omega
  · rcases Nat.lt_or_eq_of_le (show i ≤ R.idx e by omega) with h | h
    · obtain ⟨e', ht', hi'⟩ := R.idx_down e i h
      exact ⟨e', ht'.trans ht, hi'⟩
    · exact ⟨e, ht, h.symm⟩

/-- The label of an event. -/
theorem Prog.Run.lab_of (e : X.E) :
    R.lab (X.thread e) (R.idx e) = ⟨X.kind e, X.order e, X.loc e, R.val e, R.isRmw e⟩ :=
  R.lab_eq e

/-- The whole run of a thread with `n` events. -/
theorem Prog.Run.runs_all {t n : Nat} (hn : ∀ i, (∃ e, X.thread e = t ∧ R.idx e = i) ↔ i < n) :
    Runs (P.code t) (fun _ => 0) ((List.range n).map (R.lab t)) := by
  cases n with
  | zero => exact .stop
  | succ n =>
    obtain ⟨e, ht, hi⟩ := (hn n).2 (Nat.lt_succ_self n)
    have := R.trace e
    rw [ht, hi] at this
    exact this

theorem Prog.Run.rmw_inv {e f : X.E} (h : X.rmw e f) :
    X.thread e = X.thread f ∧ R.idx f = R.idx e + 1 ∧ R.isRmw f = true :=
  (R.rmw_iff e f).1 h

include R in
/-- The two events of a read-modify-write: a read, and then a write of the same place. -/
theorem Prog.Run.rmw_loc (hcode : ∀ t, (P.code t).NoRmw) {e f : X.E} (h : X.rmw e f) :
    X.kind e = .read ∧ X.loc f = X.loc e := by
  obtain ⟨ht, hi, hr⟩ := R.rmw_inv h
  have hrun := R.trace f
  unfold Trace at hrun
  have hlt : R.idx e + 1 < ((List.range (R.idx f + 1)).map (R.lab (X.thread f))).length := by
    simp; omega
  have h2 := (hrun.rmw_facts (hcode _).ok).2 (R.idx e) hlt
  simp only [List.getElem_map, List.getElem_range] at h2
  rw [← hi, R.lab_of f, ← ht, R.lab_of e] at h2
  exact h2 hr

include R in
theorem Prog.Run.rmw_unique {e f f' : X.E} (h : X.rmw e f) (h' : X.rmw e f') : f = f' := by
  obtain ⟨ht, hi, _⟩ := R.rmw_inv h
  obtain ⟨ht', hi', _⟩ := R.rmw_inv h'
  exact R.idx_inj _ _ (ht.symm.trans ht') (hi.trans hi'.symm)

include R in
theorem Prog.Run.rmw_unique' {e e' f : X.E} (h : X.rmw e f) (h' : X.rmw e' f) : e = e' := by
  obtain ⟨ht, hi, _⟩ := R.rmw_inv h
  obtain ⟨ht', hi', _⟩ := R.rmw_inv h'
  exact R.idx_inj _ _ (ht.trans ht'.symm) (by omega)

end Exec


/-! ## The order in which a run adds the events -/

section Order

open Classical

variable {P : Prog} {X : Execution} (R : P.Run X) (hcode : ∀ t, (P.code t).NoRmw)
  (wf : X.WellFormed) (hsc : X.SeqConsistent) (hco : X.CoOrder)

include R hcode wf hsc hco in
/-- An edge out of the read of a read-modify-write, to another event, also leaves its write. -/
theorem Prog.Run.redirect {e f c : X.E} (h : X.rmw e f) (hec : X.ord e c) (hcf : c ≠ f) :
    X.ord f c := by
  obtain ⟨ht, hi, _⟩ := R.rmw_inv h
  obtain ⟨hk, hloc⟩ := R.rmw_loc hcode h
  obtain ⟨_, hfw, _, _⟩ := wf.rmw_atomic e f h
  have hnw : ¬ X.IsWrite e := by unfold Execution.IsWrite; rw [hk]; nofun
  rcases hec with hpo | hrf | hco' | ⟨w, hwe, hwc⟩
  · rw [R.po_iff] at hpo
    refine Or.inl ((R.po_iff f c).2 ⟨ht.symm.trans hpo.1, ?_⟩)
    have : R.idx c ≠ R.idx f := fun h' => hcf (R.idx_inj _ _ (hpo.1.symm.trans ht) h')
    omega
  · exact absurd (wf.rf_write_read _ _ hrf).1 hnw
  · exact absurd (wf.co_writes _ _ hco').1 hnw
  · obtain ⟨_, hcw, hlc⟩ := wf.co_writes _ _ hwc
    have hlwe := (wf.rf_write_read _ _ hwe).2.2
    rcases hco.total f c hfw hcw (by rw [hloc, ← hlwe, hlc]) (Ne.symm hcf) with h1 | h1
    · exact Or.inr (Or.inr (Or.inl h1))
    · exact absurd h1 (hsc.atomicity e f c h ⟨w, hwe, hwc⟩)

/-- `y` is in the group of `x`: `x` itself, or the write of the read-modify-write that `x` reads
for. -/
def InG (X : Execution) (x y : X.E) : Prop := y = x ∨ X.rmw x y

/-- `x` is not the write of a read-modify-write. -/
def Head (X : Execution) (x : X.E) : Prop := ∀ y, ¬ X.rmw y x

/-- The last event of the group of `x`. -/
noncomputable def last (X : Execution) (x : X.E) : X.E :=
  if h : ∃ f, X.rmw x f then h.choose else x

theorem inG_last (x : X.E) : InG X x (last X x) := by
  unfold last
  split
  · exact Or.inr (Exists.choose_spec ‹∃ f, X.rmw x f›)
  · exact Or.inl rfl

include R in
theorem last_of_rmw {x f : X.E} (h : X.rmw x f) : last X x = f := by
  have he : ∃ f, X.rmw x f := ⟨f, h⟩
  simp only [last, he, dite_true]
  exact R.rmw_unique he.choose_spec h

include R in
theorem rmw_ord {e f : X.E} (h : X.rmw e f) : X.ord e f := by
  obtain ⟨ht, hi, _⟩ := R.rmw_inv h
  exact Or.inl ((R.po_iff e f).2 ⟨ht, by omega⟩)

include R in
theorem to_last {x y : X.E} (h : InG X x y) : y = last X x ∨ X.ord y (last X x) := by
  rcases h with rfl | h
  · by_cases he : ∃ f, X.rmw y f
    · obtain ⟨f, hf⟩ := he
      rw [last_of_rmw R hf]
      exact Or.inr (rmw_ord R hf)
    · left
      simp [last, he]
  · exact Or.inl (last_of_rmw R h).symm

include R hcode wf hsc hco in
theorem from_last {x y c : X.E} (h : InG X x y) (hyc : X.ord y c) (hc : ¬ InG X x c) :
    X.ord (last X x) c := by
  rcases h with rfl | h
  · by_cases he : ∃ f, X.rmw y f
    · obtain ⟨f, hf⟩ := he
      rw [last_of_rmw R hf]
      exact R.redirect hcode wf hsc hco hf hyc fun e => hc (Or.inr (e ▸ hf))
    · simpa [last, he] using hyc
  · rwa [last_of_rmw R h]

include R in
theorem head_eq {a b z : X.E} (ha : Head X a) (hb : Head X b) (hza : InG X a z) (hzb : InG X b z) :
    a = b := by
  rcases hza with rfl | hza <;> rcases hzb with rfl | hzb
  · rfl
  · exact absurd hzb (ha b)
  · exact absurd hza (hb a)
  · exact R.rmw_unique' hza hzb

/-- The group of `b` must come before the group of `a`. -/
def Before (X : Execution) (b a : X.E) : Prop :=
  Head X b ∧ Head X a ∧ b ≠ a ∧ ∃ b' a', InG X b b' ∧ InG X a a' ∧ X.ord b' a'

include R hcode wf hsc hco in
theorem before_path {b a : X.E} (h : Relation.TransGen (Before X) b a) :
    ∃ a', InG X a a' ∧ Relation.TransGen X.ord (last X b) a' := by
  induction h with
  | single h =>
    obtain ⟨hb, ha, hne, b', a', hb', ha', ho⟩ := h
    have hna : ¬ InG X b a' := fun h => hne (head_eq R hb ha h ha')
    exact ⟨a', ha', .single (from_last R hcode wf hsc hco hb' ho hna)⟩
  | @tail a c _ h ih =>
    obtain ⟨a', ha', hp⟩ := ih
    obtain ⟨ha, hc, hne, a'', c', ha'', hc', ho⟩ := h
    have hnc : ¬ InG X a c' := fun h => hne (head_eq R ha hc h hc')
    have h1 := from_last R hcode wf hsc hco ha'' ho hnc
    refine ⟨c', hc', ?_⟩
    rcases to_last R ha' with e | e
    · rw [← e] at h1
      exact hp.tail h1
    · exact (hp.tail e).tail h1

include R hcode wf hsc hco in
theorem before_acyclic (a : X.E) : ¬ Relation.TransGen (Before X) a a := by
  intro h
  obtain ⟨a', ha', hp⟩ := before_path R hcode wf hsc hco h
  rcases to_last R ha' with e | e
  · rw [← e] at hp
    exact hsc.acyclic _ hp
  · exact hsc.acyclic _ (hp.tail e)

include R hcode wf hsc hco in
/-- Take a set `D` of events that holds both events of each read-modify-write or neither, and an
event outside it. Then the group of some event `m` outside `D` can go next: each event before an
event of the group is in `D` or in the group. -/
theorem exists_next (hfin : X.Finite) {D : X.E → Prop} (hD : ∀ e f, X.rmw e f → (D e ↔ D f))
    {e : X.E} (he : ¬ D e) :
    ∃ m, Head X m ∧ ¬ D m ∧ ∀ x a, InG X m x → X.ord a x → D a ∨ InG X m a := by
  obtain ⟨l, hl⟩ := hfin
  -- The head of the group of an event.
  have head_of : ∀ x : X.E, ∃ h, Head X h ∧ InG X h x := by
    intro x
    by_cases hx : ∃ y, X.rmw y x
    · obtain ⟨y, hy⟩ := hx
      refine ⟨y, fun z hz => ?_, Or.inr hy⟩
      have h1 := (R.rmw_loc hcode hy).1
      have h2 := (wf.rmw_atomic z y hz).2.1
      unfold Execution.IsWrite at h2
      rw [h1] at h2
      cases h2
    · exact ⟨x, fun y hy => hx ⟨y, hy⟩, Or.inl rfl⟩
  have notD : ∀ h x, InG X h x → ¬ D x → ¬ D h := by
    intro h x hx hn
    rcases hx with rfl | hx
    · exact hn
    · exact fun hh => hn ((hD _ _ hx).1 hh)
  let Rel : X.E → X.E → Prop := fun x y => Before X y x ∧ ¬ D x ∧ ¬ D y
  have hrev : ∀ x y, Relation.TransGen Rel x y → Relation.TransGen (Before X) y x := by
    intro x y h
    induction h with
    | single h => exact .single h.1
    | tail _ h ih => exact Relation.TransGen.head h.1 ih
  obtain ⟨h₀, hh₀, hx₀⟩ := head_of e
  obtain ⟨m, hm, hmax⟩ := exists_maximal hl
    (fun a h => before_acyclic R hcode wf hsc hco a (hrev a a h)) h₀
  have hm' : Head X m ∧ ¬ D m := by
    rcases hm with rfl | hm
    · exact ⟨hh₀, notD _ _ hx₀ he⟩
    · have : ∃ y, Rel y m := by
        cases hm with
        | single h => exact ⟨_, h⟩
        | tail _ h => exact ⟨_, h⟩
      obtain ⟨_, h⟩ := this
      exact ⟨h.1.1, h.2.2⟩
  refine ⟨m, hm'.1, hm'.2, fun x a hx ha => ?_⟩
  by_cases hin : InG X m a
  · exact Or.inr hin
  · refine Or.inl (Classical.byContradiction fun hDa => ?_)
    obtain ⟨b, hb, hba⟩ := head_of a
    have hbm : b ≠ m := fun e => hin (e ▸ hba)
    exact hmax b ⟨⟨hb, hm'.1, hbm, a, x, hba, hx, ha⟩, hm'.2, notD _ _ hba hDa⟩

end Order


/-! ## Building the run -/

section Build

open Classical

variable {P : Prog} {X : Execution} (R : P.Run X) (n : Nat → Nat)

/-- An event with its thread and its label, as a run lists it. -/
def ev (e : X.E) : Nat × Label := (X.thread e, R.lab (X.thread e) (R.idx e))

/-- A run under construction. The events of thread `t` with a position below `k t` are in it.
When `sk t` is set, the last of them is a read that the run left out. -/
structure Built (X : Execution) where
  evs : List (Nat × Label)
  c : Cpu
  order : List X.E
  k : Nat → Nat
  sk : Nat → Bool

/-- The event `e` is in the run under construction. -/
def Built.Done (b : Built X) (e : X.E) : Prop := R.idx e < b.k (X.thread e)

structure Good (b : Built X) : Prop where
  reach : Cpu.Reach (Cpu.init P) b.evs b.c
  evs_eq : b.evs = b.order.map (ev R)
  mem_order : ∀ e, e ∈ b.order ↔
    b.Done R e ∧ ¬ (b.sk (X.thread e) = true ∧ R.idx e + 1 = b.k (X.thread e))
  ord_pos : ∀ i j (hi : i < b.order.length) (hj : j < b.order.length),
    X.ord b.order[i] b.order[j] → i < j
  down : ∀ a e, X.ord a e → b.Done R e → b.Done R a
  pair : ∀ e f, X.rmw e f → (b.Done R e ↔ b.Done R f)
  k_le : ∀ t, b.k t ≤ n t
  sk_end : ∀ t, b.sk t = true →
    b.k t = n t ∧ ∃ e, X.thread e = t ∧ R.idx e + 1 = b.k t ∧ X.kind e = .read
  acc : ∀ t, accessesOf b.evs t = (List.range (b.k t - if b.sk t then 1 else 0)).map (R.lab t)
  runs : ∀ t, b.sk t = false →
    Runs (b.c.code t) (b.c.regs t) ((List.range' (b.k t) (n t - b.k t)).map (R.lab t))
  ok : ∀ t, b.sk t = false → (b.c.code t).Ok ∧ ¬ (b.c.code t).Pending

variable {R n}

theorem Good.init (hcode : ∀ t, (P.code t).NoRmw)
    (hn : ∀ t i, (∃ e, X.thread e = t ∧ R.idx e = i) ↔ i < n t) :
    Good R n ⟨[], Cpu.init P, [], fun _ => 0, fun _ => false⟩ where
  reach := .refl
  evs_eq := rfl
  mem_order e := by simp [Built.Done]
  ord_pos _ _ hi := absurd hi (Nat.not_lt_zero _)
  down _ e _ he := absurd he (Nat.not_lt_zero _)
  pair _ _ _ := by simp [Built.Done]
  k_le _ := Nat.zero_le _
  sk_end _ h := by cases h
  acc _ := by simp [accessesOf]
  runs t _ := by
    simpa [List.range_eq_range', Cpu.init] using R.runs_all (hn t)
  ok t _ := ⟨(hcode t).ok, (hcode t).not_pending⟩

/-- A read whose events before it are all in the run gets, in the run's memory, the value that it
reads in the execution. -/
theorem Good.fits (hco : X.CoOrder) {b : Built X} (hb : Good R n b)
    {e : X.E} (he : ¬ b.Done R e) (hpre : ∀ a, X.ord a e → b.Done R a) :
    (R.lab (X.thread e) (R.idx e)).Fits (memOf b.evs) := by
  intro hk
  rw [R.lab_of] at hk ⊢
  have hr : X.IsRead e := hk
  obtain ⟨w, hw⟩ := R.rf_source e hr
  obtain ⟨hww, _, hlw⟩ := R.rf_write_read w e hw
  have hdw : b.Done R w := hpre w (Execution.ord_rf hw)
  -- `w` is in the run: the run leaves out only reads.
  have hwo : w ∈ b.order := by
    refine (hb.mem_order w).2 ⟨hdw, fun ⟨hs, hi⟩ => ?_⟩
    obtain ⟨_, e', ht', hi', hk'⟩ := hb.sk_end _ hs
    have : e' = w := R.idx_inj _ _ ht' (by omega)
    subst this
    unfold Execution.IsWrite at hww
    rw [hk'] at hww
    cases hww
  obtain ⟨p, hp, hpw⟩ := List.mem_iff_getElem.1 hwo
  have hlen : b.evs.length = b.order.length := by rw [hb.evs_eq, List.length_map]
  have hev : ∀ j (hj : j < b.evs.length), b.evs[j] = ev R (b.order[j]'(hlen ▸ hj)) := by
    intro j hj
    simp only [hb.evs_eq, List.getElem_map]
  have hwt : ∀ j (hj : j < b.evs.length), IsWriteTo b.evs[j] (X.loc e) ↔
      X.IsWrite (b.order[j]'(hlen ▸ hj)) ∧ X.loc (b.order[j]'(hlen ▸ hj)) = X.loc e := by
    intro j hj
    rw [hev, IsWriteTo, ev, R.lab_of]
    rfl
  rcases memOf_spec (X.loc e) b.evs with ⟨_, hnone⟩ | ⟨j, hj, hwj, hv, hl⟩
  · exact absurd ((hwt p (hlen ▸ hp)).2 ⟨hpw ▸ hww, hpw ▸ hlw⟩) (hnone p (hlen ▸ hp))
  · obtain ⟨hwj1, hwj2⟩ := (hwt j hj).1 hwj
    have hj' : j < b.order.length := hlen ▸ hj
    have heq : b.order[j] = w := by
      refine Classical.byContradiction fun hne => ?_
      rcases hco.total _ _ hww hwj1 (hlw.trans hwj2.symm) (Ne.symm hne) with h1 | h1
      · -- `e` reads from `w`, and the write `order[j]` comes after `w`: `e` comes before it.
        have hdj : b.Done R b.order[j] := ((hb.mem_order _).1 (List.getElem_mem hj')).1
        exact he (hb.down _ _ (Execution.ord_fr ⟨w, hw, h1⟩) hdj)
      · have hjp := hb.ord_pos j p hj' hp (by rw [hpw]; exact Execution.ord_co h1)
        exact hl p (hlen ▸ hp) hjp ((hwt p (hlen ▸ hp)).2 ⟨hpw ▸ hww, hpw ▸ hlw⟩)
    rw [hv, hev j hj]
    simp only [ev, heq, R.lab_of]
    exact congrArg some (R.rf_val w e hw)

end Build


section Extend

open Classical

variable {P : Prog} {X : Execution} {R : P.Run X} {n : Nat → Nat}

/-- Add the events `G` of thread `t`, which come next in that thread. `more` is what the run adds:
`G`, or nothing when it leaves out the last read of the thread (`skip`). -/
theorem Good.extend {b : Built X} (hb : Good R n b) {t : Nat} {G more : List X.E} {c' : Cpu}
    {skip : Bool} (hmore : (skip = false ∧ more = G) ∨ (skip = true ∧ more = []))
    (hsk : b.sk t = false)
    (hG : ∀ i (h : i < G.length), X.thread G[i] = t ∧ R.idx G[i] = b.k t + i)
    (hn : b.k t + G.length ≤ n t)
    (hpred : ∀ a i (h : i < G.length), X.ord a G[i] →
      b.Done R a ∨ ∃ j, ∃ hj : j < i, a = G[j])
    (hpair : ∀ e f, X.rmw e f → (e ∈ G ↔ f ∈ G))
    (hreach : Cpu.Reach (Cpu.init P) (b.evs ++ more.map (ev R)) c')
    (hruns : skip = false → Runs (c'.code t) (c'.regs t)
      ((List.range' (b.k t + G.length) (n t - (b.k t + G.length))).map (R.lab t)))
    (hok : skip = false → (c'.code t).Ok ∧ ¬ (c'.code t).Pending)
    (hother : ∀ t', t' ≠ t → c'.code t' = b.c.code t' ∧ c'.regs t' = b.c.regs t')
    (hskip : skip = true → G.length = 1 ∧ b.k t + 1 = n t ∧ ∃ h : 0 < G.length, X.kind G[0] = .read) :
    Good R n ⟨b.evs ++ more.map (ev R), c', b.order ++ more, setAt b.k t (b.k t + G.length),
      setAt b.sk t skip⟩ := by
  have memG : ∀ x, x ∈ G ↔ X.thread x = t ∧ b.k t ≤ R.idx x ∧ R.idx x < b.k t + G.length := by
    intro x
    constructor
    · intro hx
      obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hx
      obtain ⟨h1, h2⟩ := hG i hi
      exact ⟨h1, by omega, by omega⟩
    · intro ⟨h1, h2, h3⟩
      have hi : R.idx x - b.k t < G.length := by omega
      have := hG _ hi
      rw [R.idx_inj _ _ (h1.trans this.1.symm) (by omega)]
      exact List.getElem_mem hi
  have kt : ∀ x, setAt b.k t (b.k t + G.length) (X.thread x) =
      if X.thread x = t then b.k t + G.length else b.k (X.thread x) := fun x => rfl
  have doneNew : ∀ x, R.idx x < setAt b.k t (b.k t + G.length) (X.thread x) ↔
      b.Done R x ∨ x ∈ G := by
    intro x
    rw [kt, memG, Built.Done]
    by_cases ht : X.thread x = t
    · simp only [ht, ite_true, true_and]
      omega
    · simp [ht]
  have hG_ne : ∀ x, x ∈ G → ¬ b.Done R x := by
    intro x hx hd
    have := (memG x).1 hx
    unfold Built.Done at hd
    rw [this.1] at hd
    omega
  refine ⟨hreach, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- evs_eq
    show b.evs ++ more.map (ev R) = (b.order ++ more).map (ev R)
    rw [hb.evs_eq, List.map_append]
  · -- mem_order
    intro x
    show x ∈ b.order ++ more ↔ R.idx x < setAt b.k t (b.k t + G.length) (X.thread x) ∧
      ¬ (setAt b.sk t skip (X.thread x) = true ∧
        R.idx x + 1 = setAt b.k t (b.k t + G.length) (X.thread x))
    rw [List.mem_append, doneNew, hb.mem_order]
    by_cases ht : X.thread x = t
    · have hs : setAt b.sk t skip (X.thread x) = skip := by rw [ht]; simp
      have hsk' : b.sk (X.thread x) = false := ht ▸ hsk
      rw [hs, hsk', kt]
      simp only [ht, ite_true]
      rcases hmore with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · simp
      · obtain ⟨h1, _, _⟩ := hskip rfl
        rw [memG, h1]
        simp only [List.not_mem_nil, or_false, Bool.false_eq_true, false_and, not_false_eq_true,
          and_true, true_and, ht]
        unfold Built.Done
        rw [ht]
        omega
    · have hxG : x ∉ G := fun h => ht ((memG x).1 h).1
      have hxm : x ∉ more := by rcases hmore with ⟨_, rfl⟩ | ⟨_, rfl⟩ <;> simp [hxG]
      rw [kt]
      simp [ht, setAt, hxG, hxm]
  · -- ord_pos
    intro i j hi hj ho
    simp only [List.length_append] at hi hj
    have hmG : ∀ x, x ∈ more → x ∈ G := by
      rcases hmore with ⟨_, rfl⟩ | ⟨_, rfl⟩ <;> simp
    rcases Nat.lt_or_ge i b.order.length with hi' | hi' <;>
      rcases Nat.lt_or_ge j b.order.length with hj' | hj'
    · rw [List.getElem_append_left hi', List.getElem_append_left hj'] at ho
      exact hb.ord_pos i j hi' hj' ho
    · omega
    · exfalso
      rw [List.getElem_append_right hi', List.getElem_append_left hj'] at ho
      have hd := ((hb.mem_order _).1 (List.getElem_mem hj')).1
      exact hG_ne _ (hmG _ (List.getElem_mem _)) (hb.down _ _ ho hd)
    · rw [List.getElem_append_right hi', List.getElem_append_right hj'] at ho
      rcases hmore with ⟨_, rfl⟩ | ⟨_, rfl⟩
      · rcases hpred _ _ _ ho with hd | ⟨j', hj'', he⟩
        · exact absurd hd (hG_ne _ (List.getElem_mem _))
        · have h1 := (hG (i - b.order.length) (by omega)).2
          have h2 := (hG j' (by omega)).2
          rw [he] at h1
          omega
      · simp at hi; omega
  · -- down
    intro a e ho hd
    show R.idx a < setAt b.k t (b.k t + G.length) (X.thread a)
    have hd' : R.idx e < setAt b.k t (b.k t + G.length) (X.thread e) := hd
    rw [doneNew] at hd' ⊢
    rcases hd' with hd' | he
    · exact Or.inl (hb.down a e ho hd')
    · obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 he
      rcases hpred a i hi ho with h | ⟨j, hj, rfl⟩
      · exact Or.inl h
      · exact Or.inr (List.getElem_mem _)
  · -- pair
    intro e f h
    show R.idx e < setAt b.k t (b.k t + G.length) (X.thread e) ↔
      R.idx f < setAt b.k t (b.k t + G.length) (X.thread f)
    rw [doneNew, doneNew, hb.pair e f h, hpair e f h]
  · -- k_le
    intro t'
    by_cases ht : t' = t
    · subst ht; simpa using hn
    · simpa [setAt, ht] using hb.k_le t'
  · -- sk_end
    intro t' hs
    by_cases ht : t' = t
    · subst ht
      simp only [setAt_same] at hs ⊢
      obtain ⟨h1, h2, h0, h3⟩ := hskip hs
      refine ⟨by omega, G[0], (hG 0 h0).1, by rw [(hG 0 h0).2, h1], h3⟩
    · simp only [setAt, ht, ite_false] at hs ⊢
      exact hb.sk_end t' hs
  · -- acc
    intro t'
    have hmap : more.map (ev R) = (more.map fun x => R.lab t (R.idx x)).map (t, ·) := by
      rw [List.map_map]
      refine List.map_congr_left fun x hx => ?_
      have hx' : x ∈ G := by rcases hmore with ⟨_, rfl⟩ | ⟨_, rfl⟩ <;> simp_all
      simp [ev, ((memG x).1 hx').1]
    rw [hmap, thr_append]
    by_cases ht : t' = t
    · subst ht
      rw [hb.acc t', hsk]
      simp only [setAt_same, ite_true, Bool.false_eq_true, ite_false, Nat.sub_zero]
      rcases hmore with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
      · have hG' : more.map (fun x => R.lab t' (R.idx x)) =
            (List.range' (b.k t') more.length).map (R.lab t') := by
          apply List.ext_getElem (by simp)
          intro i h1 h2
          simp [(hG i (by simpa using h1)).2]
        simp only [Bool.false_eq_true, ite_false, Nat.sub_zero, hG', List.range_eq_range',
          ← List.map_append]
        congr 1
        have := List.range'_append (s := 0) (m := b.k t') (n := more.length) (step := 1)
        simpa using this
      · obtain ⟨h1, _⟩ := hskip rfl
        simp [h1]
    · simp [ht, setAt, hb.acc t']
  · -- runs
    intro t' hs
    by_cases ht : t' = t
    · subst ht
      simp only [setAt_same] at hs ⊢
      exact hruns hs
    · simp only [setAt, ht, ite_false] at hs ⊢
      rw [(hother t' ht).1, (hother t' ht).2]
      exact hb.runs t' hs
  · -- ok
    intro t' hs
    by_cases ht : t' = t
    · subst ht
      simp only [setAt_same] at hs
      exact hok hs
    · simp only [setAt, ht, ite_false] at hs
      rw [(hother t' ht).1]
      exact hb.ok t' hs

end Extend


section Main

open Classical

variable {P : Prog} {X : Execution} {R : P.Run X} {n : Nat → Nat}

theorem done_mono {b : Built X} {t : Nat} {len : Nat} {sk : Nat → Bool} {evs c order} (x : X.E)
    (h : b.Done R x) : Built.Done R ⟨evs, c, order, setAt b.k t (b.k t + len), sk⟩ x := by
  unfold Built.Done at h ⊢
  by_cases ht : X.thread x = t
  · rw [ht] at h ⊢; simp; omega
  · simpa [setAt, ht] using h

/-- One more step of the construction: the run takes in at least one more event. -/
theorem Good.step (hcode : ∀ t, (P.code t).NoRmw) (wf : X.WellFormed) (hsc : X.SeqConsistent)
    (hco : X.CoOrder) (hfin : X.Finite)
    (hn : ∀ t i, (∃ e, X.thread e = t ∧ R.idx e = i) ↔ i < n t) {b : Built X} (hb : Good R n b)
    {e : X.E} (he : ¬ b.Done R e) :
    ∃ b', Good R n b' ∧ (∀ x, b.Done R x → b'.Done R x) ∧ ∃ x, ¬ b.Done R x ∧ b'.Done R x := by
  obtain ⟨m, hmh, hmd, hpre⟩ := exists_next R hcode wf hsc hco hfin hb.pair he
  have hpm : ∀ a, X.ord a m → b.Done R a := by
    intro a ha
    rcases hpre m a (Or.inl rfl) ha with h | h
    · exact h
    · exfalso
      rcases h with rfl | h
      · exact hsc.acyclic _ (.single ha)
      · exact hsc.acyclic _ (Relation.TransGen.head ha (.single (rmw_ord R h)))
  generalize ht : X.thread m = t
  have hidx : R.idx m = b.k t := by
    have h1 : b.k t ≤ R.idx m := by unfold Built.Done at hmd; rw [ht] at hmd; omega
    refine Nat.le_antisymm (Nat.le_of_not_lt fun h => ?_) h1
    obtain ⟨p, hpt, hpi⟩ := R.idx_down m (b.k t) h
    have hd := hpm p (Or.inl ((R.po_iff p m).2 ⟨hpt, by omega⟩))
    unfold Built.Done at hd
    rw [hpt, ht] at hd
    omega
  have hmn : b.k t < n t := hidx ▸ (hn t (R.idx m)).1 ⟨m, ht, rfl⟩
  have hsk : b.sk t = false := by
    cases h : b.sk t
    · rfl
    · have := (hb.sk_end t h).1; omega
  obtain ⟨d, hd⟩ : ∃ d, n t - b.k t = d + 1 := ⟨n t - b.k t - 1, by omega⟩
  have hruns := hb.runs t hsk
  rw [hd, List.range'_succ, List.map_cons] at hruns
  obtain ⟨s₁, r₁, s₂, r₂, hsil, hst, hrest⟩ := hruns.cons_inv
  obtain ⟨c₁, hreach₁, hc₁, hr₁, hm₁, ho₁⟩ := hb.reach.silent hsil
  have hok₁ : s₁.Ok := hsil.ok (hb.ok t hsk).1
  have hnp₁ : ¬ s₁.Pending := hsil.not_pending (hb.ok t hsk).1 (hb.ok t hsk).2
  have hlab : R.lab t (b.k t) = R.lab (X.thread m) (R.idx m) := by rw [ht, hidx]
  have hfits : (R.lab t (b.k t)).Fits c₁.mem := by
    rw [hm₁, hb.reach.inv.mem, hlab]
    exact hb.fits hco hmd hpm
  have hevm : ev R m = (t, R.lab t (b.k t)) := by simp [ev, ht, hidx]
  rw [← hc₁, ← hr₁] at hst
  have hmono : ∀ {len : Nat} {sk : Nat → Bool} {evs c order} (x : X.E), b.Done R x →
      Built.Done R ⟨evs, c, order, setAt b.k t (b.k t + len), sk⟩ x := done_mono
  have hnew : ∀ {len : Nat} {sk : Nat → Bool} {evs c order}, 0 < len →
      Built.Done R ⟨evs, c, order, setAt b.k t (b.k t + len), sk⟩ m := by
    intro len _ _ _ _ hl
    show R.idx m < setAt b.k t (b.k t + len) (X.thread m)
    rw [ht, setAt_same]
    omega
  by_cases hp₂ : s₂.Pending
  · obtain ⟨l, hl, hkr, _⟩ := hst.to_pending (hc₁ ▸ hok₁) (hc₁ ▸ hnp₁) hp₂
    cases hl
    have hread : X.kind m = .read := by rw [hlab, R.lab_of] at hkr; exact hkr
    cases d with
    | zero =>
      -- The read is the last access of its thread: the run leaves it out.
      have hlone : ∀ f, ¬ X.rmw m f := by
        intro f hf
        obtain ⟨hft, hfi, _⟩ := R.rmw_inv hf
        have := (hn t (R.idx f)).1 ⟨f, hft ▸ ht, rfl⟩
        omega
      refine ⟨_, hb.extend (G := [m]) (more := []) (skip := true) (c' := b.c) (Or.inr ⟨rfl, rfl⟩)
        hsk (fun i h => by simp at h; subst h; exact ⟨ht, by simp [hidx]⟩) (by simp; omega)
        (fun a i h ho => by simp at h; subst h; exact Or.inl (hpm a ho))
        (fun e f h => by
          simp only [List.mem_singleton]
          exact ⟨fun h' => absurd (h' ▸ h) (hlone f), fun h' => absurd (h' ▸ h) (hmh e)⟩)
        (by simpa using hb.reach) nofun nofun (fun _ _ => ⟨rfl, rfl⟩)
        (fun _ => ⟨rfl, by omega, by simp, hread⟩), fun x => hmono x, m, hmd, hnew (by simp)⟩
    | succ d =>
      -- The write of the read-modify-write comes next.
      rw [List.range'_succ, List.map_cons] at hrest
      obtain ⟨s₃, r₃, s₄, r₄, hsil₂, hst₂, hrest₂⟩ := hrest.cons_inv
      cases hsil₂ with
      | step h _ => obtain ⟨_, h', _⟩ := h.of_pending hp₂; cases h'
      | refl =>
      obtain ⟨l₂, hl₂, hk₂, hr₂, _, hnp₄⟩ := hst₂.of_pending hp₂
      cases hl₂
      obtain ⟨f, hft, hfi⟩ := (hn t (b.k t + 1)).2 (by omega)
      have hlabf : R.lab t (b.k t + 1) = R.lab (X.thread f) (R.idx f) := by rw [hft, hfi]
      have hrmw : X.rmw m f := by
        refine (R.rmw_iff m f).2 ⟨ht.trans hft.symm, by omega, ?_⟩
        rw [hlabf, R.lab_of] at hr₂
        exact hr₂
      have hevf : ev R f = (t, R.lab t (b.k t + 1)) := by simp [ev, hft, hfi]
      have hcs := Cpu.Step.two hst (hc₁ ▸ hnp₁) hp₂ hfits hst₂
        (fun h => by rw [hk₂] at h; cases h)
      have hreach₂ := hreach₁.tail hcs
      refine ⟨_, hb.extend (G := [m, f]) (more := [m, f]) (skip := false) (Or.inl ⟨rfl, rfl⟩) hsk
        (fun i h => ?_) (by simp; omega) (fun a i h ho => ?_) (fun e f' h => ?_)
        (by simpa [hevm, hevf] using hreach₂) (fun _ => ?_) (fun _ => ?_) (fun t' h => ?_) nofun,
        fun x => hmono x, m, hmd, hnew (by simp)⟩
      · simp at h
        rcases (by omega : i = 0 ∨ i = 1) with rfl | rfl
        · exact ⟨ht, by simp [hidx]⟩
        · exact ⟨hft, by simp [hfi]⟩
      · simp at h
        rcases (by omega : i = 0 ∨ i = 1) with rfl | rfl
        · exact Or.inl (hpm a ho)
        · rcases hpre f a (Or.inr hrmw) ho with h' | h'
          · exact Or.inl h'
          · rcases h' with rfl | h'
            · exact Or.inr ⟨0, by omega, rfl⟩
            · rw [R.rmw_unique h' hrmw] at ho
              exact absurd (.single ho) (hsc.acyclic f)
      · have hfr : X.kind f = .write := (wf.rmw_atomic m f hrmw).2.1
        have her : X.kind e = .read := (R.rmw_loc hcode h).1
        simp only [List.mem_cons, List.not_mem_nil, or_false]
        constructor
        · rintro (rfl | rfl)
          · exact Or.inr (R.rmw_unique h hrmw)
          · rw [hfr] at her; cases her
        · rintro (rfl | rfl)
          · exact absurd h (hmh e)
          · exact Or.inl (R.rmw_unique' h hrmw)
      · simp only [Cpu.move, setAt_same, List.length_cons, List.length_nil]
        rw [show n t - (b.k t + (0 + 1 + 1)) = d by omega]
        simpa [Nat.add_assoc] using hrest₂
      · simp only [Cpu.move, setAt_same]
        exact ⟨hst₂.ok (hst.ok (hc₁ ▸ hok₁)), hnp₄⟩
      · simp only [Cpu.move, setAt_other _ _ h]
        exact ho₁ t' h
  · -- A step that is not part of a read-modify-write.
    have hlone : ∀ f, ¬ X.rmw m f := by
      intro f hf
      obtain ⟨hft, hfi, hfr⟩ := R.rmw_inv hf
      cases d with
      | zero => have := (hn t (R.idx f)).1 ⟨f, hft ▸ ht, rfl⟩; omega
      | succ d =>
        rw [List.range'_succ, List.map_cons] at hrest
        have := (hrest.rmw_facts (hst.ok (hc₁ ▸ hok₁))).1 _ rfl
        rw [show b.k t + 1 = R.idx f by omega, ← ht, hft, R.lab_of] at this
        exact hp₂ (this hfr).1
    have hcs := Cpu.Step.one hst (hc₁ ▸ hnp₁) hp₂ hfits
    have hreach₂ := hreach₁.tail hcs
    refine ⟨_, hb.extend (G := [m]) (more := [m]) (skip := false) (Or.inl ⟨rfl, rfl⟩) hsk
      (fun i h => by simp at h; subst h; exact ⟨ht, by simp [hidx]⟩) (by simp; omega)
      (fun a i h ho => by simp at h; subst h; exact Or.inl (hpm a ho))
      (fun e f h => by
        simp only [List.mem_singleton]
        exact ⟨fun h' => absurd (h' ▸ h) (hlone f), fun h' => absurd (h' ▸ h) (hmh e)⟩)
      (by simpa [hevm] using hreach₂) (fun _ => ?_) (fun _ => ?_) (fun t' h => ?_) nofun,
      fun x => hmono x, m, hmd, hnew (by simp)⟩
    · simp only [Cpu.move, setAt_same, List.length_cons, List.length_nil]
      rw [show n t - (b.k t + (0 + 1)) = d by omega]
      simpa using hrest
    · simp only [Cpu.move, setAt_same]
      exact ⟨hst.ok (hc₁ ▸ hok₁), hp₂⟩
    · simp only [Cpu.move, setAt_other _ _ h]
      exact ho₁ t' h

end Main


open Classical in
/-- Each finite execution of a program in which the accesses happen one at a time comes from a
run of the CPU node. In each thread `t` with `n` events, the run makes the accesses of the
execution, in order. It can leave out the last access when that access is a read. -/
theorem Cpu.complete {P : Prog} {X : Execution} (hcode : ∀ t, (P.code t).NoRmw) (R : P.Run X)
    (wf : X.WellFormed) (hfin : X.Finite) (hsc : X.SeqConsistent) (hco : X.CoOrder) :
    ∃ evs c, Cpu.Reach (Cpu.init P) evs c ∧ ∀ t, ∃ n,
      (∀ i, (∃ e, X.thread e = t ∧ R.idx e = i) ↔ i < n) ∧
      (accessesOf evs t = (List.range n).map (R.lab t) ∨
        0 < n ∧ accessesOf evs t = (List.range (n - 1)).map (R.lab t) ∧
          (R.lab t (n - 1)).kind = .read) := by
  let n := fun t => (R.exists_len hfin t).choose
  have hn : ∀ t i, (∃ e, X.thread e = t ∧ R.idx e = i) ↔ i < n t :=
    fun t => (R.exists_len hfin t).choose_spec
  obtain ⟨l, hl⟩ := hfin
  have key : ∀ N (b : Built X), Good R n b → count l (fun x => ¬ b.Done R x) ≤ N →
      ∃ b', Good R n b' ∧ ∀ x, b'.Done R x := by
    intro N
    induction N with
    | zero =>
      intro b hb hc
      refine ⟨b, hb, fun x => Classical.byContradiction fun hx => ?_⟩
      have := count_pos (p := fun x => ¬ b.Done R x) (hl x) hx
      omega
    | succ N ih =>
      intro b hb hc
      by_cases hall : ∀ x, b.Done R x
      · exact ⟨b, hb, hall⟩
      · obtain ⟨e, he⟩ := Classical.not_forall.1 hall
        obtain ⟨b', hb', hmono, x, hx, hx'⟩ := hb.step hcode wf hsc hco ⟨l, hl⟩ hn he
        have := count_lt (p := fun x => ¬ b'.Done R x) (q := fun x => ¬ b.Done R x)
          (fun y hy hd => hy (hmono y hd)) (hl x) hx (fun h => h hx')
        exact ih b' hb' (by omega)
  obtain ⟨b, hb, hall⟩ := key _ _ (Good.init hcode hn) (Nat.le_refl _)
  refine ⟨b.evs, b.c, hb.reach, fun t => ⟨n t, hn t, ?_⟩⟩
  have hk : b.k t = n t := by
    refine Nat.le_antisymm (hb.k_le t) (Nat.le_of_not_lt fun h => ?_)
    obtain ⟨e, het, hei⟩ := (hn t (n t - 1)).2 (by omega)
    have := hall e
    unfold Built.Done at this
    rw [het] at this
    omega
  rw [hb.acc t, hk]
  cases hs : b.sk t
  · exact Or.inl (by simp)
  · obtain ⟨_, e, het, hei, hek⟩ := hb.sk_end t hs
    refine Or.inr ⟨by omega, by simp, ?_⟩
    rw [← hk, ← hei, Nat.add_sub_cancel, ← het, R.lab_of]
    exact hek


/-- With DRF-SC: take a program whose atomics are sequentially consistent, with no fences, and no
data race in any interleaving. Then each finite execution that the model allows comes from a run
of the CPU node, as in `Cpu.complete`. -/
theorem Cpu.complete_of_raceFree {P : Prog} {X : Execution} (hcode : ∀ t, (P.code t).NoRmw)
    (hsc : P.SeqCstOnly)
    (hP : ∀ Y, P.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.Finite → Y.CoOrder → Y.RaceFree)
    (R : P.Run X) (wf : X.WellFormed) (hfin : X.Finite) (hco : X.CoOrder) (hc : X.Consistent) :
    ∃ evs c, Cpu.Reach (Cpu.init P) evs c ∧ ∀ t, ∃ n,
      (∀ i, (∃ e, X.thread e = t ∧ R.idx e = i) ↔ i < n) ∧
      (accessesOf evs t = (List.range n).map (R.lab t) ∨
        0 < n ∧ accessesOf evs t = (List.range (n - 1)).map (R.lab t) ∧
          (R.lab t (n - 1)).kind = .read) :=
  Cpu.complete hcode R wf hfin (P.drf_sc_co hsc hP ⟨R⟩ wf hfin hco hc) hco

end Vx
