import VxMemoryModel.Language

/-!
# Tasks

A program with tasks, as in `tasks { .. }` of Vx. The main code is thread 0, and task `t` runs
as thread `t + 1`. Code starts task `t` with `spawn t` and waits for it with `join t`. The main
code and each task can start tasks, so the tasks make a tree. `spawnAt` and `joinAt` compute the
task id, as a `spawn` in a loop does with a new task on each pass.

Each task has two flags: `start t` and `done t`. `spawn t` sets `start t` with an `sc` write, and
the task first waits for that flag. At its end the task sets `done t`, and `join t` waits for it.
So a fork is a release and a join is an acquire, as in `../../vx.cat`.

Each `spawn` also sets a register, the mark of its task, so the code knows which tasks it
started. At its end, a `tasks` block joins each task that has its mark set. `Blocks.lean` shows
that each task that a block started happens before the code after the block.

`fork_hb` and `join_hb` show that each run orders a task after the code before its `spawn`, and
before the code after its `join`. This holds for a task at any depth of the tree.
-/

namespace Vx

/-- Each access of `s` is to a place below `n`, and each task that `s` starts or joins has an id
below `tasks`. -/
def Stmt.Below (n tasks : Nat) : Stmt → Prop
  | load _ x _ | store x _ _ | fetchAdd _ x _ _ | cas _ x _ _ _ | rmwStore x _ _ | wait x => x < n
  | loadAt _ base len _ _ | storeAt base len _ _ _ | waitAt base len _ => base + len ≤ n
  | spawnAt base len _ | joinAt base len _ => base + len ≤ tasks
  | seq s t | ite _ s t => s.Below n tasks ∧ t.Below n tasks
  | loop _ s | tasksBlock s => s.Below n tasks
  | skip | fence _ | assign .. | assignAt .. => True

/-- `s` writes the place `y` only in its last step. -/
def Stmt.EndsWith (y : Nat) : Stmt → Prop
  | skip | store .. => True
  | seq s t => s.Accesses (fun k _ x => k = .write → x ≠ y) ∧ t.EndsWith y
  | s => s.Accesses fun k _ x => k = .write → x ≠ y

theorem Stmt.Accesses.endsWith {y} {s : Stmt} (h : s.Accesses fun k _ x => k = .write → x ≠ y) :
    s.EndsWith y := by
  induction s with
  | skip | store => trivial
  | seq _ _ _ ih2 => exact ⟨h.1, ih2 h.2⟩
  | _ => exact h

theorem Step.endsWith {y s regs a s' regs'} (h : Step s regs a s' regs') (hs : s.EndsWith y) :
    s'.EndsWith y ∧ ∀ l, a = some l → l.kind = .write → l.loc = y → s' = .skip := by
  cases h with
  | seq h =>
    have h' := (accessRule _).step h hs.1
    exact ⟨⟨h'.1, hs.2⟩, fun l hl hk hy => absurd hy (h'.2 l hl hk)⟩
  | seq_skip => exact ⟨hs.2, nofun⟩
  | load | store | cas_ne | rmwStore | fence | assign | loadAt | storeAt | assignAt =>
    exact ⟨trivial, fun _ _ _ _ => rfl⟩
  | waitAt _ v =>
    refine ⟨?_, fun l hl hk => by cases hl; cases hk⟩
    by_cases hv : v = 0 <;> simp [hv, Stmt.EndsWith] <;> exact hs
  | fetchAdd | cas_eq => exact ⟨hs.2, fun l hl hk => by cases hl; cases hk⟩
  | ite_true => exact ⟨Stmt.Accesses.endsWith hs.1, nofun⟩
  | ite_false => exact ⟨Stmt.Accesses.endsWith hs.2, nofun⟩
  | loop => exact ⟨⟨⟨hs, hs⟩, trivial⟩, nofun⟩
  | wait v =>
    refine ⟨?_, fun l hl hk => by cases hl; cases hk⟩
    by_cases hv : v = 0 <;> simp [hv, Stmt.EndsWith] <;> exact hs

theorem Runs.skip_nil {regs L} (h : Runs .skip regs L) : L = [] := by
  cases h with
  | stop => rfl
  | silent hs => cases hs
  | visible hs => cases hs

/-- In a run of code that ends with its write of `y`, nothing comes after that write. -/
theorem Runs.endsWith {y s regs L} (h : Runs s regs L) (hs : s.EndsWith y) :
    ∀ L1 l L2, L = L1 ++ l :: L2 → l.kind = .write → l.loc = y → L2 = [] := by
  induction h with
  | stop => intro L1 l L2 hL; simp at hL
  | silent hstep _ ih => exact ih (hstep.endsWith hs).1
  | visible hstep hr ih =>
    intro L1 l L2 hL hk hy
    rcases L1 with _ | ⟨b, L1⟩
    · obtain ⟨rfl, rfl⟩ := List.cons.inj hL
      rw [(hstep.endsWith hs).2 _ rfl hk hy] at hr
      exact hr.skip_nil
    · obtain ⟨rfl, hL'⟩ := List.cons.inj hL
      exact ih (hstep.endsWith hs).1 L1 l L2 hL' hk hy

theorem Runs.wait_first {x k regs l L} (h : Runs (.seq (.wait x) k) regs (l :: L)) :
    l.kind = .read ∧ l.order = .sc ∧ l.loc = x := by
  cases h with
  | silent hs => cases hs with | seq hs => cases hs
  | visible hs => cases hs with | seq hs => cases hs; exact ⟨rfl, rfl, rfl⟩

structure TaskProg where
  main : Stmt
  /-- The code of task `t`. -/
  task : Nat → Stmt
  /-- The task ids are below `count`. -/
  count : Nat
  /-- The code uses the places below `flags`. The places from `flags` up hold the flags. -/
  flags : Nat
  /-- The code uses the registers below `marks`. The register `marks + t` is the mark of task
  `t`. -/
  marks : Nat

namespace TaskProg

variable (P : TaskProg)

/-- The `start` flags are an array of `count` places, and the `done` flags follow it. -/
def start (t : Nat) : Nat := P.flags + t
def done (t : Nat) : Nat := P.flags + P.count + t

/-- The tasks that `s` can start, but not in a `tasks` block inside `s`. -/
def _root_.Vx.Stmt.spawnIds : Stmt → List Nat
  | .spawnAt base len _ => (List.range len).map (base + ·)
  | .seq s t | .ite _ s t => s.spawnIds ++ t.spawnIds
  | .loop _ s => s.spawnIds
  | _ => []

/-- Clear the marks of the tasks `ids`. -/
def clearMarks (ids : List Nat) : Stmt :=
  ids.foldr (fun t s => .seq (.assign (P.marks + t) (.const 0)) s) .skip

/-- Join each task of `ids` that has its mark set. -/
def joinMarked (ids : List Nat) : Stmt :=
  ids.foldr (fun t s => .seq (.ite (.reg (P.marks + t)) (.waitAt (P.done t) 1 (.const 0)) .skip) s)
    .skip

/-- Replace `spawn` and `join` with the writes and the waits of the flags. A `spawn` sets the
mark of its task first. -/
def lower : Stmt → Stmt
  | .spawnAt base len i =>
    .seq (.assignAt (P.marks + base) len i (.const 1)) (.storeAt (P.start base) len i (.const 1) .sc)
  | .joinAt base len i => .waitAt (P.done base) len i
  | .tasksBlock s => .seq (P.clearMarks s.spawnIds) (.seq (lower s) (P.joinMarked s.spawnIds))
  | .assignAt base len i e => .assignAt base len i e
  | .seq s u => .seq (lower s) (lower u)
  | .ite c s u => .ite c (lower s) (lower u)
  | .loop c s => .loop c (lower s)
  | .skip => .skip
  | .load r x o => .load r x o
  | .store x e o => .store x e o
  | .fetchAdd r x e o => .fetchAdd r x e o
  | .cas r x e n o => .cas r x e n o
  | .rmwStore x v o => .rmwStore x v o
  | .fence o => .fence o
  | .assign r e => .assign r e
  | .wait x => .wait x
  | .loadAt r base len i o => .loadAt r base len i o
  | .storeAt base len i e o => .storeAt base len i e o
  | .waitAt base len i => .waitAt base len i

/-- The code of thread `t + 1`: task `t` between its two flags. A thread with no task id runs no
code. -/
def taskCode (t : Nat) : Stmt :=
  if t < P.count then .seq (.wait (P.start t)) (.seq (P.lower (P.task t)) (.signal (P.done t)))
  else .skip

def toProg : Prog where
  code
    | 0 => P.lower P.main
    | t + 1 => P.taskCode t

/-- The rules for the code: no code uses the flags, and each task id is below `count`. -/
structure Ok : Prop where
  main : P.main.Below P.flags P.count
  task : ∀ t, (P.task t).Below P.flags P.count

variable {P}

theorem spawnIds_lt {n tasks} {s : Stmt} (h : s.Below n tasks) : ∀ u ∈ s.spawnIds, u < tasks := by
  induction s with
  | spawnAt =>
    intro u hu
    simp only [Stmt.spawnIds, List.mem_map, List.mem_range] at hu
    obtain ⟨j, hj, rfl⟩ := hu
    simp only [Stmt.Below] at h
    omega
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 =>
    intro u hu
    rcases List.mem_append.1 hu with hu | hu
    · exact ih1 h.1 u hu
    · exact ih2 h.2 u hu
  | loop _ _ ih => exact ih h
  | _ => nofun

theorem clearMarks_accesses (p : Kind → Order → Nat → Prop) (ids : List Nat) :
    (P.clearMarks ids).Accesses p := by
  induction ids with
  | nil => trivial
  | cons _ _ ih => exact ⟨trivial, ih⟩

theorem joinMarked_accesses {p : Kind → Order → Nat → Prop} {ids : List Nat}
    (h : ∀ u ∈ ids, p .read .sc (P.done u)) : (P.joinMarked ids).Accesses p := by
  induction ids with
  | nil => trivial
  | cons u _ ih =>
    refine ⟨⟨fun j hj => ?_, trivial⟩, ih fun v hv => h v (List.mem_cons_of_mem _ hv)⟩
    obtain rfl : j = 0 := by omega
    exact h u List.mem_cons_self

theorem clearMarks_seqCstOnly (ids : List Nat) : (P.clearMarks ids).SeqCstOnly := by
  induction ids with
  | nil => trivial
  | cons _ _ ih => exact ⟨trivial, ih⟩

theorem joinMarked_seqCstOnly (ids : List Nat) : (P.joinMarked ids).SeqCstOnly := by
  induction ids with
  | nil => trivial
  | cons _ _ ih => exact ⟨⟨trivial, trivial⟩, ih⟩

theorem taskCode_of_lt {t} (ht : t < P.count) :
    P.taskCode t = .seq (.wait (P.start t)) (.seq (P.lower (P.task t)) (.signal (P.done t))) :=
  by simp [taskCode, ht]

/-- The lowered code writes only its own places and the `start` flags, and it reads only its own
places and the `done` flags. -/
theorem lower_accesses {s : Stmt} (h : s.Below P.flags P.count) :
    (P.lower s).Accesses fun k o x => k = .fence ∨ x < P.flags ∨
      o = .sc ∧ (k = .write ∧ (∃ u < P.count, x = P.start u) ∨
        k = .read ∧ ∃ u < P.count, x = P.done u) := by
  induction s with
  | spawnAt base =>
    refine ⟨trivial, fun j hj => Or.inr (Or.inr ⟨rfl, Or.inl ⟨rfl, base + j, ?_, ?_⟩⟩)⟩
    · simp only [Stmt.Below] at h; omega
    · simp only [start]; omega
  | tasksBlock s ih =>
    refine ⟨clearMarks_accesses _ _, ih h, joinMarked_accesses fun u hu => ?_⟩
    exact Or.inr (Or.inr ⟨rfl, Or.inr ⟨rfl, u, spawnIds_lt (s := s) h u hu, rfl⟩⟩)
  | joinAt base =>
    refine fun j hj => Or.inr (Or.inr ⟨rfl, Or.inr ⟨rfl, base + j, ?_, ?_⟩⟩)
    · simp only [Stmt.Below] at h; omega
    · simp only [done]; omega
  | load | store | rmwStore | wait => exact Or.inr (Or.inl h)
  | fetchAdd | cas => exact ⟨Or.inr (Or.inl h), Or.inr (Or.inl h)⟩
  | loadAt | storeAt | waitAt =>
    exact fun j hj => Or.inr (Or.inl (by simp only [Stmt.Below] at h; omega))
  | fence => exact Or.inl rfl
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | loop _ _ ih => exact ih h
  | skip | assign | assignAt => trivial

theorem lower_seqCstOnly {s : Stmt} (h : s.SeqCstOnly) : (P.lower s).SeqCstOnly := by
  induction s with
  | spawnAt => exact ⟨trivial, Or.inr rfl⟩
  | joinAt => trivial
  | tasksBlock _ ih => exact ⟨clearMarks_seqCstOnly _, ih h, joinMarked_seqCstOnly _⟩
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | loop _ _ ih => exact ih h
  | _ => exact h

theorem code_succ {t} : P.toProg.code (t + 1) = P.taskCode t := rfl

theorem seqCstOnly (hm : P.main.SeqCstOnly) (ht : ∀ t, (P.task t).SeqCstOnly) :
    P.toProg.SeqCstOnly
  | 0 => lower_seqCstOnly hm
  | t + 1 => by
    rw [code_succ]
    unfold taskCode
    split
    · exact ⟨trivial, lower_seqCstOnly (ht t), Or.inr rfl⟩
    · trivial

/-- Each thread that makes an access runs a task with an id below `count`. -/
theorem task_lt {X : Execution} (r : P.toProg.Run X) {e : X.E} {t} (ht : X.thread e = t + 1) :
    t < P.count := by
  refine Nat.lt_of_not_le fun hc => ?_
  have h := r.trace e
  rw [ht, code_succ] at h
  simp only [taskCode, Nat.not_lt.2 hc] at h
  unfold Trace at h
  have := h.skip_nil
  simp [List.range_succ] at this

/-- The accesses of a run to the flags. Each thread can write `start` and read `done`, and task
`u` reads `start u` and writes `done u`. All these accesses are `sc`. -/
theorem access (hP : P.Ok) {X : Execution} (r : P.toProg.Run X) (e : X.E) :
    X.kind e = .fence ∨ X.loc e < P.flags ∨ X.IsSeqCst e ∧
      (X.IsWrite e ∧ (∃ u < P.count, X.loc e = P.start u) ∨
        X.IsRead e ∧ (∃ u < P.count, X.loc e = P.done u) ∨
        ∃ u < P.count, X.thread e = u + 1 ∧ (X.IsRead e ∧ X.loc e = P.start u ∨
          X.IsWrite e ∧ X.loc e = P.done u)) := by
  have key : ∀ t, (P.toProg.code t).Accesses fun k o x => k = .fence ∨ x < P.flags ∨ o = .sc ∧
      (k = .write ∧ (∃ u < P.count, x = P.start u) ∨ k = .read ∧ (∃ u < P.count, x = P.done u) ∨
        ∃ u < P.count, t = u + 1 ∧ (k = .read ∧ x = P.start u ∨ k = .write ∧ x = P.done u)) := by
    intro t
    have lowered {s : Stmt} (h : s.Below P.flags P.count) := (lower_accesses h).mono
      (q := fun k o x => k = .fence ∨ x < P.flags ∨ o = .sc ∧
        (k = .write ∧ (∃ u < P.count, x = P.start u) ∨ k = .read ∧ (∃ u < P.count, x = P.done u) ∨
          ∃ u < P.count, t = u + 1 ∧ (k = .read ∧ x = P.start u ∨ k = .write ∧ x = P.done u)))
      fun _ _ _ h => by
        rcases h with h | h | ⟨ho, h | h⟩
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr ⟨ho, Or.inl h⟩)
        · exact Or.inr (Or.inr ⟨ho, Or.inr (Or.inl h)⟩)
    cases t with
    | zero => exact lowered hP.main
    | succ u =>
      rw [code_succ]
      unfold taskCode
      split
      · rename_i hu
        exact ⟨Or.inr (Or.inr ⟨rfl, Or.inr (Or.inr ⟨u, hu, rfl, Or.inl ⟨rfl, rfl⟩⟩)⟩),
          lowered (hP.task u), Or.inr (Or.inr ⟨rfl, Or.inr (Or.inr ⟨u, hu, rfl, Or.inr ⟨rfl, rfl⟩⟩)⟩)⟩
      · trivial
  have h := (accessRule _).trace (r.trace e) (key (X.thread e))
  simp only [accessRule, r.lab_eq e] at h
  exact h

/-- The first access of task `t` is a read of `start t`. -/
theorem first (r : P.toProg.Run X) {e : X.E} {t} (ht : X.thread e = t + 1)
    (hi : r.idx e = 0) : X.IsRead e ∧ X.IsSeqCst e ∧ X.loc e = P.start t := by
  have htc := task_lt r ht
  have h := r.trace e
  rw [ht, hi, code_succ, taskCode_of_lt htc] at h
  unfold Trace at h
  have hl := Runs.wait_first (L := []) h
  have hlab := r.lab_eq e
  rw [ht, hi] at hlab
  rw [hlab] at hl
  exact hl

/-- In task `t`, each access comes before the write of `done t`, or is that write. -/
theorem before_done (hP : P.Ok) (r : P.toProg.Run X) {a w : X.E} {t}
    (ha : X.thread a = t + 1) (hw : X.thread w = t + 1) (hww : X.IsWrite w)
    (hwl : X.loc w = P.done t) : a = w ∨ X.po a w := by
  rcases Nat.lt_trichotomy (r.idx a) (r.idx w) with h | h | h
  · exact Or.inr ((r.po_iff a w).2 ⟨ha.trans hw.symm, h⟩)
  · exact Or.inl (r.idx_inj a w (ha.trans hw.symm) h)
  · exfalso
    have hends : (P.toProg.code (t + 1)).EndsWith (P.done t) := by
      rw [code_succ, taskCode_of_lt (task_lt r ha)]
      refine ⟨nofun, ?_, trivial⟩
      refine (lower_accesses (hP.task t)).mono fun k _ x hx hk => ?_
      rcases hx with rfl | hx | ⟨-, ⟨-, u, hu, rfl⟩ | ⟨hk', -⟩⟩
      · cases hk
      · unfold done
        omega
      · unfold start done
        omega
      · subst hk
        cases hk'
    have htr := (r.trace a).prefix (m := r.idx w + 2) (by omega)
    rw [ha] at htr
    unfold Trace at htr
    rw [List.range_succ, List.range_succ, List.map_append, List.map_append] at htr
    have hlab := r.lab_eq w
    rw [hw] at hlab
    have := htr.endsWith hends ((List.range (r.idx w)).map (r.lab (t + 1))) (r.lab (t + 1) (r.idx w)) [r.lab (t + 1) (r.idx w + 1)]
      (by simp) (by rw [hlab]; exact hww) (by rw [hlab]; exact hwl)
    cases this

/-- A task starts after the code before its `spawn`: the code before each write of `start t`
comes before each access of task `t`. The code that starts the task can be in any task. -/
theorem fork_hb (hP : P.Ok) {X : Execution} (r : P.toProg.Run X) {t} {a b : X.E}
    (hb : X.thread b = t + 1)
    (ha : ∀ w, X.IsWrite w → X.loc w = P.start t → X.po a w) : X.hb a b := by
  have htc := task_lt r hb
  obtain ⟨e, he, hi⟩ : ∃ e, X.thread e = X.thread b ∧ r.idx e = 0 := by
    rcases Nat.eq_zero_or_pos (r.idx b) with h | h
    · exact ⟨b, rfl, h⟩
    · exact r.idx_down b 0 h
  obtain ⟨her, hes, hel⟩ := first r (he.trans hb) hi
  have heb : e = b ∨ X.po e b := by
    rcases Nat.eq_zero_or_pos (r.idx b) with h | h
    · exact Or.inl (r.idx_inj e b he (hi.trans h.symm))
    · exact Or.inr ((r.po_iff e b).2 ⟨he, hi ▸ h⟩)
  obtain ⟨w, hw⟩ := r.rf_source e her
  obtain ⟨hww, -, hwl⟩ := r.rf_write_read w e hw
  rw [hel] at hwl
  have hwk : X.kind w = .write := hww
  rcases access hP r w with hk | hlt | ⟨hsc, ⟨-, -⟩ | ⟨hr, -⟩ | ⟨u, -, -, ⟨hr, -⟩ | ⟨-, hx⟩⟩⟩
  · rw [hwk] at hk
    cases hk
  · unfold start at hwl
    omega
  · exact X.hb_of_signal (Or.inr (ha w hww hwl)) hww hsc hw hes heb
  · unfold Execution.IsRead at hr
    rw [hwk] at hr
    cases hr
  · unfold Execution.IsRead at hr
    rw [hwk] at hr
    cases hr
  · unfold start done at *
    omega

/-- The code after a `join` comes after the task: each access of task `t` comes before the
code after each read of `done t`. Only `join t` reads it. -/
theorem join_hb (hP : P.Ok) {X : Execution} (r : P.toProg.Run X) {t} {a j b : X.E}
    (ha : X.thread a = t + 1) (hjr : X.IsRead j)
    (hjl : X.loc j = P.done t) (hjb : j = b ∨ X.po j b) : X.hb a b := by
  have htc := task_lt r ha
  have hjk : X.kind j = .read := hjr
  have hjs : X.IsSeqCst j := by
    rcases access hP r j with hk | hlt | ⟨hsc, -⟩
    · rw [hjk] at hk
      cases hk
    · unfold done at hjl
      omega
    · exact hsc
  obtain ⟨w, hw⟩ := r.rf_source j hjr
  obtain ⟨hww, -, hwl⟩ := r.rf_write_read w j hw
  rw [hjl] at hwl
  have hwk : X.kind w = .write := hww
  rcases access hP r w with hk | hlt | ⟨hsc, ⟨-, u, hu, hx⟩ | ⟨hr, -⟩ | ⟨u, -, hwu, ⟨hr, -⟩ | ⟨-, hx⟩⟩⟩
  · rw [hwk] at hk
    cases hk
  · unfold done at hwl
    omega
  · unfold start done at *
    omega
  · unfold Execution.IsRead at hr
    rw [hwk] at hr
    cases hr
  · unfold Execution.IsRead at hr
    rw [hwk] at hr
    cases hr
  · have hut : u = t := by
      unfold done at *
      omega
    subst hut
    exact X.hb_of_signal (before_done hP r ha hwu hww hwl) hww hsc hw hjs hjb

/-- DRF-SC for programs with tasks. -/
theorem drf_sc (P : TaskProg) (hm : P.main.SeqCstOnly) (ht : ∀ t, (P.task t).SeqCstOnly)
    (hP : ∀ Y, P.toProg.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.RaceFree)
    {X : Execution} (hX : P.toProg.Execs X) (wf : X.WellFormed) (hfin : X.Finite)
    (hc : X.Consistent) : X.SeqConsistent :=
  P.toProg.drf_sc (seqCstOnly hm ht) hP hX wf hfin hc

/-- `drf_sc`, when `co` orders the writes to each place. -/
theorem drf_sc_co (P : TaskProg) (hm : P.main.SeqCstOnly) (ht : ∀ t, (P.task t).SeqCstOnly)
    (hP : ∀ Y, P.toProg.Execs Y → Y.WellFormed → Y.SeqConsistent → Y.Finite → Y.CoOrder →
      Y.RaceFree)
    {X : Execution} (hX : P.toProg.Execs X) (wf : X.WellFormed) (hfin : X.Finite)
    (ho : X.CoOrder) (hc : X.Consistent) : X.SeqConsistent :=
  P.toProg.drf_sc_co (seqCstOnly hm ht) hP hX wf hfin ho hc

end TaskProg

end Vx
