import VxMemoryModel.Blocks

/-!
# Race freedom from footprints

`TaskProg.drf_sc` needs the hypothesis that no SC execution of the program has a data race.
`raceFree` proves that each execution has no data race, from a rule on the code.

The main code is a list of phases. A phase is code that starts no task, or a `tasks` block. The
rule for a block has two parts. Two tasks of the block make no accesses that can race. The code of
the block makes no access that can race with a task of the block. Code outside the block can
access any place: the code before the block happens before its tasks, and its tasks happen before
the code after it. Two accesses can race only when they touch one place, one of them writes, and
one of them is plain.

This is the first part of the rule. Tasks do not start tasks, and the code has no channels and no
locks.
-/

namespace Vx

/-- Two accesses that can never make a data race: one is a fence, they touch two places, both
read, or both are atomic. -/
def Compat (k : Kind) (o : Order) (x : Nat) (k' : Kind) (o' : Order) (x' : Nat) : Prop :=
  k = .fence ∨ k' = .fence ∨ x ≠ x' ∨ (k = .read ∧ k' = .read) ∨ (o ≠ .plain ∧ o' ≠ .plain)

/-- No access of `s` can race with an access of `t`. -/
def Stmt.Apart (s t : Stmt) : Prop := s.Accesses fun k o x => t.Accesses (Compat k o x)

/-- `s` starts no task and joins no task. -/
def Stmt.NoTasks : Stmt → Prop
  | spawnAt .. | joinAt .. | tasksBlock _ => False
  | seq s t | ite _ s t => s.NoTasks ∧ t.NoTasks
  | loop _ s => s.NoTasks
  | _ => True

theorem Stmt.Accesses.forall_mem {q : Nat → Kind → Order → Nat → Prop} {ids : List Nat}
    {s : Stmt} (h : ∀ v ∈ ids, s.Accesses (q v)) :
    s.Accesses fun k o x => ∀ v ∈ ids, q v k o x := by
  induction s with
  | load | store | rmwStore | fence | wait => exact fun v hv => h v hv
  | fetchAdd | cas => exact ⟨fun v hv => (h v hv).1, fun v hv => (h v hv).2⟩
  | loadAt | storeAt | waitAt => exact fun j hj v hv => h v hv j hj
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 =>
    exact ⟨ih1 fun v hv => (h v hv).1, ih2 fun v hv => (h v hv).2⟩
  | loop _ _ ih => exact ih h
  | _ => trivial

/-- A run of `seq s t` stops in `s`, or runs `s` to its end and then stops in `t`. -/
theorem Runs.seq_split {s t regs L} (h : Runs (.seq s t) regs L) :
    Runs s regs L ∨ ∃ L1 L2 regs', Steps s regs L1 .skip regs' ∧ Runs t regs' L2 ∧ L = L1 ++ L2 := by
  generalize hx : Stmt.seq s t = x at h
  induction h generalizing s with
  | stop => exact Or.inl .stop
  | silent hs h' ih =>
    subst hx
    cases hs with
    | seq_skip => exact Or.inr ⟨[], _, _, .refl, h', rfl⟩
    | seq hs1 =>
      rcases ih rfl with h1 | ⟨L1, L2, r', h1, h2, rfl⟩
      · exact Or.inl (.silent hs1 h1)
      · exact Or.inr ⟨L1, L2, r', .silent hs1 h1, h2, rfl⟩
  | visible hs h' ih =>
    subst hx
    cases hs with
    | seq hs1 =>
      rcases ih rfl with h1 | ⟨L1, L2, r', h1, h2, rfl⟩
      · exact Or.inl (.visible hs1 h1)
      · exact Or.inr ⟨_ :: L1, L2, r', .visible hs1 h1, h2, rfl⟩

theorem getElem?_rangeMap {n i : Nat} {f : Nat → Label} (hi : i < n) :
    ((List.range n).map f)[i]? = some (f i) := by
  simp [List.getElem?_map, List.getElem?_range hi]

namespace TaskProg

/-- A part of the main code: code that starts no task, or a `tasks` block. -/
inductive Phase where
  | code (s : Stmt)
  | block (s : Stmt)

def Phase.stmt : Phase → Stmt
  | .code s => s
  | .block s => .tasksBlock s

/-- The tasks that a phase can start. -/
def Phase.ids : Phase → List Nat
  | .code _ => []
  | .block s => s.spawnIds

/-- The phases, one after the other. -/
def seqAll : List Phase → Stmt
  | [] => .skip
  | p :: ps => .seq p.stmt (seqAll ps)

variable (P : TaskProg)

/-- The rule for one phase. The code of a block and each task of the block are apart, and two
tasks of the block are apart. -/
def Phase.Ok : Phase → Prop
  | .code s => s.NoTasks
  | .block s => s.NoBlocks ∧ s.RegsBelow P.marks ∧ (∀ u ∈ s.spawnIds, s.Apart (P.task u)) ∧
      ∀ u ∈ s.spawnIds, ∀ v ∈ s.spawnIds, u ≠ v → (P.task u).Apart (P.task v)

/-- Each phase follows its rule, and no two phases start the same task. -/
def PhasesOk : List Phase → Prop
  | [] => True
  | p :: ps => p.Ok P ∧ (∀ u ∈ p.ids, ∀ q ∈ ps, u ∉ q.ids) ∧ PhasesOk ps

/-- The rule on the code: the main code is the phases `ps`, and the tasks start no tasks. -/
structure Footprints (ps : List Phase) : Prop where
  ok : P.Ok
  main : P.main = seqAll ps
  phases : P.PhasesOk ps
  tasks : ∀ t, (P.task t).NoTasks

/-- What a block allows for an access in its lowered code. Below `flags`, the access cannot race
with a task of the block. It writes `start v` only for a task `v` of the block. -/
def InBlock (B : Stmt) (k : Kind) (o : Order) (x : Nat) : Prop :=
  (x < P.flags → ∀ v ∈ B.spawnIds, (P.task v).Accesses (Compat k o x)) ∧
    (k = .write → ∀ v, x = P.start v → v ∈ B.spawnIds)

/-- What a phase allows for an access in its lowered code. -/
def PhasePred : Phase → Kind → Order → Nat → Prop
  | .code _ => fun k _ x => k = .fence ∨ x < P.flags
  | .block B => P.InBlock B

/-- The label writes the flag `start u`. -/
def StartW (u : Nat) (l : Label) : Prop := l.kind = .write ∧ l.loc = P.start u

/-- Where the position `m` of a run `L` of the main code is, for the task `u`. It comes before
each write of `start u`, or it is in the block of `u`, or it comes after a run of that block that
wrote `start u`. -/
def Status (u : Nat) (L : List Label) (m : Nat) : Prop :=
  (∀ m' l, L[m']? = some l → P.StartW u l → m < m') ∨
  (∃ B, u ∈ B.spawnIds ∧ (Phase.block B).Ok P ∧
    ∀ l, L[m]? = some l → P.InBlock B l.kind l.order l.loc) ∨
  (∃ B n L' regs regs', (Phase.block B).Ok P ∧ B.Below P.flags P.count ∧
    Steps (P.lower (.tasksBlock B)) regs L' .skip regs' ∧
    (∀ i (hi : i < L'.length), L[n + i]? = some L'[i]) ∧ n + L'.length ≤ m ∧
    ∃ i, ∃ hi : i < L'.length, P.StartW u L'[i])

variable {P}

theorem lower_noTasks {s : Stmt} (h : s.NoTasks) : P.lower s = s := by
  induction s with
  | spawnAt | joinAt | tasksBlock => simp only [Stmt.NoTasks] at h
  | seq _ _ ih1 ih2 => simp only [lower, ih1 h.1, ih2 h.2]
  | ite _ _ _ ih1 ih2 => simp only [lower, ih1 h.1, ih2 h.2]
  | loop _ _ ih => simp only [lower, ih h]
  | _ => rfl

/-- Code that starts no task accesses only the places below `flags`. -/
theorem code_accesses {s : Stmt} (hb : s.Below P.flags P.count) (hn : s.NoTasks) :
    s.Accesses fun k _ x => k = .fence ∨ x < P.flags := by
  induction s with
  | spawnAt | joinAt | tasksBlock => simp only [Stmt.NoTasks] at hn
  | load | store | rmwStore | wait => exact Or.inr hb
  | fence => exact Or.inl rfl
  | fetchAdd | cas => exact ⟨Or.inr hb, Or.inr hb⟩
  | loadAt | storeAt | waitAt =>
    exact fun j hj => Or.inr (by simp only [Stmt.Below] at hb; omega)
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 hb.1 hn.1, ih2 hb.2 hn.2⟩
  | loop _ _ ih => exact ih hb hn
  | _ => trivial

/-- The lowered body of a block keeps the accesses of the body below `flags`, and writes
`start v` only for a task `v` that the body starts. -/
theorem lower_inBlock {q : Kind → Order → Nat → Prop} {ids : List Nat} {s : Stmt}
    (hb : s.Below P.flags P.count) (hn : s.NoBlocks) (hsub : ∀ u ∈ s.spawnIds, u ∈ ids)
    (hq : s.Accesses q) :
    (P.lower s).Accesses fun k o x => (x < P.flags → q k o x) ∧
      (k = .write → ∀ v, x = P.start v → v ∈ ids) := by
  induction s with
  | spawnAt base len i =>
    refine ⟨trivial, fun j hj => ⟨fun hx => ?_, fun _ v hv => ?_⟩⟩
    · simp only [start] at hx
      omega
    · have : v = base + j := by
        simp only [start] at hv
        omega
      subst this
      exact hsub _ (List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩)
  | joinAt base len i =>
    refine fun j hj => ⟨fun hx => ?_, nofun⟩
    simp only [done] at hx
    omega
  | tasksBlock => simp only [Stmt.NoBlocks] at hn
  | load | fence | wait => exact ⟨fun _ => hq, nofun⟩
  | store | rmwStore =>
    refine ⟨fun _ => hq, fun _ v hv => ?_⟩
    simp only [Stmt.Below] at hb
    simp only [start] at hv
    omega
  | fetchAdd | cas =>
    refine ⟨⟨fun _ => hq.1, nofun⟩, fun _ => hq.2, fun _ v hv => ?_⟩
    simp only [Stmt.Below] at hb
    simp only [start] at hv
    omega
  | loadAt | waitAt => exact fun j hj => ⟨fun _ => hq j hj, nofun⟩
  | storeAt =>
    refine fun j hj => ⟨fun _ => hq j hj, fun _ v hv => ?_⟩
    simp only [Stmt.Below] at hb
    simp only [start] at hv
    omega
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 =>
    exact ⟨ih1 hb.1 hn.1 (fun u hu => hsub u (List.mem_append_left _ hu)) hq.1,
      ih2 hb.2 hn.2 (fun u hu => hsub u (List.mem_append_right _ hu)) hq.2⟩
  | loop _ _ ih => exact ih hb hn hsub hq
  | skip | assign | assignAt => trivial

/-- Each access of the lowered code of a phase follows `PhasePred`. -/
theorem phase_accesses {p : Phase} (hp : p.Ok P) (hb : p.stmt.Below P.flags P.count) :
    (P.lower p.stmt).Accesses (P.PhasePred p) := by
  cases p with
  | code s =>
    show (P.lower s).Accesses _
    rw [lower_noTasks hp]
    exact code_accesses hb hp
  | block B =>
    obtain ⟨hn, -, hap, -⟩ := hp
    show (P.lower (.tasksBlock B)).Accesses (P.InBlock B)
    refine ⟨P.clearMarks_accesses _ _, (lower_inBlock (s := B) (ids := B.spawnIds) hb hn (fun _ h => h)
      (Stmt.Accesses.forall_mem (q := fun v k o x => (P.task v).Accesses (Compat k o x)) hap)).mono
        fun _ _ _ h => h, P.joinMarked_accesses fun u _ => ⟨fun hx => ?_, nofun⟩⟩
    simp only [done] at hx
    omega

theorem PhasePred.ids_of_start {p : Phase} {k o x} (h : P.PhasePred p k o x) (hk : k = .write)
    {u} (hx : x = P.start u) : u ∈ p.ids := by
  cases p with
  | code s =>
    simp only [PhasePred] at h
    rcases h with h | h
    · rw [hk] at h
      cases h
    · simp only [start] at hx
      omega
  | block B => exact h.2 hk u hx

/-- The main code writes `start u` only for a task `u` of one of its blocks. -/
theorem seqAll_accesses {ps : List Phase} (hok : P.PhasesOk ps)
    (hb : (seqAll ps).Below P.flags P.count) :
    (P.lower (seqAll ps)).Accesses fun k _ x =>
      k = .write → ∀ u, x = P.start u → ∃ p ∈ ps, u ∈ p.ids := by
  induction ps with
  | nil => trivial
  | cons p ps ih =>
    refine ⟨(phase_accesses hok.1 hb.1).mono fun _ _ _ h hk u hx =>
      ⟨p, List.mem_cons_self, h.ids_of_start hk hx⟩,
      (ih hok.2.2 hb.2).mono fun _ _ _ h hk u hx => ?_⟩
    obtain ⟨q, hq, hu⟩ := h hk u hx
    exact ⟨q, List.mem_cons_of_mem _ hq, hu⟩

/-- Each position of a run of the main code has a `Status` for each task. -/
theorem status {ps : List Phase} (hok : P.PhasesOk ps) (hb : (seqAll ps).Below P.flags P.count)
    {regs L} (hr : Runs (P.lower (seqAll ps)) regs L) (u : Nat) {m : Nat} (hm : m < L.length) :
    P.Status u L m := by
  induction ps generalizing regs L m with
  | nil =>
    have := Runs.skip_nil (regs := regs) hr
    subst this
    simp at hm
  | cons p ps ih =>
    have hacc := phase_accesses hok.1 hb.1
    rcases Runs.seq_split hr with hp | ⟨L1, L2, regs', h1, h2, rfl⟩
    · -- The run stops in the phase `p`.
      have hl := (accessRule _).runs hp hacc
      by_cases hu : u ∈ p.ids
      · cases p with
        | code s => simp [Phase.ids] at hu
        | block B =>
          exact Or.inr (Or.inl ⟨B, hu, hok.1, fun l hl' => hl l (List.mem_of_getElem? hl')⟩)
      · exact Or.inl fun _ l hl' hs =>
          (hu ((hl l (List.mem_of_getElem? hl')).ids_of_start hs.1 hs.2)).elim
    · have hl1 := (accessRule _).runs h1.runs hacc
      have hno2 : u ∈ p.ids → ∀ l ∈ L2, ¬ P.StartW u l := fun hu l hl hs => by
        obtain ⟨q, hq, huq⟩ := (accessRule _).runs h2 (seqAll_accesses hok.2.2 hb.2) l hl hs.1 u
          hs.2
        exact hok.2.1 u hu q hq huq
      by_cases hu : u ∈ p.ids
      · cases p with
        | code s => simp [Phase.ids] at hu
        | block B =>
          by_cases hm1 : m < L1.length
          · refine Or.inr (Or.inl ⟨B, hu, hok.1, fun l hl' => ?_⟩)
            rw [List.getElem?_append_left hm1] at hl'
            exact hl1 l (List.mem_of_getElem? hl')
          · by_cases hw : ∃ i, ∃ hi : i < L1.length, P.StartW u L1[i]
            · refine Or.inr (Or.inr ⟨B, 0, L1, regs, regs', hok.1, hb.1, h1, fun i hi => ?_,
                by omega, hw⟩)
              rw [Nat.zero_add, List.getElem?_append_left hi, List.getElem?_eq_getElem hi]
            · refine Or.inl fun m' l hl' hs => ?_
              by_cases hm' : m' < L1.length
              · rw [List.getElem?_append_left hm', List.getElem?_eq_some_iff] at hl'
                obtain ⟨hlt, rfl⟩ := hl'
                exact (hw ⟨m', hlt, hs⟩).elim
              · rw [List.getElem?_append_right (by omega)] at hl'
                exact (hno2 hu l (List.mem_of_getElem? hl') hs).elim
      · have hno1 : ∀ l ∈ L1, ¬ P.StartW u l := fun l hl hs =>
          hu ((hl1 l hl).ids_of_start hs.1 hs.2)
        have hbefore : ∀ m' l, (L1 ++ L2)[m']? = some l → P.StartW u l → L1.length ≤ m' := by
          intro m' l hl' hs
          refine Nat.le_of_not_lt fun hm' => ?_
          rw [List.getElem?_append_left hm'] at hl'
          exact hno1 l (List.mem_of_getElem? hl') hs
        by_cases hm1 : m < L1.length
        · exact Or.inl fun m' l hl' hs => by
            have := hbefore m' l hl' hs
            omega
        · simp only [List.length_append] at hm
          rcases ih hok.2.2 hb.2 h2 (m := m - L1.length) (by omega) with
            hbef | ⟨B, hu', hB, hin⟩ | ⟨B, n, L', r1, r2, hB, hBb, hs, hlab, hn, hw⟩
          · refine Or.inl fun m' l hl' hs => ?_
            have hle := hbefore m' l hl' hs
            rw [List.getElem?_append_right hle] at hl'
            have := hbef _ l hl' hs
            omega
          · refine Or.inr (Or.inl ⟨B, hu', hB, fun l hl' => hin l ?_⟩)
            rwa [List.getElem?_append_right (by omega)] at hl'
          · refine Or.inr (Or.inr ⟨B, L1.length + n, L', r1, r2, hB, hBb, hs, fun i hi => ?_,
              by omega, hw⟩)
            rw [List.getElem?_append_right (by omega),
              show L1.length + n + i - L1.length = n + i by omega]
            exact hlab i hi

/-! ## The executions of a program that follows the rule -/

variable {ps : List Phase} {X : Execution}

theorem not_compat {a b : X.E} (hc : X.Conflict a b) (hat : ¬ (X.IsAtomic a ∧ X.IsAtomic b)) :
    ¬ Compat (X.kind a) (X.order a) (X.loc a) (X.kind b) (X.order b) (X.loc b) := by
  obtain ⟨-, ha, hb, hl, hw⟩ := hc
  rintro (h | h | h | ⟨h1, h2⟩ | ⟨h1, h2⟩)
  · simp [Execution.IsAccess, Execution.IsRead, Execution.IsWrite, h] at ha
  · simp [Execution.IsAccess, Execution.IsRead, Execution.IsWrite, h] at hb
  · exact h hl
  · simp [Execution.IsWrite, h1, h2] at hw
  · exact hat ⟨h1, h2⟩

/-- An access to a place from `flags` up is atomic. -/
theorem atomic_of_flags (hF : P.Footprints ps) (r : P.toProg.Run X) {e : X.E}
    (he : X.IsAccess e) (hx : P.flags ≤ X.loc e) : X.IsAtomic e := by
  rcases access hF.ok r e with hk | hlt | ⟨hsc, -⟩
  · simp [Execution.IsAccess, Execution.IsRead, Execution.IsWrite, hk] at he
  · omega
  · unfold Execution.IsAtomic
    rw [hsc]
    nofun

/-- An access of task `t` to a place below `flags` follows each rule that the code of the task
follows. -/
theorem task_access (hF : P.Footprints ps) (r : P.toProg.Run X) {e : X.E} {t : Nat}
    (he : X.thread e = t + 1) (hx : X.loc e < P.flags) {q : Kind → Order → Nat → Prop}
    (hq : (P.task t).Accesses q) : q (X.kind e) (X.order e) (X.loc e) := by
  have htc := task_lt r he
  have hc : (P.toProg.code (X.thread e)).Accesses fun k o x => x < P.flags → q k o x := by
    rw [he, code_succ, taskCode_of_lt htc, lower_noTasks (hF.tasks t)]
    refine ⟨fun h => ?_, hq.mono (q := fun k o x => x < P.flags → q k o x) fun _ _ _ h _ => h,
      fun h => ?_⟩
    · simp only [start] at h
      omega
    · simp only [done] at h
      omega
  have h := (accessRule _).trace (r.trace e) hc
  simp only [accessRule, r.lab_eq e] at h
  exact h hx

/-- A task does not write a `start` flag. -/
theorem task_not_start (hF : P.Footprints ps) (r : P.toProg.Run X) {e : X.E} {t u : Nat}
    (he : X.thread e = t + 1) (hw : X.IsWrite e) (hu : u < P.count) : X.loc e ≠ P.start u := by
  have htc := task_lt r he
  have hc : (P.toProg.code (X.thread e)).Accesses fun k _ x => k = .write → x ≠ P.start u := by
    rw [he, code_succ, taskCode_of_lt htc, lower_noTasks (hF.tasks t)]
    refine ⟨nofun, (code_accesses (hF.ok.task t) (hF.tasks t)).mono fun k _ x h hk hx => ?_,
      fun _ h => ?_⟩
    · rcases h with h | h
      · rw [hk] at h
        cases h
      · simp only [start] at hx
        omega
    · simp only [start, done] at h
      omega
  have h := (accessRule _).trace (r.trace e) hc
  simp only [accessRule, r.lab_eq e] at h
  exact h hw

/-- The accesses of the main code up to `e` are a run of the phases. -/
theorem main_runs (hF : P.Footprints ps) (r : P.toProg.Run X) {e : X.E} (he : X.thread e = 0) :
    Runs (P.lower (seqAll ps)) (fun _ => 0) ((List.range (r.idx e + 1)).map (r.lab 0)) := by
  have h := r.trace e
  rw [he] at h
  rw [← hF.main]
  exact h

theorem lab_zero (r : P.toProg.Run X) {e : X.E} (he : X.thread e = 0) :
    r.lab 0 (r.idx e) = ⟨X.kind e, X.order e, X.loc e, r.val e, r.isRmw e⟩ := by
  rw [← he]
  exact r.lab_eq e

/-- Task `t` read `start t` from a write of the main code. -/
theorem spawn_of (hF : P.Footprints ps) (r : P.toProg.Run X) {b : X.E} {t : Nat}
    (hb : X.thread b = t + 1) :
    ∃ w e, X.thread w = 0 ∧ X.IsWrite w ∧ X.IsSeqCst w ∧ X.loc w = P.start t ∧ X.rf w e ∧
      X.IsSeqCst e ∧ (e = b ∨ X.po e b) := by
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
  have hw0 : X.thread w = 0 := by
    cases hwt : X.thread w with
    | zero => rfl
    | succ s => exact absurd hwl (task_not_start hF r hwt hww htc)
  have hws : X.IsSeqCst w := by
    rcases access hF.ok r w with hk | hlt | ⟨hsc, -⟩
    · rw [show X.kind w = .write from hww] at hk
      cases hk
    · rw [hwl] at hlt
      simp only [start] at hlt
      omega
    · exact hsc
  exact ⟨w, e, hw0, hww, hws, hwl, hw, hes, heb⟩

/-- An event `c` of the main code comes after a run of a block that wrote `start t`. Then each
access of task `t` happens before `c`. -/
theorem after_hb (hF : P.Footprints ps) (r : P.toProg.Run X) {N t : Nat} {c d : X.E}
    (hc : X.thread c = 0) (hcN : r.idx c < N) (hd : X.thread d = t + 1)
    {B : Stmt} {n : Nat} {L' : List Label} {regs regs' : Nat → Int} (hB : (Phase.block B).Ok P)
    (hBb : B.Below P.flags P.count) (hs : Steps (P.lower (.tasksBlock B)) regs L' .skip regs')
    (hlab : ∀ i (hi : i < L'.length), ((List.range N).map (r.lab 0))[n + i]? = some L'[i])
    (hn : n + L'.length ≤ r.idx c) {i : Nat} (hi : i < L'.length) (hw : P.StartW t L'[i]) :
    X.hb d c := by
  have hlab' : ∀ m (hm : m < L'.length), r.lab 0 (n + m) = L'[m] := fun m hm => by
    have := hlab m hm
    rw [getElem?_rangeMap (by omega)] at this
    exact Option.some.inj this
  obtain ⟨w, hwt, hwi⟩ := r.idx_down c (n + i) (by omega)
  have hlw := r.lab_eq w
  rw [hwt, hc, hwi, hlab' i hi] at hlw
  have hww : X.IsWrite w := by
    have := hw.1
    rw [hlw] at this
    exact this
  have hwl : X.loc w = P.start t := by
    have := hw.2
    rw [hlw] at this
    exact this
  exact block_hb hF.ok r hBb hB.2.1 hB.1 hs hlab' (hwt.trans hc) hww hwl (by omega) (by omega) hd
    hc hn

/-- An access `a` of the main code and an access `b` of task `t`, both below `flags`:
happens-before orders them, or they are `Compat`. -/
theorem main_task (hF : P.Footprints ps) (r : P.toProg.Run X) {a b : X.E} {t : Nat}
    (ha : X.thread a = 0) (hb : X.thread b = t + 1) (hxa : X.loc a < P.flags)
    (hxb : X.loc b < P.flags) :
    X.hb a b ∨ X.hb b a ∨
      Compat (X.kind a) (X.order a) (X.loc a) (X.kind b) (X.order b) (X.loc b) := by
  have htc := task_lt r hb
  have hga := getElem?_rangeMap (f := r.lab 0) (Nat.lt_succ_self (r.idx a))
  rw [lab_zero r ha] at hga
  rcases status hF.phases (hF.main ▸ hF.ok.main) (main_runs hF r ha) t
      (m := r.idx a) (by simp) with
    hbef | ⟨B, hu, -, hin⟩ | ⟨B, n, L', r1, r2, hB, hBb, hs, hlab, hn, i, hi, hw⟩
  · -- `a` comes before each `spawn` of task `t`.
    refine Or.inl (fork_hb hF.ok r hb fun w hww hwl => ?_)
    have hw0 : X.thread w = 0 := by
      cases hwt : X.thread w with
      | zero => rfl
      | succ s => exact absurd hwl (task_not_start hF r hwt hww htc)
    refine (r.po_iff a w).2 ⟨ha.trans hw0.symm, ?_⟩
    rcases Nat.lt_or_ge (r.idx w) (r.idx a + 1) with hlt | hge
    · have hgw := getElem?_rangeMap (f := r.lab 0) hlt
      rw [lab_zero r hw0] at hgw
      exact hbef _ _ hgw ⟨hww, hwl⟩
    · omega
  · -- `a` is in the block of task `t`.
    exact Or.inr (Or.inr (task_access hF r hb hxb ((hin _ hga).1 hxa t hu)))
  · -- `a` comes after the block of task `t`.
    exact Or.inr (Or.inl (after_hb hF r ha (Nat.lt_succ_self _) hb hB hBb hs hlab hn hi hw))

/-- Accesses `a` of task `u` and `b` of task `v`, both below `flags`: happens-before orders them,
or they are `Compat`. -/
theorem task_task (hF : P.Footprints ps) (r : P.toProg.Run X) {a b : X.E} {u v : Nat}
    (ha : X.thread a = u + 1) (hb : X.thread b = v + 1) (huv : u ≠ v) (hxa : X.loc a < P.flags)
    (hxb : X.loc b < P.flags) :
    X.hb a b ∨ X.hb b a ∨
      Compat (X.kind a) (X.order a) (X.loc a) (X.kind b) (X.order b) (X.loc b) ∨
      Compat (X.kind b) (X.order b) (X.loc b) (X.kind a) (X.order a) (X.loc a) := by
  obtain ⟨wu, eu, hwu0, hwuw, hwus, hwul, hwurf, heus, heua⟩ := spawn_of hF r ha
  obtain ⟨wv, ev, hwv0, hwvw, hwvs, hwvl, hwvrf, hevs, hevb⟩ := spawn_of hF r hb
  -- One run of the main code that holds both spawns.
  obtain ⟨w, hw0, hwu, hwv⟩ : ∃ w : X.E, X.thread w = 0 ∧ r.idx wu ≤ r.idx w ∧
      r.idx wv ≤ r.idx w := by
    rcases Nat.le_total (r.idx wu) (r.idx wv) with h | h
    · exact ⟨wv, hwv0, h, Nat.le_refl _⟩
    · exact ⟨wu, hwu0, Nat.le_refl _, h⟩
  have hrun := main_runs hF r hw0
  have hgu := getElem?_rangeMap (f := r.lab 0) (Nat.lt_succ_of_le hwu)
  rw [lab_zero r hwu0] at hgu
  have hgv := getElem?_rangeMap (f := r.lab 0) (Nat.lt_succ_of_le hwv)
  rw [lab_zero r hwv0] at hgv
  have hsu : P.StartW u ⟨X.kind wu, X.order wu, X.loc wu, r.val wu, r.isRmw wu⟩ := ⟨hwuw, hwul⟩
  have hsv : P.StartW v ⟨X.kind wv, X.order wv, X.loc wv, r.val wv, r.isRmw wv⟩ := ⟨hwvw, hwvl⟩
  have hb' := hF.main ▸ hF.ok.main
  have hu_hb : X.hb wu a := X.hb_of_signal (Or.inl rfl) hwuw hwus hwurf heus heua
  have hv_hb : X.hb wv b := X.hb_of_signal (Or.inl rfl) hwvw hwvs hwvrf hevs hevb
  rcases status hF.phases hb' hrun u (m := r.idx wv) (by simp; omega) with
    hbef | ⟨B, hu, hB, hin⟩ | ⟨B, n, L', r1, r2, hB, hBb, hs, hlab, hn, i, hi, hw⟩
  · -- The spawn of `v` comes before each spawn of `u`.
    have hlt : r.idx wv < r.idx wu := hbef _ _ hgu hsu
    rcases status hF.phases hb' hrun v (m := r.idx wu) (by simp; omega) with
      hbef' | ⟨B, hv, hB, hin⟩ | ⟨B, n, L', r1, r2, hB, hBb, hs, hlab, hn, i, hi, hw⟩
    · have := hbef' _ _ hgv hsv
      omega
    · have hu := (hin _ hgu).2 hsu.1 u hsu.2
      exact Or.inr (Or.inr (Or.inr (task_access hF r ha hxa
        (task_access hF r hb hxb (hB.2.2.2 v hv u hu (Ne.symm huv))))))
    · exact Or.inr (Or.inl ((after_hb hF r hwu0 (Nat.lt_succ_of_le hwu) hb hB hBb hs hlab hn hi
        hw).trans hu_hb))
  · -- Both tasks are in one block.
    have hv := (hin _ hgv).2 hsv.1 v hsv.2
    exact Or.inr (Or.inr (Or.inl (task_access hF r hb hxb
      (task_access hF r ha hxa (hB.2.2.2 u hu v hv huv)))))
  · -- The spawn of `v` comes after the block of `u`.
    exact Or.inl ((after_hb hF r hwv0 (Nat.lt_succ_of_le hwv) ha hB hBb hs hlab hn hi hw).trans
      hv_hb)

/-- A program that follows the rule has no data race, in each execution. -/
theorem raceFree (hF : P.Footprints ps) (hX : P.toProg.Execs X) : X.RaceFree := by
  obtain ⟨r⟩ := hX
  rintro a b ⟨hc, hthr, hab, hba, hat⟩
  have nc := not_compat hc hat
  obtain ⟨hne, hacc_a, hacc_b, hl, hw⟩ := hc
  have nc' := not_compat ⟨hne.symm, hacc_b, hacc_a, hl.symm, hw.symm⟩ fun h => hat ⟨h.2, h.1⟩
  have hxa : X.loc a < P.flags := Nat.lt_of_not_le fun hle =>
    hat ⟨atomic_of_flags hF r hacc_a hle, atomic_of_flags hF r hacc_b (hl ▸ hle)⟩
  have hxb : X.loc b < P.flags := hl ▸ hxa
  cases hta : X.thread a with
  | zero =>
    cases htb : X.thread b with
    | zero => exact hthr (hta.trans htb.symm)
    | succ t =>
      rcases main_task hF r hta htb hxa hxb with h | h | h
      · exact hab h
      · exact hba h
      · exact nc h
  | succ u =>
    cases htb : X.thread b with
    | zero =>
      rcases main_task hF r htb hta hxb hxa with h | h | h
      · exact hba h
      · exact hab h
      · exact nc' h
    | succ v =>
      have huv : u ≠ v := fun h => hthr (by rw [hta, htb, h])
      rcases task_task hF r hta htb huv hxa hxb with h | h | h | h
      · exact hab h
      · exact hba h
      · exact nc h
      · exact nc' h

/-- DRF-SC from the rule: for a program that follows it, each finite execution that the model
allows is an interleaving. -/
theorem drf_sc_of_footprints (hF : P.Footprints ps) (hm : P.main.SeqCstOnly)
    (ht : ∀ t, (P.task t).SeqCstOnly) (hX : P.toProg.Execs X) (wf : X.WellFormed)
    (hfin : X.Finite) (hc : X.Consistent) : X.SeqConsistent :=
  P.drf_sc hm ht (fun _ hY _ _ => raceFree hF hY) hX wf hfin hc

end TaskProg

end Vx
