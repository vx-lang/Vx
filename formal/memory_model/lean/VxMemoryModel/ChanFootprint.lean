import VxMemoryModel.Footprint
import VxMemoryModel.Recv
import VxMemoryModel.Cancel
import VxMemoryModel.ScIsRc11

/-!
# Race freedom from footprints, for programs with channels

The rule of `Footprint.lean` for a program with channels and locks. It speaks of the places that
the code of the program uses, below `places`. `send`, `recv`, `drop` and `locked` use the places
of the channels and the locks, and those never race (`chan_no_race`). So the rule needs only the
accesses of the code `b` of `Tasks.lean` in the program.

The lowered program follows the rule of `Footprint.lean` for the places below `places`. So no two
accesses to those places race. The places from `places` to `top` belong to the channels and the
locks. The places from `top` up are the flags of the tasks, and they have only `sc` accesses.

The rule does not use the lock yet: two tasks of a block cannot share a place, even in `locked`
blocks of one lock. The program has no failures.
-/

namespace Vx

/-- Each access of the code of `Tasks.lean` in `s` follows `q`. -/
def CStmt.Accesses (q : Kind → Order → Nat → Prop) : CStmt → Prop
  | .base s => s.Accesses q
  | .send .. | .recv .. | .drop _ | .fail _ | .checkCancel | .result _ => True
  | .locked _ s | .loop _ s | .tasksBlock s => s.Accesses q
  | .seq s t | .ite _ s t => s.Accesses q ∧ t.Accesses q

/-- No access of `s` can race with an access of `t`. -/
def CStmt.Apart (s t : CStmt) : Prop := s.Accesses fun k o x => t.Accesses (Compat k o x)

/-- `s` starts no task and joins no task. -/
def CStmt.NoTasks : CStmt → Prop
  | .base s => s.NoTasks
  | .tasksBlock _ => False
  | .locked _ s | .loop _ s => s.NoTasks
  | .seq s t | .ite _ s t => s.NoTasks ∧ t.NoTasks
  | _ => True

/-- `s` holds no `tasks` block. -/
def CStmt.NoBlocks : CStmt → Prop
  | .base s => s.NoBlocks
  | .tasksBlock _ => False
  | .locked _ s | .loop _ s => s.NoBlocks
  | .seq s t | .ite _ s t => s.NoBlocks ∧ t.NoBlocks
  | _ => True

/-- `s` uses only the registers below `m`. -/
def CStmt.RegsBelow (m : Nat) : CStmt → Prop
  | .base s => s.RegsBelow m
  | .send _ e => e.RegsBelow m
  | .recv r s _ => r < m ∧ s < m
  | .result r => r < m
  | .locked _ s | .tasksBlock s => s.RegsBelow m
  | .loop c s => c.RegsBelow m ∧ s.RegsBelow m
  | .seq s t => s.RegsBelow m ∧ t.RegsBelow m
  | .ite c s t => c.RegsBelow m ∧ s.RegsBelow m ∧ t.RegsBelow m
  | .drop _ | .fail _ | .checkCancel => True

theorem Stmt.Below.mono {n n' tasks : Nat} {s : Stmt} (h : s.Below n tasks) (hn : n ≤ n') :
    s.Below n' tasks := by
  induction s with
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | loop _ _ ih | tasksBlock _ ih => exact ih h
  | load | store | fetchAdd | cas | rmwStore | wait | loadAt | storeAt | waitAt =>
    simp only [Stmt.Below] at *
    omega
  | _ => exact h

namespace ChanProg

/-- A part of the main code: code that starts no task, or a `tasks` block. -/
inductive CPhase where
  | code (s : CStmt)
  | block (s : CStmt)

def CPhase.stmt : CPhase → CStmt
  | .code s => s
  | .block s => .tasksBlock s

/-- The phases, one after the other. -/
def cseqAll : List CPhase → CStmt
  | [] => .base .skip
  | p :: ps => .seq p.stmt (cseqAll ps)

variable (P : ChanProg)

/-- The phase of the lowered program. -/
def CPhase.lower : CPhase → TaskProg.Phase
  | .code s => .code (P.lowerC s)
  | .block s => .block (P.lowerC s)

/-- The rule for one phase. The code of a block and each task of the block are apart, and two
tasks of the block are apart. The code of a block uses only the registers below `tmp + 3`, since
the block keeps its marks from there. -/
def CPhase.Ok : CPhase → Prop
  | .code s => s.NoTasks
  | .block s => s.NoBlocks ∧ s.RegsBelow (P.tmp + 3) ∧
      (∀ u ∈ (P.lowerC s).spawnIds, s.Apart (P.task u)) ∧
      ∀ u ∈ (P.lowerC s).spawnIds, ∀ v ∈ (P.lowerC s).spawnIds, u ≠ v → (P.task u).Apart (P.task v)

/-- Each phase follows its rule, and no two phases start the same task. -/
def CPhasesOk : List CPhase → Prop
  | [] => True
  | p :: ps => p.Ok P ∧ (∀ u ∈ (p.lower P).ids, ∀ q ∈ ps, u ∉ (q.lower P).ids) ∧ CPhasesOk ps

/-- The rule on the code: the main code is the phases `ps`, the tasks start no tasks, and the
program has no failures. -/
structure CFootprints (ps : List CPhase) : Prop where
  ok : P.Ok
  errors : P.errors = false
  main : P.main = cseqAll ps
  phases : P.CPhasesOk ps
  tasks : ∀ t, (P.task t).NoTasks

variable {P}

/-! ## The lowered program follows the rule of `Footprint.lean` -/

theorem lowerC_cseqAll (ps : List CPhase) :
    P.lowerC (cseqAll ps) = TaskProg.seqAll (ps.map (CPhase.lower P)) := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    cases p <;> simp [cseqAll, CPhase.stmt, lowerC, CPhase.lower, TaskProg.seqAll,
      TaskProg.Phase.stmt, ih]

theorem not_fails (he : P.errors = false) {s : CStmt} (hs : P.StmtOk s) : ¬ s.Fails := by
  induction s with
  | fail => exact fun _ => by simp [StmtOk, he] at hs
  | checkCancel | result => exact fun _ => by simp [StmtOk, he] at hs
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact fun h => h.elim (ih1 hs.1) (ih2 hs.2)
  | loop _ _ ih | tasksBlock _ ih => exact ih hs
  | locked _ _ ih => exact ih hs.2.2
  | base | send | recv | drop => exact id

theorem lowerC_below (he : P.errors = false) {s : CStmt} (hs : P.StmtOk s) :
    (P.lowerC s).Below P.top P.count := by
  have hpt : P.places ≤ P.top := by unfold top; omega
  induction s with
  | base => exact Stmt.Below.mono hs hpt
  | send c =>
    have := chan_end (P := P) (c := c) hs
    simp only [lowerC, opCode, rest, fin, ctr, Stmt.Below, cnt, Ctr.idx, sig, sigLen, data,
      Op.other, size_eq] at *
    refine ⟨?_, ?_, ?_, ?_⟩ <;> omega
  | recv _ _ c =>
    have := chan_end (P := P) (c := c) hs.1
    simp only [lowerC, recvCode, recvRest, poll, polls, loadFull, recvEnd, result, fin, ctr,
      drp, Stmt.Below, cnt, Ctr.idx, sig, sigLen, data, size_eq] at *
    repeat' apply And.intro
    all_goals first | trivial | omega
  | drop c =>
    have := chan_end (P := P) (c := c) hs
    simp only [lowerC, Stmt.Below, cnt, Ctr.idx, size_eq] at *
    omega
  | locked m s ih =>
    have h1 := (cnt_inside (P := P) (q := .next) hs.1).2
    have h2 := (cnt_inside (P := P) (q := .serving) hs.1).2
    exact ⟨⟨h1, h2, h2⟩, ih hs.2.2, h2⟩
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 hs.1, ih2 hs.2⟩
  | loop _ _ ih | tasksBlock _ ih => exact ih hs
  | fail | checkCancel | result => simp [StmtOk, he] at hs

theorem stores_below {n tasks : Nat} : ∀ {ps : List (Nat × Int)}, (∀ q ∈ ps, q.1 < n) →
    (stores ps).Below n tasks
  | [], _ => trivial
  | _ :: _, h => ⟨h _ List.mem_cons_self, stores_below fun q hq => h q (List.mem_cons_of_mem _ hq)⟩

theorem stores_noTasks : ∀ {ps : List (Nat × Int)}, (stores ps).NoTasks
  | [] => trivial
  | _ :: _ => ⟨trivial, stores_noTasks⟩

/-- A program with no failures follows the rule of `Tasks.lean` for the places of its code. -/
theorem toTaskProg_ok (hP : P.Ok) (he : P.errors = false) : P.toTaskProg.Ok where
  main := ⟨stores_below fun q hq => initFirst_lt hP q (by simpa [initList, he] using hq),
    lowerC_below he hP.main⟩
  task t := lowerC_below he (hP.task t)

theorem lowerC_noTasks {s : CStmt} (h : s.NoTasks) : (P.lowerC s).NoTasks := by
  induction s with
  | base => exact h
  | tasksBlock => exact h.elim
  | locked _ _ ih => exact ⟨⟨trivial, trivial, trivial⟩, ih h, trivial⟩
  | loop _ _ ih => exact ih h
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | _ => simp [lowerC, Stmt.NoTasks]

theorem lowerC_noBlocks {s : CStmt} (h : s.NoBlocks) : (P.lowerC s).NoBlocks := by
  induction s with
  | base => exact h
  | tasksBlock => exact h.elim
  | locked _ _ ih => exact ⟨⟨trivial, trivial, trivial⟩, ih h, trivial⟩
  | loop _ _ ih => exact ih h
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | _ => simp [lowerC, Stmt.NoBlocks]

theorem lowerC_regsBelow {s : CStmt} (h : s.RegsBelow (P.tmp + 3)) :
    (P.lowerC s).RegsBelow (P.tmp + 3) := by
  induction s with
  | base => exact h
  | locked _ _ ih =>
    refine ⟨?_, ih h, ?_⟩ <;> simp [Stmt.RegsBelow, Expr.RegsBelow]
  | tasksBlock _ ih => exact ih h
  | loop _ _ ih => exact ⟨h.1, ih h.2⟩
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨h.1, ih1 h.2.1, ih2 h.2.2⟩
  | send => simpa [lowerC, Stmt.RegsBelow, Expr.RegsBelow, CStmt.RegsBelow] using h
  | recv =>
    simp only [CStmt.RegsBelow] at h
    simp [lowerC, Stmt.RegsBelow, Expr.RegsBelow]
    omega
  | result => simpa [lowerC, Stmt.RegsBelow, CStmt.RegsBelow] using h
  | _ => simp [lowerC, Stmt.RegsBelow, Expr.RegsBelow]

/-- The lowered code makes the accesses of `s` below `places`. Its other accesses are to the
places of the channels and the locks. -/
theorem lowerC_accesses (he : P.errors = false) {q : Kind → Order → Nat → Prop} {s : CStmt}
    (hs : P.StmtOk s) (h : s.Accesses q) :
    (P.lowerC s).Accesses fun k o x => x < P.places → q k o x := by
  have out : ∀ {k o x}, P.places ≤ x → x < P.places → q k o x := fun h1 h2 => by omega
  induction s with
  | base s =>
    show s.Accesses _
    exact h.mono (q := fun k o x => x < P.places → q k o x) fun _ _ _ h _ => h
  | send c e =>
    have o1 := (cnt_inside (P := P) (q := .op .send) hs).1
    exact ⟨⟨out o1, out o1⟩, fun _ hj => out (sig_inside hs hj).1,
      fun _ hj => out (data_inside hs hj).1, fun _ hj => out (sig_inside hs hj).1⟩
  | recv r s c =>
    have o1 := (cnt_inside (P := P) (q := .op .recv) hs.1).1
    have o2 := (cnt_inside (P := P) (q := .drop) hs.1).1
    exact ⟨⟨out o1, out o1⟩, ⟨⟨out o2, fun _ hj => out (sig_inside hs.1 hj).1⟩, out o2,
        fun _ hj => out (sig_inside hs.1 hj).1⟩,
      ⟨⟨fun _ hj => out (data_inside hs.1 hj).1, trivial⟩, trivial⟩,
      fun _ hj => out (sig_inside hs.1 hj).1⟩
  | drop c =>
    have o2 := (cnt_inside (P := P) (q := .drop) hs).1
    exact ⟨out o2, out o2⟩
  | locked m s ih =>
    have h1 := (cnt_inside (P := P) (q := .next) hs.1).1
    have h2 := (cnt_inside (P := P) (q := .serving) hs.1).1
    exact ⟨⟨⟨out h1, out h1⟩, out h2, out h2⟩, ih hs.2.2 h, out h2, out h2⟩
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 hs.1 h.1, ih2 hs.2 h.2⟩
  | loop _ _ ih => exact ih hs h
  | tasksBlock => trivial
  | fail | checkCancel | result => simp [StmtOk, he] at hs

theorem apart_lower (he : P.errors = false) {s t : CStmt} (hs : P.StmtOk s) (ht : P.StmtOk t)
    (h : s.Apart t) : (P.lowerC s).Apart P.places (P.lowerC t) :=
  (lowerC_accesses he hs h).mono fun _ _ x h hx =>
    (lowerC_accesses he ht (h hx)).mono fun _ _ x' h' => by
      by_cases hx' : x' < P.places
      · exact h' hx'
      · exact Or.inr (Or.inr (Or.inl (by omega)))

theorem cphase_ok (he : P.errors = false) (htask : ∀ t, P.StmtOk (P.task t)) {p : CPhase}
    (hp : p.Ok P) (hs : P.StmtOk p.stmt) : (p.lower P).Ok P.toTaskProg P.places := by
  cases p with
  | code s => exact lowerC_noTasks hp
  | block s =>
    obtain ⟨hn, hr, hap, hpair⟩ := hp
    exact ⟨lowerC_noBlocks hn, lowerC_regsBelow hr,
      fun u hu => apart_lower he hs (htask u) (hap u hu),
      fun u hu v hv huv => apart_lower he (htask u) (htask v) (hpair u hu v hv huv)⟩

theorem cphasesOk_lower (he : P.errors = false) (htask : ∀ t, P.StmtOk (P.task t)) :
    ∀ {ps : List CPhase}, P.CPhasesOk ps → P.StmtOk (cseqAll ps) →
      P.toTaskProg.PhasesOk P.places (ps.map (CPhase.lower P))
  | [], _, _ => trivial
  | _ :: _, ⟨hp, hids, hrest⟩, ⟨hs, hss⟩ => by
    refine ⟨cphase_ok he htask hp hs, fun u hu q hq => ?_, cphasesOk_lower he htask hrest hss⟩
    obtain ⟨q', hq', rfl⟩ := List.mem_map.1 hq
    exact hids u hu q' hq'

/-- The lowered program follows the rule of `Footprint.lean` for the places below `places`. -/
theorem toFootprints {ps : List CPhase} (hF : P.CFootprints ps) :
    P.toTaskProg.Footprints P.places (.code (stores P.initList) :: ps.map (CPhase.lower P)) where
  ok := toTaskProg_ok hF.ok hF.errors
  le_flags := by
    show P.places ≤ P.top
    unfold top
    omega
  main := by
    show Stmt.seq _ (P.lowerC P.main) = _
    rw [hF.main, lowerC_cseqAll]
    rfl
  phases := ⟨stores_noTasks, fun _ hu => by simp [TaskProg.Phase.ids] at hu,
    cphasesOk_lower hF.errors hF.ok.task hF.phases (hF.main ▸ hF.ok.main)⟩
  tasks t := lowerC_noTasks (hF.tasks t)

/-! ## The executions of a program that follows the rule -/

/-- A program with channels that follows the rule has no data race, in each execution whose
`co` orders the writes to each place. -/
theorem raceFree {ps : List CPhase} (hF : P.CFootprints ps) {X : Execution}
    (hX : P.toTaskProg.toProg.Execs X) (wf : X.WellFormed) (hcons : X.Consistent)
    (ho : X.CoOrder) (hfin : X.Finite) : X.RaceFree := by
  intro a b hr
  obtain ⟨run⟩ := hX
  have hT := toFootprints hF
  by_cases h1 : X.loc a < P.places
  · exact TaskProg.no_race_below hT ⟨run⟩ hr h1
  by_cases h2 : X.loc a < P.top
  · exact chan_no_race run hF.ok wf hcons ho hfin hr (by omega) h2
  · obtain ⟨⟨-, ha, hb, hl, -⟩, -, -, -, hat⟩ := hr
    exact hat ⟨TaskProg.atomic_of_flags hT run ha (by show P.top ≤ _; omega),
      TaskProg.atomic_of_flags hT run hb (by show P.top ≤ _; omega)⟩

/-- DRF-SC from the rule: for a program with channels that follows it, each finite execution
that the model allows, with a `co` that orders the writes to each place, is an interleaving. -/
theorem drf_sc_of_footprints {ps : List CPhase} (hF : P.CFootprints ps)
    (hm : ∀ b, P.main.Bases b → b.SeqCstOnly) (ht : ∀ t b, (P.task t).Bases b → b.SeqCstOnly)
    {X : Execution} (hX : P.toTaskProg.toProg.Execs X) (wf : X.WellFormed) (hfin : X.Finite)
    (ho : X.CoOrder) (hc : X.Consistent) : X.SeqConsistent :=
  P.drf_sc_co hm ht (not_fails hF.errors hF.ok.main) (fun t => not_fails hF.errors (hF.ok.task t))
    (fun _ hY wfY hsc hfinY hoY =>
      raceFree hF hY wfY (Execution.consistent_of_seqConsistent wfY hsc) hoY hfinY)
    hX wf hfin ho hc

end ChanProg

end Vx
