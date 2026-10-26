import VxMemoryModel.Footprint
import VxMemoryModel.Recv
import VxMemoryModel.Cancel

/-!
# Race freedom from footprints, for programs with channels

The rule of `Footprint.lean` for a program with channels and locks. It speaks of the places that
the code of the program uses, below `places`. `send`, `recv`, `drop` and `locked` use the places
of the channels and the locks, and those never race (`chan_no_race`). So the rule needs only the
accesses of the code `b` of `Tasks.lean` in the program.

The lowered program follows the rule of `Footprint.lean` for the places below `places` that have
no lock. So no two accesses to those places race. The places from `places` to `top` belong to the
channels and the locks. The failure slot and the cancel flag come next, and each access to them is
atomic (`fail_atomic`). The flags of the tasks come last, and they have only `sc` accesses.

The rule can give a place of the code a lock. The code then accesses that place only in
`locked` blocks of the lock, and two tasks of a block can share it. `LockFootprint.lean` shows
that such places never race.
-/

namespace Vx

/-- Each access of the code of `Tasks.lean` in `s` follows `q`. -/
def CStmt.Accesses (q : Kind → Order → Nat → Prop) : CStmt → Prop
  | .base s => s.Accesses q
  | .send .. | .recv .. | .drop _ | .fail _ | .checkCancel | .result _ => True
  | .locked _ s | .loop _ s | .tasksBlock s => s.Accesses q
  | .seq s t | .ite _ s t => s.Accesses q ∧ t.Accesses q

/-- No access of `s` can race with an access of `t`, at a place that `g` gives no lock. -/
def CStmt.Apart (g : Nat → Option Nat) (s t : CStmt) : Prop :=
  s.Accesses fun k o x => g x = none → t.Accesses (Compat k o x)

/-- Each access of the code of `Tasks.lean` in `s`, to a place that `g` gives the lock `m`, is in a
`locked` block of `m`. `held` lists the locks of the blocks around `s`. -/
def CStmt.Guarded (g : Nat → Option Nat) (held : List Nat) : CStmt → Prop
  | .base s => s.Accesses fun k _ x => k = .fence ∨ ∀ m, g x = some m → m ∈ held
  | .locked m s => s.Guarded g (m :: held)
  | .loop _ s | .tasksBlock s => s.Guarded g held
  | .seq s t | .ite _ s t => s.Guarded g held ∧ t.Guarded g held
  | _ => True

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

/-- A part of the code of a thread: code that starts no task, or a `tasks` block. With `some e`,
the block runs again in a loop while `e` holds. -/
inductive CPhase where
  | code (s : CStmt)
  | block (s : CStmt) (rep : Option Expr)

def CPhase.stmt : CPhase → CStmt
  | .code s => s
  | .block s none => .tasksBlock s
  | .block s (some e) => .loop e (.seq (.base .skip) (.seq (.tasksBlock s) (.base .skip)))

/-- The phases, one after the other. -/
def cseqAll : List CPhase → CStmt
  | [] => .base .skip
  | p :: ps => .seq p.stmt (cseqAll ps)

/-- The phases of thread `q`: `ps` for the main code, and `tps t` for task `t`. -/
def cthr (ps : List CPhase) (tps : Nat → List CPhase) : Nat → List CPhase
  | 0 => ps
  | t + 1 => tps t

variable (P : ChanProg) (g : Nat → Option Nat) (tps : Nat → List CPhase)

/-- A loop around a block with no code around the block. -/
def skipLoop (e : Expr) : TaskProg.Loop := ⟨e, .skip, .skip, trivial, trivial⟩

/-- The phase of the lowered program. -/
def CPhase.lower : CPhase → TaskProg.Phase
  | .code s => .code (P.lowerC s)
  | .block s rep => .block (P.lowerC s) (rep.map skipLoop)

/-- The phases of each task, lowered. -/
def lowerPhases (t : Nat) : List TaskProg.Phase := (tps t).map (CPhase.lower P)

/-- The rule for one phase, at the places that `g` gives no lock. The code of a block and each
task of the block are apart, and two tasks of the block are apart. The descendants of a task
count as the task. The code of a block uses only the registers below `tmp + 3`, since the block
keeps its marks from there. -/
def CPhase.Ok : CPhase → Prop
  | .code s => s.NoTasks
  | .block s _ => s.NoBlocks ∧ s.RegsBelow (P.tmp + 3) ∧
      (∀ u ∈ (P.lowerC s).spawnIds, ∀ w, TaskProg.Desc (P.lowerPhases tps) u w →
        s.Apart g (P.task w)) ∧
      ∀ u ∈ (P.lowerC s).spawnIds, ∀ v ∈ (P.lowerC s).spawnIds, u ≠ v → ∀ w w',
        TaskProg.Desc (P.lowerPhases tps) u w → TaskProg.Desc (P.lowerPhases tps) v w' →
          (P.task w).Apart g (P.task w')

/-- Each phase follows its rule, and no two phases start the same task. -/
def CPhasesOk : List CPhase → Prop
  | [] => True
  | p :: ps => p.Ok P g tps ∧ (∀ u ∈ (p.lower P).ids, ∀ q ∈ ps, u ∉ (q.lower P).ids) ∧
      CPhasesOk ps

/-- The rule on the code. The main code is the phases `ps`. Task `t` is the phases `tps t`, or
code that starts no task. One thread starts each task, and a task starts only tasks with larger
ids. `g` gives some places of the code a lock. The code of the tasks and of the blocks of the main
code accesses such a place only in a `locked` block of its lock. The other main code runs while
no task runs, so it needs no lock. -/
structure CFootprints (ps : List CPhase) : Prop where
  ok : P.Ok
  main : P.main = cseqAll ps
  phases : P.CPhasesOk g tps ps
  tasks : ∀ t, P.task t = cseqAll (tps t) ∨ (P.task t).NoTasks ∧ tps t = []
  task_phases : ∀ t, P.CPhasesOk g tps (tps t)
  one_parent : ∀ q q' v, (∃ p ∈ cthr ps tps q, v ∈ (p.lower P).ids) →
    (∃ p ∈ cthr ps tps q', v ∈ (p.lower P).ids) → q = q'
  order : ∀ t v, (∃ p ∈ tps t, v ∈ (p.lower P).ids) → t < v
  guard : ∀ x m, g x = some m → x < P.places ∧ m < P.locks
  guarded_blocks : ∀ B rep, CPhase.block B rep ∈ ps → B.Guarded g []
  guarded_task : ∀ t, (P.task t).Guarded g []

variable {P g tps}

@[simp] theorem lowerPhases_nil : P.lowerPhases (fun _ => []) = fun _ => [] := rfl

/-- When no task starts a task, only the main code starts tasks. -/
theorem one_parent_nil (ps : List CPhase) :
    ∀ q q' v, (∃ p ∈ cthr ps (fun _ => []) q, v ∈ (p.lower P).ids) →
      (∃ p ∈ cthr ps (fun _ => []) q', v ∈ (p.lower P).ids) → q = q'
  | 0, 0, _, _, _ => rfl
  | _ + 1, _, _, ⟨_, h, _⟩, _ | _, _ + 1, _, _, ⟨_, h, _⟩ => by simp [cthr] at h

/-! ## The lowered program follows the rule of `Footprint.lean` -/

theorem lowerC_cseqAll (ps : List CPhase) :
    P.lowerC (cseqAll ps) = TaskProg.seqAll (ps.map (CPhase.lower P)) := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    rcases p with _ | ⟨_, _ | _⟩ <;> simp [cseqAll, CPhase.stmt, lowerC, CPhase.lower,
      TaskProg.seqAll, TaskProg.Phase.stmt, skipLoop, ih]

theorem not_fails (he : P.errors = false) {s : CStmt} (hs : P.StmtOk s) : ¬ s.Fails := by
  induction s with
  | fail => exact fun _ => by simp [StmtOk, he] at hs
  | checkCancel | result => exact fun _ => by simp [StmtOk, he] at hs
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact fun h => h.elim (ih1 hs.1) (ih2 hs.2)
  | loop _ _ ih | tasksBlock _ ih => exact ih hs
  | locked _ _ ih => exact ih hs.2.2
  | base | send | recv | drop => exact id

theorem lowerC_below {s : CStmt} (hs : P.StmtOk s) :
    (P.lowerC s).Below (P.flag + 1) P.count := by
  have hpt : P.places ≤ P.flag + 1 := by unfold flag slot top; omega
  have hf : P.top < P.flag + 1 := by unfold flag slot; omega
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
    have lt : ∀ {x}, x < P.top → x < P.flag + 1 := fun h => Nat.lt_trans h hf
    exact ⟨⟨lt h1, lt h2, lt h2⟩, ih hs.2.2, lt h2⟩
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 hs.1, ih2 hs.2⟩
  | loop _ _ ih | tasksBlock _ ih => exact ih hs
  | fail | checkCancel | result =>
    simp only [lowerC, failCode, checkCode, checkTail, raise, Stmt.Below, flag, slot] at *
    repeat' apply And.intro
    all_goals first | trivial | omega

theorem stores_below {n tasks : Nat} : ∀ {ps : List (Nat × Int)}, (∀ q ∈ ps, q.1 < n) →
    (stores ps).Below n tasks
  | [], _ => trivial
  | _ :: _, h => ⟨h _ List.mem_cons_self, stores_below fun q hq => h q (List.mem_cons_of_mem _ hq)⟩

theorem stores_noTasks : ∀ {ps : List (Nat × Int)}, (stores ps).NoTasks
  | [] => trivial
  | _ :: _ => ⟨trivial, stores_noTasks⟩

/-- The writes at the start stay below the flags of the tasks. -/
theorem initList_lt (hP : P.Ok) : ∀ q ∈ P.initList, q.1 < P.flag + 1 := by
  intro q hq
  have hf : P.top < P.flag + 1 := by unfold flag slot; omega
  simp only [initList, List.mem_append] at hq
  rcases hq with hq | hq
  · exact Nat.lt_trans (initFirst_lt hP q hq) hf
  · split at hq
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at hq
      rcases hq with rfl | rfl <;> simp only [flag, slot] <;> omega
    · cases hq

/-- The program follows the rule of `Tasks.lean`: the code stays below the flags of the tasks. -/
theorem toTaskProg_ok (hP : P.Ok) : P.toTaskProg.Ok where
  main := ⟨stores_below (initList_lt hP), lowerC_below hP.main⟩
  task t := lowerC_below (hP.task t)

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
theorem lowerC_accesses {q : Kind → Order → Nat → Prop} {s : CStmt}
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
  | tasksBlock _ ih => exact ih hs h
  | fail | checkCancel | result =>
    have hs : P.places ≤ P.slot := by unfold slot top; omega
    have hf : P.places ≤ P.flag := by unfold flag; omega
    first
      | exact ⟨⟨out hs, out hs⟩, out hf, trivial⟩
      | exact ⟨out hf, ⟨⟨out hs, out hs⟩, out hf, trivial⟩, trivial⟩
      | exact out hs

variable (P g) in
/-- The places that the rule of `Footprint.lean` covers: the places of the code that `g` gives no
lock. -/
def Own (x : Nat) : Prop := x < P.places ∧ g x = none

theorem apart_lower {s t : CStmt} (hs : P.StmtOk s) (ht : P.StmtOk t)
    (h : s.Apart g t) : (P.lowerC s).Apart (P.Own g) (P.lowerC t) :=
  (lowerC_accesses hs h).mono fun _ _ x h hx =>
    (lowerC_accesses ht (h hx.1 hx.2)).mono fun _ _ x' h' => by
      by_cases hx' : x' < P.places
      · exact h' hx'
      · exact Or.inr (Or.inr (Or.inl (by have := hx.1; omega)))

theorem cphase_ok (htask : ∀ t, P.StmtOk (P.task t)) {p : CPhase}
    (hp : p.Ok P g tps) (hs : P.StmtOk p.stmt) :
    (p.lower P).Ok P.toTaskProg (P.Own g) (P.lowerPhases tps) := by
  cases p with
  | code s => exact lowerC_noTasks hp
  | block s rep =>
    obtain ⟨hn, hr, hap, hpair⟩ := hp
    replace hs : P.StmtOk s := by
      cases rep with
      | none => exact hs
      | some _ => exact hs.2.1
    have hall := Stmt.allIds_of_noBlocks (lowerC_noBlocks (P := P) hn)
    exact ⟨.of_noBlocks (lowerC_noBlocks hn), lowerC_regsBelow hr,
      fun u hu w hw => apart_lower hs (htask w) (hap u (hall ▸ hu) w hw),
      fun u hu v hv huv _ w w' hw hw' =>
        apart_lower (htask w) (htask w') (hpair u (hall ▸ hu) v (hall ▸ hv) huv w w' hw hw')⟩

theorem cphasesOk_lower (htask : ∀ t, P.StmtOk (P.task t)) :
    ∀ {ps : List CPhase}, P.CPhasesOk g tps ps → P.StmtOk (cseqAll ps) →
      P.toTaskProg.PhasesOk (P.Own g) (P.lowerPhases tps) (ps.map (CPhase.lower P))
  | [], _, _ => trivial
  | _ :: _, ⟨hp, hids, hrest⟩, ⟨hs, hss⟩ => by
    refine ⟨cphase_ok htask hp hs, fun u hu q hq => ?_, cphasesOk_lower htask hrest hss⟩
    obtain ⟨q', hq', rfl⟩ := List.mem_map.1 hq
    exact hids u hu q' hq'

/-- A task that a thread of the lowered program starts, the same thread starts in `cthr`. -/
theorem thr_ids {s : Stmt} {ps : List CPhase} {q v : Nat}
    (h : ∃ p ∈ TaskProg.thr (.code s :: ps.map (CPhase.lower P)) (P.lowerPhases tps) q, v ∈ p.ids) :
    ∃ p ∈ cthr ps tps q, v ∈ (p.lower P).ids := by
  obtain ⟨p, hp, hv⟩ := h
  cases q with
  | zero =>
    rcases List.mem_cons.1 hp with rfl | hp
    · simp [TaskProg.Phase.ids] at hv
    · obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hp
      exact ⟨p', hp', hv⟩
  | succ t =>
    obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hp
    exact ⟨p', hp', hv⟩

/-- The lowered program follows the rule of `Footprint.lean` for the places of `Own`. -/
theorem toFootprints {ps : List CPhase} (hF : P.CFootprints g tps ps) :
    P.toTaskProg.Footprints (P.Own g) (P.lowerPhases tps)
      (.code (stores P.initList) :: ps.map (CPhase.lower P)) where
  ok := toTaskProg_ok hF.ok
  le_flags x hx := by
    show x < P.flag + 1
    have := hx.1
    unfold flag slot top
    omega
  main := by
    show Stmt.seq _ (P.lowerC P.main) = _
    rw [hF.main, lowerC_cseqAll]
    rfl
  phases := ⟨stores_noTasks, fun _ hu => by simp [TaskProg.Phase.ids] at hu,
    cphasesOk_lower hF.ok.task hF.phases (hF.main ▸ hF.ok.main)⟩
  tasks t := by
    rcases hF.tasks t with h | ⟨hn, h⟩
    · left
      show P.lowerC (P.task t) = _
      rw [h, lowerC_cseqAll]
      rfl
    · exact Or.inr ⟨lowerC_noTasks hn, by simp [lowerPhases, h]⟩
  task_phases t := by
    rcases hF.tasks t with h | ⟨-, h⟩
    · exact cphasesOk_lower hF.ok.task (hF.task_phases t) (h ▸ hF.ok.task t)
    · simp [lowerPhases, h, TaskProg.PhasesOk]
  one_parent q q' v h1 h2 := hF.one_parent q q' v (thr_ids h1) (thr_ids h2)
  order t v := fun ⟨p, hp, hv⟩ => by
    obtain ⟨p', hp', rfl⟩ := List.mem_map.1 hp
    exact hF.order t v ⟨p', hp', hv⟩

/-! ## The executions of a program that follows the rule -/

/-- A program with channels that follows the rule has no data race, in each execution whose
`co` orders the writes to each place, if it has none at the places that `g` gives a lock.
`LockFootprint.lean` shows that it has none there. -/
theorem raceFree_of {ps : List CPhase} (hF : P.CFootprints g tps ps) {X : Execution}
    (hX : P.toTaskProg.toProg.Execs X) (wf : X.WellFormed) (hcons : X.Consistent)
    (ho : X.CoOrder) (hfin : X.Finite)
    (hg : ∀ {a b : X.E} {m}, X.DataRace a b → g (X.loc a) = some m → False) : X.RaceFree := by
  intro a b hr
  obtain ⟨run⟩ := hX
  have hT := toFootprints hF
  by_cases h1 : X.loc a < P.places
  · cases hga : g (X.loc a) with
    | none => exact TaskProg.no_race_below hT ⟨run⟩ hr ⟨h1, hga⟩
    | some m => exact hg hr hga
  by_cases h2 : X.loc a < P.top
  · exact chan_no_race run hF.ok wf hcons ho hfin hr (by omega) h2
  obtain ⟨⟨-, ha, hb, hl, -⟩, -, -, -, hat⟩ := hr
  have hk {e : X.E} (he : X.IsAccess e) : X.kind e ≠ .fence := fun h => by
    rcases he with he | he <;> rw [he] at h <;> cases h
  by_cases h3 : X.loc a < P.flag + 1
  · have hx : X.loc a = P.slot ∨ X.loc a = P.flag := by unfold flag slot at *; omega
    exact hat ⟨fail_atomic run hF.ok (hk ha) hx, fail_atomic run hF.ok (hk hb) (hl ▸ hx)⟩
  · exact hat ⟨TaskProg.atomic_of_flags hT run ha (by show P.flag + 1 ≤ _; omega),
      TaskProg.atomic_of_flags hT run hb (by show P.flag + 1 ≤ _; omega)⟩

end ChanProg

end Vx
