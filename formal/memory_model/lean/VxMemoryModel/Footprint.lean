import VxMemoryModel.Blocks

/-!
# Race freedom from footprints

`TaskProg.drf_sc` needs the hypothesis that no SC execution of the program has a data race.
`raceFree` proves that each execution has no data race, from a rule on the code.

The code of each thread is a list of phases. A phase is code that starts no task, or a `tasks`
block. A block can also run again in a loop, with code that starts no task before and after it
(`Loop`). Each pass of the loop joins its tasks, so the rule of one block holds for each pass
(`Runs.loop_pass`). The code around the block runs while no task of the block runs: each write
of a `start` flag before it is in a pass of the block that ended (`block_pos`). A task that starts
no task can also be plain code. The rule for a block has two parts.
Two tasks of the block make no accesses that can race. The code of the block makes no access that
can race with a task of the block. Here the accesses of a task include the accesses of each task
that it starts, at each depth (`Desc`). Code outside the block can access any place: the code
before the block happens before its tasks, and its tasks happen before the code after it. Two
accesses can race only when they touch one place, one of them writes, and one of them is plain.

The tasks make a tree: one thread starts each task, and a task starts only tasks with larger ids.
The proof takes two accesses and looks at where their threads are in the tree. One thread can
start the other, at some depth (`anc`). Or two tasks of one block can start them (`sib`).

The code has no channels and no locks.
-/

namespace Vx

/-- Two accesses that can never make a data race: one is a fence, they touch two places, both
read, or both are atomic. -/
def Compat (k : Kind) (o : Order) (x : Nat) (k' : Kind) (o' : Order) (x' : Nat) : Prop :=
  k = .fence ∨ k' = .fence ∨ x ≠ x' ∨ (k = .read ∧ k' = .read) ∨ (o ≠ .plain ∧ o' ≠ .plain)

/-- No access of `s` to a place of `own` can race with an access of `t`. -/
def Stmt.Apart (own : Nat → Prop) (s t : Stmt) : Prop :=
  s.Accesses fun k o x => own x → t.Accesses (Compat k o x)

/-- `s` starts no task and joins no task. -/
def Stmt.NoTasks : Stmt → Prop
  | spawnAt .. | joinAt .. | tasksBlock _ => False
  | seq s t | ite _ s t => s.NoTasks ∧ t.NoTasks
  | loop _ s => s.NoTasks
  | _ => True

theorem Stmt.Accesses.of_forall {α : Sort _} {C : α → Prop} {q : α → Kind → Order → Nat → Prop}
    {s : Stmt} (h : ∀ a, C a → s.Accesses (q a)) :
    s.Accesses fun k o x => ∀ a, C a → q a k o x := by
  induction s with
  | load | store | rmwStore | fence | wait => exact fun a ha => h a ha
  | fetchAdd | cas => exact ⟨fun a ha => (h a ha).1, fun a ha => (h a ha).2⟩
  | loadAt | storeAt | waitAt => exact fun j hj a ha => h a ha j hj
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 =>
    exact ⟨ih1 fun a ha => (h a ha).1, ih2 fun a ha => (h a ha).2⟩
  | loop _ _ ih | tasksBlock _ ih => exact ih h
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

/-- The labels `f` of a run, cut into two parts. -/
theorem lab_of_split {f : Nat → Label} {N : Nat} {L1 L2 : List Label}
    (h : (List.range N).map f = L1 ++ L2) :
    (∀ j (hj : j < L1.length), f j = L1[j]) ∧ ∀ j (hj : j < L2.length), f (L1.length + j) = L2[j] := by
  have hlen : N = L1.length + L2.length := by
    have := congrArg List.length h
    simpa using this
  refine ⟨fun j hj => ?_, fun j hj => ?_⟩
  · have := congrArg (·[j]?) h
    rw [getElem?_rangeMap (by omega), List.getElem?_append_left hj,
      List.getElem?_eq_getElem hj] at this
    exact Option.some.inj this
  · have := congrArg (·[L1.length + j]?) h
    rw [getElem?_rangeMap (by omega), List.getElem?_append_right (by omega),
      Nat.add_sub_cancel_left, List.getElem?_eq_getElem hj] at this
    exact Option.some.inj this

theorem Steps.snoc_none {s r0 L0 s' regs s'' regs'} (h : Steps s r0 L0 s' regs)
    (hs : Step s' regs none s'' regs') : Steps s r0 L0 s'' regs' := by
  induction h with
  | refl => exact .silent hs .refl
  | silent h1 _ ih => exact .silent h1 (ih hs)
  | visible h1 _ ih => exact .visible h1 (ih hs)

theorem Steps.snoc_some {s r0 L0 s' regs a s'' regs'} (h : Steps s r0 L0 s' regs)
    (hs : Step s' regs (some a) s'' regs') : Steps s r0 (L0 ++ [a]) s'' regs' := by
  induction h with
  | refl => exact .visible hs .refl
  | silent h1 _ ih => exact .silent h1 (ih hs)
  | visible h1 _ ih => exact .visible h1 (ih hs)

/-- Where a run of `loop e s` is: at the loop, at its test, after its end, or in a pass of `s`
that made the accesses `L0` so far. -/
def LoopAt (e : Expr) (s : Stmt) (L0 : List Label) (r0 : Nat → Int) (x : Stmt)
    (regs : Nat → Int) : Prop :=
  (x = .loop e s ∧ L0 = []) ∨ (x = .ite e (.seq s (.loop e s)) .skip ∧ L0 = []) ∨
    (x = .skip ∧ L0 = []) ∨ ∃ s', x = .seq s' (.loop e s) ∧ Steps s r0 L0 s' regs

/-- Each position before `n` of `L` is in a run of `s` to its end that ends by `n`. -/
def PassesBefore (s : Stmt) (L : List Label) (n : Nat) : Prop :=
  ∀ i < n, ∃ n0 L0 r1 r2, Steps s r1 L0 .skip r2 ∧
    (∀ j (hj : j < L0.length), L[n0 + j]? = some L0[j]) ∧ n0 ≤ i ∧ i < n0 + L0.length ∧
      n0 + L0.length ≤ n

theorem Runs.loop_aux {e : Expr} {s x : Stmt} {regs L} (h : Runs x regs L) :
    ∀ L0 r0, LoopAt e s L0 r0 x regs → ∀ m < L0.length + L.length, ∃ n L' regs', Runs s regs' L' ∧
      (∀ i (hi : i < L'.length), (L0 ++ L)[n + i]? = some L'[i]) ∧ n ≤ m ∧ m < n + L'.length ∧
      PassesBefore s (L0 ++ L) n := by
  induction h with
  | stop =>
    intro L0 r0 hx m hm
    rcases hx with ⟨-, rfl⟩ | ⟨-, rfl⟩ | ⟨-, rfl⟩ | ⟨s', -, hst⟩
    · simp at hm
    · simp at hm
    · simp at hm
    · refine ⟨0, L0, r0, hst.runs, fun i hi => ?_, Nat.zero_le _, by simpa using hm, nofun⟩
      simp [List.getElem?_eq_getElem hi]
  | silent hs _ ih =>
    intro L0 r0 hx m hm
    rcases hx with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨s', rfl, hst⟩
    · cases hs
      exact ih [] r0 (Or.inr (Or.inl ⟨rfl, rfl⟩)) m hm
    · cases hs with
      | ite_true => exact ih [] _ (Or.inr (Or.inr (Or.inr ⟨s, rfl, .refl⟩))) m hm
      | ite_false => exact ih [] r0 (Or.inr (Or.inr (Or.inl ⟨rfl, rfl⟩))) m hm
    · cases hs
    · cases hs with
      | seq_skip =>
        by_cases hm0 : m < L0.length
        · refine ⟨0, L0, r0, hst.runs, fun i hi => ?_, Nat.zero_le _, by omega, nofun⟩
          rw [Nat.zero_add, List.getElem?_append_left hi, List.getElem?_eq_getElem hi]
        · obtain ⟨n, L', r', hr, hlab, h1, h2, hpb⟩ :=
            ih [] r0 (Or.inl ⟨rfl, rfl⟩) (m - L0.length) (by simp only [List.length_nil]; omega)
          refine ⟨L0.length + n, L', r', hr, fun i hi => ?_, by omega, by omega, fun i hi => ?_⟩
          · rw [List.getElem?_append_right (by omega),
              show L0.length + n + i - L0.length = n + i by omega]
            simpa using hlab i hi
          · by_cases hi0 : i < L0.length
            · -- The pass that just ended.
              refine ⟨0, L0, r0, _, hst, fun j hj => ?_, Nat.zero_le _, by omega, by omega⟩
              rw [Nat.zero_add, List.getElem?_append_left hj, List.getElem?_eq_getElem hj]
            · obtain ⟨n0, L1, r1, r2, hs1, hl1, h3, h4, h5⟩ := hpb (i - L0.length) (by omega)
              refine ⟨L0.length + n0, L1, r1, r2, hs1, fun j hj => ?_, by omega, by omega, by omega⟩
              rw [List.getElem?_append_right (by omega),
                show L0.length + n0 + j - L0.length = n0 + j by omega]
              simpa using hl1 j hj
      | seq h1 => exact ih L0 r0 (Or.inr (Or.inr (Or.inr ⟨_, rfl, hst.snoc_none h1⟩))) m hm
  | visible hs _ ih =>
    intro L0 r0 hx m hm
    rcases hx with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨s', rfl, hst⟩
    · cases hs
    · cases hs
    · cases hs
    · cases hs with
      | seq h1 =>
        have := ih (L0 ++ [_]) r0 (Or.inr (Or.inr (Or.inr ⟨_, rfl, hst.snoc_some h1⟩))) m
          (by simp only [List.length_append, List.length_cons, List.length_nil] at hm ⊢; omega)
        simpa using this

/-- Each position of a run of `loop e s` is in a run of one pass of `s`. The passes before it
ended. -/
theorem Runs.loop_pass {e : Expr} {s : Stmt} {regs L} (h : Runs (.loop e s) regs L) {m : Nat}
    (hm : m < L.length) :
    ∃ n L' regs', Runs s regs' L' ∧ (∀ i (hi : i < L'.length), L[n + i]? = some L'[i]) ∧
      n ≤ m ∧ m < n + L'.length ∧ PassesBefore s L n := by
  simpa using h.loop_aux [] regs (Or.inl ⟨rfl, rfl⟩) m (by simpa using hm)

theorem Steps.loop_aux {e : Expr} {s x y : Stmt} {regs L regs''} (h : Steps x regs L y regs'')
    (hy : y = .skip) :
    ∀ L0 r0, LoopAt e s L0 r0 x regs → ∀ m < L0.length + L.length, ∃ n L' r1 r2,
      Steps s r1 L' .skip r2 ∧ (∀ i (hi : i < L'.length), (L0 ++ L)[n + i]? = some L'[i]) ∧
      n ≤ m ∧ m < n + L'.length ∧ n + L'.length ≤ L0.length + L.length := by
  induction h with
  | refl =>
    intro L0 r0 hx m hm
    subst hy
    rcases hx with ⟨h, -⟩ | ⟨h, -⟩ | ⟨-, rfl⟩ | ⟨s', h, -⟩
    · cases h
    · cases h
    · simp at hm
    · cases h
  | silent hs _ ih =>
    intro L0 r0 hx m hm
    rcases hx with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨s', rfl, hst⟩
    · cases hs
      exact ih hy [] r0 (Or.inr (Or.inl ⟨rfl, rfl⟩)) m hm
    · cases hs with
      | ite_true => exact ih hy [] _ (Or.inr (Or.inr (Or.inr ⟨s, rfl, .refl⟩))) m hm
      | ite_false => exact ih hy [] r0 (Or.inr (Or.inr (Or.inl ⟨rfl, rfl⟩))) m hm
    · cases hs
    · cases hs with
      | seq_skip =>
        by_cases hm0 : m < L0.length
        · refine ⟨0, L0, r0, _, hst, fun i hi => ?_, Nat.zero_le _, by omega, by omega⟩
          rw [Nat.zero_add, List.getElem?_append_left hi, List.getElem?_eq_getElem hi]
        · obtain ⟨n, L', r1, r2, hr, hlab, h1, h2, h3⟩ :=
            ih hy [] r0 (Or.inl ⟨rfl, rfl⟩) (m - L0.length) (by simp only [List.length_nil]; omega)
          simp only [List.length_nil, Nat.zero_add] at h3
          refine ⟨L0.length + n, L', r1, r2, hr, fun i hi => ?_, by omega, by omega, by omega⟩
          rw [List.getElem?_append_right (by omega),
            show L0.length + n + i - L0.length = n + i by omega]
          simpa using hlab i hi
      | seq h1 => exact ih hy L0 r0 (Or.inr (Or.inr (Or.inr ⟨_, rfl, hst.snoc_none h1⟩))) m hm
  | visible hs _ ih =>
    intro L0 r0 hx m hm
    rcases hx with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨s', rfl, hst⟩
    · cases hs
    · cases hs
    · cases hs
    · cases hs with
      | seq h1 =>
        have := ih hy (L0 ++ [_]) r0 (Or.inr (Or.inr (Or.inr ⟨_, rfl, hst.snoc_some h1⟩))) m
          (by simp only [List.length_append, List.length_cons, List.length_nil] at hm ⊢; omega)
        simp only [List.append_assoc, List.cons_append, List.nil_append, List.length_append,
          List.length_cons, List.length_nil] at this ⊢
        obtain ⟨n, L', r1, r2, hr, hlab, h1, h2, h3⟩ := this
        exact ⟨n, L', r1, r2, hr, hlab, h1, h2, by omega⟩

/-- Each position of a run of `loop e s` to its end is in a run of one pass of `s` to its end. -/
theorem Steps.loop_pass {e : Expr} {s : Stmt} {regs L regs''} (h : Steps (.loop e s) regs L .skip regs'')
    {m : Nat} (hm : m < L.length) :
    ∃ n L' r1 r2, Steps s r1 L' .skip r2 ∧ (∀ i (hi : i < L'.length), L[n + i]? = some L'[i]) ∧
      n ≤ m ∧ m < n + L'.length ∧ n + L'.length ≤ L.length := by
  simpa using h.loop_aux rfl [] regs (Or.inl ⟨rfl, rfl⟩) m (by simpa using hm)

namespace TaskProg

/-- A loop around a block: while `cond` holds, it runs `pre`, the block and `post`. The code
around the block starts no task. -/
structure Loop where
  cond : Expr
  pre : Stmt
  post : Stmt
  pre_ok : pre.NoTasks
  post_ok : post.NoTasks

/-- A part of the code of a thread: code that starts no task, or a `tasks` block. With
`some l`, the block runs in the loop `l`. -/
inductive Phase where
  | code (s : Stmt)
  | block (s : Stmt) (rep : Option Loop)

def Phase.stmt : Phase → Stmt
  | .code s => s
  | .block s none => .tasksBlock s
  | .block s (some l) => .loop l.cond (.seq l.pre (.seq (.tasksBlock s) l.post))

/-- The tasks that a phase can start. -/
def Phase.ids : Phase → List Nat
  | .code _ => []
  | .block s _ => s.spawnIds

/-- The phases, one after the other. -/
def seqAll : List Phase → Stmt
  | [] => .skip
  | p :: ps => .seq p.stmt (seqAll ps)

/-- The code of thread `q` before the lowering: the main code for thread 0, and task `t` for
thread `t + 1`. -/
def src (P : TaskProg) : Nat → Stmt
  | 0 => P.main
  | t + 1 => P.task t

/-- The phases of thread `q`: `ps` for the main code, and `tps t` for task `t`. -/
def thr (ps : List Phase) (tps : Nat → List Phase) : Nat → List Phase
  | 0 => ps
  | t + 1 => tps t

/-- `Desc tps u w`: task `u` starts task `w`, at some depth, or `w` is `u`. Task `u` starts the
tasks of the blocks in its phases `tps u`. -/
inductive Desc (tps : Nat → List Phase) : Nat → Nat → Prop
  | refl (u : Nat) : Desc tps u u
  | step {u v w : Nat} : (∃ p ∈ tps u, v ∈ p.ids) → Desc tps v w → Desc tps u w

theorem Desc.tail {tps : Nat → List Phase} {c t' t : Nat} (h : Desc tps c t')
    (hs : ∃ p ∈ tps t', t ∈ p.ids) : Desc tps c t := by
  induction h with
  | refl => exact .step hs (.refl t)
  | step h1 _ ih => exact .step h1 (ih hs)

/-- A task that starts no task has no other descendant. -/
theorem desc_of_nil {tps : Nat → List Phase} {u w : Nat} (h : tps u = []) :
    Desc tps u w ↔ w = u := by
  refine ⟨fun hd => ?_, fun hw => by subst hw; exact .refl _⟩
  cases hd with
  | refl => rfl
  | step hs =>
    rw [h] at hs
    simp at hs

@[simp] theorem desc_nil {u w : Nat} : Desc (fun _ => []) u w ↔ w = u := desc_of_nil rfl

variable (P : TaskProg) (own : Nat → Prop) (tps : Nat → List Phase)

/-- The rule for a block, for the places of `own`. The code of the block and each task of the
block are apart, and two tasks of the block are apart. The descendants of a task count as the
task. -/
def BlockOk (s : Stmt) : Prop :=
  s.NoBlocks ∧ s.RegsBelow P.marks ∧
    (∀ u ∈ s.spawnIds, ∀ w, Desc tps u w → s.Apart own (P.task w)) ∧
    ∀ u ∈ s.spawnIds, ∀ v ∈ s.spawnIds, u ≠ v → ∀ w w', Desc tps u w → Desc tps v w' →
      (P.task w).Apart own (P.task w')

/-- The rule for one phase. A block in a loop has the rule of one block: the passes run one after
the other, and each pass joins its tasks. -/
def Phase.Ok : Phase → Prop
  | .code s => s.NoTasks
  | .block s _ => P.BlockOk own tps s

/-- Each phase follows its rule, and no two phases start the same task. -/
def PhasesOk : List Phase → Prop
  | [] => True
  | p :: ps => p.Ok P own tps ∧ (∀ u ∈ p.ids, ∀ q ∈ ps, u ∉ q.ids) ∧ PhasesOk ps

/-- The rule on the code, for the places of `own`. The main code is the phases `ps`. Task `t` is
the phases `tps t`, or code that starts no task. One thread starts each task, and a task starts
only tasks with larger ids. Each place below `flags` that a program uses is in `own`, or other
theorems show that it cannot race. -/
structure Footprints (ps : List Phase) : Prop where
  ok : P.Ok
  le_flags : ∀ x, own x → x < P.flags
  main : P.main = seqAll ps
  phases : P.PhasesOk own tps ps
  tasks : ∀ t, P.task t = seqAll (tps t) ∨ (P.task t).NoTasks ∧ tps t = []
  task_phases : ∀ t, P.PhasesOk own tps (tps t)
  one_parent : ∀ q q' v, (∃ p ∈ thr ps tps q, v ∈ p.ids) → (∃ p ∈ thr ps tps q', v ∈ p.ids) →
    q = q'
  order : ∀ t v, (∃ p ∈ tps t, v ∈ p.ids) → t < v

/-- When no task starts a task, only the main code starts tasks. -/
theorem one_parent_nil (ps : List Phase) :
    ∀ q q' v, (∃ p ∈ thr ps (fun _ => []) q, v ∈ p.ids) →
      (∃ p ∈ thr ps (fun _ => []) q', v ∈ p.ids) → q = q'
  | 0, 0, _, _, _ => rfl
  | _ + 1, _, _, ⟨_, h, _⟩, _ | _, _ + 1, _, _, ⟨_, h, _⟩ => by simp [thr] at h

/-- What a block allows for an access in its lowered code. At a place of `own`, it cannot race
with a task of the block, or with a descendant of one. It writes `start v` only for a task `v` of
the block. -/
def InBlock (B : Stmt) (k : Kind) (o : Order) (x : Nat) : Prop :=
  (own x → ∀ v ∈ B.spawnIds, ∀ w, Desc tps v w → (P.task w).Accesses (Compat k o x)) ∧
    (k = .write → ∀ v, x = P.start v → v ∈ B.spawnIds)

/-- What a phase allows for an access in its lowered code. -/
def PhasePred : Phase → Kind → Order → Nat → Prop
  | .code _ => fun k _ x => k = .fence ∨ x < P.flags
  | .block B _ => fun k o x => P.InBlock own tps B k o x ∨ k = .fence ∨ x < P.flags

/-- The label writes the flag `start u`. -/
def StartW (u : Nat) (l : Label) : Prop := l.kind = .write ∧ l.loc = P.start u

/-- The label writes a flag. -/
def FlagW (l : Label) : Prop := l.kind = .write ∧ P.flags ≤ l.loc

/-- Where the position `m` of a run `L` of the phases `ps` is, for the task `u`. It comes before
each write of `start u`, or it is in a run of the block of `u`, or it comes after a run of that
block that wrote `start u`. -/
def Status (ps : List Phase) (u : Nat) (L : List Label) (m : Nat) : Prop :=
  (∀ m' l, L[m']? = some l → P.StartW u l → m < m') ∨
  (∃ B, u ∈ B.spawnIds ∧ (∃ rep, Phase.block B rep ∈ ps) ∧ P.BlockOk own tps B ∧
    (∀ l, L[m]? = some l → P.InBlock own tps B l.kind l.order l.loc) ∧
    ∃ n L' regs, Runs (P.lower (.tasksBlock B)) regs L' ∧
      (∀ i (hi : i < L'.length), L[n + i]? = some L'[i]) ∧ n ≤ m ∧ m < n + L'.length) ∨
  (∃ B n L' regs regs', P.BlockOk own tps B ∧ B.Below P.flags P.count ∧
    Steps (P.lower (.tasksBlock B)) regs L' .skip regs' ∧
    (∀ i (hi : i < L'.length), L[n + i]? = some L'[i]) ∧ n + L'.length ≤ m ∧
    ∃ i, ∃ hi : i < L'.length, P.StartW u L'[i])

variable {P own tps}

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

/-- The lowering keeps each access below `flags`. -/
theorem lower_below {q : Kind → Order → Nat → Prop} {s : Stmt} (hb : s.Below P.flags P.count)
    (hq : s.Accesses q) : (P.lower s).Accesses fun k o x => x < P.flags → q k o x := by
  induction s with
  | spawnAt =>
    refine ⟨trivial, fun j _ hx => ?_⟩
    simp only [start] at hx
    omega
  | joinAt =>
    refine fun j _ hx => ?_
    simp only [done] at hx
    omega
  | tasksBlock _ ih =>
    refine ⟨P.clearMarks_accesses _ _, ih hb hq, P.joinMarked_accesses fun u _ hx => ?_⟩
    simp only [done] at hx
    omega
  | load | store | rmwStore | fence | wait => exact fun _ => hq
  | fetchAdd | cas => exact ⟨fun _ => hq.1, fun _ => hq.2⟩
  | loadAt | storeAt | waitAt => exact fun j hj _ => hq j hj
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 hb.1 hq.1, ih2 hb.2 hq.2⟩
  | loop _ _ ih => exact ih hb hq
  | skip | assign | assignAt => trivial

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

omit own tps in
theorem block_below {B : Stmt} {rep : Option Loop} {n c : Nat}
    (h : (Phase.block B rep).stmt.Below n c) : B.Below n c := by
  cases rep with
  | none => exact h
  | some l => exact h.2.1

omit own tps in
theorem lower_loop (B : Stmt) (l : Loop) : P.lower (Phase.block B (some l)).stmt =
    .loop l.cond (.seq l.pre (.seq (P.lower (.tasksBlock B)) l.post)) := by
  show Stmt.loop _ (.seq (P.lower l.pre) (.seq _ (P.lower l.post))) = _
  rw [lower_noTasks l.pre_ok, lower_noTasks l.post_ok]

omit own tps in
/-- Code that starts no task writes no flag. -/
theorem code_noFlag {s : Stmt} (hb : s.Below P.flags P.count) (hn : s.NoTasks) {regs L}
    (h : Runs s regs L) : ∀ l ∈ L, ¬ P.FlagW l := fun l hl ⟨hk, hx⟩ => by
  rcases (accessRule _).runs h (code_accesses hb hn) l hl with h' | h'
  · rw [hk] at h'
    cases h'
  · omega

theorem eq_of_off {L L0 : List Label} {n0 : Nat}
    (h : ∀ j (hj : j < L0.length), L[n0 + j]? = some L0[j]) {i : Nat} (hi : i < L.length)
    (h1 : n0 ≤ i) (h2 : i < n0 + L0.length) : L[i] = L0[i - n0] := by
  have := h (i - n0) (by omega)
  rw [show n0 + (i - n0) = i by omega, List.getElem?_eq_getElem hi] at this
  exact Option.some.inj this

theorem mem_of_off {L L0 : List Label} {n0 : Nat}
    (h : ∀ j (hj : j < L0.length), L[n0 + j]? = some L0[j]) {i : Nat} (hi : i < L.length)
    (h1 : n0 ≤ i) (h2 : i < n0 + L0.length) : L[i] ∈ L0 := by
  rw [eq_of_off h hi h1 h2]
  exact List.getElem_mem _

theorem off_left {L A B : List Label} {n : Nat}
    (h : ∀ i (hi : i < (A ++ B).length), L[n + i]? = some (A ++ B)[i]) :
    ∀ i (hi : i < A.length), L[n + i]? = some A[i] := fun i hi => by
  have := h i (by simp only [List.length_append]; omega)
  rwa [List.getElem_append_left hi] at this

theorem off_right {L A B : List Label} {n : Nat}
    (h : ∀ i (hi : i < (A ++ B).length), L[n + i]? = some (A ++ B)[i]) :
    ∀ i (hi : i < B.length), L[n + A.length + i]? = some B[i] := fun i hi => by
  have := h (A.length + i) (by simp only [List.length_append]; omega)
  rw [List.getElem_append_right (by omega)] at this
  simpa [Nat.add_assoc] using this

/-- A run within a run at offset `n0`, at offset `n1`, is a run at offset `n0 + n1`. -/
theorem off_trans {L L0 L1 : List Label} {n0 n1 : Nat}
    (h0 : ∀ j (hj : j < L0.length), L[n0 + j]? = some L0[j])
    (h1 : ∀ j (hj : j < L1.length), L0[n1 + j]? = some L1[j]) (hl : n1 + L1.length ≤ L0.length) :
    ∀ j (hj : j < L1.length), L[n0 + n1 + j]? = some L1[j] := fun j hj => by
  rw [Nat.add_assoc, h0 (n1 + j) (by omega), ← List.getElem?_eq_getElem, h1 j hj]

omit own tps in
/-- In a pass of the loop to its end, each write of a flag is in the run of the block to its
end. -/
theorem body_done {pre blk post : Stmt} (hpre : pre.Below P.flags P.count) (hpn : pre.NoTasks)
    (hpost : post.Below P.flags P.count) (hqn : post.NoTasks) {r L r'}
    (h : Steps (.seq pre (.seq blk post)) r L .skip r') :
    ∀ i (hi : i < L.length), P.FlagW L[i] → ∃ n0 L0 r1 r2, Steps blk r1 L0 .skip r2 ∧
      (∀ j (hj : j < L0.length), L[n0 + j]? = some L0[j]) ∧ n0 ≤ i ∧ i < n0 + L0.length ∧
        n0 + L0.length ≤ L.length := by
  obtain ⟨L1, L2, ra, h1, h2, rfl⟩ := Steps.seq_done h
  obtain ⟨L21, L22, rb, h21, h22, rfl⟩ := Steps.seq_done h2
  have hoff : ∀ i (hi : i < (L1 ++ (L21 ++ L22)).length),
      (L1 ++ (L21 ++ L22))[0 + i]? = some (L1 ++ (L21 ++ L22))[i] := fun i hi => by
    rw [Nat.zero_add, List.getElem?_eq_getElem hi]
  intro i hi hw
  simp only [List.length_append] at hi
  by_cases hi1 : i < L1.length
  · exact absurd hw (code_noFlag hpre hpn h1.runs _ (mem_of_off (off_left hoff) _ (by omega)
      (by omega)))
  by_cases hi2 : i < L1.length + L21.length
  · refine ⟨L1.length, L21, ra, rb, h21, fun j hj => ?_, by omega, by omega,
      by simp only [List.length_append]; omega⟩
    simpa using off_left (off_right hoff) j hj
  · exact absurd hw (code_noFlag hpost hqn h22.runs _
      (mem_of_off (off_right (off_right hoff)) _ (by omega)
        (by omega)))

omit own tps in
/-- Each position of a run of a block phase is in a run of one pass of the block. Or it writes
no flag, and each write of a flag before it is in a run of the block that ended before it. -/
theorem block_pos {B : Stmt} {rep : Option Loop} (hb : (Phase.block B rep).stmt.Below P.flags P.count)
    {regs L} (h : Runs (P.lower (Phase.block B rep).stmt) regs L) {m : Nat} (hm : m < L.length) :
    (∃ n L' regs', Runs (P.lower (.tasksBlock B)) regs' L' ∧
      (∀ i (hi : i < L'.length), L[n + i]? = some L'[i]) ∧ n ≤ m ∧ m < n + L'.length) ∨
    ¬ P.FlagW L[m] ∧ ∀ i (hi : i < L.length), i < m → P.FlagW L[i] → ∃ n0 L0 r1 r2,
      Steps (P.lower (.tasksBlock B)) r1 L0 .skip r2 ∧
        (∀ j (hj : j < L0.length), L[n0 + j]? = some L0[j]) ∧ n0 ≤ i ∧ i < n0 + L0.length ∧
          n0 + L0.length ≤ m := by
  cases rep with
  | none =>
    exact Or.inl ⟨0, L, regs, h, fun i hi => by rw [Nat.zero_add, List.getElem?_eq_getElem hi],
      Nat.zero_le _, by omega⟩
  | some l =>
    obtain ⟨hpb, -, hqb⟩ := hb
    rw [lower_loop] at h
    obtain ⟨n, L', r', hr, hlab, h1, h2, hpass⟩ := Runs.loop_pass h hm
    -- A write of a flag in an earlier pass is in a run of the block that ended by `n`.
    have early : ∀ i (hi : i < L.length), i < n → P.FlagW L[i] → ∃ n0 L0 r1 r2,
        Steps (P.lower (.tasksBlock B)) r1 L0 .skip r2 ∧
          (∀ j (hj : j < L0.length), L[n0 + j]? = some L0[j]) ∧ n0 ≤ i ∧ i < n0 + L0.length ∧
            n0 + L0.length ≤ n := fun i hi hin hw => by
      obtain ⟨n0, L0, r1, r2, hs, hl0, h3, h4, h5⟩ := hpass i hin
      rw [eq_of_off hl0 hi h3 h4] at hw
      obtain ⟨n1, L1, r3, r4, hs1, hl1, h6, h7, h8⟩ :=
        body_done hpb l.pre_ok hqb l.post_ok hs (i - n0) (by omega) hw
      exact ⟨n0 + n1, L1, r3, r4, hs1, off_trans hl0 hl1 h8, by omega, by omega, by omega⟩
    -- A position of the current pass in the code around the block.
    have around : ∀ {A : List Label} {n1 : Nat} (hA : ∀ j (hj : j < A.length), L[n1 + j]? = some A[j]),
        (∀ l ∈ A, ¬ P.FlagW l) → n1 ≤ m → m < n1 + A.length →
        (∀ i (hi : i < L.length), i < m → n ≤ i → P.FlagW L[i] → ∃ n0 L0 r1 r2,
          Steps (P.lower (.tasksBlock B)) r1 L0 .skip r2 ∧
            (∀ j (hj : j < L0.length), L[n0 + j]? = some L0[j]) ∧ n0 ≤ i ∧ i < n0 + L0.length ∧
              n0 + L0.length ≤ m) →
        ¬ P.FlagW L[m] ∧ ∀ i (hi : i < L.length), i < m → P.FlagW L[i] → ∃ n0 L0 r1 r2,
          Steps (P.lower (.tasksBlock B)) r1 L0 .skip r2 ∧
            (∀ j (hj : j < L0.length), L[n0 + j]? = some L0[j]) ∧ n0 ≤ i ∧ i < n0 + L0.length ∧
              n0 + L0.length ≤ m := fun hA hno hm1 hm2 hcur =>
        ⟨hno _ (mem_of_off hA hm hm1 hm2), fun i hi him hw => by
          by_cases hin : i < n
          · obtain ⟨n0, L0, r1, r2, hs, hl0, h3, h4, h5⟩ := early i hi hin hw
            exact ⟨n0, L0, r1, r2, hs, hl0, h3, h4, by omega⟩
          · exact hcur i hi him (by omega) hw⟩
    rcases Runs.seq_split hr with hp | ⟨L1, L2, ra, hs1, hr2, rfl⟩
    · -- The pass stops in `pre`.
      exact Or.inr <| around hlab (code_noFlag hpb l.pre_ok hp) h1 h2 fun i hi him hin hw =>
        absurd hw (code_noFlag hpb l.pre_ok hp _ (mem_of_off hlab hi hin (by omega)))
    have hno1 := code_noFlag hpb l.pre_ok hs1.runs
    have hpre : ∀ i (hi : i < L.length), n ≤ i → i < n + L1.length → ¬ P.FlagW L[i] :=
      fun i hi h3 h4 => hno1 _ (mem_of_off (off_left hlab) hi h3 h4)
    simp only [List.length_append] at h2
    rcases Runs.seq_split hr2 with hp2 | ⟨L21, L22, rb, hs21, hr22, rfl⟩
    · by_cases hm1 : m < n + L1.length
      · exact Or.inr <| around (off_left hlab) hno1 h1 hm1 fun i hi him hin hw =>
          absurd hw (hpre i hi hin (by omega))
      · exact Or.inl ⟨n + L1.length, L2, ra, hp2, off_right hlab, by omega, by omega⟩
    · simp only [List.length_append] at h2
      by_cases hm1 : m < n + L1.length
      · exact Or.inr <| around (off_left hlab) hno1 h1 hm1 fun i hi him hin hw =>
          absurd hw (hpre i hi hin (by omega))
      by_cases hm2 : m < n + L1.length + L21.length
      · exact Or.inl ⟨n + L1.length, L21, ra, hs21.runs, off_left (off_right hlab), by omega, by omega⟩
      · have hno3 := code_noFlag hqb l.post_ok hr22
        exact Or.inr <| around (off_right (off_right hlab)) hno3 (by omega) (by omega) fun i hi him hin hw => by
          by_cases hi1 : i < n + L1.length
          · exact absurd hw (hpre i hi hin hi1)
          by_cases hi2 : i < n + L1.length + L21.length
          · exact ⟨n + L1.length, L21, ra, rb, hs21, off_left (off_right hlab), by omega, by omega,
              by omega⟩
          · exact absurd hw (hno3 _ (mem_of_off (off_right (off_right hlab)) hi (by omega) (by omega)))

omit own tps in
/-- Each write of a flag in a run of a block phase to its end is in a run of one pass of the
block to its end. -/
theorem block_pass_done {B : Stmt} {rep : Option Loop}
    (hb : (Phase.block B rep).stmt.Below P.flags P.count) {regs L regs''}
    (h : Steps (P.lower (Phase.block B rep).stmt) regs L .skip regs'') {m : Nat}
    (hm : m < L.length) (hw : P.FlagW L[m]) :
    ∃ n L' r1 r2, Steps (P.lower (.tasksBlock B)) r1 L' .skip r2 ∧
      (∀ i (hi : i < L'.length), L[n + i]? = some L'[i]) ∧ n ≤ m ∧ m < n + L'.length ∧
      n + L'.length ≤ L.length := by
  cases rep with
  | none =>
    exact ⟨0, L, regs, regs'', h, fun i hi => by rw [Nat.zero_add, List.getElem?_eq_getElem hi],
      Nat.zero_le _, by omega, by omega⟩
  | some l =>
    obtain ⟨hpb, -, hqb⟩ := hb
    rw [lower_loop] at h
    obtain ⟨n, L', r1, r2, hs, hl, h1, h2, h3⟩ := Steps.loop_pass h hm
    rw [eq_of_off hl hm h1 h2] at hw
    obtain ⟨n1, L1, r3, r4, hs1, hl1, h4, h5, h6⟩ :=
      body_done hpb l.pre_ok hqb l.post_ok hs (m - n) (by omega) hw
    exact ⟨n + n1, L1, r3, r4, hs1, off_trans hl hl1 h6, by omega, by omega, by omega⟩

/-- Each access of the lowered code of a block follows `InBlock`. -/
theorem block_accesses (hlim : ∀ x, own x → x < P.flags) {B : Stmt} (hp : P.BlockOk own tps B)
    (hb : B.Below P.flags P.count) : (P.lower (.tasksBlock B)).Accesses (P.InBlock own tps B) := by
  obtain ⟨hn, -, hap, -⟩ := hp
  have hq := Stmt.Accesses.of_forall (s := B)
    (C := fun vw : Nat × Nat => vw.1 ∈ B.spawnIds ∧ Desc tps vw.1 vw.2)
    (q := fun vw k o x => own x → (P.task vw.2).Accesses (Compat k o x))
    fun vw h => hap vw.1 h.1 vw.2 h.2
  refine ⟨P.clearMarks_accesses _ _, (lower_inBlock (s := B) (ids := B.spawnIds) hb hn
    (fun _ h => h) hq).mono fun _ _ _ h => ⟨fun hx v hv w hw => h.1 (hlim _ hx) (v, w) ⟨hv, hw⟩ hx,
    h.2⟩, P.joinMarked_accesses fun u _ => ⟨fun hx => ?_, nofun⟩⟩
  have := hlim _ hx
  simp only [done] at this
  omega

/-- Each access of the lowered code of a phase follows `PhasePred`. -/
theorem phase_accesses (hlim : ∀ x, own x → x < P.flags) {p : Phase} (hp : p.Ok P own tps)
    (hb : p.stmt.Below P.flags P.count) : (P.lower p.stmt).Accesses (P.PhasePred own tps p) := by
  cases p with
  | code s =>
    show (P.lower s).Accesses _
    rw [lower_noTasks hp]
    exact code_accesses hb hp
  | block B rep =>
    have hblk := block_accesses hlim hp (block_below hb)
    cases rep with
    | none => exact hblk.mono fun _ _ _ h => Or.inl h
    | some l =>
      obtain ⟨hpb, -, hqb⟩ := hb
      rw [lower_loop]
      exact ⟨(code_accesses hpb l.pre_ok).mono fun _ _ _ h => Or.inr h,
        hblk.mono fun _ _ _ h => Or.inl h, (code_accesses hqb l.post_ok).mono fun _ _ _ h => Or.inr h⟩

theorem PhasePred.ids_of_start {p : Phase} {k o x} (h : P.PhasePred own tps p k o x)
    (hk : k = .write) {u} (hx : x = P.start u) : u ∈ p.ids := by
  cases p with
  | code s =>
    simp only [PhasePred] at h
    rcases h with h | h
    · rw [hk] at h
      cases h
    · simp only [start] at hx
      omega
  | block B _ =>
    rcases h with h | h | h
    · exact h.2 hk u hx
    · rw [hk] at h
      cases h
    · simp only [start] at hx
      omega

/-- The phases write `start u` only for a task `u` of one of their blocks. -/
theorem seqAll_accesses (hlim : ∀ x, own x → x < P.flags) {ps : List Phase}
    (hok : P.PhasesOk own tps ps) (hb : (seqAll ps).Below P.flags P.count) :
    (P.lower (seqAll ps)).Accesses fun k _ x =>
      k = .write → ∀ u, x = P.start u → ∃ p ∈ ps, u ∈ p.ids := by
  induction ps with
  | nil => trivial
  | cons p ps ih =>
    refine ⟨(phase_accesses hlim hok.1 hb.1).mono fun _ _ _ h hk u hx =>
      ⟨p, List.mem_cons_self, h.ids_of_start hk hx⟩,
      (ih hok.2.2 hb.2).mono fun _ _ _ h hk u hx => ?_⟩
    obtain ⟨q, hq, hu⟩ := h hk u hx
    exact ⟨q, List.mem_cons_of_mem _ hq, hu⟩

omit own tps in
theorem flagW_of_start {u : Nat} {l : Label} (h : P.StartW u l) : P.FlagW l :=
  ⟨h.1, by rw [h.2]; simp only [start]; omega⟩

/-- A position of a run of a block phase of `u` has a `Status` for `u`, whatever comes after the
phase. -/
theorem block_status (hlim : ∀ x, own x → x < P.flags) {ps : List Phase} {B : Stmt}
    {rep : Option Loop} (hok : P.BlockOk own tps B)
    (hb : (Phase.block B rep).stmt.Below P.flags P.count) (hmem : Phase.block B rep ∈ ps) {u : Nat}
    (hu : u ∈ B.spawnIds) {regs L1} (L2 : List Label)
    (h1 : Runs (P.lower (Phase.block B rep).stmt) regs L1) {m : Nat} (hm : m < L1.length) :
    P.Status own tps ps u (L1 ++ L2) m := by
  have lift : ∀ {n : Nat} {L' : List Label}, (∀ i (hi : i < L'.length), L1[n + i]? = some L'[i]) →
      ∀ i (hi : i < L'.length), (L1 ++ L2)[n + i]? = some L'[i] := fun h i hi => by
    rw [List.getElem?_append_left (List.getElem?_eq_some_iff.1 (h i hi)).1]
    exact h i hi
  rcases block_pos hb h1 hm with ⟨n, L', r', hr', hlab', hn1, hn2⟩ | ⟨hnf, hpre⟩
  · refine Or.inr (Or.inl ⟨B, hu, ⟨rep, hmem⟩, hok, fun l hl' => ?_, n, L', r', hr', lift hlab',
      hn1, hn2⟩)
    rw [List.getElem?_append_left hm, List.getElem?_eq_getElem hm] at hl'
    cases hl'
    exact (accessRule _).runs hr' (block_accesses hlim hok (block_below hb)) _
      (mem_of_off hlab' hm hn1 hn2)
  · by_cases hex : ∃ i, ∃ hi : i < L1.length, i < m ∧ P.StartW u L1[i]
    · obtain ⟨i, hi, him, hw⟩ := hex
      obtain ⟨n0, L0, r1, r2, hs, hl0, h3, h4, h5⟩ := hpre i hi him (flagW_of_start hw)
      refine Or.inr (Or.inr ⟨B, n0, L0, r1, r2, hok, block_below hb, hs, lift hl0, h5, i - n0,
        by omega, ?_⟩)
      rw [← eq_of_off hl0 hi h3 h4]
      exact hw
    · refine Or.inl fun m' l hl' hs => Nat.lt_of_not_le fun hle => ?_
      rw [List.getElem?_append_left (by omega), List.getElem?_eq_some_iff] at hl'
      obtain ⟨hlt, rfl⟩ := hl'
      rcases Nat.lt_or_eq_of_le hle with h | rfl
      · exact hex ⟨m', hlt, h, hs⟩
      · exact hnf (flagW_of_start hs)

/-- Each position of a run of the phases has a `Status` for each task. -/
theorem status (hlim : ∀ x, own x → x < P.flags) {ps : List Phase} (hok : P.PhasesOk own tps ps)
    (hb : (seqAll ps).Below P.flags P.count) {regs L} (hr : Runs (P.lower (seqAll ps)) regs L)
    (u : Nat) {m : Nat} (hm : m < L.length) : P.Status own tps ps u L m := by
  induction ps generalizing regs L m with
  | nil =>
    have := Runs.skip_nil (regs := regs) hr
    subst this
    simp at hm
  | cons p ps ih =>
    have hacc := phase_accesses hlim hok.1 hb.1
    rcases Runs.seq_split hr with hp | ⟨L1, L2, regs', h1, h2, rfl⟩
    · -- The run stops in the phase `p`.
      have hl := (accessRule _).runs hp hacc
      by_cases hu : u ∈ p.ids
      · cases p with
        | code s => simp [Phase.ids] at hu
        | block B rep =>
          have := block_status hlim hok.1 hb.1 (List.mem_cons_self (l := ps)) hu [] hp hm
          simpa only [List.append_nil] using this
      · exact Or.inl fun _ l hl' hs =>
          (hu ((hl l (List.mem_of_getElem? hl')).ids_of_start hs.1 hs.2)).elim
    · have hl1 := (accessRule _).runs h1.runs hacc
      have hno2 : u ∈ p.ids → ∀ l ∈ L2, ¬ P.StartW u l := fun hu l hl hs => by
        obtain ⟨q, hq, huq⟩ := (accessRule _).runs h2 (seqAll_accesses hlim hok.2.2 hb.2) l hl hs.1 u
          hs.2
        exact hok.2.1 u hu q hq huq
      by_cases hu : u ∈ p.ids
      · cases p with
        | code s => simp [Phase.ids] at hu
        | block B rep =>
          by_cases hm1 : m < L1.length
          · exact block_status hlim hok.1 hb.1 (List.mem_cons_self (l := ps)) hu L2 h1.runs hm1
          · by_cases hw : ∃ i, ∃ hi : i < L1.length, P.StartW u L1[i]
            · obtain ⟨i, hi, hwi⟩ := hw
              obtain ⟨n, L', r1, r2, hs', hlab', hn1, hn2, hn3⟩ :=
                block_pass_done hb.1 h1 hi (flagW_of_start hwi)
              have hli : L1[i] = L'[i - n]'(by omega) := by
                have := hlab' (i - n) (by omega)
                rw [show n + (i - n) = i by omega, List.getElem?_eq_getElem hi] at this
                exact Option.some.inj this
              refine Or.inr (Or.inr ⟨B, n, L', r1, r2, hok.1, block_below hb.1, hs', fun j hj => ?_,
                by omega, i - n, by omega, hli ▸ hwi⟩)
              rw [List.getElem?_append_left (by omega)]
              exact hlab' j hj
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
            hbef | ⟨B, hu', ⟨rep, hmem⟩, hB, hin, n, L', r1, hs, hlab, hn1, hn2⟩ |
              ⟨B, n, L', r1, r2, hB, hBb, hs, hlab, hn, hw⟩
          · refine Or.inl fun m' l hl' hs => ?_
            have hle := hbefore m' l hl' hs
            rw [List.getElem?_append_right hle] at hl'
            have := hbef _ l hl' hs
            omega
          · refine Or.inr (Or.inl ⟨B, hu', ⟨rep, List.mem_cons_of_mem _ hmem⟩, hB, fun l hl' => hin l ?_,
              L1.length + n, L', r1, hs, fun i hi => ?_, by omega, by omega⟩)
            · rwa [List.getElem?_append_right (by omega)] at hl'
            · rw [List.getElem?_append_right (by omega),
                show L1.length + n + i - L1.length = n + i by omega]
              exact hlab i hi
          · refine Or.inr (Or.inr ⟨B, L1.length + n, L', r1, r2, hB, hBb, hs, fun i hi => ?_,
              by omega, hw⟩)
            rw [List.getElem?_append_right (by omega),
              show L1.length + n + i - L1.length = n + i by omega]
            exact hlab i hi

/-- In a run of the phases to their end, each write of `start u` is in a run of a block to its
end. -/
theorem start_in_block (hlim : ∀ x, own x → x < P.flags) {ps : List Phase}
    (hok : P.PhasesOk own tps ps) (hb : (seqAll ps).Below P.flags P.count) {regs L regs'}
    (hr : Steps (P.lower (seqAll ps)) regs L .skip regs') {u i : Nat} (hi : i < L.length)
    (hw : P.StartW u L[i]) :
    ∃ B n L' r1 r2, P.BlockOk own tps B ∧ B.Below P.flags P.count ∧
      Steps (P.lower (.tasksBlock B)) r1 L' .skip r2 ∧
      (∀ j (hj : j < L'.length), L[n + j]? = some L'[j]) ∧ n ≤ i ∧ i < n + L'.length ∧
      n + L'.length ≤ L.length := by
  induction ps generalizing regs L i with
  | nil =>
    obtain ⟨rfl, -⟩ := Steps.skip hr
    simp at hi
  | cons p ps ih =>
    obtain ⟨L1, L2, r', h1, h2, rfl⟩ := Steps.seq_done hr
    by_cases hi1 : i < L1.length
    · rw [List.getElem_append_left hi1] at hw
      have hacc := (accessRule _).runs h1.runs (phase_accesses hlim hok.1 hb.1) L1[i]
        (List.getElem_mem hi1)
      have hu := hacc.ids_of_start hw.1 hw.2
      cases p with
      | code s => simp [Phase.ids] at hu
      | block B rep =>
        obtain ⟨n, L', r1, r2, hs', hlab', hn1, hn2, hn3⟩ :=
          block_pass_done hb.1 h1 hi1 (flagW_of_start hw)
        refine ⟨B, n, L', r1, r2, hok.1, block_below hb.1, hs', fun j hj => ?_, hn1, hn2,
          by simp only [List.length_append]; omega⟩
        rw [List.getElem?_append_left (by omega)]
        exact hlab' j hj
    · have hi2 : i - L1.length < L2.length := by simp only [List.length_append] at hi; omega
      rw [List.getElem_append_right (by omega)] at hw
      obtain ⟨B, n, L', r1, r2, hB, hBb, hs, hlab, hn, hin, hle⟩ := ih hok.2.2 hb.2 h2 hi2 hw
      refine ⟨B, L1.length + n, L', r1, r2, hB, hBb, hs, fun j hj => ?_, by omega, by omega,
        by simp only [List.length_append]; omega⟩
      rw [List.getElem?_append_right (by omega), show L1.length + n + j - L1.length = n + j by omega]
      exact hlab j hj

theorem seqAll_ids_lt {ps : List Phase} (hb : (seqAll ps).Below P.flags P.count) :
    ∀ p ∈ ps, ∀ v ∈ p.ids, v < P.count := by
  induction ps with
  | nil => nofun
  | cons p ps ih =>
    intro p' hp' v hv
    rcases List.mem_cons.1 hp' with rfl | hp'
    · cases p' with
      | code => simp [Phase.ids] at hv
      | block B _ => exact spawnIds_lt (s := B) (block_below hb.1) v hv
    · exact ih hb.2 p' hp' v hv

/-! ## The threads of a program that follows the rule -/

variable {ps : List Phase} {X : Execution}

/-- The phases of each thread follow the rule. -/
theorem thr_ok (hF : P.Footprints own tps ps) : ∀ q, P.PhasesOk own tps (thr ps tps q)
  | 0 => hF.phases
  | t + 1 => hF.task_phases t

theorem src_below (hP : P.Ok) : ∀ q, (P.src q).Below P.flags P.count
  | 0 => hP.main
  | t + 1 => hP.task t

theorem src_cases (hF : P.Footprints own tps ps) :
    ∀ q, P.src q = seqAll (thr ps tps q) ∨ (P.src q).NoTasks ∧ thr ps tps q = []
  | 0 => Or.inl hF.main
  | t + 1 => hF.tasks t

theorem thr_below (hF : P.Footprints own tps ps) {q : Nat} (h : P.src q = seqAll (thr ps tps q)) :
    (seqAll (thr ps tps q)).Below P.flags P.count := by
  rw [← h]
  exact src_below hF.ok q

/-- A task that a thread starts has an id below `count`. -/
theorem starts_lt (hF : P.Footprints own tps ps) {q v : Nat} (h : ∃ p ∈ thr ps tps q, v ∈ p.ids) :
    v < P.count := by
  obtain ⟨p, hp, hv⟩ := h
  rcases src_cases hF q with hs | ⟨-, he⟩
  · exact seqAll_ids_lt (thr_below hF hs) p hp v hv
  · rw [he] at hp
    cases hp

/-- The lowered code of thread `q` writes `start u` only for a task `u` that it starts. -/
theorem src_starts (hF : P.Footprints own tps ps) (q : Nat) :
    (P.lower (P.src q)).Accesses fun k _ x =>
      k = .write → ∀ u, x = P.start u → ∃ p ∈ thr ps tps q, u ∈ p.ids := by
  rcases src_cases hF q with h | ⟨hn, -⟩
  · rw [h]
    exact seqAll_accesses hF.le_flags (thr_ok hF q) (thr_below hF h)
  · rw [lower_noTasks hn]
    refine (code_accesses (src_below hF.ok q) hn).mono fun k _ x h hk u hx => ?_
    rcases h with h | h
    · rw [hk] at h
      cases h
    · simp only [start] at hx
      omega

/-- Each position of a run of the lowered code of thread `q` has a `Status` for each task. -/
theorem src_status (hF : P.Footprints own tps ps) (q : Nat) {regs L}
    (hr : Runs (P.lower (P.src q)) regs L) (u : Nat) {m : Nat} (hm : m < L.length) :
    P.Status own tps (thr ps tps q) u L m := by
  rcases src_cases hF q with h | ⟨hn, -⟩
  · rw [h] at hr
    exact status hF.le_flags (thr_ok hF q) (thr_below hF h) hr u hm
  · rw [lower_noTasks hn] at hr
    refine Or.inl fun m' l hl hs => ?_
    have h : l.kind = .fence ∨ l.loc < P.flags :=
      (accessRule _).runs hr (code_accesses (src_below hF.ok q) hn) l (List.mem_of_getElem? hl)
    rcases h with h | h
    · rw [hs.1] at h
      cases h
    · rw [hs.2] at h
      simp only [start] at h
      omega

/-- `start_in_block` for the lowered code of thread `q`. -/
theorem src_start_in_block (hF : P.Footprints own tps ps) (q : Nat) {regs L regs'}
    (hr : Steps (P.lower (P.src q)) regs L .skip regs') {u i : Nat} (hi : i < L.length)
    (hw : P.StartW u L[i]) :
    ∃ B n L' r1 r2, P.BlockOk own tps B ∧ B.Below P.flags P.count ∧
      Steps (P.lower (.tasksBlock B)) r1 L' .skip r2 ∧
      (∀ j (hj : j < L'.length), L[n + j]? = some L'[j]) ∧ n ≤ i ∧ i < n + L'.length ∧
      n + L'.length ≤ L.length := by
  rcases src_cases hF q with h | ⟨hn, -⟩
  · rw [h] at hr
    exact start_in_block hF.le_flags (thr_ok hF q) (thr_below hF h) hr hi hw
  · rw [lower_noTasks hn] at hr
    have h : L[i].kind = .fence ∨ L[i].loc < P.flags :=
      (accessRule _).runs hr.runs (code_accesses (src_below hF.ok q) hn) _ (List.getElem_mem hi)
    rcases h with h | h
    · rw [hw.1] at h
      cases h
    · rw [hw.2] at h
      simp only [start] at h
      omega

theorem lab_of (r : P.toProg.Run X) (e : X.E) {q : Nat} (h : X.thread e = q) :
    r.lab q (r.idx e) = ⟨X.kind e, X.order e, X.loc e, r.val e, r.isRmw e⟩ := by
  subst h
  exact r.lab_eq e

/-- The label of `e` in a run `L` of its thread that starts at the position `n`. -/
theorem getElem_of_lab (r : P.toProg.Run X) {q n : Nat} {L : List Label}
    (hlab : ∀ j (hj : j < L.length), r.lab q (n + j) = L[j]) {e : X.E} (he : X.thread e = q)
    (h1 : n ≤ r.idx e) (h2 : r.idx e < n + L.length) :
    L[r.idx e - n]? = some ⟨X.kind e, X.order e, X.loc e, r.val e, r.isRmw e⟩ := by
  rw [List.getElem?_eq_getElem (by omega), ← hlab (r.idx e - n) (by omega), Nat.add_sub_of_le h1,
    lab_of r e he]

/-- The labels of the thread of `e`, up to `e`: reads of the flags first, and then a run of the
lowered code of the thread. The label of `e` is in the run, if `e` accesses a place below `flags`
or writes a `start` flag. -/
theorem thread_split (r : P.toProg.Run X) {e : X.E}
    (he : X.loc e < P.flags ∨ X.IsWrite e ∧ ∃ u < P.count, X.loc e = P.start u) :
    ∃ n regs L, Runs (P.lower (P.src (X.thread e))) regs L ∧ n + L.length = r.idx e + 1 ∧
      (∀ j (hj : j < L.length), r.lab (X.thread e) (n + j) = L[j]) ∧
      (∀ j < n, (r.lab (X.thread e) j).kind = .read ∧ P.flags ≤ (r.lab (X.thread e) j).loc) ∧
      n ≤ r.idx e := by
  have htr := r.trace e
  have hlab := r.lab_eq e
  generalize hq : X.thread e = q at htr hlab
  cases q with
  | zero =>
    refine ⟨0, _, _, htr, by simp, fun j hj => ?_, nofun, Nat.zero_le _⟩
    simp
  | succ t =>
    have htc := task_lt r hq
    rw [code_succ, taskCode_of_lt htc] at htr
    unfold Trace at htr
    -- The label of `e` is not a read of a flag, and not a write of `done t`.
    have hnr : ¬ ((r.lab (t + 1) (r.idx e)).kind = .read ∧
        P.flags ≤ (r.lab (t + 1) (r.idx e)).loc) := by
      rw [hlab]
      rintro ⟨h1, h2⟩
      simp only at h1 h2
      rcases he with h | ⟨h, -⟩
      · omega
      · rw [show X.kind e = .write from h] at h1
        cases h1
    have hnd : (r.lab (t + 1) (r.idx e)).loc ≠ P.done t := by
      rw [hlab]
      intro h
      simp only at h
      rcases he with h' | ⟨-, u, hu, h'⟩
      · rw [h] at h'
        unfold done at h'
        omega
      · rw [h] at h'
        unfold start done at h'
        omega
    have hlast : r.idx e < ((List.range (r.idx e + 1)).map (r.lab (t + 1))).length := by simp
    have hrd : ∀ {regs L}, Runs (.wait (P.start t)) regs L →
        ∀ l ∈ L, l.kind = .read ∧ P.flags ≤ l.loc := fun h l hl =>
      (accessRule fun k _ x => k = .read ∧ P.flags ≤ x).runs h ⟨rfl, by unfold start; omega⟩ l hl
    rcases Runs.seq_split htr with hw | ⟨L1, L2, r1, h1, h2, hL⟩
    · exfalso
      refine hnr (hrd hw _ ?_)
      exact List.mem_map.2 ⟨r.idx e, List.mem_range.2 (Nat.lt_succ_self _), rfl⟩
    · obtain ⟨L2a, L2b, h2a, h2b, rfl⟩ : ∃ L2a L2b, Runs (P.lower (P.task t)) r1 L2a ∧
          (∀ l ∈ L2b, l.loc = P.done t) ∧ L2 = L2a ++ L2b := by
        rcases Runs.seq_split h2 with hm | ⟨L2a, L2b, r2, h2a, h2b, rfl⟩
        · exact ⟨L2, [], hm, nofun, by simp⟩
        · exact ⟨L2a, L2b, h2a.runs, fun l hl =>
            (accessRule fun _ _ x => x = P.done t).runs h2b rfl l hl, rfl⟩
      rw [← List.append_assoc] at hL
      have hN : r.idx e + 1 = (L1 ++ L2a).length + L2b.length := by
        rw [← List.length_append, ← hL]
        simp
      obtain ⟨-, hf2⟩ := lab_of_split hL
      have hb0 : L2b = [] := by
        refine List.eq_nil_of_length_eq_zero (Nat.eq_zero_of_not_pos fun hpos => hnd ?_)
        have hj : r.idx e - (L1 ++ L2a).length < L2b.length := by omega
        rw [show r.idx e = (L1 ++ L2a).length + (r.idx e - (L1 ++ L2a).length) by omega,
          hf2 _ hj]
        exact h2b _ (List.getElem_mem hj)
      subst hb0
      rw [List.append_nil] at hL
      simp only [List.length_nil, Nat.add_zero] at hN
      obtain ⟨hg1, hg2⟩ := lab_of_split hL
      have hn : L1.length ≤ r.idx e := Nat.le_of_not_lt fun h => by
        refine hnr ?_
        rw [hg1 _ h]
        exact hrd h1.runs _ (List.getElem_mem h)
      refine ⟨L1.length, r1, L2a, h2a, by simp only [List.length_append] at hN; omega, hg2,
        fun j hj => ?_, hn⟩
      rw [hg1 j hj]
      exact hrd h1.runs _ (List.getElem_mem hj)

/-- The write of `done t` comes after a run of the lowered code of task `t` to its end. -/
theorem done_split (hP : P.Ok) (r : P.toProg.Run X) {d : X.E} {t : Nat}
    (hd : X.thread d = t + 1) (hw : X.IsWrite d) (hl : X.loc d = P.done t) :
    ∃ n regs regs' L, Steps (P.lower (P.src (t + 1))) regs L .skip regs' ∧
      n + L.length ≤ r.idx d ∧ (∀ j (hj : j < L.length), r.lab (t + 1) (n + j) = L[j]) ∧
      (∀ j < n, (r.lab (t + 1) j).kind = .read) ∧
      ∀ j, n + L.length ≤ j → j ≤ r.idx d → (r.lab (t + 1) j).loc = P.done t := by
  have htc := task_lt r hd
  have htr := r.trace d
  have hlab := r.lab_eq d
  rw [hd] at htr hlab
  rw [code_succ, taskCode_of_lt htc] at htr
  unfold Trace at htr
  -- The label of `d` is not a read, and the lowered code does not make it.
  have hk : (r.lab (t + 1) (r.idx d)).kind = .write := by rw [hlab]; exact hw
  have hx : (r.lab (t + 1) (r.idx d)).loc = P.done t := by rw [hlab]; exact hl
  have hnl : ∀ {regs L}, Runs (P.lower (P.task t)) regs L → (r.lab (t + 1) (r.idx d)) ∉ L := by
    intro regs L h hm
    have h' : _ ∨ _ ∨ _ := (accessRule _).runs h (lower_accesses (hP.task t)) _ hm
    rw [hk, hx] at h'
    rcases h' with h' | h' | ⟨-, ⟨-, u, hu, h'⟩ | ⟨h', -⟩⟩
    · cases h'
    · unfold done at h'
      omega
    · unfold start done at h'
      omega
    · cases h'
  have hrd : ∀ {regs L}, Runs (.wait (P.start t)) regs L → ∀ l ∈ L, l.kind = .read :=
    fun h l hl => (accessRule fun k _ _ => k = .read).runs h rfl l hl
  have hnr : ∀ {regs L}, Runs (.wait (P.start t)) regs L → (r.lab (t + 1) (r.idx d)) ∉ L :=
    fun h hm => by
      have := hrd h _ hm
      rw [hk] at this
      cases this
  have hmem : r.lab (t + 1) (r.idx d) ∈ (List.range (r.idx d + 1)).map (r.lab (t + 1)) :=
    List.mem_map.2 ⟨r.idx d, List.mem_range.2 (Nat.lt_succ_self _), rfl⟩
  rcases Runs.seq_split htr with hw' | ⟨L1, L2, r1, h1, h2, hL⟩
  · exact absurd hmem (hnr hw')
  · rcases Runs.seq_split h2 with hm | ⟨L2a, L2b, r2, h2a, h2b, rfl⟩
    · rw [hL] at hmem
      rcases List.mem_append.1 hmem with h | h
      · exact absurd h (hnr h1.runs)
      · exact absurd h (hnl hm)
    · rw [← List.append_assoc] at hL
      have hN : r.idx d + 1 = (L1 ++ L2a).length + L2b.length := by
        rw [← List.length_append, ← hL]
        simp
      obtain ⟨hg1, hg2⟩ := lab_of_split hL
      have hpos : (L1 ++ L2a).length ≤ r.idx d := Nat.le_of_not_lt fun h => by
        have hm : r.lab (t + 1) (r.idx d) ∈ L1 ++ L2a := by
          rw [hg1 _ h]
          exact List.getElem_mem h
        rcases List.mem_append.1 hm with h' | h'
        · exact hnr h1.runs h'
        · exact hnl h2a.runs h'
      simp only [List.length_append] at hpos hN hg1 hg2
      refine ⟨L1.length, r1, r2, L2a, h2a, hpos, fun j hj => ?_, fun j hj => ?_,
        fun j hj1 hj2 => ?_⟩
      · rw [hg1 _ (by omega), List.getElem_append_right (by omega)]
        simp
      · rw [hg1 _ (by omega), List.getElem_append_left hj]
        exact hrd h1.runs _ (List.getElem_mem hj)
      · have hj : j - (L1.length + L2a.length) < L2b.length := by omega
        rw [show j = L1.length + L2a.length + (j - (L1.length + L2a.length)) by omega, hg2 _ hj]
        exact (accessRule fun _ _ x => x = P.done t).runs h2b rfl _ (List.getElem_mem hj)

/-- A write of `start u` is in a thread that starts `u`. -/
theorem start_thread (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {w : X.E} {u : Nat}
    (hww : X.IsWrite w) (hwl : X.loc w = P.start u) (hu : u < P.count) :
    ∃ p ∈ thr ps tps (X.thread w), u ∈ p.ids := by
  have hc : (P.toProg.code (X.thread w)).Accesses fun k _ x =>
      k = .write → x = P.start u → ∃ p ∈ thr ps tps (X.thread w), u ∈ p.ids := by
    cases X.thread w with
    | zero => exact (src_starts hF 0).mono fun _ _ _ h hk hx => h hk u hx
    | succ t =>
      rw [code_succ]
      unfold taskCode
      split
      · refine ⟨nofun, (src_starts hF (t + 1)).mono fun _ _ _ h hk hx => h hk u hx, fun _ hx => ?_⟩
        simp only [start, done] at hx
        omega
      · trivial
  have h := (accessRule _).trace (r.trace w) hc
  simp only [accessRule, r.lab_eq w] at h
  exact h hww hwl

/-- Task `t` read `start t` from a write of the thread that starts it. -/
theorem spawn_of (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {b : X.E} {t : Nat}
    (hb : X.thread b = t + 1) :
    ∃ w, (∃ p ∈ thr ps tps (X.thread w), t ∈ p.ids) ∧ X.IsWrite w ∧ X.loc w = P.start t ∧
      X.hb w b := by
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
  have hws : X.IsSeqCst w := by
    rcases access hF.ok r w with hk | hlt | ⟨hsc, -⟩
    · rw [show X.kind w = .write from hww] at hk
      cases hk
    · rw [hwl] at hlt
      simp only [start] at hlt
      omega
    · exact hsc
  exact ⟨w, start_thread hF r hww hwl htc, hww, hwl, X.hb_of_signal (Or.inl rfl) hww hws hw hes heb⟩

/-- Only task `t` writes `done t`, with an `sc` write. -/
theorem done_thread (hP : P.Ok) (r : P.toProg.Run X) {w : X.E} {t : Nat} (hww : X.IsWrite w)
    (hwl : X.loc w = P.done t) (ht : t < P.count) : X.thread w = t + 1 ∧ X.IsSeqCst w := by
  have hwk : X.kind w = .write := hww
  rcases access hP r w with hk | hlt | ⟨hsc, ⟨-, u, hu, hx⟩ | ⟨hr, -⟩ | ⟨u, -, hwu, ⟨hr, -⟩ | ⟨-, hx⟩⟩⟩
  · rw [hwk] at hk
    cases hk
  · rw [hwl] at hlt
    unfold done at hlt
    omega
  · rw [hwl] at hx
    unfold start done at hx
    omega
  · unfold Execution.IsRead at hr
    rw [hwk] at hr
    cases hr
  · unfold Execution.IsRead at hr
    rw [hwk] at hr
    cases hr
  · rw [hwl] at hx
    unfold done at hx
    obtain rfl : u = t := by omega
    exact ⟨hwu, hsc⟩

/-- A read of `done t` read a write of task `t`, which happens before the code after the read. -/
theorem done_signal (hP : P.Ok) (r : P.toProg.Run X) {j b : X.E} {t : Nat} (hjr : X.IsRead j)
    (hjl : X.loc j = P.done t) (ht : t < P.count) (hjb : j = b ∨ X.po j b) :
    ∃ w, X.thread w = t + 1 ∧ X.IsWrite w ∧ X.loc w = P.done t ∧ X.hb w b := by
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
  obtain ⟨hwt, hws⟩ := done_thread hP r hww hwl ht
  exact ⟨w, hwt, hww, hwl, X.hb_of_signal (Or.inl rfl) hww hws hw hjs hjb⟩

/-- The writes of `start v`, for a task `v` that task `u` starts, are in task `u`. -/
theorem start_in_task (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {u v : Nat}
    (hs : ∃ p ∈ tps u, v ∈ p.ids) {w : X.E} (hww : X.IsWrite w) (hwl : X.loc w = P.start v) :
    X.thread w = u + 1 :=
  hF.one_parent _ (u + 1) _ (start_thread hF r hww hwl (starts_lt hF (q := u + 1) hs)) hs

/-- If `a` happens before each write of `start c`, it happens before each access of each
descendant of `c`. -/
theorem fork_down (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {a : X.E} {c d : Nat}
    (hd : Desc tps c d) :
    (∀ w, X.IsWrite w → X.loc w = P.start c → X.hb a w) → ∀ {b : X.E}, X.thread b = d + 1 →
      X.hb a b := by
  induction hd with
  | refl =>
    intro ha b hb
    obtain ⟨w, -, hww, hwl, hwb⟩ := spawn_of hF r hb
    exact (ha w hww hwl).trans hwb
  | @step u v _ hs _ ih =>
    intro ha b hb
    refine ih (fun w hww hwl => ?_) hb
    obtain ⟨w', -, hww', hwl', hw'b⟩ := spawn_of hF r (start_in_task hF r hs hww hwl)
    exact (ha w' hww' hwl').trans hw'b

/-- A descendant `d` of `c` runs after a write of `start c`. -/
theorem spawn_desc (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {c d : Nat}
    (hd : Desc tps c d) :
    ∀ {e : X.E}, X.thread e = d + 1 → ∃ w, X.IsWrite w ∧ X.loc w = P.start c ∧ X.hb w e := by
  induction hd with
  | refl =>
    intro e he
    obtain ⟨w, -, hww, hwl, hwe⟩ := spawn_of hF r he
    exact ⟨w, hww, hwl, hwe⟩
  | @step u v _ hs _ ih =>
    intro e he
    obtain ⟨w1, hw1, hl1, h1⟩ := ih he
    obtain ⟨w0, -, hw0, hl0, h0⟩ := spawn_of hF r (start_in_task hF r hs hw1 hl1)
    exact ⟨w0, hw0, hl0, h0.trans h1⟩

/-- A block of thread `q` that ran to its end, and wrote `start c` at `w`. Each access of each
descendant of `c` happens before the code `b` after the block. -/
theorem desc_after (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {c d : Nat}
    (hd : Desc tps c d) :
    ∀ {e : X.E}, X.thread e = d + 1 →
    ∀ {q n : Nat} {B : Stmt} {L' : List Label} {regs regs' : Nat → Int},
      B.Below P.flags P.count → B.RegsBelow P.marks → B.NoBlocks →
      Steps (P.lower (.tasksBlock B)) regs L' .skip regs' →
      (∀ m (hm : m < L'.length), r.lab q (n + m) = L'[m]) →
      ∀ {w b : X.E}, X.thread w = q → X.IsWrite w → X.loc w = P.start c → n ≤ r.idx w →
        r.idx w < n + L'.length → X.thread b = q → n + L'.length ≤ r.idx b → X.hb e b := by
  induction hd with
  | refl =>
    intro e he q n B L' regs regs' hBb hBr hBn hs hlab w b hw hww hwl hwn hwL hbq hbL
    exact block_hb hF.ok r hBb hBr hBn hs hlab hw hww hwl hwn hwL he hbq hbL
  | @step u v _ hs hvd ih =>
    intro e he q n B L' regs regs' hBb hBr hBn hrun hlab w b hw hww hwl hwn hwL hbq hbL
    -- The block joins `u`, so the write of `done u` happens before `b`.
    obtain ⟨j, -, hjr, hjl, hjb, hu⟩ :=
      block_join r hBb hBr hBn hrun hlab hw hww hwl hwn hwL hbq hbL
    have huc : u < P.count := spawnIds_lt hBb u hu
    obtain ⟨dw, hdt, hdw, hdl, hdb⟩ := done_signal hF.ok r hjr hjl huc (Or.inr hjb)
    -- Task `u` ran to its end before it wrote `done u`.
    obtain ⟨n0, r0, r0', L0, hs0, hlen0, hlab0, hrd0, hend0⟩ := done_split hF.ok r hdt hdw hdl
    -- Task `u` wrote `start v` in that run.
    obtain ⟨w1, hw1, hl1, -⟩ := spawn_desc hF r hvd he
    have hvc : v < P.count := starts_lt hF (q := u + 1) hs
    have ht1 : X.thread w1 = u + 1 := start_in_task hF r hs hw1 hl1
    have hlt : r.idx w1 < r.idx dw := by
      rcases before_done hF.ok r ht1 hdt hdw hdl with h | h
      · rw [h, hdl] at hl1
        unfold start done at hl1
        omega
      · exact ((r.po_iff _ _).1 h).2
    have hn0 : n0 ≤ r.idx w1 := Nat.le_of_not_lt fun h => by
      have := hrd0 _ h
      rw [lab_of r w1 ht1] at this
      rw [show X.kind w1 = .write from hw1] at this
      cases this
    have hin : r.idx w1 < n0 + L0.length := Nat.lt_of_not_le fun h => by
      have := hend0 _ h (Nat.le_of_lt hlt)
      rw [lab_of r w1 ht1] at this
      simp only at this
      rw [hl1] at this
      unfold start done at this
      omega
    have hg := getElem_of_lab r hlab0 ht1 hn0 hin
    rw [List.getElem?_eq_getElem (by omega)] at hg
    obtain ⟨B1, n1, L1, s1, s1', hB1, hB1b, hs1, hlab1, hn1, hi1, hle1⟩ :=
      src_start_in_block hF (u + 1) hs0 (i := r.idx w1 - n0) (by omega)
        (by rw [Option.some.inj hg]; exact ⟨hw1, hl1⟩)
    have hlab1' : ∀ m (hm : m < L1.length), r.lab (u + 1) (n0 + n1 + m) = L1[m] := fun m hm => by
      have := hlab1 m hm
      rw [List.getElem?_eq_getElem (by omega)] at this
      rw [Nat.add_assoc, hlab0 _ (by omega)]
      exact Option.some.inj this
    exact (ih he hB1b hB1.2.1 hB1.1 hs1 hlab1' ht1 hw1 hl1 (by omega) (by omega) hdt
      (by omega)).trans hdb

/-- An access of task `t` to a place below `flags` follows each rule that the code of the task
follows. -/
theorem task_access (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {e : X.E} {t : Nat}
    (he : X.thread e = t + 1) (hx : X.loc e < P.flags) {q : Kind → Order → Nat → Prop}
    (hq : (P.task t).Accesses q) : q (X.kind e) (X.order e) (X.loc e) := by
  have htc := task_lt r he
  have hc : (P.toProg.code (X.thread e)).Accesses fun k o x => x < P.flags → q k o x := by
    rw [he, code_succ, taskCode_of_lt htc]
    refine ⟨fun h => ?_, lower_below (hF.ok.task t) hq, fun h => ?_⟩
    · simp only [start] at h
      omega
    · simp only [done] at h
      omega
  have h := (accessRule _).trace (r.trace e) hc
  simp only [accessRule, r.lab_eq e] at h
  exact h hx

/-- The third case of `Status` at the position of `a` in a run of its thread: each access of a
descendant of `c` happens before `a`. -/
theorem after_of_status (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {q n : Nat}
    {L : List Label} (hlab : ∀ j (hj : j < L.length), r.lab q (n + j) = L[j]) {a : X.E}
    (ha : X.thread a = q) (hna : n ≤ r.idx a) (haL : r.idx a < n + L.length) {c : Nat} {B : Stmt}
    {n' : Nat} {L' : List Label} {r1 r2 : Nat → Int} (hB : P.BlockOk own tps B)
    (hBb : B.Below P.flags P.count) (hs : Steps (P.lower (.tasksBlock B)) r1 L' .skip r2)
    (hlab' : ∀ i (hi : i < L'.length), L[n' + i]? = some L'[i]) (hn' : n' + L'.length ≤ r.idx a - n)
    {i : Nat} (hi : i < L'.length) (hw : P.StartW c L'[i]) {d : Nat} (hd : Desc tps c d) {b : X.E}
    (hb : X.thread b = d + 1) : X.hb b a := by
  have hlab'' : ∀ k (hk : k < L'.length), r.lab q (n + n' + k) = L'[k] := fun k hk => by
    have := hlab' k hk
    rw [List.getElem?_eq_getElem (by omega)] at this
    rw [Nat.add_assoc, hlab _ (by omega)]
    exact Option.some.inj this
  obtain ⟨w, hwt, hwi⟩ := r.idx_down a (n + n' + i) (by omega)
  have hwq : X.thread w = q := hwt.trans ha
  have hwlab : L'[i] = ⟨X.kind w, X.order w, X.loc w, r.val w, r.isRmw w⟩ := by
    rw [← hlab'' i hi, ← hwi, lab_of r w hwq]
  rw [hwlab] at hw
  exact desc_after hF r hd hb hBb hB.2.1 hB.1 hs hlab'' hwq hw.1 hw.2 (by omega) (by omega) ha
    (by omega)

/-- An access `a` of thread `q`, and an access `b` of a descendant of a task `c` that `q`
starts: happens-before orders them, or they are `Compat`. -/
theorem anc (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {q c d : Nat}
    (hqc : ∃ p ∈ thr ps tps q, c ∈ p.ids) (hd : Desc tps c d) {a b : X.E} (ha : X.thread a = q)
    (hb : X.thread b = d + 1) (hxa : own (X.loc a)) (hxb : own (X.loc b)) :
    X.hb a b ∨ X.hb b a ∨
      Compat (X.kind a) (X.order a) (X.loc a) (X.kind b) (X.order b) (X.loc b) := by
  have hcc := starts_lt hF hqc
  obtain ⟨n, regs, L, hr, hlen, hlab, hrd, hna⟩ := thread_split r (e := a) (Or.inl (hF.le_flags _ hxa))
  rw [ha] at hr hlab hrd
  have hga := getElem_of_lab r hlab ha hna (by omega)
  rcases src_status hF q hr c (m := r.idx a - n) (by omega) with
    hbef | ⟨B, hcB, -, -, hin, -⟩ | ⟨B, n', L', r1, r2, hB, hBb, hs, hlab', hn', i, hi, hw⟩
  · -- `a` comes before each write of `start c`.
    refine Or.inl (fork_down hF r hd (fun w hww hwl => ?_) hb)
    have hwq : X.thread w = q := hF.one_parent _ _ _ (start_thread hF r hww hwl hcc) hqc
    refine Relation.TransGen.single (Or.inl ((r.po_iff a w).2 ⟨ha.trans hwq.symm, ?_⟩))
    refine Nat.lt_of_not_le fun hle => ?_
    have hnw : n ≤ r.idx w := Nat.le_of_not_lt fun h => by
      have := (hrd _ h).1
      rw [lab_of r w hwq, show X.kind w = .write from hww] at this
      cases this
    have := hbef _ _ (getElem_of_lab r hlab hwq hnw (by omega)) ⟨hww, hwl⟩
    omega
  · -- `a` is in the block of `c`.
    exact Or.inr (Or.inr (task_access hF r hb (hF.le_flags _ hxb) ((hin _ hga).1 hxa c hcB d hd)))
  · -- `a` comes after the block of `c`.
    exact Or.inr (Or.inl (after_of_status hF r hlab ha hna (by omega) hB hBb hs hlab' hn' hi hw hd
      hb))

/-- Accesses `a` and `b` of descendants of two tasks `c1` and `c2` that thread `q` starts:
happens-before orders them, or they are `Compat`. -/
theorem sib (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {q c1 c2 d1 d2 : Nat}
    (h1 : ∃ p ∈ thr ps tps q, c1 ∈ p.ids) (h2 : ∃ p ∈ thr ps tps q, c2 ∈ p.ids) (hne : c1 ≠ c2)
    (hd1 : Desc tps c1 d1) (hd2 : Desc tps c2 d2) {a b : X.E} (ha : X.thread a = d1 + 1)
    (hb : X.thread b = d2 + 1) (hxa : own (X.loc a)) (hxb : own (X.loc b)) :
    X.hb a b ∨ X.hb b a ∨
      Compat (X.kind a) (X.order a) (X.loc a) (X.kind b) (X.order b) (X.loc b) ∨
      Compat (X.kind b) (X.order b) (X.loc b) (X.kind a) (X.order a) (X.loc a) := by
  have hxa' : X.loc a < P.flags := hF.le_flags _ hxa
  have hxb' : X.loc b < P.flags := hF.le_flags _ hxb
  obtain ⟨wu, hwuw, hwul, hwua⟩ := spawn_desc hF r hd1 ha
  obtain ⟨wv, hwvw, hwvl, hwvb⟩ := spawn_desc hF r hd2 hb
  have hu : X.thread wu = q :=
    hF.one_parent _ _ _ (start_thread hF r hwuw hwul (starts_lt hF h1)) h1
  have hv : X.thread wv = q :=
    hF.one_parent _ _ _ (start_thread hF r hwvw hwvl (starts_lt hF h2)) h2
  -- One run of thread `q` that holds both writes.
  obtain ⟨w, hwq, hwu, hwv, hws⟩ : ∃ w : X.E, X.thread w = q ∧ r.idx wu ≤ r.idx w ∧
      r.idx wv ≤ r.idx w ∧ (X.IsWrite w ∧ ∃ u < P.count, X.loc w = P.start u) := by
    rcases Nat.le_total (r.idx wu) (r.idx wv) with h | h
    · exact ⟨wv, hv, h, Nat.le_refl _, hwvw, c2, starts_lt hF h2, hwvl⟩
    · exact ⟨wu, hu, Nat.le_refl _, h, hwuw, c1, starts_lt hF h1, hwul⟩
  obtain ⟨n, regs, L, hr, hlen, hlab, hrd, -⟩ := thread_split r (e := w) (Or.inr hws)
  rw [hwq] at hr hlab hrd
  have hn_of {e : X.E} (he : X.thread e = q) (hw : X.IsWrite e) : n ≤ r.idx e :=
    Nat.le_of_not_lt fun h => by
      have := (hrd _ h).1
      rw [lab_of r e he, show X.kind e = .write from hw] at this
      cases this
  have hnu := hn_of hu hwuw
  have hnv := hn_of hv hwvw
  have hgu := getElem_of_lab r hlab hu hnu (by omega)
  have hgv := getElem_of_lab r hlab hv hnv (by omega)
  have hsu : P.StartW c1 ⟨X.kind wu, X.order wu, X.loc wu, r.val wu, r.isRmw wu⟩ := ⟨hwuw, hwul⟩
  have hsv : P.StartW c2 ⟨X.kind wv, X.order wv, X.loc wv, r.val wv, r.isRmw wv⟩ := ⟨hwvw, hwvl⟩
  rcases src_status hF q hr c1 (m := r.idx wv - n) (by omega) with
    hbef | ⟨B, hc1, -, hB, hin, -⟩ | ⟨B, n', L', r1, r2, hB, hBb, hs, hlab', hn', i, hi, hw⟩
  · -- The write for `c2` comes before each write for `c1`.
    have hlt := hbef _ _ hgu hsu
    rcases src_status hF q hr c2 (m := r.idx wu - n) (by omega) with
      hbef' | ⟨B, hc2, -, hB, hin, -⟩ | ⟨B, n', L', r1, r2, hB, hBb, hs, hlab', hn', i, hi, hw⟩
    · have := hbef' _ _ hgv hsv
      omega
    · have hc1 := (hin _ hgu).2 hsu.1 c1 hsu.2
      exact Or.inr (Or.inr (Or.inr (task_access hF r ha hxa'
        ((task_access hF r hb hxb' (hB.2.2.2 c2 hc2 c1 hc1 (Ne.symm hne) d2 d1 hd2 hd1)) hxb))))
    · exact Or.inr (Or.inl ((after_of_status hF r hlab hu hnu (by omega) hB hBb hs hlab' hn' hi hw
        hd2 hb).trans hwua))
  · -- Both tasks are in one block.
    have hc2 := (hin _ hgv).2 hsv.1 c2 hsv.2
    exact Or.inr (Or.inr (Or.inl (task_access hF r hb hxb'
      ((task_access hF r ha hxa' (hB.2.2.2 c1 hc1 c2 hc2 hne d1 d2 hd1 hd2)) hxa))))
  · -- The write for `c2` comes after the block of `c1`.
    exact Or.inl ((after_of_status hF r hlab hv hnv (by omega) hB hBb hs hlab' hn' hi hw hd1
      ha).trans hwvb)

/-! ## The tree of tasks -/

/-- Happens-before orders `a` and `b`, or they are `Compat`. -/
def Ordered (X : Execution) (a b : X.E) : Prop :=
  X.hb a b ∨ X.hb b a ∨
    Compat (X.kind a) (X.order a) (X.loc a) (X.kind b) (X.order b) (X.loc b) ∨
    Compat (X.kind b) (X.order b) (X.loc b) (X.kind a) (X.order a) (X.loc a)

theorem Ordered.symm {a b : X.E} (h : Ordered X a b) : Ordered X b a := by
  rcases h with h | h | h | h
  · exact Or.inr (Or.inl h)
  · exact Or.inl h
  · exact Or.inr (Or.inr (Or.inr h))
  · exact Or.inr (Or.inr (Or.inl h))

theorem Ordered.of_anc {a b : X.E}
    (h : X.hb a b ∨ X.hb b a ∨
      Compat (X.kind a) (X.order a) (X.loc a) (X.kind b) (X.order b) (X.loc b)) :
    Ordered X a b := by
  rcases h with h | h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inl h))

/-- Each task that runs is a descendant of a task that the main code starts. -/
theorem under_main (hF : P.Footprints own tps ps) (r : P.toProg.Run X) (t : Nat) :
    ∀ {e : X.E}, X.thread e = t + 1 → ∃ c, (∃ p ∈ ps, c ∈ p.ids) ∧ Desc tps c t := by
  induction t using Nat.strongRecOn with
  | _ t ih =>
    intro e he
    obtain ⟨w, hws, -, -, -⟩ := spawn_of hF r he
    cases hwt : X.thread w with
    | zero =>
      rw [hwt] at hws
      exact ⟨t, hws, .refl t⟩
    | succ t' =>
      rw [hwt] at hws
      obtain ⟨c, hc, hd⟩ := ih t' (hF.order t' t hws) hwt
      exact ⟨c, hc, hd.tail hws⟩

/-- Accesses of two descendants of `c`: happens-before orders them, or they are `Compat`. -/
theorem lca (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {c dA : Nat}
    (hA : Desc tps c dA) :
    ∀ {dB : Nat}, Desc tps c dB → dA ≠ dB → ∀ {a b : X.E}, X.thread a = dA + 1 →
      X.thread b = dB + 1 → own (X.loc a) → own (X.loc b) → Ordered X a b := by
  induction hA with
  | refl c =>
    intro dB hB hne a b ha hb hxa hxb
    cases hB with
    | refl => exact absurd rfl hne
    | step hs hv => exact .of_anc (anc hF r (q := c + 1) hs hv ha hb hxa hxb)
  | @step u v1 _ hs1 hv1 ih =>
    intro dB hB hne a b ha hb hxa hxb
    cases hB with
    | refl => exact (Ordered.of_anc (anc hF r (q := u + 1) hs1 hv1 hb ha hxb hxa)).symm
    | @step _ v2 _ hs2 hv2 =>
      by_cases h : v1 = v2
      · subst h
        exact ih hv2 hne ha hb hxa hxb
      · exact sib hF r (q := u + 1) hs1 hs2 h hv1 hv2 ha hb hxa hxb

/-- An access `a` of the main code, below `flags`, and an access `b` of a task: happens-before
orders them, or `a` is in a run of a block of the main code. -/
theorem main_task (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {a b : X.E} {t : Nat}
    (ha : X.thread a = 0) (hb : X.thread b = t + 1) (hxa : X.loc a < P.flags) :
    X.hb a b ∨ X.hb b a ∨ ∃ B, (∃ rep, Phase.block B rep ∈ ps) ∧ P.BlockOk own tps B ∧
      ∃ n L' regs, Runs (P.lower (.tasksBlock B)) regs L' ∧
        (∀ i (hi : i < L'.length), r.lab 0 (n + i) = L'[i]) ∧ n ≤ r.idx a ∧
        r.idx a < n + L'.length := by
  obtain ⟨c, hqc, hd⟩ := under_main hF r t hb
  have hcc := starts_lt hF (q := 0) hqc
  obtain ⟨n, regs, L, hr, hlen, hlab, hrd, hna⟩ := thread_split r (e := a) (Or.inl hxa)
  rw [ha] at hr hlab hrd
  rcases src_status hF 0 hr c (m := r.idx a - n) (by omega) with
    hbef | ⟨B, -, hmem, hB, -, n', L', r1, hs, hlab', hn1, hn2⟩ |
      ⟨B, n', L', r1, r2, hB, hBb, hs, hlab', hn', i, hi, hw⟩
  · -- `a` comes before each write of `start c`.
    refine Or.inl (fork_down hF r hd (fun w hww hwl => ?_) hb)
    have hwq : X.thread w = 0 := hF.one_parent _ _ _ (start_thread hF r hww hwl hcc) hqc
    refine Relation.TransGen.single (Or.inl ((r.po_iff a w).2 ⟨ha.trans hwq.symm, ?_⟩))
    refine Nat.lt_of_not_le fun hle => ?_
    have hnw : n ≤ r.idx w := Nat.le_of_not_lt fun h => by
      have := (hrd _ h).1
      rw [lab_of r w hwq, show X.kind w = .write from hww] at this
      cases this
    have := hbef _ _ (getElem_of_lab r hlab hwq hnw (by omega)) ⟨hww, hwl⟩
    omega
  · -- `a` is in a run of the block of `c`.
    refine Or.inr (Or.inr ⟨B, hmem, hB, n + n', L', r1, hs, fun i hi => ?_, by omega, by omega⟩)
    obtain ⟨hlt, heq⟩ := List.getElem?_eq_some_iff.1 (hlab' i hi)
    rw [Nat.add_assoc, hlab _ hlt, heq]
  · -- `a` comes after the block of `c`.
    exact Or.inr (Or.inl (after_of_status hF r hlab ha hna (by omega) hB hBb hs hlab' hn' hi hw
      hd hb))

/-! ## The executions of a program that follows the rule -/

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
theorem atomic_of_flags (hF : P.Footprints own tps ps) (r : P.toProg.Run X) {e : X.E}
    (he : X.IsAccess e) (hx : P.flags ≤ X.loc e) : X.IsAtomic e := by
  rcases access hF.ok r e with hk | hlt | ⟨hsc, -⟩
  · simp [Execution.IsAccess, Execution.IsRead, Execution.IsWrite, hk] at he
  · omega
  · unfold Execution.IsAtomic
    rw [hsc]
    nofun

/-- A program that follows the rule has no data race at a place of `own`, in each execution. -/
theorem no_race_below (hF : P.Footprints own tps ps) (hX : P.toProg.Execs X) {a b : X.E}
    (hr : X.DataRace a b) (hxa : own (X.loc a)) : False := by
  obtain ⟨r⟩ := hX
  obtain ⟨hc, hthr, hab, hba, hat⟩ := hr
  have nc := not_compat hc hat
  obtain ⟨hne, hacc_a, hacc_b, hl, hw⟩ := hc
  have nc' := not_compat ⟨hne.symm, hacc_b, hacc_a, hl.symm, hw.symm⟩ fun h => hat ⟨h.2, h.1⟩
  have hxb : own (X.loc b) := hl ▸ hxa
  suffices h : Ordered X a b by
    rcases h with h | h | h | h
    · exact hab h
    · exact hba h
    · exact nc h
    · exact nc' h
  cases hta : X.thread a with
  | zero =>
    cases htb : X.thread b with
    | zero => exact absurd (hta.trans htb.symm) hthr
    | succ t =>
      obtain ⟨c, hc, hd⟩ := under_main hF r t htb
      exact .of_anc (anc hF r (q := 0) hc hd hta htb hxa hxb)
  | succ u =>
    cases htb : X.thread b with
    | zero =>
      obtain ⟨c, hc, hd⟩ := under_main hF r u hta
      exact (Ordered.of_anc (anc hF r (q := 0) hc hd htb hta hxb hxa)).symm
    | succ v =>
      have huv : u ≠ v := fun h => hthr (by rw [hta, htb, h])
      obtain ⟨c1, hc1, hd1⟩ := under_main hF r u hta
      obtain ⟨c2, hc2, hd2⟩ := under_main hF r v htb
      by_cases h : c1 = c2
      · subst h
        exact lca hF r hd1 hd2 huv hta htb hxa hxb
      · exact sib hF r (q := 0) hc1 hc2 h hd1 hd2 hta htb hxa hxb

/-- A program that follows the rule for the places below `flags` has no data race, in each
execution. The places from `flags` up have only atomic accesses. -/
theorem raceFree (hF : P.Footprints (· < P.flags) tps ps) (hX : P.toProg.Execs X) :
    X.RaceFree := by
  intro a b hr
  obtain ⟨r⟩ := hX
  obtain ⟨⟨-, hacc_a, hacc_b, hl, -⟩, -, -, -, hat⟩ := id hr
  refine no_race_below hF ⟨r⟩ hr (Nat.lt_of_not_le fun hle => ?_)
  exact hat ⟨atomic_of_flags hF r hacc_a hle, atomic_of_flags hF r hacc_b (hl ▸ hle)⟩

/-- DRF-SC from the rule: for a program that follows it, each finite execution that the model
allows is an interleaving. -/
theorem drf_sc_of_footprints (hF : P.Footprints (· < P.flags) tps ps) (hm : P.main.SeqCstOnly)
    (ht : ∀ t, (P.task t).SeqCstOnly) (hX : P.toProg.Execs X) (wf : X.WellFormed)
    (hfin : X.Finite) (hc : X.Consistent) : X.SeqConsistent :=
  P.drf_sc hm ht (fun _ hY _ _ => raceFree hF hY) hX wf hfin hc

end TaskProg

end Vx
