import VxMemoryModel.Tasks

/-!
# Tasks blocks

`tasks { s }` lowers to three parts. The first part clears the marks of the tasks that `s` can
start. Then `s` runs, and each `spawn` in it sets the mark of its task. At the end, the block
joins each task that has its mark set. So the block waits only for the tasks that it started.

`block_hb` shows that each task that a block started happens before the code after the block.
The body of the block must not hold another `tasks` block. A task can hold its own blocks.
-/

namespace Vx

/-- `Steps s regs L s' regs'`: the code `s` makes the accesses `L` and reaches the code `s'` with
the registers `regs'`. -/
inductive Steps : Stmt → (Nat → Int) → List Label → Stmt → (Nat → Int) → Prop
  | refl {s regs} : Steps s regs [] s regs
  | silent {s regs s' regs' L s'' regs''} :
    Step s regs none s' regs' → Steps s' regs' L s'' regs'' → Steps s regs L s'' regs''
  | visible {s regs s' regs' a L s'' regs''} :
    Step s regs (some a) s' regs' → Steps s' regs' L s'' regs'' → Steps s regs (a :: L) s'' regs''

theorem Steps.runs {s regs L s' regs'} (h : Steps s regs L s' regs') : Runs s regs L := by
  induction h with
  | refl => exact .stop
  | silent hs _ ih => exact .silent hs ih
  | visible hs _ ih => exact .visible hs ih

theorem Steps.skip {regs L s' regs'} (h : Steps .skip regs L s' regs') : L = [] ∧ regs' = regs := by
  cases h with
  | refl => exact ⟨rfl, rfl⟩
  | silent hs => cases hs
  | visible hs => cases hs

/-- A run of `seq s t` to its end runs `s` to its end and then `t` to its end. -/
theorem Steps.seq_done {s t regs L regs''} (h : Steps (.seq s t) regs L .skip regs'') :
    ∃ L1 L2 regs', Steps s regs L1 .skip regs' ∧ Steps t regs' L2 .skip regs'' ∧ L = L1 ++ L2 := by
  generalize hx : Stmt.seq s t = x at h
  generalize hy : Stmt.skip = y at h
  induction h generalizing s with
  | refl =>
    subst hy
    cases hx
  | silent hs h' ih =>
    subst hx hy
    cases hs with
    | seq_skip => exact ⟨[], _, _, .refl, h', rfl⟩
    | seq hs1 =>
      obtain ⟨L1, L2, r', h1, h2, rfl⟩ := ih rfl rfl
      exact ⟨L1, L2, r', .silent hs1 h1, h2, rfl⟩
  | visible hs h' ih =>
    subst hx hy
    cases hs with
    | seq hs1 =>
      obtain ⟨L1, L2, r', h1, h2, rfl⟩ := ih rfl rfl
      exact ⟨_ :: L1, L2, r', .visible hs1 h1, h2, rfl⟩

def Expr.RegsBelow (m : Nat) : Expr → Prop
  | .const _ => True
  | .reg r => r < m
  | .add a b | .sub a b | .eq a b => a.RegsBelow m ∧ b.RegsBelow m

theorem Expr.eval_update {m x : Nat} {v : Int} {regs : Nat → Int} {e : Expr} (h : e.RegsBelow m)
    (hx : m ≤ x) : e.eval (update regs x v) = e.eval regs := by
  induction e with
  | const => rfl
  | reg r =>
    have : r ≠ x := by simp only [Expr.RegsBelow] at h; omega
    simp [Expr.eval, update, this]
  | add _ _ ih1 ih2 | sub _ _ ih1 ih2 | eq _ _ ih1 ih2 => simp only [Expr.eval, ih1 h.1, ih2 h.2]

/-- The code uses only the registers below `m`. -/
def Stmt.RegsBelow (m : Nat) : Stmt → Prop
  | load r _ _ => r < m
  | store _ e _ => e.RegsBelow m
  | fetchAdd r _ e _ | assign r e => r < m ∧ e.RegsBelow m
  | cas r _ e n _ => r < m ∧ e.RegsBelow m ∧ n.RegsBelow m
  | loadAt r _ _ i _ => r < m ∧ i.RegsBelow m
  | storeAt _ _ i e _ => i.RegsBelow m ∧ e.RegsBelow m
  | waitAt _ _ i | spawnAt _ _ i | joinAt _ _ i => i.RegsBelow m
  | assignAt base len i e => base + len ≤ m ∧ i.RegsBelow m ∧ e.RegsBelow m
  | seq s t => s.RegsBelow m ∧ t.RegsBelow m
  | ite c s t => c.RegsBelow m ∧ s.RegsBelow m ∧ t.RegsBelow m
  | loop c s => c.RegsBelow m ∧ s.RegsBelow m
  | tasksBlock s => s.RegsBelow m
  | skip | rmwStore .. | fence _ | wait _ => True

/-- `s` holds no `tasks` block. -/
def Stmt.NoBlocks : Stmt → Prop
  | tasksBlock _ => False
  | seq s t | ite _ s t => s.NoBlocks ∧ t.NoBlocks
  | loop _ s => s.NoBlocks
  | _ => True

namespace TaskProg

variable {P : TaskProg}

/-- Code that comes from the lowering of the body of a block: it uses the places below `flags`,
and only a `spawn` writes a mark. -/
inductive Clean (P : TaskProg) : Stmt → Prop
  | skip : Clean P .skip
  | load {r x o} : r < P.marks → x < P.flags → Clean P (.load r x o)
  | store {x e o} : x < P.flags → Clean P (.store x e o)
  | fetchAdd {r x e o} : r < P.marks → x < P.flags → Clean P (.fetchAdd r x e o)
  | cas {r x e n o} : r < P.marks → x < P.flags → Clean P (.cas r x e n o)
  | rmwStore {x v o} : x < P.flags → Clean P (.rmwStore x v o)
  | fence {o} : Clean P (.fence o)
  | assign {r e} : r < P.marks → Clean P (.assign r e)
  | wait {x} : Clean P (.wait x)
  | loadAt {r base len i o} : r < P.marks → base + len ≤ P.flags → Clean P (.loadAt r base len i o)
  | storeAt {base len i e o} : base + len ≤ P.flags → Clean P (.storeAt base len i e o)
  | waitAt {base len i} : Clean P (.waitAt base len i)
  | assignAt {base len i e} : base + len ≤ P.marks → Clean P (.assignAt base len i e)
  | seq {s t} : Clean P s → Clean P t → Clean P (.seq s t)
  | ite {c s t} : Clean P s → Clean P t → Clean P (.ite c s t)
  | loop {c s} : Clean P s → Clean P (.loop c s)
  | spawn {base len i} : i.RegsBelow P.marks →
    Clean P (.seq (.assignAt (P.marks + base) len i (.const 1))
      (.storeAt (P.start base) len i (.const 1) .sc))

/-- The task `base + i` has its mark set. -/
def Marked (P : TaskProg) (regs : Nat → Int) (base len : Nat) (i : Expr) : Prop :=
  ∀ j : Nat, i.eval regs = (j : Int) → j < len → regs (P.marks + base + j) = 1

/-- The states of lowered code: the next write of a `start` flag comes after its mark. -/
inductive Good (P : TaskProg) (regs : Nat → Int) : Stmt → Prop
  | clean {s} : Clean P s → Good P regs s
  | seq {s t} : Good P regs s → Clean P t → Good P regs (.seq s t)
  | marked {base len i} : Marked P regs base len i →
    Good P regs (.seq .skip (.storeAt (P.start base) len i (.const 1) .sc))
  | marked' {base len i} : Marked P regs base len i →
    Good P regs (.storeAt (P.start base) len i (.const 1) .sc)

theorem update_keeps {regs : Nat → Int} {r : Nat} {v : Int} (hr : r < P.marks) :
    ∀ x, P.marks ≤ x → regs x = 1 → update regs r v x = 1 := by
  intro x hx h1
  have : x ≠ r := by omega
  simp [update, this, h1]

theorem update_one_keeps {regs : Nat → Int} {r : Nat} :
    ∀ x, P.marks ≤ x → regs x = 1 → update regs r 1 x = 1 := by
  intro x _ h1
  by_cases h : x = r <;> simp [update, h, h1]

/-- One step keeps `Good`, never clears a mark, and writes a `start` flag only when its mark is
set. -/
theorem Good.step {regs s a s' regs'} (h : Step s regs a s' regs') (hg : Good P regs s) :
    Good P regs' s' ∧ (∀ x, P.marks ≤ x → regs x = 1 → regs' x = 1) ∧
      ∀ l, a = some l → l.kind = .write → ∀ u, l.loc = P.start u → regs' (P.marks + u) = 1 := by
  induction h with
  | load =>
    rcases hg with ⟨⟨⟩⟩
    rename_i hr _
    exact ⟨.clean .skip, update_keeps hr, fun l hl hk => by cases hl; cases hk⟩
  | store =>
    rcases hg with ⟨⟨⟩⟩
    rename_i hx
    refine ⟨.clean .skip, fun _ _ h => h, fun l hl _ u hu => ?_⟩
    cases hl
    simp only [start] at hu
    omega
  | fetchAdd =>
    rcases hg with ⟨⟨⟩⟩
    rename_i hr hx
    exact ⟨.clean (.rmwStore hx), update_keeps hr, fun l hl hk => by cases hl; cases hk⟩
  | cas_eq =>
    rcases hg with ⟨⟨⟩⟩
    rename_i hr hx
    exact ⟨.clean (.rmwStore hx), update_keeps hr, fun l hl hk => by cases hl; cases hk⟩
  | cas_ne =>
    rcases hg with ⟨⟨⟩⟩
    rename_i hr _
    exact ⟨.clean .skip, update_keeps hr, fun l hl hk => by cases hl; cases hk⟩
  | rmwStore =>
    rcases hg with ⟨⟨⟩⟩
    rename_i hx
    refine ⟨.clean .skip, fun _ _ h => h, fun l hl _ u hu => ?_⟩
    cases hl
    simp only [start] at hu
    omega
  | fence => exact ⟨.clean .skip, fun _ _ h => h, fun l hl hk => by cases hl; cases hk⟩
  | assign =>
    rcases hg with ⟨⟨⟩⟩
    rename_i hr
    exact ⟨.clean .skip, update_keeps hr, nofun⟩
  | seq_skip =>
    rcases hg with ⟨⟨⟩⟩ | ⟨_, ht⟩ | ⟨hm⟩
    · rename_i ht
      exact ⟨.clean ht, fun _ _ h => h, nofun⟩
    · exact ⟨.clean ht, fun _ _ h => h, nofun⟩
    · exact ⟨.marked' hm, fun _ _ h => h, nofun⟩
  | seq hs ih =>
    rcases hg with ⟨⟨⟩⟩ | ⟨hg1, ht⟩ | ⟨hm⟩
    · rename_i h1 h2
      obtain ⟨g, hk, hw⟩ := ih (.clean h1)
      exact ⟨.seq g h2, hk, hw⟩
    · rename_i hi
      cases hs with
      | assignAt j hj hjl =>
        refine ⟨.marked fun j' hj' _ => ?_, update_one_keeps, nofun⟩
        rw [Expr.eval_update hi (by omega), hj] at hj'
        obtain rfl : j' = j := by exact_mod_cast hj'.symm
        simp [update, Expr.eval]
    · obtain ⟨g, hk, hw⟩ := ih hg1
      exact ⟨.seq g ht, hk, hw⟩
    · cases hs
  | ite_true =>
    rcases hg with ⟨⟨⟩⟩
    rename_i h1 _
    exact ⟨.clean h1, fun _ _ h => h, nofun⟩
  | ite_false =>
    rcases hg with ⟨⟨⟩⟩
    rename_i _ h2
    exact ⟨.clean h2, fun _ _ h => h, nofun⟩
  | loop =>
    rcases hg with ⟨⟨⟩⟩
    rename_i h1
    exact ⟨.clean (.ite (.seq h1 (.loop h1)) .skip), fun _ _ h => h, nofun⟩
  | wait v =>
    refine ⟨.clean ?_, fun _ _ h => h, fun l hl hk => by cases hl; cases hk⟩
    by_cases hv : v = 0 <;> simp only [hv, ↓reduceIte] <;>
      first | exact .wait | exact .skip
  | loadAt =>
    rcases hg with ⟨⟨⟩⟩
    rename_i hr _
    exact ⟨.clean .skip, update_keeps hr, fun l hl hk => by cases hl; cases hk⟩
  | storeAt j hj hjl =>
    rcases hg with ⟨⟨⟩⟩ | ⟨hm⟩
    · rename_i hx
      refine ⟨.clean .skip, fun _ _ h => h, fun l hl _ u hu => ?_⟩
      cases hl
      simp only [start] at hu
      omega
    · rename_i base hm
      refine ⟨.clean .skip, fun _ _ h => h, fun l hl _ u hu => ?_⟩
      cases hl
      have hu' : u = base + j := by simp only [start] at hu; omega
      subst hu'
      rw [← Nat.add_assoc]
      exact hm j hj hjl
  | waitAt _ v =>
    refine ⟨.clean ?_, fun _ _ h => h, fun l hl hk => by cases hl; cases hk⟩
    by_cases hv : v = 0 <;> simp only [hv, ↓reduceIte] <;>
      first | exact .waitAt | exact .skip
  | assignAt j _ hjl =>
    rcases hg with ⟨⟨⟩⟩
    rename_i hb
    exact ⟨.clean .skip, update_keeps (by omega), nofun⟩

theorem Good.steps {regs s L s' regs'} (h : Steps s regs L s' regs') (hg : Good P regs s) :
    Good P regs' s' ∧ (∀ x, P.marks ≤ x → regs x = 1 → regs' x = 1) ∧
      ∀ l ∈ L, l.kind = .write → ∀ u, l.loc = P.start u → regs' (P.marks + u) = 1 := by
  induction h with
  | refl => exact ⟨hg, fun _ _ h => h, nofun⟩
  | silent hs _ ih =>
    obtain ⟨g1, k1, -⟩ := hg.step hs
    obtain ⟨g2, k2, w2⟩ := ih g1
    exact ⟨g2, fun x hx h => k2 x hx (k1 x hx h), w2⟩
  | visible hs _ ih =>
    obtain ⟨g1, k1, w1⟩ := hg.step hs
    obtain ⟨g2, k2, w2⟩ := ih g1
    refine ⟨g2, fun x hx h => k2 x hx (k1 x hx h), fun l hl hk u hu => ?_⟩
    rcases List.mem_cons.1 hl with rfl | hl
    · exact k2 _ (by omega) (w1 l rfl hk u hu)
    · exact w2 l hl hk u hu

theorem clean_lower {s : Stmt} (hb : s.Below P.flags P.count) (hr : s.RegsBelow P.marks)
    (hn : s.NoBlocks) : Clean P (P.lower s) := by
  induction s with
  | skip => exact .skip
  | load => exact .load hr hb
  | store => exact .store hb
  | fetchAdd => exact .fetchAdd hr.1 hb
  | cas => exact .cas hr.1 hb
  | rmwStore => exact .rmwStore hb
  | fence => exact .fence
  | assign => exact .assign hr.1
  | seq _ _ ih1 ih2 => exact .seq (ih1 hb.1 hr.1 hn.1) (ih2 hb.2 hr.2 hn.2)
  | ite _ _ _ ih1 ih2 => exact .ite (ih1 hb.1 hr.2.1 hn.1) (ih2 hb.2 hr.2.2 hn.2)
  | loop _ _ ih => exact .loop (ih hb hr.2 hn)
  | wait => exact .wait
  | loadAt => exact .loadAt hr.1 hb
  | storeAt => exact .storeAt hb
  | waitAt => exact .waitAt
  | spawnAt => exact .spawn hr
  | joinAt => exact .waitAt
  | assignAt => exact .assignAt hr.1
  | tasksBlock => exact hn.elim

/-- The lowered code writes a `start` flag only for a task that it can start. -/
theorem lower_spawnIds {s : Stmt} (hb : s.Below P.flags P.count) (hn : s.NoBlocks) :
    (P.lower s).Accesses fun k _ x => k = .write → P.flags ≤ x → ∃ u ∈ s.spawnIds, x = P.start u := by
  induction s with
  | store | rmwStore => exact fun _ h => absurd h (by simp only [Stmt.Below] at hb; omega)
  | fetchAdd | cas => exact ⟨nofun, fun _ h => absurd h (by simp only [Stmt.Below] at hb; omega)⟩
  | load | fence | wait => exact nofun
  | loadAt | waitAt => exact fun _ _ => nofun
  | joinAt => exact fun _ _ => nofun
  | storeAt => exact fun _ _ _ h => absurd h (by simp only [Stmt.Below] at hb; omega)
  | spawnAt base =>
    refine ⟨trivial, fun j hj _ _ => ⟨base + j, List.mem_map.2 ⟨j, List.mem_range.2 hj, rfl⟩, ?_⟩⟩
    simp only [start]
    omega
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 =>
    refine ⟨(ih1 hb.1 hn.1).mono ?_, (ih2 hb.2 hn.2).mono ?_⟩
    · exact fun _ _ _ h hk hx => (h hk hx).imp fun _ hu => ⟨List.mem_append_left _ hu.1, hu.2⟩
    · exact fun _ _ _ h hk hx => (h hk hx).imp fun _ hu => ⟨List.mem_append_right _ hu.1, hu.2⟩
  | loop _ _ ih => exact ih hb hn
  | tasksBlock => exact hn.elim
  | skip | assign | assignAt => trivial

theorem clearMarks_silent {ids : List Nat} {regs L regs'}
    (h : Steps (P.clearMarks ids) regs L .skip regs') : L = [] := by
  induction ids generalizing regs L with
  | nil => exact h.skip.1
  | cons _ _ ih =>
    obtain ⟨L1, L2, _, h1, h2, rfl⟩ := Steps.seq_done h
    rw [ih h2]
    cases h1 with
    | silent hs h1 =>
      cases hs
      rw [h1.skip.1]
      rfl
    | visible hs => cases hs

theorem waitAt_done {x regs L regs'} (h : Steps (.waitAt x 1 (.const 0)) regs L .skip regs') :
    regs' = regs ∧ ∃ l ∈ L, l.kind = .read ∧ l.loc = x := by
  generalize hs : Stmt.waitAt x 1 (.const 0) = s at h
  generalize hk : Stmt.skip = k at h
  induction h with
  | refl =>
    subst hk
    cases hs
  | silent hs1 =>
    subst hs
    cases hs1
  | visible hs1 h' ih =>
    subst hs hk
    cases hs1 with
    | waitAt j v _ hj =>
      obtain rfl : j = 0 := by omega
      refine ⟨?_, _, List.mem_cons_self, rfl, rfl⟩
      by_cases hv : v = 0
      · simp only [hv, ↓reduceIte] at h'
        exact (ih (by simp [hv]) rfl).1
      · simp only [hv, ↓reduceIte] at h'
        exact h'.skip.2

theorem joinOne_done {y x regs L regs'}
    (h : Steps (.ite (.reg y) (.waitAt x 1 (.const 0)) .skip) regs L .skip regs') :
    regs' = regs ∧ (regs y ≠ 0 → ∃ l ∈ L, l.kind = .read ∧ l.loc = x) := by
  cases h with
  | silent hs h =>
    cases hs with
    | ite_true => exact ⟨(waitAt_done h).1, fun _ => (waitAt_done h).2⟩
    | ite_false hc => exact ⟨h.skip.2, fun hy => absurd hc hy⟩
  | visible hs => cases hs

/-- The end of a block reads `done t` for each task `t` that has its mark set. -/
theorem joinMarked_reads {ids : List Nat} {regs L regs'}
    (h : Steps (P.joinMarked ids) regs L .skip regs') :
    ∀ t ∈ ids, regs (P.marks + t) ≠ 0 → ∃ l ∈ L, l.kind = .read ∧ l.loc = P.done t := by
  induction ids generalizing regs L with
  | nil => nofun
  | cons t0 _ ih =>
    obtain ⟨L1, L2, _, h1, h2, rfl⟩ := Steps.seq_done h
    obtain ⟨rfl, h1⟩ := joinOne_done h1
    intro t ht hm
    rcases List.mem_cons.1 ht with rfl | ht
    · obtain ⟨l, hl, hl'⟩ := h1 hm
      exact ⟨l, List.mem_append_left _ hl, hl'⟩
    · obtain ⟨l, hl, hl'⟩ := ih h2 t ht hm
      exact ⟨l, List.mem_append_right _ hl, hl'⟩

/-- The end of a block joins each task that the block started.

The labels of thread `p` from the position `n` on are a run `L` of the block to its end. The
event `w` is a `spawn` of task `t` in that run, and `b` comes after the run. Then thread `p` reads
`done t` before `b`, and `t` is a task that the body can start. -/
theorem block_join {X : Execution} (r : P.toProg.Run X) {s : Stmt}
    (hb : s.Below P.flags P.count) (hr : s.RegsBelow P.marks) (hn : s.NoBlocks)
    {p n : Nat} {regs regs' : Nat → Int} {L : List Label}
    (hrun : Steps (P.lower (.tasksBlock s)) regs L .skip regs')
    (hlab : ∀ m (hm : m < L.length), r.lab p (n + m) = L[m])
    {t : Nat} {w b : X.E} (hw : X.thread w = p) (hww : X.IsWrite w)
    (hwl : X.loc w = P.start t) (hwn : n ≤ r.idx w) (hwL : r.idx w < n + L.length)
    (hbp : X.thread b = p) (hbL : n + L.length ≤ r.idx b) :
    ∃ j, X.thread j = p ∧ X.IsRead j ∧ X.loc j = P.done t ∧ X.po j b ∧ t ∈ s.spawnIds := by
  obtain ⟨L0, Lrest, ra, h0, hrest, rfl⟩ := Steps.seq_done hrun
  obtain rfl := clearMarks_silent h0
  obtain ⟨Lb, Le, rb, hbody, hend, rfl⟩ := Steps.seq_done hrest
  simp only [List.nil_append] at hlab hwL hbL
  -- The label of the `spawn`.
  have hwlab : (Lb ++ Le)[r.idx w - n]'(by simp only [List.length_append] at hwL ⊢; omega) =
      ⟨X.kind w, X.order w, X.loc w, r.val w, r.isRmw w⟩ := by
    rw [← hlab _ (by simp only [List.length_append] at hwL ⊢; omega), ← r.lab_eq w, hw]
    congr 1
    omega
  -- The end of the block only reads, so the `spawn` is in the body.
  have hin : r.idx w - n < Lb.length := by
    refine Nat.lt_of_not_le fun hle => ?_
    have hmem : (Lb ++ Le)[r.idx w - n]'(by simp only [List.length_append] at hwL ⊢; omega) ∈ Le := by
      rw [List.getElem_append_right hle]
      exact List.getElem_mem _
    have hread := (accessRule fun k _ _ => k = .read).runs hend.runs
      (joinMarked_accesses fun _ _ => rfl) _ hmem
    rw [hwlab] at hread
    exact absurd (hww.symm.trans hread) nofun
  have hmem : (Lb ++ Le)[r.idx w - n]'(by simp only [List.length_append] at hwL ⊢; omega) ∈ Lb := by
    rw [List.getElem_append_left hin]
    exact List.getElem_mem _
  rw [hwlab] at hmem
  -- The `spawn` set the mark of task `t`, and `t` is a task that the body can start.
  have hmark := (Good.steps hbody (.clean (clean_lower hb hr hn))).2.2 _ hmem hww t hwl
  obtain ⟨u, hu, hut⟩ := (accessRule _).runs hbody.runs (lower_spawnIds hb hn) _ hmem hww
    (by simp only [hwl, start]; omega)
  have htu : t = u := by
    simp only [hwl, start] at hut
    omega
  subst htu
  -- So the end of the block reads `done t`, before `b`.
  obtain ⟨l, hl, hlk, hll⟩ := joinMarked_reads hend t hu (by rw [hmark]; decide)
  obtain ⟨q, hq, rfl⟩ := List.mem_iff_getElem.1 hl
  obtain ⟨j, hjt, hji⟩ := r.idx_down b (n + (Lb.length + q))
    (by simp only [List.length_append] at hbL; omega)
  have hjlab := r.lab_eq j
  rw [hjt, hbp, hji, hlab _ (by simp only [List.length_append]; omega),
    List.getElem_append_right (by omega)] at hjlab
  simp only [Nat.add_sub_cancel_left] at hjlab
  have hjk : X.kind j = .read := by rw [← hlk, hjlab]
  have hjl : X.loc j = P.done t := by rw [← hll, hjlab]
  have hjb : X.po j b := (r.po_iff j b).2 ⟨hjt, by simp only [List.length_append] at hbL; omega⟩
  exact ⟨j, hjt.trans hbp, hjk, hjl, hjb, hu⟩

/-- A task that a block started happens before the code after the block: each access `a` of
task `t` happens before `b`. -/
theorem block_hb (hP : P.Ok) {X : Execution} (r : P.toProg.Run X) {s : Stmt}
    (hb : s.Below P.flags P.count) (hr : s.RegsBelow P.marks) (hn : s.NoBlocks)
    {p n : Nat} {regs regs' : Nat → Int} {L : List Label}
    (hrun : Steps (P.lower (.tasksBlock s)) regs L .skip regs')
    (hlab : ∀ m (hm : m < L.length), r.lab p (n + m) = L[m])
    {t : Nat} {w a b : X.E} (hw : X.thread w = p) (hww : X.IsWrite w)
    (hwl : X.loc w = P.start t) (hwn : n ≤ r.idx w) (hwL : r.idx w < n + L.length)
    (ha : X.thread a = t + 1) (hbp : X.thread b = p) (hbL : n + L.length ≤ r.idx b) :
    X.hb a b := by
  obtain ⟨j, -, hjr, hjl, hjb, -⟩ := block_join r hb hr hn hrun hlab hw hww hwl hwn hwL hbp hbL
  exact join_hb hP r ha hjr hjl (Or.inr hjb)

end TaskProg

end Vx
