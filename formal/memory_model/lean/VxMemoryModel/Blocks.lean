import VxMemoryModel.Tasks

/-!
# Tasks blocks

`tasks { s }` lowers to three parts. The first part clears the marks of the tasks that `s` can
start. Then `s` runs, and each `spawn` in it sets the mark of its task. At the end, the block
joins each task that has its mark set. So the block waits only for the tasks that it started.

`block_hb` shows that each task that a block started happens before the code after the block.
`block_hb_nested` allows a body that holds other `tasks` blocks. Each inner block, at any depth,
starts only tasks that the blocks around it do not start (`NestOk`). So an inner block clears only
the marks of its own tasks, and the outer block still joins each task that it started. A task of
an inner block is in a run of that inner block, which joined it (`nested_seg`).
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

/-- Each `tasks` block in `s`, at any depth, starts no task of `outer` and no task of a block
around it. -/
def Stmt.NestOk (outer : List Nat) : Stmt → Prop
  | tasksBlock s => (∀ u ∈ s.spawnIds, u ∉ outer) ∧ s.NestOk (outer ++ s.spawnIds)
  | seq s t | ite _ s t => s.NestOk outer ∧ t.NestOk outer
  | loop _ s => s.NestOk outer
  | _ => True

theorem Stmt.NestOk.of_noBlocks {outer : List Nat} {s : Stmt} (h : s.NoBlocks) : s.NestOk outer := by
  induction s with
  | tasksBlock => exact h.elim
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | loop _ _ ih => exact ih h
  | _ => trivial

theorem Stmt.NestOk.mono {o1 o2 : List Nat} (h12 : ∀ u ∈ o1, u ∈ o2) {s : Stmt}
    (h : s.NestOk o2) : s.NestOk o1 := by
  induction s generalizing o1 o2 with
  | tasksBlock _ ih =>
    refine ⟨fun u hu h1 => h.1 u hu (h12 u h1), ih (fun u hu => ?_) h.2⟩
    rcases List.mem_append.1 hu with hu | hu
    · exact List.mem_append_left _ (h12 u hu)
    · exact List.mem_append_right _ hu
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h12 h.1, ih2 h12 h.2⟩
  | loop _ _ ih => exact ih h12 h
  | _ => trivial

/-- The tasks that `s` can start, also in the `tasks` blocks inside `s`, at any depth. -/
def Stmt.allIds : Stmt → List Nat
  | spawnAt base len _ => (List.range len).map (base + ·)
  | seq s t | ite _ s t => s.allIds ++ t.allIds
  | loop _ s | tasksBlock s => s.allIds
  | _ => []

theorem Stmt.allIds_of_noBlocks {s : Stmt} (h : s.NoBlocks) : s.allIds = s.spawnIds := by
  induction s with
  | tasksBlock => exact h.elim
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => simp only [Stmt.allIds, Stmt.spawnIds, ih1 h.1, ih2 h.2]
  | loop _ _ ih => exact ih h
  | _ => rfl

theorem allIds_lt {n tasks} {s : Stmt} (h : s.Below n tasks) : ∀ u ∈ s.allIds, u < tasks := by
  induction s with
  | spawnAt =>
    intro u hu
    simp only [Stmt.allIds, List.mem_map, List.mem_range] at hu
    obtain ⟨j, hj, rfl⟩ := hu
    simp only [Stmt.Below] at h
    omega
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 =>
    intro u hu
    rcases List.mem_append.1 hu with hu | hu
    · exact ih1 h.1 u hu
    · exact ih2 h.2 u hu
  | loop _ _ ih | tasksBlock _ ih => exact ih h
  | _ => nofun

namespace TaskProg

variable {P : TaskProg}

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

/-- Code that comes from the lowering of the body of a block: it uses the places below `flags`.
Only a `spawn` sets a mark, and only an inner block clears one, of a task that is not in `keep`. -/
inductive Clean (P : TaskProg) (keep : List Nat) : Stmt → Prop
  | skip : Clean P keep .skip
  | load {r x o} : r < P.marks → x < P.flags → Clean P keep (.load r x o)
  | store {x e o} : x < P.flags → Clean P keep (.store x e o)
  | fetchAdd {r x e o} : r < P.marks → x < P.flags → Clean P keep (.fetchAdd r x e o)
  | cas {r x e n o} : r < P.marks → x < P.flags → Clean P keep (.cas r x e n o)
  | rmwStore {x v o} : x < P.flags → Clean P keep (.rmwStore x v o)
  | fence {o} : Clean P keep (.fence o)
  | assign {r e} : r < P.marks → Clean P keep (.assign r e)
  | wait {x} : Clean P keep (.wait x)
  | loadAt {r base len i o} : r < P.marks → base + len ≤ P.flags → Clean P keep (.loadAt r base len i o)
  | storeAt {base len i e o} : base + len ≤ P.flags → Clean P keep (.storeAt base len i e o)
  | waitAt {base len i} : Clean P keep (.waitAt base len i)
  | assignAt {base len i e} : base + len ≤ P.marks → Clean P keep (.assignAt base len i e)
  | seq {s t} : Clean P keep s → Clean P keep t → Clean P keep (.seq s t)
  | ite {c s t} : Clean P keep s → Clean P keep t → Clean P keep (.ite c s t)
  | loop {c s} : Clean P keep s → Clean P keep (.loop c s)
  | spawn {base len i} : i.RegsBelow P.marks →
    Clean P keep (.seq (.assignAt (P.marks + base) len i (.const 1))
      (.storeAt (P.start base) len i (.const 1) .sc))
  | clear {t e} : t ∉ keep → Clean P keep (.assign (P.marks + t) e)

/-- The task `base + i` has its mark set. -/
def Marked (P : TaskProg) (regs : Nat → Int) (base len : Nat) (i : Expr) : Prop :=
  ∀ j : Nat, i.eval regs = (j : Int) → j < len → regs (P.marks + base + j) = 1

/-- The states of lowered code: the next write of a `start` flag comes after its mark. -/
inductive Good (P : TaskProg) (keep : List Nat) (regs : Nat → Int) : Stmt → Prop
  | clean {s} : Clean P keep s → Good P keep regs s
  | seq {s t} : Good P keep regs s → Clean P keep t → Good P keep regs (.seq s t)
  | marked {base len i} : Marked P regs base len i →
    Good P keep regs (.seq .skip (.storeAt (P.start base) len i (.const 1) .sc))
  | marked' {base len i} : Marked P regs base len i →
    Good P keep regs (.storeAt (P.start base) len i (.const 1) .sc)

theorem update_keeps {keep : List Nat} {regs : Nat → Int} {r : Nat} {v : Int} (hr : r < P.marks) :
    ∀ u ∈ keep, regs (P.marks + u) = 1 → update regs r v (P.marks + u) = 1 := by
  intro u _ h1
  have : P.marks + u ≠ r := by omega
  simp [update, this, h1]

theorem update_one_keeps {keep : List Nat} {regs : Nat → Int} {r : Nat} :
    ∀ u ∈ keep, regs (P.marks + u) = 1 → update regs r 1 (P.marks + u) = 1 := by
  intro u _ h1
  by_cases h : P.marks + u = r <;> simp [update, h, h1]

theorem update_clear_keeps {keep : List Nat} {regs : Nat → Int} {t : Nat} {v : Int} (ht : t ∉ keep) :
    ∀ u ∈ keep, regs (P.marks + u) = 1 → update regs (P.marks + t) v (P.marks + u) = 1 := by
  intro u hu h1
  have : u ≠ t := fun h => ht (h ▸ hu)
  simp [update, this, h1]

/-- One step keeps `Good`, never clears the mark of a task of `keep`, and writes a `start` flag
only when its mark is set. -/
theorem Good.step {keep regs s a s' regs'} (h : Step s regs a s' regs') (hg : Good P keep regs s) :
    Good P keep regs' s' ∧
      (∀ u ∈ keep, regs (P.marks + u) = 1 → regs' (P.marks + u) = 1) ∧
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
    cases hg with
    | clean hc =>
      cases hc with
      | assign hr => exact ⟨.clean .skip, update_keeps hr, nofun⟩
      | clear ht => exact ⟨.clean .skip, update_clear_keeps ht, nofun⟩
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

theorem Good.steps {keep regs s L s' regs'} (h : Steps s regs L s' regs')
    (hg : Good P keep regs s) :
    Good P keep regs' s' ∧ (∀ u ∈ keep, regs (P.marks + u) = 1 → regs' (P.marks + u) = 1) ∧
      ∀ l ∈ L, l.kind = .write → ∀ u ∈ keep, l.loc = P.start u → regs' (P.marks + u) = 1 := by
  induction h with
  | refl => exact ⟨hg, fun _ _ h => h, nofun⟩
  | silent hs _ ih =>
    obtain ⟨g1, k1, -⟩ := hg.step hs
    obtain ⟨g2, k2, w2⟩ := ih g1
    exact ⟨g2, fun u hu h => k2 u hu (k1 u hu h), w2⟩
  | visible hs _ ih =>
    obtain ⟨g1, k1, w1⟩ := hg.step hs
    obtain ⟨g2, k2, w2⟩ := ih g1
    refine ⟨g2, fun u hu h => k2 u hu (k1 u hu h), fun l hl hk u hu hlu => ?_⟩
    rcases List.mem_cons.1 hl with rfl | hl
    · exact k2 u hu (w1 l rfl hk u hlu)
    · exact w2 l hl hk u hu hlu

theorem clearMarks_clean {keep ids : List Nat} (h : ∀ u ∈ ids, u ∉ keep) :
    Clean P keep (P.clearMarks ids) := by
  induction ids with
  | nil => exact .skip
  | cons u _ ih =>
    exact .seq (.clear (h u List.mem_cons_self)) (ih fun v hv => h v (List.mem_cons_of_mem _ hv))

theorem joinMarked_clean {keep ids : List Nat} : Clean P keep (P.joinMarked ids) := by
  induction ids with
  | nil => exact .skip
  | cons _ _ ih => exact .seq (.ite .waitAt .skip) ih

theorem clean_lower {keep : List Nat} {s : Stmt} (hb : s.Below P.flags P.count)
    (hr : s.RegsBelow P.marks) (hn : s.NestOk keep) : Clean P keep (P.lower s) := by
  induction s generalizing keep with
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
  | tasksBlock _ ih =>
    exact .seq (clearMarks_clean hn.1)
      (.seq (ih hb hr (hn.2.mono fun u hu => List.mem_append_left _ hu)) joinMarked_clean)

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

/-- A run of a block to its end: it clears its marks with no access, its body runs to its end,
and its end only reads. The access `w` of the run is in the body. -/
theorem block_body {X : Execution} {r : P.toProg.Run X} {s : Stmt} {p n : Nat}
    {regs regs' : Nat → Int} {L : List Label}
    (hrun : Steps (P.lower (.tasksBlock s)) regs L .skip regs')
    (hlab : ∀ m (hm : m < L.length), r.lab p (n + m) = L[m])
    {w : X.E} (hw : X.thread w = p) (hww : X.IsWrite w) (hwn : n ≤ r.idx w)
    (hwL : r.idx w < n + L.length) :
    ∃ Lb Le, (∃ ra rb, Steps (P.lower s) ra Lb .skip rb ∧
      Steps (P.joinMarked s.spawnIds) rb Le .skip regs') ∧
      L = Lb ++ Le ∧ ∃ hin : r.idx w - n < Lb.length,
        Lb[r.idx w - n] = ⟨X.kind w, X.order w, X.loc w, r.val w, r.isRmw w⟩ := by
  obtain ⟨L0, Lrest, ra, h0, hrest, rfl⟩ := Steps.seq_done hrun
  obtain rfl := clearMarks_silent h0
  obtain ⟨Lb, Le, rb, hbody, hend, rfl⟩ := Steps.seq_done hrest
  simp only [List.nil_append] at hlab hwL ⊢
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
  refine ⟨Lb, Le, ⟨ra, rb, hbody, hend⟩, rfl, hin, ?_⟩
  rw [← List.getElem_append_left hin]
  exact hwlab

/-- The end of a block joins each task of the block that the block started.

The labels of thread `p` from the position `n` on are a run `L` of the block to its end. The
event `w` is a `spawn` of task `t` in that run, and `b` comes after the run. Then thread `p` reads
`done t` before `b`. -/
theorem block_join_of {X : Execution} (r : P.toProg.Run X) {s : Stmt}
    (hb : s.Below P.flags P.count) (hr : s.RegsBelow P.marks) (hn : s.NestOk s.spawnIds)
    {p n : Nat} {regs regs' : Nat → Int} {L : List Label}
    (hrun : Steps (P.lower (.tasksBlock s)) regs L .skip regs')
    (hlab : ∀ m (hm : m < L.length), r.lab p (n + m) = L[m])
    {t : Nat} {w b : X.E} (hw : X.thread w = p) (hww : X.IsWrite w)
    (hwl : X.loc w = P.start t) (hwn : n ≤ r.idx w) (hwL : r.idx w < n + L.length)
    (hbp : X.thread b = p) (hbL : n + L.length ≤ r.idx b) (hu : t ∈ s.spawnIds) :
    ∃ j, X.thread j = p ∧ X.IsRead j ∧ X.loc j = P.done t ∧ X.po j b := by
  obtain ⟨Lb, Le, ⟨ra, rb, hbody, hend⟩, rfl, hin, hwlab⟩ := block_body hrun hlab hw hww hwn hwL
  have hmem : Lb[r.idx w - n] ∈ Lb := List.getElem_mem _
  rw [hwlab] at hmem
  -- The `spawn` set the mark of task `t`, and no inner block clears it.
  have hmark := (Good.steps hbody (.clean (clean_lower hb hr hn))).2.2 _ hmem hww t hu hwl
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
  exact ⟨j, hjt.trans hbp, hjk, hjl, hjb⟩

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
  obtain ⟨Lb, Le, ⟨ra, rb, hbody, -⟩, rfl, hin, hwlab⟩ := block_body hrun hlab hw hww hwn hwL
  have hmem : Lb[r.idx w - n] ∈ Lb := List.getElem_mem _
  rw [hwlab] at hmem
  -- A body with no inner block starts only its own tasks.
  obtain ⟨u, hu, hut⟩ := (accessRule _).runs hbody.runs (lower_spawnIds hb hn) _ hmem hww
    (by simp only [hwl, start]; omega)
  have htu : t = u := by
    simp only [hwl, start] at hut
    omega
  subst htu
  obtain ⟨j, h1, h2, h3, h4⟩ := block_join_of r hb hr (.of_noBlocks hn) hrun hlab hw hww hwl hwn
    hwL hbp hbL hu
  exact ⟨j, h1, h2, h3, h4, hu⟩

/-- A write of `start t` in a run of the lowered code `c` to its end, for a task `t` that is not
in `O`, is in a run of an inner block to its end. Each `spawn` of `c` outside its inner blocks
starts a task of `O`. -/
theorem nested_seg {O : List Nat} : ∀ {c : Stmt}, c.Below P.flags P.count → c.RegsBelow P.marks →
    c.NestOk O → (∀ u ∈ c.spawnIds, u ∈ O) → ∀ {regs L regs'}, Steps (P.lower c) regs L .skip regs' →
    ∀ {i : Nat} (hi : i < L.length) {t : Nat}, L[i].kind = .write ∧ L[i].loc = P.start t → t ∉ O →
    ∃ b n0 L0 r1 r2, sizeOf b < sizeOf c ∧ b.Below P.flags P.count ∧ b.RegsBelow P.marks ∧
      b.NestOk b.spawnIds ∧ Steps (P.lower (.tasksBlock b)) r1 L0 .skip r2 ∧
      (∀ j (hj : j < L0.length), L[n0 + j]? = some L0[j]) ∧ n0 ≤ i ∧ i < n0 + L0.length ∧
      n0 + L0.length ≤ L.length := by
  intro c
  induction c with
  | tasksBlock b _ =>
    intro hb hr hn _ regs L regs' h i hi t _ _
    refine ⟨b, 0, L, regs, regs', by simp, hb, hr,
      hn.2.mono fun u hu => List.mem_append_right _ hu, h, fun j hj => ?_, Nat.zero_le _,
      by omega, by omega⟩
    rw [Nat.zero_add, List.getElem?_eq_getElem hj]
  | seq s1 s2 ih1 ih2 =>
    intro hb hr hn hO regs L regs' h i hi t hw ht
    obtain ⟨L1, L2, ra, h1, h2, rfl⟩ := Steps.seq_done h
    have hO1 : ∀ u ∈ s1.spawnIds, u ∈ O := fun u hu => hO u (List.mem_append_left _ hu)
    have hO2 : ∀ u ∈ s2.spawnIds, u ∈ O := fun u hu => hO u (List.mem_append_right _ hu)
    simp only [List.length_append] at hi
    by_cases hi1 : i < L1.length
    · rw [List.getElem_append_left hi1] at hw
      obtain ⟨b, n0, L0, r1, r2, hsz, hb', hr', hn', hs, hl, h3, h4, h5⟩ :=
        ih1 hb.1 hr.1 hn.1 hO1 h1 hi1 hw ht
      refine ⟨b, n0, L0, r1, r2, by simp; omega, hb', hr', hn', hs, fun j hj => ?_, h3, h4,
        by simp only [List.length_append]; omega⟩
      rw [List.getElem?_append_left (by omega)]
      exact hl j hj
    · rw [List.getElem_append_right (by omega)] at hw
      obtain ⟨b, n0, L0, r1, r2, hsz, hb', hr', hn', hs, hl, h3, h4, h5⟩ :=
        ih2 hb.2 hr.2 hn.2 hO2 h2 (i := i - L1.length) (by omega) hw ht
      refine ⟨b, L1.length + n0, L0, r1, r2, by simp; omega, hb', hr', hn', hs, fun j hj => ?_,
        by omega, by omega, by simp only [List.length_append]; omega⟩
      rw [List.getElem?_append_right (by omega), show L1.length + n0 + j - L1.length = n0 + j by omega]
      exact hl j hj
  | ite c s1 s2 ih1 ih2 =>
    intro hb hr hn hO regs L regs' h i hi t hw ht
    have hO1 : ∀ u ∈ s1.spawnIds, u ∈ O := fun u hu => hO u (List.mem_append_left _ hu)
    have hO2 : ∀ u ∈ s2.spawnIds, u ∈ O := fun u hu => hO u (List.mem_append_right _ hu)
    change Steps (.ite c (P.lower s1) (P.lower s2)) regs L .skip regs' at h
    cases h with
    | silent hs h' =>
      cases hs with
      | ite_true =>
        obtain ⟨b, n0, L0, r1, r2, hsz, rest⟩ := ih1 hb.1 hr.2.1 hn.1 hO1 h' hi hw ht
        exact ⟨b, n0, L0, r1, r2, by simp; omega, rest⟩
      | ite_false =>
        obtain ⟨b, n0, L0, r1, r2, hsz, rest⟩ := ih2 hb.2 hr.2.2 hn.2 hO2 h' hi hw ht
        exact ⟨b, n0, L0, r1, r2, by simp; omega, rest⟩
    | visible hs => cases hs
  | loop c s1 ih =>
    intro hb hr hn hO regs L regs' h i hi t hw ht
    change Steps (.loop c (P.lower s1)) regs L .skip regs' at h
    obtain ⟨n, L', r1, r2, hs, hl, h1, h2, h3⟩ := Steps.loop_pass h hi
    rw [eq_of_off hl hi h1 h2] at hw
    obtain ⟨b, n0, L0, r3, r4, hsz, hb', hr', hn', hs', hl0, h4, h5, h6⟩ :=
      ih hb hr.2 hn hO hs (i := i - n) (by omega) hw ht
    exact ⟨b, n + n0, L0, r3, r4, by simp; omega, hb', hr', hn', hs', off_trans hl hl0 h6,
      by omega, by omega, by omega⟩
  | _ =>
    -- Code with no inner block starts only the tasks of its own `spawn`s.
    intro hb _ _ hO regs L regs' h i hi t hw ht
    obtain ⟨u, hu, hut⟩ := (accessRule _).runs h.runs (lower_spawnIds hb (by trivial)) _
      (List.getElem_mem hi) hw.1 (by rw [hw.2]; simp only [start]; omega)
    rw [hw.2] at hut
    simp only [start] at hut
    exact absurd (show t = u by omega ▸ hO u hu) ht

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

/-- `block_join` for a block whose code holds other `tasks` blocks. Each inner block starts only
tasks that the blocks around it do not start (`NestOk`). The block joins each of its own tasks
at its end. A task of an inner block is in a run of that inner block, which joined it. -/
theorem block_join_nested {X : Execution} (r : P.toProg.Run X) {s : Stmt}
    (hb : s.Below P.flags P.count) (hr : s.RegsBelow P.marks) (hn : s.NestOk s.spawnIds)
    {p n : Nat} {regs regs' : Nat → Int} {L : List Label}
    (hrun : Steps (P.lower (.tasksBlock s)) regs L .skip regs')
    (hlab : ∀ m (hm : m < L.length), r.lab p (n + m) = L[m])
    {t : Nat} {w b : X.E} (hw : X.thread w = p) (hww : X.IsWrite w)
    (hwl : X.loc w = P.start t) (hwn : n ≤ r.idx w) (hwL : r.idx w < n + L.length)
    (hbp : X.thread b = p) (hbL : n + L.length ≤ r.idx b) :
    ∃ j, X.thread j = p ∧ X.IsRead j ∧ X.loc j = P.done t ∧ X.po j b := by
  by_cases ht : t ∈ s.spawnIds
  · exact block_join_of r hb hr hn hrun hlab hw hww hwl hwn hwL hbp hbL ht
  · obtain ⟨Lb, Le, ⟨ra, rb, hbody, -⟩, rfl, hin, hwlab⟩ := block_body hrun hlab hw hww hwn hwL
    obtain ⟨c, n0, L0, r1, r2, hsz, hcb, hcr, hcn, hc, hl0, h1, h2, h3⟩ :=
      nested_seg hb hr hn (fun _ h => h) hbody (i := r.idx w - n) hin
        ⟨by rw [hwlab]; exact hww, by rw [hwlab]; exact hwl⟩ ht
    have hlab0 : ∀ m (hm : m < L0.length), r.lab p (n + n0 + m) = L0[m] := fun m hm => by
      have h0 := hl0 m hm
      rw [List.getElem?_eq_getElem (by omega)] at h0
      rw [Nat.add_assoc, hlab _ (by simp only [List.length_append]; omega),
        List.getElem_append_left (by omega)]
      exact Option.some.inj h0
    exact block_join_nested r hcb hcr hcn hc hlab0 hw hww hwl (by omega) (by omega) hbp
      (by simp only [List.length_append] at hbL; omega)
termination_by sizeOf s

/-- `block_hb` for a block whose code holds other `tasks` blocks. -/
theorem block_hb_nested (hP : P.Ok) {X : Execution} (r : P.toProg.Run X) {s : Stmt}
    (hb : s.Below P.flags P.count) (hr : s.RegsBelow P.marks) (hn : s.NestOk s.spawnIds)
    {p n : Nat} {regs regs' : Nat → Int} {L : List Label}
    (hrun : Steps (P.lower (.tasksBlock s)) regs L .skip regs')
    (hlab : ∀ m (hm : m < L.length), r.lab p (n + m) = L[m])
    {t : Nat} {w a b : X.E} (hw : X.thread w = p) (hww : X.IsWrite w)
    (hwl : X.loc w = P.start t) (hwn : n ≤ r.idx w) (hwL : r.idx w < n + L.length)
    (ha : X.thread a = t + 1) (hbp : X.thread b = p) (hbL : n + L.length ≤ r.idx b) :
    X.hb a b := by
  obtain ⟨j, -, hjr, hjl, hjb⟩ :=
    block_join_nested r hb hr hn hrun hlab hw hww hwl hwn hwL hbp hbL
  exact join_hb hP r ha hjr hjl (Or.inr hjb)

end TaskProg

end Vx
