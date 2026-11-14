import VxMemoryModel.Execution

/-!
# The code of one thread

A small language for the code of a thread: registers, loads, stores, read-modify-writes, fences,
`if` and loops. Each access names its memory order, as the methods of `core::sync::atomic` do.
`loadAt`, `storeAt` and `waitAt` access an element of an array, at an index that the code
computes.

A run of the code gives a list of labels, one for each memory access. A load can return any
value, because the value comes from another thread. So if a run ends with a load, a run with
another value for that load also exists. A read-modify-write is two steps, a read and then a
write, so a run can also stop between them.
-/

namespace Vx

/-- A memory access of a thread, as the thread sees it. -/
structure Label where
  kind : Kind
  order : Order
  loc : Nat
  val : Int
  /-- True for the write of a read-modify-write. -/
  rmw : Bool
  deriving DecidableEq

inductive Expr where
  | const (v : Int)
  | reg (r : Nat)
  | add (a b : Expr)
  | sub (a b : Expr)
  /-- 1 when the two values are equal, else 0. -/
  | eq (a b : Expr)

def Expr.eval (regs : Nat → Int) : Expr → Int
  | const v => v
  | reg r => regs r
  | add a b => a.eval regs + b.eval regs
  | sub a b => a.eval regs - b.eval regs
  | eq a b => if a.eval regs = b.eval regs then 1 else 0

inductive Stmt where
  | skip
  /-- Read the place `x` into the register `r`. -/
  | load (r x : Nat) (o : Order)
  /-- Write the value of `e` to the place `x`. -/
  | store (x : Nat) (e : Expr) (o : Order)
  /-- Add `e` to the place `x` in one atomic step, and put the old value in `r`. -/
  | fetchAdd (r x : Nat) (e : Expr) (o : Order)
  /-- Read `x` into `r`. If the value equals `expected`, write `new` in the same atomic step. -/
  | cas (r x : Nat) (expected new : Expr) (o : Order)
  /-- The write of a read-modify-write, after its read. -/
  | rmwStore (x : Nat) (v : Int) (o : Order)
  | fence (o : Order)
  | assign (r : Nat) (e : Expr)
  | seq (s t : Stmt)
  /-- Run `s` when `c` is not 0, else `t`. -/
  | ite (c : Expr) (s t : Stmt)
  /-- Run `s` while `c` is not 0. -/
  | loop (c : Expr) (s : Stmt)
  /-- Read the place `x` with `sc` reads until it is not 0. -/
  | wait (x : Nat)
  /-- Read element `i` of the array of `len` places that starts at the place `base`, into the
  register `r`. An index outside the array stops the thread, as a failed bounds check does. -/
  | loadAt (r base len : Nat) (i : Expr) (o : Order)
  /-- Write the value of `e` to element `i` of the array of `len` places at `base`. -/
  | storeAt (base len : Nat) (i e : Expr) (o : Order)
  /-- Read element `i` of the array of `len` places at `base` with `sc` reads until it is not 0. -/
  | waitAt (base len : Nat) (i : Expr)
  /-- Start the task `base + i`, for an `i` below `len`. `Tasks.lean` gives its meaning, so it has
  no step here. -/
  | spawnAt (base len : Nat) (i : Expr)
  /-- Wait for the task `base + i` to finish. `Tasks.lean` gives its meaning. -/
  | joinAt (base len : Nat) (i : Expr)
  /-- Put the value of `e` in the register `base + i`, for an `i` below `len`. -/
  | assignAt (base len : Nat) (i e : Expr)
  /-- `tasks { s }`: at its end, wait for each task that `s` started. `Tasks.lean` gives its
  meaning. -/
  | tasksBlock (s : Stmt)

/-- Start the task `t`. -/
abbrev Stmt.spawn (t : Nat) : Stmt := .spawnAt t 1 (.const 0)
/-- Wait for the task `t` to finish. -/
abbrev Stmt.join (t : Nat) : Stmt := .joinAt t 1 (.const 0)

/-- Set the flag `x`: an `sc` write of 1. -/
abbrev Stmt.signal (x : Nat) : Stmt := .store x (.const 1) .sc

def update (regs : Nat → Int) (r : Nat) (v : Int) : Nat → Int :=
  fun i => if i = r then v else regs i

/-- One step of a thread. The label is `none` for a step that does not access memory. -/
inductive Step : Stmt → (Nat → Int) → Option Label → Stmt → (Nat → Int) → Prop
  | load {r x o regs} (v : Int) :
    Step (.load r x o) regs (some ⟨.read, o, x, v, false⟩) .skip (update regs r v)
  | store {x e o regs} :
    Step (.store x e o) regs (some ⟨.write, o, x, e.eval regs, false⟩) .skip regs
  | fetchAdd {r x e o regs} (v : Int) :
    Step (.fetchAdd r x e o) regs (some ⟨.read, o, x, v, false⟩)
      (.rmwStore x (v + e.eval regs) o) (update regs r v)
  | cas_eq {r x expected new o regs} (v : Int) : v = expected.eval regs →
    Step (.cas r x expected new o) regs (some ⟨.read, o, x, v, false⟩)
      (.rmwStore x (new.eval regs) o) (update regs r v)
  | cas_ne {r x expected new o regs} (v : Int) : v ≠ expected.eval regs →
    Step (.cas r x expected new o) regs (some ⟨.read, o, x, v, false⟩) .skip (update regs r v)
  | rmwStore {x v o regs} : Step (.rmwStore x v o) regs (some ⟨.write, o, x, v, true⟩) .skip regs
  | fence {o regs} : Step (.fence o) regs (some ⟨.fence, o, 0, 0, false⟩) .skip regs
  | assign {r e regs} : Step (.assign r e) regs none .skip (update regs r (e.eval regs))
  | seq_skip {t regs} : Step (.seq .skip t) regs none t regs
  | seq {s s' t regs regs' a} : Step s regs a s' regs' → Step (.seq s t) regs a (.seq s' t) regs'
  | ite_true {c s t regs} : c.eval regs ≠ 0 → Step (.ite c s t) regs none s regs
  | ite_false {c s t regs} : c.eval regs = 0 → Step (.ite c s t) regs none t regs
  | loop {c s regs} : Step (.loop c s) regs none (.ite c (.seq s (.loop c s)) .skip) regs
  | wait {x regs} (v : Int) :
    Step (.wait x) regs (some ⟨.read, .sc, x, v, false⟩) (if v = 0 then .wait x else .skip) regs
  | loadAt {r base len i o regs} (j : Nat) (v : Int) : i.eval regs = (j : Int) → j < len →
    Step (.loadAt r base len i o) regs (some ⟨.read, o, base + j, v, false⟩) .skip (update regs r v)
  | storeAt {base len i e o regs} (j : Nat) : i.eval regs = (j : Int) → j < len →
    Step (.storeAt base len i e o) regs (some ⟨.write, o, base + j, e.eval regs, false⟩) .skip regs
  | waitAt {base len i regs} (j : Nat) (v : Int) : i.eval regs = (j : Int) → j < len →
    Step (.waitAt base len i) regs (some ⟨.read, .sc, base + j, v, false⟩)
      (if v = 0 then .waitAt base len i else .skip) regs
  | assignAt {base len i e regs} (j : Nat) : i.eval regs = (j : Int) → j < len →
    Step (.assignAt base len i e) regs none .skip (update regs (base + j) (e.eval regs))

/-- `Runs s regs l`: the code `s` can make the accesses `l` and then stop. -/
inductive Runs : Stmt → (Nat → Int) → List Label → Prop
  | stop {s regs} : Runs s regs []
  | silent {s regs s' regs' l} : Step s regs none s' regs' → Runs s' regs' l → Runs s regs l
  | visible {s regs s' regs' a l} :
    Step s regs (some a) s' regs' → Runs s' regs' l → Runs s regs (a :: l)

/-- The first `n` labels of `f` are a run of `s` from registers that hold 0. -/
def Trace (s : Stmt) (n : Nat) (f : Nat → Label) : Prop :=
  Runs s (fun _ => 0) ((List.range n).map f)

theorem Runs.prefix {s regs L} (h : Runs s regs L) : ∀ l l', L = l ++ l' → Runs s regs l := by
  induction h with
  | stop =>
    intro l _ hL
    rw [(List.append_eq_nil_iff.1 hL.symm).1]
    exact .stop
  | silent hs _ ih => exact fun l l' hL => .silent hs (ih l l' hL)
  | visible hs _ ih =>
    intro l l' hL
    rcases List.cons_eq_append_iff.1 hL with ⟨rfl, _⟩ | ⟨l, rfl, hl⟩
    · exact .stop
    · exact .visible hs (ih l l' hl)

/-- A thread can stop at any point. -/
theorem Trace.prefix {s n m} {f : Nat → Label} (h : Trace s n f) (hmn : m ≤ n) : Trace s m f := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hmn
  induction k with
  | zero => exact h
  | succ k ih =>
    unfold Trace at h
    rw [← Nat.add_assoc, List.range_succ, List.map_append] at h
    exact ih (h.prefix _ _ rfl) (Nat.le_add_right _ _)

/-- A wait that sees the flag set on its first read, followed by `k`. -/
theorem Runs.wait_seq {x k regs l} (h : Runs k regs l) :
    Runs (.seq (.wait x) k) regs (⟨.read, .sc, x, 1, false⟩ :: l) :=
  .visible (.seq (.wait 1)) (.silent (by simpa using (Step.seq_skip : Step (.seq .skip k) _ _ _ _)) h)

/-- A `waitAt` at element `j` that sees the flag set on its first read, followed by `k`. -/
theorem Runs.waitAt_seq {base len i k regs l} (j : Nat) (hi : i.eval regs = (j : Int)) (hj : j < len)
    (h : Runs k regs l) :
    Runs (.seq (.waitAt base len i) k) regs (⟨.read, .sc, base + j, 1, false⟩ :: l) :=
  .visible (.seq (.waitAt j 1 hi hj))
    (.silent (by simpa using (Step.seq_skip : Step (.seq .skip k) _ _ _ _)) h)

theorem Step.receptive {s regs a s' regs'} (h : Step s regs a s' regs') :
    ∀ l, a = some l → l.kind = .read → ∀ v : Int,
    ∃ s'' regs'', Step s regs (some { l with val := v }) s'' regs'' := by
  induction h with
  | load =>
    intro l hl _ v
    cases hl
    exact ⟨_, _, .load v⟩
  | fetchAdd =>
    intro l hl _ v
    cases hl
    exact ⟨_, _, .fetchAdd v⟩
  | wait =>
    intro l hl _ v
    cases hl
    exact ⟨_, _, .wait v⟩
  | loadAt j _ hi hj =>
    intro l hl _ v
    cases hl
    exact ⟨_, _, .loadAt j v hi hj⟩
  | waitAt j _ hi hj =>
    intro l hl _ v
    cases hl
    exact ⟨_, _, .waitAt j v hi hj⟩
  | @cas_eq _ _ expected _ _ regs _ _ | @cas_ne _ _ expected _ _ regs _ _ =>
    intro l hl _ v
    cases hl
    by_cases hv : v = expected.eval regs
    · exact ⟨_, _, .cas_eq v hv⟩
    · exact ⟨_, _, .cas_ne v hv⟩
  | store | rmwStore | fence | storeAt =>
    intro l hl hk _
    cases hl
    cases hk
  | seq _ ih =>
    intro l hl hk v
    obtain ⟨_, _, h⟩ := ih l hl hk v
    exact ⟨_, _, .seq h⟩
  | _ => intro _ hl; cases hl

theorem Runs.receptive {s regs L} (h : Runs s regs L) :
    ∀ l a, L = l ++ [a] → a.kind = .read → ∀ v : Int, Runs s regs (l ++ [{ a with val := v }]) := by
  induction h with
  | stop =>
    intro l _ hL
    simp at hL
  | silent hs _ ih =>
    intro l a hL hk v
    exact .silent hs (ih l a hL hk v)
  | visible hs hr ih =>
    intro l a hL hk v
    rcases List.cons_eq_append_iff.1 hL with ⟨rfl, hl⟩ | ⟨l', rfl, hl'⟩
    · cases hl
      obtain ⟨_, _, hs'⟩ := hs.receptive _ rfl hk v
      exact .visible hs' .stop
    · exact .visible hs (ih l' a hl' hk v)

theorem Trace.congr {s n} {f g : Nat → Label} (h : Trace s n f)
    (hfg : ∀ i, i < n → f i = g i) : Trace s n g := by
  unfold Trace
  rw [← List.map_congr_left (fun i hi => hfg i (List.mem_range.1 hi))]
  exact h

/-- The last access of a trace, when it is a load, can return any value. -/
theorem Trace.receptive {s n} {f : Nat → Label} (h : Trace s (n + 1) f)
    (hk : (f n).kind = .read) (v : Int) :
    Trace s (n + 1) (fun i => if i = n then { f n with val := v } else f i) := by
  unfold Trace at h ⊢
  rw [List.range_succ, List.map_append] at h ⊢
  have hmap : (List.range n).map (fun i => if i = n then { f n with val := v } else f i) =
      (List.range n).map f :=
    List.map_congr_left (fun i hi => by simp [Nat.ne_of_lt (List.mem_range.1 hi)])
  rw [hmap]
  simpa using h.receptive _ _ rfl hk v

/-! ## Rules on the code -/

/-- The atomics are sequentially consistent, and there are no fences. -/
def Stmt.SeqCstOnly : Stmt → Prop
  | load _ _ o | store _ _ o | fetchAdd _ _ _ o | cas _ _ _ _ o | rmwStore _ _ o
  | loadAt _ _ _ _ o | storeAt _ _ _ _ o => o = .plain ∨ o = .sc
  | fence _ => False
  | seq s t | ite _ s t => s.SeqCstOnly ∧ t.SeqCstOnly
  | loop _ s | tasksBlock s => s.SeqCstOnly
  | _ => True

def Label.SeqCstOnly (l : Label) : Prop := l.kind ≠ .fence ∧ (l.order = .plain ∨ l.order = .sc)

/-- A rule on the code that each step keeps, and that holds for each access the code makes. -/
structure Rule where
  onStmt : Stmt → Prop
  onLabel : Label → Prop
  step : ∀ {s regs a s' regs'}, Step s regs a s' regs' → onStmt s →
    onStmt s' ∧ ∀ l, a = some l → onLabel l

theorem Rule.runs (R : Rule) {s regs L} (h : Runs s regs L) (hs : R.onStmt s) :
    ∀ l, l ∈ L → R.onLabel l := by
  induction h with
  | stop => intro _ h; cases h
  | silent hstep _ ih => exact ih (R.step hstep hs).1
  | visible hstep _ ih =>
    intro l hl
    rcases List.mem_cons.1 hl with rfl | hl
    · exact (R.step hstep hs).2 _ rfl
    · exact ih (R.step hstep hs).1 l hl

theorem Rule.trace (R : Rule) {s n} {f : Nat → Label} (h : Trace s (n + 1) f) (hs : R.onStmt s) :
    R.onLabel (f n) :=
  R.runs h hs _ (List.mem_map.2 ⟨n, List.mem_range.2 (Nat.lt_succ_self n), rfl⟩)

def seqCstRule : Rule where
  onStmt := Stmt.SeqCstOnly
  onLabel := Label.SeqCstOnly
  step := by
    intro s regs a s' regs' h
    induction h with
    | load | store | cas_ne | rmwStore | loadAt | storeAt =>
      exact fun hs => ⟨trivial, fun l hl => by cases hl; exact ⟨nofun, hs⟩⟩
    | fetchAdd | cas_eq =>
      exact fun hs => ⟨hs, fun l hl => by cases hl; exact ⟨nofun, hs⟩⟩
    | fence => exact fun hs => hs.elim
    | seq _ ih =>
      intro hs
      exact ⟨⟨(ih hs.1).1, hs.2⟩, (ih hs.1).2⟩
    | seq_skip => exact fun hs => ⟨hs.2, nofun⟩
    | ite_true => exact fun hs => ⟨hs.1, nofun⟩
    | ite_false => exact fun hs => ⟨hs.2, nofun⟩
    | loop => exact fun hs => ⟨⟨⟨hs, hs⟩, trivial⟩, nofun⟩
    | assign | assignAt => exact fun _ => ⟨trivial, nofun⟩
    | wait v =>
      refine fun _ => ⟨?_, fun l hl => by cases hl; exact ⟨nofun, Or.inr rfl⟩⟩
      by_cases hv : v = 0 <;> simp [hv, Stmt.SeqCstOnly]
    | waitAt _ v =>
      refine fun _ => ⟨?_, fun l hl => by cases hl; exact ⟨nofun, Or.inr rfl⟩⟩
      by_cases hv : v = 0 <;> simp [hv, Stmt.SeqCstOnly]

/-- Each access of `s` of the kind `k` to the place `x` with the order `o` has `p k o x`. An
array access can touch each place of its array. The body of a `tasks` block runs in the same
thread, so its accesses count. -/
def Stmt.Accesses (p : Kind → Order → Nat → Prop) : Stmt → Prop
  | load _ x o => p .read o x
  | store x _ o | rmwStore x _ o => p .write o x
  | fetchAdd _ x _ o | cas _ x _ _ o => p .read o x ∧ p .write o x
  | fence o => p .fence o 0
  | wait x => p .read .sc x
  | loadAt _ base len _ o => ∀ j < len, p .read o (base + j)
  | storeAt base len _ _ o => ∀ j < len, p .write o (base + j)
  | waitAt base len _ => ∀ j < len, p .read .sc (base + j)
  | seq s t | ite _ s t => s.Accesses p ∧ t.Accesses p
  | loop _ s => s.Accesses p
  | tasksBlock s => s.Accesses p
  | skip | assign .. | spawnAt .. | joinAt .. | assignAt .. => True

def accessRule (p : Kind → Order → Nat → Prop) : Rule where
  onStmt := Stmt.Accesses p
  onLabel l := p l.kind l.order l.loc
  step := by
    intro s regs a s' regs' h
    induction h with
    | load | store | rmwStore | fence => exact fun hs => ⟨trivial, fun l hl => by cases hl; exact hs⟩
    | loadAt j _ _ hj | storeAt j _ hj =>
      exact fun hs => ⟨trivial, fun l hl => by cases hl; exact hs j hj⟩
    | cas_ne => exact fun hs => ⟨trivial, fun l hl => by cases hl; exact hs.1⟩
    | fetchAdd | cas_eq => exact fun hs => ⟨hs.2, fun l hl => by cases hl; exact hs.1⟩
    | assign | assignAt => exact fun _ => ⟨trivial, nofun⟩
    | seq _ ih =>
      intro hs
      exact ⟨⟨(ih hs.1).1, hs.2⟩, (ih hs.1).2⟩
    | seq_skip => exact fun hs => ⟨hs.2, nofun⟩
    | ite_true => exact fun hs => ⟨hs.1, nofun⟩
    | ite_false => exact fun hs => ⟨hs.2, nofun⟩
    | loop => exact fun hs => ⟨⟨⟨hs, hs⟩, trivial⟩, nofun⟩
    | wait v =>
      refine fun hs => ⟨?_, fun l hl => by cases hl; exact hs⟩
      by_cases hv : v = 0 <;> simp [hv, Stmt.Accesses] <;> exact hs
    | waitAt j v _ hj =>
      refine fun hs => ⟨?_, fun l hl => by cases hl; exact hs j hj⟩
      by_cases hv : v = 0 <;> simp [hv, Stmt.Accesses] <;> exact hs

theorem Stmt.Accesses.mono {p q : Kind → Order → Nat → Prop} (hpq : ∀ k o x, p k o x → q k o x)
    {s : Stmt} (h : s.Accesses p) : s.Accesses q := by
  induction s with
  | load | store | rmwStore | fence | wait => exact hpq _ _ _ h
  | fetchAdd | cas => exact ⟨hpq _ _ _ h.1, hpq _ _ _ h.2⟩
  | loadAt | storeAt | waitAt => exact fun j hj => hpq _ _ _ (h j hj)
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | loop _ _ ih | tasksBlock _ ih => exact ih h
  | _ => trivial

end Vx
