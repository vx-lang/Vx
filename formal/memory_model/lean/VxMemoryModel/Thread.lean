import VxMemoryModel.Execution

/-!
# The code of one thread

A small language for the code of a thread: registers, loads, stores, `if` and loops. The type
of a place gives the memory order of each access: an access to an atomic place is sequentially
consistent, and an access to another place is plain.

A run of the code gives a list of labels, one for each memory access. A load can return any
value, because the value comes from another thread. So if a run ends with a load, a run with
another value for that load also exists.
-/

namespace Vx

/-- A memory access of a thread, as the thread sees it. -/
structure Label where
  kind : Kind
  order : Order
  loc : Nat
  val : Int
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
  | load (r x : Nat)
  /-- Write the value of `e` to the place `x`. -/
  | store (x : Nat) (e : Expr)
  | assign (r : Nat) (e : Expr)
  | seq (s t : Stmt)
  /-- Run `s` when `c` is not 0, else `t`. -/
  | ite (c : Expr) (s t : Stmt)
  /-- Run `s` while `c` is not 0. -/
  | loop (c : Expr) (s : Stmt)

/-- The memory order of an access to the place `x`. -/
def orderOf (atomic : Nat → Bool) (x : Nat) : Order := if atomic x then .sc else .plain

def update (regs : Nat → Int) (r : Nat) (v : Int) : Nat → Int :=
  fun i => if i = r then v else regs i

/-- One step of a thread. The label is `none` for a step that does not access memory. -/
inductive Step (atomic : Nat → Bool) : Stmt → (Nat → Int) → Option Label → Stmt → (Nat → Int) → Prop
  | load {r x regs} (v : Int) :
    Step atomic (.load r x) regs (some ⟨.read, orderOf atomic x, x, v⟩) .skip (update regs r v)
  | store {x e regs} :
    Step atomic (.store x e) regs (some ⟨.write, orderOf atomic x, x, e.eval regs⟩) .skip regs
  | assign {r e regs} : Step atomic (.assign r e) regs none .skip (update regs r (e.eval regs))
  | seq_skip {t regs} : Step atomic (.seq .skip t) regs none t regs
  | seq {s s' t regs regs' a} :
    Step atomic s regs a s' regs' → Step atomic (.seq s t) regs a (.seq s' t) regs'
  | ite_true {c s t regs} : c.eval regs ≠ 0 → Step atomic (.ite c s t) regs none s regs
  | ite_false {c s t regs} : c.eval regs = 0 → Step atomic (.ite c s t) regs none t regs
  | loop {c s regs} :
    Step atomic (.loop c s) regs none (.ite c (.seq s (.loop c s)) .skip) regs

/-- `Runs atomic s regs l`: the code `s` can make the accesses `l` and then stop. -/
inductive Runs (atomic : Nat → Bool) : Stmt → (Nat → Int) → List Label → Prop
  | stop {s regs} : Runs atomic s regs []
  | silent {s regs s' regs' l} :
    Step atomic s regs none s' regs' → Runs atomic s' regs' l → Runs atomic s regs l
  | visible {s regs s' regs' a l} :
    Step atomic s regs (some a) s' regs' → Runs atomic s' regs' l → Runs atomic s regs (a :: l)

/-- The first `n` labels of `f` are a run of `s` from registers that hold 0. -/
def Trace (atomic : Nat → Bool) (s : Stmt) (n : Nat) (f : Nat → Label) : Prop :=
  Runs atomic s (fun _ => 0) ((List.range n).map f)

theorem Runs.prefix {atomic : Nat → Bool} {s regs L} (h : Runs atomic s regs L) :
    ∀ l l', L = l ++ l' → Runs atomic s regs l := by
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
theorem Trace.prefix {atomic : Nat → Bool} {s n m} {f : Nat → Label} (h : Trace atomic s n f)
    (hmn : m ≤ n) : Trace atomic s m f := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hmn
  induction k with
  | zero => exact h
  | succ k ih =>
    unfold Trace at h
    rw [← Nat.add_assoc, List.range_succ, List.map_append] at h
    exact ih (h.prefix _ _ rfl) (Nat.le_add_right _ _)

theorem Step.receptive {atomic : Nat → Bool} {s regs a s' regs'}
    (h : Step atomic s regs a s' regs') : ∀ l, a = some l → l.kind = .read → ∀ v : Int,
    ∃ s'' regs'', Step atomic s regs (some { l with val := v }) s'' regs'' := by
  induction h with
  | load =>
    intro l hl _ v
    cases hl
    exact ⟨_, _, .load v⟩
  | store =>
    intro l hl hk _
    cases hl
    cases hk
  | seq _ ih =>
    intro l hl hk v
    obtain ⟨_, _, h⟩ := ih l hl hk v
    exact ⟨_, _, .seq h⟩
  | _ => intro _ hl; cases hl

theorem Runs.receptive {atomic : Nat → Bool} {s regs L} (h : Runs atomic s regs L) :
    ∀ l a, L = l ++ [a] → a.kind = .read → ∀ v : Int,
    Runs atomic s regs (l ++ [{ a with val := v }]) := by
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

theorem Trace.congr {atomic : Nat → Bool} {s n} {f g : Nat → Label} (h : Trace atomic s n f)
    (hfg : ∀ i, i < n → f i = g i) : Trace atomic s n g := by
  unfold Trace
  rw [← List.map_congr_left (fun i hi => hfg i (List.mem_range.1 hi))]
  exact h

/-- The last access of a trace, when it is a load, can return any value. -/
theorem Trace.receptive {atomic : Nat → Bool} {s n} {f : Nat → Label}
    (h : Trace atomic s (n + 1) f) (hk : (f n).kind = .read) (v : Int) :
    Trace atomic s (n + 1) (fun i => if i = n then { f n with val := v } else f i) := by
  unfold Trace at h ⊢
  rw [List.range_succ, List.map_append] at h ⊢
  have hmap : (List.range n).map (fun i => if i = n then { f n with val := v } else f i) =
      (List.range n).map f :=
    List.map_congr_left (fun i hi => by simp [Nat.ne_of_lt (List.mem_range.1 hi)])
  rw [hmap]
  simpa using h.receptive _ _ rfl hk v

/-- Each step accesses memory with the order of the type of the place. -/
theorem Step.typed {atomic : Nat → Bool} {s regs a s' regs'} (h : Step atomic s regs a s' regs') :
    ∀ l, a = some l → l.kind ≠ .fence ∧ l.order = orderOf atomic l.loc := by
  induction h with
  | load => intro l hl; cases hl; exact ⟨nofun, rfl⟩
  | store => intro l hl; cases hl; exact ⟨nofun, rfl⟩
  | seq _ ih => exact ih
  | _ => intro _ hl; cases hl

theorem Runs.typed {atomic : Nat → Bool} {s regs L} (h : Runs atomic s regs L) :
    ∀ l, l ∈ L → l.kind ≠ .fence ∧ l.order = orderOf atomic l.loc := by
  induction h with
  | stop => intro _ h; cases h
  | silent _ _ ih => exact ih
  | visible hs _ ih =>
    intro l hl
    rcases List.mem_cons.1 hl with rfl | hl
    · exact hs.typed _ rfl
    · exact ih l hl

theorem Trace.typed {atomic : Nat → Bool} {s n} {f : Nat → Label} (h : Trace atomic s (n + 1) f) :
    (f n).kind ≠ .fence ∧ (f n).order = orderOf atomic (f n).loc :=
  Runs.typed h _ (List.mem_map.2 ⟨n, List.mem_range.2 (Nat.lt_succ_self n), rfl⟩)

end Vx
