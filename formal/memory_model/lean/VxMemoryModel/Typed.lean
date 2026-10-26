import VxMemoryModel.Footprint

/-!
# A typed layer over the footprint rule

`TaskProg.raceFree` needs code that follows the footprint rule (`Footprints`). This file adds a
small typed language with locals and tasks, and a compiler to `TaskProg`. `WellTyped.footprints`
shows that the code of a well-typed program follows the rule. So a well-typed program has no
data race (`WellTyped.raceFree`), and DRF-SC holds for it (`WellTyped.drf_sc`).

A type is a tree of leaves: a number, a cell (a `SharedCell` of a number with a safe wrapper, so
a shared borrow can write it), or an atomic. `Send` and `Sync` come from the leaves. A cell is
not `Sync`. Each leaf is `Send`, so `Send` checks nothing yet.

A `spawn` starts a task with an id. The task lists its captures, each with a mode: `move`, `ref`
or `mut`. The checker gives the task and the code of each block a permission for each local:
none, `ref` or own. Two permissions for one local fit when one of them is none, or both are `ref`
and the type is `Sync`. Then two accesses to one place cannot race (`compat_of_fit`).

Each task joins at the end of its block. A task starts no task, and a block does not run in a
loop.
-/

namespace Vx

namespace Typed

open Classical

/-- A leaf of a type. -/
inductive Leaf where
  | int
  /-- A `SharedCell` with a safe wrapper: a shared borrow can write it with a plain write. -/
  | cell
  /-- An atomic: each access through a shared borrow is atomic. -/
  | atomic

/-- A leaf is `Sync` when two tasks can use it through shared borrows. -/
def Leaf.Sync : Leaf → Prop
  | .int | .atomic => True
  | .cell => False

/-- A type: a leaf, or a struct of two fields. -/
inductive Ty where
  | leaf (l : Leaf)
  | pair (a b : Ty)

/-- The leaves of a type, in the order of its places. -/
def Ty.leaves : Ty → List Leaf
  | .leaf l => [l]
  | .pair a b => a.leaves ++ b.leaves

/-- A struct is `Sync` when each field is. -/
def Ty.Sync : Ty → Prop
  | .leaf l => l.Sync
  | .pair a b => a.Sync ∧ b.Sync

/-- A struct is `Send` when each field is. Each leaf is `Send`. -/
def Ty.Send : Ty → Prop
  | .leaf _ => True
  | .pair a b => a.Send ∧ b.Send

theorem Ty.Sync.leaf {τ : Ty} (h : τ.Sync) {l : Leaf} (hl : l ∈ τ.leaves) : l.Sync := by
  induction τ with
  | leaf l' =>
    simp only [Ty.leaves, List.mem_singleton] at hl
    subst hl
    exact h
  | pair a b iha ihb =>
    simp only [Ty.leaves, List.mem_append] at hl
    rcases hl with hl | hl
    · exact iha h.1 hl
    · exact ihb h.2 hl

/-- How a task captures a local. -/
inductive Mode where
  | move
  /-- A shared borrow. -/
  | ref
  /-- A mutable borrow. -/
  | mut
  deriving DecidableEq

/-- What a piece of code may do with a local. -/
inductive Perm where
  | none
  | ref
  | own
  deriving DecidableEq

/-- The accesses that a permission allows at a leaf. Through `ref`, a number is only read, a cell
can be written, and each access to an atomic is atomic. -/
def Allowed : Perm → Leaf → Kind → Order → Prop
  | .own, _, _, _ => True
  | .ref, .int, k, _ => k = .read
  | .ref, .cell, _, _ => True
  | .ref, .atomic, _, o => o ≠ .plain
  | .none, _, _, _ => False

/-- Two permissions for one leaf fit: one is none, or both are `ref` and the leaf is `Sync`. -/
def Fit (a b : Perm) (l : Leaf) : Prop :=
  a = .none ∨ b = .none ∨ (a = .ref ∧ b = .ref ∧ l.Sync)

/-- Two accesses to one place with permissions that fit cannot race. -/
theorem compat_of_fit {a b : Perm} {l : Leaf} {k k' : Kind} {o o' : Order} {p : Nat}
    (ha : Allowed a l k o) (hb : Allowed b l k' o') (h : Fit a b l) : Compat k o p k' o' p := by
  rcases h with rfl | rfl | ⟨rfl, rfl, hs⟩
  · exact absurd ha (by cases l <;> simp [Allowed])
  · exact absurd hb (by cases l <;> simp [Allowed])
  · cases l with
    | int => exact Or.inr (Or.inr (Or.inr (Or.inl ⟨ha, hb⟩)))
    | atomic => exact Or.inr (Or.inr (Or.inr (Or.inr ⟨ha, hb⟩)))
    | cell => exact hs.elim

/-- The statements of the typed language. A local `x` has one place for each leaf of its type,
and `i` picks the leaf. -/
inductive Src where
  | skip
  | assign (r : Nat) (e : Expr)
  /-- A plain read of leaf `i` of local `x`. -/
  | read (r x i : Nat)
  /-- A plain write of leaf `i` of local `x`. -/
  | write (x i : Nat) (e : Expr)
  /-- An atomic add to leaf `i` of local `x`. -/
  | fetchAdd (r x i : Nat) (e : Expr)
  | seq (s t : Src)
  | ite (c : Expr) (s t : Src)
  | loop (c : Expr) (s : Src)

/-- A part of a block: code, or a `spawn` of task `u`. -/
inductive Item where
  | code (c : Src)
  | spawn (u : Nat)

/-- A part of the main code: code that starts no task, or a `tasks` block. -/
inductive SPhase where
  | code (c : Src)
  | block (items : List Item)

/-- The tasks that the items start. -/
def spawnsOf : List Item → List Nat
  | [] => []
  | .code _ :: is => spawnsOf is
  | .spawn u :: is => u :: spawnsOf is

def SPhase.spawns : SPhase → List Nat
  | .code _ => []
  | .block is => spawnsOf is

/-! ## The compiler -/

def compile (W : Nat) : Src → Stmt
  | .skip => .skip
  | .assign r e => .assign r e
  | .read r x i => .load r (W * x + i) .plain
  | .write x i e => .store (W * x + i) e .plain
  | .fetchAdd r x i e => .fetchAdd r (W * x + i) e .sc
  | .seq s t => .seq (compile W s) (compile W t)
  | .ite c s t => .ite c (compile W s) (compile W t)
  | .loop c s => .loop c (compile W s)

def compileItems (W : Nat) : List Item → Stmt
  | [] => .skip
  | .code c :: is => .seq (compile W c) (compileItems W is)
  | .spawn u :: is => .seq (.spawnAt u 1 (.const 0)) (compileItems W is)

def SPhase.compile (W : Nat) : SPhase → TaskProg.Phase
  | .code c => .code (Typed.compile W c)
  | .block is => .block (compileItems W is) none

/-- A typed program. Leaf `i` of local `x` is the place `width * x + i`. -/
structure TProg where
  ty : Nat → Ty
  /-- The locals are below `locals`. -/
  locals : Nat
  /-- Each type has at most `width` leaves. -/
  width : Nat
  /-- The code uses the registers below `regs`. -/
  regs : Nat
  /-- The task ids are below `count`. -/
  count : Nat
  /-- The captures of task `u`. -/
  caps : Nat → List (Nat × Mode)
  /-- The code of task `u`. -/
  body : Nat → Src
  main : List SPhase

namespace TProg

variable (T : TProg)

/-- The permission of task `u` for local `x`. -/
def capPerm (u x : Nat) : Perm :=
  if (x, Mode.move) ∈ T.caps u ∨ (x, Mode.mut) ∈ T.caps u then .own
  else if (x, Mode.ref) ∈ T.caps u then .ref else .none

/-- The permission of the code of a block for local `x`. It has none when a task of the block
owns `x`, and `ref` when a task of the block borrows `x`. -/
def blockPerm (is : List Item) (x : Nat) : Perm :=
  if ∃ u ∈ spawnsOf is, T.capPerm u x = .own then .none
  else if ∃ u ∈ spawnsOf is, T.capPerm u x = .ref then .ref else .own

/-- `x` is a local, `i` is one of its leaves, and `P` holds for that leaf. -/
def At (x i : Nat) (P : Leaf → Prop) : Prop :=
  x < T.locals ∧ ∃ l, (T.ty x).leaves[i]? = some l ∧ P l

/-- The code follows the permissions `perm` and uses only the registers below `regs`. -/
def Fits (perm : Nat → Perm) : Src → Prop
  | .skip => True
  | .assign r e => r < T.regs ∧ e.RegsBelow T.regs
  | .read r x i => r < T.regs ∧ T.At x i fun l => Allowed (perm x) l .read .plain
  | .write x i e => e.RegsBelow T.regs ∧ T.At x i fun l => Allowed (perm x) l .write .plain
  | .fetchAdd r x i e => r < T.regs ∧ e.RegsBelow T.regs ∧
      T.At x i fun l => Allowed (perm x) l .read .sc ∧ Allowed (perm x) l .write .sc
  | .seq s t => Fits perm s ∧ Fits perm t
  | .ite c s t => c.RegsBelow T.regs ∧ Fits perm s ∧ Fits perm t
  | .loop c s => c.RegsBelow T.regs ∧ Fits perm s

/-- The rules of a phase. The code of a block follows `blockPerm`. A task that owns a local
excludes each other task of the block from it. -/
def PhaseOk : SPhase → Prop
  | .code c => T.Fits (fun _ => .own) c
  | .block is => (∀ c, Item.code c ∈ is → T.Fits (T.blockPerm is) c) ∧
      ∀ u ∈ spawnsOf is, ∀ v ∈ spawnsOf is, u ≠ v → ∀ x, T.capPerm u x = .own →
        T.capPerm v x = .none

/-- The tasks that the main code starts. -/
def spawns : List Nat := T.main.flatMap SPhase.spawns

/-- The rules of the type checker. -/
structure WellTyped : Prop where
  width : ∀ x, (T.ty x).leaves.length ≤ T.width
  /-- A task moves or borrows `mut` only a `Send` local, and borrows `ref` only a `Sync` one. -/
  caps : ∀ u x m, (x, m) ∈ T.caps u → (m = .ref → (T.ty x).Sync) ∧ (m ≠ .ref → (T.ty x).Send)
  /-- The code of each task uses only its captures. -/
  bodies : ∀ u, T.Fits (T.capPerm u) (T.body u)
  phases : ∀ p ∈ T.main, T.PhaseOk p
  /-- One place in the code starts each task. -/
  nodup : T.spawns.Nodup
  spawn_lt : ∀ u ∈ T.spawns, u < T.count

def phases : List TaskProg.Phase := T.main.map (SPhase.compile T.width)

def toTaskProg : TaskProg where
  main := TaskProg.seqAll T.phases
  task u := compile T.width (T.body u)
  count := T.count
  flags := T.width * T.locals
  marks := T.regs

/-! ## The proof -/

variable {T}

/-- The accesses that the permissions `perm` allow, at the places of the locals. -/
def Legal (perm : Nat → Perm) (k : Kind) (o : Order) (p : Nat) : Prop :=
  ∃ x i, p = T.width * x + i ∧ T.At x i fun l => Allowed (perm x) l k o

theorem Fits.accesses {perm : Nat → Perm} {c : Src} (h : T.Fits perm c) :
    (compile T.width c).Accesses (T.Legal perm) := by
  induction c with
  | skip | assign => trivial
  | read _ x i => exact ⟨x, i, rfl, h.2⟩
  | write x i => exact ⟨x, i, rfl, h.2⟩
  | fetchAdd _ x i =>
    obtain ⟨-, -, hx, l, hl, h1, h2⟩ := h
    exact ⟨⟨x, i, rfl, hx, l, hl, h1⟩, ⟨x, i, rfl, hx, l, hl, h2⟩⟩
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ih h.2

theorem Fits.regs {perm : Nat → Perm} {c : Src} (h : T.Fits perm c) :
    (compile T.width c).RegsBelow T.regs := by
  induction c with
  | skip => trivial
  | assign | write | ite _ _ _ ih1 ih2 => first
    | exact h
    | exact h.1
    | exact ⟨h.1, ih1 h.2.1, ih2 h.2.2⟩
  | read => exact h.1
  | fetchAdd => exact ⟨h.1, h.2.1⟩
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | loop _ _ ih => exact ⟨h.1, ih h.2⟩

theorem At.lt (hw : T.WellTyped) {x i : Nat} {P : Leaf → Prop} (h : T.At x i P) :
    T.width * x + i < T.width * T.locals := by
  obtain ⟨hx, l, hl, -⟩ := h
  have hi : i < T.width := Nat.lt_of_lt_of_le ((List.getElem?_eq_some_iff.1 hl).1)
    (hw.width x)
  calc T.width * x + i < T.width * x + T.width := Nat.add_lt_add_left hi _
    _ = T.width * (x + 1) := (Nat.mul_succ _ _).symm
    _ ≤ T.width * T.locals := Nat.mul_le_mul_left _ hx

theorem Fits.below (hw : T.WellTyped) {perm : Nat → Perm} {c : Src} (h : T.Fits perm c) (n : Nat) :
    (compile T.width c).Below (T.width * T.locals) n := by
  induction c with
  | skip | assign => trivial
  | read | write => exact At.lt hw h.2
  | fetchAdd => exact At.lt hw h.2.2
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ih h.2

theorem compile_noTasks (W : Nat) (c : Src) : (compile W c).NoTasks := by
  induction c with
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1, ih2⟩
  | loop _ _ ih => exact ih
  | _ => trivial

theorem compile_noBlocks (W : Nat) (c : Src) : (compile W c).NoBlocks := by
  induction c with
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1, ih2⟩
  | loop _ _ ih => exact ih
  | _ => trivial

theorem compile_seqCst (W : Nat) (c : Src) : (compile W c).SeqCstOnly := by
  induction c with
  | read | write => exact Or.inl rfl
  | fetchAdd => exact Or.inr rfl
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1, ih2⟩
  | loop _ _ ih => exact ih
  | _ => trivial

theorem compile_spawnIds (W : Nat) (c : Src) : (compile W c).spawnIds = [] := by
  induction c with
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => simp [compile, Stmt.spawnIds, ih1, ih2]
  | loop _ _ ih => simp [compile, Stmt.spawnIds, ih]
  | _ => rfl

theorem compileItems_spawnIds (W : Nat) (is : List Item) :
    (compileItems W is).spawnIds = spawnsOf is := by
  induction is with
  | nil => rfl
  | cons it is ih =>
    cases it <;> simp [compileItems, spawnsOf, Stmt.spawnIds, compile_spawnIds, ih]

/-- Two pieces of code whose permissions fit for each local are apart. -/
theorem apart_of_fit (hw : T.WellTyped) {a b : Nat → Perm} {s t : Stmt} (hs : s.Accesses (T.Legal a))
    (ht : t.Accesses (T.Legal b))
    (hfit : ∀ x l, l ∈ (T.ty x).leaves → Fit (a x) (b x) l) (own : Nat → Prop) :
    s.Apart own t := by
  refine Stmt.Accesses.mono ?_ hs
  rintro k o _ ⟨x, i, rfl, hx, l, hl, ha⟩ -
  refine Stmt.Accesses.mono ?_ ht
  rintro k' o' _ ⟨y, j, rfl, hy, l', hl', hb⟩
  have hi : i < T.width := Nat.lt_of_lt_of_le ((List.getElem?_eq_some_iff.1 hl).1)
    (hw.width x)
  have hj : j < T.width := Nat.lt_of_lt_of_le ((List.getElem?_eq_some_iff.1 hl').1)
    (hw.width y)
  by_cases he : T.width * x + i = T.width * y + j
  · have hpos : 0 < T.width := Nat.lt_of_le_of_lt (Nat.zero_le _) hi
    have hxy : x = y := by
      have := congrArg (· / T.width) he
      simp only [Nat.mul_add_div hpos, Nat.div_eq_of_lt hi, Nat.div_eq_of_lt hj] at this
      omega
    subst hxy
    have hij : i = j := by omega
    subst hij
    rw [hl] at hl'
    cases hl'
    rw [he]
    exact compat_of_fit ha hb (hfit x l (List.mem_of_getElem? hl))
  · exact Or.inr (Or.inr (Or.inl he))

theorem capPerm_ref {u x : Nat} (h : T.capPerm u x = .ref) : (x, Mode.ref) ∈ T.caps u := by
  unfold capPerm at h
  split at h
  · cases h
  · split at h
    · assumption
    · cases h

theorem sync_of_ref (hw : T.WellTyped) {u x : Nat} (h : T.capPerm u x = .ref) {l : Leaf}
    (hl : l ∈ (T.ty x).leaves) : l.Sync :=
  ((hw.caps u x .ref (capPerm_ref h)).1 rfl).leaf hl

/-- Two tasks of a block fit. -/
theorem fit_tasks (hw : T.WellTyped) {is : List Item} (hp : T.PhaseOk (.block is)) {u v : Nat}
    (hu : u ∈ spawnsOf is) (hv : v ∈ spawnsOf is) (huv : u ≠ v) (x : Nat) (l : Leaf)
    (hl : l ∈ (T.ty x).leaves) : Fit (T.capPerm u x) (T.capPerm v x) l := by
  have excl := hp.2
  cases h1 : T.capPerm u x with
  | none => exact Or.inl rfl
  | own => exact Or.inr (Or.inl (excl u hu v hv huv x h1))
  | ref =>
    cases h2 : T.capPerm v x with
    | none => exact Or.inr (Or.inl rfl)
    | own =>
      have := excl v hv u hu (Ne.symm huv) x h2
      rw [h1] at this
      cases this
    | ref => exact Or.inr (Or.inr ⟨rfl, rfl, sync_of_ref hw h1 hl⟩)

/-- The code of a block and a task of the block fit. -/
theorem fit_block (hw : T.WellTyped) {is : List Item} {u : Nat} (hu : u ∈ spawnsOf is)
    (x : Nat) (l : Leaf) (hl : l ∈ (T.ty x).leaves) : Fit (T.blockPerm is x) (T.capPerm u x) l := by
  unfold blockPerm
  split
  · exact Or.inl rfl
  · rename_i h1
    split
    · cases h : T.capPerm u x with
      | none => exact Or.inr (Or.inl rfl)
      | own => exact absurd ⟨u, hu, h⟩ h1
      | ref => exact Or.inr (Or.inr ⟨rfl, rfl, sync_of_ref hw h hl⟩)
    · rename_i h2
      cases h : T.capPerm u x with
      | none => exact Or.inr (Or.inl rfl)
      | own => exact absurd ⟨u, hu, h⟩ h1
      | ref => exact absurd ⟨u, hu, h⟩ h2

theorem compileItems_accesses {is : List Item} (h : ∀ c, Item.code c ∈ is → T.Fits (T.blockPerm is) c) :
    (compileItems T.width is).Accesses (T.Legal (T.blockPerm is)) := by
  generalize T.blockPerm is = perm at h
  induction is with
  | nil => trivial
  | cons it is ih =>
    cases it with
    | code c => exact ⟨(h c (List.mem_cons_self ..)).accesses, ih fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩
    | spawn => exact ⟨trivial, ih fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩

theorem compileItems_regs {is : List Item} (h : ∀ c, Item.code c ∈ is → T.Fits (T.blockPerm is) c) :
    (compileItems T.width is).RegsBelow T.regs := by
  generalize T.blockPerm is = perm at h
  induction is with
  | nil => trivial
  | cons it is ih =>
    cases it with
    | code c => exact ⟨(h c (List.mem_cons_self ..)).regs, ih fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩
    | spawn => exact ⟨trivial, ih fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩

theorem compileItems_noBlocks (W : Nat) (is : List Item) : (compileItems W is).NoBlocks := by
  induction is with
  | nil => trivial
  | cons it is ih => cases it <;> exact ⟨by first | exact compile_noBlocks W _ | trivial, ih⟩

theorem compileItems_seqCst (W : Nat) (is : List Item) : (compileItems W is).SeqCstOnly := by
  induction is with
  | nil => trivial
  | cons it is ih => cases it <;> exact ⟨by first | exact compile_seqCst W _ | trivial, ih⟩

theorem compileItems_below (hw : T.WellTyped) {is : List Item}
    (h : ∀ c, Item.code c ∈ is → T.Fits (T.blockPerm is) c) (hs : ∀ u ∈ spawnsOf is, u < T.count) :
    (compileItems T.width is).Below (T.width * T.locals) T.count := by
  generalize T.blockPerm is = perm at h
  induction is with
  | nil => trivial
  | cons it is ih =>
    cases it with
    | code c =>
      exact ⟨(h c (List.mem_cons_self ..)).below hw _,
        ih (fun u hu => hs u hu) (fun c' hc => h c' (List.mem_cons_of_mem _ hc))⟩
    | spawn u =>
      exact ⟨hs u (List.mem_cons_self ..),
        ih (fun v hv => hs v (List.mem_cons_of_mem _ hv)) fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩

/-- The rule of a block holds for the compiled block. -/
theorem blockOk (hw : T.WellTyped) {is : List Item} (hp : T.PhaseOk (.block is)) :
    T.toTaskProg.BlockOk (· < T.toTaskProg.flags) (fun _ => []) (compileItems T.width is) := by
  have hacc := compileItems_accesses hp.1
  refine ⟨compileItems_noBlocks _ _, compileItems_regs hp.1, ?_, ?_⟩
  · intro u hu w hd
    rw [compileItems_spawnIds] at hu
    rw [TaskProg.desc_nil] at hd
    subst w
    exact apart_of_fit hw hacc (hw.bodies u).accesses (fun x l hl => fit_block hw hu x l hl) _
  · intro u hu v hv huv w w' hd hd'
    rw [compileItems_spawnIds] at hu hv
    rw [TaskProg.desc_nil] at hd hd'
    subst w w'
    exact apart_of_fit hw (hw.bodies u).accesses (hw.bodies v).accesses
      (fun x l hl => fit_tasks hw hp hu hv huv x l hl) _

theorem phase_ids (W : Nat) (p : SPhase) : (p.compile W).ids = p.spawns := by
  cases p with
  | code => rfl
  | block is => exact compileItems_spawnIds W is

theorem phasesOk (hw : T.WellTyped) :
    ∀ ps : List SPhase, (∀ p ∈ ps, T.PhaseOk p) → (ps.flatMap SPhase.spawns).Nodup →
      T.toTaskProg.PhasesOk (· < T.toTaskProg.flags) (fun _ => []) (ps.map (SPhase.compile T.width))
  | [], _, _ => trivial
  | p :: ps, hp, hn => by
    rw [List.flatMap_cons, List.nodup_append] at hn
    refine ⟨?_, ?_, phasesOk hw ps (fun q hq => hp q (List.mem_cons_of_mem _ hq)) hn.2.1⟩
    · cases p with
      | code c => exact compile_noTasks _ c
      | block is => exact blockOk hw (hp _ (List.mem_cons_self ..))
    · intro u hu q hq huq
      rw [phase_ids] at hu
      obtain ⟨q', hq', rfl⟩ := List.mem_map.1 hq
      rw [phase_ids] at huq
      exact hn.2.2 u hu u (List.mem_flatMap.2 ⟨q', hq', huq⟩) rfl

theorem seqAll_below (hw : T.WellTyped) :
    ∀ ps : List SPhase, (∀ p ∈ ps, T.PhaseOk p) → (∀ u ∈ ps.flatMap SPhase.spawns, u < T.count) →
      (TaskProg.seqAll (ps.map (SPhase.compile T.width))).Below (T.width * T.locals) T.count
  | [], _, _ => trivial
  | p :: ps, hp, hs => by
    refine ⟨?_, seqAll_below hw ps (fun q hq => hp q (List.mem_cons_of_mem _ hq))
      fun u hu => hs u (by rw [List.flatMap_cons]; exact List.mem_append_right _ hu)⟩
    have hs' : ∀ u ∈ p.spawns, u < T.count := fun u hu =>
      hs u (by rw [List.flatMap_cons]; exact List.mem_append_left _ hu)
    cases p with
    | code c => exact (hp _ (List.mem_cons_self ..)).below hw _
    | block is => exact compileItems_below hw (hp _ (List.mem_cons_self ..)).1 hs'

theorem seqAll_seqCst (W : Nat) :
    ∀ ps : List SPhase, (TaskProg.seqAll (ps.map (SPhase.compile W))).SeqCstOnly
  | [] => trivial
  | p :: ps => ⟨by cases p with
      | code c => exact compile_seqCst W c
      | block is => exact compileItems_seqCst W is, seqAll_seqCst W ps⟩

/-- The code of a well-typed program follows the footprint rule. -/
theorem WellTyped.footprints (hw : T.WellTyped) :
    T.toTaskProg.Footprints (· < T.toTaskProg.flags) (fun _ => []) T.phases where
  ok := ⟨seqAll_below hw _ hw.phases hw.spawn_lt, fun u => (hw.bodies u).below hw _⟩
  le_flags := fun _ h => h
  main := rfl
  phases := phasesOk hw _ hw.phases hw.nodup
  tasks := fun u => Or.inr ⟨compile_noTasks _ _, rfl⟩
  task_phases := fun _ => trivial
  one_parent := TaskProg.one_parent_nil _
  order := fun _ _ ⟨_, hp, _⟩ => by simp at hp

/-- A well-typed program has no data race, in each execution. -/
theorem WellTyped.raceFree (hw : T.WellTyped) {X : Execution} (hX : T.toTaskProg.toProg.Execs X) :
    X.RaceFree :=
  TaskProg.raceFree hw.footprints hX

/-- DRF-SC for a well-typed program: each finite execution that the model allows is an
interleaving. -/
theorem WellTyped.drf_sc (hw : T.WellTyped) {X : Execution} (hX : T.toTaskProg.toProg.Execs X)
    (wf : X.WellFormed) (hfin : X.Finite) (hc : X.Consistent) : X.SeqConsistent :=
  TaskProg.drf_sc_of_footprints hw.footprints (seqAll_seqCst _ _)
    (fun _ => compile_seqCst _ _) hX wf hfin hc

end TProg

end Typed

end Vx
