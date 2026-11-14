import VxMemoryModel.Footprint

/-!
# A typed layer over the footprint rule

`TaskProg.raceFree` needs code that follows the footprint rule (`Footprints`). This file adds a
small typed language with locals and tasks, and a compiler to `TaskProg`. `WellTyped.footprints`
shows that the code of a well-typed program follows the rule. So a well-typed program has no
data race (`WellTyped.raceFree`), and DRF-SC holds for it (`WellTyped.drf_sc`).

A type is a tree of leaves: a number, a cell (a `SharedCell` of a number with a safe wrapper, so
a shared borrow can write it), or an atomic. `Send` and `Sync` come from the leaves. A cell is
not `Sync`. A local can also be a reference `&x` or `&mut x` to another local `x`. The checker
knows which local a reference borrows, so an access through it goes to the places of `x`. `&x`
is `Send` only when `x` is `Sync`, because two tasks can then read `x` at the same time.

A `spawn` starts a task with an id. The task lists its captures, each with a mode: `move`, `ref`
or `mut`. The checker gives the task and the code of each block a permission for each local:
none, `ref` or own. A task gets the permission through each capture that reaches the local
(`taskPerm`). A shared reference gives at most `ref` (`via`). Two permissions for one local
fit when one of them is none, or both are `ref` and the type is `Sync`. Then two accesses to one place cannot race (`compat_of_fit`).

Each task joins at the end of its block. A task starts no task, and a block does not run in a
loop.

A local lives in the whole function, or only in one main phase (its scope). Two locals whose
scopes are different phases can share a slot, which is the storage of a local. The code of a phase
uses only the locals that are live in it, and a task captures only the locals that are live in its
block. No reference outlives the local that it borrows. This is T3 of the design: a task cannot
leave its block, so the end of the block joins it, and then a later phase can use the slot.

A local can also be a raw pointer. The checker does not know where it points, and only code in an
`unsafe` region accesses through it. A raw pointer is not `Send` or `Sync`. Its type can opt in
with `unsafe impl Send` or `unsafe impl Sync`. The checker cannot check these claims, so the
theorems take them as a hypothesis (`Claims`). Code that runs while tasks of its block run uses
only raw pointers whose type opts in.

A task is a closure: its body and its captures. The rule for `spawn` is the rule for a closure
(S4), so closures add nothing here. The model has no functions, so it has no closure type in a
signature.
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

/-- The type of a local: a value, a shared reference `&x` or a mutable reference `&mut x` to
local `x`. The checker knows which local each reference borrows. -/
inductive LTy where
  | val (τ : Ty)
  | ref (x : Nat)
  | mutRef (x : Nat)
  /-- A raw pointer of pointer type `k`. The checker does not know where it points. -/
  | ptr (k : Nat)

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

/-- `a` allows no more than `b`. -/
def Perm.Le : Perm → Perm → Prop
  | .none, _ => True
  | .ref, b => b ≠ .none
  | .own, b => b = .own

theorem Allowed.mono {a b : Perm} {l : Leaf} {k : Kind} {o : Order} (h : Allowed a l k o)
    (hab : a.Le b) : Allowed b l k o := by
  cases a <;> cases b <;> cases l <;> simp_all [Allowed, Perm.Le]

/-- The permission through a local of type `t`, for code with permission `q` for that local. A
shared reference gives at most `ref`. -/
def via : LTy → Perm → Perm
  | .ref _, .own => .ref
  | _, q => q

theorem via_le (t : LTy) (q : Perm) : (via t q).Le q := by
  cases t <;> cases q <;> simp [via, Perm.Le]

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
  /-- An `unsafe` region: its code can access through a raw pointer. -/
  | unsafeCode (s : Src)

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

def compile (pl : Nat → Nat → Nat) : Src → Stmt
  | .skip => .skip
  | .assign r e => .assign r e
  | .read r x i => .load r (pl x i) .plain
  | .write x i e => .store (pl x i) e .plain
  | .fetchAdd r x i e => .fetchAdd r (pl x i) e .sc
  | .seq s t => .seq (compile pl s) (compile pl t)
  | .ite c s t => .ite c (compile pl s) (compile pl t)
  | .loop c s => .loop c (compile pl s)
  | .unsafeCode s => compile pl s

def compileItems (pl : Nat → Nat → Nat) : List Item → Stmt
  | [] => .skip
  | .code c :: is => .seq (compile pl c) (compileItems pl is)
  | .spawn u :: is => .seq (.spawnAt u 1 (.const 0)) (compileItems pl is)

def SPhase.compile (pl : Nat → Nat → Nat) : SPhase → TaskProg.Phase
  | .code c => .code (Typed.compile pl c)
  | .block is => .block (compileItems pl is) none

/-- A typed program. Leaf `i` of local `x` is the place `width * slot x + i`. A reference has no
place of its own: an access through it goes to the local that it borrows. -/
structure TProg where
  ty : Nat → LTy
  /-- The locals are below `locals`. -/
  locals : Nat
  /-- The scope of a local: the whole function (`none`), or only the main phase `k`. -/
  scope : Nat → Option Nat := fun _ => none
  /-- The storage of a local. Two locals whose scopes are different phases can share it. -/
  slot : Nat → Nat := id
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
  /-- The type that a raw pointer of pointer type `k` points to. -/
  pointee : Nat → Ty := fun _ => .leaf .int
  /-- Pointer type `k` has `unsafe impl Send`. -/
  implSend : Nat → Bool := fun _ => false
  /-- Pointer type `k` has `unsafe impl Sync`. -/
  implSync : Nat → Bool := fun _ => false
  /-- The slot that raw pointer `x` points to when the program runs. The checker does not read
  it. -/
  points : Nat → Nat := id

namespace TProg

variable (T : TProg)

/-- The permission of task `u` for local `x`. -/
def capPerm (u x : Nat) : Perm :=
  if (x, Mode.move) ∈ T.caps u ∨ (x, Mode.mut) ∈ T.caps u then .own
  else if (x, Mode.ref) ∈ T.caps u then .ref else .none

/-- The value type of local `x`. -/
def vty (x : Nat) : Ty :=
  match T.ty x with
  | .val τ => τ
  | .ptr k => T.pointee k
  | _ => .leaf .int

/-- The local that an access through `x` reaches: `x` itself, or the local that it borrows. A
raw pointer reaches itself, because the checker does not know where it points. -/
def target (x : Nat) : Nat :=
  match T.ty x with
  | .val _ | .ptr _ => x
  | .ref y | .mutRef y => y

/-- Local `x` is a raw pointer. -/
def IsPtr (x : Nat) : Prop := ∃ k, T.ty x = .ptr k

/-- The slot that an access to local `x` goes to. For a raw pointer, it is the slot that the
pointer points to. -/
def store (x : Nat) : Nat :=
  match T.ty x with
  | .ptr _ => T.points x
  | _ => T.slot x

/-- Code can use local `x` while other tasks run: `x` is not a raw pointer, or its type has an
`unsafe impl`. -/
def Shared (x : Nat) : Prop :=
  match T.ty x with
  | .ptr k => T.implSend k = true ∨ T.implSync k = true
  | _ => True

/-- An access through `x` to a raw pointer is in an `unsafe` region (`uns`). Its type has an
`unsafe impl`, or no task of the block can run at the same time (`raw`). -/
def PtrOk (raw uns : Bool) (x : Nat) : Prop := T.IsPtr x → uns = true ∧ (raw = true ∨ T.Shared x)

/-- The place of leaf `i` through local `x`, with one slot for each local. `TypedChan.lean` uses
this. -/
def place (x i : Nat) : Nat := T.width * T.target x + i

/-- The place of leaf `i` through local `x`, in the slot of the local that it reaches. -/
def slotPlace (x i : Nat) : Nat := T.width * T.store (T.target x) + i

/-- Local `x` is live in the main phase `k`. -/
def Live (x k : Nat) : Prop := T.scope x = none ∨ T.scope x = some k

/-- `&y` is `Send` when `y` is `Sync`, and `&mut y` is `Send` when `y` is `Send`. -/
def SendL (x : Nat) : Prop :=
  match T.ty x with
  | .val τ => τ.Send
  | .ref y => (T.vty y).Sync
  | .mutRef y => (T.vty y).Send
  | .ptr k => T.implSend k = true

/-- `&y` and `&mut y` are `Sync` when `y` is `Sync`. -/
def SyncL (x : Nat) : Prop :=
  match T.ty x with
  | .val τ => τ.Sync
  | .ref y | .mutRef y => (T.vty y).Sync
  | .ptr k => T.implSync k = true

/-- The permission of task `u` for an access through local `p`. -/
def pathPerm (u p : Nat) : Perm := via (T.ty p) (T.capPerm u p)

/-- The permission of task `u` for local `x`: the largest one through its captures that reach
`x`. -/
def taskPerm (u x : Nat) : Perm :=
  if ∃ c ∈ T.caps u, T.target c.1 = x ∧ T.pathPerm u c.1 = .own then .own
  else if ∃ c ∈ T.caps u, T.target c.1 = x ∧ T.pathPerm u c.1 = .ref then .ref else .none

/-- The permission of the code of a block for local `x`. It has none when a task of the block
owns `x`, and `ref` when a task of the block borrows `x`. -/
def blockPerm (is : List Item) (x : Nat) : Perm :=
  if ∃ u ∈ spawnsOf is, T.taskPerm u x = .own then .none
  else if ∃ u ∈ spawnsOf is, T.taskPerm u x = .ref then .ref else .own

/-- `x` is a local, `i` is one of its leaves, and `P` holds for that leaf. -/
def At (x i : Nat) (P : Leaf → Prop) : Prop :=
  x < T.locals ∧ ∃ l, (T.vty x).leaves[i]? = some l ∧ P l

/-- The code follows the permissions `perm`, uses only the locals in `live`, and uses only the
registers below `regs`. Here `perm p` is the permission for an access through local `p`. The code
is in an `unsafe` region when `uns` holds. It can access through each raw pointer when `raw`
holds. -/
def Fits (raw : Bool) (live : Nat → Prop) (perm : Nat → Perm) : Bool → Src → Prop
  | _, .skip => True
  | _, .assign r e => r < T.regs ∧ e.RegsBelow T.regs
  | uns, .read r x i => r < T.regs ∧ live x ∧ T.PtrOk raw uns x ∧
      T.At (T.target x) i fun l => Allowed (perm x) l .read .plain
  | uns, .write x i e => e.RegsBelow T.regs ∧ live x ∧ T.PtrOk raw uns x ∧
      T.At (T.target x) i fun l => Allowed (perm x) l .write .plain
  | uns, .fetchAdd r x i e => r < T.regs ∧ e.RegsBelow T.regs ∧ live x ∧ T.PtrOk raw uns x ∧
      T.At (T.target x) i fun l => Allowed (perm x) l .read .sc ∧ Allowed (perm x) l .write .sc
  | uns, .seq s t => Fits raw live perm uns s ∧ Fits raw live perm uns t
  | uns, .ite c s t => c.RegsBelow T.regs ∧ Fits raw live perm uns s ∧ Fits raw live perm uns t
  | uns, .loop c s => c.RegsBelow T.regs ∧ Fits raw live perm uns s
  | _, .unsafeCode s => Fits raw live perm true s

/-- The permission of the code of a block for an access through local `p`. -/
def blockFits (is : List Item) (p : Nat) : Perm := via (T.ty p) (T.blockPerm is (T.target p))

/-- The rules of the main phase `k`. Its code uses only the locals that are live in it. The code
of a block follows `blockPerm`, and uses only the raw pointers whose type has an `unsafe impl`.
A task that owns a local excludes each other task of the block from it. A task captures only the
locals that are live in its block. -/
def PhaseOk (k : Nat) : SPhase → Prop
  | .code c => T.Fits true (T.Live · k) (fun p => via (T.ty p) .own) false c
  | .block is => (∀ c, Item.code c ∈ is → T.Fits false (T.Live · k) (T.blockFits is) false c) ∧
      (∀ u ∈ spawnsOf is, ∀ v ∈ spawnsOf is, u ≠ v → ∀ x, T.taskPerm u x = .own →
        T.taskPerm v x = .none) ∧
      ∀ u ∈ spawnsOf is, ∀ c ∈ T.caps u, T.Live c.1 k

/-- The tasks that the main code starts. -/
def spawns : List Nat := T.main.flatMap SPhase.spawns

/-- The rules of the type checker. -/
structure WellTyped : Prop where
  width : ∀ x, (T.vty x).leaves.length ≤ T.width
  /-- A reference borrows a local that is a value. -/
  targets : ∀ x y, T.ty x = .ref y ∨ T.ty x = .mutRef y → ∃ τ, T.ty y = .val τ
  /-- A task moves or borrows `mut` only a `Send` local, and borrows `ref` only a `Sync` one. -/
  caps : ∀ u x m, (x, m) ∈ T.caps u → (m = .ref → T.SyncL x) ∧ (m ≠ .ref → T.SendL x)
  /-- The code of each task uses only its captures. -/
  bodies : ∀ u, T.Fits true (fun _ => True) (T.pathPerm u) false (T.body u)
  phases : ∀ k p, T.main[k]? = some p → T.PhaseOk k p
  slot_lt : ∀ x, x < T.locals → T.slot x < T.locals
  /-- Two locals that are live in one phase do not share a slot. -/
  slots : ∀ x y k, T.Live x k → T.Live y k → T.slot x = T.slot y → x = y
  /-- No reference outlives the local that it borrows. -/
  refs : ∀ x k, T.Live x k → T.Live (T.target x) k
  /-- One place in the code starts each task. -/
  nodup : T.spawns.Nodup
  spawn_lt : ∀ u ∈ T.spawns, u < T.count

/-- The claims of the `unsafe` code and of each `unsafe impl`. The checker cannot check them, so
the theorems take them as a hypothesis. -/
structure Claims : Prop where
  /-- Each raw pointer points to a slot. -/
  valid : ∀ x, x < T.locals → T.IsPtr x → T.points x < T.locals
  /-- A raw pointer whose type has an `unsafe impl` owns the slot that it points to: no other
  local that is live with it uses that slot. -/
  owns : ∀ x y k, T.IsPtr x → T.Shared x → T.Shared y → T.Live x k → T.Live y k →
    T.store x = T.store y → x = y
  /-- `unsafe impl Sync`: two tasks can use the value that the pointer points to through shared
  borrows. -/
  sync : ∀ x k, T.ty x = .ptr k → T.implSync k = true → (T.pointee k).Sync

def phases : List TaskProg.Phase := T.main.map (SPhase.compile T.slotPlace)

def toTaskProg : TaskProg where
  main := TaskProg.seqAll T.phases
  task u := compile T.slotPlace (T.body u)
  count := T.count
  flags := T.width * T.locals
  marks := T.regs

/-! ## The proof -/

variable {T}

/-- A program with no raw pointers makes no claim. -/
theorem Claims.of_noPtrs (h : ∀ x k, T.ty x ≠ .ptr k) : T.Claims where
  valid _ _ hp := absurd hp.choose_spec (h _ _)
  owns _ _ _ hp := absurd hp.choose_spec (h _ _)
  sync x k hx := absurd hx (h x k)

/-- The accesses that the permissions `perm` allow in the main phase `k`, at the places of the
locals that are live in it and that code can use while other tasks run. -/
def Legal (k : Nat) (perm : Nat → Perm) (kd : Kind) (o : Order) (p : Nat) : Prop :=
  ∃ x i, p = T.width * T.store x + i ∧ T.Live x k ∧ T.Shared x ∧
    T.At x i fun l => Allowed (perm x) l kd o

theorem Allowed.ne_none {q : Perm} {l : Leaf} {kd : Kind} {o : Order} (h : Allowed q l kd o) :
    q ≠ .none := by
  rintro rfl
  cases l <;> exact h

theorem ptr_target {x : Nat} (h : T.IsPtr x) : T.target x = x := by
  obtain ⟨k, hk⟩ := h
  simp [target, hk]

theorem store_of_not_ptr {x : Nat} (h : ¬ T.IsPtr x) : T.store x = T.slot x := by
  unfold store
  cases hx : T.ty x with
  | ptr k => exact absurd ⟨k, hx⟩ h
  | _ => rfl

/-- An access through a local that is not a raw pointer reaches a local that is not one. -/
theorem shared_target (hw : T.WellTyped) {x : Nat} (h : ¬ T.IsPtr x) : T.Shared (T.target x) := by
  cases hx : T.ty x with
  | val => simp [target, Shared, hx]
  | ref y | mutRef y =>
    obtain ⟨τ, hτ⟩ := hw.targets x y (by simp [hx])
    simp [target, Shared, hx, hτ]
  | ptr k => exact absurd ⟨k, hx⟩ h

/-- Two locals that are live in one phase, and that code can use while other tasks run, do not
share a slot. -/
theorem store_inj (hw : T.WellTyped) (hc : T.Claims) {x y k : Nat} (hx : T.Live x k)
    (hy : T.Live y k) (hsx : T.Shared x) (hsy : T.Shared y) (h : T.store x = T.store y) : x = y := by
  by_cases px : T.IsPtr x
  · exact hc.owns x y k px hsx hsy hx hy h
  · by_cases py : T.IsPtr y
    · exact (hc.owns y x k py hsy hsx hy hx h.symm).symm
    · rw [store_of_not_ptr px, store_of_not_ptr py] at h
      exact hw.slots x y k hx hy h

/-- The accesses of code with the permissions `perm` through each local. Each one is legal for
`tp` at the local that it reaches, when `tp` allows at least `perm` there and that local is live
in phase `k`. When the code can access through each raw pointer (`raw`), each raw pointer that
it uses must have an `unsafe impl` (`hsh`). -/
theorem Fits.accesses (hw : T.WellTyped) {k : Nat} {raw : Bool} {live : Nat → Prop}
    {perm tp : Nat → Perm} (hle : ∀ p, (perm p).Le (tp (T.target p)))
    (hlv : ∀ x, live x → perm x ≠ .none → T.Live (T.target x) k)
    (hsh : raw = true → ∀ x, perm x ≠ .none → T.IsPtr x → T.Shared x) {uns : Bool} {c : Src}
    (h : T.Fits raw live perm uns c) : (compile T.slotPlace c).Accesses (T.Legal k tp) := by
  have hs : ∀ x uns, T.PtrOk raw uns x → perm x ≠ .none → T.Shared (T.target x) :=
    fun x _ hp hn => by
      by_cases hx : T.IsPtr x
      · rw [ptr_target hx]
        rcases (hp hx).2 with hr | hr
        · exact hsh hr x hn hx
        · exact hr
      · exact shared_target hw hx
  induction c generalizing uns with
  | skip | assign => trivial
  | read _ x i | write x i =>
    obtain ⟨-, hv, hp, hx, l, hl, ha⟩ := h
    exact ⟨_, i, rfl, hlv x hv (Allowed.ne_none ha), hs x _ hp (Allowed.ne_none ha), hx, l, hl,
      ha.mono (hle x)⟩
  | fetchAdd _ x i =>
    obtain ⟨-, -, hv, hp, hx, l, hl, h1, h2⟩ := h
    have hk := hlv x hv (Allowed.ne_none h1)
    have hh := hs x _ hp (Allowed.ne_none h1)
    exact ⟨⟨_, i, rfl, hk, hh, hx, l, hl, h1.mono (hle x)⟩,
      ⟨_, i, rfl, hk, hh, hx, l, hl, h2.mono (hle x)⟩⟩
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ih h.2
  | unsafeCode _ ih => exact ih h

theorem Fits.regs {raw : Bool} {live : Nat → Prop} {perm : Nat → Perm} {uns : Bool} {c : Src}
    (h : T.Fits raw live perm uns c) : (compile T.slotPlace c).RegsBelow T.regs := by
  induction c generalizing uns with
  | skip => trivial
  | assign | write | ite _ _ _ ih1 ih2 => first
    | exact h
    | exact h.1
    | exact ⟨h.1, ih1 h.2.1, ih2 h.2.2⟩
  | read => exact h.1
  | fetchAdd => exact ⟨h.1, h.2.1⟩
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | loop _ _ ih => exact ⟨h.1, ih h.2⟩
  | unsafeCode _ ih => exact ih h

theorem At.lt (hw : T.WellTyped) (hc : T.Claims) {x i : Nat} {P : Leaf → Prop} (h : T.At x i P) :
    T.width * T.store x + i < T.width * T.locals := by
  obtain ⟨hx, l, hl, -⟩ := h
  have hx : T.store x < T.locals := by
    by_cases hp : T.IsPtr x
    · obtain ⟨k, hk⟩ := hp
      simpa [store, hk] using hc.valid x hx ⟨k, hk⟩
    · rw [store_of_not_ptr hp]
      exact hw.slot_lt x hx
  have hi : i < T.width := Nat.lt_of_lt_of_le ((List.getElem?_eq_some_iff.1 hl).1)
    (hw.width x)
  calc T.width * T.store x + i < T.width * T.store x + T.width := Nat.add_lt_add_left hi _
    _ = T.width * (T.store x + 1) := (Nat.mul_succ _ _).symm
    _ ≤ T.width * T.locals := Nat.mul_le_mul_left _ hx

theorem Fits.below (hw : T.WellTyped) (hc : T.Claims) {raw : Bool} {live : Nat → Prop}
    {perm : Nat → Perm} {uns : Bool} {c : Src} (h : T.Fits raw live perm uns c) (n : Nat) :
    (compile T.slotPlace c).Below (T.width * T.locals) n := by
  induction c generalizing uns with
  | skip | assign => trivial
  | read | write => exact At.lt hw hc h.2.2.2
  | fetchAdd => exact At.lt hw hc h.2.2.2.2
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ih h.2
  | unsafeCode _ ih => exact ih h

theorem compile_noTasks (pl : Nat → Nat → Nat) (c : Src) : (compile pl c).NoTasks := by
  induction c with
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1, ih2⟩
  | loop _ _ ih | unsafeCode _ ih => exact ih
  | _ => trivial

theorem compile_noBlocks (pl : Nat → Nat → Nat) (c : Src) : (compile pl c).NoBlocks := by
  induction c with
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1, ih2⟩
  | loop _ _ ih | unsafeCode _ ih => exact ih
  | _ => trivial

theorem compile_seqCst (pl : Nat → Nat → Nat) (c : Src) : (compile pl c).SeqCstOnly := by
  induction c with
  | read | write => exact Or.inl rfl
  | fetchAdd => exact Or.inr rfl
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1, ih2⟩
  | loop _ _ ih | unsafeCode _ ih => exact ih
  | _ => trivial

theorem compile_spawnIds (pl : Nat → Nat → Nat) (c : Src) : (compile pl c).spawnIds = [] := by
  induction c with
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => simp [compile, Stmt.spawnIds, ih1, ih2]
  | loop _ _ ih => simp [compile, Stmt.spawnIds, ih]
  | unsafeCode _ ih => simpa [compile] using ih
  | _ => rfl

theorem compileItems_spawnIds (pl : Nat → Nat → Nat) (is : List Item) :
    (compileItems pl is).spawnIds = spawnsOf is := by
  induction is with
  | nil => rfl
  | cons it is ih =>
    cases it <;> simp [compileItems, spawnsOf, Stmt.spawnIds, compile_spawnIds, ih]

/-- Two pieces of code of one phase whose permissions fit for each local are apart. Two locals
that are live in the phase do not share a slot. -/
theorem apart_of_fit (hw : T.WellTyped) (hc : T.Claims) {k : Nat} {a b : Nat → Perm} {s t : Stmt}
    (hs : s.Accesses (T.Legal k a)) (ht : t.Accesses (T.Legal k b))
    (hfit : ∀ x l, l ∈ (T.vty x).leaves → Fit (a x) (b x) l) (own : Nat → Prop) :
    s.Apart own t := by
  refine Stmt.Accesses.mono ?_ hs
  rintro kd o _ ⟨x, i, rfl, hkx, hsx, hx, l, hl, ha⟩ -
  refine Stmt.Accesses.mono ?_ ht
  rintro kd' o' _ ⟨y, j, rfl, hky, hsy, hy, l', hl', hb⟩
  have hi : i < T.width := Nat.lt_of_lt_of_le ((List.getElem?_eq_some_iff.1 hl).1)
    (hw.width x)
  have hj : j < T.width := Nat.lt_of_lt_of_le ((List.getElem?_eq_some_iff.1 hl').1)
    (hw.width y)
  by_cases he : T.width * T.store x + i = T.width * T.store y + j
  · have hpos : 0 < T.width := Nat.lt_of_le_of_lt (Nat.zero_le _) hi
    have hxy : x = y := by
      have := congrArg (· / T.width) he
      simp only [Nat.mul_add_div hpos, Nat.div_eq_of_lt hi, Nat.div_eq_of_lt hj] at this
      exact store_inj hw hc hkx hky hsx hsy (by omega)
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

theorem capPerm_own {u x : Nat} (h : T.capPerm u x = .own) :
    ∃ m, (x, m) ∈ T.caps u ∧ m ≠ .ref := by
  unfold capPerm at h
  split at h
  · rename_i hm
    rcases hm with hm | hm
    · exact ⟨_, hm, nofun⟩
    · exact ⟨_, hm, nofun⟩
  · split at h <;> cases h

theorem capPerm_mem {u x : Nat} (h : T.capPerm u x ≠ .none) : ∃ m, (x, m) ∈ T.caps u := by
  cases hc : T.capPerm u x with
  | none => exact absurd hc h
  | ref => exact ⟨_, capPerm_ref hc⟩
  | own => obtain ⟨m, hm, -⟩ := capPerm_own hc; exact ⟨m, hm⟩

/-- An access of task `u` through local `p` needs no more than the permission of the task for the
local that it reaches. -/
theorem pathPerm_le (u p : Nat) : (T.pathPerm u p).Le (T.taskPerm u (T.target p)) := by
  have hmem : T.pathPerm u p ≠ .none → ∃ c ∈ T.caps u, c.1 = p := fun h => by
    have : T.capPerm u p ≠ .none := fun hc => h (by rw [pathPerm, hc]; cases T.ty p <;> rfl)
    obtain ⟨m, hm⟩ := capPerm_mem this
    exact ⟨_, hm, rfl⟩
  unfold taskPerm
  cases h : T.pathPerm u p with
  | none => trivial
  | own =>
    obtain ⟨c, hc, rfl⟩ := hmem (by rw [h]; nofun)
    split
    · rfl
    · rename_i hn
      exact absurd ⟨c, hc, rfl, h⟩ hn
  | ref =>
    obtain ⟨c, hc, rfl⟩ := hmem (by rw [h]; nofun)
    split
    · nofun
    · split
      · nofun
      · rename_i _ hn
        exact absurd ⟨c, hc, rfl, h⟩ hn

/-- A task borrows a local with `ref` only when the local is `Sync`. The task can capture it with
`ref`, or capture a reference to it. For a raw pointer, this is the claim of `unsafe impl Sync`. -/
theorem sync_of_ref (hw : T.WellTyped) (hc : T.Claims) {u x : Nat} (h : T.taskPerm u x = .ref) :
    (T.vty x).Sync := by
  unfold taskPerm at h
  split at h
  · cases h
  · split at h
    · rename_i _ hex
      obtain ⟨⟨p, m⟩, -, rfl, hp⟩ := hex
      simp only [pathPerm] at hp ⊢
      cases ht : T.ty p with
      | val τ =>
        rw [ht] at hp
        simp only [via] at hp
        have := (hw.caps u p .ref (capPerm_ref hp)).1 rfl
        simp only [SyncL, ht] at this
        simpa [target, vty, ht] using this
      | ref y =>
        rw [ht] at hp
        have hy : (T.vty y).Sync := by
          cases hc : T.capPerm u p with
          | none => rw [hc] at hp; cases hp
          | ref =>
            have := (hw.caps u p .ref (capPerm_ref hc)).1 rfl
            simpa [SyncL, ht] using this
          | own =>
            obtain ⟨m, hm, hne⟩ := capPerm_own hc
            have := (hw.caps u p m hm).2 hne
            simpa [SendL, ht] using this
        simpa [target, ht] using hy
      | mutRef y =>
        rw [ht] at hp
        simp only [via] at hp
        have := (hw.caps u p .ref (capPerm_ref hp)).1 rfl
        simpa [SyncL, target, ht] using this
      | ptr k =>
        rw [ht] at hp
        simp only [via] at hp
        have := (hw.caps u p .ref (capPerm_ref hp)).1 rfl
        simp only [SyncL, ht] at this
        simpa [target, vty, ht] using hc.sync p k ht this
    · cases h

/-- Two tasks of a block fit. -/
theorem fit_tasks (hw : T.WellTyped) (hc : T.Claims) {k : Nat} {is : List Item} (hp : T.PhaseOk k (.block is))
    {u v : Nat}
    (hu : u ∈ spawnsOf is) (hv : v ∈ spawnsOf is) (huv : u ≠ v) (x : Nat) (l : Leaf)
    (hl : l ∈ (T.vty x).leaves) : Fit (T.taskPerm u x) (T.taskPerm v x) l := by
  have excl := hp.2.1
  cases h1 : T.taskPerm u x with
  | none => exact Or.inl rfl
  | own => exact Or.inr (Or.inl (excl u hu v hv huv x h1))
  | ref =>
    cases h2 : T.taskPerm v x with
    | none => exact Or.inr (Or.inl rfl)
    | own =>
      have := excl v hv u hu (Ne.symm huv) x h2
      rw [h1] at this
      cases this
    | ref => exact Or.inr (Or.inr ⟨rfl, rfl, (sync_of_ref hw hc h1).leaf hl⟩)

/-- The code of a block and a task of the block fit. -/
theorem fit_block (hw : T.WellTyped) (hc : T.Claims) {is : List Item} {u : Nat} (hu : u ∈ spawnsOf is)
    (x : Nat) (l : Leaf) (hl : l ∈ (T.vty x).leaves) : Fit (T.blockPerm is x) (T.taskPerm u x) l := by
  unfold blockPerm
  split
  · exact Or.inl rfl
  · rename_i h1
    split
    · cases h : T.taskPerm u x with
      | none => exact Or.inr (Or.inl rfl)
      | own => exact absurd ⟨u, hu, h⟩ h1
      | ref => exact Or.inr (Or.inr ⟨rfl, rfl, (sync_of_ref hw hc h).leaf hl⟩)
    · rename_i h2
      cases h : T.taskPerm u x with
      | none => exact Or.inr (Or.inl rfl)
      | own => exact absurd ⟨u, hu, h⟩ h1
      | ref => exact absurd ⟨u, hu, h⟩ h2

theorem compileItems_accesses (hw : T.WellTyped) {k : Nat} {live : Nat → Prop} {is : List Item}
    {perm tp : Nat → Perm} (hle : ∀ p, (perm p).Le (tp (T.target p)))
    (hlv : ∀ x, live x → perm x ≠ .none → T.Live (T.target x) k)
    (h : ∀ c, Item.code c ∈ is → T.Fits false live perm false c) :
    (compileItems T.slotPlace is).Accesses (T.Legal k tp) := by
  induction is with
  | nil => trivial
  | cons it is ih =>
    cases it with
    | code c => exact ⟨(h c (List.mem_cons_self ..)).accesses hw hle hlv (by simp),
        ih fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩
    | spawn => exact ⟨trivial, ih fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩

theorem compileItems_regs {live : Nat → Prop} {is : List Item} {perm : Nat → Perm}
    (h : ∀ c, Item.code c ∈ is → T.Fits false live perm false c) : (compileItems T.slotPlace is).RegsBelow T.regs := by
  induction is with
  | nil => trivial
  | cons it is ih =>
    cases it with
    | code c => exact ⟨(h c (List.mem_cons_self ..)).regs, ih fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩
    | spawn => exact ⟨trivial, ih fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩

theorem compileItems_noBlocks (pl : Nat → Nat → Nat) (is : List Item) : (compileItems pl is).NoBlocks := by
  induction is with
  | nil => trivial
  | cons it is ih => cases it <;> exact ⟨by first | exact compile_noBlocks pl _ | trivial, ih⟩

theorem compileItems_seqCst (pl : Nat → Nat → Nat) (is : List Item) : (compileItems pl is).SeqCstOnly := by
  induction is with
  | nil => trivial
  | cons it is ih => cases it <;> exact ⟨by first | exact compile_seqCst pl _ | trivial, ih⟩

theorem compileItems_below (hw : T.WellTyped) (hc : T.Claims) {live : Nat → Prop} {is : List Item}
    {perm : Nat → Perm} (h : ∀ c, Item.code c ∈ is → T.Fits false live perm false c) (hs : ∀ u ∈ spawnsOf is, u < T.count) :
    (compileItems T.slotPlace is).Below (T.width * T.locals) T.count := by
  induction is with
  | nil => trivial
  | cons it is ih =>
    cases it with
    | code c =>
      exact ⟨(h c (List.mem_cons_self ..)).below hw hc _,
        ih (fun c' hc => h c' (List.mem_cons_of_mem _ hc)) (fun u hu => hs u hu)⟩
    | spawn u =>
      exact ⟨hs u (List.mem_cons_self ..),
        ih (fun c' hc => h c' (List.mem_cons_of_mem _ hc)) fun v hv => hs v (List.mem_cons_of_mem _ hv)⟩

/-- An access of a task of a block reaches a local that is live in the phase of the block. -/
theorem live_of_path (hw : T.WellTyped) {k : Nat} {is : List Item} (hp : T.PhaseOk k (.block is))
    {u : Nat} (hu : u ∈ spawnsOf is) (x : Nat) (_ : True) (h : T.pathPerm u x ≠ .none) :
    T.Live (T.target x) k := by
  have : T.capPerm u x ≠ .none := fun hc => h (by rw [pathPerm, hc]; cases T.ty x <;> rfl)
  obtain ⟨m, hm⟩ := capPerm_mem this
  exact hw.refs x k (hp.2.2 u hu (x, m) hm)

/-- A task captures a raw pointer only when its type has an `unsafe impl`. -/
theorem shared_of_path (hw : T.WellTyped) (u : Nat) :
    true = true → ∀ x, T.pathPerm u x ≠ .none → T.IsPtr x → T.Shared x := by
  intro _ x h ⟨k, hk⟩
  have : T.capPerm u x ≠ .none := fun hc => h (by rw [pathPerm, hc]; cases T.ty x <;> rfl)
  obtain ⟨m, hm⟩ := capPerm_mem this
  have hc := hw.caps u x m hm
  simp only [Shared, hk]
  by_cases hr : m = .ref
  · have := hc.1 hr
    simp only [SyncL, hk] at this
    exact Or.inr this
  · have := hc.2 hr
    simp only [SendL, hk] at this
    exact Or.inl this

/-- The rule of a block holds for the compiled block. -/
theorem blockOk (hw : T.WellTyped) (hc : T.Claims) {k : Nat} {is : List Item} (hp : T.PhaseOk k (.block is)) :
    T.toTaskProg.BlockOk (· < T.toTaskProg.flags) (fun _ => []) (compileItems T.slotPlace is) := by
  have hacc := compileItems_accesses hw (k := k) (tp := T.blockPerm is) (fun p => via_le _ _)
    (fun x hx _ => hw.refs x k hx) hp.1
  have hall := Stmt.allIds_of_noBlocks (compileItems_noBlocks T.slotPlace is)
  refine ⟨.of_noBlocks (compileItems_noBlocks _ _), compileItems_regs hp.1, ?_, ?_⟩
  · intro u hu w hd
    rw [hall, compileItems_spawnIds] at hu
    rw [TaskProg.desc_nil] at hd
    subst w
    exact apart_of_fit hw hc hacc
      ((hw.bodies u).accesses hw (pathPerm_le u) (live_of_path hw hp hu) (shared_of_path hw u))
      (fun x l hl => fit_block hw hc hu x l hl) _
  · intro u hu v hv huv _ w w' hd hd'
    rw [hall, compileItems_spawnIds] at hu hv
    rw [TaskProg.desc_nil] at hd hd'
    subst w w'
    exact apart_of_fit hw hc
      ((hw.bodies u).accesses hw (pathPerm_le u) (live_of_path hw hp hu) (shared_of_path hw u))
      ((hw.bodies v).accesses hw (pathPerm_le v) (live_of_path hw hp hv) (shared_of_path hw v))
      (fun x l hl => fit_tasks hw hc hp hu hv huv x l hl) _

theorem phase_ids (pl : Nat → Nat → Nat) (p : SPhase) : (p.compile pl).ids = p.spawns := by
  cases p with
  | code => rfl
  | block is =>
    exact (Stmt.allIds_of_noBlocks (compileItems_noBlocks pl is)).trans (compileItems_spawnIds pl is)

/-- The phases from index `n` on follow their rules. -/
def PhasesFrom (n : Nat) (ps : List SPhase) : Prop := ∀ i p, ps[i]? = some p → T.PhaseOk (n + i) p

theorem PhasesFrom.head {n : Nat} {p : SPhase} {ps : List SPhase} (h : T.PhasesFrom n (p :: ps)) :
    T.PhaseOk n p := h 0 p rfl

theorem PhasesFrom.tail {n : Nat} {p : SPhase} {ps : List SPhase} (h : T.PhasesFrom n (p :: ps)) :
    T.PhasesFrom (n + 1) ps := fun i q hq => by
  have := h (i + 1) q hq
  rwa [show n + (i + 1) = n + 1 + i by omega] at this

theorem phasesOk (hw : T.WellTyped) (hc : T.Claims) :
    ∀ (ps : List SPhase) (n : Nat), T.PhasesFrom n ps → (ps.flatMap SPhase.spawns).Nodup →
      T.toTaskProg.PhasesOk (· < T.toTaskProg.flags) (fun _ => []) (ps.map (SPhase.compile T.slotPlace))
  | [], _, _, _ => trivial
  | p :: ps, n, hp, hn => by
    rw [List.flatMap_cons, List.nodup_append] at hn
    refine ⟨?_, ?_, phasesOk hw hc ps (n + 1) hp.tail hn.2.1⟩
    · cases p with
      | code c => exact compile_noTasks _ c
      | block is => exact blockOk hw hc hp.head
    · intro u hu q hq huq
      rw [phase_ids] at hu
      obtain ⟨q', hq', rfl⟩ := List.mem_map.1 hq
      rw [phase_ids] at huq
      exact hn.2.2 u hu u (List.mem_flatMap.2 ⟨q', hq', huq⟩) rfl

theorem seqAll_below (hw : T.WellTyped) (hc : T.Claims) :
    ∀ (ps : List SPhase) (n : Nat), T.PhasesFrom n ps →
      (∀ u ∈ ps.flatMap SPhase.spawns, u < T.count) →
      (TaskProg.seqAll (ps.map (SPhase.compile T.slotPlace))).Below (T.width * T.locals) T.count
  | [], _, _, _ => trivial
  | p :: ps, n, hp, hs => by
    refine ⟨?_, seqAll_below hw hc ps (n + 1) hp.tail
      fun u hu => hs u (by rw [List.flatMap_cons]; exact List.mem_append_right _ hu)⟩
    have hs' : ∀ u ∈ p.spawns, u < T.count := fun u hu =>
      hs u (by rw [List.flatMap_cons]; exact List.mem_append_left _ hu)
    cases p with
    | code c => exact hp.head.below hw hc _
    | block is => exact compileItems_below hw hc hp.head.1 hs'

theorem seqAll_seqCst (pl : Nat → Nat → Nat) :
    ∀ ps : List SPhase, (TaskProg.seqAll (ps.map (SPhase.compile pl))).SeqCstOnly
  | [] => trivial
  | p :: ps => ⟨by cases p with
      | code c => exact compile_seqCst pl c
      | block is => exact compileItems_seqCst pl is, seqAll_seqCst pl ps⟩

theorem WellTyped.phasesFrom (hw : T.WellTyped) : T.PhasesFrom 0 T.main := fun i p h => by
  rw [Nat.zero_add]
  exact hw.phases i p h

/-- The code of a well-typed program follows the footprint rule, when the claims of its `unsafe`
code and of its `unsafe impl` items hold. -/
theorem WellTyped.footprints (hw : T.WellTyped) (hc : T.Claims) :
    T.toTaskProg.Footprints (· < T.toTaskProg.flags) (fun _ => []) T.phases where
  ok := ⟨seqAll_below hw hc _ 0 hw.phasesFrom hw.spawn_lt, fun u => (hw.bodies u).below hw hc _⟩
  le_flags := fun _ h => h
  main := rfl
  phases := phasesOk hw hc _ 0 hw.phasesFrom hw.nodup
  tasks := fun u => Or.inr ⟨compile_noTasks _ _, rfl⟩
  task_phases := fun _ => trivial
  one_parent := TaskProg.one_parent_nil _
  order := fun _ _ ⟨_, hp, _⟩ => by simp at hp

/-- A well-typed program has no data race, in each execution, when its claims hold. -/
theorem WellTyped.raceFree (hw : T.WellTyped) (hc : T.Claims) {X : Execution}
    (hX : T.toTaskProg.toProg.Execs X) : X.RaceFree :=
  TaskProg.raceFree (hw.footprints hc) hX

/-- DRF-SC for a well-typed program: each finite execution that the model allows is an
interleaving. -/
theorem WellTyped.drf_sc (hw : T.WellTyped) (hcl : T.Claims) {X : Execution}
    (hX : T.toTaskProg.toProg.Execs X) (wf : X.WellFormed) (hfin : X.Finite) (hc : X.Consistent) :
    X.SeqConsistent :=
  TaskProg.drf_sc_of_footprints (hw.footprints hcl) (seqAll_seqCst _ _)
    (fun _ => compile_seqCst _ _) hX wf hfin hc

end TProg

end Typed

end Vx
