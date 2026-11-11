import VxMemoryModel.Typed
import VxMemoryModel.LockFootprint

/-!
# The typed layer with channels and a lock

`Typed.lean` compiles a typed language to `TaskProg`. This file adds channels and locks, and
compiles to `ChanProg`. `WellTyped.cfootprints` shows that the compiled program follows the rule
of `ChanFootprint.lean` and `LockFootprint.lean` (`CFootprints`). So a well-typed program with
channels and locks has no data race (`WellTyped.raceFree`).

A local can be the value of a mutex with the lock `m` (`guard`). The guard map `g` gives each
place of such a local the lock `m`. An access to a leaf of a mutex needs the lock `m` held: the
code runs in a `locked m` block. A mutex is `Send` and `Sync` when its value is `Send`. So two
tasks can borrow one mutex with `ref` and write its value, also when the value is a cell. The rule
for a local with no lock is the rule of `Typed.lean`.

`send` and `recv` use only the places of the channels, and those never race (`chan_no_race`). So
the checker only checks that a channel exists and that the registers are free. A value in a
channel is a number. A reference is not a value in this model, so no reference can go into a
channel.

Each task joins at the end of its block, a task starts no task, and a block does not run in a
loop. The code has no failures. This file has no raw pointers and no `unsafe` code.
-/

namespace Vx

namespace Typed

open Classical

/-- The code of the typed language with channels and locks. -/
inductive CSrc where
  | code (c : Src)
  /-- Send the value of `e` on the channel `ch`. -/
  | send (ch : Nat) (e : Expr)
  /-- Take the next value of the channel `ch` into `r`, and set `s` to 1, or to 0 when the
  channel closed. -/
  | recv (r s ch : Nat)
  /-- Take the lock `m`, run `s`, and let the lock go. -/
  | locked (m : Nat) (s : CSrc)
  | seq (s t : CSrc)
  | ite (c : Expr) (s t : CSrc)
  | loop (c : Expr) (s : CSrc)

/-- A part of a block: code, or a `spawn` of task `u`. -/
inductive CItem where
  | code (c : CSrc)
  | spawn (u : Nat)

/-- A part of the main code: code that starts no task, or a `tasks` block. -/
inductive CPh where
  | code (c : CSrc)
  | block (items : List CItem)

def cspawnsOf : List CItem → List Nat
  | [] => []
  | .code _ :: is => cspawnsOf is
  | .spawn u :: is => u :: cspawnsOf is

def CPh.spawns : CPh → List Nat
  | .code _ => []
  | .block is => cspawnsOf is

/-! ## The compiler -/

def ccompile (pl : Nat → Nat → Nat) : CSrc → CStmt
  | .code c => .base (compile pl c)
  | .send ch e => .send ch e
  | .recv r s ch => .recv r s ch
  | .locked m s => .locked m (ccompile pl s)
  | .seq s t => .seq (ccompile pl s) (ccompile pl t)
  | .ite c s t => .ite c (ccompile pl s) (ccompile pl t)
  | .loop c s => .loop c (ccompile pl s)

def ccompileItems (pl : Nat → Nat → Nat) : List CItem → CStmt
  | [] => .base .skip
  | .code c :: is => .seq (ccompile pl c) (ccompileItems pl is)
  | .spawn u :: is => .seq (.base (.spawnAt u 1 (.const 0))) (ccompileItems pl is)

def CPh.compile (pl : Nat → Nat → Nat) : CPh → ChanProg.CPhase
  | .code c => .code (ccompile pl c)
  | .block is => .block (ccompileItems pl is) none

/-- A typed program with channels and locks. -/
structure CTProg where
  /-- The locals, their types and the captures of each task. Its `body` and `main` are not
  used. -/
  env : TProg
  /-- `guard x = some m`: local `x` is the value of a mutex with the lock `m`. -/
  guard : Nat → Option Nat
  locks : Nat
  chans : Nat
  /-- The room of each channel. -/
  cap : Nat → Nat
  /-- The number of senders of each channel. -/
  senders : Nat → Nat
  /-- Each channel carries at most `msgs` values in a run. -/
  msgs : Nat
  /-- The code of task `u`. -/
  body : Nat → CSrc
  main : List CPh

namespace CTProg

variable (T : CTProg)

/-- The lock of the place `p`. -/
def g (p : Nat) : Option Nat :=
  if p < T.env.width * T.env.locals then T.guard (p / T.env.width) else none

/-- The permission of the code of a block with the tasks `us` for local `x`. -/
def bPerm (us : List Nat) (x : Nat) : Perm :=
  if ∃ u ∈ us, T.env.taskPerm u x = .own then .none
  else if ∃ u ∈ us, T.env.taskPerm u x = .ref then .ref else .own

/-- An access of kind `k` and order `o` to leaf `i` through local `p`, with the permissions
`perm`, while the code holds the locks `held`. A leaf of a mutex needs the lock of the mutex and
a capture of it. Another leaf needs a permission that allows the access. -/
def AccessOk (held : List Nat) (perm : Nat → Perm) (p i : Nat) (k : Kind) (o : Order) : Prop :=
  T.env.At (T.env.target p) i fun l =>
    (T.guard (T.env.target p) = none → Allowed (perm p) l k o) ∧
      ∀ m, T.guard (T.env.target p) = some m → m ∈ held ∧ perm p ≠ .none

/-- The code `c` follows the permissions `perm` while it holds the locks `held`. -/
def SFits (held : List Nat) (perm : Nat → Perm) : Src → Prop
  | .skip => True
  | .assign r e => r < T.env.regs ∧ e.RegsBelow T.env.regs
  | .read r x i => r < T.env.regs ∧ T.AccessOk held perm x i .read .plain
  | .write x i e => e.RegsBelow T.env.regs ∧ T.AccessOk held perm x i .write .plain
  | .fetchAdd r x i e => r < T.env.regs ∧ e.RegsBelow T.env.regs ∧
      T.AccessOk held perm x i .read .sc ∧ T.AccessOk held perm x i .write .sc
  | .seq s t => SFits held perm s ∧ SFits held perm t
  | .ite c s t => c.RegsBelow T.env.regs ∧ SFits held perm s ∧ SFits held perm t
  | .loop c s => c.RegsBelow T.env.regs ∧ SFits held perm s
  | .unsafeCode _ => False

/-- The code follows the permissions `perm` while it holds the locks `held`. A `locked m` block
takes a lock that the code does not hold yet. -/
def CFits (held : List Nat) (perm : Nat → Perm) : CSrc → Prop
  | .code c => T.SFits held perm c
  | .send ch e => ch < T.chans ∧ e.RegsBelow T.env.regs
  | .recv r s ch => ch < T.chans ∧ r < T.env.regs ∧ s < T.env.regs
  | .locked m s => m < T.locks ∧ m ∉ held ∧ CFits (m :: held) perm s
  | .seq s t => CFits held perm s ∧ CFits held perm t
  | .ite c s t => c.RegsBelow T.env.regs ∧ CFits held perm s ∧ CFits held perm t
  | .loop c s => c.RegsBelow T.env.regs ∧ CFits held perm s

/-- The permission of the code of a block for an access through local `p`. -/
def blockFits (is : List CItem) (p : Nat) : Perm :=
  via (T.env.ty p) (T.bPerm (cspawnsOf is) (T.env.target p))

/-- The rules of a phase, as in `Typed.lean`. -/
def PhaseOk : CPh → Prop
  | .code c => T.CFits [] (fun p => via (T.env.ty p) .own) c
  | .block is => (∀ c, CItem.code c ∈ is → T.CFits [] (T.blockFits is) c) ∧
      ∀ u ∈ cspawnsOf is, ∀ v ∈ cspawnsOf is, u ≠ v → ∀ x, T.env.taskPerm u x = .own →
        T.env.taskPerm v x = .none

def spawns : List Nat := T.main.flatMap CPh.spawns

/-- The rules of the type checker. -/
structure WellTyped : Prop where
  width : ∀ x, (T.env.vty x).leaves.length ≤ T.env.width
  /-- The lock of each mutex exists. -/
  guard_lt : ∀ x m, T.guard x = some m → m < T.locks
  /-- A mutex holds a value, not a reference. -/
  guard_val : ∀ x m, T.guard x = some m → ∃ τ, T.env.ty x = .val τ
  /-- No local is a raw pointer. -/
  noPtrs : ∀ x k, T.env.ty x ≠ .ptr k
  /-- The rules of `Typed.lean` for a local with no lock. A mutex is `Send` and `Sync` when its
  value is `Send`. -/
  caps : ∀ u x m, (x, m) ∈ T.env.caps u →
    (T.guard x = none → (m = .ref → T.env.SyncL x) ∧ (m ≠ .ref → T.env.SendL x)) ∧
      (T.guard x ≠ none → (T.env.vty x).Send)
  bodies : ∀ u, T.CFits [] (T.env.pathPerm u) (T.body u)
  phases : ∀ p ∈ T.main, T.PhaseOk p
  nodup : T.spawns.Nodup
  spawn_lt : ∀ u ∈ T.spawns, u < T.env.count
  cap : ∀ c, c < T.chans → T.cap c ≤ T.msgs

def phases : List ChanProg.CPhase := T.main.map (CPh.compile T.env.place)

def toChanProg : ChanProg where
  main := ChanProg.cseqAll T.phases
  task u := ccompile T.env.place (T.body u)
  count := T.env.count
  places := T.env.width * T.env.locals
  chans := T.chans
  locks := T.locks
  cap := T.cap
  senders := T.senders
  msgs := T.msgs
  tmp := T.env.regs

/-! ## The proof -/

variable {T}

theorem _root_.Vx.CStmt.Accesses.mono {p q : Kind → Order → Nat → Prop} (hpq : ∀ k o x, p k o x → q k o x)
    {s : CStmt} (h : s.Accesses p) : s.Accesses q := by
  induction s with
  | base s => exact Stmt.Accesses.mono hpq h
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | locked _ _ ih | loop _ _ ih | tasksBlock _ ih => exact ih h
  | _ => trivial

theorem _root_.Vx.Expr.RegsBelow.mono {m m' : Nat} {e : Expr} (h : e.RegsBelow m) (hm : m ≤ m') :
    e.RegsBelow m' := by
  induction e with
  | const => trivial
  | reg => exact Nat.lt_of_lt_of_le h hm
  | add _ _ ih1 ih2 | sub _ _ ih1 ih2 | eq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩

theorem leaf_lt (hw : T.WellTyped) {x i : Nat} {l : Leaf} (hl : (T.env.vty x).leaves[i]? = some l) :
    i < T.env.width :=
  Nat.lt_of_lt_of_le (List.getElem?_eq_some_iff.1 hl).1 (hw.width x)

theorem place_lt (hw : T.WellTyped) {x i : Nat} {l : Leaf} (hx : x < T.env.locals)
    (hl : (T.env.vty x).leaves[i]? = some l) : T.env.width * x + i < T.env.width * T.env.locals :=
  calc T.env.width * x + i < T.env.width * x + T.env.width := Nat.add_lt_add_left (leaf_lt hw hl) _
    _ = T.env.width * (x + 1) := (Nat.mul_succ _ _).symm
    _ ≤ T.env.width * T.env.locals := Nat.mul_le_mul_left _ hx

theorem g_place (hw : T.WellTyped) {x i : Nat} {l : Leaf} (hx : x < T.env.locals)
    (hl : (T.env.vty x).leaves[i]? = some l) : T.g (T.env.width * x + i) = T.guard x := by
  have hi := leaf_lt hw hl
  have hpos : 0 < T.env.width := Nat.lt_of_le_of_lt (Nat.zero_le _) hi
  unfold g
  split
  · rw [Nat.mul_add_div hpos, Nat.div_eq_of_lt hi, Nat.add_zero]
  · exact absurd (place_lt hw hx hl) ‹_›

/-- The accesses that the permissions `tp` allow at the places of the locals with no lock. -/
def CLegal (tp : Nat → Perm) (k : Kind) (o : Order) (p : Nat) : Prop :=
  ∃ x i, p = T.env.width * x + i ∧
    T.env.At x i fun l => T.guard x = none → Allowed (tp x) l k o

theorem SFits.accesses {held : List Nat} {perm tp : Nat → Perm}
    (hle : ∀ p, (perm p).Le (tp (T.env.target p))) {c : Src} (h : T.SFits held perm c) :
    (compile T.env.place c).Accesses (T.CLegal tp) := by
  induction c with
  | skip | assign => trivial
  | read _ x i | write x i =>
    obtain ⟨-, hx, l, hl, ha, -⟩ := h
    exact ⟨_, i, rfl, hx, l, hl, fun hg => (ha hg).mono (hle x)⟩
  | fetchAdd _ x i =>
    obtain ⟨-, -, ⟨hx, l, hl, h1, -⟩, ⟨hx', l', hl', h2, -⟩⟩ := h
    exact ⟨⟨_, i, rfl, hx, l, hl, fun hg => (h1 hg).mono (hle x)⟩,
      ⟨_, i, rfl, hx', l', hl', fun hg => (h2 hg).mono (hle x)⟩⟩
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ih h.2
  | unsafeCode => exact h.elim

theorem CFits.accesses {held : List Nat} {perm tp : Nat → Perm}
    (hle : ∀ p, (perm p).Le (tp (T.env.target p))) {c : CSrc} (h : T.CFits held perm c) :
    (ccompile T.env.place c).Accesses (T.CLegal tp) := by
  induction c generalizing held with
  | code c => exact SFits.accesses hle h
  | send | recv => trivial
  | locked _ _ ih => exact ih h.2.2
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ih h.2

/-- Each access to a leaf of a mutex holds the lock of the mutex. -/
theorem SFits.guarded (hw : T.WellTyped) {held : List Nat} {perm : Nat → Perm} {c : Src}
    (h : T.SFits held perm c) :
    (compile T.env.place c).Accesses fun k _ x => k = .fence ∨ ∀ m, T.g x = some m → m ∈ held := by
  have one : ∀ {p i k o}, T.AccessOk held perm p i k o →
      ∀ m, T.g (T.env.place p i) = some m → m ∈ held := fun h m hg => by
    obtain ⟨hx, l, hl, -, hm⟩ := h
    rw [TProg.place, g_place hw hx hl] at hg
    exact (hm m hg).1
  induction c with
  | skip | assign => trivial
  | read => exact Or.inr (one h.2)
  | write => exact Or.inr (one h.2)
  | fetchAdd => exact ⟨Or.inr (one h.2.2.1), Or.inr (one h.2.2.2)⟩
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ih h.2
  | unsafeCode => exact h.elim

theorem CFits.guarded (hw : T.WellTyped) {held : List Nat} {perm : Nat → Perm} {c : CSrc}
    (h : T.CFits held perm c) : (ccompile T.env.place c).Guarded T.g held := by
  induction c generalizing held with
  | code c => exact SFits.guarded hw h
  | send | recv => trivial
  | locked _ _ ih => exact ih h.2.2
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ih h.2

/-- Code that holds the lock `m` does not take it again. -/
theorem CFits.noLock {held : List Nat} {perm : Nat → Perm} {c : CSrc} (h : T.CFits held perm c)
    {m : Nat} (hm : m ∈ held) : ¬ (ccompile T.env.place c).Locks m := by
  induction c generalizing held with
  | code | send | recv => exact id
  | locked m' _ ih =>
    rintro (rfl | h')
    · exact h.2.1 hm
    · exact ih h.2.2 (List.mem_cons_of_mem _ hm) h'
  | seq _ _ ih1 ih2 => exact fun h' => h'.elim (ih1 h.1 hm) (ih2 h.2 hm)
  | ite _ _ _ ih1 ih2 => exact fun h' => h'.elim (ih1 h.2.1 hm) (ih2 h.2.2 hm)
  | loop _ _ ih => exact ih h.2 hm

theorem SFits.below (hw : T.WellTyped) {held : List Nat} {perm : Nat → Perm} {c : Src}
    (h : T.SFits held perm c) (n : Nat) :
    (compile T.env.place c).Below (T.env.width * T.env.locals) n := by
  have one : ∀ {p i k o}, T.AccessOk held perm p i k o →
      T.env.place p i < T.env.width * T.env.locals := fun h => by
    obtain ⟨hx, l, hl, -⟩ := h
    exact place_lt hw hx hl
  induction c with
  | skip | assign => trivial
  | read | write => exact one h.2
  | fetchAdd => exact one h.2.2.1
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ih h.2
  | unsafeCode => exact h.elim

theorem CFits.stmtOk (hw : T.WellTyped) {held : List Nat} {perm : Nat → Perm} {c : CSrc}
    (h : T.CFits held perm c) : T.toChanProg.StmtOk (ccompile T.env.place c) := by
  induction c generalizing held with
  | code c => exact SFits.below hw h _
  | send => exact h.1
  | recv => exact ⟨h.1, Nat.ne_of_lt h.2.1, Nat.ne_of_lt h.2.2⟩
  | locked m _ ih => exact ⟨h.1, h.2.2.noLock List.mem_cons_self, ih h.2.2⟩
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ih h.2

theorem SFits.regs {held : List Nat} {perm : Nat → Perm} {c : Src} (h : T.SFits held perm c) :
    (compile T.env.place c).RegsBelow (T.env.regs + 3) := by
  have hm : T.env.regs ≤ T.env.regs + 3 := Nat.le_add_right _ _
  induction c with
  | skip => trivial
  | assign => exact ⟨by have := h.1; omega, h.2.mono hm⟩
  | read => exact by have := h.1; simp only [compile, Stmt.RegsBelow]; omega
  | write => exact h.1.mono hm
  | fetchAdd => exact ⟨by have := h.1; omega, h.2.1.mono hm⟩
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨h.1.mono hm, ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ⟨h.1.mono hm, ih h.2⟩
  | unsafeCode => exact h.elim

theorem CFits.regs {held : List Nat} {perm : Nat → Perm} {c : CSrc} (h : T.CFits held perm c) :
    (ccompile T.env.place c).RegsBelow (T.env.regs + 3) := by
  have hm : T.env.regs ≤ T.env.regs + 3 := Nat.le_add_right _ _
  induction c generalizing held with
  | code c => exact SFits.regs h
  | send => exact h.2.mono hm
  | recv => exact ⟨by have := h.2.1; omega, by have := h.2.2; omega⟩
  | locked _ _ ih => exact ih h.2.2
  | seq _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | ite _ _ _ ih1 ih2 => exact ⟨h.1.mono hm, ih1 h.2.1, ih2 h.2.2⟩
  | loop _ _ ih => exact ⟨h.1.mono hm, ih h.2⟩

theorem ccompile_noTasks (pl : Nat → Nat → Nat) (c : CSrc) : (ccompile pl c).NoTasks := by
  induction c with
  | code c => exact TProg.compile_noTasks pl c
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1, ih2⟩
  | locked _ _ ih | loop _ _ ih => exact ih
  | _ => trivial

theorem ccompile_noBlocks (pl : Nat → Nat → Nat) (c : CSrc) : (ccompile pl c).NoBlocks := by
  induction c with
  | code c => exact TProg.compile_noBlocks pl c
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1, ih2⟩
  | locked _ _ ih | loop _ _ ih => exact ih
  | _ => trivial

theorem _root_.Vx.Stmt.NoTasks.spawnIds {s : Stmt} (h : s.NoTasks) : s.spawnIds = [] := by
  induction s with
  | spawnAt => exact h.elim
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => simp [Stmt.spawnIds, ih1 h.1, ih2 h.2]
  | loop _ _ ih => exact ih h
  | _ => rfl

theorem ccompileItems_spawnIds (P : ChanProg) (pl : Nat → Nat → Nat) (is : List CItem) :
    (P.lowerC (ccompileItems pl is)).spawnIds = cspawnsOf is := by
  induction is with
  | nil => rfl
  | cons it is ih =>
    cases it with
    | code c =>
      show (P.lowerC (ccompile pl c)).spawnIds ++ _ = _
      rw [Stmt.NoTasks.spawnIds (ChanProg.lowerC_noTasks (ccompile_noTasks pl c)), ih]
      rfl
    | spawn u =>
      show (Stmt.spawnAt u 1 (.const 0)).spawnIds ++ _ = _
      rw [ih]
      simp [Stmt.spawnIds, cspawnsOf]

/-! ## The rules of the permissions -/

/-- A task borrows a local with no lock with `ref` only when the local is `Sync`. -/
theorem sync_of_ref (hw : T.WellTyped) {u x : Nat} (h : T.env.taskPerm u x = .ref)
    (hg : T.guard x = none) : (T.env.vty x).Sync := by
  have hval : ∀ {p y}, (T.env.ty p = .ref y ∨ T.env.ty p = .mutRef y) → T.guard p = none :=
    fun {p y} hp => by
      cases hgp : T.guard p with
      | none => rfl
      | some m =>
        obtain ⟨τ, hτ⟩ := hw.guard_val p m hgp
        rcases hp with hp | hp <;> rw [hp] at hτ <;> cases hτ
  unfold TProg.taskPerm at h
  split at h
  · cases h
  · split at h
    · rename_i _ hex
      obtain ⟨⟨p, m⟩, -, rfl, hp⟩ := hex
      simp only [TProg.pathPerm] at hp ⊢
      cases ht : T.env.ty p with
      | val τ =>
        rw [ht] at hp
        simp only [via] at hp
        have hgp : T.guard p = none := by simpa [TProg.target, ht] using hg
        have := ((hw.caps u p .ref (TProg.capPerm_ref hp)).1 hgp).1 rfl
        simp only [TProg.SyncL, ht] at this
        simpa [TProg.target, TProg.vty, ht] using this
      | ref y =>
        rw [ht] at hp
        have hgp := hval (Or.inl ht)
        have hy : (T.env.vty y).Sync := by
          cases hc : T.env.capPerm u p with
          | none => rw [hc] at hp; cases hp
          | ref =>
            have := ((hw.caps u p .ref (TProg.capPerm_ref hc)).1 hgp).1 rfl
            simpa [TProg.SyncL, ht] using this
          | own =>
            obtain ⟨m, hm, hne⟩ := TProg.capPerm_own hc
            have := ((hw.caps u p m hm).1 hgp).2 hne
            simpa [TProg.SendL, ht] using this
        simpa [TProg.target, ht] using hy
      | mutRef y =>
        rw [ht] at hp
        simp only [via] at hp
        have := ((hw.caps u p .ref (TProg.capPerm_ref hp)).1 (hval (Or.inr ht))).1 rfl
        simpa [TProg.SyncL, TProg.target, ht] using this
      | ptr k => exact absurd ht (hw.noPtrs p k)
    · cases h

/-- Two tasks of a block fit at a local with no lock. -/
theorem fit_tasks (hw : T.WellTyped) {is : List CItem} (hp : T.PhaseOk (.block is)) {u v : Nat}
    (hu : u ∈ cspawnsOf is) (hv : v ∈ cspawnsOf is) (huv : u ≠ v) (x : Nat) (hg : T.guard x = none)
    (l : Leaf) (hl : l ∈ (T.env.vty x).leaves) : Fit (T.env.taskPerm u x) (T.env.taskPerm v x) l := by
  have excl := hp.2
  cases h1 : T.env.taskPerm u x with
  | none => exact Or.inl rfl
  | own => exact Or.inr (Or.inl (excl u hu v hv huv x h1))
  | ref =>
    cases h2 : T.env.taskPerm v x with
    | none => exact Or.inr (Or.inl rfl)
    | own =>
      have := excl v hv u hu (Ne.symm huv) x h2
      rw [h1] at this
      cases this
    | ref => exact Or.inr (Or.inr ⟨rfl, rfl, (sync_of_ref hw h1 hg).leaf hl⟩)

/-- The code of a block and a task of the block fit at a local with no lock. -/
theorem fit_block (hw : T.WellTyped) {us : List Nat} {u : Nat} (hu : u ∈ us) (x : Nat)
    (hg : T.guard x = none) (l : Leaf) (hl : l ∈ (T.env.vty x).leaves) :
    Fit (T.bPerm us x) (T.env.taskPerm u x) l := by
  unfold bPerm
  split
  · exact Or.inl rfl
  · rename_i h1
    split
    · cases h : T.env.taskPerm u x with
      | none => exact Or.inr (Or.inl rfl)
      | own => exact absurd ⟨u, hu, h⟩ h1
      | ref => exact Or.inr (Or.inr ⟨rfl, rfl, (sync_of_ref hw h hg).leaf hl⟩)
    · rename_i h2
      cases h : T.env.taskPerm u x with
      | none => exact Or.inr (Or.inl rfl)
      | own => exact absurd ⟨u, hu, h⟩ h1
      | ref => exact absurd ⟨u, hu, h⟩ h2

/-- Two pieces of code whose permissions fit at each local with no lock are apart. -/
theorem capart (hw : T.WellTyped) {a b : Nat → Perm} {s t : CStmt}
    (hs : s.Accesses (T.CLegal a)) (ht : t.Accesses (T.CLegal b))
    (hfit : ∀ x, T.guard x = none → ∀ l, l ∈ (T.env.vty x).leaves → Fit (a x) (b x) l) :
    s.Apart T.g t := by
  refine CStmt.Accesses.mono ?_ hs
  rintro k o _ ⟨x, i, rfl, hx, l, hl, ha⟩ hgp
  rw [g_place hw hx hl] at hgp
  refine CStmt.Accesses.mono ?_ ht
  rintro k' o' _ ⟨y, j, rfl, hy, l', hl', hb⟩
  have hi := leaf_lt hw hl
  have hj := leaf_lt hw hl'
  by_cases he : T.env.width * x + i = T.env.width * y + j
  · have hpos : 0 < T.env.width := Nat.lt_of_le_of_lt (Nat.zero_le _) hi
    have hxy : x = y := by
      have := congrArg (· / T.env.width) he
      simp only [Nat.mul_add_div hpos, Nat.div_eq_of_lt hi, Nat.div_eq_of_lt hj] at this
      omega
    subst hxy
    have hij : i = j := by omega
    subst hij
    rw [hl] at hl'
    cases hl'
    rw [he]
    exact compat_of_fit (ha hgp) (hb hgp) (hfit x hgp l (List.mem_of_getElem? hl))
  · exact Or.inr (Or.inr (Or.inl he))

/-! ## The compiled program follows the rule -/

theorem ccompileItems_accesses {is : List CItem} {perm tp : Nat → Perm}
    (hle : ∀ p, (perm p).Le (tp (T.env.target p))) (h : ∀ c, CItem.code c ∈ is → T.CFits [] perm c) :
    (ccompileItems T.env.place is).Accesses (T.CLegal tp) := by
  induction is with
  | nil => trivial
  | cons it is ih =>
    cases it with
    | code c => exact ⟨(h c List.mem_cons_self).accesses hle,
        ih fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩
    | spawn => exact ⟨trivial, ih fun c' hc => h c' (List.mem_cons_of_mem _ hc)⟩

theorem ccompileItems_props (hw : T.WellTyped) {is : List CItem} {perm : Nat → Perm}
    (h : ∀ c, CItem.code c ∈ is → T.CFits [] perm c) (hs : ∀ u ∈ cspawnsOf is, u < T.env.count) :
    (ccompileItems T.env.place is).NoBlocks ∧
      (ccompileItems T.env.place is).RegsBelow (T.env.regs + 3) ∧
      T.toChanProg.StmtOk (ccompileItems T.env.place is) ∧
      (ccompileItems T.env.place is).Guarded T.g [] := by
  induction is with
  | nil => exact ⟨trivial, trivial, trivial, trivial⟩
  | cons it is ih =>
    cases it with
    | code c =>
      have hc := h c List.mem_cons_self
      obtain ⟨h1, h2, h3, h4⟩ := ih (fun c' hc => h c' (List.mem_cons_of_mem _ hc)) hs
      exact ⟨⟨ccompile_noBlocks _ c, h1⟩, ⟨hc.regs, h2⟩, ⟨hc.stmtOk hw, h3⟩, ⟨hc.guarded hw, h4⟩⟩
    | spawn u =>
      obtain ⟨h1, h2, h3, h4⟩ := ih (fun c' hc => h c' (List.mem_cons_of_mem _ hc))
        fun v hv => hs v (List.mem_cons_of_mem _ hv)
      exact ⟨⟨trivial, h1⟩, ⟨trivial, h2⟩, ⟨hs u List.mem_cons_self, h3⟩, ⟨trivial, h4⟩⟩

/-- The rule of a block holds for the compiled block. -/
theorem cblockOk (hw : T.WellTyped) {is : List CItem} (hp : T.PhaseOk (.block is))
    (hs : ∀ u ∈ cspawnsOf is, u < T.env.count) :
    (ChanProg.CPhase.block (ccompileItems T.env.place is) none).Ok T.toChanProg T.g (fun _ => []) := by
  have hacc := ccompileItems_accesses (tp := T.bPerm (cspawnsOf is)) (fun p => via_le _ _) hp.1
  obtain ⟨h1, h2, -, -⟩ := ccompileItems_props hw hp.1 hs
  have hn := ChanProg.lowerC_noBlocks (P := T.toChanProg) h1
  refine ⟨.of_noBlocks hn, h2, ?_, ?_⟩
  · intro u hu w hd
    rw [Stmt.allIds_of_noBlocks hn, ccompileItems_spawnIds] at hu
    rw [ChanProg.lowerPhases_nil, TaskProg.desc_nil] at hd
    subst w
    exact capart hw hacc ((hw.bodies u).accesses (TProg.pathPerm_le u))
      fun x hg l hl => fit_block hw hu x hg l hl
  · intro u hu v hv huv w w' hd hd'
    rw [Stmt.allIds_of_noBlocks hn, ccompileItems_spawnIds] at hu hv
    rw [ChanProg.lowerPhases_nil, TaskProg.desc_nil] at hd hd'
    subst w w'
    exact capart hw ((hw.bodies u).accesses (TProg.pathPerm_le u))
      ((hw.bodies v).accesses (TProg.pathPerm_le v)) fun x hg l hl => fit_tasks hw hp hu hv huv x hg l hl

theorem ccompileItems_noBlocks (pl : Nat → Nat → Nat) (is : List CItem) :
    (ccompileItems pl is).NoBlocks := by
  induction is with
  | nil => trivial
  | cons it is ih => cases it <;> exact ⟨by first | exact ccompile_noBlocks pl _ | trivial, ih⟩

theorem phase_ids (P : ChanProg) (pl : Nat → Nat → Nat) (p : CPh) :
    ((p.compile pl).lower P).ids = p.spawns := by
  cases p with
  | code => rfl
  | block is =>
    exact (Stmt.allIds_of_noBlocks (ChanProg.lowerC_noBlocks (P := P) (ccompileItems_noBlocks pl is))).trans
      (ccompileItems_spawnIds P pl is)

theorem cphasesOk (hw : T.WellTyped) :
    ∀ ps : List CPh, (∀ p ∈ ps, T.PhaseOk p) → (ps.flatMap CPh.spawns).Nodup →
      (∀ u ∈ ps.flatMap CPh.spawns, u < T.env.count) →
      T.toChanProg.CPhasesOk T.g (fun _ => []) (ps.map (CPh.compile T.env.place))
  | [], _, _, _ => trivial
  | p :: ps, hp, hn, hs => by
    rw [List.flatMap_cons, List.nodup_append] at hn
    refine ⟨?_, ?_, cphasesOk hw ps (fun q hq => hp q (List.mem_cons_of_mem _ hq)) hn.2.1
      fun u hu => hs u (by rw [List.flatMap_cons]; exact List.mem_append_right _ hu)⟩
    · cases p with
      | code c => exact ccompile_noTasks _ c
      | block is =>
        exact cblockOk hw (hp _ List.mem_cons_self) fun u hu =>
          hs u (by rw [List.flatMap_cons]; exact List.mem_append_left _ hu)
    · intro u hu q hq huq
      rw [phase_ids] at hu
      obtain ⟨q', hq', rfl⟩ := List.mem_map.1 hq
      rw [phase_ids] at huq
      exact hn.2.2 u hu u (List.mem_flatMap.2 ⟨q', hq', huq⟩) rfl

theorem cseqAll_ok (hw : T.WellTyped) :
    ∀ ps : List CPh, (∀ p ∈ ps, T.PhaseOk p) → (∀ u ∈ ps.flatMap CPh.spawns, u < T.env.count) →
      T.toChanProg.StmtOk (ChanProg.cseqAll (ps.map (CPh.compile T.env.place)))
  | [], _, _ => trivial
  | p :: ps, hp, hs => by
    refine ⟨?_, cseqAll_ok hw ps (fun q hq => hp q (List.mem_cons_of_mem _ hq))
      fun u hu => hs u (by rw [List.flatMap_cons]; exact List.mem_append_right _ hu)⟩
    cases p with
    | code c => exact (hp _ List.mem_cons_self).stmtOk hw
    | block is =>
      exact (ccompileItems_props hw (hp _ List.mem_cons_self).1 fun u hu =>
        hs u (by rw [List.flatMap_cons]; exact List.mem_append_left _ hu)).2.2.1

/-- The code of a well-typed program follows the rule of `ChanFootprint.lean`. -/
theorem WellTyped.cfootprints (hw : T.WellTyped) :
    T.toChanProg.CFootprints T.g (fun _ => []) T.phases where
  ok := ⟨cseqAll_ok hw _ hw.phases hw.spawn_lt, fun u => (hw.bodies u).stmtOk hw, hw.cap⟩
  main := rfl
  phases := cphasesOk hw _ hw.phases hw.nodup hw.spawn_lt
  tasks := fun _ => Or.inr ⟨ccompile_noTasks _ _, rfl⟩
  task_phases := fun _ => trivial
  one_parent := ChanProg.one_parent_nil _
  order := fun _ _ ⟨_, hp, _⟩ => by simp at hp
  guard x m h := by
    unfold g at h
    split at h
    · exact ⟨by assumption, hw.guard_lt _ _ h⟩
    · cases h
  guarded_blocks B rep hB := by
    obtain ⟨p, hp, he⟩ := List.mem_map.1 hB
    cases p with
    | code => cases he
    | block is =>
      simp only [CPh.compile, ChanProg.CPhase.block.injEq] at he
      obtain ⟨rfl, -⟩ := he
      have hp' := hw.phases _ hp
      exact (ccompileItems_props hw hp'.1 fun u hu =>
        hw.spawn_lt u (List.mem_flatMap.2 ⟨_, hp, hu⟩)).2.2.2
  guarded_task t := (hw.bodies t).guarded hw

/-- A well-typed program with channels and locks has no data race, in each execution whose `co`
orders the writes to each place. -/
theorem WellTyped.raceFree (hw : T.WellTyped) {X : Execution}
    (hX : T.toChanProg.toTaskProg.toProg.Execs X) (wf : X.WellFormed) (hc : X.Consistent)
    (ho : X.CoOrder) (hfin : X.Finite) : X.RaceFree :=
  ChanProg.raceFree hw.cfootprints hX wf hc ho hfin

end CTProg

end Typed

end Vx
