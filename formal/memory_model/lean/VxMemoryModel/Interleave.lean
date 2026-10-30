import VxMemoryModel.Blocks
import VxMemoryModel.Machine

/-!
# A CPU node: one memory, one thread at each step

A configuration has one memory, and the code and the registers of each thread. At each step, one
thread takes one step of its code (`Thread.lean`). A read reads the value in the memory, and a
write changes it. A read-modify-write is one step: no thread moves between its read and its
write. A place that no write wrote has no value, so a read of it has no step. The model has no
initial writes either: a program writes each place before it reads it.

`Cpu.Reach.execs`: the accesses of each finite run, in order, are an execution of the program
(`Prog.Execs`) in which the accesses happen one at a time (`SeqConsistent`). So the CPU node gives
no behavior that the model of the language forbids.
-/

namespace Vx

/-- The next step of `s` is the write of a read-modify-write. -/
def Stmt.Pending : Stmt → Prop
  | rmwStore .. => True
  | seq s _ => s.Pending
  | _ => False

/-- Only the write of a read-modify-write has `rmw` set. -/
theorem Step.pending_of_rmw {s regs a s' regs'} (h : Step s regs a s' regs') :
    ∀ l, a = some l → l.rmw = true → s.Pending := by
  induction h with
  | rmwStore => intros; trivial
  | seq _ ih => exact ih
  | _ => intro l hl hr; cases hl <;> cases hr

abbrev Mem := Nat → Option Int

/-- The memory after the access `l`. -/
def Label.apply (l : Label) (mem : Mem) : Mem :=
  if l.kind = .write then setAt mem l.loc (some l.val) else mem

/-- A read gets the value in the memory. -/
def Label.Fits (l : Label) (mem : Mem) : Prop := l.kind = .read → mem l.loc = some l.val

structure Cpu where
  mem : Mem
  code : Nat → Stmt
  regs : Nat → Nat → Int

namespace Cpu

/-- The start of a run of `P`: no place has a value, and each register holds 0. -/
def init (P : Prog) : Cpu := ⟨fun _ => none, P.code, fun _ _ => 0⟩

/-- Thread `t` moves to the code `s` and the registers `regs`, and the memory becomes `mem`. -/
def move (c : Cpu) (t : Nat) (s : Stmt) (regs : Nat → Int) (mem : Mem) : Cpu :=
  ⟨mem, setAt c.code t s, setAt c.regs t regs⟩

/-- A step of thread `t`, with its accesses. -/
inductive Step : Cpu → Nat → List Label → Cpu → Prop
  | silent {c t s' regs'} : Vx.Step (c.code t) (c.regs t) none s' regs' →
    Step c t [] (c.move t s' regs' c.mem)
  | one {c t l s' regs'} : Vx.Step (c.code t) (c.regs t) (some l) s' regs' →
    ¬ (c.code t).Pending → ¬ s'.Pending → l.Fits c.mem →
    Step c t [l] (c.move t s' regs' (l.apply c.mem))
  /-- A read-modify-write: its read, and then its write in the same step. -/
  | two {c t l₁ s₁ regs₁ l₂ s₂ regs₂} : Vx.Step (c.code t) (c.regs t) (some l₁) s₁ regs₁ →
    ¬ (c.code t).Pending → s₁.Pending → l₁.Fits c.mem →
    Vx.Step s₁ regs₁ (some l₂) s₂ regs₂ → l₂.Fits (l₁.apply c.mem) →
    Step c t [l₁, l₂] (c.move t s₂ regs₂ (l₂.apply (l₁.apply c.mem)))

/-- `Reach c evs c'`: a finite run from `c` to `c'` makes the accesses `evs`. Each access names
its thread. -/
inductive Reach : Cpu → List (Nat × Label) → Cpu → Prop
  | refl {c} : Reach c [] c
  | tail {c evs c' t ls c''} : Reach c evs c' → Step c' t ls c'' →
    Reach c (evs ++ ls.map (t, ·)) c''

/-- A CPU node of a fleet. An action is a step of the thread with that number. The node sends
no messages, and a message does not change it. -/
def node (M : Type) : Node Cpu M where
  step c t c' out := out = [] ∧ ∃ ls, Step c t ls c'
  recv c _ := c

end Cpu


/-! ## Lemmas on lists -/

section Lists

variable {α : Type _}

/-- For each count below the count of `l`, an entry of `l` has that count before it. -/
theorem exists_at_count (p : α → Bool) : ∀ (l : List α) (i : Nat), i < l.countP p →
    ∃ j, ∃ hj : j < l.length, p l[j] = true ∧ (l.take j).countP p = i
  | [], i, h => by simp at h
  | a :: l, i, h => by
    by_cases ha : p a = true
    · rw [List.countP_cons_of_pos ha] at h
      cases i with
      | zero => exact ⟨0, by simp, by simpa using ha, by simp⟩
      | succ i =>
        obtain ⟨j, hj, hp, hc⟩ := exists_at_count p l i (by omega)
        exact ⟨j + 1, by simp; omega, by simpa using hp, by simp [List.countP_cons_of_pos ha, hc]⟩
    · rw [List.countP_cons_of_neg ha] at h
      obtain ⟨j, hj, hp, hc⟩ := exists_at_count p l i h
      exact ⟨j + 1, by simp; omega, by simpa using hp, by simp [List.countP_cons_of_neg ha, hc]⟩

theorem countP_take_le (p : α → Bool) (l : List α) {i j : Nat} (h : i ≤ j) :
    (l.take i).countP p ≤ (l.take j).countP p := by
  conv => rhs; rw [← List.take_append_drop i (l.take j)]
  rw [List.countP_append, List.take_take, Nat.min_eq_left h]
  omega

theorem countP_take_succ (p : α → Bool) (l : List α) {i : Nat} (h : i < l.length) :
    (l.take (i + 1)).countP p = (l.take i).countP p + if p l[i] then 1 else 0 := by
  rw [List.take_add_one, List.countP_append, List.getElem?_eq_getElem h]
  by_cases hp : p l[i] = true <;> simp [hp]

theorem countP_take_lt (p : α → Bool) (l : List α) {i j : Nat} (hi : i < l.length) (hij : i < j)
    (hp : p l[i] = true) : (l.take i).countP p < (l.take j).countP p := by
  have := countP_take_le p l (Nat.succ_le_of_lt hij)
  rw [countP_take_succ p l hi] at this
  simp [hp] at this
  omega

theorem getElem?_filter_countP (p : α → Bool) (l : List α) {i : Nat} (h : i < l.length)
    (hp : p l[i] = true) : (l.filter p)[(l.take i).countP p]? = some l[i] := by
  have e : l.filter p = (l.take i).filter p ++ l[i] :: (l.drop (i + 1)).filter p := by
    conv => lhs; rw [← List.take_append_drop i l, List.drop_eq_getElem_cons h]
    rw [List.filter_append, List.filter_cons_of_pos hp]
  rw [e, List.countP_eq_length_filter, List.getElem?_append_right (Nat.le_refl _)]
  simp

theorem map_range_getD {l : List α} {n : Nat} (d : α) (h : n ≤ l.length) :
    (List.range n).map (fun i => l.getD i d) = l.take n := by
  apply List.ext_getElem
  · simp; omega
  · intro i h1 h2
    simp at h1
    simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem (Nat.lt_of_lt_of_le h1 h)]

end Lists

/-! ## The accesses of a run -/

/-- The accesses of thread `t`, in order. -/
def accessesOf (evs : List (Nat × Label)) (t : Nat) : List Label := (evs.filter (·.1 == t)).map (·.2)

/-- The memory after the accesses `evs`. -/
def memOf (evs : List (Nat × Label)) : Mem := evs.foldl (fun m q => q.2.apply m) fun _ => none

/-- The position of the access `i` in its thread. -/
def idxOf (evs : List (Nat × Label)) (i : Nat) (t : Nat) : Nat := (evs.take i).countP (·.1 == t)

/-- A run of code can go on with one more step. -/
theorem Steps.tail {s regs L s₁ regs₁ a s₂ regs₂} (h : Steps s regs L s₁ regs₁)
    (hs : Step s₁ regs₁ a s₂ regs₂) : Steps s regs (L ++ a.toList) s₂ regs₂ := by
  induction h with
  | refl =>
    cases a with
    | none => exact .silent hs .refl
    | some l => exact .visible hs .refl
  | silent h₀ _ ih => exact .silent h₀ (ih hs)
  | visible h₀ _ ih => exact .visible h₀ (ih hs)

/-- `q` writes the place `x`. -/
def IsWriteTo (q : Nat × Label) (x : Nat) : Prop := q.2.kind = .write ∧ q.2.loc = x

theorem memOf_concat (evs : List (Nat × Label)) (q : Nat × Label) :
    memOf (evs ++ [q]) = q.2.apply (memOf evs) := by
  simp [memOf, List.foldl_append]

theorem foldl_spec (x : Nat) : ∀ (evs : List (Nat × Label)) (m : Mem),
    (evs.foldl (fun m q => q.2.apply m) m x = m x ∧ ∀ j (hj : j < evs.length), ¬ IsWriteTo evs[j] x) ∨
    ∃ j, ∃ hj : j < evs.length, IsWriteTo evs[j] x ∧
      evs.foldl (fun m q => q.2.apply m) m x = some evs[j].2.val ∧
      ∀ k (hk : k < evs.length), j < k → ¬ IsWriteTo evs[k] x
  | [], m => Or.inl ⟨rfl, fun _ h => absurd h (Nat.not_lt_zero _)⟩
  | q :: evs, m => by
    simp only [List.foldl_cons]
    rcases foldl_spec x evs (q.2.apply m) with ⟨h1, h2⟩ | ⟨j, hj, hw, hv, hl⟩
    · rw [h1]
      by_cases hq : IsWriteTo q x
      · refine Or.inr ⟨0, by simp, hq, by simp [Label.apply, hq.1, ← hq.2], fun k hk hk0 => ?_⟩
        obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
        exact h2 k (by simpa using hk)
      · refine Or.inl ⟨?_, fun j hj => ?_⟩
        · unfold Label.apply
          split
          · exact setAt_other _ _ fun h => hq ⟨‹_›, h.symm⟩
          · rfl
        · cases j with
          | zero => exact hq
          | succ j => exact h2 j (by simpa using hj)
    · refine Or.inr ⟨j + 1, by simp; omega, hw, hv, fun k hk hjk => ?_⟩
      obtain ⟨k, rfl⟩ : ∃ k', k = k' + 1 := ⟨k - 1, by omega⟩
      exact hl k (by simpa using hk) (by omega)

/-- A place has no value when no access writes it. Else it has the value of the last write. -/
theorem memOf_spec (x : Nat) (evs : List (Nat × Label)) :
    (memOf evs x = none ∧ ∀ j (hj : j < evs.length), ¬ IsWriteTo evs[j] x) ∨
    ∃ j, ∃ hj : j < evs.length, IsWriteTo evs[j] x ∧ memOf evs x = some evs[j].2.val ∧
      ∀ k (hk : k < evs.length), j < k → ¬ IsWriteTo evs[k] x :=
  foldl_spec x evs _

theorem thr_append (evs : List (Nat × Label)) (t t' : Nat) (ls : List Label) :
    accessesOf (evs ++ ls.map (t, ·)) t' = accessesOf evs t' ++ if t' = t then ls else [] := by
  by_cases h : t' = t
  · subst h
    simp [accessesOf, List.filter_append, List.filter_map, Function.comp_def]
  · have : ∀ l ∈ ls, ¬ (t = t') := fun _ _ e => h e.symm
    simp [accessesOf, List.filter_append, List.filter_map, Function.comp_def, h, Ne.symm h]

/-! ## The invariant of a run -/

/-- What a run from `Cpu.init P` keeps true. -/
structure Inv (P : Prog) (evs : List (Nat × Label)) (c : Cpu) : Prop where
  steps : ∀ t, Steps (P.code t) (fun _ => 0) (accessesOf evs t) (c.code t) (c.regs t)
  mem : c.mem = memOf evs
  /-- Each read gets the value that the memory had before it. -/
  fits : ∀ i (h : i < evs.length), evs[i].2.Fits (memOf (evs.take i))
  /-- The write of a read-modify-write comes right after an access of its own thread. -/
  rmw : ∀ i (h : i < evs.length), evs[i].2.rmw = true → ∃ _ : 0 < i, evs[i - 1].1 = evs[i].1

theorem fits_append {evs more : List (Nat × Label)}
    (h1 : ∀ i (h : i < evs.length), evs[i].2.Fits (memOf (evs.take i)))
    (h2 : ∀ k (h : k < more.length), more[k].2.Fits (memOf (evs ++ more.take k))) :
    ∀ i (h : i < (evs ++ more).length), (evs ++ more)[i].2.Fits (memOf ((evs ++ more).take i)) := by
  intro i h
  rcases Nat.lt_or_ge i evs.length with hi | hi
  · rw [List.getElem_append_left hi, List.take_append_of_le_length (Nat.le_of_lt hi)]
    exact h1 i hi
  · rw [List.getElem_append_right hi, List.take_append, List.take_of_length_le hi]
    exact h2 _ (by simp at h; omega)

theorem rmw_append {evs more : List (Nat × Label)}
    (h1 : ∀ i (h : i < evs.length), evs[i].2.rmw = true → ∃ _ : 0 < i, evs[i - 1].1 = evs[i].1)
    (h2 : ∀ k (h : k < more.length), more[k].2.rmw = true → ∃ _ : 0 < k, more[k - 1].1 = more[k].1) :
    ∀ i (h : i < (evs ++ more).length), (evs ++ more)[i].2.rmw = true →
      ∃ _ : 0 < i, (evs ++ more)[i - 1].1 = (evs ++ more)[i].1 := by
  intro i h hr
  rcases Nat.lt_or_ge i evs.length with hi | hi
  · rw [List.getElem_append_left hi] at hr ⊢
    obtain ⟨h0, he⟩ := h1 i hi hr
    exact ⟨h0, by rw [List.getElem_append_left (by omega)]; exact he⟩
  · rw [List.getElem_append_right hi] at hr ⊢
    obtain ⟨h0, he⟩ := h2 _ (by simp at h; omega) hr
    refine ⟨by omega, ?_⟩
    rw [List.getElem_append_right (by omega)]
    simpa [show i - 1 - evs.length = i - evs.length - 1 by omega] using he

theorem Inv.init (P : Prog) : Inv P [] (Cpu.init P) where
  steps _ := .refl
  mem := rfl
  fits _ h := absurd h (Nat.not_lt_zero _)
  rmw _ h := absurd h (Nat.not_lt_zero _)

theorem Inv.step {P : Prog} {evs c t ls c'} (hI : Inv P evs c) (hs : Cpu.Step c t ls c') :
    Inv P (evs ++ ls.map (t, ·)) c' := by
  have hcode : ∀ s regs mem t', (c.move t s regs mem).code t' = if t' = t then s else c.code t' :=
    fun _ _ _ _ => rfl
  have hregs : ∀ s regs mem t', (c.move t s regs mem).regs t' = if t' = t then regs else c.regs t' :=
    fun _ _ _ _ => rfl
  cases hs with
  | silent h =>
    refine ⟨fun t' => ?_, by simpa [Cpu.move] using hI.mem, by simpa using hI.fits,
      by simpa using hI.rmw⟩
    rw [thr_append, hcode, hregs]
    by_cases ht : t' = t
    · subst ht
      simpa using (hI.steps t').tail h
    · simpa [ht] using hI.steps t'
  | @one l s' regs' h hp _ hf =>
    have hr : l.rmw = false := by
      cases e : l.rmw
      · rfl
      · exact absurd (h.pending_of_rmw l rfl e) hp
    refine ⟨fun t' => ?_, ?_, ?_, ?_⟩
    · rw [thr_append, hcode, hregs]
      by_cases ht : t' = t
      · subst ht
        simpa using (hI.steps t').tail h
      · simpa [ht] using hI.steps t'
    · simp [Cpu.move, memOf_concat, hI.mem]
    · simp only [List.map_cons, List.map_nil]
      refine fits_append hI.fits fun k hk => ?_
      simp at hk
      subst hk
      simpa [← hI.mem] using hf
    · simp only [List.map_cons, List.map_nil]
      refine rmw_append hI.rmw fun k hk hk' => ?_
      simp at hk
      subst hk
      simp [hr] at hk'
  | @two l₁ s₁ regs₁ l₂ s₂ regs₂ h₁ hp hp₁ hf₁ h₂ hf₂ =>
    have hr : l₁.rmw = false := by
      cases e : l₁.rmw
      · rfl
      · exact absurd (h₁.pending_of_rmw l₁ rfl e) hp
    refine ⟨fun t' => ?_, ?_, ?_, ?_⟩
    · rw [thr_append, hcode, hregs]
      by_cases ht : t' = t
      · subst ht
        simpa using ((hI.steps t').tail h₁).tail h₂
      · simpa [ht] using hI.steps t'
    · rw [show evs ++ [l₁, l₂].map (t, ·) = evs ++ [(t, l₁)] ++ [(t, l₂)] by simp, memOf_concat,
        memOf_concat, ← hI.mem]
      rfl
    · simp only [List.map_cons, List.map_nil]
      refine fits_append hI.fits fun k hk => ?_
      simp at hk
      rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl
      · simpa [← hI.mem] using hf₁
      · simpa [memOf_concat, ← hI.mem] using hf₂
    · simp only [List.map_cons, List.map_nil]
      refine rmw_append hI.rmw fun k hk hk' => ?_
      simp at hk
      rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl
      · simp [hr] at hk'
      · exact ⟨by omega, rfl⟩

theorem Cpu.Reach.inv {P : Prog} {evs c} (h : Cpu.Reach (Cpu.init P) evs c) : Inv P evs c := by
  induction h with
  | refl => exact Inv.init P
  | tail _ hs ih => exact ih.step hs

/-! ## The execution of a run -/

/-- The execution of the accesses `evs`, in this order. A read reads the last write to its place
before it. The writes to a place are in the order of `evs`. -/
def ofEvents (evs : List (Nat × Label)) : Execution where
  E := Fin evs.length
  kind e := evs[e.1].2.kind
  order e := evs[e.1].2.order
  loc e := evs[e.1].2.loc
  thread e := evs[e.1].1
  po a b := evs[a.1].1 = evs[b.1].1 ∧ a.val < b.val
  rf w r := IsWriteTo evs[w.1] evs[r.1].2.loc ∧ evs[r.1].2.kind = .read ∧ w.val < r.val ∧
    ∀ k : Fin evs.length, w.val < k.val → k.val < r.val → ¬ IsWriteTo evs[k.1] evs[r.1].2.loc
  co a b := evs[a.1].2.kind = .write ∧ evs[b.1].2.kind = .write ∧ evs[a.1].2.loc = evs[b.1].2.loc ∧
    a.val < b.val
  rmw a b := evs[a.1].1 = evs[b.1].1 ∧ idxOf evs b.1 evs[b.1].1 = idxOf evs a.1 evs[a.1].1 + 1 ∧
    evs[b.1].2.rmw = true

namespace ofEvents

variable {evs : List (Nat × Label)}

theorem fr_lt {r w : Fin evs.length} (h : (ofEvents evs).fr r w) : r.val < w.val := by
  obtain ⟨w₀, ⟨hw₀, hr, h₀r, hlast⟩, hwk₀, hwk, hloc, h₀w⟩ := h
  rcases Nat.lt_or_ge w.val r.val with hwr | hwr
  · exact absurd ⟨hwk, hloc ▸ hw₀.2⟩ (hlast w h₀w hwr)
  · rcases Nat.lt_or_eq_of_le hwr with h | h
    · exact h
    · have : r = w := Fin.ext h
      subst this
      rw [hwk] at hr
      cases hr

theorem ord_lt {a b : Fin evs.length} (h : (ofEvents evs).ord a b) : a.val < b.val := by
  rcases h with h | h | h | h
  · exact h.2
  · exact h.2.2.1
  · exact h.2.2.2
  · exact fr_lt h

theorem transGen_lt {a b : Fin evs.length} (h : Relation.TransGen (ofEvents evs).ord a b) :
    a.val < b.val := by
  induction h with
  | single h => exact ord_lt h
  | tail _ h ih => exact Nat.lt_trans ih (ord_lt h)

theorem idx_lt {a b : Fin evs.length} (ht : evs[a.1].1 = evs[b.1].1) (h : a.val < b.val) :
    idxOf evs a.1 evs[a.1].1 < idxOf evs b.1 evs[b.1].1 := by
  unfold idxOf
  rw [ht]
  exact countP_take_lt _ evs a.isLt h (by simp [ht])

theorem idx_inj {a b : Fin evs.length} (ht : evs[a.1].1 = evs[b.1].1)
    (h : idxOf evs a.1 evs[a.1].1 = idxOf evs b.1 evs[b.1].1) : a = b := by
  rcases Nat.lt_trichotomy a.val b.val with hab | hab | hab
  · exact absurd h (Nat.ne_of_lt (idx_lt ht hab))
  · exact Fin.ext hab
  · exact absurd h (Nat.ne_of_gt (idx_lt ht.symm hab))

end ofEvents

/-- The label that `Prog.Run.lab` gives after the end of a thread. It is never read. -/
def Label.none : Label := ⟨.fence, .plain, 0, 0, false⟩

open ofEvents in
theorem Inv.run {P : Prog} {evs c} (hI : Inv P evs c) :
    P.Execs (ofEvents evs) ∧ (ofEvents evs).SeqConsistent := by
  -- The last write to the place of the read `r` before it, if the read is in `evs`.
  have last : ∀ r : Fin evs.length, evs[r.1].2.kind = .read →
      ∃ w : Fin evs.length, (ofEvents evs).rf w r ∧ evs[w.1].2.val = evs[r.1].2.val := by
    intro r hr
    have hf : memOf (evs.take r) evs[r.1].2.loc = some evs[r.1].2.val := hI.fits r r.isLt hr
    rcases memOf_spec evs[r.1].2.loc (evs.take r) with ⟨h, _⟩ | ⟨j, hj, hw, hv, hl⟩
    · rw [h] at hf
      cases hf
    · have := r.isLt
      have hjr : j < r.val ∧ j < evs.length := by simp at hj; omega
      refine ⟨⟨j, hjr.2⟩, ⟨?_, hr, hjr.1, fun k hjk hkr => ?_⟩, ?_⟩
      · simpa [List.getElem_take] using hw
      · have := hl k (by simp; omega) hjk
        simpa [List.getElem_take] using this
      · rw [hv] at hf
        simpa [List.getElem_take] using hf
  -- A read reads from one write.
  have rf_unique : ∀ w w' r, (ofEvents evs).rf w r → (ofEvents evs).rf w' r → w = w' := by
    intro w w' r ⟨hw, _, hwr, hl⟩ ⟨hw', _, hwr', hl'⟩
    rcases Nat.lt_trichotomy w.val w'.val with h | h | h
    · exact absurd hw' (hl w' h hwr')
    · exact Fin.ext h
    · exact absurd hw (hl' w h hwr)
  have po_iff : ∀ a b : Fin evs.length, (evs[a.1].1 = evs[b.1].1 ∧ a.val < b.val) ↔
      (evs[a.1].1 = evs[b.1].1 ∧ idxOf evs a.1 evs[a.1].1 < idxOf evs b.1 evs[b.1].1) := by
    refine fun a b => ⟨fun ⟨ht, h⟩ => ⟨ht, idx_lt ht h⟩, fun ⟨ht, h⟩ => ⟨ht, ?_⟩⟩
    rcases Nat.lt_or_ge a.val b.val with h' | h'
    · exact h'
    · rcases Nat.lt_or_eq_of_le h' with h' | h'
      · exact absurd h (Nat.lt_asymm (idx_lt ht.symm h'))
      · rw [Fin.ext h'] at h
        exact absurd h (Nat.lt_irrefl _)
  have idx_down : ∀ (e : Fin evs.length) i, i < idxOf evs e.1 evs[e.1].1 →
      ∃ e' : Fin evs.length, evs[e'.1].1 = evs[e.1].1 ∧ idxOf evs e'.1 evs[e'.1].1 = i := by
    intro e i hi
    obtain ⟨j, hj, hp, hc⟩ :=
      exists_at_count (fun q : Nat × Label => q.1 == evs[e.1].1) (evs.take e) i hi
    have := e.isLt
    have hje : j < e.val ∧ j < evs.length := by simp at hj; omega
    have ht : evs[j].1 = evs[e.1].1 := by simpa [List.getElem_take] using hp
    refine ⟨⟨j, hje.2⟩, ht, ?_⟩
    show idxOf evs j evs[j].1 = i
    rw [ht, ← hc, List.take_take, Nat.min_eq_left (Nat.le_of_lt hje.1)]
    rfl
  have lab_eq : ∀ e : Fin evs.length,
      (accessesOf evs evs[e.1].1).getD (idxOf evs e.1 evs[e.1].1) Label.none = evs[e.1].2 := by
    intro e
    unfold accessesOf idxOf
    rw [List.getD_eq_getElem?_getD, List.getElem?_map,
      getElem?_filter_countP _ evs e.isLt (by simp)]
    rfl
  have trace : ∀ e : Fin evs.length, Trace (P.code evs[e.1].1) (idxOf evs e.1 evs[e.1].1 + 1)
      fun i => (accessesOf evs evs[e.1].1).getD i Label.none := by
    intro e
    have hlen : idxOf evs e.1 evs[e.1].1 + 1 ≤ (accessesOf evs evs[e.1].1).length := by
      have h1 := countP_take_succ (·.1 == evs[e.1].1) evs e.isLt
      have h2 := (List.take_sublist (e.1 + 1) evs).countP_le
        (p := fun q : Nat × Label => q.1 == evs[e.1].1)
      unfold idxOf accessesOf
      rw [List.length_map, ← List.countP_eq_length_filter]
      simp at h1
      omega
    unfold Trace
    rw [map_range_getD _ hlen]
    have h := (hI.steps evs[e.1].1).runs
    exact h.prefix _ _ (List.take_append_drop _ _).symm
  have rf_val : ∀ w r : Fin evs.length, (ofEvents evs).rf w r → evs[w.1].2.val = evs[r.1].2.val := by
    intro w r h
    obtain ⟨w', hw', hv⟩ := last r h.2.1
    rw [rf_unique w w' r h hw']
    exact hv
  refine ⟨⟨{
    val := fun e => evs[e.1].2.val
    isRmw := fun e => evs[e.1].2.rmw
    idx := fun e => idxOf evs e.1 evs[e.1].1
    lab := fun t i => (accessesOf evs t).getD i Label.none
    po_iff := po_iff
    idx_inj := fun a b ht h => idx_inj ht h
    idx_down := idx_down
    lab_eq := lab_eq
    trace := trace
    rf_val := rf_val
    rf_write_read := fun w r h => ⟨h.1.1, h.2.1, h.1.2⟩
    rf_source := fun r hr => let ⟨w, hw, _⟩ := last r hr; ⟨w, hw⟩
    rmw_iff := fun a b => Iff.rfl }⟩, ⟨fun a h => Nat.lt_irrefl _ (transGen_lt h), ?_⟩⟩
  -- Atomicity: no write comes between the read and the write of a read-modify-write.
  intro (a : Fin evs.length) (b : Fin evs.length) (c' : Fin evs.length) ⟨ht, hidx, hrmw⟩ hfr
    ⟨_, _, _, hcb⟩
  obtain ⟨h0, hprev⟩ := hI.rmw b.1 b.isLt hrmw
  let a' : Fin evs.length := ⟨b.val - 1, by omega⟩
  have hb : idxOf evs b.1 evs[b.1].1 = idxOf evs a'.1 evs[a'.1].1 + 1 := by
    unfold idxOf
    have := countP_take_succ (·.1 == evs[b.1].1) evs a'.isLt
    rw [show a'.val + 1 = b.val by simp [a']; omega] at this
    rw [this, show evs[a'.1].1 = evs[b.1].1 from hprev]
    simp
  have ha : a.val = b.val - 1 := by
    rw [idx_inj (ht.trans hprev.symm) (by omega : _ = idxOf evs a'.1 _)]
  have := fr_lt hfr
  omega

theorem ofEvents.finite (evs : List (Nat × Label)) : (ofEvents evs).Finite :=
  ⟨List.finRange evs.length, List.mem_finRange⟩

theorem ofEvents.coOrder (evs : List (Nat × Label)) : (ofEvents evs).CoOrder where
  trans _ _ _ h1 h2 := ⟨h1.1, h2.2.1, h1.2.2.1.trans h2.2.2.1, Nat.lt_trans h1.2.2.2 h2.2.2.2⟩
  total a b ha hb hl hne := by
    rcases Nat.lt_trichotomy a.val b.val with h | h | h
    · exact Or.inl ⟨ha, hb, hl, h⟩
    · exact absurd (Fin.ext h) hne
    · exact Or.inr ⟨hb, ha, hl.symm, h⟩

/-- Each finite run of a CPU node from the start makes accesses that are an execution of the
program. In that execution the accesses happen one at a time, and `co` orders the writes to
each place. -/
theorem Cpu.Reach.execs {P : Prog} {evs c} (h : Cpu.Reach (Cpu.init P) evs c) :
    P.Execs (ofEvents evs) ∧ (ofEvents evs).SeqConsistent ∧ (ofEvents evs).Finite ∧
      (ofEvents evs).CoOrder :=
  ⟨h.inv.run.1, h.inv.run.2, ofEvents.finite evs, ofEvents.coOrder evs⟩

/-- In a run of a fleet, the states of a CPU node up to each point are a finite run of it. -/
theorem Fleet.IsRun.cpu_reach {M : Type} {F : Fleet Cpu M} {c₀ r} (hr : F.IsRun c₀ r) {n : Nat}
    (hn : F.node n = Cpu.node M) : ∀ i, ∃ evs, Cpu.Reach (c₀.state n) evs ((r i).state n) := by
  intro i
  induction i with
  | zero => exact ⟨[], by rw [hr.start]; exact .refl⟩
  | succ i ih =>
    obtain ⟨evs, h⟩ := ih
    rcases hr.next i with ⟨act, hs⟩ | he
    · generalize r (i + 1) = c' at hs ⊢
      cases hs with
      | @move _ n' a s' out hstep =>
        by_cases h' : n = n'
        · subst h'
          rw [hn] at hstep
          obtain ⟨_, ls, hcs⟩ := hstep
          exact ⟨_, h.tail (by simpa using hcs)⟩
        · exact ⟨evs, by simpa [setAt_other _ _ h'] using h⟩
      | deliver j hj =>
        refine ⟨evs, ?_⟩
        by_cases h' : n = (r i).flight[j].dst
        · rw [← h'] at *
          simpa [hn, Cpu.node] using h
        · simpa [setAt_other _ _ h'] using h
    · exact ⟨evs, by rw [he]; exact h⟩

end Vx
