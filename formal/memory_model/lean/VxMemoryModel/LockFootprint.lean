import VxMemoryModel.ChanFootprint
import VxMemoryModel.Lock
import VxMemoryModel.ScIsRc11

/-!
# Places that a lock guards

The rule of `ChanFootprint.lean` lets `g` give a place of the code a lock `m`. The code accesses
such a place only in a `locked` block of `m` (`CStmt.Guarded`). Two tasks of one block can then
share the place.

`HTrans` of `Lock.lean` takes a condition for the accesses while a thread does not hold the lock.
Here the condition is `Unguarded`: no access to a place that `g` gives `m`. So `guard_hist` shows
that before each access to such a place, the thread took a ticket of `m`, read `serving` as the
ticket, and did not let the lock go after it. Then `lock_excl` orders two such accesses in two
threads (`guard_hb`). `raceFree` puts this together with `raceFree_of`.
-/

namespace Vx

theorem Stmt.Accesses.and {p q : Kind → Order → Nat → Prop} {s : Stmt} (hp : s.Accesses p)
    (hq : s.Accesses q) : s.Accesses fun k o x => p k o x ∧ q k o x := by
  induction s with
  | load | store | rmwStore | fence | wait => exact ⟨hp, hq⟩
  | fetchAdd | cas => exact ⟨⟨hp.1, hq.1⟩, hp.2, hq.2⟩
  | loadAt | storeAt | waitAt => exact fun j hj => ⟨hp j hj, hq j hj⟩
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 hp.1 hq.1, ih2 hp.2 hq.2⟩
  | loop _ _ ih => exact ih hp hq
  | _ => trivial

/-- `lower` keeps the accesses of code with no `tasks` block, and adds accesses to the flags. -/
theorem TaskProg.lower_keeps {P : TaskProg} {q : Kind → Order → Nat → Prop} {s : Stmt}
    (hn : s.NoBlocks) (hb : s.Below P.flags P.count) (hq : s.Accesses q)
    (hf : ∀ k o x, P.flags ≤ x → q k o x) : (P.lower s).Accesses q := by
  induction s with
  | spawnAt =>
    exact ⟨trivial, fun j _ => hf _ _ _ (by simp only [start]; omega)⟩
  | joinAt => exact fun j _ => hf _ _ _ (by simp only [done]; omega)
  | tasksBlock => exact hn.elim
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 hn.1 hb.1 hq.1, ih2 hn.2 hb.2 hq.2⟩
  | loop _ _ ih => exact ih hn hb hq
  | _ => exact hq

namespace ChanProg

variable {P : ChanProg} {g : Nat → Option Nat} {m : Nat}

variable (g m) in
/-- An access that a thread can make while it does not hold the lock `m`: not to a place that
`g` gives `m`. -/
def Unguarded (k : Kind) (_ : Order) (x : Nat) : Prop := k = .fence ∨ g x ≠ some m

/-! ## The code of the program follows `HTrans` with `Unguarded` -/

section Code

variable (hm : m < P.locks) (hg : ∀ x m', g x = some m' → x < P.places)
include hm hg

omit hm in
theorem unguarded_of {k o x} (hx : P.places ≤ x) : Unguarded g m k o x :=
  Or.inr fun h => absurd (hg _ _ h) (by omega)

/-- Lowered code follows `HClean` with `Unguarded`, when its accesses to the places of `m` are
in blocks of `m`. -/
theorem hclean_guarded {s : CStmt} {held : List Nat} (hmh : m ∉ held) (hs : P.StmtOk s)
    (hb : ∀ b, s.Bases b → b.NoBlocks) (hgd : s.Guarded g held) :
    P.HClean m (Unguarded g m) (P.toTaskProg.lower (P.lowerC s)) := by
  have ug : ∀ {k o x}, P.places ≤ x → Unguarded g m k o x := fun h => unguarded_of hg h
  have hpt : P.places ≤ P.toTaskProg.flags := by simp only [toTaskProg, flag, slot, top]; omega
  -- The code of a channel operation accesses only the places from `places` up.
  have chan : ∀ {s : CStmt}, P.StmtOk s → ¬ s.Locks m → s.NoTasks →
      s.Accesses (fun _ _ _ => False) →
      P.HClean m (Unguarded g m) (P.toTaskProg.lower (P.lowerC s)) := fun hs hn hnt hq => by
    refine .other ((lower_notLock hm hs hn).and ?_)
    rw [TaskProg.lower_noTasks (lowerC_noTasks hnt)]
    exact (lowerC_accesses hs hq).mono fun _ _ x h =>
      ug (Nat.le_of_not_lt fun hx => h hx)
  induction s generalizing held with
  | base s =>
    have hb' : s.Below P.toTaskProg.flags P.toTaskProg.count := Stmt.Below.mono hs hpt
    refine .other ((lower_notLock hm hs nofun).and (TaskProg.lower_keeps (hb s rfl) hb'
      (hgd.mono fun _ _ _ h => ?_) fun _ _ x hx => ug (Nat.le_trans hpt hx)))
    rcases h with h | h
    · exact Or.inl h
    · exact Or.inr fun hx => hmh (h m hx)
  | send | recv | drop => exact chan hs nofun trivial trivial
  | locked m' s ih =>
    by_cases hne : m' = m
    · subst hne
      exact .locked (lower_notLock hm hs.2.2 hs.2.1)
    · have h1 := (cnt_inside (P := P) (q := .next) hs.1).1
      have h2 := (cnt_inside (P := P) (q := .serving) hs.1).1
      have hn : ∀ {k o}, P.NotLock m k o (P.cnt .next m') :=
        notLock_cnt (q := .next) hm hs.1 fun _ => hne
      have hv : ∀ {k o}, P.NotLock m k o (P.cnt .serving m') :=
        notLock_cnt (q := .serving) hm hs.1 fun _ => hne
      exact .seq (.other ⟨⟨⟨hn, ug h1⟩, hn, ug h1⟩, ⟨hv, ug h2⟩, hv, ug h2⟩)
        (.seq (ih (fun h => (List.mem_cons.1 h).elim (fun h => hne h.symm) hmh) hs.2.2 hb hgd)
          (.other ⟨⟨hv, ug h2⟩, hv, ug h2⟩))
  | seq _ _ ih1 ih2 =>
    exact .seq (ih1 hmh hs.1 (fun b h => hb b (Or.inl h)) hgd.1)
      (ih2 hmh hs.2 (fun b h => hb b (Or.inr h)) hgd.2)
  | ite _ _ _ ih1 ih2 =>
    exact .ite (ih1 hmh hs.1 (fun b h => hb b (Or.inl h)) hgd.1)
      (ih2 hmh hs.2 (fun b h => hb b (Or.inr h)) hgd.2)
  | loop _ _ ih => exact .loop (ih hmh hs hb hgd)
  | tasksBlock _ ih =>
    exact .seq (.other (P.toTaskProg.clearMarks_accesses _ _))
      (.seq (ih hmh hs hb hgd) (.other (P.toTaskProg.joinMarked_accesses fun u _ =>
        ⟨Avoid.notLock hm (Or.inr (Or.inr (by simp only [TaskProg.done, toTaskProg, flag, slot, top]; omega))),
          ug (by simp only [TaskProg.done, toTaskProg, flag, slot, top]; omega)⟩)))
  | fail | checkCancel | result => exact chan hs nofun trivial trivial

theorem hclean_guarded_body (hP : P.Ok) (hbm : ∀ b, P.main.Bases b → b.NoBlocks)
    (hbt : ∀ t b, (P.task t).Bases b → b.NoBlocks) (hgm : P.main.Guarded g [])
    (hgt : ∀ t, (P.task t).Guarded g []) (t : Nat) :
    P.HClean m (Unguarded g m) (P.body t) := by
  have ug : ∀ {k o x}, P.places ≤ x → Unguarded g m k o x := fun h => unguarded_of hg h
  have hpt : P.places ≤ P.toTaskProg.flags := by simp only [toTaskProg, flag, slot, top]; omega
  cases t with
  | zero => exact hclean_guarded hm hg (by simp) hP.main hbm hgm
  | succ t =>
    show P.HClean m (Unguarded g m) (P.toTaskProg.taskCode t)
    unfold TaskProg.taskCode
    split
    · refine .seq (.other ⟨Avoid.notLock hm (Or.inr (Or.inr ?_)), ug ?_⟩)
        (.seq (hclean_guarded hm hg (by simp) (hP.task t) (hbt t) (hgt t))
          (.other ⟨Avoid.notLock hm (Or.inr (Or.inr ?_)), ug ?_⟩))
      all_goals simp only [TaskProg.start, TaskProg.done, toTaskProg, flag, slot, top]; omega
    · exact .other trivial

end Code

/-! ## The history of an access to a place of the lock -/

variable (P m) in
/-- What a state of `Hold` says about an access to a place of `m` at the position `i`. -/
def GPre (h : Hold) (L : List Label) (i : Nat) : Prop :=
  match h with
  | .free | .leaving _ => False
  | .took t => L[0]? = some ⟨.write, .sc, P.cnt .next m, t + 1, true⟩ ∧
    ∃ iE, 0 < iE ∧ iE < i ∧ L[iE]? = some ⟨.read, .sc, P.cnt .serving m, t, false⟩ ∧
      P.NoUnlock m L 0 i
  | .wait t => ∃ iE < i, L[iE]? = some ⟨.read, .sc, P.cnt .serving m, t, false⟩ ∧
      P.NoUnlock m L 0 i
  | .held _ => P.NoUnlock m L 0 i

variable (P m) in
/-- The access at the position `i` is in a block of `m`: the block took the ticket `t` at `iT`,
read `serving` as `t` at `iE`, and wrote `serving` nowhere from the ticket to `i`. -/
def GuardHist (L : List Label) (i : Nat) : Prop :=
  ∃ (t : Int) (iT iE : Nat), iT + 1 < iE ∧ iE < i ∧
    L[iT]? = some ⟨.read, .sc, P.cnt .next m, t, false⟩ ∧
    L[iT + 1]? = some ⟨.write, .sc, P.cnt .next m, t + 1, true⟩ ∧
    L[iE]? = some ⟨.read, .sc, P.cnt .serving m, t, false⟩ ∧ P.NoUnlock m L (iT + 1) i

theorem GuardHist.cons {L i} {a : Label} (h : P.GuardHist m L i) :
    P.GuardHist m (a :: L) (i + 1) := by
  obtain ⟨t, iT, iE, h1, h2, hT, hC, hE, hno⟩ := h
  exact ⟨t, iT + 1, iE + 1, by omega, by omega, by simpa using hT, by simpa using hC,
    by simpa using hE, hno.cons⟩

/-- Before each access to a place of `m`, its thread took a ticket of `m`, read `serving` as the
ticket, and wrote `serving` nowhere from the ticket to the access. -/
theorem HAccepts.guard_hist (hm : m < P.locks) (hg : ∀ x, g x = some m → x < P.places) {h L}
    (hacc : P.HAccepts m (Unguarded g m) h L) :
    ∀ {i : Nat} {l : Label}, L[i]? = some l → l.kind ≠ .fence → g l.loc = some m →
      P.GPre m h L i ∨ P.GuardHist m L i := by
  have hnx := (cnt_inside (P := P) (q := .next) hm).1
  have hsv := (cnt_inside (P := P) (q := .serving) hm).1
  induction hacc with
  | nil => intro i l hl; simp at hl
  | @cons h h' a L htr _ ih =>
    intro i l hl hk hx
    have hlt := hg _ hx
    cases i with
    | zero =>
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hl
      subst hl
      cases htr with
      | other hc _ hG =>
        cases h with
        | free =>
          rcases hG rfl with h | h
          · exact (hk h).elim
          · exact (h hx).elim
        | held t => exact Or.inl fun _ _ _ h1 _ => absurd h1 (Nat.not_lt_zero _)
        | took | wait | leaving => exact hc.elim
      | take | count | peek | enter | leave | unlock =>
        simp only at hlt
        omega
    | succ i =>
      have notW {b : Label} (hb : b.kind ≠ .write ∨ b.loc ≠ P.cnt .serving m) :
          ¬ (b.kind = .write ∧ b.loc = P.cnt .serving m) := fun ⟨h1, h2⟩ => hb.elim (· h1) (· h2)
      rcases ih (by simpa using hl) hk hx with hpre | hist
      · cases htr with
        | other hc hav _ =>
          have ha : ¬ (a.kind = .write ∧ a.loc = P.cnt .serving m) := by
            rcases hav with h | h
            · exact notW (Or.inl (by rw [h]; nofun))
            · exact notW (Or.inr h.2)
          cases h with
          | free => exact hpre.elim
          | held t => exact Or.inl (hpre.cons_zero ha)
          | took | wait | leaving => exact hc.elim
        | @take t =>
          obtain ⟨hC, iE, h0, h1, hE, hno⟩ := hpre
          exact Or.inr ⟨t, 0, iE + 1, by omega, by omega, by simp, by simpa using hC,
            by simpa using hE, hno.cons⟩
        | count =>
          obtain ⟨iE, h1, hE, hno⟩ := hpre
          exact Or.inl ⟨by simp, iE + 1, by omega, by omega, by simpa using hE,
            hno.cons_zero (notW (Or.inr fun h =>
              absurd (cnt_inj (q := .next) (q' := .serving) h hm hm).1 nofun))⟩
        | peek =>
          obtain ⟨iE, h1, hE, hno⟩ := hpre
          exact Or.inl ⟨iE + 1, by omega, by simpa using hE, hno.cons_zero (notW (Or.inl nofun))⟩
        | enter => exact Or.inl ⟨0, by omega, by simp, hpre.cons_zero (notW (Or.inl nofun))⟩
        | leave | unlock => exact hpre.elim
      · exact Or.inr hist.cons

/-! ## The executions of a program that follows the rule -/

/-- The writes at the start write only the places of the channels, the locks and the failures. -/
theorem initList_ge (hP : P.Ok) {i : Nat} {p : Nat × Int} (h : P.initList[i]? = some p) :
    P.places ≤ p.1 := by
  rcases initList_cases h with ⟨-, rfl⟩ | ⟨-, -, rfl⟩ | ⟨c, hc, j, hj, rfl⟩ | ⟨c, hc, j, hj, rfl⟩ |
      ⟨-, rfl | rfl⟩
  · simp
  · exact Nat.le_trans places_le_chan (Nat.le_add_right _ _)
  · exact (sig_inside hc (o := .send) hj).1
  · have := hP.cap c hc
    exact (sig_inside hc (o := .recv) (j := j) (by simp only [sigLen]; omega)).1
  all_goals
    have := top_le_slot (P := P)
    have : P.places ≤ P.top := by unfold top; omega
    simp only [flag]
    omega

/-- The bases of the code have no `tasks` block. -/
theorem CStmt.NoBlocks.bases {s : CStmt} (h : s.NoBlocks) : ∀ b, s.Bases b → b.NoBlocks := by
  induction s with
  | base => intro b hb; exact hb ▸ h
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact fun b hb => hb.elim (ih1 h.1 b) (ih2 h.2 b)
  | loop _ _ ih | locked _ _ ih => exact ih h
  | tasksBlock => exact h.elim
  | _ => exact fun _ => nofun

theorem Stmt.NoTasks.noBlocks {s : Stmt} (h : s.NoTasks) : s.NoBlocks := by
  induction s with
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact ⟨ih1 h.1, ih2 h.2⟩
  | loop _ _ ih => exact ih h
  | tasksBlock => exact h.elim
  | _ => trivial

theorem CStmt.NoTasks.bases {s : CStmt} (h : s.NoTasks) : ∀ b, s.Bases b → b.NoBlocks := by
  induction s with
  | base => intro b hb; exact hb ▸ Stmt.NoTasks.noBlocks h
  | seq _ _ ih1 ih2 | ite _ _ _ ih1 ih2 => exact fun b hb => hb.elim (ih1 h.1 b) (ih2 h.2 b)
  | loop _ _ ih | locked _ _ ih => exact ih h
  | tasksBlock => exact h.elim
  | _ => exact fun _ => nofun

theorem cseqAll_bases : ∀ {ps : List CPhase}, P.CPhasesOk g ps →
    ∀ b, (cseqAll ps).Bases b → b.NoBlocks
  | [], _, b, hb => by cases hb; trivial
  | .code _ :: _, ⟨hp, _, hrest⟩, b, hb => hb.elim (CStmt.NoTasks.bases hp b) (cseqAll_bases hrest b)
  | .block _ :: _, ⟨hp, _, hrest⟩, b, hb =>
    hb.elim (CStmt.NoBlocks.bases hp.1 b) (cseqAll_bases hrest b)

variable {ps : List CPhase} {X : Execution} (run : P.toTaskProg.toProg.Run X)
include run

/-- Before each access to a place of the lock `m`, its thread took a ticket `t` of `m`, then read
`serving` as the ticket in `e`, and wrote `serving` nowhere from the ticket to the access. -/
theorem guard_access (hF : P.CFootprints g ps) (hm : m < P.locks) {d : X.E}
    (hk : X.kind d ≠ .fence) (hd : g (X.loc d) = some m) :
    ∃ t t' e, X.rmw t t' ∧ X.loc t' = P.cnt .next m ∧ X.thread t = X.thread d ∧
      X.IsRead e ∧ X.loc e = P.cnt .serving m ∧ run.val e = run.val t ∧
      X.thread e = X.thread d ∧ run.idx t < run.idx e ∧ run.idx e < run.idx d ∧
      ∀ w, X.thread w = X.thread d → run.idx t < run.idx w → run.idx w ≤ run.idx d →
        ¬ (X.IsWrite w ∧ X.loc w = P.cnt .serving m) := by
  have hg : ∀ x m', g x = some m' → x < P.places := fun x m' h => (hF.guard x m' h).1
  have hlt := hg _ _ hd
  have hp : P.pre (X.thread d) ≤ run.idx d := by
    refine Nat.le_of_not_lt fun hp => ?_
    obtain ⟨-, p, hi, -, -, hl, -⟩ := init_of run hF.ok hp
    have := initList_ge hF.ok hi
    omega
  have hacc := Runs.haccepts (trace_rest (run.trace d)).2
    (.clean (hclean_guarded_body hm hg hF.ok (hF.main ▸ cseqAll_bases hF.phases)
      (fun t => CStmt.NoTasks.bases (hF.tasks t)) hF.guarded_main hF.guarded_task _))
  have hget := view_get run (e := d) rfl (Nat.le_refl _) hp
  rcases hacc.guard_hist hm (fun x h => hg x m h) hget hk hd with
    h | ⟨t, iT, iE, h1, h2, hT, hC, hE, hno⟩
  · exact h.elim
  obtain ⟨d1, hd1t, hd1i⟩ := run.idx_down d (P.pre (X.thread d) + iT) (by omega)
  obtain ⟨d2, hd2t, hd2i⟩ := run.idx_down d (P.pre (X.thread d) + iT + 1) (by omega)
  obtain ⟨e, het, hei⟩ := run.idx_down d (P.pre (X.thread d) + iE) (by omega)
  have g1 := view_get run (e := d) hd1t (by omega) (by omega)
  rw [show run.idx d1 - P.pre (X.thread d) = iT by omega, hT] at g1
  have g2 := view_get run (e := d) hd2t (by omega) (by omega)
  rw [show run.idx d2 - P.pre (X.thread d) = iT + 1 by omega, hC] at g2
  have g3 := view_get run (e := d) het (by omega) (by omega)
  rw [show run.idx e - P.pre (X.thread d) = iE by omega, hE] at g3
  simp only [Option.some.injEq, Label.mk.injEq] at g1 g2 g3
  obtain ⟨-, -, -, hv1, -⟩ := g1
  obtain ⟨-, -, hl2, -, hr2⟩ := g2
  obtain ⟨hk3, -, hl3, hv3, -⟩ := g3
  refine ⟨d1, d2, e, (run.rmw_iff d1 d2).2 ⟨hd1t.trans hd2t.symm, by omega, hr2.symm⟩, hl2.symm,
    hd1t, hk3.symm, hl3.symm, by rw [← hv3, ← hv1], het, by omega, by omega, ?_⟩
  intro w hwt hw1 hw2 hww
  rcases Nat.lt_or_eq_of_le hw2 with hw2 | hw2
  · exact hno (run.idx w - P.pre (X.thread d)) _ (by omega) (by omega)
      (view_get run hwt (by omega) (by omega)) hww
  · have := run.idx_inj w d hwt hw2
    subst this
    have := (cnt_inside (P := P) (q := .serving) hm).1
    rw [hww.2] at hlt
    omega

/-- Two accesses to a place of the lock `m`, in two threads, are in two blocks of `m`. The block
with the smaller ticket happens before the other. -/
theorem guard_hb (hF : P.CFootprints g ps) (wf : X.WellFormed) (hcons : X.Consistent)
    (ho : X.CoOrder) (hfin : X.Finite) {a b : X.E} (hka : X.kind a ≠ .fence)
    (hkb : X.kind b ≠ .fence) (ha : g (X.loc a) = some m) (hb : g (X.loc b) = some m)
    (ht : X.thread a ≠ X.thread b) : X.hb a b ∨ X.hb b a := by
  have hm := (hF.guard _ _ ha).2
  obtain ⟨ta, ta', ea, hra, hral, htat, hear, heal, heav, heat, hia, hea, hnoa⟩ :=
    guard_access run hF hm hka ha
  obtain ⟨tb, tb', eb, hrb, hrbl, htbt, hebr, hebl, hebv, hebt, hib, heb, hnob⟩ :=
    guard_access run hF hm hkb hb
  have hN := counter run hF.ok wf (q := .next) hm
  rcases Int.lt_trichotomy (run.val ta) (run.val tb) with h | h | h
  · exact Or.inl (lock_excl run hF.ok wf hcons ho hfin hm hra hral htat.symm
      (fun w hw => hnoa w (hw.trans htat)) hebr hebl (by rw [hebv]; exact h)
      (Or.inr ((run.po_iff eb b).2 ⟨hebt, heb⟩)))
  · have := hN.tickets wf hcons ho hfin hra hral hrb hrbl h
    subst this
    exact (ht (htat.symm.trans htbt)).elim
  · exact Or.inr (lock_excl run hF.ok wf hcons ho hfin hm hrb hrbl htbt.symm
      (fun w hw => hnob w (hw.trans htbt)) hear heal (by rw [heav]; exact h)
      (Or.inr ((run.po_iff ea a).2 ⟨heat, hea⟩)))

omit run in
/-- A program with channels and locks that follows the rule has no data race, in each execution
whose `co` orders the writes to each place. -/
theorem raceFree (hF : P.CFootprints g ps) (hX : P.toTaskProg.toProg.Execs X)
    (wf : X.WellFormed) (hcons : X.Consistent) (ho : X.CoOrder) (hfin : X.Finite) :
    X.RaceFree := by
  obtain ⟨run⟩ := hX
  refine raceFree_of hF ⟨run⟩ wf hcons ho hfin fun hr ha => ?_
  obtain ⟨⟨-, hacca, haccb, hl, -⟩, ht, hab, hba, -⟩ := hr
  have nf : ∀ {e : X.E}, X.IsAccess e → X.kind e ≠ .fence := by
    rintro e (h | h) hf
    · simp [Execution.IsRead, hf] at h
    · simp [Execution.IsWrite, hf] at h
  rcases guard_hb run hF wf hcons ho hfin (nf hacca) (nf haccb) ha (hl ▸ ha) ht with h | h
  · exact hab h
  · exact hba h

omit run in
/-- DRF-SC from the rule: for a program with channels and locks that follows it and has no
failures, each finite execution that the model allows, with a `co` that orders the writes to each
place, is an interleaving. A failure uses release and acquire, so the program needs `errors`
false. -/
theorem drf_sc_of_footprints (hF : P.CFootprints g ps) (he : P.errors = false)
    (hm : ∀ b, P.main.Bases b → b.SeqCstOnly) (ht : ∀ t b, (P.task t).Bases b → b.SeqCstOnly)
    (hX : P.toTaskProg.toProg.Execs X) (wf : X.WellFormed) (hfin : X.Finite)
    (ho : X.CoOrder) (hc : X.Consistent) : X.SeqConsistent :=
  P.drf_sc_co hm ht (not_fails he hF.ok.main) (fun t => not_fails he (hF.ok.task t))
    (fun _ hY wfY hsc hfinY hoY =>
      raceFree hF hY wfY (Execution.consistent_of_seqConsistent wfY hsc) hoY hfinY)
    hX wf hfin ho hc

end ChanProg

end Vx
