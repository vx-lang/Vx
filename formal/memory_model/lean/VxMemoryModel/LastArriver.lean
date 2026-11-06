import VxMemoryModel.Scopes
import VxMemoryModel.Arc

/-!
# The last block reads every partial result

A kernel finds the last block with a counter (#2139). Each block writes its partial result, runs a
release fence and adds 1 to the counter. The block whose add reads `n - 1` is the last. It runs an
acquire fence and reads the partial results. `last_block_reads_every_partial` shows that each
partial write happens before that read when the adds and the fences have device scope. The edge
from fence to fence goes through the release sequence of the adds. `block_scope_breaks_sw` shows
what fails with block scope: an add of one block does not synchronize with a read in another.
-/

namespace Vx.ScopedExecution

open Relation Classical Execution

variable {S : ScopedExecution}

/-- If the counter starts at 0 and only the `n` adds of `adds` change it, no add reads `n` or more.
An add that read `v` would need an add for each value from 0 to `v`. -/
theorem Counter.read_lt (wf : S.WellFormed) {val : S.E → Int} {x : Nat} (hC : S.Counter val x)
    (hzero : ∀ a, S.IsWrite a → S.loc a = x → (∀ r, ¬ S.rmw r a) → val a = 0)
    {adds : List S.E} (hadds : ∀ r w, S.rmw r w → S.loc w = x → r ∈ adds)
    {r w : S.E} (hrw : S.rmw r w) (hwx : S.loc w = x) : val r < adds.length := by
  apply Int.lt_of_not_ge
  intro hge
  have hrx := (hC.add r w hrw hwx).1
  have hr := (wf.rmw_atomic r w hrw).1
  -- For each `v ≤ adds.length`, an add reads `v`.
  have pick : ∀ v : Nat, v ≤ adds.length → ∃ r', r' ∈ adds ∧ val r' = v := by
    intro v hv
    by_cases hlt : (v : Int) < val r
    · obtain ⟨r', w', h, hw', hv'⟩ := hC.exists_of_lt wf
        (fun a ha hax hi => (hzero a ha hax hi).symm ▸ Int.natCast_nonneg v) r hr hrx hlt
      exact ⟨r', hadds r' w' h hw', hv'⟩
    · exact ⟨r, hadds r w hrw hwx, by omega⟩
  let f : Nat → S.E := fun v => if h : v ≤ adds.length then choose (pick v h) else r
  have hf : ∀ v, v ≤ adds.length → f v ∈ adds ∧ val (f v) = v := fun v h => by
    simp only [f, h, dite_true]; exact choose_spec (pick v h)
  have hnd : ((List.range (adds.length + 1)).map f).Nodup := by
    refine List.pairwise_map.2 ((List.nodup_range).imp_of_mem ?_)
    intro a b ha hb hab heq
    have h1 := (hf a (by simp at ha; omega)).2
    have h2 := (hf b (by simp at hb; omega)).2
    exact hab (by rw [heq] at h1; omega)
  have hsub : (List.range (adds.length + 1)).map f ⊆ adds := by
    intro e he
    obtain ⟨v, hv, rfl⟩ := List.mem_map.1 he
    exact (hf v (by simp at hv; omega)).1
  have := hnd.length_le_of_subset hsub
  simp only [List.length_map, List.length_range] at this
  omega

/-- `n` blocks each run: a plain write `p` of a partial result, a release fence `f`, and an add
`r`, `w` to the counter `x`. The last block runs an add `rl`, `wl` that reads `n - 1`, then an
acquire fence `g`, then a read `q` of a partial result. The adds and the fences have device
scope, and all the threads run on one device. Then the partial write of each other block happens
before `q`. -/
theorem last_block_reads_every_partial (wf : S.WellFormed) (hc : S.Consistent) (ho : S.CoOrder)
    (hfin : S.Finite) {val : S.E → Int} {x : Nat} (hC : S.Counter val x)
    (hzero : ∀ a, S.IsWrite a → S.loc a = x → (∀ r, ¬ S.rmw r a) → val a = 0)
    {adds : List S.E} (hadds : ∀ r w, S.rmw r w → S.loc w = x → r ∈ adds)
    (hdev : ∀ r w, S.rmw r w → S.loc w = x → S.scope r = .device ∧ S.scope w = .device)
    (hone : ∀ a b, S.SameGroup .device a b)
    {p f r w : S.E} (hpf : S.po p f) (hf : S.IsFence f) (hfr : S.IsRelease f)
    (hfd : S.scope f = .device) (hfw : S.po f w) (hrw : S.rmw r w) (hwx : S.loc w = x)
    {rl wl g q : S.E} (hl : S.rmw rl wl) (hlx : S.loc wl = x) (hln : val rl = adds.length - 1)
    (hlg : S.po rl g) (hg : S.IsFence g) (hga : S.IsAcquire g) (hgd : S.scope g = .device)
    (hgq : S.po g q) (hne : w ≠ wl) : S.hb p q := by
  have wfR := relaxed_wellFormed wf
  have hcR := relaxed_consistent hc
  have hoR : S.relaxed.CoOrder := ⟨ho.trans, ho.total⟩
  have hfinR : S.relaxed.Finite := hfin
  have hCR : S.relaxed.Counter val x := ⟨hC.1, hC.2, hC.3, hC.4, hC.5⟩
  have hsrc : ∀ r y, S.rmw r y → S.loc y = x → S.loc r = x ∧ ∃ s, S.rf s r :=
    fun r y h hy => ⟨(hC.add r y h hy).1, hC.rf_source r (wf.rmw_atomic r y h).1⟩
  have hrmw : ∀ y, S.IsWrite y → S.loc y = x → (∃ a, S.co a y) → ∃ r, S.rmw r y := by
    intro y hy hyx ⟨a, hay⟩
    refine Classical.byContradiction fun hn => ?_
    exact hCR.init_first wfR hcR hoR hfinR hy hyx
      (fun r h => hn ⟨r, h⟩) a hay
  obtain ⟨-, hww, -, hwa⟩ := wf.rmw_atomic r w hrw
  obtain ⟨-, hlw, hla, -⟩ := wf.rmw_atomic rl wl hl
  -- `w` comes before `wl` in `co`. Otherwise `r` reads `n` or more.
  have hco : S.co w wl := by
    rcases ho.total w wl hww hlw (hwx.trans hlx.symm) hne with h | h
    · exact h
    · have hm := hCR.mono wfR hcR hoR hfinR wl w hlw hww hlx hwx h
      have := Counter.read_lt wf hC hzero hadds hrw hwx
      have := (hC.add r w hrw hwx).2
      have := (hC.add rl wl hl hlx).2
      omega
  obtain ⟨hrlx, y, hy⟩ := hsrc rl wl hl hlx
  have hloc : S.loc rl = S.loc wl := hrlx.trans hlx.symm
  have hyx : S.loc y = x := (wf.rf_write_read y rl hy).2.2.trans hrlx
  have hyw : S.IsWrite y := (wf.rf_write_read y rl hy).1
  -- `y` is the write of an add, of device scope.
  have hya : S.IsAtomic y ∧ S.scope y = .device := by
    by_cases hwy : w = y
    · exact hwy ▸ ⟨hwa, (hdev r w hrw hwx).2⟩
    · have : ∃ a, S.co a y := by
        rcases rmw_between wfR hcR hoR hy hl hloc hco with h | h
        · exact absurd h hwy
        · exact ⟨w, h⟩
      obtain ⟨r', h'⟩ := hrmw y hyw hyx this
      exact ⟨(wf.rmw_atomic r' y h').2.2.2, (hdev r' y h' hyx).2⟩
  have hchain : w = y ∨ TransGen (fun x y => ∃ r, S.rf x r ∧ S.rmw r y) w y := by
    rcases rmw_between wfR hcR hoR hy hl hloc hco with rfl | h
    · exact Or.inl rfl
    · exact Or.inr (rmw_chain_of_co wfR hcR hoR hfinR hrmw hsrc w y hyx h)
  have reach : ∀ a b, S.IsAtomic a → S.scope a = .device → S.Reaches a b :=
    fun a b ha hs => ⟨ha, Or.inr (hs ▸ hone a b)⟩
  -- Each write in the chain from `w` to `y` is on the counter, so each add in it has device
  -- scope. So the chain continues the release sequence.
  have step_loc : ∀ c d, (∃ r, S.rf c r ∧ S.rmw r d) → S.loc d = x → S.loc c = x :=
    fun c d ⟨r, h1, h2⟩ hd => (wf.rf_write_read c r h1).2.2.trans (hC.add r d h2 hd).1
  have chain_loc : ∀ c d, TransGen (fun x y => ∃ r, S.rf x r ∧ S.rmw r y) c d →
      S.loc d = x → S.loc c = x := by
    intro c d h
    induction h with
    | single h => exact step_loc _ _ h
    | tail _ h ih => exact fun hd => ih (step_loc _ _ h hd)
  have hchain' : w = y ∨ TransGen S.rsStep w y := by
    refine hchain.imp_right fun h => rsStep_chain h ?_
    intro c q d hc hq hcw hd
    have hdx : S.loc d = x := by
      rcases hd with rfl | hd
      · exact hyx
      · exact chain_loc d y hd hyx
    have hqx := (hC.add q d hq hdx).1
    refine ⟨reach c q ?_ ?_, reach q c (wf.rmw_atomic q d hq).2.2.1 (hdev q d hq hdx).1⟩
    · rcases hcw with rfl | ⟨r', hc'⟩
      · exact hwa
      · exact (wf.rmw_atomic r' c hc').2.2.2
    · rcases hcw with rfl | ⟨r', hc'⟩
      · exact (hdev r c hrw hwx).2
      · exact (hdev r' c hc' ((wf.rf_write_read c q hc).2.2.trans hqx)).2
  have hsw : S.sw f g := ⟨hfr, hga,
    ⟨reach f g (atomic_of_release hfr) hfd, reach g f (atomic_of_acquire hga) hgd⟩,
    w, y, rl, Or.inr ⟨hf, hfw⟩, ⟨w, hww, Or.inl rfl, hww, hwa, hchain'⟩, hy,
    ⟨reach y rl hya.1 hya.2, reach rl y hla (hdev rl wl hl hlx).1⟩, hla, Or.inr ⟨hlg, hg⟩⟩
  have hpg : S.hb p g := TransGen.tail (TransGen.single (Or.inl hpf)) (Or.inr (Or.inl hsw))
  exact TransGen.tail hpg (Or.inl hgq)

/-- So the read of the last block does not race with the partial write. -/
theorem last_block_no_race {p q : S.E} (h : S.hb p q) : ¬ S.DataRace p q :=
  fun hr => hr.2.2.1 h

/-- With block scope, a write and a read in two blocks do not reach each other. So the read of
the counter in the last block gives no `sw` from an add of another block. -/
theorem block_scope_breaks_sw {y z : S.E} (hy : S.scope y = .block)
    (hblk : ¬ S.SameGroup .block y z) : ¬ S.Strong y z :=
  fun h => h.1.2.elim (fun h' => by rw [hy] at h'; cases h') (fun h' => hblk (hy ▸ h'))

end Vx.ScopedExecution
