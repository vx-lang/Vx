import VxMemoryModel.Rc11

/-!
# Every interleaving is allowed

A sequentially consistent execution is one interleaving of the threads. The model must allow
it: a program that runs on one core, or under a global lock, behaves like this.
-/

namespace Vx.Execution

open Relation

variable {X : Execution}

theorem rs_ord (wf : X.WellFormed) {a b : X.E} (h : X.rs a b) : a = b ∨ TransGen X.ord a b := by
  obtain ⟨w, _, haw, _, _, hwb⟩ := h
  have hchain : w = b ∨ TransGen X.ord w b := by
    refine hwb.imp id (TransGen.flatten ?_)
    intro x y ⟨r, hxr, hry⟩
    exact (TransGen.single (ord_rf hxr)).tail (ord_po (wf.rmw_po _ _ hry))
  rcases haw with rfl | ⟨haw, _⟩
  · exact hchain
  · rcases hchain with rfl | hwb
    · exact Or.inr (TransGen.single (ord_po haw))
    · exact Or.inr (TransGen.head (ord_po haw) hwb)

theorem sw_ord (wf : X.WellFormed) {a b : X.E} (h : X.sw a b) : TransGen X.ord a b := by
  obtain ⟨_, _, x, y, z, hax, hxy, hyz, _, hzb⟩ := h
  have hyb : TransGen X.ord y b := by
    rcases hzb with rfl | ⟨hzb, _⟩
    · exact TransGen.single (ord_rf hyz)
    · exact (TransGen.single (ord_rf hyz)).tail (ord_po hzb)
  have hxb : TransGen X.ord x b := by
    rcases rs_ord wf hxy with rfl | hxy
    · exact hyb
    · exact hxy.trans hyb
  rcases hax with rfl | ⟨_, hax⟩
  · exact hxb
  · exact TransGen.head (ord_po hax) hxb

theorem hb_ord (wf : X.WellFormed) {a b : X.E} (h : X.hb a b) : TransGen X.ord a b :=
  TransGen.flatten (fun _ _ h => h.elim (fun h => TransGen.single (ord_po h)) (sw_ord wf)) h

theorem eco_ord {a b : X.E} (h : X.eco a b) : TransGen X.ord a b :=
  TransGen.mono (fun _ _ h => Or.inr h) h

theorem scb_ord (wf : X.WellFormed) {a b : X.E} (h : X.scb a b) : TransGen X.ord a b := by
  rcases h with h | ⟨c, d, hac, _, hcd, hdb, _⟩ | ⟨h, _⟩ | h | h
  · exact TransGen.single (ord_po h)
  · exact (TransGen.head (ord_po hac) (hb_ord wf hcd)).tail (ord_po hdb)
  · exact hb_ord wf h
  · exact TransGen.single (ord_co h)
  · exact TransGen.single (ord_fr h)

theorem psc_ord (wf : X.WellFormed) {a b : X.E} (h : X.psc a b) : TransGen X.ord a b := by
  rcases h with ⟨c, d, hac, hcd, hdb⟩ | ⟨_, _, _, _, hab⟩
  · have hac : a = c ∨ TransGen X.ord a c :=
      hac.imp (fun h => h.2) (fun h => hb_ord wf h.2.2)
    have hcb : TransGen X.ord c b := by
      rcases hdb with ⟨_, rfl⟩ | ⟨_, _, hdb⟩
      · exact scb_ord wf hcd
      · exact (scb_ord wf hcd).trans (hb_ord wf hdb)
    exact TransGen.of_refl hac hcb
  · rcases hab with hab | ⟨c, d, hac, hcd, hdb⟩
    · exact hb_ord wf hab
    · exact ((hb_ord wf hac).trans (eco_ord hcd)).trans (hb_ord wf hdb)

/-- Every sequentially consistent execution is consistent in the Vx model. -/
theorem consistent_of_seqConsistent (wf : X.WellFormed) (h : X.SeqConsistent) :
    X.Consistent where
  coherence a b hab := by
    rintro (rfl | hba)
    · exact h.acyclic a (hb_ord wf hab)
    · exact h.acyclic a ((hb_ord wf hab).trans (eco_ord hba))
  coherence_rmw a b hab hba :=
    h.acyclic a (TransGen.head (ord_po (wf.rmw_po _ _ hab)) (eco_ord hba))
  atomicity := h.atomicity
  seq_cst a hcyc := h.acyclic a (TransGen.flatten (fun _ _ => psc_ord wf) hcyc)
  no_thin_air a hcyc :=
    h.acyclic a (TransGen.mono (fun _ _ h => h.imp id (fun h => Or.inl h)) hcyc)

end Vx.Execution
