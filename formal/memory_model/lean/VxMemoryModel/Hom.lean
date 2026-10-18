import VxMemoryModel.Rc11

/-!
# Maps between executions

A `Hom Y X` sends each event of `Y` to an event of `X`, and keeps every label and edge. All
the rules of the model forbid a cycle or a pair of edges, so if `X` follows a rule, `Y` follows
it too. The restriction of an execution to a set of events is the main example.
-/

namespace Relation.TransGen

theorem map {α β : Sort _} {r : α → α → Prop} {s : β → β → Prop} (f : α → β)
    (h : ∀ a b, r a b → s (f a) (f b)) {a b : α} (hab : TransGen r a b) :
    TransGen s (f a) (f b) := by
  induction hab with
  | single hab => exact single (h _ _ hab)
  | tail _ hbc ih => exact ih.tail (h _ _ hbc)

end Relation.TransGen

namespace Vx.Execution

open Relation

structure Hom (Y X : Execution) where
  f : Y.E → X.E
  kind : ∀ e, X.kind (f e) = Y.kind e
  order : ∀ e, X.order (f e) = Y.order e
  loc : ∀ e, X.loc (f e) = Y.loc e
  thread : ∀ e, X.thread (f e) = Y.thread e
  po : ∀ a b, Y.po a b → X.po (f a) (f b)
  rf : ∀ a b, Y.rf a b → X.rf (f a) (f b)
  co : ∀ a b, Y.co a b → X.co (f a) (f b)
  rmw : ∀ a b, Y.rmw a b → X.rmw (f a) (f b)

namespace Hom

variable {Y X : Execution} (h : Hom Y X)

theorem isRead {e : Y.E} : X.IsRead (h.f e) ↔ Y.IsRead e := by
  unfold IsRead; rw [h.kind]
theorem isWrite {e : Y.E} : X.IsWrite (h.f e) ↔ Y.IsWrite e := by
  unfold IsWrite; rw [h.kind]
theorem isFence {e : Y.E} : X.IsFence (h.f e) ↔ Y.IsFence e := by
  unfold IsFence; rw [h.kind]
theorem isAtomic {e : Y.E} : X.IsAtomic (h.f e) ↔ Y.IsAtomic e := by
  unfold IsAtomic; rw [h.order]
theorem isAcquire {e : Y.E} : X.IsAcquire (h.f e) ↔ Y.IsAcquire e := by
  unfold IsAcquire; rw [h.order]
theorem isRelease {e : Y.E} : X.IsRelease (h.f e) ↔ Y.IsRelease e := by
  unfold IsRelease; rw [h.order]
theorem isSeqCst {e : Y.E} : X.IsSeqCst (h.f e) ↔ Y.IsSeqCst e := by
  unfold IsSeqCst; rw [h.order]

theorem fr {a b : Y.E} (hab : Y.fr a b) : X.fr (h.f a) (h.f b) :=
  let ⟨w, hwa, hwb⟩ := hab
  ⟨h.f w, h.rf _ _ hwa, h.co _ _ hwb⟩

theorem rs {a b : Y.E} (hab : Y.rs a b) : X.rs (h.f a) (h.f b) := by
  obtain ⟨w, ha, haw, hw, hwa, hwb⟩ := hab
  refine ⟨h.f w, h.isWrite.2 ha, ?_, h.isWrite.2 hw, h.isAtomic.2 hwa, ?_⟩
  · rcases haw with rfl | ⟨hp, hl⟩
    · exact Or.inl rfl
    · exact Or.inr ⟨h.po _ _ hp, by rw [h.loc, h.loc, hl]⟩
  · rcases hwb with rfl | hwb
    · exact Or.inl rfl
    · exact Or.inr (TransGen.map h.f (fun _ _ ⟨r, hxr, hry⟩ =>
        ⟨h.f r, h.rf _ _ hxr, h.rmw _ _ hry⟩) hwb)

theorem sw {a b : Y.E} (hab : Y.sw a b) : X.sw (h.f a) (h.f b) := by
  obtain ⟨ha, hb, x, y, z, hax, hxy, hyz, hz, hzb⟩ := hab
  refine ⟨h.isRelease.2 ha, h.isAcquire.2 hb, h.f x, h.f y, h.f z, ?_, h.rs hxy,
    h.rf _ _ hyz, h.isAtomic.2 hz, ?_⟩
  · rcases hax with rfl | ⟨hf, hp⟩
    · exact Or.inl rfl
    · exact Or.inr ⟨h.isFence.2 hf, h.po _ _ hp⟩
  · rcases hzb with rfl | ⟨hp, hf⟩
    · exact Or.inl rfl
    · exact Or.inr ⟨h.po _ _ hp, h.isFence.2 hf⟩

theorem hb {a b : Y.E} (hab : Y.hb a b) : X.hb (h.f a) (h.f b) :=
  TransGen.map h.f (fun _ _ s => s.imp (h.po _ _) h.sw) hab

theorem eco {a b : Y.E} (hab : Y.eco a b) : X.eco (h.f a) (h.f b) :=
  TransGen.map h.f (fun _ _ s => s.imp (h.rf _ _) (Or.imp (h.co _ _) h.fr)) hab

theorem scb {a b : Y.E} (hab : Y.scb a b) : X.scb (h.f a) (h.f b) := by
  rcases hab with hp | ⟨c, d, hac, hl1, hcd, hdb, hl2⟩ | ⟨hh, hl⟩ | hc | hf
  · exact Or.inl (h.po _ _ hp)
  · exact Or.inr (Or.inl ⟨h.f c, h.f d, h.po _ _ hac, by rw [h.loc, h.loc]; exact hl1,
      h.hb hcd, h.po _ _ hdb, by rw [h.loc, h.loc]; exact hl2⟩)
  · exact Or.inr (Or.inr (Or.inl ⟨h.hb hh, by rw [h.loc, h.loc, hl]⟩))
  · exact Or.inr (Or.inr (Or.inr (Or.inl (h.co _ _ hc))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (h.fr hf))))

theorem psc {a b : Y.E} (hab : Y.psc a b) : X.psc (h.f a) (h.f b) := by
  rcases hab with ⟨c, d, hac, hcd, hdb⟩ | ⟨hfa, hsa, hfb, hsb, hab⟩
  · refine Or.inl ⟨h.f c, h.f d, ?_, h.scb hcd, ?_⟩
    · rcases hac with ⟨hs, rfl⟩ | ⟨hf, hs, hh⟩
      · exact Or.inl ⟨h.isSeqCst.2 hs, rfl⟩
      · exact Or.inr ⟨h.isFence.2 hf, h.isSeqCst.2 hs, h.hb hh⟩
    · rcases hdb with ⟨hs, rfl⟩ | ⟨hf, hs, hh⟩
      · exact Or.inl ⟨h.isSeqCst.2 hs, rfl⟩
      · exact Or.inr ⟨h.isFence.2 hf, h.isSeqCst.2 hs, h.hb hh⟩
  · refine Or.inr ⟨h.isFence.2 hfa, h.isSeqCst.2 hsa, h.isFence.2 hfb, h.isSeqCst.2 hsb, ?_⟩
    rcases hab with hh | ⟨c, d, hac, hcd, hdb⟩
    · exact Or.inl (h.hb hh)
    · exact Or.inr ⟨h.f c, h.f d, h.hb hac, h.eco hcd, h.hb hdb⟩

theorem ord {a b : Y.E} (hab : Y.ord a b) : X.ord (h.f a) (h.f b) :=
  hab.imp (h.po _ _) (Or.imp (h.rf _ _) (Or.imp (h.co _ _) h.fr))

include h in
/-- If `X` is consistent, `Y` is consistent. -/
theorem consistent (hX : X.Consistent) : Y.Consistent where
  coherence a b hab := by
    rintro (rfl | hba)
    · exact hX.coherence _ _ (h.hb hab) (Or.inl rfl)
    · exact hX.coherence _ _ (h.hb hab) (Or.inr (h.eco hba))
  coherence_rmw a b hab hba := hX.coherence_rmw _ _ (h.rmw _ _ hab) (h.eco hba)
  atomicity a b c hab hac hcb := hX.atomicity _ _ _ (h.rmw _ _ hab) (h.fr hac) (h.co _ _ hcb)
  seq_cst a hcyc := hX.seq_cst _ (TransGen.map h.f (fun _ _ => h.psc) hcyc)
  no_thin_air a hcyc :=
    hX.no_thin_air _ (TransGen.map h.f (fun _ _ s => s.imp (h.po _ _) (h.rf _ _)) hcyc)

include h in
/-- If `X` is sequentially consistent, `Y` is sequentially consistent. -/
theorem seqConsistent (hX : X.SeqConsistent) : Y.SeqConsistent where
  acyclic _ hcyc := hX.acyclic _ (TransGen.map h.f (fun _ _ => h.ord) hcyc)
  atomicity _ _ _ hab hac hcb := hX.atomicity _ _ _ (h.rmw _ _ hab) (h.fr hac) (h.co _ _ hcb)

end Hom

/-- The execution with only the events in `S`. -/
def restrict (X : Execution) (S : X.E → Prop) : Execution where
  E := {e // S e}
  kind e := X.kind e.val
  order e := X.order e.val
  loc e := X.loc e.val
  thread e := X.thread e.val
  po a b := X.po a.val b.val
  rf a b := X.rf a.val b.val
  co a b := X.co a.val b.val
  rmw a b := X.rmw a.val b.val

def restrictHom (X : Execution) (S : X.E → Prop) : Hom (X.restrict S) X where
  f e := e.val
  kind _ := rfl
  order _ := rfl
  loc _ := rfl
  thread _ := rfl
  po _ _ h := h
  rf _ _ h := h
  co _ _ h := h
  rmw _ _ h := h

theorem WellFormed.restrict {X : Execution} (wf : X.WellFormed) (S : X.E → Prop) :
    (X.restrict S).WellFormed where
  po_trans a b c := wf.po_trans a.val b.val c.val
  po_total a b ht hne := wf.po_total a.val b.val ht (fun h => hne (Subtype.ext h))
  rf_write_read w r := wf.rf_write_read w.val r.val
  co_writes a b := wf.co_writes a.val b.val
  co_irrefl a := wf.co_irrefl a.val
  rmw_po r w := wf.rmw_po r.val w.val
  rmw_atomic r w := wf.rmw_atomic r.val w.val

end Vx.Execution
