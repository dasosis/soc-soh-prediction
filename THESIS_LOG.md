## 2026-06-08 — Pre-training predictions: SOC cross-dataset zero-shot (LG 18650HG2 ↔ Panasonic 18650PF)

Grounded in EDA only (input + label distributions). No model trained yet.

1. **Direction asymmetry.** LG→Panasonic zero-shot error (RMSE & MAE) will exceed
   Panasonic→LG. LG→Pan extrapolates into unseen SOC < 0.136 and unseen 0%-regen
   cold operation; Pan→LG only interpolates.
2. **Cross- vs within-dataset.** Both zero-shot directions will be markedly worse
   than the within-dataset sanity baseline — expect ≥1.5–2× by survey precedent,
   possibly more given the chemistry change.
3. **Error localization (LG→Pan).** Error will concentrate (a) at low SOC, below
   ~0.14 where LG never trained, and (b) on the discharge voltage plateaus, where
   the ~42 mV (max 161 mV) NMC/NCA offset maps to a wide SOC band.
4. **Bias, not just variance.** LG→Pan predictions will show a non-zero mean error
   (a directional SOC bias tracking the chemistry V–SOC offset), not merely more
   scatter.
5. **(Stage 3, not tested now.)** Marginal input alignment (normalization / MMD)
   will reduce the current-shift component but not remove the chemistry-offset bias;
   closing that needs chemistry-aware features or OCV correction.

To be revisited against Stage-1 numbers.
