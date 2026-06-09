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

## 2026-06-08 — Results vs pre-registered predictions (Stage-1, clean 5-seed)

#1 LG→Pan worst: FALSIFIED. Reversed — Pan→LG is the harder direction for 8/9 models
   (gap-asymmetry ratio <1; only PatchTST matches the predicted direction).
#2 Cross ≫ within (≥1.5–2×): CONFIRMED, 5–7× for deep models.
#3 Error localized at low SOC: PARTIAL / mislocated. Dominant pattern is a systematic
   SOC-dependent tilt peaking at HIGH SOC (~0.8–0.9), not a low-SOC spike. Low-SOC
   extrapolation penalty is real but secondary (clearest in the unbounded linear baseline).
#4 Systematic bias not variance: CONFIRMED, strongly. Directional, opposite-signed by
   transfer direction (PAN→LG +; LG→PAN −), consistent across roster, smallest for PatchTST
   — the chemistry V–SOC offset fingerprint from EDA.

Net mechanism: zero-shot failure is a chemistry-driven systematic bias, not variance.
PatchTST transfers best because it carries the least of it.
