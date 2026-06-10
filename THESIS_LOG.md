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

## 2026-06-09 — Stage 2 (fine-tuning data-efficiency ladder)

- Fine-tuning reaches the within-target oracle; recalibration plateaus far short.
  Affine recal zeroes the directional bias but closes only ~40% (PAN→LG) / ~25% (LG→PAN)
  of the gap; fine-tuning reaches ~95% by 50% target data, ~100% at full.
  → Refines #4: chemistry bias is real and removable but is the MINORITY of the transfer
    error; most of the gap is representational and needs weight adaptation, not correction.
- Architecture cross-over flips with adaptation: PatchTST best zero-shot; LSTM/GRU overtake
  under any fine-tuning (PatchTST worst fine-tuned — higher in-distribution floor).
  Rule: no target labels → PatchTST; some labels → fine-tune LSTM/GRU.
- Recalibration unstable below ~10% target, can be WORSE than zero-shot (esp. LG→PAN @5%).
  Affine > bias-only (a slope/tilt component exists, per #3).
- Caveat: fine-tune uses a fixed target VAL (12/8 profiles) recal doesn't — low-fraction
  budgets not yet apples-to-apples; tighten before reporting low-fraction data efficiency.
- Implication for Stage 3 (#5): even supervised affine correction can't close the gap →
  the generalizable method must act at the representation level / use chemistry-aware
  features, not post-hoc correction. The open regime is zero/few target labels, since
  fine-tuning already solves the with-labels case.

## 2026-06-09 — Stage 2 (shared-budget re-run; supersedes prior val-asymmetric run)

- Fine-tuning reaches the within-target oracle; recal plateaus at ~42% (PAN→LG) / ~25% (LG→PAN).
  Architecture flip holds: PatchTST best zero-shot, worst fine-tuned; LSTM/GRU best fine-tuned.
- Data efficiency (honest, shared budget): fine-tune at 10% closes ~77% of the gap but is
  HIGH-VARIANCE below ~20% (~5 drive cycles) — sample too small to early-stop reliably.
  Reliable monotone gains from ~20% to the oracle.
- RETRACTED: prior "recal worse than zero-shot at 5%, esp LG→PAN" — a small-pool artifact.
  Corrected: low-fraction recal is high-variance but ~neutral-to-slightly-helpful; still plateaus.
- Regime: ≤~3 cycles both unstable (recal more stable, caps low); ≥~5 cycles fine-tune wins to oracle.

## 2026-06-10 — Stage 3 pre-registration (OCV-informed features; logged before any S3 results)

Method: append SOC_ocv = clamp(OCV_chem^{-1}(V)) as a 4th input channel to (V,I,T);
source chemistry's measured 25C C/20 OCV curve at train, target chemistry's at test.
Label-free: one C/20 characterization per chemistry, ZERO labeled target drive cycles.
OCV source: Panasonic full-range (4.20-2.50V); LG truncated at 2.80V (clamps low-SOC).

S3-1: OCV-zero-shot beats BOTH plain zero-shot (S1) and supervised recalibration (S2),
      both directions, with no labeled target drive cycles.
S3-2: Drives directional chemistry bias toward ~0 by construction (per-voltage, not just
      the global shift recal removed).
S3-3: Does NOT reach the supervised fine-tune oracle — residual IR/polarization mismatch
      remains; lands between recal and few-label fine-tune.
S3-4: Beats a marginal-alignment foil (CORAL / target input standardization) — confirms
      gap #5 (conditional/chemistry-structured, not a marginal input shift).
S3-5: cross_A (target=LG) benefits less than cross_B (target=Panasonic), with residual
      error concentrated at low SOC, due to the truncated LG OCV (V>2.80).

## Stage 3 RESULTS — predictions largely FALSIFIED (honest negative + one positive sub-finding)
- S3-1 FALSIFIED: OCV-mean ~neutral vs zero-shot, nowhere near S2 recal plateau.
- S3-2 FALSIFIED: bias not zeroed — loaded V≠OCV (IR drop) is the limiter.
- S3-3 FALSIFIED placement: OCV sits between zero-shot and recal, not recal↔oracle.
- S3-4 PARTIAL: OCV beats CORAL on patchtst/average; CORAL wrecks patchtst (0.058→0.121) — supports #5.
- S3-5 FALSIFIED (pre-warned): cross_A (target=LG) benefits MOST; LG truncation negligible (0.93%).
- POSITIVE: discharge-leg OCV (C3) consistently beats zero-shot both directions + cuts bias,
  confirming the NCA-hysteresis mechanism. Still short of supervised recalibration.
- CONCLUSION: naive OCV-referencing of loaded voltage gives only a small label-free gain;
  uncorrected IR/polarization is the limiter. A few target labels (recal/fine-tune) remain
  necessary for chemistry transfer. Optional fix: IR-correct V via HPPC R0 before the OCV inverse.