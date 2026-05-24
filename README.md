# OSSD-Basic

> **Orthogonal Self-Similarity Decomposition — reference MATLAB implementation.**  
> Accompanying the manuscript by F. Latifoğlu and L. Latifoğlu (2026).

**Authors:** Fatma Latifoğlu¹, Levent Latifoğlu²

**Affiliations:**  
¹ Department of Biomedical Engineering, Erciyes University, Kayseri, Türkiye  
² Department of Civil Engineering, Erciyes University, Kayseri, Türkiye  

**Contact:** flatifoglu@erciyes.edu.tr

---

OSSD-Basic decomposes a one-dimensional signal into mutually orthogonal modes
by reading off recurrent structure on the `(time, delay)` self-similarity plane.
Each mode corresponds to a band of pixels on that plane, lifted back to a
time-domain trajectory subspace and projected onto the input signal.

Three key design elements distinguish OSSD-Basic from existing decompositions:

1. **Adaptive proxy-correlation band merge** — bands whose τ-supports overlap
   are fused under a permissive correlation floor; bands on disjoint delay axes
   require near-identical correlation before merging.
2. **Dominant-component cascade (OSSD-DR)** — when a single mode concentrates
   most of the signal energy (CCI ≥ γ), it is subtracted and the pipeline
   re-runs on the residual, preventing energy masking of weaker components.
3. **Double MGS + LS re-projection** — Modified Gram–Schmidt orthogonalisation
   followed by least-squares amplitude refitting is applied at *two* points
   (post-refinement and post-K-reduction), reducing the inter-mode orthogonality index
 to near machine precision after the tested merge and pruning operations.

---

## Requirements

- MATLAB R2018b or newer  
- No toolbox dependencies for the core decomposition (`ossd_decompose.m`) or the included demo scripts (`demo_OSSD_Basic.m`, `reproduce.m`).

---

## Contents

### Core pipeline

| File | Role |
|------|------|
| `ossd_decompose.m` | Top-level entry point. Pass-1 + cascade Pass-2 + K-reduction. |
| `ossd_band_pipeline.m` | Stages 1–6: trajectory matrix, similarity plane, band extraction and merge. |
| `ossd_outer_pipeline.m` | Stage 7: similarity merge, MGS+LS, joint refinement, post-MGS+LS. |
| `ossd_default_params.m` | All default parameters (Table 2 in the manuscript). Single source of truth. |
| `ossd_joint_refine.m` | Coordinate-wise amplitude refinement (V2-style). |
| `ossd_similarity_metrics.m` | Pairwise envelope / ACF similarity helpers. |
| `ossd_diagnostics.m` | Reconstruction, orthogonality, and spectral diagnostics. |
| `ossd_msr.m` | Mode Stability under Resampling (multi-seed robustness). |
| `ossd_sspi.m` | Self-Similarity Preservation Index (structural fidelity). |
| `ossd_sdi.m` | Snowflake Disjointness Index (delay-axis non-collision). |

### Reproduction

| File | Reproduces |
|------|-----------|
| `reproduce.m` | Table 4 synthetic validation headline numbers (+ MSR auxiliary metric). |
| `demo_OSSD_Basic.m` | Full synthetic demo with figures (Tables 4–5, Figures 3–5). |

Extended scripts used during manuscript preparation (ablation, sensitivity,
classical comparison, hydrological forecasting) are **not included in this
minimal release** and are available upon request.

---

## Quick start

Open MATLAB R2018b or newer in the package directory and run:

```matlab
demo_OSSD_Basic
```

This produces figures (synthetic ground truth, OSSD modes, reconstruction
overlay, tau-fingerprint, fractal scaling) and a console report including:

- per-mode energy / median IF / IF variance / bandwidth,
- reconstruction error (RRE, RMSE) and orthogonality index (OI),
- the inter-mode |corr| matrix,
- the mode-to-truth |corr| matrix,
- ground-truth recoveries after burst grouping,
- the three self-similarity-based diagnostic metrics (CCI, SSPI, SDI).

Total runtime: under one second on a modern laptop.

---

## Reproducing the headline numbers

```matlab
reproduce
```

Expected output on the synthetic benchmark
(N = 3200, fs = 400 Hz, rng(7, 'twister')):

| Metric | Value |
|--------|-------|
| Relative reconstruction error (RRE) | 0.0845 |
| Orthogonality index (OI) | 9.4 × 10⁻¹⁸ |
| Mean inter-mode \|corr\| | 3.1 × 10⁻⁵ |
| Max inter-mode \|corr\| | 1.2 × 10⁻⁴ |
| c1 recovery (abs corr) | 0.9559 |
| c_burst recovery (c2 + c4) | 0.7793 |
| c3 recovery | 0.7058 |
| Reconstruction SNR (vs noise-free) | 27.14 dB |
| CCI (Cascade Coverage Index) | 0.8146 |
| SSPI (Self-Similarity Preservation) | 0.9858 |
| SDI (Snowflake Disjointness) | 0.9060 |

All numbers are deterministic for the fixed RNG seed.

---

## How it works

OSSD-Basic runs the following stages in order:

1. **Trajectory matrix.** Build `X = Hankel(x, L)` with calibrated
   window length `L = 128` for `N ≥ 200`.
2. **Self-similarity plane.** Compute the column Gram matrix and read
   off the `(window-index, delay)` plane up to `max_tau = 120`.
3. **Robust normalisation.** Median / MAD per delay column (Eq. 6 in paper).
4. **Band extraction.** 8-connected components above a robust MAD-based
   threshold (`band_thresh_k` = 1.5), with minimum area, time-span, and tau-span constraints.
5. **Adaptive proxy-correlation merge.** Bands sharing the same delay
   structure merge under a relaxed correlation floor (c_min = 0.75);
   bands on different delay axes require near-identical correlation
   (c_max = 0.95). 
6. **Subspace decomposition.** Per-band mean trajectory + low-rank SVD
   truncation by energy target (rank_energy_target = 0.93).
7. **Outer pipeline.** Similarity merge → MGS + LS amplitudes →
   joint refinement → **first MGS + LS re-projection** (reduces OI to near machine precision). 
8. **Dominant-component cascade.** If CCI ≥ γ = 0.70, the dominant mode
   is subtracted and the pipeline re-runs on the residual to recover
   weaker components. 
9. **Similarity-aware K-reduction.** Final mode count enforced by merging
   the most similar pair or pruning the lowest-energy column.
10. **Second MGS + LS re-projection (post-K-reduction).** Restores numerical orthogonality
    perturbed by K-reduction. 

Four diagnostic frameworks accompany the algorithm:

| Metric | Description |
|--------|-------------|
| **CCI** | Single-mode energy concentration of pass 1. Triggers cascade when CCI ≥ 0.70. |
| **SSPI** | Frobenius-norm preservation of the input self-similarity plane structure. |
| **SDI** | Pairwise τ-axis disjointness: each mode in its own delay neighbourhood. |
| **MSR** | Mode Stability under independent noise Resampling (see `ossd_msr.m`). |

---

## Default parameters (Table 2 in manuscript)

All defaults are defined in `ossd_default_params.m`.  
Most users only need to set `fs` and optionally `n_modes`.

| Parameter | Default (N=3200) | Description |
|-----------|-----------------|-------------|
| `L` | 128 | Hankel window length. Requires L ≥ 1.5 × T_dominant. |
| `max_tau` | 120 | Max delay on τ-axis. Requires max_tau ≥ T_dominant. |
| `band_thresh_k` | 1.5 | MAD multiplier for band detection. Primary tuning knob. |
| `rank_energy_target` | 0.93 | Cumulative energy fraction for SVD rank selection. |
| `const_band_proxy_corr_min` | 0.75 | Merge floor for overlapping bands (c_min). |
| `const_band_proxy_corr_max` | 0.95 | Merge ceiling for disjoint bands (c_max). |
| `const_band_tau_overlap_min` | 0.40 | τ-overlap fraction at which c_min applies. |
| `const_dominant_ratio_for_cascade` | 0.70 | CCI threshold γ for cascade trigger. |
| `joint_refine_iters` | auto [4, 12] | Joint refinement iterations between the first and second MGS+LS re-projection steps. |
| `n_modes` | `[]` (auto) | Target output K. Empty = automatic energy-rank selection. |

Geometric parameters (`L`, `max_tau`, `min_band_area`, …) are derived
from `N` automatically when not overridden.

---

## Ablation study (Table 1)

Reference values from the manuscript (Section 2.4). The ablation script
(`ossd_ablation.m`) is available upon request.

| Variant | OI | Burst Recovery | Key finding |
|---------|----|----------------|-------------|
| V0 Full OSSD-Basic | 3.2×10⁻¹⁴ | 0.559 | Reference |
| V1 No JointRefine | 3.2×10⁻¹⁴ | 0.559 | No change — MGS+LS provides the primary contribution|
| V2 No MGS+LS | **0.203** | **0.194** | OI ↑ ~12 orders of magnitude; burst −65% |
| V3 Fixed rank | 7.9×10⁻¹⁴ | 0.364 | Burst −35% |
| V4 Low k=1.5 | 3.2×10⁻¹⁴ | 0.559 | Same as default (no ablation effect on this benchmark) |
| V5 No Refine + No MGS | 0.203 | 0.194 | Same as V2 |

---

## Sensitivity analysis 

Reference findings from Section 2.7. The sensitivity script
(`ossd_sensitivity_analysis.m`) is available upon request.

| Parameter | Range | Key finding |
|-----------|-------|-------------|
| L | 64–256 | K=4 only at L=128 and L=256; K=0–2 at intermediate values |
| tau_max | 60–180 | tau_max < 120 → K=1 (axis too short); tau_max ≥ 120 → K=4 stable |
| k | 1.5–3.5 | k = 1.5–2.0 → K = 4 stable; k ≥ 2.5 → K = 0 (all bands suppressed) |

---

## Limitations and scope

- OSSD-Basic targets **one-dimensional** signals. Multi-channel input is not supported.
- `max_tau = 120` is calibrated for signals where the dominant carrier period
  is below 120 samples. For very-low-frequency carriers, override `max_tau`
  (recommended: `max_tau ≥ 1.5 × T_dominant`).
- The cascade is **single-pass**: it triggers at most once per call. For signals
  with multiple comparable-energy carriers, call `ossd_decompose` iteratively
  on successive residuals, or increase `n_modes`.
- Results in the paper are from a **single noise realisation** (rng(7,'twister')).
  Multi-seed robustness is assessed via `ossd_msr.m`; full multi-seed validation
  is identified as future work.

---

## Citation

If you use this implementation in academic work, please cite:

> F. Latifoğlu, L. Latifoğlu,  
> "Orthogonal Self-Similarity Decomposition (OSSD): A Delay-Based Framework  
> for Multi-Scale Time Series Analysis with Hydrological Forecasting Application",  
> *Fractal and Fractional*, 2026.

(DOI and volume/issue will be added in the camera-ready version.)

---

## License

Released under the **MIT License** (see `LICENSE`).

Copyright © 2026 Fatma Latifoğlu (Department of Biomedical Engineering)
and Levent Latifoğlu (Department of Civil Engineering),
Erciyes University, Kayseri, Türkiye.
