function out = ossd_decompose(x, params)
%OSSD_DECOMPOSE  Orthogonal Self-Similarity Decomposition (OSSD-Basic).
%
%   Authors:
%     Fatma Latifoğlu  — Department of Biomedical Engineering,
%                        Erciyes University, Kayseri, Türkiye
%                        Contact: flatifoglu@erciyes.edu.tr
%     Levent Latifoğlu — Department of Civil Engineering,
%                        Erciyes University, Kayseri, Türkiye
%
%   In loving memory of İbrahim Dirgenali.
%
%   References:
%     [1] F. Latifoğlu, L. Latifoğlu, "Orthogonal Self-Similarity Decomposition (OSSD):
%         A Delay-Based Framework for Multi-Scale Time Series Analysis with
%         Hydrological Forecasting Application",
%         submitted to Fractal and Fractional, 2026.
%
%   Copyright (c) 2026 Fatma Latifoğlu, Levent Latifoğlu.
%   Released under the MIT License (see LICENSE in the package root).
%
%   --- Synopsis ---------------------------------------------------------
%   out = ossd_decompose(x)              uses default parameters.
%   out = ossd_decompose(x, params)      overrides selected fields.
%
%   The user-facing parameter struct PARAMS exposes only five fields
%   (see ossd_default_params for documentation):
%
%       params.fs                       sampling rate (data property)
%       params.band_thresh_k            MAD multiplier        [default 1.5]
%       params.rank_energy_target       per-band energy target [default 0.93]
%       params.min_mode_energy_ratio    energy pruning floor   [default 1e-3]
%       params.n_modes                  output K cap; [] = auto [default []]
%
%   All remaining geometric parameters are derived from N = numel(x); all
%   other algorithmic constants are fixed design constants of the method.
%
%   The OSSD-Basic pipeline executes eight stages in order:
%
%       Stage 1 — trajectory matrix and similarity plane (band pipeline)
%       Stage 2 — robust thresholding and band extraction (band pipeline)
%       Stage 3 — band merging by proxy correlation + tau overlap (band pipeline)
%       Stage 4 — band-wise subspace modelling (band pipeline)
%       Stage 5 — joint orthogonalization across band bases (band pipeline)
%       Stage 6 — diagonal averaging to obtain initial modes (band pipeline)
%       Stage 7 — OSSD outer pipeline (similarity merge → MGS+LS → joint refine)
%       Stage 8 — energy pruning and output mode selection
%
%   Output (struct):
%
%       out.modes        N x K matrix of decomposed modes (truth-blind)
%       out.residual     N x 1 residual r = x - sum(out.modes, 2)
%       out.bands        descriptive band cell (or {} if discarded)
%       out.S, out.Srob  raw and robust similarity planes
%       out.tau_axis     tau values along the similarity-plane second axis
%       out.summary      per-mode summary table
%       out.params_used  the fully resolved parameter struct
%       out.diagnostics  RRE, OI, mean IF, etc.

% --------------------------------------------------------------------------
% (0) Resolve parameters and bring helpers onto path.
% --------------------------------------------------------------------------
if nargin < 2 || isempty(params)
    params = struct();
end

x = x(:);
N = numel(x);

fs = 0;
if isfield(params, 'fs') && ~isempty(params.fs)
    fs = double(params.fs);
end

% Normalize the user struct: ossd_default_params will apply overrides.
p = ossd_default_params(N, fs, params);

% Make sure the OSSD_Basic folder is on the search path so that the
% pipeline files (ossd_band_pipeline, ossd_outer_pipeline, ossd_diagnostics,
% ossd_similarity_metrics, ossd_joint_refine) are visible.
here = fileparts(mfilename('fullpath'));
if exist(here, 'dir')
    addpath(here);
end

% --------------------------------------------------------------------------
% Pass 1 — single OSSD pass on the original signal.
% --------------------------------------------------------------------------
pass1 = local_run_single_pass(x, p);

% --------------------------------------------------------------------------
% OSSD-DR cascade decision.
%
%   When one mode carries an overwhelming fraction of the total energy,
%   its delay-domain footprint dominates every band and pushes weaker
%   self-similar components (here: c3-like harmonic families) into the
%   subspace residuals.  The cascade extracts the dominant mode, removes
%   it from x, and re-runs OSSD on the residual where the previously
%   suppressed components rise to detectable energy.
%
%   Triggered automatically when:
%     - cascade is enabled (default true)
%     - the largest mode's energy fraction exceeds the threshold
%     - more than one band exists in the original pass
%   The user can disable it via params.enable_cascade = false.
% --------------------------------------------------------------------------
cascade_enabled = true;
if isfield(params, 'enable_cascade') && ~isempty(params.enable_cascade)
    cascade_enabled = logical(params.enable_cascade);
end

modes      = pass1.modes;
outer_info = pass1.outer_info;
n_modes_info = pass1.n_modes_info;
band_meta  = pass1.band_meta;
bands      = pass1.bands;
S          = pass1.S;
Srob       = pass1.Srob;
tau_axis   = pass1.tau_axis;

% --------------------------------------------------------------------------
% Cascade Coverage Index (CCI).
%
%   CCI quantifies how strongly the input's energy is concentrated in
%   the single largest mode that the band pipeline produced:
%
%       CCI(x) = E(strongest mode) / E(x)
%
%   It is a signal-characterisation metric, not a tuning knob.  CCI is
%   computed AFTER pass 1 regardless of whether the cascade triggers,
%   so downstream code (and users) can read out one number describing
%   the input's energy concentration.  The cascade itself fires when
%   CCI exceeds a threshold (const_dominant_ratio_for_cascade); reading
%   CCI lets users see exactly how that decision was made.
% --------------------------------------------------------------------------
cci = local_compute_cci(modes, x);

cascade_info = struct('enabled', cascade_enabled, ...
                      'triggered', false, ...
                      'dominant_ratio', cci, ...
                      'dominant_mode_index', NaN, ...
                      'pass2_n_modes', 0);

if cascade_enabled && size(modes, 2) >= 1 && numel(bands) >= 2
    Em   = sum(modes .^ 2, 1);
    [~, k_dom] = max(Em);
    dom_ratio  = cci;        % same quantity, retained for legibility below

    cascade_info.dominant_mode_index = k_dom;

    if dom_ratio >= p.const_dominant_ratio_for_cascade
        cascade_info.triggered = true;

        % Extract the dominant mode and form the cascade residual.
        m_dom = modes(:, k_dom);
        x_resid = x - m_dom;

        % Pass 2 runs on the residual.  We deliberately let Pass 2 emit
        % ALL modes it finds (n_modes = [] = auto) rather than imposing
        % a K cap — the K cap is enforced once at the FINAL stage on the
        % combined [m_dom, pass-2] set so that weak rescued components
        % see the dominant mode in the comparison.  Imposing the cap
        % inside Pass 2 already prunes away exactly those weak rescued
        % components we want to surface.
        p2 = p;
        p2.n_modes = [];   % unconstrained inside Pass 2
        % Also relax the energy floor so weak rescued modes survive
        % Pass 2's own pruning step.
        p2.min_mode_energy_ratio = min(p.min_mode_energy_ratio, 1e-5);

        pass2 = local_run_single_pass(x_resid, p2);
        modes_pass2 = pass2.modes;
        cascade_info.pass2_n_modes = size(modes_pass2, 2);

        % Combine: dominant first, then pass-2 modes.
        modes = [m_dom, modes_pass2];

        % --- Joint LS amplitude refit against the ORIGINAL x ----------
        % After cascade, mode amplitudes were estimated on two different
        % targets (x for Pass 1, x_resid for Pass 2).  A single
        % least-squares refit against the original x recovers any energy
        % the dominant mode had absorbed from weaker components and
        % redistributes it correctly.  Coefficients are clipped to a
        % sensible non-negative range to avoid sign flips.
        if size(modes, 2) >= 1
            beta = pinv(modes) * x;
            beta = max(beta, 0);
            beta = min(beta, 2.5);
            for k = 1:size(modes, 2)
                modes(:, k) = beta(k) * modes(:, k);
            end
        end

        % Now apply the user n_modes cap with similarity-aware reduction.
        % Use a stricter merge threshold post-cascade: weak components
        % rescued from the residual should not be merged into burst
        % fragments unless they are clearly the same source.
        if ~isempty(p.n_modes) && size(modes, 2) > p.n_modes
            p_red = p;
            p_red.const_kreduce_merge_min = max(p.const_kreduce_merge_min, 0.50);
            [modes, red_info] = local_similarity_aware_reduce(modes, x, ...
                p.n_modes, p_red);
            n_modes_info.merges = n_modes_info.merges + red_info.merges;
            n_modes_info.prunes = n_modes_info.prunes + red_info.prunes;
            n_modes_info.kept   = size(modes, 2);
            n_modes_info.trimmed = true;
        end

        % Re-sort by energy.
        [~, ix2] = sort(sum(modes .^ 2, 1), 'descend');
        modes = modes(:, ix2);

        % Final re-orthogonalisation after cascade-induced LS refit and
        % similarity-aware reduction.  Both of those steps may have
        % re-introduced small inter-mode correlations even though the
        % outer pipeline (in Pass 1 / Pass 2) had produced an
        % orthogonal column set.  A single MGS + LS pass projects the
        % refined modes back onto an orthogonal basis and re-fits
        % amplitudes against x.
        if size(modes, 2) >= 2
            [modes, ~] = local_gram_schmidt_ls_inline(x, modes, p);
        end
    end
end

% --------------------------------------------------------------------------
% Diagnostics and per-mode summary.
% --------------------------------------------------------------------------
residual = x - sum(modes, 2);
diagnostics = ossd_diagnostics(x, modes, residual, fs);

nM = size(modes, 2);
if nM == 0
    summary = table();
else
    E = sum(modes .^ 2, 1).';
    if fs > 0
        [fmed, ifvar] = ossd_diagnostics(x, modes, residual, fs, 'per_mode_only');
        summary = table((1:nM).', E, fmed(:), ifvar(:), ...
            'VariableNames', {'Mode', 'Energy', 'MedianInstFreq_Hz', 'InstFreqVar_Hz2'});
    else
        summary = table((1:nM).', E, ...
            'VariableNames', {'Mode', 'Energy'});
    end
end

% --------------------------------------------------------------------------
% Assemble output.
% --------------------------------------------------------------------------
out = struct();
out.modes        = modes;
out.residual     = residual;
out.bands        = bands;
out.band_meta    = band_meta;
out.S            = S;
out.Srob         = Srob;
out.tau_axis     = tau_axis;
out.summary      = summary;
out.params_used  = p;
out.outer_info   = outer_info;
out.n_modes_info = n_modes_info;
out.cascade_info = cascade_info;
out.cci          = cci;
out.diagnostics  = diagnostics;
out.note         = ['OSSD-Basic with optional dominant-component cascade: ' ...
                    'trajectory & self-similarity plane, band extraction (MAD), ' ...
                    'adaptive proxy-correlation band merge, ' ...
                    'rank-controlled subspace modelling, joint orthogonalization, ' ...
                    'similarity merge, MGS+LS amplitude, joint refine, ' ...
                    'similarity-aware K reduction, dominant-component cascade, ' ...
                    'truth-blind throughout.'];

end

% ==========================================================================
% Local helpers
% ==========================================================================

function cci = local_compute_cci(modes, x)
%LOCAL_COMPUTE_CCI  Cascade Coverage Index of a decomposition.
%
%   CCI = E(strongest mode) / E(x).
%
%   A scalar in [0, 1] (modulo numerical noise — clipped here).
%   CCI -> 1 means the input is dominated by a single self-similar
%   component; CCI -> 0 means energy is spread across many components.
%   The OSSD-DR cascade fires when CCI is high enough that a residual
%   pass would expose components otherwise hidden under the dominant
%   one.

if isempty(modes) || size(modes, 2) == 0
    cci = 0;
    return;
end

Etot = sum(x(:) .^ 2) + eps;
Em   = sum(modes .^ 2, 1);
cci  = max(Em) / Etot;
cci  = max(0, min(cci, 1));     % clip against numerical noise
end

function pass = local_run_single_pass(x, p)
%LOCAL_RUN_SINGLE_PASS  One OSSD-Basic pass (Stages 1-8 of the original).
%
%   Returns a struct with the same fields the outer function used to
%   populate directly: modes, residual, bands, band_meta, S, Srob,
%   tau_axis, outer_info, n_modes_info.

x = x(:);
N = numel(x);

% Stages 1–6: band pipeline.
band_out = ossd_band_pipeline(x, p);
modes      = band_out.modes;
S          = band_out.S;
Srob       = band_out.Srob;
tau_axis   = band_out.tau_axis;
bands      = band_out.bands;
band_meta  = band_out.band_meta;

% Stage 7: outer pipeline.
outer_info = struct('similarity_merges', 0, ...
                    'gs_cols_in', size(modes, 2), ...
                    'gs_cols_out', size(modes, 2), ...
                    'joint_refine_applied', false);
if ~isempty(modes)
    [modes, outer_info] = ossd_outer_pipeline(x, modes, p);
end

% Stage 8: pruning + similarity-aware K reduction.
modes = local_prune_by_energy(modes, x, p.min_mode_energy_ratio);

n_modes_info = struct('requested', NaN, 'kept', size(modes, 2), ...
                      'trimmed', false, 'merges', 0, 'prunes', 0);
if ~isempty(p.n_modes) && size(modes, 2) > p.n_modes
    n_modes_info.requested = p.n_modes;
    [modes, red_info] = local_similarity_aware_reduce(modes, x, ...
        p.n_modes, p);
    n_modes_info.kept    = size(modes, 2);
    n_modes_info.trimmed = true;
    n_modes_info.merges  = red_info.merges;
    n_modes_info.prunes  = red_info.prunes;
    [~, ix2] = sort(sum(modes .^ 2, 1), 'descend');
    modes = modes(:, ix2);
end

% Final re-orthogonalisation after K-reduction.
%   The similarity-aware reduce step above may merge modes (energy-
%   weighted average) and thereby reintroduce small cross-talk that
%   the outer pipeline's MGS+LS had eliminated.  A final MGS+LS pass
%   restores numerical orthogonality (OI -> 0, max |corr| -> 0)
%   without changing recovery numbers materially.
if n_modes_info.trimmed && size(modes, 2) >= 2
    [modes, ~] = local_gram_schmidt_ls_inline(x, modes, p);
end

if outer_info.similarity_merges > 0 || outer_info.joint_refine_applied || ...
        n_modes_info.trimmed
    band_meta = struct([]);
end

pass = struct();
pass.modes        = modes;
pass.bands        = bands;
pass.band_meta    = band_meta;
pass.S            = S;
pass.Srob         = Srob;
pass.tau_axis     = tau_axis;
pass.outer_info   = outer_info;
pass.n_modes_info = n_modes_info;
end

function modes = local_prune_by_energy(modes, x, thr)
% Drop columns whose energy / ||x||^2 is below thr.
if isempty(modes)
    return;
end
Etot = sum(x .^ 2) + eps;
keep = sum(modes .^ 2, 1) / Etot >= thr;
modes = modes(:, keep);
end

function [modes, info] = local_similarity_aware_reduce(modes, x, K_target, p)
%LOCAL_SIMILARITY_AWARE_REDUCE  K reduction with mode merging (OSSD novelty).
%
%   Reduces the column count of MODES from K_current to K_target.  Each
%   reduction step does ONE of:
%
%     (a) MERGE  — if the most similar pair of remaining modes has
%                  similarity >= p.const_kreduce_merge_min, sum the two
%                  columns into one (energy-weighted to preserve scale).
%                  This collapses split fragments of the same physical
%                  source into a single mode.
%
%     (b) PRUNE  — otherwise drop the lowest-energy column outright.
%                  This is the classical energy-only fallback used when
%                  no two columns are similar enough to merge.
%
%   Looping (a)/(b) until K = K_target gives a similarity-aware reduction
%   that distinguishes "two fragments of the same source" from "one weak
%   spurious mode", which a pure energy threshold cannot.
%
%   Reads:
%     p.const_kreduce_merge_min   minimum similarity to merge (default 0.40)
%
%   Returns the reduced MODES and an info struct with merge / prune counts.

info = struct('merges', 0, 'prunes', 0);
if isempty(modes) || K_target <= 0
    modes = zeros(size(modes, 1), 0);
    return;
end

merge_thr = 0.40;
if isfield(p, 'const_kreduce_merge_min') && ~isempty(p.const_kreduce_merge_min)
    merge_thr = p.const_kreduce_merge_min;
end

x = x(:);

while size(modes, 2) > K_target
    K = size(modes, 2);
    if K == 1
        break;
    end

    % Pairwise similarity matrix (envelope+ACF mean).
    S = ossd_similarity_metrics(modes, 'matrix', p);

    % Find the most similar off-diagonal pair.
    S_off = S - eye(K);                  % zero out diagonal
    [s_max, lin] = max(S_off(:));
    [i, j] = ind2sub([K, K], lin);
    if i > j, [i, j] = deal(j, i); end

    if s_max >= merge_thr
        % Energy-weighted sum (preserves total energy scale).
        Ei = sum(modes(:, i) .^ 2);
        Ej = sum(modes(:, j) .^ 2);
        w = Ei / (Ei + Ej + eps);
        m_new = w * modes(:, i) + (1 - w) * modes(:, j);

        % Re-fit amplitude against x (single-mode LS).
        denom = m_new' * m_new + eps;
        beta = (m_new' * (x - sum(modes, 2) + modes(:, i) + modes(:, j))) / denom;
        beta = max(0, min(beta, 2.0));     % clip to a sane range
        m_new = beta * m_new;

        modes(:, i) = m_new;
        modes(:, j) = [];
        info.merges = info.merges + 1;
    else
        % No similar pair — drop the lowest-energy column.
        E = sum(modes .^ 2, 1);
        [~, k_drop] = min(E);
        modes(:, k_drop) = [];
        info.prunes = info.prunes + 1;
    end
end
end



function [modes_out, info] = local_gram_schmidt_ls_inline(x, modes, p)
%LOCAL_GRAM_SCHMIDT_LS_INLINE  Inline MGS + LS for post-K-reduction
%re-orthogonalisation.
%
%   Local copy of the MGS + LS step used by the outer pipeline so that
%   ossd_decompose can re-orthogonalise after K-reduction without an
%   extra cross-file dependency.

x = x(:);
modes = modes(:, :);
[N, K] = size(modes);
info = struct('cols_in', K, 'cols_out', 0, 'beta', []);
modes_out = zeros(N, 0);
if K == 0, return; end

tol = 1e-9;
if isfield(p, 'const_orth_tol') && ~isempty(p.const_orth_tol)
    tol = p.const_orth_tol;
end

Q = zeros(N, K);
for j = 1:K
    v = modes(:, j);
    for r = 1:j - 1
        qr0 = Q(:, r);
        denom = qr0' * qr0 + eps;
        v = v - (qr0' * v) / denom * qr0;
    end
    nv = norm(v);
    if nv > tol * (norm(modes(:, j)) + eps)
        Q(:, j) = v / nv;
    else
        Q(:, j) = 0;
    end
end

keep = vecnorm(Q) > tol;
Q = Q(:, keep);
if isempty(Q)
    return;
end

beta = Q' * x;
modes_out = bsxfun(@times, Q, beta(:).');
info.cols_out = size(modes_out, 2);
info.beta = beta;
end
