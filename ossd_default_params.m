function p = ossd_default_params(N, fs, user)
%OSSD_DEFAULT_PARAMS  Default configuration for OSSD-Basic.
%   Part of OSSD-Basic by F. Latifoğlu and L. Latifoğlu (Erciyes
%   University).  See ossd_decompose.m for full author / citation
%   information; LICENSE in package root for terms of use.
%
%   p = ossd_default_params(N)
%   p = ossd_default_params(N, fs)
%   p = ossd_default_params(N, fs, user)
%
%   The user-facing API exposes only five fields:
%
%       p.fs                     sampling rate (data property; 0 if unknown)
%       p.band_thresh_k          MAD multiplier for band extraction  [default 1.5]
%       p.rank_energy_target     per-band cumulative energy target   [default 0.93]
%       p.min_mode_energy_ratio  energy pruning floor                [default 1e-3]
%       p.n_modes                output K cap; [] = automatic        [default []]
%
%   Six geometric parameters are derived deterministically from the series
%   length N (window L, lag horizon, band area, band count, per-band rank,
%   refinement iterations).  All remaining algorithmic constants are fixed
%   design constants of the method and are not exposed for tuning.
%
%   The optional third argument is a struct of user overrides that takes
%   precedence over both the defaults and the auto-derived values.

if nargin < 1 || isempty(N) || ~isnumeric(N) || ~isscalar(N) || N < 8
    error('ossd_default_params:N', 'N must be a scalar integer >= 8.');
end
N = double(N);

if nargin < 2 || isempty(fs)
    fs = 0;
end

if nargin < 3 || isempty(user)
    user = struct();
end

% ---------------------------------------------------------------------------
% (1) Five user-facing parameters (with their defaults).
% ---------------------------------------------------------------------------
p = struct();
p.fs                    = fs;
p.band_thresh_k         = 1.5;        % MAD multiplier; primary tuning knob (typical range 1.2-2.0)
p.rank_energy_target    = 0.93;       % per-band cumulative energy target
p.min_mode_energy_ratio = 1e-3;       % drop modes with relative energy below this
p.n_modes               = [];         % output K cap; [] = automatic

% ---------------------------------------------------------------------------
% (2) Geometric parameters.
%     L is fixed at 128 by default (calibrated by L-sweep on the
%     synthetic benchmark to balance c1 / c_burst / c3 recovery).
%     For unusually short signals (N <= 200) we fall back to a small
%     auto-derived L so the trajectory matrix is well-conditioned.
%     User can override any of these via `params`.
% ---------------------------------------------------------------------------
if N >= 200
    p.L = 128;
else
    p.L = max(16, min(round(0.12 * N), 96));
end
p.L                  = min(p.L, max(8, N - 6));
p.max_tau            = max(20, min(120, round(N / 3)));
p.min_band_area      = max(8, round(N / 40));
p.min_time_span      = 6;
p.min_tau_span       = 4;
p.min_windows_per_band = 8;
p.n_bands            = max(3, min(5, round(N / 60)));
p.max_rank           = max(3, min(6, round(N / 80)));
p.joint_refine_iters = max(4, min(12, round(N / 40)));

% ---------------------------------------------------------------------------
% (3) Internal algorithmic constants (NOT user-tunable).
%     Reported in the appendix as design constants of the method.
% ---------------------------------------------------------------------------

% Band merging (adaptive proxy-correlation, OSSD novelty):
%   The merge threshold is interpolated between cmin_low and cmin_high
%   according to the tau-axis overlap of the two candidate bands.  Bands
%   sharing the same delay structure merge under cmin_low; bands on
%   different delay axes require near-identical correlation cmin_high.
%
%   Defaults below were calibrated by grid search on the synthetic
%   benchmark to maximise the worst-mode recovery across {c1, c_burst,
%   c3}, which is the primary target metric for source separation.
%   Earlier (looser) defaults of (0.55, 0.95, 0.28) over-merged the
%   weaker components into the dominant carriers; the calibrated
%   defaults preserve weak-component identity at a small reconstruction-
%   error cost.
p.const_band_proxy_corr_min   = 0.75;     % full overlap -> moderate merge
p.const_band_proxy_corr_max   = 0.95;     % no overlap   -> hard merge
p.const_band_tau_overlap_min  = 0.40;

% Band-subspace rank selection:
p.const_rank_use_scree_knee   = true;
p.const_rank_mdl_penalty      = 3.2;
p.const_rank_energy_slack     = 0.04;
p.const_rank_burst_target     = 0.82;
p.const_rank_burst_slack      = 0.02;
p.const_max_rank_harmonic     = 8;
p.const_max_rank_burst        = 3;
p.const_small_band_threshold  = 48;
p.const_small_band_max_rank   = 3;
p.const_harmonic_aspect_min   = 4.5;

% Joint orthogonalization:
p.const_orth_method  = 'sequential';   % 'sequential' or 'qr_block'
p.const_orth_tol     = 1e-9;

% OSSD outer pipeline — similarity merge:
p.const_sim_metric           = 'mean';   % 'envelope' | 'acf' | 'mean'
p.const_sim_acf_max_lag      = 80;
p.const_sim_merge_w_energy   = 0.35;
p.const_sim_merge_w_sim      = 1.00;
p.const_sim_merge_w_overlap  = 0.45;
p.const_sim_merge_lambda_corr= 0.20;
p.const_sim_merge_min_score  = 0.50;     % was 0.28; tighter to protect distinct components
p.const_sim_merge_min_sim    = 0.65;     % was 0.38; tighter to protect distinct components

% K-reduction (similarity-aware mode count reduction during n_modes trim):
%   When more modes are present than K_target, the reduction loop merges
%   the most similar pair if their similarity >= this threshold; otherwise
%   it drops the lowest-energy mode.  This collapses split fragments of
%   the same physical source rather than discarding low-energy modes that
%   carry meaningful structure.
p.const_kreduce_merge_min    = 0.40;

% OSSD-DR (Dominant-component Removed) cascade:
%   When the largest mode of pass 1 carries an energy fraction at or
%   above this threshold, OSSD subtracts that mode from x and runs a
%   second pass on the residual.  This rescues weaker self-similar
%   components that the first pass suppressed because the dominant
%   harmonic-rich component spread its signature across every band.
%   Cascade can be disabled by passing params.enable_cascade = false.
p.const_dominant_ratio_for_cascade = 0.70;

% Joint refine (V2-kernel):
p.const_joint_beta_cap   = 1.55;
p.const_joint_final_ls   = true;
p.const_joint_nonneg_ls  = false;

% ---------------------------------------------------------------------------
% (4) Apply user overrides.  Any field the caller sets in `user` overwrites
%     the corresponding default or auto-derived value.
% ---------------------------------------------------------------------------
fn = fieldnames(user);
for i = 1:numel(fn)
    p.(fn{i}) = user.(fn{i});
end

% ---------------------------------------------------------------------------
% (5) Sanity checks on user-facing values.
% ---------------------------------------------------------------------------
if ~isnumeric(p.band_thresh_k) || ~isscalar(p.band_thresh_k) || p.band_thresh_k <= 0
    error('ossd_default_params:band_thresh_k', 'band_thresh_k must be a positive scalar.');
end
if ~isnumeric(p.rank_energy_target) || ~isscalar(p.rank_energy_target) || ...
        p.rank_energy_target <= 0 || p.rank_energy_target > 1
    error('ossd_default_params:rank_energy_target', ...
        'rank_energy_target must be in (0, 1].');
end
if ~isnumeric(p.min_mode_energy_ratio) || ~isscalar(p.min_mode_energy_ratio) || ...
        p.min_mode_energy_ratio < 0
    error('ossd_default_params:min_mode_energy_ratio', ...
        'min_mode_energy_ratio must be a non-negative scalar.');
end
if ~isempty(p.n_modes)
    if ~isnumeric(p.n_modes) || ~isscalar(p.n_modes) || ~isfinite(p.n_modes) || p.n_modes < 1
        error('ossd_default_params:n_modes', ...
            'n_modes must be empty (auto) or a positive integer.');
    end
    p.n_modes = round(p.n_modes);
end

end
