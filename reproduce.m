% REPRODUCE  Single-script reproduction of the headline results.
%
% Authors:
%   Fatma Latifoğlu  — Department of Biomedical Engineering,
%                      Erciyes University, Kayseri, Türkiye
%                      Contact: flatifoglu@erciyes.edu.tr
%   Levent Latifoğlu — Department of Civil Engineering,
%                      Erciyes University, Kayseri, Türkiye
%
% In loving memory of İbrahim Dirgenali.
%
% Reference:
%   F. Latifoğlu, L. Latifoğlu, "Orthogonal Self-Similarity Decomposition (OSSD):
%   A Delay-Based Framework for Multi-Scale Time Series Analysis with
%   Hydrological Forecasting Application",
%   submitted to Fractal and Fractional, 2026.
%
% Copyright (c) 2026 Fatma Latifoğlu, Levent Latifoğlu.
% Released under the MIT License (see LICENSE in the package root).
%
% --- Description -----------------------------------------------------------
% This script runs OSSD-Basic on the same synthetic signal used in the
% manuscript, then computes the auxiliary stability metric (MSR) over
% multiple noise resamples and prints a self-contained results table.
%
% Expected total runtime: ~5 seconds on a modern laptop.
%
% Usage:
%   >> reproduce
%
% The first part of this script mirrors demo_OSSD_Basic.m; the second
% part runs MSR (Mode Stability under Resampling) as a separate
% measurement.  The numbers below should match the manuscript's
% Table 4 to within numerical precision.

clear; close all; clc;

fprintf('\n================================================================\n');
fprintf('  OSSD-Basic — reproduction of the manuscript headline results\n');
fprintf('================================================================\n\n');

% ----------------------------------------------------------------------------
% (1) Synthetic signal — identical generator as demo_OSSD_Basic.
% ----------------------------------------------------------------------------
fs = 400;
T  = 8;
[t, x, gt] = local_generate_synthetic_signal(fs, T);
N = numel(x);
fprintf('Synthetic signal: N=%d samples, fs=%d Hz, T=%g s\n\n', N, fs, T);

% ----------------------------------------------------------------------------
% (2) Run OSSD-Basic with default parameters.
% ----------------------------------------------------------------------------
params = struct();
params.fs      = fs;
params.n_modes = 4;

t_start = tic;
out = ossd_decompose(x, params);
elapsed = toc(t_start);
fprintf('OSSD-Basic finished in %.3f s.\n', elapsed);

K = size(out.modes, 2);

% ----------------------------------------------------------------------------
% (3) Diagnostics + recovery numbers (Table 4 in the manuscript).
% ----------------------------------------------------------------------------
d = out.diagnostics;
xhat = sum(out.modes, 2);
rre = norm(x - xhat) / (norm(x) + eps);
clean_x = gt.c1 + gt.c2 + gt.c3 + gt.c4;
snr_observed = 20 * log10(norm(x - mean(x)) / (norm(x - xhat) + eps));
% Table 4 / Table 7 SNR: reconstruction error vs noise-free ground truth.
snr_clean = 20 * log10(norm(clean_x) / (norm(clean_x - xhat) + eps));

% Inter-mode correlation matrix.
Cmm = corr(out.modes);
Cmm(isnan(Cmm)) = 0;
upper_mask = triu(true(K), 1);
mean_corr = mean(abs(Cmm(upper_mask)));
max_corr  = max(abs(Cmm(upper_mask)));

% Mode-to-truth |corr| matrix.
truth_names = {'c1', 'c_burst', 'c3', 'noise'};
nT = numel(truth_names);
Cmt = zeros(K, nT);
for k = 1:K
    m = out.modes(:, k);  m = m - mean(m);  nm = norm(m) + eps;
    for it = 1:nT
        g = gt.(truth_names{it});  g = g - mean(g);  ng = norm(g) + eps;
        Cmt(k, it) = abs((g' * m) / (nm * ng));
    end
end

% Burst grouping: c2 + c4 may be split across two modes.
score_burst = Cmt(:, strcmp(truth_names, 'c_burst'));
[best_burst, k_b1] = max(score_burst);
score_excl = score_burst;
score_excl(k_b1) = -inf;
[second_burst, k_b2] = max(score_excl);
second_c3 = 0;
if k_b2 >= 1
    second_c3 = Cmt(k_b2, strcmp(truth_names, 'c3'));
end

cond_a = second_burst >= 0.15 && second_burst >= 0.20 * best_burst;
cond_b = second_burst >= second_c3 + 0.05;

if cond_a && cond_b
    burst_idx = sort([k_b1, k_b2]);
    m_burst   = out.modes(:, burst_idx(1)) + out.modes(:, burst_idx(2));
    burst_lbl = sprintf('mode %d + mode %d', burst_idx(1), burst_idx(2));
else
    burst_idx = k_b1;
    m_burst   = out.modes(:, k_b1);
    burst_lbl = sprintf('mode %d', k_b1);
end

g = gt.c_burst;  g = g - mean(g);  ng = norm(g) + eps;
mb = m_burst - mean(m_burst);  nmb = norm(mb) + eps;
c_burst = abs((g' * mb) / (nmb * ng));

% Best (non-burst) mode for c1, c3.
remaining = setdiff(1:K, burst_idx);
c1_best = max(Cmt(remaining, strcmp(truth_names, 'c1')));
c3_best = max(Cmt(remaining, strcmp(truth_names, 'c3')));

% ----------------------------------------------------------------------------
% (4) Self-similarity-based metrics.
% ----------------------------------------------------------------------------
ssp = ossd_sspi(x, out.modes, 'L', 80, 'max_tau', 120);
sspi_val = ssp.sspi;

if isfield(out, 'bands') && iscell(out.bands) && numel(out.bands) >= 2
    sdi_out = ossd_sdi(out.bands);
    sdi_val = sdi_out.sdi;
else
    sdi_val = NaN;
end

cci_val = NaN;
if isfield(out, 'cci')
    cci_val = out.cci;
elseif isfield(out, 'cascade_info') && ...
        isfield(out.cascade_info, 'dominant_ratio')
    cci_val = out.cascade_info.dominant_ratio;
end

% ----------------------------------------------------------------------------
% (5) MSR — Mode Stability under Resampling (5 noise seeds).
%
%   Re-generates the signal with five different noise realisations,
%   re-runs OSSD-Basic, and matches the resulting modes against the
%   reference modes via Hungarian assignment.  Mean MSR averages the
%   per-mode stability scores.
% ----------------------------------------------------------------------------
fprintf('\nComputing MSR over 5 noise resamples ...\n');
n_seeds = 5;
msr_scores = nan(n_seeds, K);

for s = 1:n_seeds
    rng('default');
    rng(100 + s, 'twister');
    noise_s = 0.07 * randn(N, 1);
    x_s = clean_x + noise_s;

    out_s = ossd_decompose(x_s, params);
    K_s = size(out_s.modes, 2);
    if K_s == 0, continue; end

    % Pairwise |corr| between reference modes and resample modes.
    M = min(K, K_s);
    A = zeros(K, K_s);
    for i = 1:K
        u = out.modes(:, i);  u = u - mean(u);  nu = norm(u) + eps;
        for j = 1:K_s
            v = out_s.modes(:, j);  v = v - mean(v);  nv = norm(v) + eps;
            A(i, j) = abs((u' * v) / (nu * nv));
        end
    end

    % Greedy assignment (Hungarian-equivalent for our purposes).
    assigned = false(K_s, 1);
    for i = 1:K
        [best_v, best_j] = max(A(i, :) .* (~assigned)');
        if best_v > 0
            msr_scores(s, i) = best_v;
            assigned(best_j) = true;
        end
    end
end

msr_per_mode = mean(msr_scores, 1, 'omitnan');
msr_mean = mean(msr_per_mode, 'omitnan');

% ----------------------------------------------------------------------------
% (6) Print headline table.
% ----------------------------------------------------------------------------
fprintf('\n================================================================\n');
fprintf('  Headline results (manuscript Table 4 + auxiliary metrics)\n');
fprintf('================================================================\n');
fprintf('  K (number of modes)              = %d\n', K);
fprintf('  Reconstruction relative error    = %.4f\n', rre);
fprintf('  SNR vs observed x (dB)           = %.2f\n', snr_observed);
fprintf('  SNR vs noise-free ground truth   = %.2f dB  (Table 4)\n', snr_clean);
fprintf('  Orthogonality index              = %.2e\n', d.orthogonality_index);
fprintf('  Mean inter-mode |corr|           = %.4e\n', mean_corr);
fprintf('  Max  inter-mode |corr|           = %.4e\n', max_corr);
fprintf('  c1     recovery (abs corr)       = %.4f\n', c1_best);
fprintf('  c_burst recovery (c2+c4)         = %.4f  (%s)\n', c_burst, burst_lbl);
fprintf('  c3     recovery (abs corr)       = %.4f\n', c3_best);
fprintf('  ----------------------------------------------\n');
fprintf('  CCI (Cascade Coverage Index)     = %.4f\n', cci_val);
fprintf('  SSPI (Self-Similarity Pres.)     = %.4f\n', sspi_val);
fprintf('  SDI  (Snowflake Disjointness)    = %.4f\n', sdi_val);
fprintf('  MSR  mean (5 noise seeds)        = %.4f\n', msr_mean);
fprintf('       per mode = [%s]\n', ...
    strjoin(arrayfun(@(v) sprintf('%.3f', v), msr_per_mode, ...
                     'UniformOutput', false), ', '));
fprintf('================================================================\n\n');

fprintf('Figures: none generated by reproduce.m.  Run demo_OSSD_Basic to\n');
fprintf('         see the synthetic-truth, modes, and reconstruction figures.\n\n');

% ============================================================================
% Local helpers
% ============================================================================

function [t, x, gt] = local_generate_synthetic_signal(fs, T)
% Generates a four-component test signal whose components have distinct
% delay-domain signatures.  The RNG seed is fixed inside this helper so
% that successive calls produce the same x (modulo MATLAB version).

t = (0:1/fs:T - 1/fs)';
N = numel(t);

% --- c1: Strong self-similar harmonic stack -------------------------------
f0 = 3.15;
c1 = cos(2 * pi * f0 * t) ...
   + 0.40 * cos(2 * pi * 2 * f0 * t + 0.65) ...
   + 0.14 * cos(2 * pi * 3 * f0 * t + 1.05);
c1 = c1 .* (1 + 0.06 * sin(2 * pi * 0.11 * t));

% --- c2: Self-similar repeated bursts (same template, three centres) -----
fc_b    = 17.5;
sigma_b = 0.11;
centers = [1.05, 3.45, 5.95];
c2 = zeros(N, 1);
for tc = centers
    dt = t - tc;
    c2 = c2 + exp(-0.5 * (dt / sigma_b).^2) .* cos(2 * pi * fc_b * dt);
end
c2 = 0.52 * c2;

% --- c3: Independent harmonic family (different tau structure) ----------
f3 = 7.8;
c3 = 0.36 * (cos(2 * pi * f3 * t) + 0.28 * cos(2 * pi * 2 * f3 * t + 0.4));

% --- c4: Second repeated motif (shorter-period packets) -----------------
fc4  = 22;
sig4 = 0.065;
c4   = zeros(N, 1);
for tc = [2.55, 4.35, 7.1]
    dt = t - tc;
    c4 = c4 + exp(-0.5 * (dt / sig4).^2) .* cos(2 * pi * fc4 * dt + 0.3);
end
c4 = 0.30 * c4;

% --- Noise --------------------------------------------------------------
rng('default');
rng(7, 'twister');
noise = 0.07 * randn(N, 1);

x = c1 + c2 + c3 + c4 + noise;

gt.c1      = c1;
gt.c2      = c2;
gt.c3      = c3;
gt.c4      = c4;
gt.c_burst = c2 + c4;
gt.noise   = noise;
end
