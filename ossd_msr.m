function out = ossd_msr(x, params, varargin)
%OSSD_MSR  Mode Stability under Resampling — robustness benchmark for OSSD.
%   Part of OSSD-Basic by F. Latifoğlu and L. Latifoğlu (Erciyes
%   University).  See ossd_decompose.m for full author / citation
%   information; LICENSE in package root for terms of use.
%
%   out = ossd_msr(x, params)
%   out = ossd_msr(x, params, 'n_seeds', 30, 'noise_sigma', 0.05, ...)
%
%   Quantifies how stable an OSSD-Basic decomposition is to small
%   stochastic perturbations of the input.  For each of N_SEEDS
%   independent noise realisations
%
%       x_s = x + sigma * std(x) * randn(N, 1)
%
%   the function calls ossd_decompose(x_s, params), then aligns the
%   resulting modes to the modes obtained on a REFERENCE decomposition
%   of x itself (using a Hungarian-style permutation that maximises the
%   total |Pearson| over the assignment).
%
%   For each reference mode k it then reports:
%
%       MSR(k) = mean over seeds of |Pearson(reference_mode_k, matched_seed_mode_k)|
%
%   MSR(k) lies in [0, 1].  A value close to 1 means mode k reappears
%   with essentially the same waveform under noise perturbation
%   (= robust); values closer to 0 indicate the column is unstable
%   under resampling and should be interpreted with caution.
%
%   This is the central quantitative argument for reproducibility on
%   single-realisation tests: rather than asking whether the algorithm
%   recovers a given ground-truth component, it asks whether the
%   algorithm's own output is itself stable.
%
%   --- Inputs ---
%     x       N x 1 input signal.
%     params  parameter struct passed verbatim to ossd_decompose; the
%             same struct is used for the reference and all seeded calls.
%     ...     name-value options:
%               'n_seeds'      (default 30)  number of resampled trials.
%               'noise_sigma'  (default 0.05) noise level relative to
%                              std(x); each trial adds Gaussian noise of
%                              this RELATIVE standard deviation.
%               'seed_offset'  (default 1000) base offset for rng seeds
%                              so multiple MSR runs are reproducible
%                              and decoupled from the user's seed state.
%
%   --- Outputs (struct) ---
%     out.K               number of reference modes.
%     out.n_seeds         number of resampling trials used.
%     out.noise_sigma     relative noise level used.
%     out.msr             K x 1 — MSR per reference mode.
%     out.msr_mean        scalar — mean of out.msr.
%     out.msr_min         scalar — min of out.msr (worst mode).
%     out.recovery_table  K x N_SEEDS — per-seed |Pearson| of the
%                         best-matched mode column for each reference k.
%     out.cci             reference decomposition's CCI.
%     out.params_used     parameter struct that was applied.
%
%   --- Example ---
%     out = ossd_msr(x, struct('fs', 400, 'n_modes', 3), 'n_seeds', 50);
%     fprintf('Mean MSR over %d seeds: %.3f (worst mode: %.3f)\n', ...
%         out.n_seeds, out.msr_mean, out.msr_min);

% --------------------------------------------------------------------------
% Parse options.
% --------------------------------------------------------------------------
opt = struct('n_seeds', 30, 'noise_sigma', 0.05, 'seed_offset', 1000);
for i = 1:2:numel(varargin)
    name = lower(varargin{i});
    if isfield(opt, name)
        opt.(name) = varargin{i + 1};
    end
end

x = x(:);
N = numel(x);
sx = std(x);
if sx == 0
    sx = 1;
end

% --------------------------------------------------------------------------
% Reference decomposition on the un-perturbed signal.
% --------------------------------------------------------------------------
out_ref = ossd_decompose(x, params);
modes_ref = out_ref.modes;
K = size(modes_ref, 2);

if K == 0
    out = struct('K', 0, 'n_seeds', opt.n_seeds, ...
                 'noise_sigma', opt.noise_sigma, ...
                 'msr', [], 'msr_mean', NaN, 'msr_min', NaN, ...
                 'recovery_table', [], ...
                 'cci', getfield_safe(out_ref, 'cci', NaN), ...
                 'params_used', getfield_safe(out_ref, 'params_used', struct()));
    return;
end

% Pre-compute centred / normalised reference columns for fast cosine.
ref_centred = zeros(N, K);
ref_norm    = zeros(K, 1);
for k = 1:K
    v = modes_ref(:, k) - mean(modes_ref(:, k));
    ref_centred(:, k) = v;
    ref_norm(k) = norm(v) + eps;
end

% --------------------------------------------------------------------------
% Resampling loop.
% --------------------------------------------------------------------------
recovery = zeros(K, opt.n_seeds);
for s = 1:opt.n_seeds
    % Deterministic, decoupled per-seed RNG.
    rng(opt.seed_offset + s, 'twister');
    eps_s = opt.noise_sigma * sx * randn(N, 1);
    x_s = x + eps_s;

    out_s = ossd_decompose(x_s, params);
    modes_s = out_s.modes;
    Ks = size(modes_s, 2);

    if Ks == 0
        % Degenerate seed — record zero similarity for all reference
        % modes.  This penalises configurations that occasionally fail.
        recovery(:, s) = 0;
        continue;
    end

    % Build K x Ks |Pearson| cost matrix between reference and seed modes.
    C = zeros(K, Ks);
    for k = 1:K
        for j = 1:Ks
            v = modes_s(:, j) - mean(modes_s(:, j));
            nv = norm(v) + eps;
            C(k, j) = abs((ref_centred(:, k)' * v) / (ref_norm(k) * nv));
        end
    end

    % Assign each reference column to the best available seed column,
    % maximising the total |Pearson|.  Enumerate permutations for small
    % matrices (K, Ks <= 6 here typically).
    rec = local_optimal_assign_max(C);
    recovery(:, s) = rec;
end

% --------------------------------------------------------------------------
% Aggregate.
% --------------------------------------------------------------------------
msr = mean(recovery, 2);
out = struct();
out.K              = K;
out.n_seeds        = opt.n_seeds;
out.noise_sigma    = opt.noise_sigma;
out.msr            = msr;
out.msr_mean       = mean(msr);
out.msr_min        = min(msr);
out.recovery_table = recovery;
out.cci            = getfield_safe(out_ref, 'cci', NaN);
out.params_used    = getfield_safe(out_ref, 'params_used', struct());
end

% ==========================================================================
% Helpers
% ==========================================================================

function rec = local_optimal_assign_max(C)
% Return the per-row assignment cost (here: |Pearson|) of the
% permutation that maximises the total cost.  Rows can outnumber
% columns; rows that cannot be matched receive zero.
[K, Ks] = size(C);
rec = zeros(K, 1);
if K == 0 || Ks == 0
    return;
end

if K <= Ks
    % Try every K-subset assignment of column indices.  Brute force
    % is fine for K <= 6 (real OSSD output K is small).
    if K <= 6
        cols = 1:Ks;
        best_score = -inf;
        best_pick  = 1:K;
        % Enumerate permutations of K columns drawn from Ks.
        % For tractability we generate all length-K permutations.
        Pset = perms(cols);
        Pset = Pset(:, 1:K);
        Pset = unique(Pset, 'rows');
        for r = 1:size(Pset, 1)
            p = Pset(r, :);
            s = 0;
            for k = 1:K
                s = s + C(k, p(k));
            end
            if s > best_score
                best_score = s;
                best_pick  = p;
            end
        end
        for k = 1:K
            rec(k) = C(k, best_pick(k));
        end
    else
        % Fall back to greedy when K is unusually large.
        rec = local_greedy_assign(C);
    end
else
    % More reference columns than seed columns — at most Ks reference
    % rows can be matched; the rest stay at zero.  Use greedy to
    % match as many as possible.
    rec = local_greedy_assign(C);
end
end

function rec = local_greedy_assign(C)
[K, Ks] = size(C);
rec = zeros(K, 1);
used = false(Ks, 1);
% Order rows by descending best score so high-confidence matches win
% the columns first.
[~, row_order] = sort(max(C, [], 2), 'descend');
for ii = 1:numel(row_order)
    k = row_order(ii);
    avail = find(~used);
    if isempty(avail)
        break;
    end
    [val, jrel] = max(C(k, avail));
    j = avail(jrel);
    rec(k) = val;
    used(j) = true;
end
end

function v = getfield_safe(s, name, default)
if isstruct(s) && isfield(s, name)
    v = s.(name);
else
    v = default;
end
end
