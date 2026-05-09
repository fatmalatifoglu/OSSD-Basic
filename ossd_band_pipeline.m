function out = ossd_band_pipeline(x, p)
%OSSD_BAND_PIPELINE  Stages 1-6 of the OSSD-Basic decomposition.
%   Part of OSSD-Basic by F. Latifoğlu and L. Latifoğlu (Erciyes
%   University).  See ossd_decompose.m for full author / citation
%   information; LICENSE in package root for terms of use.
%
%   out = ossd_band_pipeline(x, p)
%
%   Builds the initial mode matrix from the trajectory matrix:
%       (1) trajectory matrix and similarity plane
%       (2) robust thresholding and band extraction
%       (3) band merging by mean-trajectory proxy correlation + tau overlap
%       (4) per-band subspace modelling (rank chosen from energy target,
%           scree-knee heuristic and an MDL-like criterion, with
%           class-aware caps)
%       (5) joint orthogonalization across band bases (sequential)
%       (6) diagonal averaging to produce the initial modes
%
%   Output (struct):
%
%       out.modes      N x K initial modes (one per surviving band)
%       out.bands      cell of struct, one entry per surviving band, with
%                      fields: Omega, U, s, rank, energy_ratio, time_span,
%                      tau_span, band_type
%       out.band_meta  same length as bands, kept for the summary table
%       out.S          raw similarity plane (pre-MAD)
%       out.Srob       robust similarity plane (post-MAD)
%       out.tau_axis   tau values along the second axis of S/Srob

x = x(:);
N = numel(x);

out = struct('modes', zeros(N, 0), 'bands', {{}}, 'band_meta', struct([]), ...
             'S', [], 'Srob', [], 'tau_axis', []);

% --------------------------------------------------------------------------
% Stage 1: trajectory matrix + similarity plane
% --------------------------------------------------------------------------
[X, ~] = local_build_trajectory(x, p.L);
[~, M] = size(X);
if M < 2
    return;
end

G = local_similarity_matrix(X);
max_tau_eff = min(p.max_tau, M - 1);
[S, tau_axis] = local_similarity_plane(G, max_tau_eff);
Srob = local_robust_normalize_plane(S);

out.S = S;
out.Srob = Srob;
out.tau_axis = tau_axis;

% --------------------------------------------------------------------------
% Stage 2: robust thresholding + 8-connected band extraction
% --------------------------------------------------------------------------
bands = local_extract_bands(Srob, p);
if isempty(bands)
    return;
end

% --------------------------------------------------------------------------
% Stage 3: merge bands by mean-trajectory proxy correlation + tau overlap.
% Uses an OSSD-novel adaptive threshold: bands with high tau-overlap merge
% under a relaxed correlation cutoff (cmin_low); bands with low overlap
% require near-identical proxy correlation (cmin_high).
% --------------------------------------------------------------------------
bands = local_merge_bands_proxy_corr(bands, X, ...
    p.const_band_proxy_corr_min, ...
    p.const_band_proxy_corr_max, ...
    p.const_band_tau_overlap_min);
if isempty(bands)
    return;
end

% Optional cap on the number of bands carried forward.
if ~isempty(p.n_bands) && p.n_bands >= 1
    nb = min(round(p.n_bands), numel(bands));
    bands = bands(1:nb);
end

% --------------------------------------------------------------------------
% Stage 4: per-band subspace modelling
% --------------------------------------------------------------------------
K0 = numel(bands);
basisCell  = cell(K0, 1);
band_meta  = repmat(struct('Omega', [], 'U', [], 's', [], 'rank', 0, ...
    'energy_ratio', 0, 'time_span', [0 0], 'tau_span', [0 0], ...
    'band_type', ''), K0, 1);

for k = 1:K0
    Omega = local_band_to_window_indices(bands{k}, M);
    Omega = Omega(Omega >= 1 & Omega <= M);
    if numel(Omega) < p.min_windows_per_band
        continue;
    end

    btype = local_classify_band_type(bands{k}, p.const_harmonic_aspect_min);

    Xk = X(:, Omega);
    [Uk, sk, rk, er] = local_band_subspace(Xk, p, btype, numel(Omega));
    if isempty(Uk)
        continue;
    end

    band_meta(k).Omega        = Omega;
    band_meta(k).U            = Uk;
    band_meta(k).s            = sk;
    band_meta(k).rank         = rk;
    band_meta(k).energy_ratio = er;
    band_meta(k).time_span    = [min(bands{k}(:, 1)), max(bands{k}(:, 1))];
    band_meta(k).tau_span     = [min(bands{k}(:, 2)), max(bands{k}(:, 2))];
    band_meta(k).band_type    = btype;
    basisCell{k} = Uk;
end

valid     = ~cellfun(@isempty, basisCell);
basisCell = basisCell(valid);
band_meta = band_meta(valid);
bands     = bands(valid);
K = numel(basisCell);
if K == 0
    return;
end

% --------------------------------------------------------------------------
% Stage 5: joint orthogonalization across band bases
% --------------------------------------------------------------------------
if strcmpi(p.const_orth_method, 'qr_block')
    Qcell = local_joint_orthogonalize_qr_block(basisCell, p.const_orth_tol);
else
    Qcell = local_joint_orthogonalize_sequential(basisCell, p.const_orth_tol);
end

% --------------------------------------------------------------------------
% Stage 6: diagonal averaging to produce initial mode time-series.
% --------------------------------------------------------------------------
modes = zeros(N, K);
for k = 1:K
    Qk = Qcell{k};
    if isempty(Qk)
        continue;
    end
    Xproj = Qk * (Qk' * X);
    mk = local_diagonal_averaging(Xproj);
    modes(:, k) = mk(:);
end

% Drop degenerate modes (all-zero columns can arise from the projection).
nz = false(1, K);
for k = 1:K
    nz(k) = norm(modes(:, k)) > 1e-12;
end
modes      = modes(:, nz);
band_meta  = band_meta(nz);
bands      = bands(nz);

out.modes     = modes;
out.bands     = bands;
out.band_meta = band_meta;

end

% ==========================================================================
% LOCAL HELPERS — geometric core (Stages 1-2)
% ==========================================================================

function [X, starts] = local_build_trajectory(x, L)
% Hankel trajectory matrix: X(:, i) = x(i:i+L-1).
x = x(:);
N = numel(x);
M = N - L + 1;
if M < 1
    X = zeros(L, 0);
    starts = zeros(1, 0);
    return;
end
X = zeros(L, M);
for i = 1:M
    X(:, i) = x(i:i + L - 1);
end
starts = 1:M;
end

function G = local_similarity_matrix(X)
% Window-to-window normalised |inner product|.
[~, M] = size(X);
if M < 1
    G = [];
    return;
end
nrm = sqrt(sum(X .^ 2, 1)) + eps;
Xn = X ./ nrm;
G = abs(Xn' * Xn);
G(1:M + 1:end) = 1;
end

function [S, tau_axis] = local_similarity_plane(G, max_tau)
% Lag self-similarity plane: S(n, tau) = G(n, n+tau).
M = size(G, 1);
max_tau = min(max_tau, M - 1);
if max_tau < 1
    S = zeros(M, 0);
    tau_axis = [];
    return;
end
tau_axis = (1:max_tau)';
S = nan(M, numel(tau_axis));
for it = 1:numel(tau_axis)
    tau = tau_axis(it);
    for n = 1:(M - tau)
        S(n, it) = G(n, n + tau);
    end
end
end

function Srob = local_robust_normalize_plane(S)
% Per-column robust scaling: (S - median) / (1.4826 * MAD).
Srob = S;
for j = 1:size(S, 2)
    v = S(:, j);
    idx = isfinite(v);
    if ~any(idx)
        continue;
    end
    vv = v(idx);
    medv = median(vv);
    madv = median(abs(vv - medv)) / 0.6745 + eps;
    Srob(idx, j) = (vv - medv) / madv;
end
Srob(~isfinite(Srob)) = -inf;
end

function bands = local_extract_bands(Srob, p)
% Threshold Srob at median + k*MAD; keep 8-connected components passing
% area / time-span / tau-span filters; sort by area descending.
bands = {};
vals = Srob(isfinite(Srob) & Srob > -inf);
if isempty(vals)
    return;
end
medv = median(vals);
md = median(abs(vals - medv)) / 0.6745 + eps;
thr = medv + p.band_thresh_k * md;

BW = isfinite(Srob) & (Srob >= thr);
cc = local_conncomp_8(BW);
if cc.NumObjects == 0
    return;
end

kout = 0;
for i = 1:cc.NumObjects
    pix = cc.PixelIdxList{i};
    if numel(pix) < p.min_band_area
        continue;
    end
    [rr, cc_] = ind2sub(size(Srob), pix);
    time_span = max(rr) - min(rr) + 1;
    tau_span = max(cc_) - min(cc_) + 1;
    if time_span < p.min_time_span || tau_span < p.min_tau_span
        continue;
    end
    kout = kout + 1;
    bands{kout, 1} = [rr(:), cc_(:)]; %#ok<AGROW>
end

if ~isempty(bands)
    areas = cellfun(@(b) size(b, 1), bands);
    [~, idx] = sort(areas, 'descend');
    bands = bands(idx);
end
end

function cc = local_conncomp_8(M)
% 8-connected components of a binary mask (no Image Processing Toolbox).
M = logical(M);
[h, w] = size(M);
L = zeros(h, w);
current = 0;
for i = 1:h
    for j = 1:w
        if M(i, j) && L(i, j) == 0
            current = current + 1;
            stack = [i, j];
            while ~isempty(stack)
                ii = stack(1, 1);
                jj = stack(1, 2);
                stack(1, :) = [];
                if ii < 1 || ii > h || jj < 1 || jj > w
                    continue;
                end
                if ~M(ii, jj) || L(ii, jj) > 0
                    continue;
                end
                L(ii, jj) = current;
                for di = -1:1
                    for dj = -1:1
                        if di == 0 && dj == 0
                            continue;
                        end
                        stack(end + 1, :) = [ii + di, jj + dj]; %#ok<AGROW>
                    end
                end
            end
        end
    end
end
cc = struct();
cc.NumObjects = current;
cc.PixelIdxList = cell(max(1, current), 1);
for k = 1:current
    cc.PixelIdxList{k} = find(L == k);
end
if current == 0
    cc.PixelIdxList = {};
end
end

function Omega = local_band_to_window_indices(bandPixels, M)
% Map (n, tau) pixel pairs to the unique set of trajectory column indices
% n and n+tau that the band touches.
n = bandPixels(:, 1);
tau = bandPixels(:, 2);
Omega = [n; n + tau];
Omega = Omega(Omega >= 1 & Omega <= M);
Omega = unique(Omega(:)', 'stable');
end

% ==========================================================================
% LOCAL HELPERS — band merge by proxy correlation (Stage 3)
% ==========================================================================

function bands = local_merge_bands_proxy_corr(bands, X, cmin_low, cmin_high, ovmin)
% Greedy merge with delay-overlap-aware adaptive threshold (OSSD novelty).
%
% For each candidate pair (b_i, b_j), the merge threshold is interpolated
% according to their tau-axis overlap:
%
%     rho_thr(b_i, b_j) = cmin_high - (cmin_high - cmin_low) * tau_overlap
%
% Bands sharing the same delay structure (overlap -> 1) merge under a
% relaxed threshold cmin_low; bands whose delay supports do not overlap
% (overlap -> 0) require near-identical proxy correlation cmin_high to
% merge.  This couples the merge decision to the self-similarity geometry
% of the bands themselves: bands that look similar but live on different
% delay axes are treated as separate physical components.
%
% This replaces the global single-threshold rule used by V2/V3.
if isempty(bands)
    return;
end
bands = bands(:);
[~, M] = size(X);

while true
    nB = numel(bands);
    W = cell(nB, 1);
    for k = 1:nB
        if isempty(bands{k})
            W{k} = [];
            continue;
        end
        Om = local_band_to_window_indices(bands{k}, M);
        Om = Om(Om >= 1 & Om <= M);
        if isempty(Om)
            W{k} = [];
            continue;
        end
        wv = mean(X(:, Om), 2);
        wv = wv - mean(wv);
        W{k} = wv / (norm(wv) + eps);
    end

    best_score = -inf;        % score = c - rho_thr (margin above threshold)
    best_i = [];
    best_j = [];
    for i = 1:nB - 1
        if isempty(W{i})
            continue;
        end
        for j = i + 1:nB
            if isempty(W{j})
                continue;
            end
            ov = local_tau_overlap_ratio(bands{i}, bands{j});
            if ov < ovmin
                continue;
            end
            c = abs(W{i}' * W{j});

            % Adaptive threshold: low overlap -> high threshold (hard to merge).
            rho_thr = cmin_high - (cmin_high - cmin_low) * ov;

            if c >= rho_thr
                margin = c - rho_thr;
                if margin > best_score
                    best_score = margin;
                    best_i = i;
                    best_j = j;
                end
            end
        end
    end

    if isempty(best_i)
        break;
    end

    bands{best_i} = [bands{best_i}; bands{best_j}];
    bands{best_j} = [];
    keep = ~cellfun(@isempty, bands);
    bands = bands(keep);
    areas = cellfun(@(b) size(b, 1), bands);
    [~, ix] = sort(areas, 'descend');
    bands = bands(ix);
end
end

function r = local_tau_overlap_ratio(b1, b2)
% Fraction of tau-axis overlap relative to the shorter range.
t1 = [min(b1(:, 2)), max(b1(:, 2))];
t2 = [min(b2(:, 2)), max(b2(:, 2))];
lo = max(t1(1), t2(1));
hi = min(t1(2), t2(2));
inter = max(0, hi - lo + 1);
w1 = t1(2) - t1(1) + 1;
w2 = t2(2) - t2(1) + 1;
mn = min(w1, w2);
r = inter / (mn + eps);
end

% ==========================================================================
% LOCAL HELPERS — band classification + subspace (Stage 4)
% ==========================================================================

function t = local_classify_band_type(bandPixels, harmonic_aspect_min)
% 'harmonic' if extended along time; 'burst' if compact.
rr = bandPixels(:, 1);
cc = bandPixels(:, 2);
time_span = max(rr) - min(rr) + 1;
tau_span = max(cc) - min(cc) + 1;
aspect = time_span / (tau_span + eps);
if aspect >= harmonic_aspect_min
    t = 'harmonic';
else
    t = 'burst';
end
end

function [Ukeep, svals, rk, er] = local_band_subspace(Xk, p, band_type, nOmega)
% Rank chosen by combining: energy target (with slack), scree-knee, and an
% MDL-like criterion; class-aware ceiling; small-band cap.
if isempty(Xk) || size(Xk, 2) < 2
    Ukeep = []; svals = []; rk = 0; er = 0;
    return;
end

% Class-aware rank ceiling.
max_r = p.max_rank;
if strcmpi(band_type, 'burst')
    max_r = min(max_r, p.const_max_rank_burst);
else
    max_r = min(max_r, p.const_max_rank_harmonic);
end

% Small-band cap.
if nargin < 4 || isempty(nOmega)
    nOmega = size(Xk, 2);
end
if nOmega <= p.const_small_band_threshold
    max_r = min(max_r, p.const_small_band_max_rank);
end

[U, S, ~] = svd(Xk, 'econ');
svals = diag(S);
if isempty(svals)
    Ukeep = []; rk = 0; er = 0;
    return;
end

e = svals .^ 2;
sume = sum(e) + eps;
ce = cumsum(e) / sume;
n = numel(svals);

% Class-aware energy target and slack.
et = p.rank_energy_target;
slack = p.const_rank_energy_slack;
if strcmpi(band_type, 'burst')
    et    = min(et, p.const_rank_burst_target);
    slack = min(slack, p.const_rank_burst_slack);
end

r_compact = find(ce >= et - slack, 1, 'first');
if isempty(r_compact), r_compact = n; end

r_energy = find(ce >= et, 1, 'first');
if isempty(r_energy), r_energy = min(max_r, n); end

r_knee = local_scree_knee_rank(svals, max_r);

pen = p.const_rank_mdl_penalty;
best = inf;
r_mdl = 1;
for r = 1:min(n, max_r)
    tail = sum(e(r + 1:end));
    obj = r * pen + tail / sume;
    if obj < best
        best = obj;
        r_mdl = r;
    end
end

if p.const_rank_use_scree_knee
    cands = [r_compact, r_energy, r_knee, r_mdl];
else
    cands = [r_compact, r_energy, r_mdl];
end

if strcmpi(band_type, 'burst')
    rk = round(min(cands));
else
    rk = round(median(cands));
end
rk = max(1, min([rk, max_r, n]));

Ukeep = U(:, 1:rk);
er = ce(rk);
end

function rk = local_scree_knee_rank(svals, max_r)
% Knee = position of the steepest curvature in log singular values.
s = max(svals(:), 0);
n = numel(s);
if n <= 2
    rk = 1; return;
end
logs = log(s + eps);
d1 = diff(logs);
if numel(d1) < 2
    rk = 1; return;
end
d2 = diff(d1);
[~, ix] = min(d2);
rk = max(1, min([ix + 1, n, max_r]));
end

% ==========================================================================
% LOCAL HELPERS — joint orthogonalization (Stage 5)
% ==========================================================================

function Qcell = local_joint_orthogonalize_sequential(basisCell, tol)
% Sequential QR: project each band basis onto the orthogonal complement of
% the previously accepted bands, then QR.
K = numel(basisCell);
Qcell = cell(K, 1);
Qprev = [];
for k = 1:K
    Uk = basisCell{k};
    if isempty(Uk), continue; end
    W = Uk;
    if ~isempty(Qprev)
        W = W - Qprev * (Qprev' * W);
    end
    [Qw, ~] = qr(W, 0);
    if isempty(Qw), continue; end
    cn = sqrt(sum(Qw .^ 2, 1));
    keep = cn > tol;
    Qk = Qw(:, keep);
    if isempty(Qk), continue; end
    Qcell{k} = Qk;
    Qprev = [Qprev, Qk]; %#ok<AGROW>
end
end

function Qcell = local_joint_orthogonalize_qr_block(basisCell, tol)
% Single global QR on the concatenated basis; provided for completeness
% but not the default (mixes band signatures more than sequential).
K = numel(basisCell);
Qcell = cell(K, 1);
sizes = zeros(K, 1);
for k = 1:K
    if ~isempty(basisCell{k})
        sizes(k) = size(basisCell{k}, 2);
    end
end
B = cell2mat(basisCell(:)');
if isempty(B)
    return;
end
[Q, ~] = qr(B, 0);
cn = sqrt(sum(Q .^ 2, 1));
Q(:, cn <= tol) = 0;
ofs = 0;
for k = 1:K
    if sizes(k) == 0
        continue;
    end
    cols = ofs + (1:sizes(k));
    Qcell{k} = Q(:, cols);
    ofs = ofs + sizes(k);
end
end

% ==========================================================================
% LOCAL HELPER — diagonal averaging (Stage 6)
% ==========================================================================

function y = local_diagonal_averaging(X)
% SSA diagonal averaging / Hankelisation: L x M -> length L+M-1.
[L, M] = size(X);
N = L + M - 1;
y = zeros(N, 1);
w = zeros(N, 1);
for i = 1:L
    for j = 1:M
        n = i + j - 1;
        y(n) = y(n) + X(i, j);
        w(n) = w(n) + 1;
    end
end
y = y ./ (w + eps);
end

