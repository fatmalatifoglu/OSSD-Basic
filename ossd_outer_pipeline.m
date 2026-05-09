function [modes, info] = ossd_outer_pipeline(x, modes, p)
%OSSD_OUTER_PIPELINE  Stage 7 — OSSD-Basic outer refinement.
%   Part of OSSD-Basic by F. Latifoğlu and L. Latifoğlu (Erciyes
%   University).  See ossd_decompose.m for full author / citation
%   information; LICENSE in package root for terms of use.
%
%   [modes, info] = ossd_outer_pipeline(x, modes, p)
%
%   Applies the three OSSD-specific refinement steps in order:
%
%     (a) SIMILARITY MERGE.  Greedily merges pairs of modes whose envelope
%         and ACF profiles indicate they belong to the same component, in
%         a manner that does not collapse pairs that are also similar to a
%         third mode (so two distinct components are not crushed together
%         just because both happen to be smooth).  Composite score:
%
%           score(i,j) = w_E * (E_i + E_j) / E_x
%                      + w_sim * sim(i, j)
%                      - w_overlap * |M_i' M_j| / (||M_i|| ||M_j||)
%                      - lambda_corr * max_{k != i,j}
%                            max( |corr(M_i, M_k)|, |corr(M_j, M_k)| )
%
%         A candidate (i, j) is admissible only if sim(i, j) >= sim_min.
%         The pair with the highest score is merged (energy-weighted mean)
%         provided the score itself exceeds score_min.  The loop terminates
%         when no admissible pair remains.
%
%     (b) MGS + LEAST-SQUARES AMPLITUDES.  The surviving modes are passed
%         through Modified Gram-Schmidt to obtain an orthonormal basis Q,
%         then the least-squares amplitudes beta = Q' * x are computed
%         and the modes set to Q .* beta'.  This decouples mode separation
%         (orthogonalisation) from amplitude estimation (LS on the input)
%         in a single, deterministic step.
%
%     (c) JOINT REFINEMENT.  A final coordinate-wise amplitude refinement
%         (V2-style) is applied to absorb any residual cross-talk that the
%         orthogonalisation could not remove.
%
%   The three constants of OSSD-Basic that govern step (a) are
%   (w_E, w_sim, w_overlap, lambda_corr) and the two thresholds
%   (sim_merge_min_score, sim_merge_min_sim).  All read from p.

info = struct( ...
    'similarity_merges',     0, ...
    'similarity_score_log', [], ...
    'gs_cols_in',            size(modes, 2), ...
    'gs_cols_out',           size(modes, 2), ...
    'gs_beta',              [], ...
    'joint_refine_applied', false, ...
    'post_refine_orth_applied', false);

if isempty(modes)
    return;
end

x = x(:);

% --------------------------------------------------------------------------
% (a) Similarity merge
% --------------------------------------------------------------------------
if size(modes, 2) >= 2
    [modes, mi] = local_similarity_merge(x, modes, p);
    info.similarity_merges    = mi.n_merges;
    info.similarity_score_log = mi.scores;
end

% --------------------------------------------------------------------------
% (b) Modified Gram-Schmidt + LS amplitudes
% --------------------------------------------------------------------------
if ~isempty(modes)
    [modes, gs] = local_gram_schmidt_ls(x, modes, p);
    info.gs_cols_in  = gs.cols_in;
    info.gs_cols_out = gs.cols_out;
    info.gs_beta     = gs.beta;
end

% --------------------------------------------------------------------------
% (c) Joint refinement
% --------------------------------------------------------------------------
if ~isempty(modes)
    p_ref = p;
    p_ref.joint_refine_iters = p.joint_refine_iters;
    modes = ossd_joint_refine(x, modes, p_ref);
    info.joint_refine_applied = true;
end

% --------------------------------------------------------------------------
% (d) Post-refinement re-orthogonalisation (MGS + LS amplitudes).
%
%   Joint refinement adjusts mode amplitudes via least-squares against
%   x; this can re-introduce small inter-mode correlations even though
%   step (b) initially produced an orthogonal column set.  A single
%   final MGS + LS pass projects the refined modes back onto an
%   orthogonal basis and re-fits amplitudes against x.  This collapses
%   the orthogonality index and the inter-mode |corr| to numerical
%   zero without disturbing the recovery of physical components: the
%   refit only redistributes amplitudes that were already nearly
%   correct, and reconstruction error typically improves slightly
%   because the LS step sees the cleaned columns.
% --------------------------------------------------------------------------
if ~isempty(modes) && size(modes, 2) >= 2
    [modes, gs2] = local_gram_schmidt_ls(x, modes, p);
    info.gs_cols_out = gs2.cols_out;     % update final column count
    info.gs_beta     = gs2.beta;         % update final amplitudes
    info.post_refine_orth_applied = true;
else
    info.post_refine_orth_applied = false;
end

end

% ==========================================================================
% LOCAL HELPERS
% ==========================================================================

function [modes, info] = local_similarity_merge(x, modes, p)
% Greedy merge: pick the pair with the highest composite score, merge it
% (energy-weighted mean), repeat.  Stops when no admissible pair has a
% score above p.const_sim_merge_min_score.

x = x(:);
Ex = sum(x .^ 2) + eps;

% Constants from p (with safe fallbacks).
w_E      = local_pget(p, 'const_sim_merge_w_energy',    0.35);
w_sim    = local_pget(p, 'const_sim_merge_w_sim',       1.00);
w_over   = local_pget(p, 'const_sim_merge_w_overlap',   0.45);
lam_corr = local_pget(p, 'const_sim_merge_lambda_corr', 0.20);
t_score  = local_pget(p, 'const_sim_merge_min_score',   0.28);
t_sim    = local_pget(p, 'const_sim_merge_min_sim',     0.38);

n_merges  = 0;
score_log = [];

while size(modes, 2) >= 2
    K = size(modes, 2);

    % Pairwise similarity matrix (mean of envelope + ACF).
    S = ossd_similarity_metrics(modes, 'matrix', p);

    best  = -inf;
    best_i = []; best_j = [];

    for i = 1:K - 1
        Ei = sum(modes(:, i) .^ 2);
        for j = i + 1:K
            Ej = sum(modes(:, j) .^ 2);
            E_term = (Ei + Ej) / Ex;
            sim_ij = S(i, j);

            ni = norm(modes(:, i)) + eps;
            nj = norm(modes(:, j)) + eps;
            overlap_ij = abs(modes(:, i)' * modes(:, j)) / (ni * nj);

            cmax_ij = ossd_similarity_metrics(modes, ...
                'max_corr_others', i, j);

            sc = w_E    * E_term ...
               + w_sim  * sim_ij ...
               - w_over * overlap_ij ...
               - lam_corr * cmax_ij;

            if sim_ij >= t_sim && sc > best
                best  = sc;
                best_i = i;
                best_j = j;
            end
        end
    end

    if isempty(best_i) || best < t_score
        break;
    end

    score_log(end + 1, 1) = best; %#ok<AGROW>

    i = best_i;
    j = best_j;
    Ei = sum(modes(:, i) .^ 2);
    Ej = sum(modes(:, j) .^ 2);
    w  = Ei / (Ei + Ej + eps);
    mnew = w * modes(:, i) + (1 - w) * modes(:, j);

    modes(:, i) = mnew;
    modes(:, j) = [];
    n_merges = n_merges + 1;
end

info = struct('n_merges', n_merges, 'scores', score_log);
end

function [modes_out, info] = local_gram_schmidt_ls(x, modes, p)
% Modified Gram-Schmidt orthogonalisation followed by LS amplitudes
% beta = Q' * x.  Output columns are Q .* beta'.

x = x(:);
modes = modes(:, :);
[N, K] = size(modes);
info = struct('cols_in', K, 'cols_out', 0, 'beta', []);
modes_out = zeros(N, 0);
if K == 0, return; end

tol = local_pget(p, 'const_orth_tol', 1e-9);

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

function v = local_pget(p, name, default_val)
% Safe field-with-default getter on a parameter struct.
if isstruct(p) && isfield(p, name) && ~isempty(p.(name))
    v = p.(name);
else
    v = default_val;
end
end
