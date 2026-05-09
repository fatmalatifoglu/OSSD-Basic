function out = ossd_sspi(x, modes, varargin)
%OSSD_SSPI  Self-Similarity Preservation Index of a decomposition.
%   Part of OSSD-Basic by F. Latifoğlu and L. Latifoğlu (Erciyes
%   University).  See ossd_decompose.m for full author / citation
%   information; LICENSE in package root for terms of use.
%
%   out = ossd_sspi(x, modes)
%   out = ossd_sspi(x, modes, 'L', 80, 'max_tau', 120, ...)
%
%   Quantifies how well a decomposition (x ~ sum_k modes(:,k))
%   preserves the input's geometric self-similarity structure:
%
%       SSPI(x, modes) = 1 - || S(x) - S(xhat) ||_F / || S(x) ||_F
%
%   where xhat = sum(modes, 2) and S(.) is the robust-normalised
%   self-similarity plane built by OSSD's own self-similarity
%   pipeline (ossd_build_trajectory + ossd_similarity_matrix +
%   ossd_similarity_plane + ossd_robust_normalize_plane).
%
%   Interpretation:
%     SSPI -> 1   modes preserve x's self-similarity structure
%                 (geometric, delay-domain content is retained)
%     SSPI -> 0   the decomposition has discarded structural
%                 information; the recombined signal looks similar
%                 in the time domain but lives on a different
%                 self-similarity manifold
%     SSPI < 0    pathological case (rarely occurs unless modes
%                 introduce spurious structure not present in x)
%
%   This metric is a structural quality criterion that
%   distinguishes self-similarity-based decompositions from purely
%   additive ones.  Frequency-domain methods (EMD, VMD, SSA) only
%   guarantee sum(modes) ~ x and therefore can score arbitrarily
%   low SSPI even when their RRE is excellent — the recomposed
%   signal matches in time but its geometric structure is not
%   preserved.  OSSD, by construction, optimises a band-wise
%   approximation of S(x) and therefore lies near SSPI -> 1.
%
%   --- Inputs ---
%     x     N x 1 input signal.
%     modes N x K matrix of decomposition modes (any method).
%     ...   name-value options:
%             'L'           (default 80)  trajectory window length
%             'max_tau'     (default 120) max lag for similarity plane
%             'use_robust'  (default true) compare on robust-normalised
%                           plane S_rob; set false to compare on raw S
%
%   --- Outputs (struct) ---
%     out.sspi              scalar in (-inf, 1]
%     out.frob_diff         || S(x) - S(xhat) ||_F
%     out.frob_ref          || S(x) ||_F
%     out.rre_time_domain   reconstruction error in the time domain
%                           (for sanity comparison: rre << 1 - sspi
%                           means the decomposition is structurally
%                           less faithful than its time-domain error
%                           would suggest)
%     out.params_used       struct of parameters used for S(.)
%
%   --- Example ---
%     out = ossd_decompose(x, params);
%     s = ossd_sspi(x, out.modes);
%     fprintf('SSPI = %.4f (RRE = %.4f)\n', s.sspi, s.rre_time_domain);

% --------------------------------------------------------------------------
% Parse options.
% --------------------------------------------------------------------------
opt = struct('L', 80, 'max_tau', 120, 'use_robust', true);
for i = 1:2:numel(varargin)
    name = lower(varargin{i});
    if isfield(opt, name)
        opt.(name) = varargin{i + 1};
    end
end

x = x(:);
N = numel(x);

if isempty(modes)
    out = struct('sspi', NaN, 'frob_diff', NaN, 'frob_ref', NaN, ...
                 'rre_time_domain', 1, 'params_used', opt);
    return;
end

xhat = sum(modes, 2);

% --------------------------------------------------------------------------
% Build self-similarity planes for x and xhat using OSSD's pipeline.
% --------------------------------------------------------------------------
[Sx, Sxhat] = local_build_planes(x, xhat, opt);

if isempty(Sx) || isempty(Sxhat)
    out = struct('sspi', NaN, 'frob_diff', NaN, 'frob_ref', NaN, ...
                 'rre_time_domain', norm(x - xhat) / (norm(x) + eps), ...
                 'params_used', opt);
    return;
end

% Both planes must have identical shape; trim to common region if not.
[Sx, Sxhat] = local_align_planes(Sx, Sxhat);

% --------------------------------------------------------------------------
% Compute Frobenius distance, ignoring NaNs (which may appear at the
% borders of the robust-normalised plane).
% --------------------------------------------------------------------------
mask = isfinite(Sx) & isfinite(Sxhat);
if ~any(mask(:))
    out = struct('sspi', NaN, 'frob_diff', NaN, 'frob_ref', NaN, ...
                 'rre_time_domain', norm(x - xhat) / (norm(x) + eps), ...
                 'params_used', opt);
    return;
end

dif = Sx(mask) - Sxhat(mask);
ref = Sx(mask);
frob_diff = sqrt(sum(dif .^ 2));
frob_ref  = sqrt(sum(ref .^ 2));
sspi = 1 - frob_diff / (frob_ref + eps);

out = struct();
out.sspi            = sspi;
out.frob_diff       = frob_diff;
out.frob_ref        = frob_ref;
out.rre_time_domain = norm(x - xhat) / (norm(x) + eps);
out.params_used     = opt;
end


% ==========================================================================
% Helpers
% ==========================================================================

function [Sx, Sxhat] = local_build_planes(x, xhat, opt)
% Use OSSD's own pipeline if available; otherwise fall back to a
% minimal local implementation so this helper stands alone.
Sx = []; Sxhat = [];

% Prefer OSSD's pipeline (consistent with the rest of the package).
have_ossd = exist('ossd_build_trajectory', 'file') == 2 && ...
            exist('ossd_similarity_matrix', 'file') == 2 && ...
            exist('ossd_similarity_plane', 'file') == 2;

if have_ossd
    try
        [X1, ~] = ossd_build_trajectory(x, opt.L);
        [X2, ~] = ossd_build_trajectory(xhat, opt.L);
        if size(X1, 2) >= 2 && size(X2, 2) >= 2
            G1 = ossd_similarity_matrix(X1);
            G2 = ossd_similarity_matrix(X2);
            mt1 = min(opt.max_tau, size(X1, 2) - 1);
            mt2 = min(opt.max_tau, size(X2, 2) - 1);
            mt  = min(mt1, mt2);
            [Sx_raw, ~]   = ossd_similarity_plane(G1, mt);
            [Sxhat_raw, ~] = ossd_similarity_plane(G2, mt);

            if opt.use_robust && exist('ossd_robust_normalize_plane', 'file') == 2
                Sx    = ossd_robust_normalize_plane(Sx_raw);
                Sxhat = ossd_robust_normalize_plane(Sxhat_raw);
            else
                Sx    = Sx_raw;
                Sxhat = Sxhat_raw;
            end
            return;
        end
    catch
        % Fall through to local implementation.
    end
end

% Local fallback: build Hankel trajectory and a simple cosine-similarity
% plane.  Used only when OSSD pipeline functions are not on the path.
[Sx,    ok1] = local_simple_plane(x,    opt.L, opt.max_tau);
[Sxhat, ok2] = local_simple_plane(xhat, opt.L, opt.max_tau);
if ~ok1 || ~ok2
    Sx = []; Sxhat = [];
end
end

function [S, ok] = local_simple_plane(x, L, max_tau)
x = x(:);
N = numel(x);
M = N - L + 1;
ok = true;
if M < 2
    S = [];
    ok = false;
    return;
end
% Hankel matrix.
X = zeros(L, M);
for j = 1:M
    X(:, j) = x(j:j + L - 1);
end
% Column-wise normalisation.
nrm = sqrt(sum(X .^ 2, 1)) + eps;
Xn = X ./ nrm;

mt = min(max_tau, M - 1);
S = zeros(M - mt, mt);
for n = 1:(M - mt)
    base = Xn(:, n);
    for tau = 1:mt
        S(n, tau) = sum(base .* Xn(:, n + tau));
    end
end
end

function [A, B] = local_align_planes(A, B)
% Trim two planes to a common rectangular shape if they differ.
if isequal(size(A), size(B))
    return;
end
nA = min(size(A, 1), size(B, 1));
mA = min(size(A, 2), size(B, 2));
A = A(1:nA, 1:mA);
B = B(1:nA, 1:mA);
end
