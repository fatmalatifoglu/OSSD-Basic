function out = ossd_similarity_metrics(arg, mode, varargin)
%OSSD_SIMILARITY_METRICS  Single-entry dispatch for mode-pair similarity.
%   Part of OSSD-Basic by F. Latifoğlu and L. Latifoğlu (Erciyes
%   University).  See ossd_decompose.m for full author / citation
%   information; LICENSE in package root for terms of use.
%
%   This file consolidates every similarity / correlation primitive used
%   by OSSD-Basic.  It exposes a single public function and selects the
%   primitive via the MODE string:
%
%   --- Pair-level scalars (input is a 2-cell {a, b} of column vectors) ---
%
%       s = ossd_similarity_metrics({a,b}, 'envelope_pair')
%           |corr|( |Hilbert(a)|, |Hilbert(b)| ), demeaned.  Phase-invariant.
%
%       s = ossd_similarity_metrics({a,b}, 'acf_pair', maxlag)
%           |corr| between short-lag |ACF| profiles.  maxlag default 80.
%
%       s = ossd_similarity_metrics({a,b}, 'mean_pair', maxlag)
%           0.5*(envelope_pair + acf_pair).  This is the OSSD default.
%
%       c = ossd_similarity_metrics({a,b}, 'abs_pearson')
%           |Pearson|(a, b) on demeaned vectors.
%
%   --- Matrix / multi-mode operations (input is the K-column mode matrix) ---
%
%       S = ossd_similarity_metrics(modes, 'matrix', p)
%           K x K symmetric matrix of mean_pair similarities.
%           Reads p.const_sim_acf_max_lag (default 80) if present.
%
%       cmax = ossd_similarity_metrics(modes, 'max_corr_others', i, j)
%           max_{k != i,j} max(|Pearson(M_i, M_k)|, |Pearson(M_j, M_k)|).
%           Returns 0 if K < 3.
%
%   The dispatcher exists because MATLAB only exposes the first function
%   in a file to the caller; gathering all primitives behind a single
%   public name keeps the package limited to one file per concept while
%   still allowing every other module of OSSD-Basic to reach every
%   primitive it needs.

mode = lower(string(mode));

switch mode
    case "envelope_pair"
        out = local_envelope_sim(arg{1}, arg{2});

    case "acf_pair"
        if isempty(varargin)
            maxlag = 80;
        else
            maxlag = varargin{1};
        end
        out = local_acf_sim(arg{1}, arg{2}, maxlag);

    case "mean_pair"
        if isempty(varargin)
            maxlag = 80;
        else
            maxlag = varargin{1};
        end
        se = local_envelope_sim(arg{1}, arg{2});
        sa = local_acf_sim(arg{1}, arg{2}, maxlag);
        out = 0.5 * (se + sa);

    case "abs_pearson"
        out = local_abs_pearson(arg{1}, arg{2});

    case "matrix"
        if isempty(varargin)
            p = struct();
        else
            p = varargin{1};
        end
        maxlag = 80;
        if isstruct(p) && isfield(p, 'const_sim_acf_max_lag') && ...
                ~isempty(p.const_sim_acf_max_lag)
            maxlag = round(p.const_sim_acf_max_lag);
        end
        modes = arg(:, :);
        K = size(modes, 2);
        out = eye(K);
        if K < 2, return; end
        for i = 1:K
            for j = i + 1:K
                se = local_envelope_sim(modes(:, i), modes(:, j));
                sa = local_acf_sim(modes(:, i), modes(:, j), maxlag);
                sij = 0.5 * (se + sa);
                out(i, j) = sij;
                out(j, i) = sij;
            end
        end

    case "max_corr_others"
        if numel(varargin) < 2
            error('ossd_similarity_metrics:args', ...
                'max_corr_others requires (modes, ''max_corr_others'', i, j).');
        end
        i = varargin{1};
        j = varargin{2};
        modes = arg(:, :);
        K = size(modes, 2);
        out = 0;
        if K < 3, return; end
        for k = 1:K
            if k == i || k == j, continue; end
            out = max(out, local_abs_pearson(modes(:, i), modes(:, k)));
            out = max(out, local_abs_pearson(modes(:, j), modes(:, k)));
        end

    otherwise
        error('ossd_similarity_metrics:mode', ...
            'Unknown mode: %s', mode);
end

end

% ==========================================================================
% LOCAL HELPERS
% ==========================================================================

function s = local_envelope_sim(a, b)
% |corr|(|Hilbert(a)|, |Hilbert(b)|) on demeaned envelopes.
a = a(:) - mean(a(:));
b = b(:) - mean(b(:));
if norm(a) < eps || norm(b) < eps
    s = 0; return;
end
ea = abs(hilbert(a));
eb = abs(hilbert(b));
ea = ea - mean(ea);
eb = eb - mean(eb);
s = abs((ea' * eb) / ((norm(ea) + eps) * (norm(eb) + eps)));
end

function s = local_acf_sim(a, b, maxlag)
% |corr| between short-lag normalised ACF profiles.
a = a(:) - mean(a(:));
b = b(:) - mean(b(:));
maxlag = max(5, round(maxlag));
Na = min(numel(a), maxlag + 50);
Nb = min(numel(b), maxlag + 50);
a = a(1:Na);
b = b(1:Nb);
if norm(a) < eps || norm(b) < eps
    s = 0; return;
end
ca = local_norm_acf_vec(a, maxlag);
cb = local_norm_acf_vec(b, maxlag);
s = abs((ca' * cb) / ((norm(ca) + eps) * (norm(cb) + eps)));
end

function v = local_norm_acf_vec(x, L)
% Normalised, demeaned, unit-norm short-lag ACF vector.
n = numel(x);
L = min(L, n - 1);
v = zeros(L, 1);
xc = x - mean(x);
s0 = sum(xc .^ 2) + eps;
for lag = 1:L
    v(lag) = (xc(1:end - lag)' * xc(1 + lag:end)) / s0;
end
v = v - mean(v);
nv = norm(v);
if nv > eps
    v = v / nv;
end
end

function c = local_abs_pearson(u, v)
% |Pearson|(u, v) on demeaned vectors.
u = u(:) - mean(u(:), 'omitnan');
v = v(:) - mean(v(:), 'omitnan');
nu = norm(u) + eps;
nv = norm(v) + eps;
c = abs((u' * v) / (nu * nv));
end
