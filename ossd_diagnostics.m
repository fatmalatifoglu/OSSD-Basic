function varargout = ossd_diagnostics(x, modes, residual, fs, mode)
%OSSD_DIAGNOSTICS  Reconstruction, separation and instantaneous-frequency
%   Part of OSSD-Basic by F. Latifoğlu and L. Latifoğlu (Erciyes
%   University).  See ossd_decompose.m for full author / citation
%   information; LICENSE in package root for terms of use.
%                  diagnostics for an OSSD-Basic decomposition.
%
%   d = ossd_diagnostics(x, modes, residual, fs)
%
%       Returns a struct d with fields:
%
%         d.rel_recon_error      ||x - sum_k m_k|| / ||x||
%         d.orthogonality_index  sum_{i<j} |m_i' m_j| / sum_k ||m_k||^2
%                                (in [0, K(K-1)/2]; lower is better)
%         d.mode_corr_matrix     K x K |Pearson| matrix on demeaned modes
%         d.mode_corr_table      mode_corr_matrix as a labelled MATLAB table
%         d.median_inst_freq_hz  K x 1 vector of median |dphi/dt|*fs/(2pi)
%                                (NaN where fs <= 0 or insufficient samples)
%         d.inst_freq_var_hz2    K x 1 vector of var(|dphi/dt|*fs/(2pi))
%         d.spectral_bandwidth_hz K x 1 vector of std-dev around centroid
%         d.residual_energy_ratio  ||residual||^2 / ||x||^2
%         d.K                    number of modes
%
%   [fmed, ifvar] = ossd_diagnostics(x, modes, residual, fs, 'per_mode_only')
%
%       Compact form returning only the two K x 1 per-mode IF vectors.
%       Used by ossd_decompose to populate the summary table.
%
%   The diagnostics are deterministic and TRUTH-BLIND: no ground truth or
%   reference signal is consulted at any point.

if nargin < 5
    mode = "full";
end
mode = lower(string(mode));

x = x(:);
modes = modes(:, :);
[N, K] = size(modes);

if nargin < 3 || isempty(residual)
    if K == 0
        residual = x;
    else
        residual = x - sum(modes, 2);
    end
end
residual = residual(:);

if nargin < 4 || isempty(fs)
    fs = 0;
end

% --------------------------------------------------------------------------
% Compact mode (called from ossd_decompose to fill the summary table).
% --------------------------------------------------------------------------
if mode == "per_mode_only"
    fmed  = nan(K, 1);
    ifvar = nan(K, 1);
    if fs > 0
        for k = 1:K
            [fmed(k), ifvar(k)] = local_if_stats(modes(:, k), fs);
        end
    end
    varargout{1} = fmed;
    varargout{2} = ifvar;
    return;
end

% --------------------------------------------------------------------------
% Full diagnostic struct.
% --------------------------------------------------------------------------
d = struct();
d.K = K;
d.rel_recon_error      = NaN;
d.orthogonality_index  = NaN;
d.mode_corr_matrix     = [];
d.mode_corr_table      = [];
d.median_inst_freq_hz  = nan(K, 1);
d.inst_freq_var_hz2    = nan(K, 1);
d.spectral_bandwidth_hz = nan(K, 1);
d.residual_energy_ratio = NaN;

if K == 0
    d.rel_recon_error = 1;
    d.orthogonality_index = 0;
    nx2 = sum(x .^ 2) + eps;
    d.residual_energy_ratio = sum(residual .^ 2) / nx2;
    varargout{1} = d;
    return;
end

% --- reconstruction error ---
xhat = sum(modes, 2);
nx = norm(x) + eps;
d.rel_recon_error = norm(x - xhat) / nx;

% --- orthogonality index ---
if K < 2
    d.orthogonality_index = 0;
else
    num = 0;
    den = 0;
    for i = 1:K
        den = den + norm(modes(:, i)) ^ 2;
        for j = i + 1:K
            num = num + abs(modes(:, i)' * modes(:, j));
        end
    end
    d.orthogonality_index = num / (den + eps);
end

% --- mode correlation matrix (|Pearson|) ---
C = zeros(K, K);
for i = 1:K
    C(i, i) = 1;
    ui = modes(:, i) - mean(modes(:, i));
    ni = norm(ui) + eps;
    for j = i + 1:K
        uj = modes(:, j) - mean(modes(:, j));
        nj = norm(uj) + eps;
        cij = abs((ui' * uj) / (ni * nj));
        C(i, j) = cij;
        C(j, i) = cij;
    end
end
d.mode_corr_matrix = C;
rn = cell(K, 1);
for k = 1:K
    rn{k} = sprintf('m%d', k);
end
d.mode_corr_table = array2table(C, 'VariableNames', rn, 'RowNames', rn);

% --- per-mode instantaneous-frequency stats and spectral bandwidth ---
if fs > 0
    for k = 1:K
        [d.median_inst_freq_hz(k), d.inst_freq_var_hz2(k)] = ...
            local_if_stats(modes(:, k), fs);
        d.spectral_bandwidth_hz(k) = local_spectral_bw_hz(modes(:, k), fs);
    end
end

% --- residual energy ratio ---
nx2 = sum(x .^ 2) + eps;
d.residual_energy_ratio = sum(residual .^ 2) / nx2;

varargout{1} = d;

end

% ==========================================================================
% LOCAL HELPERS
% ==========================================================================

function [fmed, ifvar] = local_if_stats(mk, fs)
% Median and variance of the instantaneous frequency derived from the
% Hilbert phase derivative.  Returns NaN where the signal is too short
% or fs <= 0.
mk = mk(:);
fmed = NaN;
ifvar = NaN;
if isempty(mk) || fs <= 0
    return;
end
z = hilbert(mk - mean(mk));
phi = unwrap(angle(z));
dphi = diff(phi);
fi = abs(dphi) * fs / (2 * pi);
fi = fi(isfinite(fi));
if numel(fi) < 4
    return;
end
fmed = median(fi);
ifvar = var(fi, 0);
end

function bw = local_spectral_bw_hz(y, fs)
% Frequency-weighted standard deviation around the spectral centroid
% (one-sided positive-frequency power spectrum).
y = y(:) - mean(y(:));
N = numel(y);
bw = NaN;
if N < 8 || fs <= 0
    return;
end
Y = fft(y);
P2 = abs(Y / N) .^ 2;
nh = floor(N / 2) + 1;
P1 = P2(1:nh);
if nh > 2
    P1(2:nh - 1) = 2 * P1(2:nh - 1);
end
f = ((0:nh - 1)' * fs / N);
sP = sum(P1) + eps;
fc = sum(f .* P1) / sP;
bw = sqrt(max(0, sum((f - fc) .^ 2 .* P1) / sP));
end
