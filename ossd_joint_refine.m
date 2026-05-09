function modes = ossd_joint_refine(x, modes, p)
%OSSD_JOINT_REFINE  Coordinate-wise amplitude refinement followed by LS.
%   Part of OSSD-Basic by F. Latifoğlu and L. Latifoğlu (Erciyes
%   University).  See ossd_decompose.m for full author / citation
%   information; LICENSE in package root for terms of use.
%
%   modes = ossd_joint_refine(x, modes, p)
%
%   Performs the joint amplitude refinement used by OSSD-Basic: in each
%   sweep, for every column k in MODES, the target is set to
%
%       target_k = x - sum_{j != k} M_j
%
%   and the column M_k is rescaled by the projection coefficient
%   beta = (target_k' * M_k) / (M_k' * M_k), clipped to [0, beta_cap].
%   This is repeated for `iters` sweeps.  After the final sweep, an
%   optional global least-squares step rescales every column jointly:
%
%       beta = pinv(M) * x;   (clipped to >= 0 if requested)
%       M(:, k) = beta(k) * M(:, k)
%
%   Parameters read from p:
%       p.joint_refine_iters         (number of sweeps; required)
%       p.const_joint_beta_cap       (sweep-step amplitude cap; default 1.55)
%       p.const_joint_final_ls       (apply final pinv LS; default true)
%       p.const_joint_nonneg_ls      (clip final LS coefficients to >= 0)
%
%   This function is the V2-kernel refinement step used both inside
%   ossd_band_pipeline (light pass) and inside ossd_outer_pipeline
%   (the main OSSD refinement).

x = x(:);
[~, K] = size(modes);
if K == 0
    return;
end

% Number of sweeps.
nit = 6;
if isfield(p, 'joint_refine_iters') && ~isempty(p.joint_refine_iters)
    nit = max(1, round(p.joint_refine_iters));
end

% Sweep-step amplitude cap.
bcap = 1.55;
if isfield(p, 'const_joint_beta_cap') && ~isempty(p.const_joint_beta_cap)
    bcap = p.const_joint_beta_cap;
end

M = modes;

% --- Coordinate sweeps -----------------------------------------------------
for it = 1:nit
    for k = 1:K
        other  = sum(M, 2) - M(:, k);
        target = x - other;
        mk = M(:, k);
        den = mk' * mk + eps;
        b = (target' * mk) / den;
        b = max(0, min(b, bcap));
        M(:, k) = b * mk;
    end
end

% --- Final joint LS rescaling ---------------------------------------------
do_final_ls = true;
if isfield(p, 'const_joint_final_ls') && ~isempty(p.const_joint_final_ls)
    do_final_ls = logical(p.const_joint_final_ls);
end
do_nonneg = false;
if isfield(p, 'const_joint_nonneg_ls') && ~isempty(p.const_joint_nonneg_ls)
    do_nonneg = logical(p.const_joint_nonneg_ls);
end

if do_final_ls
    beta = pinv(M) * x;
    if do_nonneg
        beta = max(beta, 0);
    end
    for k = 1:K
        M(:, k) = beta(k) * M(:, k);
    end
end

modes = M;

end
