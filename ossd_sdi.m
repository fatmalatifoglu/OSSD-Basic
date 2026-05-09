function out = ossd_sdi(bands, varargin)
%OSSD_SDI  Snowflake Disjointness Index — pairwise tau-axis disjointness.
%   Part of OSSD-Basic by F. Latifoğlu and L. Latifoğlu (Erciyes
%   University).  See ossd_decompose.m for full author / citation
%   information; LICENSE in package root for terms of use.
%
%   out = ossd_sdi(bands)
%   out = ossd_sdi(bands, 'mode', 'tau' | 'pixel')
%
%   Quantifies the extent to which OSSD's extracted bands occupy
%   non-overlapping regions of the delay (tau) axis — the "snowflake
%   principle": each band falls in its own delay-domain neighbourhood
%   without colliding with the others.
%
%   For each pair of bands (i, j) the Jaccard distance over their tau
%   support is
%
%       J(i, j)  = |T_i \cap T_j| / |T_i \cup T_j|
%       d(i, j)  = 1 - J(i, j)
%
%   where T_i is the set of (integer) tau values that band i covers.
%   The Snowflake Disjointness Index averages d(i, j) over all distinct
%   pairs:
%
%       SDI = (1 / C(K, 2)) * sum_{i < j}  d(i, j),
%
%   so SDI = 1 means every pair of bands is perfectly disjoint along
%   tau (no shared lag), and SDI = 0 means every pair shares the same
%   tau support entirely.  Values close to 1 indicate the
%   decomposition has obeyed the snowflake principle: bands settle on
%   separate delay-domain regions without colliding.
%
%   The metric is structural, not statistical: it reads off the band
%   geometry that OSSD produces, so it cannot be computed for a
%   frequency-domain decomposition, which represents
%   each component only as a 1-D time series.  This makes SDI a
%   discriminator unique to self-similarity-based decompositions.
%
%   --- Inputs ---
%     bands    {nB x 1} cell array of [npix x 2] pixel matrices in
%              (time, tau) coordinates (the field out.bands of
%              ossd_decompose).
%     ...      name-value:
%                'mode'   'tau'   (default) Jaccard on integer tau
%                                  support of each band
%                         'pixel' Jaccard on full (time, tau) pixel
%                                  set — stricter, also penalises
%                                  same-tau-different-time overlaps
%
%   --- Outputs (struct) ---
%     out.sdi              scalar in [0, 1] — overall snowflake index
%     out.pair_distances   nB x nB matrix of pairwise d(i, j); 0 on
%                          the diagonal; symmetric
%     out.pair_jaccards    nB x nB matrix of pairwise J(i, j)
%     out.pair_table       table of pairwise statistics for inspection
%     out.mode             which support mode was used
%
%   --- Example ---
%     out = ossd_decompose(x, params);
%     s = ossd_sdi(out.bands);
%     fprintf('SDI = %.4f (snowflake disjointness)\n', s.sdi);

% --------------------------------------------------------------------------
% Parse options.
% --------------------------------------------------------------------------
opt = struct('mode', 'tau');
for i = 1:2:numel(varargin)
    name = lower(varargin{i});
    if isfield(opt, name)
        opt.(name) = varargin{i + 1};
    end
end
mode_str = lower(char(opt.mode));

% --------------------------------------------------------------------------
% Defensive guards.
% --------------------------------------------------------------------------
out = struct();
out.mode = mode_str;

if isempty(bands) || numel(bands) < 2
    out.sdi            = NaN;
    out.pair_distances = [];
    out.pair_jaccards  = [];
    out.pair_table     = table();
    return;
end

nB = numel(bands);

% --------------------------------------------------------------------------
% Build per-band support sets.
% --------------------------------------------------------------------------
supports = cell(nB, 1);
for k = 1:nB
    b = bands{k};
    if isempty(b)
        supports{k} = [];
        continue;
    end
    switch mode_str
        case 'tau'
            % Unique integer tau values covered by this band.
            supports{k} = unique(b(:, 2));
        case 'pixel'
            % Full (time, tau) pixel set as a hash via linear indexing.
            % We use sub2ind on a virtual grid sized to the data extent.
            tmin = min(b(:, 1));  tmax = max(b(:, 1));
            % use a simple unique-row trick:
            supports{k} = unique(b, 'rows');
        otherwise
            error('ossd_sdi: unknown mode "%s" (use tau or pixel)', mode_str);
    end
end

% --------------------------------------------------------------------------
% Pairwise Jaccard / disjointness.
% --------------------------------------------------------------------------
J = eye(nB);   % Jaccard similarity (1 on diagonal)
D = zeros(nB); % distance     (0 on diagonal)

pair_i_list = [];
pair_j_list = [];
pair_J_list = [];
pair_d_list = [];
pair_inter  = [];
pair_union  = [];

for i = 1:nB - 1
    si = supports{i};
    for j = i + 1:nB
        sj = supports{j};

        if isempty(si) || isempty(sj)
            J(i, j) = 0;
            J(j, i) = 0;
            D(i, j) = 1;
            D(j, i) = 1;
            inter_sz = 0;
            union_sz = numel(si) + numel(sj);
        else
            switch mode_str
                case 'tau'
                    inter_set = intersect(si, sj);
                    union_set = union(si, sj);
                    inter_sz = numel(inter_set);
                    union_sz = numel(union_set);
                case 'pixel'
                    inter_set = intersect(si, sj, 'rows');
                    union_set = union(si, sj, 'rows');
                    inter_sz = size(inter_set, 1);
                    union_sz = size(union_set, 1);
            end
            jac = inter_sz / max(union_sz, 1);
            J(i, j) = jac;
            J(j, i) = jac;
            D(i, j) = 1 - jac;
            D(j, i) = 1 - jac;
        end

        pair_i_list(end + 1, 1) = i;            %#ok<AGROW>
        pair_j_list(end + 1, 1) = j;            %#ok<AGROW>
        pair_J_list(end + 1, 1) = J(i, j);      %#ok<AGROW>
        pair_d_list(end + 1, 1) = D(i, j);      %#ok<AGROW>
        pair_inter(end + 1, 1)  = inter_sz;     %#ok<AGROW>
        pair_union(end + 1, 1)  = union_sz;     %#ok<AGROW>
    end
end

% --------------------------------------------------------------------------
% Aggregate.
% --------------------------------------------------------------------------
upper_mask = triu(true(nB), 1);
ds = D(upper_mask);
out.sdi            = mean(ds);
out.pair_distances = D;
out.pair_jaccards  = J;
out.pair_table     = table(pair_i_list, pair_j_list, ...
    pair_inter, pair_union, pair_J_list, pair_d_list, ...
    'VariableNames', {'Band_i', 'Band_j', ...
                      'Intersection', 'Union', 'Jaccard', 'Distance'});
end
