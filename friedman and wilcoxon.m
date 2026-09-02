clc; clear; close all;

%% ============================================================
%  STAGE-WISE ATTRIBUTION ANALYSIS
%  Repeated-measures design: Friedman test + paired Wilcoxon
%  signed-rank post-hoc tests with Holm correction.
% ============================================================

%% -------------------------------
% 1. Load File
% -------------------------------
[file, path] = uigetfile('*.csv', 'Select your attribution file');

if isequal(file, 0)
    error('No file selected');
end

data = readtable(fullfile(path, file));
stages = unique(data.stage);
n_stages = length(stages);
n_images = sum(strcmp(data.stage, stages{1}));

fprintf('\n============================================================\n');
fprintf('STAGE-WISE ATTRIBUTION ANALYSIS\n');
fprintf('Images: %d, Stages: %d\n', n_images, n_stages);
fprintf('============================================================\n');

%% -------------------------------
% 2. Reshape Data for Repeated Measures
%    (rows = images, columns = stages)
% -------------------------------
attribution_matrix = zeros(n_images, n_stages);

for i = 1:n_stages
    idx = strcmp(data.stage, stages{i});
    vals = data.contribution_score(idx);
    if length(vals) ~= n_images
        error('Stage "%s" has %d rows, expected %d. Check for missing/duplicate images per stage.', ...
            stages{i}, length(vals), n_images);
    end
    attribution_matrix(:, i) = vals;
end

%% -------------------------------
% 3. FRIEDMAN TEST (Repeated Measures)
% -------------------------------
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('FRIEDMAN TEST\n');
fprintf('%s\n', repmat('=', 1, 60));

[p_friedman, tbl_friedman, stats_friedman] = friedman(attribution_matrix, 1, 'off');

% tbl_friedman columns are: Source, SS, df, MS, Chi-sq, Prob>Chi-sq
% Row 2 is the "Columns" (stage) effect - the one we want.
chi2 = tbl_friedman{2, 5};   % Chi-sq
df   = tbl_friedman{2, 3};   % df
kendall_W = chi2 / (n_images * (n_stages - 1));

fprintf('Chi-square (chi^2) = %.4f\n', chi2);
fprintf('Degrees of Freedom = %d\n', df);
if p_friedman < 1e-300
    fprintf('p-value < 1e-300 (underflow; effectively zero)\n');
else
    fprintf('p-value = %.4g\n', p_friedman);
end
fprintf('Kendall''s W = %.4f\n', kendall_W);

if p_friedman < 0.001
    fprintf('Significant difference across stages (p < 0.001)\n');
elseif p_friedman < 0.05
    fprintf('Significant difference across stages (p < 0.05)\n');
else
    fprintf('No significant difference\n');
end

%% -------------------------------
% 4. WILCOXON SIGNED-RANK POST-HOC (Holm Correction)
% -------------------------------
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('POST-HOC: WILCOXON SIGNED-RANK (Holm Correction)\n');
fprintf('%s\n', repmat('=', 1, 60));

n_pairs = nchoosek(n_stages, 2);
pair_names   = cell(n_pairs, 1);
p_values     = zeros(n_pairs, 1);
effect_sizes = zeros(n_pairs, 1);
z_values     = zeros(n_pairs, 1);

pair_idx = 1;
for i = 1:n_stages
    for j = i+1:n_stages
        vals_i = attribution_matrix(:, i);
        vals_j = attribution_matrix(:, j);

        % Paired differences
        diff_vals = vals_i - vals_j;
        diff_vals = diff_vals(diff_vals ~= 0);  % Remove zero differences
        n_nonzero = length(diff_vals);

        if n_nonzero > 0
            [~, sort_idx_local] = sort(abs(diff_vals));
            ranks = 1:n_nonzero;
            signed_ranks = ranks .* sign(diff_vals(sort_idx_local))';
            % (transpose above is defensive in case diff_vals is a row;
            %  attribution_matrix columns are already column vectors here)
            signed_ranks = ranks(:) .* sign(diff_vals(sort_idx_local));

            W = sum(signed_ranks(signed_ranks > 0));   % W+ : sum of positive signed ranks

            % Normal-approximation p-value
            mu    = n_nonzero * (n_nonzero + 1) / 4;
            sigma = sqrt(n_nonzero * (n_nonzero + 1) * (2 * n_nonzero + 1) / 24);
            Z = (W - mu) / sigma;
            p = 2 * (1 - normcdf(abs(Z)));

            % Matched-pairs rank-biserial correlation
            % r = (W+ - W-) / (W+ + W-), and W+ + W- = n(n+1)/2
            % => r = 4*W+ / (n*(n+1)) - 1
            % Bounded in [-1, 1] by construction.
            r_effect = (4 * W) / (n_nonzero * (n_nonzero + 1)) - 1;

            pair_names{pair_idx}   = sprintf('%s vs %s', stages{i}, stages{j});
            p_values(pair_idx)     = p;
            z_values(pair_idx)     = Z;
            effect_sizes(pair_idx) = r_effect;
        else
            pair_names{pair_idx}   = sprintf('%s vs %s', stages{i}, stages{j});
            p_values(pair_idx)     = 1.0;
            z_values(pair_idx)     = 0;
            effect_sizes(pair_idx) = 0;
        end

        pair_idx = pair_idx + 1;
    end
end

% Hard sanity check: rank-biserial correlation must be in [-1, 1].
% If this ever fires, something upstream (ranking, sign, or W) is wrong.
assert(all(effect_sizes >= -1 - 1e-9 & effect_sizes <= 1 + 1e-9), ...
    'BUG: rank-biserial effect size out of [-1,1] range. Check W computation.');

% Holm-Bonferroni correction
[sorted_p, sort_idx] = sort(p_values);
holm_alpha = zeros(n_pairs, 1);
for k = 1:n_pairs
    holm_alpha(k) = 0.05 / (n_pairs - k + 1);
end
significant_holm = sorted_p < holm_alpha;

significant = zeros(n_pairs, 1);
for i = 1:n_pairs
    significant(sort_idx(i)) = significant_holm(i);
end

results_table = table(pair_names, p_values, effect_sizes, significant, ...
    'VariableNames', {'Comparison', 'PValue', 'EffectSize_r', 'Significant'});

disp(results_table);

%% -------------------------------
% 5. Save Results
% -------------------------------
output_dir = fullfile(path, 'friedman_wilcoxon_results');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

friedman_table = table({'Chi-square'; 'df'; 'p-value'; 'Kendall_W'}, ...
    {chi2; df; p_friedman; kendall_W}, ...
    'VariableNames', {'Metric', 'Value'});
writetable(friedman_table, fullfile(output_dir, 'friedman_test_results.csv'));
writetable(results_table, fullfile(output_dir, 'wilcoxon_posthoc_results.csv'));

fprintf('\nResults saved to: %s\n', output_dir);

%% -------------------------------
% 6. Summary
% -------------------------------
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('SUMMARY\n');
fprintf('%s\n', repmat('=', 1, 60));

fprintf('Friedman Test:\n');
if p_friedman < 1e-300
    fprintf('  chi^2 = %.4f, df = %d, p < 1e-300\n', chi2, df);
else
    fprintf('  chi^2 = %.4f, df = %d, p = %.4g\n', chi2, df, p_friedman);
end
fprintf('  Kendall''s W = %.4f\n\n', kendall_W);

significant_count = sum(significant);
fprintf('Post-hoc (Wilcoxon + Holm):\n');
fprintf('  %d/%d pairs are significant\n', significant_count, n_pairs);

non_sig_idx = find(~significant);
if ~isempty(non_sig_idx)
    fprintf('  Non-significant pair(s):\n');
    for i = 1:length(non_sig_idx)
        fprintf('    - %s (p = %.4f)\n', pair_names{non_sig_idx(i)}, p_values(non_sig_idx(i)));
    end
end

fprintf('\nAnalysis complete.\n');
