%==========================================================================
% FNSLSM  Fast Non-dominated Sorting–based Leader Selection Mechanism
% FNSLSM  基于快速非支配排序的领导者选择机制
%
% EN: Select leaders by jointly considering solution quality (fitness) and
%     distribution (relative distance), using Pareto dominance.
% CN: 通过同时考虑解质量（适应度）与分布性（相对距离），基于帕累托支配关系
%     选择领导者集合，以缓解单一适应度贪婪选择导致的过度聚集。
%
% Inputs / 输入:
%   Positions   : N×D population positions / 种群位置矩阵
%   Fitness     : N×1 (or 1×N) objective values (minimization) / 适应度（最小化）
%   num_leaders : number of leaders / 领导者数量
%
% Outputs / 输出:
%   leader_indices   : indices of leaders (w.r.t. original population) / 领导者索引
%   follower_indices : remaining indices (reordered) / 跟随者索引（已重排）
%==========================================================================
function [leader_indices, follower_indices] = FNSLSM(Positions, Fitness, num_leaders)
    N = size(Positions, 1);

    % EN: Force Fitness to a column vector for consistent indexing.
    % CN: 强制将 Fitness 转为列向量，便于统一处理。
    Fitness = reshape(Fitness, [], 1); % N×1

    %----------------------------------------------------------------------
    % Step 1: Sort by Fitness (ascending, minimization).
    % 第 1 步：按适应度升序排序（最小化问题）。
    %----------------------------------------------------------------------
    [Fitness_sorted, sort_idx] = sort(Fitness);
    sorted_Positions = Positions(sort_idx, :);

    %----------------------------------------------------------------------
    % Step 2: Directional diversity (minimum distance to better individuals).
    % 第 2 步：多样性度量（到更优个体集合的最小欧氏距离）。
    %----------------------------------------------------------------------
    Diversity_sorted = zeros(N, 1);
    %K_limit = min(N-1, 20);

    for i = 2:N
        current_pos = sorted_Positions(i, :);
        start_idx = max(1, i - 1);
        prev_positions = sorted_Positions(start_idx : i-1, :);
        dists = sqrt(sum((prev_positions - current_pos).^2, 2));
        Diversity_sorted(i) = min(dists);
    end

    % EN: Assign a large diversity value to the best-fitness individual as a reference.
    % CN: 对适应度最优个体设置较大的相对距离值，用作参考个体。
    Diversity_sorted(1) = inf;
    max_div = max(Diversity_sorted(isfinite(Diversity_sorted)));
    if isempty(max_div) || max_div == 0, max_div = 1; end
    Diversity_sorted(1) = max_div * 1.1;

    %----------------------------------------------------------------------
    % Step 3: Construct bi-objective vectors [fitness, -diversity] (min-min).
    % 第 3 步：构建双目标 [适应度, -多样性]，统一为最小化形式。
    %----------------------------------------------------------------------
    Objectives_sorted = [Fitness_sorted, -Diversity_sorted];

    %----------------------------------------------------------------------
    % Step 4: Fast non-dominated sorting (NSGA-II style).
    % 第 4 步：快速非支配排序，得到多层帕累托前沿。
    %----------------------------------------------------------------------
    Fronts_sorted = NonDominatedSorting(Objectives_sorted);

    %----------------------------------------------------------------------
    % Step 5: Fill leaders front-by-front; truncate by ideal-point distance.
    % 第 5 步：按前沿逐层填充领导者；最后一层超额时按理想点距离截断。
    %----------------------------------------------------------------------
    leader_sorted_idx = [];
    front_idx = 1;

    while length(leader_sorted_idx) + length(Fronts_sorted{front_idx}) <= num_leaders
        leader_sorted_idx = [leader_sorted_idx; Fronts_sorted{front_idx}];
        front_idx = front_idx + 1;
        if front_idx > length(Fronts_sorted), break; end
    end

    remaining_slots = num_leaders - length(leader_sorted_idx);
    if remaining_slots > 0 && front_idx <= length(Fronts_sorted)
        last_front_indices = Fronts_sorted{front_idx};
        objs = Objectives_sorted(last_front_indices, :);

        % EN: Normalize objectives within the critical front, then select closest to the ideal point.
        % CN: 对关键前沿内目标归一化，并选择到理想点距离最小的个体。
        f_min = min(objs, [], 1);
        f_max = max(objs, [], 1);
        range = f_max - f_min;
        range(range==0) = 1;
        objs_norm = (objs - f_min) ./ range;

        dist_to_ideal = sum(objs_norm.^2, 2);
        [~, sorted_score_idx] = sort(dist_to_ideal, 'ascend');
        leader_sorted_idx = [leader_sorted_idx; last_front_indices(sorted_score_idx(1:remaining_slots))];
    end

    %----------------------------------------------------------------------
    % Step 6: Map back to original indices.
    % 第 6 步：映射回原始种群索引空间。
    %----------------------------------------------------------------------
    leader_indices = sort_idx(leader_sorted_idx);
    follower_indices = setdiff(1:N, leader_indices);

    %----------------------------------------------------------------------
    % Step 7: Re-order followers by (front rank, fitness) for stable updates.
    % 第 7 步：按（前沿层级、适应度）重排跟随者，提高后续更新稳定性。
    %
    % NOTE (EN):
    %   front_rank is computed in the sorted-index space; follower_indices are
    %   in the original-index space. If you require strict consistency, build
    %   an inverse map from original indices to sorted indices before indexing.
    % NOTE (CN):
    %   front_rank 在“排序索引空间”定义，而 follower_indices 属于“原始索引空间”。
    %   若需严格一致性，应先构建“原始索引→排序索引”的逆映射再取 front_rank。
    %----------------------------------------------------------------------
    if ~isempty(follower_indices)
        front_rank = zeros(N,1);
        for f = 1:length(Fronts_sorted)
            front_rank(Fronts_sorted{f}) = f;
        end

        follower_indices = reshape(follower_indices, [], 1);

        rank_col1 = front_rank(follower_indices);
        rank_col1 = reshape(rank_col1, [], 1);

        fit_col2  = Fitness(follower_indices);
        fit_col2  = reshape(fit_col2, [], 1);

        rank_data = [rank_col1, fit_col2];

        [~, order] = sortrows(rank_data, [1, 2]);
        follower_indices = follower_indices(order);
    end

    % EN: Format outputs as row vectors for downstream concatenation.
    % CN: 输出统一为行向量，便于后续拼接与索引。
    leader_indices = reshape(leader_indices, 1, []);
    follower_indices = reshape(follower_indices, 1, []);
end

%==========================================================================
% NonDominatedSorting  Fast non-dominated sorting (all objectives minimized)
% NonDominatedSorting  快速非支配排序（默认所有目标均为最小化）
%==========================================================================
function Fronts = NonDominatedSorting(Objectives)
    [N, ~] = size(Objectives);
    Fronts = {};

    % EN: domination_count(p) = number of solutions dominating p.
    % CN: domination_count(p) = 支配 p 的个体数量。
    domination_count = zeros(N, 1);

    % EN: dominated_set{p} stores solutions dominated by p.
    % CN: dominated_set{p} 存储被 p 支配的个体集合。
    dominated_set = cell(N, 1);

    for p = 1:N
        for q = p+1:N
            p_dominates_q = all(Objectives(p, :) <= Objectives(q, :)) && any(Objectives(p, :) < Objectives(q, :));
            q_dominates_p = all(Objectives(q, :) <= Objectives(p, :)) && any(Objectives(q, :) < Objectives(p, :));
            if p_dominates_q
                dominated_set{p} = [dominated_set{p}, q];
                domination_count(q) = domination_count(q) + 1;
            elseif q_dominates_p
                dominated_set{q} = [dominated_set{q}, p];
                domination_count(p) = domination_count(p) + 1;
            end
        end
    end

    % EN/CN: First front contains all non-dominated solutions.
    % EN/CN: 第一前沿包含所有非支配解。
    F1_indices = find(domination_count == 0);
    Fronts{1} = F1_indices;

    front_idx = 1;
    while ~isempty(Fronts{front_idx})
        next_front = [];
        for p = Fronts{front_idx}'
            for q = dominated_set{p}
                domination_count(q) = domination_count(q) - 1;
                if domination_count(q) == 0
                    next_front = [next_front, q];
                end
            end
        end
        front_idx = front_idx + 1;
        Fronts{front_idx} = next_front';
    end
end
