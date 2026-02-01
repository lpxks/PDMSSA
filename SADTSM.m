%==========================================================================
% SADTSM  State-Aware Directional Tabu Search Module (for PDMSSA)
% SADTSM  状态感知的方向禁忌搜索模块（用于 PDMSSA）
%
% EN: This module performs a conditional tabu-search refinement. It stores
%     historically ineffective exploration directions (unit vectors) rather than
%     forbidding specific solution coordinates, and adapts tabu tenure based on
%     an improvement-strength indicator.
% CN: 本模块实现状态触发式方向禁忌搜索：记录历史低效的探索方向（单位向量），
%     而非禁忌具体位置，并依据改进强度自适应调整禁忌期限。
%
% Inputs / 输入:
%   salp         : N×D population / 当前种群
%   Gbest        : 1×D global best position / 历史全局最优
%   salpFitness  : N×1 fitness values / 种群适应度（最小化）
%   fobj         : objective handle / 目标函数句柄
%   lb, ub       : bounds (scalar or 1×D) / 下界与上界
%   TS_max_iter  : inner iterations of tabu search / 禁忌搜索内循环次数
%
% Outputs / 输出:
%   newFoodPosition    : refined best position / 改进后的最优位置
%   bestFitnessFound   : refined best fitness / 改进后的最优适应度
%   salp, salpFitness  : possibly partially updated population / 可选的部分更新种群
%==========================================================================

function [newFoodPosition,bestFitnessFound,salp,salpFitness] = SADTSM(salp, Gbest, salpFitness, fobj, lb, ub,TS_max_iter)

[N, D] = size(salp);

% EN: 1) Core parameters of tabu search.
% CN: 1) 禁忌搜索核心参数。
neighborhood_size = ceil(N * 0.2) + ceil(N * 0.1); 

% 自适应禁忌锥参数
theta_min = 0.70; 
theta_max = 0.99; 

% EN: 2) Initialization.
% CN: 2) 初始化。
if isscalar(lb), lb = lb * ones(1, D); end
if isscalar(ub), ub = ub * ones(1, D); end

% EN: Initialize tabu list (direction + expiry).
% CN: 初始化禁忌表（方向 + 过期迭代）。
tabuList = struct('direction', {}, 'expireIter', {});

% EN: Parameters for partial population update during TS.
% CN: 禁忌搜索中的种群部分更新参数。
elite_rate = 0.08;         

% EN: Historical global best (long-term elite).
% CN: 历史全局最优（长期精英）。
bestFitnessFound = fobj(Gbest);
bestSolutionFound = Gbest;

eps_safe = 1e-12;
Lmin = 5; Lmax = 10; 
norm_range = norm(ub - lb) + eps_safe;

% EN: Initialize momentum vector for direction smoothing.
% CN: 初始化动量向量，用于方向平滑与惯性利用。
momentum_vector = zeros(1, D); 
momentum_weight = 0.5; % 初始动量权重

% EN: 3) Main tabu-search loop.
% CN: 3) 禁忌搜索主循环。
for t = 1 : TS_max_iter
    
    % EN: Current population best (short-term elite).
    % CN: 当前种群最优（短期精英）。
    [~,idxs] = sort(salpFitness);
    Pbest = salp(idxs(1),:);
    
    % EN: Adapt cosine-similarity threshold based on diversity.
    % CN: 基于多样性自适应调整余弦阈值 theta_cos。
    [diversity, diversity_max_possible] = calculate_diversity(salp, lb, ub);
    diversity_norm = diversity / (diversity_max_possible + eps_safe); 
    theta_cos = theta_min + (theta_max - theta_min) * diversity_norm;

    update_prob = 0.2 + 0.8 * (1 - diversity_norm); % 自适应更新概率
    
    is_move_found = false;

    % EN: Fitness-weighted centroid (for quasi-opposition learning).
    % CN: 适应度加权质心（用于准反向学习）。
    fworst = max(salpFitness) * ones(1, N);
    weights = fworst - salpFitness(1:N);
    weights_sum = sum(weights);
    if weights_sum == 0, weights_sum = 1; end % 防止除零
    weights = weights / weights_sum;
    weights = (ones(D,1) * weights)';  
    Gcenter = sum(weights .* salp(1:N,:), 1);

    % EN: Cache the best admissible move in this TS iteration.
    % CN: 缓存本次 TS 迭代中最优的可接受移动。
    best_move_direction = zeros(1, D);
    best_candidate_fitness = inf; 
    best_candidate_in_iter = [];
    best_base_solution = [];
    best_base_fitness = inf;
    
    % EN: Generate and evaluate neighborhood candidates.
    % CN: 生成并评估邻域候选解。
    for j = 1 : neighborhood_size
          
        if j <= neighborhood_size * 0.2
            candi_idx = idxs(j); % 精英个体
        else 
            candi_idx = idxs(randi([ceil(neighborhood_size * 0.2), N])); % 非精英个体
        end
        currentSolution = salp(candi_idx,:);

        % EN: (a) Differential perturbation with momentum term.
        % CN: (a) 含动量项的差分扰动生成。
        ids = randperm(N,2);
        
        r1 = rand(1,D); r2 = rand(1,D); r3 = rand(1,D);
        %r4 用于控制动量项  % EN: Added/modified behavior as marked. | CN: 如标注为新增/修改。
        r4 = rand(1,D); 
        
        % --- 加入 momentum_vector * momentum_weight ---  % EN: Added behavior as marked. | CN: 如标注为新增/修改。
        candidateSolution = currentSolution + r1 .* (salp(ids(1),:) - salp(ids(2),:)) ...
                           + r2 .* (Gbest - currentSolution) ...
                           + r3 .* (Pbest - currentSolution) ...
                           + r4 .* momentum_vector * momentum_weight; 
        
        candidateSolution = handlebound(candidateSolution,lb,ub);
        candidateFitness = fobj(candidateSolution);
        

        
        % EN: Quasi-opposition learning with probabilistic dimension selection.
        % CN: 准反向学习：按概率选择维度进行反向采样。
        inverse_point = 2.0 * Gcenter - currentSolution;

        point_A = Gcenter;
        point_B = inverse_point;
        
        full_opp_candi = currentSolution;
        Jr = rand(1, D) < 0.5;   % 自然随机选择
        if ~any(Jr)
            Jr(randperm(D, ceil(D*0.2))) = true;
        end
        % 获取选中维度的上下界，确保 rand 正确插值
        idx = find(Jr); % 获取逻辑索引对应的下标，方便后续操作
        min_p = min(point_A(idx), point_B(idx));
        max_p = max(point_A(idx), point_B(idx));
        
        % 在 [Gcenter, Inverse] 之间随机采样
        full_opp_candi(idx) = min_p + rand(1, length(idx)) .* (max_p - min_p);
        
        % ... 后面的边界处理不变 ...
        full_opp_candi = handlebound(full_opp_candi, lb, ub);
        full_opp_fit = fobj(full_opp_candi);
        % EN: Greedy selection between the two candidates.
        % CN: 在两类候选中贪婪择优。
         if full_opp_fit < candidateFitness
             candidateSolution = full_opp_candi;
             candidateFitness = full_opp_fit;
         end

        % EN: (c) Compute move direction (unit vector).
        % CN: (c) 计算移动方向（单位向量）。
        move_vector = candidateSolution - currentSolution;
        if norm(move_vector) < 1e-9
            continue;
        end
        move_direction_hat = move_vector / norm(move_vector); % 单位方向化

        % EN: (d) Directional tabu check via cosine similarity.
        % CN: (d) 方向禁忌判定（余弦相似度）。
        is_tabu = false;
        for k = 1:length(tabuList)
            if t < tabuList(k).expireIter
                similarity = dot(move_direction_hat, tabuList(k).direction);
                if similarity > theta_cos 
                    is_tabu = true;
                    break;
                end
            end
        end

        % EN: (e) Aspiration: allow tabu move if it improves the global best.
        % CN: (e) 赦免准则：若优于全局最优则允许禁忌移动。
        if is_tabu && candidateFitness < bestFitnessFound
            is_tabu = false;
        end

        % EN: (f) Select the best non-tabu candidate in this iteration.
        % CN: (f) 选取本轮最优非禁忌候选。
        if ~is_tabu && candidateFitness < best_candidate_fitness
            best_candidate_in_iter = candidateSolution;
            best_candidate_fitness = candidateFitness;
            best_move_direction = move_direction_hat;
            best_base_solution = currentSolution;
            best_base_fitness = salpFitness(candi_idx);
            is_move_found = true;
        end
        
        % EN: Partial population update to inject improvements without full replacement.
        % CN: 种群部分更新：在不完全替换的前提下注入改进。
        if ~is_tabu
            if j <= elite_rate * neighborhood_size
                continue; % 精英不更新
            else
                % 非精英个体按概率更新
                if rand() < update_prob && candidateFitness < salpFitness(candi_idx)
                    salp(candi_idx,:) = candidateSolution;
                    salpFitness(candi_idx) = candidateFitness;
                end
            end
        end
    end

% EN: Update best solution and tabu memory.
% CN: 更新最优解与禁忌记忆。
    if is_move_found
        norm_v = norm(best_candidate_in_iter - best_base_solution);
        M = norm_v / norm_range;

        f_curr = best_base_fitness;
        f_cand = best_candidate_fitness;

        if f_curr == 0
            I = 0.01;
        else
            delta = f_curr - f_cand;
            if delta > 0
                I = delta / (abs(f_curr) + eps_safe);
            else
                I = 0.01;
            end
        end

        w = 0.5; 
        IF = M * w + I * (1 - w);
        IF = min(max(IF, 0), 1);

        % EN: Exempt highly beneficial directions from being tabu (avoid over-restriction).
        % CN: 对高收益方向进行豁免，避免过度限制有利下降方向。
        tabu_skip_threshold = 0.75; 

        if IF < tabu_skip_threshold
            % 只有当改进幅度没那么大时，才加入禁忌表
            
            % 逻辑: 改进越大(IF大) -> 禁忌越短
            tenure = Lmin + round((Lmax - Lmin) * (1 - IF));
            
            newTabu.direction = best_move_direction;
            newTabu.expireIter = t + tenure;
            tabuList(end+1) = newTabu;
        else
            % IF >= 0.75，触发超级豁免，不加入禁忌表，允许惯性冲刺
            % disp('Great move! Tabu skipped.'); 
        end

        % 更新当前解 
        currentSolution = best_candidate_in_iter;
        currentFitness = best_candidate_fitness;
        
        % EN: Update momentum based on the accepted move.
        % CN: 根据已接受移动更新动量。
        % 1. 计算本次真实位移
        current_move_vec = best_candidate_in_iter - best_base_solution;
        
        % 2. 动量更新 (平滑滤波: 30%历史 + 70%当前)
        momentum_vector = 0.3 * momentum_vector + 0.7 * current_move_vec;
        
        % 3. 动态调整动量权重
        if IF > 0.75
             momentum_weight = 0.8; % 改进巨大，加大下一次利用动量的比例
        else
             momentum_weight = 0.5; % 恢复正常
        end

        % 更新全局最优
        if currentFitness < bestFitnessFound
            bestSolutionFound = currentSolution;
            bestFitnessFound = currentFitness;
            Gbest = currentSolution; 
        end
    else
        % EN: No improving move found -> decay momentum.
        % CN: 未找到改进移动（停滞）→ 衰减动量。
        momentum_vector = momentum_vector * 0.5;
        momentum_weight = 0.1;
    end

% EN: Remove expired tabu entries.
% CN: 清理过期的禁忌项。
    if ~isempty(tabuList)
        active_indices = [tabuList.expireIter] > t;
        tabuList = tabuList(active_indices);
    end
end

% EN: 4) Return results.
% CN: 4) 返回结果。
newFoodPosition = bestSolutionFound;

end


% EN: Helper: estimate population diversity.
% CN: 辅助函数：计算种群多样性。
function [diversity, max_diversity] = calculate_diversity(salp, lb, ub)
    centroid = mean(salp, 1);
    distances = sqrt(sum((salp - centroid).^2, 2));
    diversity = mean(distances);

    space_range = ub - lb;
    max_diversity = norm(space_range) / 2; % 修正为空间对角线的一半，更科学
    if max_diversity == 0, max_diversity = 1; end
end
