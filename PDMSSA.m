%==========================================================================
% PDMSSA  Pareto-Guided Dynamic Memetic Salp Swarm Algorithm (core loop)
% PDMSSA  帕累托引导的动态模因樽海鞘群算法（主循环）
%
% EN: This file contains the main optimization loop integrating:
%     (i)  Pareto-based leader selection (FNSLSM),
%     (ii) Scale-free (BA) network neighborhood guidance for followers, and
%     (iii) State-aware directional tabu search (SADTSM) as a conditional local intensifier.
% CN: 本文件实现 PDMSSA 的主循环，包含：
%     (i)  基于非支配排序的领导者选择（FNSLSM），
%     (ii) 基于 BA 无标度网络的邻域协同引导（跟随者更新），
%     (iii) 以 SADTSM 为核心的状态触发式方向禁忌搜索（局部强化）。
%
% Notes / 说明:
%   - fobj is assumed to be a minimization objective. / 默认 fobj 为最小化目标。
%==========================================================================

function [FoodFitness,FoodPosition,Convergence_curve] = PDMSSA(N,Max_iter,lb,ub,dim,fobj)


% EN: 1) Initialization.
% CN: 1) 初始化。
SalpPositions = initialization(N,dim,ub,lb);
FoodPosition   = zeros(1,dim);
FoodFitness    = inf;

% EN: Evaluate initial fitness.
% CN: 计算初始适应度。
SalpFitness = zeros(1,N);
for i=1:size(SalpPositions,1)
    SalpFitness(1,i) = fobj(SalpPositions(i,:));
end

[sorted_salps_fitness,sorted_indexes] = sort(SalpFitness);
SalpPositions = SalpPositions(sorted_indexes, :);
FoodPosition = SalpPositions(1,:);
FoodFitness  = sorted_salps_fitness(1);

Convergence_curve = zeros(1,Max_iter);
Convergence_curve(1) = FoodFitness;

history_len = 10; % 轨迹滑动窗口大小
Food_History = repmat(FoodPosition, history_len, 1); % 初始化历史记录
history_ptr = 1;  % 环形缓冲区指针
TS_cooldown = 0;  % 触发冷却时间


% EN: Main loop.
% CN: 主循环。
l = 2;
while l < Max_iter + 1

    c1 = 2 * exp(-(4 * l / Max_iter)^2);
    
    % EN: Leader selection + population reordering.
    % CN: 领导者选择与种群重排。
    [leader_indices, follower_indices] = FNSLSM(SalpPositions, SalpFitness, ceil(N / 2));
    combined = [leader_indices, follower_indices];
    SalpPositions = SalpPositions(combined,:);
    SalpFitness   = SalpFitness(combined);
    SalpPositions_old = SalpPositions; 
    SalpFitness_old = SalpFitness;


    % EN: (Re)build BA scale-free network periodically.
    % CN: 按固定周期重建 BA 无标度网络。
    if l ==  2 || mod(l,200) == 0
        adj = Build_ScaleFreeNetwork(N);
    end

    [~,Pbest_idx] = min(SalpFitness); %当前种群的最优位置
    Pbest = SalpPositions(Pbest_idx,:); 

    % EN: Population position update.
    % CN: 种群位置更新。
    for i = 1 : N
        tag = (rand(1,dim) < 0.5) * 2 - 1;
        c2 = rand(1,dim);

        if i <= ceil(N / 2)  % 领导者更新

           SalpPositions(i,:) = FoodPosition + tag .* c1 .* ((ub - lb) .* c2 + lb);
            
        else   % 跟随者更新
             % ---- 找最佳邻居 ----
            neighbor = find_best_neighbor(i, adj, SalpPositions_old, SalpFitness_old);
            SalpPositions(i, :) = (SalpPositions(i, :) + SalpPositions(i - 1,:)) / 2 ...
                + rand(1,dim) .* (neighbor - SalpPositions(i,:)) + ...
                rand(1,dim) .* ((FoodPosition + Pbest) / 2 - SalpPositions(i,:));
        end

        SalpPositions(i, :) = handlebound(SalpPositions(i, :),lb,ub);%处理边界
        SalpFitness(i) = fobj(SalpPositions(i,:));
    end

    [tmp,idx]= min(SalpFitness);
    if tmp < FoodFitness %如果更优 更新历史最优
        FoodPosition = SalpPositions(idx,:);
        FoodFitness = tmp;
    end
    

    
% EN: Update sliding window history of the global best.
% CN: 更新历史最优轨迹滑动窗口。
    Food_History(history_ptr, :) = FoodPosition;
    history_ptr = mod(history_ptr, history_len) + 1;
    
    trigger_TS = false; %是否触发禁忌搜索
    
% EN: Detect stagnation only if cooldown ends and history is available.
% CN: 仅在冷却结束且历史记录充足时检测停滞。
    if TS_cooldown == 0 && l > history_len
        
        % A. 计算累积路径长度 (Path Length) - 粒子实际跑过的路程
        L_path = 0;
        for h = 1 : history_len-1
            % 倒推索引处理环形缓冲
            idx_curr = mod(history_ptr - 1 - h - 1, history_len) + 1; 
            idx_prev = mod(history_ptr - 1 - h - 1 - 1, history_len) + 1;
            dist = norm(Food_History(idx_curr, :) - Food_History(idx_prev, :));
            L_path = L_path + dist;
        end
        
        % B. 计算净位移 (Net Displacement) - 起点到终点的直线距离
        idx_now = mod(history_ptr - 1 - 1, history_len) + 1;
        idx_old = mod(history_ptr - 1 - history_len, history_len) + 1;
        D_net = norm(Food_History(idx_now, :) - Food_History(idx_old, :));
        
        % C. 计算曲率指数 (Tortuosity Index, TI)
        % TI 接近 1 表示高效搜索(直线)，TI 接近 0 表示低效震荡(原地打转)
        TI = D_net / (L_path + 1e-10);
        
% EN: D) Fitness improvement between consecutive iterations.
% CN: D) 计算适应度改善幅度。
        fit_improvement = abs(Convergence_curve(l-1) - FoodFitness) ;%/  abs(FoodFitness) + 1e-9;
        
% EN: Trigger criteria (trajectory + fitness stagnation).
% CN: 触发判据（轨迹 + 适应度双条件）。
        % 条件1: TI < 0.1 (路径极其卷曲，做了很多无用功)
        % 条件2: 适应度改善极小 (说明真的卡住了，而不是在绕坑边缘优化)
        if TI < 0.1 && fit_improvement < 1e-5
            trigger_TS = true;
        end
    end
    
    % EN: Execute tabu search if triggered.
    % CN: 若触发则执行禁忌搜索。
    if trigger_TS
       
        
        % EN: Tabu search as a rescue operator to escape stagnation.
        % CN: 禁忌搜索作为“救援”算子以跳出停滞。
        [FoodPosition, FoodFitness, ~, ~] = ...
            FNSLSM(SalpPositions, FoodPosition, SalpFitness, fobj, lb, ub, 50);
        % EN: Enter cooldown to avoid triggering every iteration.
        % CN: 进入冷却期，避免每代连续触发。
        TS_cooldown = 50;
        
        % 更新收敛曲线 (因为 TS 可能改进了 Food)
        Convergence_curve(l) = FoodFitness; 
        
        % 触发后最好重置历史记录，避免旧的震荡数据影响下一次判断
        Food_History = repmat(FoodPosition, history_len, 1);
    end
    
    % EN: Cooldown countdown.
    % CN: 冷却倒计时。
    if TS_cooldown > 0
        TS_cooldown = TS_cooldown - 1;
    end
    % =========================================================================

    Convergence_curve(l) = FoodFitness;
    l = l + 1;
end
end

function [adj, sorted_degrees] = Build_ScaleFreeNetwork(N)
% Build_ScaleFreeNetwork 构建基于BA模型的无标度网络
% 输入:
%   N: 种群规模 (节点总数)
% 输出:
%   adj: 排序后的邻接矩阵 (N x N)，节点1对应度最大的节点
%   sorted_degrees: 排序后的度数向量

    % "network is initiated from a fully-connected undirected subgraph containing 5 nodes" 
    %初始化5个节点
    M0 = 5; 
    
    % "each newly added node stochastically establishes 3 connections" 
    m = 3;  
    
    % 确保 N 大于 M0
    if N < M0
        error('种群规模 N 必须大于初始节点数 M0 (5)');
    end
    
    % 初始化邻接矩阵
    raw_adj = zeros(N, N);
    
    % ---------------------------------------------------------
    % 2. 初始化全连接网络 (Step 1 in Algorithm 3) [cite: 418]
    % ---------------------------------------------------------
    for i = 1:M0
        for j = i+1:M0
            raw_adj(i,j) = 1;
            raw_adj(j,i) = 1;
        end
    end
    
    % 记录当前节点的度数
    degrees = sum(raw_adj, 2);
    
    % ---------------------------------------------------------
    % 3. 增长机制 (BA Model Growth) 
    % ---------------------------------------------------------
    % 逐个添加新节点
    for newNode = M0+1 : N
        
        % 获取当前已存在的节点列表
        existingNodes = 1:(newNode-1);
        
        % 计算连接概率 Pj = Dj / sum(D)
        % "degree-based attachment probability Pj... Dj denotes the degree of node j" 
        currentDegrees = degrees(existingNodes);
        totalDegree = sum(currentDegrees);
        
        if totalDegree == 0
            probs = ones(size(existingNodes)) / length(existingNodes);
        else
            probs = currentDegrees / totalDegree;
        end
        
        % -----------------------------------------------------
        % 赌轮盘选择 (Roulette Wheel Selection) 选择 m 个不同的节点
        % -----------------------------------------------------
        targets = zeros(1, m);
        count = 0;
        
        % 为了避免重复连接同一个节点，进行不放回抽样逻辑
        temp_probs = probs; 
        
        while count < m
            % 生成随机数
            r = rand();
            
            % 计算累积概率
            cumProbs = cumsum(temp_probs / sum(temp_probs)); % 归一化确保总和为1
            
            % 找到选中的索引
            selectedIdx = find(cumProbs >= r, 1, 'first');
            selectedNode = existingNodes(selectedIdx);
            
            % 记录连接目标
            count = count + 1;
            targets(count) = selectedNode;
            
            % 将选中节点的概率置0，避免再次选中 (实现无放回)
            temp_probs(selectedIdx) = 0;
        end
        
        % -----------------------------------------------------
        % 更新网络连接
        % -----------------------------------------------------
        for tNode = targets
            raw_adj(newNode, tNode) = 1;
            raw_adj(tNode, newNode) = 1;
            
            % 更新度数
            degrees(newNode) = degrees(newNode) + 1;
            degrees(tNode) = degrees(tNode) + 1;
        end
    end
    
    % ---------------------------------------------------------
    % 4. 节点重排序 (Mapping Strategy)
    % ---------------------------------------------------------
    % 这意味着度数最大的节点被重新编号为1，度数最小的为N。
    
    [sorted_degrees, sortIdx] = sort(degrees, 'descend');
    
    % 根据排序后的索引重构邻接矩阵
    adj = raw_adj(sortIdx, sortIdx);
end


function neighbor_pos = find_best_neighbor(i, adj, Positions, Fitness)
%
% 输入:
%   i         : 当前个体的索引（直接对应网络节点索引 ）
%   adj       : 无标度网络的邻接矩阵
%   Positions : 种群位置矩阵 (N x D)
%   Fitness   : 种群适应度向量 (N x 1)
%
% 输出:
%   neighbor_pos : 最佳邻居的位置向量
% 

    % --- 1. 执行 BFS 获取一跳和二跳邻居 ---
    % 范围为 "first-hop and second-hop" 
    max_hops = 2; 
    neighbor_indices = bfs(adj, i, max_hops);

    % --- 2. 贪婪择优 (Greedy Selection) ---
    if isempty(neighbor_indices)
        % 如果网络极其稀疏导致无邻居（理论上BA网络不会发生），返回自身
        neighbor_pos = Positions(i, :);
    else
        % 获取所有邻居的适应度
        neigh_fitness = Fitness(neighbor_indices);
        
        % 找到适应度最小（最好）的那个邻居的索引
        [~, best_local_idx] = min(neigh_fitness);
        best_global_idx = neighbor_indices(best_local_idx);
        
        % 获取其位置
        neighbor_pos = Positions(best_global_idx, :);
    end
end

% -----------------------------------------------------------
% 子函数: 广度优先搜索 (BFS)
% -----------------------------------------------------------
function all_neighbors = bfs(adj, start_node, max_hops)
    % 初始化
    N = size(adj, 1);
    visited = false(1, N);
    visited(start_node) = true; % 标记起始点，避免将自己算作邻居
    
    current_layer = start_node; % 当前层节点
    all_neighbors = [];         % 存储所有遍历到的邻居
    
    % 按跳数（层级）遍历，循环次数为 max_hops (即 2)
    for hop = 1:max_hops
        next_layer = [];
        
        % 遍历当前层的所有节点
        for u = current_layer
            % 查找 u 的所有直接连接节点（adj中为1的位置）
            % adj(u, :) > 0 找出所有相连的节点索引
            connected_nodes = find(adj(u, :) > 0);
            
            for v = connected_nodes
                if ~visited(v)
                    visited(v) = true;       % 标记为已访问
                    next_layer(end+1) = v;   % 加入下一层队列
                    all_neighbors(end+1) = v;% 记录到结果列表中
                end
            end
        end
        
        % 更新当前层，准备进入下一跳
        current_layer = next_layer;
        
        % 如果下一层没有节点，说明网络不连通，提前结束
        if isempty(current_layer)
            break;
        end
    end
end

function [salp] = handlebound(salp, lb, ub)
    % 确保 lb、ub 与 salp 维度匹配（假设均为向量）
    % salp：待处理的个体（向量）
    % lb：各维度的下界（向量，与 salp 同长度）
    % ub：各维度的上界（向量，与 salp 同长度）
    
    for i = 1 : size(salp,2)
        if salp(i) > ub(i)
            % 超出第 i 维上界时，使用第 i 维的 ub 反弹
            salp(i) = 2 * ub(i) - salp(i);
        elseif salp(i) < lb(i)
            % 超出第 i 维下界时，使用第 i 维的 lb 反弹
            salp(i) = 2 * lb(i) - salp(i);
        end
        % 若在范围内，保持原值不变
    end
end
