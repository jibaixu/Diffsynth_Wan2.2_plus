为了在引入分割网络后优化点集的质量并保持 $N$ 个点的恒定规模，建议采用一种**基于优先级分配与动态掩码对齐（Mask Alignment）**的机制。

这种机制的核心是将点集视为动态资产，通过“先剪裁、后校准、再补充”的逻辑进行维护。

---

### 1. 点集分配策略 (Budget Allocation)

首先，预设一个目标分配比例 $\alpha$，以确保模型既关注操作主体（机械臂），又能捕捉环境语义：
* **$N_{arm}$ (机械臂点目标):** $\alpha \cdot N$ (建议 $\alpha = 0.6 \sim 0.7$)。
* **$N_{bg}$ (背景点目标):** $(1-\alpha) \cdot N$。

---

### 2. 详细的淘汰与校准逻辑 (Chunk Boundary Process)

在每个 Chunk 的最后一帧 $T_{end}$，执行以下流水线：

#### 第一步：硬淘汰 (Hard Pruning)
* **边界检查：** 遍历所有预测点，只要坐标 $x, y \notin [0, 1]$，立即剔除。
* **重复点去重：** 如果多个点在预测后重合度过高（像素级距离），保留一个，剔除其余点以维持采样多样性。

#### 第二步：语义校准与标签修正 (Label Re-alignment)
利用分割网络获取 $T_{end}$ 的 Mask，对剩余的有效点进行采样比对。此时需要处理**点位偏移**：

| 原始标签 | 当前 Mask 归属 | 策略 | 决策权重逻辑 |
| :--- | :--- | :--- | :--- |
| **机械臂** | **机械臂** | **保留** | 理想状态，继续追踪。 |
| **背景** | **背景** | **保留** | 理想状态。 |
| **背景** | **机械臂** | **修改标签** | 说明机械臂运动覆盖了该背景点。**权重：高**（优先保留，因为这种点往往位于接触面边缘）。 |
| **机械臂** | **背景** | **权衡：抛弃** | 说明点已经脱离机械臂表面或发生严重漂移。 |

> **针对“机械臂 $\to$ 背景”偏移的权衡建议：**
> 由于轨迹模型误差导致的“掉队点”通常不再具备代表性。建议**直接抛弃**此类点，而不是将其修改为背景点。
> *原因：* 机械臂上的采样通常需要较高的空间分布均匀性，掉落到背景的点会导致背景点集在机械臂轨迹边缘过度堆积，破坏全局分布。

---

### 3. 点集新增与初始化机制 (Point Resampling)

在完成上述淘汰和校准后，假设剩余点数为 $M$。我们需要补充 $N - M$ 个新点。

#### 场景 A：首帧机械臂缺失
如果在第一帧（或当前帧）Mask 中没有检测到机械臂：
* 将所有 $N$ 个点的配额全部用于**全局均匀格点采样**。
* 标签全部设为 "背景"。

#### 场景 B：正常补齐逻辑 (Priority Filling)
按照以下顺序填充至 $N$ 个点：

1.  **机械臂空位补偿：** * 计算 $Gap_{arm} = N_{arm} - \text{Count}(Current\_Arm\_Points)$。
    * 在机械臂 Mask 区域内，使用 **FPS (最远点采样)** 或 **均匀随机采样** 补齐。
2.  **背景空位补偿：** * 计算剩余所需总数。
    * 在非机械臂区域进行采样。
    * *优化技巧：* 优先在之前点数稀疏的区域补充，以保证空间覆盖率。

---

### 4. 算法伪代码流程

```python
def update_point_set(predicted_points, labels, current_mask, N, alpha=0.7):
    # 1. 边界淘汰
    valid_mask = (predicted_points >= 0) & (predicted_points <= 1)
    valid_mask = valid_mask.all(dim=-1)
    p_rem = predicted_points[valid_mask]
    l_rem = labels[valid_mask]

    # 2. 获取 Mask 属性 (假设 1 为 arm, 0 为 bg)
    actual_mask_values = sample_mask(current_mask, p_rem) # 在点位处采样 mask
    
    # 3. 标签修正与过滤
    final_p, final_l = [], []
    for p, old_l, curr_l in zip(p_rem, l_rem, actual_mask_values):
        if old_l == "arm" and curr_l == 0:
            continue # 机械臂掉队的点，抛弃
        else:
            final_p.append(p)
            final_l.append("arm" if curr_l == 1 else "bg")
    
    # 4. 补齐点数
    num_arm = sum(1 for l in final_l if l == "arm")
    num_to_add_arm = max(0, int(N * alpha) - num_arm)
    
    # 如果 mask 为空，特殊处理
    if not has_arm(current_mask):
        return global_grid_sample(N)

    new_arm_pts = sample_from_mask(current_mask == 1, num_to_add_arm)
    new_bg_pts = sample_from_mask(current_mask == 0, N - len(final_p) - len(new_arm_pts))
    
    return concatenate([final_p, new_arm_pts, new_bg_pts])
```

---

### 总结要点

* **对于偏移点：** 采取“严出宽进”策略。离开机械臂的点直接淘汰，被机械臂覆盖的背景点可以转正。
* **恒定点数控制：** 通过计算 $N - M$ 的差值，在每个 Chunk 结束时强制重置至 $N$。
* **首帧鲁棒性：** 只要 Mask 为空，退化为全局均匀采样，确保模型不会因为初始无目标而崩溃。