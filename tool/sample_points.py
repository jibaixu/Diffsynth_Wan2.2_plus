import torch

def add_new_points_grid_based(points_last_frame, num_new_points, grid_size=32):
    """
    使用网格占据法在未覆盖区域新增查询点。
    
    参数:
        points_last_frame: Tensor, shape [N, 2], 最后一帧存活的点坐标。
        num_new_points: int, 需要新增的点数。
        grid_size: int, 网格划分的分辨率 GxG。
        
    返回:
        new_points: Tensor, shape [num_new_points, 2], 新生成的点坐标 (范围在 0~1 之间)。
    """
    # 1. 过滤越界点 (只保留 0~1 范围内的有效点)
    valid_mask = (points_last_frame[:, 0] > 0) & (points_last_frame[:, 0] < 1) & \
                 (points_last_frame[:, 1] > 0) & (points_last_frame[:, 1] < 1)
    valid_points = points_last_frame[valid_mask]
    
    if valid_points.shape[0] == 0:
        # 如果没有存活点，直接在全局随机生成
        return torch.rand((num_new_points, 2))
    
    # 2. 映射存活点到网格索引
    # 乘以 grid_size 并向下取整，得到 [0, grid_size-1] 的索引
    grid_indices = torch.floor(valid_points * grid_size).long()
    
    # 防止极个别浮点精度问题导致的越界
    grid_indices = torch.clamp(grid_indices, 0, grid_size - 1)
    
    # 3. 生成占据掩码 (Mask)
    occupancy_mask = torch.zeros((grid_size, grid_size), dtype=torch.bool, device=points_last_frame.device)
    occupancy_mask[grid_indices[:, 0], grid_indices[:, 1]] = True
    
    # 4. 提取空位 (找到 mask 中为 False 的二维索引)
    empty_indices = (~occupancy_mask).nonzero(as_tuple=False)
    num_empty = empty_indices.shape[0]
    
    if num_empty == 0:
        print("警告：网格空间已被完全占满，无法生成新点！返回空张量。")
        return torch.empty((0, 2), device=points_last_frame.device)
        
    # 5. 采样空余网格
    # 如果需要的点数超过空余网格数，允许重复采样(有放回)；否则随机无放回采样
    replace = num_new_points > num_empty
    if replace:
        sample_idx = torch.randint(0, num_empty, (num_new_points,), device=points_last_frame.device)
    else:
        sample_idx = torch.randperm(num_empty, device=points_last_frame.device)[:num_new_points]
        
    selected_empty_indices = empty_indices[sample_idx].float()
    
    # 6. 反映射回连续坐标系
    # 加上 [0, 1) 的随机偏移量，使得新点在空余网格内部呈现随机分布，而不是死板地站在网格左上角
    random_offsets = torch.rand((num_new_points, 2), device=points_last_frame.device)
    new_points = (selected_empty_indices + random_offsets) / grid_size
    
    return new_points

# ==========================================
# 模拟测试用例
# ==========================================
if __name__ == "__main__":
    # 假设设备是 CPU，实际业务中可改为 'cuda'
    device = torch.device('cuda:0') 
    
    # 模拟最后一帧存活下来的 200 个点，故意让它们全部集中在画面的左下角 (0.0 ~ 0.5 之间)
    # 这样整个画面的右半边和上半边都是“未覆盖区域”
    surviving_points = torch.rand(200, 2, device=device) * 0.5 
    
    # 我们希望新增 50 个点来填补未覆盖区域
    num_to_add = 50
    grid_resolution = 16 # 将空间划分为 16x16 的网格
    
    new_pts = add_new_points_grid_based(surviving_points, num_to_add, grid_resolution)
    
    print(f"原有存活点数量: {surviving_points.shape[0]}")
    print(f"成功新增点数量: {new_pts.shape[0]}")
    
    # 简单验证：因为老点都在左下角，新点应该会被分配到右侧和上方
    print("-" * 30)
    print(f"老点 X 坐标均值: {surviving_points[:, 0].mean().item():.3f}")
    print(f"老点 Y 坐标均值: {surviving_points[:, 1].mean().item():.3f}")
    print(f"新点 X 坐标均值: {new_pts[:, 0].mean().item():.3f} (预期明显变大)")
    print(f"新点 Y 坐标均值: {new_pts[:, 1].mean().item():.3f} (预期明显变大)")
