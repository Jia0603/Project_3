import numpy as np
import os
from mpi4py import MPI  # 直接导入MPI，强制并行环境
from common.utils import get_default_dx, get_default_dy, get_room_grid_info
from common.boundary_config import get_boundary_conditions
from core.mpi_solver import dirichlet_neumann_iterate
import matplotlib.pyplot as plt

# 初始化MPI环境（非MPI环境将直接报错）
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

def main():
    # 固定参数（与MPI solver逻辑匹配）
    dx = get_default_dx()
    dy = get_default_dy()
    omega = 0.8
    num_iters = 10

    # 调用MPI并行求解（子进程返回None，主进程返回结果字典）
    result = dirichlet_neumann_iterate(dx, dy, omega=omega, num_iters=num_iters)

    # 子进程无结果，直接返回（不执行后续输出/保存）
    if RANK != 0:
        return

    # 提取主进程的求解结果
    u1 = result["u1"]
    u2 = result["u2"]
    u3 = result["u3"]
    gamma1 = result["gamma1"]
    gamma2 = result["gamma2"]

    '''
    # 输出尺寸信息（仅主进程显示）
    print("u1 shape:", u1.shape)
    print("u2 shape:", u2.shape)
    print("u3 shape:", u3.shape)
    print("gamma1 shape:", gamma1.shape, "gamma2 shape:", gamma2.shape)
    '''

    # 温度判断逻辑
    def judge_heating_adequacy():
        """
        判断供暖是否充足：简化逻辑（聚焦整体平均温度）
        - 打印关键温度指标（最低/平均/最高温）
        - 核心标准：平均温度 > 18℃ 即判定为供暖充足（符合冬季舒适温度要求）
        """
        # 收集所有房间的有效温度（排除可视化时填充的NaN）
        all_valid_temps = []
        for room_temp in [u1, u2, u3]:
            valid_temps = room_temp[~np.isnan(room_temp)]  # 筛选有效温度点
            all_valid_temps.extend(valid_temps)
        all_valid_temps = np.array(all_valid_temps)

        # 计算核心温度指标
        min_temp = np.min(all_valid_temps)
        avg_temp = np.mean(all_valid_temps)
        max_temp = np.max(all_valid_temps)

        # 输出结果（简洁清晰，仅主进程显示）
        print("\n" + "="*50)
        print("Task2: 公寓供暖充足性判断（简化版）")
        print("="*50)
        print(f"1. 所有房间最低温度：{min_temp:.2f}℃")  # 保留参考，体现局部情况
        print(f"2. 所有房间平均温度：{avg_temp:.2f}℃")  # 核心判断依据
        print(f"3. 所有房间最高温度：{max_temp:.2f}℃")  # 保留参考，体现加热效果
        print("-"*50)

        # 核心判断逻辑（仅看平均温度是否>18℃）
        if avg_temp > 18.0:
            print("结论：供暖充足 [√]（平均温度>18℃，满足冬季舒适温度要求）")
        else:
            print("结论：供暖不足 [×]（平均温度≤18℃，整体未达舒适温度标准）")
        
        print("="*50 + "\n")

        # 可选：返回指标供后续使用（如保存到文件）
        return {
            "min_temp": min_temp, 
            "avg_temp": avg_temp, 
            "max_temp": max_temp,
            "is_adequate": avg_temp > 18.0  # 标记是否充足
        }

    # 调用供暖判断函数（仅主进程执行）
    judge_heating_adequacy()

    # 创建输出目录并保存结果数组
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "u1.npy"), u1)
    np.save(os.path.join(out_dir, "u2.npy"), u2)
    np.save(os.path.join(out_dir, "u3.npy"), u3)
    np.save(os.path.join(out_dir, "gamma1.npy"), gamma1)
    np.save(os.path.join(out_dir, "gamma2.npy"), gamma2)
    
    
    # 主进程可视化（若有matplotlib）
    def reconstruct_full(u_interior, room_id, gamma1_arr, gamma2_arr):
        """重构包含边界的完整温度场（用于可视化）"""
        info = get_room_grid_info(room_id, dx, dy)
        Nx, Ny = info["Nx"], info["Ny"]
        bc_types, bc_values = get_boundary_conditions(room_id, gamma1_arr, gamma2_arr, dx, dy)

        # 确定求解域索引（与matrix_builder逻辑一致）
        i_start = 1 
        i_end = Nx - 1 
        j_start = 1 
        j_end = Ny - 1 

        # 初始化完整温度场并填充内部求解结果
        u_full = np.full((Nx, Ny), np.nan, dtype=float)
        # u_interior 现在是 (ny_solve, nx_solve)，需要转置为 (nx_solve, ny_solve)
        u_full[i_start:i_end, j_start:j_end] = u_interior.T

        # 辅助函数：获取边界值（兼容标量/数组）
        def get_val(v, idx):
            return v if np.isscalar(v) else v[idx]

        # 填充Dirichlet边界
        if bc_types.get("left") == "Dirichlet":
            for jg in range(j_start, j_end):
                u_full[0, jg] = get_val(bc_values["left"], jg)
        if bc_types.get("right") == "Dirichlet":
            for jg in range(j_start, j_end):
                u_full[Nx-1, jg] = get_val(bc_values["right"], jg)
        if bc_types.get("bottom") == "Dirichlet":
            for ig in range(i_start, i_end):
                u_full[ig, 0] = get_val(bc_values["bottom"], ig)
        if bc_types.get("top") == "Dirichlet":
            for ig in range(i_start, i_end):
                u_full[ig, Ny-1] = get_val(bc_values["top"], ig)

        # Neumann边界：复制相邻内部点（仅用于可视化颜色连续性）
        if bc_types.get("left") == "Neumann":
            u_full[0, j_start:j_end] = u_full[1, j_start:j_end]
        if bc_types.get("right") == "Neumann":
            u_full[Nx-1, j_start:j_end] = u_full[Nx-2, j_start:j_end]
        if bc_types.get("bottom") == "Neumann":
            u_full[i_start:i_end, 0] = u_full[i_start:i_end, 1]
        if bc_types.get("top") == "Neumann":
            u_full[i_start:i_end, Ny-1] = u_full[i_start:i_end, Ny-2]

        return u_full

    # 重构各房间完整温度场并保存图像
    u1_full = reconstruct_full(u1, "room1", gamma1, gamma2)
    u2_full = reconstruct_full(u2, "room2", gamma1, gamma2)
    u3_full = reconstruct_full(u3, "room3", gamma1, gamma2)

    # 统一色轴范围（匹配文档中边界温度：5℃-40℃）
    vmin, vmax = 5.0, 40.0

    # 保存Room1图像
    plt.imshow(u1_full.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.colorbar(); plt.title("Room1 Temperature (with boundaries)")
    room1_path = os.path.join(out_dir, "room1.png")
    plt.savefig(room1_path, dpi=200); plt.close()
    print(f"Saved figure: {room1_path}")

    # 保存Room2图像
    plt.figure(figsize=(4, 8))  # 长方形：长宽比 2:1
    plt.imshow(u2_full.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.colorbar(); plt.title("Room2 Temperature (with boundaries)")
    room2_path = os.path.join(out_dir, "room2.png")
    plt.savefig(room2_path, dpi=200); plt.close()
    print(f"Saved figure: {room2_path}")

    # 保存Room3图像
    plt.imshow(u3_full.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.colorbar(); plt.title("Room3 Temperature (with boundaries)")
    room3_path = os.path.join(out_dir, "room3.png")
    plt.savefig(room3_path, dpi=200); plt.close()
    print(f"Saved figure: {room3_path}")
    


if __name__ == "__main__":
    main()