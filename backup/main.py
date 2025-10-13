import numpy as np
import os
from mpi4py import MPI 
from common.utils import get_default_dx, get_default_dy, get_room_grid_info
from common.boundary_config import get_boundary_conditions
from core.mpi_solver import dirichlet_neumann_iterate
from core.ext_mpi_solver import ext_dirichlet_neumann_iterate
from core.visualizer import visualize_all_rooms
import matplotlib.pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Apartment Temperature Simulation')
    parser.add_argument('--new-apartment', '-n', 
                       action='store_true',
                       help='Run simulation for new apartment layout (default: old apartment)')
    return parser.parse_args()


# 初始化MPI环境（非MPI环境将直接报错）
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

def main(apt_new=False):
    # 固定参数（与MPI solver逻辑匹配）
    dx = get_default_dx()
    dy = get_default_dy()
    omega = 0.8
    num_iters = 10

    # 调用MPI并行求解（子进程返回None，主进程返回结果字典）
    if apt_new == False:
        result = dirichlet_neumann_iterate(dx, dy, omega=omega, num_iters=num_iters)
    else:
        result = ext_dirichlet_neumann_iterate(dx, dy, omega=omega, num_iters=num_iters)

    # 子进程无结果，直接返回（不执行后续输出/保存）
    if RANK != 0:
        return

    # 提取主进程的求解结果
    u1 = result["u1"]
    u2 = result["u2"]
    u3 = result["u3"]
    gamma1 = result["gamma1"]
    gamma2 = result["gamma2"]
    gamma3 = None
    if apt_new == True:
        u4 = result["u4"]
        gamma3 = result["gamma3"]

    '''
    # 输出尺寸信息（仅主进程显示）
    print("u1 shape:", u1.shape)
    print("u2 shape:", u2.shape)
    print("u3 shape:", u3.shape)
    print("gamma1 shape:", gamma1.shape, "gamma2 shape:", gamma2.shape)
    '''

    all_valid_temps = []
    if apt_new == False:
        u = [u1, u2, u3]
    else:
        u = [u1, u2, u3, u4]
    for room_temp in u:
        valid_temps = room_temp[~np.isnan(room_temp)] 
        all_valid_temps.extend(valid_temps)
    all_valid_temps = np.array(all_valid_temps)

    min_temp = np.min(all_valid_temps)
    avg_temp = np.mean(all_valid_temps)
    max_temp = np.max(all_valid_temps)


    print("\n" + "="*50)
    print("Is the heating in the flat adequate?")
    print("="*50)
    print(f"1. Lowest temp:{min_temp:.2f} ˚C")  
    print(f"2. Averaged temp:{avg_temp:.2f}˚C")  
    print(f"3. Highest temp:{max_temp:.2f}˚C")  
    print("-"*50)

    if avg_temp > 18.0:
        print("Averged tempture > 18˚C. The heating in the flat is adequate.")
    else:
        print("Averged tempture < 18˚C. The flat is not warm enough.")
            
    print("="*50 + "\n")

    # 创建输出目录并保存结果数组
    if apt_new == False:
        out_dir = os.path.join(os.path.dirname(__file__), "output")
    else:
        out_dir = os.path.join(os.path.dirname(__file__), "ext_output")

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "u1.npy"), u1)
    np.save(os.path.join(out_dir, "u2.npy"), u2)
    np.save(os.path.join(out_dir, "u3.npy"), u3)
    np.save(os.path.join(out_dir, "gamma1.npy"), gamma1)
    np.save(os.path.join(out_dir, "gamma2.npy"), gamma2)
    
    if apt_new == True:
        np.save(os.path.join(out_dir, "u4.npy"), u4)
        np.save(os.path.join(out_dir, "gamma3.npy"), gamma3)
    
    
    # 主进程可视化（若有matplotlib）
    def reconstruct_full(u_interior, room_id, gamma1_arr, gamma2_arr, gamma3_arr=None):
        """重构包含边界的完整温度场（用于可视化）"""
        new_apt=False if gamma3_arr is None else True
        info = get_room_grid_info(room_id, dx, dy, new_apt)
        Nx, Ny = info["Nx"], info["Ny"]
        bc_types, bc_values = get_boundary_conditions(room_id, gamma1_arr, gamma2_arr, gamma3_arr, dx, dy)

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
    u1_full = reconstruct_full(u1, "room1", gamma1, gamma2, gamma3)
    u2_full = reconstruct_full(u2, "room2", gamma1, gamma2, gamma3)
    u3_full = reconstruct_full(u3, "room3", gamma1, gamma2, gamma3)

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

    # 保存Room4图像
    if apt_new == True:
        u4_full = reconstruct_full(u4, "room4", gamma1, gamma2, gamma3)
        plt.imshow(u4_full.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        plt.xlabel('x [m]'); plt.ylabel('y [m]')
        plt.colorbar(); plt.title("Room4 Temperature (with boundaries)")
        room4_path = os.path.join(out_dir, "room4.png")
        plt.savefig(room4_path, dpi=200); plt.close()
        print(f"Saved figure: {room4_path}")
    


if __name__ == "__main__":
    args = parse_arguments()
    main(apt_new=args.new_apartment)