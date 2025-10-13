import numpy as np
import os
from mpi4py import MPI 
from common.utils import get_default_dx, get_default_dy, get_room_grid_info
from common.boundary_config import get_boundary_conditions
from core.mpi_solver import dirichlet_neumann_iterate
from core.ext_mpi_solver import ext_dirichlet_neumann_iterate
from core.visualizer import visualize_pipeline
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

def main(apt_new=False, heater_temp=40.0, window_temp=5.0, wall_temp=15.0, num_iters=10):
    #################################这里我加了一些参数，方便修改
    from common import utils  # 确保导入在函数内
    utils.set_boundary_conditions(heater=heater_temp, window=window_temp, wall=wall_temp)

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
    print(f"1. Lowest temp:{min_temp:.2f} °C")
    print(f"2. Averaged temp:{avg_temp:.2f}°C")
    print(f"3. Highest temp:{max_temp:.2f}°C")
    print("-"*50)

    if avg_temp > 18.0:
        print("Averged tempture > 18°C. The heating in the flat is adequate.")
    else:
        print("Averged tempture < 18°C. The flat is not warm enough.")
            
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



    # --- Visualization ---
    if RANK == 0:
        visualize_pipeline(result, dx, dy, apt_new)


if __name__ == "__main__":
    args = parse_arguments()
    #main(apt_new=args.new_apartment)
    ########################################这里我也改了
    main(
        apt_new=args.new_apartment,
        heater_temp=20.0,  # 改暖气
        window_temp=5.0,  # 改窗温
        wall_temp=15.0,  # 改墙温
        num_iters=20  # 改迭代次数
    )
