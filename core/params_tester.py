import os
import importlib
import numpy as np
from mpi4py import MPI

# 导入主程序和底层模块
import main
import common.utils as utils

def run_experiment(heater_temp, num_iters):
    """
    在给定 heater_temp 和 num_iters 下运行一次模拟，并保存在独立文件夹中
    """
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()

    # 1️⃣ 动态修改 heater 温度
    utils.HEATER_TEMP = heater_temp
    importlib.reload(utils)  # 使 boundary_config 读取新温度

    # 2️⃣ 修改 main.py 中的参数（传入 num_iters）
    # 为了尽量少改动，不直接改 main，而是 monkey patch 变量
    dx = utils.get_default_dx()
    dy = utils.get_default_dy()
    omega = 0.8

    from core.mpi_solver import dirichlet_neumann_iterate
    result = dirichlet_neumann_iterate(dx, dy, omega=omega, num_iters=num_iters)

    # 3️⃣ 仅主进程负责保存结果
    if RANK == 0:
        # 创建输出目录（每个组合一个文件夹）
        combo_dir = f"results/T{heater_temp:.0f}_Iter{num_iters}"
        os.makedirs(combo_dir, exist_ok=True)

        # 保存所有房间温度矩阵
        for key, value in result.items():
            np.save(os.path.join(combo_dir, f"{key}.npy"), value)

        # 计算平均温度
        all_valid_temps = []
        for k, room_temp in result.items():
            if k.startswith("u"):
                valid_temps = room_temp[~np.isnan(room_temp)]
                all_valid_temps.extend(valid_temps)
        avg_temp = np.mean(all_valid_temps)

        print(f"[Heater {heater_temp}°C | Iter {num_iters}] → AvgTemp = {avg_temp:.3f}°C")
        return heater_temp, num_iters, avg_temp

    else:
        return None


if __name__ == "__main__":
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()

    heater_temps = [30, 35, 40, 45, 50]
    iter_counts = [5, 10, 20]
    results = []

    # 循环运行所有组合
    for T in heater_temps:
        for it in iter_counts:
            res = run_experiment(T, it)
            if RANK == 0 and res is not None:
                results.append(res)

    # 4️⃣ 主进程保存汇总表格
    if RANK == 0:
        results_arr = np.array(results, dtype=[("HeaterTemp", float),
                                               ("NumIterations", int),
                                               ("AverageTemp", float)])
        out_csv = "results/summary.csv"
        os.makedirs("results", exist_ok=True)
        np.savetxt(out_csv, results_arr, delimiter=",",
                   header="HeaterTemp,NumIterations,AverageTemp", comments='')
        print(f"\n✅ Saved summary table to {out_csv}")
