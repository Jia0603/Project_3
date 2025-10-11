import numpy as np
from mpi4py import MPI  # 直接导入MPI，不再兼容无MPI环境
from common.utils import get_room_grid_info
from common.boundary_config import get_boundary_conditions
from core.matrix_builder import build_laplace_matrix_mixed, build_b_mixed


def _solve_room(room_id, gamma1, gamma2, dx, dy):
    """
    组装房间 `room_id` 的离散方程并直接求解（子进程内部调用）。
    返回二维温度场 u_2d。
    """
    info = get_room_grid_info(room_id, dx, dy)
    bc_types, bc_values = get_boundary_conditions(room_id, gamma1, gamma2, dx, dy)

    A, nx_solve, ny_solve = build_laplace_matrix_mixed(info["Nx"], info["Ny"], dx, bc_types)
    b = build_b_mixed(info["Nx"], info["Ny"], dx, bc_types, bc_values)

    # 检查矩阵维度
    if A.shape[0] != len(b):
        raise ValueError(f"{room_id}: 矩阵维度不匹配! A: {A.shape}, b: {len(b)}")
    
    # 检查矩阵条件数
    cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"警告：{room_id} 矩阵条件数很大: {cond:.2e}")
    
    u = np.linalg.solve(A, b).reshape(ny_solve, nx_solve)
    return u


def dirichlet_neumann_iterate(dx, dy, omega=0.8, num_iters=10):
    """
    Dirichlet–Neumann 迭代（仅MPI并行模式）：
    按 “Ω2 -> (Ω1, Ω3) -> 松弛更新” 顺序同步三个子域的接口数据。
    进程布局固定为4个进程：rank0（主进程）、rank1（Ω1）、rank2（Ω2）、rank3（Ω3）。

    返回：
        dict: 仅主进程返回 {"u1": ndarray, "u2": ndarray, "u3": ndarray, "gamma1": ndarray, "gamma2": ndarray}
        子进程返回None
    """
    # 初始化MPI环境（强制并行，无MPI环境将报错）
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # 校验进程数（必须为4，否则终止）
    if size != 4:
        if rank == 0:
            raise ValueError("并行计算需启动4个进程！请使用 'mpiexec -n 4 python 脚本名.py' 运行")
        comm.Abort()  # 非主进程直接终止

    # 初始化接口变量（长度取决于room1的y方向网格数）
    room1_info = get_room_grid_info("room1", dx, dy)
    Ny_interface = room1_info["Ny"]
    gamma1 = np.full(Ny_interface, 20.0, dtype=float)
    gamma2 = np.full(Ny_interface, 20.0, dtype=float)

    # 定义MPI通信标签
    TAG_G1_SEND = 10  # 迭代阶段：子进程向主进程发送解（用于接口更新）
    TAG_G2_SEND = 11  # 最终阶段：子进程向主进程发送最终解
    TAG_DONE = 99     # 主进程通知子进程迭代结束

    # 主进程逻辑（rank0）：调度迭代、更新接口、收集结果
    if rank == 0:
        u1 = u2 = u3 = None
        # 迭代更新接口与子域解
        for _ in range(num_iters):
            # 广播当前接口温度到所有子进程
            comm.bcast(gamma1, root=0)
            comm.bcast(gamma2, root=0)

            # 接收子进程发送的解（用于更新接口）
            u1 = comm.recv(source=1, tag=TAG_G1_SEND)
            _ = comm.recv(source=2, tag=TAG_G1_SEND)  # Ω2解暂不参与接口更新
            u3 = comm.recv(source=3, tag=TAG_G1_SEND)

            # 松弛更新接口温度（仅更新内部点，避免边界效应）
            u1_right = u1[-1, :]  # Ω1右边界（与Ω2左接口重合）
            u3_left = u3[0, :]    # Ω3左边界（与Ω2右接口重合）
            
            gamma1_new = gamma1.copy()
            gamma2_new = gamma2.copy()
            gamma1_new[1:-1] = omega * u1_right + (1 - omega) * gamma1[1:-1]
            gamma2_new[1:-1] = omega * u3_left + (1 - omega) * gamma2[1:-1]
            
            gamma1, gamma2 = gamma1_new, gamma2_new

        # 通知所有子进程迭代结束
        for dst_rank in (1, 2, 3):
            comm.send(True, dest=dst_rank, tag=TAG_DONE)

        # 广播最终接口温度，收集子进程的最终解
        comm.bcast(gamma1, root=0)
        comm.bcast(gamma2, root=0)
        u1 = comm.recv(source=1, tag=TAG_G2_SEND)
        u2 = comm.recv(source=2, tag=TAG_G2_SEND)
        u3 = comm.recv(source=3, tag=TAG_G2_SEND)

        return {"u1": u1, "u2": u2, "u3": u3, "gamma1": gamma1, "gamma2": gamma2}

    # 子进程逻辑（rank1:Ω1、rank2:Ω2、rank3:Ω3）：求解子域、响应主进程指令
    if rank in (1, 2, 3):
        # 映射rank到房间ID
        room_id_map = {1: "room1", 2: "room2", 3: "room3"}
        room_id = room_id_map[rank]

        # 迭代阶段：接收接口温度、求解子域、发送解给主进程
        for _ in range(num_iters):
            # 接收主进程广播的接口温度
            gamma1 = comm.bcast(None, root=0)
            gamma2 = comm.bcast(None, root=0)

            # 求解当前子域的温度场
            u = _solve_room(room_id, gamma1, gamma2, dx, dy)

            # 发送解给主进程（用于接口更新）
            comm.send(u, dest=0, tag=TAG_G1_SEND)

            # 检查是否收到结束指令
        
        _ = comm.recv(source=0, tag=TAG_DONE)

        # 最终阶段：接收最终接口温度，求解并发送最终解
        gamma1 = comm.bcast(None, root=0)
        gamma2 = comm.bcast(None, root=0)
        final_u = _solve_room(room_id, gamma1, gamma2, dx, dy)
        comm.send(final_u, dest=0, tag=TAG_G2_SEND)
        
        return None