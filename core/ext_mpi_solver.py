import numpy as np
from mpi4py import MPI 
from common.utils import get_room_grid_info
from common.boundary_config import get_boundary_conditions
from core.matrix_builder import build_laplace_matrix_mixed, build_b_mixed

def solve_room(room_id, gamma1, gamma2, gamma3, dx, dy):
    """
    Solve Au = b.
    """
    info = get_room_grid_info(room_id, dx, dy, new_apt=True)
    bc_types, bc_values = get_boundary_conditions(room_id, gamma1, gamma2, gamma3, dx, dy)

    A, nx_solve, ny_solve = build_laplace_matrix_mixed(info["Nx"], info["Ny"], dx, bc_types)
    b = build_b_mixed(info["Nx"], info["Ny"], dx, bc_types, bc_values)

    if A.shape[0] != len(b):
        raise ValueError(f"{room_id}: Unmatchable matrices! A: {A.shape}, b: {len(b)}")
    
    cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"Warning: {room_id} large condition number: {cond:.2e}")
    
    u = np.linalg.solve(A, b).reshape(ny_solve, nx_solve)

    return u


def ext_dirichlet_neumann_iterate(dx, dy, omega=0.8, num_iters=10):
    """
    Dirichlet-Neumann iteration using MPI.
    Steps: “Ω2 -> (Ω1, Ω3, Ω4) -> relaxation” 

    Return:
        dict: only for rank0 process {"u1": ndarray, "u2": ndarray, "u3": ndarray, "gamma1": ndarray, "gamma2": ndarray}
        sub-processes return None
    """
    # Initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # check number of precesses == 5
    if size != 5:
        if rank == 0:
            raise ValueError("5 processes required! Run the program with 'mpiexec -n 5 python file_name.py -n' instead.")
        comm.Abort() 

    room1_info = get_room_grid_info("room1", dx, dy, new_apt=True)
    room4_info = get_room_grid_info("room4", dx, dy, new_apt=True)
    Ny_interface = room1_info["Ny"]
    Ny_interface_new = room4_info["Ny"]
    gamma1 = np.full(Ny_interface, 20.0, dtype=float)
    gamma2 = np.full(Ny_interface, 20.0, dtype=float)
    gamma3 = np.full(Ny_interface_new, 20.0, dtype=float)

    TAG_G1_SEND = 10  # tag for iteration stages
    TAG_G2_SEND = 11  # tag for sending final solution
    TAG_DONE = 99     # interation completed

    # rank0
    if rank == 0:
        u1 = u2 = u3 = u4 = None
        for _ in range(num_iters):
            # broadcast current temp on the interfaces to sub-processes
            comm.bcast(gamma1, root=0)
            comm.bcast(gamma2, root=0)
            comm.bcast(gamma3, root=0)

            # iterfaces update
            u1 = comm.recv(source=1, tag=TAG_G1_SEND)
            _ = comm.recv(source=2, tag=TAG_G1_SEND)  # Ω2: room 2 skip updates
            u3 = comm.recv(source=3, tag=TAG_G1_SEND)
            u4 = comm.recv(source=4, tag=TAG_G1_SEND)

            # relaxation
            u1_right = u1[-1, :] 
            u3_left = u3[0, :]
            u4_left = u4[0, :]   
            
            gamma1_new = gamma1.copy()
            gamma2_new = gamma2.copy()
            gamma3_new = gamma3.copy()
            gamma1_new[1:-1] = omega * u1_right + (1 - omega) * gamma1[1:-1]
            gamma2_new[1:-1] = omega * u3_left + (1 - omega) * gamma2[1:-1]
            gamma3_new[1:-1] = omega * u4_left + (1 - omega) * gamma3[1:-1]
            
            gamma1, gamma2, gamma3 = gamma1_new, gamma2_new, gamma3_new

        # terminate sub-processes
        for dst_rank in (1, 2, 3, 4):
            comm.send(True, dest=dst_rank, tag=TAG_DONE)

        # 广播最终接口温度，收集子进程的最终解
        comm.bcast(gamma1, root=0)
        comm.bcast(gamma2, root=0)
        comm.bcast(gamma3, root=0)
        u1 = comm.recv(source=1, tag=TAG_G2_SEND)
        u2 = comm.recv(source=2, tag=TAG_G2_SEND)
        u3 = comm.recv(source=3, tag=TAG_G2_SEND)
        u4 = comm.recv(source=4, tag=TAG_G2_SEND)

        return {"u1": u1, "u2": u2, "u3": u3, "u4": u4, "gamma1": gamma1, "gamma2": gamma2, "gamma3": gamma3}

    # Solve sub-spaces（rank1:Ω1, rank2:Ω2, rank3:Ω3, rank4:Ω4）
    if rank in (1, 2, 3, 4):
        
        room_id_map = {1: "room1", 2: "room2", 3: "room3", 4: "room4"}
        room_id = room_id_map[rank]

        for _ in range(num_iters):
            # get interface temps
            gamma1 = comm.bcast(None, root=0)
            gamma2 = comm.bcast(None, root=0)
            gamma3 = comm.bcast(None, root=0)

            # solve Ax=b for {room_id}
            u = solve_room(room_id, gamma1, gamma2, gamma3, dx, dy)

            # send to main process
            comm.send(u, dest=0, tag=TAG_G1_SEND)
        
        _ = comm.recv(source=0, tag=TAG_DONE) # check if terminate

        # final solve
        gamma1 = comm.bcast(None, root=0)
        gamma2 = comm.bcast(None, root=0)
        gamma3 = comm.bcast(None, root=0)
        final_u = solve_room(room_id, gamma1, gamma2, gamma3, dx, dy)
        comm.send(final_u, dest=0, tag=TAG_G2_SEND)
        
        return None