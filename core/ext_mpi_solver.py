import numpy as np
from mpi4py import MPI 
from common.utils import get_room_grid_info
from common.boundary_config import get_boundary_conditions
from core.matrix_builder import build_laplace_matrix_mixed, build_b_mixed

from scipy.sparse.linalg import cg, gmres, spsolve
from scipy.sparse import csr_matrix

def solve_room(room_id, gamma1, gamma2, gamma3, dx, dy, solver_config=None):
    """
    Solve Au = b.
    """

    if solver_config is None:
        solver_config = {'solver_type': 'direct', 'tol': 1e-6, 'maxiter': 10000}


    info = get_room_grid_info(room_id, dx, dy, new_apt=True)
    bc_types, bc_values = get_boundary_conditions(room_id, gamma1, gamma2, gamma3, dx, dy)

    A, nx_solve, ny_solve = build_laplace_matrix_mixed(info["Nx"], info["Ny"], dx, bc_types)
    b = build_b_mixed(info["Nx"], info["Ny"], dx, bc_types, bc_values)

    if A.shape[0] != len(b):
        raise ValueError(f"{room_id}: Unmatchable matrices! A: {A.shape}, b: {len(b)}")

    """cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"Warning: {room_id} large condition number: {cond:.2e}")"""

    solver_type = solver_config.get('solver_type', 'direct')

    if solver_type == 'cg':
        A_sparse = csr_matrix(A)
        u_flat, info = cg(A_sparse, b, rtol=solver_config['tol'], maxiter=solver_config['maxiter'])
        if info > 0:
            print(f"Warning: {room_id} CG did not converge after {info} iterations")

    elif solver_type == 'spsolve':
        A_sparse = csr_matrix(A)
        u_flat = spsolve(A_sparse, b)

    else:  # 'direct'
        if hasattr(A, 'toarray'):
            u_flat = np.linalg.solve(A.toarray(), b)
        else:
            u_flat = np.linalg.solve(A, b)

    u = u_flat.reshape(ny_solve, nx_solve)
    return u


def ext_dirichlet_neumann_iterate(dx, dy, omega=0.8, num_iters=10, solver_config=None):
    """
    Dirichlet–Neumann iteration for 4-room apartment:
      Phase A: solve Ω2 with Dirichlet (gamma1,gamma2,gamma3) -> compute Neumann fluxes for Ω1/Ω3/Ω4
      Phase B: solve Ω1, Ω3, Ω4 with those Neumann -> return interface temps -> relax gamma

    MPI ranks: 5 processes (rank0: root, rank1: room1, rank2: room2, rank3: room3, rank4: room4)
    
    Returns:
        dict: Root returns {"u1", "u2", "u3", "u4", "gamma1", "gamma2", "gamma3"}
        Workers return None
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 5:
        if rank == 0:
            raise ValueError("This simulation requires 5 MPI ranks. Run with 'mpiexec -n 5 python main.py -n'")
        comm.Abort()

    # Initialize interface arrays
    room1_info = get_room_grid_info("room1", dx, dy, new_apt=True)
    room4_info = get_room_grid_info("room4", dx, dy, new_apt=True)
    Ny_interface = room1_info["Ny"]
    Ny_interface_new = room4_info["Ny"]
    
    # MPI communication tags
    TAG_SOLUTION = 12
    TAG_DONE = 99

    # ========== ROOT PROCESS (rank 0) ==========
    if rank == 0:
        # Initialize interface values
        gamma1 = np.full(Ny_interface, 20.0, dtype=float)  # Ω₁-Ω₂ interface
        gamma2 = np.full(Ny_interface, 20.0, dtype=float)  # Ω₂-Ω₃ interface
        gamma3 = np.full(Ny_interface_new, 20.0, dtype=float)  # Ω₂-Ω₄ interface
        
        print(f"Starting 4-room Dirichlet-Neumann iteration: {num_iters} iters, ω={omega}, dx={dx}")
        
        for iter_num in range(num_iters):
            # ===== PHASE 1: All workers receive current interface values =====
            comm.bcast(gamma1, root=0)
            comm.bcast(gamma2, root=0)
            comm.bcast(gamma3, root=0)
            
            # ===== PHASE 2: Room2 solves with Dirichlet BC =====
            u2 = comm.recv(source=2, tag=TAG_SOLUTION)
            
            # Compute Neumann BC from room2's solution and SLICE to interface lengths
            # Room1 interface segment (lower segment along Ω2 left boundary)
            neumann1 = (u2[:Ny_interface, 1] - u2[:Ny_interface, 0]) / dx
            # Room3 interface segment (upper segment along Ω2 right boundary)
            neumann2 = -(u2[-Ny_interface:, -1] - u2[-Ny_interface:, -2]) / dx
            # Room4 interface segment (middle segment along Ω2 right boundary)
            neumann3 = -(u2[-(Ny_interface + Ny_interface_new):-Ny_interface, -1] - 
                        u2[-(Ny_interface + Ny_interface_new):-Ny_interface, -2]) / dx

            # Sanity check
            assert neumann1.shape[0] == Ny_interface, f"neumann1 len {neumann1.shape[0]} != Ny_interface {Ny_interface}"
            assert neumann2.shape[0] == Ny_interface, f"neumann2 len {neumann2.shape[0]} != Ny_interface {Ny_interface}"
            assert neumann3.shape[0] == Ny_interface_new, f"neumann3 len {neumann3.shape[0]} != Ny_interface_new {Ny_interface_new}"

            # Broadcast Neumann BC
            comm.bcast(neumann1, root=0)
            comm.bcast(neumann2, root=0)
            comm.bcast(neumann3, root=0)
            
            # ===== PHASE 3: Room1, Room3, Room4 solve with Neumann BC =====
            u1 = comm.recv(source=1, tag=TAG_SOLUTION)
            u3 = comm.recv(source=3, tag=TAG_SOLUTION)
            u4 = comm.recv(source=4, tag=TAG_SOLUTION)
            
            # ===== PHASE 4: Relaxation update on interface values =====
            # Extract new interface values from outer rooms
            gamma1_from_room1 = u1[:, -1].copy()  # Room1 right boundary
            gamma2_from_room3 = u3[:, 0].copy()   # Room3 left boundary
            gamma3_from_room4 = u4[:, 0].copy()   # Room4 left boundary
            
            # Relaxation: blend new values from outer rooms with old interface values
            gamma1_new = gamma1.copy()
            gamma2_new = gamma2.copy()
            gamma3_new = gamma3.copy()
            gamma1_new[1:-1] = omega * gamma1_from_room1 + (1 - omega) * gamma1[1:-1]
            gamma2_new[1:-1] = omega * gamma2_from_room3 + (1 - omega) * gamma2[1:-1]
            gamma3_new[1:-1] = omega * gamma3_from_room4 + (1 - omega) * gamma3[1:-1]
            
            gamma1 = gamma1_new
            gamma2 = gamma2_new
            gamma3 = gamma3_new
            
            # Progress logging
            if (iter_num + 1) % 2 == 0 or iter_num == num_iters - 1:
                avg_gamma = (np.mean(gamma1) + np.mean(gamma2) + np.mean(gamma3)) / 3
                print(f"Iteration {iter_num + 1}/{num_iters}: avg_interface_temp={avg_gamma:.2f}°C")
        
        # Signal workers that iterations are done
        for worker_rank in [1, 2, 3, 4]:
            comm.send(True, dest=worker_rank, tag=TAG_DONE)
        
        print("4-room iterations completed successfully!")
        return {
            "u1": u1,
            "u2": u2,
            "u3": u3,
            "u4": u4,
            "gamma1": gamma1,
            "gamma2": gamma2,
            "gamma3": gamma3
        }
    
    # ========== WORKER PROCESSES (rank 1, 2, 3, 4) ==========
    room_map = {1: "room1", 2: "room2", 3: "room3", 4: "room4"}
    room_id = room_map[rank]
    
    for _ in range(num_iters):
        # PHASE 1: All workers receive interface Dirichlet values
        gamma1 = comm.bcast(None, root=0)
        gamma2 = comm.bcast(None, root=0)
        gamma3 = comm.bcast(None, root=0)
        
        # PHASE 2: Only room2 solves with Dirichlet BC
        if rank == 2:
            u = solve_room(room_id, gamma1, gamma2, gamma3, dx, dy, solver_config)
            comm.send(u, dest=0, tag=TAG_SOLUTION)
        
        # PHASE 3: All workers receive Neumann values
        neumann1 = comm.bcast(None, root=0)
        neumann2 = comm.bcast(None, root=0)
        neumann3 = comm.bcast(None, root=0)
        
        # PHASE 4: Room1, room3, room4 solve with Neumann BC
        if rank == 1:
            # Room1: right boundary is Neumann. boundary_config expects this in gamma1 position.
            u = solve_room(room_id, neumann1, None, None, dx, dy, solver_config)
            comm.send(u, dest=0, tag=TAG_SOLUTION)
        elif rank == 3:
            # Room3: left boundary is Neumann. boundary_config expects this in gamma2 position.
            u = solve_room(room_id, None, neumann2, None, dx, dy, solver_config)
            comm.send(u, dest=0, tag=TAG_SOLUTION)
        elif rank == 4:
            # Room4: left boundary is Neumann. boundary_config expects this in gamma3 position.
            u = solve_room(room_id, None, None, neumann3, dx, dy, solver_config)
            comm.send(u, dest=0, tag=TAG_SOLUTION)
    
    # Wait for termination signal
    _ = comm.recv(source=0, tag=TAG_DONE)
    return None
