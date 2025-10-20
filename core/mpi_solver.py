import numpy as np
from mpi4py import MPI
from common.utils import get_room_grid_info
from common.boundary_config import get_boundary_conditions
from core.matrix_builder import build_laplace_matrix_mixed, build_b_mixed
from scipy.sparse.linalg import cg, spsolve
from scipy.sparse import csr_matrix

def _solve_room(room_id, gamma1, gamma2, dx, dy, solver_config=None):
    """
    Assemble and solve the discrete system for a given `room_id`.
    Returns the 2D temperature field u_2d with shape (ny_solve, nx_solve).
    """
    if solver_config is None:
        solver_config = {'solver_type': 'spsolve', 'tol': 1e-4, 'maxiter': 10000}

    info = get_room_grid_info(room_id, dx, dy)
    bc_types, bc_values = get_boundary_conditions(room_id, gamma1, gamma2, None, dx, dy)

    A, nx_solve, ny_solve = build_laplace_matrix_mixed(info["Nx"], info["Ny"], dx, bc_types)
    b = build_b_mixed(info["Nx"], info["Ny"], dx, bc_types, bc_values)

    if A.shape[0] != len(b):
        raise ValueError(f"{room_id}: Dimension mismatch! A: {A.shape}, b: {len(b)}")

    solver_type = solver_config.get('solver_type', 'direct')

    if solver_type == 'cg':
        A_sparse = csr_matrix(A)
        u_flat, info = cg(A_sparse, b, rtol=solver_config['tol'], maxiter=solver_config['maxiter'])
        if info > 0:
            print(f"Warning: {room_id} CG did not converge after {info} iterations")
        elif info < 0:
            print(f"Error: {room_id} CG illegal input or breakdown")

    elif solver_type == 'spsolve':
        A_sparse = csr_matrix(A)
        u_flat = spsolve(A_sparse, b)

    else:  # 'direct' or default
        if hasattr(A, 'toarray'):
            u_flat = np.linalg.solve(A.toarray(), b)
        else:
            u_flat = np.linalg.solve(A, b)

    # Reshape: matrix_builder stores as row-major, output is (ny_solve, nx_solve)
    u = u_flat.reshape(ny_solve, nx_solve)
    return u


def dirichlet_neumann_iterate(dx, dy, omega=0.8, num_iters=100, solver_config=None):
    """
    Dirichlet–Neumann iteration following the PDF specification:
    
    Each iteration:
    1. Solve Ω₂ (room2) with Dirichlet BC from gamma1, gamma2 (values from Ω₁, Ω₃)
    2. Update gamma1, gamma2 from Ω₂'s solution (provides Dirichlet BC for next iteration)
    3. Solve Ω₁ (room1) and Ω₃ (room3) with Neumann BC computed from Ω₂
    4. Apply relaxation to gamma1, gamma2 using new values from Ω₁, Ω₃
    
    MPI ranks: 4 processes (rank0: root, rank1: room1, rank2: room2, rank3: room3)
    
    Returns:
        dict: Root returns {"u1", "u2", "u3", "gamma1", "gamma2"}
        Workers return None
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 4:
        if rank == 0:
            raise ValueError("This simulation requires 4 MPI ranks. Run with 'mpiexec -n 4 python main.py'")
        comm.Abort()

    # Initialize interface arrays
    room1_info = get_room_grid_info("room1", dx, dy)
    Ny_interface = room1_info["Ny"]
    
    # MPI communication tags
    TAG_SOLUTION = 12
    TAG_DONE = 99

    # ========== ROOT PROCESS (rank 0) ==========
    if rank == 0:
        # Initialize interface values with reasonable starting guess
        gamma1 = np.full(Ny_interface, 20.0, dtype=float)  # Ω₁-Ω₂ interface
        gamma2 = np.full(Ny_interface, 20.0, dtype=float)  # Ω₂-Ω₃ interface
        
        print(f"[rank0] Start D-N: iters={num_iters}, ω={omega}, dx={dx}")
        
        for iter_num in range(num_iters):
            # print(f"[rank0] iter {iter_num+1}: bcast gammas")
            # ===== PHASE 1: All workers receive current interface values =====
            comm.bcast(gamma1, root=0)
            comm.bcast(gamma2, root=0)
            
            # ===== PHASE 2: Room2 solves with Dirichlet BC =====
            u2 = comm.recv(source=2, tag=TAG_SOLUTION)
            # print(f"[rank0] iter {iter_num+1}: recv u2 shape={None if u2 is None else u2.shape}")
            
            # Extract interface values from room2's solution for next Dirichlet BC
            # Room2's left boundary → interface gamma1 (but we'll update it after getting room1)
            # Room2's right boundary → interface gamma2 (but we'll update it after getting room3)
            gamma1_from_room2 = u2[:, 0].copy()   # Room2 left boundary (interior)
            gamma2_from_room2 = u2[:, -1].copy()  # Room2 right boundary (interior)
            
            # Compute Neumann BC from room2's solution and SLICE to interface length
            # Room1 interface segment (lower segment along Ω2 left boundary)
            neumann1 = (u2[:Ny_interface, 1] - u2[:Ny_interface, 0]) / dx
            # Room3 interface segment (upper segment along Ω2 right boundary)
            neumann2 = -(u2[-Ny_interface:, -1] - u2[-Ny_interface:, -2]) / dx

            # Sanity check
            assert neumann1.shape[0] == Ny_interface, f"neumann1 len {neumann1.shape[0]} != Ny_interface {Ny_interface}"
            assert neumann2.shape[0] == Ny_interface, f"neumann2 len {neumann2.shape[0]} != Ny_interface {Ny_interface}"

            # Broadcast Neumann BC (length Ny_interface)
            comm.bcast(neumann1, root=0)
            comm.bcast(neumann2, root=0)
            # print(f"[rank0] iter {iter_num+1}: bcast neumann done")
            
            # ===== PHASE 3: Room1 and Room3 solve with Neumann BC =====
            u1 = comm.recv(source=1, tag=TAG_SOLUTION)
            # print(f"[rank0] iter {iter_num+1}: recv u1 shape={None if u1 is None else u1.shape}")
            u3 = comm.recv(source=3, tag=TAG_SOLUTION)
            # print(f"[rank0] iter {iter_num+1}: recv u3 shape={None if u3 is None else u3.shape}")
            
            # ===== PHASE 4: Relaxation update on interface values =====
            # Extract new interface values from outer rooms
            # u1 shape: (ny_solve, nx_solve), right boundary is u1[:, -1]
            # u3 shape: (ny_solve, nx_solve), left boundary is u3[:, 0]
            gamma1_from_room1 = u1[:, -1].copy()  # Room1 right boundary (length Ny_interface-2)
            #gamma1_from_room1 = u1[-1, :].copy()
            gamma2_from_room3 = u3[:, 0].copy()   # Room3 left boundary (length Ny_interface-2)
            
            # Relaxation: blend new values from outer rooms with old interface values
            # Only update interior points (not corners)
            gamma1_new = gamma1.copy()
            gamma2_new = gamma2.copy()
            # Map u1/u3 interior boundary (Ny_interface-2) directly to gamma interior
            gamma1_new[1:-1] = omega * gamma1_from_room1 + (1 - omega) * gamma1[1:-1]
            gamma2_new[1:-1] = omega * gamma2_from_room3 + (1 - omega) * gamma2[1:-1]
            
            gamma1 = gamma1_new
            gamma2 = gamma2_new
            
            # Progress logging
            if (iter_num + 1) % 2 == 0 or iter_num == num_iters - 1:
                avg_gamma = (np.mean(gamma1) + np.mean(gamma2)) / 2
                print(f"Iteration {iter_num + 1}/{num_iters}: avg_interface_temp={avg_gamma:.2f}°C")
        
        # Signal workers that iterations are done
        for worker_rank in [1, 2, 3]:
            comm.send(True, dest=worker_rank, tag=TAG_DONE)
        
        # print("[rank0] Iterations completed successfully!")
        return {
            "u1": u1,
            "u2": u2,
            "u3": u3,
            "gamma1": gamma1,
            "gamma2": gamma2
        }
    
    # ========== WORKER PROCESSES (rank 1, 2, 3) ==========
    room_map = {1: "room1", 2: "room2", 3: "room3"}
    room_id = room_map[rank]
    
    for iter_num in range(num_iters):
        # PHASE 1: All workers receive interface Dirichlet values
        gamma1 = comm.bcast(None, root=0)
        gamma2 = comm.bcast(None, root=0)
        # if rank == 2:
        #     print(f"[rank2] iter {iter_num+1}: got gammas len={len(gamma1)},{len(gamma2)}")
        
        # PHASE 2: Only room2 solves with Dirichlet BC
        if rank == 2:
            u = _solve_room(room_id, gamma1, gamma2, dx, dy, solver_config)
            # print(f"[rank2] iter {iter_num+1}: send u2 shape={u.shape}")
            comm.send(u, dest=0, tag=TAG_SOLUTION)
        
        # PHASE 3: All workers receive Neumann values
        neumann1 = comm.bcast(None, root=0)
        neumann2 = comm.bcast(None, root=0)
        # if rank in (1,3):
        #     print(f"[rank{rank}] iter {iter_num+1}: got neumann len={len(neumann1)},{len(neumann2)}")
        
        # PHASE 4: Room1 and room3 solve with Neumann BC
        if rank == 1:
            # Room1: right boundary is Neumann. boundary_config expects this in gamma1 position.
            assert neumann1.shape[0] == get_room_grid_info("room1", dx, dy)["Ny"], "rank1 neumann length mismatch"
            u = _solve_room(room_id, neumann1, None, dx, dy, solver_config)
            # print(f"[rank1] iter {iter_num+1}: send u1 shape={u.shape}")
            comm.send(u, dest=0, tag=TAG_SOLUTION)
        elif rank == 3:
            # Room3: left boundary is Neumann. boundary_config expects this in gamma2 position.
            assert neumann2.shape[0] == get_room_grid_info("room3", dx, dy)["Ny"], "rank3 neumann length mismatch"
            u = _solve_room(room_id, None, neumann2, dx, dy, solver_config)
            # print(f"[rank3] iter {iter_num+1}: send u3 shape={u.shape}")
            comm.send(u, dest=0, tag=TAG_SOLUTION)
    
    # Wait for termination signal
    _ = comm.recv(source=0, tag=TAG_DONE)
    return None