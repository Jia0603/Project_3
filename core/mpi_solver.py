import numpy as np
from mpi4py import MPI
from common.utils import get_room_grid_info
from common.boundary_config import get_boundary_conditions
from core.matrix_builder import build_laplace_matrix_mixed, build_b_mixed
from scipy.sparse.linalg import cg, gmres, spsolve
from scipy.sparse import csr_matrix

def _solve_room(room_id, gamma1, gamma2, dx, dy, solver_config=None):
    """
    Assemble and solve the discrete system for a given `room_id`.
    Returns the 2D temperature field u_2d.
    """
    if solver_config is None:
        solver_config = {'solver_type': 'direct', 'tol': 1e-6, 'maxiter': 10000}

    info = get_room_grid_info(room_id, dx, dy)
    bc_types, bc_values = get_boundary_conditions(room_id, gamma1, gamma2, None,  dx, dy)

    A, nx_solve, ny_solve = build_laplace_matrix_mixed(info["Nx"], info["Ny"], dx, bc_types)
    b = build_b_mixed(info["Nx"], info["Ny"], dx, bc_types, bc_values)

    # Validate matrix dimensions
    if A.shape[0] != len(b):
        raise ValueError(f"{room_id}: Dimension mismatch! A: {A.shape}, b: {len(b)}")
    
    # Optional: condition number diagnostics
    """cond = np.linalg.cond(A)
    if cond > 1e10:
        print(f"Warning: {room_id} matrix is ill-conditioned: {cond:.2e}")"""

    # u = np.linalg.solve(A, b).reshape(ny_solve, nx_solve)
    # return u

    # Select solver
    solver_type = solver_config.get('solver_type', 'direct')

    if solver_type == 'cg':
        # Conjugate Gradient (for SPD matrices)
        A_sparse = csr_matrix(A)
        u_flat, info = cg(A_sparse, b, rtol=solver_config['tol'], maxiter=solver_config['maxiter'])
        if info > 0:
            print(f"Warning: {room_id} CG did not converge after {info} iterations")
        elif info < 0:
            print(f"Error: {room_id} CG illegal input or breakdown")

    elif solver_type == 'spsolve':
        # Sparse direct solve
        A_sparse = csr_matrix(A)
        u_flat = spsolve(A_sparse, b)

    else:  # 'direct' or default
        if hasattr(A, 'toarray'):
            # Convert sparse to dense then solve
            u_flat = np.linalg.solve(A.toarray(), b)
        else:
            # Dense matrix
            u_flat = np.linalg.solve(A, b)

    u = u_flat.reshape(ny_solve, nx_solve)
    return u


def dirichlet_neumann_iterate(dx, dy, omega=0.8, num_iters=10, solver_config=None):
    """
    Dirichlet–Neumann iteration (MPI-only):
    Synchronizes interface data across three subdomains in the order
    "Ω2 -> (Ω1, Ω3) -> relaxation update".
    The process layout requires 4 ranks: rank0 (root), rank1 (Ω1), rank2 (Ω2), rank3 (Ω3).

    Returns:
        dict: Only the root returns {"u1", "u2", "u3", "gamma1", "gamma2"}
        Non-root ranks return None
    """
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Validate number of processes (must be 4)
    if size != 4:
        if rank == 0:
            raise ValueError("This simulation requires 4 MPI ranks. Run with 'mpiexec -n 4 python main.py'")
        comm.Abort()

    # Initialize interface arrays (length follows room1 Ny)
    room1_info = get_room_grid_info("room1", dx, dy)
    Ny_interface = room1_info["Ny"]
    gamma1 = np.full(Ny_interface, 20.0, dtype=float)
    gamma2 = np.full(Ny_interface, 20.0, dtype=float)

    # MPI tags
    TAG_G1_SEND = 10  # Iteration: send partial solutions to root for interface update
    TAG_G2_SEND = 11  # Final: send final solutions to root
    TAG_DONE = 99     # Root signals all workers that iterations are done

    # Root logic (rank 0): orchestrate iterations, update interfaces, gather results
    if rank == 0:
        u1 = u2 = u3 = None
        # Iterate interface updates and subdomain solves
        for _ in range(num_iters):
            # Broadcast current interface temperatures
            comm.bcast(gamma1, root=0)
            comm.bcast(gamma2, root=0)

            # Receive solutions from workers
            u1 = comm.recv(source=1, tag=TAG_G1_SEND)
            _ = comm.recv(source=2, tag=TAG_G1_SEND)  # Room2 solution not used in interface update
            u3 = comm.recv(source=3, tag=TAG_G1_SEND)

            # Relaxation update on interfaces (interior points only)
            u1_right = u1[-1, :]  # Room1 right boundary (matches Room2 left interface)
            u3_left = u3[0, :]    # Room3 left boundary (matches Room2 right interface)
            
            gamma1_new = gamma1.copy()
            gamma2_new = gamma2.copy()
            gamma1_new[1:-1] = omega * u1_right + (1 - omega) * gamma1[1:-1]
            gamma2_new[1:-1] = omega * u3_left + (1 - omega) * gamma2[1:-1]
            
            gamma1, gamma2 = gamma1_new, gamma2_new

        # Notify workers that iterations are complete
        for dst_rank in (1, 2, 3):
            comm.send(True, dest=dst_rank, tag=TAG_DONE)

        # Broadcast final interfaces and gather final solutions
        comm.bcast(gamma1, root=0)
        comm.bcast(gamma2, root=0)
        u1 = comm.recv(source=1, tag=TAG_G2_SEND)
        u2 = comm.recv(source=2, tag=TAG_G2_SEND)
        u3 = comm.recv(source=3, tag=TAG_G2_SEND)

        return {"u1": u1, "u2": u2, "u3": u3, "gamma1": gamma1, "gamma2": gamma2}

    # Worker logic (rank1: room1, rank2: room2, rank3: room3)
    if rank in (1, 2, 3):
        # Map rank to room id
        room_id_map = {1: "room1", 2: "room2", 3: "room3"}
        room_id = room_id_map[rank]

        # Iteration phase: receive interfaces, solve subdomain, send to root
        for _ in range(num_iters):
            # Receive interface temperatures
            gamma1 = comm.bcast(None, root=0)
            gamma2 = comm.bcast(None, root=0)

            # Solve current subdomain
            u = _solve_room(room_id, gamma1, gamma2, dx, dy, solver_config)
            # Send solution back to root
            comm.send(u, dest=0, tag=TAG_G1_SEND)

            # Check for termination signal (polled after loop)
        
        _ = comm.recv(source=0, tag=TAG_DONE)

        # Final phase: receive final interfaces, solve, and send final solution
        gamma1 = comm.bcast(None, root=0)
        gamma2 = comm.bcast(None, root=0)
        final_u = _solve_room(room_id, gamma1, gamma2, dx, dy, solver_config)
        comm.send(final_u, dest=0, tag=TAG_G2_SEND)
        
        return None