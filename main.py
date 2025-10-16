import numpy as np
import os
from mpi4py import MPI 
from common.utils import get_default_dx, get_default_dy, get_room_grid_info
from common.boundary_config import get_boundary_conditions
from core.mpi_solver import dirichlet_neumann_iterate
from core.ext_mpi_solver import ext_dirichlet_neumann_iterate
from core.visualizer import visualize_pipeline
import argparse
import time
from scipy.sparse.linalg import cg, gmres, spsolve

# Initialize MPI environment
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Apartment Temperature Simulation')

    parser.add_argument('--new-apartment', '-n',
                        action='store_true',
                        help='Use the new apartment layout (default: old apartment)')
    parser.add_argument('--heater-temp', type=float, default=40.0,
                        help='Heater temperature (°C)')
    parser.add_argument('--window-temp', type=float, default=5.0,
                        help='Window temperature (°C)')
    parser.add_argument('--wall-temp', type=float, default=15.0,
                        help='Wall temperature (°C)')
    parser.add_argument('--iters', type=int, default=10,
                        help='Number of Dirichlet-Neumann iterations')
    parser.add_argument('--dx', type=float, default=None,
                        help='Grid spacing Δx (default: get_default_dx())')
    parser.add_argument('--dy', type=float, default=None,
                        help='Grid spacing Δy (default: get_default_dy())')
    parser.add_argument('--solver', type=str, default='direct',
                        choices=['direct', 'cg', 'gmres', 'spsolve'],
                        help='Linear solver type')
    parser.add_argument('--solver-tol', type=float, default=1e-6,
                        help='Solver tolerance for iterative methods')

    return parser.parse_args()


def main(apt_new=False, heater_temp=40.0, window_temp=5.0, wall_temp=15.0,
         num_iters=10, dx=None, dy=None, solver='direct', solver_tol=1e-6):
    """
    Main entry: run apartment temperature simulation

    Args:
        apt_new: whether to use the new apartment layout (4 rooms)
        heater_temp: heater temperature (°C)
        window_temp: window temperature (°C)
        wall_temp: wall temperature (°C)
        num_iters: number of Dirichlet–Neumann iterations
        dx, dy: grid spacing
        solver: linear solver type ('direct', 'cg', 'gmres', 'spsolve')
        solver_tol: tolerance for iterative solvers
    """

    from common import utils
    utils.set_boundary_conditions(heater=heater_temp, window=window_temp, wall=wall_temp)

    # Set grid spacing
    dx = get_default_dx() if dx is None else dx
    dy = get_default_dy() if dy is None else dy
    omega = 0.8

    # Solver configuration
    solver_config = {
        'solver_type': solver,
        'tol': solver_tol,
        'maxiter': 10000
    }

    if RANK == 0:
        print(f"\n{'='*60}")
        print(f"Simulation Configuration:")
        print(f"  - Apartment: {'New (4 rooms)' if apt_new else 'Old (3 rooms)'}")
        print(f"  - Grid spacing: dx={dx}, dy={dy}")
        print(f"  - D-N iterations: {num_iters}")
        print(f"  - Solver: {solver} (tol={solver_tol})")
        print(f"  - Temperatures: Heater={heater_temp}°C, Window={window_temp}°C, Wall={wall_temp}°C")
        print(f"{'='*60}\n")

    t_start = time.time()

    # Call MPI solver (workers return None, root returns results dict)
    if apt_new == False:
        result = dirichlet_neumann_iterate(dx, dy, omega=omega, num_iters=num_iters, solver_config=solver_config)
    else:
        result = ext_dirichlet_neumann_iterate(dx, dy, omega=omega, num_iters=num_iters, solver_config=solver_config)

    # Workers return here (no output/visualization)
    if RANK != 0:
        return


    # Extract root results
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
    # Optional: print shapes (root only)
    print("u1 shape:", u1.shape)
    print("u2 shape:", u2.shape)
    print("u3 shape:", u3.shape)
    print("gamma1 shape:", gamma1.shape, "gamma2 shape:", gamma2.shape)
    '''

    # all_valid_temps = []
    # if apt_new == False:
    #     u = [u1, u2, u3]
    # else:
    #     u = [u1, u2, u3, u4]
    # for room_temp in u:
    #     valid_temps = room_temp[~np.isnan(room_temp)] 
    #     all_valid_temps.extend(valid_temps)
    # all_valid_temps = np.array(all_valid_temps)

    # min_temp = np.min(all_valid_temps)
    # avg_temp = np.mean(all_valid_temps)
    # max_temp = np.max(all_valid_temps)


    # print("\n" + "="*50)
    # print("Is the heating in the flat adequate?")
    # print("="*50)
    # print(f"1. Lowest temp:{min_temp:.2f} °C")
    # print(f"2. Averaged temp:{avg_temp:.2f}°C")
    # print(f"3. Highest temp:{max_temp:.2f}°C")
    # print("-"*50)

    # if avg_temp > 18.0:
    #     print("Averged tempture > 18°C. The heating in the flat is adequate.")
    # else:
    #     print("Averged tempture < 18°C. The flat is not warm enough.")
            
    # print("="*50 + "\n")

    # Create output directory and save arrays
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
    # Customized CLI wiring
    main(
        apt_new=args.new_apartment,
        heater_temp=args.heater_temp,
        window_temp=args.window_temp,
        wall_temp=args.wall_temp,
        num_iters=args.iters,
        dx=args.dx,
        dy=args.dy,
        solver=args.solver,
        solver_tol=args.solver_tol
    )
