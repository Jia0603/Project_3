import numpy as np
import os
from mpi4py import MPI 
from common.utils import get_default_dx, get_default_dy, get_room_grid_info
from common.boundary_config import get_boundary_conditions
from core.mpi_solver import dirichlet_neumann_iterate
from core.ext_mpi_solver import ext_dirichlet_neumann_iterate
import matplotlib.pyplot as plt
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Apartment Temperature Simulation')
    parser.add_argument('--new-apartment', '-n', 
                       action='store_true',
                       help='Run simulation for new apartment layout (default: old apartment)')
    return parser.parse_args()


# Initialize MPI environment (will raise error if non-MPI environment)
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

def main(apt_new=False):
    # Fixed parameters (matching MPI solver logic)
    dx = get_default_dx()
    dy = get_default_dy()
    omega = 0.8
    num_iters = 10

    # Call MPI parallel solver (subprocess returns None, main process returns result dict)
    if apt_new == False:
        result = dirichlet_neumann_iterate(dx, dy, omega=omega, num_iters=num_iters)
    else:
        result = ext_dirichlet_neumann_iterate(dx, dy, omega=omega, num_iters=num_iters)

    # Subprocess has no result, return directly (skip output/save)
    if RANK != 0:
        return

    # Extract main process solution results
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
    # Output dimension information (main process only)
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
        print("Averaged tempture > 18˚C. The heating in the flat is adequate.")
    else:
        print("Averaged tempture < 18˚C. The flat is not warm enough.")
            
    print("="*50 + "\n")

    # Create output directory and save result arrays
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
    
    
    # Main process visualization (if matplotlib available)
    def reconstruct_full(u_interior, room_id, gamma1_arr, gamma2_arr, gamma3_arr=None):
        """Reconstruct full temperature field with boundary (for visualization)"""
        new_apt=False if gamma3_arr is None else True
        info = get_room_grid_info(room_id, dx, dy, new_apt)
        Nx, Ny = info["Nx"], info["Ny"]
        bc_types, bc_values = get_boundary_conditions(room_id, gamma1_arr, gamma2_arr, gamma3_arr, dx, dy)

        # Determine solution domain indices (consistent with matrix_builder logic)
        i_start = 1 
        i_end = Nx - 1 
        j_start = 1 
        j_end = Ny - 1 

        # Initialize full temperature field and fill interior solution results
        u_full = np.full((Nx, Ny), np.nan, dtype=float)
        # u_interior is now (ny_solve, nx_solve), transpose to (nx_solve, ny_solve)
        u_full[i_start:i_end, j_start:j_end] = u_interior.T

        # Helper function: get boundary value (compatible with scalar/array)
        def get_val(v, idx):
            return v if np.isscalar(v) else v[idx]

        # Fill Dirichlet boundaries
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

        # Neumann boundaries: copy adjacent interior points (for visualization color continuity only)
        if bc_types.get("left") == "Neumann":
            u_full[0, j_start:j_end] = u_full[1, j_start:j_end]
        if bc_types.get("right") == "Neumann":
            u_full[Nx-1, j_start:j_end] = u_full[Nx-2, j_start:j_end]
        if bc_types.get("bottom") == "Neumann":
            u_full[i_start:i_end, 0] = u_full[i_start:i_end, 1]
        if bc_types.get("top") == "Neumann":
            u_full[i_start:i_end, Ny-1] = u_full[i_start:i_end, Ny-2]

        return u_full

    # Reconstruct complete temperature field for each room and save images
    u1_full = reconstruct_full(u1, "room1", gamma1, gamma2, gamma3)
    u2_full = reconstruct_full(u2, "room2", gamma1, gamma2, gamma3)
    u3_full = reconstruct_full(u3, "room3", gamma1, gamma2, gamma3)

    # Unified colorbar range (matching document boundary temperatures: 5℃-40℃)
    vmin, vmax = 5.0, 40.0

    # Save Room1 image
    plt.imshow(u1_full.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.colorbar(); plt.title("Room1 Temperature (with boundaries)")
    room1_path = os.path.join(out_dir, "room1.png")
    plt.savefig(room1_path, dpi=200); plt.close()
    print(f"Saved figure: {room1_path}")

    # Save Room2 image
    plt.figure(figsize=(4, 8))  # Long rectangle: aspect ratio 2:1
    plt.imshow(u2_full.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.colorbar(); plt.title("Room2 Temperature (with boundaries)")
    room2_path = os.path.join(out_dir, "room2.png")
    plt.savefig(room2_path, dpi=200); plt.close()
    print(f"Saved figure: {room2_path}")

    # Save Room3 image
    plt.imshow(u3_full.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.xlabel('x [m]'); plt.ylabel('y [m]')
    plt.colorbar(); plt.title("Room3 Temperature (with boundaries)")
    room3_path = os.path.join(out_dir, "room3.png")
    plt.savefig(room3_path, dpi=200); plt.close()
    print(f"Saved figure: {room3_path}")

    # Save Room4 image
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