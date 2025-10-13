"""
core/visualizer.py
-----------------------------------
Handles visualization for Project 3 & 3a:
 - Fill corner NaNs in each room
 - Draw each room individually
 - Compute overall heating adequacy (no image stitching)
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from common.utils import get_room_grid_info
from common.boundary_config import get_boundary_conditions



# ===================================
# --- reconstruct single room ---
# ===================================
def reconstruct_full(u_interior, room_id, gamma1_arr, gamma2_arr, gamma3_arr=None, dx=1/20, dy=1/20):
    """
    Reconstruct full temperature field (including boundary).
    Works for both 3-room and 4-room layouts.
    """
    new_apt = False if gamma3_arr is None else True
    info = get_room_grid_info(room_id, dx, dy, new_apt)
    Nx, Ny = info["Nx"], info["Ny"]
    bc_types, bc_values = get_boundary_conditions(room_id, gamma1_arr, gamma2_arr, gamma3_arr, dx, dy)

    i_start, i_end = 1, Nx - 1
    j_start, j_end = 1, Ny - 1
    u_full = np.full((Nx, Ny), np.nan, dtype=float)
    u_full[i_start:i_end, j_start:j_end] = u_interior.T

    def get_val(v, idx):
        return v if np.isscalar(v) else v[idx]

    # Dirichlet boundaries
    for side in ["left", "right", "bottom", "top"]:
        if bc_types.get(side) == "Dirichlet":
            if side in ["left", "right"]:
                i = 0 if side == "left" else Nx - 1
                for jg in range(j_start, j_end):
                    u_full[i, jg] = get_val(bc_values[side], jg)
            else:
                j = 0 if side == "bottom" else Ny - 1
                for ig in range(i_start, i_end):
                    u_full[ig, j] = get_val(bc_values[side], ig)

    # Neumann boundaries: copy neighbor cells
    if bc_types.get("left") == "Neumann":
        u_full[0, j_start:j_end] = u_full[1, j_start:j_end]
    if bc_types.get("right") == "Neumann":
        u_full[-1, j_start:j_end] = u_full[-2, j_start:j_end]
    if bc_types.get("bottom") == "Neumann":
        u_full[i_start:i_end, 0] = u_full[i_start:i_end, 1]
    if bc_types.get("top") == "Neumann":
        u_full[i_start:i_end, -1] = u_full[i_start:i_end, -2]

    # --- Fill corners (avoid NaN) ---
    u_full[0, 0] = u_full[1, 1]
    u_full[-1, 0] = u_full[-2, 1]
    u_full[0, -1] = u_full[1, -2]
    u_full[-1, -1] = u_full[-2, -2]
    return u_full


# ===================================
# --- draw each room individually ---
# ===================================
def visualize_all_rooms(result, dx, dy, apt_new=False):
    """
    Visualize each room separately (3-room or 4-room version).
    Return reconstructed u_fulls for later heating evaluation.
    """
    out_dir = os.path.join(os.path.dirname(__file__), "..", "ext_output" if apt_new else "output")
    os.makedirs(out_dir, exist_ok=True)

    u1, u2, u3 = result["u1"], result["u2"], result["u3"]
    gamma1, gamma2 = result["gamma1"], result["gamma2"]
    gamma3 = result.get("gamma3", None)
    u4 = result.get("u4", None)

    u1_full = reconstruct_full(u1, "room1", gamma1, gamma2, gamma3, dx, dy)
    u2_full = reconstruct_full(u2, "room2", gamma1, gamma2, gamma3, dx, dy)
    u3_full = reconstruct_full(u3, "room3", gamma1, gamma2, gamma3, dx, dy)

    vmin, vmax = 5.0, 40.0
    def save_plot(u_full, title, path, figsize=(4, 4)):
        plt.figure(figsize=figsize)
        plt.imshow(u_full.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(label="Temperature (°C)")
        plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"Saved: {path}")

    save_plot(u1_full, "Room 1 Temperature", os.path.join(out_dir, "room1.png"))
    save_plot(u2_full, "Room 2 Temperature", os.path.join(out_dir, "room2.png"), figsize=(4, 8))
    save_plot(u3_full, "Room 3 Temperature", os.path.join(out_dir, "room3.png"))
    if apt_new and u4 is not None:
        u4_full = reconstruct_full(u4, "room4", gamma1, gamma2, gamma3, dx, dy)
        save_plot(u4_full, "Room 4 Temperature", os.path.join(out_dir, "room4.png"))
    else:
        u4_full = None

    return u1_full, u2_full, u3_full, u4_full


# ===================================
# --- compute & judge heating ---
# ===================================
def judge_heating(u1_full, u2_full, u3_full, u4_full=None, threshold=18):
    """
    Compute average temperature across all rooms
    and judge heating adequacy based only on average > threshold.
    """
    all_vals = []
    for arr in [u1_full, u2_full, u3_full, u4_full]:
        if arr is not None:
            all_vals.extend(arr[~np.isnan(arr)].ravel())

    all_vals = np.array(all_vals)
    avg_temp = np.mean(all_vals)
    min_temp = np.min(all_vals)
    max_temp = np.max(all_vals)

    print("\n===== Heating Evaluation =====")
    print(f"Average temperature: {avg_temp:.2f} °C")
    print(f"Minimum temperature: {min_temp:.2f} °C")
    print(f"Maximum temperature: {max_temp:.2f} °C")
    print("==============================")

    if avg_temp > threshold:
        print(f"Averaged temperature > {threshold}°C → Heating is adequate.\n")
    else:
        print(f"Averaged temperature < {threshold}°C → The flat is not warm enough.\n")

    return avg_temp, min_temp, max_temp



# ===================================
# --- one-call convenience pipeline ---
# ===================================
def visualize_pipeline(result, dx, dy, apt_new=False):
    """
    One-call pipeline to visualize rooms and evaluate heating.
    Does NOT stitch or plot a full-apartment image.
    """
    u1_full, u2_full, u3_full, u4_full = visualize_all_rooms(result, dx, dy, apt_new)
    judge_heating(u1_full, u2_full, u3_full, u4_full)
