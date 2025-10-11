# core/matrix_builder.py
import numpy as np

def build_laplace_matrix_mixed(nx, ny, h, bc_types):
    
    nx_solve = nx - 2
    ny_solve = ny - 2
    N = nx_solve * ny_solve
    A = np.zeros((N, N))
    
    for j in range(ny_solve):
        for i in range(nx_solve):
            p = i + j * nx_solve
            
            # 标准五点模板的中心系数
            central_coeff = -4.0 / h**2
            
            # 检查是否在边界上，若是 Neumann 边界则修改系数
            # Ghost point 方法：u_ghost = u_interior，使该方向导数项消失
            
            # 左边界 (i = 0)
            if i == 0 and bc_types.get("left") == "Neumann":
                central_coeff += 1.0 / h**2  
            
            # 右边界 (i = nx_solve - 1)
            if i == nx_solve - 1 and bc_types.get("right") == "Neumann":
                central_coeff += 1.0 / h**2
            
            # 下边界 (j = 0)
            if j == 0 and bc_types.get("bottom") == "Neumann":
                central_coeff += 1.0 / h**2
            
            # 上边界 (j = ny_solve - 1)
            if j == ny_solve - 1 and bc_types.get("top") == "Neumann":
                central_coeff += 1.0 / h**2
            
            A[p, p] = central_coeff
            
            # 右邻居 (i+1, j)
            if i < nx_solve - 1:
                A[p, p+1] = 1.0 / h**2
            
            # 左邻居 (i-1, j)
            if i > 0:
                A[p, p-1] = 1.0 / h**2
            
            # 上邻居 (i, j+1)
            if j < ny_solve - 1:
                A[p, p+nx_solve] = 1.0 / h**2
            
            # 下邻居 (i, j-1)
            if j > 0:
                A[p, p-nx_solve] = 1.0 / h**2
    
    return A, nx_solve, ny_solve


def build_b_mixed(nx, ny, h, bc_types, bc_values):

    nx_solve = nx - 2
    ny_solve = ny - 2
    b = np.zeros(nx_solve * ny_solve)
    
    for j in range(ny_solve):
        for i in range(nx_solve):
            p = i + j * nx_solve
            i_global = i + 1
            j_global = j + 1
            
            # --- 左边界 ---
            if i == 0 and "left" in bc_values:
                if bc_types.get("left") == "Dirichlet":
                    val = _get_bc_value(bc_values["left"], j_global)
                    b[p] -= (1/h**2) * val
                elif bc_types.get("left") == "Neumann":
                    g = _get_bc_value(bc_values["left"], j_global)
                    b[p] -= (1/h) * g
            
            # --- 右边界 ---
            if i == nx_solve - 1 and "right" in bc_values:
                if bc_types.get("right") == "Dirichlet":
                    val = _get_bc_value(bc_values["right"], j_global)
                    b[p] -= (1/h**2) * val
                elif bc_types.get("right") == "Neumann":
                    g = _get_bc_value(bc_values["right"], j_global)
                    b[p] += (1/h) * g
            
            # --- 下边界 ---
            if j == 0 and "bottom" in bc_values:
                if bc_types.get("bottom") == "Dirichlet":
                    val = _get_bc_value(bc_values["bottom"], i_global)
                    b[p] -= (1/h**2) * val
                elif bc_types.get("bottom") == "Neumann":
                    g = _get_bc_value(bc_values["bottom"], i_global)
                    b[p] -= (1/h) * g
            
            # --- 上边界 ---
            if j == ny_solve - 1 and "top" in bc_values:
                if bc_types.get("top") == "Dirichlet":
                    val = _get_bc_value(bc_values["top"], i_global)
                    b[p] -= (1/h**2) * val
                elif bc_types.get("top") == "Neumann":
                    g = _get_bc_value(bc_values["top"], i_global)
                    b[p] += (1/h) * g
    
    return b


def _get_bc_value(value, index):
    """
    辅助函数：从边界值中提取特定索引的值
    
    参数:
        value : 标量或数组
        index : 索引
    
    返回:
        标量值
    """
    if np.isscalar(value):
        return value
    elif isinstance(value, (np.ndarray, list)):
        return value[index]
    else:
        return value


def print_laplace_matrix(A, room_id, nx_solve, ny_solve, h):
    """
    Print detailed information about the Laplace matrix
    
    Parameters:
        A : Laplace matrix
        room_id : Room ID (string, e.g. "room1")
        nx_solve : Number of grid points in x-direction
        ny_solve : Number of grid points in y-direction
        h : Grid spacing
    """
    print(f"\n{'='*70}")
    print(f"Room: {room_id}")
    print(f"Grid spacing h: {h}")
    print(f"Solution domain size: nx_solve = {nx_solve}, ny_solve = {ny_solve}")
    print(f"Matrix dimension: {A.shape[0]} × {A.shape[1]}")
    print(f"{'='*70}")
    
    # Set numpy print options
    np.set_printoptions(precision=4, suppress=True, linewidth=150, threshold=10000)
    
    print("Laplace Matrix A:")
    print(A)
    
    # Print matrix statistics
    print(f"\nMatrix Statistics:")
    print(f"  - Diagonal elements range: [{np.min(np.diag(A)):.4f}, {np.max(np.diag(A)):.4f}]")
    print(f"  - Number of non-zero elements: {np.count_nonzero(A)}")
    print(f"  - Matrix sparsity: {100 * (1 - np.count_nonzero(A) / A.size):.2f}%")
    print(f"{'='*70}\n")
