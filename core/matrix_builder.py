# core/matrix_builder.py
import numpy as np

def build_laplace_matrix_mixed(nx, ny, h, bc_types):
    """
    构造二维五点差分矩阵 A（混合边界条件）
    
    注意: 
    - Dirichlet边界: 边界点值已知，不在求解域内
    - Neumann边界: 边界点未知，在求解域内，使用单边差分
    
    参数:
        nx, ny : 网格点数（包括所有Neumann边界点）
        h      : 网格间距
        bc_types : 边界类型字典
                   {"left": "Dirichlet"/"Neumann", 
                    "right": "Dirichlet"/"Neumann",
                    "top": "Dirichlet"/"Neumann", 
                    "bottom": "Dirichlet"/"Neumann"}
    
    返回:
        A : 差分矩阵
        注意: 矩阵大小取决于边界条件
    """
    # 确定实际求解的网格范围
    i_start = 1 if bc_types.get("left") == "Dirichlet" else 0
    i_end = nx - 1 if bc_types.get("right") == "Dirichlet" else nx
    j_start = 1 if bc_types.get("bottom") == "Dirichlet" else 0
    j_end = ny - 1 if bc_types.get("top") == "Dirichlet" else ny
    
    nx_solve = i_end - i_start
    ny_solve = j_end - j_start
    N = nx_solve * ny_solve
    A = np.zeros((N, N))
    
    for j in range(ny_solve):
        for i in range(nx_solve):
            p = i + j * nx_solve
            diag = 0.0
            
            # 全局坐标
            i_global = i + i_start
            j_global = j + j_start
            
            # --- 左邻居 ---
            if i > 0:  # 内部左邻居
                A[p, p-1] = 1.0
                diag -= 1.0
            elif i_global > 0:  # 左边是Dirichlet边界
                diag -= 1.0  # 边界贡献在右端项
            # else: 左边是Neumann边界（i_global == 0）
                
            # --- 右邻居 ---
            if i < nx_solve - 1:  # 内部右邻居
                A[p, p+1] = 1.0
                diag -= 1.0
            elif i_global < nx - 1:  # 右边是Dirichlet边界
                diag -= 1.0
            # else: 右边是Neumann边界
                
            # --- 下邻居 ---
            if j > 0:
                A[p, p-nx_solve] = 1.0
                diag -= 1.0
            elif j_global > 0:
                diag -= 1.0
                
            # --- 上邻居 ---
            if j < ny_solve - 1:
                A[p, p+nx_solve] = 1.0
                diag -= 1.0
            elif j_global < ny - 1:
                diag -= 1.0
            
            A[p, p] = diag
    
    return A / (h*h), nx_solve, ny_solve

def build_b_dirichlet(nx, ny, h, bc):
    """
    构造全Dirichlet边界条件的右端项 b
    
    参数:
        nx, ny : 总网格点数（包括边界）
        h      : 网格间距
        bc     : 边界条件字典
        f      : 源项（可选），形状 (nx, ny) 或标量
    
    返回:
        b : ((nx-2)*(ny-2),) 右端项向量
    """
    nx_inner = nx - 2
    ny_inner = ny - 2
    b = np.zeros(nx_inner * ny_inner)
    
    # 处理边界条件
    for j in range(ny_inner):
        for i in range(nx_inner):
            p = i + j * nx_inner
            j_global = j + 1  # 对应全局网格的j索引
            i_global = i + 1
            
            # --- 左边界 (i_global = 1, 邻居是边界) ---
            if i == 0 and "left" in bc:
                val = _get_bc_value(bc["left"]["value"], j_global)
                b[p] -= (1/h**2) * val
            
            # --- 右边界 (i_global = nx-2, 邻居是边界) ---
            if i == nx_inner - 1 and "right" in bc:
                val = _get_bc_value(bc["right"]["value"], j_global)
                b[p] -= (1/h**2) * val
            
            # --- 下边界 (j_global = 1, 邻居是边界) ---
            if j == 0 and "bottom" in bc:
                val = _get_bc_value(bc["bottom"]["value"], i_global)
                b[p] -= (1/h**2) * val
            
            # --- 上边界 (j_global = ny-2, 邻居是边界) ---
            if j == ny_inner - 1 and "top" in bc:
                val = _get_bc_value(bc["top"]["value"], i_global)
                b[p] -= (1/h**2) * val
    
    return b


def build_b_mixed(nx, ny, h, bc_types, bc_values):
    """
    构造混合边界条件的右端项 b
    
    参数:
        nx, ny : 网格点数
        h      : 网格间距
        bc_types : 边界类型字典 {"left": "Dirichlet"/"Neumann", ...}
        bc_values : 边界值字典 {"left": value/array, ...}
        f      : 源项（可选）
    
    返回:
        b : 右端项向量（大小取决于边界条件）
    """
    # 确定求解域
    i_start = 1 if bc_types.get("left") == "Dirichlet" else 0
    i_end = nx - 1 if bc_types.get("right") == "Dirichlet" else nx
    j_start = 1 if bc_types.get("bottom") == "Dirichlet" else 0
    j_end = ny - 1 if bc_types.get("top") == "Dirichlet" else ny
    
    nx_solve = i_end - i_start
    ny_solve = j_end - j_start
    b = np.zeros(nx_solve * ny_solve)
    

    # 处理边界条件
    for j in range(ny_solve):
        for i in range(nx_solve):
            p = i + j * nx_solve
            i_global = i + i_start
            j_global = j + j_start
            
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