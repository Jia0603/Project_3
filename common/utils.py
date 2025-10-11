# common/utils.py
import numpy as np

# ==================== 全局参数定义 ====================

# 房间尺寸（单位：米）
ROOM1_SIZE = (1.0, 1.0)    # Room 1: 1m × 1m (正方形)
ROOM2_SIZE = (1.0, 2.0)    # Room 2: 1m × 2m (矩形，高度是Room1的2倍)
ROOM3_SIZE = (1.0, 1.0)    # Room 3: 1m × 1m (正方形)

# 网格参数
DEFAULT_DX = 1/20           # 默认网格间距
DEFAULT_DY = 1/20           # 默认网格间距

# 边界条件参数
HEATER_TEMP = 40.0          # 暖气温度
WINDOW_TEMP = 5.0           # 窗户温度
WALL_TEMP = 15.0            # 普通墙温度

# 接口参数
INTERFACE_INIT_TEMP = 20.0  # 接口初始温度

def get_room_dimensions():
    """返回所有房间的尺寸"""
    return {
        "room1": ROOM1_SIZE,
        "room2": ROOM2_SIZE, 
        "room3": ROOM3_SIZE
    }

def get_default_dx():
    """返回默认网格间距"""
    return DEFAULT_DX

def get_default_dy():
    """返回默认网格间距"""
    return DEFAULT_DY

def compute_grid_size(Lx, Ly, dx=None, dy=None):
    """根据区域尺寸和步长计算网格点数量"""
    if dx is None:
        dx = DEFAULT_DX
    if dy is None:
        dy = DEFAULT_DY
    Nx = int(Lx / dx) + 1
    Ny = int(Ly / dy) + 1
    return Nx, Ny

def get_room_grid_info(room_id, dx=None, dy=None):
    """
    获取指定房间的网格信息
    
    参数:
        room_id : str  ('room1' / 'room2' / 'room3')
        dx, dy  : float  网格间距（可选）
    
    返回:
        dict : {
            'Lx': float, 'Ly': float,     # 房间尺寸
            'Nx': int, 'Ny': int,         # 网格点数
            'dx': float, 'dy': float,     # 网格间距
            'x': ndarray, 'y': ndarray    # 网格坐标
        }
    """
    if dx is None:
        dx = DEFAULT_DX
    if dy is None:
        dy = DEFAULT_DY
    
    room_sizes = get_room_dimensions()
    Lx, Ly = room_sizes[room_id]
    Nx, Ny = compute_grid_size(Lx, Ly, dx, dy)
    
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    
    return {
        'Lx': Lx, 'Ly': Ly,
        'Nx': Nx, 'Ny': Ny,
        'dx': dx, 'dy': dy,
        'x': x, 'y': y
    }

def get_interface_grid_info(dx=None, dy=None):
    """
    获取接口的网格信息
    返回:
        dict : {
            'Ny_interface': int,          # 接口网格点数
            # 'y_interface': ndarray,        # 接口y坐标
        }
    """
    if dx is None:
        dx = DEFAULT_DX
    if dy is None:
        dy = DEFAULT_DY
    
    Ly_interface = ROOM1_SIZE[1] 
    Ny_interface = int(Ly_interface / dy) + 1
    # y_interface = np.linspace(0, Ly_interface, Ny_interface)
    
    return {
        'Ny_interface': Ny_interface,
    }

def generate_grid(Lx, Ly, dx):
    """生成笛卡尔网格坐标"""
    Nx, Ny = compute_grid_size(Lx, Ly, dx)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return X, Y, Nx, Ny


def extract_left_boundary(u_2d):
    """
    从二维解矩阵中提取左边界的值
    """
    return u_2d[0, :].copy()


def extract_right_boundary(u_2d):
    """
    从二维解矩阵中提取右边界的值
    """
    return u_2d[-1, :].copy()


def extract_bottom_boundary(u_2d):
    """
    从二维解矩阵中提取下边界的值
    """
    return u_2d[:, 0].copy()


def extract_top_boundary(u_2d):
    """
    从二维解矩阵中提取上边界的值
    """
    return u_2d[:, -1].copy()