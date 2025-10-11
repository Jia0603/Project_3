# common/boundary_config.py
# 采用数学坐标系，从下往上，从左往右

import numpy as np
from .utils import (
    HEATER_TEMP, WINDOW_TEMP, WALL_TEMP, INTERFACE_INIT_TEMP,
    get_interface_grid_info, get_room_grid_info
)

def get_boundary_conditions(room_id, gamma1, gamma2, dx=None, dy=None):
    """
    返回某个房间的边界条件字典。

    参数:
        room_id : str  ('room1' / 'room2' / 'room3')
        gamma1  : ndarray (Ny_interface,) 
                  接口Γ1的温度/热流分布（沿y方向）
                  用于：room1右边界 ↔ room2左边界
        gamma2  : ndarray (Ny_interface,) 
                  接口Γ2的温度/热流分布（沿y方向）
                  用于：room2右边界 ↔ room3左边界
        dx, dy  : float  网格间距（可选）
    
    返回:
        bc_types  : dict 边界类型字典 {"left": "Dirichlet"/"Neumann", ...}
        bc_values : dict 边界值字典 {"left": value/array, ...}
    """

    # --- 左房 (Ω1): 左暖气墙, 上下普通, 右边界是接口Γ1 ---
    if room_id == "room1":
        bc_types = {
            "left":   "Dirichlet",
            "right":  "Neumann",      # 接口Γ1（热流边界）
            "top":    "Dirichlet",
            "bottom": "Dirichlet",
        }
        bc_values = {
            "left":   HEATER_TEMP,    # 暖气墙（标量）
            "right":  gamma1,         # 接口Γ1（整个高度）
            "top":    WALL_TEMP,      # 普通墙
            "bottom": WALL_TEMP,
        }
        return bc_types, bc_values

    # --- 中房 (Ω2): 上暖气, 下窗户, 左边界Γ1, 右边界混合 ---
    elif room_id == "room2":
        # 从 utils 获取网格信息
        room2_info = get_room_grid_info("room2", dx, dy)
        interface_info = get_interface_grid_info(dx, dy)
        
        Ny_room2 = room2_info['Ny']  # Room 2 的 y 方向网格点数
        Ny_interface = interface_info['Ny_interface']  # 接口的网格点数
        
        # 需要构造混合边界：下半部分是接口，上半部分是墙
        # 构造左边界：下半部分是接口Γ1，上半部分是恒温
        left_bc = np.zeros(Ny_room2)
        left_bc[:Ny_interface] = gamma1  # 下半部分是接口Γ1
        left_bc[Ny_interface:] = WALL_TEMP  # 上半部分是恒温
        
        # 构造右边界：上半部分是接口Γ2，下半部分是恒温
        right_bc = np.zeros(Ny_room2)
        right_bc[:-Ny_interface] = WALL_TEMP  # 下半部分是恒温
        right_bc[-Ny_interface:] = gamma2     # 上半部分是接口Γ2
        
        bc_types = {
            "left":   "Dirichlet",    # 混合边界（部分接口，部分恒温）
            "right":  "Dirichlet",    # 混合边界（部分接口，部分恒温）
            "top":    "Dirichlet",
            "bottom": "Dirichlet",
        }
        bc_values = {
            "left":   left_bc,        # 拼接数组（下半接口，上半恒温）
            "right":  right_bc,       # 拼接数组（下半恒温，上半接口）
            "top":    HEATER_TEMP,    # 顶部暖气
            "bottom": WINDOW_TEMP,    # 底部窗户
        }
        return bc_types, bc_values

    # --- 右房 (Ω3): 右外墙, 上下普通, 左边界是接口Γ2 ---
    elif room_id == "room3":
        bc_types = {
            "left":   "Neumann",      # 接口Γ2（热流边界）
            "right":  "Dirichlet",
            "top":    "Dirichlet",
            "bottom": "Dirichlet",
        }
        bc_values = {
            "left":   gamma2,        
            "right":  HEATER_TEMP,    
            "top":    WALL_TEMP,
            "bottom": WALL_TEMP,
        }
        return bc_types, bc_values

    else:
        raise ValueError(f"未知的房间 ID: {room_id}")


def initialize_interface_variables(dx=None, dy=None):
    """
    初始化接口温度/热流分布变量
    
    参数:
        dx, dy : float  网格间距（可选）
    
    返回:
        gamma1 : ndarray (Ny_interface,) 接口Γ1初始值
        gamma2 : ndarray (Ny_interface,) 接口Γ2初始值
        interface_info : dict  接口网格信息
    """
    interface_info = get_interface_grid_info(dx, dy)
    Ny_interface = interface_info['Ny_interface']
    
    gamma1 = np.full(Ny_interface, INTERFACE_INIT_TEMP)
    gamma2 = np.full(Ny_interface, INTERFACE_INIT_TEMP)
    
    return gamma1, gamma2, interface_info
