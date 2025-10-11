# common/boundary_config.py
# 采用数学坐标系，从下往上，从左往右

import numpy as np
from .utils import (
    HEATER_TEMP, WINDOW_TEMP, WALL_TEMP, INTERFACE_INIT_TEMP,
    get_interface_grid_info, get_room_grid_info
)

def get_boundary_conditions(room_id, gamma1, gamma2, gamma3=None, dx=None, dy=None):
    """
    Args:
        room_id : str  ('room1' / 'room2' / 'room3')
        gamma1  : ndarray (Ny_interface,) 
        gamma2  : ndarray (Ny_interface,) 
        gamma3  : ndarray (Ny_interface_new,) , for the extension task, default as None
        dx, dy  : float  
    
    Return:
        bc_types  : dict  {"left": "Dirichlet"/"Neumann", ...}
        bc_values : dict  {"left": value/array, ...}
    """

    # Ω1
    if room_id == "room1":
        bc_types = {
            "left":   "Dirichlet",
            "right":  "Neumann",      
            "top":    "Dirichlet",
            "bottom": "Dirichlet",
        }
        bc_values = {
            "left":   HEATER_TEMP,    
            "right":  gamma1,         
            "top":    WALL_TEMP,      
            "bottom": WALL_TEMP,
        }
        return bc_types, bc_values

    # Ω2
    elif room_id == "room2":
        
        room2_info = get_room_grid_info("room2", dx, dy)
        interface_info = get_interface_grid_info(dx, dy)
        
        Ny_room2 = room2_info['Ny']  # number of grids along y axis in room 2
        Ny_interface = interface_info['Ny_interface'][0]  # Iterface with room 1 or room 3
       
        
        left_bc = np.zeros(Ny_room2)
        left_bc[:Ny_interface] = gamma1  # bottom to the middle
        left_bc[Ny_interface:] = WALL_TEMP  
        
        if gamma3 is not None: # if interface with room4 exists
            Ny_interface_new = interface_info['Ny_interface'][1]  # Iterface with room 4
            right_bc = np.full(Ny_room2, WALL_TEMP)
            right_bc[:-Ny_interface-Ny_interface_new] = WALL_TEMP  # wall temp
            right_bc[-Ny_interface:] = gamma2     # interface with room3
            right_bc[-(Ny_interface + Ny_interface_new):-Ny_interface] = gamma3  # interface with room4
            # print(f"DEBUG: Running NEW apartment layout with Room4")

        else: # if 2 bedroom apt for project 3
            right_bc = np.full(Ny_room2, WALL_TEMP)
            right_bc[-Ny_interface:] = gamma2     # the middle to the top
            # print(f"DEBUG: Running OLD apartment layout - only Room3 interface")
  
        bc_types = {
            "left":   "Dirichlet",    
            "right":  "Dirichlet",    
            "top":    "Dirichlet",
            "bottom": "Dirichlet",
        }
        bc_values = {
            "left":   left_bc,        
            "right":  right_bc,      
            "top":    HEATER_TEMP,    
            "bottom": WINDOW_TEMP,    
        }
        # print(f"DEBUG Room2: Ny_room2={Ny_room2}, Ny_interface={Ny_interface}")
        # print(f"DEBUG Room2 right_bc: {right_bc}")

        return bc_types, bc_values

    # Ω3
    elif room_id == "room3":
        bc_types = {
            "left":   "Neumann",      
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
    
    # Ω4
    elif room_id == "room4":
        bc_types = {
            "left":   "Neumann",      
            "right":  "Dirichlet",
            "top":    "Dirichlet",
            "bottom": "Dirichlet",
        }
        bc_values = {
            "left":   gamma3,        
            "right":  WALL_TEMP,    
            "top":    WALL_TEMP,
            "bottom": HEATER_TEMP,
        }
        return bc_types, bc_values
    
    else:
        raise ValueError(f"Error: {room_id} does not exist!")


# def initialize_interface_variables(dx=None, dy=None):
#     """
#     初始化接口温度/热流分布变量
    
#     参数:
#         dx, dy : float  网格间距（可选）
    
#     返回:
#         gamma1 : ndarray (Ny_interface,) 接口Γ1初始值
#         gamma2 : ndarray (Ny_interface,) 接口Γ2初始值
#         interface_info : dict  接口网格信息
#     """
#     interface_info = get_interface_grid_info(dx, dy)
#     Ny_interface = interface_info['Ny_interface']
    
#     gamma1 = np.full(Ny_interface, INTERFACE_INIT_TEMP)
#     gamma2 = np.full(Ny_interface, INTERFACE_INIT_TEMP)
    
#     return gamma1, gamma2, interface_info
