# common/utils.py
import numpy as np

# Global parameters

# Room sizes
ROOM1_SIZE = (1.0, 1.0)    
ROOM2_SIZE = (1.0, 2.0)    
ROOM3_SIZE = (1.0, 1.0)    
ROOM4_SIZE = (0.5, 0.5)  # Room 4 size, for the extension task

# Default delta_x
DEFAULT_DX = 1/20          
DEFAULT_DY = 1/20           

# Boundary conditions
HEATER_TEMP = 40.0
WINDOW_TEMP = 5.0           
WALL_TEMP = 15.0

def set_boundary_conditions(heater=None, window=None, wall=None):
    """
    Allow dynamic override of boundary temperatures.
    Example:
        set_boundary_conditions(heater=45, window=10)
    """
    global HEATER_TEMP, WINDOW_TEMP, WALL_TEMP
    if heater is not None:
        HEATER_TEMP = float(heater)
    if window is not None:
        WINDOW_TEMP = float(window)
    if wall is not None:
        WALL_TEMP = float(wall)
    print(f"[Config] Updated boundary temps → Heater={HEATER_TEMP}°C, Window={WINDOW_TEMP}°C, Wall={WALL_TEMP}°C")

# Interface temp
INTERFACE_INIT_TEMP = 20.0  # initial temp

def get_room_dimensions(new_apt=False):
    
    if new_apt == False:
        return {
            "room1": ROOM1_SIZE,
            "room2": ROOM2_SIZE, 
            "room3": ROOM3_SIZE
        }
    else:
        return {
            "room1": ROOM1_SIZE,
            "room2": ROOM2_SIZE, 
            "room3": ROOM3_SIZE,
            "room4": ROOM4_SIZE
        }

def get_default_dx():
    """Return default grid spacing"""
    return DEFAULT_DX

def get_default_dy():
    """Return default grid spacing"""
    return DEFAULT_DY

def compute_grid_size(Lx, Ly, dx=None, dy=None):
    
    if dx is None:
        dx = DEFAULT_DX
    if dy is None:
        dy = DEFAULT_DY
    Nx = int(Lx / dx) + 1
    Ny = int(Ly / dy) + 1
    return Nx, Ny

def get_room_grid_info(room_id, dx=None, dy=None, new_apt=False):
    """
    Args:
        room_id : str  ('room1' / 'room2' / 'room3')
        dx, dy  : float  (grid size)
    
    Return:
        dict : {
            'Lx': float, 'Ly': float,     # room size
            'Nx': int, 'Ny': int,         # num of grids
            'dx': float, 'dy': float,     
            'x': ndarray, 'y': ndarray    # coordinates
        }
    """
    if dx is None:
        dx = DEFAULT_DX
    if dy is None:
        dy = DEFAULT_DY
    if new_apt is True:
        room_sizes = get_room_dimensions(new_apt=True) # for extension
    else:
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
    Return:
        dict : {
            'Ny_interface': list,         
            # 'y_interface': ndarray,     
        }
    """
    if dx is None:
        dx = DEFAULT_DX
    if dy is None:
        dy = DEFAULT_DY
    
    Ly_interface = ROOM1_SIZE[1]
    Ly_interface_new = ROOM4_SIZE[1] # for extension
    Ny_interface = int(Ly_interface / dy) + 1
    Ny_interface_new = int(Ly_interface_new / dy) + 1 # for extension
    # y_interface = np.linspace(0, Ly_interface, Ny_interface)
    
    return {
        'Ny_interface': [Ny_interface, Ny_interface_new],
    }

