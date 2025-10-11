project_3_temp_simulation/
├── core/                  
│   ├── matrix_builder.py  # 系数矩阵构造（A）已完成
│   ├── mpi_solver.py      # MPI并行迭代求解（B）已完成
│   ├── visualizer.py      # 温度可视化与供暖判断（C）
│   └── params_tester.py   # 参数敏感性测试（C）
│   └── ext_mpi_solver.py  # 适配4子域的MPI求解（D）已完成
├── common/                
│   ├── boundary_config.py # 边界条件配置（A）已修改适用extension
│   └── utils.py           # 网格计算、数据格式转换工具(A) 已修改适用extension
└── main.py                # 主程序（B）已修改适用extension

初始化：

from common.utils import *
from common.boundary_config import *
from core.matrix_builder import *

dx = dy = 1/20
omega = 0.8

gamma1, gamma2, interface_info = initialize_interface_variables(dx, dy)

info1 = get_room_grid_info("room1", dx, dy)
info2 = get_room_grid_info("room2", dx, dy)
info3 = get_room_grid_info("room3", dx, dy)

获取边界信息

get_boundary_conditions("room2", gamma1, gamma2)

拉普拉斯方程A,b

A2, nx_solve2, ny_solve2 = build_laplace_matrix_mixed(info2["Nx"], info2["Ny"], dx, bc2[0])

b2 = build_b_mixed(info2["Nx"], info2["Ny"], dx, bc2[0], bc2[1])

u2 = np.linalg.solve(A2, b2).reshape(nx_solve2, ny_solve2)


Dirichlet条件下，A2尺寸是Nx-2, Ny-2不包含边界

Neumann边界会被包含用来求解


**************** mpi_solver.py / main.py ****************

运行： 
    mpiexec -n 4 python main.py
    mpiexec -n 5 python main.py -n for the extention task

结果： 
1. 3个房间的温度场，分别保存在u1.npy, u2.npy, u3.npy中；
2. gamma1.npy / gamma2.npy两个子域接口的温度场；
3. 判断是否达到供暖效果，仅看平均温度是否>18℃;
4. 作图这一部分在main.py中已注释，srz可以根据注释的作图代码调整 or 自己再重新写个;
5. 以上结果都保存在output文件夹中。extension的结果报存在ext_output文件夹中
