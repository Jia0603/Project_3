project_3_temp_simulation/
├── core/                  # 基础任务核心代码（A B C)
│   ├── matrix_builder.py  # 系数矩阵构造（A）
│   ├── mpi_solver.py      # MPI并行迭代求解（B）
│   ├── visualizer.py      # 温度可视化与供暖判断（C）
│   └── params_tester.py   # 参数敏感性测试（C）
├── extension/             # 扩展任务代码（D E）
│   ├── ext_matrix.py      # 新增Ω4的矩阵构造
│   └── ext_mpi_solver.py  # 适配4子域的MPI求解
├── common/                # 公共工具（A B C）
│   ├── boundary_config.py # 边界条件配置（A）
│   └── utils.py           # 网格计算、数据格式转换工具(A)
└── main.py                # 主程序（B）
