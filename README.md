project_3_temp_simulation/
├── core/                  # 基础任务核心代码（3人负责）A B C
│   ├── matrix_builder.py  # 系数矩阵构造（对应Task1）
│   ├── mpi_solver.py      # MPI并行迭代求解（核心计算）
│   ├── visualizer.py      # 温度可视化与供暖判断（对应Task2、3）
│   └── params_tester.py   # 参数敏感性测试（对应Task4）
├── extension/             # 扩展任务代码（2人负责）
│   ├── ext_matrix.py      # 新增Ω4的矩阵构造（对应3a Task1）
│   └── ext_mpi_solver.py  # 适配4子域的MPI求解（对应3a Task1）
├── common/                # 公共工具（3人+2人共用）
│   ├── boundary_config.py # 边界条件配置（含ΓH/ΓWF定义）
│   └── utils.py           # 网格计算、数据格式转换工具
└── main.py                # 主程序（调用各模块，3人团队中1人负责整合）
