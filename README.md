## Project Structure

    project_3_temp_simulation/
    ├── core/
    │   ├── matrix_builder.py   # Coefficient matrix
    │   ├── mpi_solver.py       # MPI parallel Dirichlet–Neumann iteration
    │   ├── visualizer.py       # Visualization and heating adequacy check
    │   ├── params_tester.py    # Parameter sensitivity tests
    │   └── ext_mpi_solver.py   # MPI solver for 4 subdomains (extension)
    ├── common/
    │   ├── boundary_config.py  # Boundary condition configuration (extension-ready)
    │   └── utils.py            # Grid utilities and helpers (extension-ready)
    └── main.py                 # Entrypoint (extension-ready)

## Run
    mpiexec -n 4 python main.py
    mpiexec -n 5 python main.py -n   # for the 4-room extension

Note: You can run MPI processes in project_3.ipynb instead of using command above.

Outputs:
1) Temperature fields for rooms saved to u1.npy, u2.npy, u3.npy (and u4.npy for extension)
2) Interface arrays gamma1.npy / gamma2.npy (and gamma3.npy for extension)
3) Heating adequacy check based on the overall average temperature (> 18°C)
4) Visualization is implemented in visualizer.py (handles 3/4 rooms and corner fill)
5) Results are saved under output/ (and ext_output/ for the extension)

CLI examples for varying parameters (heater temperature):

    mpiexec -n 4 python main.py --heater-temp 20
    
Other parameters can be provided similarly; defaults match the assignment specification.


## Group contributions
Zhe Zhang: implemented mpi_solver.py for parallel Dirichlet–Neumann iteration with MPI and main.py as the entrypoint for orchestrating the thermal simulation and result analysis.

Jiazhuang Chen: Writing the notebook and code of Extension Task 2 section

Ruizhen Shen: implemented the visualization for rooms, the heating evaluation and tested different parameters.

Jiuen Feng: Writing the initialized settings. Room config and matrix build.

Jia Gu: Implemented ext_mpi_solver.py, and modified utils.py, boundary_config.py, and main.py for the extension task.
