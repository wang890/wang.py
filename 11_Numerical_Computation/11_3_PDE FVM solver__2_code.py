# 导入基本计算和绘图库
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# --- 物理和模拟参数设置 ---
# 砖块尺寸 (单位: 米)
LENGTH = 0.10  # x方向, 10cm
HEIGHT = 0.05  # y方向, 5cm
DEPTH = 0.07   # z方向, 7cm

# 材料属性 (以普通红砖为例)
THERMAL_CONDUCTIVITY = 0.6  # 导热系数 k (W/m·K)
DENSITY = 1900.0            # 密度 rho (kg/m^3)
SPECIFIC_HEAT = 840.0       # 比热容 c_p (J/kg·K)

# 计算热扩散率 alpha
THERMAL_DIFFUSIVITY = THERMAL_CONDUCTIVITY / (DENSITY * SPECIFIC_HEAT)

# 初始温度和边界温度 (单位: 摄氏度)
T_INITIAL = 20.0
T_LEFT_BOUNDARY = 200.0

# 空间离散化设置
NX = 20  # x方向的网格节点数
NY = 10  # y方向的网格节点数
NZ = 14  # z方向的网格节点数

# 时间离散化设置
TOTAL_SIM_TIME = 3600 * 2  # 总模拟时间 (2小时)

# --- 任务 (1): 空间离散化与网格可视化 ---

def create_grid(length, height, depth, nx, ny, nz):
    """
    创建三维笛卡尔网格的节点坐标和步长。

    返回:
    x, y, z : 1D numpy arrays，包含每个方向上节点的中心坐标。
    dx, dy, dz : float，每个方向上的网格间距（控制体尺寸）。
    """
    # 解释: np.linspace生成一个等差数列，从0到尺寸上限，共nx个点。
    # 这代表了每个控制体中心的x, y, z坐标。
    x = np.linspace(0, length, nx)
    y = np.linspace(0, height, ny)
    z = np.linspace(0, depth, nz)

    # 解释: 计算每个方向上的网格间距(步长)。
    # 这是相邻节点之间的距离。
    dx = length / (nx - 1)
    dy = height / (ny - 1)
    dz = depth / (nz - 1)
    
    return x, y, z, dx, dy, dz

def plot_grid(x, y, z):
    """
    将离散后的三维网格节点进行可视化。
    """
    # 解释: 创建一个3D图形对象。
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 解释: np.meshgrid从一维坐标数组创建三维坐标网格。
    # X, Y, Z是三维数组，存储了每个节点的(x,y,z)坐标。
    X, Y, Z = np.meshgrid(x, y, z)

    # 解释: ax.scatter在3D空间中绘制所有节点，'s=5'设置点的大小。
    ax.scatter(X, Y, Z, c='b', s=5, label='Grid Nodes')

    # 解释: 设置坐标轴标签，清晰地表示方向。
    ax.set_xlabel('X-axis (Length = 10cm)')
    ax.set_ylabel('Y-axis (Height = 5cm)')
    ax.set_zlabel('Z-axis (Depth = 7cm)')

    # 解释: 设置图标题。
    ax.set_title('3D Discretized Grid')
    
    # 解释: 标记坐标原点(0,0,0)。
    ax.scatter(0, 0, 0, c='r', s=50, label='Origin (0,0,0)')
    ax.legend()
    
    # 解释: 显示图形。
    plt.show()

# --- 任务 (2): 有限体积法传热过程实现 ---

def solve_heat_conduction_fvm(grid_params, sim_params, material_props, conditions):
    """
    使用有限体积法和显式欧拉格式求解三维瞬态热传导问题。
    """
    # --- 解包所有传入的参数 ---
    # 解释: 从元组中提取网格参数，方便使用。
    x, y, z, dx, dy, dz = grid_params
    nx, ny, nz = len(x), len(y), len(z)

    # 解释: 提取模拟时间参数。
    total_time = sim_params['total_time']
    dt = sim_params['dt']

    # 解释: 提取材料和边界条件参数。
    alpha = material_props['alpha']
    t_initial = conditions['t_initial']
    t_left_bc = conditions['t_left_bc']
    
    # --- 初始化温度场 ---
    # 解释: 创建一个(nx, ny, nz)的三维数组来存储每个节点的温度。
    # np.full用指定的初始温度t_initial填充整个数组。
    T = np.full((nx, ny, nz), t_initial)

    # --- 核心时间迭代循环 ---
    # 解释: time_steps是模拟时间的离散点。我们从t=0循环到总模拟时间。
    time_steps = np.arange(0, total_time, dt)
    for t_step, t in enumerate(time_steps):
        # 解释: 在每个时间步开始时，复制当前的温度场。
        # 在显式格式中，计算新温度必须使用前一时间步的旧温度，
        # 如果不复制，计算一个点会立即影响到下一个点的计算，这是错误的。
        T_old = T.copy()

        # --- 遍历所有内部节点 ---
        # 解释: 这三个嵌套循环遍历了除边界外的所有内部控制体。
        # 边界节点将由边界条件特殊处理。
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    # 解释: 这是3D热传导方程的FVM离散形式。
                    # laplacian_x/y/z 分别计算x,y,z三个方向上的二阶导数近似。
                    # 这是通过中心差分实现的，代表了流入/流出节点的净热流。
                    laplacian_x = (T_old[i + 1, j, k] - 2 * T_old[i, j, k] + T_old[i - 1, j, k]) / (dx**2)
                    laplacian_y = (T_old[i, j + 1, k] - 2 * T_old[i, j, k] + T_old[i, j - 1, k]) / (dy**2)
                    laplacian_z = (T_old[i, j, k + 1] - 2 * T_old[i, j, k] + T_old[i, j, k - 1]) / (dz**2)

                    # 解释: 这是显式欧拉法的更新公式: T_new = T_old + dt * (变化率)。
                    # 变化率是 热扩散率 * (总的拉普拉斯项)。
                    T[i, j, k] = T_old[i, j, k] + alpha * dt * (laplacian_x + laplacian_y + laplacian_z)
        
        # --- 应用边界条件 ---
        # 解释: 左侧面 (x=0) 是第一类边界条件（恒定温度）。
        # 我们强制将这个面上的所有节点的温度设为指定值。
        T[0, :, :] = t_left_bc

        # 解释: 其他五个面是第二类边界条件（绝热，热通量为0）。
        # 我们通过令边界节点的温度等于其相邻内部节点的温度来实现这一点。
        # 这相当于在边界上创建了一个虚拟的“镜像”节点，从而使得温度梯度为0。
        T[nx - 1, :, :] = T[nx - 2, :, :]  # 右侧面 (x=L)
        T[:, ny - 1, :] = T[:, ny - 2, :]  # 顶面 (y=H)
        T[:, 0, :]      = T[:, 1, :]      # 底面 (y=0)
        T[:, :, nz - 1] = T[:, :, nz - 2]  # 前面 (z=D)
        T[:, :, 0]      = T[:, :, 1]      # 后面 (z=0)
        
        # --- 打印进度 (可选) ---
        if (t_step + 1) % 100 == 0:
            print(f"Time: {t:.2f} s / {total_time} s, Progress: {((t+dt)/total_time)*100:.1f}%")

    # 解释: 循环结束后，返回最终的温度分布场。
    return T, x, y, z

# --- 主执行程序 ---
if __name__ == '__main__':
    # (1) 创建并可视化网格
    grid_data = create_grid(LENGTH, HEIGHT, DEPTH, NX, NY, NZ)
    plot_grid(grid_data[0], grid_data[1], grid_data[2])

    # (2) 设置并执行FVM求解
    # 解释: 首先计算时间步长的稳定性上限。
    dx, dy, dz = grid_data[3], grid_data[4], grid_data[5]
    dt_stability_limit = 1 / (2 * THERMAL_DIFFUSIVITY * (1/dx**2 + 1/dy**2 + 1/dz**2))
    
    # 解释: 选择一个比上限小的时间步长以确保稳定。
    # 通常取上限的90%或更小。
    DT_SIM = 0.9 * dt_stability_limit
    
    print(f"--- Simulation Setup ---")
    print(f"Material Thermal Diffusivity (alpha): {THERMAL_DIFFUSIVITY:.3e} m^2/s")
    print(f"Stability Limit for dt: {dt_stability_limit:.4f} s")
    print(f"Chosen dt for Simulation: {DT_SIM:.4f} s")
    print(f"Total timesteps: {int(TOTAL_SIM_TIME / DT_SIM)}")
    print(f"------------------------\n")

    # 解释: 将所有参数打包到字典或元组中，方便传递给求解函数。
    grid_parameters = grid_data
    simulation_parameters = {'total_time': TOTAL_SIM_TIME, 'dt': 我}
    material_properties = {'alpha': THERMAL_DIFFUSIVITY}
    boundary_conditions = {'t_initial': T_INITIAL, 't_left_bc': T_LEFT_BOUNDARY}
    
    # 解释: 调用主求解器函数，开始计算。
    final_T, x_coords, y_coords, z_coords = solve_heat_conduction_fvm(
        grid_parameters, 
        simulation_parameters, 
        material_properties, 
        boundary_conditions
    )
    
    print("\n--- Simulation Finished ---")
    
    # (3) 可视化最终结果
    # 解释: 由于3D场难以直接可视化，我们选择几个有代表性的2D切面来展示温度分布。
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # 切面1: 砖块的中心水平面 (y = H/2)
    center_y_index = NY // 2
    im1 = axes[0].contourf(x_coords, z_coords, final_T[:, center_y_index, :].T, levels=20, cmap='hot')
    axes[0].set_title(f'Temperature at center plane (Y = {HEIGHT/2:.3f} m)')
    axes[0].set_xlabel('X-axis (m)')
    axes[0].set_ylabel('Z-axis (m)')
    fig.colorbar(im1, ax=axes[0], label='Temperature (°C)')
    
    # 切面2: 砖块的中心垂直面 (z = D/2)
    center_z_index = NZ // 2
    im2 = axes[1].contourf(x_coords, y_coords, final_T[:, :, center_z_index].T, levels=20, cmap='hot')
    axes[1].set_title(f'Temperature at center plane (Z = {DEPTH/2:.3f} m)')
    axes[1].set_xlabel('X-axis (m)')
    axes[1].set_ylabel('Y-axis (m)')
    fig.colorbar(im2, ax=axes[1], label='Temperature (°C)')

    # 切面3: 靠近热源的垂直切面 (x = 0.01m)
    hot_x_index = int(0.1 * NX) # 取x方向10%位置的切面
    im3 = axes[2].contourf(y_coords, z_coords, final_T[hot_x_index, :, :].T, levels=20, cmap='hot')
    axes[2].set_title(f'Temperature at cross-section (X = {x_coords[hot_x_index]:.3f} m)')
    axes[2].set_xlabel('Y-axis (m)')
    axes[2].set_ylabel('Z-axis (m)')
    fig.colorbar(im3, ax=axes[2], label='Temperature (°C)')

    fig.suptitle(f'Final Temperature Distribution after {TOTAL_SIM_TIME/3600:.1f} hours', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

print("\n\n---- End ----")