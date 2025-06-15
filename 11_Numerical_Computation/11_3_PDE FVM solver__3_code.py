# 导入基本计算和绘图库
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

# --- 物理和模拟参数设置 ---
# 解释: 定义一个常量用于摄氏度和开尔文之间的转换。
KELVIN_OFFSET = 273.15

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

# 解释: 将初始温度和边界温度从摄氏度转换为开尔文。
# 初始温度和边界温度 (单位: 开尔文 K)
T_INITIAL = 20.0 + KELVIN_OFFSET
T_LEFT_BOUNDARY = 200.0 + KELVIN_OFFSET

# 空间离散化设置
NX = 20  # x方向的网格节点数
NY = 10  # y方向的网格节点数
NZ = 14  # z方向的网格节点数

# 时间离散化设置
TOTAL_SIM_TIME = 3600 * 5  # 总模拟时间上限 (5小时)

# --- 任务 (1): 空间离散化与网格可视化 (与之前相同) ---

def create_grid(length, height, depth, nx, ny, nz):
    """
    创建三维笛卡尔网格的节点坐标和步长。
    """
    x = np.linspace(0, length, nx)
    y = np.linspace(0, height, ny)
    z = np.linspace(0, depth, nz)
    dx = length / (nx - 1)
    dy = height / (ny - 1)
    dz = depth / (nz - 1)
    return x, y, z, dx, dy, dz

def plot_grid(x, y, z):
    """
    将离散后的三维网格节点进行可视化。
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = np.meshgrid(x, y, z)
    ax.scatter(X, Y, Z, c='b', s=5, label='Grid Nodes')
    ax.set_xlabel('X-axis (Length = 10cm)')
    ax.set_ylabel('Y-axis (Height = 5cm)')
    ax.set_zlabel('Z-axis (Depth = 7cm)')
    ax.set_title('3D Discretized Grid')
    ax.scatter(0, 0, 0, c='r', s=50, label='Origin (0,0,0)')
    ax.legend()
    plt.show()

# --- 任务 (2): 有限体积法传热过程实现 (改进版) ---

def solve_heat_conduction_fvm(grid_params, sim_params, material_props, conditions, 
                              stop_time=None, steady_state_tol=None):
    """
    使用有限体积法和显式欧拉格式求解三维瞬态热传导问题。
    此版本增加了对停止时间、稳态检测的支持。
    """
    # --- 解包所有传入的参数 ---
    x, y, z, dx, dy, dz = grid_params
    nx, ny, nz = len(x), len(y), len(z)

    # 解释: 如果指定了stop_time，它将覆盖总模拟时间。
    # 这使得我们可以精确地运行到某个特定时刻。
    total_time = stop_time if stop_time is not None else sim_params['total_time']
    dt = sim_params['dt']

    alpha = material_props['alpha']
    t_initial = conditions['t_initial']
    t_left_bc = conditions['t_left_bc']
    
    # --- 初始化温度场 ---
    T = np.full((nx, ny, nz), t_initial)

    # --- 核心时间迭代循环 ---
    # 解释: 创建时间步数组。使用 total_time + dt 确保循环能达到并包含 total_time。
    time_steps = np.arange(0, total_time + dt, dt)
    final_t = 0
    
    for t_step, t in enumerate(time_steps):
        T_old = T.copy()

        # --- 内部节点计算 (与之前相同) ---
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                for k in range(1, nz - 1):
                    laplacian_x = (T_old[i + 1, j, k] - 2 * T_old[i, j, k] + T_old[i - 1, j, k]) / (dx**2)
                    laplacian_y = (T_old[i, j + 1, k] - 2 * T_old[i, j, k] + T_old[i, j - 1, k]) / (dy**2)
                    laplacian_z = (T_old[i, j, k + 1] - 2 * T_old[i, j, k] + T_old[i, j, k - 1]) / (dz**2)
                    T[i, j, k] = T_old[i, j, k] + alpha * dt * (laplacian_x + laplacian_y + laplacian_z)
        
        # --- 应用边界条件 (与之前相同) ---
        T[0, :, :] = t_left_bc
        T[nx - 1, :, :] = T[nx - 2, :, :]
        T[:, ny - 1, :] = T[:, ny - 2, :]
        T[:, 0, :]      = T[:, 1, :]
        T[:, :, nz - 1] = T[:, :, nz - 2]
        T[:, :, 0]      = T[:, :, 1]
        
        # 解释: 持续更新最后计算的时间。
        final_t = t

        # 解释: 如果传入了稳态容差，则进行稳态检测。
        if steady_state_tol is not None:
            # 解释: 计算整个砖块温度场在一步长内的最大绝对变化值。
            max_temp_change = np.max(np.abs(T - T_old))
            # 解释: 如果最大温度变化小于我们设定的容差，就认为系统达到稳态。
            if max_temp_change < steady_state_tol:
                # 解释: 打印稳态信息，并立即返回结果，终止模拟。
                print(f"\n* 稳态已达到，时刻 t = {t:.2f} s. (最大温度变化: {max_temp_change:.3e})")
                return T, t

        # 解释: 减少打印频率，避免刷屏。
        if (t_step + 1) % 5000 == 0:
            print(f"  ...模拟进行中... t = {t:.2f} s / {total_time:.2f} s")
            
    # 解释: 如果循环正常结束（未提前达到稳态），返回最终结果。
    return T, final_t

# --- 主执行程序 ---
if __name__ == '__main__':
    # (1) 创建并可视化网格
    grid_data = create_grid(LENGTH, HEIGHT, DEPTH, NX, NY, NZ)
    x_coords, y_coords, z_coords, dx, dy, dz = grid_data
    plot_grid(x_coords, y_coords, z_coords)

    # (2) 设置并计算时间步长
    dt_stability_limit = 1 / (2 * THERMAL_DIFFUSIVITY * (1/dx**2 + 1/dy**2 + 1/dz**2))
    DT_SIM = 0.9 * dt_stability_limit
    
    print(f"--- 模拟设置 ---")
    print(f"材料热扩散率 (alpha): {THERMAL_DIFFUSIVITY:.3e} m^2/s")
    print(f"稳定性要求 dt < {dt_stability_limit:.4f} s")
    print(f"选用模拟步长 dt = {DT_SIM:.4f} s")
    print(f"--------------------\n")

    # 解释: 将参数打包到字典中，方便传递。
    simulation_parameters = {'total_time': TOTAL_SIM_TIME, 'dt': DT_SIM}
    material_properties = {'alpha': THERMAL_DIFFUSIVITY}
    boundary_conditions = {'t_initial': T_INITIAL, 't_left_bc': T_LEFT_BOUNDARY}

    # --- 任务 (3a): 寻找稳态时间 t_steady ---
    print("--- 步骤1: 寻找稳态时间 (t_steady) ---")
    # 解释: 运行一次模拟以确定达到稳态所需的时间。
    # 我们传入一个很小的容差值(1e-5 K/s)来判断稳态。
    # 为了防止模拟时间过长，我们仍然使用 TOTAL_SIM_TIME 作为上限。
    _, t_steady = solve_heat_conduction_fvm(
        grid_data,
        simulation_parameters,
        material_properties,
        boundary_conditions,
        steady_state_tol=1e-5 # K/s, 稳态容差
    )
    
    if t_steady >= TOTAL_SIM_TIME:
        print(f"\n警告: 在 {TOTAL_SIM_TIME} s 内未达到稳态，将使用此时间作为 t_steady 进行后续计算。")

    # --- 任务 (3b): 获取三个特定时刻的温度场快照 ---
    # 解释: 根据找到的稳态时间t_steady，计算三个快照的时刻，并取整。
    snapshot_times = [int(t_steady / 4), int(t_steady / 2), int(3 * t_steady / 4)]
    snapshots = []
    print(f"\n--- 步骤2: 生成快照，时刻 t = {snapshot_times[0]}s, {snapshot_times[1]}s, {snapshot_times[2]}s ---")

    for t_snap in snapshot_times:
        print(f"正在运行模拟直到 t = {t_snap:.2f} s...")
        # 解释: 为获取每个快照，我们重新运行模拟直到目标时间点。
        # 这是通过向求解器传入 stop_time 参数实现的。
        T_snap, _ = solve_heat_conduction_fvm(
            grid_data,
            simulation_parameters,
            material_properties,
            boundary_conditions,
            stop_time=t_snap
        )
        snapshots.append(T_snap)

    # --- 任务 (3c): 3D 可视化快照 ---
    print("\n--- 步骤3: 可视化三维温度云图 ---")
    
    # 解释: np.meshgrid 从一维坐标数组创建三维坐标网格。
    # X, Y, Z 是三维数组，存储了每个节点的(x,y,z)坐标，用于绘图。
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # 解释: 创建一个大的图形，包含三个3D子图。
    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(f'3D Temperature Distribution (Steady State at t_steady={t_steady:.2f}s)', fontsize=16)

    # 解释: 确定所有快照中的最低和最高温度，以统一颜色条的范围。
    vmin = min(T.min() for T in snapshots)
    vmax = max(T.max() for T in snapshots)

    # 解释: 创建一个列表来存储所有子图的句柄，用于颜色条。
    all_axes = []

    for i, t_snap in enumerate(snapshot_times):
        # 解释: 为每个快照创建一个3D子图。
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        all_axes.append(ax) # 将句柄添加到列表中
        T_snap = snapshots[i]

        # 解释: 使用3D散点图(scatter)来可视化温度场。
        # X.flatten()等将三维坐标网格展平为一维数组。
        # c=T_snap.flatten() 将每个点的温度值映射为颜色。
        # cmap='coolwarm' 是一个适合温度场的可视化颜色映射表。
        # vmin和vmax确保三个图的颜色刻度一致，便于比较。
        # 我们将温度单位从开尔文转回摄氏度进行显示。
        p = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=T_snap.flatten() - KELVIN_OFFSET, cmap='coolwarm', vmin=vmin - KELVIN_OFFSET, vmax=vmax - KELVIN_OFFSET, s=15, alpha=0.7)

        # 解释: 设置子图标题和坐标轴标签。
        ax.set_title(f'Snapshot at t = {t_snap} s')
        ax.set_xlabel('X-axis (m)')
        ax.set_ylabel('Y-axis (m)')
        ax.set_zlabel('Z-axis (m)')
        
        # 解释: 设置一个合适的视角以便观察。elev是仰角，azim是方位角。
        ax.view_init(elev=25., azim=-135)

    # 解释: 在图形右侧添加一个总的颜色条。
    # 我们将colorbar附加到所有子图(all_axes)上。
    # p 是最后一个scatter图的返回对象，可用于颜色映射。
    cbar = fig.colorbar(p, ax=all_axes, shrink=0.7, aspect=20)
    cbar.set_label('Temperature (°C)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()

print("\n\n---- End ----")
