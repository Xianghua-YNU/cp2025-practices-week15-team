"""
学生模板：双摆模拟
课程：计算物理
说明：请实现标记为 TODO 的函数。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

# 常量
G_CONST = 9.81  # 重力加速度 (m/s²)
L_CONST = 0.4   # 单摆臂长度 (m)
M_CONST = 1.0   # 单摆球质量 (kg)

def derivatives(y, t, L1, L2, m1, m2, g_param):
    """
    返回双摆状态向量y的时间导数。

    参数:
        y (list或np.array): 当前状态向量 [theta1, omega1, theta2, omega2]
        t (float): 当前时间（在自治方程中未直接使用，但odeint需要）
        L1 (float): 第一摆臂长度
        L2 (float): 第二摆臂长度
        m1 (float): 第一摆球质量
        m2 (float): 第二摆球质量
        g (float): 重力加速度

    返回:
        list: 时间导数 [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]
    
    运动方程（L1=L2=L, m1=m2=m的简化形式）:
    dtheta1_dt = omega1
    dtheta2_dt = omega2
    domega1_dt = (-omega1**2*np.sin(2*theta1-2*theta2) - 2*omega2**2*np.sin(theta1-theta2) - 
                  (g/L) * (np.sin(theta1-2*theta2) + 3*np.sin(theta1))) / (3 - np.cos(2*theta1-2*theta2))
    domega2_dt = (4*omega1**2*np.sin(theta1-theta2) + omega2**2*np.sin(2*theta1-2*theta2) + 
                  2*(g/L) * (np.sin(2*theta1-theta2) - np.sin(theta2))) / (3 - np.cos(2*theta1-2*theta2))
    """
    theta1, omega1, theta2, omega2 = y

    # 本问题中假设L1=L2=L且m1=m2=M，直接使用问题描述的方程
    # 若m1,m2,L1,L2不同，则需要更通用的方程
    
    dtheta1_dt = omega1
    dtheta2_dt = omega2

    # domega1_dt的分子和分母
    num1 = -omega1**2 * np.sin(2*theta1 - 2*theta2) \
           - 2 * omega2**2 * np.sin(theta1 - theta2) \
           - (g_param/L1) * (np.sin(theta1 - 2*theta2) + 3*np.sin(theta1))
    den1 = 3 - np.cos(2*theta1 - 2*theta2)  # 假设m1=m2, L1=L2
    
    domega1_dt = num1 / den1

    # domega2_dt的分子和分母
    num2 = 4 * omega1**2 * np.sin(theta1 - theta2) \
           + omega2**2 * np.sin(2*theta1 - 2*theta2) \
           + 2 * (g_param/L1) * (np.sin(2*theta1 - theta2) - np.sin(theta2))
    den2 = 3 - np.cos(2*theta1 - 2*theta2)  # 假设m1=m2, L1=L2
    
    domega2_dt = num2 / den2
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

def solve_double_pendulum(initial_conditions, t_span, t_points, L_param=L_CONST, g_param=G_CONST):
    """
    求解双摆常微分方程组。

    参数:
        initial_conditions (dict): 初始条件字典 {'theta1': 值, 'omega1': 值, 'theta2': 值, 'omega2': 值}（弧度制）
        t_span (tuple): 模拟时间范围 (t_start, t_end)
        t_points (int): 生成的时间点数
        L_param (float): 摆臂长度
        g_param (float): 重力加速度

    返回:
        tuple: (t_arr, sol_arr)
               t_arr: 时间点的一维数组
               sol_arr: 状态矩阵 [theta1, omega1, theta2, omega2]
    """
    y0 = [initial_conditions['theta1'], initial_conditions['omega1'], 
          initial_conditions['theta2'], initial_conditions['omega2']]
    t_arr = np.linspace(t_span[0], t_span[1], t_points)
    
    # 假设L1=L2=L_param且m1=m2=M_CONST
    sol_arr = odeint(derivatives, y0, t_arr, args=(L_param, L_param, M_CONST, M_CONST, g_param), rtol=1e-9, atol=1e-9)
    return t_arr, sol_arr

def calculate_energy(sol_arr, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST):
    """
    计算双摆系统总能量。

    参数:
        sol_arr (np.array): odeint的解矩阵
        L_param (float): 摆臂长度
        m_param (float): 摆球质量
        g_param (float): 重力加速度

    返回:
        np.array: 各时间点总能量的一维数组
    """
    theta1 = sol_arr[:, 0]
    omega1 = sol_arr[:, 1]
    theta2 = sol_arr[:, 2]
    omega2 = sol_arr[:, 3]

    # 势能 (V)
    V = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))

    # 动能 (T)
    T = m_param * L_param**2 * (omega1**2 + 0.5 * omega2**2 + omega1 * omega2 * np.cos(theta1 - theta2))
    
    return T + V

def animate_double_pendulum(t_arr, sol_arr, L_param=L_CONST, skip_frames=int):
    """
    创建双摆动画。

    参数:
        t_arr (np.array): 时间数组
        sol_arr (np.array): odeint的解矩阵
        L_param (float): 摆臂长度
        skip_frames (int): 动画帧间隔步数

    返回:
        matplotlib.animation.FuncAnimation: 动画对象
    """
    theta1_all = sol_arr[:, 0]
    theta2_all = sol_arr[:, 2]

    # 动画帧选择
    theta1_anim = theta1_all[::skip_frames]
    theta2_anim = theta2_all[::skip_frames]
    t_anim = t_arr[::skip_frames]

    # 笛卡尔坐标转换
    x1 = L_param * np.sin(theta1_anim)
    y1 = -L_param * np.cos(theta1_anim)
    x2 = x1 + L_param * np.sin(theta2_anim)
    y2 = y1 - L_param * np.cos(theta2_anim)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2*L_param - 0.1, 2*L_param + 0.1), ylim=(-2*L_param - 0.1, 0.1))
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Double Pendulum')

    line, = ax.plot([], [], 'o-', lw=2, markersize=8, color='blue')  # 摆臂和摆球
    line2, = ax.plot([],[],'k')
    time_template = 'Time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        """动画初始化函数"""
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        """动画帧更新函数"""
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        sx = [x1[:i],x2[:i]]
        sy = [y1[:i],y2[:i]]
        line.set_data(thisx, thisy)
        line2.set_data(sx,sy)
        time_text.set_text(time_template % t_anim[i])
        return line, time_text, line2

    ani = animation.FuncAnimation(fig, animate, frames=len(t_anim),
                                  interval=25, blit=True, init_func=init)
    return ani

if __name__ == "__main__":
    # 初始条件（弧度制）
    initial_conditions_rad = {
        'theta1': np.pi/2,  # 90度
        'omega1': 0.0,
        'theta2': np.pi/2,  # 90度
        'omega2': 0.0
    }
    t_start = 0
    t_end = 100
    t_points_sim = 2000  # 时间点数（提高精度）

    # 1. 求解常微分方程组
    print(f"求解常微分方程组 (t = {t_start}s 到 {t_end}s)...")
    t_solution, sol_solution = solve_double_pendulum(initial_conditions_rad, (t_start, t_end), t_points_sim)
    print("求解完成")

    # 2. 计算能量
    print("计算能量...")
    energy_solution = calculate_energy(sol_solution)
    print("能量计算完成")

    # 3. 绘制能量-时间图
    plt.figure(figsize=(10, 5))
    plt.plot(t_solution, energy_solution, label='总能量')
    plt.xlabel('时间 (s)')
    plt.ylabel('能量 (焦耳)')
    plt.title('Total Energy vs. Time')
    plt.grid(True)
    plt.legend()
    
    # 能量守恒分析
    initial_energy = energy_solution[0]
    final_energy = energy_solution[-1]
    energy_variation = np.max(energy_solution) - np.min(energy_solution)
    print(f"初始能量: {initial_energy:.7f} J")
    print(f"最终能量: {final_energy:.7f} J")
    print(f"最大能量波动: {energy_variation:.7e} J")
    
    if energy_variation < 1e-5:
        print("达到能量守恒目标 (< 1e-5 J)")
    else:
        print(f"未达到能量守恒目标 (< 1e-5 J)。波动: {energy_variation:.2e} J")
    
    plt.ylim(initial_energy - max(5*energy_variation, 1e-5), 
             initial_energy + max(5*energy_variation, 1e-5))
    plt.show()

    run_animation = True
    if run_animation:
        print("创建动画中...")
        anim_object = animate_double_pendulum(t_solution, sol_solution, skip_frames=1)
        plt.show()
        print("动画展示完成")
    else:
        print("跳过动画")

    print("双摆模拟完成")
