#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：二阶常微分方程边值问题数值解法 - 学生代码模板

本项目要求实现两种数值方法求解边值问题：
1. 有限差分法 (Finite Difference Method)
2. scipy.integrate.solve_bvp 方法

问题设定：
y''(x) + sin(#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：二阶常微分方程边值问题数值解法 - 学生代码模板

本项目要求实现两种数值方法求解边值问题：
1. 有限差分法 (Finite Difference Method)
2. scipy.integrate.solve_bvp 方法

问题设定：
y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
边界条件：y(0) = 0, y(5) = 3

学生需要完成所有标记为 TODO 的函数实现。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.linalg import solve


# ============================================================================
# 方法1：有限差分法 (Finite Difference Method)
# ============================================================================

def solve_bvp_finite_difference(n):
    """
    使用有限差分法求解二阶常微分方程边值问题。

    方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
    边界条件：y(0) = 0, y(5) = 3

    Args:
        n (int): 内部网格点数量

    Returns:
        tuple: (x_grid, y_solution)
            x_grid (np.ndarray): 包含边界点的完整网格
            y_solution (np.ndarray): 对应的解值

    TODO: 实现有限差分法
    Hints:
    1. 创建网格点 x_i = i*h, i=0,1,...,n+1, 其中 h = 5/(n+1)
    2. 对于内部点 i=1,2,...,n，使用中心差分近似：
       y''_i ≈ (y_{i+1} - 2*y_i + y_{i-1}) / h^2
       y'_i ≈ (y_{i+1} - y_{i-1}) / (2*h)
    3. 构建线性系统 A*y = b，其中 y = [y_1, y_2, ..., y_n]
    4. 边界条件：y_0 = 0, y_{n+1} = 3
    5. 对于每个内部点，重新整理方程得到系数
    6. 处理边界条件对右端向量的影响
    """
    # TODO: 在此实现有限差分法 (预计30-40行代码)
    # [STUDENT_CODE_HERE]
    #计算步长，创建表格
    h = 5.0 / (n + 1)
    x_grid = np.linspace(0,5,n + 2)
    #初始化系统矩阵和右端向量
    A = np.zero((n,n))
    b = np.zeros(n)
    #填充系数矩阵和右端向量
    for i in range(n):
        x_i = x_grid[i+1]#内部索引从1到n
        #中心查分系数
        coef_left = 1/(h**2) - np.sin(x_i)/(2*h)
        coef_center = -2/(h**2) + np.exp(x_i)
        coef_right = 1/(h**2) + np.sin(x_i)/(2*h)
        #填充矩阵
        if i > 0:
            A[i,i-1] = coef_left
        A[i,i] = coef_center
        if i < n-1:
            A[i,i+1] = coef_right
        #填充右端向量
        b[i] = x_i**2
        #处理边界条件
        if i == 0:
            b[i] -= coef_left*0
        if i == n-1:
            b[i] -= coef_right*3
    #应用边界条件
    #左边界y(0)=0
    #b[0] -= coef_left * 0
    #右边界y(5)=3
    #b[-1] -= coef_right * 3
    #求解线性系统
    y_interior = solve(A,b)
    #添加边界值
    y_solution = np.zeros(n+2)
    y_solution[0] = 0  #y(0)
    y_solution[1:-1] = y_interior
    y_solution[-1] = 3  #y(5)
    return x_grid,y_solution






# ============================================================================
# 方法2：scipy.integrate.solve_bvp 方法
# ============================================================================

def ode_system_for_solve_bvp(x, y):
    """
    为 scipy.integrate.solve_bvp 定义ODE系统。

    将二阶ODE转换为一阶系统：
    y[0] = y(x)
    y[1] = y'(x)

    系统方程：
    dy[0]/dx = y[1]
    dy[1]/dx = -sin(x) * y[1] - exp(x) * y[0] + x^2

    Args:
        x (float or array): 自变量
        y (array): 状态变量 [y, y']

    Returns:
        array: 导数 [dy/dx, dy'/dx]

    TODO: 实现ODE系统的右端项
    Hints:
    1. 提取 y[0] 和 y[1] 分别表示 y(x) 和 y'(x)
    2. 根据一阶系统方程计算导数
    3. 使用 np.vstack 组合返回结果
    """
    # TODO: 在此实现一阶ODE系统 (预计5-8行代码)
    # [STUDENT_CODE_HERE]
    y0 = y[0]  #y(x)
    y1 = y[1]  #y'(x)
    dy0_dx = y1
    dy1_dx = -np.sin(x) * y1 - np.exp(x) * y0 + x**2
    return np.vstack([dy0_dx,dy1_dx])





def boundary_conditions_for_solve_bvp(ya, yb):
    """
    为 scipy.integrate.solve_bvp 定义边界条件。

    Args:
        ya (array): 左边界处的状态 [y(0), y'(0)]
        yb (array): 右边界处的状态 [y(5), y'(5)]

    Returns:
        array: 边界条件残差 [y(0) - 0, y(5) - 3]

    TODO: 实现边界条件
    Hints:
    1. ya[0] 是左边界的 y 值，应该等于 0
    2. yb[0] 是右边界的 y 值，应该等于 3
    3. 返回残差数组
    """
    # TODO: 在此实现边界条件 (预计1-2行代码)
    # [STUDENT_CODE_HERE]

    return np.array([ya[0],yb[0] - 3])


def solve_bvp_scipy(n_initial_points=11):
    """
    使用 scipy.integrate.solve_bvp 求解BVP。

    Args:
        n_initial_points (int): 初始网格点数

    Returns:
        tuple: (x_solution, y_solution)
            x_solution (np.ndarray): 解的 x 坐标数组
            y_solution (np.ndarray): 解的 y 坐标数组

    TODO: 实现 solve_bvp 方法
    Hints:
    1. 创建初始网格 x_initial
    2. 创建初始猜测 y_initial (2×n 数组)
    3. 调用 solve_bvp 函数
    4. 检查求解是否成功并提取解
    """
    # TODO: 在此实现 solve_bvp 方法 (预计10-15行代码)
    # [STUDENT_CODE_HERE]

    #创建初始网格
    x_initial = np.linspace(0,5,n_initial_points)
    #创建初始猜测(y和dydx)
    #线性猜测：从(0,0)到(5,3),倒数为常数(3-0)/5=0.6
    y_initial = np.zeros((2,n_initial_points))
    y_initial[0] = np.linspace(0,3,n_initial_points)  #y(x)的初始值猜想
    y_initial[1] = 0.6 * np.ones(n_initial_points)  #y'(x)的初始值猜想
    #求解边值问题
    sol = solve_bvp(ode_system_for_solve_bvp, boundary_conditions_for_solve_bvp,x_initial,y_initial,max_nodes=1000)
    if sol.success:
        x_solution = sol.x
        y_solution = sol.y[0]  # 只取 y(x)，不要 y'(x)
        return x_solution, y_solution
    else:
        raise RuntimeError("solve_bvp failed to converge")


# ============================================================================
# 主程序：测试和比较两种方法
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("二阶常微分方程边值问题数值解法比较")
    print("方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2")
    print("边界条件：y(0) = 0, y(5) = 3")
    print("=" * 60)

    # 设置参数
    n_points = 50  # 有限差分法的内部网格点数

    try:
        # 方法1：有限差分法
        print("\n1. 有限差分法求解...")
        x_fd, y_fd = solve_bvp_finite_difference(n_points)
        print(f"   网格点数：{len(x_fd)}")
        print(f"   y(0) = {y_fd[0]:.6f}, y(5) = {y_fd[-1]:.6f}")

    except NotImplementedError:
        print("   有限差分法尚未实现")
        x_fd, y_fd = None, None

    try:
        # 方法2：scipy.integrate.solve_bvp
        print("\n2. scipy.integrate.solve_bvp 求解...")
        x_scipy, y_scipy = solve_bvp_scipy()
        print(f"   网格点数：{len(x_scipy)}")
        print(f"   y(0) = {y_scipy[0]:.6f}, y(5) = {y_scipy[-1]:.6f}")

    except NotImplementedError:
        print("   solve_bvp 方法尚未实现")
        x_scipy, y_scipy = None, None

    # 绘图比较
    plt.figure(figsize=(12, 8))

    # 子图1：解的比较
    plt.subplot(2, 1, 1)
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-o', markersize=3, label='Finite Difference Method', linewidth=2)
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', label='scipy.integrate.solve_bvp', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Comparison of Numerical Solutions for BVP')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：解的差异（如果两种方法都实现了）
    plt.subplot(2, 1, 2)
    if (x_fd is not None and y_fd is not None and
            x_scipy is not None and y_scipy is not None):

        # 将 scipy 解插值到有限差分网格上进行比较
        y_scipy_interp = np.interp(x_fd, x_scipy, y_scipy)
        difference = np.abs(y_fd - y_scipy_interp)

        plt.semilogy(x_fd, difference, 'g-', linewidth=2, label='|Finite Diff - solve_bvp|')
        plt.xlabel('x')
        plt.ylabel('Absolute Difference')
        plt.title('Absolute Difference Between Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 数值比较
        max_diff = np.max(difference)
        mean_diff = np.mean(difference)
        print(f"\n数值比较：")
        print(f"   最大绝对误差：{max_diff:.2e}")
        print(f"   平均绝对误差：{mean_diff:.2e}")
    else:
        plt.text(0.5, 0.5, 'Need both methods implemented\nfor comparison',
                 ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Difference Plot (Not Available)')

    plt.tight_layout()
    plt.show()

    print("\n=" * 60)
    print("实验完成！")
    print("请在实验报告中分析两种方法的精度、效率和适用性。")
    print("=" * 60)x) * y'(x) + exp(x) * y(x) = x^2
边界条件：y(0) = 0, y(5) = 3

学生需要完成所有标记为 TODO 的函数实现。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.linalg import solve


# ============================================================================
# 方法1：有限差分法 (Finite Difference Method)
# ============================================================================

def solve_bvp_finite_difference(n):
    """
    使用有限差分法求解二阶常微分方程边值问题。
    
    方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
    边界条件：y(0) = 0, y(5) = 3
    
    Args:
        n (int): 内部网格点数量
    
    Returns:
        tuple: (x_grid, y_solution)
            x_grid (np.ndarray): 包含边界点的完整网格
            y_solution (np.ndarray): 对应的解值
    
    TODO: 实现有限差分法
    Hints:
    1. 创建网格点 x_i = i*h, i=0,1,...,n+1, 其中 h = 5/(n+1)
    2. 对于内部点 i=1,2,...,n，使用中心差分近似：
       y''_i ≈ (y_{i+1} - 2*y_i + y_{i-1}) / h^2
       y'_i ≈ (y_{i+1} - y_{i-1}) / (2*h)
    3. 构建线性系统 A*y = b，其中 y = [y_1, y_2, ..., y_n]
    4. 边界条件：y_0 = 0, y_{n+1} = 3
    5. 对于每个内部点，重新整理方程得到系数
    6. 处理边界条件对右端向量的影响
    """
    # TODO: 在此实现有限差分法 (预计30-40行代码)
    # [STUDENT_CODE_HERE]
    
    raise NotImplementedError("请在此处实现有限差分法")


# ============================================================================
# 方法2：scipy.integrate.solve_bvp 方法
# ============================================================================

def ode_system_for_solve_bvp(x, y):
    """
    为 scipy.integrate.solve_bvp 定义ODE系统。
    
    将二阶ODE转换为一阶系统：
    y[0] = y(x)
    y[1] = y'(x)
    
    系统方程：
    dy[0]/dx = y[1]
    dy[1]/dx = -sin(x) * y[1] - exp(x) * y[0] + x^2
    
    Args:
        x (float or array): 自变量
        y (array): 状态变量 [y, y']
    
    Returns:
        array: 导数 [dy/dx, dy'/dx]
    
    TODO: 实现ODE系统的右端项
    Hints:
    1. 提取 y[0] 和 y[1] 分别表示 y(x) 和 y'(x)
    2. 根据一阶系统方程计算导数
    3. 使用 np.vstack 组合返回结果
    """
    # TODO: 在此实现一阶ODE系统 (预计5-8行代码)
    # [STUDENT_CODE_HERE]
    
    raise NotImplementedError("请在此处实现ODE系统")


def boundary_conditions_for_solve_bvp(ya, yb):
    """
    为 scipy.integrate.solve_bvp 定义边界条件。
    
    Args:
        ya (array): 左边界处的状态 [y(0), y'(0)]
        yb (array): 右边界处的状态 [y(5), y'(5)]
    
    Returns:
        array: 边界条件残差 [y(0) - 0, y(5) - 3]
    
    TODO: 实现边界条件
    Hints:
    1. ya[0] 是左边界的 y 值，应该等于 0
    2. yb[0] 是右边界的 y 值，应该等于 3
    3. 返回残差数组
    """
    # TODO: 在此实现边界条件 (预计1-2行代码)
    # [STUDENT_CODE_HERE]
    
    raise NotImplementedError("请在此处实现边界条件")


def solve_bvp_scipy(n_initial_points=11):
    """
    使用 scipy.integrate.solve_bvp 求解BVP。
    
    Args:
        n_initial_points (int): 初始网格点数
    
    Returns:
        tuple: (x_solution, y_solution)
            x_solution (np.ndarray): 解的 x 坐标数组
            y_solution (np.ndarray): 解的 y 坐标数组
    
    TODO: 实现 solve_bvp 方法
    Hints:
    1. 创建初始网格 x_initial
    2. 创建初始猜测 y_initial (2×n 数组)
    3. 调用 solve_bvp 函数
    4. 检查求解是否成功并提取解
    """
    # TODO: 在此实现 solve_bvp 方法 (预计10-15行代码)
    # [STUDENT_CODE_HERE]
    
    raise NotImplementedError("请在此处实现 solve_bvp 方法")


# ============================================================================
# 主程序：测试和比较两种方法
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("二阶常微分方程边值问题数值解法比较")
    print("方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2")
    print("边界条件：y(0) = 0, y(5) = 3")
    print("=" * 60)
    
    # 设置参数
    n_points = 50  # 有限差分法的内部网格点数
    
    try:
        # 方法1：有限差分法
        print("\n1. 有限差分法求解...")
        x_fd, y_fd = solve_bvp_finite_difference(n_points)
        print(f"   网格点数：{len(x_fd)}")
        print(f"   y(0) = {y_fd[0]:.6f}, y(5) = {y_fd[-1]:.6f}")
        
    except NotImplementedError:
        print("   有限差分法尚未实现")
        x_fd, y_fd = None, None
    
    try:
        # 方法2：scipy.integrate.solve_bvp
        print("\n2. scipy.integrate.solve_bvp 求解...")
        x_scipy, y_scipy = solve_bvp_scipy()
        print(f"   网格点数：{len(x_scipy)}")
        print(f"   y(0) = {y_scipy[0]:.6f}, y(5) = {y_scipy[-1]:.6f}")
        
    except NotImplementedError:
        print("   solve_bvp 方法尚未实现")
        x_scipy, y_scipy = None, None
    
    # 绘图比较
    plt.figure(figsize=(12, 8))
    
    # 子图1：解的比较
    plt.subplot(2, 1, 1)
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-o', markersize=3, label='Finite Difference Method', linewidth=2)
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', label='scipy.integrate.solve_bvp', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Comparison of Numerical Solutions for BVP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：解的差异（如果两种方法都实现了）
    plt.subplot(2, 1, 2)
    if (x_fd is not None and y_fd is not None and 
        x_scipy is not None and y_scipy is not None):
        
        # 将 scipy 解插值到有限差分网格上进行比较
        y_scipy_interp = np.interp(x_fd, x_scipy, y_scipy)
        difference = np.abs(y_fd - y_scipy_interp)
        
        plt.semilogy(x_fd, difference, 'g-', linewidth=2, label='|Finite Diff - solve_bvp|')
        plt.xlabel('x')
        plt.ylabel('Absolute Difference')
        plt.title('Absolute Difference Between Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 数值比较
        max_diff = np.max(difference)
        mean_diff = np.mean(difference)
        print(f"\n数值比较：")
        print(f"   最大绝对误差：{max_diff:.2e}")
        print(f"   平均绝对误差：{mean_diff:.2e}")
    else:
        plt.text(0.5, 0.5, 'Need both methods implemented\nfor comparison', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Difference Plot (Not Available)')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=" * 60)
    print("实验完成！")
    print("请在实验报告中分析两种方法的精度、效率和适用性。")
    print("=" * 60)
