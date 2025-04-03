import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
# 定义时间符号变量
t = sp.symbols('t')


class Model:
    """模型基类"""
    pass


class Connector(Model):
    """接口基类"""
    pass


class Component(Model):
    """设备元件基类"""
    def __init__(self, name):
        self.name = name
        self.equations = []

    def add_equation(self, eq):
        self.equations.append(eq)


class Pin(Connector):
    """电气接口模型"""
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        # 改用普通符号变量，避免函数符号带来的复杂性
        self.v = sp.symbols(f"{parent.name}_{name}_v")
        self.i = sp.symbols(f"{parent.name}_{name}_i")


class Ground(Component):
    """接地模型"""
    def __init__(self, name):
        super().__init__(name)
        self.g = Pin(self, 'g')
        self.add_equation(sp.Eq(self.g.v, 0))  # 接地点电压固定为0


class OnePort(Component):
    """二接口模型"""
    def __init__(self, name):
        super().__init__(name)
        self.p = Pin(self, 'p')
        self.n = Pin(self, 'n')
        # 定义端口变量
        self.v = self.p.v - self.n.v
        self.i = self.p.i
        # 基础方程
        self.add_equation(sp.Eq(self.v, self.p.v - self.n.v))
        self.add_equation(sp.Eq(0, self.p.i + self.n.i))
        self.add_equation(sp.Eq(self.i, self.p.i))


class Resistor(OnePort):
    """电阻模型"""
    def __init__(self, name, R):
        super().__init__(name)
        self.R = sp.symbols(f"{name}_R")
        self.add_equation(sp.Eq(self.v, self.i * self.R))


class Capacitor(OnePort):
    """电容模型"""
    def __init__(self, name, C):
        super().__init__(name)
        self.C = sp.symbols(f"{name}_C")
        # 使用显式微分变量
        self.v_t = sp.Function('V_C')(t)  # 定义电容电压为时间函数
        self.add_equation(sp.Eq(self.v, self.v_t))  # 关联符号变量与时间函数
        self.add_equation(sp.Eq(sp.Derivative(self.v_t, t), self.i / self.C))


class ConstantVoltage(OnePort):
    """恒压源"""
    def __init__(self, name, V):
        super().__init__(name)
        self.V = sp.symbols(f"{name}_V")
        self.add_equation(sp.Eq(self.V, self.v))


def connect(pin1, pin2):
    """生成连接方程（包含电压相等和电流守恒）"""
    return [
        sp.Eq(pin1.v, pin2.v),
        sp.Eq(pin1.i + pin2.i, 0)
    ]


class RCsys(Component):
    """完整的RC电路系统模型"""
    def __init__(self):
        super().__init__('sys')
        # 实例化元件
        self.resistor = Resistor('R', R=1.0)
        self.capacitor = Capacitor('C', C=1.0)
        self.source = ConstantVoltage('V', V=1.0)
        self.ground = Ground('GND')

        # 收集所有方程
        components = [self.resistor, self.capacitor, self.source, self.ground]
        for comp in components:
            self.equations.extend(comp.equations)

        # 添加连接关系
        self.equations.extend(connect(self.source.p, self.resistor.p))
        self.equations.extend(connect(self.resistor.n, self.capacitor.p))
        self.equations.extend(connect(self.capacitor.n, self.source.n))
        self.equations.extend(connect(self.capacitor.n, self.ground.g))


def save_outs(solution, y_index, col_x_name, col_y_name):
    """仿真结果保存到Excel"""
    import pandas as pd
    try:
        # 创建数据框
        df = pd.DataFrame({
            col_x_name: solution.t,
            col_y_name: solution.y[y_index]
        })
        # 写入Excel文件
        filename = '../3020/sim_outs.xlsx'
        df.to_excel(filename, index=False, engine='openpyxl')  # 明确指定引擎
        print(f"\033[32m数据已成功保存至 {filename}\033[0m")
    except Exception as e:
        print(f"\033[31m保存失败: {str(e)}\033[0m")
        print("请确认已安装依赖库：pip install pandas openpyxl")


def plot(x, y, label, x_label, y_label):
    """图示"""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')  # 强制设置matplotlib后端

    plt.figure(figsize=(4, 3))
    plt.plot(x, y, 'b-', label=label)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    # plt.title('RC Circuit Transient Response', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # 实例化模型
    sys = RCsys()

    # 参数替换字典
    params = {
        sp.symbols('R_R'): 1.0,  # 电阻值
        sp.symbols('V_V'): 1.0,  # 电源电压
        sp.symbols('C_C'): 1.0  # 电容值
    }

    # 代入参数，分类方程
    equations = [eq.subs(params) for eq in sys.equations]
    eqs_diff = [eq for eq in equations if eq.has(sp.Derivative)]
    eqs_algebraic = [eq for eq in equations if not eq.has(sp.Derivative)]

    # 收集所有代数变量，识别微分变量
    vrs = set().union(*[eq.free_symbols for eq in equations])
    vrs_diff = {eq.lhs.args[0] for eq in eqs_diff}
    vrs_algebraic = list(vrs - vrs_diff - {t})

    # 求解代数方程组
    try:
        sol_algebraic = sp.solve(eqs_algebraic, vrs_algebraic, dict=True)[0]
    except IndexError:  # 添加存在性检查
        print("代数方程求解失败，当前方程组：")
        for i, eq in enumerate(eqs_algebraic):
            print(f"Eq{i}: {eq}")
        raise RuntimeError("无法求解代数方程组，请检查设备元件连接关系")

    # 化简微分方程
    eqs_ode = [eq.subs(sol_algebraic).doit() for eq in eqs_diff]

    # 提取ODE表达式
    v_c = sp.Function('V_C')(t)
    ode_rhs = eqs_ode[0].rhs

    # 创建数值计算函数
    dydt = sp.lambdify((t, v_c), ode_rhs, modules='numpy')

    # 求解ODE
    t_span = (0, 10.0)  # 仿真时间段 0-10秒
    u0 = [0.0]          # 微分变量初始值
    sol = solve_ivp(dydt, t_span, u0, t_eval=np.linspace(*t_span, 100))

    # 保存仿真结果数据
    save_outs(sol, 0, "time\\s", "Vc\\V")

    # 可视化结果
    plot(sol.t, sol.y[0], label="Capacitor Voltage Vc",
         x_label="time / s", y_label="Vc / V")

    print("---- End ----")
