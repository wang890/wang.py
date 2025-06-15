长10cm高5cm深7cm的一个砖头，左侧面的初始温度200摄氏度，其他所有侧面的初始温度都是20摄氏度，所有侧面都为绝热（不对外放热），请你自己设置传热系数，以此为例采用有限体积法实现砖头的传热过程及代码：(1) 给我python代码，只能使用math、numpy这样的基本计算库 不能使用现有现成的库函数，实现砖头几何体的空间离散和时间离散，并且把离散网格图示出来 标注坐标方向和原点；（2）编写python代码，只能使用math、numpy等基本计算库 不要使用现成的库函数，在问题(1)离散网格和离散时间的基础上，采用有限体积法实现砖头的传热过程，注释每一行代码，（3）并介绍(1)和(2)的计算过程原理。——回复以Markdown源码的形式给我（要求latex数学公式以$...$或$$....$$为边界）

笔者注：上述第次提问得到第1份代码文件(结果为二维图)，下面两次提问得到第2份代码文件（结果三维图）

---

请你修改上述代码，直接修改这个文件的代码，实现下述改善要求：(1) 可视化最终结果采用xyz三维显示（以较好的视角进行展示），你还是画的二维图吗？不要二维图 要xyz三维云图，只画 t2/4、t2/2、3*t2/4 三个时刻的xyz三维结果云图（这三个时刻 你可以取整），t2为达到传热稳态的时间点（t2这个符号你可自行取名）。(2) 解决错误 xception has occurred: NameError    name 'axes' is not defined   File "E:\wang.py\11_Numerical_Computation\11_3_PDE FVM solver__3_code.py", line 230, in <module>
    cbar = fig.colorbar(p, ax=axes.ravel().tolist(), shrink=0.7, aspect=20)
NameError: name 'axes' is not defined

---

请你修改@11_3_PDE FVM solver__3_code.py 的代码，直接修改这个文件的代码，实现下述改善要求：结果三维云图 使用红色到蓝色的渐变色，高温使用红色、低温使用蓝色

---

这是一个非常经典的计算传热学问题，下面我将为您详细拆解并实现整个过程。在编写代码之前，我们首先需要理解其背后的物理和数学原理。

### .1 物理模型与控制方程

我们要模拟的是一个三维物体内的非稳态（瞬态）热传导过程。这个过程由傅里叶热传导定律描述，其微分形式的控制方程（也称为热扩散方程）如下：

$$
\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + S
$$

其中：
*   $T$ 是温度 (K 或 °C)，是时间 $t$ 和空间坐标 $(x, y, z)$ 的函数。
*   $\rho$ 是材料密度 (kg/m³)。
*   $c_p$ 是比热容 (J/(kg·K))。
*   $k$ 是导热系数 (W/(m·K))。
*   $\nabla \cdot$ 是散度算子，$\nabla$ 是梯度算子。
*   $S$ 是源项 (W/m³)，代表内部热源，本问题中为0。

为了简化，我们假设材料属性 $(k, \rho, c_p)$ 是均匀且恒定的。这样，方程可以写为：

$$
\frac{\partial T}{\partial t} = \frac{k}{\rho c_p} \nabla^2 T = \alpha \nabla^2 T
$$

这里 $\alpha = \frac{k}{\rho c_p}$ 被称为**热扩散率** (m²/s)，它描述了热量在材料中扩散的速度。$\nabla^2$ 是拉普拉斯算子，在笛卡尔坐标系下展开为：

$$
\frac{\partial T}{\partial t} = \alpha \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} + \frac{\partial^2 T}{\partial z^2} \right)
$$

### .2 有限体积法 (FVM)

有限体积法的核心思想是将计算区域（我们的砖块）划分为许多不重叠的微小控制体（Control Volume, CV），然后对每个控制体积分上述的控制方程。

对方程在一个控制体 $P$（体积为 $\Delta V$）上进行积分，并对时间从 $t$ 到 $t+\Delta t$ 积分：

$$
\int_{t}^{t+\Delta t} \left( \int_{\Delta V} \frac{\partial T}{\partial t} dV \right) dt = \int_{t}^{t+\Delta t} \left( \int_{\Delta V} \alpha \nabla^2 T dV \right) dt
$$

#### (1) 时间项（左侧）:

假设在一个时间步 $\Delta t$ 内，控制体 $P$ 内的温度 $T_P$ 是均匀变化的，我们可以将其离散为：

$$
\int_{\Delta V} (T_P^{t+\Delta t} - T_P^{t}) dV \approx (T_P^{new} - T_P^{old}) \Delta V
$$
这里的 $T_P$ 指的是控制体中心节点的温度。

#### (2) 空间项（右侧）:

利用高斯散度定理，体积分可以转化为面积分：

$$
\int_{\Delta V} \nabla^2 T dV = \int_{\Delta V} \nabla \cdot (\nabla T) dV = \oint_{A} (\nabla T) \cdot d\mathbf{A} = \sum_{f} (\nabla T)_f \cdot \mathbf{A}_f
$$

这里的 $A$ 是控制体的总表面积，$f$ 代表控制体的各个面（东、西、南、北、上、下），$\mathbf{A}_f$ 是面的面积矢量。这代表了通过所有面流入控制体的净热通量。

#### (3) 离散方程:

对于一个内部控制体P，它有六个邻居：东(E)、西(W)、北(N)、南(S)、上(T, Top)、下(B, Bottom)。通过P和E之间交界面的热流（x方向）可以近似为：

$$
q_e \approx k A_{yz} \frac{T_E - T_P}{\Delta x}
$$
这里 $A_{yz}$ 是东西向的截面积，$\Delta x$ 是东西节点间的距离。

将所有六个面的热流通量加起来，再结合时间项，我们就得到了一个内部节点的温度更新公式（使用显式时间格式）：

$$
\frac{T_P^{new} - T_P^{old}}{\Delta t} \Delta V = \alpha \left( A_{yz}\frac{T_E^{old} - T_P^{old}}{\Delta x} + A_{yz}\frac{T_W^{old} - T_P^{old}}{\Delta x} + A_{xz}\frac{T_N^{old} - T_P^{old}}{\Delta y} + \dots \right)
$$

对于均匀的立方体网格（$\Delta x = \Delta y = \Delta z = h$），这个公式可以简化为我们熟悉的有限差分形式：
$$
T_P^{new} = T_P^{old} + \frac{\alpha \Delta t}{h^2} \left( (T_E^{old} - 2T_P^{old} + T_W^{old}) + (T_N^{old} - 2T_P^{old} + T_S^{old}) + \dots \right)
$$
我们的代码将实现这个更新逻辑。

### .3 边界条件

#### 第一类（Dirichlet）边界条件

直接给定边界上的温度值。在我们的问题中，左侧面 ($x=0$) 的温度始终为 200°C。在每一时间步计算后，我们强制将这些边界节点的温度设为200。


#### 第二类（Neumann）边界条件: 

给定边界上的热流密度。绝热边界意味着热流为0，即 $\frac{\partial T}{\partial n} = 0$（温度在边界法线方向上的导数为0）。在离散格式中，最简单的实现方法是令边界节点温度等于其相邻的内部节点的温度。例如，在右侧绝热边界上，我们设置 $T_{boundary} = T_{interior\_neighbor}$。

### .4 稳定性

我们使用的显式时间推进格式有一个重要的限制，即时间步长 $\Delta t$ 必须足够小才能保证计算稳定。对于三维问题，稳定性条件为：

$$
\Delta t \le \frac{1}{2\alpha \left( \frac{1}{(\Delta x)^2} + \frac{1}{(\Delta y)^2} + \frac{1}{(\Delta z)^2} \right)}
$$

如果 $\Delta t$ 超过这个阈值，计算结果会出现无意义的振荡并迅速发散。在代码中，我们必须计算这个值并选择一个比它小的 $\Delta t$。

---

**Python 代码实现**

下面是完整的 Python 代码，包含了网格生成、可视化、FVM 求解器以及最终结果的展示。
