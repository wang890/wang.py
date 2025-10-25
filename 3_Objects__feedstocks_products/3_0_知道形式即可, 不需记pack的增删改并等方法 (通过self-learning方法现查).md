3_7节，待学完 chapter 4 后再回来学习。

### .1 objects：feedstock and product of Function machine
+ 单个的且初级的 single and primary: None, Number, Bool
+ 单个的且自定义 single: self-defined by `class`，类的部件、部件的元件都可能是自定义的类

+ 非单个的一堆的 pack and iterable, 五种：str字符串, list列表, tuple元组, set集合, dict字典

+ 有序的 ordered pack，称为 **sequence**: str, list, tuple
+ 无序的 unordered pack: set, dict

+ 可修改的 mutable pack: list, set, dict
+ 不可修改的 imtable pack: str, tuple
+ 有序且可修改的 ordered and mutable pack: list

上述 objects 满足了所有应用场景，选用何种类型的 feedstock object, 直接影响其处理函数算法的复杂性。

### .2 程序和算法
+ 程序就是 feedstock objects → Machine processes → product objects；
+ Machine processes，由operator运算符、function函数、class.function类的函数组成，他们都可称为算法；
+ 算法：就是 处理步骤 或者 系列处理步骤 Machine processes；
+ 算法的封装：语句表达式expr → 函数fenction → 类lib → 库lib → 软件software。


### .3 pack
+ 由一堆元素孩子组成，元素element 之间用**英文逗号分隔**，字典的元素是 键值对`key:value`；
+ pack可以无限嵌套，但嵌套多的pack，其处理函数写起来复杂；

+ 形式：str `''` or `""`, list `[]`, tuple `()`, set `{}`, dict `{key1:value1, key2:value2, ...}`;
+ 形式，特殊：空的 set `set()`, 只有1个 元素element 的tuple `(ele,)`;

+ 五种pack都可迭代iterable，dict在迭代时默认以 key为主；
+ 文中的 iterable 即指 str, list, tuple, set, dict 这五种当中的任何一种；
+ 文中的 mutalbe  即指 list, set, dict 这三种当中的任何一种。

### .4 如何学习 pack
+ 知道 五种pack 的形式即可；
+ 虽然教程代码包括 pack的增删改并 等方法，通过 self-learning方法 现查，如 `pack.`后IDE(编程软件，集成开发环境)就能显示有哪些方法函数；
+ 有某种场景需求时，可以按场景百度搜索。

### .5 Python语言特点
+ 用tab缩进体现语句代码之间的层级关系；
+ 除了缩进，其他地方的空格可以是多个 (有时可以利用这个特点做对齐)；
+ 体现语法的标点符号均是英文标点，标点符号后可以留0至多个空格，规范的形式为留一个空格 (行的末尾时不留空格)。