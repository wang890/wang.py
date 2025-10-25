def func():
    pass


class Human(object):
    def __init__(self, name):
        self.name = name


# 所有的self表示当前实例, 可以全部替换成其他词语 如this, 有的语言就是this
class Student(Human):          # 类,模板
    total = 0                  # 共性 类变量

    def __init__(self, name):  # 实例方法, 具体来说 这个init函数是初始化函数即构造函数
        self.univ = 'NCEPU'    # 实例变量
        self.__class__.total += 1
        super(Student, self).__init__(name)  # 由于继承有父类，父类Human中有构造函数, 在此要调用父类的构造函数
        print(self.name)

    @staticmethod
    def func():
        pass

    def chi_fan(self):     # 实例(对象)函数, self 实例化后的具体的某个Student, 对象
        print(self.name + '拿了标有"' + self.name + '"的饭盒') # 这个是个性化的 用了个体实例的名字，所以此函数为实例函数

    @classmethod           # 类方法，共用 这个类能够使用的方法
    def kai_ban_hui(cls):  # cls可以替换成其他词语，指代的是这个类Student
        cls.total += 0
        print('开班会是所有学生的事情，是共性，不是某个学生的个性')


if __name__ == '__main__':

    Zhangfei = Student('Zhangfei') # 实例化一个学生，以模板Student()造出一位学生
    Zhangfei.chi_fan()
    Zhangfei.kai_ban_hui()
    Student.kai_ban_hui()

    print('total='+str(Student.total))

    Guanerye = Student('Guanerye')
    Guanerye.chi_fan()
    Guanerye.kai_ban_hui()
    Student.kai_ban_hui()

    print('total='+str(Student.total))