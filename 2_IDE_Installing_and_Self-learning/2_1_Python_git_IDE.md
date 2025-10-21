一边观看笔者录制的视频，一边安装，有如下环节：
+ 电脑文件夹整理
+ 安装 Python
+ 安装 vscode，为vscode安装Python插件，这样vscode才能认识 Python代码
+ 安装 git, 有一步是选择默认的代码编辑器，选vscode
+ 设置 Python库的镜像下载地址，默认国外下载（你所在机构的国外下载速度可能有点慢）
+ 克隆或新建project，为project设置虚拟环境：
  项目之间libs独立，项目之间的Python版本可以不同
+ Project用到了哪些lib库，就 pip install libname

这样就可以啦，本md文件的如下内容，是以前写的参考资料，现在不需要啦。

### .1 interpreter Python
+ https://www.bilibili.com/video/BV1TN411K7sn/?p=2，
  Python多版本共存，修改Python.exe的文件名（作者注：需核实）。
  此为系列视频，还包括VScode虚拟环境设置。
  
### .2 version control Git

### .3 code storage website Github

### .4 IDE Pycharm
集成开发环境IDE, 优先使用Pycharm, 而不是VScode。
对于Python编程来说，PyCharm的功能更为强大。

+ https://www.bilibili.com/video/BV1EQ4y1V7yx <br> pycharm创建虚拟环境
+ https://www.bilibili.com/cheese/play/ss1852 <br> Pycharm收费课程(没必要购买), 含git操作

### .5 IDE VScode
+ vscode, 下载System Installer, 点×64不要点arm64
  - https://code.visualstudio.com/download
  - https://code.visualstudio.com/updates
+ 32位下载， 1.83.0 (sep 2023)之后不再更新32位
   + https://update.code.visualstudio.com/1.83.0/win32/stable 跳转到如下
   + https://vscode.download.prss.microsoft.com/dbazure/download/stable/e7e037083ff4455cf320e344325dacb480062c3c/VSCodeSetup-ia32-1.83.0.exe 

+ Previous release versions
  https://code.visualstudio.com/docs/supporting/faq#_previous-release-versions

+ https://www.bilibili.com/video/BV18jq8YNE8t/?p=7, 系列：vscode的python环境安装,讲复杂啦。
+ https://www.bilibili.com/video/BV1TN411K7sn/?p=3, 系列：5分钟搞定VScode中配置Python运行环境 
+ https://www.bilibili.com/video/BV1tF411M7hy, VScode中配置Python运行环境
+ https://docs.pingcode.com/ask/1010434.html, vscode如何运行python交互
+ https://www.cnblogs.com/txwtech/p/18026597, vscode python 快捷键大全

### .6 lib management pip
+ https://www.baidu.com/s?wd=pycharm%20生成requirements, <br>
  百度AI显示了几种方法，其中之一为：**pip freeze > requirements.txt**
+ https://www.baidu.com/s?wd=python%20pip%20requirements, <br>
百度AI结果中有，**pip install -r requirements.txt** 即可安装所有依赖包。
+ https://blog.csdn.net/Stromboli/article/details/143824330, <br>
较为详细的 pip包管理命令