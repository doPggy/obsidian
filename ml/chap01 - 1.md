# 环境搭建
1. anacoda
2. cuda
3. cmake
4. docker NVIDIA docker

conda 可能不同包之间有矛盾。 

conda instal python=3.7

miniconda

## Docker
轻量虚拟机

docker 的安装

docker pull

docker run --it --rm  -gpus  all -v ... -p 888:888 (各自什么意思？)

-v 本地文件夹映射进来
-p 把本地端口映射到  docker

## NVIDIA Docker
disable nouveau


## cuda
显卡驱动

用 `runfile` 安装, sh ....run --no-opengl-libs

pytorch 要什么样的 cuda 版本



# python/R 基本语法与常见坑
1. 变量与赋值
2. 控制循环
3. 函数
4. 类
	1. 普通
	2. 抽象
5. 异常

## 异常处理
好处？防止崩溃

坏处？底层有一个异常，外层函数要不停 try

monad。一个函数返回的东西可能是正常值或者是异常，外层调用需要判断。而利用 monad 方式就是，对于异常情况做一个默认的操作，这样就不用去管了。 

## python 对象拷贝
副作用
浅拷贝：引用的传递
深拷贝

## ghost bus
可变对象做参数默认值，会保留上一次结果。（为什么？这个可变对象是全局的


## *args **kargs


