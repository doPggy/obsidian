# profiler
代码瓶颈在某一段上

但是数据库读取。

要先通过 profiler 去找瓶颈

cProfiler - function profiler


## 优化原则
1. 算法本身
2. 复杂度：指数计算比加复杂得多 
3. 实现细节： hash、simd、内存存储顺序
4. 并行

## 并行

## hotspots
利用 profiler 找。

但是寻找原因比较难。高级语言底层不好控制。

内存读取是瓶颈核心。因为 cpu gpu 已经很快了。

## 内存层次
cpu 只能在 rigister 工作。

cache 无法控制。

寄存器还能用汇编

## cache 问题
1. 不好读。成批读邻近数据，但会不命中，所以数据到处都是很麻烦。

存二维矩阵：存一个向量。以 row 来存储，但是读取以 col 来读取，要不断跳过一个 col 长度，很容易就 cache 不命中。

2. 反复读。条件语句前，会先预先读数据入 cache。多线程不断改邻近内存。


## simd
openmp 不适合高级编程语言，用于加速

gpu 怕树结构。



## target encoding
把一些离散量编码成不同类别

但是 target leakage。本质是 x 吧 y 包括进来

如果一个类样本数据太少，假如只有 3 个，其中一个的修改会大大影响结果，也等于把  x 和 y 包括了。



## cython
python 没有数组概念，numpy 比较接近

代码保护：代码混淆容易有问题

建议弄成一个 module ，用 python setup.py install 方式。

pyx -> .cpp -> .so 弄成了 c++ 之后生成 .so 文件在本地

python setup.py build_ext -i install

注意，以上是自定义第三方打包流程。


```c
#pragma once
```

c++ 中尽量不用 bool，他是一个 bit

python 看不到 cdef

cpdef 定义了两个函数，一个 python 能看到，cython 能看到，利用 def 包装 cython

不用 c 的问题：如何把数据传给 c？但是 python 可以把数据给 cython

变量基本 cdef

cimport 和 import：numpy 有 c 实现，如果想用，就用 cimport。

python -> cython -> c

所以资源分配不要跨阶层释放，所以在 cython 或者 python 生成一些内存，例如 numpy。

python 多线程加速要用 cython 到 c++ 中。