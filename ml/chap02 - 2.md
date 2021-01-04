# 预习内容
```
@cython.boundscheck(False)
@cython.wraparound(False)
```
分别是什么意思。

来自官网查到的，对于 array 类型数据来说，即使利用 cdef 来明确各个变量，使用更确切的类型生命来提高效率。但高效索引的瓶颈还有两个：
```
The array lookups are still slowed down by two factors:

1.  Bounds checking is performed.
2.  Negative indices are checked for and handled correctly. The code above is explicitly coded so that it doesn’t use negative indices, and it (hopefully) always access within bounds.
```

# 上课
把 cython 写成 c。但是要定义静态类

cython 做数据预处理，然后把指针传入 c 。

cimport 导入 c 底层实现


clog 关键

优化到后头会有付出收益比不佳的情况

边界检查和负索引都关闭，就需要保证代码明确，反正不要像 Python 那样写了。


## fortran order 
二维指针，动态指针分配的内存可能是不连续的。这样就联系到 cache miss 的问题。那么行先排列在数组，还是列先排列。

fortran 是 col 优先，一列一列存
c 是 row 优先，一行一行存
例如一张表，对他行操作多，例如算员工总薪资 用 c mode


## cython connect c
cdef extern from

最好的方式：在 cython 分配资源例如生成 vector，然后穿指针进 c 里头。

在 cython 中其实也是在 python 中，所以内存回收机制可以管理。

numpy 内积 c 实现慢了一百万倍哈哈哈哈。

可以使用 eigen 的 map 来接 cython 的内存


simd：提速方法? 这个要去查


## ray
优势：
1. 调用 Python 函数
2. 共享数据(只读)。一般多进程不能传，只能 copy。因为存在 race condition。


DAG ?

Akka stream?

actor 实现