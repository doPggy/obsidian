本笔记总结来自==**极客时间王然老师的机器学习训练营，再次感谢王然老师的教学与指导**==，这里务必给极客时间王然老师的机器学习训练营打个广告。

这一周的学习主要围绕着优化 python 代码性能来展开的。

那么如何提升 py 代码的性能，首先需要一套指导思想，然后才能有方法论。 

---
# 思想
一般思路是这样：
1. 找到代码瓶颈。
2. 宏观上找修改策略。(这是什么)
3. 利用更底层的语言来实现功能从而提速。
4. 最终才想到并行操作。

当然还是实践出真知， 同时当你发现代码瓶颈：
1. 可修改，那么就改了。
2. 无法修改。数据库的瓶颈，框架的问题。那么完全可以换一个或者提 bug 给社区或者就不管啦。

## 优化基本原则
第一个原则：不要提前优化。这样很容易导致代码写不出来，可取的做法是，哪怕笨一点的办法，**先实现出来保证正确性**，然后我们在接着优化。（利用作业展示这一个过程

第二个原则：已经找到瓶颈着手优化的时候， 有一个优先顺序：
1. 算法本身。A 算法怎么优化都不如 B 算法，那直接用 B 算法不就好了。
2. 算法复杂度的优化。不换算法了，对于现有算法本身优化，首先想到的就是复杂度。例如指数计算远复杂于加法运算。或者能一个循环就不用两个循环了。
3. 实现细节。如果设计上已经找不到优化点，那么落地这个算法的各个实现方法可能就是有问题的。例如算法主要以 `col major` 读取方式，内存存储的顺序以 `row major`（数组的 row major/col major）或者是没有用 SIMD([SIMD 简介](https://zh.wikipedia.org/wiki/%E5%8D%95%E6%8C%87%E4%BB%A4%E6%B5%81%E5%A4%9A%E6%95%B0%E6%8D%AE%E6%B5%81))（也就是利用多个处理器实现相同计算的并行）
4. 并行。这是最后才需要想到的操作，而不是一开始就想到并行。并行需要注意的点很多，所以不优先考虑，很容易会抢占他人资源。

# 方法论
指导思想已经知道了，重要的还是如何实践，从而形成一套方法论。
## 找代码瓶颈
根据思想指导，那么首先我们就需要找代码瓶颈。

个人实践经历和老师提供的方法，可以总结成两个：

1. 通过经验和对代码可读性的理解来排查代码拖慢的点
2. **基于 profile 查找代码瓶颈**

那么 profiler 一般就是分析程序的执行的各个指标。例如执行时间等。而 profiler 分为两种:
1. function profiler。用于函数级别的分析，不进入，有点像单步调试中的 step over。
2. line profiler。会分析每一个语句，并且可能有他的调用栈，消耗时间等等。py 语句往往一句就有很多处理，这样导致不断地 step into，很有侵入性。

目前介绍了有 cprofile/Vtune/line_profiler， 由于 Vtune 的特殊性，暂时就不实践了，但是这个是很好的 profiler。

### cprofile 
这是 function profiler。

这是 cprofile 的命令行方式调用，其实还有程序内使用的方式，这个完全可以查看官方文档，不再赘述。

![cprofile 使用](http://img.multiparam.com/dapao/ai/20210103155609.png)

里头的一些指标，[拷贝自官方文档](https://docs.python.org/3/library/profile.html)，多看文档是好习惯。

```
ncalls:  被调用的次数

tottime: 指定函数花费时间，但不包括函数中调用其他函数的时间。

percall: 是 tottime 除以 ncalls 的商

cumtime: 是此功能和所有子功能（从调用到退出）花费的累积时间。 即使对于递归函数，此数字也是准确的。
```

但是它依旧存在缺点：
1. 没有 line profiler
2. 缺少其他底层信息，从上图也能发现，那些指标基本都是时间上的消耗，没有内存，cpu 耗时等信息。
3. 没有图形化界面(这个我还没注意，可能是因为 Vtune 有且强大)
4. 有时候时间不准确。(这个还没遇到过，不知道怎么去判定。)，但是有这个问题就很致命。
5. 多语言情况难处理。（这个也是没遇到过。也许是字符串带中文比如，就有问题。)

### line_profiler
主要是利用 @profile 装饰器来使用。

一直安不下来，windows 是真的坑，不过 colab 是有支持的，十分舒服。不过使用 colab 的方式就得使用 `%lprun` 的方式了，具体会写到代码里，大致如下：

```
%load_ext line_profiler

def xx():
	print(112312
	
%lprun -f xx xx()
```

执行结果大致如下：

![结果](http://img.multiparam.com/dapao/ai/20210103170826.png)

在 colab 上我也找到了一种方式，就是在文件中是可以新建文件的，把相关文件新建，然后利用：

```
!kernprof -l -v xxx.py
```

![文件位置](http://img.multiparam.com/dapao/ai/20210103173857.png)

但是在网页上的编写效果体验不好，可以本地写好复制上去。

这个最好再写一个操作文档。  [看这个例子应该够了](https://colab.research.google.com/drive/1HSC11nr5TeTN1lS3j3Fe_NmDpoeBIRdZ#scrollTo=GEtz8lVMpZMY)

### 代码瓶颈原因分析
上头说了不少利用 profiler 来分析代码瓶颈。但是有几个问题：
1. python 没有特别好的 profiler，Vtune 可能是。
2. 瓶颈的原因难以找到。cprofile 和 line_profile 只能看到底层的表现(例如时间)，但是不能知道造成的原因。同时 py 这样的语言，一句可能就包含很多事情，你也不知道是哪里的问题。

所以可以推断，（注意我说的是推断，因为我没有真实践过，希望以后我有丰富的实践经验来补充）我们的优化，由于难以控制底层，还有算法本身的原因，导致容易找不到症结。

但是根据老师提供的思路，不知道造成瓶颈原因可以优先考虑：
1. 优化内存方面的表现
2. 使用 SIMD 来提速

#### 优化内存
为什么考虑内存，因为现在 GPU/CPU 已经很快了，所以往往慢得那个会成为瓶颈，所以会想到内存读取。

而内存读取往往是可以控制的，但是 cache 是直接由操作系统控制，导致无法控制，就可能出现瓶颈。

我们这么考虑，CPU 读取 cache 内存数据，但是

1. cache 是成批读邻近数据（这个学操作系统的时候讲过，可以问程序锅），如 cache 中的数据不连续或者 CPU 读方式对于 cache 数据来说不连续。容易造成 cache miss 问题。依旧使用 `col major` 和 `row major` 的例子来说，如果矩阵用 `fortran` 方式存储，也就是 `col major` 存储。而程序使用 `row major` 方式读，明显是不连续的，这样 cache 读的邻近数据就是无效的。

2. 反复读取问题。条件语句下，cache 是会预先读取其中一个判断分支需要的数据，要是其实多走的判断分支与预先读取的不一致，就要不断改。还有就是多线程反复修改问题。

#### SIMD

SIMD([SIMD 简介](https://zh.wikipedia.org/wiki/%E5%8D%95%E6%8C%87%E4%BB%A4%E6%B5%81%E5%A4%9A%E6%95%B0%E6%8D%AE%E6%B5%81)) 利用多个处理器实现相同计算的并行，许多操作可以使用这个方式来达到提速。例如对于矩阵每一列的累加，完全可以分摊到不同的处理器上。

实现方式：OpenMp

## 宏观修改
这个先放着，还没有想法。应该就是框架或者算法的更换。



---
以下两个是优化方法论：用底层语言实现和并行。这两个需要更多的实践，故先总结在 iynb 文件或相关 py 文件中，之后再整理出来，故先写一些关键点。

## 用底层语言去实现
来到关键 cython。

关于 cython 其实 iynb 文件上已经有了很多内容，而且更多需要自己的实践。后头给一个 colab 链接或者给一些代码片段即可。可以单独写一篇文档。

主要围绕这几个：
1. cython 能做什么。
2. cython 关键语法。
3. cython 编译的推荐做法。
4. cython 优化方法。

fortrain order 是什么。

cython 中 numpy 的类型声明。

## 并行
openMp、Eigen、oneTBB、Denpendecy graph。

ray


# 实例分析

由这次作业我真的理解了什么叫做优化，基于 profiler 的优化。

我先通过经验，其实这里就要用 profiler，然后开始调整算法。然后通过 profiler 找瓶颈，然后发现 loc 可以替换成 at。