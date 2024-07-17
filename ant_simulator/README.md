# Simulator Read
目前阅读下来，（已阅读的）文件的目录功能大致如下：

1. [bitfusion](#bitfusion)
    * [sram](#sram)
    * [src](#src)
        + [optimizer](#optimizer)
2. [dnnweaver2](#dnnweaver2)
    * [compiler](#compiler)
    * [fpga](#fpga)
    * [tf_utils](#tf_utils)

## bitfusion
里面大概都是性能计算的模块

### sram
+ 包含了用于模拟和分析缓存和存储器性能特性的 cacti 工具。

### src
包括了有关模拟器实例的基本组件。

#### optimizer
+ 模块的主要功能是优化卷积神经网络在硬件加速器上的执行性能。通过调整数据分块和循环顺序，最小化计算周期和能量消耗，提升整体计算效率。

## dnnweaver2
这个目录包括了硬件模拟的一些代码。

### compiler
+ 定义了一些处理单元的运算（figure 2 中的 compute 模块可能就在这）。

### fpga
+ 用于管理 FPGA 的资源。

### tf_utils
+ 包括一些自定义的量化内容，不同于论文中的实现。可以按 quant 为前缀检索相关内容。


# ANT Simulator

This repository contains the code for the ANT simulator based on DNNWeaver and BitFusion.

## Prerequisite

+ Ubuntu 18.04.5 LTS
+ Andconda 4.10.1
+ Python 3.8
+ gcc 7.5.0

## Getting Started

```shell
$ # Environment.
$ conda create -n ant_sim python=3.8
$ conda activate ant_sim  
$ pip install -r  requirements.txt
$ # Cacti for the memory simulation.
$ git clone https://github.com/HewlettPackard/cacti ./bitfusion/sram/cacti/
$ make -C ./bitfusion/sram/cacti/
$ # Run ANT simulation.
$ python ./run_ant.py       # About 15 minutes
```

## Evaluation

The script `run_ant.py` generates statistic data and stores it in file `./result/ant_res.csv`. Note that BiScaled only test on VGG16 and ResNet50.

In `./result/ant_res.csv`, Line 3 shows the **cycle** data that normalized with AdaFloat. Line 7-10 shows the **energy** data that normalized with AdaFloat.

As shown below, the `./result/ANT-simulator.xlsx` provides the template. You can fill it with the numbers of `./result/ant_res.csv` to generate Figure 13 in the paper.

<div>
<img src=./docs/img/evaluation.png width=100%>
</div>
