# Simulator Read
目前阅读下来，（已阅读的）文件的目录功能大致如下：
## bitfusion
里面大概都是性能计算的模块
### sram
+ 里面包含了用于模拟和分析缓存和存储器的性能特性的cacti工具
### src
+ 里面包括了有关simulator实例的基本组件

## dnnweaver
这个里面的话就是包括硬件模拟的一些代码
### compiler
+ 里面定义了一些处理单元的运算（那个figure 2中的compute模块我认为就在这）

### fpga
+ 里面用于管理fpga的资源
### tf_utils
+ 这个里面就是我说的，包括一些他自己量化的内容，不同于论文。可以按quant为前缀检索


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
