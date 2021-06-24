**<center><font size='3'>YOLOv5剪枝算法测试报告</center>**

[1 工程准备](#1-测试环境搭建)

[2 测试结果](#2-测试结果)

[3 结果分析](#3-结果分析)

# 概要

​		为满足硬件部署条件，利用[剪枝工具包](https://github.com/VainF/Torch-Pruning)（[理论基础](https://arxiv.org/abs/1608.08710)）实现通道剪枝。模型大小与精度、推理时间是互斥条件，由此提供多组不同模型大小的剪枝效果测试，额外补充了通道数减半的效果对比。最终剪枝效果基本达到预期，结果详见[表1](#all0)及[报告末总结](#3-结果分析)。

# 1 工程准备

## 1.1 测试算法框架及运行环境

算法框架选取[YOLOv5_v4.0](https://github.com/ultralytics/yolov5/tree/v4.0),在原有框架上加载剪枝工具包API

主要依赖环境为服务器vision001

## 1.2 工程代码及变动

详见git(todo整合上传至git)

- 添加剪枝工具包源码至根目录
- 修改train.py
  - 增添稀疏训练调用参数及log （[稀疏训练理论基础](https://arxiv.org/abs/1708.06519)）
  - 增添微调调用参数及模型加载代码
- 添加剪枝代码pruning.py
- 在loss.py中添加关于稀疏损失计算sl_loss的定义

## 1.3 测试数据集

训练集&验证集：[dmcode](http://172.16.102.80/#/image-display?storeId=37&type=1): 5736张； [qrcode](http://172.16.102.80/#/image-display?storeId=38&type=1): 5538张； [barcode](http://172.16.102.80/#/image-display?storeId=36&type=1): 8728张； [multicode(去除测试集)](http://172.16.102.80/#/image-display?storeId=46&type=1): 8128张； 共28130张   

测试集： 3112张

# 2 测试结果

## 2.1 运行流程

<center><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-%E5%89%AA%E6%9E%9D%E6%B5%81%E7%A8%8B.png" style="zoom:25%;" /></center>

大参数模型（original模型）采用YOLOv5s（epoch=578），此时模型拟合。

训练流程首先设定稀疏阈值对original模型进行稀疏训练；然后稀疏训练得到的模型，通过设置剪枝阈值完成剪枝；最后，为恢复原有精度，对剪枝后的模型进行微调。

⭐剪枝整个过程关键在于找到合适的剪枝阈值，从而合理地缩减推理时间。

## 2.2 稀疏训练（稀疏阈值6e-4）

不同epoch下，稀疏分布可视化如左图（Y轴权重数，X轴稀疏度），mAP变化曲线如右图。

<center><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-%E7%A8%80%E7%96%8F%E5%BA%A6%E5%8F%98%E5%8C%96.gif" alt="稀疏度变化" style="zoom: 50%;" align=lift/><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-%E7%A8%80%E7%96%8F%E8%AE%AD%E7%BB%83%E6%8D%9F%E5%A4%B1%E5%8F%98%E5%8C%96.jpg" alt="稀疏训练损失变化" style="zoom: 50%;" align=right/></center>

依据[预剪枝测试结果](#剪枝预测试)确定合适的剪枝阈值后，增加epoch至449，此时稀疏分布和mAP趋于稳定。

<center><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-epoch171%E7%A8%80%E7%96%8F%E5%BA%A6%E5%88%86%E5%B8%83.jpg" alt="epoch171稀疏度分布" style="zoom: 50%;" align=lift /><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-%E5%A2%9E%E5%8A%A0epoch%E5%90%8E%E7%9A%84%E7%A8%80%E7%96%8F%E5%BA%A6%E5%88%86%E5%B8%83.jpg" alt="增加epoch后的稀疏度分布" style="zoom: 50%;" align=right/></center>

## 2.3 剪枝效果

预剪枝：探索不同剪枝阈值的效果，取稀疏训练epoch=171，从精度损失最小化以及极限剪枝的角度，剪枝阈值分别选取0.01与0.2作为测试组，后续补充了折中阈值0.1。

极限剪枝：根据预剪枝确定剪枝阈值，在预期精度内尽可能压缩模型大小，从而获取更短的推理时间。

### 2.3.1 精度及速度

表1统计<a name="all0">各模型的测试效果</a>，备注为模型拟合情况或epoch。置信度阈值为0.01，推理耗时默认图片640×640(batch_size=16)下进行测试。

| 序号 |             模型             |           mAP@0.5            |     ▲      |               mmAP               | 模型大小  |             推理耗时             | FLOPs |     备注     |
| :--: | :--------------------------: | :--------------------------: | :--------: | :------------------------------: | :-------: | :------------------------------: | :---: | :----------: |
|  1   |    [original](#混淆矩阵0)    | <font color=red>0.982</font> |     0      |              0.862               |   15.1M   | **<font color=red>2.0ms</font>** | 18.6G |     拟合     |
|  2   |            sl171             |            0.961             |   -0.021   |              0.813               |   15.1M   |              2.0ms               | 18.6G |     171      |
|  3   |          sl171_0.01          |            0.966             |   -0.016   |              0.828               |   3.33M   |              1.3ms               | 6.9G  |      50      |
|  4   |          sl171_0.1           |            0.955             |   -0.027   |              0.780               |   1.53M   |              1.0ms               | 3.7G  |      50      |
|  5   |          sl171_0.1+          |            0.957             |   -0.025   |              0.803               |   1.53M   |              1.0ms               | 3.7G  |   基本拟合   |
|  6   |          sl171_0.2           |            0.934             |   -0.048   | **<font color=red>0.661</font>** |   488K    |              0.7ms               | 1.3G  |      50      |
|  7   |            sl449             |            0.963             |   -0.019   |              0.836               |   15.1M   |              2.0ms               | 18.6G |     449      |
|  8   | **[sl449_0.01](#混淆矩阵5)** |          **0.965**           | **-0.017** |            **0.838**             | **2.48M** |            **1.2ms**             | 5.5G  | **基本拟合** |
|  9   | **[sl449_0.08](#混淆矩阵6)** |          **0.962**           | **-0.020** |            **0.825**             | **1.62M** |            **1.0ms**             | 3.8G  | **基本拟合** |
|  10  |           1/2通道            |            0.956             |   -0.027   |              0.837               |   3.9M    |              1.0ms               | 4.4G  |     拟合     |
|  11  |           1/4通道            |            0.961             |   -0.021   |              0.808               |   1.2M    |              0.6ms               | 1.1G  |     拟合     |

<p align="right">p.s.  “模型sl171_0.01” 指稀疏训练epoch=171以及剪枝阈值为0.01</p>
<p align="right"> “基本拟合” 指模型随着epoch增加几乎不再更新模型</p>

### 2.3.2 混淆矩阵(IoU阈值0.5)

根据不同剪枝阈值，客观分析AP变化。值得注意，各模型类间均不存在误检，因此FP主要为残缺码识别，详见[误检率说明](#31 关于误检率较高的说明)。

#### 极限剪枝

表2 <a name="混淆矩阵5">sl449_0.01</a>

|  模型   |  TP  | FP分布(误检) | FN(漏检) |  AP   |
| :-----: | :--: | :----------: | :------: | :---: |
| dmcode  |  1   |     0.09     |    0     | 0.994 |
| qrcode  |  1   |     0.15     |    0     | 0.995 |
| barcode | 0.93 |     0.76     |   0.07   | 0.906 |

表3 <a name="混淆矩阵6">sl449_0.08</a>

|  类型   |  TP  | FP分布(误检) | FN(漏检) |  AP   |
| :-----: | :--: | :----------: | :------: | :---: |
| dmcode  |  1   |     0.06     |    0     | 0.994 |
| qrcode  |  1   |     0.18     |    0     | 0.995 |
| barcode | 0.93 |     0.76     |   0.07   | 0.896 |

# 3 结果分析

### 3.1 关于误检率较高的说明

​		当置信度阈值较低（conf_thres=0.01）时，只有条形码出现较大比例漏检，而且误检主要占比也是条形码，如表4。另一方面，在original模型的图片检测可视化中（如下图），网络将大部分**残缺码**充当正确识别。综合表2-表3，剪枝会导致该条形码误检率放大。

表4 <a name="混淆矩阵0">original</a>-置信度阈值为0.01

|  类型   |  TP  | FP分布(误检) | FN(漏检) |                AP                |
| :-----: | :--: | :----------: | :------: | :------------------------------: |
| dmcode  | 0.97 |     0.07     |   0.03   |              0.994               |
| qrcode  |  1   |     0.14     |    0     |              0.995               |
| barcode | 0.94 |     0.8      |   0.06   | **<font color=red>0.959</font>** |

<center><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-original%E6%9D%A1%E7%A0%81%E6%AE%8B%E7%BC%BA%E7%A0%81%E8%AF%AF%E6%A3%80(1).jpg" alt="original条码残缺码误检" style="zoom: 10%;" align=lift/><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-original%E6%9D%A1%E7%A0%81%E6%AE%8B%E7%BC%BA%E7%A0%81%E8%AF%AF%E6%A3%80(2).jpg" alt="original条码残缺码误检2" style="zoom: 35%;" align=center/><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-original%20DM%E6%AE%8B%E7%BC%BA%E7%A0%81%E8%AF%AF%E6%A3%80.jpg" alt="original DM残缺码误检" style="zoom: 40%;" align=right/></center>

------

### 3.2 剪枝效果分析

综上，得到以下结论：

​		在稀疏训练中，结合[表1](#all0)，**<u>①单次稀疏训练会降低精度，因此不宜进行人工迭代剪枝，只能通过增加稀疏训练epoch来获取较理想的稀疏分布</u>**

​		在<u>预剪枝测试</u>中，结合[表1](#all0)，sl171_0.01剪枝效果良好，模型压缩至3.3M（压缩率为78%），微调50epoch后精度总体下降1.6%（增加epoch有望恢复原有精度），推理时间由原来>8ms降至7ms。**<a name="剪枝预测试">考虑到剪枝后通道数为奇数会导致海思平台不能完全并行</a>**，即后处理时间未达预期，进一步增大剪枝阈值，得到sl171_0.1模型（压缩率为90%），其推理时间进一步下降到6ms，但精度只能恢复到0.960附近（相对original下降2.2%），基本与通道数减半及减至1/4的精度接近。<u>**②根据推理时间确定剪枝后模型大小为1.6M，由此推断剪枝阈值在0.01-0.1。**</u>

​		另一方面，结合[表1](#all0)中模型4及模型5可知，**<u>③单纯地增加微调epoch只能提升高阈值下的mAP（主要体现在条形码AP上），极大概率不能最终恢复至original的精度，而只能趋近或略高于稀疏模型的精度。</u>**

​		最后，通过增大稀疏训练epoch，得到极限剪枝模型sl449_0.01（推理速度6.41ms，mAP@0.5=0.965）及sl449_0.08（推理速度6.23ms，mAP@0.5=0.962）。

<center><img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-ft_449_0.01%E5%90%84%E5%8D%B7%E7%A7%AF%E5%B1%82%E5%89%AA%E6%9E%9D%E6%AF%94%E4%BE%8B.png" alt="ft_449_0.01各卷积层剪枝比例." style="zoom:50%;" /> <img src="https://whiskey-tuku.oss-cn-beijing.aliyuncs.com/img/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3&%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A/YOLOv5%E5%89%AA%E6%9E%9D%E6%B5%8B%E8%AF%95%E6%8A%A5%E5%91%8A-ft_449_0.08%E5%90%84%E5%8D%B7%E7%A7%AF%E5%B1%82%E5%89%AA%E6%9E%9D%E6%AF%94%E4%BE%8B.png" alt="ft_449_0.08各卷积层剪枝比例." style="zoom:50%;" /></center>

​		各层通道的修剪比例如上图，蓝色为保留部分，橙色为修剪部分。其中conv_123及conv_144为反卷积层，该剪枝方案并不支持反卷积剪枝。

​		**<u>④YOLOv5网络中head部分参数量大但计算量小，backbone恰好与之相反；目前方案优先剪裁head部分，当增大剪枝阈值时，backbone才得到剪裁。因此，相较于通道减半的模型，sl449_0.01及sl449_0.08虽然具有较小的模型体积，但推理耗时并没有显著优势。</u>**