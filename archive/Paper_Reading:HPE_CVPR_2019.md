# HPE in CVPR 2019


- [HPE in CVPR 2019](#hpe-in-cvpr-2019)
  - [Fast Human Pose Estimation](#fast-human-pose-estimation)
    - [介绍](#%e4%bb%8b%e7%bb%8d)
    - [贡献](#%e8%b4%a1%e7%8c%ae)
    - [方法](#%e6%96%b9%e6%b3%95)
      - [知识蒸馏 (Knowledge Distillation)](#%e7%9f%a5%e8%af%86%e8%92%b8%e9%a6%8f-knowledge-distillation)
      - [网络结构](#%e7%bd%91%e7%bb%9c%e7%bb%93%e6%9e%84)
    - [实验](#%e5%ae%9e%e9%aa%8c)
    - [总结](#%e6%80%bb%e7%bb%93)
  - [Multi-Person Pose Estimation with Enhanced Channel-wise and Spatial Information](#multi-person-pose-estimation-with-enhanced-channel-wise-and-spatial-information)
    - [介绍](#%e4%bb%8b%e7%bb%8d-1)
    - [贡献](#%e8%b4%a1%e7%8c%ae-1)
    - [方法](#%e6%96%b9%e6%b3%95-1)
      - [CSM: Channel Shuffle Module](#csm-channel-shuffle-module)
      - [ARB: Attention Residual Bottleneck](#arb-attention-residual-bottleneck)
      - [整体网络结构](#%e6%95%b4%e4%bd%93%e7%bd%91%e7%bb%9c%e7%bb%93%e6%9e%84)
    - [实验](#%e5%ae%9e%e9%aa%8c-1)
    - [总结](#%e6%80%bb%e7%bb%93-1)
  - [Deep High-Resolution Representation Learning for Human Pose Estimation](#deep-high-resolution-representation-learning-for-human-pose-estimation)
    - [介绍](#%e4%bb%8b%e7%bb%8d-2)
    - [贡献](#%e8%b4%a1%e7%8c%ae-2)
    - [方法](#%e6%96%b9%e6%b3%95-2)
      - [高分辨率网络](#%e9%ab%98%e5%88%86%e8%be%a8%e7%8e%87%e7%bd%91%e7%bb%9c)
      - [重复多尺度融合](#%e9%87%8d%e5%a4%8d%e5%a4%9a%e5%b0%ba%e5%ba%a6%e8%9e%8d%e5%90%88)
    - [实验](#%e5%ae%9e%e9%aa%8c-2)
      - [标准数据集性能](#%e6%a0%87%e5%87%86%e6%95%b0%e6%8d%ae%e9%9b%86%e6%80%a7%e8%83%bd)
      - [重复多尺度融合](#%e9%87%8d%e5%a4%8d%e5%a4%9a%e5%b0%ba%e5%ba%a6%e8%9e%8d%e5%90%88-1)
      - [表征分辨率](#%e8%a1%a8%e5%be%81%e5%88%86%e8%be%a8%e7%8e%87)
    - [总结](#%e6%80%bb%e7%bb%93-2)
  - [Does Learning Specific Features for Related Parts Help Human Pose Estimation?](#does-learning-specific-features-for-related-parts-help-human-pose-estimation)
    - [介绍](#%e4%bb%8b%e7%bb%8d-3)
    - [贡献](#%e8%b4%a1%e7%8c%ae-3)
    - [方法](#%e6%96%b9%e6%b3%95-3)
      - [相关身体部位](#%e7%9b%b8%e5%85%b3%e8%ba%ab%e4%bd%93%e9%83%a8%e4%bd%8d)
      - [基于部分的分支网络(PBN)](#%e5%9f%ba%e4%ba%8e%e9%83%a8%e5%88%86%e7%9a%84%e5%88%86%e6%94%af%e7%bd%91%e7%bb%9cpbn)
    - [实验](#%e5%ae%9e%e9%aa%8c-3)
      - [基准数据集结果](#%e5%9f%ba%e5%87%86%e6%95%b0%e6%8d%ae%e9%9b%86%e7%bb%93%e6%9e%9c)
      - [Ablation study](#ablation-study)
    - [总结](#%e6%80%bb%e7%bb%93-3)
  - [Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views](#fast-and-robust-multi-person-3d-pose-estimation-from-multiple-views)
    - [介绍](#%e4%bb%8b%e7%bb%8d-4)
    - [贡献](#%e8%b4%a1%e7%8c%ae-4)
    - [方法](#%e6%96%b9%e6%b3%95-4)
      - [整体框架](#%e6%95%b4%e4%bd%93%e6%a1%86%e6%9e%b6)
      - [多视图对应](#%e5%a4%9a%e8%a7%86%e5%9b%be%e5%af%b9%e5%ba%94)
    - [实验](#%e5%ae%9e%e9%aa%8c-4)
    - [总结](#%e6%80%bb%e7%bb%93-4)


##  Fast Human Pose Estimation

这篇文章是电子科技大学和 Vision Semantics 共同完成的,关注于构建一个轻量化的人体姿态估计模型,在保证精度的同时提高模型效率,是一个偏向于应用层面的研究。

### 介绍

现有的人体姿态估计方法往往只考虑如何提高模型的泛化性能,而忽略了效率问题,导致重型模型很难在实际中使用。本文提出了一种新的快速姿态蒸馏(Fast Pose Distillation , FPD) 模型学习策略,训练能够以低计算成本快速执行的轻量级姿态神经网络结构。

### 贡献

本文的主要贡献可以总结为以下三点:

1. 研究了人体姿态估计模型效率问题,有助于将现有深度姿态估计方法扩展到实际应用。

2. 提出 Fast Pose Distillation (FPD) 模型训练方法,能够更有效地训练超小型人体姿态神经网络。

3. 设计了一个轻量级 Hourglass 网络,在提升成本效益同时保持足够准确率。

### 方法

#### 知识蒸馏 (Knowledge Distillation)

知识蒸馏的概念由 Hinton 在他的文章Distilling the Knowledge in a Neural Network 中首次提出，简要总结为通过教师网络 (teacher network) 输出作为 soft label 来训练一个学生网络 (student network) ，实现知识迁移。

#### 网络结构

文章采用快速姿态提取模型学习策略，整体框架如图1所示。首先训练一个大型教师姿态估计模型，本文采用原始沙漏模型。然后借助教师模型所学到的知识来训练目标学生模型，论文中学生网络为轻量级沙漏网络，训练时，教师网络为学生网络提供额外的监督指导，知识蒸馏便发生在这一过程中。在测试时，轻量学生姿态估计模型便能够实现快速且经济高效的部署，而计算量很大的教师模型最终被丢弃，因为它的辨别知识已经转移到目标模型中。

![图1 快速姿态提取模型学习策略](https://i.loli.net/2019/09/01/zX2pQdVLr8gWCNO.png)

<center>图3.1 快速姿态提取模型学习策略</center>

为能够有效地将教师网络的知识转移到学生模型的训练，本文设计一个合适的损失函数, 总体损失函数由姿态蒸馏损失函数和均方差损失函数组成，并通过系数α平衡两者。最终便可让网络结合 ground-truth 和教师模型的预测结果来进行训练。

![1](https://i.loli.net/2019/09/01/IEcROtwbu6h2orU.png)

![2](https://i.loli.net/2019/09/01/ON9eghZ4zlIYExi.png)

![3](https://i.loli.net/2019/09/01/X8kibEhQRr9zVWd.png)

关于教师网络的有效性，可作如下的解释

1. 人工标记的关节点坐标标签可能存在一定错误，教师模型能够通过学习和推理来减少一些错误，从而减少错误标记的训练样本的误导。

2. 在复杂背景和遮挡情形下，教师模型可以通过用模型推理解释这些困难点，从而降低学习难度。

3. 无标记关节点坐标可能会被教师网络标记。

### 实验

作者在MPII和LSP数据集上进行了实验，所提出的FPD 方法都取得了与最好结果可比较的精度。表1 展示了在MPII数据集上的测试结果，发现 FPD 模型size只有3M，计算复杂度也大幅降低，而且没有出现明显的模型泛化能力降低，在 LSP 数据集上也展现了同样的结果。同时作者也对 loss 函数的选择和 loss 函数中平衡系数的选择做了实验验证，确定了论文中 loss 函数的选择是最佳的。还做了实验验证了 knowledge distillation 的方法对最终精度有提升。这部分的实验不算复杂，在此就不作过多展示。

<center>表1 PCKh@0.5and AUC (%) rates on the MPII test dataset</center>

![表1 PCKh@0.5and AUC (%) rates on the MPII test dataset](https://i.loli.net/2019/09/01/hOjPxC2GJ4qWrNi.png)

### 总结

本文提出了一种新的快速姿态提取学习策略。与大多数现有的人体姿态估计方法相比，FPD旨在解决研究不足和实际意义重大的模型成本效益质量问题，以便将人体姿态估计模型扩展到现实中的大规模部署。

总体来说，论文的出发点确实是很有意义且很有必要，现在的深度学习方法确实大幅提高了很多计算机视觉任务中的精度，但同时模型也越来越复杂，导致了不能很好的应用到轻量级的设备上，使理论研究与工业应用不能很好的结合。轻量快速高效的网络也会一直是一个值得研究的方向。

论文中 knowledge distillation 的概念听着很高端，但实际应用并不复杂。而文中设计的新的 loss 函数也只是简单的将 teacher network 的输出作为 label 计算 loss，与传统的姿态估计的 loss 计算方法没有差别，将两个 loss 相结合以及研究平衡系数还是值得借鉴，总的来说这个损失函数的设计其实是很自然的。

文章中 teacher network 选用 Hourglass 网络，student network模型的构造也很直接，直接将Hourglass模型缩小一倍，将8个stage 缩小为4个, conv层通道数从256减小到128。虽然实验中对不同 Stage 数和 Channel 数的模型都做了对比试验分析，但如果考虑其他网络结构的小模型作为 student network ，而不仅限于 Hourglass 网络，不知道效果会不会有提升。还有论文一直强调参数少消耗低，这仅针对最终的 student network， 但是整个流程还是需要先训练一个复杂的 teacher network，而这部分的资源消耗和网络复杂度作者却未提及。

---

## Multi-Person Pose Estimation with Enhanced Channel-wise and Spatial Information

本文由字节跳动和东南大学组队完成，提出了基于增强通道和空间信息的人体姿态估计网络，获得了今年 Look Into Person (LIP) 竞赛 Single-Person Human Pose Estimation 挑战赛的冠军。

### 介绍

多人姿态估计即定位图像中的所有人的身体部位，例如手臂、躯干和面部的关键点。然而在计算机视觉中，光照影响、尺度差异、遮挡等问题的存在，使得很难获得准确的人体关键点定位结果。一般来说，具有较大感受野的高层特征图可以用来推断不可见和遮挡的关键点，而低层的高分辨率的特征图对适应尺度变化推导小尺度的关键点十分重要，因此两者之间的平衡就很重要了。如何结合并利用这两部分特征也是一个难点。基于此，本文作者提出了一个 Channel Shuffle Module (CSM) 来进一步增强所有比例的特征图之间的跨通道通信。此外，还设计了一个 Spatial＆Channel-wise Attention Residual Bottleneck (SCARB)，用于在空间和通道中自适应地增强融合的特征图信息。

### 贡献

本文的主要贡献可以总结为以下三点：

1. 提出了 Channel Shuffle Module (CSM), 可以增强低层和高层特征图之间的跨通道信息通信。

2. 提出了 Spatial, Channel-wise Attention Residual Bottleneck (SCARB), 可以在空间和通道上自适应地增强融合金字塔特征信息。

3. 此方法在 COCO 关键点基准上达到了 state-of-the-art 的结果。

### 方法

#### CSM: Channel Shuffle Module

对于多人姿态估计，低级和高级特征图之间的折衷仍然存在限制。不同层间具有不同特征的通道信息可以相互补充和加强。基于此，本论文提出了 Channel Shuffle Module 来进一步重新校准低级和高级特征图之间的相互关系。如图1所示。对原始提取出的多个特征块进行1*1卷积，变换到相同的通道数，再对低分辨率的特征进行上采样到相同的分辨率，这些特征被连接在一起后执行信道混洗操作，以融合不同级别之间的互补通道信息。最后将混洗的特征分开，并分别下采样到原始分辨率。

![图1 Channel Shuffle Module](https://i.loli.net/2019/09/01/7aV9lIoOPd6W3r2.png)

<center>图1 Channel Shuffle Module</center>

#### ARB: Attention Residual Bottleneck

特征融合的过程是动态多变的，融合后的特征往往也存在许多冗余。网络中那些对姿态估计任务更加有效的特征信息应该被自适应地突显出来。因此，论文提出基于空间和通道注意力机制的残差模块，自适应地从空间和通道两个维度增强刚刚融合后的金字塔特征信息。如图2所示，本文最后采用了 Spatial, Channel-wise Attention Residual Bottleneck (SCARB)。

![图2 Attention Residual Bottleneck](https://i.loli.net/2019/09/01/oeDpRHkQIwSx5Jv.png)

<center>图2 Attention Residual Bottleneck</center>

#### 整体网络结构

本文以 Cascaded Pyramid Network (CPN) 作为基础网络结构，探索Channel Shuffle Module (CSM) 和Spatial, Channel-wise Attention Residual Bottleneck (SCARB) 对多人姿势估计的影响。具体网络结构如图3所示。

![图3 整体网络结构](https://i.loli.net/2019/09/01/r3eHl4LYx7n5pOw.png)

<center>图3 整体网络结构</center>

### 实验

本文做了大量的实验来探究最佳的网络结构和参数，如Channel Shuffle Module (CSM) 中进行 Shuffle 操作组数的确定，还有在添加注意力机制时是先 Spatial 后 Channel-wise Attention还是先Channel-wise 后 Spatial Attention顺序的确定，以及单独增加 CSM、SCARB 模块还是两者共同作用，都进行了实验探索最佳的网络结构，才得到了图1.3中最终的网络设计。

<center>表1 COCO测试开发数据集最终结果的比较</center>

![表1 COCO测试开发数据集最终结果的比较](https://i.loli.net/2019/09/01/aQN5FV9ztipIlfW.png)

最后论文将提出的模型与COCO测试数据集上的其他最先进的方法进行比较。结果如表1所示，超越了先前所有的方法，在COCO关键点基准上实现了 state-of-the-art 的性能。

### 总结

本篇论文的主要贡献在于对CPN 网络加入 Channel Shuffle 操作和引入注意力机制，在COCO关键点基准上实现了 state-of-the-art 的性能。虽然这两个模块的操作都不算作者的原创，Channel Shuffle 的操作思想源于 ShuffleNet，注意力机制在很多领域的应用也很普遍。但本文的工作还是有很多值得借鉴的地方，在Channel Shuffle Module 的操作中，将 ResNet 中提取的金字塔特征进行上采样到相同的分辨率，再进行 Channel Shuffle 操作，最后分别下采样到原始分辨率，并将最终混洗得到的特征图与原始金字塔特图连接起来，最终获得增强的金字塔特征， 这些增强的金字塔特征图不仅包含来自原始金字塔特征的信息，还包含来自混叠金字塔特征图的融合的跨通道信息，这个操作对于那些以 ResNet 作为 Backbone 的网络都有参考意义。同时注意力机制也是一个很值得尝试的操作，在很多任务中，注意力机制都被证明对性能提升有一定帮助。

本论文给人最终的感觉就是简单高效，通过并不复杂的操作实现高性能，而且本文通过大量实验来确定网络参数的最优选择，实验考虑也很全面周到，有足够的说服力。

---

## Deep High-Resolution Representation Learning for Human Pose Estimation

本文由中国科学技术大学和微软亚洲研究院视觉计算组完成，提出高分辨率深度神经网络（HRNet），与上篇论文 Multi-Person Pose Estimation with Enhanced Channel-wise and Spatial Information 并列为今年 Look Into Person (LIP) 竞赛 Single-Person Human Pose Estimation 挑战赛的冠军。

### 介绍

对于视觉识别中的区域层次和像素层次问题，分类网络（如ResNet、VGGNet等）学到的表征分辨率比较低，在此基础上恢复的高分辨率表征空间区分度仍然不够强，使其在对空间精度敏感的任务上很难取得准确的预测结果。本文提出高分辨率深度神经网络（HRNet），对网络结构做了基础性的改变，由传统的串行连接高低分辨率卷积，改成并行连接高低分辨率卷积，通过全程保持高分辨率和对高低分辨率表征的多次信息交换来学到丰富的高分辨率表征，在多个数据集的人体姿态估计任务中取得了最佳的性能。

### 贡献

本文的主要贡献可以总结为以下两点：

1. 提出将高分辨率到低分辨率的子网络并联，而不是像大多数现有的串联解决方案，从而此方法能够保持高分辨率，而不是通过从低到高的过程恢复分辨率，因此预测的热图在空间上可能更精确。
2. 大多数现有的融合方案集合了低级和高级表示。而本文执行重复的多尺度融合并借助于相同深度和相似级别的低分辨率表示来增强高分辨率表示，从而高分辨率表示对于姿态估计也提供了丰富的信息。

### 方法

#### 高分辨率网络

HRNet 的结构如图1所示，HRNet整个网络中始终保持高分辨率表征，逐步引入低分辨率卷积，并且将不同分辨率的卷积并行连接。同时，通过不断在多分辨率表征之间进行信息交换，来提升高分辨率和低分辨率表征的表达能力，让多分辨率表征之间更好地相互促进。HRNet 将高低分辨率并行连接是较大的创新。

![图1 HRNet 结构图](https://i.loli.net/2019/09/02/uQs8Cf23zLIRWeB.png)

<center>图1 HRNet 结构图</center>

![图2 多分辨率信息交换](https://i.loli.net/2019/09/02/kojQit8Y4wCXVxl.png)

<center>图2 多分辨率信息交换</center>

#### 重复多尺度融合

论文在并行子网中引入交换单元，这样每个子网可以重复接收来自其他并行子网的信息。多分辨率表征信息交换如图2所示，每一个分辨率的输出表征都会融合三个分辨率输入的表征，以保证信息的充分利用和交互。将高分辨率特征降到低分辨率特征时，采用stride为2的3x3卷积；低分辨率特征到高分辨率特征时，先利用1x1卷积进行通道数的匹配，再利用最近邻插值的方式来提高分辨率。相同分辨率的表征则采用恒等映射的形式。

### 实验

#### 标准数据集性能

本文在 COCO Keypoint Detection、MPII Human Pose Estimation 和 Pose Tracking 三个数据集上均做了大量的实验验证 HRNet 的性能。如下表1 展示了在 COCO test-dev上与最先进方法的性能比较，可以看到小模型HRNet-W32和大模型HRNet-W48都取得了性能提升，在引入额外数据的情况下，大模型展现了更强的表达能力，有更显著的提升。在其他数据集上也展示了很好的性能，在此不做展示了。

<center>表1 Comparisons on the COCO validation set</center>

![表1 Comparisons on the COCO validation set](https://i.loli.net/2019/09/02/pMTv6KVlfeUzHZ9.png)



#### 重复多尺度融合

本文分析了重复多尺度融合的效果，研究了网络的三种变体，如表2所示。验证集的结果表明，多分辨率表征信息交换可以将不同分辨率的表征信息进行充分的交换利用，对表征增强的作用十分明显，可以到达2.6% AP的提升。

<center>表2 多尺度融合研究</center>

![表2 多尺度融合研究](https://i.loli.net/2019/09/02/ktnzWVK7Igr6YDA.png)

#### 表征分辨率

本文研究了表征分辨率如何影响姿态估计性能：从高到低检查从每个分辨率的特征映射估计的热图质量，从图3 可以清楚地看到，网络输出表征的分辨率降低会使得模型的性能有巨大的损失。这体现了表征分辨率对于空间精度的重要性。

![图3 表征分辨率研究](https://i.loli.net/2019/09/02/CNku9sL4vUjKaZ6.png)

<center>图3 表征分辨率研究</center>

### 总结

视觉识别主要包括三大类问题：图像层次（图像分类），区域层次（目标检测）和像素层次（图像分割、人体姿态估计等）。用于图像分类的卷积神经网络成为解决视觉识别问题的标准结构。这类网络的特点是学到的表征在空间分辨率上逐渐变小。但分类网络并不适合区域层次和像素层次的问题，因为学到的表征本质上具有低分辨率的特点，在分辨率上的巨大损失使得其在对空间精度敏感的任务上很难取得准确的预测结果。

为了弥补空间精度的损失，在分类卷积神经网络结构的基础上，通过引入上采样操作和/或组合空洞卷积减少降采样次数来提升表征的分辨率，典型的结构包括Hourglass、U-Net等。在这类网络结构中，最终的高分辨表征主要来源于两个部分：第一是原本的高分辨率表征，但是由于只经过了少量的卷积操作，其本身只能提供低层次的语义表达；第二是低分辨率表征通过上采样得到的高分辨率表征，其本身虽然拥有很好的语义表达能力，但是上采样本身并不能完整地弥补空间分辨率的损失。所以，最终输出的高分辨率表征所具有的空间敏感度并不高，很大程度上受限于语义表达力强的表征所对应的分辨率。

本论文改变了现有的基于分类网络的人体姿态估计的网络结构，提出了高分辨率深度神经网络（HRNet）。整个过程保持高分辨率，不需要从低分辨率恢复高分辨率，以及多次对高低分辨率表征进行信息补足，成功学到足够丰富的高分辨率表征。

HRNet 不仅对人体姿态估计任务有着很好的性能表现，目前 HRNet 在图像分割、目标检测、图像分类等等任务上都展现了很好的性能，是大家可以去尝试的一个方法。具体可以参考 [GitHub HRNet](https://github.com/HRNet) 页面 。最近，本文作者又放出新论文 Bottom-up Higher-Resolution Networks for Multi-Person Pose Estimation，是 HRNet 的升级版。

新论文中作者采用自下而上的方法进行多人姿态估计。典型的自下而上流程包括热图预测和关键点分组两步。作者主要关注提高热图预测精度。提出了 HigherHRNet。通过对HRNet输出的高分辨率特征图进行解卷积来生成高分辨率特征图，这对于中小尺寸人体姿态估计来说在空间上更加精确。然后构建高质量的多层次特征并进行多尺度姿态预测。HigherHRNet 性能超越了 COCO 数据集上所有现有的自下而上的方法。

---

## Does Learning Specific Features for Related Parts Help Human Pose Estimation?

本文是西北大学提出的在人体姿态估计任务中，学习相关部位的特定特征能提高人体姿态估计的精度。

### 介绍

人体姿态估计(HPE)本质上是一个同质多任务学习问题，每个身体部位的定位是一个不同的任务。最近的HPE方法普遍学习所有身体部位的共享表示，从该表示中它们的位置被线性回归。统计分析表明，并非所有部分都相互关联。结果，这种共享机制会导致负迁移并降低性能。基于此，本文提出一个有趣的问题。能否通过识别相关部位并学习它们的具体特征来提高姿态估计吗？由于不相关的任务不再共享高层次的表达，希望能避免负迁移的反作用。

本文首先提出了一种数据驱动的方法，根据相关部位共享的信息量对它们进行分组。然后引入基于部件的分支网络(PBN)来学习每个部位组特有的表示。再进一步提出了该网络的多阶段版本，以反复改进中间特征和姿态估计。实验表明，学习特定特征显著改善了遮挡部位的定位，从而有利于HPE。在两个基准数据集上达到了最先进的方法，在发生遮挡时具有突出的优势。

### 贡献

本文的贡献总结为以下几点：

1. 本文是第一个发现所有身体部位的特征应该完全共享是存在问题的，并通过简单有效的基于部位的分支网络来解决这个问题

2. 本文首次尝试利用身体部位位置的概率分布及其相互信息来对相关部位进行分组，并证明比基于人体结构的方法更有效

3. 本文的模型在定位遮挡部位方面具有突出的优势

### 方法

论文首先介绍两种识别相关身体部位的策略。然后提出一个基于部分的分支网络来学习它们的特定特征。最后，提出了该网络的多阶段版本，以重复地改进中间特征和身体部位定位。

#### 相关身体部位

**基于人体身体结构**

识别相关部位最直接的方法是利用人体结构，直觉上，自然界中相连的部分是相关的，如图1所示，展示了相关部位的分组。

![图1 基于身体结构分组](https://i.loli.net/2019/09/04/Twm4fk3dQbxOovL.png)

<center>图1 基于身体结构分组</center>

**基于数据驱动**

将每个部位的位置视为随机变量，由此计算两者之间的相关性，公式如下：

![1](https://i.loli.net/2019/09/04/jSW4eTxFwnEKfC9.png)

I值越高表示与部位m密切相关的特征也为部位n提供了信息线索，反之亦然。

#### 基于部分的分支网络(PBN)

基于部分的分支网络(PBN)是一个 CNN 网络结构，由两阶段组成，一个学习对所有身体部分通用的共享表示的阶段，另一个学习对每组相关部分专用的高级特征的阶段，整体结构如图2所示。

![图2 PBN 网络结构](https://i.loli.net/2019/09/04/9SakNlOFY8wTbcu.png)

<center>图2 PBN 网络结构</center>

### 实验

#### 基准数据集结果

论文在MPII和LSP数据集上进行了实验，均取得了较好的结果，具体实验结果如下表1、2、3所示：

![表1 Comparisons of PCKh@0.5 scores on the invisible parts in the MPII validation set](https://i.loli.net/2019/09/04/nVqh6ugxeoQ71D3.png)

<center>表1 Comparisons of PCKh@0.5 scores on the invisible parts in the MPII validation set</center>

![表2 Comparisons of PCK@0.2 scores on the LSP testing set](https://i.loli.net/2019/09/04/bgBi9eNcC4hGksj.png)

<center>表2 Comparisons of PCK@0.2 scores on the LSP testing set</center>

![表3 Comparisons of PCK@0.2 scores on the corrected LSP testing set](https://i.loli.net/2019/09/04/xUuLWRMQ1rjdlN4.png)

<center>表3 Comparisons of PCK@0.2 scores on the corrected LSP testing set</center>

#### Ablation study

论文还做了大量实验，分析了特定的feature层的深度和宽度选取多少合适，讨论了身体部位分组的策略以及分组中存在部位重叠的效果等等大量的实验内容，更多的实验内容可以参考论文。

![Ablation study using variants of three-stack PBNs](https://i.loli.net/2019/09/04/SreOqp2TU6gBi35.png)

<center>Ablation study using variants of three-stack PBNs</center>

### 总结

论文通过大量的基准数据集实验和消融研究，得出结论，学习相关身体部位的特定特征可以显著改善遮挡部位的定位，从而有利于人体姿态估计。

似乎这是一个很自然的想法，但先前的工作的都是对各个部位共享参数来定位的，但论文中做　Ablation study　时，证明对不同身体部位分组有重叠的情况下，反而对最后结果没有提升帮助，个人认为此部分可以再去做更多实验来验证。而且多个分组得到的结果直接取均值来算最后结果并不是一个可靠的方法，也许不同的部位的权重也需要考虑，注意力机制也许是一个很好的方法，可以应用到本文中。

---

## Fast and Robust Multi-Person 3D Pose Estimation from Multiple Views

本论文由浙大CAD实验室三维视觉研究组提出，完成了快速且鲁棒的多视角下多人三维姿态估计，该方法比之前的最先进的方法在相应的基准数据集上效果有非常明显的提升。

### 介绍

三维人体姿态估计通常分两个阶段：第一阶段在单独的2D视图中检测人体关键点，这些关键点在第二阶段被聚集以重建3D姿态。目前基于深度学习的2D关键点检测技术已经取得了显著的性能，而找到检测到的关键点之间的交叉视图对应关系，来完成三维姿态估计还是一个很有挑战性的任务。

本文提出了一种新的多人三维姿态估计方法。通过在多个视图中匹配检测到的2D姿态，从而在单个人的层面解决对应问题，产生2D姿态的聚类，其中每个聚类包括同一个人在不同视图中的2D姿态。然后，可以从匹配的2D姿态中分别推断出每个人的3D姿态，大幅减少了状态空间，能快速的完成3D姿态估计。

### 贡献

本文的主要贡献和创新可以总结为以下三点：

1. 提出了一种快速鲁棒多人三维姿态估计的新方法。将在所有视图中属于同一个人的检测到的2D姿态分组，可以极大地减少状态空间，从而提高3D姿态估计的效率和鲁棒性。

2. 提出了一种多向匹配算法，用于在多个视图中找到检测到的2D姿态的一致的对应关系。

3. 将几何和外观信息结合起来，以匹配不同视图中检测到的2D姿态，而以前的方法大多忽略了外观信息。

### 方法

#### 整体框架

如图1展示了整个工作流程。首先，采用现成的2D人体姿态检测器 (CPN)来产生每个视图中人的边界框和2D关键点位置。针对噪声2D检测，提出了一种多向匹配算法来建立检测到的边界框在视图间的对应关系，消除误检测。最后，使用3DS模型从相应的2D边界框和关键点为每个人重建3D姿态。

![图1 整体网络框架](https://i.loli.net/2019/09/03/jYPlAkCavTu8ecy.png)

<center>图1 整体网络框架</center>

#### 多视图对应

重建3D姿态之前，检测到的2D姿态应该在视图之间匹配，即在所有视图中找到属于同一个人的边界框。

论文中结合外观相似性和几何兼容性来计算边界框之间的相似性分数，基于此计算得到相似性矩阵。外观相似性的衡量用一个预训练好的re-ID network，前面网络输出的 bounding box 作为输入，re-ID network得到一个feature，计算两个视角得到的 feature 之间的欧氏距离，映射到0到1之间，得到了外观的相似性。几何信息衡量相似性，使用极线约束来测量。

循环一致性的多路匹配，如图4.2所示，绿色的bounding box在每个视角中都是匹配对的，会形成一个闭环。而红色的bounding box在这几个视角中存在配错了，形成不了闭环。本文利用这个信息保证一个全局的连续结果。最后通过给定同一个人在不同视图中的估计2D姿态完成3D姿态重建。

![图2 循环一致性解释](https://i.loli.net/2019/09/03/oM6nsqabJ4rDSN7.png)

<center>图2 循环一致性解释</center>

### 实验

论文在Campus 和Shelf数据集上进行了实验验证。Campus 数据集只有三个相机视角的数据集，Shelf 数据集有五个相机视角。表1 展示了论文提出的相似性计算和多视角匹配能有效提升性能。表2 展示了论文方法与目前state-of-the-art 的方法对比，获得了最好的性能表现。同时论文也分析了运行时间，它能够以大于20 fps的速度实现实时性能。

<center>表1 Ablative study on the Campus and Shelf datasets</center>

![表1 Ablative study on the Campus and Shelf datasets](https://i.loli.net/2019/09/03/2v4xQmA3JyCIdc7.png)

<center>表2 Quantitative comparison on the Campus and Shelf datasets</center>

![表2 Quantitative comparison on the Campus and Shelf datasets](https://i.loli.net/2019/09/03/kxitZVnsMHdg28I.png)

### 总结

本论文提出了一种新的多视角三维姿态估计方法，该方法可以快速、鲁棒地恢复多人的三维姿态。本文的核心思想是使用一种多向匹配算法对检测到的2D姿态进行聚类，以减少3DS模型的状态空间，从而提高效率和鲁棒性。

三维姿态估计是建立在二维姿态估计得到的二维姿态的基础上的，相比二维姿态估计，三维姿态估计会涉及到更多空间信息计算和更多数学知识，也会更加复杂。本文中采用的匹配2D姿态来降低状态空间的方法，也给其他任务一定的启发，类似于二维姿态估计中Top-down 和 Bottom-up 方法的差异，也就是先利用目标检测算法检测出单个人再对进行关键点的检测还是先检测关键点再匹配到单人，不同的思路对精度和处理速度都有着不一样的影响。

---

> 20190905 汇总