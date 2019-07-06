---
title: '[论文总结] 理解FishNet'
tags:
  - Deep Learning
  - Computer Vision
  - Reviews
date: 2019-04-21 15:50:36
---

<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML' async></script>

从VALSE2019回来后，感觉自己俨然变成了欧阳万里老师的脑残粉呀╰( ᐖ╰)！会上欧阳老师介绍的FishNet简直让我眼前一亮，这么好的点子，我怎么就没想到呐！回来好好读了一下文章和代码，简单总结一下。

<div align="center" class="figure">
  <img src="/images/fishnet/fish.jpg" width="15%" alt="我咸甚，此鱼何能及也"/>

</div>

<!-- more -->

## 0 问题

### 0.1 各层特征的融合方式
较早的典型深度CNN结构大多为漏斗状，不断地进行卷积、下采样来提取、浓缩图像特征，最后用一些全连接层之类的结构来计算具体任务的输出结果。这样的设计很自然地被用于图像分类任务，因为较深的神经网络更能学习高级语义特征，最后将图像浓缩到一个像素而变成一个向量时，这个像素的每一个通道的值则代表了整个图像在这个语义特征上的表现。
<div align="center" class="figure"><img src="/images/fishnet/vgg.png" width="40%" alt="VGG-16">
Fig. 1 漏斗状卷积神经网络，以VGG-16为例

</div>

但是呢，这样的结构原封不动地应用到其他任务上，效果就不是很好了。比如在分割任务中，细节特征保留得好的话，分割的效果则会更佳（例如FCN-8s的效果远好于FCN-32s）。又如在anchor-based目标检测模型中，用尺寸更大的特征图能够更好地回归较小目标的候选框（例如YOLOv3加入FPN后显著提升小物体的检测效果）。因此，出现了一些沙漏状甚至多沙漏堆叠的网络结构（U-Net，FPN，Stacked Hourglass等等）来更好地处理这些任务。
<div align="center" class="figure"><img src="/images/fishnet/unet.png" width="40%" alt="U-Net">
    Fig. 2 沙漏状卷积神经网络，以U-Net为例

</div>

可以看到，类似这样的工作大多出于这样的一个想法：底层细节特征很重要，我们要把它融合到顶层语义特征里去。这样就有人问了：那语义特征是不是也能融合到细节特征里去，从而增强高分辨率特征图的效果呢？FishNet就做到了这样的融合，让网络最后一部分的各个分辨率的特征图中的底层、中层、顶层特征（作者原话为pixel-level, region-level, image-level）都能“你中有我，我中有你”。

### 0.2 梯度反向传播的阻碍
在ResNet中，作者用一种巧妙的办法让较浅的层也能得到有效的梯度信息——在每层层的输出上加一个identity mapping。也就是该层的输入\\(x_l\\)、下一层的输入\\(x_{l+1}\\)以及本层的运算\\(\mathcal{F}\(x, \mathcal{W}_l\)\\)之间的关系是$$x_{l+1}=x_l+\mathcal{F}(x_l, \mathcal{W_l})$$
再下一层的话：
$$x_{l+2}=x_l + \mathcal{F}(x_l, \mathcal{W_l}) + \mathcal{F}(x_{l+1}, \mathcal{W_{l+1}})$$
要是一直写到最后一层\\(x_L\\)：
$$x_{L}=x_l+\sum_{i=l}^{L-1}\mathcal{F}(x_i, \mathcal{W_i})$$
那么梯度反传时则有：
$$\begin{split}
\frac{\partial{\mathcal{E}}}{\partial{x_l}} & = \frac{\partial{\mathcal{E}}}{\partial{x_L}}\frac{\partial{x_L}}{\partial{x_l}}\\\\
& = \frac{\partial{\mathcal{E}}}{\partial{x_L}}\Big(1+\frac{\partial{}}{\partial{x_l}}\sum_{i=l}^{L-1}\mathcal{F}(x_i,\mathcal{W}_i)\Big)
\end{split}$$

然而现实是：因为中间涉及了几次下采样，采样后的特征图尺寸发生了变化，这时，那个恒等映射\\(x\\)上不得不加一个\\(\mathcal{M}(x)\\)（一般为一个\\((1\times 1)\\)尺寸的卷积，作者称之为I-conv，即Isolated convolution）来改变尺寸和通道数。因此，不是每一层都能保证简单的\\(x_{l+1}=x_l+\mathcal{F}(x_l, \mathcal{W_l})\\)，上边的梯度公式也只是一种理想情况而已。
<div align="center" class="figure"><img src="/images/fishnet/bottleneck_alter.png" width="50%" alt="Bottlenecks in ResNet">
    Fig. 3 ResNet中，理想的Bottleneck模块与现实中某些Bottleneck模块

</div>

在ResNet本身里面倒还好。到了FPN甚至Stacked Hourglass中，这样的I-conv在每次特征图融合时都被使用，这就有点违背ResNet保持梯度有效反传的初衷了。而FishNet在这种情况下采用了一种更“平滑”的方式使得梯度反传受到的影响降到最低。
<div align="center" class="figure"><img src="/images/fishnet/fish_block.png" width="36%" alt="Bottlenecks in FishNet">
    Fig. 4 FishNet中涉及采样的Bottleneck模块（除tail部分外）

</div>


## 1 整体方案
妙啊（👏）！那我们就来看一眼FishNet的全貌：
<div align="center" class="figure"><img src="/images/fishnet/fishnet.png" width="70%" alt="FishNet">
    Fig. 5 FishNet

</div>

<!--这才是真正的Fishnet！（斜眼
<div align="center" class="figure">
<img src="/images/fishnet/real_fishnet.png" width="40%" alt="">
</div>
-->

<s>整条鱼</s>整个FishNet由三部分构成：tail（尾巴），body（躯干）和head（头）。tail部分之前，图像先过了三层卷积层，初步从\\(\(224\times 225 \times 3\)\\)尺寸的原图像提取出\\(\(56\times 56 \times 64\)\\)尺寸的特征图。作者把不同阶段内同一分辨率的特征图分为同一个stage，\\(\(56\times 56\)\\)的是stage 1，\\(\(28\times 28\)\\)的是stage 2，\\(\(14\times 14\)\\)的是stage 3，\\(\(7\times 7\)\\)的是stage 4。因为分辨率相同，三个部分的特征图可以不用上/下采样而直接在channel维度上concat起来。

tail部分就是一个漏斗状的网络，涉及三次最大池化，每次池化前，最后一个卷积层输出的特征图被留下来供body部分使用。这一部分的结果就是经典的漏斗状网络，作者使用的是一个三阶段的ResNet。tail部分的最后，作者用了一个Squeeze-Excitation模块\[2\]，先把\\(\(7\times 7 \times 512\)\\)尺寸的特征图用Global Average Pooling再加几个卷积层（实际上和全连接层并无本质区别）映射成一个\\(\(1\times 1\times 512\)\\)的向量，再把这个向量的每一个值作为一个权重，乘到之前\\(\(7\times 7\times 512\)\\)的特征图对应的通道上去。

body部分像FPN一样不断地用上采样来放大特征图，同时融合之前tail部分保留下来的同一分辨率的特征。

head部分则是FishNet的独创性工作，它像是body部分的反过程。以往的沙漏形网络将高层语义特征用来精化低层细节特征，而head网络反其道而行之，又用精化过的低层细节特征反过来精化高层特征。这样，再次采样得到的高层特征的质量被有效提高。

## 2 实现细节
### 2.1 网络参数
FishNet-99整体的各个部分的参数见下表。

| Part-Stage |       Input shape      |      Output shape      | Bottlenecks | I-convs | Convs in total|
|:----------:|:----------------------:|:----------------------:|:-----------:|:-------:|:-------------:|
|   Input    | \\(3\times 224 \times 224\\) | \\(64\times 56 \times 56\\)  | \\(0\\) | \\(0\\) | \\(3\\) |
|   Tail-1   | \\(64\times 56 \times 56\\)  | \\(128\times 28 \times 28\\) | \\(2\\) | \\(1\\) | \\(7\\) |
|   Tail-2   | \\(128\times 28 \times 28\\) | \\(256\times 14 \times 14\\) | \\(2\\) | \\(1\\) | \\(7\\) |
|   Tail-3   | \\(256\times 14 \times 14\\) | \\(512\times 7 \times 7\\)   | \\(6\\) | \\(1\\) |\\(19\\) |
|  SE-block  | \\(512\times 7 \times 7\\)   | \\(512\times 7 \times 7\\)   | \\(2\\) | \\(1\\) |\\(11\\) |
|   Body-3   | \\(512\times 7 \times 7\\)   | \\(256\times 14 \times 14\\) | \\(1 + 1\\) | \\(0\\) | \\(6\\) |
|   Body-2   | \\(\(512+256\)\times 14 \times 14\\) | \\(384\times 28 \times 28\\) | \\(1 + 1\\) | \\(0\\) | \\(6\\) |
|   Body-1   | \\(\(384+128\)\times 28 \times 28\\) | \\(256\times 56 \times 56\\) | \\(1 + 1\\) | \\(0\\) | \\(6\\) |
|   Head-1   | \\(\(256+64\)\times 56 \times 56\\)  | \\(320\times 28 \times 28\\) | \\(1 + 1\\) | \\(0\\) | \\(6\\) |
|   Head-2   | \\(\(320+512\)\times 28 \times 28\\) | \\(832\times 14 \times 14\\) | \\(2 + 1\\) | \\(0\\) | \\(9\\) |
|   Head-3   | \\(\(832+768\)\times 14 \times 14\\) | \\(1600\times 7 \times 7\\)  | \\(2 + 4\\) | \\(0\\) | \\(18\\) |
| Score-Conv | \\(\(1600+512\)\times 7 \times 7\\)| \\(1056\times 7 \times 7\\)  | \\(0\\) | \\(0\\) | \\(1\\) |
|  Score-FC  | \\(1056\times 7 \times 7\\)| \\(1000\times 1 \times 1\\)  | \\(0\\) | \\(0\\) | \\(1\\) |

说明：
  + 第一列的Tail-1代表Tail部分的stage \\(1\\)。
  + Body-3至Head-3的Bottleneck模块数量包括两种：网络主干上的和特征图迁移模块上的。迁移模块用于将上一部分同一stage的特征图进行变换。

FishNet-150的参数见下表，与FishNet-99相比而言只是各个部分Bottleneck块的数量不同，没有太大差异。

| Part-Stage |       Input shape      |      Output shape      | Bottlenecks | I-convs | Convs in total|
|:----------:|:----------------------:|:----------------------:|:-----------:|:-------:|:-------------:|
|   Input    | \\(3\times 224 \times 224\\) | \\(64\times 56 \times 56\\)  | \\(0\\) | \\(0\\) | \\(3\\) |
|   Tail-1   | \\(64\times 56 \times 56\\)  | \\(128\times 28 \times 28\\) | \\(2\\) | \\(1\\) | \\(7\\) |
|   Tail-2   | \\(128\times 28 \times 28\\) | \\(256\times 14 \times 14\\) | \\(4\\) | \\(1\\) |\\(13\\) |
|   Tail-3   | \\(256\times 14 \times 14\\) | \\(512\times 7 \times 7\\)   | \\(8\\) | \\(1\\) |\\(25\\) |
|  SE-block  | \\(512\times 7 \times 7\\)   | \\(512\times 7 \times 7\\)   | \\(4\\) | \\(1\\) |\\(17\\) |
|   Body-3   | \\(512\times 7 \times 7\\)   | \\(256\times 14 \times 14\\) | \\(2 + 2\\) | \\(0\\) |\\(12\\) |
|   Body-2   | \\(\(512+256\)\times 14 \times 14\\) | \\(384\times 28 \times 28\\) | \\(2 + 2\\) | \\(0\\) | \\(12\\) |
|   Body-1   | \\(\(384+128\)\times 28 \times 28\\) | \\(256\times 56 \times 56\\) | \\(2 + 2\\) | \\(0\\) | \\(12\\) |
|   Head-1   | \\(\(256+64\)\times 56 \times 56\\)  | \\(320\times 28 \times 28\\) | \\(2 + 2\\) | \\(0\\) | \\(12\\) |
|   Head-2   | \\(\(320+512\)\times 28 \times 28\\) | \\(832\times 14 \times 14\\) | \\(2 + 2\\) | \\(0\\) | \\(12\\) |
|   Head-3   | \\(\(832+768\)\times 14 \times 14\\) | \\(1600\times 7 \times 7\\)  | \\(4 + 4\\) | \\(0\\) | \\(24\\) |
| Score-Conv | \\(\(1600+512\)\times 7 \times 7\\)| \\(1056\times 7 \times 7\\)  | \\(0\\) | \\(0\\) | \\(1\\) |
|  Score-FC  | \\(1056\times 7 \times 7\\)| \\(1000\times 1 \times 1\\)  | \\(0\\) | \\(0\\) | \\(1\\) |

tail，body和head三部分的主要成分都是Bottleneck模块，即下表所示的结构：

|    Layer  |          Type          |     Output channels    |   Kernel Size   |
|:---------:|:----------------------:|:----------------------:|:---------------:|
|(shortcut) | (take shortcut)        |           -            |        -        |
|   relu    | ReLU                   |           \\(C\\)            |        -        |
|    bn1    | Batch Normalization    |           \\(C\\)            |        -        |
|   conv1   | Convolution            |        \\(C / 4\\)           | \\(1\times 1\\) |
|    bn2    | Batch Normalization    |        \\(C / 4\\)           |        -        |
|   conv2   | Convolution            |        \\(C / 4\\)           | \\(3\times 3\\) |
|    bn3    | Batch Normalization    |        \\(C / 4\\)           |        -        |
|   conv3   | Convolution            |          \\(C'\\)            | \\(1\times 1\\) |
|(addition) | (add shortcut)         |          \\(C'\\)            |        -        |

在tail部分的每一个stage中，第一个Bottleneck模块会涉及通道数的变化（即\\(C'\neq C\\)）。这时shortcut需要经过一个卷积层来变换identity mapping的通道数。因此，这三个shortcut上依旧无法避免使用Isolated convolution。在SE-block中也存在类似的情况。而在head部分中，尽管特征图仍在不断地下采样，其通道数并没有被改变，所以不需要使用这样的Isolated convolution来干扰梯度的直接反传（direct back-propagation）。

（PS：可是我数了数，FishNet-99里有100个卷积，FishNet-150里有151个卷积呀😂？个人猜测是因为Score-FC层不应该算在FishNet主干内？对了，虽然它叫做FC层，但作者代码里还是用卷积层的形式定义的哦。因为\\(7\times 7\\)尺寸的特征图过了一层Global Average Pooling变成了\\(1\times 1\\)尺寸，所以它本质上变成了一个长度为通道数的向量。）

### 2.2 采样&精化模块
从body部分的stage 3开始直到head部分的stage3，每个stage的特征图将与之前部分的特征图融合（也就是图中的红色虚线和红框所表示的内容）。为了保证梯度直接反传，作者设计了UR-block (Upsampling & Refinement) 和DR-block (Downsampling & Refinement) 来“保持和精化”（preserve and refine）各个部分的特征。

#### 2.2.1 上采样&精化（UR）模块
上边提到，FishNet中的stage号不是从浅到深依次增大的，而是与特征图的尺度相对应。设tail部分和body部分的stage \\(s\\)的**第一层**输出特征分别为\\(x^t_s\\)和\\(x^b_s\\)，则\\(x^t_s\\)和\\(x^b_s\\)的宽度和高度应该是一致的（尽管通道数可能不同）。\\(x^t_s\\)经过一个迁移模块\\(\mathcal{T}(x)\\)（transferring block，同样是带shortcut的Bottleneck模块）后与\\(x^b_s\\)进行连接构成融合的特征图\\(\widetilde{x}^b_s\\):
$$\widetilde{x}^b_s = concat(x^b_s, \mathcal{T}(x^t_s))$$

\\(\widetilde{x}^b_s\\)将继续作为body部分的stage \\(s\\)中后面的卷积层\\(\mathcal{M}(x)\\)的输入。同时，为了梯度的直接反传，另有一条恒等映射与\\(\mathcal{M}(\widetilde{x}^b_s)\\)相加。这里的思路与ResNet中\\(\mathcal{H}(x)=x+\mathcal{F}(x)\\)是一致的：
$$\widetilde{x}'^b_s = r(\widetilde{x}^b_s) + \mathcal{M}(\widetilde{x}^b_s)$$

在body部分的stage 1中，\\(\mathcal{M}(x)\\)的输出值通道数与\\(x\\)相同，此时\\(r(x)\\)即为\\(x\\)。而stage 2和stage 1中，由于\\(\mathcal{M}(x)\\)中通道数会产生变化（在作者代码中，通道数减半，\\(k=2\\)），所以这里的\\(r(x)\\)需要起到缩小通道数（channel-wise reduction）的作用。**还是为了梯度直接反传**，这里甚至没有使用\\((1\times 1)\\)的卷积来变换通道数，而是直接把每\\(k\\)个通道求和（element-wise summation）而压缩成一个通道。\\(\widetilde{x}’^b_s\\)再进行一下上采样就成为body部分下一个stage（即stage \\(s-1\\)）的输入了：
$$x^b_{s-1}=up(\widetilde{x}'^b_s)$$

<div align="center" class="figure"><img src="/images/fishnet/ur.png" width="20%" alt="Upsampling & Refinement blocks">
    Fig. 6 上采样&精化模块

</div>

（PS：为什么这里不用\\((1\times 1)\\)卷积，而前面tail部分要用呢？个人猜测是因为tail部分要扩大通道数而不得不用这样的方式。或许在tail部分使用与这里的\\(r(x)\\)相反的过程——通过把每个通道duplicate一下来达成通道数增加一倍的效果也能work呢？有兴趣的可以试一下。）

#### 2.2.2 下采样&精化（DR）模块
head部分的下采样&精化模块比上采样&精化模块更加简单，因为这里所有的\\(\mathcal{M}(x)\\)都不会导致通道数的变化，UR模块用于的\\(r(x)\\)也就不需要了。其他的公式与UR模块基本相同：
$$\widetilde{x}^b_s = concat(x^b_s, \mathcal{T}(x^t_s)) \\\\
\widetilde{x}'^b_s = \widetilde{x}^b_s + \mathcal{M}(\widetilde{x}^b_s) \\\\
x^b_{s+1}=down(\widetilde{x}'^b_s)$$

<div align="center" class="figure"><img src="/images/fishnet/dr.png" width="20%" alt="Downsampling & Refinement blocks">
    Fig. 7 下采样&精化模块

</div>

## 3 经验总结
### 3.1 低层特征对高层特征的加强
漏斗状卷积网络里，较浅卷积层中的特征往往是较简单、像素级的特征，而更深的卷积层中的特征由于感受域较大，是更抽象、泛化的特征。由于FishNet中上采样、下采样的存在，直接以“浅层”“深层”特征来区分不同分辨率的特征似乎并不妥当。因此，这里我用“低层特征”来指代分辨率较大、较具体的特征，用“高层特征”指代分辨率较小、抽象程度较高，或者说“浓缩程度”较高的特征。

分类任务里，图像通过一个漏斗状的卷积网络即可回归出它的类别；检测任务里，通过用高层特征加强低层特征的方式可以有效提升检测效果；如果反过来再用低层特征增强高层特征，网络则可同时被用于图像级、区域级和像素级的不同任务。
### 3.2 避免Isolated Convolution
尽量避免在shortcut上使用I-conv。FishNet除了tail部分在涉及通道数变化的残差模块上使用了I-conv外，在body和head部分的融合时都避免使用I-conv，从而最大限度地保证了梯度的直接反传。
### 3.3 上采样的方式
上采样方式的选择上，尽可能**不使用带权值的反卷积**，而是用最近邻插值等方式。此举同样是为了保证梯度的直接反传。
### 3.4 下采样的方式
**用kernel尺寸为\\(\(2\times 2\)\\)、stride也为\\(2\\)的MaxPooling进行下采样**与其他几种典型的下采样方式相比，效果更好。用来对(diao)比(da)的另外几种下采样方式包括：
  + 最后一层卷积stride=\\(2\\)（干扰了梯度直接反传）
  + kernel size=\\(\(3\times 3\)\\)、stride=\\(2\\)的MaxPooling（滑动窗口有交叠，扰乱了结构信息）
  + kernel size=\\(\(3\times 3\)\\)、stride=\\(2\\)的AveragePooling（原文没讲，个人认为与最后一层卷积加stride=\\(2\\)效果类似）

## 4 个人感悟
“老僧三十年前，未参禅时，见山是山，见水是水。

及至后来亲见知识，有个入处，见山不是山，见水不是水。

而今得个休歇处，依前见山只是山，见水只是水。

大众，这三般见解，是同是别，有人缁素得出。” 

——吉州青原惟信禅师[3]

FishNet的思想，似乎与这三重境界有什么关联？池化，插值，融合，再池化，再融合，这个过程，仿佛一个人脑海中对知识的建构、解构和重构的过程。

我在初识某些新事物时，由于对它还没有形成充分的了解，只是大致地形成了一个印象。比如十多年前，“屏幕，主机，鼠标，键盘”，这就是我脑海中一台计算机的样子，所谓“计算机科学”，在当时的自己看来也不过是用一些软件写写文档画画图之类的工作。

随着学习的逐渐深入，我从一个使用者成为了一个开发者后，关注点也不断地深入、细化：当看到一个网页的动效，我想到按F12看看它是怎么用js实现的，想到这个异步请求是怎么响应的，想到网络请求的TCP报文是怎样的，想到报文是如何经历一系列路由器传输到服务器的。在对计算机的了解不断深入的过程中，我却又对它产生了一种陌生感——这门科学还藏有多少的奥秘，其中是否有些我甚至还无法想象？

至于再将学习深入下去我会对计算机产生怎样的认识，才疏学浅，尚不得而知。也许某一天我会恍然大悟——哦，原来计算机科学就是这个样子的呀。

## 5 参考文献
[1] [Sun S, Pang J, Shi J, et al. Fishnet: A versatile backbone for image, region, and pixel level prediction[C]//Advances in Neural Information Processing Systems. 2018: 754-764.](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf)

[2] [Hu J, Shen L, Sun G. Squeeze-and-excitation networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2018: 7132-7141.](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)

[3] [瞿汝稷. 指月录[M]. 出版信息不详. 卷二十八 六祖下第十四世](http://www.shixiu.net/wenhua/tuijian/zyl/4802.html)

<style type="text/css" rel="stylesheet">
.markdown-body p {
    text-indent: 2em
}   
</style>
