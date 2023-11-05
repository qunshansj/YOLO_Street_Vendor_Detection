
# 1.识别效果展示
![1.png](https://img-blog.csdnimg.cn/img_convert/1aaac339bec37ecc41e12c5bd78a895b.png
![2.png](dd10d6047e516cb772a576b2ee6fa103.png)

![3.png](6666ee61e2f5763bd3d829be0d6da989.png)



# 2.视频演示

[[YOLOv7]基于YOLOv7的游商游贩检测系统(源码＆部署教程＆数据集)_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1YB4y1g7Ee/?vd_source=bc9aec86d164b67a7004b996143742dc)


# 3.算法简介

无论梯度路径长度和大规模 ELAN 中计算块的堆叠数量如何，它都达到了稳定状态。如果无限堆叠更多的计算块，可能会破坏这种稳定状态，参数利用率会降低。新提出的 E-ELAN 使用 expand、shuffle、merge cardinality 在不破坏原有梯度路径的情况下让网络的学习能力不断增强。
![5.png](e2031f9a87a3f9448dc1f2505b94cc5a.png)

此外， YOLOv7 的在速度和精度上的表现也优于 YOLOR、YOLOX、Scaled-YOLOv4、YOLOv5、DETR 等多种目标检测器。

# 4.技术方法
近年来，实时目标检测器仍在针对不同的边缘设备进行开发。例如，MCUNet 和 NanoDet 的开发专注于生产低功耗单芯片并提高边缘 CPU 的推理速度；YOLOX、YOLOR 等方法专注于提高各种 GPU 的推理速度；


## 该研究的主要贡献包括：
[参考该博客提出的创新点](https://mbd.pub/o/bread/Y5WVm5hu)
(1) 设计了几种可训练的 bag-of-freebies 方法，使得实时目标检测可以在不增加推理成本的情况下大大提高检测精度；

(2) 对于目标检测方法的演进，研究者发现了两个新问题：一是重参数化的模块如何替换原始模块，二是动态标签分配策略如何处理分配给不同输出层的问题，并提出了解决这两个问题的方法； 

(3) 提出了实时目标检测器的「扩充（extend）」和「复合扩展（compound scale）」方法，以有效地利用参数和计算； 

(4) 该研究提出的方法可以有效减少 SOTA 实时目标检测器约 40% 的参数和 50% 的计算量，并具有更快的推理速度和更高的检测精度。

在大多数关于设计高效架构的文献中，人们主要考虑的因素包括参数的数量、计算量和计算密度。下图 2（b）中 CSPVoVNet 的设计是 VoVNet 的变体。CSPVoVNet 的架构分析了梯度路径，以使不同层的权重能够学习更多不同的特征，使推理更快、更准确。图 2 (c) 中的 ELAN 则考虑了「如何设计一个高效网络」的问题。

YOLOv7 研究团队提出了基于 ELAN 的扩展 E-ELAN，其主要架构如图所示。
![6.png](5d8f9125877a90f816346c99feda7c8e.png)
新的 E-ELAN 完全没有改变原有架构的梯度传输路径，其中使用组卷积来增加添加特征的基数（cardinality），并以 shuffle 和 merge cardinality 的方式组合不同组的特征。这种操作方式可以增强不同特征图学得的特征，改进参数的使用和计算效率。


[参考该博客，改变了计算块的架构](https://afdian.net/item?plan_id=92c8275860a311edab9c52540025c377)，而过渡层（transition layer）的架构完全没有改变。YOLOv7 的策略是使用组卷积来扩展计算块的通道和基数。研究者将对计算层的所有计算块应用相同的组参数和通道乘数。然后，每个计算块计算出的特征图会根据设置的组参数 g 被打乱成 g 个组，再将它们连接在一起。此时，每组特征图的通道数将与原始架构中的通道数相同。最后，该方法添加 g 组特征图来执行 merge cardinality。除了保持原有的 ELAN 设计架构，E-ELAN 还可以引导不同组的计算块学习更多样化的特征。
因此，对基于串联的模型，我们不能单独分析不同的扩展因子，而必须一起考虑。该研究提出图 （c），即在对基于级联的模型进行扩展时，只需要对计算块中的深度进行扩展，其余传输层进行相应的宽度扩展。这种复合扩展方法可以保持模型在初始设计时的特性和最佳结构。

此外，该研究使用梯度流传播路径来分析如何重参数化卷积，以与不同的网络相结合。下图展示了该研究设计的用于 PlainNet 和 ResNet 的「计划重参数化卷积」。
![7.png](a453b65f772444d4f27e2c5f2dce308d.png)

# 5.数据集的准备
## 标注收集到的图片制作YOLO格式数据集


![11.png](72ddc8234d67a250aaabfb8509c875b1.png)
自己创建一个myself.yaml文件用来配置路径，路径格式与之前的V5、V6不同，只需要配置txt路径就可以
![8.png](abafd23beaafe64c06ce9308e859df7f.png)

![9.png](7f87b0024db736efa776ab99f7ed1f4c.png)
 train-list.txt和val-list.txt文件里存放的都是图片的绝对路径（也可以放入相对路径）
![12.png](f0e8d78f41e619f3c88fd36148e71198.png)
 如何获取图像的绝对路径，脚本写在下面了（也可以获取相对路径）
```
# From Mr. Dinosaur
 
import os
 
 
def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
 
 
list_name = []
path = 'D:/PythonProject/data/'  # 文件夹路径
listdir(path, list_name)
print(list_name)
 
with open('./list.txt', 'w') as f:  # 要存入的txt
    write = ''
    for i in list_name:
        write = write + str(i) + '\n'
    f.write(write)
```
# 6.训练过程

## 运行train.py
train文件还是和V5一样，为了方便，我将需要用到的文件放在了根目录下
![13.png](2d0296a1f13d91e5f016d6801e14e65d.png)

路径修改完之后右击运行即可
![14.png](fd2efe305ef67a13ac864443348a4501.png)
## 经过漫长的训练过程，YOLOv7相比YOLOv5训练更吃配置尤其是显存，实测GPU 3080ti训练长达40小时以上，建议电脑显存8G以下的谨慎尝试，可能训练的过程低配置的电脑会出现蓝屏等现象皆为显卡过载，使用本文提供的训练好的权重进行预测则不吃配置，CPU也能取得很好的预测结果且不会损伤电脑
附上本文实验设备配置
![16.jpg](d9d69fd9bbc7d9f803745748b803339a.jpeg)


# 7.测试验证
下面放上对比图：（上面V7，下面V5）
![15.png](b89d75bddbfeb3dca5ecff5ef38c4914.png)


# 8.系统整合
下图[完整源码&环境部署视频教程&数据集&自定义UI界面](https://s.xiaocichang.com/s/12253c)
![4.png](8570256eff204c37d484db5fe635df91.png)
参考博客《[YOLOv7]基于YOLOv7的游商游贩检测系统(源码＆部署教程＆数据集)》









---
#### 如果您需要更详细的【源码和环境部署教程】，除了通过【系统整合】小节的链接获取之外，还可以通过邮箱以下途径获取:
#### 1.请先在GitHub上为该项目点赞（Star），编辑一封邮件，附上点赞的截图、项目的中文描述概述（About）以及您的用途需求，发送到我们的邮箱
#### sharecode@yeah.net
#### 2.我们收到邮件后会定期根据邮件的接收顺序将【完整源码和环境部署教程】发送到您的邮箱。
#### 【免责声明】本文来源于用户投稿，如果侵犯任何第三方的合法权益，可通过邮箱联系删除。