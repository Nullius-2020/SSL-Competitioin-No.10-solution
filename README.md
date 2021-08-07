## 第三届中国AI+创新创业大赛：半监督学习目标定位竞赛第十名方案


## 项目描述
采用Self-training和Consistency learning结合的思路：训练采用Self-training方法，伪标签的处理采用Consistency learning方法，多次迭代训练，推理时多尺度和翻转增强，最后融合多次最好的结果。



## 项目结构
-README.MD       说明文件

-2162311.ipynb   可在AiStuio中运行的Notebook文件

-check.py        数据检查和分析

-fcn2.py         用于PaddleSeg中的模型文件

-fcn2.yml        有监督训练的配置文件

-fcn2_pl.yml     半监督训练的配置文件

-vote.py         用于融合多个模型结果的文件


```
## 一键运行
在AI Studio上一键[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/2162311)

