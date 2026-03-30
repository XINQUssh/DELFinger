## Training DeLFinger

There are 2 steps for training: (1) global stage, and (2) local keypoint stage.  

**(1) training global stage:**

$ cd train_res

$python main.py

参数可以进config.py自行查看修改

**(2) training local stage:**

+ load_from: 需要读取global阶段训练好的模型 (<model_name>.pth.tar)
+ expr: 保持和global阶段的一致就行

$ cd train_res
$ python main.py  --load_from <path to model> 



## Train PCA

首先将local阶段训练好的模型bestshot.pth.tar cp 一份成 fix.pth.tar

$ cd extract
$ python extractor.py

参数进入extractor.py自行修改

训练完成会得到.h5文件



**[parameter]**

extract/extractor.py

​	rf(感受野大小)

​	ATTN_THRES(注意力分数阈值)

​	PCA_DIMS(PCA降维维度)

​	SCALE_LIST(多尺度)

helper/matcher.py

​	cKDTree的距离阈值——_DISTANCE_THRESHOLD

​	LPM的几个parameters

## Extractor Feature

使用local阶段训练出的模型<bestshot.pth.tar>和PCA后的特征<out.h5>进行特征矩阵的构建

$ python run.py



## SVM

将生成的feature.txt文件读入并训练svm分类器

$ python svm.py





