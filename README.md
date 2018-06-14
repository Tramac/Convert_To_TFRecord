# Convert_To_TFRecord
这是一份如何将数据转为tfrecord格式的教程

**在开始之前有一个坑需要说明一下(超级坑啊!)**<br>
**一般来说,python读取图像时,常用的方式有以下三种:**
```python
   import numpy as np
   import scipy.misc as misc
   import matplot.pyplot as plt
   from PIL import Image
       
   # method 1
   image1 = misc.imread(image_path)
       
   # method 2
   image2 = plt.imread(image_path)
       
   # method 3
   image3 = Image.open(image_path)
   image3 = np.array(image3)
```

## Tensorflow提供的三种数据读取的方式

* Feeding: Python产生数据，再把数据喂给后端(这个应该是最常见到也是最熟悉的方式了)。
* Preloaded data (预加载数据): 在Tensorflow的图中定义常量或变量来保存所有数据(仅适用于数据量比较小的情况).
* Reading from file (从文件中直接读取): 在Tensorflow图的起始,让一个输入管线从文件中读取数据.

对于数据量较小的情况,我们可以一次性的将数据加载内存中,然后分为batch输入网络进行训练.但是当数据量太大时,由于内存限制此方法就不再适用了.对于数据量较大的情况,用python写一个数据接口,然后feeding给网络也是常见的方式.首先可以用python将所有图像读入并且保存,格式可以为```.npy```也可以是其他的,也可以实时的从硬盘批量读取图像(这种方式效率太低,不建议使用).那么为什么要适用tfrecord格式呢?
