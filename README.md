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
   image3 = np.array(image3) # 这里将Image对象转换为array,不影响数值大小
```
为什么说是坑呢?<br>
<br>
对于`.jpg`格式图像来说,三种读取方法没什么区别.假设一幅尺寸为(224, 224, 3)的`jpg`图像,每个方法读取后都是一个大小为(224, 224, 3)的三维矩阵,且数据类型均为`uint8`.所以,对于`jpg`图像,大家都很正常!<br>
<br>
但是,对于`png`格式而言,问题就全都来了!以Pascal VOC2007中分割任务中的label为例,假设一张label图的大小为(224, 224),且数据类型为uint8,其中除了边缘为255,其余的像素点灰度值范围应该在0~20之间(背景为0).那么三种读取方式格式什么样的呢?<br>
* `misc`读取后,矩阵大小为(224, 224, 3),数据类型为uint8,但是灰度值不再是在0~20之间.
* `plt`读取后,矩阵大小为(224, 224, 3),数据类型为float32,灰度值同样发生变化,并且乘以255后和`misc`读取的相同.
* `Image`读取后,矩阵大小为(224, 224),数据类型为uint8,灰度值正常,均在0~20之间(边缘为255除外).

到此为止,应该清楚问题所在了.如要要做分割任务,读取标签数据时,如果用`misc`或`plt`两种方式,由于数值变化可能得不到理想的结果!<br>
之所以提到这个坑,是因为在将数据转换为tfrecord格式时,其中一部分出现了类似的问题,在下面的说明中会指出问题所在.

## Tensorflow提供的三种数据读取的方式

* Feeding: Python产生数据，再把数据喂给后端(这个应该是最常见到也是最熟悉的方式了)。
* Preloaded data (预加载数据): 在Tensorflow的图中定义常量或变量来保存所有数据(仅适用于数据量比较小的情况).
* Reading from file (从文件中直接读取): 在Tensorflow图的起始,让一个输入管线从文件中读取数据.

对于数据量较小的情况,我们可以一次性的将数据加载内存中,然后分为batch输入网络进行训练.但是当数据量太大时,由于内存限制此方法就不再适用了.对于数据量较大的情况,用python写一个数据接口,然后feeding给网络也是常见的方式.首先可以用python将所有图像读入并且保存,格式可以为```.npy```也可以是其他的,也可以实时的从硬盘批量读取图像(这种方式效率太低,不建议使用).那么为什么要适用tfrecord格式呢?
