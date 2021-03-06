# Convert_To_TFRecord

### Update
当出现以下错误时，请先检查路径是否有错误。
```
OutOfRangeError (see above for traceback): RandomShuffleQueue '_0_shuffle_batch/random_shuffle_queue' is closed and has insufficient elements (requested 16, current size 0)
	 [[Node: shuffle_batch = QueueDequeueManyV2[component_types=[DT_FLOAT, DT_UINT8], timeout_ms=-1, _device="/job:localhost/replica:0/task:0/device:CPU:0"](shuffle_batch/random_shuffle_queue, shuffle_batch/n)]]

```

----------

这是一份如何将数据转为tfrecord格式的教程

**在开始之前有一个坑需要说明一下(超级坑啊!)**<br>
一般来说,python读取图像时,常用的方式有以下三种:
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
注：python中，如果用opencv读取图片，那么读进来的就是BGR,矩阵大小为(h,w,3)，如果是PIL.Image读取的图片就是RGB，矩阵大小为(w,h,3)。<br>
为什么说是坑呢?<br>
<br>
对于`.jpg`格式图像来说,三种读取方法没什么区别.假设一幅尺寸为(224, 224, 3)的`jpg`图像,每个方法读取后都是一个大小为(224, 224, 3)的三维矩阵,且数据类型均为`uint8`.所以,对于`jpg`图像,大家都很正常!<br>
<br>
但是,对于`png`格式而言,问题就全都来了!以Pascal VOC2007中分割任务中的label为例,假设一张label图的大小为(224, 224),且数据类型为uint8,其中除了边缘为255,其余的像素点灰度值范围应该在0~20之间(背景为0).那么三种读取方式格式什么样的呢?<br>
* `misc`读取后,矩阵大小为(224, 224, 3),数据类型为uint8,但是灰度值不再是在0~20之间.
* `plt`读取后,矩阵大小为(224, 224, 3),数据类型为float32,灰度值同样发生变化,并且乘以255后和`misc`读取的相同.
* `Image`读取后,矩阵大小为(224, 224),数据类型为uint8,灰度值正常,均在0~20之间(边缘为255除外).

到此为止,应该清楚问题所在了.如要要做分割任务,读取标签数据时,如果用`misc`或`plt`两种方式,由于数值变化可能得不到理想的结果!<br>
之所以提到这个坑,是因为在将数据转换为tfrecord格式时,其中一部分出现了类似的问题,在下面的说明中会指出问题所在.<br>
<br>

下面回到正题!
## Tensorflow提供的三种数据读取的方式

* Feeding: Python产生数据，再把数据喂给后端(这个应该是最常见到也是最熟悉的方式了)。
* Preloaded data (预加载数据): 在Tensorflow的图中定义常量或变量来保存所有数据(仅适用于数据量比较小的情况).
* Reading from file (从文件中直接读取): 在Tensorflow图的起始,让一个输入管线从文件中读取数据.

对于数据量较小的情况,我们可以一次性的将数据加载内存中,然后分为batch输入网络进行训练.但是当数据量太大时,由于内存限制此方法就不再适用了.对于数据量较大的情况,用python写一个数据接口,然后feeding给网络也是常见的方式.首先可以用python将所有图像读入并且保存,格式可以为```.npy```也可以是其他的,也可以实时的从硬盘批量读取图像(这种方式效率太低,不建议使用).但是即便如此,遇到大型数据时也会很吃力,中间环节的增加也是不小的开销,比如数据类型转换等等.<br>
那么相对而言,tfrecord格式有哪些优势呢?

## TFRecords
TFRecords是一种二进制文件,能够更好的利用内存,方便复制和移动,并且可以将图像和标签以及其他的一些属性(如文件名,图像大小等)集成在一起.<br>
<br>
关于一些TFRecords格式的一些其他性质,这里不过多介绍了,网上一搜一大把.<br>
<br>
下面以Pascal VOC2007的分割任务为例,来说明如何利用TFRecords,当然自己的数据格式也可以灵活运用!<br>

为了方便起见,只对部分代码片做出解释,代码片和完整代码并不完全一样.<br>
## 将数据转化为TFRecord格式
### 准备数据
* 首先,下载VOC2007数据集,并将其解压到`./datasets`文件夹.
### 生成TFRecords文件
```Python
   import os
   import random
   import numpy as np
   import tensorflow as tf
   
   from PIL import Image
   
   data_dir = "./datasets/VOC2007"
   output_dir = os.path.join(data_dir, "tfrecord_data")
   if not tf.gfile.Exists(output_dir):
       tf.gfile.MakeDirs(output_dir)
       
   # 给你生成的tfrecord文件起一个有辨识度的文件名
   tfrecord_filename = os.path.join(output_dir, '{}_{}.tfrecord'.format("VOC2007", "train"))
   
   # 创建一个writer来写TFRecord文件
   writer = tf.python_io.TFRecordWriter(tfrecord_filename)
   
   # 获取要处理的数据
   examples_path = os.path.join(data_dir, 'ImageSets', 'Segmentation', 'train' + '.txt') # 根据自己的数据格式更改
   examples_list = read_examples_list(examples_path)
   
   # 获取image,label的文件名
   for idx, example in enumerate(examples_list):
       image_path = os.path.join(data_dir, "JPEGImages", example + '.jpg')
       label_path = os.path.join(data_dir, "SegmentationClass", example + '.png')
       
       # 下面提供了两种对数据读取以及编码的方式,也就是最开始提到的坑出现的地方,其中method 1是正常的方法,建议使用此方法
       # method 1
       image = np.array(Image.open(image_path)) 
       label = np.array(Image.open(label_path)) # 用Image读取图像,小心这里的坑
       label[label == 255] = 0 # 针对语义分割,要求标签的值为[0,21),但是标注的边界部分都是255,这里将其置为0,这可能不是一个好的解决方法,包括后处理中,将添加的背景也是置为0
       height, width = image.shape[0], image.shape[1] # 
       image_raw = image.tobytes() # 将图像转化为生成一个字符串,注意,对于(224, 224, 3)的图像而言,转换之后是一个224x224x3的列向量(?)
       label_raw = label.tobytes() # 不再是三维矩阵,这也是为什么上面要获取height和width,为了解码的时候对列向量reshape,恢复原图大小
       
       '''
       方法2是一个更为直接的编码方式,解码时也很方便,但是存在的一个问题是,tf.gfile.FastGFile对png图像编码时出现了和misc和plt读取png时相同的问题,使得label的数值发生更改,所以这里方法2并不合适,但是如果你的标签不是png格式,用这个方法应该不错(请自己尝试).
       '''
       # method 2
       # image = np.array(Image.open(image_path)) #
       # height, width = image.shape[0], image.shape[1] # 这里获取height,width不是必须的
       # image_raw = tf.gfile.FastGFile(image_path, 'r').read()
       # label_raw = tf.gfile.FastGFile(label_path, 'r').read()
       
       # 将一个样例转化为Example Protocol Buffer,并将所有的信息写出这个数据结构,这里可以看出tfrecord的巨大优势了,因为不必将image和label分开,而且可以集成更多的属性
       example = tf.train.Example(features=tf.train.Features(feature={
        'label_raw': bytes_feature(label),
        'image_raw': bytes_feature(image),
        'height': int64_feature(height),
        'width': int64_feature(width)
    }))
    
    # 将一个Example写入TFRecord文件
    writer.write(example.SerializeToString())
    
  writer.close()
```
执行上面的代码就可以把数据生成tfrecord文件了,那么下面应该就是在训练时,如何从tfrecord文件中解析出图像了.<br>

### Tensorflow从TFRecord中读取数据
生成了tfrecord文件之后,为了高效的读取数据,TF中使用队列(queue)读取数据
```Python
   def read_and_decode(self, filename):
       # 创建一个队列
       filename_queue = tf.train.string_input_producer([filename]) # 
       # 创建一个reader来读取tfrecord文件中的样例
       reader = tf.TFRecordReader()
       # 从文件中读取一个样例
       _, serialized_example = reader.read(filename_queue)
       # 解析读入的一个样例
       features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label_raw': tf.FixedLenFeature([], tf.string),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'height': tf.FixedLenFeature([], tf.int64),
                                               'width': tf.FixedLenFeature([], tf.int64),
                                           })
       
       '''
       下面的两种解析方法和前面的编码方法保持对应关系,即前面如果用method1编码,这里也应该用method1解码,method2同理.
       '''
       # method 1
       height = tf.cast(features['height'], tf.int32)
       width = tf.cast(features['width'], tf.int32)
       image = tf.decode_raw(features['image_raw'], tf.uint8) # 将字符串解析成图像对应的像素数组,注意这里并不是三维矩阵,而是一个列向量,所以后面需要reshape恢复原图像大小
       image = tf.reshape(image, [height, width, 3]) # 当然,如果你的图像允许,可以在编码之前就将你的图像resize到一个固定尺寸,这里解析的时候直接reshape就可以了,这样一来,就不用获取height,width属性了.
       label = tf.decode_raw(features['label_raw'], tf.uint8)
       label = tf.reshape(label, [height, width, 1])
       
       '''
       方法2是直接解析出jpg,png图像,但是这种方式有它固定的编码方式,即编码方法2,该方法不用reshape,但是如前面提到的,由于对png格式的图像不太友好,所以慎用
       '''
       # method 2
       # image = tf.image.decode_jpeg(features['image_raw'])
       # label = tf.image.decode_png(features['label_raw'])
       
      # 解析出图像之后,可以做你想要的预处理,比如归一化,数据扩增等等
      image, label = preprocess_image(image, label, is_training="train")
      
      return image, label
```
训练时的数据接口,就可以很方便的用下面的方式
```python
   image, label = read_and_decode("VOC2007_train.tfrecord")
   image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                     batch_size=64,
                                                     capacity=2000,
                                                     min_after_dequeue=5) # capacity,min_after_dequeue两个参数详细看官网,其中min_after_dequeue参数对获取数据速度有很大的影响,请根据自己的数据情况调节
   
   init = tf.initialize_all_variables()
   with tf.Session() as sess:
       sess.run(init) # 相当于初始化定义的graph,完整的模型肯定会包含的
       # 启动多线程处理数据,很重要,虽然不知道为什么,但是不用会出错
       coord = tf.train.Coordinator()
       threads = tf.train.start_queue_runners(sess=sess, coord=coord) # 必不可少
       img, label = sess.run([img_batch, label_batch])
       
       # 再往下就可以对你的批数据进行操作了,比如你要打印形状
       print(img.shape)
       
       coord.request_stop()
       coord.join(threads)
```

### 流程总结
* 生成tfrecord格式文件
* 定义record reader解析tfrecord文件,也就是`read_and_decode`函数
* 构建一个批生成器`tf.train.shuffle_batch`
* 初始化操作
* 启动QueueRunner

**以上是整个过程总结,这只是一个数据接口,完整的项目运用请见后续工作~**
