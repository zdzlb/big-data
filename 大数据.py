# #导入包
# import tensorflow as tf
# import numpy as np
# from tensorflow import keras
# import matplotlib.pyplot as plt
#
# #加载数据
# fashion_mnist = keras.datasets.fashion_mnist
# (train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
# # print(len(test_images))
#
# #打印图像
# plt.figure()
# plt.xticks()
# plt.yticks()
# plt.imshow(train_images[12])
# plt.show()



#导入模块
import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

#加载数据

fashion_mnist = keras.datasets.fashion_mnist
#进行训练集与测试集的划分

(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
# print(train_images.shape)

#数据预处理，将每个像素点压缩在0和1之间
train_images = train_images/255.0
test_images = test_images/255.0

#搭建简单的神经网络
def create_model():
    model =tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation="relu"),
        keras.layers.Dense(10)]
    )

#编译模型
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
     )
#还回模型xc
    return model

#构建模型
new_model = create_model()

#训练模型
new_model.fit(train_images,train_labels,epochs=10)

#保存模型
new_model.save("model/...new_model.H5")






