import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt



#使图像显示中文
plt.rcParams["font.sans-serif"]=["FangSong"]
#加载数据
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()
#构建标签列表
class_names = ['T恤/上衣', '裤子', '套衫', '连衣裙', '外套','凉鞋', '衬衫', '运动鞋', '手提包', '踝靴']

#进行数据预处理,将每个像素点都压缩在0和1之间
train_images  =  train_images/255.0
test_images = test_images/255.0
#加载神经网络模型
new_model = keras.models.load_model("model/...new_model.H5")
#对刚才训练的模型进行测试
test_loss,test_acc = new_model.evaluate(test_images,test_labels,verbose=2)
print("\nTest accuracy:{:5.2f}%".format(100*test_acc))
#对数据进行预测
probability_model=keras.Sequential([new_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
# 答应所有的预测结果
plt.figure()
plt.subplot(1,2,1)
plt.xticks([])
plt.yticks([])
plt.imshow(test_images[0],cmap=plt.cm.binary),plt.xlabel("{} 预测正确率：{:2.0f}%".format(class_names[np.argmax(predictions[0])],
                               100*np.max(predictions[0])),fontsize=20,color="blue")
plt.subplot(1,2,2)
plt.xticks(range(10),class_names)
plt.yticks([])
thisplot = plt.bar(range(10), predictions[0], color="#777777")
plt.ylim([0, 1])
predicted_label = np.argmax(predictions[0])

thisplot[predicted_label].set_color('blue')
# thisplot[true_label].set_color('blue')
plt.show()
print("模型预测的结果为：{}".format(class_names[np.argmax(predictions[12])]))
