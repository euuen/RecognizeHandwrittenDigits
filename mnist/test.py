import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
# 下载数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 数据预处处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255.
X_test = X_test.reshape(X_test.shape[0], -1) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
# 不使用model.add()，用以下方式也可以构建网络
# model = Sequential([
#     Dense(400, input_dim=784),
#     Activation('relu'),
#     Dense(512),
#     Activation('relu'),
#     Dense(512),
#     Activation('relu'),
#     Dense(10),
#     Activation('softmax'),
# ])
# # 定义优化器
# rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# model.compile(optimizer=rmsprop,
#               loss='categorical_crossentropy',
#               metrics=['accuracy']) # metrics赋值为'accuracy'，会在训练过程中输出正确率
# # 这次我们用fit()来训练网路
# print('Training over')
# model.fit(X_train, y_train, epochs=4, batch_size=32)
# model.save("mnist.h5")
model = load_model("mnist.h5")
model.fit(X_train, y_train, epochs=4, batch_size=32)
model.save("mnist.h5")
a,  b = model.evaluate(X_test, y_test)
print(b)



# def max_index(array):
#     maxIndex = 0
#     for i in range(0, len(array)):
#         if array[i] > array[maxIndex]:
#             maxIndex = i
#     return maxIndex
#
# plt.subplot()
# plt.imshow(X_train[2].reshape(28, 28))
# plt.show()


# print(model.predict(X_test[0]))
# print(max_index(y_test[0]))

























































