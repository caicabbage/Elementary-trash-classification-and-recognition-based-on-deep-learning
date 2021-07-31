import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('./results/garbage2.h5')
filename = r'datasets\test\metal1.jpg'

test_img = tf.keras.preprocessing.image.load_img(filename, target_size=(150, 150, 3))  # 此处得到的是pillow图像Image实例
test_img = tf.keras.preprocessing.image.img_to_array(test_img)  # 将Image实例转化为多维数组

model_path = './results/knn.h5'
try:
    model_path = os.path.realpath(__file__).replace('main.py', model_path)
except NameError:
    model_path = './' + model_path

x = np.expand_dims(test_img, axis=0)

y = model.predict(x)

labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

predict = labels[np.argmax(y)]

print('预测结果：', end='')
print(predict)