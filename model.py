# -*- coding: utf-8 -*-  

from keras import optimizers
import csv
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D


# 获取记录文件，driving_log.csv，得到每一行数据
def read_log(filepath):
    lines = []
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    # 每一行： 中间图片，左边图片，右边图片，专项较多，，， 速度
    return lines


# Read original data
alllines = read_log('/Users/admin/auto/datas/1/driving_log.csv')
#lines_curves = read_log('../data_curves/driving_log.csv')
# 二维数组
lines = np.array(alllines)

# Balance data
nbins = 2000
max_examples = 200

# 选择数据， lines.shape[1]为每行数据个数
# 随机抽样，尽量使数据平衡，可以将整个Steering Angle范围划分成2000个bucket，保证每个bucket中数据样本不超过200个
# 从-1.0到1.0
balanced = np.empty([0, lines.shape[1]], dtype=lines.dtype)
for i in range(0, nbins):
    begin = i * (1.0 / nbins)
    end = begin + 1.0 / nbins
    # 偏向角度选择范围
    extracted = lines[(abs(lines[:, 3].astype(float)) >= begin)
                      & (abs(lines[:, 3].astype(float)) < end)]
    # 随机选择
    np.random.shuffle(extracted)
    extracted = extracted[0:max_examples, :]
    balanced = np.concatenate((balanced, extracted), axis=0)

# 输入图片，输出角度
imgs, angles = [], []

# 左右两边图片角度修正
offset = 0.2
correction = [0, offset, -offset]  # center, left, right cameras
for line in balanced:
    for i in range(3): #0,1,2
        img_path = line[i]
        img = cv2.imread(img_path)
        imgs.append(img)

        angle = float(line[3])
        angles.append(angle + correction[i])

# 左右翻转图片，增加训练样本
flip_imgs, flip_angles = [], []
for img, angle in zip(imgs, angles):
    flip_imgs.append(cv2.flip(img, 1))
    flip_angles.append(-1.0 * angle)

# 合并输入输出
augmented_imgs = imgs + flip_imgs
augmented_angles = angles + flip_angles

X_train = np.array(augmented_imgs)
y_train = np.array(augmented_angles)

# Build the model
model = Sequential()
# 输入为3张图片160高, 320宽的图片， 颜色值灰度值归一化到[-0.5, 0.5]
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160, 320, 3)))  
# 对2D输入（图像）进行裁剪，即宽和高的方向上裁剪， 保留中间有效部分
# (top_crop, bottom_crop), (left_crop, right_crop), 65高度，320宽度
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
# 二维卷积层，即对图像卷积, filters = 8,  kernel_size = {3, 3}, strides=(1, 1),
model.add(Convolution2D(8, 3, 3, activation='relu'))
# 为空域信号施加最大值池化
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合
model.add(Dropout(0.2))
model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
# Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Flatten())
# Dense就是常用的全连接层
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))         #1维输出

model.compile(loss='mse', optimizer = optimizers.Adam(lr=0.0001))
best_model = ModelCheckpoint('model_best.h5', verbose=2, save_best_only=True)
model.fit(X_train, y_train, validation_split=0.2,
          shuffle = True, epochs = 30, callbacks= [ best_model ])

model.save('model_last.h5')
