'''
Author: LetMeFly
Date: 2022-06-12 23:28:36
LastEditors: LetMeFly
LastEditTime: 2022-06-13 15:01:56
'''
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# 避免plt警告
os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

# 支持中文显示
plt.rcParams[ 'font.sans-serif' ] = [ 'SimHei' ]

# 读入图像
img = cv2.imread("img/9.36.jpg", 0)
rows, cols = img.shape

# 二值化
_, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))  # 开运算
img_open_backup = img_open.copy()

# 把图像分成一个个连通块儿
num, img_label = cv2.connectedComponents(img_open)
labels = [i for i in range(1, num)]

bounded_labels = set()  # 处在边界的标签
for col in range(cols):
    if img_label[0][col]:
        bounded_labels.add(img_label[0][col])
    if img_label[rows - 1][col]:
        bounded_labels.add(img_label[rows - 1][col])
for row in range(rows):
    if img_label[row][0]:
        bounded_labels.add(img_label[row][0])
    if img_label[row][cols - 1]:
        bounded_labels.add(img_label[row][cols - 1])

# 与边界重合的部分
img_bounds = np.zeros((rows, cols), dtype=np.uint8)
for row in range(rows):
    for col in range(cols):
        if img_label[row][col] in bounded_labels:
            img_bounds[row][col] = 255
img_open -= img_bounds

# 获取各个标签的面积
area_dict = {}
for label in labels:
    area_dict[label] = 0
for row in range(rows):
    for col in range(cols):
        if img_open[row][col]:
            area_dict[img_label[row][col]] += 1

# 设置单个颗粒的面积阈值
single_area = 420  # 经过调试，420是个不错的选择

# 相互重叠的图像（面积 > 单个颗粒的图像）
img_overlap = np.zeros((rows, cols), np.uint8)
for row in range(rows):
    for col in range(cols):
        if img_label[row][col] and area_dict[img_label[row][col]] > single_area:
            img_overlap[row][col] = 255

# 剩下的就是单个颗粒的部分
img_single = img_open - img_overlap

# 显示结果
_, ax_list = plt.subplots(1, 5, figsize=(20, 10))
ax_list[0].set_title("原图")
ax_list[0].imshow(img, cmap="gray")
ax_list[1].set_title("开运算")
ax_list[1].imshow(img_open_backup, cmap="gray")
ax_list[2].set_title("与边界融合")
ax_list[2].imshow(img_bounds, cmap="gray")
ax_list[3].set_title("相互重叠")
ax_list[3].imshow(img_overlap, cmap="gray")
ax_list[4].set_title("没有重叠")
ax_list[4].imshow(img_single, cmap="gray")
plt.show()
