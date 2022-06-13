'''
Author: LetMeFly
Date: 2022-06-13 00:15:41
LastEditors: LetMeFly
LastEditTime: 2022-06-13 00:15:41
'''
import cv2
# 读取图片
img = cv2.imread('./9.36.jpg',0)
# 定义核结构元素
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
# 腐蚀图像
eroded = cv2.erode(img,kernel)
# 显示腐蚀后的图像
cv2.imshow("Eroded Image",eroded)
#膨胀图像
dilated = cv2.dilate(img,kernel)
#显示膨胀后的图像
cv2.imshow("Dilated Image",dilated)
# 开运算
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# 显示开运算后的图像
cv2.imshow("Open", opened)
#闭运算
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
#显示闭运算后的图像
cv2.imshow("Close",closed)
cv2.waitKey(0)