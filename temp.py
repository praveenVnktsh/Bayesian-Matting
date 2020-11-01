import cv2
import numpy as np
import glob
index = 9
scale = 0.3
# gt = cv2.resize(cv2.imread('trainingData/GT/GT' + str(index).zfill(2) + '.png'),(0,0), fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)


index = 15
# img = cv2.resize(cv2.imread('trainingData/input/GT' + str(index).zfill(2) + '.png'),(0,0), fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
# cv2.imwrite('OUTPUT/' + str(index) + '-IMAGE.png', img)
# img = cv2.resize(cv2.imread('trainingData/trimap/Trimap1/GT' + str(index).zfill(2) + '.png'),(0,0), fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
# cv2.imwrite('OUTPUT/' + str(index) + '-TRIMAP.png', img)
# img = cv2.resize(cv2.imread('trainingData/GT/GT' + str(index).zfill(2) + '.png'),(0,0), fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
# cv2.imwrite('OUTPUT/' + str(index) + '-GT.png', img)
# print(img.shape)


for path in glob.glob('OUTPUT/TRIMAP/*'):
    # print(path)
    name = path[::-1][4:6][::-1]
    # print(name)
    img = cv2.imread(path)
    # img = cv2.resize(img, (0,0), fx = 0.3, fy = 0.3)
    # cv2.imwrite('OUTPUT/TRIMAP/' + name + '-TRIMAP.png', img)
    print(path, img.size/3)

# for path in glob.glob('OUTPUT/TRIMAP/*'):
# 	img = cv2.imread(path)
# 	img = cv2.resize(path, (0,0), fx = 0.3, fy =  0.3 )
#     o = cv2.imwrite(path, img)
    

