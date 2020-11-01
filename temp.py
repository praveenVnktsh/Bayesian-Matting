import cv2

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
