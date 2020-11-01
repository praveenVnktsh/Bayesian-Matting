import cv2
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import json
from numpy.linalg.linalg import solve
from sklearn.cluster import KMeans
import time

def showMultiImages(arr, name = 'MULTIIMAGE'):
	arrNew = [cv2.resize(x, (0,0), fx = 1.5, fy = 1.5) for x in arr]
	cv2.imshow(name, np.concatenate(arrNew, axis=1))

def getImages(index, scale = 0.3):
	gt = cv2.resize(cv2.imread('trainingData/GT/GT' + str(index).zfill(2) + '.png'),(0,0), fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
	input = cv2.resize(cv2.imread('trainingData/input/GT' + str(index).zfill(2) + '.png'),(0,0), fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
	trimap = cv2.resize(cv2.imread('trainingData/trimap/Trimap1/GT' + str(index).zfill(2) + '.png'),(0,0), fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
	trimap2 = cv2.resize(cv2.imread('trainingData/trimap/Trimap2/GT' + str(index).zfill(2) + '.png'), (0,0),fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
	print(gt.shape)
	return input, trimap, trimap2, gt


def getFBM(trimap):
	fgSure = trimap.copy()
	bgSure = trimap.copy()
	mid = trimap.copy()
	fgSure[fgSure != 255] = 0
	bgSure = 255 - fgSure

	mid[mid != 128] = 0
	mid[mid == 128] = 255
	bgSure = 255 - mid
	bgSure = bgSure - fgSure
	return bgSure, fgSure, mid



def getGaussianWeights(size = 3, sigma=8):
 # Gives a filter that is similar to the gaussian blurring filter
		gaussTemp = cv2.getGaussianKernel(size, sigma)
		kernel = np.multiply(gaussTemp.T, gaussTemp)
		return kernel

def showImage(image, name = 'showimage'):
	cv2.imshow(name, cv2.resize(image, (350,350)))




def getPixelNeighbourhood(img, location,size, threechannel = False):
	y, x = location
	if threechannel:
		h, w, _ = img.shape
	else:
		h, w = img.shape


	xlo = max(0, x - size//2)
	ylo = max(0, y - size//2)
	xhi = min(w, x + (size//2+1))
	yhi = min(h, y + (size//2+1))


	ret = img[ylo:yhi, xlo:xhi]
	if ret.shape[0] != size or ret.shape[1] != size:
		ret = cv2.resize(ret, (size, size))


	return ret



def getMeanAndCov(pixels, weights, nClusters = 5):

	kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(pixels)
	predictedClusters = np.array(kmeans.predict(pixels))
	means = []
	covs = []
	for i in range(nClusters):
		trutharray = (predictedClusters == i)
		pixelsbin = pixels[trutharray]
		weightsbin = np.reshape(weights[trutharray], (weights[trutharray].size,1))
		weightsSqrt = np.sqrt(weightsbin)
		meanBin = np.array([np.dot(pixelsbin[:,0], weightsbin),np.dot(pixelsbin[:,1], weightsbin),np.dot(pixelsbin[:,2], weightsbin)])/np.sum(weightsbin)
		meanBin = np.reshape(meanBin, (3,))
		means.append(meanBin)
		diff = (pixelsbin - meanBin)
		diff = np.multiply(weightsSqrt, diff)
		covBin = np.eye(3)
		with np.errstate(divide='raise'):
			try:
				covBin = np.dot(diff.T, diff)/np.sum(weightsbin) + np.eye(3)*(10**(-5)) #adding to prevent singular matrices
			except Exception as e:
				print(covBin.shape)
				print(covBin)
		
		covs.append(covBin)  
	
	means = np.array(means).T
	covs = np.array(cv2.merge(covs))
	# exit()
	return means, covs

def iterativeSolver(fgMean, fgCov, bgMean, bgCov, colMean, colCov, initAlpha, maxIterations = 10, minLikelihoodDelta = 10**(-3)):
 
	#Some global required constants
	fgSolved = np.zeros(3)
	bgSolved = np.zeros(3)
	alphaSolved = 0
	I = np.eye(3)
	nIterations = 0

	maxLikelihood = - np.inf
	
	for i in range(nClusters):
		try:
			ifgCov = np.linalg.inv(fgCov[:,:,i]) # inverse of ith FG cluster covariance
		except Exception as e:
			# print(fgCov[:,:,i])
			f.write('FGCOV ERROR = ')
			f.write(json.dumps(fgCov[:, :, i].tolist()))
			f.write('\n')
			continue

		curfgMean = fgMean[:, i] #mean of ith FG Cluster
		for j in range(nClusters):

			try:
				ibgCov = np.linalg.inv(bgCov[:,:,j]) # inverse of ith BG cluster covariance
			except Exception as e:
				# print(bgCov[:,:,j])
				f.write('BGCOV ERROR = ')
				f.write(json.dumps(bgCov[:, :, i].tolist()))
				f.write('\n')
				continue

			curbgMean = bgMean[:, j] #mean of ith BG Cluster

			#initialize before starting the iterative solver
			alpha = initAlpha
			nIterations = 0
			prevLikelihood = -np.inf
			while True:
				nIterations += 1
				
				#SOLVE FOR Foreground and background
				# define the equation Ax = b
				
				#define A
				a11 = ifgCov + I*(alpha/colCov)**2
				a12 = I*(alpha*(1-alpha)/colCov**2)
				a22 = ibgCov + I*((1 - alpha)/colCov)**2
				r1 = np.hstack((a11, a12))
				r2 = np.hstack((a12, a22))
				A = np.vstack((r1, r2))

				#define b
				b11 = np.dot(ifgCov, curfgMean) + colMean*(alpha/colCov**2)
				b12 = np.dot(ibgCov, curbgMean) + colMean*(1-alpha)/(colCov**2)
				b = np.concatenate((b11, b12)).T

				#solve Ax = b
				try:
					x = np.linalg.solve(A, b)
				except Exception as e:
					# print(e)
					f.write('LINALG ERROR  A= ')
					f.write(json.dumps(A.tolist()))
					f.write('\n')
					f.write('LINALG ERROR  B= ')
					f.write(json.dumps(b.tolist()))
					f.write('\n')
					f.write(str(e) )
					f.write('\n')
					print("ERROR SOLVING")
					break


				#assign fg and bg that are solved
				fgCol = x[0:3]
				bgCol = x[3:6]
				# print(x.shape, fgCol, bgCol)

				# SOLVE FOR ALPHA using estimated F, B, C        
				alpha = np.dot((colMean.T - bgCol).T, (fgCol - bgCol))/(np.sum(np.square(fgCol - bgCol)))
				
				#sometimes alpha goes beyond 0 and 1 range. Lets constrain that
				alpha = np.minimum(alpha, 1)
				alpha = np.maximum(0, alpha)
				
				
				# FIND LIKELIHOOD
				L = [None, None, None] #likelihoods
				L[0] = -np.sum((colMean.T - alpha*fgCol - (1-alpha)*bgCol)**2)/(colCov**2) # L(C|F,B, alpha)
				L[1] = -(np.dot(np.dot((fgCol - curfgMean.T).T, ifgCov), (fgCol - curfgMean.T).T)/2) #L(F)
				L[2] = -(np.dot(np.dot((bgCol - curbgMean.T).T, ibgCov), (bgCol - curbgMean.T).T)/2) #L(B)
				L = np.array(L)

				likelihood = np.sum(L)


				if likelihood > maxLikelihood:
					alphaSolved = alpha
					maxLikelihood = likelihood
					fgSolved = fgCol
					bgSolved = bgCol
				
				if abs(prevLikelihood - likelihood) < minLikelihoodDelta or nIterations > maxIterations:
					break

				prevLikelihood = likelihood
	
	# f.close()
	# print( initAlpha == alphaSolved)
	return fgSolved, bgSolved, alphaSolved

# f = open('temp.txt','w')

def findSolution(orig, trimap, threshold = 5):
	global fg, bg, alphaMask, nClusters
	
 
	windowSize = 7
	stdDevGaussian = 4

	bgMask, fgMask, alphaMask = getFBM(trimap)

	alphaMask = alphaMask[:,:,0].astype(np.float)
	
	Y,X = np.nonzero(alphaMask)
	
	unsolvedLocations = list(set(zip(Y,X)))
	unsolvedLocations = sorted(unsolvedLocations, key=itemgetter(0))

	alphaMask[alphaMask != 0] = np.nan

	alphaMask[fgMask[:,:,0] != 0] = 1.0
	
	fg = cv2.bitwise_and(fgMask,orig)
	bg = cv2.bitwise_and(bgMask, orig)
	iterations = 0
	solvedLocations = []
	locationsToSolve = unsolvedLocations.copy()
	startTime = time.time()
	sad = 0
	while len(locationsToSolve) != 0:
		locationsToSolve = []
		for loce in unsolvedLocations:
			if loce not in solvedLocations:
				locationsToSolve.append(loce)
		for location in locationsToSolve:

			# cv2.waitKey(1)
			# FOR VISUALIZATION
			
			
			# print(location)
			# if iterations == 1000:
			# 	break

			if startTime - time.time() > 300:
				# return fg, bg, alphaMask, sad
				threshold = 0
				nClusters = 2
			# if startTime - time.time() > 500:
			# 	# return fg, bg, alphaMask, sad
			# 	threshold = 0


			gaussianWeights = getGaussianWeights(windowSize, stdDevGaussian) #Gaussian Weighting to weight close by pixels more. 
			alphaWeightingWindow = getPixelNeighbourhood(alphaMask, location, windowSize) #Get the alpha values from the trimap
			
			
			weightingWindowFg = np.multiply(np.square(alphaWeightingWindow),gaussianWeights) #We weight by alpha^2 g
			fgWindow = getPixelNeighbourhood(fg, location, windowSize, threechannel = True)
			knownIndices = np.nan_to_num(weightingWindowFg) > 0
			fgPixels = fgWindow[knownIndices]
			fgWeights = weightingWindowFg[knownIndices]


			weightingWindowBg = np.multiply(np.square(1 - alphaWeightingWindow),gaussianWeights) #We weight by (1 - alpha^2) g
			bgWindow = getPixelNeighbourhood(bg, location, windowSize, threechannel = True)
			knownIndices = np.nan_to_num(weightingWindowBg) > 0
			bgPixels = bgWindow[knownIndices]
			bgWeights = weightingWindowBg[knownIndices]


			if len(bgPixels) < threshold or len(fgPixels) < threshold:
				continue
			else:
				solvedLocations.append(location)
			
			fgMean, fgCov = getMeanAndCov(fgPixels, fgWeights)
			bgMean, bgCov = getMeanAndCov(bgPixels, bgWeights)

			colorMean = orig[location]
			colorCov = 3
			
			initialAlpha = np.nanmean(alphaWeightingWindow.flatten())

			fgCol, bgCol, alphaVal = iterativeSolver(fgMean, fgCov, bgMean, bgCov, colorMean, colorCov, initialAlpha, 50)

			# if iterations == 0:
			#   break
			
			# print(alphaMask[location], alphaVal, end = ' ')
			alphaMask[location] = alphaVal
			# print(alphaMask[location])
			fg[location] = fgCol
			bg[location] = bgCol
		

			
			if iterations % 10 == 0:
				print('SOLVED LOCATIONS = ', len(solvedLocations))
				
				print('REMAINING LOCATIONS = ', len(unsolvedLocations) - len(solvedLocations))
				sad = np.sum(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY ) - (alphaMask*255).astype(np.uint8))
				print('SAD = ', sad)
				print('---------------------------------------------')
				
				showMultiImages((fg, bg, gt, cv2.cvtColor((alphaMask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR) ), 'CURRENT')
				# showMultiImages((), 'ALPHA')
				cv2.waitKey(1)

			iterations += 1
	print(iterations)


	# cv2.imshow('NAME', alphaMask)
	return fg, bg, alphaMask, sad


nClusters = 5
f1 = open('SAD.txt','w')
# for index in range(1, 27):
index = 5
image, trimap, trimap2, gt = getImages(index)

f=open("singulars.txt", "w")
fg, bg, alphaMask,sad = findSolution(image, trimap)


print(gt.shape, alphaMask.shape)
f1.write(str(index) + '=' + str(sad))


cv2.imwrite('OUTPUT/'+  str(index) + '-MATTE.png',(alphaMask*255).astype(np.uint8))
np.save('OUTPUT/'+  str(index) + '-MATTE.npy', alphaMask)
cv2.imwrite('OUTPUT/'+  str(index) + '-FG.png',fg)
cv2.imwrite('OUTPUT/'+  str(index) + '-BG.png',bg)

f.close()
	# cv2.waitKey(0)

# f1.close()

# showMultiImages((fg, bg ))
# cv2.waitKey(0)