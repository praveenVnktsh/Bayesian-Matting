import cv2
import numpy as np
from operator import itemgetter
import json
from numpy.linalg.linalg import solve
from sklearn.cluster import KMeans

def showMultiImages(arr, name = 'MULTIIMAGE'):
	"""
	Helper function that helps show images.
	"""
	arrNew = [cv2.resize(x, (0,0), fx = 1.5, fy = 1.5) for x in arr]
	cv2.imshow(name, np.concatenate(arrNew, axis=1))

def showImage(image, name = 'showimage'):
	"""
	Helper function that helps show images.
	"""
	cv2.imshow(name, cv2.resize(image, (350,350)))

def getImages(index, scale = 0.3):
	"""
	Grabs image from dataset and returns the ground truth, input, trimap, trimap 2 after rescaling to 30%.
	"""
	gt = cv2.resize(cv2.imread('trainingData/GT/GT' + str(index).zfill(2) + '.png'),(0,0), fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
	input = cv2.resize(cv2.imread('trainingData/input/GT' + str(index).zfill(2) + '.png'),(0,0), fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
	trimap = cv2.resize(cv2.imread('trainingData/trimap/Trimap1/GT' + str(index).zfill(2) + '.png'),(0,0), fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
	trimap2 = cv2.resize(cv2.imread('trainingData/trimap/Trimap2/GT' + str(index).zfill(2) + '.png'), (0,0),fx = scale, fy = scale, interpolation = cv2.INTER_NEAREST)
	return input, trimap, trimap2, gt


def getFBM(trimap):
	"""
	Helper function that returns the maps of surely background, surely foreground and unknown regions.
	"""

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
	"""
	Uses an opencv api to return a Gaussian kernel for processing.
	"""
	gaussTemp = cv2.getGaussianKernel(size, sigma) # returns a 1d gaussian filter
	kernel = np.dot(gaussTemp, gaussTemp.T) # since gaussTemp is 1d, we multiply to get a 2d kernel
	return kernel



def getPixelNeighbourhood(img, location,size, threechannel = False):
	"""
	Returns pixel neighbourhood around a given location
	"""

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
		ret = cv2.resize(ret, (size, size)) #resizing if the region isn't square. This happens at the edges of the image. 
		#Another possibility could be to pad the image with zeros or the mean value of the window.
	return ret



def getMeanAndCov(pixels, weights, nClusters = 5):
	"""
	Clusters the region into n clusters and then returns the weighted mean and covariances between RGB channels for each cluster stacked in a ort
	orthognal direction
	"""
	means = []
	covs = []
	kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(pixels) # Finds the centers of the clusters
	predictedClusters = np.array(kmeans.predict(pixels)) # Find the clusters
	
	for i in range(nClusters):
		trutharray = (predictedClusters == i) #Returns array where the cluster index equals the predicted cluster
		pixelsbin = pixels[trutharray]
		weightsbin = np.reshape(weights[trutharray], (weights[trutharray].size,1))
		weightsSqrt = np.sqrt(weightsbin) #for later, when computing covariance

		#compute the weighted mean
		meanBin = np.array([np.dot(pixelsbin[:,0], weightsbin),np.dot(pixelsbin[:,1], weightsbin),np.dot(pixelsbin[:,2], weightsbin)])/np.sum(weightsbin)
		meanBin = np.reshape(meanBin, (3,))
		means.append(meanBin)

		diff = (pixelsbin - meanBin)
		diff = np.multiply(weightsSqrt, diff)
		covBin = np.eye(3)
		with np.errstate(divide='raise'): # raise errors, but dont stop execution when low number of clusters, and other cases
			try:
				covBin = np.dot(diff.T, diff)/np.sum(weightsbin) + np.eye(3)*(10**(-5)) #adding small value to prevent singular matrices
			except Exception as e:
				print(covBin.shape)
				print(covBin)
		covs.append(covBin)  
	means = np.array(means).T
	covs = np.array(cv2.merge(covs))
	return means, covs

def iterativeSolver(fgMean, fgCov, bgMean, bgCov, colMean, colCov, initAlpha,  minLikelihoodDelta = 10**(-3), maxIterations = 10):
	"""
	Solves the matting equations iteratively and returns the best values of fg, bg, and alpha.
	We solve for the best pair of (FG, BG) clusters that maximizes the likelihood of a given F, B, alpha!

	"""
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
		except Exception as e: # Sometimes its singular, so we dump the matrix and skip for now!
			f.write('FGCOV ERROR = ')
			f.write(json.dumps(fgCov[:, :, i].tolist()))
			f.write('\n')
			continue

		curfgMean = fgMean[:, i] #mean of ith FG Cluster

		for j in range(nClusters):
			try:
				ibgCov = np.linalg.inv(bgCov[:,:,j]) # inverse of ith BG cluster covariance
			except Exception as e: # Sometimes its singular, so we dump the matrix and skip for now!
				f.write('BGCOV ERROR = ')
				f.write(json.dumps(bgCov[:, :, i].tolist()))
				f.write('\n')
				continue

			curbgMean = bgMean[:, j] #mean of ith BG Cluster

			#initialize values before starting the iterative solver
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
				except Exception as e: # Sometimes there is an issue with solving if A is non invertible. This did not occur after singular matrices were skipped, but is still there for safety.
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

				# SOLVE FOR ALPHA using estimated F, B, C        
				alpha = np.dot((colMean.T - bgCol).T, (fgCol - bgCol))/(np.sum(np.square(fgCol - bgCol)))
				
				#sometimes alpha goes beyond 0 and 1 range. Lets constrain that
				alpha = np.minimum(alpha, 1)
				alpha = np.maximum(0, alpha)
				
				
				# FIND LIKELIHOODS from the Gaussian Models
				L = [None, None, None] #likelihoods
				L[0] = -np.sum((colMean.T - alpha*fgCol - (1-alpha)*bgCol)**2)/(colCov**2) # L(C|F,B, alpha)
				L[1] = -(np.dot(np.dot((fgCol - curfgMean.T).T, ifgCov), (fgCol - curfgMean.T).T)/2) #L(F)
				L[2] = -(np.dot(np.dot((bgCol - curbgMean.T).T, ibgCov), (bgCol - curbgMean.T).T)/2) #L(B)
				L = np.array(L)

				likelihood = np.sum(L)

				if likelihood > maxLikelihood: # If best likelihood so far, use that
					alphaSolved = alpha
					maxLikelihood = likelihood
					fgSolved = fgCol
					bgSolved = bgCol
				
				# Stop solving if the solver saturates or if it exceeds the maximum number of iterations per cluster pair
				if abs(prevLikelihood - likelihood) < minLikelihoodDelta or nIterations > maxIterations: # Stop solving
					break

				prevLikelihood = likelihood

	return fgSolved, bgSolved, alphaSolved


def findSolution(orig, trimap, threshold = 10):
	"""
	Solves the bayesian matting problem given an image and the trimap. The threshold value defines the lowest number of datapoints 
	required for the solver to attempt solving. If the solver goes into an infinite loop due to low number of datapoints , the window size is increased.
	Returns the estimated foreground, background, alpha matte, and the value of SAD from the ground truth.
	"""
	global fg, bg, alphaMask, nClusters
	
	#define some constants
	origWindowSize = 9
	stdDevGaussian = 4
	bgMask, fgMask, alphaMask = getFBM(trimap)


	alphaMask = alphaMask[:,:,0].astype(np.float) #convert 3 channel image to one channel image
	Y,X = np.nonzero(alphaMask)
	unsolvedLocations = list(set(zip(Y,X))) #list of unsolved locations (y, x)
	unsolvedLocations = sorted(unsolvedLocations, key=itemgetter(0)) #to solve from top to bottom

	alphaMask[alphaMask != 0] = np.nan #if the place is unknown, replace with nan
	alphaMask[fgMask[:,:,0] != 0] = 1.0 #normalize values to one
	
	fg = cv2.bitwise_and(fgMask,orig) #find fg and bg
	bg = cv2.bitwise_and(bgMask, orig)

	iterations = 0
	solvedLocations = []
	locationsToSolve = unsolvedLocations.copy()
	sad = 0
	passNumber = 0
	
	while len(locationsToSolve) != 0: # for multiple passes over unsolved locations
		passNumber += 1
		locationsToSolve = []
		insufficientDatapoints = [] # use this to store if there is insufficient data (some pixel is skipped)
		for loce in unsolvedLocations: #find unsolved locations and solve them!
			if loce not in solvedLocations:
				locationsToSolve.append(loce)
				insufficientDatapoints.append(0)

		#If the window has too many unknowns, expand the window and finish solving.
		if passNumber >= 100:
			windowSize = ((origWindowSize + passNumber -100)//2)*2 + 1 #window size is an odd number, so //2 *2 + 1 gives odd
			# this will increase size over each pass even if it is not required, but still gives reasonable results
		else:
			windowSize = origWindowSize
		# windowSize = ((origWindowSize + passNumber -100)//2)*2 + 1
		print("UPDATING WINDOWSIZE TO ", windowSize)

		

		for location in locationsToSolve:
			
			gaussianWeights = getGaussianWeights(windowSize, stdDevGaussian) #Gaussian Weighting to weight close by pixels more. 
			alphaWeightingWindow = getPixelNeighbourhood(alphaMask, location, windowSize) #Get the alpha values from the trimap
			
			#find fg pixels and fg weights to cluster
			weightingWindowFg = np.multiply(np.square(alphaWeightingWindow),gaussianWeights) #We weight fg by alpha^2 g
			fgWindow = getPixelNeighbourhood(fg, location, windowSize, threechannel = True)
			knownIndices = np.nan_to_num(weightingWindowFg) > 0 #pixels where alpha is known is taken
			fgPixels = fgWindow[knownIndices]
			fgWeights = weightingWindowFg[knownIndices]

			#find bg pixels and bg weights to cluster
			weightingWindowBg = np.multiply(np.square(1 - alphaWeightingWindow),gaussianWeights) #We weight bg by (1 - alpha^2) g
			bgWindow = getPixelNeighbourhood(bg, location, windowSize, threechannel = True)
			knownIndices = np.nan_to_num(weightingWindowBg) > 0 #pixels where alpha is known is taken
			bgPixels = bgWindow[knownIndices]
			bgWeights = weightingWindowBg[knownIndices]



			# if the number of unknowns is too many, then skip those ones for now. Solve them at the very end when there is enough 
			# data!
			if len(bgPixels) < threshold or len(fgPixels) < threshold:
				insufficientDatapoints[locationsToSolve.index(location)] = 1
				continue
			else:
				insufficientDatapoints[locationsToSolve.index(location)] = 0
				solvedLocations.append(location)
			

			#Cluster bg and fg, and then get the weighted means and covariances of each cluster.
			fgMeans, fgCovs = getMeanAndCov(fgPixels, fgWeights)
			bgMeans, bgCovs = getMeanAndCov(bgPixels, bgWeights)
			
			colorMean = orig[location] #observed pixel C
			colorCov = 3 #covariance of the colour. This is a tunable parameter
			
			initialAlpha = np.nanmean(alphaWeightingWindow.flatten()) # Take initial guess as the mean. This makes sense since pixels nearby an alpha will be closer to alpha

			#iteratively solve!
			fgCol, bgCol, alphaVal = iterativeSolver(fgMeans, fgCovs, bgMeans, bgCovs, colorMean, colorCov, initialAlpha, maxIterations = 50)


			#update the estimated values
			alphaMask[location] = alphaVal
			fg[location] = fgCol
			bg[location] = bgCol


			#print occasional status updates
			if iterations % 10 == 0:
				print("PASS", passNumber)
				print("WINDOW SIZE = ", windowSize)
				print('SOLVED LOCATIONS = ', len(solvedLocations))
				print('REMAINING LOCATIONS = ', len(unsolvedLocations) - len(solvedLocations))
				print('INSUFFICIENT POINTS = ',  sum(insufficientDatapoints))
				sad = np.sum(np.absolute(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY ) - (alphaMask*255).astype(np.uint8)))/255
				print('SAD = ', sad)
				print('---------------------------------------------')
				showMultiImages((fg, bg, gt, cv2.cvtColor((alphaMask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR) ), 'CURRENT')
				cv2.waitKey(1)

			iterations += 1
	sad = np.sum(np.absolute(cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY ) - (alphaMask*255).astype(np.uint8)))/255
	print('SOLVER COMPLETED IN ', iterations, 'ITERATIONS')
	print('---------------------------------------------')
	print('---------------------------------------------')
	print('---------------------------------------------')


	return fg, bg, alphaMask, sad


nClusters = 5
f1 = open('SAD.txt','a')
f=open("singulars.txt", "w")

index = 1

print('STARTING WITH IMAGE', str(index))
print('---------------------------------------------')
image, trimap, trimap2, gt = getImages(index)
fg, bg, alphaMask,sad = findSolution(image, trimap)
f1.write('SAD-' + str(index) + ' = ' + str(sad) + ' in ' + str(alphaMask.size) +  'pixels \n' )
cv2.imwrite('OUTPUT/MATTE/'+  str(index) + '-MATTE.png',(alphaMask*255).astype(np.uint8))
np.save('OUTPUT/NUMPY/'+  str(index) + '.npy', alphaMask)
cv2.imwrite('OUTPUT/FG/'+  str(index) + '-FG.png',fg)
cv2.imwrite('OUTPUT/BG/'+  str(index) + '-BG.png',bg)
cv2.imwrite('OUTPUT/ORIG/'+  str(index) + '-ORIG.png',image)
cv2.imwrite('OUTPUT/GT/'+  str(index) + '-GT.png',gt)
cv2.imwrite('OUTPUT/TRIMAP/'+  str(index) + '-TRIMAP.png',gt)

f1.close()
