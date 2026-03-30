# author : zhangru                    time : 2020-08-04
# description : calculate the score of two match images

import os
import cv2
import random
import numpy as np
import scipy as sp
import scipy.linalg

	
	
# linear least square model for ransac
class LinearLeastSquareModel:
    
	# inputCols is the dimension of input, outputCols is the dimension of output
    def __init__(self, inputCols, outputCols):
        self.inputCols = inputCols
        self.outputCols = outputCols
    
	# compute the temp model
    def fit(self, data):
        A = np.vstack( [data[:,i] for i in self.inputCols] ).T
        B = np.vstack( [data[:,i] for i in self.outputCols] ).T
        x, resids, rank, s = scipy.linalg.lstsq(A, B)
        return x
		
	# compute the error of every input
    def getError(self, data, model):
        A = np.vstack( [data[:,i] for i in self.inputCols] ).T
        B = np.vstack( [data[:,i] for i in self.outputCols] ).T
        BFit = sp.dot(A, model)
        errPerPoint = np.sum( (B - BFit) ** 2, axis = 1 )
        return errPerPoint
		
		

# ransac function to fit affine transformation coefficient
# data : sample point 
# model: use LinearLeastSquareModel here
# leastNum : the least point to fit a model
# maxIter : max iter number
# threshold : the threshold to judge the point is inlier
# leastInlierNum : the least inlier points to judge the bestfit is fair 
def _ransac(data, model, leastNum, maxIter, threshold, leastInlierNum):

    # initial parameter setting 
	iterations = 0
	bestFit = None
	bestErr = np.inf

	# update within iterations
	while iterations < maxIter:
	
		# obtain the train and test data
		trainIdxs, testIdxs = randomPartition(leastNum, data.shape[0])
		trainInliers = data[trainIdxs, :]
		testPoints = data[testIdxs]

		# train model and compute error
		curModel = model.fit(trainInliers)
		testErr = model.getError(testPoints, curModel) 
		
		# get inlier test data
		testInlierIdxs = testIdxs[testErr < threshold]
		testInlierPoints = data[testInlierIdxs,:]
		
		# if leastInlierNum is fair, compute betterData betterModel and betterErrs
		if len(testInlierPoints) > leastInlierNum:
			betterData = np.concatenate((trainInliers, testInlierPoints))
			betterModel = model.fit(betterData)
			betterErrs = model.getError(betterData, betterModel)
			curErr = np.mean(betterErrs)
			# if curErr is lower, update
			if curErr < bestErr:
				bestFit = betterModel
				bestErr = curErr
			
		iterations += 1
		
	if bestFit is None:
		raise ValueError("Did't meet fit acceptance criteria")
	return bestFit,{'inliers':testInlierIdxs}

 
 
# upset data to select some train points, and the other as test
def randomPartition(breakPoint, dataSum):

	# upset data index
    allIdxs = np.arange(dataSum)
    np.random.shuffle(allIdxs)
	
    trainIdxs = allIdxs[:breakPoint]
    testIdxs = allIdxs[breakPoint:]
	
    return trainIdxs, testIdxs
	
	

# list the X and Y of feature
def listFeature(featureSrc):

	listSrcX = []
	listSrcY = []
	
	for i in range(len(featureSrc)):
		listSrcX.append(featureSrc[i][0])
		listSrcY.append(featureSrc[i][1])
		
	return listSrcX, listSrcY
	

	
# get some random unique points in range
def randomPoints(srcData, matchData, number):

	rangeIndex = len(srcData)
	
	# make sure the value is valid
	if (rangeIndex < number):
		raise ValueError("SrcData rangeIndex don't meet the request") 
	
	# get some random unique index in range
	randomIndex = random.sample(range(0, rangeIndex), number)
	randomSrcPoints = [srcData[i] for i in randomIndex]
	randomMatchPoints = [matchData[i] for i in randomIndex]
	
	return np.float32(randomSrcPoints), np.float32(randomMatchPoints)
	
	
	

# calculate the center of list	
def calMeans(data):
	
	sumHalf, sumAll, cntAll = 0, 0, 0
	
	# sort data
	data.sort()
	
	# calculate 1/4-3/4 as initial mean
	for i in range(len(data)//4, len(data)*3//4):
		sumHalf += data[i]
	avgHalf = sumHalf / (len(data)//2)

	# calculate final mean when meet the request
	for i in range(len(data)):
		if abs(data[i]-avgHalf) <= data[len(data)*3//4] - data[len(data)//4]:
			sumAll += data[i]
			cntAll += 1
	
	return sumAll / cntAll
	
	
	
# read feature pair from text	
def readTxt(fp):
	
	# read txt 
	txtSrc = open(fp + '/locations_1_to_use.txt', 'r')
	txtMatch = open(fp + '/locations_2_to_use.txt', 'r')
	
	featureSrc, featureMatch = [], []
	
	# save to list
	for line in txtSrc.readlines():
		line = line.rstrip('\n')
		line = [np.float32(i) for i in line.split(',')]
		featureSrc.append(line)
		
	for line in txtMatch.readlines():
		line = line.rstrip('\n')
		line = [np.float32(i) for i in line.split(',')]
		featureMatch.append(line)	
	
	return featureSrc, featureMatch